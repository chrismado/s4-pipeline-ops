"""Ollama inference backend for benchmarking.

Communicates with a running Ollama instance via its HTTP API.
Provides a common InferenceBackend interface used by all backends.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import httpx
from loguru import logger
from pydantic import BaseModel

from src.benchmarks.configs import BenchmarkConfig


class GenerationResult(BaseModel):
    """Result from a single inference call."""

    text: str
    tokens_generated: int
    ttft_ms: float
    total_ms: float
    log_probs: list[float] | None = None


class InferenceBackend:
    """Base protocol for inference backends."""

    def generate(self, prompt: str, max_tokens: int = 256) -> GenerationResult:
        raise NotImplementedError

    def start(self) -> None:
        """Start the backend server if needed."""

    def stop(self) -> None:
        """Stop the backend server if needed."""

    def is_ready(self) -> bool:
        """Check if the backend is ready to accept requests."""
        return True


@dataclass
class OllamaBenchmark(InferenceBackend):
    """Ollama inference backend using the /api/generate endpoint."""

    config: BenchmarkConfig
    base_url: str = "http://localhost:11434"
    _client: httpx.Client | None = None

    def __post_init__(self) -> None:
        self._client = httpx.Client(base_url=self.base_url, timeout=300.0)

    def start(self) -> None:
        """Ensure the model is pulled and loaded."""
        logger.info(f"Pulling Ollama model: {self.config.model}")
        try:
            self._client.post("/api/pull", json={"name": self.config.model}, timeout=600.0)
        except httpx.HTTPError as e:
            logger.warning(f"Failed to pull model (may already exist): {e}")

    def stop(self) -> None:
        if self._client:
            self._client.close()

    def is_ready(self) -> bool:
        try:
            resp = self._client.get("/api/tags")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    def generate(self, prompt: str, max_tokens: int = 256) -> GenerationResult:
        """Generate text via Ollama API with streaming for TTFT measurement."""
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "num_predict": max_tokens,
                "num_ctx": self.config.context_length,
            },
        }

        tokens_generated = 0
        response_text = ""
        ttft_ms = 0.0
        start = time.perf_counter()
        first_token_seen = False

        with self._client.stream("POST", "/api/generate", json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)

                if not first_token_seen and chunk.get("response"):
                    ttft_ms = (time.perf_counter() - start) * 1000.0
                    first_token_seen = True

                response_text += chunk.get("response", "")

                if chunk.get("done"):
                    tokens_generated = chunk.get("eval_count", len(response_text.split()))
                    break

        total_ms = (time.perf_counter() - start) * 1000.0

        return GenerationResult(
            text=response_text,
            tokens_generated=tokens_generated,
            ttft_ms=ttft_ms,
            total_ms=total_ms,
        )
