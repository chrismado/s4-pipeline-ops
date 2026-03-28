"""llama.cpp server inference backend for benchmarking.

Starts a llama-server process with configurable KV cache quantization
flags and communicates via the OpenAI-compatible API.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
from loguru import logger

from src.benchmarks.configs import BenchmarkConfig
from src.benchmarks.inference.ollama_bench import GenerationResult, InferenceBackend


@dataclass
class LlamaCppBenchmark(InferenceBackend):
    """llama.cpp server backend with KV cache quantization control."""

    config: BenchmarkConfig
    server_port: int = 8080
    model_dir: str = "models"
    _process: subprocess.Popen | None = None
    _client: httpx.Client | None = None

    def __post_init__(self) -> None:
        self._client = httpx.Client(
            base_url=f"http://localhost:{self.server_port}",
            timeout=300.0,
        )

    def start(self) -> None:
        """Start llama-server with the configured KV cache flags."""
        model_path = Path(self.model_dir) / self.config.model
        if not model_path.exists():
            logger.error(
                f"Model file not found: {model_path}. "
                f"Download from HuggingFace and place in {self.model_dir}/"
            )
            raise FileNotFoundError(f"GGUF model not found: {model_path}")

        cmd = [
            "llama-server",
            "--model", str(model_path),
            "--ctx-size", str(self.config.context_length),
            "--n-gpu-layers", "99",
            "--port", str(self.server_port),
            "--log-disable",
        ]

        if self.config.kv_cache_type_k:
            cmd.extend(["--cache-type-k", self.config.kv_cache_type_k])
        if self.config.kv_cache_type_v:
            cmd.extend(["--cache-type-v", self.config.kv_cache_type_v])

        logger.info(f"Starting llama-server: {' '.join(cmd)}")
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self._wait_for_ready()

    def stop(self) -> None:
        """Terminate the llama-server process."""
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
        if self._client:
            self._client.close()

    def is_ready(self) -> bool:
        try:
            resp = self._client.get("/health")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    def _wait_for_ready(self, timeout_s: float = 120.0) -> None:
        """Wait for llama-server to be ready to accept requests."""
        start = time.perf_counter()
        while time.perf_counter() - start < timeout_s:
            if self.is_ready():
                logger.info("llama-server is ready")
                return
            time.sleep(1.0)
        raise TimeoutError(f"llama-server not ready after {timeout_s}s")

    def generate(self, prompt: str, max_tokens: int = 256) -> GenerationResult:
        """Generate text via the OpenAI-compatible /v1/chat/completions endpoint."""
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": True,
        }

        tokens_generated = 0
        response_text = ""
        ttft_ms = 0.0
        start = time.perf_counter()
        first_token_seen = False
        log_probs: list[float] = []

        with self._client.stream(
            "POST", "/v1/chat/completions", json=payload
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break

                chunk = json.loads(data_str)
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content", "")

                if not first_token_seen and content:
                    ttft_ms = (time.perf_counter() - start) * 1000.0
                    first_token_seen = True

                if content:
                    response_text += content
                    tokens_generated += 1

                # Collect log probs if available
                logprob_info = chunk.get("choices", [{}])[0].get("logprobs")
                if logprob_info and logprob_info.get("content"):
                    for token_lp in logprob_info["content"]:
                        log_probs.append(token_lp.get("logprob", 0.0))

        total_ms = (time.perf_counter() - start) * 1000.0

        return GenerationResult(
            text=response_text,
            tokens_generated=tokens_generated,
            ttft_ms=ttft_ms,
            total_ms=total_ms,
            log_probs=log_probs if log_probs else None,
        )
