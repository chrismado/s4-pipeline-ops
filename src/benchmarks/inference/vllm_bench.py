"""vLLM inference backend for benchmarking (optional).

Starts a vLLM OpenAI-compatible server if available. Falls back
gracefully if vLLM is not installed or not compatible with
the current GPU.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass

import httpx
from loguru import logger

from src.benchmarks.configs import BenchmarkConfig
from src.benchmarks.inference.ollama_bench import GenerationResult, InferenceBackend


@dataclass
class VLLMBenchmark(InferenceBackend):
    """vLLM server backend with PagedAttention."""

    config: BenchmarkConfig
    server_port: int = 8000
    _process: subprocess.Popen | None = None
    _client: httpx.Client | None = None

    def __post_init__(self) -> None:
        self._client = httpx.Client(
            base_url=f"http://localhost:{self.server_port}",
            timeout=300.0,
        )

    def start(self) -> None:
        """Start vLLM server."""
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.config.model,
            "--port", str(self.server_port),
            "--max-model-len", str(self.config.context_length),
        ]

        # INT8 KV cache quantization in vLLM
        if self.config.kv_cache_type_k == "q8_0":
            cmd.extend(["--kv-cache-dtype", "int8"])

        logger.info(f"Starting vLLM server: {' '.join(cmd)}")
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self._wait_for_ready()

    def stop(self) -> None:
        """Terminate the vLLM server process."""
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=15)
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

    def _wait_for_ready(self, timeout_s: float = 180.0) -> None:
        """Wait for vLLM to finish loading the model."""
        start = time.perf_counter()
        while time.perf_counter() - start < timeout_s:
            if self.is_ready():
                logger.info("vLLM server is ready")
                return
            time.sleep(2.0)
        raise TimeoutError(f"vLLM server not ready after {timeout_s}s")

    def generate(self, prompt: str, max_tokens: int = 256) -> GenerationResult:
        """Generate text via vLLM's OpenAI-compatible endpoint."""
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

        total_ms = (time.perf_counter() - start) * 1000.0

        return GenerationResult(
            text=response_text,
            tokens_generated=tokens_generated,
            ttft_ms=ttft_ms,
            total_ms=total_ms,
        )
