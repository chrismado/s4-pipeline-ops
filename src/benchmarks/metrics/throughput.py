"""Throughput metrics: sequential and concurrent request benchmarking.

Measures requests-per-minute and tokens-per-second under various
concurrency levels and context lengths.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from src.benchmarks.inference.ollama_bench import InferenceBackend


class ConcurrencyResult(BaseModel):
    """Throughput at a given concurrency level."""

    concurrency: int
    requests_per_minute: float
    total_tokens: int
    aggregate_tps: float
    avg_latency_ms: float


class ContextScalingResult(BaseModel):
    """Throughput at a given context length."""

    context_length: int
    tps: float
    vram_mb: float
    ttft_ms: float


class ThroughputMetrics(BaseModel):
    """Aggregated throughput measurements across concurrency and context tests."""

    sequential_rpm: float
    concurrency_results: list[ConcurrencyResult]
    context_scaling: list[ContextScalingResult]


async def _run_single_request(
    backend: InferenceBackend,
    prompt: str,
    max_tokens: int,
) -> tuple[float, int]:
    """Run a single inference request, return (latency_ms, token_count)."""
    loop = asyncio.get_event_loop()
    start = time.perf_counter()
    result = await loop.run_in_executor(None, backend.generate, prompt, max_tokens)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return elapsed_ms, result.tokens_generated


async def measure_concurrent_throughput(
    backend: InferenceBackend,
    prompt: str,
    max_tokens: int,
    concurrency: int,
    num_requests: int = 16,
) -> ConcurrencyResult:
    """Measure throughput at a specific concurrency level.

    Dispatches `num_requests` total, with up to `concurrency` in flight at once.
    """
    semaphore = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    total_tokens = 0

    async def _bounded_request() -> None:
        nonlocal total_tokens
        async with semaphore:
            latency_ms, tokens = await _run_single_request(backend, prompt, max_tokens)
            latencies.append(latency_ms)
            total_tokens += tokens

    wall_start = time.perf_counter()
    await asyncio.gather(*[_bounded_request() for _ in range(num_requests)])
    wall_elapsed_s = time.perf_counter() - wall_start

    rpm = (num_requests / wall_elapsed_s) * 60.0 if wall_elapsed_s > 0 else 0.0
    agg_tps = total_tokens / wall_elapsed_s if wall_elapsed_s > 0 else 0.0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    return ConcurrencyResult(
        concurrency=concurrency,
        requests_per_minute=round(rpm, 2),
        total_tokens=total_tokens,
        aggregate_tps=round(agg_tps, 2),
        avg_latency_ms=round(avg_latency, 2),
    )


def measure_sequential_rpm(
    backend: InferenceBackend,
    prompt: str,
    max_tokens: int,
    num_requests: int = 10,
) -> float:
    """Measure sequential requests-per-minute."""
    start = time.perf_counter()
    for _ in range(num_requests):
        backend.generate(prompt, max_tokens)
    elapsed_s = time.perf_counter() - start
    return round((num_requests / elapsed_s) * 60.0, 2) if elapsed_s > 0 else 0.0
