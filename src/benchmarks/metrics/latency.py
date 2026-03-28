"""Latency metrics collection for inference benchmarks.

Measures TTFT, tokens-per-second, end-to-end latency, and computes
percentile distributions (p50/p95/p99) across multiple runs.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field

from pydantic import BaseModel


class LatencyMetrics(BaseModel):
    """Aggregated latency statistics across multiple benchmark runs."""

    ttft_ms_mean: float
    ttft_ms_p50: float
    ttft_ms_p95: float
    ttft_ms_p99: float
    ttft_ms_std: float

    tps_mean: float
    tps_p50: float
    tps_p95: float
    tps_p99: float
    tps_std: float

    e2e_ms_mean: float
    e2e_ms_p50: float
    e2e_ms_p95: float
    e2e_ms_p99: float
    e2e_ms_std: float

    num_runs: int


@dataclass
class LatencyTracker:
    """Collects per-run latency samples and computes aggregate statistics."""

    ttft_samples: list[float] = field(default_factory=list)
    tps_samples: list[float] = field(default_factory=list)
    e2e_samples: list[float] = field(default_factory=list)

    def record(self, ttft_ms: float, tokens_generated: int, e2e_ms: float) -> None:
        """Record a single run's latency measurements.

        Args:
            ttft_ms: Time to first token in milliseconds.
            tokens_generated: Number of tokens generated in this run.
            e2e_ms: End-to-end latency in milliseconds.
        """
        self.ttft_samples.append(ttft_ms)
        generation_time_s = (e2e_ms - ttft_ms) / 1000.0
        tps = tokens_generated / generation_time_s if generation_time_s > 0 else 0.0
        self.tps_samples.append(tps)
        self.e2e_samples.append(e2e_ms)

    def compute(self) -> LatencyMetrics:
        """Compute aggregate statistics from recorded samples."""
        if not self.ttft_samples:
            raise ValueError("No latency samples recorded — cannot compute metrics")
        return LatencyMetrics(
            ttft_ms_mean=statistics.mean(self.ttft_samples),
            ttft_ms_p50=_percentile(self.ttft_samples, 50),
            ttft_ms_p95=_percentile(self.ttft_samples, 95),
            ttft_ms_p99=_percentile(self.ttft_samples, 99),
            ttft_ms_std=statistics.stdev(self.ttft_samples) if len(self.ttft_samples) > 1 else 0.0,
            tps_mean=statistics.mean(self.tps_samples),
            tps_p50=_percentile(self.tps_samples, 50),
            tps_p95=_percentile(self.tps_samples, 95),
            tps_p99=_percentile(self.tps_samples, 99),
            tps_std=statistics.stdev(self.tps_samples) if len(self.tps_samples) > 1 else 0.0,
            e2e_ms_mean=statistics.mean(self.e2e_samples),
            e2e_ms_p50=_percentile(self.e2e_samples, 50),
            e2e_ms_p95=_percentile(self.e2e_samples, 95),
            e2e_ms_p99=_percentile(self.e2e_samples, 99),
            e2e_ms_std=statistics.stdev(self.e2e_samples) if len(self.e2e_samples) > 1 else 0.0,
            num_runs=len(self.ttft_samples),
        )


class TimingContext:
    """Context manager for timing operations with millisecond precision."""

    def __init__(self) -> None:
        self.start: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> TimingContext:
        self.start = time.perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        self.elapsed_ms = (time.perf_counter() - self.start) * 1000.0

    def mark(self) -> float:
        """Return elapsed time in ms since start without stopping the timer."""
        return (time.perf_counter() - self.start) * 1000.0


def _percentile(data: list[float], pct: int) -> float:
    """Compute a percentile value from a sorted list of samples."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (pct / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[-1]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])
