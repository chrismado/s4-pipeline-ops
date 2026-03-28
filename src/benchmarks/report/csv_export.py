"""CSV export for raw benchmark data.

Exports results in a flat format suitable for analysis in
pandas, Excel, or other tools.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from src.benchmarks.runner import BenchmarkResult


def export_csv(
    results: list[BenchmarkResult],
    output_path: str = "benchmarks/results/benchmark_results.csv",
) -> str:
    """Export benchmark results to a flat CSV file.

    Returns the output path.
    """
    if not results:
        return output_path

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "config_name",
        "backend",
        "model",
        "model_quant",
        "kv_cache_type_k",
        "kv_cache_type_v",
        "context_length",
        # Latency
        "tps_mean",
        "tps_p50",
        "tps_p95",
        "tps_p99",
        "tps_std",
        "ttft_ms_mean",
        "ttft_ms_p50",
        "ttft_ms_p95",
        "ttft_ms_p99",
        "ttft_ms_std",
        "e2e_ms_mean",
        "e2e_ms_p50",
        "e2e_ms_p95",
        "e2e_ms_p99",
        "e2e_ms_std",
        "num_runs",
        # Memory
        "vram_baseline_mb",
        "vram_peak_mb",
        "vram_kv_cache_estimated_mb",
        "vram_total_mb",
        "max_batch_estimate",
        # Throughput
        "sequential_rpm",
        # Quality
        "avg_output_length",
        "output_length_std",
        "similarity_to_baseline",
        "perplexity_proxy",
        # Meta
        "timestamp",
    ]

    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            row = {
                "config_name": r.config.name,
                "backend": r.config.backend,
                "model": r.config.model,
                "model_quant": r.config.model_quant,
                "kv_cache_type_k": r.config.kv_cache_type_k or "",
                "kv_cache_type_v": r.config.kv_cache_type_v or "",
                "context_length": r.config.context_length,
                "tps_mean": r.latency.tps_mean,
                "tps_p50": r.latency.tps_p50,
                "tps_p95": r.latency.tps_p95,
                "tps_p99": r.latency.tps_p99,
                "tps_std": r.latency.tps_std,
                "ttft_ms_mean": r.latency.ttft_ms_mean,
                "ttft_ms_p50": r.latency.ttft_ms_p50,
                "ttft_ms_p95": r.latency.ttft_ms_p95,
                "ttft_ms_p99": r.latency.ttft_ms_p99,
                "ttft_ms_std": r.latency.ttft_ms_std,
                "e2e_ms_mean": r.latency.e2e_ms_mean,
                "e2e_ms_p50": r.latency.e2e_ms_p50,
                "e2e_ms_p95": r.latency.e2e_ms_p95,
                "e2e_ms_p99": r.latency.e2e_ms_p99,
                "e2e_ms_std": r.latency.e2e_ms_std,
                "num_runs": r.latency.num_runs,
                "vram_baseline_mb": r.memory.vram_baseline_mb,
                "vram_peak_mb": r.memory.vram_peak_mb,
                "vram_kv_cache_estimated_mb": r.memory.vram_kv_cache_estimated_mb,
                "vram_total_mb": r.memory.vram_total_mb,
                "max_batch_estimate": r.memory.max_batch_estimate,
                "sequential_rpm": r.throughput.sequential_rpm,
                "avg_output_length": r.quality.avg_output_length,
                "output_length_std": r.quality.output_length_std,
                "similarity_to_baseline": r.quality.similarity_to_baseline or "",
                "perplexity_proxy": r.quality.perplexity_proxy or "",
                "timestamp": r.timestamp,
            }
            writer.writerow(row)

    logger.info(f"CSV exported to {output_path}")
    return output_path
