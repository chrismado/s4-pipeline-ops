"""Prometheus metrics export for inference benchmark results.

Appends benchmark-specific gauge metrics to the existing Prometheus
exposition output from src/api/prometheus.py.
"""

from __future__ import annotations

import json
from pathlib import Path


def generate_benchmark_metrics(results_dir: str = "benchmarks/results") -> str:
    """Generate Prometheus exposition format metrics from benchmark results.

    Reads the summary.json from the results directory and exposes
    key metrics as Prometheus gauges.
    """
    summary_path = Path(results_dir) / "summary.json"
    if not summary_path.exists():
        return ""

    try:
        data = json.loads(summary_path.read_text())
    except (json.JSONDecodeError, OSError):
        return ""

    results = data.get("results", [])
    if not results:
        return ""

    lines: list[str] = []

    # Tokens per second
    lines.append("# HELP s4_inference_tokens_per_second Inference throughput in tokens/sec")
    lines.append("# TYPE s4_inference_tokens_per_second gauge")
    for r in results:
        config = r["config"]["name"]
        tps = r["latency"]["tps_mean"]
        lines.append(f's4_inference_tokens_per_second{{config="{config}"}} {tps:.2f}')

    # Time to first token
    lines.append("# HELP s4_inference_ttft_ms Time to first token in milliseconds")
    lines.append("# TYPE s4_inference_ttft_ms gauge")
    for r in results:
        config = r["config"]["name"]
        ttft = r["latency"]["ttft_ms_p50"]
        lines.append(f's4_inference_ttft_ms{{config="{config}"}} {ttft:.2f}')

    # VRAM usage
    lines.append("# HELP s4_inference_vram_mb VRAM peak usage in MB")
    lines.append("# TYPE s4_inference_vram_mb gauge")
    for r in results:
        config = r["config"]["name"]
        vram = r["memory"]["vram_peak_mb"]
        lines.append(f's4_inference_vram_mb{{config="{config}"}} {vram:.1f}')

    # KV cache size
    lines.append("# HELP s4_kv_cache_size_mb Estimated KV cache size in MB")
    lines.append("# TYPE s4_kv_cache_size_mb gauge")
    for r in results:
        config = r["config"]["name"]
        kv = r["memory"]["vram_kv_cache_estimated_mb"]
        lines.append(f's4_kv_cache_size_mb{{config="{config}"}} {kv:.1f}')

    # Sequential throughput
    lines.append("# HELP s4_inference_sequential_rpm Sequential requests per minute")
    lines.append("# TYPE s4_inference_sequential_rpm gauge")
    for r in results:
        config = r["config"]["name"]
        rpm = r["throughput"]["sequential_rpm"]
        lines.append(f's4_inference_sequential_rpm{{config="{config}"}} {rpm:.2f}')

    lines.append("")
    return "\n".join(lines)
