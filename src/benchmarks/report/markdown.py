"""Markdown report generator for benchmark results.

Produces a publication-ready markdown document with tables,
key findings, and methodology documentation.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from src.benchmarks.runner import BenchmarkResult


def generate_markdown_report(
    results: list[BenchmarkResult],
    output_path: str = "docs/kv_cache_benchmarks.md",
) -> str:
    """Generate a complete markdown benchmark report.

    Returns the markdown content and writes to output_path.
    """
    if not results:
        return "# No benchmark results available.\n"

    hardware = results[0].hardware
    gpu_name = "Unknown GPU"
    gpu_vram = "Unknown"
    if hardware.get("gpus"):
        gpu_name = hardware["gpus"][0].get("name", "Unknown GPU")
        gpu_vram = f"{hardware['gpus'][0].get('vram_total_mb', 0):,} MB"

    sections = [
        _header(hardware, gpu_name, gpu_vram),
        _results_summary_table(results),
        _key_findings(results),
        _concurrency_table(results),
        _context_scaling_table(results),
        _quality_table(results),
        _methodology(),
    ]

    content = "\n\n".join(sections) + "\n"

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content)
    logger.info(f"Report written to {output_path}")

    return content


def _header(hardware: dict, gpu_name: str, gpu_vram: str) -> str:
    return f"""# KV Cache Quantization Benchmarks — {datetime.now().strftime("%B %Y")}

Inspired by Google Research's TurboQuant (March 25, 2026), which demonstrated
6× KV cache memory reduction and 8× attention speedup with no measurable
accuracy loss on H100 GPUs. These benchmarks measure the real-world impact of
KV cache quantization on Mistral 7B inference using consumer hardware.

## Hardware

| Component | Specification |
|-----------|--------------|
| GPU | {gpu_name} ({gpu_vram} VRAM) |
| CPU | {hardware.get("processor", "N/A")} |
| RAM | {hardware.get("ram_total_gb", "N/A")} GB |
| OS | {hardware.get("platform", "N/A")} |
| Python | {hardware.get("python_version", "N/A")} |"""


def _results_summary_table(results: list[BenchmarkResult]) -> str:
    lines = [
        "## Results Summary",
        "",
        "| Config | TPS (mean) | TPS (p95) | TTFT (ms) | VRAM Peak (MB) | KV Cache Est. (MB) | Quality Sim. |",
        "|--------|-----------|-----------|-----------|----------------|-------------------|-------------|",
    ]

    for r in results:
        sim = f"{r.quality.similarity_to_baseline:.3f}" if r.quality.similarity_to_baseline is not None else "baseline"
        lines.append(
            f"| {r.config.name} "
            f"| {r.latency.tps_mean:.1f} "
            f"| {r.latency.tps_p95:.1f} "
            f"| {r.latency.ttft_ms_p50:.0f} "
            f"| {r.memory.vram_peak_mb:,.0f} "
            f"| {r.memory.vram_kv_cache_estimated_mb:,.0f} "
            f"| {sim} |"
        )

    return "\n".join(lines)


def _key_findings(results: list[BenchmarkResult]) -> str:
    lines = ["## Key Findings", ""]

    # Find baseline (first FP16 KV config or first config overall)
    baseline = None
    for r in results:
        if r.config.kv_cache_type_k in (None, "f16"):
            baseline = r
            break
    if baseline is None and results:
        baseline = results[0]

    if baseline is None:
        return "## Key Findings\n\nNo baseline found for comparison."

    for r in results:
        if r is baseline:
            continue
        tps_delta = ((r.latency.tps_mean / baseline.latency.tps_mean) - 1) * 100 if baseline.latency.tps_mean else 0
        mem_ratio = baseline.memory.vram_kv_cache_estimated_mb / r.memory.vram_kv_cache_estimated_mb if r.memory.vram_kv_cache_estimated_mb > 0 else 0
        sim = r.quality.similarity_to_baseline

        finding = f"- **{r.config.name}**: "
        if mem_ratio > 1:
            finding += f"{mem_ratio:.1f}× KV cache memory reduction, "
        finding += f"{tps_delta:+.1f}% TPS"
        if sim is not None:
            finding += f", {sim:.3f} similarity to baseline"
        lines.append(finding)

    return "\n".join(lines)


def _concurrency_table(results: list[BenchmarkResult]) -> str:
    lines = [
        "## Throughput Under Concurrency",
        "",
        "| Config | Seq. RPM | 1 Conc. | 2 Conc. | 4 Conc. | 8 Conc. |",
        "|--------|---------|---------|---------|---------|---------|",
    ]

    for r in results:
        conc_map = {cr.concurrency: cr.requests_per_minute for cr in r.throughput.concurrency_results}
        lines.append(
            f"| {r.config.name} "
            f"| {r.throughput.sequential_rpm:.1f} "
            f"| {conc_map.get(1, 0):.1f} "
            f"| {conc_map.get(2, 0):.1f} "
            f"| {conc_map.get(4, 0):.1f} "
            f"| {conc_map.get(8, 0):.1f} |"
        )

    return "\n".join(lines)


def _context_scaling_table(results: list[BenchmarkResult]) -> str:
    lines = [
        "## Context Length Scaling",
        "",
        "VRAM usage (MB) at different context lengths:",
        "",
        "| Config | 512 | 1024 | 2048 | 4096 |",
        "|--------|-----|------|------|------|",
    ]

    for r in results:
        ctx_map = {cs.context_length: cs.vram_mb for cs in r.throughput.context_scaling}
        lines.append(
            f"| {r.config.name} "
            f"| {ctx_map.get(512, 0):,.0f} "
            f"| {ctx_map.get(1024, 0):,.0f} "
            f"| {ctx_map.get(2048, 0):,.0f} "
            f"| {ctx_map.get(4096, 0):,.0f} |"
        )

    return "\n".join(lines)


def _quality_table(results: list[BenchmarkResult]) -> str:
    lines = [
        "## Quality Metrics",
        "",
        "| Config | Avg Output Len (words) | Similarity to Baseline | Perplexity |",
        "|--------|----------------------|----------------------|------------|",
    ]

    for r in results:
        sim = f"{r.quality.similarity_to_baseline:.4f}" if r.quality.similarity_to_baseline is not None else "baseline"
        ppl = f"{r.quality.perplexity_proxy:.2f}" if r.quality.perplexity_proxy is not None else "N/A"
        lines.append(
            f"| {r.config.name} "
            f"| {r.quality.avg_output_length:.0f} ± {r.quality.output_length_std:.0f} "
            f"| {sim} "
            f"| {ppl} |"
        )

    return "\n".join(lines)


def _methodology() -> str:
    return """## Methodology

- **Warmup**: 5 runs discarded before measurement to eliminate cold-start effects
- **Sample size**: 50 runs per configuration for statistical significance
- **Prompts**: Standardized eval set (short, medium, long generation tasks)
- **Metrics**: TTFT, TPS, e2e latency reported as mean with p50/p95/p99 percentiles
- **Memory**: VRAM sampled at 50ms intervals via pynvml; KV cache estimated as peak minus baseline
- **Quality**: Word-overlap similarity (Jaccard) against FP16 baseline; perplexity via log-probs where supported
- **Concurrency**: Async requests with semaphore-bounded parallelism at 1, 2, 4, 8 levels

### Reproducibility

All benchmark configurations, prompts, and raw results (JSON + CSV) are stored in `benchmarks/`.
Re-run with `s4ops benchmark --all --report` to reproduce.

---

*Generated by s4-pipeline-ops benchmark suite*"""
