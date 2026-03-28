"""Matplotlib chart generation for benchmark reports.

Generates PNG charts for TPS comparison, VRAM usage, context scaling,
throughput curves, and quality-vs-speed tradeoffs.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from src.benchmarks.runner import BenchmarkResult


def generate_all_plots(
    results: list[BenchmarkResult],
    output_dir: str = "benchmarks/reports",
) -> list[str]:
    """Generate all benchmark charts. Returns list of saved file paths."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping chart generation")
        return []

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths: list[str] = []

    paths.append(_plot_tps_comparison(results, plt, out))
    paths.append(_plot_vram_comparison(results, plt, out))
    paths.append(_plot_context_scaling(results, plt, out))
    paths.append(_plot_throughput_concurrency(results, plt, out))
    paths.append(_plot_quality_vs_speed(results, plt, out))

    return [p for p in paths if p]


def _plot_tps_comparison(results: list[BenchmarkResult], plt, out: Path) -> str:
    """Bar chart: tokens per second across configurations."""
    fig, ax = plt.subplots(figsize=(12, 6))

    names = [r.config.name for r in results]
    tps_means = [r.latency.tps_mean for r in results]
    tps_stds = [r.latency.tps_std for r in results]

    colors = []
    for r in results:
        if r.config.backend == "ollama":
            colors.append("#4CAF50")
        elif r.config.backend == "llamacpp":
            colors.append("#2196F3")
        else:
            colors.append("#FF9800")

    ax.bar(range(len(names)), tps_means, yerr=tps_stds, color=colors, alpha=0.85, capsize=4)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Tokens per Second")
    ax.set_title("Inference Throughput: Tokens per Second by Configuration")
    ax.grid(axis="y", alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4CAF50", label="Ollama"),
        Patch(facecolor="#2196F3", label="llama.cpp"),
        Patch(facecolor="#FF9800", label="vLLM"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    fig.tight_layout()
    path = str(out / "tps_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Chart saved: {path}")
    return path


def _plot_vram_comparison(results: list[BenchmarkResult], plt, out: Path) -> str:
    """Stacked bar chart: VRAM breakdown (model vs KV cache)."""
    fig, ax = plt.subplots(figsize=(12, 6))

    names = [r.config.name for r in results]
    baselines = [r.memory.vram_baseline_mb for r in results]
    kv_caches = [r.memory.vram_kv_cache_estimated_mb for r in results]

    x = range(len(names))
    ax.bar(x, baselines, label="Model Weights", color="#5C6BC0", alpha=0.85)
    ax.bar(x, kv_caches, bottom=baselines, label="KV Cache (est.)", color="#EF5350", alpha=0.85)

    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("VRAM (MB)")
    ax.set_title("VRAM Usage: Model Weights vs KV Cache")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = str(out / "vram_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Chart saved: {path}")
    return path


def _plot_context_scaling(results: list[BenchmarkResult], plt, out: Path) -> str:
    """Line chart: VRAM usage vs context length."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for r in results:
        if not r.throughput.context_scaling:
            continue
        ctx_lens = [cs.context_length for cs in r.throughput.context_scaling]
        vram_vals = [cs.vram_mb for cs in r.throughput.context_scaling]
        ax.plot(ctx_lens, vram_vals, marker="o", label=r.config.name, linewidth=2)

    ax.set_xlabel("Context Length (tokens)")
    ax.set_ylabel("VRAM Usage (MB)")
    ax.set_title("VRAM vs Context Length — KV Cache Compression Extends Max Context")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = str(out / "context_scaling.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Chart saved: {path}")
    return path


def _plot_throughput_concurrency(results: list[BenchmarkResult], plt, out: Path) -> str:
    """Line chart: requests/min vs concurrency level."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for r in results:
        if not r.throughput.concurrency_results:
            continue
        levels = [cr.concurrency for cr in r.throughput.concurrency_results]
        rpms = [cr.requests_per_minute for cr in r.throughput.concurrency_results]
        ax.plot(levels, rpms, marker="s", label=r.config.name, linewidth=2)

    ax.set_xlabel("Concurrent Requests")
    ax.set_ylabel("Requests per Minute")
    ax.set_title("Throughput vs Concurrency — KV Cache Compression Enables Higher Parallelism")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_xticks([1, 2, 4, 8])

    fig.tight_layout()
    path = str(out / "throughput_concurrency.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Chart saved: {path}")
    return path


def _plot_quality_vs_speed(results: list[BenchmarkResult], plt, out: Path) -> str:
    """Scatter plot: quality (similarity) vs speed (TPS) tradeoff."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for r in results:
        sim = r.quality.similarity_to_baseline
        if sim is None:
            sim = 1.0  # baseline
        ax.scatter(
            r.latency.tps_mean,
            sim,
            s=100,
            alpha=0.8,
            label=r.config.name,
        )
        ax.annotate(
            r.config.name,
            (r.latency.tps_mean, sim),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=7,
        )

    ax.set_xlabel("Tokens per Second")
    ax.set_ylabel("Quality (Similarity to Baseline)")
    ax.set_title("Quality vs Speed Tradeoff — The Sweet Spot of KV Cache Quantization")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = str(out / "quality_vs_speed.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Chart saved: {path}")
    return path
