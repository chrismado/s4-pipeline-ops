"""Benchmark orchestrator for KV cache quantization experiments.

Runs all configured benchmarks, collects metrics, and saves results
incrementally as JSON for later report generation.
"""

from __future__ import annotations

import asyncio
import json
import platform
from datetime import datetime
from pathlib import Path

from loguru import logger
from pydantic import BaseModel

from src.benchmarks.configs import BenchmarkConfig, BenchmarkParams
from src.benchmarks.inference.llamacpp_bench import LlamaCppBenchmark
from src.benchmarks.inference.ollama_bench import InferenceBackend, OllamaBenchmark
from src.benchmarks.inference.vllm_bench import VLLMBenchmark
from src.benchmarks.metrics.latency import LatencyMetrics, LatencyTracker
from src.benchmarks.metrics.memory import MemoryMetrics, VRAMMonitor, get_vram_usage_mb, measure_memory
from src.benchmarks.metrics.quality import (
    QualityMetrics,
    compute_similarity,
    estimate_perplexity,
    measure_output_quality,
)
from src.benchmarks.metrics.throughput import (
    ContextScalingResult,
    ThroughputMetrics,
    measure_concurrent_throughput,
    measure_sequential_rpm,
)


class BenchmarkResult(BaseModel):
    """Complete result set for a single benchmark configuration."""

    config: BenchmarkConfig
    latency: LatencyMetrics
    memory: MemoryMetrics
    throughput: ThroughputMetrics
    quality: QualityMetrics
    timestamp: str
    hardware: dict


class BenchmarkRunner:
    """Orchestrates benchmark runs across all configured backends.

    Usage::

        runner = BenchmarkRunner(configs=BENCHMARK_SUITE)
        results = runner.run_all()
    """

    def __init__(
        self,
        configs: list[BenchmarkConfig],
        params: BenchmarkParams | None = None,
        output_dir: str = "benchmarks/results",
        prompts_path: str = "src/benchmarks/prompts/eval_prompts.json",
    ) -> None:
        self.configs = configs
        self.params = params or BenchmarkParams()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prompts = self._load_prompts(prompts_path)
        self.hardware = self._detect_hardware()
        self._baseline_outputs: list[str] | None = None

    def run_all(self) -> list[BenchmarkResult]:
        """Run all benchmark configurations and return results."""
        results: list[BenchmarkResult] = []

        for i, config in enumerate(self.configs, 1):
            logger.info(f"[{i}/{len(self.configs)}] Benchmarking: {config.name}")
            logger.info(f"  {config.description}")

            try:
                result = self._run_single(config)
                results.append(result)
                self._save_result(result)
                logger.info(
                    f"  Done: {result.latency.tps_mean:.1f} TPS, "
                    f"{result.memory.vram_peak_mb:.0f} MB VRAM"
                )
            except Exception as e:
                logger.error(f"  Failed: {e}")
                continue

        self._save_summary(results)
        return results

    def _run_single(self, config: BenchmarkConfig) -> BenchmarkResult:
        """Run a complete benchmark for a single configuration."""
        backend = self._get_backend(config)

        try:
            backend.start()
            gpu_id = config.gpu_ids[0] if config.gpu_ids else 0

            # Measure baseline VRAM after model load
            vram_baseline = get_vram_usage_mb(gpu_id)

            # Warmup
            logger.info(f"  Warming up ({self.params.warmup_runs} runs)...")
            warmup_prompt = self.prompts["short_generation"][0]
            for _ in range(self.params.warmup_runs):
                backend.generate(warmup_prompt, max_tokens=32)

            # Latency
            logger.info(f"  Measuring latency ({self.params.eval_runs} runs)...")
            latency = self._measure_latency(backend, gpu_id)

            # Memory
            logger.info("  Measuring memory...")
            memory = self._measure_memory(backend, gpu_id, vram_baseline)

            # Throughput
            logger.info("  Measuring throughput...")
            throughput = self._measure_throughput(backend)

            # Quality
            logger.info("  Measuring quality...")
            quality = self._measure_quality(backend, config)

            return BenchmarkResult(
                config=config,
                latency=latency,
                memory=memory,
                throughput=throughput,
                quality=quality,
                timestamp=datetime.now().isoformat(),
                hardware=self.hardware,
            )
        finally:
            backend.stop()

    def _measure_latency(
        self,
        backend: InferenceBackend,
        gpu_id: int,
    ) -> LatencyMetrics:
        """Run eval prompts and collect latency statistics."""
        tracker = LatencyTracker()
        # Cycle through short + medium prompts for variety
        prompts = self.prompts["short_generation"] + self.prompts["medium_generation"]

        for i in range(self.params.eval_runs):
            prompt = prompts[i % len(prompts)]
            result = backend.generate(prompt, self.params.max_tokens)
            tracker.record(
                ttft_ms=result.ttft_ms,
                tokens_generated=result.tokens_generated,
                e2e_ms=result.total_ms,
            )

        return tracker.compute()

    def _measure_memory(
        self,
        backend: InferenceBackend,
        gpu_id: int,
        vram_baseline: float,
    ) -> MemoryMetrics:
        """Measure VRAM peak during inference."""
        monitor = VRAMMonitor(gpu_index=gpu_id)
        monitor.start()

        # Generate a long output to maximize KV cache usage
        long_prompt = self.prompts["long_generation"][0]
        backend.generate(long_prompt, max_tokens=self.params.max_tokens)

        monitor.stop()

        return measure_memory(
            gpu_index=gpu_id,
            vram_baseline_mb=vram_baseline,
            vram_peak_mb=monitor.peak_mb,
        )

    def _measure_throughput(self, backend: InferenceBackend) -> ThroughputMetrics:
        """Measure sequential and concurrent throughput."""
        prompt = self.prompts["short_generation"][0]

        # Sequential
        seq_rpm = measure_sequential_rpm(
            backend, prompt, self.params.max_tokens, num_requests=10
        )

        # Concurrent at different levels
        concurrency_results = []
        for level in self.params.concurrency_levels:
            result = asyncio.run(
                measure_concurrent_throughput(
                    backend, prompt, self.params.max_tokens,
                    concurrency=level, num_requests=max(level * 2, 8),
                )
            )
            concurrency_results.append(result)

        # Context scaling (throughput at different context lengths)
        context_scaling = []
        for ctx_len in self.params.context_lengths:
            # Generate a prompt that exercises the context
            result = backend.generate(prompt, max_tokens=min(ctx_len // 4, 256))
            tps = result.tokens_generated / (result.total_ms / 1000.0) if result.total_ms > 0 else 0
            vram = get_vram_usage_mb(0)
            context_scaling.append(ContextScalingResult(
                context_length=ctx_len,
                tps=round(tps, 2),
                vram_mb=round(vram, 1),
                ttft_ms=round(result.ttft_ms, 2),
            ))

        return ThroughputMetrics(
            sequential_rpm=seq_rpm,
            concurrency_results=concurrency_results,
            context_scaling=context_scaling,
        )

    def _measure_quality(
        self,
        backend: InferenceBackend,
        config: BenchmarkConfig,
    ) -> QualityMetrics:
        """Measure output quality and compare to baseline."""
        # Use short + medium prompts for quality eval
        eval_prompts = self.prompts["short_generation"] + self.prompts["medium_generation"]

        outputs, quality = measure_output_quality(
            backend, eval_prompts, self.params.max_tokens
        )

        # Compute similarity to baseline if we have one
        if self._baseline_outputs is not None:
            quality.similarity_to_baseline = compute_similarity(
                self._baseline_outputs, outputs
            )
        elif config.kv_cache_type_k is None or config.kv_cache_type_k == "f16":
            # This is a baseline config — store outputs for comparison
            self._baseline_outputs = outputs

        # Attempt perplexity estimation
        perplexity = estimate_perplexity(backend, self.prompts["perplexity_eval"])
        quality.perplexity_proxy = perplexity

        return quality

    def _get_backend(self, config: BenchmarkConfig) -> InferenceBackend:
        """Instantiate the appropriate backend for a config."""
        if config.backend == "ollama":
            return OllamaBenchmark(config=config)
        elif config.backend == "llamacpp":
            return LlamaCppBenchmark(config=config)
        elif config.backend == "vllm":
            return VLLMBenchmark(config=config)
        else:
            raise ValueError(f"Unknown backend: {config.backend}")

    def _save_result(self, result: BenchmarkResult) -> None:
        """Save a single result as JSON."""
        path = self.output_dir / f"{result.config.name}.json"
        path.write_text(result.model_dump_json(indent=2))
        logger.info(f"  Saved: {path}")

    def _save_summary(self, results: list[BenchmarkResult]) -> None:
        """Save all results as a combined summary."""
        path = self.output_dir / "summary.json"
        summary = {
            "timestamp": datetime.now().isoformat(),
            "hardware": self.hardware,
            "results": [r.model_dump() for r in results],
        }
        path.write_text(json.dumps(summary, indent=2, default=str))
        logger.info(f"Summary saved: {path}")

    @staticmethod
    def _load_prompts(path: str) -> dict[str, list[str]]:
        """Load eval prompts from JSON."""
        prompts_path = Path(path)
        if not prompts_path.exists():
            logger.warning(f"Prompts file not found: {path}, using defaults")
            return {
                "short_generation": ["Explain quantum entanglement in one paragraph."],
                "medium_generation": ["Describe the architecture of a transformer neural network."],
                "long_generation": ["Write a guide to training a neural network from scratch."],
                "s4_domain": ["What did Bob Lazar claim about Element 115?"],
                "perplexity_eval": ["The quick brown fox jumps over the lazy dog."],
            }
        return json.loads(prompts_path.read_text())

    @staticmethod
    def _detect_hardware() -> dict:
        """Detect hardware specs for the report."""
        hw: dict = {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
        }
        try:
            import pynvml

            pynvml.nvmlInit()
            gpu_count = pynvml.nvmlDeviceGetCount()
            gpus = []
            for i in range(gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode("utf-8")
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpus.append({
                    "index": i,
                    "name": name,
                    "vram_total_mb": round(mem.total / (1024 * 1024)),
                })
            hw["gpus"] = gpus
        except Exception:
            hw["gpus"] = [{"index": 0, "name": "Unknown", "vram_total_mb": 0}]

        try:
            import psutil

            hw["ram_total_gb"] = round(psutil.virtual_memory().total / (1024**3), 1)
            hw["cpu_count"] = psutil.cpu_count()
        except ImportError:
            pass

        return hw
