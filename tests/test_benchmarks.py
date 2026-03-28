"""Tests for the benchmark module — configs, metrics, report generation."""

import json
import math
from pathlib import Path

import pytest

from src.benchmarks.configs import (
    BENCHMARK_SUITE,
    LLAMACPP_CONFIGS,
    OLLAMA_CONFIGS,
    VLLM_CONFIGS,
    BenchmarkConfig,
    BenchmarkParams,
)
from src.benchmarks.inference.ollama_bench import GenerationResult, InferenceBackend
from src.benchmarks.metrics.latency import LatencyMetrics, LatencyTracker, TimingContext, _percentile
from src.benchmarks.metrics.memory import MemoryMetrics, measure_memory
from src.benchmarks.metrics.quality import QualityMetrics, compute_similarity, _std
from src.benchmarks.metrics.throughput import ConcurrencyResult, ThroughputMetrics


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestBenchmarkConfigs:
    def test_suite_not_empty(self):
        assert len(BENCHMARK_SUITE) > 0

    def test_all_configs_have_names(self):
        for config in BENCHMARK_SUITE:
            assert config.name
            assert config.backend in ("ollama", "llamacpp", "vllm")
            assert config.model

    def test_unique_config_names(self):
        names = [c.name for c in BENCHMARK_SUITE]
        assert len(names) == len(set(names)), "Config names must be unique"

    def test_ollama_configs_no_kv_cache(self):
        for config in OLLAMA_CONFIGS:
            assert config.kv_cache_type_k is None
            assert config.kv_cache_type_v is None

    def test_llamacpp_configs_have_kv_cache(self):
        for config in LLAMACPP_CONFIGS:
            assert config.kv_cache_type_k is not None
            assert config.kv_cache_type_v is not None

    def test_config_serialization(self):
        config = BENCHMARK_SUITE[0]
        data = config.model_dump()
        restored = BenchmarkConfig(**data)
        assert restored.name == config.name

    def test_benchmark_params_defaults(self):
        params = BenchmarkParams()
        assert params.warmup_runs == 5
        assert params.eval_runs == 50
        assert params.max_tokens == 256
        assert 1 in params.concurrency_levels
        assert 8 in params.concurrency_levels


# ---------------------------------------------------------------------------
# Latency metric tests
# ---------------------------------------------------------------------------


class TestLatencyMetrics:
    def test_tracker_single_sample(self):
        tracker = LatencyTracker()
        tracker.record(ttft_ms=50.0, tokens_generated=100, e2e_ms=1050.0)
        metrics = tracker.compute()
        assert metrics.num_runs == 1
        assert metrics.ttft_ms_mean == 50.0
        assert metrics.tps_mean == pytest.approx(100.0, abs=0.1)

    def test_tracker_multiple_samples(self):
        tracker = LatencyTracker()
        for i in range(10):
            tracker.record(
                ttft_ms=40.0 + i,
                tokens_generated=100,
                e2e_ms=1000.0 + i * 10,
            )
        metrics = tracker.compute()
        assert metrics.num_runs == 10
        assert metrics.ttft_ms_std > 0

    def test_percentile_basic(self):
        data = list(range(1, 101))
        assert _percentile(data, 50) == pytest.approx(50.5, abs=0.1)
        assert _percentile(data, 99) == pytest.approx(99.01, abs=0.5)

    def test_percentile_empty(self):
        assert _percentile([], 50) == 0.0

    def test_tracker_empty_raises(self):
        tracker = LatencyTracker()
        with pytest.raises(ValueError, match="No latency samples"):
            tracker.compute()

    def test_timing_context(self):
        with TimingContext() as timer:
            total = sum(range(1000))
        assert timer.elapsed_ms > 0
        assert timer.elapsed_ms < 1000  # Should be very fast

    def test_timing_context_mark(self):
        with TimingContext() as timer:
            mid = timer.mark()
            total = sum(range(10000))
        assert mid <= timer.elapsed_ms


# ---------------------------------------------------------------------------
# Memory metric tests
# ---------------------------------------------------------------------------


class TestMemoryMetrics:
    def test_measure_memory_with_values(self):
        metrics = measure_memory(
            gpu_index=0,
            vram_baseline_mb=5000.0,
            vram_peak_mb=7000.0,
        )
        assert metrics.vram_baseline_mb == 5000.0
        assert metrics.vram_peak_mb == 7000.0
        assert metrics.vram_kv_cache_estimated_mb == 2000.0
        assert metrics.max_batch_estimate > 0

    def test_kv_cache_estimate_no_negative(self):
        metrics = measure_memory(
            gpu_index=0,
            vram_baseline_mb=8000.0,
            vram_peak_mb=7000.0,
        )
        assert metrics.vram_kv_cache_estimated_mb == 0.0


# ---------------------------------------------------------------------------
# Quality metric tests
# ---------------------------------------------------------------------------


class TestQualityMetrics:
    def test_similarity_identical(self):
        outputs = ["hello world foo bar", "test output data"]
        assert compute_similarity(outputs, outputs) == 1.0

    def test_similarity_different(self):
        a = ["the quick brown fox"]
        b = ["completely different text"]
        sim = compute_similarity(a, b)
        assert 0.0 <= sim < 1.0

    def test_similarity_mismatched_lengths(self):
        assert compute_similarity(["a"], ["b", "c"]) == 0.0

    def test_similarity_empty(self):
        assert compute_similarity([""], [""]) == 1.0

    def test_std_single_value(self):
        assert _std([5]) == 0.0

    def test_std_multiple_values(self):
        assert _std([2, 4, 4, 4, 5, 5, 7, 9]) == pytest.approx(2.0, abs=0.2)


# ---------------------------------------------------------------------------
# Throughput metric tests
# ---------------------------------------------------------------------------


class TestThroughputMetrics:
    def test_concurrency_result_model(self):
        result = ConcurrencyResult(
            concurrency=4,
            requests_per_minute=120.5,
            total_tokens=1000,
            aggregate_tps=50.2,
            avg_latency_ms=250.0,
        )
        assert result.concurrency == 4

    def test_throughput_metrics_model(self):
        metrics = ThroughputMetrics(
            sequential_rpm=30.0,
            concurrency_results=[],
            context_scaling=[],
        )
        assert metrics.sequential_rpm == 30.0


# ---------------------------------------------------------------------------
# Generation result tests
# ---------------------------------------------------------------------------


class TestGenerationResult:
    def test_result_model(self):
        result = GenerationResult(
            text="Hello world",
            tokens_generated=2,
            ttft_ms=50.0,
            total_ms=100.0,
        )
        assert result.tokens_generated == 2
        assert result.log_probs is None

    def test_result_with_log_probs(self):
        result = GenerationResult(
            text="Hello",
            tokens_generated=1,
            ttft_ms=30.0,
            total_ms=80.0,
            log_probs=[-0.5, -0.3],
        )
        assert len(result.log_probs) == 2


# ---------------------------------------------------------------------------
# Report tests
# ---------------------------------------------------------------------------


class TestReportGeneration:
    def _make_result(self, name: str = "test-config"):
        """Create a minimal BenchmarkResult for testing."""
        from src.benchmarks.runner import BenchmarkResult

        return BenchmarkResult(
            config=BenchmarkConfig(
                name=name,
                backend="llamacpp",
                model="test.gguf",
                model_quant="q8_0",
                kv_cache_type_k="f16",
                kv_cache_type_v="f16",
            ),
            latency=LatencyMetrics(
                ttft_ms_mean=50.0, ttft_ms_p50=48.0, ttft_ms_p95=65.0,
                ttft_ms_p99=80.0, ttft_ms_std=5.0,
                tps_mean=45.0, tps_p50=44.0, tps_p95=50.0,
                tps_p99=52.0, tps_std=3.0,
                e2e_ms_mean=1200.0, e2e_ms_p50=1180.0, e2e_ms_p95=1400.0,
                e2e_ms_p99=1500.0, e2e_ms_std=80.0,
                num_runs=50,
            ),
            memory=MemoryMetrics(
                vram_baseline_mb=5000.0, vram_peak_mb=7000.0,
                vram_kv_cache_estimated_mb=2000.0,
                vram_total_mb=24576.0, max_batch_estimate=9,
            ),
            throughput=ThroughputMetrics(
                sequential_rpm=30.0,
                concurrency_results=[
                    ConcurrencyResult(
                        concurrency=1, requests_per_minute=30.0,
                        total_tokens=500, aggregate_tps=45.0, avg_latency_ms=1200.0,
                    ),
                ],
                context_scaling=[],
            ),
            quality=QualityMetrics(
                avg_output_length=50.0, output_length_std=10.0,
            ),
            timestamp="2026-03-27T12:00:00",
            hardware={"platform": "test", "gpus": [{"name": "RTX 4090", "vram_total_mb": 24576}]},
        )

    def test_markdown_report(self, tmp_path):
        from src.benchmarks.report.markdown import generate_markdown_report

        results = [self._make_result("config-a"), self._make_result("config-b")]
        output = tmp_path / "report.md"
        content = generate_markdown_report(results, str(output))

        assert output.exists()
        assert "KV Cache Quantization Benchmarks" in content
        assert "config-a" in content
        assert "config-b" in content

    def test_csv_export(self, tmp_path):
        from src.benchmarks.report.csv_export import export_csv

        results = [self._make_result()]
        output = tmp_path / "results.csv"
        export_csv(results, str(output))

        assert output.exists()
        content = output.read_text()
        assert "test-config" in content
        assert "tps_mean" in content

    def test_markdown_empty_results(self):
        from src.benchmarks.report.markdown import generate_markdown_report

        content = generate_markdown_report([])
        assert "No benchmark results" in content


# ---------------------------------------------------------------------------
# Prometheus metric tests
# ---------------------------------------------------------------------------


class TestBenchmarkPrometheus:
    def test_no_results_returns_empty(self):
        from src.benchmarks.prometheus import generate_benchmark_metrics

        result = generate_benchmark_metrics(results_dir="/nonexistent")
        assert result == ""

    def test_metrics_from_summary(self, tmp_path):
        from src.benchmarks.prometheus import generate_benchmark_metrics

        summary = {
            "results": [
                {
                    "config": {"name": "test-config"},
                    "latency": {"tps_mean": 45.0, "ttft_ms_p50": 50.0},
                    "memory": {"vram_peak_mb": 7000.0, "vram_kv_cache_estimated_mb": 2000.0},
                    "throughput": {"sequential_rpm": 30.0},
                }
            ]
        }
        (tmp_path / "summary.json").write_text(json.dumps(summary))

        output = generate_benchmark_metrics(str(tmp_path))
        assert "s4_inference_tokens_per_second" in output
        assert "s4_inference_ttft_ms" in output
        assert "s4_kv_cache_size_mb" in output
        assert 'config="test-config"' in output


# ---------------------------------------------------------------------------
# Eval prompts tests
# ---------------------------------------------------------------------------


class TestEvalPrompts:
    def test_prompts_file_valid_json(self):
        path = Path("src/benchmarks/prompts/eval_prompts.json")
        data = json.loads(path.read_text())
        assert "short_generation" in data
        assert "medium_generation" in data
        assert "long_generation" in data
        assert "perplexity_eval" in data

    def test_prompts_non_empty(self):
        path = Path("src/benchmarks/prompts/eval_prompts.json")
        data = json.loads(path.read_text())
        for key, prompts in data.items():
            assert len(prompts) > 0, f"Prompt category '{key}' is empty"
