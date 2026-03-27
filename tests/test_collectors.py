"""Tests for GPU collectors — mock mode always available."""

from src.collectors.gpu import collect_gpu_metrics, collect_system_metrics


class TestGPUCollector:
    def test_collect_returns_metrics(self):
        """Should always return metrics (real or mock)."""
        metrics = collect_gpu_metrics()
        assert isinstance(metrics, list)
        assert len(metrics) > 0

    def test_metrics_have_valid_ranges(self):
        metrics = collect_gpu_metrics()
        for gpu in metrics:
            assert 0 <= gpu.temperature_c <= 120
            assert 0 <= gpu.utilization_pct <= 100
            assert gpu.memory_total_mb > 0

    def test_system_metrics(self):
        metrics = collect_system_metrics()
        assert metrics.gpu_count > 0
        assert metrics.ram_total_gb > 0
        assert metrics.disk_total_gb > 0
