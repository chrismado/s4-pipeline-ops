"""Tests for data models — validation, properties, edge cases."""

from datetime import UTC, datetime, timedelta

from src.models.schemas import (
    GPUMetrics,
    Job,
    JobConfig,
    JobStatus,
    PipelineHealth,
    SystemMetrics,
)


class TestGPUMetrics:
    def test_vram_fraction(self):
        gpu = GPUMetrics(
            gpu_index=0,
            temperature_c=55.0,
            utilization_pct=80.0,
            memory_used_mb=18000.0,
            memory_total_mb=24576.0,
            memory_utilization_pct=73.2,
        )
        assert 0.73 < gpu.vram_fraction < 0.74

    def test_vram_fraction_zero_total(self):
        gpu = GPUMetrics(
            gpu_index=0,
            temperature_c=40.0,
            utilization_pct=0.0,
            memory_used_mb=0.0,
            memory_total_mb=0.0,
            memory_utilization_pct=0.0,
        )
        # Should not divide by zero
        assert gpu.vram_fraction == 0.0

    def test_is_idle(self):
        gpu = GPUMetrics(
            gpu_index=0,
            temperature_c=35.0,
            utilization_pct=2.0,
            memory_used_mb=500.0,
            memory_total_mb=24576.0,
            memory_utilization_pct=2.0,
        )
        assert gpu.is_idle is True

    def test_not_idle(self):
        gpu = GPUMetrics(
            gpu_index=0,
            temperature_c=70.0,
            utilization_pct=95.0,
            memory_used_mb=20000.0,
            memory_total_mb=24576.0,
            memory_utilization_pct=81.0,
        )
        assert gpu.is_idle is False


class TestSystemMetrics:
    def test_gpu_count(self):
        metrics = SystemMetrics(
            gpus=[
                GPUMetrics(
                    gpu_index=i,
                    temperature_c=50.0,
                    utilization_pct=50.0,
                    memory_used_mb=10000.0,
                    memory_total_mb=24576.0,
                    memory_utilization_pct=40.0,
                )
                for i in range(4)
            ]
        )
        assert metrics.gpu_count == 4

    def test_total_vram(self):
        metrics = SystemMetrics(
            gpus=[
                GPUMetrics(
                    gpu_index=0,
                    temperature_c=50.0,
                    utilization_pct=50.0,
                    memory_used_mb=12000.0,
                    memory_total_mb=24576.0,
                    memory_utilization_pct=48.8,
                ),
                GPUMetrics(
                    gpu_index=1,
                    temperature_c=50.0,
                    utilization_pct=50.0,
                    memory_used_mb=8000.0,
                    memory_total_mb=24576.0,
                    memory_utilization_pct=32.6,
                ),
            ]
        )
        assert abs(metrics.total_vram_used_gb - 19.53125) < 0.01
        assert abs(metrics.total_vram_gb - 48.0) < 0.01


class TestJob:
    def test_duration_running(self):
        job = Job(
            id="test1",
            name="Test Job",
            config=JobConfig(command="echo hello"),
            status=JobStatus.RUNNING,
            started_at=datetime.now(UTC) - timedelta(seconds=120),
        )
        assert job.duration_seconds is not None
        assert job.duration_seconds >= 119  # Allow small timing margin

    def test_duration_not_started(self):
        job = Job(
            id="test2",
            name="Test Job",
            config=JobConfig(command="echo hello"),
        )
        assert job.duration_seconds is None

    def test_is_terminal(self):
        for status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            job = Job(
                id="test",
                name="Test",
                config=JobConfig(command="echo"),
                status=status,
            )
            assert job.is_terminal is True

    def test_is_not_terminal(self):
        for status in (JobStatus.PENDING, JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.RETRYING):
            job = Job(
                id="test",
                name="Test",
                config=JobConfig(command="echo"),
                status=status,
            )
            assert job.is_terminal is False


class TestPipelineHealth:
    def test_failure_rate(self):
        health = PipelineHealth(
            stages=[],
            total_jobs_24h=100,
            failed_jobs_24h=15,
        )
        assert health.failure_rate_24h == 15.0

    def test_failure_rate_zero_jobs(self):
        health = PipelineHealth(stages=[], total_jobs_24h=0, failed_jobs_24h=0)
        assert health.failure_rate_24h == 0.0
