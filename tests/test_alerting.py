"""Tests for alert engine — threshold evaluation and cooldown."""

from src.alerting.engine import AlertEngine
from src.models.schemas import (
    AlertType,
    GPUMetrics,
    Job,
    JobConfig,
    JobStatus,
    SystemMetrics,
)


class TestAlertEngine:
    def test_no_alerts_normal_metrics(self):
        engine = AlertEngine()
        metrics = SystemMetrics(
            gpus=[
                GPUMetrics(
                    gpu_index=0,
                    temperature_c=55.0,
                    utilization_pct=50.0,
                    memory_used_mb=8000.0,
                    memory_total_mb=24576.0,
                    memory_utilization_pct=32.0,
                )
            ],
            disk_used_gb=100,
            disk_total_gb=500,
        )
        alerts = engine.check_gpu_metrics(metrics)
        assert len(alerts) == 0

    def test_temp_warning(self):
        engine = AlertEngine()
        metrics = SystemMetrics(
            gpus=[
                GPUMetrics(
                    gpu_index=0,
                    temperature_c=85.0,
                    utilization_pct=90.0,
                    memory_used_mb=8000.0,
                    memory_total_mb=24576.0,
                    memory_utilization_pct=32.0,
                )
            ]
        )
        alerts = engine.check_gpu_metrics(metrics)
        temp_alerts = [a for a in alerts if a.type == AlertType.GPU_TEMP]
        assert len(temp_alerts) == 1
        assert "warning" in temp_alerts[0].severity.value.lower() or "critical" in temp_alerts[0].severity.value.lower()

    def test_vram_critical(self):
        engine = AlertEngine()
        metrics = SystemMetrics(
            gpus=[
                GPUMetrics(
                    gpu_index=0,
                    temperature_c=50.0,
                    utilization_pct=99.0,
                    memory_used_mb=23500.0,
                    memory_total_mb=24576.0,
                    memory_utilization_pct=95.6,
                )
            ]
        )
        alerts = engine.check_gpu_metrics(metrics)
        mem_alerts = [a for a in alerts if a.type == AlertType.GPU_MEMORY]
        assert len(mem_alerts) >= 1

    def test_cooldown_prevents_duplicate(self):
        engine = AlertEngine()
        metrics = SystemMetrics(
            gpus=[
                GPUMetrics(
                    gpu_index=0,
                    temperature_c=92.0,
                    utilization_pct=99.0,
                    memory_used_mb=8000.0,
                    memory_total_mb=24576.0,
                    memory_utilization_pct=32.0,
                )
            ]
        )
        alerts1 = engine.check_gpu_metrics(metrics)
        alerts2 = engine.check_gpu_metrics(metrics)
        # Second call should be suppressed by cooldown
        assert len(alerts1) > 0
        assert len(alerts2) == 0

    def test_job_failed_alert(self):
        engine = AlertEngine()
        job = Job(
            id="test",
            name="Training",
            config=JobConfig(command="train.py"),
            status=JobStatus.FAILED,
            exit_code=1,
            error_message="OOM",
            attempt=3,
            max_attempts=3,
        )
        alert = engine.check_job_event(job)
        assert alert is not None
        assert alert.type == AlertType.JOB_FAILED
