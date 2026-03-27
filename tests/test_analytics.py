"""Tests for historical job analytics."""

from datetime import datetime, timedelta

from src.dashboard.analytics import JobAnalytics
from src.models.schemas import Job, JobConfig, JobStatus, JobPriority


def _make_job(
    name="Test",
    status=JobStatus.COMPLETED,
    tags=None,
    duration=60.0,
    priority=JobPriority.NORMAL,
    error_message=None,
    gpu_ids=None,
):
    now = datetime.utcnow()
    return Job(
        id=name[:8],
        name=name,
        config=JobConfig(command="echo", tags=tags or []),
        status=status,
        priority=priority,
        created_at=now - timedelta(seconds=duration + 10),
        started_at=now - timedelta(seconds=duration),
        completed_at=now if status in (JobStatus.COMPLETED, JobStatus.FAILED) else None,
        assigned_gpus=gpu_ids or [],
        error_message=error_message,
    )


class TestJobAnalytics:
    def test_avg_duration_by_tag(self):
        jobs = {
            "a": _make_job("a", tags=["train"], duration=100),
            "b": _make_job("b", tags=["train"], duration=200),
            "c": _make_job("c", tags=["render"], duration=50),
        }
        analytics = JobAnalytics(jobs)
        result = analytics.avg_duration_by_tag()
        assert result["train"]["avg_seconds"] == 150.0
        assert result["train"]["count"] == 2
        assert result["render"]["count"] == 1

    def test_failure_analysis(self):
        jobs = {
            "a": _make_job("a", status=JobStatus.COMPLETED),
            "b": _make_job("b", status=JobStatus.FAILED, error_message="Exit code 1 after 3 attempts"),
            "c": _make_job("c", status=JobStatus.FAILED, error_message="Timed out"),
            "d": _make_job("d", status=JobStatus.COMPLETED),
        }
        analytics = JobAnalytics(jobs)
        result = analytics.failure_analysis()
        assert result["total_terminal"] == 4
        assert result["completed"] == 2
        assert result["failed"] == 2
        assert result["overall_failure_rate_pct"] == 50.0
        assert "Timeout" in result["failure_reasons"]

    def test_priority_breakdown(self):
        jobs = {
            "a": _make_job("a", priority=JobPriority.HIGH),
            "b": _make_job("b", priority=JobPriority.HIGH),
            "c": _make_job("c", priority=JobPriority.LOW),
        }
        analytics = JobAnalytics(jobs)
        result = analytics.priority_breakdown()
        assert result["high"] == 2
        assert result["low"] == 1

    def test_gpu_utilization_stats(self):
        jobs = {
            "a": _make_job("a", duration=3600, gpu_ids=[0]),
            "b": _make_job("b", duration=7200, gpu_ids=[0, 1]),
        }
        analytics = JobAnalytics(jobs)
        result = analytics.gpu_utilization_stats()
        assert result["gpus"]["0"]["job_count"] == 2
        assert result["gpus"]["1"]["job_count"] == 1

    def test_empty_jobs(self):
        analytics = JobAnalytics({})
        result = analytics.summary()
        assert result["total_jobs"] == 0
        assert result["failure_analysis"]["total_terminal"] == 0
