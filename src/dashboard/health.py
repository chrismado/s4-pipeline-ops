"""
Pipeline health monitoring — tracks stage status, throughput, error rates.

Each pipeline stage (ingest, preprocess, train, render, export) is monitored
independently. The overall pipeline status is the worst of any individual stage.
"""

from datetime import UTC, datetime, timedelta
from typing import Optional


from config.settings import settings
from src.models.schemas import (
    Job,
    JobStatus,
    PipelineHealth,
    PipelineStage,
    StageStatus,
)


class HealthMonitor:
    """
    Tracks pipeline health across all stages.

    Updates from completed jobs — maps job tags to pipeline stages,
    computes error rates and throughput.
    """

    def __init__(self):
        self.stages: dict[str, PipelineStage] = {
            name: PipelineStage(name=name)
            for name in settings.pipeline_stages
        }
        self._job_history: list[dict] = []

    def record_job(self, job: Job):
        """Record a completed job's result for health tracking."""
        if not job.is_terminal:
            return

        # Map job to stage via tags
        stage_name = self._job_to_stage(job)
        if not stage_name or stage_name not in self.stages:
            return

        stage = self.stages[stage_name]
        now = datetime.now(UTC)

        self._job_history.append({
            "stage": stage_name,
            "status": job.status.value,
            "duration": job.duration_seconds,
            "timestamp": now,
        })

        if job.status == JobStatus.COMPLETED:
            stage.last_success = now
        elif job.status == JobStatus.FAILED:
            stage.last_failure = now

        # Recalculate stage metrics from recent history
        self._recalculate_stage(stage_name)

    def get_health(self) -> PipelineHealth:
        """Get current pipeline health summary."""
        stages = list(self.stages.values())

        # Determine overall status
        statuses = [s.status for s in stages if s.status != StageStatus.UNKNOWN]
        if StageStatus.DOWN in statuses:
            overall = StageStatus.DOWN
        elif StageStatus.DEGRADED in statuses:
            overall = StageStatus.DEGRADED
        elif statuses:
            overall = StageStatus.HEALTHY
        else:
            overall = StageStatus.UNKNOWN

        # 24h stats
        cutoff = datetime.now(UTC) - timedelta(hours=24)
        recent = [h for h in self._job_history if h["timestamp"] > cutoff]
        total_24h = len(recent)
        failed_24h = sum(1 for h in recent if h["status"] == "failed")

        return PipelineHealth(
            stages=stages,
            overall_status=overall,
            total_jobs_24h=total_24h,
            failed_jobs_24h=failed_24h,
        )

    def _job_to_stage(self, job: Job) -> Optional[str]:
        """Map a job to a pipeline stage based on tags or name."""
        for tag in job.config.tags:
            if tag in self.stages:
                return tag

        # Fallback: match by job name keywords
        name_lower = job.name.lower()
        for stage_name in self.stages:
            if stage_name in name_lower:
                return stage_name

        return None

    def _recalculate_stage(self, stage_name: str):
        """Recalculate metrics for a stage from job history."""
        stage = self.stages[stage_name]
        cutoff = datetime.now(UTC) - timedelta(hours=24)
        recent = [
            h for h in self._job_history
            if h["stage"] == stage_name and h["timestamp"] > cutoff
        ]

        if not recent:
            stage.status = StageStatus.UNKNOWN
            return

        total = len(recent)
        failed = sum(1 for h in recent if h["status"] == "failed")
        durations = [h["duration"] for h in recent if h["duration"] is not None]

        stage.error_rate_pct = (failed / total) * 100 if total > 0 else 0
        stage.avg_duration_seconds = sum(durations) / len(durations) if durations else 0
        stage.throughput_per_hour = total / 24.0

        # Determine status
        if stage.error_rate_pct > 50:
            stage.status = StageStatus.DOWN
        elif stage.error_rate_pct > 10:
            stage.status = StageStatus.DEGRADED
        else:
            stage.status = StageStatus.HEALTHY
