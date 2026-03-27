"""
Historical job analytics — aggregates job data for insights.

Computes:
  - Average duration by tag/stage
  - Failure rates and patterns
  - GPU utilization trends
  - Throughput over time
"""

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional

from src.models.schemas import Job, JobStatus


class JobAnalytics:
    """
    Analyzes historical job data from the JobManager.

    Provides aggregate statistics for understanding pipeline
    performance, failure patterns, and resource utilization.
    """

    def __init__(self, jobs: dict[str, Job]):
        self._jobs = jobs

    def summary(self, hours: int = 168) -> dict:
        """Full analytics summary for the given time window."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        recent = [j for j in self._jobs.values() if j.created_at > cutoff]

        return {
            "time_window_hours": hours,
            "total_jobs": len(recent),
            "duration_by_tag": self.avg_duration_by_tag(recent),
            "failure_analysis": self.failure_analysis(recent),
            "throughput": self.throughput_by_hour(recent),
            "gpu_utilization": self.gpu_utilization_stats(recent),
            "priority_breakdown": self.priority_breakdown(recent),
        }

    def avg_duration_by_tag(self, jobs: Optional[list[Job]] = None) -> dict[str, dict]:
        """Average job duration grouped by tag."""
        jobs = jobs if jobs is not None else list(self._jobs.values())
        tag_durations: dict[str, list[float]] = defaultdict(list)

        for job in jobs:
            if job.status != JobStatus.COMPLETED or job.duration_seconds is None:
                continue
            tags = job.config.tags or ["untagged"]
            for tag in tags:
                tag_durations[tag].append(job.duration_seconds)

        return {
            tag: {
                "avg_seconds": round(sum(durs) / len(durs), 1),
                "min_seconds": round(min(durs), 1),
                "max_seconds": round(max(durs), 1),
                "count": len(durs),
            }
            for tag, durs in tag_durations.items()
        }

    def failure_analysis(self, jobs: Optional[list[Job]] = None) -> dict:
        """Analyze failure patterns."""
        jobs = jobs if jobs is not None else list(self._jobs.values())
        terminal = [j for j in jobs if j.is_terminal]
        failed = [j for j in terminal if j.status == JobStatus.FAILED]
        completed = [j for j in terminal if j.status == JobStatus.COMPLETED]

        # Failure reasons
        reasons: dict[str, int] = defaultdict(int)
        for j in failed:
            reason = j.error_message or "Unknown"
            # Normalize common reasons
            if "timed out" in reason.lower():
                reasons["Timeout"] += 1
            elif "exit code" in reason.lower():
                reasons[f"Exit code failure"] += 1
            elif "unclean shutdown" in reason.lower():
                reasons["Unclean shutdown"] += 1
            else:
                reasons[reason[:50]] += 1

        # Failure rate by tag
        tag_stats: dict[str, dict] = defaultdict(lambda: {"total": 0, "failed": 0})
        for j in terminal:
            tags = j.config.tags or ["untagged"]
            for tag in tags:
                tag_stats[tag]["total"] += 1
                if j.status == JobStatus.FAILED:
                    tag_stats[tag]["failed"] += 1

        tag_failure_rates = {
            tag: {
                "total": s["total"],
                "failed": s["failed"],
                "rate_pct": round(s["failed"] / s["total"] * 100, 1) if s["total"] > 0 else 0,
            }
            for tag, s in tag_stats.items()
        }

        return {
            "total_terminal": len(terminal),
            "completed": len(completed),
            "failed": len(failed),
            "overall_failure_rate_pct": round(len(failed) / len(terminal) * 100, 1) if terminal else 0,
            "failure_reasons": dict(reasons),
            "failure_rate_by_tag": tag_failure_rates,
        }

    def throughput_by_hour(self, jobs: Optional[list[Job]] = None, hours: int = 24) -> list[dict]:
        """Job completions per hour over the given window."""
        jobs = jobs if jobs is not None else list(self._jobs.values())
        now = datetime.utcnow()
        buckets: list[dict] = []

        for h in range(hours, 0, -1):
            start = now - timedelta(hours=h)
            end = now - timedelta(hours=h - 1)
            completed_in_hour = [
                j for j in jobs
                if j.completed_at and start <= j.completed_at < end
                and j.status == JobStatus.COMPLETED
            ]
            failed_in_hour = [
                j for j in jobs
                if j.completed_at and start <= j.completed_at < end
                and j.status == JobStatus.FAILED
            ]
            buckets.append({
                "hour": start.strftime("%Y-%m-%d %H:00"),
                "completed": len(completed_in_hour),
                "failed": len(failed_in_hour),
            })

        return buckets

    def gpu_utilization_stats(self, jobs: Optional[list[Job]] = None) -> dict:
        """GPU utilization statistics from job assignments."""
        jobs = jobs if jobs is not None else list(self._jobs.values())
        gpu_jobs: dict[int, int] = defaultdict(int)
        gpu_hours: dict[int, float] = defaultdict(float)

        for job in jobs:
            if job.duration_seconds is None:
                continue
            for gpu_id in job.assigned_gpus:
                gpu_jobs[gpu_id] += 1
                gpu_hours[gpu_id] += job.duration_seconds / 3600

        return {
            "gpus": {
                str(gpu_id): {
                    "job_count": gpu_jobs[gpu_id],
                    "total_hours": round(gpu_hours[gpu_id], 2),
                }
                for gpu_id in sorted(set(gpu_jobs) | set(gpu_hours))
            }
        }

    def priority_breakdown(self, jobs: Optional[list[Job]] = None) -> dict[str, int]:
        """Count of jobs by priority level."""
        jobs = jobs if jobs is not None else list(self._jobs.values())
        counts: dict[str, int] = defaultdict(int)
        for job in jobs:
            counts[job.priority.value] += 1
        return dict(counts)
