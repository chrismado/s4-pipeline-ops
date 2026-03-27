"""
Job scheduler — queue, dispatch, and track render/training jobs.

Manages job lifecycle: submit → queue → assign GPUs → run → complete/fail/retry.
Jobs are dispatched based on priority and GPU availability. Failed jobs retry
with exponential backoff up to max_attempts.
"""

import json
import os
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

from config.settings import settings
from src.collectors.gpu import collect_gpu_metrics
from src.models.schemas import Job, JobConfig, JobPriority, JobStatus
from src.scheduler.progress import parse_progress


class JobManager:
    """
    Manages the lifecycle of pipeline jobs.

    Jobs are persisted to a JSON file so state survives restarts.
    GPU assignment is based on current utilization — idle GPUs
    are assigned to the highest-priority queued job.
    """

    def __init__(self):
        self.jobs: dict[str, Job] = {}
        self._processes: dict[str, subprocess.Popen] = {}
        self._load_state()

    def submit(
        self,
        name: str,
        config: JobConfig,
        priority: JobPriority = JobPriority.NORMAL,
    ) -> Job:
        """Submit a new job to the queue."""
        job = Job(
            id=str(uuid.uuid4())[:8],
            name=name,
            config=config,
            priority=priority,
            max_attempts=settings.retry_max_attempts,
        )
        self.jobs[job.id] = job
        self._save_state()
        logger.info(f"Job submitted: {job.id} ({job.name}) priority={priority.value}")
        return job

    def dispatch(self) -> list[Job]:
        """Check queue and dispatch jobs to available GPUs."""
        dispatched = []

        # Get available GPUs (idle ones not assigned to running jobs)
        assigned_gpus = set()
        for job in self.jobs.values():
            if job.status == JobStatus.RUNNING:
                assigned_gpus.update(job.assigned_gpus)

        gpu_metrics = collect_gpu_metrics()
        available_gpus = [
            g.gpu_index for g in gpu_metrics
            if g.gpu_index not in assigned_gpus and g.is_idle
        ]

        # Count running jobs
        running_count = sum(
            1 for j in self.jobs.values() if j.status == JobStatus.RUNNING
        )

        # Get queued jobs sorted by priority
        queued = sorted(
            [j for j in self.jobs.values() if j.status in (JobStatus.PENDING, JobStatus.QUEUED, JobStatus.RETRYING)],
            key=lambda j: list(JobPriority).index(j.priority),
            reverse=True,  # CRITICAL first
        )

        for job in queued:
            if running_count >= settings.max_concurrent_jobs:
                break

            needed = job.config.gpu_count
            if job.config.gpu_ids:
                # Specific GPUs requested
                if all(g in available_gpus for g in job.config.gpu_ids):
                    gpus_to_assign = job.config.gpu_ids
                else:
                    continue
            elif len(available_gpus) >= needed:
                gpus_to_assign = available_gpus[:needed]
            else:
                job.status = JobStatus.QUEUED
                continue

            # Assign and start
            job.assigned_gpus = gpus_to_assign
            for g in gpus_to_assign:
                if g in available_gpus:
                    available_gpus.remove(g)

            self._start_job(job)
            dispatched.append(job)
            running_count += 1

        if dispatched:
            self._save_state()

        return dispatched

    def check_running(self) -> list[Job]:
        """Check status of running jobs, handle completions and failures."""
        changed = []

        for job_id, proc in list(self._processes.items()):
            job = self.jobs.get(job_id)
            if not job:
                continue

            # Read available stdout and parse progress
            self._read_output(job, proc)

            retcode = proc.poll()
            if retcode is None:
                # Still running — check timeout
                if job.duration_seconds and job.duration_seconds > (
                    (job.config.timeout_hours or settings.job_timeout_hours) * 3600
                ):
                    logger.warning(f"Job {job_id} timed out after {job.duration_seconds:.0f}s")
                    proc.kill()
                    job.status = JobStatus.FAILED
                    job.error_message = "Timed out"
                    job.completed_at = datetime.utcnow()
                    job.exit_code = -9
                    del self._processes[job_id]
                    changed.append(job)
                continue

            # Process finished
            job.exit_code = retcode
            job.completed_at = datetime.utcnow()

            if retcode == 0:
                job.status = JobStatus.COMPLETED
                logger.info(f"Job {job_id} completed in {job.duration_seconds:.1f}s")
            else:
                if job.attempt < job.max_attempts:
                    job.status = JobStatus.RETRYING
                    job.attempt += 1
                    logger.warning(
                        f"Job {job_id} failed (exit {retcode}), "
                        f"retrying {job.attempt}/{job.max_attempts}"
                    )
                else:
                    job.status = JobStatus.FAILED
                    job.error_message = f"Exit code {retcode} after {job.attempt} attempts"
                    logger.error(f"Job {job_id} permanently failed: {job.error_message}")

            del self._processes[job_id]
            changed.append(job)

        if changed:
            self._save_state()

        return changed

    def cancel(self, job_id: str) -> Optional[Job]:
        """Cancel a pending or running job."""
        job = self.jobs.get(job_id)
        if not job or job.is_terminal:
            return None

        if job_id in self._processes:
            self._processes[job_id].kill()
            del self._processes[job_id]

        job.status = JobStatus.CANCELLED
        job.completed_at = datetime.utcnow()
        self._save_state()
        logger.info(f"Job {job_id} cancelled")
        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

    def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        limit: int = 50,
    ) -> list[Job]:
        """List jobs, optionally filtered by status."""
        jobs = list(self.jobs.values())
        if status:
            jobs = [j for j in jobs if j.status == status]
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)[:limit]

    def get_queue_stats(self) -> dict:
        """Summary stats for the job queue."""
        by_status = {}
        for job in self.jobs.values():
            by_status[job.status.value] = by_status.get(job.status.value, 0) + 1

        return {
            "total": len(self.jobs),
            "by_status": by_status,
            "running": by_status.get("running", 0),
            "queued": by_status.get("queued", 0) + by_status.get("pending", 0),
        }

    def _start_job(self, job: Job):
        """Launch a job subprocess."""
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in job.assigned_gpus)
        env.update(job.config.env_vars)

        logger.info(
            f"Starting job {job.id} ({job.name}) on GPU(s) {job.assigned_gpus}: "
            f"{job.config.command}"
        )

        try:
            proc = subprocess.Popen(
                job.config.command,
                shell=True,
                cwd=job.config.working_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            self._processes[job.id] = proc
            job.status = JobStatus.RUNNING
            job.started_at = datetime.utcnow()
        except Exception as e:
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
            logger.error(f"Failed to start job {job.id}: {e}")

    def _read_output(self, job: Job, proc: subprocess.Popen) -> None:
        """Read available stdout from a process and parse progress."""
        if not proc.stdout:
            return

        # Non-blocking read of available output
        try:
            if sys.platform == "win32":
                # Windows: peek at pipe to check if data is available
                import msvcrt
                import ctypes
                from ctypes import wintypes
                kernel32 = ctypes.windll.kernel32
                handle = msvcrt.get_osfhandle(proc.stdout.fileno())
                avail = wintypes.DWORD()
                if kernel32.PeekNamedPipe(handle, None, 0, None, ctypes.byref(avail), None) and avail.value > 0:
                    data = proc.stdout.read(avail.value)
                else:
                    data = b""
            else:
                # Unix: use select for non-blocking read
                ready, _, _ = select.select([proc.stdout], [], [], 0)
                data = proc.stdout.read(4096) if ready else b""
        except Exception:
            data = b""

        if not data:
            return

        text = data.decode("utf-8", errors="replace")
        job.output_log = (job.output_log + text)[-10000:]  # Keep last 10KB

        # Parse progress from each line
        for line in text.splitlines():
            pct = parse_progress(line)
            if pct is not None:
                job.progress_pct = round(pct, 1)

    def _save_state(self):
        """Persist job state to disk."""
        path = Path(settings.job_db_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {jid: job.model_dump(mode="json") for jid, job in self.jobs.items()}
        path.write_text(json.dumps(data, indent=2, default=str))

    def _load_state(self):
        """Load persisted job state."""
        path = Path(settings.job_db_path)
        if not path.exists():
            return

        try:
            data = json.loads(path.read_text())
            for jid, jdata in data.items():
                job = Job(**jdata)
                # Mark previously running jobs as failed (unclean shutdown)
                if job.status == JobStatus.RUNNING:
                    job.status = JobStatus.FAILED
                    job.error_message = "Process died (unclean shutdown)"
                    job.completed_at = datetime.utcnow()
                self.jobs[jid] = job
            logger.info(f"Loaded {len(self.jobs)} jobs from state file")
        except Exception as e:
            logger.error(f"Failed to load job state: {e}")
