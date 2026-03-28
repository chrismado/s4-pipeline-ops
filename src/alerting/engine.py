"""
Alert engine — monitors metrics, fires alerts, dispatches to backends.

Checks GPU metrics against configured thresholds and generates alerts.
Supports multiple dispatch backends: logging, Slack webhooks, email, generic webhooks.
Alert cooldown prevents spamming repeated alerts for the same condition.
"""

import uuid
from datetime import UTC, datetime
from typing import Optional

import httpx
from loguru import logger

from config.settings import settings
from src.models.schemas import (
    Alert,
    AlertSeverity,
    AlertType,
    Job,
    JobStatus,
    SystemMetrics,
)


class AlertEngine:
    """
    Evaluates metrics against thresholds and dispatches alerts.

    Maintains a cooldown map to prevent alert fatigue — the same alert type
    for the same GPU won't fire again within alert_cooldown_seconds.
    """

    def __init__(self):
        self.history: list[Alert] = []
        self._cooldowns: dict[str, datetime] = {}

    def check_gpu_metrics(self, metrics: SystemMetrics) -> list[Alert]:
        """Evaluate system metrics and generate alerts for threshold breaches."""
        alerts = []

        for gpu in metrics.gpus:
            # Temperature checks
            if gpu.temperature_c >= settings.gpu_temp_critical:
                alert = self._create_alert(
                    AlertType.GPU_TEMP,
                    AlertSeverity.CRITICAL,
                    f"GPU {gpu.gpu_index} critical temperature: {gpu.temperature_c:.0f}°C",
                    f"GPU {gpu.name} (index {gpu.gpu_index}) reached {gpu.temperature_c:.0f}°C, "
                    f"exceeding critical threshold of {settings.gpu_temp_critical}°C. "
                    f"Risk of thermal throttling or hardware damage.",
                    gpu_index=gpu.gpu_index,
                    metric_value=gpu.temperature_c,
                    threshold=settings.gpu_temp_critical,
                )
                if alert:
                    alerts.append(alert)
            elif gpu.temperature_c >= settings.gpu_temp_warning:
                alert = self._create_alert(
                    AlertType.GPU_TEMP,
                    AlertSeverity.WARNING,
                    f"GPU {gpu.gpu_index} high temperature: {gpu.temperature_c:.0f}°C",
                    f"GPU {gpu.name} (index {gpu.gpu_index}) at {gpu.temperature_c:.0f}°C, "
                    f"above warning threshold of {settings.gpu_temp_warning}°C.",
                    gpu_index=gpu.gpu_index,
                    metric_value=gpu.temperature_c,
                    threshold=settings.gpu_temp_warning,
                )
                if alert:
                    alerts.append(alert)

            # VRAM checks
            if gpu.vram_fraction >= settings.vram_usage_critical:
                alert = self._create_alert(
                    AlertType.GPU_MEMORY,
                    AlertSeverity.CRITICAL,
                    f"GPU {gpu.gpu_index} VRAM critical: {gpu.vram_fraction * 100:.0f}%",
                    f"GPU {gpu.name} VRAM at {gpu.memory_used_mb:.0f}MB / {gpu.memory_total_mb:.0f}MB "
                    f"({gpu.vram_fraction * 100:.1f}%). OOM crash likely.",
                    gpu_index=gpu.gpu_index,
                    metric_value=gpu.vram_fraction,
                    threshold=settings.vram_usage_critical,
                )
                if alert:
                    alerts.append(alert)
            elif gpu.vram_fraction >= settings.vram_usage_warning:
                alert = self._create_alert(
                    AlertType.GPU_MEMORY,
                    AlertSeverity.WARNING,
                    f"GPU {gpu.gpu_index} VRAM high: {gpu.vram_fraction * 100:.0f}%",
                    f"GPU {gpu.name} VRAM at {gpu.memory_used_mb:.0f}MB / {gpu.memory_total_mb:.0f}MB.",
                    gpu_index=gpu.gpu_index,
                    metric_value=gpu.vram_fraction,
                    threshold=settings.vram_usage_warning,
                )
                if alert:
                    alerts.append(alert)

        # Disk space check
        if metrics.disk_total_gb > 0:
            disk_pct = metrics.disk_used_gb / metrics.disk_total_gb
            if disk_pct > 0.90:
                alert = self._create_alert(
                    AlertType.DISK_SPACE,
                    AlertSeverity.CRITICAL if disk_pct > 0.95 else AlertSeverity.WARNING,
                    f"Disk space {'critical' if disk_pct > 0.95 else 'low'}: {disk_pct * 100:.0f}% used",
                    f"{metrics.disk_used_gb:.1f}GB / {metrics.disk_total_gb:.1f}GB used.",
                    metric_value=disk_pct,
                    threshold=0.90,
                )
                if alert:
                    alerts.append(alert)

        # Dispatch all new alerts
        for alert in alerts:
            self._dispatch(alert)

        return alerts

    def check_job_event(self, job: Job) -> Optional[Alert]:
        """Generate alerts for job state changes."""
        alert = None

        if job.status == JobStatus.FAILED:
            alert = Alert(
                id=str(uuid.uuid4())[:8],
                type=AlertType.JOB_FAILED,
                severity=AlertSeverity.CRITICAL if job.attempt >= job.max_attempts else AlertSeverity.WARNING,
                title=f"Job failed: {job.name}",
                message=(
                    f"Job {job.id} ({job.name}) failed with exit code {job.exit_code}. "
                    f"Attempt {job.attempt}/{job.max_attempts}. "
                    f"Error: {job.error_message or 'Unknown'}"
                ),
                job_id=job.id,
            )
        elif job.status == JobStatus.COMPLETED:
            alert = Alert(
                id=str(uuid.uuid4())[:8],
                type=AlertType.JOB_COMPLETED,
                severity=AlertSeverity.INFO,
                title=f"Job completed: {job.name}",
                message=(
                    f"Job {job.id} ({job.name}) completed in "
                    f"{job.duration_seconds:.0f}s on GPU(s) {job.assigned_gpus}."
                    if job.duration_seconds is not None
                    else f"Job {job.id} ({job.name}) completed on GPU(s) {job.assigned_gpus}."
                ),
                job_id=job.id,
            )

        if alert:
            self.history.append(alert)
            self._dispatch(alert)

        return alert

    def get_recent_alerts(self, limit: int = 50) -> list[Alert]:
        """Return recent alerts, newest first."""
        return sorted(self.history, key=lambda a: a.timestamp, reverse=True)[:limit]

    def _create_alert(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        title: str,
        message: str,
        gpu_index: Optional[int] = None,
        metric_value: Optional[float] = None,
        threshold: Optional[float] = None,
        job_id: Optional[str] = None,
    ) -> Optional[Alert]:
        """Create an alert if not in cooldown."""
        cooldown_key = f"{alert_type.value}:{gpu_index if gpu_index is not None else 'system'}"

        now = datetime.now(UTC)
        last_fired = self._cooldowns.get(cooldown_key)
        if last_fired and (now - last_fired).total_seconds() < settings.alert_cooldown_seconds:
            return None

        self._cooldowns[cooldown_key] = now

        alert = Alert(
            id=str(uuid.uuid4())[:8],
            type=alert_type,
            severity=severity,
            title=title,
            message=message,
            gpu_index=gpu_index,
            metric_value=metric_value,
            threshold=threshold,
            job_id=job_id,
        )
        self.history.append(alert)
        return alert

    def _dispatch(self, alert: Alert):
        """Send alert to all configured backends."""
        for backend in settings.alert_backends:
            try:
                if backend == "log":
                    self._dispatch_log(alert)
                elif backend == "slack":
                    self._dispatch_slack(alert)
                elif backend == "webhook":
                    self._dispatch_webhook(alert)
                elif backend == "email":
                    self._dispatch_email(alert)
            except Exception as e:
                logger.error(f"Alert dispatch to {backend} failed: {e}")

    def _dispatch_log(self, alert: Alert):
        """Log alert with appropriate level."""
        msg = f"[{alert.severity.value.upper()}] {alert.title}: {alert.message}"
        if alert.severity == AlertSeverity.CRITICAL:
            logger.critical(msg)
        elif alert.severity == AlertSeverity.WARNING:
            logger.warning(msg)
        else:
            logger.info(msg)

    def _dispatch_slack(self, alert: Alert):
        """Send alert to Slack via incoming webhook."""
        if not settings.slack_webhook_url:
            return

        emoji = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}
        payload = {
            "text": f"{emoji.get(alert.severity.value, '')} *{alert.title}*\n{alert.message}",
        }
        httpx.post(settings.slack_webhook_url, json=payload, timeout=10)

    def _dispatch_webhook(self, alert: Alert):
        """Send alert to a generic webhook endpoint."""
        if not settings.webhook_url:
            return
        httpx.post(
            settings.webhook_url,
            json=alert.model_dump(mode="json"),
            timeout=10,
        )

    def _dispatch_email(self, alert: Alert):
        """Send alert via email (placeholder — implement with smtplib)."""
        if not settings.email_smtp_host or not settings.email_to:
            return
        # Email dispatch would go here — smtplib with TLS
        logger.info(f"Email alert would be sent to {settings.email_to}: {alert.title}")
