"""
Pipeline Ops configuration — Pydantic Settings with env var overrides.

All settings can be overridden via environment variables prefixed with S4OPS_.
Example: S4OPS_POLL_INTERVAL=10 overrides poll_interval.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="S4OPS_")

    # --- GPU Monitoring ---
    poll_interval: int = Field(5, description="GPU metrics polling interval in seconds")
    gpu_temp_warning: float = Field(80.0, description="GPU temperature warning threshold (°C)")
    gpu_temp_critical: float = Field(90.0, description="GPU temperature critical threshold (°C)")
    gpu_util_idle_threshold: float = Field(5.0, description="GPU utilization below this = idle (%)")
    vram_usage_warning: float = Field(0.85, description="VRAM usage warning threshold (fraction)")
    vram_usage_critical: float = Field(0.95, description="VRAM usage critical threshold (fraction)")

    # --- Job Scheduler ---
    max_concurrent_jobs: int = Field(4, description="Max simultaneous render/training jobs")
    job_timeout_hours: float = Field(24.0, description="Kill jobs running longer than this")
    retry_max_attempts: int = Field(3, description="Max retry attempts for failed jobs")
    retry_backoff_seconds: float = Field(60.0, description="Base backoff between retries")
    job_db_path: str = Field("data/jobs.json", description="Job state persistence file")

    # --- Alerting ---
    alert_cooldown_seconds: int = Field(300, description="Min seconds between repeated alerts")
    alert_backends: list[str] = Field(
        default=["log"],
        description="Alert backends: log, slack, email, webhook",
    )
    slack_webhook_url: str = Field("", description="Slack incoming webhook URL")
    email_smtp_host: str = Field("", description="SMTP host for email alerts")
    email_smtp_port: int = Field(587, description="SMTP port")
    email_from: str = Field("", description="Alert sender email")
    email_to: list[str] = Field(default=[], description="Alert recipient emails")
    webhook_url: str = Field("", description="Generic webhook URL for alerts")

    # --- Dashboard API ---
    api_host: str = Field("0.0.0.0", description="API bind host")
    api_port: int = Field(8100, description="API port")

    # --- Pipeline Health ---
    health_check_interval: int = Field(60, description="Pipeline health check interval (seconds)")
    pipeline_stages: list[str] = Field(
        default=["ingest", "preprocess", "train", "render", "export"],
        description="Named pipeline stages to monitor",
    )

    # --- Metrics Retention ---
    metrics_retention_hours: int = Field(168, description="Keep metrics for this many hours (7 days)")
    metrics_db_path: str = Field("data/metrics.json", description="Metrics persistence file")


settings = Settings()
