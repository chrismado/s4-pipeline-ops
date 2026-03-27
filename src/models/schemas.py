"""
Data models for pipeline ops — GPU metrics, jobs, alerts, pipeline health.

Pydantic models provide validation, serialization, and clean API contracts.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# GPU Metrics
# ──────────────────────────────────────────────

class GPUMetrics(BaseModel):
    """Snapshot of a single GPU's state."""
    gpu_index: int
    name: str = ""
    temperature_c: float
    utilization_pct: float = Field(ge=0, le=100)
    memory_used_mb: float
    memory_total_mb: float
    memory_utilization_pct: float = Field(ge=0, le=100)
    power_draw_w: float = 0.0
    power_limit_w: float = 0.0
    fan_speed_pct: float = 0.0
    clock_sm_mhz: int = 0
    clock_mem_mhz: int = 0
    pcie_tx_kbps: int = 0
    pcie_rx_kbps: int = 0
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @property
    def vram_fraction(self) -> float:
        return self.memory_used_mb / max(self.memory_total_mb, 1.0)

    @property
    def is_idle(self) -> bool:
        from config.settings import settings
        return self.utilization_pct < settings.gpu_util_idle_threshold


class SystemMetrics(BaseModel):
    """Full system snapshot — all GPUs + host stats."""
    gpus: list[GPUMetrics]
    cpu_pct: float = 0.0
    ram_used_gb: float = 0.0
    ram_total_gb: float = 0.0
    disk_used_gb: float = 0.0
    disk_total_gb: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @property
    def gpu_count(self) -> int:
        return len(self.gpus)

    @property
    def total_vram_used_gb(self) -> float:
        return sum(g.memory_used_mb for g in self.gpus) / 1024

    @property
    def total_vram_gb(self) -> float:
        return sum(g.memory_total_mb for g in self.gpus) / 1024


# ──────────────────────────────────────────────
# Job Scheduling
# ──────────────────────────────────────────────

class JobStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class JobPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class JobConfig(BaseModel):
    """Configuration for a render/training job."""
    command: str = Field(description="Shell command to execute")
    working_dir: str = Field(default=".", description="Working directory for the job")
    gpu_ids: list[int] = Field(default=[], description="Specific GPUs to use (empty = any available)")
    gpu_count: int = Field(default=1, description="Number of GPUs needed")
    env_vars: dict[str, str] = Field(default={}, description="Extra environment variables")
    timeout_hours: Optional[float] = Field(default=None, description="Override default timeout")
    tags: list[str] = Field(default=[], description="Tags for filtering/grouping")


class Job(BaseModel):
    """A scheduled pipeline job (render, training, export, etc.)."""
    id: str
    name: str
    config: JobConfig
    status: JobStatus = JobStatus.PENDING
    priority: JobPriority = JobPriority.NORMAL
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    assigned_gpus: list[int] = Field(default=[])
    attempt: int = 0
    max_attempts: int = 3
    exit_code: Optional[int] = None
    error_message: Optional[str] = None
    progress_pct: Optional[float] = None
    output_log: str = ""

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.started_at:
            end = self.completed_at or datetime.utcnow()
            return (end - self.started_at).total_seconds()
        return None

    @property
    def is_terminal(self) -> bool:
        return self.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)


# ──────────────────────────────────────────────
# Alerting
# ──────────────────────────────────────────────

class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(str, Enum):
    GPU_TEMP = "gpu_temperature"
    GPU_MEMORY = "gpu_memory"
    GPU_IDLE = "gpu_idle"
    GPU_ERROR = "gpu_error"
    JOB_FAILED = "job_failed"
    JOB_TIMEOUT = "job_timeout"
    JOB_COMPLETED = "job_completed"
    PIPELINE_STALL = "pipeline_stall"
    DISK_SPACE = "disk_space"
    SYSTEM_ERROR = "system_error"


class Alert(BaseModel):
    """An alert event triggered by a threshold breach or job event."""
    id: str = Field(default="")
    type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    gpu_index: Optional[int] = None
    job_id: Optional[str] = None
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    acknowledged: bool = False


# ──────────────────────────────────────────────
# Pipeline Health
# ──────────────────────────────────────────────

class StageStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"
    UNKNOWN = "unknown"


class PipelineStage(BaseModel):
    """Health status of a single pipeline stage."""
    name: str
    status: StageStatus = StageStatus.UNKNOWN
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    error_rate_pct: float = 0.0
    avg_duration_seconds: float = 0.0
    throughput_per_hour: float = 0.0
    active_jobs: int = 0


class PipelineHealth(BaseModel):
    """Overall pipeline health summary."""
    stages: list[PipelineStage]
    overall_status: StageStatus = StageStatus.UNKNOWN
    total_jobs_24h: int = 0
    failed_jobs_24h: int = 0
    avg_queue_wait_seconds: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @property
    def failure_rate_24h(self) -> float:
        if self.total_jobs_24h == 0:
            return 0.0
        return (self.failed_jobs_24h / self.total_jobs_24h) * 100
