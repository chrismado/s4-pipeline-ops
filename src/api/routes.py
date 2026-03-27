"""
API routes for pipeline ops dashboard.

Endpoints:
    GET  /gpu/metrics          — Current GPU metrics snapshot
    GET  /gpu/history          — GPU metrics history (time series)
    POST /jobs                 — Submit a new job
    GET  /jobs                 — List jobs (filter by status)
    GET  /jobs/{id}            — Get job details
    DELETE /jobs/{id}          — Cancel a job
    GET  /health               — Pipeline health summary
    GET  /alerts               — Recent alerts
    GET  /status               — Quick overall status check
"""

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel

from src.collectors.gpu import collect_system_metrics
from src.models.schemas import JobConfig, JobPriority, JobStatus

router = APIRouter()

_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "dashboard" / "templates"

# ── Singletons (initialized on first request) ──

_job_manager = None
_alert_engine = None
_health_monitor = None
_metrics_store = None


def _get_job_manager():
    global _job_manager
    if _job_manager is None:
        from src.scheduler.manager import JobManager
        _job_manager = JobManager()
    return _job_manager


def _get_alert_engine():
    global _alert_engine
    if _alert_engine is None:
        from src.alerting.engine import AlertEngine
        _alert_engine = AlertEngine()
    return _alert_engine


def _get_health_monitor():
    global _health_monitor
    if _health_monitor is None:
        from src.dashboard.health import HealthMonitor
        _health_monitor = HealthMonitor()
    return _health_monitor


def _get_metrics_store():
    global _metrics_store
    if _metrics_store is None:
        from src.collectors.metrics_store import MetricsStore
        _metrics_store = MetricsStore()
    return _metrics_store


# ── Dashboard ──

@router.get("/", response_class=HTMLResponse)
def dashboard():
    """Live dashboard page."""
    html = (_TEMPLATE_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)


# ── GPU Metrics ──

@router.get("/gpu/metrics")
def get_gpu_metrics():
    """Current GPU and system metrics snapshot."""
    metrics = collect_system_metrics()

    # Record to time series store
    store = _get_metrics_store()
    store.record(metrics)

    # Check for alerts
    engine = _get_alert_engine()
    alerts = engine.check_gpu_metrics(metrics)

    return {
        "metrics": metrics.model_dump(mode="json"),
        "new_alerts": [a.model_dump(mode="json") for a in alerts],
    }


@router.get("/gpu/history")
def get_gpu_history(hours: float = 24, gpu_index: Optional[int] = None):
    """GPU metrics time series for charting."""
    store = _get_metrics_store()
    return {"history": store.get_history(hours=hours, gpu_index=gpu_index)}


# ── Job Management ──

class SubmitJobRequest(BaseModel):
    name: str
    command: str
    working_dir: str = "."
    gpu_ids: list[int] = []
    gpu_count: int = 1
    priority: str = "normal"
    tags: list[str] = []
    env_vars: dict[str, str] = {}
    timeout_hours: Optional[float] = None


@router.post("/jobs")
def submit_job(req: SubmitJobRequest):
    """Submit a new render/training job."""
    manager = _get_job_manager()

    config = JobConfig(
        command=req.command,
        working_dir=req.working_dir,
        gpu_ids=req.gpu_ids,
        gpu_count=req.gpu_count,
        env_vars=req.env_vars,
        timeout_hours=req.timeout_hours,
        tags=req.tags,
    )

    try:
        priority = JobPriority(req.priority)
    except ValueError:
        priority = JobPriority.NORMAL

    job = manager.submit(req.name, config, priority)
    return {"job": job.model_dump(mode="json")}


@router.get("/jobs")
def list_jobs(status: Optional[str] = None, limit: int = 50):
    """List jobs, optionally filtered by status."""
    manager = _get_job_manager()

    status_filter = None
    if status:
        try:
            status_filter = JobStatus(status)
        except ValueError:
            raise HTTPException(400, f"Invalid status: {status}")

    jobs = manager.list_jobs(status=status_filter, limit=limit)
    return {
        "jobs": [j.model_dump(mode="json") for j in jobs],
        "stats": manager.get_queue_stats(),
    }


@router.get("/jobs/{job_id}")
def get_job(job_id: str):
    """Get details of a specific job."""
    manager = _get_job_manager()
    job = manager.get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job not found: {job_id}")
    return {"job": job.model_dump(mode="json")}


@router.delete("/jobs/{job_id}")
def cancel_job(job_id: str):
    """Cancel a pending or running job."""
    manager = _get_job_manager()
    job = manager.cancel(job_id)
    if not job:
        raise HTTPException(404, f"Job not found or already complete: {job_id}")
    return {"job": job.model_dump(mode="json")}


# ── Pipeline Health ──

@router.get("/health")
def get_pipeline_health():
    """Pipeline health summary across all stages."""
    monitor = _get_health_monitor()
    health = monitor.get_health()
    return health.model_dump(mode="json")


# ── Alerts ──

@router.get("/alerts")
def get_alerts(limit: int = 50):
    """Recent alerts."""
    engine = _get_alert_engine()
    alerts = engine.get_recent_alerts(limit=limit)
    return {"alerts": [a.model_dump(mode="json") for a in alerts]}


# ── Quick Status ──

@router.get("/metrics", response_class=PlainTextResponse)
def prometheus_metrics():
    """Prometheus metrics endpoint."""
    from src.api.prometheus import generate_prometheus_metrics
    return PlainTextResponse(
        content=generate_prometheus_metrics(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


# ── Job Analytics ──

@router.get("/analytics")
def get_analytics(hours: int = 168):
    """Historical job analytics — durations, failure patterns, throughput."""
    from src.dashboard.analytics import JobAnalytics
    manager = _get_job_manager()
    analytics = JobAnalytics(manager.jobs)
    return analytics.summary(hours=hours)


# ── Multi-Node Cluster ──

_aggregator = None


def _get_aggregator():
    global _aggregator
    if _aggregator is None:
        from src.multinode.aggregator import NodeAggregator
        _aggregator = NodeAggregator()
    return _aggregator


@router.post("/nodes/register")
def register_node(url: str, node_id: str = ""):
    """Register a remote agent node."""
    agg = _get_aggregator()
    node = agg.add_node(url, node_id)
    return {"registered": node.url, "node_id": node.node_id}


@router.delete("/nodes/{node_url:path}")
def unregister_node(node_url: str):
    """Remove a registered node."""
    agg = _get_aggregator()
    if agg.remove_node(node_url):
        return {"removed": node_url}
    raise HTTPException(404, f"Node not found: {node_url}")


@router.get("/nodes")
def list_nodes():
    """List all registered agent nodes and their status."""
    agg = _get_aggregator()
    return {"nodes": agg.get_node_status()}


@router.get("/cluster/metrics")
def cluster_metrics():
    """Aggregated GPU metrics from all nodes in the cluster."""
    agg = _get_aggregator()
    return agg.collect_all()


# ── Quick Status ──

@router.get("/status")
def get_status():
    """Quick status check — GPU count, running jobs, pipeline health."""
    metrics = collect_system_metrics()
    manager = _get_job_manager()
    monitor = _get_health_monitor()
    health = monitor.get_health()

    return {
        "gpu_count": metrics.gpu_count,
        "total_vram_gb": round(metrics.total_vram_gb, 1),
        "total_vram_used_gb": round(metrics.total_vram_used_gb, 1),
        "cpu_pct": metrics.cpu_pct,
        "ram_used_gb": round(metrics.ram_used_gb, 1),
        "queue": manager.get_queue_stats(),
        "pipeline_status": health.overall_status.value,
        "failure_rate_24h": round(health.failure_rate_24h, 1),
    }
