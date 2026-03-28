"""
Multi-node agent — runs on each GPU node, exposes local metrics via HTTP.

Each node runs a lightweight agent that:
  - Polls local GPU metrics at the configured interval
  - Serves them at GET /agent/metrics for the central aggregator
  - Reports node identity (hostname, IP, GPU count)

Usage:
    s4ops agent --port 8101

The central aggregator polls each agent's /agent/metrics endpoint.
"""

import socket
from datetime import UTC, datetime

from fastapi import APIRouter

from src.collectors.gpu import collect_gpu_metrics, collect_system_metrics

router = APIRouter(prefix="/agent", tags=["agent"])


def get_node_id() -> str:
    """Return a stable node identifier."""
    return socket.gethostname()


@router.get("/metrics")
def agent_metrics():
    """Return this node's current metrics for the aggregator."""
    metrics = collect_system_metrics()
    return {
        "node_id": get_node_id(),
        "hostname": socket.gethostname(),
        "timestamp": datetime.now(UTC).isoformat(),
        "gpu_count": metrics.gpu_count,
        "metrics": metrics.model_dump(mode="json"),
    }


@router.get("/health")
def agent_health():
    """Quick health check for the aggregator."""
    gpus = collect_gpu_metrics()
    return {
        "node_id": get_node_id(),
        "status": "ok",
        "gpu_count": len(gpus),
        "timestamp": datetime.now(UTC).isoformat(),
    }
