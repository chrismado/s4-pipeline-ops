"""
Time series storage for GPU metrics — JSON file backed.

Stores GPU metric snapshots at each poll interval with automatic
retention-based cleanup. Enables the /gpu/history endpoint for
time-series charts in the dashboard.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock

from loguru import logger

from config.settings import settings


class MetricsStore:
    """
    Append-only JSON store for GPU metric history.

    Each entry is a timestamped snapshot of all GPU metrics.
    Old entries are pruned based on metrics_retention_hours.
    Thread-safe for concurrent API access.
    """

    def __init__(self):
        self._path = Path(settings.metrics_db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._history: list[dict] = self._load()

    def record(self, system_metrics) -> None:
        """Append a system metrics snapshot to history."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "gpus": [
                {
                    "gpu_index": g.gpu_index,
                    "name": g.name,
                    "temperature_c": g.temperature_c,
                    "utilization_pct": g.utilization_pct,
                    "memory_used_mb": g.memory_used_mb,
                    "memory_total_mb": g.memory_total_mb,
                    "memory_utilization_pct": g.memory_utilization_pct,
                    "power_draw_w": g.power_draw_w,
                    "fan_speed_pct": g.fan_speed_pct,
                }
                for g in system_metrics.gpus
            ],
            "cpu_pct": system_metrics.cpu_pct,
            "ram_used_gb": round(system_metrics.ram_used_gb, 2),
            "ram_total_gb": round(system_metrics.ram_total_gb, 2),
            "disk_used_gb": round(system_metrics.disk_used_gb, 1),
            "disk_total_gb": round(system_metrics.disk_total_gb, 1),
        }

        with self._lock:
            self._history.append(entry)
            self._prune()
            self._save()

    def get_history(self, hours: float = 24, gpu_index: int | None = None) -> list[dict]:
        """Return metric history for the given time window."""
        cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()

        with self._lock:
            results = [e for e in self._history if e["timestamp"] >= cutoff]

        if gpu_index is not None:
            filtered = []
            for entry in results:
                copy = dict(entry)
                copy["gpus"] = [g for g in entry["gpus"] if g["gpu_index"] == gpu_index]
                filtered.append(copy)
            results = filtered

        return results

    def _prune(self) -> None:
        """Remove entries older than retention period."""
        cutoff = (datetime.utcnow() - timedelta(hours=settings.metrics_retention_hours)).isoformat()
        before = len(self._history)
        self._history = [e for e in self._history if e["timestamp"] >= cutoff]
        pruned = before - len(self._history)
        if pruned > 0:
            logger.debug(f"Pruned {pruned} old metric entries")

    def _save(self) -> None:
        """Persist to disk."""
        try:
            self._path.write_text(json.dumps(self._history, indent=None))
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")

    def _load(self) -> list[dict]:
        """Load from disk."""
        if not self._path.exists():
            return []
        try:
            data = json.loads(self._path.read_text())
            if isinstance(data, list):
                logger.info(f"Loaded {len(data)} metric entries from store")
                return data
        except Exception as e:
            logger.error(f"Failed to load metrics store: {e}")
        return []
