"""GPU memory metrics collection using pynvml.

Tracks VRAM baseline (model loaded), peak usage during inference,
and estimated KV cache footprint.
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field

from pydantic import BaseModel

try:
    import pynvml

    pynvml.nvmlInit()
    _NVML_AVAILABLE = True
except Exception:
    _NVML_AVAILABLE = False


class MemoryMetrics(BaseModel):
    """VRAM usage snapshot for a benchmark configuration."""

    vram_baseline_mb: float
    vram_peak_mb: float
    vram_kv_cache_estimated_mb: float
    vram_total_mb: float
    max_batch_estimate: int


def get_vram_usage_mb(gpu_index: int = 0) -> float:
    """Return current VRAM usage in MB for the given GPU."""
    if not _NVML_AVAILABLE:
        return 0.0
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info.used / (1024 * 1024)


def get_vram_total_mb(gpu_index: int = 0) -> float:
    """Return total VRAM in MB for the given GPU."""
    if not _NVML_AVAILABLE:
        return 24576.0  # Default 24 GB for RTX 4090
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info.total / (1024 * 1024)


@dataclass
class VRAMMonitor:
    """Background thread that samples VRAM usage to find peak values.

    Usage::

        monitor = VRAMMonitor(gpu_index=0)
        monitor.start()
        # ... run inference ...
        monitor.stop()
        print(monitor.peak_mb)
    """

    gpu_index: int = 0
    poll_interval_s: float = 0.05  # 50ms sampling
    peak_mb: float = 0.0
    _samples: list[float] = field(default_factory=list)
    _running: bool = False
    _thread: threading.Thread | None = None

    def start(self) -> None:
        """Begin background VRAM monitoring."""
        self._running = True
        self.peak_mb = 0.0
        self._samples = []
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop monitoring."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _poll_loop(self) -> None:
        while self._running:
            usage = get_vram_usage_mb(self.gpu_index)
            self._samples.append(usage)
            if usage > self.peak_mb:
                self.peak_mb = usage
            time.sleep(self.poll_interval_s)


def measure_memory(
    gpu_index: int = 0,
    vram_baseline_mb: float | None = None,
    vram_peak_mb: float | None = None,
) -> MemoryMetrics:
    """Build a MemoryMetrics snapshot.

    If baseline/peak are not provided, takes a single-point measurement.
    KV cache size is estimated as peak minus baseline.
    Max batch estimate assumes each additional concurrent request uses
    roughly the same KV cache allocation.
    """
    vram_total = get_vram_total_mb(gpu_index)
    baseline = vram_baseline_mb if vram_baseline_mb is not None else get_vram_usage_mb(gpu_index)
    peak = vram_peak_mb if vram_peak_mb is not None else get_vram_usage_mb(gpu_index)
    kv_estimate = max(0.0, peak - baseline)
    free_after_model = vram_total - baseline
    max_batch = int(free_after_model / kv_estimate) if kv_estimate > 0 else 1

    return MemoryMetrics(
        vram_baseline_mb=round(baseline, 1),
        vram_peak_mb=round(peak, 1),
        vram_kv_cache_estimated_mb=round(kv_estimate, 1),
        vram_total_mb=round(vram_total, 1),
        max_batch_estimate=max_batch,
    )
