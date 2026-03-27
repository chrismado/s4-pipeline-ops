"""
GPU metrics collector using NVIDIA Management Library (NVML).

Polls GPU state at configurable intervals: temperature, utilization,
VRAM, power draw, clocks, PCIe throughput. Falls back to mock data
when no NVIDIA GPU is available (for development/testing).
"""

import sys
from datetime import datetime

import psutil
from loguru import logger

from config.settings import settings
from src.models.schemas import GPUMetrics, SystemMetrics

# Try to import pynvml — gracefully degrade if no GPU
try:
    import pynvml

    pynvml.nvmlInit()
    _NVML_AVAILABLE = True
    _GPU_COUNT = pynvml.nvmlDeviceGetCount()
    logger.info(f"NVML initialized: {_GPU_COUNT} GPU(s) detected")
except Exception:
    _NVML_AVAILABLE = False
    _GPU_COUNT = 0
    logger.warning("NVML not available — using mock GPU metrics")


def collect_gpu_metrics() -> list[GPUMetrics]:
    """Collect metrics from all available GPUs."""
    if _NVML_AVAILABLE:
        return _collect_nvml()
    return _collect_mock()


def collect_system_metrics() -> SystemMetrics:
    """Collect full system snapshot: GPUs + host resources."""
    gpus = collect_gpu_metrics()

    cpu_pct = psutil.cpu_percent(interval=0.1)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("C:\\" if sys.platform == "win32" else "/")

    return SystemMetrics(
        gpus=gpus,
        cpu_pct=cpu_pct,
        ram_used_gb=mem.used / (1024 ** 3),
        ram_total_gb=mem.total / (1024 ** 3),
        disk_used_gb=disk.used / (1024 ** 3),
        disk_total_gb=disk.total / (1024 ** 3),
        timestamp=datetime.utcnow(),
    )


def _collect_nvml() -> list[GPUMetrics]:
    """Real GPU metrics via NVML."""
    metrics = []
    for i in range(_GPU_COUNT):
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")

            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW → W
                power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
            except pynvml.NVMLError:
                power, power_limit = 0.0, 0.0

            try:
                fan = pynvml.nvmlDeviceGetFanSpeed(handle)
            except pynvml.NVMLError:
                fan = 0.0

            try:
                clk_sm = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
                clk_mem = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
            except pynvml.NVMLError:
                clk_sm, clk_mem = 0, 0

            try:
                pcie = pynvml.nvmlDeviceGetPcieThroughput(
                    handle, pynvml.NVML_PCIE_UTIL_TX_BYTES
                )
                pcie_rx = pynvml.nvmlDeviceGetPcieThroughput(
                    handle, pynvml.NVML_PCIE_UTIL_RX_BYTES
                )
            except pynvml.NVMLError:
                pcie, pcie_rx = 0, 0

            metrics.append(GPUMetrics(
                gpu_index=i,
                name=name,
                temperature_c=float(temp),
                utilization_pct=float(util.gpu),
                memory_used_mb=mem_info.used / (1024 ** 2),
                memory_total_mb=mem_info.total / (1024 ** 2),
                memory_utilization_pct=float(util.memory),
                power_draw_w=power,
                power_limit_w=power_limit,
                fan_speed_pct=float(fan),
                clock_sm_mhz=clk_sm,
                clock_mem_mhz=clk_mem,
                pcie_tx_kbps=pcie,
                pcie_rx_kbps=pcie_rx,
            ))
        except pynvml.NVMLError as e:
            logger.error(f"Failed to read GPU {i}: {e}")

    return metrics


def _collect_mock() -> list[GPUMetrics]:
    """Mock GPU metrics for development without NVIDIA hardware."""
    import random

    return [
        GPUMetrics(
            gpu_index=0,
            name="Mock RTX 4090 (dev mode)",
            temperature_c=45.0 + random.uniform(0, 20),
            utilization_pct=random.uniform(0, 100),
            memory_used_mb=random.uniform(1000, 20000),
            memory_total_mb=24576.0,
            memory_utilization_pct=random.uniform(10, 90),
            power_draw_w=random.uniform(50, 350),
            power_limit_w=450.0,
            fan_speed_pct=random.uniform(20, 80),
            clock_sm_mhz=random.randint(1500, 2520),
            clock_mem_mhz=random.randint(9000, 10500),
        ),
        GPUMetrics(
            gpu_index=1,
            name="Mock RTX 3090 (dev mode)",
            temperature_c=42.0 + random.uniform(0, 25),
            utilization_pct=random.uniform(0, 100),
            memory_used_mb=random.uniform(500, 22000),
            memory_total_mb=24576.0,
            memory_utilization_pct=random.uniform(5, 95),
            power_draw_w=random.uniform(40, 350),
            power_limit_w=350.0,
            fan_speed_pct=random.uniform(15, 90),
            clock_sm_mhz=random.randint(1400, 1950),
            clock_mem_mhz=random.randint(8500, 9750),
        ),
    ]
