"""
Prometheus metrics exporter — exposes GPU and pipeline metrics in Prometheus text format.

Endpoint: GET /metrics
Format: https://prometheus.io/docs/instrumenting/exposition_formats/
"""

from src.collectors.gpu import collect_system_metrics


def generate_prometheus_metrics() -> str:
    """Generate Prometheus exposition format metrics."""
    lines: list[str] = []

    metrics = collect_system_metrics()

    # GPU metrics
    lines.append("# HELP s4ops_gpu_temperature_celsius GPU temperature in Celsius")
    lines.append("# TYPE s4ops_gpu_temperature_celsius gauge")
    for g in metrics.gpus:
        lines.append(f's4ops_gpu_temperature_celsius{{gpu="{g.gpu_index}",name="{g.name}"}} {g.temperature_c}')

    lines.append("# HELP s4ops_gpu_utilization_percent GPU utilization percentage")
    lines.append("# TYPE s4ops_gpu_utilization_percent gauge")
    for g in metrics.gpus:
        lines.append(f's4ops_gpu_utilization_percent{{gpu="{g.gpu_index}",name="{g.name}"}} {g.utilization_pct}')

    lines.append("# HELP s4ops_gpu_memory_used_bytes GPU memory used in bytes")
    lines.append("# TYPE s4ops_gpu_memory_used_bytes gauge")
    for g in metrics.gpus:
        lines.append(f's4ops_gpu_memory_used_bytes{{gpu="{g.gpu_index}",name="{g.name}"}} {g.memory_used_mb * 1024 * 1024:.0f}')

    lines.append("# HELP s4ops_gpu_memory_total_bytes GPU memory total in bytes")
    lines.append("# TYPE s4ops_gpu_memory_total_bytes gauge")
    for g in metrics.gpus:
        lines.append(f's4ops_gpu_memory_total_bytes{{gpu="{g.gpu_index}",name="{g.name}"}} {g.memory_total_mb * 1024 * 1024:.0f}')

    lines.append("# HELP s4ops_gpu_memory_utilization_percent GPU memory utilization percentage")
    lines.append("# TYPE s4ops_gpu_memory_utilization_percent gauge")
    for g in metrics.gpus:
        lines.append(f's4ops_gpu_memory_utilization_percent{{gpu="{g.gpu_index}",name="{g.name}"}} {g.memory_utilization_pct}')

    lines.append("# HELP s4ops_gpu_power_draw_watts GPU power draw in watts")
    lines.append("# TYPE s4ops_gpu_power_draw_watts gauge")
    for g in metrics.gpus:
        lines.append(f's4ops_gpu_power_draw_watts{{gpu="{g.gpu_index}",name="{g.name}"}} {g.power_draw_w}')

    lines.append("# HELP s4ops_gpu_power_limit_watts GPU power limit in watts")
    lines.append("# TYPE s4ops_gpu_power_limit_watts gauge")
    for g in metrics.gpus:
        lines.append(f's4ops_gpu_power_limit_watts{{gpu="{g.gpu_index}",name="{g.name}"}} {g.power_limit_w}')

    lines.append("# HELP s4ops_gpu_fan_speed_percent GPU fan speed percentage")
    lines.append("# TYPE s4ops_gpu_fan_speed_percent gauge")
    for g in metrics.gpus:
        lines.append(f's4ops_gpu_fan_speed_percent{{gpu="{g.gpu_index}",name="{g.name}"}} {g.fan_speed_pct}')

    lines.append("# HELP s4ops_gpu_clock_sm_mhz GPU SM clock in MHz")
    lines.append("# TYPE s4ops_gpu_clock_sm_mhz gauge")
    for g in metrics.gpus:
        lines.append(f's4ops_gpu_clock_sm_mhz{{gpu="{g.gpu_index}",name="{g.name}"}} {g.clock_sm_mhz}')

    lines.append("# HELP s4ops_gpu_clock_mem_mhz GPU memory clock in MHz")
    lines.append("# TYPE s4ops_gpu_clock_mem_mhz gauge")
    for g in metrics.gpus:
        lines.append(f's4ops_gpu_clock_mem_mhz{{gpu="{g.gpu_index}",name="{g.name}"}} {g.clock_mem_mhz}')

    # System metrics
    lines.append("# HELP s4ops_gpu_count Number of GPUs")
    lines.append("# TYPE s4ops_gpu_count gauge")
    lines.append(f"s4ops_gpu_count {metrics.gpu_count}")

    lines.append("# HELP s4ops_cpu_utilization_percent CPU utilization percentage")
    lines.append("# TYPE s4ops_cpu_utilization_percent gauge")
    lines.append(f"s4ops_cpu_utilization_percent {metrics.cpu_pct}")

    lines.append("# HELP s4ops_ram_used_bytes RAM used in bytes")
    lines.append("# TYPE s4ops_ram_used_bytes gauge")
    lines.append(f"s4ops_ram_used_bytes {metrics.ram_used_gb * 1024 * 1024 * 1024:.0f}")

    lines.append("# HELP s4ops_ram_total_bytes RAM total in bytes")
    lines.append("# TYPE s4ops_ram_total_bytes gauge")
    lines.append(f"s4ops_ram_total_bytes {metrics.ram_total_gb * 1024 * 1024 * 1024:.0f}")

    lines.append("# HELP s4ops_disk_used_bytes Disk used in bytes")
    lines.append("# TYPE s4ops_disk_used_bytes gauge")
    lines.append(f"s4ops_disk_used_bytes {metrics.disk_used_gb * 1024 * 1024 * 1024:.0f}")

    lines.append("# HELP s4ops_disk_total_bytes Disk total in bytes")
    lines.append("# TYPE s4ops_disk_total_bytes gauge")
    lines.append(f"s4ops_disk_total_bytes {metrics.disk_total_gb * 1024 * 1024 * 1024:.0f}")

    # Job queue metrics (lazy singleton)
    from src.api.routes import _get_job_manager
    manager = _get_job_manager()
    stats = manager.get_queue_stats()

    lines.append("# HELP s4ops_jobs_total Total number of jobs")
    lines.append("# TYPE s4ops_jobs_total gauge")
    lines.append(f"s4ops_jobs_total {stats['total']}")

    lines.append("# HELP s4ops_jobs_running Number of running jobs")
    lines.append("# TYPE s4ops_jobs_running gauge")
    lines.append(f"s4ops_jobs_running {stats['running']}")

    lines.append("# HELP s4ops_jobs_queued Number of queued jobs")
    lines.append("# TYPE s4ops_jobs_queued gauge")
    lines.append(f"s4ops_jobs_queued {stats['queued']}")

    for status_name, count in stats.get("by_status", {}).items():
        lines.append(f's4ops_jobs_by_status{{status="{status_name}"}} {count}')

    lines.append("")
    return "\n".join(lines)
