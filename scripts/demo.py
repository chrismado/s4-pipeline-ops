#!/usr/bin/env python3
"""
S4 Pipeline Ops — End-to-End Demo

Demonstrates the full system working together:
  1. Starts the FastAPI dashboard server in a background thread
  2. Submits several fake jobs with progress output
  3. Runs the dispatch/monitor loop to process them
  4. Generates alerts by simulating high GPU readings
  5. Records metrics history for dashboard charts
  6. Prints a final summary

Usage:
    python scripts/demo.py

Then open http://localhost:8100 to see the live dashboard.
"""

import sys
import time
import threading
from pathlib import Path

# Ensure project root is on path
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def banner(msg: str):
    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    console.print(f"[bold cyan]  {msg}[/bold cyan]")
    console.print(f"[bold cyan]{'='*60}[/bold cyan]\n")


def start_api_server():
    """Start the FastAPI server in a background thread."""
    import uvicorn
    from src.api.app import create_app

    app = create_app()
    config = uvicorn.Config(app, host="127.0.0.1", port=8100, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    time.sleep(1)  # Let it start
    return server


def run_demo():
    from config.settings import settings
    from src.scheduler.manager import JobManager
    from src.alerting.engine import AlertEngine
    from src.dashboard.health import HealthMonitor
    from src.collectors.gpu import collect_system_metrics
    from src.collectors.metrics_store import MetricsStore
    from src.models.schemas import (
        JobConfig, JobPriority, GPUMetrics, SystemMetrics,
    )

    # Clean stale data so each demo run starts fresh
    for path in [Path(settings.job_db_path), Path(settings.metrics_db_path)]:
        if path.exists():
            path.unlink()

    manager = JobManager()
    engine = AlertEngine()
    health_monitor = HealthMonitor()
    metrics_store = MetricsStore()

    # ──────────────────────────────────────────────
    banner("S4 PIPELINE OPS — DEMO")
    console.print("[dim]This demo shows the GPU monitoring, job scheduling,\nalerting, and pipeline health features working together.[/dim]\n")

    # ──────────────────────────────────────────────
    banner("STEP 1: Collecting GPU Metrics")

    metrics = collect_system_metrics()
    metrics_store.record(metrics)

    console.print(f"  GPUs detected: [bold]{metrics.gpu_count}[/bold]")
    for gpu in metrics.gpus:
        console.print(
            f"  GPU {gpu.gpu_index}: {gpu.name} — "
            f"{gpu.temperature_c:.0f}°C, {gpu.utilization_pct:.0f}% util, "
            f"{gpu.memory_used_mb:.0f}/{gpu.memory_total_mb:.0f} MB VRAM"
        )
    console.print(f"  CPU: {metrics.cpu_pct:.0f}% | RAM: {metrics.ram_used_gb:.1f}/{metrics.ram_total_gb:.1f} GB")
    console.print()

    # ──────────────────────────────────────────────
    banner("STEP 2: Submitting Jobs")

    jobs_to_submit = [
        ("StyleGAN2 Training",    'python -c "import time; [print(f\'[{i}/10]\') or time.sleep(0.3) for i in range(1,11)]"',
         "high", ["train"]),
        ("Neural Matte Export",   'python -c "import time; [print(f\'Frame {i}/5 rendered\') or time.sleep(0.4) for i in range(1,6)]"',
         "normal", ["render"]),
        ("Gaussian Splat Ingest", 'python -c "import time; [print(f\'{i*20}%\') or time.sleep(0.3) for i in range(1,6)]"',
         "normal", ["ingest"]),
        ("Data Preprocessing",    'python -c "import time; [print(f\'Step {i}/8\') or time.sleep(0.25) for i in range(1,9)]"',
         "low", ["preprocess"]),
        ("Failing Job (expected)", 'python -c "import time; time.sleep(0.5); exit(1)"',
         "normal", ["export"]),
    ]

    submitted = []
    for name, cmd, priority, tags in jobs_to_submit:
        config = JobConfig(command=cmd, gpu_count=0, tags=tags)
        prio = JobPriority(priority)
        job = manager.submit(name, config, prio)
        submitted.append(job)
        console.print(f"  [green]+[/green] {job.id} — {name} [dim](priority={priority})[/dim]")

    console.print(f"\n  Submitted {len(submitted)} jobs to the queue.\n")

    # ──────────────────────────────────────────────
    banner("STEP 3: Dispatching & Running Jobs")

    console.print("  Starting all jobs and monitoring progress...\n")

    # Force-start all jobs (bypass GPU idle check since we're demoing)
    for job in submitted:
        if not job.is_terminal:
            manager._start_job(job)

    # Monitor loop — poll until all jobs finish
    max_wait = 30
    tick = 0
    while tick < max_wait:
        changed = manager.check_running()

        for job in changed:
            # Record to health monitor
            health_monitor.record_job(job)
            # Generate alerts for job events
            engine.check_job_event(job)

        # Show running job progress
        running = [j for j in manager.jobs.values() if j.status.value == "running"]
        for j in running:
            pct_str = f"{j.progress_pct:.0f}%" if j.progress_pct is not None else "..."
            console.print(f"  [blue]Running[/blue] {j.id} {j.name[:30]:30s} progress={pct_str}")

        # Collect and record metrics
        metrics = collect_system_metrics()
        metrics_store.record(metrics)

        # Check if all done
        all_terminal = all(j.is_terminal for j in submitted)
        if all_terminal:
            break

        time.sleep(0.5)
        tick += 1

    console.print()

    # ──────────────────────────────────────────────
    banner("STEP 4: Simulating Alert Conditions")

    # Create synthetic high-temp metrics to trigger alerts
    fake_gpus = [
        GPUMetrics(
            gpu_index=0,
            name="Simulated RTX 4090 (overheating)",
            temperature_c=92.0,
            utilization_pct=99.0,
            memory_used_mb=23500.0,
            memory_total_mb=24576.0,
            memory_utilization_pct=95.6,
            power_draw_w=420.0,
            power_limit_w=450.0,
        ),
    ]
    fake_metrics = SystemMetrics(
        gpus=fake_gpus,
        cpu_pct=85.0,
        ram_used_gb=58.0,
        ram_total_gb=64.0,
        disk_used_gb=450.0,
        disk_total_gb=500.0,
    )
    alerts = engine.check_gpu_metrics(fake_metrics)
    metrics_store.record(fake_metrics)

    for a in alerts:
        severity_color = {"critical": "red", "warning": "yellow", "info": "blue"}
        color = severity_color.get(a.severity.value, "white")
        console.print(f"  [{color}][{a.severity.value.upper()}][/{color}] {a.title}")

    if not alerts:
        console.print("  [dim](Alerts suppressed by cooldown)[/dim]")
    console.print()

    # ──────────────────────────────────────────────
    banner("STEP 5: Final Summary")

    # Job results table
    table = Table(title="Job Results")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Status")
    table.add_column("Duration")
    table.add_column("Progress")

    status_colors = {
        "completed": "green", "failed": "red", "cancelled": "dim",
        "running": "blue", "pending": "dim", "queued": "yellow",
    }

    for job in submitted:
        color = status_colors.get(job.status.value, "white")
        dur = f"{job.duration_seconds:.1f}s" if job.duration_seconds else "—"
        pct = f"{job.progress_pct:.0f}%" if job.progress_pct is not None else "—"
        table.add_row(
            job.id,
            job.name[:35],
            f"[{color}]{job.status.value}[/{color}]",
            dur,
            pct,
        )

    console.print(table)

    # Pipeline health
    health = health_monitor.get_health()
    console.print(f"\n  Pipeline status: [bold]{health.overall_status.value.upper()}[/bold]")
    console.print(f"  24h jobs: {health.total_jobs_24h} total, {health.failed_jobs_24h} failed")

    # Alert summary
    all_alerts = engine.get_recent_alerts()
    console.print(f"  Total alerts: {len(all_alerts)}")

    # Queue stats
    stats = manager.get_queue_stats()
    console.print(f"  Queue: {stats['total']} total, {stats['running']} running, {stats['queued']} queued")

    # Metrics history
    history = metrics_store.get_history(hours=1)
    console.print(f"  Metric snapshots recorded: {len(history)}")

    console.print(Panel.fit(
        "[bold green]Demo complete![/bold green]\n\n"
        "Open [bold]http://localhost:8100[/bold] to see the live dashboard.\n"
        "The dashboard shows GPU charts, job queue, alerts, and pipeline health.\n"
        "Press Ctrl+C to stop.",
        title="S4 Pipeline Ops",
    ))


if __name__ == "__main__":
    try:
        banner("Starting API server on http://127.0.0.1:8100")
        start_api_server()
        console.print("  [green]API server running[/green] — dashboard at http://localhost:8100\n")

        run_demo()

        # Keep alive for dashboard browsing
        console.print("\n[dim]Server running. Press Ctrl+C to stop.[/dim]")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        console.print("\n[dim]Demo stopped.[/dim]")
