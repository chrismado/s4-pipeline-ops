"""
CLI for S4 Pipeline Ops.

Usage:
    s4ops status                         — Quick system status
    s4ops gpu                            — GPU metrics snapshot
    s4ops submit "python train.py" -n "Training run"
    s4ops jobs                           — List all jobs
    s4ops cancel <job_id>
    s4ops health                         — Pipeline health
    s4ops alerts                         — Recent alerts
    s4ops serve                          — Start dashboard API
    s4ops monitor                        — Live monitoring loop
    s4ops benchmark --all                — Run all inference benchmarks
"""

import sys
from pathlib import Path
from typing import Optional

# Ensure the project root is on sys.path so `config` resolves to our config package,
# not a same-named package from another editable install.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()
app = typer.Typer(name="s4ops", help="GPU monitoring and pipeline operations")


@app.command()
def status():
    """Quick system status overview."""
    from src.collectors.gpu import collect_system_metrics

    metrics = collect_system_metrics()

    console.print(Panel.fit(
        f"[bold]GPUs:[/bold] {metrics.gpu_count}  "
        f"[bold]VRAM:[/bold] {metrics.total_vram_used_gb:.1f}/{metrics.total_vram_gb:.1f} GB  "
        f"[bold]CPU:[/bold] {metrics.cpu_pct:.0f}%  "
        f"[bold]RAM:[/bold] {metrics.ram_used_gb:.1f}/{metrics.ram_total_gb:.1f} GB  "
        f"[bold]Disk:[/bold] {metrics.disk_used_gb:.0f}/{metrics.disk_total_gb:.0f} GB",
        title="System Status",
    ))

    table = Table(title="GPU Details")
    table.add_column("GPU", style="cyan")
    table.add_column("Name")
    table.add_column("Temp", justify="right")
    table.add_column("Util", justify="right")
    table.add_column("VRAM", justify="right")
    table.add_column("Power", justify="right")

    for gpu in metrics.gpus:
        temp_style = "red" if gpu.temperature_c > 80 else "yellow" if gpu.temperature_c > 65 else "green"
        util_style = "green" if gpu.utilization_pct > 80 else "yellow" if gpu.utilization_pct > 20 else "dim"

        table.add_row(
            str(gpu.gpu_index),
            gpu.name,
            f"[{temp_style}]{gpu.temperature_c:.0f}°C[/{temp_style}]",
            f"[{util_style}]{gpu.utilization_pct:.0f}%[/{util_style}]",
            f"{gpu.memory_used_mb:.0f}/{gpu.memory_total_mb:.0f} MB",
            f"{gpu.power_draw_w:.0f}W" if gpu.power_draw_w > 0 else "—",
        )

    console.print(table)


@app.command()
def gpu():
    """Detailed GPU metrics snapshot."""
    status()


@app.command()
def submit(
    command: str = typer.Argument(help="Command to run"),
    name: str = typer.Option("", "--name", "-n", help="Job name"),
    gpu_count: int = typer.Option(1, "--gpus", "-g", help="Number of GPUs"),
    priority: str = typer.Option("normal", "--priority", "-p", help="Priority: low, normal, high, critical"),
    tags: Optional[str] = typer.Option(None, "--tags", "-t", help="Comma-separated tags"),
):
    """Submit a job to the scheduler."""
    from src.scheduler.manager import JobManager
    from src.models.schemas import JobConfig, JobPriority

    manager = JobManager()
    config = JobConfig(
        command=command,
        gpu_count=gpu_count,
        tags=tags.split(",") if tags else [],
    )

    try:
        prio = JobPriority(priority)
    except ValueError:
        prio = JobPriority.NORMAL

    job_name = name or command[:40]
    job = manager.submit(job_name, config, prio)
    console.print(f"[green]Job submitted:[/green] {job.id} ({job.name})")


@app.command()
def jobs(
    status_filter: Optional[str] = typer.Option(None, "--status", "-s", help="Filter by status"),
):
    """List all jobs."""
    from src.scheduler.manager import JobManager
    from src.models.schemas import JobStatus

    manager = JobManager()

    sf = None
    if status_filter:
        try:
            sf = JobStatus(status_filter)
        except ValueError:
            console.print(f"[red]Invalid status: {status_filter}[/red]")
            raise typer.Exit(1)

    job_list = manager.list_jobs(status=sf)

    if not job_list:
        console.print("[dim]No jobs found.[/dim]")
        return

    table = Table(title="Jobs")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Status")
    table.add_column("Priority")
    table.add_column("GPUs")
    table.add_column("Duration")

    status_colors = {
        "pending": "dim", "queued": "yellow", "running": "blue",
        "completed": "green", "failed": "red", "cancelled": "dim",
        "retrying": "yellow",
    }

    for job in job_list:
        color = status_colors.get(job.status.value, "white")
        duration = f"{job.duration_seconds:.0f}s" if job.duration_seconds else "—"
        table.add_row(
            job.id,
            job.name[:30],
            f"[{color}]{job.status.value}[/{color}]",
            job.priority.value,
            ",".join(str(g) for g in job.assigned_gpus) or "—",
            duration,
        )

    console.print(table)
    stats = manager.get_queue_stats()
    console.print(f"\n[dim]Running: {stats['running']} | Queued: {stats['queued']} | Total: {stats['total']}[/dim]")


@app.command()
def cancel(job_id: str = typer.Argument(help="Job ID to cancel")):
    """Cancel a job."""
    from src.scheduler.manager import JobManager

    manager = JobManager()
    job = manager.cancel(job_id)
    if job:
        console.print(f"[yellow]Cancelled:[/yellow] {job.id} ({job.name})")
    else:
        console.print(f"[red]Job not found or already complete: {job_id}[/red]")


@app.command()
def health():
    """Pipeline health summary."""
    from src.dashboard.health import HealthMonitor

    monitor = HealthMonitor()
    h = monitor.get_health()

    status_colors = {"healthy": "green", "degraded": "yellow", "down": "red", "unknown": "dim"}
    color = status_colors.get(h.overall_status.value, "white")
    console.print(f"\n[bold]Pipeline Status:[/bold] [{color}]{h.overall_status.value.upper()}[/{color}]")

    table = Table(title="Stage Health")
    table.add_column("Stage")
    table.add_column("Status")
    table.add_column("Error Rate", justify="right")
    table.add_column("Avg Duration", justify="right")
    table.add_column("Throughput/hr", justify="right")

    for stage in h.stages:
        sc = status_colors.get(stage.status.value, "white")
        table.add_row(
            stage.name,
            f"[{sc}]{stage.status.value}[/{sc}]",
            f"{stage.error_rate_pct:.1f}%",
            f"{stage.avg_duration_seconds:.0f}s",
            f"{stage.throughput_per_hour:.1f}",
        )

    console.print(table)


@app.command()
def alerts(limit: int = typer.Option(20, help="Number of alerts to show")):
    """Show recent alerts."""
    from src.alerting.engine import AlertEngine

    engine = AlertEngine()
    alert_list = engine.get_recent_alerts(limit=limit)

    if not alert_list:
        console.print("[dim]No alerts.[/dim]")
        return

    for alert in alert_list:
        severity_style = {"info": "blue", "warning": "yellow", "critical": "red"}
        style = severity_style.get(alert.severity.value, "white")
        console.print(
            f"[{style}][{alert.severity.value.upper()}][/{style}] "
            f"{alert.title} — {alert.message} "
            f"[dim]({alert.timestamp.strftime('%H:%M:%S')})[/dim]"
        )


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind host"),
    port: int = typer.Option(8100, help="Bind port"),
):
    """Start the dashboard API server."""
    import uvicorn
    from src.api.app import create_app

    api = create_app()
    console.print(f"[green]Starting dashboard API on {host}:{port}[/green]")
    uvicorn.run(api, host=host, port=port)


@app.command()
def monitor(
    interval: int = typer.Option(5, help="Polling interval in seconds"),
):
    """Live monitoring loop — polls GPU metrics and checks alerts."""
    import time
    from src.collectors.gpu import collect_system_metrics
    from src.alerting.engine import AlertEngine

    engine = AlertEngine()
    console.print("[bold]Starting live monitor[/bold] (Ctrl+C to stop)\n")

    try:
        while True:
            metrics = collect_system_metrics()
            new_alerts = engine.check_gpu_metrics(metrics)

            # Clear and redraw
            table = Table(title=f"GPU Monitor (polling every {interval}s)")
            table.add_column("GPU", style="cyan")
            table.add_column("Temp", justify="right")
            table.add_column("Util", justify="right")
            table.add_column("VRAM", justify="right")
            table.add_column("Power", justify="right")

            for gpu in metrics.gpus:
                temp_style = "red" if gpu.temperature_c > 80 else "yellow" if gpu.temperature_c > 65 else "green"
                table.add_row(
                    f"{gpu.gpu_index}: {gpu.name}",
                    f"[{temp_style}]{gpu.temperature_c:.0f}°C[/{temp_style}]",
                    f"{gpu.utilization_pct:.0f}%",
                    f"{gpu.memory_used_mb:.0f}/{gpu.memory_total_mb:.0f} MB",
                    f"{gpu.power_draw_w:.0f}W" if gpu.power_draw_w else "—",
                )

            console.clear()
            console.print(table)

            if new_alerts:
                for a in new_alerts:
                    console.print(f"[red]ALERT: {a.title}[/red]")

            time.sleep(interval)
    except KeyboardInterrupt:
        console.print("\n[dim]Monitor stopped.[/dim]")


@app.command()
def agent(
    host: str = typer.Option("0.0.0.0", help="Bind host"),
    port: int = typer.Option(8101, help="Bind port"),
):
    """Start a multi-node agent — exposes local GPU metrics for the central aggregator."""
    import socket
    import uvicorn
    from src.api.app import create_app

    api = create_app()
    node_name = socket.gethostname()
    console.print(f"[green]Starting agent on {host}:{port}[/green] (node: {node_name})")
    console.print(f"[dim]Register this node: POST /nodes/register?url=http://{node_name}:{port}[/dim]")
    uvicorn.run(api, host=host, port=port)


@app.command()
def benchmark(
    all_configs: bool = typer.Option(False, "--all", help="Run all benchmark configurations"),
    backend: Optional[str] = typer.Option(None, "--backend", "-b", help="Backend: ollama, llamacpp, vllm"),
    config_name: Optional[str] = typer.Option(None, "--config", "-c", help="Specific config name"),
    report: bool = typer.Option(False, "--report", help="Generate markdown report after benchmarks"),
    quick: bool = typer.Option(False, "--quick", help="Quick mode: fewer runs (10 instead of 50)"),
    export: Optional[str] = typer.Option(None, "--export", help="Export format: csv"),
    output_dir: str = typer.Option("benchmarks/results", "--output", "-o", help="Results output directory"),
):
    """Run KV cache quantization benchmarks on inference backends."""
    from src.benchmarks.configs import (
        BENCHMARK_SUITE,
        BenchmarkParams,
        LLAMACPP_CONFIGS,
        OLLAMA_CONFIGS,
        VLLM_CONFIGS,
    )
    from src.benchmarks.runner import BenchmarkRunner

    # Select configs
    if config_name:
        configs = [c for c in BENCHMARK_SUITE if c.name == config_name]
        if not configs:
            console.print(f"[red]Config not found: {config_name}[/red]")
            console.print("Available configs:")
            for c in BENCHMARK_SUITE:
                console.print(f"  {c.name} — {c.description}")
            raise typer.Exit(1)
    elif backend:
        backend_map = {"ollama": OLLAMA_CONFIGS, "llamacpp": LLAMACPP_CONFIGS, "vllm": VLLM_CONFIGS}
        configs = backend_map.get(backend, [])
        if not configs:
            console.print(f"[red]Unknown backend: {backend}[/red]")
            raise typer.Exit(1)
    elif all_configs:
        configs = BENCHMARK_SUITE
    else:
        console.print("[yellow]Specify --all, --backend, or --config[/yellow]")
        console.print("\nAvailable configurations:")
        for c in BENCHMARK_SUITE:
            console.print(f"  [cyan]{c.name}[/cyan] — {c.description}")
        raise typer.Exit(0)

    params = BenchmarkParams(eval_runs=10 if quick else 50, warmup_runs=2 if quick else 5)
    console.print(f"\n[bold]Running {len(configs)} benchmark(s)[/bold] ({params.eval_runs} runs each)\n")

    runner = BenchmarkRunner(configs=configs, params=params, output_dir=output_dir)
    results = runner.run_all()

    if not results:
        console.print("[red]No benchmarks completed successfully.[/red]")
        raise typer.Exit(1)

    # Display summary table
    table = Table(title="Benchmark Results")
    table.add_column("Config", style="cyan")
    table.add_column("TPS", justify="right")
    table.add_column("TTFT (ms)", justify="right")
    table.add_column("VRAM Peak (MB)", justify="right")
    table.add_column("KV Cache (MB)", justify="right")

    for r in results:
        table.add_row(
            r.config.name,
            f"{r.latency.tps_mean:.1f}",
            f"{r.latency.ttft_ms_p50:.0f}",
            f"{r.memory.vram_peak_mb:,.0f}",
            f"{r.memory.vram_kv_cache_estimated_mb:,.0f}",
        )

    console.print(table)

    if export == "csv":
        from src.benchmarks.report.csv_export import export_csv

        csv_path = export_csv(results)
        console.print(f"\n[green]CSV exported:[/green] {csv_path}")

    if report:
        from src.benchmarks.report.markdown import generate_markdown_report
        from src.benchmarks.report.plots import generate_all_plots

        md_path = generate_markdown_report(results)
        console.print(f"[green]Report generated:[/green] {md_path}")
        plot_paths = generate_all_plots(results)
        for p in plot_paths:
            console.print(f"[green]Chart:[/green] {p}")


if __name__ == "__main__":
    app()
