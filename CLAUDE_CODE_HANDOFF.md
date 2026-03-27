# Claude Code Handoff — S4 Pipeline Ops

## Who You're Building This For

Chris Matteau — Montreal ML engineer who built GPU-heavy ML pipelines for the documentary "S4: The Bob Lazar Story" (StyleGAN2-ADA de-aging, neural matting, Gaussian Splatting). This project open-sources the infrastructure and monitoring tooling that kept those pipelines running reliably. Primary targets: **CGI** (infrastructure consulting), **National Bank/Desjardins** (ML platform engineering), **Palantir** (production ML systems), **Autodesk** (pipeline TD/automation).

## What This Project Does

ML pipelines crash. GPUs overheat. VRAM fills up. Training runs die at hour 23. This project monitors and manages GPU render farms and ML training infrastructure:

1. **GPU Monitoring** — polls NVIDIA GPUs via NVML (temperature, utilization, VRAM, power, clocks, PCIe)
2. **Job Scheduling** — queues render/training jobs, assigns GPUs based on availability, retries failures
3. **Alerting** — threshold-based alerts (temp, VRAM, disk) dispatched to Slack, email, webhooks
4. **Pipeline Health** — tracks stage status (ingest → preprocess → train → render → export), error rates, throughput

## What's Already Scaffolded

```
s4-pipeline-ops/
├── config/settings.py              # Pydantic settings with env var overrides
├── src/
│   ├── models/schemas.py           # GPUMetrics, SystemMetrics, Job, Alert, PipelineHealth
│   ├── collectors/gpu.py           # NVML GPU collector with mock fallback
│   ├── scheduler/manager.py        # Job lifecycle: submit → queue → assign → run → retry
│   ├── alerting/engine.py          # Threshold evaluation, cooldown, multi-backend dispatch
│   ├── dashboard/health.py         # Pipeline stage monitoring and health scoring
│   ├── api/
│   │   ├── app.py                  # FastAPI factory
│   │   └── routes.py               # REST endpoints for all features
│   └── cli.py                      # Typer CLI: status, gpu, submit, jobs, health, alerts, serve, monitor
├── tests/
│   ├── test_models.py              # Schema validation, properties, edge cases
│   ├── test_collectors.py          # GPU collector (mock mode)
│   └── test_alerting.py            # Threshold evaluation, cooldown, job alerts
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml          # With NVIDIA GPU passthrough
├── .env.example
├── .gitignore
└── pyproject.toml
```

## What Needs To Be Done

### Phase 1: Get It Running

1. **Install deps, fix imports**
   - `pip install -e ".[dev]"` and resolve any issues
   - Run `pytest` and fix any test failures
   - Add `conftest.py` if test imports need path setup
   - Add `data/` directory creation in the startup path

2. **Run the mock monitoring loop**
   - `s4ops status` should show mock GPU data with Rich formatting
   - `s4ops monitor` should start the live monitoring loop
   - `s4ops serve` should start the FastAPI server on port 8100
   - Test all API endpoints with curl/httpx

3. **Test job submission flow**
   - Submit a job: `s4ops submit "echo hello" -n "Test job"`
   - Verify it appears in `s4ops jobs`
   - Verify the dispatch loop picks it up and runs it
   - Test cancellation, retry logic

### Phase 2: Make It Demo-Ready

4. **Add metrics time series storage**
   - Store GPU metrics history in a simple JSON/SQLite store
   - Add `GET /gpu/history?hours=24` endpoint
   - This enables the time-series charts in the dashboard

5. **Add a live dashboard page**
   - Serve a single HTML page at `GET /` with auto-refreshing GPU charts
   - Options: Chart.js with fetch polling, or a simple Jinja2 template
   - Should show: GPU utilization/temp/VRAM over time, job queue, alerts
   - This is the visual that makes the project shine on GitHub

6. **Add render job progress tracking**
   - Parse stdout from running jobs for progress indicators
   - Support patterns like `[50/100]`, `50%`, `Epoch 5/10`
   - Update `job.progress_pct` in real time

7. **Add scripts/demo.py**
   - A self-contained demo that:
     - Submits several fake jobs (short sleep commands with progress output)
     - Shows them flowing through the scheduler
     - Generates alerts by simulating high-temp GPU readings
     - Produces a final summary showing the system working end-to-end

### Phase 3: Production Polish

8. **Add Prometheus metrics export** (`GET /metrics` in Prometheus format)
9. **Add Grafana dashboard JSON** (importable dashboard definition)
10. **Add multi-node support** — agent running on each GPU node, central aggregator
11. **Add historical job analytics** — average duration by type, failure patterns, GPU utilization over time

## Key Architecture Decisions (Don't Change)

- **NVML with mock fallback**: The GPU collector gracefully degrades to mock data when no NVIDIA GPU is available. This means the project runs everywhere (CI, laptops, servers) without modification. Don't make NVML required.

- **Job state persisted to JSON**: Jobs survive process restarts. Previously-running jobs are marked as failed on recovery. This is intentionally simple — don't add a database dependency unless Phase 3 demands it.

- **Alert cooldown is per-type-per-GPU**: The same alert (e.g., GPU 0 temperature warning) won't fire again within `alert_cooldown_seconds`. This prevents alert fatigue. Don't remove or weaken this.

- **Pipeline stages are configurable**: The list of stages in `settings.pipeline_stages` should match whatever the user's actual pipeline looks like. Don't hardcode stage names in the health monitor.

- **FastAPI singletons are lazy**: The job manager, alert engine, and health monitor are created on first request, not at import time. This keeps tests fast and avoids side effects during import.

## Tech Stack Mapping to Target Requirements

| Requirement (CGI/Banks/Palantir/Autodesk) | Project Implementation |
|---|---|
| Infrastructure automation | GPU monitoring, job scheduling, alerting |
| Python production systems | FastAPI, Pydantic, async-ready architecture |
| ML/GPU experience | NVML integration, CUDA_VISIBLE_DEVICES management, VRAM tracking |
| Monitoring & observability | Multi-backend alerting, pipeline health scoring, metrics |
| Docker & deployment | Docker Compose with GPU passthrough, env-based config |
| REST APIs | FastAPI with full CRUD for jobs, metrics, alerts |
| Testing | pytest with mock GPU support, threshold validation |
| CI/CD awareness | Works in CI without GPUs (mock mode), structured for automation |
