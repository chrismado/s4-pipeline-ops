# S4 Pipeline Ops

**GPU render farm monitoring and ML pipeline automation.**

ML pipelines are fragile — GPUs overheat, VRAM fills up, training runs die at hour 23, and nobody knows until someone checks manually. This project automates monitoring, job scheduling, and alerting for GPU-heavy workloads.

Built from the infrastructure that kept the **"S4: The Bob Lazar Story"** ML pipeline running through weeks of StyleGAN2-ADA de-aging renders, neural matting passes, and Gaussian Splatting reconstructions.

---

## Features

- **GPU Monitoring** — real-time metrics via NVIDIA NVML: temperature, utilization, VRAM, power draw, clocks, PCIe throughput. Graceful fallback to mock data when no GPU is available.
- **Job Scheduler** — submit render/training jobs with GPU requirements and priority levels. Automatic dispatch to available GPUs, failure retry with backoff, timeout handling. Real-time progress parsing from stdout (`[50/100]`, `Epoch 5/10`, `75%`).
- **Threshold Alerting** — configurable alerts for GPU temperature, VRAM usage, disk space, job failures. Multi-backend dispatch: logging, Slack webhooks, email, generic webhooks. Per-GPU cooldown prevents alert fatigue.
- **Pipeline Health** — tracks named stages (ingest, preprocess, train, render, export) with error rates, throughput, and overall health scoring.
- **Live Dashboard** — auto-refreshing web UI at `GET /` with Chart.js GPU charts, job queue with progress bars, alerts, and pipeline health.
- **Prometheus + Grafana** — `GET /metrics` in Prometheus exposition format. Importable Grafana dashboard JSON with 9 pre-built panels.
- **Multi-Node** — lightweight agent on each GPU node, central aggregator collects cluster-wide metrics via HTTP.
- **Job Analytics** — historical analysis: average duration by type, failure patterns, GPU utilization, throughput over time.
- **REST API** — FastAPI with endpoints for metrics, jobs, health, alerts, analytics, and cluster management.
- **CLI** — Rich-formatted terminal interface for status, monitoring, job management, and alerting.

## Tech Stack

| Component | Implementation |
|---|---|
| GPU Metrics | NVIDIA NVML (pynvml) with mock fallback |
| Job Management | subprocess with CUDA_VISIBLE_DEVICES, JSON persistence |
| Progress Tracking | Regex parsing of stdout (`[N/M]`, `Epoch N/M`, `N%`, `progress: 0.N`) |
| Alerting | Multi-backend: Slack, email, webhook, log |
| Health Monitoring | Stage-based tracking with error rate and throughput |
| Metrics Storage | JSON time series with configurable retention |
| Observability | Prometheus exporter, Grafana dashboard |
| API | FastAPI + Pydantic |
| CLI | Typer + Rich (tables, panels, live display) |
| Config | Pydantic Settings with S4OPS_ env prefix |
| Deployment | Docker Compose with NVIDIA GPU passthrough |

## Quick Start

```bash
# Install
git clone https://github.com/chrismatteau/s4-pipeline-ops.git
cd s4-pipeline-ops
pip install -e ".[dev]"

# Check system status
s4ops status

# Start the dashboard API + live web UI
s4ops serve
# Open http://localhost:8100

# Run the full end-to-end demo
python scripts/demo.py
```

## CLI Commands

```
s4ops status                          Quick system status overview
s4ops gpu                             Detailed GPU metrics snapshot
s4ops submit "python train.py" -n "Training" -g 2 -p high
s4ops jobs                            List all jobs
s4ops jobs --status running           Filter by status
s4ops cancel <job_id>                 Cancel a job
s4ops health                          Pipeline health summary
s4ops alerts                          Recent alerts
s4ops serve                           Start dashboard API (port 8100)
s4ops monitor                         Live GPU monitoring loop
s4ops agent --port 8101               Start multi-node agent
s4ops benchmark --all                 Run all inference benchmarks
s4ops benchmark --backend llamacpp    Benchmark specific backend
s4ops benchmark --all --report        Generate report with charts
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Live dashboard (HTML) |
| `GET` | `/status` | Quick system status |
| `GET` | `/gpu/metrics` | Current GPU + system metrics |
| `GET` | `/gpu/history?hours=24` | GPU metrics time series |
| `GET` | `/metrics` | Prometheus exposition format |
| `POST` | `/jobs` | Submit a job |
| `GET` | `/jobs` | List jobs (filter by `?status=running`) |
| `GET` | `/jobs/{id}` | Job details |
| `DELETE` | `/jobs/{id}` | Cancel a job |
| `GET` | `/health` | Pipeline health summary |
| `GET` | `/alerts` | Recent alerts |
| `GET` | `/analytics?hours=168` | Historical job analytics |
| `GET` | `/agent/metrics` | This node's metrics (multi-node) |
| `GET` | `/agent/health` | Agent health check |
| `GET` | `/nodes` | List registered cluster nodes |
| `POST` | `/nodes/register?url=...` | Register a remote agent |
| `GET` | `/cluster/metrics` | Aggregated cluster metrics |

### API Usage

```bash
# GPU metrics
curl http://localhost:8100/gpu/metrics

# Submit a job
curl -X POST http://localhost:8100/jobs \
  -H "Content-Type: application/json" \
  -d '{"name": "Training run", "command": "python train.py", "gpu_count": 2, "priority": "high", "tags": ["train"]}'

# Job queue
curl http://localhost:8100/jobs

# Pipeline health
curl http://localhost:8100/health

# Recent alerts
curl http://localhost:8100/alerts

# Job analytics (last 7 days)
curl http://localhost:8100/analytics

# Prometheus metrics
curl http://localhost:8100/metrics
```

## Alert Configuration

Alerts fire when GPU metrics cross configurable thresholds:

| Metric | Warning | Critical |
|---|---|---|
| GPU Temperature | 80°C | 90°C |
| VRAM Usage | 85% | 95% |
| Disk Space | 90% | 95% |

Alert backends are configured via environment variables:

```bash
S4OPS_ALERT_BACKENDS='["log","slack"]'
S4OPS_SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
```

Cooldown (default 5 minutes) prevents the same alert from firing repeatedly.

## Multi-Node Setup

Run an agent on each GPU node:

```bash
# On each GPU server
s4ops agent --port 8101
```

Register nodes with the central server:

```bash
curl -X POST "http://central:8100/nodes/register?url=http://node1:8101"
curl -X POST "http://central:8100/nodes/register?url=http://node2:8101"
curl http://central:8100/cluster/metrics
```

## Grafana

Import `grafana/dashboard.json` into Grafana. Add Prometheus scrape config:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 's4ops'
    scrape_interval: 5s
    static_configs:
      - targets: ['localhost:8100']
```

## Docker

```bash
cd docker
docker compose up -d
```

GPU passthrough via NVIDIA Container Toolkit. The compose file mounts `data/` for job state persistence.

## Testing

```bash
pytest                    # Run all 32 tests
pytest -v                 # Verbose output
pytest --cov=src          # With coverage
```

## Configuration

All settings via environment variables with `S4OPS_` prefix:

```bash
S4OPS_POLL_INTERVAL=5               # GPU polling interval (seconds)
S4OPS_GPU_TEMP_WARNING=80           # Temperature warning (°C)
S4OPS_GPU_TEMP_CRITICAL=90          # Temperature critical (°C)
S4OPS_VRAM_USAGE_WARNING=0.85       # VRAM warning (fraction)
S4OPS_VRAM_USAGE_CRITICAL=0.95      # VRAM critical (fraction)
S4OPS_MAX_CONCURRENT_JOBS=4         # Max simultaneous jobs
S4OPS_RETRY_MAX_ATTEMPTS=3          # Max retry attempts
S4OPS_ALERT_BACKENDS=["log"]        # log, slack, email, webhook
S4OPS_API_PORT=8100                 # Dashboard API port
```

See `.env.example` for the full list.

## Project Structure

```
s4-pipeline-ops/
├── config/settings.py              Pydantic settings with env var overrides
├── src/
│   ├── models/schemas.py           GPUMetrics, Job, Alert, PipelineHealth
│   ├── collectors/
│   │   ├── gpu.py                  NVML collector with mock fallback
│   │   └── metrics_store.py        JSON time series storage
│   ├── scheduler/
│   │   ├── manager.py              Job lifecycle management
│   │   └── progress.py             Stdout progress parsing
│   ├── alerting/engine.py          Threshold alerts + multi-backend dispatch
│   ├── dashboard/
│   │   ├── health.py               Pipeline stage health monitoring
│   │   ├── analytics.py            Historical job analytics
│   │   └── templates/index.html    Live dashboard
│   ├── api/
│   │   ├── app.py                  FastAPI factory
│   │   ├── routes.py               REST endpoints
│   │   └── prometheus.py           Prometheus exporter
│   ├── benchmarks/
│   │   ├── configs.py                 Benchmark configurations
│   │   ├── runner.py                  Benchmark orchestrator
│   │   ├── prometheus.py              Prometheus metrics for benchmarks
│   │   ├── inference/                 Backend wrappers (Ollama, llama.cpp, vLLM)
│   │   ├── metrics/                   Latency, memory, throughput, quality
│   │   ├── report/                    Markdown, CSV, matplotlib charts
│   │   └── prompts/                   Standardized eval prompts
│   ├── multinode/
│   │   ├── agent.py                Per-node agent
│   │   └── aggregator.py           Central metrics aggregator
│   └── cli.py                      Typer CLI
├── tests/                          32 tests
├── scripts/demo.py                 End-to-end demo
├── grafana/dashboard.json          Importable Grafana dashboard
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── .github/workflows/ci.yml        CI pipeline
└── pyproject.toml
```

## KV Cache Quantization Benchmarks (March 2026)

Inspired by [Google Research's TurboQuant](https://research.google/blog/turboquant) (March 25, 2026),
which demonstrated 6× KV cache memory reduction and 8× attention speedup with no measurable accuracy
loss on H100 GPUs. This module benchmarks the real-world impact of KV cache quantization on Mistral 7B
inference performance using RTX 4090 consumer hardware.

### What Gets Benchmarked

- **Ollama**: Q4_K_M, Q5_K_M, Q8_0, FP16 model variants
- **llama.cpp**: 6 configs — 2 model quants (Q8_0, Q4_K_M) × 3 KV cache types (FP16, INT8, INT4)
- **vLLM** (optional): FP16 and INT8 KV cache with PagedAttention

### Metrics Collected

| Category | Measurements |
|----------|-------------|
| **Latency** | TTFT, tokens/sec, e2e latency, p50/p95/p99 across 50+ runs |
| **Memory** | VRAM baseline, peak, estimated KV cache size via pynvml |
| **Throughput** | Sequential RPM, concurrent throughput at 1/2/4/8 parallel requests |
| **Quality** | Output similarity to FP16 baseline, perplexity estimation |

### Run Benchmarks

```bash
# Run all benchmarks
s4ops benchmark --all

# Run specific backend
s4ops benchmark --backend llamacpp

# Quick mode (10 runs instead of 50)
s4ops benchmark --all --quick

# Generate report with charts
s4ops benchmark --all --report

# Export raw CSV
s4ops benchmark --all --export csv
```

[Full benchmark report →](docs/kv_cache_benchmarks.md)

## Background

During production of **"S4: The Bob Lazar Story"**, the ML pipeline processed thousands of video frames through multiple GPU-intensive models — StyleGAN2-ADA for de-aging, U-Net with ControlNet for compositing, and Gaussian Splatting for 3D reconstruction. Each stage could run for hours, consuming multiple GPUs.

Without automated monitoring, failures went unnoticed for hours. This project extracts the monitoring and scheduling infrastructure into a reusable toolkit for any GPU-heavy ML pipeline.

## Related Projects

- [**s4-research-intelligence**](https://github.com/chrismatteau/s4-research-intelligence) — RAG pipeline for documentary research with source-weighted retrieval
- [**s4-temporal-flow**](https://github.com/chrismatteau/s4-temporal-flow) — Temporal consistency for AI-generated video via optical flow warping

## License

MIT
