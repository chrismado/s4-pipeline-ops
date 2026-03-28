# Claude Code Handoff — Run KV Cache Benchmarks

## Status

The benchmark **code is complete** on branch `feature/kv-cache-benchmarks` (PR #1). All 65 tests pass, lint is clean. What remains is **running the benchmarks on the 4090 hardware** and populating `docs/kv_cache_benchmarks.md` with real numbers.

## Who This Is For

Chris Matteau — Montreal-based ML engineer. This benchmark suite proves he understands inference economics at the hardware level. Target companies: CGI, Torc Robotics, Samsung AI, Runway. The benchmarks are inspired by Google Research's TurboQuant (March 25, 2026) — implementing the concept on consumer hardware within days of publication.

## What's Already Built

Everything under `src/benchmarks/`:
- **12 benchmark configs** in `configs.py`: 4 Ollama + 6 llama.cpp + 2 vLLM
- **3 backend wrappers** in `inference/`: Ollama (HTTP streaming), llama.cpp (OpenAI-compatible API with `--cache-type-k`/`--cache-type-v` flags), vLLM (optional)
- **4 metrics modules** in `metrics/`: latency (TTFT/TPS/p50/p95/p99), memory (pynvml VRAM profiling), throughput (async concurrent), quality (Jaccard similarity + perplexity)
- **Report generators** in `report/`: markdown tables, CSV export, 5 matplotlib charts
- **CLI command**: `s4ops benchmark --all/--backend/--config` with `--report`, `--quick`, `--export csv`
- **Prometheus integration**: benchmark results exported via existing `/metrics` endpoint
- **33 tests** in `tests/test_benchmarks.py`

## What You Need To Do

### Step 1: Prerequisites

```bash
# Ensure the project is installed
cd s4-pipeline-ops
git checkout feature/kv-cache-benchmarks
pip install -e ".[dev]"

# Verify tests pass
pytest tests/ -v
```

### Step 2: Run Ollama Benchmarks

Ollama should already be installed and running on the machine.

```bash
# Pull models if not already present
ollama pull mistral:7b
ollama pull mistral:7b-instruct-q5_K_M
ollama pull mistral:7b-instruct-q8_0
ollama pull mistral:7b-instruct-fp16

# Run benchmarks (--quick for 10 runs, drop --quick for full 50-run suite)
s4ops benchmark --backend ollama --quick --report
```

If a model variant tag doesn't exist in Ollama's registry, adjust the model name in `src/benchmarks/configs.py` (`OLLAMA_CONFIGS`). Ollama model tags change — check `ollama list` and the Ollama model library for current Mistral 7B tags.

### Step 3: Run llama.cpp Benchmarks

This is the core benchmark — it tests KV cache quantization directly via `--cache-type-k` and `--cache-type-v` flags.

**Install llama.cpp:**
- Build from source: https://github.com/ggerganov/llama.cpp
- Or install a prebuilt release
- `llama-server` must be on PATH (or adjust `LlamaCppBenchmark` to use a full path)

**Download GGUF models:**
```bash
pip install huggingface-hub
python scripts/download_models.py
```

**IMPORTANT:** The `download_models.py` script references `MistralAI/Mistral-7B-Instruct-v0.3-GGUF` as the HuggingFace repo. This may not be the correct repo ID — GGUF files are often hosted by community quantizers. If the download fails:
1. Search HuggingFace for `Mistral-7B-Instruct-v0.3 GGUF`
2. Update the `repo` field in `scripts/download_models.py`
3. Ensure the filenames match what's in `src/benchmarks/configs.py` (`LLAMACPP_CONFIGS`)

**Run benchmarks:**
```bash
s4ops benchmark --backend llamacpp --quick --report
```

The runner will start/stop `llama-server` automatically for each config. If a config fails (e.g., INT4 KV cache not supported on this llama.cpp build), it logs the error and continues to the next config.

### Step 4: vLLM Benchmarks (Optional)

Skip if vLLM is difficult to install on Windows or RTX 4090. The Ollama + llama.cpp results are the core story.

```bash
pip install vllm
s4ops benchmark --backend vllm --quick --report
```

### Step 5: Full Run With Statistical Rigor

Once quick runs validate everything works:

```bash
s4ops benchmark --all --report --export csv
```

This runs 50 iterations per config with 5 warmup runs. Takes a while.

### Step 6: Review and Polish the Report

After benchmarks complete, the report is at `docs/kv_cache_benchmarks.md` and charts are in `benchmarks/reports/`. Review the auto-generated report for:

- **Key Findings section** — verify the auto-computed memory ratios and TPS deltas make sense
- **Quality metrics** — if similarity to baseline is low for INT4 KV cache, that's noteworthy and should be called out
- **Charts** — regenerated each run, should show clear visual differences

If the auto-generated analysis is too thin, expand the Key Findings section manually with insights specific to the results.

### Step 7: Commit Results and Update PR

```bash
# The report and charts are gitignored by default.
# To include the report in the repo:
git add docs/kv_cache_benchmarks.md
git add benchmarks/reports/*.png  # if you want charts in the repo
git commit -m "docs: add benchmark results from RTX 4090 runs"
git push
```

## Architecture Gotchas

| Thing | Detail |
|-------|--------|
| CLI framework | **Typer** (not Click/argparse). Commands are `@app.command()` on `src.cli:app` |
| Prometheus | **Hand-rolled** string formatting in `src/api/prometheus.py`, not `prometheus_client` library |
| GPU metrics | **pynvml** with mock fallback — tests pass without a GPU but benchmarks need real hardware |
| Model files | `models/` directory is **gitignored** — GGUF files won't be in the repo |
| Backend lifecycle | Runner calls `backend.start()` before and `backend.stop()` after each config — llama-server is started/killed per config |
| Async throughput | Uses `asyncio.run()` inside a sync function — works but can't be called from an existing async context |
| Baseline comparison | First config with `kv_cache_type_k is None` or `"f16"` becomes the quality baseline — run order matters |
| Server ports | Ollama: 11434 (default), llama.cpp: 8080, vLLM: 8000 — don't run conflicting services |

## If Something Goes Wrong

**"Model not found" from Ollama:** Check `ollama list` for available model tags. Ollama model naming changes frequently. Update model names in `OLLAMA_CONFIGS`.

**"GGUF model not found" from llama.cpp:** Models need to be in `models/` relative to where you run the command. Check `scripts/download_models.py` ran successfully.

**"llama-server not ready after 120s":** The model is too large for VRAM, or llama-server isn't on PATH. Check with `llama-server --help`. For FP16 models on a single 4090 (24GB), you may need to reduce `--ctx-size` or skip that config.

**Throughput test hangs:** The async concurrency test dispatches parallel requests. If the backend doesn't support concurrent requests (Ollama default), requests queue up. This is expected — it measures real queuing behavior.

**Negative KV cache estimate:** If VRAM peak < baseline, the KV cache estimate is clamped to 0. This can happen if the GPU was doing other work during baseline measurement. Re-run with the GPU idle.

## Definition of Done

- [ ] Ollama benchmarks run successfully with real numbers
- [ ] llama.cpp benchmarks run with at least 4 of 6 configs (FP16 model may not fit)
- [ ] `docs/kv_cache_benchmarks.md` has actual data, not placeholder "XX.X" values
- [ ] Charts generated in `benchmarks/reports/`
- [ ] Results committed and PR updated
- [ ] README benchmark section updated with actual key results (memory reduction ratios, TPS changes)
