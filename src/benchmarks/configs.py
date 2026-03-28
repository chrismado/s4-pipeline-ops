"""Benchmark configurations for KV cache quantization experiments.

Defines the full benchmark matrix: model quantizations × KV cache types
across Ollama and llama.cpp backends.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class BenchmarkConfig(BaseModel):
    """Configuration for a single benchmark run."""

    name: str
    backend: str  # "ollama" | "llamacpp" | "vllm"
    model: str  # model name (ollama) or GGUF filename (llamacpp)
    model_quant: str  # "fp16" | "q4_k_m" | "q5_k_m" | "q8_0"
    kv_cache_type_k: str | None = None  # "f16" | "q8_0" | "q4_0"
    kv_cache_type_v: str | None = None  # "f16" | "q8_0" | "q4_0"
    context_length: int = 4096
    gpu_ids: list[int] = Field(default_factory=lambda: [0])
    num_parallel: int = 1
    description: str = ""


class BenchmarkParams(BaseModel):
    """Parameters controlling benchmark execution."""

    warmup_runs: int = 5
    eval_runs: int = 50
    max_tokens: int = 256
    concurrency_levels: list[int] = Field(default_factory=lambda: [1, 2, 4, 8])
    context_lengths: list[int] = Field(default_factory=lambda: [512, 1024, 2048, 4096])


# ---------------------------------------------------------------------------
# Pre-defined benchmark suite
# ---------------------------------------------------------------------------

OLLAMA_CONFIGS = [
    BenchmarkConfig(
        name="ollama-q4km",
        backend="ollama",
        model="mistral:7b",
        model_quant="q4_k_m",
        description="Ollama default Mistral 7B Q4_K_M",
    ),
    BenchmarkConfig(
        name="ollama-q5km",
        backend="ollama",
        model="mistral:7b-instruct-q5_K_M",
        model_quant="q5_k_m",
        description="Ollama Mistral 7B Q5_K_M",
    ),
    BenchmarkConfig(
        name="ollama-q8",
        backend="ollama",
        model="mistral:7b-instruct-q8_0",
        model_quant="q8_0",
        description="Ollama Mistral 7B Q8_0",
    ),
    BenchmarkConfig(
        name="ollama-fp16",
        backend="ollama",
        model="mistral:7b-instruct-fp16",
        model_quant="fp16",
        description="Ollama Mistral 7B FP16 (baseline)",
    ),
]

LLAMACPP_CONFIGS = [
    # Q8 model × KV cache variants
    BenchmarkConfig(
        name="llamacpp-q8-kv-fp16",
        backend="llamacpp",
        model="mistral-7b-instruct-v0.3.Q8_0.gguf",
        model_quant="q8_0",
        kv_cache_type_k="f16",
        kv_cache_type_v="f16",
        description="llama.cpp Q8 model + FP16 KV cache (baseline)",
    ),
    BenchmarkConfig(
        name="llamacpp-q8-kv-int8",
        backend="llamacpp",
        model="mistral-7b-instruct-v0.3.Q8_0.gguf",
        model_quant="q8_0",
        kv_cache_type_k="q8_0",
        kv_cache_type_v="q8_0",
        description="llama.cpp Q8 model + INT8 KV cache",
    ),
    BenchmarkConfig(
        name="llamacpp-q8-kv-int4",
        backend="llamacpp",
        model="mistral-7b-instruct-v0.3.Q8_0.gguf",
        model_quant="q8_0",
        kv_cache_type_k="q4_0",
        kv_cache_type_v="q4_0",
        description="llama.cpp Q8 model + INT4 KV cache (TurboQuant-inspired)",
    ),
    # Q4_K_M model × KV cache variants
    BenchmarkConfig(
        name="llamacpp-q4km-kv-fp16",
        backend="llamacpp",
        model="mistral-7b-instruct-v0.3.Q4_K_M.gguf",
        model_quant="q4_k_m",
        kv_cache_type_k="f16",
        kv_cache_type_v="f16",
        description="llama.cpp Q4_K_M model + FP16 KV cache",
    ),
    BenchmarkConfig(
        name="llamacpp-q4km-kv-int8",
        backend="llamacpp",
        model="mistral-7b-instruct-v0.3.Q4_K_M.gguf",
        model_quant="q4_k_m",
        kv_cache_type_k="q8_0",
        kv_cache_type_v="q8_0",
        description="llama.cpp Q4_K_M model + INT8 KV cache",
    ),
    BenchmarkConfig(
        name="llamacpp-q4km-kv-int4",
        backend="llamacpp",
        model="mistral-7b-instruct-v0.3.Q4_K_M.gguf",
        model_quant="q4_k_m",
        kv_cache_type_k="q4_0",
        kv_cache_type_v="q4_0",
        description="llama.cpp Q4_K_M model + INT4 KV cache (max compression)",
    ),
]

VLLM_CONFIGS = [
    BenchmarkConfig(
        name="vllm-fp16",
        backend="vllm",
        model="mistralai/Mistral-7B-Instruct-v0.3",
        model_quant="fp16",
        description="vLLM FP16 with PagedAttention",
    ),
    BenchmarkConfig(
        name="vllm-kv-int8",
        backend="vllm",
        model="mistralai/Mistral-7B-Instruct-v0.3",
        model_quant="fp16",
        kv_cache_type_k="q8_0",
        kv_cache_type_v="q8_0",
        description="vLLM INT8 KV cache with PagedAttention",
    ),
]

BENCHMARK_SUITE: list[BenchmarkConfig] = OLLAMA_CONFIGS + LLAMACPP_CONFIGS + VLLM_CONFIGS
