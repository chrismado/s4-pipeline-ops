"""Quality metrics: perplexity comparison and output similarity.

Compares quantized model outputs against FP16 baseline to measure
quality degradation from KV cache compression.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from src.benchmarks.inference.ollama_bench import InferenceBackend


class QualityMetrics(BaseModel):
    """Quality measurements for a benchmark configuration."""

    avg_output_length: float
    output_length_std: float
    similarity_to_baseline: float | None = None  # cosine similarity if baseline available
    perplexity_proxy: float | None = None  # log-prob based approximation
    factual_score: float | None = None  # fraction of factual answers correct


def measure_output_quality(
    backend: InferenceBackend,
    prompts: list[str],
    max_tokens: int = 256,
) -> tuple[list[str], QualityMetrics]:
    """Generate outputs for all prompts and compute basic quality stats.

    Returns the raw outputs (for later comparison) and quality metrics.
    """
    outputs: list[str] = []
    lengths: list[int] = []

    for prompt in prompts:
        result = backend.generate(prompt, max_tokens)
        outputs.append(result.text)
        lengths.append(len(result.text.split()))

    avg_len = sum(lengths) / len(lengths) if lengths else 0.0
    std_len = _std(lengths) if len(lengths) > 1 else 0.0

    return outputs, QualityMetrics(
        avg_output_length=round(avg_len, 1),
        output_length_std=round(std_len, 1),
    )


def compute_similarity(outputs_a: list[str], outputs_b: list[str]) -> float:
    """Compute average word-overlap similarity between two sets of outputs.

    Uses Jaccard similarity on word sets as a lightweight proxy for
    semantic similarity (avoids heavy embedding dependencies).
    """
    if len(outputs_a) != len(outputs_b):
        return 0.0

    similarities: list[float] = []
    for a, b in zip(outputs_a, outputs_b):
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a and not words_b:
            similarities.append(1.0)
            continue
        union = words_a | words_b
        intersection = words_a & words_b
        similarities.append(len(intersection) / len(union) if union else 0.0)

    return round(sum(similarities) / len(similarities), 4) if similarities else 0.0


def estimate_perplexity(
    backend: InferenceBackend,
    eval_texts: list[str],
) -> float | None:
    """Estimate perplexity using the backend's log-probability output.

    Not all backends support this. Returns None if log probs unavailable.
    """
    total_log_prob = 0.0
    total_tokens = 0

    for text in eval_texts:
        result = backend.generate(text, max_tokens=1)
        if result.log_probs is not None and result.log_probs:
            total_log_prob += sum(result.log_probs)
            total_tokens += len(result.log_probs)

    if total_tokens == 0:
        return None

    avg_neg_log_prob = -total_log_prob / total_tokens
    return round(math.exp(avg_neg_log_prob), 4)


def _std(values: list[int | float]) -> float:
    """Standard deviation of a list of numbers."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)
