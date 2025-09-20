"""Generation metrics for retrieval-augmented workflows."""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, Sequence

from eval_agent.metrics.base import Metric
from eval_agent.registry import METRIC_REGISTRY
from eval_agent.types import Example, MetricResult, ModelResponse


def _tokenize(text: str) -> list[str]:
    return [token for token in text.lower().split() if token]


def _lcs_length(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    if not reference or not hypothesis:
        return 0
    lengths = [[0] * (len(hypothesis) + 1) for _ in range(len(reference) + 1)]
    for i, ref_token in enumerate(reference, start=1):
        for j, hyp_token in enumerate(hypothesis, start=1):
            if ref_token == hyp_token:
                lengths[i][j] = lengths[i - 1][j - 1] + 1
            else:
                lengths[i][j] = max(lengths[i - 1][j], lengths[i][j - 1])
    return lengths[-1][-1]


@METRIC_REGISTRY.register("rouge-l")
class RougeLMetric(Metric):
    """Compute ROUGE-L F1 scores averaged across examples."""

    def compute(
        self,
        *,
        examples: Sequence[Example],
        responses: Sequence[ModelResponse],
    ) -> MetricResult:
        per_example: Dict[str, float] = {}
        for example, response in zip(examples, responses):
            reference_tokens = _tokenize(str(example.expected_output))
            hypothesis_tokens = _tokenize(str(response.output))
            if not reference_tokens and not hypothesis_tokens:
                score = 1.0
            else:
                lcs = _lcs_length(reference_tokens, hypothesis_tokens)
                precision = lcs / len(hypothesis_tokens) if hypothesis_tokens else 0.0
                recall = lcs / len(reference_tokens) if reference_tokens else 0.0
                score = 0.0
                if precision + recall > 0:
                    score = (2 * precision * recall) / (precision + recall)
            per_example[example.uid] = score

        value = sum(per_example.values()) / len(per_example) if per_example else 0.0
        details = {"per_example": per_example}
        return MetricResult(name=self.name, value=value, details=details)


def _modified_precision(candidate: list[str], reference: list[str], n: int) -> float:
    if len(candidate) < n:
        return 0.0
    candidate_counts = Counter(tuple(candidate[i : i + n]) for i in range(len(candidate) - n + 1))
    reference_counts = Counter(tuple(reference[i : i + n]) for i in range(len(reference) - n + 1))
    clipped = {
        ngram: min(count, reference_counts.get(ngram, 0)) for ngram, count in candidate_counts.items()
    }
    numerator = sum(clipped.values())
    denominator = sum(candidate_counts.values())
    if not denominator:
        return 0.0
    return numerator / denominator


def _brevity_penalty(candidate_len: int, reference_len: int) -> float:
    if candidate_len == 0:
        return 0.0
    if reference_len == 0 or candidate_len > reference_len:
        return 1.0
    return math.exp(1 - reference_len / candidate_len)


@METRIC_REGISTRY.register("bleu")
class BleuMetric(Metric):
    """Compute corpus-level BLEU with smoothing to avoid zero scores."""

    def __init__(self, *, max_n: int = 4, smoothing: float = 1e-9, name: str | None = None) -> None:
        super().__init__(name=name)
        self.max_n = max(1, max_n)
        self.smoothing = smoothing

    def compute(
        self,
        *,
        examples: Sequence[Example],
        responses: Sequence[ModelResponse],
    ) -> MetricResult:
        per_example: Dict[str, float] = {}
        for example, response in zip(examples, responses):
            reference_tokens = _tokenize(str(example.expected_output))
            candidate_tokens = _tokenize(str(response.output))
            if not candidate_tokens:
                per_example[example.uid] = 0.0
                continue

            precisions: list[float] = []
            for n in range(1, self.max_n + 1):
                precision = _modified_precision(candidate_tokens, reference_tokens, n)
                if precision <= 0:
                    precision = self.smoothing
                precisions.append(precision)

            geo_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
            bp = _brevity_penalty(len(candidate_tokens), len(reference_tokens))
            per_example[example.uid] = bp * geo_mean

        value = sum(per_example.values()) / len(per_example) if per_example else 0.0
        details = {"per_example": per_example, "max_n": self.max_n}
        return MetricResult(name=self.name, value=value, details=details)


@METRIC_REGISTRY.register("context-precision")
class ContextPrecisionMetric(Metric):
    """Approximate hallucination checks by measuring context token overlap."""

    def compute(
        self,
        *,
        examples: Sequence[Example],
        responses: Sequence[ModelResponse],
    ) -> MetricResult:
        per_example: Dict[str, float] = {}
        for example, response in zip(examples, responses):
            predicted_tokens = _tokenize(str(response.output))
            if not predicted_tokens:
                per_example[example.uid] = 0.0
                continue

            context_texts: list[str] = []
            metadata = response.metadata or {}
            retrieved = metadata.get("retrieved_documents")
            if isinstance(retrieved, list):
                for entry in retrieved:
                    if isinstance(entry, dict):
                        text = entry.get("text")
                        if isinstance(text, str):
                            context_texts.append(text)

            if not context_texts and example.metadata:
                references = example.metadata.get("reference_contexts")
                if isinstance(references, list):
                    context_texts.extend(str(item) for item in references)

            if not context_texts:
                per_example[example.uid] = 0.0
                continue

            context_tokens = set(_tokenize(" ".join(context_texts)))
            if not context_tokens:
                per_example[example.uid] = 0.0
                continue

            hits = sum(1 for token in predicted_tokens if token in context_tokens)
            per_example[example.uid] = hits / len(predicted_tokens)

        value = sum(per_example.values()) / len(per_example) if per_example else 0.0
        details = {"per_example": per_example}
        return MetricResult(name=self.name, value=value, details=details)
