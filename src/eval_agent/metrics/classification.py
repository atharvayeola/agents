"""Classification metrics."""

from __future__ import annotations

from collections import Counter
from typing import Dict, Sequence

from eval_agent.metrics.base import Metric
from eval_agent.registry import METRIC_REGISTRY
from eval_agent.types import Example, MetricResult, ModelResponse


@METRIC_REGISTRY.register("accuracy")
class AccuracyMetric(Metric):
    """Compute simple accuracy."""

    def compute(
        self,
        *,
        examples: Sequence[Example],
        responses: Sequence[ModelResponse],
    ) -> MetricResult:
        correct = 0
        total = len(responses)
        for example, response in zip(examples, responses):
            if example.expected_output == response.output:
                correct += 1
        value = (correct / total) if total else 0.0
        details = {"correct": correct, "total": total}
        return MetricResult(name=self.name, value=value, details=details)


@METRIC_REGISTRY.register("label-distribution")
class LabelDistributionMetric(Metric):
    """Summarize the distribution of predicted and reference labels."""

    def compute(
        self,
        *,
        examples: Sequence[Example],
        responses: Sequence[ModelResponse],
    ) -> MetricResult:
        gold_counter: Counter[str] = Counter()
        pred_counter: Counter[str] = Counter()
        for example, response in zip(examples, responses):
            gold_counter[str(example.expected_output)] += 1
            pred_counter[str(response.output)] += 1
        details: Dict[str, Dict[str, int]] = {
            "gold": dict(sorted(gold_counter.items())),
            "predicted": dict(sorted(pred_counter.items())),
        }
        return MetricResult(name=self.name, value=1.0, details=details)
