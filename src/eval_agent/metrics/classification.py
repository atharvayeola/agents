"""Classification metrics."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, Sequence

from eval_agent.metrics.base import Metric
from eval_agent.registry import METRIC_REGISTRY
from eval_agent.types import Example, MetricResult, ModelResponse


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _gather_classification_stats(
    examples: Sequence[Example],
    responses: Sequence[ModelResponse],
):
    labels_set: set[str] = set()
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    support = defaultdict(int)
    confusion = defaultdict(lambda: defaultdict(int))

    for example, response in zip(examples, responses):
        gold = str(example.expected_output)
        pred = str(response.output)
        labels_set.add(gold)
        labels_set.add(pred)
        support[gold] += 1
        confusion[gold][pred] += 1
        if gold == pred:
            tp[gold] += 1
        else:
            fn[gold] += 1
            fp[pred] += 1

    labels = sorted(labels_set)
    totals = {
        "tp": sum(tp.values()),
        "fp": sum(fp.values()),
        "fn": sum(fn.values()),
    }
    return labels, tp, fp, fn, support, confusion, totals


def _aggregate_scores(
    per_label: Dict[str, float],
    support: Dict[str, int],
    average: str,
    *,
    totals: Dict[str, int],
    score: str,
) -> float:
    labels = list(per_label.keys())
    if not labels:
        return 0.0

    if average == "macro":
        return sum(per_label[label] for label in labels) / len(labels)
    if average == "weighted":
        total_support = sum(support.get(label, 0) for label in labels)
        if not total_support:
            return 0.0
        return sum(per_label[label] * support.get(label, 0) for label in labels) / total_support
    if average == "micro":
        tp = totals["tp"]
        fp = totals["fp"]
        fn = totals["fn"]
        if score == "precision":
            return _safe_div(tp, tp + fp)
        if score == "recall":
            return _safe_div(tp, tp + fn)
        if score == "f1":
            precision_micro = _safe_div(tp, tp + fp)
            recall_micro = _safe_div(tp, tp + fn)
            if precision_micro + recall_micro == 0:
                return 0.0
            return 2 * precision_micro * recall_micro / (precision_micro + recall_micro)
        raise ValueError(f"Unsupported score '{score}' for micro average")
    raise ValueError(f"Unsupported average '{average}'. Use 'macro', 'weighted', or 'micro'.")


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


@METRIC_REGISTRY.register("precision")
class PrecisionMetric(Metric):
    """Compute precision for classification tasks."""

    def __init__(self, *, average: str = "macro", name: str | None = None) -> None:
        super().__init__(name=name)
        self.average = average

    def compute(
        self,
        *,
        examples: Sequence[Example],
        responses: Sequence[ModelResponse],
    ) -> MetricResult:
        labels, tp, fp, fn, support, _confusion, totals = _gather_classification_stats(examples, responses)
        per_label = {label: _safe_div(tp[label], tp[label] + fp[label]) for label in labels}
        value = _aggregate_scores(per_label, support, self.average, totals=totals, score="precision")
        details = {
            "average": self.average,
            "per_label": per_label,
            "support": {label: int(support[label]) for label in labels},
        }
        return MetricResult(name=self.name, value=value, details=details)


@METRIC_REGISTRY.register("recall")
class RecallMetric(Metric):
    """Compute recall for classification tasks."""

    def __init__(self, *, average: str = "macro", name: str | None = None) -> None:
        super().__init__(name=name)
        self.average = average

    def compute(
        self,
        *,
        examples: Sequence[Example],
        responses: Sequence[ModelResponse],
    ) -> MetricResult:
        labels, tp, fp, fn, support, _confusion, totals = _gather_classification_stats(examples, responses)
        per_label = {label: _safe_div(tp[label], tp[label] + fn[label]) for label in labels}
        value = _aggregate_scores(per_label, support, self.average, totals=totals, score="recall")
        details = {
            "average": self.average,
            "per_label": per_label,
            "support": {label: int(support[label]) for label in labels},
        }
        return MetricResult(name=self.name, value=value, details=details)


@METRIC_REGISTRY.register("f1")
class F1ScoreMetric(Metric):
    """Compute the F1-score for classification tasks."""

    def __init__(self, *, average: str = "macro", name: str | None = None) -> None:
        super().__init__(name=name)
        self.average = average

    def compute(
        self,
        *,
        examples: Sequence[Example],
        responses: Sequence[ModelResponse],
    ) -> MetricResult:
        labels, tp, fp, fn, support, _confusion, totals = _gather_classification_stats(examples, responses)
        per_label_precision = {label: _safe_div(tp[label], tp[label] + fp[label]) for label in labels}
        per_label_recall = {label: _safe_div(tp[label], tp[label] + fn[label]) for label in labels}
        per_label = {}
        for label in labels:
            precision = per_label_precision[label]
            recall = per_label_recall[label]
            if precision + recall == 0:
                per_label[label] = 0.0
            else:
                per_label[label] = 2 * precision * recall / (precision + recall)
        value = _aggregate_scores(per_label, support, self.average, totals=totals, score="f1")
        details = {
            "average": self.average,
            "per_label": per_label,
            "support": {label: int(support[label]) for label in labels},
        }
        return MetricResult(name=self.name, value=value, details=details)


@METRIC_REGISTRY.register("confusion-matrix")
class ConfusionMatrixMetric(Metric):
    """Compute a confusion matrix."""

    def compute(
        self,
        *,
        examples: Sequence[Example],
        responses: Sequence[ModelResponse],
    ) -> MetricResult:
        labels, _tp, _fp, _fn, _support, confusion, _totals = _gather_classification_stats(examples, responses)
        matrix = [
            [int(confusion[gold].get(pred, 0)) for pred in labels]
            for gold in labels
        ]
        details = {
            "labels": labels,
            "matrix": matrix,
        }
        return MetricResult(name=self.name, value=1.0, details=details)


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
