"""Core evaluation agent runner."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Sequence

from eval_agent.config import EvaluationConfig, MetricConfig
from eval_agent.registry import DATASET_REGISTRY, METRIC_REGISTRY, MODEL_REGISTRY, TASK_REGISTRY
from eval_agent.types import MetricResult, PredictionRecord

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    name: str
    task: str
    metrics: Sequence[MetricResult]
    predictions: Sequence[PredictionRecord]
    started_at: datetime
    completed_at: datetime
    output_path: Path | None = None

    def to_dict(self, *, include_predictions: bool = True) -> dict:
        payload = {
            "name": self.name,
            "task": self.task,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "metrics": [metric.to_dict() for metric in self.metrics],
            "output_path": str(self.output_path) if self.output_path else None,
        }
        if include_predictions:
            payload["predictions"] = [prediction.to_dict() for prediction in self.predictions]
        return payload


class EvaluationAgent:
    """Orchestrates dataset loading, model execution, and metric computation."""

    def __init__(self, config: EvaluationConfig) -> None:
        self.config = config

    def run(self) -> EvaluationResult:
        logger.info("Starting evaluation for config '%s'", self.config.name)
        dataset = DATASET_REGISTRY.create(
            self.config.dataset.type,
            **self.config.dataset.parameters,
        )
        model = MODEL_REGISTRY.create(self.config.model.type, **self.config.model.parameters)
        task = TASK_REGISTRY.create(self.config.task, dataset, model)

        metric_instances = self._build_metrics(self.config.metrics)

        start_time = datetime.now(timezone.utc)
        responses = task.run()
        completed_time = datetime.now(timezone.utc)

        metrics: List[MetricResult] = []
        for metric in metric_instances:
            metrics.append(metric.compute(examples=dataset.examples(), responses=responses))

        predictions = [
            PredictionRecord(
                uid=example.uid,
                inputs=example.inputs,
                expected_output=example.expected_output,
                predicted_output=response.output,
                metadata=response.metadata,
            )
            for example, response in zip(dataset.examples(), responses)
        ]

        result = EvaluationResult(
            name=self.config.name,
            task=self.config.task,
            metrics=metrics,
            predictions=predictions,
            started_at=start_time,
            completed_at=completed_time,
        )

        output_path = self._persist_results(result)
        result.output_path = output_path

        logger.info("Completed evaluation for config '%s'", self.config.name)
        return result

    def _build_metrics(self, configs: Sequence[MetricConfig]):
        return [
            METRIC_REGISTRY.create(
                config.type,
                name=config.name,
                **config.parameters,
            )
            for config in configs
        ]

    def _persist_results(self, result: EvaluationResult) -> Path | None:
        output_dir = self.config.output.directory
        if not output_dir:
            return None
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = result.completed_at.strftime("%Y%m%dT%H%M%S")
        file_path = output_dir / f"{self.config.name}_{timestamp}.json"
        payload = result.to_dict(include_predictions=self.config.output.save_predictions)
        with file_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return file_path
