"""Classification task implementation."""

from __future__ import annotations

from typing import List

from eval_agent.registry import TASK_REGISTRY
from eval_agent.tasks.base import Task
from eval_agent.types import ModelResponse


@TASK_REGISTRY.register("text-classification")
class TextClassificationTask(Task):
    """Run text classification evaluations using the provided dataset and model."""

    def run(self) -> List[ModelResponse]:
        responses: List[ModelResponse] = []
        for example in self.dataset:
            responses.append(self.model.predict(example))
        return responses
