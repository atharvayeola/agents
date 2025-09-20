"""Task abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Sequence

from eval_agent.datasets.base import Dataset
from eval_agent.models.base import ModelAdapter
from eval_agent.types import Example, ModelResponse


class Task(ABC):
    """Base class for evaluation tasks."""

    def __init__(self, dataset: Dataset, model: ModelAdapter) -> None:
        self.dataset = dataset
        self.model = model

    @abstractmethod
    def run(self) -> Sequence[ModelResponse]:
        """Execute the evaluation task and return responses."""

    def warmup(self, examples: Iterable[Example] | None = None) -> None:
        self.model.warmup(examples)
