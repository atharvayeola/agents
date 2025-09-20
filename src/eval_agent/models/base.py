"""Base interfaces for model adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List, Sequence

from eval_agent.types import Example, ModelResponse


class ModelAdapter(ABC):
    """Abstract base class for model integrations."""

    def __init__(self, *, name: str | None = None) -> None:
        self.name = name or self.__class__.__name__

    @abstractmethod
    def predict(self, example: Example) -> ModelResponse:
        """Run inference for a single example."""

    def predict_batch(self, examples: Sequence[Example]) -> List[ModelResponse]:
        return [self.predict(example) for example in examples]

    def warmup(self, examples: Iterable[Example] | None = None) -> None:
        """Hook for preparing the model before evaluation."""

        _ = examples
