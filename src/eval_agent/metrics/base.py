"""Metric abstractions for evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from eval_agent.types import Example, MetricResult, ModelResponse


class Metric(ABC):
    """Base metric interface."""

    def __init__(self, *, name: str | None = None) -> None:
        self.name = name or self.__class__.__name__.lower()

    @abstractmethod
    def compute(
        self,
        *,
        examples: Sequence[Example],
        responses: Sequence[ModelResponse],
    ) -> MetricResult:
        """Compute the metric for the provided examples and responses."""
