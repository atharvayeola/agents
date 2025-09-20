"""Common data structures used across the evaluation agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(slots=True)
class Example:
    """Represents a single evaluation example."""

    uid: str
    inputs: Dict[str, Any]
    expected_output: Any
    metadata: Dict[str, Any] = field(default_factory=dict)

    def text(self) -> str:
        """Return the primary text input if available."""

        value = self.inputs.get("text")
        if isinstance(value, str):
            return value
        raise KeyError("Example does not contain a 'text' field in inputs.")


@dataclass(slots=True)
class ModelResponse:
    """Represents the output from a model adapter."""

    uid: str
    output: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PredictionRecord:
    """Stores the combined information about an evaluation prediction."""

    uid: str
    inputs: Dict[str, Any]
    expected_output: Any
    predicted_output: Any
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uid": self.uid,
            "inputs": self.inputs,
            "expected_output": self.expected_output,
            "predicted_output": self.predicted_output,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class MetricResult:
    """Container for the outcome of a metric computation."""

    name: str
    value: float
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "value": self.value, "details": self.details}
