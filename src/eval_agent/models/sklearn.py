"""scikit-learn pipeline adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import joblib

from eval_agent.models.base import ModelAdapter
from eval_agent.registry import MODEL_REGISTRY
from eval_agent.types import Example, ModelResponse


@MODEL_REGISTRY.register("sklearn-pipeline")
class SklearnPipelineModel(ModelAdapter):
    """Model adapter that loads a serialized scikit-learn pipeline via joblib."""

    def __init__(
        self,
        *,
        artifact_path: str | Path,
        label_mapping: dict[str, str] | None = None,
        probability_field: str = "probabilities",
        name: str | None = None,
        warmup_examples: int = 0,
    ) -> None:
        super().__init__(name=name)
        self.artifact_path = Path(artifact_path)
        if not self.artifact_path.exists():
            raise FileNotFoundError(f"Model artifact not found at {self.artifact_path}")
        self.pipeline = joblib.load(self.artifact_path)
        self.label_mapping = {str(key): value for key, value in (label_mapping or {}).items()}
        self.probability_field = probability_field
        self.warmup_examples = warmup_examples
        self._classes: list[str] | None = None
        classes = getattr(self.pipeline, "classes_", None)
        if classes is not None:
            self._classes = [self._map_label(label) for label in classes]

    def _map_label(self, label: Any) -> Any:
        return self.label_mapping.get(str(label), label)

    def warmup(self, examples: Iterable[Example] | None = None) -> None:
        if not self.warmup_examples:
            return
        iterable = list(examples or [])[: self.warmup_examples]
        if not iterable:
            return
        texts = [example.text() for example in iterable]
        if hasattr(self.pipeline, "predict"):
            self.pipeline.predict(texts)
        if hasattr(self.pipeline, "predict_proba"):
            try:
                self.pipeline.predict_proba(texts)
            except Exception:
                # Some estimators raise when predict_proba is not available; ignore.
                pass

    def predict(self, example: Example) -> ModelResponse:
        text = example.text()
        raw_output = self.pipeline.predict([text])[0]
        mapped_output = self._map_label(raw_output)

        metadata: dict[str, Any] = {"model_name": self.name, "artifact_path": str(self.artifact_path)}
        if hasattr(self.pipeline, "predict_proba"):
            try:
                probabilities = self.pipeline.predict_proba([text])[0]
                classes = self._classes or getattr(self.pipeline, "classes_", None)
                if classes is not None:
                    metadata[self.probability_field] = {
                        str(self._map_label(label)): float(prob)
                        for label, prob in zip(classes, probabilities)
                    }
            except Exception:
                # Ignore probability errors for estimators that do not implement it.
                pass

        return ModelResponse(uid=example.uid, output=mapped_output, metadata=metadata)
