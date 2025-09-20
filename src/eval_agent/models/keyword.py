"""Simple keyword-based text classifier for demonstration purposes."""

from __future__ import annotations

from typing import Iterable, Sequence

from eval_agent.models.base import ModelAdapter
from eval_agent.registry import MODEL_REGISTRY
from eval_agent.types import Example, ModelResponse


@MODEL_REGISTRY.register("keyword-matching")
class KeywordMatchingModel(ModelAdapter):
    """A lightweight rule-based classifier.

    Parameters
    ----------
    positive_keywords:
        Keywords mapped to the positive label.
    negative_keywords:
        Keywords mapped to the negative label.
    default_label:
        Fallback label when no keyword is matched.
    case_sensitive:
        Whether the matching should be case sensitive.
    priority:
        Order of evaluation between "positive" and "negative" categories. The first
        match wins.
    """

    def __init__(
        self,
        *,
        positive_keywords: Iterable[str] | None = None,
        negative_keywords: Iterable[str] | None = None,
        default_label: str = "neutral",
        case_sensitive: bool = False,
        priority: Sequence[str] | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.case_sensitive = case_sensitive
        self.positive_keywords = [kw if case_sensitive else kw.lower() for kw in (positive_keywords or [])]
        self.negative_keywords = [kw if case_sensitive else kw.lower() for kw in (negative_keywords or [])]
        self.default_label = default_label
        self.priority = list(priority or ("positive", "negative"))

    def _normalize(self, text: str) -> str:
        return text if self.case_sensitive else text.lower()

    def predict(self, example: Example) -> ModelResponse:
        text = example.inputs.get("text", "")
        normalized = self._normalize(text)
        match_label: str | None = None
        matched_keyword: str | None = None

        for category in self.priority:
            keywords = self.positive_keywords if category == "positive" else self.negative_keywords
            label = "positive" if category == "positive" else "negative"
            for keyword in keywords:
                if keyword in normalized:
                    match_label = label
                    matched_keyword = keyword
                    break
            if match_label is not None:
                break

        output = match_label if match_label is not None else self.default_label
        metadata = {
            "matched_keyword": matched_keyword,
            "matched_category": match_label,
            "model_name": self.name,
        }
        return ModelResponse(uid=example.uid, output=output, metadata=metadata)
