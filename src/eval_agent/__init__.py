"""Evaluation agent package initialization."""

from __future__ import annotations

from eval_agent.config import EvaluationConfig, load_config
from eval_agent.runner import EvaluationAgent, EvaluationResult

# Import modules for registry side-effects.
from eval_agent.datasets import jsonl as _datasets_jsonl  # noqa: F401
from eval_agent.metrics import classification as _metrics_classification  # noqa: F401
from eval_agent.models import keyword as _models_keyword  # noqa: F401
from eval_agent.models import sklearn as _models_sklearn  # noqa: F401
from eval_agent.tasks import classification as _tasks_classification  # noqa: F401

__all__ = [
    "EvaluationAgent",
    "EvaluationResult",
    "EvaluationConfig",
    "load_config",
]
