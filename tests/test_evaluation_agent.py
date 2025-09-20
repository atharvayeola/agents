"""Integration tests for the evaluation agent."""

from __future__ import annotations

import json
from pathlib import Path

from eval_agent import EvaluationAgent, load_config


def test_keyword_model_evaluation(tmp_path: Path) -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "sentiment_keyword.json"
    config = load_config(config_path)
    config.output.directory = tmp_path

    agent = EvaluationAgent(config)
    result = agent.run()

    assert result.output_path is not None
    assert result.output_path.parent == tmp_path
    assert len(result.predictions) == 6

    accuracy = next(metric for metric in result.metrics if metric.name == "accuracy")
    assert accuracy.value == 1.0
    assert accuracy.details["total"] == 6
    assert accuracy.details["correct"] == 6

    metrics = {metric.name: metric for metric in result.metrics}
    assert metrics["precision_macro"].value == 1.0
    assert metrics["recall_macro"].value == 1.0
    assert metrics["f1_macro"].value == 1.0

    confusion = metrics["confusion_matrix"].details
    labels = confusion["labels"]
    matrix = confusion["matrix"]
    assert len(labels) == len(matrix)
    assert sum(matrix[i][i] for i in range(len(labels))) == len(result.predictions)

    saved_payload = json.loads(result.output_path.read_text(encoding="utf-8"))
    assert saved_payload["name"] == "sentiment-keyword-baseline"
    assert saved_payload["metrics"][0]["name"] == "accuracy"
    assert saved_payload["metrics"][0]["value"] == 1.0
    assert len(saved_payload["predictions"]) == 6
