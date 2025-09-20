"""Configuration loading for the evaluation agent."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class ModelConfig:
    type: str
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetConfig:
    type: str
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricConfig:
    type: str
    name: str | None = None
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OutputConfig:
    directory: Path
    save_predictions: bool = True


@dataclass
class EvaluationConfig:
    name: str
    task: str
    dataset: DatasetConfig
    model: ModelConfig
    metrics: List[MetricConfig]
    output: OutputConfig


def _resolve_path(value: str | Path, *, base_dir: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        return (base_dir / path).resolve()
    return path


def _resolve_parameter_paths(parameters: Dict[str, Any], *, base_dir: Path) -> Dict[str, Any]:
    resolved: Dict[str, Any] = {}
    for key, value in parameters.items():
        if isinstance(value, str) and (key.endswith("_path") or key.endswith("_dir") or key in {"path", "directory"}):
            resolved[key] = str(_resolve_path(value, base_dir=base_dir))
        else:
            resolved[key] = value
    return resolved


def load_config(path: str | Path) -> EvaluationConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    base_dir = path.parent

    model_params = raw["model"].get("parameters", {})
    model = ModelConfig(
        type=raw["model"]["type"],
        parameters=_resolve_parameter_paths(model_params, base_dir=base_dir),
    )

    dataset_params = raw["dataset"].get("parameters", {})
    dataset = DatasetConfig(
        type=raw["dataset"]["type"],
        parameters=_resolve_parameter_paths(dataset_params, base_dir=base_dir),
    )

    metrics = [
        MetricConfig(
            type=item["type"],
            name=item.get("name"),
            parameters=item.get("parameters", {}),
        )
        for item in raw.get("metrics", [])
    ]

    output_raw = raw.get("output", {})
    output_dir = output_raw.get("directory", "runs")
    output_config = OutputConfig(
        directory=_resolve_path(output_dir, base_dir=base_dir),
        save_predictions=output_raw.get("save_predictions", True),
    )

    return EvaluationConfig(
        name=raw["name"],
        task=raw["task"],
        dataset=dataset,
        model=model,
        metrics=metrics,
        output=output_config,
    )
