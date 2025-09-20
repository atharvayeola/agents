"""JSONL dataset implementations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from eval_agent.datasets.base import Dataset
from eval_agent.registry import DATASET_REGISTRY
from eval_agent.types import Example


@DATASET_REGISTRY.register("jsonl-classification")
class JsonlClassificationDataset(Dataset):
    """Dataset for text classification stored as JSON Lines."""

    def __init__(self, path: str | Path, *, base_dir: Path | None = None) -> None:
        super().__init__()
        self.path = self.resolve_path(path, base_dir=base_dir)

    def _load(self) -> Iterable[Example]:
        resolved_path = Path(self.path)
        with resolved_path.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                uid = str(payload.get("id", idx))
                text = payload.get("text") or payload.get("input")
                if text is None:
                    raise ValueError(
                        f"Example {uid} is missing a 'text' or 'input' field in {resolved_path}."
                    )
                expected = payload.get("label")
                if expected is None:
                    raise ValueError(
                        f"Example {uid} is missing a 'label' field in {resolved_path}."
                    )
                metadata = {
                    key: value
                    for key, value in payload.items()
                    if key not in {"id", "text", "input", "label"}
                }
                yield Example(
                    uid=uid,
                    inputs={"text": text},
                    expected_output=expected,
                    metadata=metadata,
                )
