"""JSONL dataset tailored for retrieval-augmented generation tasks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

from eval_agent.datasets.base import Dataset
from eval_agent.registry import DATASET_REGISTRY
from eval_agent.types import Example


def _load_context_store(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Context file not found at {path}")

    def _normalize_entry(entry: dict[str, object], fallback_id: int) -> tuple[str, str]:
        identifier = entry.get("id", fallback_id)
        text = entry.get("text") or entry.get("content")
        if text is None:
            raise ValueError(f"Context entry {identifier!r} is missing a 'text' field")
        return str(identifier), str(text)

    store: dict[str, str] = {}
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError("Each JSONL context entry must be an object")
                identifier, text = _normalize_entry(payload, idx)
                store[identifier] = text
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            for idx, entry in enumerate(payload):
                if not isinstance(entry, dict):
                    raise ValueError("Each context entry must be an object with 'id' and 'text'")
                identifier, text = _normalize_entry(entry, idx)
                store[identifier] = text
        elif isinstance(payload, dict):
            for key, value in payload.items():
                if not isinstance(value, str):
                    raise ValueError("Context dictionary values must be strings")
                store[str(key)] = value
        else:
            raise ValueError("Unsupported context format. Use JSONL, a list of objects, or a mapping.")

    return store


@DATASET_REGISTRY.register("jsonl-rag")
class JsonlRagDataset(Dataset):
    """Dataset that yields question/answer pairs with optional context identifiers."""

    def __init__(
        self,
        path: str | Path,
        *,
        contexts_path: str | Path | None = None,
        base_dir: Path | None = None,
    ) -> None:
        super().__init__()
        self.path = self.resolve_path(path, base_dir=base_dir)
        self.contexts_path = (
            self.resolve_path(contexts_path, base_dir=base_dir) if contexts_path is not None else None
        )
        self._context_store: dict[str, str] | None = None

    def _load(self) -> Iterable[Example]:
        with self.path.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError("Each example must be a JSON object")

                uid = str(payload.get("id", idx))
                question = payload.get("question") or payload.get("input") or payload.get("text")
                if question is None:
                    raise ValueError(f"Example {uid} is missing a 'question' field")

                expected = payload.get("answer") or payload.get("expected_answer")
                if expected is None:
                    raise ValueError(f"Example {uid} is missing an 'answer' field")

                context_ids_iter = payload.get("context_ids") or payload.get("contexts") or []
                if isinstance(context_ids_iter, (str, bytes)):
                    context_ids_iter = [context_ids_iter]
                context_ids: list[str] = [str(item) for item in context_ids_iter]

                metadata = {
                    key: value
                    for key, value in payload.items()
                    if key not in {"id", "question", "input", "text", "answer", "expected_answer"}
                }
                if context_ids:
                    metadata["context_ids"] = context_ids
                    metadata["reference_contexts"] = self._resolve_reference_contexts(context_ids)

                yield Example(
                    uid=uid,
                    inputs={"question": str(question), "text": str(question)},
                    expected_output=expected,
                    metadata=metadata,
                )

    def _resolve_reference_contexts(self, context_ids: Iterable[str]) -> list[str]:
        store = self.context_store
        resolved: list[str] = []
        for identifier in context_ids:
            if identifier in store:
                resolved.append(store[identifier])
        return resolved

    @property
    def context_store(self) -> dict[str, str]:
        if self._context_store is None:
            if self.contexts_path is None:
                self._context_store = {}
            else:
                self._context_store = _load_context_store(Path(self.contexts_path))
        return self._context_store

    def iter_contexts(self) -> Iterator[tuple[str, str]]:
        """Return an iterator over (context_id, text) pairs."""

        for key, value in self.context_store.items():
            yield key, value
