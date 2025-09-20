"""Dataset abstractions for the evaluation agent."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, List

from eval_agent.types import Example


class Dataset(ABC):
    """Base class for datasets."""

    def __init__(self) -> None:
        self._cache: List[Example] | None = None

    def __iter__(self) -> Iterable[Example]:
        return iter(self.examples())

    def __len__(self) -> int:
        return len(self.examples())

    def examples(self) -> List[Example]:
        if self._cache is None:
            self._cache = list(self._load())
        return self._cache

    @abstractmethod
    def _load(self) -> Iterable[Example]:
        """Load the dataset examples."""

    @staticmethod
    def resolve_path(path: str | Path, *, base_dir: Path | None = None) -> Path:
        path = Path(path)
        if not path.is_absolute() and base_dir is not None:
            path = base_dir / path
        return path
