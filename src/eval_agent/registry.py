"""Simple registries to keep track of pluggable components."""

from __future__ import annotations

from typing import Any, Callable, Dict, Type, TypeVar

T = TypeVar("T")


class Registry:
    """A minimal registry implementation."""

    def __init__(self, *, name: str) -> None:
        self._name = name
        self._items: Dict[str, Type[Any]] = {}

    def register(self, key: str) -> Callable[[Type[T]], Type[T]]:
        def decorator(cls: Type[T]) -> Type[T]:
            if key in self._items:
                raise ValueError(f"{self._name} already contains an entry for '{key}'.")
            self._items[key] = cls
            return cls

        return decorator

    def create(self, key: str, *args: Any, **kwargs: Any) -> Any:
        try:
            cls = self._items[key]
        except KeyError as exc:
            available = ", ".join(sorted(self._items)) or "<empty>"
            raise KeyError(
                f"Unknown entry '{key}' for registry '{self._name}'. Available: {available}."
            ) from exc
        return cls(*args, **kwargs)

    def get(self, key: str) -> Type[Any]:
        try:
            return self._items[key]
        except KeyError as exc:
            available = ", ".join(sorted(self._items)) or "<empty>"
            raise KeyError(
                f"Unknown entry '{key}' for registry '{self._name}'. Available: {available}."
            ) from exc

    def keys(self) -> list[str]:
        return list(self._items.keys())


MODEL_REGISTRY = Registry(name="model")
DATASET_REGISTRY = Registry(name="dataset")
METRIC_REGISTRY = Registry(name="metric")
TASK_REGISTRY = Registry(name="task")
