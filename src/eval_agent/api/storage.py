"""SQLite persistence for evaluation runs."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

from eval_agent.runner import EvaluationResult


@dataclass
class RunRecord:
    id: int
    name: str
    task: str
    config_name: str
    config_path: str
    started_at: datetime
    completed_at: datetime
    duration: float
    metrics: list[dict]
    predictions_path: str | None


class RunStore:
    """Lightweight SQLite-backed store for evaluation runs."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    task TEXT NOT NULL,
                    config_name TEXT NOT NULL,
                    config_path TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT NOT NULL,
                    duration REAL NOT NULL,
                    metrics_json TEXT NOT NULL,
                    predictions_path TEXT
                );
                """
            )
            connection.commit()

    def record_run(
        self,
        *,
        config_name: str,
        config_path: Path,
        result: EvaluationResult,
    ) -> int:
        metrics_json = json.dumps([metric.to_dict() for metric in result.metrics])
        predictions_path = str(result.output_path) if result.output_path else None
        duration = (result.completed_at - result.started_at).total_seconds()
        with sqlite3.connect(self.path) as connection:
            cursor = connection.execute(
                """
                INSERT INTO runs (
                    name,
                    task,
                    config_name,
                    config_path,
                    started_at,
                    completed_at,
                    duration,
                    metrics_json,
                    predictions_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.name,
                    result.task,
                    config_name,
                    str(config_path),
                    result.started_at.isoformat(),
                    result.completed_at.isoformat(),
                    duration,
                    metrics_json,
                    predictions_path,
                ),
            )
            connection.commit()
            return int(cursor.lastrowid)

    def list_runs(self) -> List[RunRecord]:
        with sqlite3.connect(self.path) as connection:
            cursor = connection.execute(
                """
                SELECT id, name, task, config_name, config_path, started_at, completed_at, duration, metrics_json, predictions_path
                FROM runs
                ORDER BY completed_at DESC
                """
            )
            rows = cursor.fetchall()
        return [self._row_to_record(row) for row in rows]

    def get_run(self, run_id: int) -> Optional[RunRecord]:
        with sqlite3.connect(self.path) as connection:
            cursor = connection.execute(
                """
                SELECT id, name, task, config_name, config_path, started_at, completed_at, duration, metrics_json, predictions_path
                FROM runs
                WHERE id = ?
                """,
                (run_id,),
            )
            row = cursor.fetchone()
        return self._row_to_record(row) if row else None

    def _row_to_record(self, row: Iterable) -> RunRecord:
        (
            run_id,
            name,
            task,
            config_name,
            config_path,
            started_at,
            completed_at,
            duration,
            metrics_json,
            predictions_path,
        ) = row
        metrics = json.loads(metrics_json) if metrics_json else []
        return RunRecord(
            id=int(run_id),
            name=str(name),
            task=str(task),
            config_name=str(config_name),
            config_path=str(config_path),
            started_at=datetime.fromisoformat(str(started_at)),
            completed_at=datetime.fromisoformat(str(completed_at)),
            duration=float(duration),
            metrics=metrics,
            predictions_path=str(predictions_path) if predictions_path else None,
        )
