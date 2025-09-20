"""FastAPI application for orchestrating evaluation runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from fastapi.middleware.cors import CORSMiddleware

from eval_agent import EvaluationAgent, load_config
from eval_agent.api.schemas import (
    ConfigInfoSchema,
    MetricSchema,
    PredictionSchema,
    RunCreateRequest,
    RunDetailSchema,
    RunSummarySchema,
)
from eval_agent.api.storage import RunStore
from eval_agent.runner import EvaluationResult

BASE_DIR = Path(__file__).resolve().parents[2]
FRONTEND_DIST = BASE_DIR / "frontend" / "dist"
RUNS_DIR = BASE_DIR / "runs"
DEFAULT_DB_PATH = RUNS_DIR / "evaluations.db"

PRESET_CONFIGS = {
    "sentiment-keyword": BASE_DIR / "configs" / "sentiment_keyword.json",
    "sentiment-sklearn": BASE_DIR / "configs" / "sentiment_sklearn.json",
}

app = FastAPI(title="Evaluation Agent API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

store = RunStore(DEFAULT_DB_PATH)


if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
else:

    @app.get("/", include_in_schema=False)
    def serve_index() -> None:
        raise HTTPException(status_code=404, detail="Frontend bundle not found")


    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_frontend(full_path: str) -> None:
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API route not found")
        raise HTTPException(status_code=404, detail="Frontend bundle not found")

@app.on_event("startup")
def _startup() -> None:
    store.initialize()


def _resolve_config(config_identifier: str) -> tuple[str, Path]:
    if config_identifier in PRESET_CONFIGS:
        return config_identifier, PRESET_CONFIGS[config_identifier]

    candidate = Path(config_identifier)
    if not candidate.is_absolute():
        potential = BASE_DIR / candidate
        candidate = potential if potential.exists() else (Path.cwd() / candidate)
    candidate = candidate.resolve()
    if not candidate.exists():
        raise HTTPException(status_code=404, detail=f"Configuration '{config_identifier}' not found")
    return config_identifier, candidate


def _result_to_schema(
    *,
    run_id: int,
    config_name: str,
    config_path: Path,
    result: EvaluationResult,
) -> RunDetailSchema:
    metrics = [MetricSchema(**metric.to_dict()) for metric in result.metrics]
    predictions = [PredictionSchema(**prediction.to_dict()) for prediction in result.predictions]
    duration = (result.completed_at - result.started_at).total_seconds()
    return RunDetailSchema(
        id=run_id,
        name=result.name,
        task=result.task,
        config_name=config_name,
        config_path=str(config_path),
        started_at=result.started_at,
        completed_at=result.completed_at,
        duration=duration,
        metrics=metrics,
        predictions=predictions,
    )


def _record_to_summary(record) -> RunSummarySchema:
    metrics = [MetricSchema(**metric) for metric in record.metrics]
    return RunSummarySchema(
        id=record.id,
        name=record.name,
        task=record.task,
        config_name=record.config_name,
        config_path=record.config_path,
        started_at=record.started_at,
        completed_at=record.completed_at,
        duration=record.duration,
        metrics=metrics,
    )


def _record_to_detail(record) -> RunDetailSchema:
    predictions_payload = _load_predictions(record.predictions_path)
    metrics = [MetricSchema(**metric) for metric in record.metrics]
    predictions = [PredictionSchema(**item) for item in predictions_payload]
    return RunDetailSchema(
        id=record.id,
        name=record.name,
        task=record.task,
        config_name=record.config_name,
        config_path=record.config_path,
        started_at=record.started_at,
        completed_at=record.completed_at,
        duration=record.duration,
        metrics=metrics,
        predictions=predictions,
    )


def _load_predictions(path: str | None) -> List[dict]:
    if not path:
        return []
    candidate = Path(path)
    if not candidate.exists():
        return []
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    predictions = payload.get("predictions")
    if isinstance(predictions, list):
        return predictions
    return []


@app.get("/api/health")
def healthcheck() -> dict:
    return {"status": "ok"}


@app.get("/api/configs", response_model=List[ConfigInfoSchema])
def list_configs() -> List[ConfigInfoSchema]:
    return [
        ConfigInfoSchema(name=name, path=str(path), description=f"Preset config located at {path}")
        for name, path in PRESET_CONFIGS.items()
    ]


@app.get("/api/runs", response_model=List[RunSummarySchema])
def list_runs() -> List[RunSummarySchema]:
    records = store.list_runs()
    return [_record_to_summary(record) for record in records]


@app.get("/api/runs/{run_id}", response_model=RunDetailSchema)
def get_run(run_id: int) -> RunDetailSchema:
    record = store.get_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Run with id {run_id} not found")
    return _record_to_detail(record)


@app.post("/api/runs", response_model=RunDetailSchema)
def create_run(request: RunCreateRequest) -> RunDetailSchema:
    config_name, config_path = _resolve_config(request.config)
    config = load_config(config_path)
    if request.save_predictions is not None:
        config.output.save_predictions = request.save_predictions

    agent = EvaluationAgent(config)
    result = agent.run()

    run_id = store.record_run(config_name=config_name, config_path=config_path, result=result)
    return _result_to_schema(run_id=run_id, config_name=config_name, config_path=config_path, result=result)
