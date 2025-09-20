"""Pydantic schemas for the evaluation API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class MetricSchema(BaseModel):
    name: str
    value: float
    details: Dict[str, Any]


class PredictionSchema(BaseModel):
    uid: str
    inputs: Dict[str, Any]
    expected_output: Any
    predicted_output: Any
    metadata: Dict[str, Any]


class RunSummarySchema(BaseModel):
    id: int
    name: str
    task: str
    config_name: str
    config_path: str
    started_at: datetime
    completed_at: datetime
    duration: float = Field(..., description="Duration of the run in seconds")
    metrics: List[MetricSchema]


class RunDetailSchema(RunSummarySchema):
    predictions: List[PredictionSchema]


class RunCreateRequest(BaseModel):
    config: str = Field(..., description="Either a preset name or an absolute/relative path to a config file")
    save_predictions: Optional[bool] = Field(
        None,
        description="Override the configuration output.save_predictions flag.",
    )


class ConfigInfoSchema(BaseModel):
    name: str
    path: str
    description: Optional[str] = None
