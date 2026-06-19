"""Pydantic schemas for the experimental Web Console API."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """Task lifecycle states persisted by the Web Console."""

    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class HealthResponse(BaseModel):
    status: str
    service: str
    python_versions: str = "3.10-3.12"
    experimental: bool = True


class DatasetRecord(BaseModel):
    id: str
    filename: str
    original_filename: str
    path: str
    size_bytes: int
    extension: str
    created_at: str


class DatasetPreview(BaseModel):
    dataset_id: str
    point_count: int
    sampled_count: int
    points: list[list[float]]


class TaskCreateRequest(BaseModel):
    dataset_id: str | None = None
    source_dataset_id: str | None = None
    target_dataset_id: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class TaskRecord(BaseModel):
    id: str
    task_type: str
    status: TaskStatus
    request: dict[str, Any]
    result: dict[str, Any] | None = None
    error: str | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)
    created_at: str
    updated_at: str


class ErrorEnvelope(BaseModel):
    task: str
    success: bool = False
    error: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    path: str | None = None
