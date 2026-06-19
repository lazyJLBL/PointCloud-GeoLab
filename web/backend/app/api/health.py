"""Health endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from web.backend.app.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="pointcloud-geolab-web")
