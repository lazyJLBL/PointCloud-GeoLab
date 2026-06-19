"""FastAPI entry point for the experimental PointCloud-GeoLab Web Console."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from web.backend.app.api import artifacts, datasets, health, tasks
from web.backend.app.config import Settings, get_settings
from web.backend.app.services.geolab_service import GeolabService
from web.backend.app.storage import WebStorage
from web.backend.app.tasks import TaskManager


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create a configured FastAPI application."""

    active_settings = settings or get_settings()
    storage = WebStorage(active_settings)
    service = GeolabService(storage)
    task_manager = TaskManager(storage, service)

    app = FastAPI(
        title="PointCloud-GeoLab Experimental Web Console",
        version="1.1.0",
        description=(
            "Experimental reviewer console for PointCloud-GeoLab. "
            "This is not a production LiDAR platform."
        ),
    )
    app.state.settings = active_settings
    app.state.storage = storage
    app.state.task_manager = task_manager

    app.add_middleware(
        CORSMiddleware,
        allow_origins=active_settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router, prefix="/api", tags=["health"])
    app.include_router(datasets.router, prefix="/api", tags=["datasets"])
    app.include_router(tasks.router, prefix="/api", tags=["tasks"])
    app.include_router(artifacts.router, prefix="/api", tags=["artifacts"])
    return app


app = create_app()
