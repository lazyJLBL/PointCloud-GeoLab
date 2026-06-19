"""Runtime configuration for the experimental Web Console backend."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

ALLOWED_UPLOAD_EXTENSIONS = (".ply", ".pcd", ".xyz", ".txt", ".bin", ".off")


class Settings:
    """Small settings object intentionally kept dependency-light."""

    def __init__(self) -> None:
        self.output_root = Path(os.getenv("PCG_WEB_OUTPUT_ROOT", "outputs/web"))
        self.max_upload_bytes = int(os.getenv("PCG_WEB_MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
        self.preview_point_limit = int(os.getenv("PCG_WEB_PREVIEW_POINT_LIMIT", "10000"))
        self.allowed_extensions = ALLOWED_UPLOAD_EXTENSIONS
        self.cors_origins = [
            origin.strip()
            for origin in os.getenv(
                "PCG_WEB_CORS_ORIGINS",
                "http://localhost:5173,http://127.0.0.1:5173",
            ).split(",")
            if origin.strip()
        ]

    @property
    def uploads_root(self) -> Path:
        return self.output_root / "uploads"

    @property
    def tasks_root(self) -> Path:
        return self.output_root / "tasks"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings so tests can reset via cache_clear()."""

    return Settings()
