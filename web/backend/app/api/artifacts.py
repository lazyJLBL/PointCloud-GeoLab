"""Artifact download endpoint."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

router = APIRouter()


@router.get("/artifacts/{task_id}/{artifact_name}")
def get_artifact(task_id: str, artifact_name: str, request: Request) -> FileResponse:
    try:
        path = request.app.state.storage.artifact_path(task_id, artifact_name)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(path)
