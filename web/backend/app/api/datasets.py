"""Dataset upload, listing, deletion, and preview endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from web.backend.app.schemas import DatasetPreview, DatasetRecord
from web.backend.app.services.preview_service import preview_point_cloud
from web.backend.app.storage import WebStorage, sanitize_filename

router = APIRouter()


def get_storage(request: Request) -> WebStorage:
    return request.app.state.storage


@router.post("/datasets/upload", response_model=DatasetRecord)
async def upload_dataset(
    request: Request,
    file: Annotated[UploadFile, File(...)],
) -> DatasetRecord:
    storage = get_storage(request)
    settings = request.app.state.settings
    try:
        original = sanitize_filename(file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    extension = Path(original).suffix.lower()
    if extension not in settings.allowed_extensions:
        raise HTTPException(status_code=400, detail=f"unsupported upload extension: {extension}")

    dataset_id, destination = storage.create_dataset_path(original)
    try:
        total = await write_upload_file(file, destination, settings.max_upload_bytes)
        if total == 0:
            raise HTTPException(status_code=400, detail="uploaded point cloud file is empty")
        return storage.save_dataset_record(dataset_id, original, original, destination, total)
    except HTTPException:
        storage.cleanup_dataset_dir(dataset_id)
        raise
    except Exception as exc:
        storage.cleanup_dataset_dir(dataset_id)
        raise HTTPException(status_code=500, detail=f"failed to store upload: {exc}") from exc


@router.get("/datasets", response_model=list[DatasetRecord])
def list_datasets(request: Request) -> list[DatasetRecord]:
    return get_storage(request).list_datasets()


@router.get("/datasets/{dataset_id}", response_model=DatasetRecord)
def get_dataset(dataset_id: str, request: Request) -> DatasetRecord:
    try:
        return get_storage(request).get_dataset(dataset_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/datasets/{dataset_id}")
def delete_dataset(dataset_id: str, request: Request) -> dict[str, object]:
    try:
        get_storage(request).delete_dataset(dataset_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"deleted": True, "dataset_id": dataset_id}


@router.get("/datasets/{dataset_id}/preview", response_model=DatasetPreview)
def preview_dataset(dataset_id: str, request: Request) -> DatasetPreview:
    storage = get_storage(request)
    settings = request.app.state.settings
    try:
        record = storage.get_dataset(dataset_id)
        preview = preview_point_cloud(record.path, settings.preview_point_limit)
        return DatasetPreview(dataset_id=dataset_id, **preview)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


async def write_upload_file(file: UploadFile, destination: Path, max_upload_bytes: int) -> int:
    """Write an upload with a hard size limit and return bytes written."""

    total = 0
    with destination.open("wb") as handle:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_upload_bytes:
                raise HTTPException(status_code=413, detail="upload exceeds size limit")
            handle.write(chunk)
    return total
