"""Task execution endpoints for the experimental Web Console."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from web.backend.app.schemas import TaskCreateRequest, TaskRecord
from web.backend.app.tasks import TaskManager

router = APIRouter()


def get_manager(request: Request) -> TaskManager:
    return request.app.state.task_manager


@router.post("/tasks/preprocessing", response_model=TaskRecord)
def run_preprocessing_task(payload: TaskCreateRequest, request: Request) -> TaskRecord:
    return _run("preprocessing", payload, request)


@router.post("/tasks/registration/icp", response_model=TaskRecord)
def run_icp_task(payload: TaskCreateRequest, request: Request) -> TaskRecord:
    return _run("registration/icp", payload, request)


@router.post("/tasks/registration/robust-icp", response_model=TaskRecord)
def run_robust_icp_task(payload: TaskCreateRequest, request: Request) -> TaskRecord:
    return _run("registration/robust-icp", payload, request)


@router.post("/tasks/registration/multiscale-icp", response_model=TaskRecord)
def run_multiscale_icp_task(payload: TaskCreateRequest, request: Request) -> TaskRecord:
    return _run("registration/multiscale-icp", payload, request)


@router.post("/tasks/segmentation", response_model=TaskRecord)
def run_segmentation_task(payload: TaskCreateRequest, request: Request) -> TaskRecord:
    return _run("segmentation", payload, request)


@router.post("/tasks/segmentation/ground-object", response_model=TaskRecord)
def run_ground_object_task(payload: TaskCreateRequest, request: Request) -> TaskRecord:
    return _run("segmentation/ground-object", payload, request)


@router.post("/tasks/geometry", response_model=TaskRecord)
def run_geometry_task(payload: TaskCreateRequest, request: Request) -> TaskRecord:
    return _run("geometry", payload, request)


@router.post("/tasks/primitives/plane", response_model=TaskRecord)
def run_plane_task(payload: TaskCreateRequest, request: Request) -> TaskRecord:
    return _run("primitives/plane", payload, request)


@router.post("/tasks/primitives/fit", response_model=TaskRecord)
def run_fit_task(payload: TaskCreateRequest, request: Request) -> TaskRecord:
    return _run("primitives/fit", payload, request)


@router.post("/tasks/primitives/extract", response_model=TaskRecord)
def run_extract_task(payload: TaskCreateRequest, request: Request) -> TaskRecord:
    return _run("primitives/extract", payload, request)


@router.post("/tasks/benchmark", response_model=TaskRecord)
def run_benchmark_task(payload: TaskCreateRequest, request: Request) -> TaskRecord:
    return _run("benchmark", payload, request)


@router.post("/tasks/portfolio", response_model=TaskRecord)
def run_portfolio_task(payload: TaskCreateRequest, request: Request) -> TaskRecord:
    return _run("portfolio", payload, request)


@router.get("/tasks", response_model=list[TaskRecord])
def list_tasks(request: Request) -> list[TaskRecord]:
    return request.app.state.storage.list_tasks()


@router.get("/tasks/{task_id}", response_model=TaskRecord)
def get_task(task_id: str, request: Request) -> TaskRecord:
    try:
        return request.app.state.storage.get_task(task_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _run(task_type: str, payload: TaskCreateRequest, request: Request) -> TaskRecord:
    return get_manager(request).create_and_run(task_type, payload.model_dump())
