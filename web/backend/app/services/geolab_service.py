"""Adapter from Web Console task requests to the stable PointCloud-GeoLab API."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from pointcloud_geolab import (
    TaskResult,
    run_benchmark,
    run_extract_primitives,
    run_geometry_analysis,
    run_ground_object_segmentation,
    run_icp,
    run_multiscale_icp,
    run_plane_segmentation,
    run_portfolio_verification,
    run_preprocessing,
    run_primitive_fitting,
    run_robust_icp,
    run_segmentation,
)
from web.backend.app.storage import WebStorage


class GeolabService:
    """Run stable API tasks for Web Console requests."""

    def __init__(self, storage: WebStorage) -> None:
        self.storage = storage

    def run_task(
        self,
        task_type: str,
        request: dict[str, Any],
        artifacts_dir: Path,
    ) -> TaskResult:
        parameters = dict(request.get("parameters") or {})
        try:
            if task_type == "preprocessing":
                return self._run_single_dataset(
                    run_preprocessing,
                    request,
                    artifacts_dir,
                    parameters,
                    output=artifacts_dir / "preprocessed.ply",
                    save_results=True,
                )
            if task_type == "registration/icp":
                return self._run_registration(run_icp, request, artifacts_dir, parameters)
            if task_type == "registration/robust-icp":
                return self._run_registration(run_robust_icp, request, artifacts_dir, parameters)
            if task_type == "registration/multiscale-icp":
                return self._run_registration(
                    run_multiscale_icp,
                    request,
                    artifacts_dir,
                    parameters,
                    output=artifacts_dir / "aligned_source.ply",
                    save_diagnostics=artifacts_dir / "diagnostics.json",
                )
            if task_type == "segmentation":
                return self._run_single_dataset(
                    run_segmentation,
                    request,
                    artifacts_dir,
                    parameters,
                    output=artifacts_dir / "segmentation.ply",
                    export_html=artifacts_dir / "segmentation.html",
                )
            if task_type == "segmentation/ground-object":
                return self._run_single_dataset(
                    run_ground_object_segmentation,
                    request,
                    artifacts_dir,
                    parameters,
                    output=artifacts_dir / "ground_object.ply",
                    export_html=artifacts_dir / "ground_object.html",
                    export_report=artifacts_dir / "cluster_report.md",
                )
            if task_type == "geometry":
                return self._run_single_dataset(
                    run_geometry_analysis,
                    request,
                    artifacts_dir,
                    parameters,
                    save_results=True,
                )
            if task_type == "primitives/plane":
                return self._run_single_dataset(
                    run_plane_segmentation,
                    request,
                    artifacts_dir,
                    parameters,
                    save_results=True,
                )
            if task_type == "primitives/fit":
                model = str(parameters.pop("model", "plane"))
                return self._run_single_dataset(
                    run_primitive_fitting,
                    request,
                    artifacts_dir,
                    parameters,
                    model=model,
                    output=artifacts_dir / f"{model}_inliers.ply",
                    export_html=artifacts_dir / f"{model}_fit.html",
                )
            if task_type == "primitives/extract":
                return self._run_single_dataset(
                    run_extract_primitives,
                    request,
                    artifacts_dir,
                    parameters,
                    output=artifacts_dir / "primitives.ply",
                    export_html=artifacts_dir / "primitives.html",
                )
            if task_type == "benchmark":
                suite = str(parameters.pop("suite", parameters.pop("benchmark", "all")))
                parameters["quick"] = bool(parameters.pop("quick", True))
                parameters["full"] = False
                return run_benchmark(suite, output_dir=artifacts_dir, **parameters)
            if task_type == "portfolio":
                quick = bool(parameters.pop("quick", True))
                return run_portfolio_verification(output_dir=artifacts_dir, quick=quick)
            raise ValueError(f"unsupported task type: {task_type}")
        except Exception as exc:
            return TaskResult(
                task=task_type,
                success=False,
                parameters=request,
                error=str(exc),
                path=_request_path(request),
            )

    def _run_single_dataset(
        self,
        function: Callable[..., TaskResult],
        request: dict[str, Any],
        artifacts_dir: Path,
        parameters: dict[str, Any],
        **extra: Any,
    ) -> TaskResult:
        dataset = self.storage.get_dataset(str(request.get("dataset_id") or ""))
        return function(
            dataset.path,
            output_dir=artifacts_dir,
            **_allowed_parameters(function, parameters),
            **extra,
        )

    def _run_registration(
        self,
        function: Callable[..., TaskResult],
        request: dict[str, Any],
        artifacts_dir: Path,
        parameters: dict[str, Any],
        **extra: Any,
    ) -> TaskResult:
        source = self.storage.get_dataset(str(request.get("source_dataset_id") or ""))
        target = self.storage.get_dataset(str(request.get("target_dataset_id") or ""))
        return function(
            source.path,
            target.path,
            output_dir=artifacts_dir,
            **_allowed_parameters(function, parameters),
            **extra,
        )


def _allowed_parameters(
    function: Callable[..., TaskResult],
    parameters: dict[str, Any],
) -> dict[str, Any]:
    allowed = set(function.__annotations__)
    return {key: value for key, value in parameters.items() if key in allowed}


def _request_path(request: dict[str, Any]) -> str | None:
    for key in ("dataset_id", "source_dataset_id", "target_dataset_id"):
        value = request.get(key)
        if value:
            return str(value)
    return None
