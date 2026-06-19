from __future__ import annotations

import importlib
import json
from pathlib import Path

import pointcloud_geolab as pcg
from pointcloud_geolab import api

STABLE_API = {
    "TaskResult",
    "run_benchmark",
    "run_extract_primitives",
    "run_geometry_analysis",
    "run_ground_object_segmentation",
    "run_icp",
    "run_multiscale_icp",
    "run_plane_segmentation",
    "run_portfolio_verification",
    "run_preprocessing",
    "run_primitive_fitting",
    "run_robust_icp",
    "run_segmentation",
}


def test_api_all_exports_only_stable_task_api() -> None:
    assert set(api.__all__) == STABLE_API


def test_package_top_level_reexports_stable_api() -> None:
    assert set(pcg.__all__) == STABLE_API | {"__version__"}
    for name in STABLE_API:
        assert getattr(pcg, name) is getattr(api, name)


def test_experimental_and_optional_wrappers_are_not_stable_exports() -> None:
    assert "run_feature_registration" not in api.__all__
    assert "run_global_registration" not in api.__all__
    assert "run_reconstruction" not in api.__all__
    assert "run_train_pointnet" not in api.__all__


def test_public_api_imports_are_explicit() -> None:
    module = importlib.import_module("pointcloud_geolab")

    assert module.__version__ == "1.0.0"
    assert callable(module.run_icp)
    assert module.TaskResult is api.TaskResult


def test_public_api_benchmark_error_path_is_structured(tmp_path: Path) -> None:
    result = api.run_benchmark("kdtree", output_dir=tmp_path, points=[20], queries=2, repeat=0)

    assert not result.success
    assert result.error == "repeat must be at least 1"
    assert result.task == "benchmark:kdtree"


def test_task_result_to_json_preserves_error_contract() -> None:
    result = api.TaskResult(
        task="demo",
        success=False,
        parameters={"input": "missing.ply"},
        error="missing file",
    )

    payload = json.loads(result.to_json())

    assert payload["task"] == "demo"
    assert payload["success"] is False
    assert payload["error"] == "missing file"
    assert payload["parameters"]["input"] == "missing.ply"
    assert payload["path"] == "missing.ply"
