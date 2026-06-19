from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pointcloud_geolab import api as api_module
from pointcloud_geolab.api import (
    run_benchmark,
    run_icp,
    run_plane_segmentation,
    run_preprocessing,
)
from pointcloud_geolab.io.pointcloud_io import save_point_cloud


def test_run_icp_returns_structured_result(tmp_path: Path) -> None:
    axis = np.linspace(-0.5, 0.5, 4)
    target = np.asarray(np.meshgrid(axis, axis, axis), dtype=float).reshape(3, -1).T
    source = target + np.asarray([0.02, -0.01, 0.015])
    source_path = tmp_path / "source.ply"
    target_path = tmp_path / "target.ply"
    output_dir = tmp_path / "icp_result"
    save_point_cloud(source_path, source)
    save_point_cloud(target_path, target)

    result = run_icp(
        source_path,
        target_path,
        output_dir=output_dir,
        max_iterations=30,
        tolerance=1e-10,
    )

    assert result.success
    assert result.metrics["final_rmse"] < 1e-8
    assert result.artifacts["metrics_json"] == str(output_dir / "metrics.json")
    assert (output_dir / "metrics.json").exists()


def test_run_plane_segmentation_returns_metrics(tmp_path: Path) -> None:
    rng = np.random.default_rng(21)
    xy = rng.uniform(-1, 1, size=(120, 2))
    points = np.column_stack([xy, np.zeros(120)])
    input_path = tmp_path / "room.pcd"
    output_dir = tmp_path / "plane_result"
    save_point_cloud(input_path, points)

    result = run_plane_segmentation(
        input_path,
        output_dir=output_dir,
        threshold=0.005,
        max_iterations=100,
        seed=21,
    )

    assert result.success
    assert result.metrics["inlier_ratio"] == 1.0
    assert (output_dir / "metrics.json").exists()


def test_run_preprocessing_writes_output_and_metrics(tmp_path: Path) -> None:
    rng = np.random.default_rng(22)
    points = rng.normal(size=(90, 3))
    input_path = tmp_path / "input.ply"
    output_path = tmp_path / "clean.ply"
    output_dir = tmp_path / "preprocess_result"
    save_point_cloud(input_path, points)

    result = run_preprocessing(
        input_path,
        output=output_path,
        output_dir=output_dir,
        voxel_size=0.0,
    )

    assert result.success
    assert result.metrics["original_points"] == len(points)
    assert output_path.exists()
    assert (output_dir / "metrics.json").exists()


def test_run_preprocessing_default_metrics_are_complete(tmp_path: Path) -> None:
    points = np.asarray([[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]])
    input_path = tmp_path / "tiny.ply"
    save_point_cloud(input_path, points)

    result = run_preprocessing(input_path, output_dir=tmp_path / "preprocess_default")

    assert result.success, result.error
    for key in [
        "original_points",
        "after_crop",
        "after_voxel_downsample",
        "after_statistical_filter",
        "after_radius_filter",
        "after_sampling",
        "kept_statistical_inliers",
        "final_points",
    ]:
        assert key in result.metrics
    assert result.metrics["after_crop"] == 2
    assert result.metrics["after_voxel_downsample"] == 2
    assert result.metrics["after_radius_filter"] == 2
    assert result.metrics["after_sampling"] == 2


def test_run_preprocessing_disabled_optional_steps_keep_metrics(tmp_path: Path) -> None:
    points = np.asarray([[0.0, 0.0, 0.0], [0.2, 0.1, 0.0], [0.4, 0.0, 0.2]])
    input_path = tmp_path / "small.xyz"
    save_point_cloud(input_path, points)

    result = run_preprocessing(
        input_path,
        output_dir=tmp_path / "preprocess_disabled",
        voxel_size=0.0,
        statistical_nb_neighbors=0,
        radius=0.0,
        sample_count=None,
        crop_min=None,
        crop_max=None,
    )

    assert result.success, result.error
    assert result.metrics["original_points"] == 3
    assert result.metrics["after_crop"] == 3
    assert result.metrics["after_voxel_downsample"] == 3
    assert result.metrics["after_statistical_filter"] == 3
    assert result.metrics["after_radius_filter"] == 3
    assert result.metrics["after_sampling"] == 3
    assert result.metrics["kept_statistical_inliers"] == 3


def test_run_benchmark_kdtree_small_case(tmp_path: Path) -> None:
    output_dir = tmp_path / "benchmark_result"
    md_path = tmp_path / "benchmark.md"
    json_path = tmp_path / "benchmark.json"

    result = run_benchmark(
        "kdtree",
        output_dir=output_dir,
        quick=True,
        points=[100],
        queries=5,
        save_json=json_path,
        save_md=md_path,
    )

    assert result.success
    assert result.metrics["cases"] == 1
    assert result.metrics["all_correct"] is True
    assert md_path.exists()
    assert json_path.exists()
    assert result.artifacts["benchmark_json_custom"] == str(json_path)
    assert (output_dir / "metrics.json").exists()


def test_run_benchmark_repeat_records_stats_and_memory(tmp_path: Path) -> None:
    output_dir = tmp_path / "benchmark_repeat"

    result = run_benchmark(
        "kdtree",
        output_dir=output_dir,
        quick=True,
        points=[60],
        queries=4,
        repeat=2,
    )

    assert result.success, result.error
    assert result.metrics["repeat"] == 2
    assert result.metrics["peak_memory_bytes"] >= 0
    row = result.data["rows"][0]
    assert row["repeat_count"] == 2
    for aggregate in ["mean", "std", "min", "max"]:
        assert f"kd_time_{aggregate}" in row
    payload = json.loads((output_dir / "kdtree_benchmark.json").read_text(encoding="utf-8"))
    assert payload["metadata"]["repeat"]["count"] == 2
    assert "kd_time" in payload["metadata"]["repeat"]["timing_fields"]
    assert payload["metadata"]["memory"]["available"] is True
    assert payload["metadata"]["memory"]["method"] == "tracemalloc"


def test_run_benchmark_ransac_repeat_records_timing_stats(tmp_path: Path) -> None:
    result = run_benchmark("ransac", output_dir=tmp_path, quick=True, repeat=2)

    assert result.success, result.error
    row = result.data["rows"][0]
    assert row["repeat_count"] == 2
    assert "custom_ransac_time_mean" in row
    payload = json.loads((tmp_path / "ransac_benchmark.json").read_text(encoding="utf-8"))
    assert "custom_ransac_time" in payload["metadata"]["repeat"]["timing_fields"]


def test_run_benchmark_registration_repeat_records_timing_stats(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeGlobalRegistration:
        refined_transform = np.eye(4)

    monkeypatch.setattr(
        api_module,
        "register_fpfh_ransac_icp",
        lambda *args, **kwargs: FakeGlobalRegistration(),
    )
    monkeypatch.setattr(
        api_module,
        "evaluate_registration",
        lambda *args, **kwargs: {"rmse": 0.0, "fitness": 1.0},
    )

    result = run_benchmark("registration", output_dir=tmp_path, quick=True, repeat=2)

    assert result.success, result.error
    rows = result.data["rows"]
    assert any("custom_icp_time_mean" in row for row in rows)
    assert any("global_registration_time_mean" in row for row in rows)
    payload = json.loads((tmp_path / "registration_benchmark.json").read_text(encoding="utf-8"))
    assert payload["metadata"]["repeat"]["timing_fields"] == ["runtime"]


def test_run_benchmark_all_writes_summary_metadata(tmp_path: Path) -> None:
    result = run_benchmark(
        "all",
        output_dir=tmp_path,
        quick=True,
        points=[60],
        queries=4,
    )

    assert result.success, result.error
    assert result.metrics["benchmark"] == "all"
    assert result.metrics["repeat"] == 1
    summary = json.loads((tmp_path / "benchmark_summary.json").read_text(encoding="utf-8"))
    assert len(summary["suites"]) == 6
    assert summary["metadata"]["repeat"]["count"] == 1
    assert summary["metadata"]["memory"]["available"] is True


def test_run_benchmark_all_repeat_matches_suite_repeat_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run_once(**kwargs):
        assert kwargs["benchmark"] == "all"
        rows = [
            {
                "suite": "kdtree",
                "kd_time": 0.001,
                "repeat_count": 2,
                "kd_time_mean": 0.001,
                "kd_time_std": 0.0,
                "kd_time_min": 0.001,
                "kd_time_max": 0.001,
            },
            {
                "suite": "icp",
                "icp_time": 0.01,
                "repeat_count": 2,
                "icp_time_mean": 0.01,
                "icp_time_std": 0.0,
                "icp_time_min": 0.01,
                "icp_time_max": 0.01,
            },
        ]
        return (
            rows,
            "| suite | time |\n|---|---:|",
            "all done",
            [{"suite": "kdtree", "cases": 1, "repeat": 2, "conclusion": "ok"}],
        )

    monkeypatch.setattr("pointcloud_geolab.api._run_benchmark_once", fake_run_once)

    result = run_benchmark(
        "all",
        output_dir=tmp_path,
        quick=True,
        points=[40],
        queries=3,
        repeat=2,
    )

    assert result.success, result.error
    payload = json.loads((tmp_path / "all_benchmark.json").read_text(encoding="utf-8"))
    assert payload["metadata"]["repeat"]["count"] == 2
    assert payload["metadata"]["memory"]["method"] == "tracemalloc"
    assert payload["metadata"]["repeat"]["statistics"]["aggregates"] == [
        "mean",
        "std",
        "min",
        "max",
    ]
    for row in payload["rows"]:
        assert row["repeat_count"] == 2
        base_timing_fields = [
            key
            for key, value in row.items()
            if (key.endswith("_time") or key in {"runtime", "total_seconds"})
            and isinstance(value, (int, float))
        ]
        assert base_timing_fields
        for field in base_timing_fields:
            for aggregate in ["mean", "std", "min", "max"]:
                assert f"{field}_{aggregate}" in row
