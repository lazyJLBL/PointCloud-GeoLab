from __future__ import annotations

import json
from pathlib import Path

import numpy as np

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
