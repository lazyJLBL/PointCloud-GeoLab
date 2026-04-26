from __future__ import annotations

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

    result = run_benchmark(
        "kdtree",
        output_dir=output_dir,
        quick=True,
        points=[100],
        queries=5,
        save_md=md_path,
    )

    assert result.success
    assert result.metrics["cases"] == 1
    assert result.metrics["all_correct"] is True
    assert md_path.exists()
    assert (output_dir / "metrics.json").exists()
