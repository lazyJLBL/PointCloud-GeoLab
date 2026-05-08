from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from pointcloud_geolab.api import TaskResult
from pointcloud_geolab.io.pointcloud_io import save_point_cloud
from pointcloud_geolab.pipeline import run_portfolio_pipeline
from pointcloud_geolab.utils.transform import apply_transform, rotation_matrix_from_euler

ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "pointcloud_geolab", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def make_pipeline_input(input_dir: Path) -> None:
    input_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(123)
    cluster_a = rng.normal(loc=[0.0, 0.0, 0.0], scale=0.015, size=(45, 3))
    cluster_b = rng.normal(loc=[0.7, 0.0, 0.1], scale=0.015, size=(45, 3))
    slab = rng.uniform([-0.25, -0.18, -0.02], [0.25, 0.18, 0.02], size=(50, 3))
    main_cloud = np.vstack([cluster_a, cluster_b, slab])

    target = rng.normal(loc=[0.0, 0.0, 0.0], scale=[0.25, 0.09, 0.12], size=(100, 3))
    rotation = rotation_matrix_from_euler(0.04, -0.03, 0.05)
    translation = np.asarray([0.05, -0.03, 0.02])
    source = apply_transform(target, rotation, translation)

    segmentation_noise = rng.uniform([-0.4, -0.4, -0.2], [1.1, 0.4, 0.4], size=(8, 3))
    segmentation_cloud = np.vstack([cluster_a, cluster_b, segmentation_noise])

    save_point_cloud(input_dir / "object.ply", main_cloud)
    save_point_cloud(input_dir / "bunny_source.ply", source)
    save_point_cloud(input_dir / "bunny_target.ply", target)
    save_point_cloud(input_dir / "lidar_scene.ply", segmentation_cloud)


def run_pipeline(tmp_path: Path) -> tuple[subprocess.CompletedProcess[str], Path]:
    input_dir = tmp_path / "demo_data"
    output_dir = tmp_path / "portfolio_demo"
    make_pipeline_input(input_dir)
    completed = run_cli(
        "pipeline",
        "--input",
        str(input_dir),
        "--output",
        str(output_dir),
        "--eps",
        "0.08",
        "--min-points",
        "5",
        "--format",
        "json",
    )
    return completed, output_dir


def run_pipeline_direct(tmp_path: Path) -> tuple[TaskResult, Path]:
    input_dir = tmp_path / "demo_data"
    output_dir = tmp_path / "portfolio_demo"
    make_pipeline_input(input_dir)
    result = run_portfolio_pipeline(
        input_path=input_dir,
        output_dir=output_dir,
        eps=0.08,
        min_points=5,
    )
    return result, output_dir


def test_pipeline_cli_help() -> None:
    completed = run_cli("pipeline", "--help")

    assert completed.returncode == 0
    assert "run the portfolio demo pipeline" in completed.stdout
    assert "--input" in completed.stdout
    assert "--output" in completed.stdout


def test_pipeline_smoke(tmp_path: Path) -> None:
    completed, output_dir = run_pipeline(tmp_path)

    assert completed.returncode == 0, completed.stderr
    assert (output_dir / "report.md").exists()
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "figures").is_dir()
    assert (output_dir / "artifacts" / "processed_cloud.ply").exists()
    assert (output_dir / "artifacts" / "transformation.json").exists()
    for name in [
        "raw_pointcloud.png",
        "downsampled.png",
        "registration_before_after.png",
        "segmentation_result.png",
        "bounding_box_or_normals.png",
    ]:
        assert (output_dir / "figures" / name).exists()


def test_metrics_schema(tmp_path: Path) -> None:
    result, output_dir = run_pipeline_direct(tmp_path)

    assert result.success, result.error
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    for section in ["input", "preprocessing", "registration", "segmentation", "runtime"]:
        assert section in metrics
    assert {"num_points", "bounds"} <= set(metrics["input"])
    assert {"num_points_before", "num_points_after", "downsample_ratio"} <= set(
        metrics["preprocessing"]
    )
    assert {"rmse_before", "rmse_after", "transformation"} <= set(metrics["registration"])
    assert {"num_clusters", "cluster_sizes", "noise_ratio"} <= set(metrics["segmentation"])
    assert metrics["runtime"]["total_seconds"] >= 0
    assert metrics["registration"]["rmse_after"] <= metrics["registration"]["rmse_before"]


def test_pipeline_does_not_write_to_repo_root(tmp_path: Path) -> None:
    forbidden_paths = [
        ROOT / "report.md",
        ROOT / "metrics.json",
        ROOT / "figures",
        ROOT / "artifacts",
        ROOT / "processed_cloud.ply",
        ROOT / "transformation.json",
    ]
    before = {path: path.stat().st_mtime_ns if path.exists() else None for path in forbidden_paths}

    result, output_dir = run_pipeline_direct(tmp_path)

    assert result.success, result.error
    after = {path: path.stat().st_mtime_ns if path.exists() else None for path in forbidden_paths}
    assert after == before
    for artifact_path in result.artifacts.values():
        assert Path(artifact_path).resolve().is_relative_to(output_dir.resolve())
