from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from pointcloud_geolab.io.pointcloud_io import save_point_cloud

ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "main.py"), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_cli_geometry_smoke(tmp_path: Path) -> None:
    rng = np.random.default_rng(8)
    points = rng.normal(size=(80, 3))
    input_path = tmp_path / "object.ply"
    results_dir = tmp_path / "results"
    save_point_cloud(input_path, points)

    completed = run_cli(
        "--mode",
        "geometry",
        "--input",
        str(input_path),
        "--save-results",
        "--results-dir",
        str(results_dir),
    )

    assert completed.returncode == 0, completed.stderr
    assert "Point Cloud Geometry" in completed.stdout
    assert (results_dir / "obb_visualization.png").exists()


def test_cli_icp_smoke(tmp_path: Path) -> None:
    axis = np.linspace(-0.5, 0.5, 4)
    target = np.asarray(np.meshgrid(axis, axis, axis), dtype=float).reshape(3, -1).T
    source = target + np.asarray([0.02, 0.01, -0.015])
    source_path = tmp_path / "source.ply"
    target_path = tmp_path / "target.ply"
    save_point_cloud(source_path, source)
    save_point_cloud(target_path, target)

    completed = run_cli(
        "--mode",
        "icp",
        "--source",
        str(source_path),
        "--target",
        str(target_path),
        "--max-iterations",
        "20",
        "--results-dir",
        str(tmp_path / "legacy_icp_results"),
    )

    assert completed.returncode == 0, completed.stderr
    assert "ICP Registration Result" in completed.stdout
    assert "Final RMSE" in completed.stdout


def test_cli_plane_smoke(tmp_path: Path) -> None:
    rng = np.random.default_rng(9)
    xy = rng.uniform(-1, 1, size=(120, 2))
    points = np.column_stack([xy, np.zeros(120)])
    input_path = tmp_path / "room.pcd"
    save_point_cloud(input_path, points)

    completed = run_cli(
        "--mode",
        "plane",
        "--input",
        str(input_path),
        "--threshold",
        "0.005",
        "--max-iterations",
        "100",
        "--seed",
        "9",
        "--results-dir",
        str(tmp_path / "legacy_plane_results"),
    )

    assert completed.returncode == 0, completed.stderr
    assert "RANSAC Plane Fitting Result" in completed.stdout
    assert "Inlier Ratio" in completed.stdout


def test_cli_geometry_subcommand_json(tmp_path: Path) -> None:
    rng = np.random.default_rng(18)
    points = rng.normal(size=(60, 3))
    input_path = tmp_path / "object.ply"
    results_dir = tmp_path / "geometry_results"
    save_point_cloud(input_path, points)

    completed = run_cli(
        "geometry",
        "--input",
        str(input_path),
        "--output-dir",
        str(results_dir),
        "--format",
        "json",
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["success"] is True
    assert payload["task"] == "geometry"
    assert payload["metrics"]["point_count"] == len(points)
    assert (results_dir / "metrics.json").exists()


def test_cli_config_values_can_be_overridden(tmp_path: Path) -> None:
    rng = np.random.default_rng(19)
    points = rng.normal(size=(80, 3))
    input_path = tmp_path / "room.pcd"
    results_dir = tmp_path / "plane_results"
    config_path = tmp_path / "plane.yaml"
    save_point_cloud(input_path, points)
    config_path.write_text(
        "\n".join(
            [
                f"input: {input_path}",
                "threshold: 0.0001",
                "max_iterations: 10",
                f"output_dir: {results_dir}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    completed = run_cli(
        "plane",
        "--config",
        str(config_path),
        "--threshold",
        "0.2",
        "--format",
        "json",
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["success"] is True
    assert payload["parameters"]["threshold"] == 0.2
    assert (results_dir / "metrics.json").exists()


def test_cli_batch_manifest_json(tmp_path: Path) -> None:
    rng = np.random.default_rng(20)
    points = rng.normal(size=(50, 3))
    input_path = tmp_path / "object.ply"
    output_path = tmp_path / "clean.ply"
    manifest_path = tmp_path / "batch.yaml"
    batch_dir = tmp_path / "batch_results"
    save_point_cloud(input_path, points)
    manifest_path.write_text(
        "\n".join(
            [
                "jobs:",
                "  - name: geometry_job",
                "    task: geometry",
                f"    input: {input_path}",
                "  - name: preprocess_job",
                "    task: preprocess",
                f"    input: {input_path}",
                f"    output: {output_path}",
                "    voxel_size: 0.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    completed = run_cli(
        "--batch",
        str(manifest_path),
        "--output-dir",
        str(batch_dir),
        "--format",
        "json",
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["success"] is True
    assert [job["task"] for job in payload["jobs"]] == ["geometry", "preprocess"]
    assert (batch_dir / "geometry_job" / "metrics.json").exists()
    assert (batch_dir / "preprocess_job" / "metrics.json").exists()
    assert output_path.exists()


def test_cli_benchmark_kdtree_quick(tmp_path: Path) -> None:
    results_dir = tmp_path / "benchmark_results"
    md_path = tmp_path / "kdtree.md"

    completed = run_cli(
        "benchmark",
        "kdtree",
        "--quick",
        "--points",
        "100",
        "200",
        "--queries",
        "5",
        "--output-dir",
        str(results_dir),
        "--save-md",
        str(md_path),
    )

    assert completed.returncode == 0, completed.stderr
    assert "Benchmark Result: kdtree" in completed.stdout
    assert (results_dir / "metrics.json").exists()
    assert md_path.exists()


def test_cli_missing_file_returns_error_and_metrics(tmp_path: Path) -> None:
    results_dir = tmp_path / "missing_results"

    completed = run_cli(
        "geometry",
        "--input",
        str(tmp_path / "missing.ply"),
        "--output-dir",
        str(results_dir),
    )

    assert completed.returncode == 1
    assert "Error:" in completed.stdout
    payload = json.loads((results_dir / "metrics.json").read_text(encoding="utf-8"))
    assert payload["success"] is False
