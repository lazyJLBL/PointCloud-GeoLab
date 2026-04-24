from __future__ import annotations

from pathlib import Path
import subprocess
import sys

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
    )

    assert completed.returncode == 0, completed.stderr
    assert "RANSAC Plane Fitting Result" in completed.stdout
    assert "Inlier Ratio" in completed.stdout

