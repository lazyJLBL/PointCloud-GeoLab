from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pointcloud_geolab.api import TaskResult
from pointcloud_geolab.cli import _format_text_result, _load_yaml, _run_batch
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
        "--repeat",
        "2",
        "--output-dir",
        str(results_dir),
        "--save-md",
        str(md_path),
    )

    assert completed.returncode == 0, completed.stderr
    assert "Benchmark Result: kdtree" in completed.stdout
    assert (results_dir / "metrics.json").exists()
    assert md_path.exists()
    payload = json.loads((results_dir / "kdtree_benchmark.json").read_text(encoding="utf-8"))
    assert payload["metadata"]["repeat"]["count"] == 2
    assert "kd_time_mean" in payload["rows"][0]


def test_cli_fit_primitive_and_segment(tmp_path: Path) -> None:
    rng = np.random.default_rng(23)
    xy = rng.uniform(-1, 1, size=(80, 2))
    plane = np.column_stack([xy, np.zeros(80)])
    input_path = tmp_path / "plane.ply"
    save_point_cloud(input_path, plane)

    fit = run_cli(
        "fit-primitive",
        "--input",
        str(input_path),
        "--model",
        "plane",
        "--threshold",
        "0.01",
        "--output-dir",
        str(tmp_path / "fit"),
        "--format",
        "json",
    )
    assert fit.returncode == 0, fit.stderr
    assert json.loads(fit.stdout)["metrics"]["inliers"] == len(plane)

    clusters = np.vstack(
        [
            rng.normal([0, 0, 0], 0.01, size=(20, 3)),
            rng.normal([1, 0, 0], 0.01, size=(20, 3)),
        ]
    )
    cluster_path = tmp_path / "clusters.ply"
    save_point_cloud(cluster_path, clusters)
    segmented_path = tmp_path / "segmented.ply"
    segment = run_cli(
        "segment",
        "--input",
        str(cluster_path),
        "--method",
        "dbscan",
        "--eps",
        "0.08",
        "--min-points",
        "3",
        "--output",
        str(segmented_path),
        "--output-dir",
        str(tmp_path / "segment"),
        "--format",
        "json",
    )
    assert segment.returncode == 0, segment.stderr
    assert json.loads(segment.stdout)["metrics"]["cluster_count"] == 2
    assert segmented_path.exists()


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


def test_cli_missing_file_json_error_is_machine_readable(tmp_path: Path) -> None:
    completed = run_cli(
        "geometry",
        "--input",
        str(tmp_path / "missing.ply"),
        "--output-dir",
        str(tmp_path / "missing_json"),
        "--format",
        "json",
    )

    payload = json.loads(completed.stdout)
    assert completed.returncode == 1
    assert payload["success"] is False
    assert payload["task"] == "geometry"
    assert "missing.ply" in payload["error"]
    assert payload["path"] == str(tmp_path / "missing.ply")


def test_cli_benchmark_json_error_contract_includes_path(tmp_path: Path) -> None:
    completed = run_cli(
        "benchmark",
        "kdtree",
        "--points",
        "0",
        "--output-dir",
        str(tmp_path / "bad_benchmark"),
        "--format",
        "json",
    )

    payload = json.loads(completed.stdout)
    assert completed.returncode == 1
    assert payload["task"] == "benchmark:kdtree"
    assert payload["success"] is False
    assert payload["parameters"]["points"] == [0]
    assert payload["path"] == str(tmp_path / "bad_benchmark")


def test_cli_subcommand_help_is_available() -> None:
    for command in ["geometry", "benchmark", "pipeline", "register"]:
        completed = run_cli(command, "--help")

        assert completed.returncode == 0
        assert "usage:" in completed.stdout
        assert command in completed.stdout


def test_load_yaml_rejects_scalar_document(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("42\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must contain a YAML mapping or list"):
        _load_yaml(config_path)


def test_batch_manifest_rejects_non_job_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest_path = tmp_path / "bad_batch.yaml"
    manifest_path.write_text("name: not-a-jobs-list\n", encoding="utf-8")

    code = _run_batch(
        argparse.Namespace(
            batch=manifest_path,
            output_dir=None,
            format="text",
            save_results=None,
            visualize=None,
            seed=None,
        )
    )

    assert code == 1
    assert "top-level jobs list" in capsys.readouterr().err


def test_batch_manifest_reports_invalid_jobs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest_path = tmp_path / "invalid_jobs.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                "jobs:",
                "  - not-a-mapping",
                "  - task: not-real",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    code = _run_batch(
        argparse.Namespace(
            batch=manifest_path,
            output_dir=tmp_path / "batch",
            format="text",
            save_results=None,
            visualize=None,
            seed=None,
        )
    )

    output = capsys.readouterr().out
    assert code == 1
    assert "Batch Summary" in output
    assert "batch job must be a mapping" in output
    assert "batch job task must be one of" in output


def test_cli_text_formatters_cover_reviewer_outputs() -> None:
    error = _format_text_result(TaskResult(task="geometry", success=False, error="bad input"))
    icp = _format_text_result(
        TaskResult(
            task="icp",
            success=True,
            metrics={
                "iterations": 2,
                "initial_rmse": 0.5,
                "final_rmse": 0.1,
                "fitness": 0.9,
                "converged": True,
            },
            data={
                "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "translation": [0.0, 0.0, 0.0],
                "transformation": _identity(),
            },
        )
    )
    plane = _format_text_result(
        TaskResult(
            task="plane",
            success=True,
            metrics={"inliers": 9, "outliers": 1, "inlier_ratio": 0.9},
            data={"plane_model": [0.0, 0.0, 1.0, -0.1]},
        )
    )
    geometry = _format_text_result(
        TaskResult(
            task="geometry",
            success=True,
            metrics={
                "center": [0.0, 0.0, 0.0],
                "aabb_extent": [1.0, 2.0, 3.0],
                "obb_extent": [1.0, 2.0, 3.0],
                "pca_eigenvalues": [3.0, 2.0, 1.0],
                "main_direction": [1.0, 0.0, 0.0],
            },
        )
    )
    preprocess = _format_text_result(
        TaskResult(
            task="preprocess",
            success=True,
            metrics={
                "original_points": 10,
                "after_voxel_downsample": 8,
                "after_statistical_filter": 7,
                "after_radius_filter": 6,
                "kept_statistical_inliers": 7,
                "estimated_normals": True,
            },
        )
    )
    primitive = _format_text_result(
        TaskResult(
            task="fit-primitive",
            success=True,
            metrics={
                "model": "plane",
                "inliers": 10,
                "outliers": 2,
                "inlier_ratio": 0.8,
                "residual_mean": 0.01,
            },
            data={"model_params": {"normal": [0.0, 0.0, 1.0]}},
        )
    )
    extracted = _format_text_result(
        TaskResult(
            task="extract-primitives",
            success=True,
            metrics={"model_count": 1, "remaining_points": 4},
            data={
                "primitives": [
                    {
                        "model_type": "plane",
                        "inlier_ratio": 0.9,
                        "residual_mean": 0.01,
                    }
                ]
            },
        )
    )
    segmented = _format_text_result(
        TaskResult(
            task="segment",
            success=True,
            metrics={"cluster_count": 2, "noise_points": 3},
            data={"clusters": [{"label": 0, "size": 5}]},
        )
    )
    visualize = _format_text_result(
        TaskResult(task="visualize", success=True, metrics={"mode": "html"})
    )
    train = _format_text_result(
        TaskResult(task="train-pointnet", success=True, metrics={"loss": 0.1, "accuracy": 0.9})
    )
    reconstruct = _format_text_result(
        TaskResult(
            task="reconstruct",
            success=True,
            metrics={"method": "alpha_shape", "vertices": 4, "triangles": 2},
        )
    )
    verify = _format_text_result(
        TaskResult(
            task="verify-portfolio",
            success=True,
            metrics={
                "passed_commands": 3,
                "failed_commands": 0,
                "generated_artifacts": 8,
                "missing_readme_artifacts": 0,
            },
        )
    )
    infer = _format_text_result(
        TaskResult(
            task="infer-pointnet",
            success=True,
            metrics={"class": "sphere", "confidence": 0.8},
        )
    )
    unknown = _format_text_result(TaskResult(task="unknown", success=True, metrics={"ok": 1}))

    benchmark = _format_text_result(
        TaskResult(
            task="benchmark:kdtree",
            success=True,
            metrics={"benchmark": "kdtree"},
            data={"markdown": "## Run Metadata"},
        )
    )
    register = _format_text_result(
        TaskResult(
            task="register",
            success=True,
            metrics={
                "coarse_fitness": 0.9,
                "coarse_inlier_rmse": 0.2,
                "refined_fitness": 0.95,
                "final_rmse": 0.1,
            },
            data={"refined_transform": _identity()},
            artifacts={"transform": "transform.json"},
        )
    )
    pipeline = _format_text_result(
        TaskResult(
            task="pipeline",
            success=True,
            metrics={
                "input": {"num_points": 100},
                "preprocessing": {"num_points_after": 80},
                "registration": {"rmse_before": 0.4, "rmse_after": 0.1},
                "segmentation": {"num_clusters": 2, "noise_ratio": 0.05},
            },
        )
    )

    assert error == "Error: bad input"
    assert "ICP Registration Result" in icp
    assert "RANSAC Plane Fitting Result" in plane
    assert "Point Cloud Geometry" in geometry
    assert "Estimated normals: true" in preprocess
    assert "Primitive Fitting Result" in primitive
    assert "Sequential Primitive Extraction Result" in extracted
    assert "Segmentation Result" in segmented
    assert "Visualization Result" in visualize
    assert "PointNet Training Result" in train
    assert "Surface Reconstruction Result" in reconstruct
    assert "Portfolio Verification Result" in verify
    assert "PointNet Inference Result" in infer
    assert '"task": "unknown"' in unknown
    assert "Benchmark Result: kdtree" in benchmark
    assert "Refined Transformation" in register
    assert "- transform: transform.json" in register
    assert "Portfolio Pipeline Result" in pipeline


def _identity() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
