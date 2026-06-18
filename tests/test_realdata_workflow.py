from __future__ import annotations

import argparse
import json
from pathlib import Path

from examples.kitti_lidar_segmentation import main as kitti_main
from examples.kitti_lidar_segmentation import run_kitti_segmentation
from scripts.verify_realdata_workflow import (
    main,
    verify_kitti_workflow_outputs,
    write_tiny_kitti_frame,
)


def test_kitti_example_missing_frame_returns_preparation_code(tmp_path: Path) -> None:
    code = kitti_main(
        [
            "--frame",
            str(tmp_path / "missing.bin"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert code == 2


def test_kitti_workflow_writes_report_metrics_and_figures(tmp_path: Path) -> None:
    frame = write_tiny_kitti_frame(tmp_path / "000000.bin")
    output_dir = tmp_path / "kitti_out"

    result = run_kitti_segmentation(
        argparse.Namespace(
            frame=frame,
            output_dir=output_dir,
            eps=0.55,
            min_points=8,
            max_points=5000,
            ground_threshold=0.18,
            ground_angle_threshold=35.0,
            seed=7,
        )
    )

    issues = verify_kitti_workflow_outputs(output_dir)
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))

    assert result["metrics"]["workflow"] == "kitti_lidar_segmentation"
    assert issues == []
    assert metrics["memory"]["available"] is True
    assert "not an official KITTI benchmark" in " ".join(metrics["limitations"])


def test_realdata_verifier_dry_run_main_passes() -> None:
    assert main(["--dry-run"]) == 0


def test_realdata_verifier_reports_missing_artifact(tmp_path: Path) -> None:
    output_dir = tmp_path / "empty"
    output_dir.mkdir()

    issues = verify_kitti_workflow_outputs(output_dir)

    assert any("metrics.json" in issue for issue in issues)
    assert any("report.md" in issue for issue in issues)


def test_realdata_verifier_rejects_bad_metrics_json(tmp_path: Path) -> None:
    output_dir = tmp_path / "bad"
    output_dir.mkdir()
    for name in [
        "report.md",
        "report.html",
        "kitti_bev.png",
        "kitti_clusters.png",
        "kitti_height_histogram.png",
        "cluster_report.md",
        "kitti_clusters.ply",
    ]:
        (output_dir / name).write_text("placeholder", encoding="utf-8")
    (output_dir / "metrics.json").write_text("{not-json", encoding="utf-8")

    issues = verify_kitti_workflow_outputs(output_dir)

    assert any("invalid JSON" in issue for issue in issues)
