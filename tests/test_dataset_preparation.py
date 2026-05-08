from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from scripts.prepare_datasets import read_off, sample_off_mesh, validate_layout

ROOT = Path(__file__).resolve().parents[1]


def test_read_and_sample_off_mesh(tmp_path: Path) -> None:
    off_path = tmp_path / "triangle.off"
    off_path.write_text(
        "\n".join(
            [
                "OFF",
                "3 1 0",
                "0 0 0",
                "1 0 0",
                "0 1 0",
                "3 0 1 2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    vertices, faces = read_off(off_path)
    points = sample_off_mesh(off_path, count=16, seed=1)

    assert vertices.shape == (3, 3)
    assert faces.shape == (1, 3)
    assert points.shape == (16, 3)
    assert np.all(points[:, 2] == 0.0)


def test_validate_layout_reports_missing_external_data(tmp_path: Path) -> None:
    report = validate_layout(tmp_path)

    assert set(report) == {"stanford", "kitti", "modelnet"}
    assert all(item["present"] is False for item in report.values())


def test_real_examples_fail_cleanly_when_data_is_missing(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "examples" / "real_bunny_registration.py"),
            "--data-dir",
            str(tmp_path / "missing_bunny"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "Stanford Bunny pair not found" in completed.stderr
