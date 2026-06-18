from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from pointcloud_geolab.datasets.fixtures import (
    load_kitti_like_bin,
    load_modelnet_like_off,
    validate_fixture_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "datasets"


def test_kitti_like_fixture_loads_xyzi_shape() -> None:
    cloud = load_kitti_like_bin(FIXTURE_ROOT / "mini_kitti_like.bin", expected_points=4)

    assert cloud.shape == (4, 4)
    assert cloud[0, :3].tolist() == [0.0, 0.0, 0.0]
    assert cloud[0, 3] == pytest.approx(0.1)


def test_modelnet_like_fixture_loads_vertices_and_faces() -> None:
    mesh = load_modelnet_like_off(FIXTURE_ROOT / "mini_modelnet_like.off")

    assert mesh.vertices.shape == (5, 3)
    assert mesh.faces.shape == (4, 3)
    assert mesh.faces[0].tolist() == [0, 1, 2]


def test_fixture_manifest_sha256_validation_passes() -> None:
    result = validate_fixture_manifest(FIXTURE_ROOT / "manifest.json")

    assert result.success, result.issues
    assert len(result.checked_files) == 2


def test_fixture_manifest_checksum_mismatch_fails(tmp_path: Path) -> None:
    _copy_fixture_tree(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["fixtures"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    result = validate_fixture_manifest(manifest_path)

    assert not result.success
    assert any("checksum mismatch" in issue for issue in result.issues)
    assert any("mini_kitti_like.bin" in issue for issue in result.issues)


def test_bad_kitti_like_bin_reports_path_and_reason(tmp_path: Path) -> None:
    bad_path = tmp_path / "bad.bin"
    bad_path.write_bytes(b"123")

    with pytest.raises(ValueError, match=r"bad\.bin.*not divisible by 16"):
        load_kitti_like_bin(bad_path)


def test_bad_off_reports_path_and_reason(tmp_path: Path) -> None:
    bad_path = tmp_path / "bad.off"
    bad_path.write_text("NOFF\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"bad\.off.*bad OFF header"):
        load_modelnet_like_off(bad_path)


def test_off_count_mismatch_reports_clear_error(tmp_path: Path) -> None:
    bad_path = tmp_path / "short.off"
    bad_path.write_text("OFF\n3 1 0\n0 0 0\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"short\.off.*unexpected end of file"):
        load_modelnet_like_off(bad_path)


def test_dataset_fixture_script_runs() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/check_dataset_fixtures.py"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Dataset fixture checks passed" in completed.stdout


def test_verify_core_runs_fixture_check() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    assert "check-fixtures:" in makefile
    assert "verify-core:" in makefile
    assert "check-fixtures" in makefile.split("verify-core:", 1)[1].splitlines()[0]


def _copy_fixture_tree(target: Path) -> None:
    for path in FIXTURE_ROOT.iterdir():
        if path.is_file():
            shutil.copy2(path, target / path.name)
