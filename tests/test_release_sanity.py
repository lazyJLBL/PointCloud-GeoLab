from __future__ import annotations

import json
import subprocess
from pathlib import Path

from scripts import check_devcontainer, check_packaging
from scripts.check_devcontainer import run_devcontainer_checks
from scripts.check_packaging import (
    check_packaging_metadata,
    load_packaging_metadata,
    run_packaging_checks,
)

ROOT = Path(__file__).resolve().parents[1]


def test_devcontainer_checks_repository(monkeypatch) -> None:
    monkeypatch.setattr(check_devcontainer.shutil, "which", lambda name: None)

    result = run_devcontainer_checks(ROOT)

    assert result.success, result.issues
    assert any("Docker CLI was not found" in warning for warning in result.warnings)


def test_devcontainer_finds_heavy_dependency_markers(tmp_path: Path, monkeypatch) -> None:
    _write_devcontainer(tmp_path)
    dockerfile = tmp_path / ".devcontainer" / "Dockerfile"
    dockerfile.write_text(
        "FROM nvidia/cuda:12.4.0-runtime\nRUN pip install torch\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(check_devcontainer.shutil, "which", lambda name: None)

    result = run_devcontainer_checks(tmp_path)

    assert not result.success
    assert any("CUDA" in issue for issue in result.issues)
    assert any("PyTorch" in issue for issue in result.issues)


def test_devcontainer_reports_docker_daemon_skip(tmp_path: Path, monkeypatch) -> None:
    _write_devcontainer(tmp_path)
    monkeypatch.setattr(check_devcontainer.shutil, "which", lambda name: "docker")

    def runner(command, **kwargs):
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="daemon unavailable")

    result = run_devcontainer_checks(tmp_path, runner=runner)

    assert result.success, result.issues
    assert result.warnings == ["Docker daemon probe skipped: daemon unavailable"]


def test_packaging_metadata_parser_reads_project() -> None:
    metadata = load_packaging_metadata(ROOT / "pyproject.toml")

    assert metadata.name == "pointcloud-geolab"
    assert metadata.version == "0.1.1"
    assert metadata.scripts["pointcloud-geolab"] == "pointcloud_geolab.cli:main"
    assert {"dev", "vis", "bench"} <= set(metadata.optional_dependencies)


def test_packaging_checker_finds_missing_optional_dependency() -> None:
    metadata = check_packaging.PackagingMetadata(
        name="pointcloud-geolab",
        version="0.1.1",
        requires_python=">=3.10",
        scripts={"pointcloud-geolab": "pointcloud_geolab.cli:main"},
        optional_dependencies={"dev": [], "vis": []},
    )

    issues = check_packaging_metadata(metadata)

    assert "pyproject.toml: missing optional dependency group `bench`" in issues


def test_packaging_skips_missing_build_module(tmp_path: Path) -> None:
    _write_pyproject(tmp_path)

    result = run_packaging_checks(tmp_path, build_available=False)

    assert result.success, result.issues
    assert result.built_artifacts == []
    assert result.warnings == [
        "Python module `build` is not installed; package build smoke skipped."
    ]


def test_packaging_build_smoke_uses_temporary_dist(tmp_path: Path) -> None:
    _write_pyproject(tmp_path)
    (tmp_path / "pointcloud_geolab").mkdir()
    (tmp_path / "pointcloud_geolab" / "__init__.py").write_text("", encoding="utf-8")

    def runner(command, **kwargs):
        outdir = Path(command[command.index("--outdir") + 1])
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "pointcloud_geolab-0.1.0.tar.gz").write_bytes(b"sdist")
        (outdir / "pointcloud_geolab-0.1.0-py3-none-any.whl").write_bytes(b"wheel")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    result = run_packaging_checks(tmp_path, runner=runner, build_available=True)

    assert result.success, result.issues
    assert sorted(path.suffix for path in result.built_artifacts) == [".gz", ".whl"]
    assert not (tmp_path / "dist").exists()
    assert not list(tmp_path.glob("*.egg-info"))


def test_makefile_runs_release_sanity_checks() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    assert "check-packaging:" in makefile
    assert "check-devcontainer:" in makefile
    verify_core = makefile.split("verify-core:", 1)[1].splitlines()[0]
    assert "check-devcontainer" in verify_core
    assert "check-packaging" in verify_core


def _write_devcontainer(root: Path) -> None:
    folder = root / ".devcontainer"
    folder.mkdir()
    (folder / "Dockerfile").write_text(
        "FROM python:3.12-slim\nRUN apt-get update && apt-get install -y make git\n",
        encoding="utf-8",
    )
    (folder / "devcontainer.json").write_text(
        json.dumps(
            {
                "name": "test",
                "description": "Open3D/ML extras are optional.",
                "build": {"dockerfile": "Dockerfile"},
                "workspaceFolder": "/workspaces/PointCloud-GeoLab",
                "postCreateCommand": 'python -m pip install -e ".[dev,vis,bench]"',
                "postAttachCommand": (
                    "python -m pytest --version && make verify-core -n "
                    "&& make verify-portfolio -n"
                ),
            }
        ),
        encoding="utf-8",
    )


def _write_pyproject(root: Path) -> None:
    root.joinpath("pyproject.toml").write_text(
        """
[project]
name = "pointcloud-geolab"
version = "0.1.0"
requires-python = ">=3.10"

[project.optional-dependencies]
dev = []
vis = []
bench = []

[project.scripts]
pointcloud-geolab = "pointcloud_geolab.cli:main"
""".strip(),
        encoding="utf-8",
    )
