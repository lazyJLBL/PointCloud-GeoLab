from __future__ import annotations

import json
from pathlib import Path

from scripts.check_v1_ready import (
    CURRENT_VERSION,
    check_boundary_wording,
    check_generated_paths,
    check_release_manifest,
    check_version_consistency,
    check_web_presentation,
    main,
    run_v1_ready,
)

ROOT = Path(__file__).resolve().parents[1]


def test_v1_ready_script_runs_on_repository() -> None:
    assert main(["--root", str(ROOT)]) == 0


def test_v1_ready_result_passes_repository_static_checks() -> None:
    result = run_v1_ready(ROOT)

    assert result.success, result.issues


def test_v1_ready_detects_version_mismatch(tmp_path: Path) -> None:
    _write_minimal_v1_repo(tmp_path)
    (tmp_path / "CITATION.cff").write_text('version: "0.9.0"\n', encoding="utf-8")

    issues = check_version_consistency(tmp_path)

    assert any("expected version 1.1.0" in issue for issue in issues)


def test_v1_ready_detects_generated_path() -> None:
    issues = check_generated_paths(ROOT, tracked_files=["outputs/kitti_segmentation/report.md"])

    assert issues == ["generated path is tracked by git: outputs/kitti_segmentation/report.md"]


def test_v1_ready_detects_bad_release_manifest(tmp_path: Path) -> None:
    _write_minimal_v1_repo(tmp_path)
    manifest = tmp_path / "docs" / "releases" / "v1.1.0_artifacts.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["version"] = "0.1.1"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    issues = check_release_manifest(tmp_path)

    assert any(f"version must be {CURRENT_VERSION}" in issue for issue in issues)


def test_v1_ready_detects_missing_web_assets(tmp_path: Path) -> None:
    _write_minimal_v1_repo(tmp_path)
    (tmp_path / "docs" / "assets" / "web_console_dashboard.svg").unlink()

    result = run_v1_ready(tmp_path, tracked_files=[])

    assert not result.success
    assert any("web_console_dashboard.svg" in issue for issue in result.issues)


def test_v1_ready_checks_web_presentation_text(tmp_path: Path) -> None:
    _write_minimal_v1_repo(tmp_path)
    (tmp_path / "README.md").write_text("No Web hero here.\n", encoding="utf-8")

    issues = check_web_presentation(tmp_path)

    assert any("README.md: missing Web presentation phrase" in issue for issue in issues)


def test_v1_ready_detects_missing_boundary_wording(tmp_path: Path) -> None:
    _write_minimal_v1_repo(tmp_path)
    for relative in [
        "README.md",
        "docs/limitations.md",
        "docs/project_boundary.md",
        "docs/releases/v1.1.0.md",
        "docs/case_studies/kitti_lidar_result.md",
        "docs/web_console.md",
        "docs/web_api.md",
        "web/README.md",
    ]:
        (tmp_path / relative).write_text("Everything is complete.\n", encoding="utf-8")

    issues = check_boundary_wording(tmp_path)

    assert any("full nonlinear gicp" in issue for issue in issues)


def test_v1_ready_detects_stale_issue_two_wording(tmp_path: Path) -> None:
    _write_minimal_v1_repo(tmp_path)
    release = tmp_path / "docs" / "releases" / "v1.1.0.md"
    release.write_text(
        "not a full nonlinear GICP; not an official KITTI benchmark; "
        "not a SLAM backend; not CUDA accelerated; not a PointNet training "
        "release; synthetic fixture boundary; issue #2 remains open.\n",
        encoding="utf-8",
    )

    issues = check_boundary_wording(tmp_path)

    assert any("issue #2" in issue for issue in issues)


def test_v1_ready_warns_without_git_metadata(tmp_path: Path) -> None:
    _write_minimal_v1_repo(tmp_path)

    result = run_v1_ready(tmp_path)

    assert result.success, result.issues
    assert any(".git metadata not found" in warning for warning in result.warnings)


def test_v1_ready_can_require_git_metadata(tmp_path: Path) -> None:
    _write_minimal_v1_repo(tmp_path)

    result = run_v1_ready(tmp_path, require_git=True)

    assert not result.success
    assert any(".git metadata not found" in issue for issue in result.issues)


def _write_minimal_v1_repo(root: Path) -> None:
    (root / "docs" / "releases").mkdir(parents=True)
    (root / "docs" / "case_studies").mkdir(parents=True)
    (root / "docs" / "gallery").mkdir(parents=True)
    (root / "docs" / "assets").mkdir(parents=True)
    (root / "web" / "backend" / "tests").mkdir(parents=True)
    (root / "web" / "frontend").mkdir(parents=True)
    (root / "pointcloud_geolab").mkdir()
    (root / "pyproject.toml").write_text(
        '[project]\nname = "demo"\nversion = "1.1.0"\n',
        encoding="utf-8",
    )
    (root / "pointcloud_geolab" / "__init__.py").write_text(
        '__version__ = "1.1.0"\n',
        encoding="utf-8",
    )
    (root / "CITATION.cff").write_text('version: "1.1.0"\n', encoding="utf-8")
    (root / "CHANGELOG.md").write_text(
        "# Changelog\n\n"
        "## v1.1.0 - 2026-06-19\n\n"
        "## v1.0.0 - 2026-06-18\n\n"
        "## v0.1.1 - 2026-06-18\n\n"
        "## v0.1.0 Portfolio Release\n",
        encoding="utf-8",
    )
    for relative in [
        "README.md",
        "docs/limitations.md",
        "docs/project_boundary.md",
        "docs/releases/v1.1.0.md",
        "docs/case_studies/kitti_lidar_result.md",
        "docs/versioning.md",
        "docs/api_stability.md",
        "docs/cli_reference.md",
        "docs/gallery/README.md",
        "docs/scale_benchmark.md",
        "docs/web_console.md",
        "docs/web_api.md",
        "web/README.md",
        "web/backend/tests/test_web_backend.py",
    ]:
        text = (
            "not a full nonlinear GICP; not an official KITTI benchmark; "
            "not a SLAM backend; not CUDA accelerated; not a PointNet training "
            "release; not a production web platform; synthetic fixture boundary.\n"
        )
        if relative == "README.md":
            text += (
                "PointCloud-GeoLab = point-cloud geometry core + reports + "
                "Experimental Web Console.\n"
                "![Web](docs/assets/web_console_dashboard.svg)\n"
                "make web-backend\nnpm run dev\n"
            )
        if relative == "docs/gallery/README.md":
            text += (
                "## Experimental Web Console\n"
                "![Dashboard](../assets/web_console_dashboard.svg)\n"
                "![Preview](../assets/web_console_dataset_preview.svg)\n"
                "![Artifacts](../assets/web_console_task_artifacts.svg)\n"
            )
        (root / relative).write_text(text, encoding="utf-8")
    (root / "web" / "frontend" / "package.json").write_text(
        '{\n  "version": "1.1.0"\n}\n',
        encoding="utf-8",
    )
    (root / "web" / "backend" / "app").mkdir(parents=True)
    (root / "web" / "backend" / "app" / "main.py").write_text(
        'app_config = dict(version="1.1.0")\nversion="1.1.0"\n',
        encoding="utf-8",
    )
    for asset in [
        "portfolio_raw_pointcloud.png",
        "portfolio_downsampled.png",
        "portfolio_registration_before_after.png",
        "portfolio_segmentation_result.png",
        "portfolio_bbox_normals.png",
        "kitti_case_study_tiny.png",
        "scale_benchmark_quick.png",
        "web_console_dashboard.svg",
        "web_console_dataset_preview.svg",
        "web_console_task_artifacts.svg",
    ]:
        suffix = b"svg" if asset.endswith(".svg") else b"png"
        (root / "docs" / "assets" / asset).write_bytes(suffix)
    (root / "Makefile").write_text(
        "verify-realdata:\n\nverify-scale-benchmark:\n\n" "verify-v1-candidate:\n\nverify-web:\n",
        encoding="utf-8",
    )
    (root / "docs" / "releases" / "v1.1.0_artifacts.json").write_text(
        json.dumps(
            {
                "version": "1.1.0",
                "release_date": "2026-06-19",
                "commit": "HEAD",
                "local_verification_commands": [
                    "python scripts/check_release_ready.py",
                    "make verify-release-candidate",
                    "make verify-v1-candidate",
                    "make verify-web",
                ],
                "expected_generated_artifacts": {
                    "portfolio": ["outputs/portfolio_demo/report.html"],
                    "benchmarks": ["outputs/scale_benchmark/scale_benchmark.json"],
                    "realdata": ["outputs/kitti_segmentation/report.md"],
                    "web": ["outputs/web/tasks/{task_id}/result.json"],
                },
                "ignored_artifact_paths": ["outputs/"],
                "limitations": [
                    "not a full nonlinear gicp",
                    "not an official kitti benchmark",
                    "not a slam backend",
                    "not a cuda stack",
                    "not a pointnet training release",
                    "not a production web platform",
                ],
                "open_roadmap_items": ["future real KITTI benchmark"],
            }
        ),
        encoding="utf-8",
    )
