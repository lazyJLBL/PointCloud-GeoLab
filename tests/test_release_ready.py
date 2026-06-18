from __future__ import annotations

import json
from pathlib import Path

from scripts.check_release_ready import (
    CURRENT_VERSION,
    check_changelog_section,
    load_artifact_manifest,
    main,
    run_release_ready,
)

ROOT = Path(__file__).resolve().parents[1]


def test_release_ready_script_runs_on_repository() -> None:
    assert main(["--root", str(ROOT)]) == 0


def test_release_manifest_json_parses() -> None:
    manifest, issues = load_artifact_manifest(ROOT / "docs" / "releases" / "v1.0.0_artifacts.json")

    assert issues == []
    assert manifest is not None
    assert manifest["version"] == CURRENT_VERSION
    assert "make verify-release-candidate" in manifest["local_verification_commands"]
    assert "make verify-v1-candidate" in manifest["local_verification_commands"]


def test_release_ready_finds_version_mismatch(tmp_path: Path) -> None:
    _write_minimal_release_repo(tmp_path)
    (tmp_path / "CITATION.cff").write_text(
        'version: "0.2.0"\ndate-released: "2026-06-18"\n',
        encoding="utf-8",
    )

    result = run_release_ready(tmp_path, tracked_files=[], status_output="")

    assert not result.success
    assert any("expected version 1.0.0" in issue for issue in result.issues)


def test_release_ready_finds_missing_changelog_section(tmp_path: Path) -> None:
    _write_minimal_release_repo(tmp_path)
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(
        "# Changelog\n\n## Unreleased\n\n## v0.1.0 Portfolio Release\n",
        encoding="utf-8",
    )

    issues = check_changelog_section(changelog, CURRENT_VERSION)

    assert any("missing `## v1.0.0 - 2026-06-18`" in issue for issue in issues)


def test_release_ready_finds_tracked_generated_path(tmp_path: Path) -> None:
    _write_minimal_release_repo(tmp_path)

    result = run_release_ready(
        tmp_path,
        tracked_files=["outputs/portfolio_demo/report.md"],
        status_output="",
    )

    assert not result.success
    assert any("generated path is tracked" in issue for issue in result.issues)


def test_release_ready_finds_overclaim_wording(tmp_path: Path) -> None:
    _write_minimal_release_repo(tmp_path)
    (tmp_path / "docs" / "bad.md").write_text(
        "The release implemented full nonlinear GICP for real KITTI benchmark support.\n",
        encoding="utf-8",
    )

    result = run_release_ready(tmp_path, tracked_files=[], status_output="")

    assert not result.success
    assert any("overclaim term" in issue for issue in result.issues)
    assert any("roadmap item" in issue for issue in result.issues)


def test_release_ready_reports_clean_workspace_prompt(tmp_path: Path) -> None:
    _write_minimal_release_repo(tmp_path)

    result = run_release_ready(tmp_path, tracked_files=[], status_output="")

    assert result.success, result.issues
    assert result.workspace_clean
    assert result.warnings == []


def test_verify_release_candidate_target_exists() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    assert "check-release-ready:" in makefile
    assert "verify-release-candidate:" in makefile
    target = makefile.split("verify-release-candidate:", 1)[1].splitlines()[0]
    assert "verify-core" in target
    assert "verify-portfolio" in target
    assert "verify-benchmarks" in target
    assert "check-release-ready" in target


def _write_minimal_release_repo(root: Path) -> None:
    (root / ".github" / "workflows").mkdir(parents=True)
    (root / "docs" / "releases").mkdir(parents=True)
    (root / "pointcloud_geolab").mkdir()

    (root / "README.md").write_text(
        "# Demo\n\n[Release](docs/releases/v1.0.0.md)\n",
        encoding="utf-8",
    )
    (root / "docs" / "releases" / "v0.1.0.md").write_text(
        "# v0.1.0 Portfolio Release\n",
        encoding="utf-8",
    )
    (root / "docs" / "releases" / "v0.1.1.md").write_text(
        "# v0.1.1 Hardening Release\n\n"
        "This is not a full nonlinear GICP optimizer and not real KITTI data.\n",
        encoding="utf-8",
    )
    (root / "docs" / "releases" / "v1.0.0.md").write_text(
        "# v1.0.0 Portfolio-Stable Release Candidate\n\n"
        "This is not a full nonlinear GICP optimizer and not real KITTI data.\n",
        encoding="utf-8",
    )
    (root / "docs" / "releases" / "v1.0.0_artifacts.json").write_text(
        json.dumps(_artifact_manifest(), indent=2),
        encoding="utf-8",
    )
    (root / "pyproject.toml").write_text(
        '[project]\nname = "demo"\nversion = "1.0.0"\n',
        encoding="utf-8",
    )
    (root / "pointcloud_geolab" / "__init__.py").write_text(
        '__version__ = "1.0.0"\n',
        encoding="utf-8",
    )
    (root / "CITATION.cff").write_text(
        'version: "1.0.0"\ndate-released: "2026-06-18"\n',
        encoding="utf-8",
    )
    (root / "CHANGELOG.md").write_text(
        "# Changelog\n\n"
        "## Unreleased\n\n"
        "No unreleased changes yet.\n\n"
        "## v1.0.0 - 2026-06-18\n\n"
        "### Added\n\n"
        "- Release checks.\n\n"
        "## v0.1.1 - 2026-06-18\n\n"
        "### Added\n\n"
        "- Historical checks.\n\n"
        "## v0.1.0 Portfolio Release\n",
        encoding="utf-8",
    )
    (root / ".github" / "workflows" / "tests.yml").write_text(
        "name: Tests\n",
        encoding="utf-8",
    )


def _artifact_manifest() -> dict[str, object]:
    return {
        "version": "1.0.0",
        "release_date": "2026-06-18",
        "commit": "HEAD",
        "local_verification_commands": [
            "python scripts/check_release_ready.py",
            "make verify-release-candidate",
            "make verify-v1-candidate",
        ],
        "expected_generated_artifacts": {
            "portfolio": ["outputs/portfolio_demo/report.md"],
            "benchmarks": ["outputs/benchmarks/all_benchmark.json"],
        },
        "ignored_artifact_paths": [
            "outputs/",
            "results/",
            "examples/demo_data/",
            "benchmark_results/",
        ],
        "limitations": [
            "Synthetic fixtures are not real benchmarks.",
            "This is not a full nonlinear GICP optimizer.",
        ],
        "open_roadmap_items": [
            "Implement full nonlinear GICP optimizer.",
            "Add real KITTI benchmark report.",
        ],
    }
