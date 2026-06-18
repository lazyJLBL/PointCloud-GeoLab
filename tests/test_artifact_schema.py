from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.check_artifact_schema import (
    DEFAULT_RELEASE_MANIFEST,
    main,
    validate_benchmark_json,
    validate_file,
    validate_portfolio_metrics,
    validate_release_manifest,
)


def test_artifact_schema_script_help_runs() -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0


def test_release_manifest_schema_passes_repository_manifest() -> None:
    check = validate_file(DEFAULT_RELEASE_MANIFEST, "release")

    assert check.success, check.issues


def test_release_manifest_schema_reports_missing_keys(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    payload = {"version": "0.1.1"}

    issues = validate_release_manifest(payload, path)

    assert any("local_verification_commands" in issue for issue in issues)
    assert any("expected_generated_artifacts" in issue for issue in issues)


def test_portfolio_schema_validates_core_metrics(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"
    payload = {
        "input": {"num_points": 10},
        "preprocessing": {"num_points_after": 8},
        "features": {},
        "registration": {"rmse_after": 0.1, "fitness": 0.9},
        "segmentation": {"num_clusters": 2, "noise_ratio": 0.1},
        "runtime": {"total_seconds": 0.2},
    }

    assert validate_portfolio_metrics(payload, path) == []


def test_portfolio_schema_reports_invalid_ranges(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"
    payload = {
        "input": {"num_points": -1},
        "preprocessing": {"num_points_after": 8},
        "features": {},
        "registration": {"rmse_after": 0.1, "fitness": 2.0},
        "segmentation": {"num_clusters": 2, "noise_ratio": 0.1},
        "runtime": {"total_seconds": 0.2},
    }

    issues = validate_portfolio_metrics(payload, path)

    assert any("input.num_points" in issue for issue in issues)
    assert any("registration.fitness" in issue for issue in issues)


def test_benchmark_schema_requires_repeat_and_memory_metadata(tmp_path: Path) -> None:
    path = tmp_path / "benchmark.json"
    payload = {
        "benchmark": "kdtree",
        "metadata": {
            "parameters": {},
            "repeat": {"count": 2},
            "memory": {"available": True, "peak_bytes": 1234},
        },
        "conclusion": "local run",
        "rows": [{"kd_time_mean": 0.01}],
    }

    assert validate_benchmark_json(payload, path) == []


def test_benchmark_schema_reports_missing_memory(tmp_path: Path) -> None:
    path = tmp_path / "benchmark.json"
    payload = {
        "benchmark": "kdtree",
        "metadata": {"parameters": {}, "repeat": {"count": 1}},
        "conclusion": "local run",
        "rows": [],
    }

    issues = validate_benchmark_json(payload, path)

    assert any("metadata.memory" in issue for issue in issues)
    assert any("rows must not be empty" in issue for issue in issues)


def test_artifact_schema_cli_validates_explicit_files(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "version": "0.1.1",
                "release_date": "2026-06-18",
                "commit": "HEAD",
                "local_verification_commands": ["make verify-release-candidate"],
                "expected_generated_artifacts": {
                    "portfolio": ["outputs/portfolio_demo/report.md"],
                    "benchmarks": ["outputs/benchmarks/all_benchmark.json"],
                },
                "ignored_artifact_paths": ["outputs/"],
                "limitations": ["not real benchmark data"],
                "open_roadmap_items": ["future full nonlinear GICP"],
            }
        ),
        encoding="utf-8",
    )

    assert main(["--release-manifest", str(manifest)]) == 0
