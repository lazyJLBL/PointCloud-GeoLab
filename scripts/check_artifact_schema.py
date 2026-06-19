"""Validate lightweight JSON artifact schemas without external dependencies."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE_MANIFEST = ROOT / "docs" / "releases" / "v1.0.0_artifacts.json"


@dataclass(frozen=True, slots=True)
class SchemaCheck:
    """Result for one artifact schema validation."""

    path: Path
    schema: str
    issues: list[str]

    @property
    def success(self) -> bool:
        return not self.issues


def validate_release_manifest(payload: dict[str, Any], path: Path) -> list[str]:
    """Validate the release artifact manifest shape."""

    issues: list[str] = []
    _require_type(payload, "version", str, issues)
    _require_type(payload, "release_date", str, issues)
    _require_type(payload, "commit", str, issues)
    _require_list(payload, "local_verification_commands", str, issues)
    _require_list(payload, "ignored_artifact_paths", str, issues)
    _require_list(payload, "limitations", str, issues)
    _require_list(payload, "open_roadmap_items", str, issues)

    artifacts = payload.get("expected_generated_artifacts")
    if not isinstance(artifacts, dict):
        issues.append("expected_generated_artifacts must be an object")
    else:
        _require_list(artifacts, "portfolio", str, issues, "expected_generated_artifacts")
        _require_list(artifacts, "benchmarks", str, issues, "expected_generated_artifacts")
        _require_list(artifacts, "realdata", str, issues, "expected_generated_artifacts")

    version = payload.get("version")
    if isinstance(version, str) and not re.match(r"^\d+\.\d+\.\d+$", version):
        issues.append("version must be a semantic project version such as 1.0.0")
    if not str(payload.get("commit", "")).strip():
        issues.append("commit must be a hash or explicit placeholder")
    return _prefix(path, issues)


def validate_portfolio_metrics(payload: dict[str, Any], path: Path) -> list[str]:
    """Validate the portfolio metrics JSON produced by the pipeline."""

    issues: list[str] = []
    for key in ["input", "preprocessing", "features", "registration", "segmentation", "runtime"]:
        _require_type(payload, key, dict, issues)
    if isinstance(payload.get("input"), dict):
        _require_nonnegative_number(payload["input"], "num_points", issues, "input")
    if isinstance(payload.get("preprocessing"), dict):
        _require_nonnegative_number(
            payload["preprocessing"], "num_points_after", issues, "preprocessing"
        )
    if isinstance(payload.get("registration"), dict):
        _require_nonnegative_number(payload["registration"], "rmse_after", issues, "registration")
        _require_number_range(payload["registration"], "fitness", 0.0, 1.0, issues, "registration")
    if isinstance(payload.get("segmentation"), dict):
        _require_nonnegative_number(payload["segmentation"], "num_clusters", issues, "segmentation")
        _require_number_range(
            payload["segmentation"], "noise_ratio", 0.0, 1.0, issues, "segmentation"
        )
    if isinstance(payload.get("runtime"), dict):
        _require_nonnegative_number(payload["runtime"], "total_seconds", issues, "runtime")
    return _prefix(path, issues)


def validate_benchmark_json(payload: dict[str, Any], path: Path) -> list[str]:
    """Validate benchmark JSON emitted by the benchmark API/CLI."""

    issues: list[str] = []
    _require_type(payload, "benchmark", str, issues)
    _require_type(payload, "metadata", dict, issues)
    _require_type(payload, "conclusion", str, issues)
    _require_type(payload, "rows", list, issues)

    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        _require_type(metadata, "parameters", dict, issues, "metadata")
        _require_type(metadata, "repeat", dict, issues, "metadata")
        _require_type(metadata, "memory", dict, issues, "metadata")
        repeat = metadata.get("repeat")
        if isinstance(repeat, dict):
            repeat_count = repeat.get("count")
            if isinstance(repeat_count, (int, float)) and not isinstance(repeat_count, bool):
                if repeat_count < 1:
                    issues.append("metadata.repeat.count must be at least 1")
                if repeat_count > 1:
                    _require_type(repeat, "statistics", dict, issues, "metadata.repeat")
                    _validate_repeat_rows(payload, repeat, int(repeat_count), issues)
            else:
                issues.append("metadata.repeat.count must be a positive number")
        memory = metadata.get("memory")
        if isinstance(memory, dict):
            _require_type(memory, "available", bool, issues, "metadata.memory")
            _require_type(memory, "method", str, issues, "metadata.memory")
            if memory.get("available") is True:
                _require_nonnegative_number(memory, "peak_bytes", issues, "metadata.memory")

    rows = payload.get("rows")
    if isinstance(rows, list) and not rows:
        issues.append("rows must not be empty")
    return _prefix(path, issues)


def _validate_repeat_rows(
    payload: dict[str, Any],
    repeat: dict[str, Any],
    repeat_count: int,
    issues: list[str],
) -> None:
    rows = payload.get("rows")
    timing_fields = repeat.get("timing_fields")
    statistics = repeat.get("statistics")
    aggregates = statistics.get("aggregates") if isinstance(statistics, dict) else []
    if not isinstance(rows, list) or not rows:
        return
    if not isinstance(timing_fields, list) or not timing_fields:
        issues.append("metadata.repeat.timing_fields must list timing fields when repeat > 1")
        return
    if not isinstance(aggregates, list) or not aggregates:
        issues.append("metadata.repeat.statistics.aggregates must list repeat aggregates")
        return
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        if row.get("repeat_count") != repeat_count:
            issues.append(f"rows[{row_index}].repeat_count must equal {repeat_count}")
        row_fields = [field for field in timing_fields if field in row]
        if "suite" not in row:
            row_fields = timing_fields
        if not row_fields:
            issues.append(f"rows[{row_index}] has no declared timing fields")
        for field in row_fields:
            if field not in row:
                issues.append(f"rows[{row_index}] missing timing field {field}")
                continue
            for aggregate in aggregates:
                _require_nonnegative_number(
                    row,
                    f"{field}_{aggregate}",
                    issues,
                    f"rows[{row_index}]",
                )


def validate_file(path: str | Path, schema: str) -> SchemaCheck:
    """Load and validate one artifact file."""

    artifact = Path(path)
    try:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return SchemaCheck(artifact, schema, [f"{artifact}: missing JSON artifact"])
    except json.JSONDecodeError as exc:
        return SchemaCheck(artifact, schema, [f"{artifact}: invalid JSON ({exc})"])
    if not isinstance(payload, dict):
        return SchemaCheck(artifact, schema, [f"{artifact}: JSON root must be an object"])

    validators = {
        "release": validate_release_manifest,
        "portfolio": validate_portfolio_metrics,
        "benchmark": validate_benchmark_json,
    }
    return SchemaCheck(artifact, schema, validators[schema](payload, artifact))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-manifest",
        type=Path,
        default=DEFAULT_RELEASE_MANIFEST,
        help="Release artifact manifest JSON to validate.",
    )
    parser.add_argument("--portfolio-metrics", type=Path, help="Portfolio metrics JSON.")
    parser.add_argument("--benchmark-json", type=Path, help="Benchmark JSON report.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    checks = [validate_file(args.release_manifest, "release")]
    if args.portfolio_metrics:
        checks.append(validate_file(args.portfolio_metrics, "portfolio"))
    if args.benchmark_json:
        checks.append(validate_file(args.benchmark_json, "benchmark"))

    for check in checks:
        if check.issues:
            print(f"{check.schema} schema failed for {check.path}:")
            for issue in check.issues:
                print(f"- {issue}")
        else:
            print(f"{check.schema} schema passed for {check.path}.")
    return 0 if all(check.success for check in checks) else 1


def _require_type(
    payload: dict[str, Any],
    key: str,
    expected_type: type,
    issues: list[str],
    prefix: str | None = None,
) -> None:
    value = payload.get(key)
    if not isinstance(value, expected_type) or (
        expected_type is not bool and isinstance(value, bool)
    ):
        issues.append(f"{_field(prefix, key)} must be {expected_type.__name__}")


def _require_list(
    payload: dict[str, Any],
    key: str,
    item_type: type,
    issues: list[str],
    prefix: str | None = None,
) -> None:
    value = payload.get(key)
    field = _field(prefix, key)
    if not isinstance(value, list) or not value:
        issues.append(f"{field} must be a non-empty list")
        return
    if any(not isinstance(item, item_type) for item in value):
        issues.append(f"{field} must contain only {item_type.__name__} values")


def _require_nonnegative_number(
    payload: dict[str, Any],
    key: str,
    issues: list[str],
    prefix: str | None = None,
) -> None:
    value = payload.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
        issues.append(f"{_field(prefix, key)} must be a non-negative number")


def _require_number_range(
    payload: dict[str, Any],
    key: str,
    minimum: float,
    maximum: float,
    issues: list[str],
    prefix: str | None = None,
) -> None:
    value = payload.get(key)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or value < minimum
        or value > maximum
    ):
        issues.append(f"{_field(prefix, key)} must be between {minimum} and {maximum}")


def _field(prefix: str | None, key: str) -> str:
    return f"{prefix}.{key}" if prefix else key


def _prefix(path: Path, issues: list[str]) -> list[str]:
    return [f"{path}: {issue}" for issue in issues]


if __name__ == "__main__":
    raise SystemExit(main())
