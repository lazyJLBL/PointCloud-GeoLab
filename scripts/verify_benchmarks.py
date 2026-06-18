"""Verify benchmark artifacts produced by ``pointcloud_geolab benchmark``."""

from __future__ import annotations

import argparse
import csv
import json
import struct
from dataclasses import dataclass
from pathlib import Path

BENCHMARK_SUITES = ("kdtree", "icp", "ransac", "registration", "gicp", "segmentation")
ARTIFACT_EXTENSIONS = ("csv", "json", "md", "png")


@dataclass(frozen=True, slots=True)
class BenchmarkVerification:
    """Result of checking a benchmark output directory."""

    output_dir: Path
    checked_files: list[Path]
    missing_files: list[Path]
    invalid_files: list[str]

    @property
    def success(self) -> bool:
        return not self.missing_files and not self.invalid_files


def verify_benchmark_outputs(
    output_dir: str | Path = "outputs/benchmarks",
    suite: str = "all",
) -> BenchmarkVerification:
    """Check that benchmark CSV/JSON/Markdown/PNG artifacts are present and valid."""

    root = Path(output_dir)
    checked: list[Path] = []
    missing: list[Path] = []
    invalid: list[str] = []

    suites = list(BENCHMARK_SUITES) if suite == "all" else [suite]
    if suite == "all":
        _check_suite_files(root, "all", checked, missing, invalid)
        _check_summary_files(root, checked, missing, invalid)
    for name in suites:
        suite_dir = root / name if (root / name).exists() else root
        _check_suite_files(suite_dir, name, checked, missing, invalid)

    return BenchmarkVerification(
        output_dir=root,
        checked_files=checked,
        missing_files=missing,
        invalid_files=invalid,
    )


def _check_suite_files(
    directory: Path,
    suite: str,
    checked: list[Path],
    missing: list[Path],
    invalid: list[str],
) -> None:
    for extension in ARTIFACT_EXTENSIONS:
        path = directory / f"{suite}_benchmark.{extension}"
        if not path.exists():
            missing.append(path)
            continue
        checked.append(path)
        if path.stat().st_size == 0:
            invalid.append(f"{path}: file is empty")
            continue
        if extension == "csv":
            _validate_csv(path, invalid)
        elif extension == "json":
            _validate_json(path, invalid)
        elif extension == "md":
            _validate_markdown(path, invalid)
        elif extension == "png":
            _validate_png(path, invalid)


def _check_summary_files(
    directory: Path,
    checked: list[Path],
    missing: list[Path],
    invalid: list[str],
) -> None:
    for path in [directory / "benchmark_summary.md", directory / "benchmark_summary.json"]:
        if not path.exists():
            missing.append(path)
            continue
        checked.append(path)
        if path.stat().st_size == 0:
            invalid.append(f"{path}: file is empty")
            continue
        if path.suffix == ".json":
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                invalid.append(f"{path}: invalid JSON ({exc})")
                continue
            if "suites" not in payload:
                invalid.append(f"{path}: missing suites")
            elif not isinstance(payload["suites"], list) or not payload["suites"]:
                invalid.append(f"{path}: suites must be a non-empty list")
            metadata = payload.get("metadata")
            if metadata is None:
                invalid.append(f"{path}: missing metadata")
            else:
                _validate_metadata(path, metadata, invalid)
        elif "Benchmark Summary" not in path.read_text(encoding="utf-8"):
            invalid.append(f"{path}: missing Benchmark Summary heading")


def _validate_csv(path: Path, invalid: list[str]) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not reader.fieldnames:
        invalid.append(f"{path}: missing CSV header")
    if not rows:
        invalid.append(f"{path}: no benchmark rows")
    if rows and all(not any(str(value).strip() for value in row.values()) for row in rows):
        invalid.append(f"{path}: benchmark rows are empty")


def _validate_json(path: Path, invalid: list[str]) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        invalid.append(f"{path}: invalid JSON ({exc})")
        return
    if "metadata" not in payload:
        invalid.append(f"{path}: missing metadata")
    if "rows" not in payload or not payload["rows"]:
        invalid.append(f"{path}: missing benchmark rows")
    elif not isinstance(payload["rows"], list):
        invalid.append(f"{path}: rows must be a list")
    metadata = payload.get("metadata", {})
    if isinstance(metadata, dict):
        _validate_metadata(path, metadata, invalid)
        rows = payload.get("rows")
        if isinstance(rows, list):
            _validate_repeat_stats(path, rows, metadata, invalid)
    elif metadata:
        invalid.append(f"{path}: metadata must be an object")


def _validate_metadata(path: Path, metadata: object, invalid: list[str]) -> None:
    if not isinstance(metadata, dict):
        invalid.append(f"{path}: metadata must be an object")
        return
    for key in ["parameters", "data_scale", "platform", "python"]:
        if key not in metadata:
            invalid.append(f"{path}: metadata missing {key}")
    repeat = metadata.get("repeat")
    if repeat is None:
        invalid.append(f"{path}: metadata missing repeat")
    elif isinstance(repeat, dict):
        _validate_repeat_metadata(path, repeat, invalid)
    else:
        invalid.append(f"{path}: metadata.repeat must be an object")
    memory = metadata.get("memory")
    if memory is None:
        invalid.append(f"{path}: metadata missing memory")
    elif isinstance(memory, dict):
        _validate_memory_metadata(path, memory, invalid)
    else:
        invalid.append(f"{path}: metadata.memory must be an object")


def _validate_repeat_metadata(path: Path, repeat: dict[str, object], invalid: list[str]) -> None:
    count = repeat.get("count")
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        invalid.append(f"{path}: metadata.repeat missing valid count")
    timing_fields = repeat.get("timing_fields")
    if not isinstance(timing_fields, list):
        invalid.append(f"{path}: metadata.repeat missing timing_fields list")
    statistics = repeat.get("statistics")
    if not isinstance(statistics, dict):
        invalid.append(f"{path}: metadata.repeat missing statistics object")
    elif count and count > 1:
        aggregates = statistics.get("aggregates")
        for name in ["mean", "std", "min", "max"]:
            if not isinstance(aggregates, list) or name not in aggregates:
                invalid.append(f"{path}: metadata.repeat.statistics missing aggregate `{name}`")


def _validate_memory_metadata(path: Path, memory: dict[str, object], invalid: list[str]) -> None:
    available = memory.get("available")
    if not isinstance(available, bool):
        invalid.append(f"{path}: metadata.memory missing available flag")
        return
    if "method" not in memory:
        invalid.append(f"{path}: metadata.memory missing method")
    if available:
        for key in ["current_bytes", "peak_bytes"]:
            value = memory.get(key)
            if not _is_number(value) or float(value) < 0:
                invalid.append(f"{path}: metadata.memory missing non-negative `{key}`")
    elif "reason" not in memory:
        invalid.append(f"{path}: metadata.memory unavailable without reason")


def _validate_repeat_stats(
    path: Path,
    rows: list[object],
    metadata: dict[str, object],
    invalid: list[str],
) -> None:
    repeat = metadata.get("repeat", {})
    count = repeat.get("count", 1) if isinstance(repeat, dict) else 1
    if not isinstance(repeat, dict) or not isinstance(count, int) or count <= 1:
        return
    timing_fields = repeat.get("timing_fields", [])
    if not isinstance(timing_fields, list) or not timing_fields:
        invalid.append(f"{path}: repeat count > 1 but no timing fields were recorded")
        return
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            invalid.append(f"{path}: row {row_index} must be an object")
            continue
        for field in timing_fields:
            if field not in row:
                continue
            for aggregate in ["mean", "std", "min", "max"]:
                stat_key = f"{field}_{aggregate}"
                if not _is_number(row.get(stat_key)):
                    invalid.append(f"{path}: row {row_index} missing repeat stat `{stat_key}`")


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_markdown(path: Path, invalid: list[str]) -> None:
    text = path.read_text(encoding="utf-8")
    if "| " not in text:
        invalid.append(f"{path}: missing benchmark table")
    if "## Run Metadata" not in text:
        invalid.append(f"{path}: missing Run Metadata section")


def _validate_png(path: Path, invalid: list[str]) -> None:
    data = path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        invalid.append(f"{path}: missing PNG signature")
        return
    if len(data) < 33 or data[12:16] != b"IHDR":
        invalid.append(f"{path}: missing PNG IHDR chunk")
        return
    width, height = struct.unpack(">II", data[16:24])
    if width <= 0 or height <= 0:
        invalid.append(f"{path}: invalid PNG dimensions")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        "--output",
        dest="output_dir",
        type=Path,
        default=Path("outputs/benchmarks"),
    )
    parser.add_argument(
        "--suite",
        choices=[*BENCHMARK_SUITES, "all"],
        default="all",
        help="Check one suite output directory or the full benchmark output tree.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = verify_benchmark_outputs(args.output_dir, suite=args.suite)
    print(f"Checked {len(result.checked_files)} benchmark files under {result.output_dir}")
    if result.missing_files:
        print("Missing files:")
        for path in result.missing_files:
            print(f"- {path}")
    if result.invalid_files:
        print("Invalid files:")
        for message in result.invalid_files:
            print(f"- {message}")
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
