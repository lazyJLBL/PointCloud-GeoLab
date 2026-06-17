"""Verify benchmark artifacts produced by ``pointcloud_geolab benchmark``."""

from __future__ import annotations

import argparse
import csv
import json
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
    metadata = payload.get("metadata", {})
    if isinstance(metadata, dict):
        for key in ["parameters", "data_scale", "platform", "python"]:
            if key not in metadata:
                invalid.append(f"{path}: metadata missing {key}")


def _validate_markdown(path: Path, invalid: list[str]) -> None:
    text = path.read_text(encoding="utf-8")
    if "| " not in text:
        invalid.append(f"{path}: missing benchmark table")
    if "## Run Metadata" not in text:
        invalid.append(f"{path}: missing Run Metadata section")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/benchmarks"))
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
