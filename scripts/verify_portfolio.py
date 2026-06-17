"""Run portfolio smoke checks and validate key portfolio artifacts."""

from __future__ import annotations

import argparse
import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pointcloud_geolab.api import run_portfolio_verification

EXPECTED_GALLERY_ARTIFACTS = (
    "registration_before_after.png",
    "icp_convergence_curve.png",
    "kdtree_benchmark.png",
    "ransac_outlier_benchmark.png",
    "segmentation_result.png",
    "primitive_extraction.html",
)

EXPECTED_PIPELINE_ARTIFACTS = (
    "report.md",
    "metrics.json",
    "figures/raw_pointcloud.png",
    "figures/registration_before_after.png",
    "figures/segmentation_result.png",
    "artifacts/transformation.json",
)


@dataclass(frozen=True, slots=True)
class PortfolioArtifactVerification:
    """Result of checking generated portfolio artifacts."""

    checked_files: list[Path]
    missing_files: list[Path]
    invalid_files: list[str]

    @property
    def success(self) -> bool:
        return not self.missing_files and not self.invalid_files


def missing_portfolio_artifacts(
    root: Path = ROOT,
    output_dir: str | Path = "outputs",
) -> list[Path]:
    """Return expected portfolio artifacts that are missing or empty."""

    result = verify_portfolio_outputs(root=root, output_dir=output_dir)
    return result.missing_files


def verify_portfolio_outputs(
    root: Path = ROOT,
    output_dir: str | Path = "outputs",
) -> PortfolioArtifactVerification:
    """Validate report, metrics, key images, and transform JSON."""

    output_root = _resolve_output_root(root, output_dir)
    pipeline_root = output_root / "portfolio_demo"
    gallery_root = root / "outputs" / "gallery"
    checked: list[Path] = []
    missing: list[Path] = []
    invalid: list[str] = []

    for name in EXPECTED_GALLERY_ARTIFACTS:
        _check_file(gallery_root / name, checked, missing, invalid)
    for name in EXPECTED_PIPELINE_ARTIFACTS:
        _check_file(pipeline_root / name, checked, missing, invalid)

    report = pipeline_root / "report.md"
    metrics = pipeline_root / "metrics.json"
    transform = pipeline_root / "artifacts" / "transformation.json"
    if report.exists() and report.stat().st_size > 0:
        _validate_report(report, invalid)
    if metrics.exists() and metrics.stat().st_size > 0:
        _validate_metrics(metrics, invalid)
    if transform.exists() and transform.stat().st_size > 0:
        _validate_transform_json(transform, invalid)

    for image in [
        gallery_root / "registration_before_after.png",
        gallery_root / "icp_convergence_curve.png",
        gallery_root / "kdtree_benchmark.png",
        gallery_root / "ransac_outlier_benchmark.png",
        gallery_root / "segmentation_result.png",
        pipeline_root / "figures" / "raw_pointcloud.png",
        pipeline_root / "figures" / "registration_before_after.png",
        pipeline_root / "figures" / "segmentation_result.png",
    ]:
        if image.exists() and image.stat().st_size > 0:
            _validate_png(image, invalid)

    html = gallery_root / "primitive_extraction.html"
    if html.exists() and html.stat().st_size > 0:
        text = html.read_text(encoding="utf-8", errors="ignore").lower()
        if "<html" not in text:
            invalid.append(f"{html}: expected HTML document")

    return PortfolioArtifactVerification(
        checked_files=checked,
        missing_files=missing,
        invalid_files=invalid,
    )


def _resolve_output_root(root: Path, output_dir: str | Path) -> Path:
    output_root = Path(output_dir)
    if not output_root.is_absolute():
        output_root = root / output_root
    return output_root


def _check_file(
    path: Path,
    checked: list[Path],
    missing: list[Path],
    invalid: list[str],
) -> None:
    if not path.exists():
        missing.append(path)
        return
    checked.append(path)
    if path.stat().st_size == 0:
        invalid.append(f"{path}: file is empty")


def _validate_report(path: Path, invalid: list[str]) -> None:
    text = path.read_text(encoding="utf-8", errors="ignore")
    required = [
        "Registration Results",
        "Segmentation Results",
        "Transformation matrix",
        "Current Limitations",
    ]
    for marker in required:
        if marker not in text:
            invalid.append(f"{path}: missing report section `{marker}`")


def _validate_metrics(path: Path, invalid: list[str]) -> None:
    payload = _load_json_object(path, invalid)
    if payload is None:
        return
    for section in ["input", "preprocessing", "registration", "segmentation", "runtime"]:
        if section not in payload:
            invalid.append(f"{path}: missing metrics section `{section}`")
    registration = payload.get("registration", {})
    if isinstance(registration, dict):
        _validate_transform_matrix(
            registration.get("transformation"),
            invalid,
            f"{path}: registration.transformation",
        )
        for key in ["rmse_before", "rmse_after", "fitness"]:
            if not _is_number(registration.get(key)):
                invalid.append(f"{path}: registration missing numeric `{key}`")
    segmentation = payload.get("segmentation", {})
    if isinstance(segmentation, dict):
        if not isinstance(segmentation.get("cluster_sizes"), list):
            invalid.append(f"{path}: segmentation.cluster_sizes must be a list")
        if not _is_number(segmentation.get("num_clusters")):
            invalid.append(f"{path}: segmentation missing numeric `num_clusters`")


def _validate_transform_json(path: Path, invalid: list[str]) -> None:
    payload = _load_json_object(path, invalid)
    if payload is None:
        return
    _validate_transform_matrix(payload.get("transformation"), invalid, f"{path}: transformation")
    for key in ["rmse_before", "rmse_after"]:
        if not _is_number(payload.get(key)):
            invalid.append(f"{path}: missing numeric `{key}`")


def _load_json_object(path: Path, invalid: list[str]) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        invalid.append(f"{path}: invalid JSON ({exc})")
        return None
    if not isinstance(payload, dict):
        invalid.append(f"{path}: expected JSON object")
        return None
    return payload


def _validate_transform_matrix(value: Any, invalid: list[str], label: str) -> None:
    if not isinstance(value, list) or len(value) != 4:
        invalid.append(f"{label} must be a 4x4 matrix")
        return
    for row in value:
        if not isinstance(row, list) or len(row) != 4 or not all(_is_number(item) for item in row):
            invalid.append(f"{label} must be a numeric 4x4 matrix")
            return


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify PointCloud-GeoLab portfolio evidence")
    parser.add_argument("--quick", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    result = run_portfolio_verification(output_dir=args.output_dir, quick=args.quick)
    artifacts = verify_portfolio_outputs(output_dir=args.output_dir)
    print(f"Portfolio report: {result.artifacts.get('report')}")
    print(f"Checked {len(artifacts.checked_files)} portfolio files")
    if artifacts.missing_files:
        print("Missing expected portfolio artifacts:")
        for path in artifacts.missing_files:
            print(f"- {path}")
    if artifacts.invalid_files:
        print("Invalid portfolio artifacts:")
        for message in artifacts.invalid_files:
            print(f"- {message}")
    if not result.success:
        if result.error:
            print(f"Portfolio smoke failed: {result.error}")
        failed_commands = result.data.get("failed", [])
        if failed_commands:
            print("Failed portfolio commands:")
            for item in failed_commands:
                print(f"- `{item['command']}` returned {item['returncode']}")
                stderr_lines = str(item.get("stderr") or "").splitlines()
                if stderr_lines:
                    print(f"  stderr tail: {stderr_lines[-1]}")
        missing_readme_artifacts = result.data.get("missing_readme_artifacts", [])
        if missing_readme_artifacts:
            print("Missing README artifacts:")
            for item in missing_readme_artifacts:
                print(f"- {item}")
    return 0 if result.success and artifacts.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
