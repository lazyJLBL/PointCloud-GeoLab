"""Verify the user-provided real-data workflow without downloading datasets."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.kitti_lidar_segmentation import build_parser as build_kitti_parser
from examples.kitti_lidar_segmentation import run_kitti_segmentation

EXPECTED_ARTIFACTS = (
    "metrics.json",
    "report.md",
    "report.html",
    "kitti_bev.png",
    "kitti_clusters.png",
    "kitti_height_histogram.png",
    "cluster_report.md",
    "kitti_clusters.ply",
)


def write_tiny_kitti_frame(path: str | Path, seed: int = 7) -> Path:
    """Write a synthetic KITTI-like frame for CI format and workflow smoke tests."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    xs = np.linspace(-6.0, 6.0, 14)
    ys = np.linspace(-4.0, 4.0, 10)
    xx, yy = np.meshgrid(xs, ys)
    ground = np.column_stack(
        [
            xx.ravel(),
            yy.ravel(),
            rng.normal(0.0, 0.015, xx.size),
            rng.uniform(0.2, 0.6, xx.size),
        ]
    )
    cluster_a = rng.normal(scale=[0.16, 0.14, 0.08, 0.05], size=(34, 4))
    cluster_a += np.array([2.0, 1.3, 0.8, 0.7])
    cluster_b = rng.normal(scale=[0.18, 0.15, 0.09, 0.05], size=(36, 4))
    cluster_b += np.array([-2.5, -1.2, 1.0, 0.8])
    points = np.vstack([ground, cluster_a, cluster_b]).astype(np.float32)
    points.tofile(output)
    return output


def verify_kitti_workflow_outputs(output_dir: str | Path) -> list[str]:
    """Return validation issues for KITTI-like workflow artifacts."""

    root = Path(output_dir)
    issues: list[str] = []
    for name in EXPECTED_ARTIFACTS:
        path = root / name
        if not path.exists():
            issues.append(f"{path}: missing artifact")
            continue
        if path.stat().st_size == 0:
            issues.append(f"{path}: artifact is empty")

    metrics_path = root / "metrics.json"
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return sorted(issues)
    except json.JSONDecodeError as exc:
        issues.append(f"{metrics_path}: invalid JSON ({exc})")
        return sorted(issues)

    issues.extend(_validate_metrics(metrics_path, metrics))
    for png_name in ["kitti_bev.png", "kitti_clusters.png", "kitti_height_histogram.png"]:
        issues.extend(_validate_png(root / png_name))

    report_text = (
        (root / "report.md").read_text(encoding="utf-8") if (root / "report.md").exists() else ""
    )
    html_text = (
        (root / "report.html").read_text(encoding="utf-8")
        if (root / "report.html").exists()
        else ""
    )
    boundary = "not an official KITTI benchmark"
    if boundary not in report_text:
        issues.append(f"{root / 'report.md'}: missing KITTI benchmark boundary wording")
    if boundary not in html_text:
        issues.append(f"{root / 'report.html'}: missing KITTI benchmark boundary wording")
    return sorted(issues)


def run_realdata_verification(
    frame: str | Path | None,
    output_dir: str | Path,
    dry_run: bool = False,
) -> list[str]:
    """Run or inspect the KITTI-like workflow and return validation issues."""

    output = Path(output_dir)
    if dry_run:
        with tempfile.TemporaryDirectory(prefix="pcgl-kitti-") as temp:
            temp_root = Path(temp)
            tiny_frame = write_tiny_kitti_frame(temp_root / "000000.bin")
            args = _kitti_args(tiny_frame, temp_root / "outputs")
            run_kitti_segmentation(args)
            return verify_kitti_workflow_outputs(args.output_dir)

    if frame is None:
        frame_path = ROOT / "data" / "external" / "kitti" / "velodyne" / "000000.bin"
    else:
        frame_path = Path(frame)
    if not frame_path.exists():
        print(
            "KITTI frame not found; skipping real-data verification. "
            f"Expected user-provided data at {frame_path}.",
        )
        return []

    args = _kitti_args(frame_path, output)
    run_kitti_segmentation(args)
    return verify_kitti_workflow_outputs(output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frame",
        type=Path,
        default=ROOT / "data" / "external" / "kitti" / "velodyne" / "000000.bin",
        help="User-provided KITTI-like xyzi .bin frame.",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "kitti_segmentation")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use a temporary synthetic KITTI-like frame; no external data required.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    issues = run_realdata_verification(args.frame, args.output_dir, dry_run=args.dry_run)
    if issues:
        print("Real-data workflow verification failed:")
        for issue in issues:
            print(f"- {issue}")
        return 1
    mode = "dry-run tiny fixture" if args.dry_run else "user-provided data or friendly skip"
    print(f"Real-data workflow verification passed ({mode}).")
    return 0


def _kitti_args(frame: Path, output_dir: Path) -> argparse.Namespace:
    parser = build_kitti_parser()
    return parser.parse_args(
        [
            "--frame",
            str(frame),
            "--output-dir",
            str(output_dir),
            "--eps",
            "0.55",
            "--min-points",
            "8",
            "--max-points",
            "5000",
        ]
    )


def _validate_metrics(path: Path, payload: object) -> list[str]:
    issues: list[str] = []
    if not isinstance(payload, dict):
        return [f"{path}: JSON root must be an object"]
    for key in ["workflow", "scope", "input", "segmentation", "timing", "memory", "artifacts"]:
        if key not in payload:
            issues.append(f"{path}: missing {key}")
    if payload.get("workflow") != "kitti_lidar_segmentation":
        issues.append(f"{path}: workflow must be kitti_lidar_segmentation")
    input_section = payload.get("input", {})
    if not isinstance(input_section, dict) or int(input_section.get("points", 0)) <= 0:
        issues.append(f"{path}: input.points must be positive")
    segmentation = payload.get("segmentation", {})
    if not isinstance(segmentation, dict) or int(segmentation.get("ground_points", 0)) <= 0:
        issues.append(f"{path}: segmentation.ground_points must be positive")
    memory = payload.get("memory", {})
    if not isinstance(memory, dict) or memory.get("available") is not True:
        issues.append(f"{path}: memory metadata must be available")
    elif int(memory.get("peak_bytes", -1)) < 0:
        issues.append(f"{path}: memory.peak_bytes must be non-negative")
    limitations = " ".join(str(item) for item in payload.get("limitations", []))
    if "not an official KITTI benchmark" not in limitations:
        issues.append(f"{path}: limitations must state this is not an official KITTI benchmark")
    return issues


def _validate_png(path: Path) -> list[str]:
    if not path.exists():
        return []
    data = path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return [f"{path}: missing PNG signature"]
    if len(data) < 33 or data[12:16] != b"IHDR":
        return [f"{path}: missing PNG IHDR chunk"]
    return []


if __name__ == "__main__":
    raise SystemExit(main())
