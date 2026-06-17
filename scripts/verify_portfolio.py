"""Run portfolio smoke checks and write outputs/portfolio_check_report.md."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
    "figures/registration_before_after.png",
    "figures/segmentation_result.png",
    "artifacts/transformation.json",
)


def missing_portfolio_artifacts(
    root: Path = ROOT,
    output_dir: str | Path = "outputs",
) -> list[Path]:
    """Return expected portfolio artifacts that are missing or empty."""

    output_root = Path(output_dir)
    if not output_root.is_absolute():
        output_root = root / output_root
    candidates = [
        *(root / "outputs" / "gallery" / name for name in EXPECTED_GALLERY_ARTIFACTS),
        *(output_root / "portfolio_demo" / name for name in EXPECTED_PIPELINE_ARTIFACTS),
    ]
    return [path for path in candidates if not path.exists() or path.stat().st_size == 0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify PointCloud-GeoLab portfolio evidence")
    parser.add_argument("--quick", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    result = run_portfolio_verification(output_dir=args.output_dir, quick=args.quick)
    missing = missing_portfolio_artifacts(output_dir=args.output_dir)
    print(f"Portfolio report: {result.artifacts.get('report')}")
    if missing:
        print("Missing expected portfolio artifacts:")
        for path in missing:
            print(f"- {path}")
    return 0 if result.success and not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
