"""Run portfolio smoke checks and write outputs/portfolio_check_report.md."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pointcloud_geolab.api import run_portfolio_verification


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify PointCloud-GeoLab portfolio evidence")
    parser.add_argument("--quick", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    result = run_portfolio_verification(output_dir=args.output_dir, quick=args.quick)
    print(f"Portfolio report: {result.artifacts.get('report')}")
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
