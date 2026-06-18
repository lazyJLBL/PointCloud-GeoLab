"""Validate tiny synthetic dataset fixtures committed for format smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

from pointcloud_geolab.datasets.fixtures import validate_fixture_manifest

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "tests" / "fixtures" / "datasets" / "manifest.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = validate_fixture_manifest(args.manifest)
    print(f"Checked {len(result.checked_files)} dataset fixture files.")
    if result.issues:
        print("Dataset fixture issues:")
        for issue in result.issues:
            print(f"- {issue}")
    else:
        print("Dataset fixture checks passed.")
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
