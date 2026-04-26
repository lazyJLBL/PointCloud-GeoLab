"""Demo: optional PointNet training on synthetic shapes."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pointcloud_geolab.api import run_train_pointnet


def main() -> int:
    out = ROOT / "outputs" / "ml"
    result = run_train_pointnet(
        output=out / "pointnet_model.pt",
        output_dir=out,
        epochs=1,
        samples_per_class=4,
        points_per_sample=32,
    )
    print(result.to_dict()["metrics"] if result.success else result.error)
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
