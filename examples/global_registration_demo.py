"""Demo: FPFH global registration followed by ICP refinement."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pointcloud_geolab.api import run_global_registration
from pointcloud_geolab.io import save_point_cloud
from pointcloud_geolab.utils.transform import apply_transform, rotation_matrix_from_euler


def main() -> int:
    rng = np.random.default_rng(100)
    target = np.vstack(
        [
            rng.normal([0, 0, 0], 0.08, size=(120, 3)),
            rng.normal([0.6, 0.15, 0.1], 0.05, size=(120, 3)),
            rng.normal([-0.25, 0.45, -0.15], 0.04, size=(120, 3)),
        ]
    )
    rotation = rotation_matrix_from_euler(0.1, -0.05, 0.35)
    translation = np.asarray([0.18, -0.08, 0.05])
    source = apply_transform(target, rotation, translation)

    out = ROOT / "outputs" / "registration"
    out.mkdir(parents=True, exist_ok=True)
    source_path = out / "source_demo.ply"
    target_path = out / "target_demo.ply"
    save_point_cloud(source_path, source)
    save_point_cloud(target_path, target)

    result = run_global_registration(
        source_path,
        target_path,
        output=out / "aligned_source.ply",
        save_transform=out / "transform.txt",
        output_dir=out,
        voxel_size=0.08,
        save_results=True,
        export_html=out / "registration.html",
    )
    print(result.to_dict()["metrics"])
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
