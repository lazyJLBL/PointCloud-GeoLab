"""Demo: robust plane, sphere, and cylinder fitting with RANSAC."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pointcloud_geolab.api import run_primitive_fitting
from pointcloud_geolab.io import save_point_cloud


def main() -> int:
    rng = np.random.default_rng(110)
    out = ROOT / "outputs" / "primitives"
    out.mkdir(parents=True, exist_ok=True)

    center = np.asarray([0.3, -0.2, 0.1])
    dirs = rng.normal(size=(220, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    sphere = center + 0.6 * dirs + rng.normal(0, 0.003, size=(220, 3))
    scene = np.vstack([sphere, rng.uniform(-1.5, 1.5, size=(80, 3))])
    input_path = out / "sphere_scene.ply"
    save_point_cloud(input_path, scene)

    result = run_primitive_fitting(
        input_path,
        model="sphere",
        output=out / "sphere_inliers.ply",
        output_dir=out,
        threshold=0.03,
        save_results=True,
        export_html=out / "sphere_fit.html",
    )
    print(result.to_dict()["data"]["model_params"])
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
