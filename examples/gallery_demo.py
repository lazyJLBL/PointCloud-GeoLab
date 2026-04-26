"""Generate small gallery assets for README or portfolio use."""

# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pointcloud_geolab.api import run_benchmark, run_primitive_fitting, run_segmentation
from pointcloud_geolab.datasets import make_mixed_scene, make_sphere
from pointcloud_geolab.io import save_point_cloud
from pointcloud_geolab.io.visualization import save_point_cloud_projection
from pointcloud_geolab.visualization import label_colors


def main() -> int:
    gallery = ROOT / "outputs" / "gallery"
    gallery.mkdir(parents=True, exist_ok=True)

    sphere = make_sphere(220, center=[0.2, -0.1, 0.3], radius=0.7, noise=0.003, random_state=200)
    outliers = np.random.default_rng(201).uniform(-1.5, 1.5, size=(70, 3))
    primitive_scene = np.vstack([sphere, outliers])
    primitive_path = gallery / "primitive_scene.ply"
    save_point_cloud(primitive_path, primitive_scene)
    run_primitive_fitting(
        primitive_path,
        "sphere",
        output_dir=gallery / "primitive_fit",
        threshold=0.03,
        save_results=True,
    )

    scene, _ = make_mixed_scene(random_state=210, outliers=40)
    scene_path = gallery / "segmentation_scene.ply"
    save_point_cloud(scene_path, scene)
    segmentation = run_segmentation(
        scene_path,
        output=gallery / "segmented_scene.ply",
        output_dir=gallery / "segmentation",
        method="dbscan",
        eps=0.18,
        min_points=10,
    )
    labels = np.asarray(segmentation.to_dict()["data"]["labels"], dtype=int)
    colors = label_colors(labels)
    save_point_cloud_projection(
        gallery / "segmentation_preview.png",
        [scene],
        colors=[colors[0] if len(colors) else [0.1, 0.45, 0.85]],
        title="Segmentation Preview",
    )

    run_benchmark("ransac", output_dir=gallery / "benchmarks", quick=True)
    print(f"Gallery assets written to {gallery}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
