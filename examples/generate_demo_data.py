"""Generate deterministic synthetic point cloud data for demos."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pointcloud_geolab.datasets import make_mixed_scene
from pointcloud_geolab.io.pointcloud_io import save_point_cloud
from pointcloud_geolab.utils.transform import apply_transform, rotation_matrix_from_euler


def make_object_points(rng: np.random.Generator, n: int = 900) -> np.ndarray:
    phi = rng.uniform(0, np.pi, n)
    theta = rng.uniform(0, 2 * np.pi, n)
    radius = 1.0 + 0.08 * np.sin(4 * theta) * np.sin(3 * phi)
    body = np.column_stack(
        [
            0.65 * radius * np.sin(phi) * np.cos(theta),
            0.38 * radius * np.sin(phi) * np.sin(theta),
            0.85 * radius * np.cos(phi),
        ]
    )
    ear_a = rng.normal(loc=[0.18, 0.10, 0.82], scale=[0.08, 0.04, 0.18], size=(120, 3))
    ear_b = rng.normal(loc=[-0.18, 0.10, 0.82], scale=[0.08, 0.04, 0.18], size=(120, 3))
    points = np.vstack([body, ear_a, ear_b])
    points += rng.normal(scale=0.004, size=points.shape)
    return points


def make_room_points(rng: np.random.Generator) -> np.ndarray:
    floor_xy = rng.uniform([-2.0, -1.5], [2.0, 1.5], size=(1200, 2))
    floor = np.column_stack([floor_xy, rng.normal(0.0, 0.006, size=1200)])
    wall_yz = rng.uniform([-1.5, 0.0], [1.5, 1.8], size=(500, 2))
    wall = np.column_stack([rng.normal(-2.0, 0.006, size=500), wall_yz])
    table_xy = rng.uniform([-0.8, -0.5], [0.8, 0.5], size=(350, 2))
    table = np.column_stack([table_xy, rng.normal(0.75, 0.006, size=350)])
    outliers = rng.uniform([-2.2, -1.7, -0.2], [2.2, 1.7, 1.9], size=(120, 3))
    return np.vstack([floor, wall, table, outliers])


def make_rotated_box(rng: np.random.Generator) -> np.ndarray:
    local = rng.uniform([-0.8, -0.25, -0.15], [0.8, 0.25, 0.15], size=(1000, 3))
    rotation = rotation_matrix_from_euler(0.35, -0.18, 0.55)
    translation = np.asarray([0.2, -0.1, 0.35])
    points = apply_transform(local, rotation, translation)
    return points + rng.normal(scale=0.003, size=points.shape)


def make_lidar_scene(rng: np.random.Generator) -> np.ndarray:
    ground_xy = rng.uniform([-2.5, -2.0], [2.5, 2.0], size=(600, 2))
    ground = np.column_stack([ground_xy, rng.normal(0.0, 0.004, size=len(ground_xy))])
    objects = []
    for center, scale in [
        ([0.7, 0.5, 0.35], [0.12, 0.10, 0.18]),
        ([-0.9, -0.4, 0.45], [0.16, 0.12, 0.22]),
        ([1.4, -0.8, 0.25], [0.10, 0.08, 0.12]),
    ]:
        objects.append(rng.normal(loc=center, scale=scale, size=(120, 3)))
    outliers = rng.uniform([-2.5, -2.0, 0.05], [2.5, 2.0, 1.2], size=(50, 3))
    return np.vstack([ground, *objects, outliers])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic demo point clouds.")
    parser.add_argument(
        "--output",
        "--output-dir",
        dest="output_dir",
        type=Path,
        default=ROOT / "data",
        help="directory for generated demo point clouds",
    )
    args = parser.parse_args(argv)

    rng = np.random.default_rng(42)
    data_dir = args.output_dir
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    target = make_object_points(rng)
    rotation = rotation_matrix_from_euler(0.08, -0.06, 0.12)
    translation = np.asarray([0.28, -0.12, 0.18])
    source = apply_transform(target, rotation, translation)

    room = make_room_points(rng)
    object_points = make_rotated_box(rng)
    synthetic_scene, _ = make_mixed_scene(random_state=44, noise=0.006, outliers=60)
    lidar_scene = make_lidar_scene(rng)

    save_point_cloud(data_dir / "bunny_target.ply", target)
    save_point_cloud(data_dir / "bunny_source.ply", source)
    save_point_cloud(data_dir / "room.pcd", room)
    save_point_cloud(data_dir / "room.xyz", room)
    save_point_cloud(data_dir / "object.ply", object_points)
    save_point_cloud(data_dir / "synthetic_scene.ply", synthetic_scene)
    save_point_cloud(data_dir / "lidar_scene.ply", lidar_scene)
    print(f"Generated demo data in {data_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
