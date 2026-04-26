from __future__ import annotations

from pathlib import Path

import numpy as np

from pointcloud_geolab.io import load_kitti_bin, load_point_cloud, save_point_cloud
from pointcloud_geolab.preprocessing import (
    crop_by_aabb,
    estimate_normals,
    farthest_point_sample,
    normalize_point_cloud,
    random_sample,
    voxel_downsample,
)


def test_txt_point_cloud_io(tmp_path: Path) -> None:
    points = np.asarray([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    path = tmp_path / "points.txt"

    save_point_cloud(path, points)
    loaded = load_point_cloud(path)

    assert np.allclose(loaded, points)


def test_kitti_bin_reader_supports_intensity(tmp_path: Path) -> None:
    path = tmp_path / "frame.bin"
    raw = np.asarray([[1.0, 2.0, 3.0, 0.5], [4.0, 5.0, 6.0, 0.7]], dtype=np.float32)
    raw.tofile(path)

    xyz = load_kitti_bin(path)
    xyzi = load_kitti_bin(path, include_intensity=True)

    assert xyz.shape == (2, 3)
    assert xyzi.shape == (2, 4)


def test_preprocessing_filters_shapes() -> None:
    rng = np.random.default_rng(50)
    points = rng.normal(size=(100, 3))

    down = voxel_downsample(points, voxel_size=0.5)
    cropped, crop_ids = crop_by_aabb(points, [-1, -1, -1], [1, 1, 1])
    sampled, sample_ids = random_sample(points, 10, random_state=51)
    farthest, farthest_ids = farthest_point_sample(points, 12, random_state=52)
    normalized, center, scale = normalize_point_cloud(points)
    normals = estimate_normals(points, k=8)

    assert len(down) <= len(points)
    assert len(cropped) == len(crop_ids)
    assert sampled.shape == (10, 3)
    assert len(sample_ids) == 10
    assert farthest.shape == (12, 3)
    assert len(farthest_ids) == 12
    assert normalized.shape == points.shape
    assert center.shape == (3,)
    assert scale > 0
    assert normals.shape == points.shape
