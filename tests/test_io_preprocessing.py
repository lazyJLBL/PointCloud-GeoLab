from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from pointcloud_geolab.io import load_kitti_bin, load_point_cloud, pointcloud_io, save_point_cloud
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


def test_ascii_ply_roundtrip_with_colors(tmp_path: Path) -> None:
    points = np.asarray([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    colors = np.asarray([[1.0, 0.0, 0.0], [0.0, 0.5, 1.0]])
    path = tmp_path / "points.ply"

    save_point_cloud(path, points, colors=colors)
    loaded = load_point_cloud(path)

    assert np.allclose(loaded, points)
    assert "property uchar red" in path.read_text(encoding="utf-8")


def test_ascii_pcd_roundtrip(tmp_path: Path) -> None:
    points = np.asarray([[0.0, 0.0, 0.0], [1.0, -2.0, 3.5]])
    path = tmp_path / "points.pcd"

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


def test_kitti_bin_rejects_incomplete_frame(tmp_path: Path) -> None:
    path = tmp_path / "bad.bin"
    np.asarray([1.0, 2.0, 3.0], dtype=np.float32).tofile(path)

    with pytest.raises(ValueError, match="divisible by 4"):
        load_kitti_bin(path)


def test_point_cloud_io_error_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="unsupported point cloud format"):
        save_point_cloud(tmp_path / "cloud.bad", np.zeros((2, 3)))

    with pytest.raises(ValueError, match="colors must have shape"):
        save_point_cloud(tmp_path / "cloud.ply", np.zeros((2, 3)), colors=np.zeros((1, 3)))

    bad_ply = tmp_path / "bad.ply"
    bad_ply.write_text("not-ply\n", encoding="utf-8")
    with pytest.raises(ValueError, match=rf"{re.escape(str(bad_ply))}.*bad PLY header"):
        load_point_cloud(bad_ply)

    monkeypatch.setattr(pointcloud_io, "_optional_open3d", lambda: None)
    with pytest.raises(ImportError, match="Optional dependency `open3d` is unavailable"):
        pointcloud_io.to_open3d_point_cloud(np.zeros((3, 3)))


def test_point_cloud_load_errors_include_path_for_common_formats(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = tmp_path / "missing.xyz"
    with pytest.raises(
        FileNotFoundError,
        match=rf"{re.escape(str(missing))}.*missing point cloud file",
    ):
        load_point_cloud(missing)

    unsupported = tmp_path / "cloud.abc"
    unsupported.write_text("1 2 3\n", encoding="utf-8")
    with pytest.raises(
        ValueError,
        match=rf"{re.escape(str(unsupported))}.*unsupported point cloud format",
    ):
        load_point_cloud(unsupported)

    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match=rf"{re.escape(str(empty))}.*empty"):
        load_point_cloud(empty)

    bad_xyz = tmp_path / "bad.xyz"
    bad_xyz.write_text("1 2 nope\n", encoding="utf-8")
    with pytest.raises(ValueError, match=rf"{re.escape(str(bad_xyz))}.*bad numeric"):
        load_point_cloud(bad_xyz)

    bad_bin = tmp_path / "bad.bin"
    np.asarray([1.0, 2.0, 3.0], dtype=np.float32).tofile(bad_bin)
    with pytest.raises(ValueError, match=rf"{re.escape(str(bad_bin))}.*divisible by 4"):
        load_point_cloud(bad_bin)

    bad_off = tmp_path / "bad.off"
    bad_off.write_text("NOFF\n", encoding="utf-8")
    with pytest.raises(ValueError, match=rf"{re.escape(str(bad_off))}.*bad OFF header"):
        load_point_cloud(bad_off)

    monkeypatch.setattr(pointcloud_io, "_optional_open3d", lambda: None)
    bad_ply = tmp_path / "numeric.ply"
    bad_ply.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 1",
                "property float x",
                "property float y",
                "property float z",
                "end_header",
                "1 2 nope",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=rf"{re.escape(str(bad_ply))}.*bad numeric"):
        load_point_cloud(bad_ply)

    bad_pcd = tmp_path / "numeric.pcd"
    bad_pcd.write_text(
        "\n".join(
            [
                "# .PCD v0.7",
                "FIELDS x y z",
                "SIZE 4 4 4",
                "TYPE F F F",
                "COUNT 1 1 1",
                "WIDTH 1",
                "HEIGHT 1",
                "POINTS 1",
                "DATA ascii",
                "1 2 nope",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=rf"{re.escape(str(bad_pcd))}.*bad numeric"):
        load_point_cloud(bad_pcd)


def test_stack_point_clouds_handles_empty_and_multiple_sets() -> None:
    assert pointcloud_io.stack_point_clouds([]).shape == (0, 3)

    stacked = pointcloud_io.stack_point_clouds(
        [
            np.asarray([[0.0, 0.0, 0.0]]),
            np.asarray([[1.0, 2.0, 3.0, 4.0]]),
        ]
    )

    assert stacked.shape == (2, 3)
    assert np.allclose(stacked[1], [1.0, 2.0, 3.0])


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
