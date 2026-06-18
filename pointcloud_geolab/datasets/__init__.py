"""Dataset readers for real-world point cloud formats."""

from .fixtures import (
    FixtureValidation,
    OffMesh,
    load_kitti_like_bin,
    load_modelnet_like_off,
    validate_fixture_manifest,
)
from .kitti import load_velodyne_frame
from .synthetic import make_box, make_cylinder, make_mixed_scene, make_plane, make_sphere

__all__ = [
    "FixtureValidation",
    "OffMesh",
    "load_kitti_like_bin",
    "load_modelnet_like_off",
    "load_velodyne_frame",
    "make_box",
    "make_cylinder",
    "make_mixed_scene",
    "make_plane",
    "make_sphere",
    "validate_fixture_manifest",
]
