"""KITTI Velodyne point cloud helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pointcloud_geolab.io.pointcloud_io import load_kitti_bin


def load_velodyne_frame(path: str | Path, include_intensity: bool = False) -> np.ndarray:
    """Load one KITTI Velodyne frame from a ``.bin`` file."""

    return load_kitti_bin(path, include_intensity=include_intensity)
