"""Preview sampling for uploaded point clouds."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pointcloud_geolab.io.pointcloud_io import load_point_cloud


def preview_point_cloud(path: str | Path, point_limit: int = 10000) -> dict[str, object]:
    """Load a point cloud and return a deterministic sample for the frontend."""

    points = load_point_cloud(path)
    if point_limit < 1:
        raise ValueError("point_limit must be positive")
    sampled = points
    if len(points) > point_limit:
        indices = np.linspace(0, len(points) - 1, num=point_limit, dtype=int)
        sampled = points[indices]
    return {
        "point_count": int(len(points)),
        "sampled_count": int(len(sampled)),
        "points": sampled[:, :3].astype(float).tolist(),
    }
