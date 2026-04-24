"""Visualization helpers for point clouds and algorithm outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .pointcloud_io import to_open3d_point_cloud


DEFAULT_COLORS = np.asarray(
    [
        [0.0, 0.45, 0.95],
        [0.95, 0.25, 0.20],
        [0.20, 0.70, 0.35],
        [0.55, 0.55, 0.55],
    ],
    dtype=float,
)


def visualize_point_clouds(
    point_sets: Sequence[np.ndarray],
    colors: Sequence[Sequence[float]] | None = None,
    window_name: str = "PointCloud-GeoLab",
) -> None:
    """Open an interactive Open3D window for one or more point clouds."""

    if colors is None:
        colors_arr = DEFAULT_COLORS
    else:
        colors_arr = np.asarray(colors, dtype=float)
    geometries = []
    for i, points in enumerate(point_sets):
        pts = np.asarray(points, dtype=float)
        color = np.tile(colors_arr[i % len(colors_arr)], (len(pts), 1))
        geometries.append(to_open3d_point_cloud(pts, color))

    import open3d as o3d  # type: ignore

    o3d.visualization.draw_geometries(geometries, window_name=window_name)


def save_point_cloud_projection(
    path: str | Path,
    point_sets: Sequence[np.ndarray],
    colors: Sequence[Sequence[float]] | None = None,
    labels: Sequence[str] | None = None,
    title: str | None = None,
    max_points_per_set: int = 4000,
) -> None:
    """Save a static 3D scatter projection as PNG."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    colors_arr = DEFAULT_COLORS if colors is None else np.asarray(colors, dtype=float)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    all_points = []
    for i, points in enumerate(point_sets):
        pts = np.asarray(points, dtype=float)
        if pts.size == 0:
            continue
        all_points.append(pts)
        if len(pts) > max_points_per_set:
            step = int(np.ceil(len(pts) / max_points_per_set))
            pts = pts[::step]
        label = labels[i] if labels is not None and i < len(labels) else None
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            s=3,
            alpha=0.75,
            color=colors_arr[i % len(colors_arr)],
            label=label,
        )

    if title:
        ax.set_title(title)
    if labels:
        ax.legend(loc="best")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    if all_points:
        _set_axes_equal(ax, np.vstack(all_points))
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def save_error_curve(path: str | Path, errors: Sequence[float], title: str = "ICP RMSE") -> None:
    """Save a convergence curve image."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(len(errors)), errors, marker="o", linewidth=1.5, markersize=3)
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("RMSE")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _set_axes_equal(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0)
    if radius == 0:
        radius = 1.0
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)

