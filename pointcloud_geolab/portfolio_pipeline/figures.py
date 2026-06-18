"""Figure writers for the portfolio pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from pointcloud_geolab.portfolio_pipeline.metrics import _sample_indices
from pointcloud_geolab.visualization import label_colors


def _save_registration_figure(
    path: Path,
    source: np.ndarray,
    target: np.ndarray,
    aligned: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(11, 5))
    before = fig.add_subplot(121, projection="3d")
    after = fig.add_subplot(122, projection="3d")
    _scatter_cloud(before, target, "#1f77b4", "target")
    _scatter_cloud(before, source, "#d62728", "source before")
    before.set_title("Before ICP")
    before.legend(loc="best")
    _scatter_cloud(after, target, "#1f77b4", "target")
    _scatter_cloud(after, aligned, "#2ca02c", "source after")
    after.set_title("After ICP")
    after.legend(loc="best")
    all_points = np.vstack([source, target, aligned])
    _set_axes_equal(before, all_points)
    _set_axes_equal(after, all_points)
    for axis in [before, after]:
        axis.set_xlabel("X")
        axis.set_ylabel("Y")
        axis.set_zlabel("Z")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_segmentation_figure(
    path: Path,
    points: np.ndarray,
    labels: np.ndarray,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = np.asarray(points, dtype=float)
    colors = label_colors(labels)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    for label in sorted(int(value) for value in np.unique(labels)):
        mask = labels == label
        name = "noise" if label < 0 else f"cluster {label}"
        color = colors[np.flatnonzero(mask)[0]]
        ax.scatter(
            pts[mask, 0],
            pts[mask, 1],
            pts[mask, 2],
            s=4,
            alpha=0.8,
            color=color,
            label=name,
        )
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="best")
    _set_axes_equal(ax, pts)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_bounding_box_or_normals_figure(
    path: Path,
    points: np.ndarray,
    corners: np.ndarray,
    normals: np.ndarray,
    rng: np.random.Generator,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = np.asarray(points, dtype=float)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    _scatter_cloud(ax, pts, "#1f77b4", "processed")
    for start, end in _aabb_edges():
        segment = corners[[start, end]]
        ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], color="#d62728", linewidth=1.5)
    normal_indices = _sample_indices(len(pts), min(40, len(pts)), rng)
    normal_scale = max(float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))) / 25.0, 0.02)
    ax.quiver(
        pts[normal_indices, 0],
        pts[normal_indices, 1],
        pts[normal_indices, 2],
        normals[normal_indices, 0],
        normals[normal_indices, 1],
        normals[normal_indices, 2],
        length=normal_scale,
        color="#2ca02c",
        linewidth=0.8,
        normalize=True,
    )
    ax.set_title("AABB and estimated normals")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    _set_axes_equal(ax, np.vstack([pts, corners]))
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _scatter_cloud(ax: Any, points: np.ndarray, color: str, label: str) -> None:
    pts = np.asarray(points, dtype=float)
    if len(pts) > 3000:
        step = int(np.ceil(len(pts) / 3000))
        pts = pts[::step]
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=3, alpha=0.72, color=color, label=label)


def _aabb_edges() -> list[tuple[int, int]]:
    return [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 3),
        (4, 5),
        (4, 6),
        (5, 7),
        (6, 7),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]


def _set_axes_equal(ax: Any, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0)
    if radius <= 0:
        radius = 1.0
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)
