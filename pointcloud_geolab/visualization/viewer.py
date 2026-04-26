"""High-level visualization helpers for point clouds and algorithm outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pointcloud_geolab.visualization.export import (
    export_point_cloud_html,
    export_registration_html,
    label_colors,
)


def visualize_point_cloud(
    points: np.ndarray,
    colors: np.ndarray | None = None,
    output_path: str | Path | None = None,
    title: str = "Point Cloud",
) -> None:
    """Visualize a point cloud by exporting HTML or opening an Open3D window."""

    if output_path:
        export_point_cloud_html(points, colors, output_path, title=title)
        return
    _open3d_draw([points], colors=[colors] if colors is not None else None, title=title)


def visualize_registration(
    source: np.ndarray,
    target: np.ndarray,
    transform: np.ndarray | None = None,
    output_path: str | Path | None = None,
    title: str = "Registration",
) -> None:
    """Visualize source/target registration before or after a transform."""

    if output_path:
        export_registration_html(source, target, transform, output_path, title=title)
        return

    from pointcloud_geolab.utils.transform import apply_homogeneous_transform

    shown_source = (
        apply_homogeneous_transform(source, transform) if transform is not None else source
    )
    _open3d_draw(
        [target, shown_source],
        colors=[
            np.tile([0.1, 0.45, 0.85], (len(target), 1)),
            np.tile([0.9, 0.25, 0.2], (len(shown_source), 1)),
        ],
        title=title,
    )


def visualize_clusters(
    points: np.ndarray,
    labels: np.ndarray,
    output_path: str | Path | None = None,
    title: str = "Clusters",
) -> None:
    """Visualize cluster labels with deterministic colors."""

    colors = label_colors(labels)
    visualize_point_cloud(points, colors=colors, output_path=output_path, title=title)


def visualize_inliers_outliers(
    points: np.ndarray,
    inlier_mask: np.ndarray,
    output_path: str | Path | None = None,
    title: str = "Inliers and Outliers",
) -> None:
    """Visualize inliers in blue and outliers in red."""

    mask = np.asarray(inlier_mask, dtype=bool)
    colors = np.zeros((len(points), 3), dtype=float)
    colors[mask] = [0.1, 0.45, 0.85]
    colors[~mask] = [0.9, 0.25, 0.2]
    visualize_point_cloud(points, colors=colors, output_path=output_path, title=title)


def _open3d_draw(
    point_sets: list[np.ndarray],
    colors: list[np.ndarray | None] | None,
    title: str,
) -> None:
    try:
        import open3d as o3d  # type: ignore

        from pointcloud_geolab.io.pointcloud_io import to_open3d_point_cloud
    except ImportError as exc:
        raise ImportError(
            "Open3D is required for interactive windows. Use output_path for HTML export."
        ) from exc

    geometries = []
    for i, points in enumerate(point_sets):
        color = None if colors is None else colors[i]
        geometries.append(to_open3d_point_cloud(points, color))
    o3d.visualization.draw_geometries(geometries, window_name=title)
