"""Export point cloud visualizations to files."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pointcloud_geolab.io.pointcloud_io import save_point_cloud
from pointcloud_geolab.utils.transform import apply_homogeneous_transform


def label_colors(labels: np.ndarray) -> np.ndarray:
    """Map integer labels to RGB colors in ``[0, 1]``."""

    labs = np.asarray(labels, dtype=int)
    colors = np.zeros((len(labs), 3), dtype=float)
    palette = np.asarray(
        [
            [0.12, 0.47, 0.71],
            [1.00, 0.50, 0.05],
            [0.17, 0.63, 0.17],
            [0.84, 0.15, 0.16],
            [0.58, 0.40, 0.74],
            [0.55, 0.34, 0.29],
            [0.89, 0.47, 0.76],
            [0.50, 0.50, 0.50],
            [0.74, 0.74, 0.13],
            [0.09, 0.75, 0.81],
        ],
        dtype=float,
    )
    for i, label in enumerate(labs):
        colors[i] = [0.2, 0.2, 0.2] if label < 0 else palette[label % len(palette)]
    return colors


def save_colored_point_cloud(path: str | Path, points: np.ndarray, labels: np.ndarray) -> None:
    """Save a colored PLY/PCD/XYZ point cloud from cluster labels."""

    save_point_cloud(path, points, colors=label_colors(labels))


def export_point_cloud_html(
    points: np.ndarray,
    colors: np.ndarray | None,
    output_path: str | Path,
    title: str = "PointCloud-GeoLab",
) -> None:
    """Export an interactive Plotly point cloud HTML file."""

    fig = _make_scatter3d(points, colors, title=title, name="points")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output), include_plotlyjs="cdn")


def export_registration_html(
    source: np.ndarray,
    target: np.ndarray,
    transform: np.ndarray | None,
    output_path: str | Path,
    title: str = "Registration",
) -> None:
    """Export an interactive registration comparison HTML file."""

    transformed = (
        apply_homogeneous_transform(source, transform) if transform is not None else source
    )
    go = _require_plotly()
    fig = go.Figure()
    _add_cloud(fig, target, [0.1, 0.45, 0.85], "target")
    _add_cloud(fig, transformed, [0.9, 0.25, 0.2], "source")
    fig.update_layout(title=title, scene_aspectmode="data")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output), include_plotlyjs="cdn")


def _make_scatter3d(points: np.ndarray, colors: np.ndarray | None, title: str, name: str):
    go = _require_plotly()
    pts = np.asarray(points, dtype=float)
    color_values = None
    if colors is not None:
        cols = np.asarray(colors, dtype=float)
        color_values = [f"rgb({int(r*255)},{int(g*255)},{int(b*255)})" for r, g, b in cols]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            mode="markers",
            name=name,
            marker={"size": 2, "color": color_values or "#1f77b4", "opacity": 0.85},
        )
    )
    fig.update_layout(title=title, scene_aspectmode="data")
    return fig


def _add_cloud(fig, points: np.ndarray, color: list[float], name: str) -> None:
    go = _require_plotly()
    pts = np.asarray(points, dtype=float)
    rgb = f"rgb({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)})"
    fig.add_trace(
        go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            mode="markers",
            name=name,
            marker={"size": 2, "color": rgb, "opacity": 0.8},
        )
    )


def _require_plotly():
    try:
        import plotly.graph_objects as go  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Plotly is required for HTML export. Install with `python -m pip install plotly`."
        ) from exc
    return go
