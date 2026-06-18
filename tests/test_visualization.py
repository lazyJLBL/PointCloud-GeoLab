from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pointcloud_geolab.visualization import export_point_cloud_html, label_colors, viewer


def test_label_colors_shape() -> None:
    labels = np.asarray([0, 1, -1, 2])

    colors = label_colors(labels)

    assert colors.shape == (4, 3)
    assert np.all((colors >= 0) & (colors <= 1))


def test_export_point_cloud_html_smoke(tmp_path: Path) -> None:
    pytest.importorskip("plotly")
    points = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    output = tmp_path / "cloud.html"

    export_point_cloud_html(points, None, output)

    assert output.exists()
    assert "html" in output.read_text(encoding="utf-8").lower()


def test_viewer_helpers_use_html_output_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, Path]] = []

    def fake_point_cloud_html(points, colors, output_path, title="Point Cloud"):
        calls.append((title, Path(output_path)))
        Path(output_path).write_text("<html></html>", encoding="utf-8")

    def fake_registration_html(source, target, transform, output_path, title="Registration"):
        calls.append((title, Path(output_path)))
        Path(output_path).write_text("<html></html>", encoding="utf-8")

    monkeypatch.setattr(viewer, "export_point_cloud_html", fake_point_cloud_html)
    monkeypatch.setattr(viewer, "export_registration_html", fake_registration_html)

    points = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    labels = np.asarray([0, 1])
    mask = np.asarray([True, False])
    viewer.visualize_point_cloud(points, output_path=tmp_path / "cloud.html", title="Cloud")
    viewer.visualize_clusters(points, labels, output_path=tmp_path / "clusters.html")
    viewer.visualize_inliers_outliers(points, mask, output_path=tmp_path / "inliers.html")
    viewer.visualize_registration(
        points,
        points,
        output_path=tmp_path / "registration.html",
    )

    assert [name for name, _ in calls] == [
        "Cloud",
        "Clusters",
        "Inliers and Outliers",
        "Registration",
    ]
    assert all(path.exists() for _, path in calls)
