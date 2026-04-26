from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pointcloud_geolab.visualization import export_point_cloud_html, label_colors


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
