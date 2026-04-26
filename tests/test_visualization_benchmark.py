from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pointcloud_geolab.api import run_benchmark
from pointcloud_geolab.visualization import export_point_cloud_html, label_colors


def test_label_colors_noise_is_dark() -> None:
    labels = np.asarray([0, 1, -1])
    colors = label_colors(labels)

    assert colors.shape == (3, 3)
    assert np.allclose(colors[-1], [0.2, 0.2, 0.2])


def test_export_point_cloud_html_smoke(tmp_path: Path) -> None:
    pytest.importorskip("plotly")
    points = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    output = tmp_path / "cloud.html"

    export_point_cloud_html(points, None, output)

    assert output.exists()
    assert "Plotly" in output.read_text(encoding="utf-8")


def test_benchmark_ransac_smoke(tmp_path: Path) -> None:
    result = run_benchmark("ransac", output_dir=tmp_path, quick=True)

    assert result.success
    assert result.metrics["cases"] == 3
    assert (tmp_path / "ransac_benchmark.csv").exists()
    assert (tmp_path / "ransac_benchmark.png").exists()
