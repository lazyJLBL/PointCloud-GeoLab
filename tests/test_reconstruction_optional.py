from __future__ import annotations

import pytest

from pointcloud_geolab.datasets import make_sphere
from pointcloud_geolab.reconstruction import reconstruct_surface


def test_open3d_reconstruction_optional_smoke(tmp_path) -> None:
    pytest.importorskip("open3d")
    points = make_sphere(90, radius=0.5, random_state=170)
    output = tmp_path / "sphere_mesh.ply"

    result = reconstruct_surface(
        points,
        method="alpha_shape",
        output=output,
        alpha=0.8,
        normal_radius=0.25,
    )

    assert output.exists()
    assert len(result.vertices) > 0
