from __future__ import annotations

import pytest

from pointcloud_geolab.ml.datasets import SyntheticShapeDataset


def test_synthetic_shape_dataset_shape() -> None:
    dataset = SyntheticShapeDataset(samples_per_class=2, points_per_sample=16, random_state=70)

    points, label = dataset[0]

    assert len(dataset) == 8
    assert points.shape == (16, 3)
    assert 0 <= label < 4


def test_pointnet_forward_smoke_when_torch_installed() -> None:
    torch = pytest.importorskip("torch")
    from pointcloud_geolab.ml.pointnet import build_pointnet

    model = build_pointnet(num_classes=4)
    output = model(torch.zeros((2, 16, 3), dtype=torch.float32))

    assert output.shape == (2, 4)
