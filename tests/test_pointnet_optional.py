from __future__ import annotations

import pytest


def test_pointnet_forward_smoke() -> None:
    torch = pytest.importorskip("torch")
    from pointcloud_geolab.ml.datasets import SyntheticShapeDataset, torch_dataset
    from pointcloud_geolab.ml.pointnet import build_pointnet

    dataset = SyntheticShapeDataset(samples_per_class=2, points_per_sample=16, seed=60)
    points, label = torch_dataset(dataset)[0]
    model = build_pointnet(num_classes=4)
    logits = model(points.unsqueeze(0))

    assert points.shape == (16, 3)
    assert int(label) in {0, 1, 2, 3}
    assert logits.shape == (1, 4)
    assert torch.isfinite(logits).all()
