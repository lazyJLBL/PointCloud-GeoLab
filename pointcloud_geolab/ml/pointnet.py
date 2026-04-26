"""Minimal PointNet classifier for optional synthetic shape demos."""

from __future__ import annotations

from pointcloud_geolab.ml import require_torch


def build_pointnet(num_classes: int = 4):
    """Build a small PointNet-style classifier.

    Input shape is ``(B, N, 3)``. The model applies shared MLP layers to each
    point, max-pools over points, then classifies the global feature.
    """

    torch = require_torch()
    nn = torch.nn

    class PointNetClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.point_mlp = nn.Sequential(
                nn.Linear(3, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
            )
            self.classifier = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(128, num_classes),
            )

        def forward(self, points):
            features = self.point_mlp(points)
            global_feature = torch.max(features, dim=1).values
            return self.classifier(global_feature)

    return PointNetClassifier()
