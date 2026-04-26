"""Inference helper for the optional PointNet demo."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pointcloud_geolab.io import load_point_cloud
from pointcloud_geolab.ml import require_torch
from pointcloud_geolab.ml.pointnet import build_pointnet
from pointcloud_geolab.preprocessing import normalize_point_cloud, random_sample


def infer_pointnet(
    model_path: str | Path,
    input_path: str | Path,
    points_per_sample: int | None = None,
) -> dict[str, object]:
    """Run PointNet classification on one point cloud."""

    torch = require_torch()
    checkpoint = torch.load(model_path, map_location="cpu")
    if points_per_sample is None:
        points_per_sample = int(checkpoint.get("points_per_sample", 128))
    class_names = list(checkpoint.get("class_names", ["sphere", "box", "cylinder", "plane"]))
    points = load_point_cloud(input_path)
    if len(points) > points_per_sample:
        points, _ = random_sample(points, points_per_sample, random_state=0)
    elif len(points) < points_per_sample:
        repeat = int(np.ceil(points_per_sample / max(len(points), 1)))
        points = np.tile(points, (repeat, 1))[:points_per_sample]
    points, _, _ = normalize_point_cloud(points)
    model = build_pointnet(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(points.astype("float32")).unsqueeze(0))
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    index = int(np.argmax(probabilities))
    return {
        "class_index": index,
        "class_name": class_names[index],
        "probability": float(probabilities[index]),
    }
