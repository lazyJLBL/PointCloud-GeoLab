"""Training helper for the optional PointNet synthetic demo."""

from __future__ import annotations

from pathlib import Path

from pointcloud_geolab.ml import require_torch
from pointcloud_geolab.ml.datasets import SyntheticShapeDataset, torch_dataset
from pointcloud_geolab.ml.pointnet import build_pointnet


def train_pointnet(
    output: str | Path,
    epochs: int = 5,
    batch_size: int = 16,
    samples_per_class: int = 24,
    points_per_sample: int = 128,
    seed: int = 7,
) -> dict[str, float]:
    """Train a minimal PointNet classifier on synthetic shapes."""

    torch = require_torch()
    torch.manual_seed(seed)
    dataset = torch_dataset(
        SyntheticShapeDataset(
            samples_per_class=samples_per_class,
            points_per_sample=points_per_sample,
            random_state=seed,
        )
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = build_pointnet(num_classes=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    final_loss = 0.0
    final_accuracy = 0.0
    for _ in range(epochs):
        correct = 0
        total = 0
        for points, labels in loader:
            logits = model(points)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu())
            correct += int((logits.argmax(dim=1) == labels).sum())
            total += int(labels.numel())
        final_accuracy = correct / max(total, 1)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "class_names": SyntheticShapeDataset.class_names,
            "points_per_sample": points_per_sample,
        },
        output_path,
    )
    return {"loss": final_loss, "accuracy": final_accuracy}
