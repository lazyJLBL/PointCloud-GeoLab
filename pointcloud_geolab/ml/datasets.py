"""Synthetic datasets for optional PointNet demos."""

from __future__ import annotations

import numpy as np

from pointcloud_geolab.datasets import make_box, make_cylinder, make_plane, make_sphere
from pointcloud_geolab.preprocessing import normalize_point_cloud


class SyntheticShapeDataset:
    """Small synthetic classification dataset for sphere/box/cylinder/plane."""

    class_names = ["sphere", "box", "cylinder", "plane"]

    def __init__(
        self,
        samples_per_class: int = 32,
        points_per_sample: int = 128,
        random_state: int = 7,
        seed: int | None = None,
        normalize: bool = True,
    ) -> None:
        if seed is not None:
            random_state = seed
        self.samples_per_class = samples_per_class
        self.points_per_sample = points_per_sample
        self.random_state = random_state
        self.normalize = normalize
        self._samples, self._labels = self._build()

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, index: int):
        points = self._samples[index].astype("float32")
        label = int(self._labels[index])
        return points, label

    def _build(self) -> tuple[np.ndarray, np.ndarray]:
        samples = []
        labels = []
        seed = self.random_state
        generators = [make_sphere, make_box, make_cylinder, make_plane]
        for label, generator in enumerate(generators):
            for _ in range(self.samples_per_class):
                if generator is make_box:
                    points = generator(self.points_per_sample, random_state=seed)
                    points += np.random.default_rng(seed).normal(scale=0.01, size=points.shape)
                else:
                    points = generator(self.points_per_sample, random_state=seed, noise=0.01)
                seed += 1
                if self.normalize:
                    points, _, _ = normalize_point_cloud(points)
                samples.append(points)
                labels.append(label)
        return np.asarray(samples, dtype="float32"), np.asarray(labels, dtype=int)


def torch_dataset(dataset: SyntheticShapeDataset):
    """Wrap ``SyntheticShapeDataset`` as a PyTorch ``Dataset``."""

    from pointcloud_geolab.ml import require_torch

    torch = require_torch()

    class TorchSyntheticDataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return len(dataset)

        def __getitem__(self, index: int):
            points, label = dataset[index]
            return torch.from_numpy(points), torch.tensor(label, dtype=torch.long)

    return TorchSyntheticDataset()
