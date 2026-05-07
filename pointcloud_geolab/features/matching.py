"""Descriptor matching utilities."""

from __future__ import annotations

import numpy as np


def descriptor_distances(
    source_descriptors: np.ndarray, target_descriptors: np.ndarray
) -> np.ndarray:
    """Return pairwise Euclidean descriptor distances."""

    src = _ensure_descriptors(source_descriptors, "source_descriptors")
    tgt = _ensure_descriptors(target_descriptors, "target_descriptors")
    if src.shape[1] != tgt.shape[1]:
        raise ValueError("descriptor dimensions must match")
    if len(src) == 0 or len(tgt) == 0:
        return np.empty((len(src), len(tgt)), dtype=float)
    diff = src[:, None, :] - tgt[None, :, :]
    return np.linalg.norm(diff, axis=2)


def match_descriptors(
    source_descriptors: np.ndarray,
    target_descriptors: np.ndarray,
    ratio: float = 0.85,
    mutual: bool = True,
) -> np.ndarray:
    """Match descriptors with Lowe ratio test and optional mutual NN check."""

    if ratio <= 0:
        raise ValueError("ratio must be positive")
    distances = descriptor_distances(source_descriptors, target_descriptors)
    if distances.size == 0:
        return np.empty((0, 2), dtype=int)

    target_best_for_source = np.argsort(distances, axis=1, kind="mergesort")
    source_best_for_target = np.argmin(distances, axis=0)
    matches = []
    for source_index, order in enumerate(target_best_for_source):
        best = int(order[0])
        if len(order) > 1:
            second = float(distances[source_index, int(order[1])])
            if second > 1e-12 and float(distances[source_index, best]) / second > ratio:
                continue
        if mutual and int(source_best_for_target[best]) != source_index:
            continue
        matches.append((source_index, best))
    return np.asarray(matches, dtype=int)


def _ensure_descriptors(descriptors: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(descriptors, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"{name} must have shape (N, D)")
    return values
