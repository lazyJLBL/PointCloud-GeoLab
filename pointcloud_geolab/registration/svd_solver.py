"""SVD-based rigid transform estimation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pointcloud_geolab.utils.transform import make_transform


@dataclass(frozen=True, slots=True)
class RigidTransformResult:
    rotation: np.ndarray
    translation: np.ndarray
    transformation: np.ndarray


def estimate_rigid_transform(source: np.ndarray, target: np.ndarray) -> RigidTransformResult:
    """Estimate the least-squares rigid transform from source to target."""

    src = np.asarray(source, dtype=float)
    tgt = np.asarray(target, dtype=float)
    if src.shape != tgt.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError("source and target must both have shape (N, 3)")
    if len(src) < 3:
        raise ValueError("at least 3 point pairs are required")

    src_mean = src.mean(axis=0)
    tgt_mean = tgt.mean(axis=0)
    src_centered = src - src_mean
    tgt_centered = tgt - tgt_mean
    covariance = src_centered.T @ tgt_centered

    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    translation = tgt_mean - rotation @ src_mean
    return RigidTransformResult(rotation, translation, make_transform(rotation, translation))
