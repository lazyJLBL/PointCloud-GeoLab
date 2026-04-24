"""PCA-based point cloud analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class PCAResult:
    center: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray


def pca_analysis(points: np.ndarray) -> PCAResult:
    """Compute PCA center, eigenvalues, and principal directions."""

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if len(pts) < 2:
        raise ValueError("at least 2 points are required")
    center = pts.mean(axis=0)
    centered = pts - center
    covariance = centered.T @ centered / len(pts)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, -1] *= -1.0
    return PCAResult(center=center, eigenvalues=eigenvalues, eigenvectors=eigenvectors)


def shape_type_from_eigenvalues(eigenvalues: np.ndarray, eps: float = 1e-9) -> str:
    """Classify a point cloud as line-like, plane-like, or volume-like."""

    vals = np.asarray(eigenvalues, dtype=float)
    if vals.shape[0] != 3:
        raise ValueError("eigenvalues must contain 3 values")
    l1, l2, l3 = np.maximum(vals, eps)
    if l1 / l2 > 10 and l2 / l3 < 10:
        return "line-like"
    if l2 / l3 > 10:
        return "plane-like"
    return "volume-like"

