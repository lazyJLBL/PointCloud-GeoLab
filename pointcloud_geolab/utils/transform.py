"""Rigid transform helpers."""

from __future__ import annotations

import numpy as np


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Create a 4x4 homogeneous transform from ``R`` and ``t``."""

    r = np.asarray(rotation, dtype=float)
    t = np.asarray(translation, dtype=float).reshape(3)
    if r.shape != (3, 3):
        raise ValueError("rotation must have shape (3, 3)")
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = r
    transform[:3, 3] = t
    return transform


def apply_transform(
    points: np.ndarray, rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    """Apply ``R p + t`` to an ``(N, 3)`` point cloud."""

    pts = np.asarray(points, dtype=float)
    r = np.asarray(rotation, dtype=float)
    t = np.asarray(translation, dtype=float).reshape(3)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    return pts @ r.T + t


def apply_homogeneous_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous transform to points."""

    matrix = np.asarray(transform, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError("transform must have shape (4, 4)")
    return apply_transform(points, matrix[:3, :3], matrix[:3, 3])


def invert_transform(transform: np.ndarray) -> np.ndarray:
    """Invert a rigid 4x4 homogeneous transform."""

    matrix = np.asarray(transform, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError("transform must have shape (4, 4)")
    r = matrix[:3, :3]
    t = matrix[:3, 3]
    inverse = np.eye(4, dtype=float)
    inverse[:3, :3] = r.T
    inverse[:3, 3] = -r.T @ t
    return inverse


def rotation_matrix_from_euler(rx: float, ry: float, rz: float) -> np.ndarray:
    """Create a rotation matrix from XYZ Euler angles in radians."""

    sx, cx = np.sin(rx), np.cos(rx)
    sy, cy = np.sin(ry), np.cos(ry)
    sz, cz = np.sin(rz), np.cos(rz)
    rx_m = np.asarray([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=float)
    ry_m = np.asarray([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=float)
    rz_m = np.asarray([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=float)
    return rz_m @ ry_m @ rx_m
