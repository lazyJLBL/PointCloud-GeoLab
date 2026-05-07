"""Registration metrics."""

from __future__ import annotations

import numpy as np


def rmse(distances: np.ndarray) -> float:
    values = np.asarray(distances, dtype=float)
    if len(values) == 0:
        return float("nan")
    return float(np.sqrt(np.mean(values**2)))


def mean_error(distances: np.ndarray) -> float:
    values = np.asarray(distances, dtype=float)
    if len(values) == 0:
        return float("nan")
    return float(np.mean(values))


def fitness(distances: np.ndarray, threshold: float | None) -> float:
    values = np.asarray(distances, dtype=float)
    if len(values) == 0:
        return 0.0
    if threshold is None:
        return 1.0
    return float(np.mean(values <= threshold))


def rotation_error_deg(estimated: np.ndarray, ground_truth: np.ndarray) -> float:
    """Return geodesic rotation error in degrees."""

    r_est = np.asarray(estimated, dtype=float)
    r_gt = np.asarray(ground_truth, dtype=float)
    if r_est.shape != (3, 3) or r_gt.shape != (3, 3):
        raise ValueError("rotations must have shape (3, 3)")
    delta = r_est @ r_gt.T
    cos_angle = float(np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def translation_error(estimated: np.ndarray, ground_truth: np.ndarray) -> float:
    """Return Euclidean translation error."""

    t_est = np.asarray(estimated, dtype=float).reshape(-1)
    t_gt = np.asarray(ground_truth, dtype=float).reshape(-1)
    if t_est.shape != (3,) or t_gt.shape != (3,):
        raise ValueError("translations must have shape (3,)")
    return float(np.linalg.norm(t_est - t_gt))


def registration_success(
    rotation_error_degrees: float,
    translation_error_value: float,
    rmse_value: float,
    rotation_threshold_degrees: float = 5.0,
    translation_threshold: float = 0.1,
    rmse_threshold: float = 0.05,
) -> bool:
    """Return whether transform and residual errors are below thresholds."""

    return bool(
        rotation_error_degrees <= rotation_threshold_degrees
        and translation_error_value <= translation_threshold
        and rmse_value <= rmse_threshold
    )
