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

