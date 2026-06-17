from __future__ import annotations

import logging

import numpy as np
import pytest

from pointcloud_geolab.utils.logger import get_logger
from pointcloud_geolab.utils.transform import (
    apply_homogeneous_transform,
    apply_transform,
    invert_transform,
    make_transform,
    rotation_matrix_from_euler,
)


def test_get_logger_configures_single_stream_handler() -> None:
    logger_name = "pointcloud_geolab.test_logger"
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()

    first = get_logger(logger_name, level=logging.DEBUG)
    second = get_logger(logger_name, level=logging.WARNING)

    assert first is second
    assert len(first.handlers) == 1
    assert first.level == logging.WARNING


def test_transform_inverse_round_trip() -> None:
    points = np.asarray([[0.0, 0.0, 0.0], [0.3, -0.2, 0.5], [1.0, 0.4, -0.1]])
    rotation = rotation_matrix_from_euler(0.2, -0.1, 0.3)
    translation = np.asarray([0.4, -0.2, 0.1])
    transform = make_transform(rotation, translation)

    moved = apply_homogeneous_transform(points, transform)
    restored = apply_homogeneous_transform(moved, invert_transform(transform))

    assert np.allclose(restored, points)


def test_transform_helpers_validate_shapes() -> None:
    with pytest.raises(ValueError, match="rotation"):
        make_transform(np.eye(2), np.zeros(3))
    with pytest.raises(ValueError, match="points"):
        apply_transform(np.zeros((3, 2)), np.eye(3), np.zeros(3))
    with pytest.raises(ValueError, match="transform"):
        apply_homogeneous_transform(np.zeros((3, 3)), np.eye(3))
    with pytest.raises(ValueError, match="transform"):
        invert_transform(np.eye(3))
