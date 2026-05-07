from __future__ import annotations

import numpy as np

from pointcloud_geolab.features import (
    compute_local_geometric_descriptors,
    detect_iss_keypoints,
    match_descriptors,
)


def test_iss_keypoints_and_descriptors_have_stable_shapes() -> None:
    rng = np.random.default_rng(130)
    points = rng.normal(scale=0.2, size=(120, 3))
    points[:20] += np.asarray([0.6, 0.0, 0.0])

    keypoints = detect_iss_keypoints(
        points,
        salient_radius=0.35,
        non_max_radius=0.2,
        gamma21=0.99,
        gamma32=0.99,
        min_neighbors=5,
    )
    descriptors = compute_local_geometric_descriptors(
        points,
        keypoints.indices[: min(5, len(keypoints.indices))],
        radius=0.4,
    )

    assert keypoints.indices.ndim == 1
    assert descriptors.shape[1] == 8


def test_descriptor_matching_ratio_and_mutual() -> None:
    source = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    target = source + 0.01

    matches = match_descriptors(source, target, ratio=0.95, mutual=True)

    assert matches.shape[1] == 2
    assert len(matches) >= 2
