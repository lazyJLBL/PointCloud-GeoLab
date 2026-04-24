"""3D geometry utilities."""

from .bounding_box import AABB, OBB, compute_aabb, compute_obb
from .distance import point_to_line_distances, point_to_plane_distances
from .pca import PCAResult, pca_analysis

__all__ = [
    "AABB",
    "OBB",
    "PCAResult",
    "compute_aabb",
    "compute_obb",
    "pca_analysis",
    "point_to_line_distances",
    "point_to_plane_distances",
]

