"""3D geometry utilities."""

from .bounding_box import AABB, OBB, compute_aabb, compute_obb
from .distance import point_to_line_distances, point_to_plane_distances
from .pca import PCAResult, pca_analysis
from .primitive_fitting import (
    CylinderModel,
    PlaneModel,
    RANSACResult,
    SphereModel,
    ransac_fit_primitive,
)

__all__ = [
    "AABB",
    "CylinderModel",
    "OBB",
    "PCAResult",
    "PlaneModel",
    "RANSACResult",
    "SphereModel",
    "compute_aabb",
    "compute_obb",
    "pca_analysis",
    "point_to_line_distances",
    "point_to_plane_distances",
    "ransac_fit_primitive",
]
