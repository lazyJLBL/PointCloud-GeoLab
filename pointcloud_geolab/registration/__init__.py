"""Point cloud registration algorithms."""

from .feature_registration import (
    RansacTransformResult,
    estimate_rigid_transform_ransac,
    register_iss_descriptor_ransac_icp,
)
from .gicp import CovarianceEstimationResult, estimate_local_covariances, generalized_icp
from .global_registration import (
    GlobalRegistrationResult,
    RegistrationStage,
    evaluate_registration,
    execute_global_registration,
    refine_registration_icp,
    register_fpfh_ransac_icp,
)
from .icp import (
    ICPResult,
    MultiScaleICPResult,
    multiscale_icp,
    point_to_plane_icp,
    point_to_point_icp,
    robust_icp,
)
from .metrics import registration_success, rotation_error_deg, translation_error
from .svd_solver import estimate_rigid_transform

__all__ = [
    "GlobalRegistrationResult",
    "ICPResult",
    "MultiScaleICPResult",
    "CovarianceEstimationResult",
    "RansacTransformResult",
    "RegistrationStage",
    "estimate_rigid_transform",
    "estimate_rigid_transform_ransac",
    "estimate_local_covariances",
    "evaluate_registration",
    "execute_global_registration",
    "generalized_icp",
    "multiscale_icp",
    "point_to_plane_icp",
    "point_to_point_icp",
    "registration_success",
    "refine_registration_icp",
    "register_fpfh_ransac_icp",
    "register_iss_descriptor_ransac_icp",
    "robust_icp",
    "rotation_error_deg",
    "translation_error",
]
