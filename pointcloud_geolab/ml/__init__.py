"""Optional point cloud learning modules."""

from pointcloud_geolab.utils.optional_deps import require_optional

__all__ = ["require_torch"]


def require_torch():
    """Import PyTorch or raise a helpful optional-dependency error."""

    return require_optional("torch")
