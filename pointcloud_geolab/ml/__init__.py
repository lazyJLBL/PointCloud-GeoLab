"""Optional point cloud learning modules."""

__all__ = ["require_torch"]


def require_torch():
    """Import PyTorch or raise a helpful optional-dependency error."""

    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "PointNet demos require PyTorch. Install the optional ML dependencies with "
            "`python -m pip install -e .[ml]`."
        ) from exc
    return torch
