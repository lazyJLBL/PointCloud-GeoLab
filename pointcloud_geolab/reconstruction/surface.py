"""Open3D-backed point cloud surface reconstruction demos."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class MeshReconstructionResult:
    """Summary of a reconstructed triangle mesh."""

    method: str
    vertices: np.ndarray
    triangles: np.ndarray
    output_path: str | None = None


def reconstruct_surface(
    points: np.ndarray,
    method: str = "poisson",
    output: str | Path | None = None,
    normal_radius: float = 0.15,
    poisson_depth: int = 6,
    alpha: float = 0.2,
) -> MeshReconstructionResult:
    """Reconstruct a mesh with Open3D Poisson, BPA, or Alpha Shape."""

    pts = _ensure_points(points)
    o3d = _require_open3d()
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pts)
    point_cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30)
    )
    point_cloud.orient_normals_consistent_tangent_plane(min(20, max(3, len(pts) - 1)))

    normalized = method.lower()
    if normalized == "poisson":
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            point_cloud,
            depth=poisson_depth,
        )
        densities = np.asarray(densities)
        if len(densities):
            keep = densities > np.quantile(densities, 0.05)
            mesh.remove_vertices_by_mask(~keep)
    elif normalized in {"ball_pivoting", "bpa"}:
        distances = point_cloud.compute_nearest_neighbor_distance()
        avg_distance = float(np.mean(distances)) if distances else normal_radius
        radii = o3d.utility.DoubleVector([avg_distance * 1.5, avg_distance * 2.5])
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(point_cloud, radii)
        normalized = "ball_pivoting"
    elif normalized == "alpha_shape":
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(point_cloud, alpha)
    else:
        raise ValueError("method must be one of: poisson, ball_pivoting, alpha_shape")

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    output_path = None
    if output is not None:
        output_path = str(output)
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_triangle_mesh(str(output), mesh)
    return MeshReconstructionResult(
        method=normalized,
        vertices=np.asarray(mesh.vertices),
        triangles=np.asarray(mesh.triangles),
        output_path=output_path,
    )


def _ensure_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if len(pts) < 4:
        raise ValueError("surface reconstruction requires at least 4 points")
    return pts


def _require_open3d():
    try:
        import open3d as o3d  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Open3D is required for surface reconstruction. Install with "
            "`python -m pip install open3d`."
        ) from exc
    return o3d
