"""Ground removal and object clustering pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pointcloud_geolab.geometry import compute_aabb, compute_obb, ransac_fit_primitive
from pointcloud_geolab.segmentation.clustering import dbscan_clustering, euclidean_clustering


@dataclass(slots=True)
class GroundRemovalResult:
    """Ground/non-ground split from constrained RANSAC plane fitting."""

    ground_indices: np.ndarray
    non_ground_indices: np.ndarray
    plane_model: dict[str, object]
    normal_angle_degrees: float


@dataclass(slots=True)
class ObjectCluster:
    """JSON-friendly object cluster summary."""

    label: int
    indices: np.ndarray
    point_count: int
    centroid: np.ndarray
    aabb_min: np.ndarray
    aabb_max: np.ndarray
    aabb_extent: np.ndarray
    obb_center: np.ndarray
    obb_extent: np.ndarray
    obb_rotation: np.ndarray
    volume: float

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "point_count": self.point_count,
            "centroid": self.centroid.tolist(),
            "aabb": {
                "min_bound": self.aabb_min.tolist(),
                "max_bound": self.aabb_max.tolist(),
                "extent": self.aabb_extent.tolist(),
            },
            "obb": {
                "center": self.obb_center.tolist(),
                "extent": self.obb_extent.tolist(),
                "rotation": self.obb_rotation.tolist(),
            },
            "volume": self.volume,
        }


@dataclass(slots=True)
class GroundObjectSegmentationResult:
    """Ground removal plus object clustering output."""

    labels: np.ndarray
    ground: GroundRemovalResult
    clusters: list[ObjectCluster]
    noise_indices: np.ndarray


def remove_ground_plane(
    points: np.ndarray,
    threshold: float = 0.03,
    max_iterations: int = 500,
    ground_axis: str = "z",
    angle_threshold_degrees: float = 20.0,
    seed: int | None = 7,
) -> GroundRemovalResult:
    """Split ground and non-ground points using constrained plane RANSAC."""

    pts = _ensure_points(points)
    axis = _axis_vector(ground_axis)
    result = ransac_fit_primitive(
        pts,
        "plane",
        threshold=threshold,
        max_iterations=max_iterations,
        min_inliers=max(3, min(30, len(pts) // 5)),
        random_state=seed,
    )
    params = result.model.get_params()
    normal = np.asarray(params["normal"], dtype=float)
    angle = float(np.degrees(np.arccos(np.clip(abs(float(normal @ axis)), -1.0, 1.0))))
    if angle > angle_threshold_degrees:
        raise RuntimeError(
            "dominant plane does not satisfy ground normal constraint: "
            f"{angle:.2f} > {angle_threshold_degrees:.2f} degrees"
        )
    ground_mask = np.zeros(len(pts), dtype=bool)
    ground_mask[result.inlier_indices] = True
    return GroundRemovalResult(
        ground_indices=result.inlier_indices,
        non_ground_indices=np.flatnonzero(~ground_mask),
        plane_model=params,
        normal_angle_degrees=angle,
    )


def ground_object_segmentation(
    points: np.ndarray,
    ground_threshold: float = 0.03,
    ground_axis: str = "z",
    ground_angle_threshold: float = 20.0,
    cluster_method: str = "euclidean",
    eps: float = 0.15,
    min_points: int = 10,
    seed: int | None = 7,
) -> GroundObjectSegmentationResult:
    """Run preprocess-free ground removal followed by object clustering."""

    pts = _ensure_points(points)
    ground = remove_ground_plane(
        pts,
        threshold=ground_threshold,
        ground_axis=ground_axis,
        angle_threshold_degrees=ground_angle_threshold,
        seed=seed,
    )
    objects = pts[ground.non_ground_indices]
    if cluster_method == "euclidean":
        clustering = euclidean_clustering(objects, tolerance=eps, min_points=min_points)
    elif cluster_method == "dbscan":
        clustering = dbscan_clustering(objects, eps=eps, min_points=min_points)
    else:
        raise ValueError("cluster_method must be 'euclidean' or 'dbscan'")

    labels = np.full(len(pts), -1, dtype=int)
    labels[ground.ground_indices] = -2
    clusters = []
    for cluster in clustering.clusters:
        original_indices = ground.non_ground_indices[cluster.indices]
        labels[original_indices] = cluster.label
        cluster_points = pts[original_indices]
        clusters.append(_cluster_summary(cluster.label, original_indices, cluster_points))
    noise_indices = ground.non_ground_indices[clustering.noise_indices]
    return GroundObjectSegmentationResult(
        labels=labels,
        ground=ground,
        clusters=clusters,
        noise_indices=noise_indices,
    )


def write_cluster_report(
    result: GroundObjectSegmentationResult,
    output_path: str | Path,
) -> None:
    """Write a Markdown report for ground/object segmentation."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Object Clustering Report",
        "",
        f"- Ground points: {len(result.ground.ground_indices)}",
        f"- Non-ground noise points: {len(result.noise_indices)}",
        f"- Object clusters: {len(result.clusters)}",
        f"- Ground normal angle: {result.ground.normal_angle_degrees:.2f} deg",
        "",
        "| Label | Points | Centroid | Extent | Volume |",
        "|---:|---:|---|---|---:|",
    ]
    for cluster in result.clusters:
        lines.append(
            "| {label} | {count} | {centroid} | {extent} | {volume:.6f} |".format(
                label=cluster.label,
                count=cluster.point_count,
                centroid=_fmt(cluster.centroid),
                extent=_fmt(cluster.aabb_extent),
                volume=cluster.volume,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _cluster_summary(label: int, indices: np.ndarray, points: np.ndarray) -> ObjectCluster:
    aabb = compute_aabb(points)
    obb = compute_obb(points)
    return ObjectCluster(
        label=int(label),
        indices=indices,
        point_count=len(points),
        centroid=points.mean(axis=0),
        aabb_min=aabb.min_bound,
        aabb_max=aabb.max_bound,
        aabb_extent=aabb.extent,
        obb_center=obb.center,
        obb_extent=obb.extent,
        obb_rotation=obb.rotation,
        volume=float(np.prod(np.maximum(aabb.extent, 0.0))),
    )


def _axis_vector(axis: str) -> np.ndarray:
    axes = {
        "x": np.asarray([1.0, 0.0, 0.0]),
        "y": np.asarray([0.0, 1.0, 0.0]),
        "z": np.asarray([0.0, 0.0, 1.0]),
    }
    try:
        return axes[axis.lower()]
    except KeyError as exc:
        raise ValueError("ground_axis must be one of: x, y, z") from exc


def _fmt(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(value):.3f}" for value in values) + "]"


def _ensure_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if len(pts) < 3:
        raise ValueError("at least 3 points are required")
    return pts
