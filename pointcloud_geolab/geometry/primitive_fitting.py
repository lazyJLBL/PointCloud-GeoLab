"""Robust geometric primitive fitting with RANSAC.

The module implements explicit model geometry instead of delegating fitting to
external libraries:

- plane distance: ``|n^T x + d|``
- sphere distance: ``abs(||x - c|| - r)``
- cylinder distance: ``abs(distance_to_axis(x) - r)``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class PrimitiveModel(Protocol):
    """Common interface for fitted primitives."""

    def predict(self, points: np.ndarray) -> np.ndarray:
        """Return model-specific predictions for points."""

    def residuals(self, points: np.ndarray) -> np.ndarray:
        """Return geometric residuals for points."""

    def inlier_mask(self, points: np.ndarray, threshold: float) -> np.ndarray:
        """Return ``True`` for points whose residual is within threshold."""

    def get_params(self) -> dict[str, object]:
        """Return JSON-friendly model parameters."""


@dataclass(frozen=True, slots=True)
class PlaneModel:
    """Plane represented by unit normal ``n`` and offset ``d``."""

    normal: np.ndarray
    d: float

    @classmethod
    def fit(cls, points: np.ndarray) -> "PlaneModel":
        """Fit a plane from at least three points using SVD."""

        pts = _ensure_points(points, min_points=3)
        if len(pts) == 3:
            normal = np.cross(pts[1] - pts[0], pts[2] - pts[0])
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                raise ValueError("degenerate plane sample")
            normal = normal / norm
            center = pts[0]
        else:
            center = pts.mean(axis=0)
            _, _, vh = np.linalg.svd(pts - center, full_matrices=False)
            normal = vh[-1]
        if normal[np.argmax(np.abs(normal))] < 0:
            normal = -normal
        d = -float(normal @ center)
        return cls(normal=normal.astype(float), d=d)

    def predict(self, points: np.ndarray) -> np.ndarray:
        pts = _ensure_points(points)
        return pts @ self.normal + self.d

    def residuals(self, points: np.ndarray) -> np.ndarray:
        return np.abs(self.predict(points))

    def inlier_mask(self, points: np.ndarray, threshold: float) -> np.ndarray:
        return self.residuals(points) <= threshold

    def get_params(self) -> dict[str, object]:
        return {"normal": self.normal.tolist(), "d": self.d}


@dataclass(frozen=True, slots=True)
class SphereModel:
    """Sphere represented by center ``c`` and radius ``r``."""

    center: np.ndarray
    radius: float

    @classmethod
    def fit(cls, points: np.ndarray) -> "SphereModel":
        """Fit a sphere from at least four points by linear least squares."""

        pts = _ensure_points(points, min_points=4)
        a = np.column_stack((2.0 * pts, np.ones(len(pts))))
        b = np.sum(pts * pts, axis=1)
        solution, *_ = np.linalg.lstsq(a, b, rcond=None)
        center = solution[:3]
        radius_sq = float(solution[3] + center @ center)
        if radius_sq <= 0:
            raise ValueError("degenerate sphere sample")
        return cls(center=center.astype(float), radius=float(np.sqrt(radius_sq)))

    def predict(self, points: np.ndarray) -> np.ndarray:
        pts = _ensure_points(points)
        return np.linalg.norm(pts - self.center, axis=1)

    def residuals(self, points: np.ndarray) -> np.ndarray:
        return np.abs(self.predict(points) - self.radius)

    def inlier_mask(self, points: np.ndarray, threshold: float) -> np.ndarray:
        return self.residuals(points) <= threshold

    def get_params(self) -> dict[str, object]:
        return {"center": self.center.tolist(), "radius": self.radius}


@dataclass(frozen=True, slots=True)
class CylinderModel:
    """Infinite cylinder represented by axis point, axis direction, and radius."""

    axis_point: np.ndarray
    axis_direction: np.ndarray
    radius: float

    @classmethod
    def fit(cls, points: np.ndarray) -> "CylinderModel":
        """Fit an infinite cylinder using PCA axis and median radial distance.

        This is intentionally lightweight and geometry-first: RANSAC supplies
        robustness, while each candidate estimates the cylinder axis as the
        dominant PCA direction of the sample.
        """

        pts = _ensure_points(points, min_points=6)
        center = pts.mean(axis=0)
        _, _, vh = np.linalg.svd(pts - center, full_matrices=False)
        direction = vh[0]
        direction = direction / np.linalg.norm(direction)
        if direction[np.argmax(np.abs(direction))] < 0:
            direction = -direction
        u, v = _orthonormal_basis(direction)
        local = np.column_stack([(pts - center) @ u, (pts - center) @ v])
        circle_a = np.column_stack((2.0 * local, np.ones(len(local))))
        circle_b = np.sum(local * local, axis=1)
        solution, *_ = np.linalg.lstsq(circle_a, circle_b, rcond=None)
        circle_center = solution[:2]
        radius_sq = float(solution[2] + circle_center @ circle_center)
        if radius_sq <= 0:
            raise ValueError("degenerate cylinder sample")
        axis_point = center + circle_center[0] * u + circle_center[1] * v
        radius = float(np.sqrt(radius_sq))
        if radius <= 1e-12:
            raise ValueError("degenerate cylinder sample")
        return cls(
            axis_point=axis_point.astype(float),
            axis_direction=direction.astype(float),
            radius=radius,
        )

    def predict(self, points: np.ndarray) -> np.ndarray:
        pts = _ensure_points(points)
        return _point_to_axis_distances(pts, self.axis_point, self.axis_direction)

    def residuals(self, points: np.ndarray) -> np.ndarray:
        return np.abs(self.predict(points) - self.radius)

    def inlier_mask(self, points: np.ndarray, threshold: float) -> np.ndarray:
        return self.residuals(points) <= threshold

    def get_params(self) -> dict[str, object]:
        return {
            "axis_point": self.axis_point.tolist(),
            "axis_direction": self.axis_direction.tolist(),
            "radius": self.radius,
        }


@dataclass(slots=True)
class RANSACResult:
    """Best primitive model and inlier statistics."""

    model: PrimitiveModel
    inlier_indices: np.ndarray
    outlier_indices: np.ndarray
    score: float
    residual_mean: float
    iterations: int


@dataclass(slots=True)
class ExtractedPrimitive:
    """One primitive extracted from a mixed scene."""

    model_type: str
    model: PrimitiveModel
    inlier_indices: np.ndarray
    residual_mean: float
    inlier_ratio: float
    selection_score: float

    def get_params(self) -> dict[str, object]:
        """Return JSON-friendly primitive parameters."""

        return {
            "model_type": self.model_type,
            "params": self.model.get_params(),
            "inliers": self.inlier_indices.tolist(),
            "residual_mean": self.residual_mean,
            "inlier_ratio": self.inlier_ratio,
            "selection_score": self.selection_score,
        }


@dataclass(slots=True)
class PrimitiveExtractionResult:
    """Sequential RANSAC extraction result."""

    primitives: list[ExtractedPrimitive]
    remaining_indices: np.ndarray


def ransac_fit_primitive(
    points: np.ndarray,
    model_type: str,
    threshold: float = 0.02,
    max_iterations: int = 1000,
    min_inliers: int = 0,
    random_state: int | None = None,
) -> RANSACResult:
    """Fit a plane, sphere, or cylinder with a generic RANSAC loop."""

    pts = _ensure_points(points)
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    model_cls, sample_size = _model_spec(model_type)
    if len(pts) < sample_size:
        raise ValueError(f"{model_type} fitting requires at least {sample_size} points")

    rng = np.random.default_rng(random_state)
    best_model: PrimitiveModel | None = None
    best_inliers = np.asarray([], dtype=int)
    best_residual_mean = float("inf")
    iterations = 0

    for iteration in range(1, max_iterations + 1):
        sample_ids = rng.choice(len(pts), size=sample_size, replace=False)
        try:
            candidate = model_cls.fit(pts[sample_ids])
        except (ValueError, np.linalg.LinAlgError):
            continue
        residuals = candidate.residuals(pts)
        inliers = np.flatnonzero(residuals <= threshold)
        residual_mean = float(residuals[inliers].mean()) if len(inliers) else float("inf")
        is_better = len(inliers) > len(best_inliers) or (
            len(inliers) == len(best_inliers) and residual_mean < best_residual_mean
        )
        if is_better:
            best_model = candidate
            best_inliers = inliers
            best_residual_mean = residual_mean
            iterations = iteration

    if best_model is None or len(best_inliers) < min_inliers:
        raise RuntimeError("failed to fit a primitive with enough inliers")

    try:
        best_model = model_cls.fit(pts[best_inliers])
    except (ValueError, np.linalg.LinAlgError):
        pass
    final_residuals = best_model.residuals(pts)
    best_inliers = np.flatnonzero(final_residuals <= threshold)
    outlier_mask = np.ones(len(pts), dtype=bool)
    outlier_mask[best_inliers] = False
    residual_mean = (
        float(final_residuals[best_inliers].mean()) if len(best_inliers) else float("inf")
    )
    return RANSACResult(
        model=best_model,
        inlier_indices=best_inliers,
        outlier_indices=np.flatnonzero(outlier_mask),
        score=float(len(best_inliers) / len(pts)),
        residual_mean=residual_mean,
        iterations=iterations,
    )


def extract_primitives(
    points: np.ndarray,
    model_types: list[str] | tuple[str, ...] = ("plane", "sphere", "cylinder"),
    threshold: float = 0.03,
    max_models: int = 5,
    min_inliers: int = 30,
    max_iterations: int = 1000,
    random_state: int | None = None,
) -> PrimitiveExtractionResult:
    """Sequentially extract multiple primitives from a mixed point cloud."""

    pts = _ensure_points(points)
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    if max_models <= 0:
        raise ValueError("max_models must be positive")
    if min_inliers <= 0:
        raise ValueError("min_inliers must be positive")
    normalized_types = [model.lower() for model in model_types]
    for model_type in normalized_types:
        _model_spec(model_type)

    remaining = np.arange(len(pts), dtype=int)
    primitives: list[ExtractedPrimitive] = []
    seed = random_state
    for model_index in range(max_models):
        if len(remaining) < min_inliers:
            break
        best: ExtractedPrimitive | None = None
        best_local_inliers: np.ndarray | None = None
        for offset, model_type in enumerate(normalized_types):
            try:
                result = ransac_fit_primitive(
                    pts[remaining],
                    model_type=model_type,
                    threshold=threshold,
                    max_iterations=max_iterations,
                    min_inliers=min_inliers,
                    random_state=None if seed is None else seed + model_index * 17 + offset,
                )
            except (RuntimeError, ValueError, np.linalg.LinAlgError):
                continue
            inlier_ratio = float(len(result.inlier_indices) / len(remaining))
            score = _model_selection_score(
                model_type,
                inlier_ratio=inlier_ratio,
                residual_mean=result.residual_mean,
                point_count=len(remaining),
            )
            candidate = ExtractedPrimitive(
                model_type=model_type,
                model=result.model,
                inlier_indices=remaining[result.inlier_indices],
                residual_mean=result.residual_mean,
                inlier_ratio=inlier_ratio,
                selection_score=score,
            )
            if best is None or candidate.selection_score < best.selection_score:
                best = candidate
                best_local_inliers = result.inlier_indices
        if best is None or best_local_inliers is None:
            break
        primitives.append(best)
        keep_mask = np.ones(len(remaining), dtype=bool)
        keep_mask[best_local_inliers] = False
        remaining = remaining[keep_mask]

    return PrimitiveExtractionResult(primitives=primitives, remaining_indices=remaining)


def _model_selection_score(
    model_type: str,
    inlier_ratio: float,
    residual_mean: float,
    point_count: int,
) -> float:
    parameter_count = {"plane": 4, "sphere": 4, "cylinder": 7}[model_type]
    residual_term = np.log(max(residual_mean, 1e-12))
    complexity = parameter_count * np.log(max(point_count, 2)) / max(point_count, 1)
    return float(residual_term + complexity - 2.0 * inlier_ratio)


def _model_spec(model_type: str):
    normalized = model_type.lower()
    if normalized == "plane":
        return PlaneModel, 3
    if normalized == "sphere":
        return SphereModel, 4
    if normalized == "cylinder":
        return CylinderModel, 12
    raise ValueError("model_type must be one of: plane, sphere, cylinder")


def _point_to_axis_distances(
    points: np.ndarray,
    axis_point: np.ndarray,
    axis_direction: np.ndarray,
) -> np.ndarray:
    vec = points - axis_point
    projected = np.outer(vec @ axis_direction, axis_direction)
    return np.linalg.norm(vec - projected, axis=1)


def _orthonormal_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.asarray([1.0, 0.0, 0.0]) if abs(direction[0]) < 0.9 else np.asarray([0.0, 1.0, 0.0])
    u = np.cross(direction, helper)
    u /= np.linalg.norm(u)
    v = np.cross(direction, u)
    return u, v


def _ensure_points(points: np.ndarray, min_points: int = 1) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if len(pts) < min_points:
        raise ValueError(f"at least {min_points} points are required")
    return pts
