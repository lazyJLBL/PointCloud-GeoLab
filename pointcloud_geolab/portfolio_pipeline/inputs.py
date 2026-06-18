"""Input discovery helpers for the portfolio pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pointcloud_geolab.io.pointcloud_io import load_point_cloud
from pointcloud_geolab.utils.transform import apply_transform, rotation_matrix_from_euler

POINT_EXTENSIONS = {".ply", ".pcd", ".xyz", ".txt", ".bin", ".off", ".las", ".laz"}


@dataclass(frozen=True, slots=True)
class PipelineInputs:
    """Resolved input files used by the portfolio pipeline."""

    requested_root: Path
    resolved_root: Path
    main_cloud: Path
    segmentation_cloud: Path
    registration_source: Path | None
    registration_target: Path | None


def _resolve_pipeline_inputs(input_path: str | Path) -> PipelineInputs:
    requested = Path(input_path)
    root = requested
    if not root.exists():
        repo_root = Path(__file__).resolve().parents[1]
        normalized = requested.as_posix().strip("/")
        if normalized == "examples/demo_data" and (repo_root / "data").exists():
            root = repo_root / "data"
        else:
            raise FileNotFoundError(f"input path does not exist: {requested}")

    if root.is_file():
        main_cloud = root
        return PipelineInputs(
            requested_root=requested,
            resolved_root=root.parent,
            main_cloud=main_cloud,
            segmentation_cloud=main_cloud,
            registration_source=None,
            registration_target=None,
        )

    if not root.is_dir():
        raise ValueError(f"input path must be a point cloud file or directory: {requested}")

    main_cloud = _first_existing(
        root,
        ["object.ply", "synthetic_scene.ply", "lidar_scene.ply", "room.pcd", "room.xyz"],
    )
    segmentation_cloud = _first_existing(
        root,
        ["lidar_scene.ply", "synthetic_scene.ply", "room.pcd", "object.ply", main_cloud.name],
    )
    registration_source = _optional_existing(root, ["bunny_source.ply", "source.ply"])
    registration_target = _optional_existing(root, ["bunny_target.ply", "target.ply"])
    return PipelineInputs(
        requested_root=requested,
        resolved_root=root,
        main_cloud=main_cloud,
        segmentation_cloud=segmentation_cloud,
        registration_source=registration_source,
        registration_target=registration_target,
    )


def _first_existing(root: Path, names: list[str]) -> Path:
    for name in names:
        candidate = root / name
        if candidate.exists():
            return candidate
    discovered = sorted(path for path in root.iterdir() if path.suffix.lower() in POINT_EXTENSIONS)
    if discovered:
        return discovered[0]
    raise FileNotFoundError(f"no supported point cloud files found in {root}")


def _optional_existing(root: Path, names: list[str]) -> Path | None:
    for name in names:
        candidate = root / name
        if candidate.exists():
            return candidate
    return None


def _load_registration_pair(
    inputs: PipelineInputs,
    fallback_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    if inputs.registration_source and inputs.registration_target:
        return (
            load_point_cloud(inputs.registration_source),
            load_point_cloud(inputs.registration_target),
            "using source/target demo clouds from the input directory",
        )

    target = np.asarray(fallback_points, dtype=float)
    rotation = rotation_matrix_from_euler(0.08, -0.05, 0.10)
    translation = np.asarray([0.16, -0.08, 0.10], dtype=float)
    source = apply_transform(target, rotation, translation)
    return source, target, "using a deterministic transformed copy of the input cloud"
