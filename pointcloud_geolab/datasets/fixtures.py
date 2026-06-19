"""Tiny synthetic dataset fixtures and validators for format smoke tests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class OffMesh:
    """Minimal OFF mesh container used by dataset fixture tests."""

    vertices: np.ndarray
    faces: np.ndarray


@dataclass(frozen=True, slots=True)
class FixtureValidation:
    """Result of validating a tiny fixture manifest."""

    manifest_path: Path
    checked_files: list[Path]
    issues: list[str]

    @property
    def success(self) -> bool:
        return not self.issues


def load_kitti_like_bin(
    path: str | Path,
    expected_points: int | None = None,
    include_intensity: bool = True,
) -> np.ndarray:
    """Load a tiny KITTI-like ``float32 x y z intensity`` binary fixture."""

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    byte_size = file_path.stat().st_size
    if byte_size == 0:
        raise ValueError(f"{file_path}: KITTI-like .bin is empty")
    if byte_size % 16 != 0:
        raise ValueError(
            f"{file_path}: KITTI-like .bin byte size {byte_size} is not divisible by 16"
        )
    data = np.fromfile(file_path, dtype=np.float32)
    if data.size % 4 != 0:
        raise ValueError(f"{file_path}: KITTI-like .bin value count is not divisible by 4")
    cloud = data.reshape(-1, 4).astype(float)
    if not np.all(np.isfinite(cloud[:, :3])):
        raise ValueError(f"{file_path}: KITTI-like .bin contains NaN or infinite coordinates")
    if expected_points is not None and len(cloud) != expected_points:
        raise ValueError(
            f"{file_path}: expected {expected_points} KITTI-like points, found {len(cloud)}"
        )
    return cloud if include_intensity else cloud[:, :3].copy()


def load_modelnet_like_off(path: str | Path) -> OffMesh:
    """Load a tiny ModelNet-like OFF mesh fixture."""

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    if file_path.stat().st_size == 0:
        raise ValueError(f"{file_path}: OFF file is empty")

    with file_path.open("r", encoding="utf-8") as handle:
        header = _next_data_line(handle, file_path)
        if header != "OFF":
            raise ValueError(f"{file_path}: bad OFF header `{header}`")
        counts = _parse_counts(_next_data_line(handle, file_path), file_path)
        vertex_count, face_count = counts
        vertices = []
        for index in range(vertex_count):
            parts = _next_data_line(handle, file_path).split()
            if len(parts) < 3:
                raise ValueError(f"{file_path}: vertex {index} has fewer than 3 values")
            try:
                vertices.append([float(value) for value in parts[:3]])
            except ValueError as exc:
                raise ValueError(f"{file_path}: vertex {index} contains a non-number") from exc
            if not np.all(np.isfinite(vertices[-1])):
                raise ValueError(f"{file_path}: vertex {index} contains NaN or infinite values")

        faces = []
        for index in range(face_count):
            parts = _next_data_line(handle, file_path).split()
            try:
                values = [int(value) for value in parts]
            except ValueError as exc:
                raise ValueError(f"{file_path}: face {index} contains a non-integer") from exc
            if not values or values[0] < 3:
                raise ValueError(f"{file_path}: face {index} must contain at least 3 vertices")
            face = values[1 : values[0] + 1]
            if len(face) != values[0]:
                raise ValueError(f"{file_path}: face {index} count does not match indices")
            if any(vertex < 0 or vertex >= vertex_count for vertex in face):
                raise ValueError(f"{file_path}: face {index} references an invalid vertex")
            faces.append(face)

    face_width = max((len(face) for face in faces), default=0)
    face_array = np.asarray(
        [face + [-1] * (face_width - len(face)) for face in faces],
        dtype=int,
    )
    return OffMesh(vertices=np.asarray(vertices, dtype=float), faces=face_array)


def validate_fixture_manifest(
    manifest_path: str | Path = "tests/fixtures/datasets/manifest.json",
) -> FixtureValidation:
    """Validate fixture checksums, declared counts, and synthetic-boundary metadata."""

    manifest = Path(manifest_path)
    checked: list[Path] = []
    issues: list[str] = []
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return FixtureValidation(manifest, checked, [f"{manifest}: missing manifest"])
    except json.JSONDecodeError as exc:
        return FixtureValidation(manifest, checked, [f"{manifest}: invalid JSON ({exc})"])

    fixtures = payload.get("fixtures") if isinstance(payload, dict) else None
    if not isinstance(fixtures, list) or not fixtures:
        return FixtureValidation(manifest, checked, [f"{manifest}: missing fixtures list"])

    for entry in fixtures:
        if not isinstance(entry, dict):
            issues.append(f"{manifest}: fixture entry must be an object")
            continue
        _validate_manifest_entry(manifest, entry, checked, issues)
    return FixtureValidation(manifest, checked, sorted(issues))


def sha256_file(path: str | Path) -> str:
    """Return a file SHA256 digest."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_manifest_entry(
    manifest: Path,
    entry: dict[str, Any],
    checked: list[Path],
    issues: list[str],
) -> None:
    root = manifest.parent
    filename = entry.get("filename")
    fixture_format = entry.get("format")
    if not isinstance(filename, str) or not filename:
        issues.append(f"{manifest}: fixture missing filename")
        return
    path = root / filename
    checked.append(path)
    if not path.exists():
        issues.append(f"{path}: missing fixture file")
        return

    if entry.get("synthetic") is not True:
        issues.append(f"{path}: manifest must mark fixture as synthetic")
    note = str(entry.get("note", "")).lower()
    if "not real" not in note:
        issues.append(f"{path}: manifest note must state this is not real benchmark data")

    expected_sha = entry.get("sha256")
    actual_sha = sha256_file(path)
    if expected_sha != actual_sha:
        issues.append(f"{path}: checksum mismatch expected {expected_sha}, got {actual_sha}")

    try:
        if fixture_format == "kitti-like-bin":
            expected_points = _int_field(entry, "points")
            cloud = load_kitti_like_bin(path, expected_points=expected_points)
            if cloud.shape != (expected_points, 4):
                issues.append(f"{path}: expected shape ({expected_points}, 4), got {cloud.shape}")
        elif fixture_format == "modelnet-like-off":
            expected_vertices = _int_field(entry, "vertices")
            expected_faces = _int_field(entry, "faces")
            mesh = load_modelnet_like_off(path)
            if mesh.vertices.shape != (expected_vertices, 3):
                issues.append(
                    f"{path}: expected vertices ({expected_vertices}, 3), "
                    f"got {mesh.vertices.shape}"
                )
            if mesh.faces.shape[0] != expected_faces:
                issues.append(f"{path}: expected {expected_faces} faces, got {mesh.faces.shape[0]}")
        else:
            issues.append(f"{path}: unsupported fixture format `{fixture_format}`")
    except (FileNotFoundError, ValueError) as exc:
        issues.append(str(exc))


def _int_field(entry: dict[str, Any], key: str) -> int:
    value = entry.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{entry.get('filename', '<fixture>')}: manifest field `{key}` missing")
    return value


def _parse_counts(line: str, path: Path) -> tuple[int, int]:
    parts = line.split()
    if len(parts) < 2:
        raise ValueError(f"{path}: OFF counts line must include vertices and faces")
    try:
        vertex_count = int(parts[0])
        face_count = int(parts[1])
    except ValueError as exc:
        raise ValueError(f"{path}: OFF counts line contains a non-integer") from exc
    if vertex_count <= 0:
        raise ValueError(f"{path}: OFF vertex count must be positive")
    if face_count < 0:
        raise ValueError(f"{path}: OFF face count cannot be negative")
    return vertex_count, face_count


def _next_data_line(handle, path: Path) -> str:
    for line in handle:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped
    raise ValueError(f"{path}: unexpected end of file")
