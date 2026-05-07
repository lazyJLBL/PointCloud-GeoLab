"""Point cloud loading and saving utilities.

The public API returns NumPy arrays so the algorithm modules stay independent
from Open3D. Open3D is used when available, with small ASCII readers/writers as
fallbacks for demo data and tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


def _optional_open3d():
    try:
        import open3d as o3d  # type: ignore
    except ImportError:
        return None
    return o3d


def _ensure_points(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError("point cloud must be an array with shape (N, 3) or wider")
    return arr[:, :3].copy()


def load_point_cloud(path: str | Path) -> np.ndarray:
    """Load a point cloud as an array.

    Supported formats are ``.ply``, ``.pcd``, ``.xyz``, ``.txt``, KITTI
    Velodyne ``.bin`` and optional LAS/LAZ via ``laspy``.
    """

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    suffix = file_path.suffix.lower()
    if suffix in {".xyz", ".txt"}:
        return _load_xyz(file_path)
    if suffix == ".bin":
        return load_kitti_bin(file_path, include_intensity=False)
    if suffix in {".las", ".laz"}:
        return load_las(file_path)["points"]

    o3d = _optional_open3d()
    if o3d is not None and suffix in {".ply", ".pcd"}:
        pcd = o3d.io.read_point_cloud(str(file_path))
        points = np.asarray(pcd.points, dtype=float)
        if len(points) > 0:
            return _ensure_points(points)

    if suffix == ".ply":
        return _load_ascii_ply(file_path)
    if suffix == ".pcd":
        return _load_ascii_pcd(file_path)
    raise ValueError(f"unsupported point cloud format: {suffix}")


def save_point_cloud(
    path: str | Path,
    points: np.ndarray,
    colors: np.ndarray | None = None,
    prefer_open3d: bool = False,
) -> None:
    """Save points to .ply, .pcd, .xyz, .txt, or optional .las/.laz.

    ASCII output is the default because it is deterministic and works in test
    environments without Open3D.
    """

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    pts = _ensure_points(points)
    cols = None if colors is None else np.asarray(colors)
    if cols is not None:
        if cols.shape[0] != pts.shape[0] or cols.shape[1] < 3:
            raise ValueError("colors must have shape (N, 3)")
        cols = cols[:, :3]

    suffix = file_path.suffix.lower()
    if prefer_open3d:
        o3d = _optional_open3d()
        if o3d is not None and suffix in {".ply", ".pcd"}:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            if cols is not None:
                if cols.max(initial=0) > 1:
                    cols = cols / 255.0
                pcd.colors = o3d.utility.Vector3dVector(cols)
            o3d.io.write_point_cloud(str(file_path), pcd)
            return

    if suffix == ".ply":
        _save_ascii_ply(file_path, pts, cols)
    elif suffix == ".pcd":
        _save_ascii_pcd(file_path, pts)
    elif suffix in {".xyz", ".txt"}:
        np.savetxt(file_path, pts, fmt="%.8f")
    elif suffix in {".las", ".laz"}:
        save_las(file_path, pts)
    else:
        raise ValueError(f"unsupported point cloud format: {suffix}")


def load_kitti_bin(path: str | Path, include_intensity: bool = False) -> np.ndarray:
    """Load a KITTI Velodyne ``.bin`` frame.

    KITTI stores repeated float32 tuples ``x, y, z, intensity``. By default the
    function returns ``(N, 3)``; set ``include_intensity=True`` for ``(N, 4)``.
    """

    file_path = Path(path)
    data = np.fromfile(file_path, dtype=np.float32)
    if data.size % 4 != 0:
        raise ValueError("KITTI .bin file length must be divisible by 4")
    cloud = data.reshape(-1, 4).astype(float)
    return cloud if include_intensity else cloud[:, :3].copy()


def load_las(path: str | Path) -> dict[str, np.ndarray]:
    """Load LAS/LAZ points and common attributes using optional ``laspy``."""

    try:
        import laspy  # type: ignore
    except ImportError as exc:
        raise ImportError("LAS/LAZ support requires `python -m pip install laspy`.") from exc

    las = laspy.read(str(path))
    points = np.column_stack([las.x, las.y, las.z]).astype(float)
    result: dict[str, np.ndarray] = {"points": points}
    if hasattr(las, "intensity"):
        result["intensity"] = np.asarray(las.intensity)
    if hasattr(las, "classification"):
        result["classification"] = np.asarray(las.classification)
    return result


def save_las(path: str | Path, points: np.ndarray) -> None:
    """Save points to a simple LAS file using optional ``laspy``."""

    try:
        import laspy  # type: ignore
    except ImportError as exc:
        raise ImportError("LAS/LAZ support requires `python -m pip install laspy`.") from exc

    pts = _ensure_points(points)
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = pts.min(axis=0) if len(pts) else np.zeros(3)
    header.scales = np.asarray([0.001, 0.001, 0.001])
    las = laspy.LasData(header)
    las.x = pts[:, 0]
    las.y = pts[:, 1]
    las.z = pts[:, 2]
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    las.write(str(output))


def _load_xyz(path: Path) -> np.ndarray:
    data = np.loadtxt(path, dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return _ensure_points(data)


def _load_ascii_ply(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as f:
        first = f.readline().strip()
        if first != "ply":
            raise ValueError(f"{path} is not a PLY file")
        vertex_count = None
        is_ascii = False
        while True:
            line = f.readline()
            if not line:
                raise ValueError("invalid PLY header: missing end_header")
            stripped = line.strip()
            if stripped == "format ascii 1.0":
                is_ascii = True
            elif stripped.startswith("element vertex"):
                vertex_count = int(stripped.split()[-1])
            elif stripped == "end_header":
                break
        if not is_ascii:
            raise ValueError("fallback PLY reader only supports ASCII PLY")
        if vertex_count is None:
            raise ValueError("invalid PLY header: missing vertex count")
        rows = []
        for _ in range(vertex_count):
            parts = f.readline().strip().split()
            if len(parts) < 3:
                raise ValueError("invalid PLY vertex row")
            rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return np.asarray(rows, dtype=float)


def _save_ascii_ply(path: Path, points: np.ndarray, colors: np.ndarray | None) -> None:
    has_colors = colors is not None
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if has_colors:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")
        if has_colors:
            cols = np.asarray(colors, dtype=float)
            if cols.max(initial=0) <= 1:
                cols = np.rint(cols * 255)
            cols = np.clip(cols, 0, 255).astype(int)
            for point, color in zip(points, cols, strict=True):
                f.write(
                    f"{point[0]:.8f} {point[1]:.8f} {point[2]:.8f} "
                    f"{color[0]} {color[1]} {color[2]}\n"
                )
        else:
            for point in points:
                f.write(f"{point[0]:.8f} {point[1]:.8f} {point[2]:.8f}\n")


def _load_ascii_pcd(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as f:
        fields: list[str] | None = None
        data_mode = None
        rows: list[list[float]] = []
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            upper = stripped.upper()
            if upper.startswith("FIELDS"):
                fields = stripped.split()[1:]
            elif upper.startswith("DATA"):
                data_mode = stripped.split()[1].lower()
                if data_mode != "ascii":
                    raise ValueError("fallback PCD reader only supports DATA ascii")
                break
        if fields is None:
            raise ValueError("invalid PCD header: missing FIELDS")
        if data_mode != "ascii":
            raise ValueError("invalid PCD header: missing DATA ascii")
        try:
            x_i, y_i, z_i = fields.index("x"), fields.index("y"), fields.index("z")
        except ValueError as exc:
            raise ValueError("PCD file must contain x, y, z fields") from exc
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            rows.append([float(parts[x_i]), float(parts[y_i]), float(parts[z_i])])
    return np.asarray(rows, dtype=float)


def _save_ascii_pcd(path: Path, points: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write("FIELDS x y z\n")
        f.write("SIZE 4 4 4\n")
        f.write("TYPE F F F\n")
        f.write("COUNT 1 1 1\n")
        f.write(f"WIDTH {len(points)}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {len(points)}\n")
        f.write("DATA ascii\n")
        for point in points:
            f.write(f"{point[0]:.8f} {point[1]:.8f} {point[2]:.8f}\n")


def to_open3d_point_cloud(points: np.ndarray, colors: np.ndarray | None = None):
    """Convert a NumPy point cloud to Open3D, raising if Open3D is absent."""

    o3d = _optional_open3d()
    if o3d is None:
        raise ImportError("Open3D is required for interactive visualization")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(_ensure_points(points))
    if colors is not None:
        cols = np.asarray(colors, dtype=float)
        if cols.max(initial=0) > 1:
            cols = cols / 255.0
        pcd.colors = o3d.utility.Vector3dVector(cols[:, :3])
    return pcd


def stack_point_clouds(point_sets: Iterable[np.ndarray]) -> np.ndarray:
    arrays = [_ensure_points(points) for points in point_sets]
    if not arrays:
        return np.empty((0, 3), dtype=float)
    return np.vstack(arrays)
