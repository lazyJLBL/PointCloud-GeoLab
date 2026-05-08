"""Prepare external point-cloud datasets without committing large assets.

The script documents expected locations, validates local files, converts common
formats into the small ASCII point-cloud formats used by the examples, and
writes SHA256 manifests for reproducibility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pointcloud_geolab.datasets import load_velodyne_frame
from pointcloud_geolab.io.pointcloud_io import load_point_cloud, save_point_cloud
from pointcloud_geolab.utils.transform import apply_transform, rotation_matrix_from_euler

DATASET_LAYOUT = {
    "stanford": {
        "path": "data/external/stanford",
        "description": "Stanford Bunny / Armadillo PLY files",
        "required_suffixes": [".ply"],
    },
    "kitti": {
        "path": "data/external/kitti/velodyne",
        "description": "KITTI Velodyne .bin frames",
        "required_suffixes": [".bin"],
    },
    "modelnet": {
        "path": "data/external/modelnet_small",
        "description": "ModelNet OFF meshes or converted XYZ point samples",
        "required_suffixes": [".off", ".xyz", ".txt", ".ply"],
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    summary = subparsers.add_parser("summary", help="print expected dataset layout")
    summary.add_argument("--root", type=Path, default=ROOT)

    validate = subparsers.add_parser("validate", help="validate local dataset layout")
    validate.add_argument("--root", type=Path, default=ROOT)
    validate.add_argument("--write-manifest", type=Path)

    kitti = subparsers.add_parser("convert-kitti-bin", help="convert one KITTI .bin frame to PLY")
    kitti.add_argument("--input", type=Path, required=True)
    kitti.add_argument("--output", type=Path, required=True)

    modelnet = subparsers.add_parser("convert-modelnet-off", help="sample one OFF mesh to XYZ")
    modelnet.add_argument("--input", type=Path, required=True)
    modelnet.add_argument("--output", type=Path, required=True)
    modelnet.add_argument("--points", type=int, default=2048)
    modelnet.add_argument("--seed", type=int, default=42)

    bunny = subparsers.add_parser(
        "make-bunny-pair",
        help="create source/target PLY files from a single Stanford mesh or point cloud",
    )
    bunny.add_argument("--input", type=Path, required=True)
    bunny.add_argument("--output-dir", type=Path, required=True)
    bunny.add_argument("--angle-deg", type=float, default=12.0)
    bunny.add_argument("--translation", nargs=3, type=float, default=[0.05, -0.03, 0.02])

    checksum = subparsers.add_parser("checksum", help="write SHA256 manifest for a directory")
    checksum.add_argument("--input", type=Path, required=True)
    checksum.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "summary":
        print(dataset_summary(args.root))
        return 0
    if args.command == "validate":
        report = validate_layout(args.root)
        print(format_validation_report(report))
        if args.write_manifest:
            write_manifest(args.root / "data" / "external", args.write_manifest)
        return 0 if all(item["present"] for item in report.values()) else 1
    if args.command == "convert-kitti-bin":
        points = load_velodyne_frame(args.input)
        save_point_cloud(args.output, points)
        print(f"Converted {len(points)} KITTI points to {args.output}")
        return 0
    if args.command == "convert-modelnet-off":
        points = sample_off_mesh(args.input, count=args.points, seed=args.seed)
        save_point_cloud(args.output, points)
        print(f"Sampled {len(points)} points from {args.input} to {args.output}")
        return 0
    if args.command == "make-bunny-pair":
        make_registration_pair(args.input, args.output_dir, args.angle_deg, args.translation)
        print(f"Wrote bunny_source.ply and bunny_target.ply to {args.output_dir}")
        return 0
    if args.command == "checksum":
        write_manifest(args.input, args.output)
        print(f"Wrote checksum manifest to {args.output}")
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


def dataset_summary(root: Path) -> str:
    lines = [
        "# External Dataset Layout",
        "",
        "Large datasets stay outside git. Place or convert files under:",
        "",
    ]
    for name, spec in DATASET_LAYOUT.items():
        lines.append(f"- {name}: `{root / spec['path']}`")
        lines.append(f"  {spec['description']}")
    lines.extend(
        [
            "",
            "Useful commands:",
            "",
            "- `python scripts/prepare_datasets.py validate`",
            "- `python scripts/prepare_datasets.py convert-kitti-bin --input frame.bin "
            "--output frame.ply`",
            "- `python scripts/prepare_datasets.py convert-modelnet-off --input chair.off "
            "--output chair.xyz`",
            "- `python scripts/prepare_datasets.py make-bunny-pair --input bunny.ply "
            "--output-dir data/external/stanford/bunny_pair`",
        ]
    )
    return "\n".join(lines)


def validate_layout(root: Path) -> dict[str, dict[str, object]]:
    report = {}
    for name, spec in DATASET_LAYOUT.items():
        dataset_root = root / str(spec["path"])
        suffixes = set(spec["required_suffixes"])
        files = []
        if dataset_root.exists():
            files = [
                path
                for path in dataset_root.rglob("*")
                if path.is_file() and path.suffix.lower() in suffixes
            ]
        report[name] = {
            "path": str(dataset_root),
            "present": bool(files),
            "files": len(files),
            "suffixes": sorted(suffixes),
        }
    return report


def format_validation_report(report: dict[str, dict[str, object]]) -> str:
    lines = [
        "| Dataset | Present | Files | Path |",
        "|---|:---:|---:|---|",
    ]
    for name, item in report.items():
        present = "yes" if item["present"] else "no"
        lines.append(f"| {name} | {present} | {item['files']} | `{item['path']}` |")
    return "\n".join(lines)


def sample_off_mesh(path: Path, count: int, seed: int = 42) -> np.ndarray:
    if count <= 0:
        raise ValueError("count must be positive")
    vertices, faces = read_off(path)
    rng = np.random.default_rng(seed)
    if len(faces) == 0:
        indices = rng.choice(len(vertices), size=count, replace=len(vertices) < count)
        return vertices[indices]

    triangles = vertices[faces[:, :3]]
    areas = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    probabilities = areas / np.sum(areas) if np.sum(areas) > 0 else None
    face_ids = rng.choice(len(triangles), size=count, replace=True, p=probabilities)
    selected = triangles[face_ids]
    u = rng.random(count)
    v = rng.random(count)
    flip = u + v > 1.0
    u[flip] = 1.0 - u[flip]
    v[flip] = 1.0 - v[flip]
    return (
        selected[:, 0]
        + u[:, None] * (selected[:, 1] - selected[:, 0])
        + v[:, None] * (selected[:, 2] - selected[:, 0])
    )


def read_off(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8") as f:
        header = f.readline().strip()
        if header != "OFF":
            raise ValueError(f"{path} is not an OFF file")
        counts_line = _next_data_line(f)
        vertex_count, face_count, *_ = [int(value) for value in counts_line.split()]
        vertex_rows = []
        for _ in range(vertex_count):
            vertex_rows.append([float(value) for value in _next_data_line(f).split()[:3]])
        vertices = np.asarray(vertex_rows, dtype=float)
        faces = []
        for _ in range(face_count):
            parts = [int(value) for value in _next_data_line(f).split()]
            if parts[0] >= 3:
                faces.append(parts[1:4])
        return vertices, np.asarray(faces, dtype=int)


def make_registration_pair(
    input_path: Path,
    output_dir: Path,
    angle_degrees: float,
    translation: list[float],
) -> None:
    points = load_point_cloud(input_path)
    rotation = rotation_matrix_from_euler(0.0, 0.0, np.radians(angle_degrees))
    source = apply_transform(points, rotation, np.asarray(translation, dtype=float))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_point_cloud(output_dir / "bunny_target.ply", points)
    save_point_cloud(output_dir / "bunny_source.ply", source)


def write_manifest(input_dir: Path, output_path: Path) -> None:
    if not input_dir.exists():
        raise FileNotFoundError(input_dir)
    rows = []
    for path in sorted(item for item in input_dir.rglob("*") if item.is_file()):
        rows.append(
            {
                "path": path.relative_to(input_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _next_data_line(handle) -> str:
    for line in handle:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped
    raise ValueError("unexpected end of OFF file")


if __name__ == "__main__":
    raise SystemExit(main())
