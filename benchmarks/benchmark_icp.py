"""Benchmark ICP convergence under different perturbations."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.generate_demo_data import make_object_points
from pointcloud_geolab.registration import point_to_point_icp
from pointcloud_geolab.utils.transform import apply_transform, rotation_matrix_from_euler


def rotation_error_degrees(estimated: np.ndarray, expected: np.ndarray) -> float:
    delta = estimated @ expected.T
    cos_angle = (np.trace(delta) - 1.0) / 2.0
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def case_result(
    target: np.ndarray,
    rotation_degrees: float,
    translation_magnitude: float,
    noise: float,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    rotation = rotation_matrix_from_euler(
        np.radians(rotation_degrees) * 0.6,
        -np.radians(rotation_degrees) * 0.35,
        np.radians(rotation_degrees),
    )
    direction = np.asarray([1.0, -0.45, 0.32], dtype=float)
    direction /= np.linalg.norm(direction)
    translation = direction * translation_magnitude
    source = apply_transform(target, rotation, translation)
    if noise > 0:
        source = source + rng.normal(scale=noise, size=source.shape)

    result = point_to_point_icp(source, target, max_iterations=80, tolerance=1e-7)
    expected_rotation = rotation.T
    expected_translation = -rotation.T @ translation
    return {
        "rotation_degrees": rotation_degrees,
        "translation": translation_magnitude,
        "noise": noise,
        "converged": result.converged,
        "iterations": result.iterations,
        "final_rmse": result.final_rmse,
        "rotation_error": rotation_error_degrees(result.rotation, expected_rotation),
        "translation_error": float(np.linalg.norm(result.translation - expected_translation)),
    }


def format_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "| Rotation (deg) | Translation | Noise | Converged | Iterations | Final RMSE | Rotation Error (deg) | Translation Error |",
        "|---:|---:|---:|:---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {rotation_degrees:.0f} | {translation:.2f} | {noise:.2f} | {converged} | "
            "{iterations} | {final_rmse:.6f} | {rotation_error:.4f} | {translation_error:.6f} |".format(
                rotation_degrees=row["rotation_degrees"],
                translation=row["translation"],
                noise=row["noise"],
                converged="yes" if row["converged"] else "no",
                iterations=row["iterations"],
                final_rmse=row["final_rmse"],
                rotation_error=row["rotation_error"],
                translation_error=row["translation_error"],
            )
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="run a small README-friendly grid")
    mode.add_argument("--full", action="store_true", help="run the full perturbation grid")
    parser.add_argument("--save", type=Path, help="optional markdown output path")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rng = np.random.default_rng(args.seed)
    target = make_object_points(rng, n=500)

    if args.full:
        rotations = [5.0, 15.0, 30.0, 60.0]
        translations = [0.1, 0.3, 0.5, 1.0]
        noises = [0.0, 0.01, 0.03]
    else:
        rotations = [5.0, 15.0, 30.0]
        translations = [0.1, 0.3]
        noises = [0.0, 0.01]

    rows = []
    seed = args.seed
    for rotation in rotations:
        for translation in translations:
            for noise in noises:
                rows.append(case_result(target, rotation, translation, noise, seed))
                seed += 1

    table = format_table(rows)
    print(table)
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(table + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

