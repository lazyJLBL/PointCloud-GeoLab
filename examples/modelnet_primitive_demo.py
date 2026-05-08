"""Run primitive and PCA diagnostics on a small converted ModelNet sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pointcloud_geolab.geometry import compute_obb, pca_analysis, ransac_fit_primitive
from pointcloud_geolab.io.pointcloud_io import load_point_cloud
from pointcloud_geolab.io.visualization import save_point_cloud_projection


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "external" / "modelnet_small" / "sample.xyz",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "modelnet_demo")
    parser.add_argument("--model", choices=["plane", "sphere", "cylinder"], default="plane")
    parser.add_argument("--threshold", type=float, default=0.03)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.input.exists():
        print(
            "ModelNet sample not found. Convert an OFF mesh first, for example: "
            "`python scripts/prepare_datasets.py convert-modelnet-off --input "
            "data/external/modelnet_small/chair.off --output "
            "data/external/modelnet_small/sample.xyz`.",
            file=sys.stderr,
        )
        return 2

    points = load_point_cloud(args.input)
    obb = compute_obb(points)
    pca = pca_analysis(points)
    try:
        primitive = ransac_fit_primitive(
            points,
            args.model,
            threshold=args.threshold,
            max_iterations=800,
            min_inliers=max(10, len(points) // 10),
            random_state=42,
        )
        primitive_payload = {
            "model": args.model,
            "inliers": len(primitive.inlier_indices),
            "inlier_ratio": len(primitive.inlier_indices) / len(points),
            "residual_mean": primitive.residual_mean,
            "params": primitive.model.get_params(),
        }
    except RuntimeError as exc:
        primitive_payload = {"model": args.model, "error": str(exc)}

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    save_point_cloud_projection(
        output_dir / "modelnet_pca_obb.png",
        [points, obb.corners],
        labels=["points", "OBB corners"],
        title="ModelNet PCA / OBB",
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "input": str(args.input),
                "points": len(points),
                "pca_eigenvalues": pca.eigenvalues.tolist(),
                "obb_extent": obb.extent.tolist(),
                "primitive": primitive_payload,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Points: {len(points)}")
    print(f"OBB extent: {obb.extent}")
    print(f"Primitive result: {primitive_payload}")
    print(f"Artifacts: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
