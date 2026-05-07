# Registration

PointCloud-GeoLab has two registration tracks:

- Open3D FPFH + RANSAC + project ICP baseline.
- Self-implemented ISS keypoints + local descriptor + matching + RANSAC + ICP.

The first is an industrial baseline. The second exists to make the underlying
geometry explicit.

## ICP Variants

Point-to-point ICP alternates nearest-neighbor matching with SVD rigid
alignment. It is simple and accurate near the correct pose, but sensitive to
bad correspondences.

Point-to-plane ICP uses target normals:

```text
n^T((R p + t) - q) ~= n^T((w x p) + t + p - q)
```

The implementation estimates normals when not provided, requires enough
correspondences, and stops on ill-conditioned least-squares systems.

Robust ICP supports:

- `--trim-ratio`: keep only the closest correspondence fraction;
- `--robust-kernel huber`;
- `--robust-kernel tukey`.

Multi-scale ICP runs from coarse to fine voxel sizes and reports per-level
diagnostics.

Run:

```bash
python -m pointcloud_geolab register \
  --source data/bunny_source.ply \
  --target data/bunny_target.ply \
  --method fpfh_ransac_icp \
  --icp-method point_to_point \
  --multiscale \
  --voxel-sizes 0.2 0.1 0.05 \
  --robust-kernel huber \
  --trim-ratio 0.9 \
  --save-diagnostics outputs/registration/diagnostics.json \
  --output outputs/registration/aligned_source.ply
```

## ISS + Descriptor Registration

Pipeline:

```text
ISS keypoints -> covariance descriptors -> mutual NN matching -> 3-point RANSAC -> ICP
```

ISS uses local covariance eigenvalue ratios. The descriptor includes linearity,
planarity, scattering, curvature, anisotropy, omnivariance, eigenentropy, and
local density. RANSAC samples three matched pairs and scores the transform using
correspondence residuals.

Run:

```bash
python -m pointcloud_geolab register \
  --source data/bunny_source.ply \
  --target data/bunny_target.ply \
  --method iss_descriptor_ransac_icp \
  --threshold 0.15 \
  --output outputs/registration/iss_aligned.ply
```

## Evaluation

The registration metrics module provides:

- `rotation_error_deg(R_est, R_gt)`;
- `translation_error(t_est, t_gt)`;
- `registration_success(...)`.

Gallery evidence:

```bash
python examples/gallery_demo.py
```

Generated assets include `multiscale_icp_curve.png` and
`robust_icp_outlier_comparison.png`.

## Failure Cases

- Low overlap produces wrong nearest neighbors and weak descriptor matches.
- Symmetric objects admit multiple plausible transforms.
- Point-to-plane ICP fails when normals are noisy or geometry is degenerate.
- Robust ICP still fails if most kept correspondences are wrong.
