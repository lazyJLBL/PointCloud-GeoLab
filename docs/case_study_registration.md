# Case Study: Registration Pipeline

This case study explains the registration path used by the portfolio demo and
the feature-registration examples.

## Why ICP Needs a Good Initial Pose

ICP alternates nearest-neighbor matching and rigid least squares. The nearest
neighbor step is only meaningful when the source cloud is already close to the
target. With a bad initial pose, points are matched to unrelated surfaces, then
SVD solves a clean least-squares problem for the wrong correspondences. The
algorithm can converge, but to a local minimum.

In this repository, `point_to_point_icp`, `point_to_plane_icp`, `robust_icp`,
`multiscale_icp`, and `generalized_icp` all share that local-optimizer
assumption.

## How FPFH/RANSAC Expands the Convergence Basin

Feature registration builds a coarse transform before ICP:

1. Downsample the source and target.
2. Estimate local normals.
3. Compute feature descriptors, such as Open3D FPFH or the project's custom
   covariance-spectrum descriptor.
4. Match descriptors with ratio tests.
5. Run RANSAC over minimal correspondence sets.
6. Refine the coarse transform with ICP.

RANSAC rejects bad feature matches by scoring a candidate transform on all
correspondences. This makes the initial transform good enough that nearest
neighbors become meaningful for ICP.

## Why Robust ICP Helps With Outliers

Least squares is sensitive to large residuals. A small number of outliers can
pull the SVD update away from the real overlap. PointCloud-GeoLab implements
three robust variants:

| Variant | Idea | Good Use Case |
|---|---|---|
| Huber ICP | Quadratic near zero, linear for large residuals | Moderate noise and some outliers |
| Tukey ICP | Gives very large residuals near-zero weight | Strong outliers when overlap is still clear |
| Trimmed ICP | Keeps only the closest correspondence ratio | Partial overlap and source-only outliers |

The benchmark suite reports plain ICP, Huber, trimmed ICP, and optional Open3D
ICP baseline metrics.

## KDTree Role in Correspondence Search

Correspondence search is the hot loop. A brute-force search checks every target
point for every source point. The custom KDTree prunes subtrees whose splitting
plane is already farther than the current best candidate. This keeps the math
visible and makes ICP, normals, DBSCAN, ISS, and descriptor registration share
the same self-implemented spatial primitive.

The VoxelHashGrid complements KDTree for fixed-radius locality, especially in
segmentation and downsampling.

## Failure Modes

- Large initial rotation with symmetric geometry can lead to wrong local minima.
- Low overlap makes nearest-neighbor correspondences ambiguous.
- Bad normal estimates hurt point-to-plane ICP and GICP.
- Repetitive geometry can produce plausible but wrong feature matches.
- Very high outlier ratios require more RANSAC iterations or stronger priors.

## Reproducible Commands

```bash
python examples/generate_demo_data.py --output examples/demo_data
python -m pointcloud_geolab pipeline --input examples/demo_data --output outputs/portfolio_demo
python -m pointcloud_geolab benchmark --suite registration --quick --output outputs/benchmarks
python -m pointcloud_geolab benchmark --suite icp --quick --output outputs/benchmarks
python -m pointcloud_geolab benchmark --suite gicp --quick --output outputs/benchmarks
```
