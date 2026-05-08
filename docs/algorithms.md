# Algorithms

This document maps each algorithmic claim to implementation files and runnable
commands.

## KDTree

Problem: nearest-neighbor, kNN, and radius queries over `N` points in `D`
dimensions.

Core idea: recursively median-split points by `axis = depth mod D`. During
query, visit the near branch first and prune the far branch when
`(query[axis] - split)^2 > best_distance^2`.

Complexity: build is approximately `O(N log N)`. Balanced nearest-neighbor
queries are often `O(log N)` in low dimensions but degrade toward `O(N)` in high
dimensions or badly distributed data.

Implementation: `pointcloud_geolab/kdtree/kdtree.py`.

Demo:

```bash
python -m pointcloud_geolab benchmark --suite kdtree --quick --output outputs/benchmarks
```

Failure cases: high-dimensional descriptor spaces, many duplicate points, and
queries with a very large radius reduce pruning effectiveness.

## Voxel Hash Grid

Problem: efficient fixed-radius, nearest-neighbor, kNN, and box queries for
point clouds.

Core idea: hash `floor(point / voxel_size)` to a bucket. Radius search only
checks buckets within `ceil(radius / voxel_size)` cells from the query voxel.

Complexity: build is `O(N)`. Query time depends on local density and voxel size.
It is often faster than a tree for fixed-radius locality. The implementation
also supports bounded nearest-neighbor and kNN queries by ordering non-empty
voxel buckets by their minimum possible distance to the query point.

Implementation: `pointcloud_geolab/spatial/voxel_hash.py`.

Demo:

```bash
python examples/voxel_hash_grid_demo.py
python examples/gallery_demo.py
```

Failure cases: too-large voxels overload buckets; too-small voxels increase
bucket traversal overhead.

## SVD Rigid Alignment

Problem: estimate `R, t` minimizing `sum ||R p_i + t - q_i||^2`.

Formula:

1. Compute centroids `p_bar`, `q_bar`.
2. Build covariance `H = (P - p_bar)^T (Q - q_bar)`.
3. SVD `H = U S V^T`.
4. `R = V U^T`, with determinant correction for reflections.
5. `t = q_bar - R p_bar`.

Implementation: `pointcloud_geolab/registration/svd_solver.py`.

Failure cases: fewer than three pairs, collinear points, or bad correspondences.

## ICP Variants

Problem: align a source cloud to a target cloud when correspondences are unknown.

Point-to-point ICP alternates:

1. KDTree nearest-neighbor correspondences.
2. SVD rigid alignment.
3. Transform update and convergence check.

Point-to-plane ICP minimizes:

```text
n^T ((R p + t) - q) ~= n^T ((w x p) + t + p - q)
```

The implementation solves the small-angle linear least-squares system, requires
a minimum correspondence count, and rejects ill-conditioned systems.

Robust ICP adds:

- trimmed correspondences: keep only the closest ratio;
- Huber or Tukey weights: reduce influence of large residuals.

Multi-scale ICP downsamples with coarse voxels first, then refines with smaller
voxels.

Generalized ICP estimates local source/target covariance matrices and weights
each correspondence by the Mahalanobis residual:

```text
e_i^T (C_qi + R C_pi R^T)^-1 e_i
```

This project solves a compact GICP-style loop with custom KDTree
correspondences and weighted SVD updates.

Implementation: `pointcloud_geolab/registration/icp.py` and
`pointcloud_geolab/registration/gicp.py`.

Demo:

```bash
python examples/gallery_demo.py
python examples/gicp_demo.py
python -m pointcloud_geolab register --source data/bunny_source.ply --target data/bunny_target.ply --multiscale --voxel-sizes 0.2 0.1 0.05
```

Failure cases: symmetric geometry, too-large initial pose error, insufficient
overlap, unstable normals, and excessive outliers.

## Feature Registration

Problem: obtain a coarse transform before ICP when initial alignment is poor.

Pipeline:

1. ISS keypoints from local covariance eigenvalue ratios.
2. Local descriptors from linearity, planarity, scattering, curvature, density,
   and eigenvalue statistics.
3. Ratio-test and mutual-nearest descriptor matching.
4. RANSAC over 3 correspondence pairs.
5. ICP refinement.

Implementation:

- `pointcloud_geolab/features/iss.py`
- `pointcloud_geolab/features/descriptors.py`
- `pointcloud_geolab/features/matching.py`
- `pointcloud_geolab/registration/feature_registration.py`

Demo:

```bash
python -m pointcloud_geolab register --source data/bunny_source.ply --target data/bunny_target.ply --method iss_descriptor_ransac_icp --threshold 0.15
```

Failure cases: repetitive local geometry, too few salient points, descriptors
without enough distinctiveness, or very low overlap.

## RANSAC Primitive Fitting

Problem: estimate geometric models under outliers.

Residuals:

- plane: `|n^T x + d|`;
- sphere: `abs(||x - c|| - r)`;
- cylinder: `abs(distance_to_axis(x) - r)`.

Sequential extraction fits candidate model types to remaining points, scores
them by inlier ratio, residual, and a BIC-like complexity penalty, removes
inliers, and repeats.

Implementation: `pointcloud_geolab/geometry/primitive_fitting.py`.

Demo:

```bash
python -m pointcloud_geolab extract-primitives --input data/synthetic_scene.ply --models plane sphere cylinder --threshold 0.04 --max-models 3
```

Failure cases: overlapping primitives, wrong thresholds, degenerate samples, and
very high outlier ratios.

## PCA / OBB

Problem: summarize point cloud orientation and extents.

Core idea: compute covariance eigenvectors, project points onto those axes, and
use min/max projected coordinates to form the oriented bounding box.

Implementation: `pointcloud_geolab/geometry/pca.py` and
`pointcloud_geolab/geometry/bounding_box.py`.

Failure cases: near-spherical data gives unstable axes; thin/linear data has
near-zero eigenvalues.

## Clustering and Ground Removal

DBSCAN expands density-reachable components using radius neighborhoods.
Euclidean clustering computes connected components in the radius graph. Region
growing adds a normal-angle constraint.

Ground removal fits a dominant plane with RANSAC and constrains the plane normal
against a configured axis before clustering non-ground points and computing
AABB/OBB object summaries.

Implementation:

- `pointcloud_geolab/segmentation/clustering.py`
- `pointcloud_geolab/segmentation/region_growing.py`
- `pointcloud_geolab/segmentation/ground.py`

Demo:

```bash
python -m pointcloud_geolab segment --input data/lidar_scene.ply --method euclidean --remove-ground --eps 0.18 --min-points 20 --export-report outputs/segmentation/cluster_report.md
```

Failure cases: sloped roads without adjusted axis thresholds, touching objects,
variable point density, and small objects below `min_points`.

## Benchmark Methodology

Benchmarks use deterministic synthetic data and quick modes for CI. Each suite
emits CSV, Markdown, JSON, and PNG artifacts. The benchmark goal is not to hide
that optimized libraries are faster; it is to prove correctness, expose
trade-offs, and show where custom implementations are educational.

Implementation: `pointcloud_geolab/api.py` benchmark helpers.

Demo:

```bash
python -m pointcloud_geolab benchmark --suite all --quick --output outputs/benchmarks
```
