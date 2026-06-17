# Interview Notes

These notes are written for a reviewer or interviewer who wants to understand
what was implemented, what is optional, and where the limits are.

## 30-Second Project Pitch

PointCloud-GeoLab is a point-cloud geometry portfolio project. The core
algorithms are intentionally visible in Python and NumPy: KDTree,
VoxelHashGrid, ICP variants, RANSAC primitive fitting, PCA/OBB geometry,
clustering, and a small feature-registration pipeline. Optional libraries such
as Open3D, Plotly, SciPy, scikit-learn, and PyTorch are used as baselines,
visualization helpers, or demos, not as the proof of the core implementation.

## What Is Custom

- KDTree nearest, kNN, radius, batch, and bounded queries.
- VoxelHashGrid for fixed-radius spatial lookup and voxel downsampling.
- SVD rigid alignment, point-to-point ICP, point-to-plane ICP, robust ICP, and
  multi-scale ICP.
- GICP-style covariance-weighted ICP using covariance-derived scalar weights
  with weighted SVD updates.
- Plane, sphere, cylinder, and sequential primitive RANSAC.
- DBSCAN, Euclidean clustering, region growing, ground/object segmentation, and
  cluster reports.
- ISS keypoints, local geometric descriptors, descriptor matching, transform
  RANSAC, and ICP refinement.

## What Is Optional Baseline Or Demo

- Open3D FPFH registration is an optional baseline, not required by core CI.
- Open3D mesh reconstruction is an optional demo path.
- Plotly powers optional HTML visualization exports.
- PyTorch powers the optional PointNet demo.
- SciPy and scikit-learn are optional benchmark baselines.

## What Is Documented Workflow Only

- Stanford Bunny, KITTI, and ModelNet workflows describe expected local data
  layouts under `data/external/`.
- The repository does not commit those datasets and does not claim broad
  real-data validation from the synthetic demo outputs.
- Large LiDAR-scale processing still needs streaming, chunking, and profiling.

## KDTree

KDTree recursively splits the point set by coordinate axes. During nearest
neighbor search, it visits the branch that contains the query first, keeps the
current best distance, and prunes the opposite branch when the distance to the
split plane is already larger than that best distance.

How to say it in an interview:

- "Brute force computes every distance. KDTree uses geometry to avoid visiting
  subtrees that cannot contain a closer point."
- "It works best in low dimensions. In high dimensions or huge radius queries,
  pruning becomes weak and it can approach brute force."
- "The tests compare KDTree answers against brute force, including duplicate,
  empty, radius, and high-dimensional cases."

## VoxelHashGrid

VoxelHashGrid maps each point to an integer voxel coordinate and stores points
in a hash map keyed by that coordinate. For fixed-radius queries, it only checks
neighboring voxels that can intersect the search radius.

How to say it:

- "KDTree is general nearest-neighbor search. Voxel hashing is useful when I
  repeatedly need local neighborhoods at a known radius."
- "It trades exact global tree traversal for simple spatial bucketing."
- "It is also a natural basis for voxel downsampling."

## ICP

Point-to-point ICP alternates two steps:

1. Find nearest-neighbor correspondences from transformed source points to the
   target.
2. Solve the best rigid transform with centered SVD.

Key caveat:

- ICP is a local optimizer. It needs enough overlap and a reasonable initial
  pose; otherwise nearest-neighbor correspondences can be wrong.

Diagnostics now include initial RMSE, final RMSE, fitness, correspondence
count, residual history, and step-norm history.

## Point-To-Plane ICP

Point-to-plane ICP minimizes residuals along target normals:

```text
n_i^T (R p_i + t - q_i)
```

Using small-angle updates turns the optimization into a linear least-squares
step. It often converges faster on smooth surfaces, but it depends on stable
normals and can be ill-conditioned on degenerate geometry such as nearly flat
or symmetric point sets.

## GICP-Style ICP

This project includes GICP-style covariance-weighted ICP. It estimates local
covariances and turns them into scalar correspondence weights, then solves a
weighted SVD update.

What to say clearly:

- "This is useful for explaining covariance weighting."
- "This is not a full nonlinear GICP optimizer."
- "The diagnostics explicitly include `full_nonlinear_gicp: false`."

## RANSAC

RANSAC repeatedly samples minimal point sets, fits a candidate model, counts
inliers under a threshold, and keeps the best model. It is used here for plane,
sphere, cylinder, and feature-registration transform estimation.

Important formula:

```text
P = 1 - (1 - w^s)^N
```

`w` is the inlier ratio, `s` is the minimal sample size, and `N` is iterations.
As outliers increase or the model needs more sample points, required iterations
rise quickly.

## DBSCAN And Segmentation

DBSCAN finds dense connected regions and labels sparse points as noise.
Euclidean clustering is simpler: it connects points within a fixed radius.
Ground removal first removes a dominant plane so object clustering is not
connected through the road or floor.

Honest limits:

- Fixed radii are sensitive to LiDAR density changes.
- Slopes, curbs, ramps, walls, and touching objects can break simple ground and
  cluster assumptions.
- The current implementation is suitable for small scenes and review demos, not
  a production autonomy segmentation stack.

## Feature Registration Fallback

The feature path uses ISS keypoints, local descriptors, ratio matching,
transform RANSAC, and ICP refinement. If descriptor matching does not produce
enough correspondences, a geometry fallback can be enabled for diagnostics.

How to say it:

- "Fallback is not descriptor registration success."
- "It is disabled by default and records diagnostics when explicitly enabled."
- "For mature production descriptors, Open3D/PCL feature stacks are the
  baseline to compare against."

## Good Reviewer Questions

- "Which functions are custom and which are optional baselines?"
- "What happens when optional Open3D or PyTorch is not installed?"
- "How do you know KDTree or VoxelHashGrid is correct?"
- "Why does ICP fail with poor initialization?"
- "Why is the GICP implementation not full GICP?"
- "How would you scale this to a full KITTI sequence?"
