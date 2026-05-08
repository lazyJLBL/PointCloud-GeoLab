# Generalized ICP

Generalized ICP (GICP) keeps the ICP loop but changes the residual model. Plain
point-to-point ICP treats every correspondence residual with the same isotropic
metric:

```text
min_R,t sum ||R p_i + t - q_i||^2
```

GICP estimates a local covariance around each source point and target point.
For correspondence `(p_i, q_i)`, the residual is:

```text
e_i = R p_i + t - q_i
```

and the covariance-aware cost is:

```text
e_i^T (C_qi + R C_pi R^T)^-1 e_i
```

Intuition: if a point lies on a locally planar patch, residuals tangent to the
surface are less informative than residuals along the surface normal. The
covariance matrix encodes that local shape.

## Implementation

File: `pointcloud_geolab/registration/gicp.py`

The implementation is intentionally compact:

1. Estimate local covariance matrices with the custom KDTree kNN query.
2. Find correspondences with the custom KDTree.
3. Convert the Mahalanobis residual into scalar correspondence weights.
4. Solve each rigid update with weighted SVD.
5. Compose transforms until RMSE convergence.

This is not a wrapper around Open3D. Open3D is only used as an optional
industrial baseline in benchmarks.

## Trade-Offs

GICP costs more per iteration than point-to-point ICP because every
correspondence evaluates a covariance system. It can be more stable on smooth
surface data, but it is still local: it needs overlap and a reasonable initial
pose. Bad normals/covariances, thin structures, repeated geometry, and very poor
initialization remain failure modes.

## Commands

```bash
python examples/gicp_demo.py
python -m pointcloud_geolab benchmark --suite gicp --quick --output outputs/benchmarks
```

## Tests

```bash
python -m pytest tests/test_gicp.py
```
