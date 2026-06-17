# Limitations

PointCloud-GeoLab is a compact geometry lab for learning, testing, and portfolio
review. It is not a replacement for Open3D, PCL, or a production autonomy stack.

## Algorithmic Limits

- ICP, robust ICP, multi-scale ICP, and GICP-style covariance-weighted ICP are
  local optimizers. They need enough overlap and a reasonable initial pose.
- Point-to-plane ICP depends on stable target normals and can become
  ill-conditioned on perfectly planar or otherwise degenerate geometry.
- GICP-style covariance-weighted ICP uses covariance-informed scalar weights
  with weighted SVD updates. This is not a full nonlinear GICP optimizer.
- RANSAC success drops quickly as the inlier ratio falls, especially for models
  with larger minimal samples such as cylinders.
- PCA/OBB axes are unstable when eigenvalues are close, for example near-sphere
  point clouds.
- DBSCAN and Euclidean clustering use fixed radii. Uneven LiDAR density often
  needs range-aware clustering or adaptive parameters.
- Ground removal assumes a dominant plane whose normal is close to a configured
  axis. Slopes, ramps, curbs, and walls can violate that assumption.

## Data and Scale Limits

- Demo data is intentionally small enough for CI and laptop review.
- Large LiDAR sequences should use streaming, chunking, tiling, or a dedicated
  spatial backend. This repository currently uses in-memory NumPy arrays.
- Real datasets are not committed. The examples document expected layouts and
  exit with preparation hints when data is missing.
- Benchmark numbers are machine-specific. Regenerate them locally and inspect
  the emitted CSV/JSON/Markdown/PNG artifacts.

## Dependency Limits

- Open3D is optional and used for comparison baselines, reconstruction, and some
  feature workflows.
- Plotly is optional for interactive HTML exports.
- PyTorch is optional for the PointNet demo and is not part of the core geometry
  validation path.
- LAS/LAZ support requires `laspy`.

## Engineering Limits

- The public API is typed and documented where it is used by CLI/examples, but
  this is still a small research-style project rather than a long-term stable
  SDK.
- The C++ KDTree demo is independent from the Python package and does not
  accelerate the Python implementation.
- Coverage is useful for regression confidence, but correctness still relies on
  geometry-specific tests against brute force or known transforms.
