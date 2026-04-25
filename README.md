# PointCloud-GeoLab：点云配准与三维几何处理系统

![Tests](https://github.com/lazyJLBL/PointCloud-GeoLab/actions/workflows/python-tests.yml/badge.svg)

PointCloud-GeoLab is a point cloud registration and 3D geometry processing system for 3D reconstruction, SLAM, robotics perception, and 3D vision practice. It combines custom algorithm implementations with Open3D-based I/O and visualization.

PointCloud-GeoLab 是一个面向三维重建、SLAM、机器人感知和 3D 视觉岗位的点云算法实验系统，重点展示 KD-Tree、ICP、RANSAC、PCA、AABB/OBB 等底层算法能力。

## Features

- Read and write `.ply`, `.pcd`, and `.xyz` point clouds.
- Voxel downsampling, statistical outlier removal, radius outlier removal, and normal estimation.
- Custom KD-Tree with nearest neighbor, KNN, and radius search.
- Point-to-point ICP implemented from scratch with SVD rigid transform estimation.
- RANSAC plane fitting for robust dominant-plane extraction.
- AABB, PCA-based OBB, PCA principal direction analysis, point-to-plane distance, and point-to-line distance.
- Open3D interactive visualization when installed, plus saved Matplotlib result images.
- CLI, Python API, deterministic demo data, examples, docs, tests, and benchmarks.

## Results

### ICP Registration

Before:

![ICP Before](results/icp_before.png)

After:

![ICP After](results/icp_after.png)

RMSE Curve:

![ICP Error Curve](results/icp_error_curve.png)

### RANSAC Plane Fitting

![RANSAC Plane](results/ransac_plane.png)

### PCA-based OBB

![OBB](results/obb_visualization.png)

### Preprocessing

![Preprocessing](results/preprocessing.png)

## Installation

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Windows CMD:

```bat
.venv\Scripts\activate.bat
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Open3D is optional for the core tests because this repo includes ASCII `.ply`, `.pcd`, and `.xyz` fallback readers/writers. Install Open3D for interactive visualization and broader real-world point cloud format support.

For editable package usage:

```bash
python -m pip install -e .
pointcloud-geolab --mode icp --source data/bunny_source.ply --target data/bunny_target.ply
```

## Generate Demo Data

```bash
python examples/generate_demo_data.py
```

This creates small deterministic synthetic point clouds:

- `data/bunny_source.ply`
- `data/bunny_target.ply`
- `data/room.pcd`
- `data/room.xyz`
- `data/object.ply`

## Usage

ICP registration:

```bash
python main.py --mode icp --source data/bunny_source.ply --target data/bunny_target.ply --save-results
```

RANSAC plane fitting:

```bash
python main.py --mode plane --input data/room.pcd --threshold 0.02 --max-iterations 1000 --save-results
```

Geometry analysis:

```bash
python main.py --mode geometry --input data/object.ply --save-results
```

Preprocessing:

```bash
python main.py --mode preprocess --input data/room.pcd --output results/room_clean.ply --voxel-size 0.04 --radius 0.12 --estimate-normals --save-results
```

Add `--visualize` to open an Open3D window when Open3D is installed.

## Demo Scripts

```bash
python examples/demo_kdtree.py
python examples/demo_icp.py
python examples/demo_ransac_plane.py
python examples/demo_bounding_box.py
python examples/demo_preprocessing.py
```

## Python API

```python
from pointcloud_geolab.io import load_point_cloud
from pointcloud_geolab.kdtree import KDTree
from pointcloud_geolab.registration import point_to_point_icp

source = load_point_cloud("data/bunny_source.ply")
target = load_point_cloud("data/bunny_target.ply")

tree = KDTree(target)
idx, dist = tree.nearest_neighbor(source[0])

result = point_to_point_icp(source, target, max_iterations=60, tolerance=1e-7)
print(result.transformation)
print(result.final_rmse)
```

## Why Implement ICP and KD-Tree from Scratch?

This project uses Open3D mainly for point cloud I/O, optional preprocessing support, and visualization. The core algorithms, including KD-Tree search, point-to-point ICP, SVD-based rigid transform estimation, RANSAC plane fitting, PCA analysis, and PCA-based OBB computation, are implemented manually with NumPy.

本项目使用 Open3D 主要完成点云读取、保存和可视化；核心算法如 KD-Tree、ICP、SVD 刚体变换估计、RANSAC 平面拟合和 PCA OBB 均使用 NumPy 自实现。

## ICP Registration

The ICP module implements classic point-to-point ICP without calling Open3D registration APIs.

For matched pairs:

```text
P = {p_i}
Q = {q_i}
```

Compute centroids:

```text
p_mean = mean(P)
q_mean = mean(Q)
```

Center the points:

```text
P_centered = P - p_mean
Q_centered = Q - q_mean
```

Build the covariance matrix and solve with SVD:

```text
H = P_centered^T Q_centered
H = U S V^T
R = V U^T
t = q_mean - R p_mean
```

If `det(R) < 0`, the final singular vector is flipped to avoid reflection.

## KD-Tree Implementation

The KD-Tree is built by recursively splitting points along alternating axes using a median split. It supports:

- `nearest_neighbor(query_point)`
- `knn_search(query_point, k)`
- `radius_search(query_point, radius)`

ICP uses this custom KD-Tree for correspondence search.

## RANSAC Plane Fitting

RANSAC repeatedly samples 3 points, estimates:

```text
ax + by + cz + d = 0
```

Then scores the model by point-to-plane distance:

```text
distance = |ax + by + cz + d| / sqrt(a^2 + b^2 + c^2)
```

The model with the largest inlier set is returned with inliers, outliers, and inlier ratio.

## 3D Geometry Processing

The geometry module includes:

- AABB from coordinate-wise min/max bounds.
- PCA eigenvalues/eigenvectors and principal directions.
- PCA-based OBB using principal axes as the local box frame.
- Point-to-plane and point-to-line distances.

## Benchmarks

Run:

```bash
python benchmarks/benchmark_kdtree.py --save results/kdtree_benchmark.md
python benchmarks/benchmark_icp.py --quick --save results/icp_benchmark.md
```

Representative KD-Tree output from this repository:

| Points | Queries | Build Time (s) | Brute Force (s) | KD-Tree (s) | Speedup | Correct |
|---:|---:|---:|---:|---:|---:|:---:|
| 1,000 | 100 | 0.0050 | 0.0023 | 0.0319 | 0.07x | yes |
| 5,000 | 100 | 0.0174 | 0.0069 | 0.0415 | 0.17x | yes |
| 10,000 | 100 | 0.0353 | 0.0153 | 0.0437 | 0.35x | yes |
| 50,000 | 100 | 0.1874 | 0.1158 | 0.0444 | 2.61x | yes |

Representative ICP quick benchmark output:

| Rotation (deg) | Translation | Noise | Converged | Iterations | Final RMSE | Rotation Error (deg) | Translation Error |
|---:|---:|---:|:---:|---:|---:|---:|---:|
| 5 | 0.10 | 0.00 | yes | 7 | 0.000000 | 0.0000 | 0.000000 |
| 5 | 0.30 | 0.01 | yes | 13 | 0.016755 | 0.0922 | 0.000530 |
| 15 | 0.30 | 0.01 | yes | 15 | 0.016667 | 0.0708 | 0.000955 |
| 30 | 0.30 | 0.01 | yes | 18 | 0.016763 | 0.0805 | 0.001085 |

The exact timings depend on CPU, Python version, and BLAS build, so benchmark tables are generated locally instead of hard-coded.

## Technical Highlights

- 自实现 KD-Tree，支持最近邻、KNN 和半径查询。
- 从零实现 point-to-point ICP，使用 SVD 求解刚体变换。
- 实现基于 RANSAC 的三维平面拟合，可从含噪点云中鲁棒提取主平面。
- 实现 AABB、基于 PCA 的 OBB、点到平面距离、点到直线距离和主方向分析。
- 使用 Open3D 完成点云读取、可视化、降采样、法向量估计相关展示能力。
- 构建模块化点云处理 pipeline，覆盖预处理、配准、分割和几何分析。

## Interview Notes

1. ICP 的核心瓶颈是最近邻搜索，所以使用 KD-Tree 加速每轮对应点匹配。
2. SVD 用于求解两组匹配点之间的最优刚体变换。
3. RANSAC 用于在含噪声和离群点的点云中鲁棒拟合平面。
4. PCA 可以求点云主方向，并用于构造更贴合物体方向的 OBB。
5. Open3D 主要用于 I/O 和可视化，核心算法由 NumPy 实现。

## Resume Description

PointCloud-GeoLab：点云配准与三维几何处理系统

- 基于 Python、NumPy 和 Open3D 构建点云处理 pipeline，支持 `.ply` / `.pcd` / `.xyz` 点云读取、体素降采样、离群点去除、法向量估计和可视化。
- 自实现 KD-Tree，支持最近邻、KNN 和半径搜索，并用于 ICP 配准中的对应点搜索。
- 从零实现 point-to-point ICP，基于 SVD 求解最优刚体变换，输出旋转矩阵、平移向量、RMSE 和收敛曲线。
- 实现 RANSAC 平面拟合算法，支持从含噪点云中提取主平面并区分内点/离群点。
- 实现 AABB、PCA-based OBB、点到平面距离、点到直线距离和 PCA 主方向分析。
- 提供 CLI、Python API、demo scripts、单元测试和算法文档，覆盖点云配准、分割、空间搜索和几何处理流程。

## Project Structure

```text
PointCloud-GeoLab/
├── main.py
├── pointcloud_geolab/
│   ├── io/
│   ├── preprocessing/
│   ├── kdtree/
│   ├── registration/
│   ├── segmentation/
│   ├── geometry/
│   └── utils/
├── examples/
├── benchmarks/
├── tests/
├── configs/
├── data/
├── results/
└── docs/
```

## Tests

```bash
python -m pytest
```

The tests cover KD-Tree correctness against brute force, edge cases, SVD transform recovery, ICP convergence, RANSAC plane fitting, geometry utilities, and CLI smoke flows.

## Future Work

- Add point-to-plane ICP.
- Add FPFH feature extraction and global registration.
- Add mesh reconstruction demos.
- Add larger real-world datasets with clear licenses.
- Extend benchmarks with Open3D or scikit-learn KD-Tree comparisons.
