| Points | Queries | Build Time (s) | Brute Force (s) | KD-Tree (s) | KD Radius (s) | Voxel Radius (s) | Open3D (s) | SciPy cKDTree (s) | sklearn KDTree (s) | Speedup | Correct |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 1,000 | 100 | 0.0032 | 0.0021 | 0.0282 | 0.0025 | 0.0022 | 0.0004 | 0.0004 | 0.0005 | 0.08x | yes |
| 10,000 | 100 | 0.0343 | 0.0142 | 0.0409 | 0.0087 | 0.0083 | 0.0026 | 0.0018 | 0.0031 | 0.35x | yes |
| 100,000 | 100 | 0.4127 | 0.2457 | 0.0465 | 0.0564 | 0.0846 | 0.0286 | 0.0241 | 0.0434 | 5.28x | yes |

Conclusion: Custom KDTree demonstrates pruning logic; SciPy/sklearn are expected to win raw throughput at large N, while voxel hash is competitive for fixed-radius locality.
