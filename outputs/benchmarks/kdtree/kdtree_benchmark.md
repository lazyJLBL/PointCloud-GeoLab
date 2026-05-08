| Points | Queries | Build Time (s) | Brute Force (s) | KD-Tree (s) | KD Radius (s) | Voxel Radius (s) | Open3D (s) | SciPy cKDTree (s) | sklearn KDTree (s) | Speedup | Correct |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 1,000 | 100 | 0.0037 | 0.0021 | 0.0295 | 0.0027 | 0.0021 | 0.0004 | 0.0004 | 0.0010 | 0.07x | yes |
| 10,000 | 100 | 0.0367 | 0.0159 | 0.0424 | 0.0087 | 0.0093 | 0.0025 | 0.0021 | 0.0032 | 0.37x | yes |
| 100,000 | 100 | 0.4423 | 0.2419 | 0.0482 | 0.0611 | 0.0835 | 0.0297 | 0.0234 | 0.0426 | 5.02x | yes |

Conclusion: Custom KDTree demonstrates pruning logic; SciPy/sklearn are expected to win raw throughput at large N, while voxel hash is competitive for fixed-radius locality.
