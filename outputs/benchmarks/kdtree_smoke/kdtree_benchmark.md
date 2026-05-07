| Points | Queries | Build Time (s) | Brute Force (s) | KD-Tree (s) | KD Radius (s) | Voxel Radius (s) | Open3D (s) | SciPy cKDTree (s) | sklearn KDTree (s) | Speedup | Correct |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 1,000 | 100 | 0.0032 | 0.0020 | 0.0279 | 0.0025 | 0.0036 | 0.0004 | 0.0006 | 0.0005 | 0.07x | yes |
| 10,000 | 100 | 0.0351 | 0.0138 | 0.0401 | 0.0082 | 0.0086 | 0.0029 | 0.0022 | 0.0030 | 0.34x | yes |
| 100,000 | 100 | 0.4140 | 0.2426 | 0.0477 | 0.0564 | 0.0834 | 0.0287 | 0.0233 | 0.0446 | 5.09x | yes |

Conclusion: Custom KDTree demonstrates pruning logic; SciPy/sklearn are expected to win raw throughput at large N, while voxel hash is competitive for fixed-radius locality.
