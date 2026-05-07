| Points | Queries | Build Time (s) | Brute Force (s) | KD-Tree (s) | KD Radius (s) | Voxel Radius (s) | Open3D (s) | SciPy cKDTree (s) | sklearn KDTree (s) | Speedup | Correct |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 1,000 | 25 | 0.0033 | 0.0007 | 0.0092 | 0.0023 | 0.0023 | 0.0003 | 0.0004 | 0.0005 | 0.08x | yes |
| 10,000 | 25 | 0.0362 | 0.0044 | 0.0084 | 0.0089 | 0.0083 | 0.0023 | 0.0020 | 0.0028 | 0.52x | yes |
| 100,000 | 25 | 0.4518 | 0.0577 | 0.0110 | 0.0569 | 0.0871 | 0.0309 | 0.0245 | 0.0445 | 5.22x | yes |

Conclusion: Custom KDTree demonstrates pruning logic; SciPy/sklearn are expected to win raw throughput at large N, while voxel hash is competitive for fixed-radius locality.
