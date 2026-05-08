| Points | Queries | Build Time (s) | Brute Force (s) | KD-Tree (s) | KD Radius (s) | Voxel Radius (s) | Open3D (s) | SciPy cKDTree (s) | sklearn KDTree (s) | Speedup | Correct |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 1,000 | 100 | 0.0037 | 0.0022 | 0.0285 | 0.0023 | 0.0023 | 0.0004 | 0.0004 | 0.0005 | 0.08x | yes |
| 10,000 | 100 | 0.0359 | 0.0142 | 0.0386 | 0.0080 | 0.0078 | 0.0024 | 0.0018 | 0.0034 | 0.37x | yes |
| 100,000 | 100 | 0.4035 | 0.2247 | 0.0452 | 0.0560 | 0.0869 | 0.0278 | 0.0221 | 0.0430 | 4.97x | yes |

Conclusion: Custom KDTree demonstrates pruning logic; SciPy/sklearn are expected to win raw throughput at large N, while voxel hash is competitive for fixed-radius locality.
