| Points | Queries | Build Time (s) | Brute Force (s) | KD-Tree (s) | KD Radius (s) | Voxel Radius (s) | Open3D (s) | SciPy cKDTree (s) | sklearn KDTree (s) | Speedup | Correct |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 1,000 | 25 | 0.0044 | 0.0009 | 0.0132 | 0.0048 | 0.0020 | 0.0002 | 0.0005 | 0.0008 | 0.07x | yes |
| 10,000 | 25 | 0.0486 | 0.0060 | 0.0094 | 0.0114 | 0.0173 | 0.0032 | 0.0017 | 0.0039 | 0.64x | yes |
| 100,000 | 25 | 0.5968 | 0.0697 | 0.0234 | 0.0823 | 0.1139 | 0.0304 | 0.0294 | 0.0520 | 2.98x | yes |

Conclusion: Custom KDTree demonstrates pruning logic; SciPy/sklearn are expected to win raw throughput at large N, while voxel hash is competitive for fixed-radius locality.
