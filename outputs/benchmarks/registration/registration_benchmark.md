| fitness | initial_angle_degrees | method | rmse | success |
| --- | --- | --- | --- | --- |
| 1.000000 | 5.000000 | icp | 0.000000 | True |
| 1.000000 | 5.000000 | fpfh_ransac_icp | 0.000000 | True |
| 1.000000 | 35.000000 | icp | 0.094691 | False |
| 1.000000 | 35.000000 | fpfh_ransac_icp | 0.000000 | True |

Conclusion: Feature-based global registration expands ICP's basin of convergence by estimating a coarse pose first.
