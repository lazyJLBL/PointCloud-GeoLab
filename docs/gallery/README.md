# Gallery

This gallery gives reviewers a quick map of the visual outputs that the project
can generate. Images are synthetic, tiny fixture, or user-provided workflow
examples as labeled.

## Experimental Web Console

![Web Console dashboard mockup](../assets/web_console_dashboard.svg)

Documentation mockup of the reviewer dashboard. The Web Console is an
experimental FastAPI/Vue presentation layer over the stable Python task API,
not a production web platform.

![Web Console dataset preview mockup](../assets/web_console_dataset_preview.svg)

Documentation mockup of the upload and sampled-preview workflow. Supported
formats include `.ply`, `.pcd`, `.xyz`, `.txt`, KITTI-like `.bin`, and
ModelNet-like `.off`.

![Web Console task artifacts mockup](../assets/web_console_task_artifacts.svg)

Documentation mockup of task status, metrics JSON, logs, and nested artifact
downloads. Web tasks currently run synchronously, so long portfolio or
benchmark runs may block until completion.

## Portfolio Pipeline

![Raw synthetic point cloud](../assets/portfolio_raw_pointcloud.png)

Synthetic demo input from the portfolio pipeline.

![Downsampled synthetic point cloud](../assets/portfolio_downsampled.png)

Synthetic downsampled view used for documentation.

![Registration before and after](../assets/portfolio_registration_before_after.png)

Synthetic registration before/after view from the portfolio pipeline.

![Segmentation result](../assets/portfolio_segmentation_result.png)

Synthetic segmentation result from the portfolio pipeline.

![Bounding box and normals](../assets/portfolio_bbox_normals.png)

Synthetic geometry view showing bbox and normals-style annotations.

## KITTI-Like Workflow

![Tiny KITTI-like smoke view](../assets/kitti_case_study_tiny.png)

Tiny synthetic KITTI-like format smoke view. This is not a real KITTI frame and
not an official KITTI benchmark.

## Benchmark

![Scale benchmark quick chart](../assets/scale_benchmark_quick.png)

Illustrative scale-benchmark trend chart. Actual timings should be regenerated
locally and treated as machine-specific.
