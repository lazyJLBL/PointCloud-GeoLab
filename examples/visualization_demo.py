"""Demo: export an interactive Plotly point cloud visualization."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pointcloud_geolab.visualization import export_point_cloud_html, label_colors


def main() -> int:
    rng = np.random.default_rng(140)
    points = np.vstack(
        [
            rng.normal([0, 0, 0], 0.05, size=(80, 3)),
            rng.normal([0.6, 0.2, 0.0], 0.05, size=(80, 3)),
        ]
    )
    labels = np.asarray([0] * 80 + [1] * 80)
    out = ROOT / "outputs" / "visualization"
    out.mkdir(parents=True, exist_ok=True)
    export_point_cloud_html(points, label_colors(labels), out / "clusters.html", title="Clusters")
    print(out / "clusters.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
