from __future__ import annotations

import json
from pathlib import Path

from scripts.verify_benchmarks import verify_benchmark_outputs
from scripts.verify_portfolio import (
    EXPECTED_GALLERY_ARTIFACTS,
    EXPECTED_PIPELINE_ARTIFACTS,
    missing_portfolio_artifacts,
)


def test_verify_benchmark_outputs_accepts_complete_suite(tmp_path: Path) -> None:
    _write_benchmark_suite(tmp_path, "kdtree")

    result = verify_benchmark_outputs(tmp_path, suite="kdtree")

    assert result.success
    assert len(result.checked_files) == 4


def test_verify_benchmark_outputs_reports_missing_files(tmp_path: Path) -> None:
    _write_benchmark_suite(tmp_path, "kdtree")
    (tmp_path / "kdtree_benchmark.png").unlink()

    result = verify_benchmark_outputs(tmp_path, suite="kdtree")

    assert not result.success
    assert result.missing_files == [tmp_path / "kdtree_benchmark.png"]


def test_missing_portfolio_artifacts_accepts_expected_files(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    for name in EXPECTED_GALLERY_ARTIFACTS:
        path = tmp_path / "outputs" / "gallery" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ok", encoding="utf-8")
    for name in EXPECTED_PIPELINE_ARTIFACTS:
        path = output_dir / "portfolio_demo" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ok", encoding="utf-8")

    assert missing_portfolio_artifacts(root=tmp_path, output_dir=output_dir) == []


def _write_benchmark_suite(directory: Path, suite: str) -> None:
    (directory / f"{suite}_benchmark.csv").write_text(
        "points,queries,kd_time\n100,10,0.001\n",
        encoding="utf-8",
    )
    (directory / f"{suite}_benchmark.json").write_text(
        json.dumps(
            {
                "benchmark": suite,
                "metadata": {
                    "parameters": {"seed": 7},
                    "data_scale": {"points": [100]},
                    "platform": "test",
                    "python": "3.12",
                },
                "rows": [{"points": 100, "queries": 10, "kd_time": 0.001}],
            }
        ),
        encoding="utf-8",
    )
    (directory / f"{suite}_benchmark.md").write_text(
        "| points | kd_time |\n|---:|---:|\n| 100 | 0.001 |\n\n## Run Metadata\n",
        encoding="utf-8",
    )
    (directory / f"{suite}_benchmark.png").write_bytes(b"png")
