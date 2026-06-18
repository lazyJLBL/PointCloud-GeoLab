from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

from scripts.verify_benchmarks import verify_benchmark_outputs
from scripts.verify_portfolio import (
    EXPECTED_GALLERY_ARTIFACTS,
    EXPECTED_PIPELINE_ARTIFACTS,
    missing_portfolio_artifacts,
    verify_portfolio_outputs,
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


def test_verify_benchmark_outputs_rejects_invalid_png(tmp_path: Path) -> None:
    _write_benchmark_suite(tmp_path, "kdtree")
    (tmp_path / "kdtree_benchmark.png").write_bytes(b"not a png")

    result = verify_benchmark_outputs(tmp_path, suite="kdtree")

    assert not result.success
    assert any("PNG signature" in message for message in result.invalid_files)


def test_verify_benchmark_outputs_rejects_invalid_json(tmp_path: Path) -> None:
    _write_benchmark_suite(tmp_path, "kdtree")
    (tmp_path / "kdtree_benchmark.json").write_text("{not-json", encoding="utf-8")

    result = verify_benchmark_outputs(tmp_path, suite="kdtree")

    assert not result.success
    assert any("invalid JSON" in message for message in result.invalid_files)


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


def test_verify_portfolio_outputs_validates_report_metrics_images_and_transform(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "outputs"
    gallery = tmp_path / "outputs" / "gallery"
    pipeline = output_dir / "portfolio_demo"
    for name in EXPECTED_GALLERY_ARTIFACTS:
        path = gallery / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".png":
            _write_png(path)
        else:
            path.write_text("<html><body>ok</body></html>", encoding="utf-8")
    for name in EXPECTED_PIPELINE_ARTIFACTS:
        path = pipeline / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".png":
            _write_png(path)
        elif path.name == "report.md":
            path.write_text(
                "# Report\n\n"
                "## Registration Results\n"
                "Transformation matrix\n"
                "## Segmentation Results\n"
                "## Current Limitations\n",
                encoding="utf-8",
            )
        elif path.name == "metrics.json":
            path.write_text(
                json.dumps(
                    {
                        "input": {},
                        "preprocessing": {},
                        "registration": {
                            "rmse_before": 1.0,
                            "rmse_after": 0.1,
                            "fitness": 1.0,
                            "transformation": _identity(),
                        },
                        "segmentation": {"num_clusters": 2, "cluster_sizes": [10, 12]},
                        "runtime": {},
                    }
                ),
                encoding="utf-8",
            )
        else:
            path.write_text(
                json.dumps(
                    {
                        "rmse_before": 1.0,
                        "rmse_after": 0.1,
                        "transformation": _identity(),
                    }
                ),
                encoding="utf-8",
            )

    result = verify_portfolio_outputs(root=tmp_path, output_dir=output_dir)

    assert result.success, result.invalid_files


def test_verify_portfolio_outputs_rejects_invalid_metrics(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    gallery = tmp_path / "outputs" / "gallery"
    pipeline = output_dir / "portfolio_demo"
    for name in EXPECTED_GALLERY_ARTIFACTS:
        path = gallery / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".png":
            _write_png(path)
        else:
            path.write_text("<html></html>", encoding="utf-8")
    for name in EXPECTED_PIPELINE_ARTIFACTS:
        path = pipeline / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".png":
            _write_png(path)
        else:
            path.write_text("{}", encoding="utf-8")

    result = verify_portfolio_outputs(root=tmp_path, output_dir=output_dir)

    assert not result.success
    assert any("missing metrics section" in message for message in result.invalid_files)


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
    _write_png(directory / f"{suite}_benchmark.png")


def _write_png(path: Path) -> None:
    width = 1
    height = 1
    raw = b"\x00\x00\x00\x00"
    png = b"\x89PNG\r\n\x1a\n"
    png += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += _png_chunk(b"IDAT", zlib.compress(raw))
    png += _png_chunk(b"IEND", b"")
    path.write_bytes(png)


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    )


def _identity() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
