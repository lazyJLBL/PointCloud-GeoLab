from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_scale_benchmark import main, run_scale_benchmark
from scripts.verify_benchmarks import verify_benchmark_outputs


def test_scale_benchmark_quick_outputs_repeat_and_memory(tmp_path: Path) -> None:
    payload = run_scale_benchmark(
        output_dir=tmp_path,
        quick=True,
        repeat=2,
        sizes=[120],
        queries=6,
    )

    result = verify_benchmark_outputs(tmp_path, suite="scale")
    saved = json.loads((tmp_path / "scale_benchmark.json").read_text(encoding="utf-8"))

    assert result.success, result.invalid_files
    assert payload["metadata"]["repeat"]["count"] == 2
    assert saved["metadata"]["memory"]["available"] is True
    assert saved["rows"][0]["kdtree_query_time_mean"] >= 0


def test_scale_benchmark_cli_rejects_bad_repeat(tmp_path: Path) -> None:
    code = main(["--output-dir", str(tmp_path), "--repeat", "0"])

    assert code == 1


def test_scale_benchmark_cli_rejects_bad_points_and_queries(tmp_path: Path) -> None:
    bad_points = main(["--output-dir", str(tmp_path / "points"), "--points", "0"])
    bad_queries = main(["--output-dir", str(tmp_path / "queries"), "--queries", "0"])

    assert bad_points == 1
    assert bad_queries == 1


def test_scale_verifier_reports_missing_repeat_stat(tmp_path: Path) -> None:
    run_scale_benchmark(output_dir=tmp_path, quick=True, repeat=2, sizes=[80], queries=4)
    path = tmp_path / "scale_benchmark.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["rows"][0]["kdtree_query_time_mean"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = verify_benchmark_outputs(tmp_path, suite="scale")

    assert not result.success
    assert any("kdtree_query_time_mean" in issue for issue in result.invalid_files)


def test_scale_verifier_reports_missing_declared_timing_field(tmp_path: Path) -> None:
    run_scale_benchmark(output_dir=tmp_path, quick=True, repeat=2, sizes=[80], queries=4)
    path = tmp_path / "scale_benchmark.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["rows"][0]["kdtree_query_time"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = verify_benchmark_outputs(tmp_path, suite="scale")

    assert not result.success
    assert any("kdtree_query_time" in issue for issue in result.invalid_files)


def test_scale_benchmark_cli_help_runs() -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0
