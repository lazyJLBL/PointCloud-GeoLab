from __future__ import annotations

import json
from pathlib import Path

import pytest

from pointcloud_geolab import cli
from pointcloud_geolab.api import TaskResult


def test_cli_main_geometry_json_invokes_api(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    captured: dict[str, object] = {}

    def fake_geometry(**kwargs):
        captured.update(kwargs)
        return TaskResult(
            task="geometry",
            success=True,
            metrics={"point_count": 3},
            data={"center": [0.0, 0.0, 0.0]},
            artifacts={"metrics": "metrics.json"},
            parameters=kwargs,
        )

    monkeypatch.setattr(cli, "run_geometry_analysis", fake_geometry)

    exit_code = cli.main(
        [
            "geometry",
            "--input",
            "cloud.ply",
            "--output-dir",
            "out",
            "--format",
            "json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["success"] is True
    assert payload["task"] == "geometry"
    assert payload["metrics"]["point_count"] == 3
    assert captured["input_path"] == "cloud.ply"
    assert captured["output_dir"] == Path("out")


def test_cli_legacy_mode_invokes_legacy_api(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    captured: dict[str, object] = {}

    def fake_icp(**kwargs):
        captured.update(kwargs)
        return TaskResult(
            task="icp",
            success=True,
            metrics={"iterations": 1},
            data={"transformation": [[1.0, 0.0, 0.0, 0.0]]},
            parameters=kwargs,
        )

    monkeypatch.setattr(cli, "run_icp", fake_icp)

    exit_code = cli.main(
        [
            "--mode",
            "icp",
            "--source",
            "source.ply",
            "--target",
            "target.ply",
            "--format",
            "json",
        ]
    )

    captured_output = capsys.readouterr()
    payload = json.loads(captured_output.out)
    assert exit_code == 0
    assert payload["task"] == "icp"
    assert captured["source"] == "source.ply"
    assert captured["target"] == "target.ply"
    assert "deprecated" in captured_output.err


def test_cli_batch_reports_invalid_jobs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys,
) -> None:
    def fake_geometry(**kwargs):
        return TaskResult(
            task="geometry",
            success=True,
            metrics={"point_count": 5},
            data={},
            parameters=kwargs,
        )

    monkeypatch.setattr(cli, "run_geometry_analysis", fake_geometry)
    manifest = tmp_path / "batch.yaml"
    manifest.write_text(
        "\n".join(
            [
                "jobs:",
                "  - name: geometry-job",
                "    task: geometry",
                "    input: cloud.ply",
                "  - 42",
                "  - task: unknown",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code = cli.main(
        [
            "--batch",
            str(manifest),
            "--output-dir",
            str(tmp_path / "batch-output"),
            "--format",
            "json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["success"] is False
    assert [job["success"] for job in payload["jobs"]] == [True, False, False]
    assert "must be a mapping" in payload["jobs"][1]["error"]
    assert "must be one of" in payload["jobs"][2]["error"]


def test_cli_invalid_config_returns_error(tmp_path: Path, capsys) -> None:
    config = tmp_path / "bad.yaml"
    config.write_text("42\n", encoding="utf-8")

    exit_code = cli.main(["geometry", "--config", str(config), "--format", "json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 1
    assert payload["success"] is False
    assert payload["task"] == "cli"
    assert payload["parameters"]["config"] == str(config)
    assert "must contain a YAML mapping or list" in payload["error"]
    assert captured.err == ""


def test_cli_requires_command_or_batch(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main([])

    assert exc_info.value.code == 2
    assert "a subcommand or --batch is required" in capsys.readouterr().err
