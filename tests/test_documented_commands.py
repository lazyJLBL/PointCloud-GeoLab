from __future__ import annotations

import subprocess
from pathlib import Path

from scripts.check_documented_commands import (
    ROOT,
    documented_help_commands,
    documented_web_commands,
    main,
    run_documented_command_checks,
    run_static_documented_command_checks,
)


def test_documented_command_script_runs_on_repository() -> None:
    assert main(["--root", str(ROOT)]) == 0


def test_documented_help_commands_include_examples() -> None:
    commands = [" ".join(command) for command in documented_help_commands(ROOT)]

    assert any("kitti_lidar_segmentation.py --help" in command for command in commands)
    assert any("run_scale_benchmark.py --help" in command for command in commands)


def test_documented_web_commands_are_checked() -> None:
    commands = documented_web_commands()
    static_checks = run_static_documented_command_checks(ROOT)

    assert "make web-backend" in commands
    assert "make web-frontend" in commands
    assert "make verify-web" in commands
    assert static_checks
    assert all(check.success for check in static_checks)


def test_documented_web_command_checker_reports_missing_doc(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "web").mkdir()
    (tmp_path / "README.md").write_text("make web-backend\n", encoding="utf-8")
    (tmp_path / "docs" / "web_console.md").write_text("", encoding="utf-8")
    (tmp_path / "web" / "README.md").write_text("", encoding="utf-8")
    (tmp_path / "Makefile").write_text(
        "web-backend:\n\nweb-frontend:\n\nverify-web:\n",
        encoding="utf-8",
    )

    checks = run_static_documented_command_checks(tmp_path)

    assert any(not check.success and check.location == "docs/web_console.md" for check in checks)


def test_documented_command_checker_reports_failed_help(tmp_path: Path) -> None:
    def fake_runner(command, **kwargs):  # noqa: ANN001, ANN202
        return subprocess.CompletedProcess(command, 2, stdout="", stderr="bad help")

    checks = run_documented_command_checks(tmp_path, runner=fake_runner)

    assert checks
    assert not checks[0].success
    assert checks[0].stderr == "bad help"
