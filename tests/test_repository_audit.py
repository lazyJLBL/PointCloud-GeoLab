from __future__ import annotations

import subprocess
from pathlib import Path

from scripts.audit_repository_state import collect_audit, format_audit, main


def test_audit_repository_help_runs() -> None:
    try:
        main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0


def test_audit_repository_skips_github_when_gh_unavailable(tmp_path: Path) -> None:
    _write_minimal_repo(tmp_path)

    audit = collect_audit(tmp_path, runner=_fake_runner, gh_available=False)
    rendered = format_audit(audit)

    assert audit.version == "0.1.1"
    assert audit.release_summary == "skipped: GitHub CLI unavailable"
    assert "GitHub CLI unavailable" in audit.warnings[0]
    assert "Repository Audit Snapshot" in rendered


def test_audit_repository_reports_tracked_generated_path(tmp_path: Path) -> None:
    _write_minimal_repo(tmp_path)

    audit = collect_audit(tmp_path, runner=_fake_runner_with_generated, gh_available=False)

    assert not audit.success
    assert audit.generated_tracked == ["outputs/demo/report.md"]


def _fake_runner(
    command: list[str],
    text: bool,
    capture_output: bool,
    check: bool,
) -> subprocess.CompletedProcess[str]:
    del text, capture_output, check
    joined = " ".join(command)
    if "rev-list" in joined:
        return subprocess.CompletedProcess(command, 0, stdout="0\t0\n", stderr="")
    if "tag -l" in joined:
        return subprocess.CompletedProcess(command, 0, stdout="v0.1.1\n", stderr="")
    return subprocess.CompletedProcess(command, 0, stdout="", stderr="")


def _fake_runner_with_generated(
    command: list[str],
    text: bool,
    capture_output: bool,
    check: bool,
) -> subprocess.CompletedProcess[str]:
    joined = " ".join(command)
    if "ls-files" in joined:
        return subprocess.CompletedProcess(command, 0, stdout="outputs/demo/report.md\n", stderr="")
    return _fake_runner(command, text=text, capture_output=capture_output, check=check)


def _write_minimal_repo(root: Path) -> None:
    (root / "pyproject.toml").write_text(
        '[project]\nname = "demo"\nversion = "0.1.1"\n\n'
        "[tool.coverage.report]\nfail_under = 70\n",
        encoding="utf-8",
    )
