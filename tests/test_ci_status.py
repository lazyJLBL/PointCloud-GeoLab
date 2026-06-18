from __future__ import annotations

import json
import subprocess

from scripts import check_ci_status
from scripts.check_ci_status import check_latest_ci, parse_ci_run


def test_parse_ci_run_success() -> None:
    run = parse_ci_run(
        json.dumps(
            {
                "databaseId": 123,
                "name": "Tests",
                "headSha": "abcdef123456",
                "status": "completed",
                "conclusion": "success",
                "url": "https://example.test/run",
            }
        )
    )

    assert run.is_success
    assert "abcdef1" == run.head_sha[:7]


def test_check_latest_ci_handles_missing_gh(monkeypatch) -> None:
    monkeypatch.setattr(check_ci_status.shutil, "which", lambda name: None)

    code, message = check_latest_ci()

    assert code == 2
    assert "GitHub CLI `gh` was not found" in message


def test_check_latest_ci_uses_run_list_and_view(monkeypatch) -> None:
    monkeypatch.setattr(check_ci_status.shutil, "which", lambda name: "gh")
    commands: list[list[str]] = []

    def runner(command, **kwargs):
        commands.append(command)
        if command[:3] == ["gh", "run", "list"]:
            return _completed(json.dumps([{"databaseId": 42}]))
        if command[:3] == ["gh", "run", "view"]:
            return _completed(
                json.dumps(
                    {
                        "databaseId": 42,
                        "name": "Tests",
                        "headSha": "abcdef123456",
                        "status": "completed",
                        "conclusion": "success",
                        "url": "https://example.test/run",
                    }
                )
            )
        raise AssertionError(command)

    code, message = check_latest_ci(runner=runner)

    assert code == 0
    assert "completed/success" in message
    assert commands[0][:3] == ["gh", "run", "list"]
    assert commands[1][:3] == ["gh", "run", "view"]


def test_check_latest_ci_reports_failure(monkeypatch) -> None:
    monkeypatch.setattr(check_ci_status.shutil, "which", lambda name: "gh")

    def runner(command, **kwargs):
        if command[:3] == ["gh", "run", "list"]:
            return _completed(json.dumps([{"databaseId": 43}]))
        return _completed(
            json.dumps(
                {
                    "databaseId": 43,
                    "name": "Tests",
                    "headSha": "fedcba654321",
                    "status": "completed",
                    "conclusion": "failure",
                    "url": "https://example.test/fail",
                }
            )
        )

    code, message = check_latest_ci(runner=runner)

    assert code == 1
    assert "completed/failure" in message


def _completed(stdout: str, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["gh"],
        returncode=returncode,
        stdout=stdout,
        stderr="",
    )
