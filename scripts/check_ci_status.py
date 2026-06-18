"""Check the latest GitHub Actions Tests workflow status with gh."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True, slots=True)
class CiRun:
    """Small view of a GitHub Actions workflow run."""

    database_id: int
    name: str
    head_sha: str
    status: str
    conclusion: str | None
    url: str

    @property
    def is_success(self) -> bool:
        return self.status == "completed" and self.conclusion == "success"


CommandRunner = Callable[[list[str]], subprocess.CompletedProcess[str]]


def check_latest_ci(
    branch: str = "main",
    workflow: str = "tests.yml",
    runner: CommandRunner = subprocess.run,
) -> tuple[int, str]:
    """Return exit code and human-readable status for the latest workflow run."""

    if shutil.which("gh") is None:
        return (
            2,
            "GitHub CLI `gh` was not found. Install gh and authenticate to check CI status.",
        )

    listed = run_gh(
        [
            "gh",
            "run",
            "list",
            "--branch",
            branch,
            "--workflow",
            workflow,
            "--limit",
            "1",
            "--json",
            "databaseId",
        ],
        runner,
    )
    if listed.returncode != 0:
        return 2, format_command_error("gh run list", listed)

    try:
        runs = json.loads(listed.stdout or "[]")
    except json.JSONDecodeError as exc:
        return 2, f"Could not parse `gh run list` JSON: {exc}"
    if not runs:
        return 1, f"No workflow runs found for {workflow} on branch {branch}."

    run_id = runs[0].get("databaseId")
    if not isinstance(run_id, int):
        return 2, "`gh run list` did not return a numeric databaseId."

    viewed = run_gh(
        [
            "gh",
            "run",
            "view",
            str(run_id),
            "--json",
            "databaseId,name,headSha,status,conclusion,url",
        ],
        runner,
    )
    if viewed.returncode != 0:
        return 2, format_command_error("gh run view", viewed)

    try:
        run = parse_ci_run(viewed.stdout)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return 2, f"Could not parse `gh run view` JSON: {exc}"

    summary = (
        f"{run.name} run {run.database_id} for {run.head_sha[:7]} is "
        f"{run.status}/{run.conclusion or 'none'}: {run.url}"
    )
    if run.is_success:
        return 0, summary
    return 1, summary


def parse_ci_run(raw_json: str) -> CiRun:
    """Parse the JSON returned by `gh run view`."""

    payload = json.loads(raw_json)
    return CiRun(
        database_id=int(payload["databaseId"]),
        name=str(payload.get("name") or "workflow"),
        head_sha=str(payload.get("headSha") or ""),
        status=str(payload["status"]),
        conclusion=payload.get("conclusion"),
        url=str(payload.get("url") or ""),
    )


def run_gh(command: list[str], runner: CommandRunner) -> subprocess.CompletedProcess[str]:
    """Run a gh command with text output enabled."""

    return runner(command, text=True, capture_output=True, check=False)


def format_command_error(label: str, completed: subprocess.CompletedProcess[str]) -> str:
    detail = (completed.stderr or completed.stdout or "").strip()
    return f"{label} failed with exit code {completed.returncode}: {detail}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", default="main", help="Branch to inspect.")
    parser.add_argument("--workflow", default="tests.yml", help="Workflow file or name.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    code, message = check_latest_ci(branch=args.branch, workflow=args.workflow)
    print(message)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
