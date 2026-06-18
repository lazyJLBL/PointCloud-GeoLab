"""Validate the lightweight DevContainer setup used for reviewer workflows."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]

Runner = Callable[..., subprocess.CompletedProcess[str]]

BANNED_HEAVY_PATTERNS = (
    ("CUDA base/toolkit", re.compile(r"\b(nvidia/cuda|cuda-toolkit|cudnn)\b", re.I)),
    ("PointNet-specific setup", re.compile(r"\bpointnet\b", re.I)),
    ("PyTorch install", re.compile(r"\b(pytorch|torch)\b", re.I)),
    (
        "real-data download",
        re.compile(r"\b(wget|curl)\b.*\b(kitti|modelnet|stanford|nuscenes)\b", re.I),
    ),
)


@dataclass(frozen=True, slots=True)
class DevcontainerCheck:
    """Result of checking DevContainer files and optional Docker availability."""

    checked_files: list[Path]
    issues: list[str]
    warnings: list[str]

    @property
    def success(self) -> bool:
        return not self.issues


def run_devcontainer_checks(
    root: str | Path = ROOT,
    runner: Runner = subprocess.run,
) -> DevcontainerCheck:
    """Run static DevContainer checks and a lightweight Docker availability probe."""

    repo = Path(root).resolve()
    devcontainer = repo / ".devcontainer" / "devcontainer.json"
    dockerfile = repo / ".devcontainer" / "Dockerfile"
    checked: list[Path] = []
    issues: list[str] = []
    warnings: list[str] = []

    if not devcontainer.exists():
        issues.append(".devcontainer/devcontainer.json: missing file")
    else:
        checked.append(devcontainer)
    if not dockerfile.exists():
        issues.append(".devcontainer/Dockerfile: missing file")
    else:
        checked.append(dockerfile)
    if issues:
        return DevcontainerCheck(checked, sorted(issues), warnings)

    devcontainer_text = devcontainer.read_text(encoding="utf-8")
    dockerfile_text = dockerfile.read_text(encoding="utf-8")
    payload = _load_json(devcontainer, devcontainer_text, issues)
    combined_text = f"{devcontainer_text}\n{dockerfile_text}"

    _check_base_image(dockerfile_text, issues)
    _check_dependency_light_text(combined_text, issues)
    if payload is not None:
        _check_devcontainer_payload(payload, issues)
    warnings.extend(_docker_probe(runner))

    return DevcontainerCheck(checked, sorted(issues), sorted(warnings))


def _load_json(path: Path, text: str, issues: list[str]) -> dict[str, object] | None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        issues.append(f"{path}: invalid JSON ({exc})")
        return None
    if not isinstance(payload, dict):
        issues.append(f"{path}: expected JSON object")
        return None
    return payload


def _check_base_image(dockerfile_text: str, issues: list[str]) -> None:
    if not re.search(r"^FROM\s+python:3\.(11|12)-slim\b", dockerfile_text, re.I | re.M):
        issues.append(".devcontainer/Dockerfile: base image must be python 3.11/3.12 slim")
    for package in ["make", "git"]:
        if not re.search(rf"\b{re.escape(package)}\b", dockerfile_text):
            issues.append(f".devcontainer/Dockerfile: missing system package `{package}`")


def _check_dependency_light_text(text: str, issues: list[str]) -> None:
    for label, pattern in BANNED_HEAVY_PATTERNS:
        if pattern.search(text):
            issues.append(f".devcontainer: heavy dependency marker found: {label}")
    lowered = text.lower()
    if ".[dev,vis,bench]" not in text:
        issues.append(".devcontainer: missing editable install for .[dev,vis,bench]")
    if "open3d" not in lowered or "ml" not in lowered or "optional" not in lowered:
        issues.append(".devcontainer: must state that Open3D/ML extras are optional")
    for command in ["python -m pytest", "make verify-core", "make verify-portfolio"]:
        if command not in text:
            issues.append(f".devcontainer: missing reviewer command `{command}`")


def _check_devcontainer_payload(payload: dict[str, object], issues: list[str]) -> None:
    build = payload.get("build")
    if not isinstance(build, dict) or build.get("dockerfile") != "Dockerfile":
        issues.append(".devcontainer/devcontainer.json: build.dockerfile must be Dockerfile")
    if payload.get("workspaceFolder") != "/workspaces/PointCloud-GeoLab":
        issues.append(".devcontainer/devcontainer.json: unexpected workspaceFolder")
    post_create = payload.get("postCreateCommand")
    if not isinstance(post_create, str) or "pip install -e" not in post_create:
        issues.append(".devcontainer/devcontainer.json: postCreateCommand must install package")


def _docker_probe(runner: Runner = subprocess.run) -> list[str]:
    if shutil.which("docker") is None:
        return ["Docker CLI was not found; Docker smoke probe skipped."]
    completed = runner(
        ["docker", "version", "--format", "{{.Server.Version}}"],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip() or "unknown Docker error"
        return [f"Docker daemon probe skipped: {detail}"]
    return [f"Docker daemon available: {completed.stdout.strip()}"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_devcontainer_checks(args.root)
    print(f"Checked {len(result.checked_files)} DevContainer files.")
    for warning in result.warnings:
        print(f"Warning: {warning}")
    if result.issues:
        print("DevContainer issues:")
        for issue in result.issues:
            print(f"- {issue}")
    else:
        print("DevContainer checks passed.")
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
