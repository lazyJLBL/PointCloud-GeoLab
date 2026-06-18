"""Small helpers for optional dependency diagnostics."""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
from dataclasses import dataclass
from types import ModuleType
from typing import Any

OPTIONAL_DEPENDENCIES: dict[str, dict[str, str]] = {
    "open3d": {
        "module": "open3d",
        "purpose": "interactive visualization, FPFH registration, and reconstruction",
        "install_hint": "python -m pip install open3d",
    },
    "plotly": {
        "module": "plotly.graph_objects",
        "distribution": "plotly",
        "purpose": "interactive HTML visualization exports",
        "install_hint": "python -m pip install plotly",
    },
    "laspy": {
        "module": "laspy",
        "purpose": "LAS/LAZ point cloud IO",
        "install_hint": "python -m pip install laspy",
    },
    "torch": {
        "module": "torch",
        "purpose": "optional synthetic PointNet demos",
        "install_hint": "install PyTorch from the official CPU or platform-specific wheel index",
    },
    "scipy": {
        "module": "scipy",
        "purpose": "optional benchmark baseline comparisons",
        "install_hint": "python -m pip install scipy",
    },
    "sklearn": {
        "module": "sklearn",
        "distribution": "scikit-learn",
        "purpose": "optional benchmark baseline comparisons",
        "install_hint": "python -m pip install scikit-learn",
    },
    "pandas": {
        "module": "pandas",
        "purpose": "optional notebook/report analysis workflows",
        "install_hint": "python -m pip install pandas",
    },
}


@dataclass(frozen=True, slots=True)
class OptionalDependencyStatus:
    """Availability details for one optional dependency."""

    name: str
    module: str
    installed: bool
    version: str | None
    purpose: str
    install_hint: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly status record."""

        return {
            "name": self.name,
            "module": self.module,
            "installed": self.installed,
            "version": self.version,
            "purpose": self.purpose,
            "install_hint": self.install_hint,
            "error": self.error,
        }


def optional_dependency_status(name: str) -> OptionalDependencyStatus:
    """Return availability metadata without importing heavy packages."""

    info = _dependency_info(name)
    module_name = info["module"]
    distribution = info.get("distribution", module_name.split(".", 1)[0])
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, ValueError) as exc:
        return _status(name, info, installed=False, error=str(exc))
    if spec is None:
        return _status(name, info, installed=False, error="module not installed")
    try:
        version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        version = None
    return _status(name, info, installed=True, version=version)


def require_optional(name: str) -> ModuleType:
    """Import an optional dependency or raise a helpful ImportError."""

    status = optional_dependency_status(name)
    if not status.installed:
        raise ImportError(
            f"Optional dependency `{name}` is unavailable for {status.purpose}. "
            f"Install with `{status.install_hint}`. Core tests skip this path."
        )
    return importlib.import_module(status.module)


def optional_dependency_report(names: list[str] | None = None) -> dict[str, dict[str, Any]]:
    """Return status records for optional dependencies."""

    selected = names or sorted(OPTIONAL_DEPENDENCIES)
    return {name: optional_dependency_status(name).to_dict() for name in selected}


def _dependency_info(name: str) -> dict[str, str]:
    try:
        return OPTIONAL_DEPENDENCIES[name]
    except KeyError as exc:
        known = ", ".join(sorted(OPTIONAL_DEPENDENCIES))
        raise KeyError(f"unknown optional dependency `{name}`; known: {known}") from exc


def _status(
    name: str,
    info: dict[str, str],
    installed: bool,
    version: str | None = None,
    error: str | None = None,
) -> OptionalDependencyStatus:
    return OptionalDependencyStatus(
        name=name,
        module=info["module"],
        installed=installed,
        version=version,
        purpose=info["purpose"],
        install_hint=info["install_hint"],
        error=error,
    )
