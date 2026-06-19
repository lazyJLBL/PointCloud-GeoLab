"""Filesystem storage for uploads, task records, and artifacts."""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from web.backend.app.config import Settings
from web.backend.app.schemas import DatasetRecord, TaskRecord, TaskStatus


def utc_now() -> str:
    """Return an ISO timestamp with timezone information."""

    return datetime.now(timezone.utc).isoformat()


def make_id(prefix: str) -> str:
    """Return a short opaque identifier suitable for path names."""

    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def assert_safe_child(root: Path, candidate: Path) -> Path:
    """Resolve candidate and ensure it stays inside root."""

    resolved_root = root.resolve()
    resolved_candidate = candidate.resolve()
    try:
        resolved_candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"path escapes storage root: {candidate}") from exc
    return resolved_candidate


def sanitize_filename(filename: str | None) -> str:
    """Reject path-like upload names and return a safe base name."""

    if not filename:
        raise ValueError("upload filename is required")
    if "/" in filename or "\\" in filename:
        raise ValueError("upload filename must not contain path separators")
    name = Path(filename).name
    if name in {"", ".", ".."} or name != filename:
        raise ValueError("upload filename is not safe")
    return name


class WebStorage:
    """Small JSON-backed storage layer for the Web Console MVP."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.output_root = settings.output_root
        self.uploads_root = settings.uploads_root
        self.tasks_root = settings.tasks_root
        self.uploads_root.mkdir(parents=True, exist_ok=True)
        self.tasks_root.mkdir(parents=True, exist_ok=True)

    def create_dataset_path(self, filename: str) -> tuple[str, Path]:
        dataset_id = make_id("ds")
        upload_dir = assert_safe_child(self.uploads_root, self.uploads_root / dataset_id)
        upload_dir.mkdir(parents=True, exist_ok=False)
        path = assert_safe_child(upload_dir, upload_dir / filename)
        return dataset_id, path

    def save_dataset_record(
        self,
        dataset_id: str,
        filename: str,
        original_filename: str,
        path: Path,
        size_bytes: int,
    ) -> DatasetRecord:
        record = DatasetRecord(
            id=dataset_id,
            filename=filename,
            original_filename=original_filename,
            path=str(path),
            size_bytes=size_bytes,
            extension=path.suffix.lower(),
            created_at=utc_now(),
        )
        self._dataset_metadata_path(dataset_id).write_text(
            record.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        return record

    def list_datasets(self) -> list[DatasetRecord]:
        records: list[DatasetRecord] = []
        for metadata in sorted(self.uploads_root.glob("*/dataset.json")):
            records.append(DatasetRecord.model_validate_json(metadata.read_text(encoding="utf-8")))
        return records

    def get_dataset(self, dataset_id: str) -> DatasetRecord:
        metadata = self._dataset_metadata_path(dataset_id)
        if not metadata.exists():
            raise KeyError(f"dataset not found: {dataset_id}")
        return DatasetRecord.model_validate_json(metadata.read_text(encoding="utf-8"))

    def delete_dataset(self, dataset_id: str) -> None:
        record = self.get_dataset(dataset_id)
        dataset_dir = assert_safe_child(self.uploads_root, Path(record.path).parent)
        shutil.rmtree(dataset_dir)

    def init_task(self, task_type: str, request: dict[str, Any]) -> TaskRecord:
        task_id = make_id("task")
        task_dir = self.task_dir(task_id)
        (task_dir / "artifacts").mkdir(parents=True, exist_ok=False)
        request_path = task_dir / "request.json"
        request_path.write_text(json.dumps(request, indent=2, ensure_ascii=False) + "\n")
        (task_dir / "logs.txt").write_text("", encoding="utf-8")
        now = utc_now()
        record = TaskRecord(
            id=task_id,
            task_type=task_type,
            status=TaskStatus.pending,
            request=request,
            created_at=now,
            updated_at=now,
        )
        self.save_task(record)
        return record

    def save_task(self, record: TaskRecord) -> None:
        self._task_metadata_path(record.id).write_text(
            record.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )

    def update_task(
        self,
        record: TaskRecord,
        status: TaskStatus,
        result: dict[str, Any] | None = None,
        error: str | None = None,
        artifacts: dict[str, str] | None = None,
    ) -> TaskRecord:
        updated = record.model_copy(
            update={
                "status": status,
                "result": result,
                "error": error,
                "artifacts": artifacts or record.artifacts,
                "updated_at": utc_now(),
            }
        )
        self.save_task(updated)
        return updated

    def list_tasks(self) -> list[TaskRecord]:
        records: list[TaskRecord] = []
        for metadata in sorted(self.tasks_root.glob("*/task.json")):
            records.append(TaskRecord.model_validate_json(metadata.read_text(encoding="utf-8")))
        return records

    def get_task(self, task_id: str) -> TaskRecord:
        metadata = self._task_metadata_path(task_id)
        if not metadata.exists():
            raise KeyError(f"task not found: {task_id}")
        return TaskRecord.model_validate_json(metadata.read_text(encoding="utf-8"))

    def task_dir(self, task_id: str) -> Path:
        if "/" in task_id or "\\" in task_id or task_id in {"", ".", ".."}:
            raise ValueError("task_id is not safe")
        return assert_safe_child(self.tasks_root, self.tasks_root / task_id)

    def artifacts_dir(self, task_id: str) -> Path:
        return self.task_dir(task_id) / "artifacts"

    def artifact_path(self, task_id: str, artifact_name: str) -> Path:
        if "/" in artifact_name or "\\" in artifact_name or artifact_name in {"", ".", ".."}:
            raise ValueError("artifact_name is not safe")
        task_dir = self.task_dir(task_id)
        if artifact_name in {"request.json", "result.json", "logs.txt", "task.json"}:
            path = task_dir / artifact_name
        else:
            path = task_dir / "artifacts" / artifact_name
        resolved = assert_safe_child(task_dir, path)
        if not resolved.exists() or not resolved.is_file():
            raise FileNotFoundError(f"artifact not found: {artifact_name}")
        return resolved

    def collect_artifacts(self, task_id: str) -> dict[str, str]:
        task_dir = self.task_dir(task_id)
        artifacts = {
            "request": "request.json",
            "result": "result.json",
            "logs": "logs.txt",
        }
        for path in sorted((task_dir / "artifacts").rglob("*")):
            if path.is_file():
                artifacts[path.stem if path.name == "metrics.json" else path.name] = path.name
        return artifacts

    def _dataset_metadata_path(self, dataset_id: str) -> Path:
        return assert_safe_child(self.uploads_root, self.uploads_root / dataset_id / "dataset.json")

    def _task_metadata_path(self, task_id: str) -> Path:
        return assert_safe_child(self.tasks_root, self.tasks_root / task_id / "task.json")
