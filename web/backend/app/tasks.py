"""Synchronous task execution for the Web Console MVP."""

from __future__ import annotations

import json
from typing import Any

from web.backend.app.schemas import TaskRecord, TaskStatus
from web.backend.app.services.geolab_service import GeolabService
from web.backend.app.storage import WebStorage


class TaskManager:
    """Run tasks immediately while persisting status and artifacts."""

    def __init__(self, storage: WebStorage, service: GeolabService) -> None:
        self.storage = storage
        self.service = service

    def create_and_run(self, task_type: str, request: dict[str, Any]) -> TaskRecord:
        record = self.storage.init_task(task_type, request)
        record = self.storage.update_task(record, TaskStatus.running)
        task_dir = self.storage.task_dir(record.id)
        artifacts_dir = self.storage.artifacts_dir(record.id)
        log_path = task_dir / "logs.txt"
        try:
            log_path.write_text(f"Started task {task_type}\n", encoding="utf-8")
            result = self.service.run_task(task_type, request, artifacts_dir)
            payload = result.to_dict()
            (task_dir / "result.json").write_text(
                json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            artifacts = self.storage.collect_artifacts(record.id)
            status = TaskStatus.completed if payload.get("success") else TaskStatus.failed
            message = "completed" if status == TaskStatus.completed else "failed"
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"Task {message}\n")
            return self.storage.update_task(
                record,
                status,
                result=payload,
                error=payload.get("error"),
                artifacts=artifacts,
            )
        except Exception as exc:
            error_payload = {
                "task": task_type,
                "success": False,
                "error": str(exc),
                "parameters": request,
                "path": request.get("dataset_id")
                or request.get("source_dataset_id")
                or request.get("target_dataset_id"),
            }
            (task_dir / "result.json").write_text(
                json.dumps(error_payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"Task failed: {exc}\n")
            artifacts = self.storage.collect_artifacts(record.id)
            return self.storage.update_task(
                record,
                TaskStatus.failed,
                result=error_payload,
                error=str(exc),
                artifacts=artifacts,
            )
