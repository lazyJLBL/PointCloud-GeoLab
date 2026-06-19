from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from web.backend.app.api import datasets as dataset_api
from web.backend.app.config import Settings
from web.backend.app.main import create_app


def test_health_endpoint(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["python_versions"] == "3.10-3.12"


def test_upload_list_preview_and_delete_dataset(tmp_path: Path) -> None:
    client = _client(tmp_path)

    upload = _upload_xyz(client)
    dataset_id = upload["id"]

    listing = client.get("/api/datasets")
    assert listing.status_code == 200
    assert listing.json()[0]["id"] == dataset_id

    preview = client.get(f"/api/datasets/{dataset_id}/preview")
    assert preview.status_code == 200
    assert preview.json()["point_count"] == 4
    assert preview.json()["sampled_count"] == 4

    deleted = client.delete(f"/api/datasets/{dataset_id}")
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] is True
    assert client.get(f"/api/datasets/{dataset_id}").status_code == 404


def test_upload_rejects_unsupported_extension(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.post(
        "/api/datasets/upload",
        files={"file": ("cloud.npy", b"0 0 0\n", "application/octet-stream")},
    )

    assert response.status_code == 400
    assert "unsupported upload extension" in response.json()["detail"]


def test_upload_accepts_off_and_preview(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.post(
        "/api/datasets/upload",
        files={"file": ("mini.off", _mini_off_bytes(), "application/octet-stream")},
    )

    assert response.status_code == 200, response.text
    dataset_id = response.json()["id"]
    preview = client.get(f"/api/datasets/{dataset_id}/preview")
    assert preview.status_code == 200, preview.text
    assert preview.json()["point_count"] == 5
    assert preview.json()["sampled_count"] == 5


def test_upload_rejects_path_like_filename(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.post(
        "/api/datasets/upload",
        files={"file": ("../cloud.xyz", b"0 0 0\n", "text/plain")},
    )

    assert response.status_code == 400
    assert "path separators" in response.json()["detail"]


def test_upload_size_failure_cleans_dataset_directory(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.max_upload_bytes = 4
    client = TestClient(create_app(settings))

    response = client.post(
        "/api/datasets/upload",
        files={"file": ("large.xyz", b"0 0 0\n1 1 1\n", "text/plain")},
    )

    assert response.status_code == 413
    assert not list((tmp_path / "web" / "uploads").glob("*"))


def test_upload_write_failure_cleans_dataset_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client(tmp_path)

    async def fail_write(*args, **kwargs) -> int:
        raise OSError("disk write failed")

    monkeypatch.setattr(dataset_api, "write_upload_file", fail_write)
    response = client.post(
        "/api/datasets/upload",
        files={"file": ("cloud.xyz", b"0 0 0\n", "text/plain")},
    )

    assert response.status_code == 500
    assert "disk write failed" in response.json()["detail"]
    assert not list((tmp_path / "web" / "uploads").glob("*"))


def test_preview_samples_large_cloud(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.preview_point_limit = 3
    client = TestClient(create_app(settings))
    payload = "\n".join(f"{i} {i + 1} {i + 2}" for i in range(8)).encode()

    upload = client.post(
        "/api/datasets/upload",
        files={"file": ("large.xyz", payload, "text/plain")},
    ).json()
    preview = client.get(f"/api/datasets/{upload['id']}/preview")

    assert preview.status_code == 200
    assert preview.json()["point_count"] == 8
    assert preview.json()["sampled_count"] == 3


def test_nested_artifact_download_and_traversal_guard(tmp_path: Path) -> None:
    client = _client(tmp_path)
    storage = client.app.state.storage
    task = storage.init_task("dummy", {"parameters": {}})
    nested = storage.artifacts_dir(task.id) / "figures" / "plot.png"
    nested.parent.mkdir(parents=True)
    nested.write_bytes(b"png")
    storage.save_task(
        storage.update_task(
            task,
            status=task.status,
            artifacts=storage.collect_artifacts(task.id),
        )
    )

    response = client.get(f"/api/artifacts/{task.id}/artifacts/figures/plot.png")
    assert response.status_code == 200
    assert response.content == b"png"
    assert "artifacts/figures/plot.png" in storage.collect_artifacts(task.id).values()

    traversal = client.get(f"/api/artifacts/{task.id}/artifacts/%2E%2E/result.json")
    assert traversal.status_code == 404


def test_preprocessing_task_writes_required_files(tmp_path: Path) -> None:
    client = _client(tmp_path)
    dataset = _upload_xyz(client)

    response = client.post(
        "/api/tasks/preprocessing",
        json={
            "dataset_id": dataset["id"],
            "parameters": {
                "statistical_nb_neighbors": 0,
                "voxel_size": 0.0,
                "sample_count": 3,
                "seed": 7,
            },
        },
    )

    assert response.status_code == 200
    task = response.json()
    assert task["status"] == "completed"
    assert task["result"]["task"] == "preprocess"
    assert task["result"]["success"] is True
    task_dir = tmp_path / "web" / "tasks" / task["id"]
    assert (task_dir / "request.json").exists()
    assert (task_dir / "result.json").exists()
    assert (task_dir / "logs.txt").exists()
    assert (task_dir / "artifacts" / "metrics.json").exists()

    artifact = client.get(f"/api/artifacts/{task['id']}/result.json")
    assert artifact.status_code == 200
    assert artifact.json()["metrics"]["after_sampling"] == 3


def test_task_failure_uses_json_envelope(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.post(
        "/api/tasks/preprocessing",
        json={"dataset_id": "missing", "parameters": {}},
    )

    assert response.status_code == 200
    task = response.json()
    assert task["status"] == "failed"
    assert task["result"]["task"] == "preprocessing"
    assert task["result"]["success"] is False
    assert "error" in task["result"]
    assert task["result"]["path"] == "missing"


def test_web_benchmark_rejects_long_repeat(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.post(
        "/api/tasks/benchmark",
        json={"parameters": {"suite": "kdtree", "repeat": 4}},
    )

    assert response.status_code == 200
    task = response.json()
    assert task["status"] == "failed"
    assert "repeat must be between 1 and 3" in task["result"]["error"]


def _client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(_settings(tmp_path)))


def _settings(tmp_path: Path) -> Settings:
    settings = Settings()
    settings.output_root = tmp_path / "web"
    settings.max_upload_bytes = 1024 * 1024
    return settings


def _upload_xyz(client: TestClient) -> dict[str, object]:
    response = client.post(
        "/api/datasets/upload",
        files={
            "file": (
                "cloud.xyz",
                b"0 0 0\n1 0 0\n0 1 0\n0 0 1\n",
                "text/plain",
            )
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def _mini_off_bytes() -> bytes:
    return (
        "OFF\n"
        "5 4 0\n"
        "0 0 0\n"
        "1 0 0\n"
        "0 1 0\n"
        "0 0 1\n"
        "1 1 0.5\n"
        "3 0 1 2\n"
        "3 0 1 3\n"
        "3 0 2 3\n"
        "3 1 2 4\n"
    ).encode()
