# Experimental Web API

The Web API is provided by the FastAPI backend under `web/backend`. It is for
the v1.1.0 experimental Web Console and reviewer workflows. It is not a
production service boundary or production web platform.

Supported backend Python versions are 3.10-3.12.

See [Experimental Web Console](web_console.md) for the upload -> preview ->
task -> artifacts reviewer workflow and mockup images.

## Start

```bash
python -m pip install -r web/backend/requirements.txt
make web-backend
```

The default base URL is:

```text
http://127.0.0.1:8000/api
```

## Health

- `GET /api/health`

Returns service status and marks the Web Console as experimental.

## Datasets

- `POST /api/datasets/upload`
- `GET /api/datasets`
- `GET /api/datasets/{dataset_id}`
- `DELETE /api/datasets/{dataset_id}`
- `GET /api/datasets/{dataset_id}/preview`

Uploads accept `.ply`, `.pcd`, `.xyz`, `.txt`, `.bin`, and `.off` files.
Preview responses sample at most 10,000 points.

## Tasks

Task requests use this shape:

```json
{
  "dataset_id": "ds_x",
  "source_dataset_id": "ds_source",
  "target_dataset_id": "ds_target",
  "parameters": {}
}
```

Single-dataset routes use `dataset_id`. Registration routes use
`source_dataset_id` and `target_dataset_id`.

Available task routes:

- `POST /api/tasks/preprocessing`
- `POST /api/tasks/registration/icp`
- `POST /api/tasks/registration/robust-icp`
- `POST /api/tasks/registration/multiscale-icp`
- `POST /api/tasks/segmentation`
- `POST /api/tasks/segmentation/ground-object`
- `POST /api/tasks/geometry`
- `POST /api/tasks/primitives/plane`
- `POST /api/tasks/primitives/fit`
- `POST /api/tasks/primitives/extract`
- `POST /api/tasks/benchmark`
- `POST /api/tasks/portfolio`
- `GET /api/tasks`
- `GET /api/tasks/{task_id}`

Tasks are currently executed synchronously and may block long requests, but
they still persist a lifecycle state: `pending`, `running`, `completed`, or
`failed`.

All PointCloud-GeoLab task results are serialized through
`TaskResult.to_dict()` before being stored in `result.json`.

Benchmark tasks default to quick mode. Timing and memory metadata are local
machine references only.

Long tasks currently run synchronously in the backend process. The API is meant
for local review and should not be treated as a production web platform.

## Artifacts

- `GET /api/artifacts/{task_id}/{artifact_name}`

The backend only serves files from the selected task directory and rejects path
traversal attempts. Nested artifact paths such as
`artifacts/figures/result.png` are supported.
