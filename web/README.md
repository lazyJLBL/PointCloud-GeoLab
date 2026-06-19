# PointCloud-GeoLab Web Console

This directory contains the v1.1.0 experimental Web Console MVP for reviewer
workflows. It is intentionally separate from the stable `pointcloud_geolab`
package.

Backend Python support is 3.10-3.12. The DevContainer and local backend docs use
Python 3.12 as the recommended default.

## Backend

```bash
python -m pip install -e ".[dev,vis,bench]"
python -m pip install -r web/backend/requirements.txt
make web-backend
```

The backend starts at `http://127.0.0.1:8000` and exposes `/api` routes.

## Frontend

```bash
cd web/frontend
npm install
npm run dev
```

The Vite dev server starts at `http://127.0.0.1:5173`.

## Tests

```bash
make web-test
```

The Web tests are independent from `verify-core`. Core repository checks do not
require Node.js or npm.

## Boundaries

The Web Console is not a production LiDAR platform, not a SLAM backend, not a
CUDA stack, not a PointNet training app, and not an official KITTI benchmark.
It calls stable PointCloud-GeoLab task API functions and keeps generated files
under ignored `outputs/web/` paths.

The Web Console is also not a production web platform. Tasks currently execute
synchronously, so long portfolio or benchmark requests may block until they
finish.
