PYTHON ?= python

.PHONY: install install-dev compile data pipeline test lint format-check
.PHONY: check-hygiene check-packaging check-devcontainer check-fixtures
.PHONY: check-documented-commands
.PHONY: check-artifact-schema check-release-ready check-v1-ready audit-repository
.PHONY: benchmark scale-benchmark verify-core verify-portfolio verify-benchmarks
.PHONY: verify-realdata verify-scale-benchmark verify-release-candidate
.PHONY: verify-v1-candidate verify-full verify cpp-demo
.PHONY: web-backend web-frontend web-build web-test verify-web

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev,vis,bench]"

compile:
	$(PYTHON) -m compileall -q main.py pointcloud_geolab tests examples scripts benchmarks web/backend

data:
	$(PYTHON) examples/generate_demo_data.py --output examples/demo_data

pipeline: data
	$(PYTHON) -m pointcloud_geolab pipeline --input examples/demo_data --output outputs/portfolio_demo

test:
	$(PYTHON) -m pytest --cov=pointcloud_geolab

lint:
	$(PYTHON) -m ruff check .

format-check:
	$(PYTHON) -m black --check .

check-hygiene:
	$(PYTHON) scripts/check_repo_hygiene.py

check-packaging:
	$(PYTHON) scripts/check_packaging.py

check-devcontainer:
	$(PYTHON) scripts/check_devcontainer.py

check-fixtures:
	$(PYTHON) scripts/check_dataset_fixtures.py

check-artifact-schema:
	$(PYTHON) scripts/check_artifact_schema.py

check-documented-commands:
	$(PYTHON) scripts/check_documented_commands.py

check-release-ready:
	$(PYTHON) scripts/check_release_ready.py

check-v1-ready:
	$(PYTHON) scripts/check_v1_ready.py

audit-repository:
	$(PYTHON) scripts/audit_repository_state.py

benchmark:
	$(PYTHON) -m pointcloud_geolab benchmark --suite all --quick --output outputs/benchmarks

scale-benchmark:
	$(PYTHON) scripts/run_scale_benchmark.py --quick --repeat 2 --output-dir outputs/scale_benchmark

verify-core: compile lint format-check test check-hygiene check-devcontainer check-packaging check-fixtures check-artifact-schema
	$(PYTHON) scripts/check_documented_commands.py

verify-portfolio: data pipeline
	$(PYTHON) scripts/verify_portfolio.py --quick --output-dir outputs

verify-benchmarks: benchmark
	$(PYTHON) scripts/verify_benchmarks.py --output-dir outputs/benchmarks

verify-realdata:
	$(PYTHON) scripts/verify_realdata_workflow.py --dry-run

verify-scale-benchmark: scale-benchmark
	$(PYTHON) scripts/verify_benchmarks.py --output-dir outputs/scale_benchmark --suite scale

verify-release-candidate: verify-core verify-portfolio verify-benchmarks check-release-ready check-artifact-schema

verify-v1-candidate: verify-core verify-portfolio verify-benchmarks verify-realdata verify-scale-benchmark check-v1-ready

verify-full: verify-core verify-portfolio verify-benchmarks

verify: verify-full

cpp-demo:
	cmake -S cpp -B cpp/build
	cmake --build cpp/build --config Release

web-backend:
	$(PYTHON) -m uvicorn web.backend.app.main:app --reload --host 127.0.0.1 --port 8000

web-frontend:
	cd web/frontend && npm run dev -- --host 127.0.0.1

web-build:
	cd web/frontend && npm install && npm test && npm run build

web-test:
	$(PYTHON) -m pytest web/backend/tests

verify-web: web-test web-build
