PYTHON ?= python

.PHONY: install install-dev compile data pipeline test lint format-check
.PHONY: check-hygiene check-packaging check-devcontainer check-fixtures
.PHONY: check-release-ready
.PHONY: benchmark verify-core verify-portfolio verify-benchmarks
.PHONY: verify-release-candidate verify-full verify cpp-demo

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev,vis,bench]"

compile:
	$(PYTHON) -m compileall -q main.py pointcloud_geolab tests examples scripts benchmarks

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

check-release-ready:
	$(PYTHON) scripts/check_release_ready.py

benchmark:
	$(PYTHON) -m pointcloud_geolab benchmark --suite all --quick --output outputs/benchmarks

verify-core: compile lint format-check test check-hygiene check-devcontainer check-packaging check-fixtures

verify-portfolio: data pipeline
	$(PYTHON) scripts/verify_portfolio.py --quick --output-dir outputs

verify-benchmarks: benchmark
	$(PYTHON) scripts/verify_benchmarks.py --output-dir outputs/benchmarks

verify-release-candidate: verify-core verify-portfolio verify-benchmarks check-release-ready

verify-full: verify-core verify-portfolio verify-benchmarks

verify: verify-full

cpp-demo:
	cmake -S cpp -B cpp/build
	cmake --build cpp/build --config Release
