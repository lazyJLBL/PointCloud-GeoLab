PYTHON ?= python

.PHONY: install install-dev data pipeline test lint format-check benchmark verify cpp-demo

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev,vis,bench]"

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

benchmark:
	$(PYTHON) -m pointcloud_geolab benchmark --suite all --quick --output outputs/benchmarks

verify: lint format-check data pipeline test

cpp-demo:
	cmake -S cpp -B cpp/build
	cmake --build cpp/build --config Release
