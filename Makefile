install:
	python -m pip install -r requirements.txt

install-dev:
	python -m pip install -e .[dev]

data:
	python examples/generate_demo_data.py

test:
	python -m pytest

demo:
	python examples/demo_kdtree.py
	python examples/demo_icp.py
	python examples/demo_ransac_plane.py
	python examples/demo_bounding_box.py
	python examples/demo_preprocessing.py

benchmark:
	python benchmarks/benchmark_kdtree.py --save results/kdtree_benchmark.md
	python benchmarks/benchmark_icp.py --quick --save results/icp_benchmark.md

