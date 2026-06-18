from __future__ import annotations

from pointcloud_geolab.pipeline import PipelineInputs, run_portfolio_pipeline
from pointcloud_geolab.portfolio_pipeline import run_portfolio_pipeline as package_runner
from pointcloud_geolab.portfolio_pipeline.runner import run_portfolio_pipeline as module_runner


def test_pipeline_compatibility_imports_are_preserved() -> None:
    assert run_portfolio_pipeline is module_runner
    assert package_runner is module_runner
    assert PipelineInputs.__name__ == "PipelineInputs"
