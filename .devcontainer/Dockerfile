FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspaces/PointCloud-GeoLab

LABEL org.opencontainers.image.description="Dependency-light PointCloud-GeoLab reviewer environment. Open3D/ML extras and external datasets are optional and not installed."
