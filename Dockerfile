# Wildlife training environments — multi-stage Dockerfile
#
# Build a specific stage with:
#   docker build --target main       -t wildlife-training   .
#   docker build --target speciesnet -t wildlife-speciesnet .
#   docker build --target yolov5     -t wildlife-yolov5     .
#
# Or via the root Makefile:
#   make build
#   make build TARGET=speciesnet IMAGE=wildlife-speciesnet
#   make build TARGET=yolov5    IMAGE=wildlife-yolov5

# ─── Stage: main — YOLO family (Python 3.13, uv) ─────────────────────────────
#
# Base: nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04
#   CUDA 13.x official images are not yet on Docker Hub. The host driver
#   (595.45.04, CUDA 13.x capable) injects libcuda.so at container startup
#   via the NVIDIA container runtime — the 12.8 toolkit baked here is only
#   used at build time. torch+cu130 wheels call into the host driver at
#   runtime, so CUDA 13.x features are fully available inside the container.
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04 AS main

# uv binary from official image (faster and more reliable than pip install uv)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# System dependencies:
#   - deadsnakes PPA for Python 3.13 (not available in nvidia CUDA images)
#   - libgl1 libglib2.0-0 libsm6 libxext6: OpenCV headless runtime requirements
#   - git make htop tmux: development utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.13 \
        python3.13-venv \
        python3.13-dev \
        git \
        make \
        htop \
        tmux \
        curl \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency manifest first for better layer caching
# (pyproject.toml changes less frequently than scripts)
COPY pyproject.toml ./

# Place the venv outside /app so it survives -v /host/repo:/app bind mounts
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV UV_PYTHON=python3.13
ENV PYTHONUNBUFFERED=1

# Install dependencies
# --no-dev: skip dev tools (pytest, black, isort) in the training image
# --compile-bytecode: faster startup inside the container
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --compile-bytecode

# Make installed packages available to Python
ENV PATH="/opt/venv/bin:$PATH"

# Copy scripts (separate layer — changes more frequently than deps)
COPY scripts ./scripts

# scripts/ root is on PYTHONPATH so that _image_utils.py is importable
ENV PYTHONPATH=/app/scripts

# NVIDIA container runtime environment variables
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

CMD ["/bin/bash"]


# ─── Stage: speciesnet — SpeciesNet teacher pipeline (Python 3.11) ────────────
#
# Python 3.11 is required because the speciesnet PyPI package constrains
# to Python <3.13, and 3.11 is the most stable option for all dependencies.
#
# Provides: MegaDetector v5a (PytorchWildlife) + SpeciesNet EfficientNetV2-M
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04 AS speciesnet

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3.11-dev \
        git \
        make \
        htop \
        tmux \
        curl \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install all dependencies with pip directly (simpler than uv for a
# Python-version-pinned auxiliary environment)
# torch cu130: same version as the host for consistency
RUN python3.11 -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip --quiet

RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/venv/bin/pip install \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
        torch==2.11.0+cu130 \
        torchvision==0.26.0+cu130 \
        --quiet

RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/venv/bin/pip install \
        speciesnet \
        PytorchWildlife \
        numpy \
        pillow \
        tqdm \
        pandas \
        opencv-python-headless \
        --quiet

ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH=/app/scripts
ENV PYTHONUNBUFFERED=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

COPY scripts ./scripts

CMD ["/bin/bash"]


# ─── Stage: yolov5 — YOLOv5s baseline (Python 3.10, PyTorch 2.0.1) ───────────
#
# Pinned to PyTorch 2.0.1 + CUDA 11.8 for compatibility with YOLOv5@5cdad89
# (later commits of YOLOv5 require an additional commercial license).
#
# Setup:
#   git clone https://github.com/ultralytics/yolov5.git /opt/yolov5
#   cd /opt/yolov5 && git checkout 5cdad89
#   make build TARGET=yolov5 IMAGE=wildlife-yolov5
#   make run IMAGE=wildlife-yolov5
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 AS yolov5

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3.10-dev python3-pip \
        git make curl wget \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/* \
    && python3.10 -m pip install --upgrade pip

# PyTorch 2.0.1 with CUDA 11.8 — last version fully supported by YOLOv5@5cdad89.
# The host driver (CUDA 12+) injects libcuda.so at runtime via NVIDIA container
# runtime, so CUDA 11.8 toolkit here is only used at build time.
RUN pip3 install \
    torch==2.0.1+cu118 \
    torchvision==0.15.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# YOLOv5 Python dependencies (from requirements.txt at commit 5cdad89).
# The yolov5 source itself is mounted at /opt/yolov5 at runtime — not baked in
# so the host checkout (with any patches) is always used.
RUN pip3 install \
    matplotlib>=3.2.2 \
    numpy>=1.18.5 \
    "opencv-python-headless>=4.1.2" \
    Pillow \
    "PyYAML>=5.3.1" \
    "requests>=2.23.0" \
    "scipy>=1.4.1" \
    "tqdm>=4.41.0" \
    "seaborn>=0.11.0" \
    pandas \
    thop \
    "mlflow>=3.0.0"

# yolov5 source is expected at /opt/yolov5 (bind-mounted by docker run -v)
# repo is expected at /app (bind-mounted by docker run -v)
WORKDIR /opt/yolov5
