# Wildlife object detection — GPU training environment
#
# Base: nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04
#   CUDA 13.x official images are not yet on Docker Hub. The host driver
#   (595.45.04, CUDA 13.x capable) injects libcuda.so at container startup
#   via the NVIDIA container runtime — the 12.8 toolkit baked here is only
#   used at build time. torch+cu130 wheels call into the host driver at
#   runtime, so CUDA 13.x features are fully available inside the container.
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

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

# Default: interactive shell.
# Run training with: docker run ... python scripts/training/2-train_teacher.py
CMD ["/bin/bash"]
