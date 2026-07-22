# Use the Debian Bookworm base image
FROM debian:bookworm

# Set environment variables to avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies needed by OpenCV and other native libs
RUN apt-get update && apt-get install -y \
    git \
    make \
    wget \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Set the working directory
WORKDIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Keep the venv outside /app — the repo is bind-mounted over /app at runtime,
# which would shadow a venv baked at /app/.venv with the host's stale one.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv

# Sync dependencies — uv downloads and manages Python 3.13 automatically
RUN uv sync --frozen --no-dev

# Add the venv to PATH so 'python' resolves to the venv's Python
ENV PATH="/opt/venv/bin:$PATH"

CMD ["/bin/bash"]
