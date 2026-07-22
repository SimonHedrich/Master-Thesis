REPO_ROOT := $(shell git rev-parse --show-toplevel)
DATA_DIR  := $(REPO_ROOT)/data

YV5_DIR ?= /opt/yolov5

# Image name
IMAGE_NAME = training

# Container name
CONTAINER_NAME = training-container

# Volume mount path
OUTPUT_PATH = ${PWD}/source

# GPU flag
GPU_FLAG = --gpus all

execute: build run

# Build the Docker image
build:
	docker build -t $(IMAGE_NAME) .

# Run the Docker container
run: clean
	docker run -d $(GPU_FLAG) \
	--shm-size 16G \
	-v $(REPO_ROOT):/app \
	-v $(DATA_DIR):/app/data \
	-v $(YV5_DIR):/opt/yolov5 \
	-e PYTHONPATH=/app/scripts \
	--name $(CONTAINER_NAME) \
	$(IMAGE_NAME) tail -f /dev/null
	docker exec -it $(CONTAINER_NAME) /bin/bash

# Stop and remove the Docker container if it exists
stop:
	-@docker rm -f $(CONTAINER_NAME) 2>/dev/null || true

# Alias for stop (kept for backwards compatibility)
clean: stop

# Remove the Docker image if needed
clean-image:
	-@docker rmi $(IMAGE_NAME) 2>/dev/null || true

# ─── Training: yolov5s ────────────────────────────────────────────────────────

# Run the yolov5s fine-tuning pipeline inside the training Docker container.
# Requires scripts/training/yolov5s/.env (MLflow credentials) to exist.
# Extra CLI flags via YOLOV5S_ARGS, e.g.:
#   make yolov5s-train YOLOV5S_ARGS="--resume-from scripts/training/yolov5s/model_exports/<run>/last.pt"
YOLOV5S_ARGS ?=
yolov5s-train:
	set -a && . scripts/training/yolov5s/.env && set +a && \
	PYTHONPATH=/app uv run python -m scripts.training.yolov5s.run_training_pipeline $(YOLOV5S_ARGS)

# ─── Dependencies ─────────────────────────────────────────────────────────────

# Pin all dependencies to uv.lock (commit for reproducibility)
uv-lock:
	uv lock

# ─── Sync ─────────────────────────────────────────────────────────────────────

# Remote server configuration
REMOTE_HOST := gpu.local
REMOTE_PATH := ~/Master-Thesis/

# NAS backup (RAID 1, always-on) — prerequisite: ssh-copy-id debian@data-server.taile550ef.ts.net
NAS_HOST := debian@data-server.taile550ef.ts.net
NAS_PATH  := /data/raid1_6t/Master-Thesis/

# ics-server — second repo instance (repo already cloned, only missing gitignored files)
# prerequisite: ssh-copy-id ubuntu@thesis.taile550ef.ts.net
ICS_HOST := ubuntu@thesis.taile550ef.ts.net
ICS_PATH  := /home/ubuntu/Master-Thesis/

# Set to true to skip files already present on the remote, false to overwrite
IGNORE_EXISTING := true

# Resolve rsync flag
ifeq ($(IGNORE_EXISTING),true)
  _IGNORE_FLAG := --ignore-existing
else
  _IGNORE_FLAG :=
endif

_RSYNC := rsync -avh --progress $(_IGNORE_FLAG)

# Sync large files excluded by .gitignore to the remote server.
# Add new paths here as additional rsync lines.
sync:
# 	$(_RSYNC) data/ $(REMOTE_HOST):$(REMOTE_PATH)data/
# 	$(_RSYNC) resources/SNPredictions_all* $(REMOTE_HOST):$(REMOTE_PATH)resources/
	$(_RSYNC) reports/wikimedia_categories/* $(REMOTE_HOST):$(REMOTE_PATH)reports/wikimedia_categories/

# Sync all gitignored files to the remote server.
# Uses git to discover ignored files, then rsyncs each one preserving directory structure.
sync-ignored:
	@git ls-files --others --ignored --exclude-standard | rsync -avh --progress --ignore-existing --files-from=- . $(REMOTE_HOST):$(REMOTE_PATH)

# Populate gitignored files on ics-server (repo already cloned there).
# Skips existing files; safe to re-run. Excludes data/ — use sync-ics-data for that.
sync-ics:
	@git ls-files --others --ignored --exclude-standard \
	  | grep -v '^data/' \
	  | rsync -avh --progress --ignore-existing --inplace \
	      --files-from=- . $(ICS_HOST):$(ICS_PATH)

# Sync the 186 GB dataset to ics-server. Safe to interrupt and resume.
sync-ics-data:
	rsync -a --no-i-r --no-owner --no-group --info=progress2 --ignore-existing --inplace \
	  /home/debian/Master-Thesis/data/ $(ICS_HOST):$(ICS_PATH)data/


# ─── Disk ─────────────────────────────────────────────────────────────────────

df:
	df -h .

ncdu:
	ncdu
