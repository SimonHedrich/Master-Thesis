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

# ─── Training: SpeciesNet teacher fine-tune ───────────────────────────────────

# Separate Python 3.11 Docker environment (the `speciesnet` PyPI package
# constrains to Python <3.13) — see Dockerfile.speciesnet and
# scripts/training/teacher_finetune/README.md.
SPECIESNET_IMAGE_NAME = wildlife-speciesnet
SPECIESNET_CONTAINER_NAME = speciesnet-container

# Build the SpeciesNet training image.
speciesnet-build:
	docker build -f Dockerfile.speciesnet -t $(SPECIESNET_IMAGE_NAME) .

# Start a persistent container with the repo/data live-mounted and drop into
# a shell. Requires scripts/training/teacher_finetune/.env (MLflow
# credentials) to exist.
#
# --dns 100.100.100.100 = Tailscale MagicDNS (resolves *.taile550ef.ts.net,
# e.g. the MLflow server); --dns 192.168.178.2 = this host's LAN resolver,
# used as fallback for everything else (e.g. api.kaggle.com) since MagicDNS
# answers SERVFAIL for non-tailnet names on this host (no default route).
speciesnet-start: speciesnet-stop
	docker run -d $(GPU_FLAG) \
	--shm-size 16G \
	--dns 100.100.100.100 --dns 192.168.178.2 \
	-v $(REPO_ROOT):/app \
	-v $(DATA_DIR):/app/data \
	--name $(SPECIESNET_CONTAINER_NAME) \
	$(SPECIESNET_IMAGE_NAME) tail -f /dev/null
	docker exec -it $(SPECIESNET_CONTAINER_NAME) /bin/bash

# Attach another shell to the already-running container.
speciesnet-shell:
	docker exec -it $(SPECIESNET_CONTAINER_NAME) /bin/bash

# Stop and remove the SpeciesNet container if it exists.
speciesnet-stop:
	-@docker rm -f $(SPECIESNET_CONTAINER_NAME) 2>/dev/null || true

# Run the SpeciesNet classifier-head fine-tuning pipeline inside the running
# container (requires `make speciesnet-start` first).
#   make speciesnet-finetune SPECIESNET_ARGS="--resume-from scripts/training/teacher_finetune/model_exports/<run>/last.pt"
SPECIESNET_ARGS ?=
speciesnet-finetune:
	docker exec -it $(SPECIESNET_CONTAINER_NAME) bash -c \
	  "set -a && . scripts/training/teacher_finetune/.env && set +a && cd /app && python -m scripts.training.teacher_finetune.run_finetune $(SPECIESNET_ARGS)"

# ─── Dependencies ─────────────────────────────────────────────────────────────

# Pin all dependencies to uv.lock (commit for reproducibility)
uv-lock:
	uv lock

# ─── Thesis: Overleaf ─────────────────────────────────────────────────────────

# Two-way sync between thesis/manuscript/ and the Overleaf project that compiles
# it. Runs on this host (not in the container) and must run on the
# `thesis/overleaf-sync` branch — the Overleaf git bridge has no branch support,
# so exactly one branch may own the manuscript. Requires OVERLEAF_TOKEN in .env
# and a clone at ~/overleaf/master-thesis (override with OVERLEAF_CLONE).
# See scripts/thesis/README.md and thesis/docs/2026-09-19_overleaf-sync.md.
#
#   make overleaf                                      <- the usual one
#   make overleaf-status
#   make overleaf-pull
#   make overleaf-push OVERLEAF_MSG="rewrote 3.3 intro"
#   make overleaf-force-push                           <- local wins, no questions
#   make overleaf OVERLEAF_ARGS=--allow-any-branch
OVERLEAF_MSG  ?=
OVERLEAF_ARGS ?=
_OVERLEAF := uv run python -m scripts.thesis.sync_overleaf

# Sync, whichever way it needs to go: pulls if Overleaf moved, pushes if the
# manuscript did, does nothing if neither. Stops and explains if BOTH moved,
# rather than guessing which version wins.
overleaf:
	$(_OVERLEAF) sync $(if $(OVERLEAF_MSG),-m "$(OVERLEAF_MSG)") $(OVERLEAF_ARGS)

# Report what each side has. Read-only.
overleaf-status:
	$(_OVERLEAF) status $(OVERLEAF_ARGS)

# Bring Overleaf edits into thesis/manuscript/. Refuses if that tree is dirty;
# leaves what it pulled uncommitted for `git diff` review.
overleaf-pull:
	$(_OVERLEAF) pull $(OVERLEAF_ARGS)

# Send thesis/manuscript/ to Overleaf. Refuses if Overleaf is ahead.
overleaf-push:
	$(_OVERLEAF) push $(if $(OVERLEAF_MSG),-m "$(OVERLEAF_MSG)") $(OVERLEAF_ARGS)

# Same, but the local manuscript always wins: overwrites whatever is on Overleaf,
# including edits made there that were never pulled, and discards leftover changes
# in the clone. Overleaf keeps its history — the overwrite lands as a new commit on
# top, nothing is rewritten — so `make overleaf-pull` on an older repo state can
# still recover what was replaced. Use it when Overleaf-side edits are known junk.
overleaf-force-push:
	$(_OVERLEAF) push --force $(if $(OVERLEAF_MSG),-m "$(OVERLEAF_MSG)") $(OVERLEAF_ARGS)

# ─── Sync ─────────────────────────────────────────────────────────────────────

# Remote server configuration
REMOTE_HOST := gpu.local
REMOTE_PATH := ~/Master-Thesis/

# NAS backup (RAID 1, always-on) — prerequisite: ssh-copy-id debian@data-server.taile550ef.ts.net
NAS_HOST := debian@data-server.taile550ef.ts.net
NAS_PATH  := /data/raid1_6t/Master-Thesis/

# ics-server — second repo instance (repo already cloned, only missing gitignored files)
# prerequisite: ssh-copy-id ubuntu@ics-server.taile550ef.ts.net
# Note: the A40's OS hostname is "thesis", but its Tailscale DNS name is
# "ics-server" (confirmed via `tailscale status --json`) — use the latter.
ICS_HOST := ubuntu@ics-server.taile550ef.ts.net
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
