REPO_ROOT := $(shell git rev-parse --show-toplevel)
DATA_DIR  := $(REPO_ROOT)/data

# Overridable: select which stage/image/container to work with.
# Stages defined in Dockerfile: main, speciesnet, yolov5
#
#   make build TARGET=speciesnet IMAGE=wildlife-speciesnet
#   make run   IMAGE=wildlife-speciesnet
#   make build TARGET=yolov5    IMAGE=wildlife-yolov5
#   make run   IMAGE=wildlife-yolov5
IMAGE     ?= wildlife-training
TARGET    ?= main
CONTAINER ?= wildlife-train

# YOLOv5 source directory (bind-mounted into the yolov5 container)
YV5_DIR ?= /opt/yolov5

# GPU flag — set to empty to disable GPU access (e.g. for dataset prep without CUDA):
#   make run GPUS=
GPUS ?= all

# ─── Docker ───────────────────────────────────────────────────────────────────

build:
	docker build --target $(TARGET) -t $(IMAGE) .

# Open an interactive shell inside the container.
# The full repo and dataset are mounted; training scripts are at /app/scripts.
# Run training commands from inside: make -f /app/scripts/training/Makefile <target>
run:
	docker run --rm -it \
	  $(if $(GPUS),--gpus $(GPUS)) \
	  --shm-size=8g \
	  -v $(REPO_ROOT):/app \
	  -v $(DATA_DIR):/app/data \
	  -v $(YV5_DIR):/opt/yolov5 \
	  -e PYTHONPATH=/app/scripts \
	  -w /app \
	  $(IMAGE) bash

stop:
	docker stop $(CONTAINER) && docker rm $(CONTAINER)

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

# ─── Backup ───────────────────────────────────────────────────────────────────

# Backup this directory to the USB HDD (incremental, mirrors source exactly)
# Mount:   sudo mount /dev/sda2 /mnt/backup-hdd
# Unmount: sudo umount /mnt/backup-hdd
# Excludes /data/ (186 GB of publicly re-downloadable dataset images) for speed.
# Run 'make backup-hdd-data' separately for a one-time dataset backup.
backup-hdd:
	rsync -a --no-i-r --no-owner --no-group --info=progress2 \
	  --exclude='/data/' \
	  /home/debian/Master-Thesis/ /mnt/backup-hdd/Master-Thesis/

# One-time dataset backup to HDD — safe to interrupt and resume.
# Uses --inplace and --ignore-existing since dataset images never change.
backup-hdd-data:
	rsync -a --no-i-r --no-owner --no-group --info=progress2 --inplace --ignore-existing \
	  /home/debian/Master-Thesis/data/ /mnt/backup-hdd/Master-Thesis/data/

# Backup this directory to the USB SSD (incremental, mirrors source exactly)
# Mount:   sudo mount /dev/sdb2 /mnt/backup-ssd
# Unmount: sudo umount /mnt/backup-ssd
# Excludes /data/ (186 GB of publicly re-downloadable dataset images) for speed.
# Run 'make backup-ssd-data' separately for a one-time dataset backup.
backup-ssd:
	rsync -a --no-i-r --no-owner --no-group --info=progress2 \
	  --exclude='/data/' \
	  /home/debian/Master-Thesis/ /mnt/backup-ssd/Master-Thesis/

# One-time dataset backup to SSD — safe to interrupt and resume.
# Uses --inplace and --ignore-existing since dataset images never change.
backup-ssd-data:
	rsync -a --no-i-r --no-owner --no-group --info=progress2 --inplace --ignore-existing \
	  /home/debian/Master-Thesis/data/ /mnt/backup-ssd/Master-Thesis/data/

# Backup code/docs/scripts/models to NAS RAID 1 (excludes 186 GB dataset)
# ~90-100 MB/s over Gigabit LAN; RAID 1 survives single drive failure
backup-nas:
	rsync -a --no-i-r --no-owner --no-group --info=progress2 --delete \
	  --exclude='/data/' \
	  /home/debian/Master-Thesis/ $(NAS_HOST):$(NAS_PATH)

# One-time dataset backup to NAS — safe to interrupt and resume
backup-nas-data:
	rsync -a --no-i-r --no-owner --no-group --info=progress2 --inplace --ignore-existing \
	  /home/debian/Master-Thesis/data/ $(NAS_HOST):$(NAS_PATH)data/

# Quick size check — safe to run while backup is in progress
backup-status:
	@echo "--- HDD (/mnt/backup-hdd)     ---" && df -h /mnt/backup-hdd
	@echo "--- SSD (/mnt/backup-ssd) ---" && df -h /mnt/backup-ssd
	@echo "--- Source size           ---" && du -sh /home/debian/Master-Thesis
	@echo "--- HDD backup size       ---" && du -sh /mnt/backup-hdd/Master-Thesis 2>/dev/null || echo "(not found)"
	@echo "--- SSD backup size       ---" && du -sh /mnt/backup-ssd/Master-Thesis 2>/dev/null || echo "(not found)"

# ─── Disk ─────────────────────────────────────────────────────────────────────

df:
	df -h .

ncdu:
	ncdu
