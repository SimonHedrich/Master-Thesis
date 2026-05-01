OUTPUT_DIR := output

# ─── Docker training environment ──────────────────────────────────────────────

IMAGE           := wildlife-training
IMAGE_SN        := wildlife-speciesnet
IMAGE_NANODET   := wildlife-nanodet
IMAGE_PADDLE    := wildlife-paddle

CONTAINER       := wildlife-train
CONTAINER_SN    := wildlife-speciesnet-run
CONTAINER_ND    := wildlife-nanodet-run
CONTAINER_PD    := wildlife-paddle-run

REPO_ROOT := $(shell git rev-parse --show-toplevel)
DATA_DIR  := $(REPO_ROOT)/data

# Overridable on the command line:
#   make speciesnet-demo      IMAGE=data/test/deer.jpg
#   make speciesnet-labels    DIR=data/training  OUT=output/kd_soft_labels.jsonl
#   make speciesnet-validate  OUT=output/species_validation.jsonl
#   make speciesnet-classify  SOURCE=gbif
IMAGE  ?= data/test/sample.jpg
OUT    ?= output/teacher_labels.jsonl
DIR    ?= data/training
SOURCE ?= all

# ─── Main Docker image (YOLO family — Python 3.13) ───────────────────────────

# Build the GPU-enabled main Docker image
docker-build:
	docker build -t $(IMAGE) .

# Interactive shell inside the container with GPU and full repo mounted
docker-shell:
	docker run --rm -it \
	  --gpus all \
	  --shm-size=8g \
	  -v $(REPO_ROOT):/app \
	  -v $(DATA_DIR):/app/data \
	  -e PYTHONPATH=/app/scripts \
	  $(IMAGE) bash

# Run model smoke test on the host (uses system-wide Python packages)
smoke:
	python $(REPO_ROOT)/scripts/training/1-smoke_test_models.py

# Run model smoke test inside Docker with GPU access
smoke-docker:
	docker run --rm \
	  --gpus all \
	  --shm-size=8g \
	  -v $(REPO_ROOT):/app \
	  -e PYTHONPATH=/app/scripts \
	  $(IMAGE) \
	  python /app/scripts/training/1-smoke_test_models.py

# Start long-running teacher training in a detached container
docker-train:
	@mkdir -p $(REPO_ROOT)/output
	docker run -d --name $(CONTAINER) \
	  --gpus all \
	  --shm-size=8g \
	  -v $(REPO_ROOT):/app \
	  -v $(DATA_DIR):/app/data \
	  -e PYTHONPATH=/app/scripts \
	  $(IMAGE) \
	  bash -c 'python scripts/training/2-train_teacher.py 2>&1 | tee /app/output/train.log'
	@echo "Training started in container '$(CONTAINER)'. Run 'make docker-logs' to follow."

docker-logs:
	docker logs -f $(CONTAINER)

docker-stop:
	docker stop $(CONTAINER) && docker rm $(CONTAINER)

docker-attach:
	docker exec -it $(CONTAINER) bash

# Pin all dependencies to uv.lock (commit for reproducibility)
uv-lock:
	uv lock

# ─── SpeciesNet teacher pipeline (Python 3.11, Dockerfile.speciesnet) ─────────

speciesnet-build:
	docker build -t $(IMAGE_SN) -f Dockerfile.speciesnet .

# Verify the full SpeciesNet two-stage pipeline loads and runs
speciesnet-smoke:
	docker run --rm \
	  --gpus all \
	  --shm-size=8g \
	  -v $(REPO_ROOT):/app \
	  -e PYTHONPATH=/app/scripts \
	  $(IMAGE_SN) \
	  python /app/scripts/training/0-teacher_speciesnet_pipeline.py --smoke-test

# Demo: run pipeline on a single image (prints JSON to stdout)
# Usage: make speciesnet-demo IMAGE=path/to/image.jpg
speciesnet-demo:
	docker run --rm \
	  --gpus all \
	  --shm-size=8g \
	  -v $(REPO_ROOT):/app \
	  -e PYTHONPATH=/app/scripts \
	  $(IMAGE_SN) \
	  python /app/scripts/training/0-teacher_speciesnet_pipeline.py --image /app/$(IMAGE)

# Batch soft-label generation (writes JSONL for use as KD targets)
# Usage: make speciesnet-labels DIR=data/training OUT=output/kd_soft_labels.jsonl
speciesnet-labels:
	@mkdir -p $(REPO_ROOT)/output
	docker run --rm \
	  --gpus all \
	  --shm-size=8g \
	  -v $(REPO_ROOT):/app \
	  -e PYTHONPATH=/app/scripts \
	  $(IMAGE_SN) \
	  python /app/scripts/training/0-teacher_speciesnet_pipeline.py \
	    --dir /app/$(DIR) \
	    --output /app/$(OUT) \
	    --classes /app/resources/2026-03-19_student_model_labels.txt \
	    --soft-labels

# Usage: make speciesnet-validate OUT=output/species_validation.jsonl
speciesnet-validate:
	@mkdir -p $(REPO_ROOT)/output
	docker run --rm \
	  --gpus all \
	  --shm-size=8g \
	  -v $(REPO_ROOT):/app \
	  -e PYTHONPATH=/app/scripts \
	  $(IMAGE_SN) \
	  python /app/scripts/training/0-teacher_speciesnet_pipeline.py \
	    --validate-labels \
	    --classes /app/resources/2026-03-19_student_model_labels.txt \
	    --output /app/$(or $(OUT),output/species_validation.jsonl)

# Start a persistent SpeciesNet container and exec into its shell.
# The container keeps running after you exit so it can be re-entered later.
speciesnet-start:
	docker run -d --name $(CONTAINER_SN) \
	  --gpus all \
	  --shm-size=8g \
	  -v $(REPO_ROOT):/app \
	  -e PYTHONPATH=/app/scripts \
	  $(IMAGE_SN) sleep infinity
	docker exec -it $(CONTAINER_SN) bash

# Stop and remove the persistent SpeciesNet container.
speciesnet-stop:
	docker stop $(CONTAINER_SN) && docker rm $(CONTAINER_SN)

# Run SpeciesNet classification inside the running container with nohup.
# Output is logged to output/speciesnet_classify_<SOURCE>.log on the host.
# Requires: make speciesnet-start first.
# Usage: make speciesnet-classify SOURCE=gbif
#        make speciesnet-classify SOURCE=all
speciesnet-classify:
	@mkdir -p $(REPO_ROOT)/output
	docker exec $(CONTAINER_SN) bash -c \
	  'nohup python /app/scripts/dataset_quality/6-classify_speciesnet.py \
	    --source $(SOURCE) \
	    > /app/output/speciesnet_classify_$(SOURCE).log 2>&1 &'
	@echo "Classification started — follow with: make speciesnet-classify-logs SOURCE=$(SOURCE)"

# Tail the classification log from inside the container.
# Usage: make speciesnet-classify-logs SOURCE=gbif
speciesnet-classify-logs:
	docker exec -it $(CONTAINER_SN) tail -f /app/output/speciesnet_classify_$(SOURCE).log

speciesnet-shell:
	docker run --rm -it \
	  --gpus all \
	  --shm-size=8g \
	  -v $(REPO_ROOT):/app \
	  -e PYTHONPATH=/app/scripts \
	  $(IMAGE_SN) bash

# ─── NanoDet-Plus-m (Python 3.11, Dockerfile.nanodet) ────────────────────────

nanodet-build:
	docker build -t $(IMAGE_NANODET) -f Dockerfile.nanodet .

# Run NanoDet forward-pass smoke test
nanodet-smoke:
	docker run --rm \
	  --gpus all \
	  --shm-size=8g \
	  -v $(REPO_ROOT):/app \
	  -e PYTHONPATH=/app/scripts:/opt/nanodet \
	  $(IMAGE_NANODET) \
	  python /app/scripts/training/nanodet_smoke_test.py

# Interactive shell for NanoDet training
# Training: cd /opt/nanodet && python tools/train.py /app/scripts/training/configs/nanodet-plus-m-wildlife225.yml
nanodet-shell:
	docker run --rm -it \
	  --gpus all \
	  --shm-size=8g \
	  -v $(REPO_ROOT):/app \
	  -v $(DATA_DIR):/app/data \
	  -e PYTHONPATH=/app/scripts:/opt/nanodet \
	  $(IMAGE_NANODET) bash

# ─── PicoDet-S (PaddlePaddle, Dockerfile.paddle) ──────────────────────────────

paddle-build:
	docker build -t $(IMAGE_PADDLE) -f Dockerfile.paddle .

# Run PicoDet forward-pass smoke test
paddle-smoke:
	docker run --rm \
	  --gpus all \
	  --shm-size=8g \
	  -v $(REPO_ROOT):/app \
	  -e PYTHONPATH=/app/scripts:/opt/PaddleDetection \
	  $(IMAGE_PADDLE) \
	  python /app/scripts/training/picodet_smoke_test.py

# Interactive shell for PicoDet training
# Training: cd /opt/PaddleDetection && python tools/train.py -c /app/scripts/training/configs/picodet-s-wildlife225.yml
paddle-shell:
	docker run --rm -it \
	  --gpus all \
	  --shm-size=8g \
	  -v $(REPO_ROOT):/app \
	  -v $(DATA_DIR):/app/data \
	  -e PYTHONPATH=/app/scripts:/opt/PaddleDetection \
	  $(IMAGE_PADDLE) bash

# ─── General utilities ────────────────────────────────────────────────────────

# Run a Python script with nohup, logging to output/<timestamp>_<basename>.txt
# Usage: make run SCRIPT=scripts/filter_wikimedia_categories.py
run:
	@mkdir -p $(OUTPUT_DIR)
	@ts=$$(date +%Y%m%d_%H%M%S); \
	base=$$(basename $(SCRIPT) .py); \
	logfile=$(OUTPUT_DIR)/$${ts}_$${base}.txt; \
	echo "Logging to $$logfile"; \
	nohup python $(SCRIPT) 2>&1 | tee "$$logfile" &

# Remote server configuration
# REMOTE_HOST := gpu-server.taile550ef.ts.net
REMOTE_HOST := gpu.local
REMOTE_PATH := ~/Master-Thesis/

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

# Show available disk space
df:
	df -h .

ncdu:
	ncdu

# ── Image quality filtering ───────────────────────────────────────────────────
# Run each target in sequence: metadata → heuristics → megadetector → report
# Override SOURCE to filter a single source, e.g.: make filter-metadata SOURCE=wikimedia

SOURCE ?= all

filter-metadata:
	python scripts/filter_dataset_quality.py metadata --source $(SOURCE)

filter-heuristics:
	python scripts/filter_dataset_quality.py heuristics --source $(SOURCE)

filter-megadetector:
	python scripts/filter_dataset_quality.py megadetector --source $(SOURCE)

filter-vlm:
	python scripts/filter_dataset_quality.py vlm --source wikimedia

filter-report:
	python scripts/filter_dataset_quality.py report --source all