# Training commands — run these INSIDE the container.
#
# Start a shell first (from repo root):
#   make run                            # main environment (wildlife-training)
#   make run IMAGE=wildlife-speciesnet  # speciesnet environment
#   make run IMAGE=wildlife-yolov5      # yolov5 environment
#
# Then run targets here:
#   make -f /app/scripts/training/Makefile <target>
#
# Or cd into the directory and just: make <target>

# ─── Defaults (override on the command line) ──────────────────────────────────

DIR      ?= data/training
OUT      ?= output/teacher_labels.jsonl
SOURCE   ?= all
YV5_SEED ?= 42

# ─── Smoke tests ──────────────────────────────────────────────────────────────

smoke:
	python /app/scripts/training/1-smoke_test_models.py

# ─── SpeciesNet teacher pipeline ──────────────────────────────────────────────

speciesnet-smoke:
	python /app/scripts/training/0-teacher_speciesnet_pipeline.py --smoke-test

# Usage: make speciesnet-demo IMG=data/test/deer.jpg
speciesnet-demo:
	python /app/scripts/training/0-teacher_speciesnet_pipeline.py --image /app/$(IMG)

# Batch soft-label generation — writes JSONL for use as KD targets
# Usage: make speciesnet-labels DIR=data/training OUT=output/kd_soft_labels.jsonl
speciesnet-labels:
	@mkdir -p /app/output
	python /app/scripts/training/0-teacher_speciesnet_pipeline.py \
	  --dir /app/$(DIR) \
	  --output /app/$(OUT) \
	  --classes /app/resources/2026-03-19_student_model_labels.txt \
	  --soft-labels

# Usage: make speciesnet-validate OUT=output/species_validation.jsonl
speciesnet-validate:
	@mkdir -p /app/output
	python /app/scripts/training/0-teacher_speciesnet_pipeline.py \
	  --validate-labels \
	  --classes /app/resources/2026-03-19_student_model_labels.txt \
	  --output /app/$(or $(OUT),output/species_validation.jsonl)

# Usage: make speciesnet-classify SOURCE=gbif
speciesnet-classify:
	@mkdir -p /app/output
	nohup python /app/scripts/dataset_quality/6-classify_speciesnet.py \
	  --source $(SOURCE) \
	  > /app/output/speciesnet_classify_$(SOURCE).log 2>&1 &
	@echo "Classification running in background — tail /app/output/speciesnet_classify_$(SOURCE).log"

# ─── YOLOv5s baseline ─────────────────────────────────────────────────────────
# Prerequisite: yolov5 source mounted at /opt/yolov5 (see Dockerfile yolov5 stage)
#   git clone https://github.com/ultralytics/yolov5.git /opt/yolov5
#   cd /opt/yolov5 && git checkout 5cdad89

# Prepare YOLO TXT dataset from COCO annotations (run once before training)
yolov5-prepare:
	python /app/scripts/training/1-prepare_yolov5_dataset.py

# Full training run — logs to /app/output/yolov5_seed<SEED>.log
# Usage: make yolov5-train
#        make yolov5-train YV5_SEED=1
#        make yolov5-train YV5_SEED=7
yolov5-train:
	@mkdir -p /app/output
	cd /opt/yolov5 && nohup python train.py \
	  --weights yolov5s.pt \
	  --cfg models/yolov5s.yaml \
	  --data /app/data/training/wildlife225_yolov5.yaml \
	  --hyp /app/scripts/training/configs/hyp.finetune-wildlife.yaml \
	  --epochs 150 --batch-size 64 --imgsz 640 \
	  --patience 50 --cos-lr --multi-scale --label-smoothing 0.1 \
	  --project /app/output/yolov5_wildlife \
	  --name yolov5s_wildlife225_seed$(YV5_SEED) \
	  --workers 8 --seed $(YV5_SEED) \
	  > /app/output/yolov5_seed$(YV5_SEED).log 2>&1 &
	@echo "Training running in background — tail /app/output/yolov5_seed$(YV5_SEED).log"

# Evaluate best checkpoint on the test split
# Usage: make yolov5-eval YV5_SEED=42
yolov5-eval:
	cd /opt/yolov5 && python val.py \
	  --weights /app/output/yolov5_wildlife/yolov5s_wildlife225_seed$(YV5_SEED)/weights/best.pt \
	  --data /app/data/training/wildlife225_yolov5.yaml \
	  --imgsz 640 --batch-size 32 --task test --verbose --save-json \
	  --project /app/output/yolov5_wildlife \
	  --name yolov5s_wildlife225_seed$(YV5_SEED)_eval
