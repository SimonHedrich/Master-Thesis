"""Single source of truth for all paths and hyperparameters.

Every knob the training pipeline exposes lives here so a single MLflow
`log_params` call captures the full configuration of a run.
"""
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = REPO_ROOT / "data"

ANNOTATIONS_TRAIN = REPO_ROOT / "data" / "real" / "annotations_train.json"
ANNOTATIONS_VAL = REPO_ROOT / "data" / "real" / "annotations_val.json"
ANNOTATIONS_TEST = REPO_ROOT / "data" / "real" / "annotations_test.json"

# `file_name` fields in the COCO JSONs start with `data/...` and resolve
# relative to the repo root.
IMAGE_ROOT = REPO_ROOT

PRETRAINED_WEIGHTS = REPO_ROOT / "weights" / "yolov5s.pt"
OUTPUT_DIR = REPO_ROOT / "scripts" / "training" / "yolov5s" / "model_exports"

# ─── Model ────────────────────────────────────────────────────────────────────

NUM_CLASSES = 225
IMAGE_SIZE = 640
MODEL_CONFIG = "yolov5s.yaml"  # ships inside the yolov5 PyPI package

# ─── Optimizer (SGD, YOLOv5 reference defaults for fine-tuning) ──────────────

OPTIMIZER = "SGD"  # alt: "AdamW"
LEARNING_RATE = 1e-3  # lr0 for fine-tuning (1/10 of training-from-scratch)
MOMENTUM = 0.937
WEIGHT_DECAY = 5e-4
NESTEROV = True

# ─── Scheduler ────────────────────────────────────────────────────────────────

LRF = 0.01  # final lr fraction (cosine end = lr0 * LRF)
WARMUP_EPOCHS = 3
WARMUP_MOMENTUM = 0.8
WARMUP_BIAS_LR = 0.1

# ─── ComputeLoss hyperparameters ──────────────────────────────────────────────
# Surfaced here so MLflow logs them as run params.

HYP_BOX = 0.05  # box loss gain
HYP_CLS = 0.5  # cls loss gain
HYP_CLS_PW = 1.0  # cls BCELoss positive_weight
HYP_OBJ = 1.0  # obj loss gain
HYP_OBJ_PW = 1.0  # obj BCELoss positive_weight
HYP_IOU_T = 0.20  # IoU training threshold
HYP_ANCHOR_T = 4.0  # anchor-multiple threshold
HYP_FL_GAMMA = 0.0  # focal loss gamma (0 = disabled)
HYP_LABEL_SMOOTHING = 0.0

# ─── Data / loop ──────────────────────────────────────────────────────────────

EPOCH_COUNT = 50
BATCH_SIZE = 16
NUM_WORKERS = 8
SEED = 42

# ─── Augmentation toggles (all off for baseline — only resize is active) ─────

AUG_MOSAIC = False
AUG_HSV = False
AUG_HFLIP = False

# ─── Evaluation ───────────────────────────────────────────────────────────────

EVAL_CONF_THRES = 0.001
EVAL_IOU_THRES = 0.6
EVAL_MAX_DET = 300

# ─── MLflow ───────────────────────────────────────────────────────────────────
# URI / experiment / credentials are loaded from `.env` at run start; these
# are only fallbacks if the env vars are missing.

MLFLOW_EXPERIMENT_DEFAULT = "yolov5s-wildlife225"
MLFLOW_LOG_EVERY_N_STEPS = 50


def as_dict() -> dict[str, object]:
    """All public module-level constants, for `mlflow.log_params`."""
    import sys

    module = sys.modules[__name__]
    return {
        name: getattr(module, name)
        for name in dir(module)
        if name.isupper() and not name.startswith("_")
    }
