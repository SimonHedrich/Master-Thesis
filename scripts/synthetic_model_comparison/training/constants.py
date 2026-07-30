"""Single source of truth for all paths and hyperparameters.

Self-contained package for the synthetic-generator comparison experiment
(docs/synthetic-model-comparison/) — copied and adapted from
scripts/training/yolo26n/constants.py, not imported, so this experiment's
code never silently drifts if the main 225-class pipeline changes (see
docs/synthetic-model-comparison/11_detector-architecture-selection.md for
why YOLO26n was chosen as the fixed detector here).

Differences from scripts/training/yolo26n/constants.py:
  - NUM_CLASSES = 12, not 225.
  - No fixed ANNOTATIONS_TRAIN/VAL — this experiment has one train set per
    generator x prompt-regime "cell", not a single fixed dataset. The cell's
    full annotations.json (produced by
    scripts/synthetic_model_comparison/5-export_coco.py) is split into an
    internal train/val pair by split_dataset.py at runtime (see VAL_FRACTION/
    SPLIT_SEED below); run_training_pipeline.py resolves the cell path from
    its --generator/--prompt-regime CLI args.
  - ANNOTATIONS_TEST points at the experiment's fixed REAL test set
    (data/synthetic_model_comparison/test/annotations_test.json), evaluated
    exactly once per run, at the end — never used for early-stopping/model
    selection (see docs/synthetic-model-comparison/11_detector-architecture-selection.md
    §7: evaluating the full ~9,742-image real test set every epoch would
    dominate wall-clock for no benefit).
  - MLFLOW_EXPERIMENT_DEFAULT is this experiment's own name.
  - No KD-related constants — this experiment is direct-fine-tune-only
    (comparing generators, not distilling), so KD_TEMPERATURE/KD_ALPHA/
    KD_APPLY_TO are dropped entirely along with the --kd training mode.
  - Optimizer/scheduler/augmentation/early-stop values are otherwise the same
    as yolo26n/constants.py — no reason to re-litigate the main pipeline's
    own tuning choices for this side experiment.
"""
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = REPO_ROOT / "data"

EXPERIMENT_DATA_ROOT = DATA_ROOT / "synthetic_model_comparison"
TRAIN_ROOT = EXPERIMENT_DATA_ROOT / "train"
ANNOTATIONS_TEST = EXPERIMENT_DATA_ROOT / "test" / "annotations_test.json"

# `file_name` fields in the COCO JSONs start with `data/...` and resolve
# relative to the repo root.
IMAGE_ROOT = REPO_ROOT

PRETRAINED_WEIGHTS = REPO_ROOT / "weights" / "yolo26n.pt"
# Base directory holding one timestamped sub-directory per training run
# (named after the run, e.g.
# ``yolo26n-gemini-3-1-flash-image-preview-full-seed42-20260721-233434/``).
OUTPUT_DIR = REPO_ROOT / "scripts" / "synthetic_model_comparison" / "training" / "model_exports"


def latest_run_dir() -> "Path | None":
    """Most recent training-run dir under ``OUTPUT_DIR`` (one containing
    ``best.pt``), or ``None`` if there are none."""
    if not OUTPUT_DIR.exists():
        return None
    runs = [d for d in OUTPUT_DIR.iterdir() if d.is_dir() and (d / "best.pt").exists()]
    return max(runs, key=lambda d: d.stat().st_mtime, default=None)


def cell_dir(generator: str, prompt_regime: str) -> Path:
    """Directory holding one generator x prompt-regime cell's data."""
    return TRAIN_ROOT / generator / prompt_regime


# ─── Model ────────────────────────────────────────────────────────────────────

NUM_CLASSES = 12
IMAGE_SIZE = 640
MODEL_CONFIG = "yolo26n.yaml"  # ships inside the ultralytics PyPI package

# ─── Internal train/val split (this experiment has no separate val set) ─────
# split_dataset.py carves this fraction off each cell's exported
# annotations.json, stratified per class, using SPLIT_SEED — independent of
# the training-run SEED below, so the split stays identical across the >=3
# training seeds recommended per cell
# (docs/synthetic-model-comparison/06_evaluation-methodology.md).

VAL_FRACTION = 0.2
SPLIT_SEED = 42

# ─── Optimizer (SGD, kept identical to yolo26n's own choices) ────────────────

OPTIMIZER = "SGD"  # alt: "AdamW"
LEARNING_RATE = 1e-3  # lr0 for fine-tuning (1/10 of training-from-scratch)
MOMENTUM = 0.937
WEIGHT_DECAY = 5e-4
NESTEROV = True

# ─── Data / loop ──────────────────────────────────────────────────────────────

EPOCH_COUNT = 200  # safety ceiling only — early stopping is expected to end the run
BATCH_SIZE = 64  # re-run find_max_batch_size.py if this doesn't fit the training GPU
NUM_WORKERS = 8
SEED = 42  # training-run seed — override per run via --seed for the >=3-seeds recommendation

# ─── Scheduler (OneCycleLR) ────────────────────────────────────────────────────
# model_optimizer/model_scheduler (optim.py) read these directly (unlike
# yolo26n's package, which re-exports them from yolov5s_model.py — this
# package doesn't import from either main-pipeline package at all).

WARMUP_EPOCHS = 3  # early-stop patience gating; also sets ONE_CYCLE_PCT_START
ONE_CYCLE_MAX_LR = 1e-2
ONE_CYCLE_PCT_START = WARMUP_EPOCHS / EPOCH_COUNT
ONE_CYCLE_DIV_FACTOR = 10.0
ONE_CYCLE_FINAL_DIV_FACTOR = 100.0

# ─── Loss hyperparameters (Ultralytics' own defaults, same as yolo26n) ───────
# YOLO26 (yolo26.yaml: end2end=True, reg_max=1) is anchor-free and NMS-free.

HYP_BOX = 7.5  # box loss gain
HYP_CLS = 0.5  # cls loss gain
HYP_DFL = 1.5  # distribution focal loss gain (numerically near-inert: reg_max=1 disables DFL)

# ─── Auto-stop + selection metric ──────────────────────────────────────────────

SELECTION_METRIC = "mAP50_95"
EARLY_STOP = True
EARLY_STOP_PATIENCE = 20
EARLY_STOP_MIN_DELTA = 0.001

# ─── Training-quality add-ons ──────────────────────────────────────────────────

USE_EMA = True
USE_AMP = True

# ─── Basic shared augmentation (same defaults as yolo26n) ────────────────────

AUG_HFLIP = True
AUG_HSV = True
AUG_HSV_H = 0.015
AUG_HSV_S = 0.7
AUG_HSV_V = 0.4
AUG_SCALE = 0.5
AUG_TRANSLATE = 0.1
AUG_DEGREES = 0.0
AUG_SHEAR = 0.0
AUG_PERSPECTIVE = 0.0
AUG_FLIPUD = 0.0

# ─── Compositing augmentation — off by default, same as yolo26n ─────────────

AUG_MOSAIC = 0.0
AUG_MIXUP = 0.0
AUG_COPY_PASTE = 0.0
AUG_CLOSE_MOSAIC = 0

# ─── Evaluation ───────────────────────────────────────────────────────────────

EVAL_CONF_THRES = 0.001
EVAL_IOU_THRES = 0.6
EVAL_MAX_DET = 100

# ─── MLflow ───────────────────────────────────────────────────────────────────

MLFLOW_EXPERIMENT_DEFAULT = "yolo26n-synthetic-model-comparison"
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
