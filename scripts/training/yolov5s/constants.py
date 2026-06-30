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
# Base directory holding one timestamped sub-directory per training run
# (named after the run, e.g. ``yolov5s-20260602-233434/``). Checkpoints, the
# training log, and eval reports for a run all live inside its run dir.
OUTPUT_DIR = REPO_ROOT / "scripts" / "training" / "yolov5s" / "model_exports"


def latest_run_dir() -> "Path | None":
    """Most recent training-run dir under ``OUTPUT_DIR`` (one containing
    ``best.pt``), or ``None`` if there are none.

    "Most recent" is the directory with the largest mtime — robust to the
    ``smoke-`` prefix breaking lexicographic ordering, and the ``best.pt``
    filter skips smoke-only / log-only / half-finished run dirs.
    """
    if not OUTPUT_DIR.exists():
        return None
    runs = [d for d in OUTPUT_DIR.iterdir() if d.is_dir() and (d / "best.pt").exists()]
    return max(runs, key=lambda d: d.stat().st_mtime, default=None)

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

# ─── Scheduler (linear warmup → ReduceLROnPlateau) ─────────────────────────────
# Horizon-agnostic by design so it composes with early stopping: the same
# validation metric (SELECTION_METRIC) drives the LR drops and the auto-stop.
# See docs/plans/2026-06-10_training-hyperparameters-autostop-lr-schedule.md §4.

WARMUP_EPOCHS = 3  # linear lr warmup (lr0 * (epoch+1)/WARMUP_EPOCHS), then plateau takes over
PLATEAU_FACTOR = 0.5  # multiply lr by this when the metric plateaus
PLATEAU_PATIENCE = 5  # epochs without improvement before an lr drop (< EARLY_STOP_PATIENCE)
PLATEAU_MIN_LR = 1e-5  # lower bound on lr (≈ lr0 * 0.01)

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

EPOCH_COUNT = 200  # safety ceiling only — early stopping is expected to end the run
BATCH_SIZE = 32
NUM_WORKERS = 8
SEED = 42

# ─── Auto-stop + selection metric ──────────────────────────────────────────────
# One metric drives best-checkpoint selection, plateau LR drops, and early stop
# so all three behaviours read the same signal (§5 of the plan).

SELECTION_METRIC = "mAP50_95"  # one of the keys returned by evaluate(): "mAP50" | "mAP50_95"
EARLY_STOP = True
EARLY_STOP_PATIENCE = 20  # epochs without improvement before stopping (> PLATEAU_PATIENCE)
EARLY_STOP_MIN_DELTA = 0.001  # min metric gain to count as an improvement (filters noise)

# ─── Training-quality add-ons ──────────────────────────────────────────────────

USE_EMA = True  # evaluate/checkpoint an exponential moving average of the weights
USE_AMP = True  # automatic mixed precision (CUDA only; no-op on CPU)

# ─── Basic shared augmentation — setups A & B (single-image, distillation-safe) ──
# Applied identically in the shared dataloader; A and B differ ONLY in the loss.
# The scale/translate transform reuses yolov5.utils.augmentations.random_perspective
# (tested) rather than hand-rolled bbox math, to minimise bug surface.

AUG_HFLIP        = True    # horizontal flip, p=0.5  (bbox x -> 1-x; ~zero risk)
AUG_HSV          = True    # photometric jitter; no bbox math at all
AUG_HSV_H        = 0.015   # hue gain        (YOLOv5 reference default)
AUG_HSV_S        = 0.7     # saturation gain
AUG_HSV_V        = 0.4     # value gain
AUG_SCALE        = 0.5     # random scale gain (±0.5), via random_perspective
AUG_TRANSLATE    = 0.1     # random translation fraction, via random_perspective
AUG_DEGREES      = 0.0     # rotation OFF       — low value for upright animals
AUG_SHEAR        = 0.0     # shear OFF          — bbox-distortion risk
AUG_PERSPECTIVE  = 0.0     # perspective OFF    — bug-prone, low value
AUG_FLIPUD       = 0.0     # vertical flip OFF  — animals are upright

# ─── Compositing augmentation — setup C ONLY (multi-image; breaks distillation) ──
# No valid teacher view of a composite, so these are NEVER enabled in setup A.
# Shared default = OFF (setups A & B). Comments show the setup-C override values.

AUG_MOSAIC       = 0.0     # setup C: 1.0   (4-image stitch)
AUG_MIXUP        = 0.0     # setup C: 0.1   (image+label blend)
AUG_COPY_PASTE   = 0.0     # setup C: 0.0   — needs masks; NO-OP on box-only GT
AUG_CLOSE_MOSAIC = 0       # setup C: 10    — epochs of mosaic-off tail before end

# ─── Evaluation ───────────────────────────────────────────────────────────────

EVAL_CONF_THRES = 0.001
EVAL_IOU_THRES = 0.6
EVAL_MAX_DET = 100  # COCO-standard cap; test sets max out at 28 GT boxes/img, so 100 ≫ enough

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
