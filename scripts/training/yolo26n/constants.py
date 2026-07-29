"""Single source of truth for all paths and hyperparameters.

Every knob the training pipeline exposes lives here so a single MLflow
`log_params` call captures the full configuration of a run.

Kept byte-identical to `scripts/training/yolov5s/constants.py` wherever the
comparability contract requires it (see
docs/plans/2026-06-30_yolo26-kd-and-teacher-finetune-implementation-plan.md §0):
data paths/splits, seed, optimizer family, LR schedule, early-stop, every
`AUG_*` value, eval thresholds. Two things are *intentionally* exempted —
`BATCH_SIZE` and the `HYP_*` loss gains — see the comments at each below.
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

PRETRAINED_WEIGHTS = REPO_ROOT / "weights" / "yolo26n.pt"
# Base directory holding one timestamped sub-directory per training run
# (named after the run, e.g. ``yolo26n-20260701-233434/``). Checkpoints, the
# training log, and eval reports for a run all live inside its run dir.
OUTPUT_DIR = REPO_ROOT / "scripts" / "training" / "yolo26n" / "model_exports"


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
MODEL_CONFIG = "yolo26n.yaml"  # ships inside the ultralytics PyPI package

# ─── Optimizer (SGD, kept identical to yolov5s for comparability) ────────────

OPTIMIZER = "SGD"  # alt: "AdamW"
LEARNING_RATE = 1e-3  # lr0 for fine-tuning (1/10 of training-from-scratch)
MOMENTUM = 0.937
WEIGHT_DECAY = 5e-4
NESTEROV = True

# ─── Data / loop ──────────────────────────────────────────────────────────────

EPOCH_COUNT = 200  # safety ceiling only — early stopping is expected to end the run
# NOT required to match yolov5s' 32 — VRAM footprint differs by architecture
# (docs/2026-04-29_gpu_training_options.md). Run find_max_batch_size.py and set
# this to the largest power-of-two that fits before a real (non-smoke) run.
BATCH_SIZE = 64 # on ics
NUM_WORKERS = 8
SEED = 42

# ─── Scheduler (OneCycleLR, kept identical to yolov5s for comparability) ─────
# Warmup → peak → cosine annealing, stepped every batch. model_optimizer/
# model_scheduler are re-exported from yolov5s_model.py (see yolo26n_model.py),
# so the values actually applied come from yolov5s.constants, not these — these
# are mirrored here (like LEARNING_RATE/MOMENTUM above) so as_dict() logs the
# real schedule to MLflow. Keep in sync with yolov5s/constants.py.

WARMUP_EPOCHS = 3  # early-stop patience gating; also sets ONE_CYCLE_PCT_START
ONE_CYCLE_MAX_LR = 1e-2          # peak LR (10× LEARNING_RATE; super-convergence)
ONE_CYCLE_PCT_START = WARMUP_EPOCHS / EPOCH_COUNT  # warmup fraction ≈ 0.015
ONE_CYCLE_DIV_FACTOR = 10.0      # initial_lr = max_lr / div_factor = LEARNING_RATE
ONE_CYCLE_FINAL_DIV_FACTOR = 100.0  # min_lr = initial_lr / final_div_factor = 1e-5

# ─── Loss hyperparameters (Ultralytics' own defaults — NOT yolov5s' HYP_* values) ──
# YOLO26 (yolo26.yaml: end2end=True, reg_max=1) is anchor-free and NMS-free:
# TaskAlignedAssigner replaces anchor matching, and there is no objectness term,
# so yolov5s' HYP_CLS_PW/HYP_OBJ/HYP_OBJ_PW/HYP_IOU_T/HYP_ANCHOR_T/HYP_FL_GAMMA/
# HYP_LABEL_SMOOTHING have no equivalent here. Reusing yolov5s' anchor-based
# gains (HYP_BOX=0.05 etc.) on this structurally different loss would not make
# the comparison more rigorous — it would silently mis-scale it. These are
# Ultralytics' own calibrated defaults (ultralytics/cfg/default.yaml: box=7.5,
# cls=0.5, dfl=1.5), used as-is per the source plan's §0 exemption.
HYP_BOX = 7.5  # box loss gain
HYP_CLS = 0.5  # cls loss gain
HYP_DFL = 1.5  # distribution focal loss gain (numerically near-inert: reg_max=1 disables DFL)

# ─── Auto-stop + selection metric ──────────────────────────────────────────────
# One metric drives best-checkpoint selection and early stop (§5 of the plan).

SELECTION_METRIC = "mAP50_95"  # one of the keys returned by evaluate(): "mAP50" | "mAP50_95"
EARLY_STOP = True
EARLY_STOP_PATIENCE = 20  # epochs without improvement before stopping
EARLY_STOP_MIN_DELTA = 0.001  # min metric gain to count as an improvement (filters noise)

# ─── Training-quality add-ons ──────────────────────────────────────────────────

USE_EMA = True  # evaluate/checkpoint an exponential moving average of the weights
USE_AMP = True  # automatic mixed precision (CUDA only; no-op on CPU)

# ─── Basic shared augmentation — setups A & B (single-image, distillation-safe) ──
# Applied identically in the shared dataloader (scripts.training.yolov5s.dataset/
# transforms, imported unchanged — see yolo26n/README.md). A and B differ ONLY
# in the loss.

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
# CONF_THRES/IOU_THRES are kept for signature parity with yolov5s' evaluate()
# call site (training_pipeline.py passes them positionally regardless of which
# package's evaluation.py is in scope) but are UNUSED by yolo26n's decode path:
# yolo26n's Detect head (end2end=True) is NMS-free — postprocess() already does
# score/top-k filtering with no separate IoU-suppression or confidence-threshold
# knob exposed at this layer, only EVAL_MAX_DET (wired via Detect.max_det).

EVAL_CONF_THRES = 0.001
EVAL_IOU_THRES = 0.6
EVAL_MAX_DET = 100  # COCO-standard cap; test sets max out at 28 GT boxes/img, so 100 ≫ enough

# ─── MLflow ───────────────────────────────────────────────────────────────────
# URI / experiment / credentials are loaded from `.env` at run start; these
# are only fallbacks if the env vars are missing.

MLFLOW_EXPERIMENT_DEFAULT = "yolo26n-wildlife225"
MLFLOW_LOG_EVERY_N_STEPS = 50

# ─── Knowledge distillation (Goal B; frozen Goal C teacher soft-label cache) ──
# Only used when run_training_pipeline.py --kd is set; zero effect on the
# direct-FT path. See kd_dataset.py / kd_loss.py and
# docs/plans/2026-06-30_yolo26-kd-and-teacher-finetune-implementation-plan.md §3.

KD_TEMPERATURE = 4.0  # sweep target: {4, 8} per the strategy doc's starting grid
KD_ALPHA = 0.5  # sweep target: {0.5, 0.7} — weight on the teacher distribution in the assigned-target blend
# Which E2ELoss head(s) receive the KD blend. "one2one" is the head actually
# used at inference (§3.4 of the plan doc); "one2many"/"both" are a one-line
# change here for a future ablation, not exposed as a CLI flag.
KD_APPLY_TO = "one2one"  # one of: "one2one" | "one2many" | "both"


def as_dict() -> dict[str, object]:
    """All public module-level constants, for `mlflow.log_params`."""
    import sys

    module = sys.modules[__name__]
    return {
        name: getattr(module, name)
        for name in dir(module)
        if name.isupper() and not name.startswith("_")
    }
