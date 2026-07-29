"""Single source of truth for all paths and hyperparameters.

Fine-tunes SpeciesNet's classifier head (EfficientNetV2-M, 2,498-class native
taxonomy) on the project's 225-class wildlife dataset. This package deliberately
does **not** mirror `scripts/training/yolov5s/constants.py` byte-for-byte the
way `scripts/training/yolo26n/constants.py` does — SpeciesNet is a classifier,
not a detector, and several of the detector pipelines' constants have no
meaning here. Every deviation is called out below and restated in
`README.md`'s "Deviations from the detector pipelines" section:

- **No `AUG_*` / `HYP_*` constants.** There is no detection loss and no
  bounding-box augmentation. Training uses `SpeciesNetClassifier.preprocess_crop()`
  verbatim (see `dataset.py`) with no extra augmentation, to keep train-time and
  inference-time preprocessing identical (parent strategy doc §2.1).
- **`OPTIMIZER="AdamW"`, `LEARNING_RATE=1e-4`** — not SGD/1e-3. Classifier
  fine-tuning on a pretrained ImageNet/iNat-scale backbone conventionally uses a
  lower AdamW learning rate; the detector pipelines' SGD hyperparameters were
  calibrated for a from-scratch-anchor detection loss and don't transfer here.
- **`IMAGE_SIZE=480`** — SpeciesNet's native classifier input resolution, not
  640 (the detectors' input size).
- **`FREEZE_PARAM_FRACTION`** has no detector analog: the backbone is large
  (EfficientNetV2-M, ~54M params) relative to the 225-class fine-tuning set, so
  a fraction of its parameters (by `named_parameters()` iteration order) are
  frozen to reduce catastrophic forgetting of the native 2,498-class taxonomy.
  See `teacher_model.py` and `README.md` for the rationale and how to change it.
"""
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = REPO_ROOT / "data"

# Reused directly — same file YOLOv5s/YOLO26n train against, not
# filter_results.jsonl (see README.md's "Deviations" section for why: this is
# the downstream, contamination-reviewed, final-label source of truth).
ANNOTATIONS_TRAIN = REPO_ROOT / "data" / "real" / "annotations_train.json"
ANNOTATIONS_VAL = REPO_ROOT / "data" / "real" / "annotations_val.json"
ANNOTATIONS_TEST = REPO_ROOT / "data" / "real" / "annotations_test.json"

# `file_name` fields in the COCO JSONs start with `data/...` and resolve
# relative to the repo root.
IMAGE_ROOT = REPO_ROOT

TAXONOMY_PATH = REPO_ROOT / "resources" / "speciesnet_taxonomy_release.txt"
CLASSES_225_PATH = REPO_ROOT / "reports" / "classes_225.csv"

# Base directory holding one timestamped sub-directory per fine-tuning run
# (named after the run, e.g. ``teacher_finetune-20260701-233434/``).
# Checkpoints and the training log for a run all live inside its run dir.
OUTPUT_DIR = REPO_ROOT / "scripts" / "training" / "teacher_finetune" / "model_exports"


def latest_run_dir() -> "Path | None":
    """Most recent fine-tuning run dir under ``OUTPUT_DIR`` (one containing
    ``best.pt``), or ``None`` if there are none. Same convention as
    `yolov5s/constants.py`'s `latest_run_dir()`.
    """
    if not OUTPUT_DIR.exists():
        return None
    runs = [d for d in OUTPUT_DIR.iterdir() if d.is_dir() and (d / "best.pt").exists()]
    return max(runs, key=lambda d: d.stat().st_mtime, default=None)


# ─── Model ────────────────────────────────────────────────────────────────────

NUM_CLASSES_LEAF = 2498  # SpeciesNet classifier's native output dimensionality
NUM_CLASSES_225 = 225  # project's wildlife taxonomy (species/genus/family rollups)
IMAGE_SIZE = 480  # SpeciesNet's native EfficientNetV2-M input resolution

# Fraction of the backbone's parameters (by `named_parameters()` iteration
# order — earliest layers first) to FREEZE. The remaining (1 - fraction) —
# later blocks + the classification head — are fine-tuned. Default is a
# partial-fine-tune compromise (frozen low-level features, adapted high-level
# features) chosen because neither the implementation plan nor the parent
# strategy doc pins this down, and full end-to-end fine-tuning risks eroding
# the native 2,498-class taxonomy's general species-ID behaviour, which §2.2
# of the implementation plan explicitly wants preserved (the `prob_225_sum`
# diagnostic and the "still recognizably SpeciesNet" framing both depend on
# it). See `teacher_model.py` for the freeze-boundary implementation and
# README.md for the full rationale. Set to 0.0 for full fine-tune, or 1.0 minus
# a tiny epsilon for a near-linear-probe.
FREEZE_PARAM_FRACTION = 0.5

# ─── Optimizer (AdamW — classifier fine-tuning convention, NOT yolov5s' SGD) ──

OPTIMIZER = "AdamW"
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# ─── Data / loop ──────────────────────────────────────────────────────────────

EPOCH_COUNT = 100  # safety ceiling only — early stopping is expected to end the run
BATCH_SIZE = 32  # conservative default; tune with find_max_batch_size.py
NUM_WORKERS = 8
SEED = 42  # same seed, same data/real/ splits as every other model in the comparison

# ─── Scheduler (OneCycleLR) ────────────────────────────────────────────────────
# Same warmup → peak → cosine-anneal design as the detector pipelines
# (docs/plans/2026-06-10_training-hyperparameters-autostop-lr-schedule.md §4),
# stepped every batch. pct_start is computed from WARMUP_EPOCHS / EPOCH_COUNT so
# the warmup window matches the patience-gating window used by early stopping.

WARMUP_EPOCHS = 3  # early-stop patience gating; also sets ONE_CYCLE_PCT_START
ONE_CYCLE_MAX_LR = 1e-3             # peak LR (10× LEARNING_RATE; super-convergence)
ONE_CYCLE_PCT_START = WARMUP_EPOCHS / EPOCH_COUNT  # warmup fraction
ONE_CYCLE_DIV_FACTOR = 10.0         # initial_lr = max_lr / div_factor = LEARNING_RATE
ONE_CYCLE_FINAL_DIV_FACTOR = 100.0  # min_lr = initial_lr / final_div_factor = 1e-6

# ─── Auto-stop + selection metric ──────────────────────────────────────────────

SELECTION_METRIC = "f1_macro"  # one of the keys returned by evaluate(): "accuracy_top1" | "f1_macro" | "f1_micro"
EARLY_STOP = True
EARLY_STOP_PATIENCE = 15
EARLY_STOP_MIN_DELTA = 0.001

# ─── Training-quality add-ons ──────────────────────────────────────────────────

USE_EMA = True  # evaluate/checkpoint an exponential moving average of the weights
USE_AMP = True  # automatic mixed precision (CUDA only; no-op on CPU)

# ─── MLflow ───────────────────────────────────────────────────────────────────
# URI / experiment / credentials are loaded from `.env` at run start; these
# are only fallbacks if the env vars are missing.

MLFLOW_EXPERIMENT_DEFAULT = "teacher-finetune-speciesnet225"
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
