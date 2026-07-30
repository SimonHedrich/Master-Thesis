# YOLO26n Training Pipeline — Synthetic-Generator Comparison

Self-contained training pipeline for the synthetic-generator comparison
experiment (`docs/synthetic-model-comparison/`). Copied and adapted from
`scripts/training/yolo26n/` (not imported — see
[`11_detector-architecture-selection.md`](../../../docs/synthetic-model-comparison/11_detector-architecture-selection.md)
for why YOLO26n was chosen as the fixed detector here, and why this package
is a full copy rather than a thin wrapper: so this experiment's code can
never silently drift if the main 225-class pipeline changes).

## What's different from `scripts/training/yolo26n/`

| | Main pipeline (`yolo26n/`) | This package |
|---|---|---|
| Classes | 225 | 12 (fixed ids — see below) |
| Dataset | one fixed `data/real/` split | one **cell** per generator × prompt-regime under `data/synthetic_model_comparison/train/<generator>/<regime>/` |
| Train/val | separate `annotations_{train,val}.json` | one cell's `annotations.json`, auto-split 80/20 by `split_dataset.py` |
| Test | `data/real/annotations_test.json` | the experiment's fixed **real** test set, `data/synthetic_model_comparison/test/annotations_test.json` — evaluated once at the end, never for early-stopping |
| KD mode | `--kd` (distillation) | not present — this experiment directly fine-tunes per generator, no distillation |
| Seeds | fixed `constants.SEED` | `--seed` CLI override, so the ≥3-seeds-per-cell recommendation doesn't need editing constants |

## The frozen 12-class category ids

Every cell's `annotations.json` (built by `scripts/synthetic_model_comparison/5-export_coco.py`)
must declare exactly these ids/names — taken verbatim from the master
225-class taxonomy, **not** renumbered — so the model's class-index mapping
stays identical across cells and compatible with `reports/lookalike_groups_v2.csv`:

`12`=american black bear, `19`=aye-aye, `107`=grevy's zebra, `123`=kinkajou,
`132`=lion, `146`=mountain zebra, `162`=pangolin family, `166`=plains zebra,
`174`=red fox, `184`=ringtail, `189`=saiga, `214`=water deer.

## Module map

Same shape as `scripts/training/yolo26n/`'s (see that package's README for
the general design rationale — MLflow contract, checkpoint format,
early-stop/EMA/AMP behavior are unchanged):

| File | Role |
|---|---|
| `constants.py` | Paths + hyperparameters. `NUM_CLASSES=12`, no fixed train/val paths (resolved per-cell at runtime), `ANNOTATIONS_TEST` = the fixed real test set, `VAL_FRACTION`/`SPLIT_SEED` for the internal split. |
| `dataset.py`, `transforms.py`, `logging_setup.py`, `training_pipeline.py` | Copied verbatim from `yolov5s/` (already fully generic — dynamic category-id mapping, injectable eval functions), only internal imports repointed to this package. |
| `optim.py` | `model_optimizer`/`model_scheduler`, extracted from `yolov5s_model.py` (architecture-generic — no yolov5-specific code needed here). |
| `yolo26n_model.py`, `loss.py`, `evaluation.py` | Copied verbatim from `yolo26n/` (already class-count-agnostic), imports repointed. |
| `split_dataset.py` | Carves a stratified train/val split out of one cell's `annotations.json` (`VAL_FRACTION`/`SPLIT_SEED`), idempotent, auto-invoked by `run_training_pipeline.py`. |
| `run_training_pipeline.py` | Entry point. `--generator`/`--prompt-regime` (required) select the cell; `--seed` overrides the training seed; `dl_test` is always the fixed real test set. |
| `find_max_batch_size.py`, `smoke_test_loss_and_decode.py` | Same sanity checks as `yolo26n/`'s, imports repointed. |
| `eval_suite/predict.py` | Copied from `yolo26n/eval_suite/`, imports repointed. |
| `eval_suite/grouping.py`, `eval_suite/scoring.py` | Copied from `yolov5s/eval_suite/`, then trimmed to drop unused granularity/domain-merge helpers (`load_detect_remap`, `merge_domains`, `filter_image_ids_by_band`, `domain_shift_delta`) — this experiment only ever scores real-only, fine + coarse granularity. Still valid for these 12 classes since their ids aren't renumbered. |
| `eval_suite/report.py` | Rewritten (not a straight copy) to cover only this experiment's Axis C ask: headline real-test mAP, per-class AP table, within-look-alike-group confusion. The production suite's mixed/real duplication, detect-analog, granularity-gap decomposition, band×granularity grid, and domain-shift-delta tiers are dropped — see the module docstring for why each doesn't apply to a generator comparison. |
| `eval_suite/run_evaluation.py` | Adapted: real test set only (no synthetic/mixed domain — this experiment's test set is real-only by design). |

Excluded on purpose (not applicable to a direct-fine-tune generator
comparison): KD machinery (`kd_dataset.py`/`kd_loss.py`/`--kd` mode),
`predict_ensemble.py`.

## Data prerequisite: labeled cells

Training needs a cell's `annotations.json`, produced by the labeling
pipeline in `scripts/synthetic_model_comparison/` (numbered `2`-`5`):

```bash
uv run python scripts/synthetic_model_comparison/2-run_megadetector.py \
    --generator gemini-3.1-flash-image-preview --prompt-regime full
# optional interactive review (scripts 3/4) — see their docstrings
uv run python scripts/synthetic_model_comparison/5-export_coco.py \
    --generator gemini-3.1-flash-image-preview --prompt-regime full
```

## Running

### 0. Loss/decode smoke test (cheap, CPU-only, seconds)

```bash
PYTHONPATH=. uv run -m scripts.synthetic_model_comparison.training.smoke_test_loss_and_decode
# with a real cell, also exercises the dataset/eval plumbing:
PYTHONPATH=. uv run -m scripts.synthetic_model_comparison.training.smoke_test_loss_and_decode \
    --generator gemini-3.1-flash-image-preview --prompt-regime full
```

### 1. Training smoke test (1 epoch on the cell's val split — wiring check)

```bash
PYTHONPATH=. uv run -m scripts.synthetic_model_comparison.training.run_training_pipeline \
    --generator gemini-3.1-flash-image-preview --prompt-regime full --smoke
```

### 2. Full run

```bash
PYTHONPATH=. uv run -m scripts.synthetic_model_comparison.training.run_training_pipeline \
    --generator gemini-3.1-flash-image-preview --prompt-regime full --seed 42
```

Repeat with `--seed 43`, `--seed 44`, … for the ≥3-seeds-per-cell
recommendation (`docs/synthetic-model-comparison/06_evaluation-methodology.md`).
The internal train/val split (`constants.SPLIT_SEED`) stays fixed across
seeds — only model init / dataloader shuffling varies.

Outputs land in `model_exports/<run_name>/` (e.g.
`yolo26n-gemini-3-1-flash-image-preview-full-seed42-20260721-233434/`):
`best.pt`, `last.pt`, the run log, and (with `--full-eval`) `evaluation/`.

### 3. Evaluation report

```bash
PYTHONPATH=. uv run -m scripts.synthetic_model_comparison.training.eval_suite.run_evaluation \
    --run-dir scripts/synthetic_model_comparison/training/model_exports/<run_name>
```

Writes `evaluation_report.{md,json}`, `eval_per_class.csv`, and
`eval_confusion_pairs.csv` to `<run_dir>/eval_<checkpoint_stem>/` (or
`evaluation/` when triggered via `--full-eval`). Covers exactly this
experiment's Axis C ask (`docs/synthetic-model-comparison/
06_evaluation-methodology.md` §4) — headline real-test mAP (COCO-12 vector),
a 12-row per-class AP table flagging test-limited (<30 real test image)
classes, and the zebra-style within-look-alike-group confusion rate/pairs.
This is deliberately narrower than `scripts/training/`'s eval suite, which
also reports a mixed/real domain split, a class-agnostic detect-only mAP, a
fine/coarse/detect granularity-gap decomposition, and a band×granularity
grid — none of which this experiment (comparing image generators, not
detector architectures, on a real-only test set) needs.

## Setup

Same dependencies as `scripts/training/yolo26n/` (`ultralytics`, `yolov5`,
`torchmetrics`, `mlflow`, etc. — already pinned in `pyproject.toml`). Download
`weights/yolo26n.pt` per that package's README if not already present.
Copy `.env.example` to `.env` and fill in MLflow credentials.

## Known limitations

- **Partially-populated cells**: only 7 of 12 classes currently have
  incumbent-generator images (`docs/synthetic-model-comparison/10_train-subset-incumbent-selection.md`).
  The 5 missing classes stay in the 12-class head with zero training
  annotations — same "never predicted, visible in the per-class AP table"
  pattern the main pipeline already tolerates for 50/225 classes.
- **Auto-label yield not required for export**: `5-export_coco.py` will
  export a cell even before human review (stages 3/4) finishes, falling back
  to MegaDetector's best-effort box with a warning — useful for pipeline
  verification, but full review is still required before a cell's numbers
  are thesis-final (`01_experiment-design.md` §5 point 2, "same review rules").
