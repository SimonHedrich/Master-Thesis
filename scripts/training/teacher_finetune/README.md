# SpeciesNet Teacher Fine-Tuning Pipeline

Fine-tunes SpeciesNet's classifier head (EfficientNetV2-M, 2,498-class native
taxonomy) on the project's 225-class wildlife dataset (`data/real/`). This is
Phase 2 (teacher fine-tuning) of
`docs/plans/2026-06-30_knowledge-distillation-and-teacher-finetuning-strategy.md`'s
experimental ladder — see
`docs/plans/2026-06-30_yolo26-kd-and-teacher-finetune-implementation-plan.md`
§2 for the implementation plan this package follows. The resulting checkpoint
is a prerequisite for Goal B (KD training): `cache_soft_labels.py` (also in
this package, since it needs the same `speciesnet` 3.11 environment) uses it
once, offline, to regenerate SpeciesNet soft labels for the student's
distillation loss — see "Caching soft labels for Goal B" below.

Runs in a separate Python 3.11 Docker environment (`Dockerfile.speciesnet`),
not the main project's Python 3.13/`uv` environment, because the `speciesnet`
PyPI package constrains to Python `<3.13`.

## Deviations from the detector pipelines

`scripts/training/yolov5s/` and `scripts/training/yolo26n/` share a
byte-identical training-protocol "comparability contract" (see the
implementation plan's §0). This package does **not** attempt that — SpeciesNet
is a classifier, not a detector, and most of the detector pipelines' constants
have no meaning here. Every deviation is intentional; documented once here and
mirrored as a comment banner at the relevant constant in `constants.py`:

- **Training data source: `data/real/annotations_{train,val,test}.json`
  directly, not `filter_results.jsonl`.** The parent strategy doc's §5
  engineering table says to source crops from `filter_results.jsonl`.
  `annotations_*.json` is **downstream and more authoritative**: it's produced
  after the contamination-review pipeline
  (`14-flag_multi_animal_contamination.py` → `15-apply_contamination_decisions.py`
  + human FiftyOne review), already carries the final reconciled
  `category_id` and `bbox`, and — critically — is the **exact same file
  YOLOv5s/YOLO26n train against**. Using it directly guarantees the teacher
  and every student train on literally identical `(image, bbox, label)`
  triples, not just an identical split. `dataset.py` normalizes the
  absolute-pixel COCO bboxes by each image's `width`/`height` before handing
  them to `SpeciesNetClassifier.preprocess_crop()`'s normalized-bbox
  convention.
- **No person-class downweighting.** The parent strategy doc's §3 says to
  carry the detector pipelines' 0.3× person-class loss weight over
  "unchanged" — but that mechanism (`yolo26n_model.py`'s
  `model.class_weights`) downweights MegaDetector's own `person` output class
  (animal/person/vehicle), which is **not** one of the 225 project classes at
  all (confirmed: not present in `reports/classes_225.csv`). This classifier
  only ever sees **animal-class** MegaDetector crops (that's all
  `annotations_*.json` contains), so there is no person-crop training signal
  to downweight. `loss.py` has no such term.
- **No training-time image augmentation beyond `clf.preprocess()`'s fixed
  crop+resize.** The parent doc §2.1 explicitly wants train-time and
  inference-time preprocessing identical ("reuse this verbatim for
  fine-tuning"); `dataset.py` calls the exact same preprocessing path
  `6-classify_speciesnet.py` uses at inference, with no extra transforms.
- **`OPTIMIZER="AdamW"`, `LEARNING_RATE=1e-4`** — not SGD/1e-3. Standard
  classifier-fine-tuning convention on a pretrained backbone; the detector
  pipelines' SGD hyperparameters were calibrated for a from-scratch-anchor
  detection loss with no equivalent here.
- **`IMAGE_SIZE=480`** — SpeciesNet's native classifier input resolution, not
  640.
- **`FREEZE_PARAM_FRACTION=0.5`** (partial fine-tune). Neither the
  implementation plan nor the parent strategy doc pins down how much of the
  EfficientNetV2-M backbone to unfreeze — a judgment call made during
  implementation (no user response received when asked; proceeded with the
  standard transfer-learning compromise). Freezing the first half of the
  backbone's parameters (by `named_parameters()` iteration order) and
  fine-tuning the back half + head balances domain adaptation against
  preserving the native 2,498-class taxonomy's general species-ID behaviour —
  which the implementation plan's §2.2 explicitly wants kept, for the
  `prob_225_sum` diagnostic and the "still recognizably SpeciesNet" framing.
  Set to `0.0` for full end-to-end fine-tune, or close to `1.0` for a
  near-linear-probe. See `teacher_model.py`'s `_freeze_backbone()` — flagged
  there as needing a real look at EfficientNetV2-M's actual module structure
  once `speciesnet` is installed, since a flat-tensor-count split may not
  align cleanly with a fraction-of-depth split.
- **Per-source breakdown computed directly**, not via
  `8-class_distribution_report.py`'s machinery. That script measures trusted
  dataset-composition tiers for dataset-build decisions — a different purpose
  from scoring model predictions against ground truth. `evaluate.py` instead
  groups by each sample's `source` field (already in `annotations_*.json`'s
  `images` list).

**Circularity caveat** (parent strategy doc §1.1, restated here): part of the
test set's OpenImages/ImagesCV portion was originally filtered using the
*pre-fine-tuning* SpeciesNet, so that portion's contribution to any reported
accuracy improvement is optimistic by an unknown (likely small) amount. This
is disclosed via `evaluate.py`'s per-source breakdown, not engineered around —
every model in the comparison matrix is scored on the same fixed test set.

## Module map

| File | Role |
|------|------|
| `constants.py` | Single source of truth for paths and hyperparameters (see deviations above). `as_dict()` is logged to MLflow at run start. |
| `taxonomy.py` | `build_group_table()` — inverts `7-filter_speciesnet.py`'s species→genus→family lookup dicts (imported via the existing numeric-filename `importlib` pattern) into `idx_225 -> list[leaf_idx]`, for the grouped-CE loss. `projection_tables()` exposes the same cached lookup dicts for `evaluate.py`'s `compute_probs_225` projection. |
| `dataset.py` | `SpeciesNetCropDataset` reads `data/real/annotations_{split}.json` directly (see deviations), normalizes bboxes, delegates cropping to an injected `preprocess_fn`. `collate_fn` stacks crops + labels. |
| `teacher_model.py` | `speciesnet_model()` loads the SpeciesNet classifier, applies the freeze split, returns `(model, preprocess_fn, labels)` — same factory shape as `yolov5s_model.py`. `model_optimizer()` (single-group AdamW), `model_scheduler()` (`OneCycleLR`; stepped every batch in the training loop). |
| `loss.py` | `GroupedCrossEntropyLoss` — the log-sum-exp-over-group technique from the implementation plan's §2.2, reducing to plain cross-entropy for the 178 species-level (1:1) classes. |
| `training_pipeline.py` | New `TrainingPipeline` class (not imported from `yolov5s` — no bbox decode, no mAP, different eval contract) mirroring the same EMA/AMP/`OneCycleLR`/early-stop/checkpoint/MLflow-cadence conventions. Includes a small self-contained `ModelEMA` (not `yolov5.utils.torch_utils.ModelEMA`, to avoid pulling the detector-only `yolov5` package into this image). |
| `evaluate.py` | `evaluate()` — projects native 2,498-way softmax to the 225-class vector via `compute_probs_225`, computes top-1 accuracy + macro/micro F1 (`torchmetrics`), reports the per-source breakdown. `eval_log_mlflow()` logs scalars + a per-source table. |
| `run_finetune.py` | Entry point — same `--smoke` / `.env` / MLflow-wiring structure as `run_training_pipeline.py`. Builds its own `DataLoader`s directly rather than importing `yolov5s.dataset`'s wrapper, since that module pulls in `yolov5.utils.augmentations` (not installed in this image). |
| `find_max_batch_size.py` | Same doubling-until-OOM structure as `yolov5s`'s, probing forward+backward (training) memory on the classifier. |
| `smoke_test_taxonomy_and_loss.py` | Risk-mitigation smoke test for the grouped-CE mechanism — group-table self-consistency, CE-equivalence at group size 1, numerical stability, `compute_probs_225` round-trip. Must pass before any real training run. |
| `cache_soft_labels.py` | **Goal B prerequisite.** Loads a fine-tuned `best.pt` into a fresh `speciesnet_model()`, runs it over a COCO split via `SpeciesNetCropDataset`, projects each detection's softmax to the 225-class vector (`taxonomy.projection_tables()` + `compute_probs_225`), writes `data/real/teacher_soft_labels_{split}.jsonl`. Run once, offline, after `run_finetune.py` — the student's KD training loop (`scripts/training/yolo26n --kd`) only ever reads this file. |

## Data contract

Same `data/real/annotations_{train,val,test}.json` COCO files the detector
pipelines use — see `yolov5s/README.md`'s data contract for the general
shape. The one thing this package additionally relies on:
`images[*].source` (e.g. `inaturalist`, `gbif`, `openimages`) drives
`evaluate.py`'s per-source breakdown and the circularity-caveat disclosure
above.

## Setup

### 1. Build the SpeciesNet Docker image

```bash
make speciesnet-build
```

Builds `Dockerfile.speciesnet` (Python 3.11, CUDA 12.8 base, `speciesnet` +
`PytorchWildlife` + `mlflow`/`python-dotenv`/`torchmetrics`).

### 2. MLflow credentials

```bash
cp scripts/training/teacher_finetune/.env.example scripts/training/teacher_finetune/.env
# then edit it
```

### 3. Start the persistent container

```bash
make speciesnet-start
```

Drops you into a shell inside the container with the repo mounted at `/app`.
`make speciesnet-stop` tears it down when you're done.

## Running

### Smoke test (1 epoch on val set — wiring check)

Inside the container (or via `make speciesnet-finetune` after adding
`--smoke` to the target — see below):

```bash
cd /app && python -m scripts.training.teacher_finetune.run_finetune --smoke
```

Uses the val split as a tiny train set, `num_workers=0`, 1 epoch. Run
`smoke_test_taxonomy_and_loss.py` first — it's cheap (seconds, mostly CPU) and
catches grouped-CE bugs before they hide inside a plausible-looking accuracy
number:

```bash
python -m scripts.training.teacher_finetune.smoke_test_taxonomy_and_loss
```

### Full run

```bash
make speciesnet-finetune
```

(Requires `make speciesnet-start` to already be running.) Outputs land in a
per-run directory, same convention as the detector pipelines:

- `scripts/training/teacher_finetune/model_exports/<run_name>/best.pt` (highest val `SELECTION_METRIC`; EMA weights when `USE_EMA`) — the artifact Goal B needs
- `scripts/training/teacher_finetune/model_exports/<run_name>/last.pt`
- `scripts/training/teacher_finetune/model_exports/<run_name>/<run_name>.log`
- All of the above + per-epoch metrics on the MLflow server

### Caching soft labels for Goal B

Once a real `best.pt` exists (from the full run above), regenerate the
teacher soft-label cache the student's `--kd` mode reads — inside the same
container:

```bash
python -m scripts.training.teacher_finetune.cache_soft_labels --split train
python -m scripts.training.teacher_finetune.cache_soft_labels --split val   # for yolo26n --kd --smoke
```

Defaults to `constants.latest_run_dir()/best.pt` and writes
`data/real/teacher_soft_labels_{split}.jsonl` (pass `--checkpoint`/`--output`
to override either). This is a one-time, offline step — the ~54M-param
teacher never runs inside the student's training loop; it only reads the
resulting JSONL cache.

## MLflow logging contract

| When | What |
|---|---|
| Run start | All `constants.py` values + dataset sizes + git SHA + device, as `mlflow.log_params` |
| Every `MLFLOW_LOG_EVERY_N_STEPS` (default 50) | `train/step/loss_grouped_ce`, `train/lr` at `step=global_step` |
| End of each epoch | `train/epoch_loss_grouped_ce`, `val/accuracy_top1`, `val/f1_macro`, `val/f1_micro` at `step=epoch`, plus a per-source accuracy table artifact |
| End of run | `test/accuracy_top1`, `test/f1_macro`, `test/f1_micro`, `<run_name>/best.pt`, `<run_name>/last.pt`, `<run_name>/<run_name>.log` |

## Tuning

| Constant | Default | Notes |
|---|---|---|
| `EPOCH_COUNT` | 100 | Safety ceiling only — early stopping is expected to end the run. |
| `BATCH_SIZE` | 32 | Conservative default; tune with `find_max_batch_size.py`. |
| `LEARNING_RATE` / `OPTIMIZER` | 1e-4 / `"AdamW"` | See "Deviations" above. |
| `FREEZE_PARAM_FRACTION` | 0.5 | See "Deviations" above. |
| `SELECTION_METRIC` | `"f1_macro"` | Drives best.pt + plateau + early stop. Macro (not micro/accuracy) weights rare classes equally, matching the project's long-tail concern. |
| `EARLY_STOP_PATIENCE` / `EARLY_STOP_MIN_DELTA` | 15 / 1e-3 | |
| `USE_EMA` / `USE_AMP` | `True` / `True` | |
| `IMAGE_SIZE` | 480 | SpeciesNet's native input resolution. |

## Limitations / known issues

- **11 of 225 project classes have NO corresponding leaf class anywhere in
  SpeciesNet's native 2,498-class taxonomy** — discovered by running
  `smoke_test_taxonomy_and_loss.py` against the real classifier (verified not
  a bug: 0 species-level classes have an *ambiguous*, >1-leaf match; these 11
  simply have 0): **blackbuck, eared seals, elephant seal, japanese macaque,
  kob, pinniped clade, ring-tailed lemur, saiga, sea otter, walrus, yak.**
  SpeciesNet's own training distribution apparently never covered these
  species (e.g. it has other `macaca`/`bos`/`kobus` species but not
  `fuscata`/`grunniens`/`kob` specifically). This is a **structural ceiling**,
  not a training-fixable bug: `compute_probs_225` (the existing production
  projection this package reuses unmodified) has always given these 11
  classes exactly zero probability mass, so top-1 recall on them is
  necessarily 0% regardless of fine-tuning quality. `loss.py`'s
  `GroupedCrossEntropyLoss` excludes samples with these labels from the loss
  (`ignore_index`-style) rather than crashing on an empty-group `logsumexp`.
  Report this explicitly alongside any headline accuracy/F1 number — it caps
  the achievable ceiling by `11/225 ≈ 4.9` percentage points of macro-averaged
  metrics even with a perfect model.
- **No multi-GPU / DDP**, **no resume-from-checkpoint** — same as the
  detector pipelines.
- **Freeze-boundary implementation is a flat parameter-tensor-count split**,
  not a verified depth-aligned split — see the `FREEZE_PARAM_FRACTION` note
  above. Worth double-checking against SpeciesNet's actual module structure
  once training is run for real.
- **`GroupedCrossEntropyLoss` loops per-sample in Python** rather than a
  vectorized masked computation — fine at the batch sizes here (tens, not
  thousands), see `loss.py`'s docstring for the reasoning.
- **Circularity caveat** (see above) — always report per-source breakdown
  alongside any headline accuracy/F1 number from this pipeline.
