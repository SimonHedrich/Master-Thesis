# YOLO26n Fine-Tuning Pipeline

Direct fine-tune of YOLO26n on the 225-class wildlife dataset (`data/real/`),
engineered for direct comparability with `scripts/training/yolov5s/` — see
`docs/plans/2026-06-30_yolo26-kd-and-teacher-finetune-implementation-plan.md`
§0–1 for the exact comparability contract (what's kept identical vs. what's
necessarily exempted). This is Phase 1 (direct fine-tune baseline) of
`docs/plans/2026-06-30_knowledge-distillation-and-teacher-finetuning-strategy.md`'s
experimental ladder. This package also implements Phase 3 (KD training, Goal
B of the implementation plan's §3) as a `--kd` mode — see "KD training mode
(Goal B)" below.

YOLO26-specific machinery (model architecture, loss) is imported from the
`ultralytics` PyPI package; everything else — dataset, dataloader, optimizer,
scheduler, training loop, logging — is **reused directly from
`scripts.training.yolov5s`**, not copied, so there is only ever one
implementation of the shared pipeline pieces.

## Comparability contract

Per the implementation plan's §0, every training-protocol constant in
`constants.py` is kept **byte-identical** to `yolov5s/constants.py`: data
paths/splits, `SEED`, optimizer family/LR schedule, early-stop, every `AUG_*`
value, eval thresholds. Two things are *intentionally* exempted — treat these
as documented design decisions, not drift:

- **`BATCH_SIZE`**: not forced to match yolov5s's value. VRAM footprint
  differs by architecture (`docs/2026-04-29_gpu_training_options.md`). Run
  `PYTHONPATH=. uv run -m scripts.training.yolo26n.find_max_batch_size` and
  set `BATCH_SIZE` in `constants.py` to the largest power-of-two that fits
  this GPU. The chosen value is still logged to MLflow via `constants.as_dict()`,
  so the comparison report shows what batch size each model actually used.
- **`HYP_BOX` / `HYP_CLS` / `HYP_DFL`**: use Ultralytics' own calibrated
  defaults (`box=7.5, cls=0.5, dfl=1.5`, from `ultralytics/cfg/default.yaml`),
  **not** yolov5s's `HYP_BOX=0.05/...` values. YOLO26 (`yolo26.yaml`:
  `end2end: True`, `reg_max: 1`) is anchor-free and NMS-free — no anchor
  matching step, no objectness term — so yolov5s's anchor-based gains have no
  equivalent here and reusing them would mis-scale a structurally different
  loss rather than improve comparability. There is likewise no `HYP_OBJ` /
  `HYP_IOU_T` / `HYP_ANCHOR_T` / `HYP_FL_GAMMA` / `HYP_LABEL_SMOOTHING`
  equivalent in this package.

## Module map

| File | Role |
|------|------|
| `constants.py` | Single source of truth for paths and hyperparameters (see comparability contract above). `as_dict()` is logged to MLflow at run start. |
| `transforms.py`, `dataset.py`, `logging_setup.py` | **(imported from `scripts.training.yolov5s` — not duplicated)**. `CocoYoloDataset`'s `targets[N,6]` = `[batch_idx, cls, cx, cy, w, h]` (normalized) is exactly the convention Ultralytics' loss expects — no dataset changes needed. |
| `yolo26n_model.py` | `yolo26n_model()` builds `ultralytics.nn.tasks.DetectionModel("yolo26n.yaml", nc=225)`, loads pretrained COCO weights (shape-filtered), sets `model.args` (an `IterableSimpleNamespace`, not a dict — see note below), `model.model[-1].max_det`. `model_optimizer` / `model_scheduler` are **re-exported from `yolov5s_model.py`** (verified `nn.Module`-generic). |
| `loss.py` | `Yolo26Loss` — thin wrapper around `model.init_criterion()` (resolves to Ultralytics' `E2ELoss` since `end2end=True`). Same `(preds, targets)` call signature as yolov5s's `YoloLoss`, returning `{"loss_box","loss_cls","loss_dfl","loss_total"}` (no `loss_obj` — anchor-free TAL assignment replaces objectness). `update()` anneals `E2ELoss`'s one2many/one2one weight mix once per epoch. |
| `evaluation.py` | `evaluate()` — same `MeanAveragePrecision` + un-letterbox harness as yolov5s, but **NMS-free**: YOLO26's `Detect` head (`end2end=True`) already does score-ranked top-k filtering (`Detect.max_det`), so there is no `non_max_suppression()` call. |
| `training_pipeline.py` | **(imported from `scripts.training.yolov5s` — not duplicated)**. `TrainingPipeline`'s loss-key handling is generic (keyed off whatever `loss_fn` returns) and it calls `getattr(self.loss_fn, "update", lambda: None)()` once per epoch — a no-op for yolov5s's `YoloLoss`, the annealing hook for `Yolo26Loss`. |
| `run_training_pipeline.py` | Entry point — same structure as yolov5s's, imports repointed to this package's `constants`/`loss`/`yolo26n_model`. |
| `find_max_batch_size.py` | Same power-of-two-doubling search as yolov5s's, swapped to `yolo26n_model`/`Yolo26Loss`. |
| `smoke_test_loss_and_decode.py` | **New, yolo26n-specific.** Validates the loss adapter and the NMS-free decode path on synthetic data before any real training run (see below). |
| `kd_dataset.py` | **Goal B.** `KDCocoYoloDataset` — composition wrapper over `CocoYoloDataset` that adds a per-image `teacher_probs` tensor loaded from a `teacher_soft_labels_{split}.jsonl` cache (zeros if uncached). `kd_collate_fn` stacks it alongside the base 4-tuple. Only used for `dl_train` when `--kd` is set; `dl_val`/`dl_test` are unaffected. |
| `kd_loss.py` | **Goal B.** `KDv8DetectionLoss`/`KDE2ELoss`/`KDYolo26Loss` — blends `TaskAlignedAssigner`'s soft `target_scores` toward the cached teacher distribution before the BCE cls loss, per-image (not per-instance). See "KD training mode" below. |
| `smoke_test_kd_loss.py` | **Goal B.** Synthetic-tensor checks for the KD blend: all-zero teacher_probs is a bit-identical fallback, the blend measurably changes `loss_cls` as `KD_ALPHA` sweeps, temperature scaling behaves correctly, and `KDE2ELoss` still anneals `o2m`. |
| `eval_suite/predict.py` | Model-specific inference → predictions-JSON. Same frozen contract as yolov5s's, NMS-free decode substitution identical to `evaluation.py`'s. Unchanged by `--kd` — KD checkpoints are saved in the same format as direct-FT ones. |
| `eval_suite/{grouping,scoring,report}.py` | **(imported from `scripts.training.yolov5s.eval_suite` — not duplicated)**. Confirmed model-agnostic: pure consumers of the frozen predictions-JSON + COCO annotation contract. |
| `eval_suite/run_evaluation.py` | Thin entry point wiring this package's `predict` into the shared `grouping`/`scoring`/`report`. |

## `model.args` note

Ultralytics' loss classes (`v8DetectionLoss`, `E2ELoss`) read hyperparameters
via **attribute** access (`model.args.box`, `.cls`, `.dfl`, `.epochs`). This
pipeline bypasses Ultralytics' high-level `Model`/`Trainer` API entirely
(constructing `DetectionModel` directly, the same bypass pattern
`yolov5s_model.py` uses for yolov5's `Trainer`), so `model.args` is never
auto-populated — `yolo26n_model()` sets it explicitly as an
`ultralytics.utils.IterableSimpleNamespace` (not a plain dict, which would
raise `AttributeError` inside the loss).

## KD training mode (Goal B)

`run_training_pipeline.py --kd` distills `scripts/training/teacher_finetune/`'s
frozen, fine-tuned SpeciesNet teacher into this student, reusing every other
piece of the direct-FT recipe unchanged (dataset, augmentation, optimizer,
scheduler, early-stop) so the KD-vs-direct-FT comparison isolates the loss,
not the recipe. See
`docs/plans/2026-06-30_yolo26-kd-and-teacher-finetune-implementation-plan.md`
§3 for the full design.

**Prerequisite: the teacher soft-label cache.** KD reads a precomputed
`data/real/teacher_soft_labels_{split}.jsonl` — the ~54M-param teacher
classifier never runs inside this package's training loop. Build it once,
inside `Dockerfile.speciesnet` (after `teacher_finetune/run_finetune.py` has
produced a `best.pt`):

```bash
# inside `make speciesnet-shell` / `speciesnet-start`
PYTHONPATH=. python -m scripts.training.teacher_finetune.cache_soft_labels --split train
PYTHONPATH=. python -m scripts.training.teacher_finetune.cache_soft_labels --split val   # for --kd --smoke
```

**The blend mechanism.** `TaskAlignedAssigner` already produces a soft,
IoU-quality-weighted `target_scores` tensor for every foreground anchor
(`ultralytics/utils/loss.py`'s `v8DetectionLoss.get_assigned_targets_and_loss`).
For anchors belonging to an image with a cached teacher record,
`KDv8DetectionLoss` blends this per-image (not per-instance — Goal B is
scoped to single-label KD; multi-animal pseudo-GT is a documented follow-on)
before the BCE cls loss:

```
target_scores[image, anchor, :] = (1 - KD_ALPHA) * target_scores[...] + KD_ALPHA * teacher_probs_225[image]
```

`KD_TEMPERATURE` sharpens/flattens the cached teacher distribution first (via
simplex re-normalization — the cache stores post-softmax probabilities, not
logits, so this is an approximation of true logit-temperature scaling).
Images with no cached record fall through to the unmodified hard-label
target — bit-identical to plain `Yolo26Loss` (verified by
`smoke_test_kd_loss.py`'s check 1). `KD_APPLY_TO` controls which `E2ELoss`
head(s) receive the blend; defaults to `"one2one"`, the head actually used at
inference.

**Usage:**

```bash
# 0. KD loss smoke test (synthetic tensors, cheap, run first)
PYTHONPATH=. uv run -m scripts.training.yolo26n.smoke_test_kd_loss

# 1. KD training smoke test (wiring check — needs teacher_soft_labels_val.jsonl)
PYTHONPATH=. uv run -m scripts.training.yolo26n.run_training_pipeline --kd --smoke

# 2. Full KD run (COCO-pretrained init by default — see --init-from below)
PYTHONPATH=. uv run -m scripts.training.yolo26n.run_training_pipeline --kd
```

`--init-from {coco,phase1}` (default `coco`) controls weight init: KD always
starts from the COCO-pretrained checkpoint by default, **not** this
package's own Phase-1 (direct-FT) `best.pt` — mixing the two would confound
the KD signal with a head start from hard-label fine-tuning. `phase1` is an
explicit, logged opt-in for a separate ablation, not the default.

Runs land in the same `model_exports/` directory as direct-FT runs, prefixed
`yolo26n-kd-...`, tagged `mode=kd` in MLflow (vs. `mode=direct-ft`) in the
same experiment — so both conditions are directly comparable side by side,
per the parent doc's §5 requirement.

**No separate batch-size tuning needed.** KD adds only a `[225]`-float
vector per training sample and a masked blend inside the loss — negligible
GPU memory versus direct-FT. Reuse whatever `BATCH_SIZE` `find_max_batch_size.py`
already found for the direct-FT run; there is no `--kd`-specific batch-size
script.

**Status:** code-complete and validated on synthetic data
(`smoke_test_kd_loss.py`); the real end-to-end run requires
`teacher_finetune/run_finetune.py` to have actually been executed first (no
checkpoint exists yet on a fresh checkout) — see that package's README.

## Data contract

Same as `yolov5s/README.md`'s: `data/real/annotations_{train,val,test}.json`
are standard COCO JSON, `file_name` relative to the repo root, `bbox` is
pixel `[x, y, w, h]`, `categories[*].id` is 1..225 (mapped to YOLO indices
0..224 inside the shared dataset).

**Known gap on some hosts:** the actual image bytes for `data/real/` may
ship as unextracted zips (`data/inaturalist.zip`, `data/gbif.zip`,
`data/wikimedia.zip`, `data/openimages.zip`) rather than already-unpacked
directories — extract them before running anything beyond the synthetic
smoke test. This is separate from the already-documented
`data/synthetic/annotations_test.json` gap below.

## `--full-eval` / `data/synthetic/` caveat

`data/synthetic/annotations_test.json` does not exist on every host.
`--full-eval` and any `mixed`-domain evaluation degrade gracefully to
real-only with a warning (same behavior as
`yolov5s/eval_suite/run_evaluation.py`'s `evaluate_checkpoint`), but for the
genuine mixed-domain report, sync the synthetic data in first (see the
`Makefile`'s `sync`/`sync-ics-data` targets, or extract `data/synthetic.zip`
if present). This blocks the full evaluation report, not training itself
(training only touches `data/real/`).

## MLflow logging contract

Same shape as `yolov5s/README.md`'s, with `loss_obj` replaced by `loss_dfl`:

| When | What |
|---|---|
| Run start | All `constants.py` values + dataset sizes + git SHA + device, as tags + `mlflow.log_params` |
| Every `MLFLOW_LOG_EVERY_N_STEPS` (default 50) | `train/step/loss_{box,cls,dfl,total}`, `train/lr` at `step=global_step` |
| End of each epoch | `train/epoch_loss_*`, `val/mAP50`, `val/mAP50_95` at `step=epoch` |
| End of each epoch (if torchmetrics returns per-class) | Per-class AP table as a JSON artifact |
| End of run | `test/mAP50`, `test/mAP50_95`, `<run_name>/best.pt`, `<run_name>/last.pt`, `<run_name>/<run_name>.log` |

`loss_dfl` will be numerically ~0 throughout — `reg_max=1` disables DFL for
this architecture; it is logged anyway for parity/debugging, not because
it's expected to move.

## Setup

### 1. Dependencies

`pyproject.toml` already pins `ultralytics>=8.4.33` (which ships `yolo26n.yaml`)
alongside the existing `yolov5>=7.0.13`, `torchmetrics`, `mlflow`, etc. — no
separate environment is needed.

```bash
uv lock && uv sync
```

### 2. Pretrained weights

Download COCO-pretrained YOLO26n into `weights/yolo26n.pt` at the repo root
(check the current release asset name/URL at
https://github.com/ultralytics/assets/releases — mirrors the yolov5 release-asset
pattern already used for `weights/yolov5s.pt`):

```bash
mkdir -p weights
curl -L <yolo26n.pt release URL> -o weights/yolo26n.pt
```

If `weights/yolo26n.pt` is missing the run still works — training starts from
random initialization; shape-mismatched detect-head tensors (`nc=80` COCO vs.
`nc=225` project) are filtered out automatically, same mechanism as yolov5s.

### 3. MLflow credentials

```bash
cp scripts/training/yolo26n/.env.example scripts/training/yolo26n/.env
# then edit it
```

```
MLFLOW_TRACKING_URI=http://mlflow-server:5000
MLFLOW_TRACKING_USERNAME=...
MLFLOW_TRACKING_PASSWORD=...
MLFLOW_EXPERIMENT_NAME=yolo26n-wildlife225
```

## Running

### 0. Loss/decode smoke test (run first — cheap, CPU-only, seconds)

```bash
PYTHONPATH=. uv run -m scripts.training.yolo26n.smoke_test_loss_and_decode
```

Validates the loss adapter (train-mode forward, forward/backward, the
per-epoch anneal hook) and the NMS-free eval-decode path (output shape,
class-index/score ranges, coordinate-space sanity) on synthetic data, plus a
4-image real-data plumbing check of `evaluation.evaluate()`. Must pass
*before* the `--smoke` training run below — a silently-wrong decode would
otherwise produce a plausible-looking but meaningless mAP number.

### 1. Training smoke test (1 epoch on val set — wiring check)

```bash
PYTHONPATH=. uv run -m scripts.training.yolo26n.run_training_pipeline --smoke
```

Uses the val split as a tiny train set, `num_workers=0`, 1 epoch. Use this
before any real run to verify dataset paths, MLflow connectivity, and
gradient flow end to end.

### 2. Full run

```bash
PYTHONPATH=. uv run -m scripts.training.yolo26n.run_training_pipeline
```

Outputs land in a per-run directory `model_exports/<run_name>/` (e.g.
`yolo26n-20260701-233434`):

- `scripts/training/yolo26n/model_exports/<run_name>/best.pt` (highest val `SELECTION_METRIC`; EMA weights when `USE_EMA`)
- `scripts/training/yolo26n/model_exports/<run_name>/last.pt` (final epoch)
- `scripts/training/yolo26n/model_exports/<run_name>/<run_name>.log` (full terminal log)
- `scripts/training/yolo26n/model_exports/<run_name>/evaluation/` (only with `--full-eval`)
- All of the above + per-epoch metrics on the MLflow server

`eval_suite.run_evaluation` defaults to the **latest** run dir, or accepts
`--run-dir` / `--checkpoint` to target a specific run.

## Tuning

Every knob lives in `constants.py` — see the comparability contract above
for which values are pinned to yolov5s's and which are architecture-specific.
The architecture-specific ones:

| Constant | Default | Notes |
|---|---|---|
| `MODEL_CONFIG` | `"yolo26n.yaml"` | Ships inside the `ultralytics` package. |
| `HYP_BOX` / `HYP_CLS` / `HYP_DFL` | 7.5 / 0.5 / 1.5 | Ultralytics' own defaults — see comparability contract. |
| `BATCH_SIZE` | (run `find_max_batch_size.py`) | Not forced to match yolov5s's. |
| `KD_TEMPERATURE` | 4.0 | `--kd` only. Sweep target: `{4, 8}` per the strategy doc's starting grid. |
| `KD_ALPHA` | 0.5 | `--kd` only. Sweep target: `{0.5, 0.7}` — weight on the teacher distribution in the blend. |
| `KD_APPLY_TO` | `"one2one"` | `--kd` only. Which `E2ELoss` head(s) get the blend: `"one2one"` \| `"one2many"` \| `"both"`. |

## Limitations / known issues

Same as `yolov5s/README.md`'s, plus:

- **`loss_dfl` is near-inert**: `reg_max=1` (declared in `yolo26.yaml`)
  disables the distribution-focal-loss branch; the term is logged for parity
  but will not move meaningfully during training.
- **`conf_thres`/`iou_thres` are unused at eval time**: kept in
  `evaluate()`/`predict.py`'s signatures for call-site parity with yolov5s,
  but YOLO26's NMS-free decode only exposes `max_det` as a filtering knob.
