# Implementation Plan: YOLO26n Direct Fine-Tune, KD Training, and MD+SN Teacher Fine-Tune

**Date:** 2026-06-30
**Status:** Goal A implemented (2026-07-01) — see §1a. Goal C implemented
(2026-07-01) — see §2a. Goal B remains plan-only.
**Depends on:** `docs/plans/2026-06-30_knowledge-distillation-and-teacher-finetuning-strategy.md`
(this document turns its §4 experimental ladder and §5 engineering table into concrete
module specs), `scripts/training/yolov5s/` (the pipeline being mirrored),
`docs/plans/2026-06-10_model-evaluation-strategy.md`,
`docs/plans/2026-06-10_training-hyperparameters-autostop-lr-schedule.md`,
`docs/plans/2026-04-30_speciesnet-classification-strategy.md`.

**Scope — three concrete implementation goals, in dependency order:**

1. **Goal A** — `scripts/training/yolo26n/`: direct fine-tune of YOLO26n on `data/real/`,
   engineered to be **as directly comparable as possible** to
   `scripts/training/yolov5s/` (Phase 1 of the strategy doc's ladder).
2. **Goal C** — `scripts/training/teacher_finetune/`: fine-tune SpeciesNet's classifier
   head on the project's 225-class taxonomy (Phase 2). Listed before Goal B below because
   Goal B depends on its output.
3. **Goal B** — extends the Goal A package with a KD training mode that distills the
   frozen Goal C teacher into YOLO26n, reusing Goal A's data prep/config unchanged so the
   KD-vs-direct-FT comparison isolates the loss, not the recipe (Phase 3).

Quantization/QAT/SNPE export remain out of scope (owned by
`docs/2026-03-10_object-detection-models-for-embedded-systems.md`), per the parent
strategy doc's stated boundary.

---

## 0. What "directly comparable to yolov5s" means here

The user's instruction is to make Goal A "directly comparable" to `yolov5s/` in **data
preparation and hyperparameter/autostop**. This plan treats that as: **identical training
protocol, not identical loss-internal numbers.** Concretely:

**Kept byte-for-byte identical** (copied from `yolov5s/constants.py` unchanged):
`NUM_CLASSES=225`, `IMAGE_SIZE=640`, the train/val/test annotation paths, `SEED=42`,
`OPTIMIZER="SGD"`, `LEARNING_RATE=1e-3`, `MOMENTUM=0.937`, `WEIGHT_DECAY=5e-4`,
`NESTEROV=True`, `WARMUP_EPOCHS=3`, `PLATEAU_FACTOR=0.5`, `PLATEAU_PATIENCE=5`,
`PLATEAU_MIN_LR=1e-5`, `EPOCH_COUNT=200` (safety ceiling), `SELECTION_METRIC="mAP50_95"`,
`EARLY_STOP=True`, `EARLY_STOP_PATIENCE=20`, `EARLY_STOP_MIN_DELTA=0.001`, `USE_EMA=True`,
`USE_AMP=True`, every `AUG_*` flag and value (HFlip/HSV/scale/translate on; mosaic/mixup/
rotation/shear/perspective off — the shared "setup A/B" augmentation set per
`docs/plans/2026-06-07_data-augmentation-strategy.md`), `EVAL_CONF_THRES=0.001`,
`EVAL_IOU_THRES=0.6`, `EVAL_MAX_DET=100`, `NUM_WORKERS=8`, `MLFLOW_LOG_EVERY_N_STEPS=50`.
This is what makes the comparison meaningful: same images, same splits, same augmentation
pixels, same optimizer family/LR schedule/early-stop criterion, same selection metric.

**Explicitly exempted — `BATCH_SIZE`.** VRAM footprint differs by architecture
(`docs/2026-04-29_gpu_training_options.md` already tunes batch size per model family).
Run `scripts/training/yolo26n/find_max_batch_size.py` (a copy of the existing script,
swapped to the new model factory) and use the largest power-of-two that fits, exactly the
methodology already used for YOLOv5s. Note the chosen value in the run's MLflow params; do
not force it to match `yolov5s`'s 32.

**Necessarily different — `ComputeLoss` hyperparameters.** YOLOv5s's `HYP_BOX/CLS/OBJ/...`
(`yolov5s/constants.py` lines 68–76) are calibrated for an anchor-based loss with an
objectness term; YOLO26 has no objectness term (§1 below) and ships its own calibrated
defaults (`box=7.5, cls=0.5, dfl=1.5` — verified in
`ultralytics/cfg/default.yaml`). Reusing YOLOv5's gains on YOLO26's loss would not produce
a *more* comparable result — it would silently mis-scale a structurally different loss.
Use Ultralytics' own defaults for the new `HYP_*` constants and document this explicitly
in `yolo26n/README.md` so the difference is visible, not glossed over, exactly like the
`BATCH_SIZE` exemption above.

---

## 1. Goal A — `scripts/training/yolo26n/` (direct fine-tune, Phase 1)

### 1a. Implementation status — DONE (2026-07-01)

Goal A is implemented per this plan's §0–1. Detailed build plan, verified
Ultralytics-internals citations, deviations found during implementation (a
loss-key-genericization fix in the shared `training_pipeline.py` beyond the
single hook line this doc describes, and an `E2ELoss.__call__`-returns-a-3-vector
`.sum()` fix in the new loss adapter), and the verification-gate results are
recorded in `/home/ubuntu/.claude/plans/please-implement-goal-a-parsed-fog.md`.

Summary: all files in the §1.2 module map exist under `scripts/training/yolo26n/`,
the one shared-file edit to `scripts/training/yolov5s/training_pipeline.py` is
in place, `smoke_test_loss_and_decode.py` (the §1.4-mandated risk-mitigation
test) passes all 6 checks, a full (non-abbreviated) `yolov5s --smoke` run
confirms zero regression to the existing pipeline, a full `yolo26n --smoke`
run completed train→val→test with zero errors, and
`eval_suite.run_evaluation --limit 40` produced a full report with all
predicted `category_id`s validated in the correct COCO range. **The full
verification gate has passed end-to-end** — see the linked plan file for the
complete results table. The real (non-smoke) training run — which needs
`weights/yolo26n.pt` downloaded first (not yet done on this host; the
verification runs above trained from random init) — and
`find_max_batch_size.py`-driven `BATCH_SIZE` tuning are deferred to whenever
training is actually launched. Goals B and C are unaffected — still plan-only.

### 1.1 What was verified in this environment

- `ultralytics==8.4.60` is already installed (`pyproject.toml` pins `>=8.4.33`) and ships
  `yolo26n.yaml` (`ultralytics/cfg/models/26/yolo26.yaml`, scale `n`).
- `yolo26.yaml` declares `end2end: True` and `reg_max: 1` at the top of the file. This is
  the architectural fact that drives every difference from YOLOv5s below: YOLO26 is
  **anchor-free, NMS-free, and DFL-disabled** (`reg_max=1` ⇒ `use_dfl = reg_max > 1` is
  `False` inside `v8DetectionLoss`, so the box-regression head predicts box offsets
  directly rather than a distribution).
- `docs/progress_notes/2026-04-24_training-setup-and-model-smoke-test.md` already
  confirmed (RTX 3060, this repo's actual smoke-test script) that the `nc=80→225` head
  swap works for `yolo26n` (2.57M→2.61M params) using the manual swap pattern shown there
  (replace the final `Conv2d` in each `cv3` branch; `cv2`/box-regression branches are
  untouched). That note's elaborate Docker/Makefile zoo for the other model families was
  **never built** — only `scripts/training/yolov5s/` and the repo-root `Dockerfile`/
  `Makefile` exist today. Goal A needs to add the `yolo26n/` package; it does not need to
  build the abandoned multi-image Docker setup.
- Confirmed by reading `ultralytics/nn/tasks.py` (`BaseModel.loss`, `DetectionModel`) and
  `ultralytics/utils/loss.py`: the model exposes `model.loss(batch, preds=None)` /
  `model.init_criterion()` as a **manual-loop-compatible** API — it does **not** require
  Ultralytics' own `Trainer`/`model.train()`. This is what makes mirroring the existing
  custom `TrainingPipeline` loop (rather than switching to `model.train()`) feasible.

### 1.2 New module map (mirrors `yolov5s/` file-for-file)

| File | Role | Relationship to `yolov5s/` |
|---|---|---|
| `constants.py` | Same constants as §0, plus `MODEL_CONFIG="yolo26n.yaml"`, `PRETRAINED_WEIGHTS=weights/yolo26n.pt`, `HYP_BOX=7.5`, `HYP_CLS=0.5`, `HYP_DFL=1.5`, `MLFLOW_EXPERIMENT_DEFAULT="yolo26n-wildlife225"`. No anchor/`HYP_IOU_T`/`HYP_ANCHOR_T`/`HYP_FL_GAMMA` (anchor-free model, no anchor matching). | Copy + trim |
| `transforms.py`, `dataset.py`, `logging_setup.py` | **Reused verbatim, unmodified** — re-export or `import` from `scripts.training.yolov5s`. `CocoYoloDataset`'s output format (image tensor, `targets[N,6]` = `[batch_idx, cls, cx, cy, w, h]` normalized) needs **no changes**: §1.3 shows Ultralytics' loss consumes the identical normalized-xywh convention. | Import, not copy — one shared implementation, zero drift risk |
| `yolo26n_model.py` | Factories: `yolo26n_model()` (load `DetectionModel("yolo26n.yaml", nc=225)`, swap-load COCO weights with the same shape-filtered `load_state_dict(strict=False)` pattern as `yolov5s_model.py`, set `model.args`/`model.names`), `model_optimizer()` (reuse the 3-param-group BN/conv/bias split verbatim — it is `nn.Module`-generic, not YOLOv5-specific), `model_scheduler()` (verbatim `ReduceLROnPlateau`, no changes). | New file, mostly copy |
| `loss.py` | `Yolo26Loss` wraps `model.init_criterion()` (resolves to `E2ELoss` because `end2end=True`); see §1.3 for the exact adapter and the one genuinely new piece of state (`loss_fn.update()` per epoch). | New file, structurally analogous |
| `evaluation.py` | `evaluate()` — same `torchmetrics.MeanAveragePrecision` harness, but the YOLOv5-specific `non_max_suppression()` call is replaced; see §1.4. | New file, one substantive change |
| `training_pipeline.py` | **Reused with one hook added**: after `_train_one_epoch`, call `getattr(self.loss_fn, "update", lambda: None)()` (mirrors the existing `getattr(_ds, "set_epoch", ...)` guard pattern already used for close-mosaic — same defensive style, zero risk to the YOLOv5s path). Everything else (warmup, plateau step, early-stop bookkeeping, EMA, checkpointing, final test eval) is unchanged. | Import + 1 hook, not a copy |
| `run_training_pipeline.py`, `find_max_batch_size.py` | Same structure, swapped imports (`yolo26n_model`, `Yolo26Loss`). | Copy + import swap |
| `eval_suite/predict.py` | New — model-specific inference → predictions-JSON (frozen contract per `eval_suite/README.md`). NMS-free decode; see §1.4. | New file |
| `eval_suite/{grouping,scoring,report,run_evaluation}.py` | **Reused via import from `yolov5s.eval_suite`** — these are already documented as model-agnostic ("Identical contract will be reused by the NanoDet / PicoDet pipelines for apples-to-apples comparison"). Only `run_evaluation.py`'s checkpoint-loading call needs a thin re-export pointed at the new `predict.py`. | Import, not copy |

### 1.3 The loss adapter — the one real engineering task

`v8DetectionLoss.preprocess()` (`ultralytics/utils/loss.py:370`) builds its GT tensor from
`batch["batch_idx"]`, `batch["cls"]`, `batch["bboxes"]` and then does
`xywh2xyxy(out[..., 1:5].mul_(scale_tensor))` — i.e. it expects **normalized-xywh boxes**,
the exact convention `CocoYoloDataset` already emits in `targets[:, 2:6]`. So **no dataset
change is needed** — only a ~5-line reshape from the existing flat `[N,6]` tensor into the
three separate dict entries Ultralytics wants:

```python
# Adapter inside Yolo26Loss.__call__, given the existing (imgs, targets) from the dataloader:
batch = {
    "img": imgs,
    "batch_idx": targets[:, 0],
    "cls": targets[:, 1],
    "bboxes": targets[:, 2:6],
}
preds = model(imgs)                       # {"one2many": {...}, "one2one": {...}}
total, parts = model.criterion(preds, batch)   # E2ELoss.__call__ → (scalar, tensor[3])
```

Two things `model.args` must carry that `yolov5`'s `model.hyp` dict didn't need to:
- `args.box / args.cls / args.dfl` — read by `v8DetectionLoss.__init__` as `h = model.args`.
- `args.epochs` — `E2ELoss.decay()` reads `self.one2one.hyp.epochs` to anneal the
  one-to-many/one-to-one loss-weight mix (`o2m: 0.8→0.1`) over the training horizon. Set it
  to `constants.EPOCH_COUNT` (the safety ceiling) — the decay schedule degrades gracefully
  if early stopping ends the run sooner (it simply never reaches `final_o2m`).
- `loss_fn.update()` must be called once per epoch (increments `E2ELoss`'s internal step
  counter and recomputes `o2m`/`o2o`) — this is the one behavior with no YOLOv5s analog;
  §1.2's `training_pipeline.py` hook covers it.

`E2ELoss.__call__` returns `(total_loss, loss_detach[3])` where the 3 parts are
**box, cls, dfl** (no `obj` — replaced by the anchor-free task-aligned assignment; `dfl`
will be numerically ~0 throughout since `reg_max=1` disables it, but it is logged anyway
for parity/debugging). `Yolo26Loss.__call__` returns the same
`{"loss_box": ..., "loss_cls": ..., "loss_dfl": ..., "loss_total": ...}` dict shape as
`YoloLoss`, so `training_pipeline.py`'s logging code (`for k, v in parts.items(): sums[k] += v`)
needs no change beyond the key names.

**Person-class downweighting for free.** `v8DetectionLoss.__init__` already reads
`getattr(model, "class_weights", None)` and folds it into the BCE cls loss. The existing
project convention (0.3× loss weight on the person class, carried over unchanged per the
strategy doc §3) is therefore a **one-line set** (`model.class_weights = torch.ones(225); model.class_weights[person_idx] = 0.3`)
rather than new loss code — note this in `yolo26n_model.py`.

### 1.4 Evaluation — the other real engineering task

`Detect.forward()` (`ultralytics/nn/modules/head.py`) at inference time runs
`self._inference(preds["one2one"])` then, when `end2end`, `self.postprocess(y)` — a
score-threshold + top-k filter with **no IoU-suppression step**, because the one-to-one
head is trained (via `tal_topk2=1`) to emit at most one prediction per object already.
This means `yolov5s/evaluation.py`'s `from yolov5.utils.general import non_max_suppression`
call has no equivalent need in the new `evaluation.py` / `eval_suite/predict.py` — replace
it with direct consumption of the model's own decoded output (already
`[batch, N, 4+nc]`-shaped boxes+scores, top-k filtered). Concretely: call
`model(imgs)` in eval mode, take the returned tensor (not the `(y, preds)` tuple — `.eval()`
inference returns `y` directly when `model.export` is `False`... verify the exact return
shape against `Detect.postprocess`'s output columns during implementation and unit-test it
against a handful of synthetic boxes before trusting it on the real eval suite), and reuse
the existing un-letterbox + `MeanAveragePrecision` accumulation code unchanged.

This is flagged as a real implementation risk, not a one-liner: get it wrong and per-class
AP silently looks fine but is computed on mis-decoded boxes. **Verification step:** before
running any real training, write a tiny smoke check that feeds a few known synthetic boxes
through `model.eval()` + the new decode path and confirms the recovered boxes match within
letterbox-rounding tolerance — mirrors the existing
`scripts/training/yolov5s/smoke_test_augmentation.py` philosophy (verify the mechanism on
synthetic data before trusting it on real images).

### 1.5 Pre-flight

- `weights/yolo26n.pt` does not exist yet — download the COCO checkpoint into `weights/`
  (the project root already has the analogous `weights/yolov5s.pt`); mirror
  `yolov5s/README.md`'s download-instructions section.
- Run `python -m scripts.training.yolo26n.run_training_pipeline --smoke` (1 epoch on the
  val split, `num_workers=0`) before any real run — same gate `yolov5s/README.md`
  documents.
- `data/synthetic/` is **empty on this host** (verified: `data/synthetic/annotations_test.json`
  does not exist here, despite `docs/plans/2026-06-10_model-evaluation-strategy.md` §11.3
  recording it as built). It must be `rsync`'d in from wherever it actually lives (the
  `Makefile`'s `sync`/`sync-ics-data`/`backup-*` targets reference several hosts) before
  `--full-eval` or any `mixed`-domain evaluation can run on this machine. This blocks
  Phase 4 reporting, not Phase 1 training (training only touches `data/real/`).

---

## 2. Goal C — `scripts/training/teacher_finetune/` (MD+SN fine-tune, Phase 2)

### 2a. Implementation status — DONE (2026-07-01)

Goal C is implemented per this plan's §2. `Dockerfile.speciesnet` and the
`speciesnet-build`/`speciesnet-start`/`speciesnet-stop`/`speciesnet-shell`/
`speciesnet-finetune` Makefile targets were **restored from git history**
(they existed once, were deleted in commit `9731c18` "feat: docker clean,"
and the docs/error-messages referencing them were never updated) and adapted
to add `mlflow`/`python-dotenv`/`torchmetrics` and point at the new
`run_finetune.py` entry point instead of the never-built
`scripts/training/0-teacher_speciesnet_pipeline.py`. All files in this
section's §2.3 module map exist under `scripts/training/teacher_finetune/`.

**Deviations found during implementation** (each documented in depth in
`teacher_finetune/README.md`'s "Deviations from the detector pipelines"
section):
1. **Training data source is `data/real/annotations_{train,val,test}.json`
   directly, not `filter_results.jsonl`** as this section and the parent
   strategy doc's §5 engineering table say — `annotations_*.json` is
   downstream of contamination review and is the exact file every detector
   already trains against, so reusing it guarantees identical
   `(image, bbox, label)` triples across every model, not just an identical
   split.
2. **No person-class downweighting** — tracing the mechanism showed it
   downweights MegaDetector's own `person` output class, which isn't one of
   the 225 project classes; this classifier only ever sees animal-class crops,
   so the mechanism has nothing to apply to here.
3. **A genuine taxonomy-coverage finding, not a bug**: running the
   risk-mitigation smoke test (`smoke_test_taxonomy_and_loss.py`) against the
   real SpeciesNet classifier found **11 of 225 project classes have no
   corresponding leaf class anywhere in SpeciesNet's native 2,498-class
   taxonomy** (blackbuck, eared seals, elephant seal, japanese macaque, kob,
   pinniped clade, ring-tailed lemur, saiga, sea otter, walrus, yak) —
   verified clean (zero species-level classes have an *ambiguous* >1-leaf
   match; these 11 simply have zero). This also means the existing production
   `compute_probs_225` projection has always given these 11 classes exactly
   zero probability mass — a pre-existing, structural recall ceiling, not
   something this package introduces. `GroupedCrossEntropyLoss` was extended
   to exclude samples with these labels from the loss (`ignore_index`-style)
   rather than crashing on an empty-group `logsumexp`. Documented in both
   `loss.py`'s docstring and the README's Limitations section.
4. **`FREEZE_PARAM_FRACTION=0.5`** (partial backbone fine-tune) was a judgment
   call made during implementation — the user was asked which fine-tune scope
   to use (full / partial / linear-probe) and did not respond in time, so the
   plan's own recommended default (partial) was used, exposed as a single
   easily-changed constant.

**Verification-gate results**: `docker build -f Dockerfile.speciesnet` builds
a working `wildlife-speciesnet` image (6.64GB, confirmed `speciesnet`/`torch`
import and CUDA visibility). `smoke_test_taxonomy_and_loss.py` passes all 5
checks inside the container (group-table self-consistency, CE-equivalence at
group size 1, numerical stability, empty-group handling, `compute_probs_225`
round-trip). `run_finetune.py --smoke` was run against the real dataset and
GPU: model load, dataset construction (12,519 crops), taxonomy/group-table
build, optimizer/scheduler construction, and `mlflow.log_params` all
succeeded, and the training loop reached a real forward → grouped-CE loss →
backward → optimizer step → MLflow step-log cycle with finite, plausible loss
values on real data. The **full** 1-epoch smoke cycle (through checkpoint save
and final eval) did not finish in-session: this host's A40 GPU is shared with
other tenants consuming ~16–22GB of its 24GB at the time of testing, forcing
`BATCH_SIZE` down to 4 (from the default 32) to avoid CUDA OOM, and at
`BATCH_SIZE=4` with `num_workers=0` (the `--smoke` default) a full epoch over
all three 12.5k-crop splits would take on the order of hours — an
infrastructure/timing constraint of this verification session, not a code
defect. Re-run `run_finetune.py --smoke` (at the default `BATCH_SIZE=32`, or
tuned via `find_max_batch_size.py`) once the shared GPU has adequate free
memory to confirm the full checkpoint/eval cycle before launching real
training. The real (non-smoke) fine-tuning run and `find_max_batch_size.py`
tuning are deferred to whenever training is actually launched, mirroring
Goal A's §1a status.

### 2.1 What was verified in this environment

- `speciesnet` is **not installed** in the main 3.13 venv (confirmed:
  `ModuleNotFoundError`). `scripts/dataset_quality/6-classify_speciesnet.py` and
  `7-filter_speciesnet.py` both guard on this and instruct
  `make speciesnet-build && make speciesnet-start` — but **no `Dockerfile.speciesnet` and
  no such Makefile targets exist in this repo today.** This is the same gap as Goal A's
  abandoned-note infrastructure: documented as if built, never committed. **First concrete
  task of Goal C is building this environment**, not just writing the fine-tuning script.
- The classifier is loaded as
  `SpeciesNet(DEFAULT_MODEL, components="classifier", geofence=False).classifier`, and
  critically, **`clf.model` is a plain accessible `nn.Module`** — confirmed directly from
  `6-classify_speciesnet.py`'s `classify_batch()`: `logits = self._clf.model(tensor)` is an
  ordinary forward pass on a batched tensor, fully compatible with a standard PyTorch
  fine-tuning loop (`loss.backward()`, `optimizer.step()`, `torch.save(clf.model.state_dict(), ...)`).
  Architecture: EfficientNetV2-M, 2,498-class output, 480×480 input (per
  `docs/progress_notes/2026-03-18_speciesnet-pipeline-and-experiment-design.md` §3),
  ~214 MB checkpoint auto-downloaded by the `SpeciesNet(...)` constructor.
- MegaDetector itself runs in the **main 3.13 env** via `PytorchWildlife>=1.2.4`
  (confirmed: `scripts/dataset_quality/1-filter_dataset_quality.py` already imports it
  there) — only the SpeciesNet classifier needs the separate 3.11 environment. Per the
  strategy doc §4, MegaDetector fine-tuning is optional/secondary; this plan does not
  build it.
- Crops are already defined: `filter_results.jsonl["detections"][i]["bbox"]` is
  `[xmin, ymin, w, h]` normalized, and `SpeciesNetClassifier.preprocess_crop()`
  (`6-classify_speciesnet.py:150`) already shows the exact crop+resize call
  (`clf.preprocess(img, bboxes=[BBox(*bbox_norm)])`) — reuse this verbatim for fine-tuning
  so train-time and inference-time preprocessing are identical.

### 2.2 Open design decision: keep the native 2,498-way head, or swap to a 225-way head?

The strategy doc §5 says Goal C's output should go "via the existing `classes_225.csv`
projection" (i.e. `7-filter_speciesnet.py`'s `compute_probs_225()`), which only makes
sense if the **native 2,498-class head is kept** and projected down at use-time (rather
than replacing it with a fresh 225-way head, the way the YOLO student head-swap works).
This plan recommends keeping the native head, for two reasons:

1. It is what the parent doc already committed to, and what `compute_probs_225()` is
   built for — reusing it (as instructed) requires the native taxonomy to still exist.
2. A 225-way head swap would discard SpeciesNet's ability to discriminate species
   *outside* the 225-class set, which matters for two things this plan needs: (a) the
   `prob_225_sum` diagnostic already used in script 7 (mass falling outside the 225 set is
   informative, not noise — a head-swapped model can't report it) and (b) keeping the
   teacher's generalization behavior closer to its publicly-evaluated form, which matters
   for the thesis's "is this still recognizably SpeciesNet" framing.

**The catch, and the recommended fix.** Of the 225 classes, 178 are species-level (a 1:1
match to exactly one of the 2,498 leaf classes — found by inverting
`genus_species_to_225`), but 35 are genus-level and 12 are family-level rollups with **no
single correct leaf index** — a genus-level project label like "weasel species" maps to
every `mustela *` leaf class at once. Plain single-label cross-entropy is undefined for
these 47 classes. Use **grouped/marginal CE**: for a genus/family-level label, the target
is "any leaf class in this group," implemented as
`loss = -log(sum(softmax(logits)[i] for i in group_leaf_indices))` (log-sum-exp over the
group, a standard hierarchical-classification technique) rather than picking one
arbitrary member or falling back to a uniform distribution. Build the
species/genus/family→leaf-indices group table once, by reusing
`7-filter_speciesnet.py`'s `load_taxonomy()` + `load_classes_225()` (already implemented,
already parses `resources/speciesnet_taxonomy_release.txt` and `reports/classes_225.csv`)
— no new parsing code needed, just a new small function that inverts the existing lookup
dicts into `idx_225 → list[leaf_idx]`.

**Document this decision before building**, since it is a genuine fork not fully pinned
down by the parent strategy doc — if grouped-CE proves awkward to validate, the fallback
is the simpler 225-way head swap (option A in the earlier draft of this analysis),
documented here so a future reader knows it was considered and why it was not the default.

### 2.3 New module map

| File | Role |
|---|---|
| `requirements-speciesnet.txt` (or a `pyproject.toml` under the new dir) | Pins `speciesnet`, `torch`, `Pillow` for the Python 3.11 environment — separate from the main `uv` project per the existing Python-version constraint. |
| `Dockerfile.speciesnet` (repo root, alongside the existing `Dockerfile`) | **New** — Python 3.11 base, installs `speciesnet`. The April note describes the intended shape; it was never committed. |
| `Makefile` additions | `speciesnet-build`, `speciesnet-start`/`speciesnet-stop`, `speciesnet-finetune` targets, mirroring the existing `yolov5s-train` target's `.env`-sourcing pattern. |
| `scripts/training/teacher_finetune/dataset.py` | `SpeciesNetCropDataset`: reads `filter_results.jsonl` (animal detections) filtered to images present in `data/real/annotations_{train,val,test}.json` (**reuse the exact same split** — critical so the teacher and every student train/eval on identical data partitions), crops via `clf.preprocess()`, label = leaf index (species-level) or group-indices list (genus/family-level, §2.2). |
| `scripts/training/teacher_finetune/loss.py` | Grouped CE per §2.2; falls back to plain CE when the group has one member. |
| `scripts/training/teacher_finetune/training_pipeline.py` | Same engineering conventions as `yolov5s/training_pipeline.py` — EMA, AMP, `ReduceLROnPlateau`, single-metric early stop, per-run checkpoint dir, MLflow logging contract — but **not** hyperparameter-identical (different model family; classifier fine-tuning conventionally uses AdamW at a lower LR, e.g. `1e-4`, vs. the detector pipelines' SGD `1e-3`). State this divergence explicitly in `teacher_finetune/README.md`, same spirit as §0's `BATCH_SIZE`/`HYP_*` exemptions for Goal A. |
| `scripts/training/teacher_finetune/evaluate.py` | Computes species accuracy/F1 (via `compute_probs_225` projection, top-1 over the 225-class vector) on the **same fixed test set as every other model** (`data/real/annotations_test.json` ∪ synthetic) — no special circularity-safe slice, per strategy doc §1.1. Report per-source composition (reusing the per-class/per-source breakdown machinery already in `8-class_distribution_report.py`/`9-class_distribution_with_reviews.py`) alongside the headline number. |
| `scripts/training/teacher_finetune/run_finetune.py` | Entry point, same `.env`/MLflow/logging wiring pattern as `run_training_pipeline.py`. |

### 2.4 Output and handoff to Goal B

`teacher_finetune/model_exports/<run>/best.pt` (a plain `clf.model.state_dict()`) is the
artifact Goal B needs. Goal B does **not** load this checkpoint live during student
training (§3) — it is used once, offline, to regenerate `speciesnet_results.jsonl`-style
soft labels with the fine-tuned weights (re-running `6-classify_speciesnet.py`'s batching
code with the new checkpoint swapped in, per the strategy doc §1.3's "precompute once and
cache" design — the ~196M-param MD+SN ensemble never runs inside the student's training
loop).

---

## 3. Goal B — KD training (Phase 3, extends Goal A's package)

### 3.1 Structural choice: extend `yolo26n/`, do not fork a new package

The user's instruction is that the KD run use "the same data preparation / config" as the
direct-FT run, for the cleanest possible comparison. The strongest way to guarantee that is
literal code reuse, not parallel maintenance: add a `--kd` mode to the **existing**
`yolo26n/` package (new `kd_dataset.py`, `kd_loss.py`, `--teacher-cache` flag on
`run_training_pipeline.py`) rather than a sibling `yolo26n_kd/` directory. `constants.py`,
`yolo26n_model.py`, `evaluation.py`, and the augmentation pipeline are shared automatically
by construction — there is no second copy of the recipe that could silently drift from the
first. Only `KD_TEMPERATURE` and `KD_ALPHA` are added to `constants.py`, following the
existing `as_dict()`-logged convention (per strategy doc §5's engineering table).

Per the strategy doc §4 (Phase 3), the KD run initializes from **COCO-pretrained weights,
not Phase 1's fine-tuned checkpoint** — `run_training_pipeline.py --kd` must not
default to loading the Goal A run's `best.pt`; make this an explicit, logged choice
(`init_from=coco` vs. `init_from=phase1`) so it can't silently regress to the wrong
init.

### 3.2 Teacher soft-label cache (prerequisite step, run once)

New script `scripts/training/teacher_finetune/cache_soft_labels.py` (lives next to the
teacher, since it needs the fine-tuned `speciesnet` env, not the main env): re-runs
`6-classify_speciesnet.py`'s `SpeciesNetClassifier` batching/threading machinery with the
fine-tuned checkpoint loaded, over the **training-split images only**
(`data/real/annotations_train.json`), writing one record per `(filepath, detection_idx)`
with the **already-225-projected** soft-label vector (`compute_probs_225()` output,
reused directly) rather than the raw 2,498-class scores — this keeps the cache small and
means the student-side dataloader never needs the taxonomy-lookup tables, only a flat
225-vector per detection. Output: `data/real/teacher_soft_labels_train.jsonl`, schema
`{"filepath": ..., "detection_idx": 0, "probs_225": [...], "prob_225_sum": ...}`.

### 3.3 KD dataset — extending, not replacing, `CocoYoloDataset`

`kd_dataset.py`'s `KDCocoYoloDataset` wraps `CocoYoloDataset` (composition, not
inheritance-with-overrides, so the base augmentation/letterbox code path is untouched):
loads the soft-label cache once into a dict keyed by `file_name` (the same key
`CocoYoloDataset` already uses to resolve images), and `__getitem__` returns the base
4-tuple plus a `teacher_probs` tensor (`[225]`, zeros if the image has no cached teacher
record — e.g. it failed MD/SN's own confidence floors at cache time). `collate_fn` gains a
`teacher_probs` stack alongside the existing `imgs`/`targets`/`paths`/`shapes` — additive,
so the base `collate_fn` used by Goal A is unaffected.

### 3.4 KD loss — the central engineering risk of Goal B

This is flagged explicitly because the strategy doc's §3/§5 description ("extends
`loss.py`'s wrapper pattern," "adds the soft-CE/KL term to the existing `YoloLoss`
output") understates the integration difficulty for an anchor-free, task-aligned-assigned
detector. The complication: SpeciesNet's soft label is **one distribution per
image/detection** (its best guess for the one annotated animal), but YOLO26's classification
loss is computed **per dense anchor** after `TaskAlignedAssigner` decides which anchors are
"foreground" for which ground-truth instance. A bolt-on global KL term with no connection
to the assigner has no well-defined anchors to apply it to.

**Recommended mechanism**, grounded in how `v8DetectionLoss` already works
(`ultralytics/utils/loss.py`, `get_assigned_targets_and_loss`): `TaskAlignedAssigner`
already produces a *soft* `target_scores` tensor (IoU-quality-weighted, not one-hot) for
every foreground anchor, which is what the BCE cls loss is computed against. For an
anchor assigned to a GT instance that has a cached teacher `probs_225` vector, **blend the
assigned-class one-hot/IoU-weighted target with the teacher distribution** before the BCE
call:

```
target_scores[anchor, :] = (1 - α) * target_scores[anchor, :] + α * teacher_probs_225[instance]
```

applied only to the 225-class dimension of foreground anchors whose matched GT instance has
a cached teacher record (most will, since the teacher cache covers the same training set);
anchors without a cached record fall back to the unmodified hard-label target. This needs a
small subclass of `v8DetectionLoss` (override the cls-loss section of
`get_assigned_targets_and_loss`, not the whole class) inside `kd_loss.py`, wired into a
`KDE2ELoss` that otherwise delegates to the stock `E2ELoss` for the one-to-many head and
applies the blend only on the one-to-one head (the one actually used at inference) — or
applies it to both; this is a concrete experimental choice to settle during implementation,
not assumed here. Temperature `T` scales the teacher distribution's sharpness before
blending (`softmax(teacher_logits / T)` — note the cache stores **probabilities**, not
logits, so recovering a temperature-scaled distribution from a stored softmax output
requires either caching logits instead of probabilities, or approximating via
re-temperature on the probability simplex; **decide which during cache-script
implementation** in §3.2 — caching logits is more correct and only marginally larger).

**Secondary-detection pseudo-GT (multi-animal KD advantage, §3 of strategy doc) is scoped
as a documented follow-on ablation, not part of Goal B's primary run** — the strategy doc
itself calls this "worth an explicit ablation," not a headline requirement. Building it
means materializing an augmented `annotations_train_kd_multianimal.json` with secondary MD
boxes + their own SpeciesNet soft label appended wherever `multi_animal=true`; defer this
to a follow-on once the primary KD-vs-direct-FT comparison (single-label KD only) is
validated and shows a plausible result.

**Person-class downweighting**: same one-line `model.class_weights` mechanism as Goal A
(§1.3) — already consistent between the hard-label and KD loss branches automatically,
since both read the same `model.class_weights` attribute.

### 3.5 Hyperparameter sweep

`KD_TEMPERATURE ∈ {4, 8}`, `KD_ALPHA ∈ {0.5, 0.7}` per the strategy doc's starting grid —
4 runs. Recommend validating the mechanism end-to-end at one fixed point
(`T=4, α=0.5`) first via `--smoke`, then launching the full grid once the loss is confirmed
numerically sane (KD loss term comparable in magnitude to the hard-label cls loss, not
exploding/vanishing).

---

## 4. Cross-cutting pre-flight checklist (all three goals)

| Item | Status | Action |
|---|---|---|
| `weights/yolo26n.pt` | Missing | Download (Goal A §1.5) |
| `data/synthetic/annotations_test.json` | Missing on this host | `rsync` from wherever it lives (`Makefile` sync targets) before any `mixed`-domain eval |
| `Dockerfile.speciesnet` + Makefile targets | Missing (documented in an abandoned note, never committed) | Build (Goal C §2.3) |
| `reports/lookalike_groups_v2.csv` | **Exists** (built 2026-06-11) | No action — reusable as-is by `eval_suite/grouping.py` |
| `ultralytics==8.4.60`, `yolo26n.yaml`, COCO pretrained weights | **Verified present/downloadable** in this env | No action |
| `speciesnet` package | Not installed in main 3.13 env (expected — needs the new 3.11 image) | Build per Goal C |

---

## 5. Verification

- **Goal A sanity gate**: same as the parent strategy doc §6 — the direct-FT run must
  reach a non-degenerate `mAP50_95` on val before KD is attempted. Additionally: the
  loss-adapter and eval-decode smoke checks in §1.3/§1.4 must pass *before* the sanity
  gate, since a silently-wrong decode would otherwise produce a plausible-looking but
  meaningless mAP number.
- **Goal C success criterion**: unchanged from the parent doc §6 — species accuracy/F1 on
  the 225-class projection improves over the off-the-shelf checkpoint, measured on the
  same fixed test set as every other model, with per-source composition reported and the
  circularity caveat (§1.1 of the parent doc) stated explicitly.
- **Goal B comparison**: both the direct-FT (Goal A) and KD (Goal B) checkpoints run
  through the **same** `eval_suite/run_evaluation.py` — no bespoke evaluation path for KD.
  Report the headline `mAP50_95` delta; per the parent doc, a KD result that underperforms
  direct-FT is a valid, reportable finding, not a bug to chase away.
- **No duplication check**: this plan does not re-specify the evaluation-axis design
  (`docs/plans/2026-06-10_model-evaluation-strategy.md`) or quantization/QAT
  (`docs/2026-03-10_object-detection-models-for-embedded-systems.md`) — it only wires new
  checkpoints into the first and explicitly excludes the second.
