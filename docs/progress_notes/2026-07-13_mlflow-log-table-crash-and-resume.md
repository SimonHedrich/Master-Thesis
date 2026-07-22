# MLflow `log_table` tag overflow crash + resume-from-checkpoint support

**Date of incident:** 2026-07-05 (run `yolov5s-20260629-235646`)
**Date of fix:** 2026-07-13

## What happened

The first full YOLOv5s training run (`yolov5s-20260629-235646`, MLflow run
`cc54f94eceb640cd9ce16f85de353869`) crashed at **epoch 137/200**, immediately
after a successful validation pass (val mAP50=0.5828, mAP50_95=0.4885):

```
ERROR __main__ | run failed: Expecting value: line 1 column 8001 (char 8000)
  ...
  File ".../mlflow/tracking/client.py", line 3498, in log_table
    current_tag_value = json.loads(run.data.tags.get(MLFLOW_LOGGED_ARTIFACTS, "[]"))
json.decoder.JSONDecodeError: Expecting value: line 1 column 8001 (char 8000)
```

Training itself was healthy — the failure was entirely inside MLflow logging.

## Root cause

`eval_log_mlflow()` logged the per-class AP table once per epoch via
`mlflow.log_table()` with a unique filename per epoch
(`val_per_class_ap_step0.json`, `step1.json`, …).

`mlflow.log_table` is not a plain artifact upload: it also maintains a hidden
run tag, `mlflow.loggedArtifacts`, holding a JSON list of every table ever
logged to the run, and **reads + re-parses that tag on every call** to append
the new entry.

MLflow caps tag values at **8000 characters** (`MAX_TAG_VAL_LENGTH` in
`mlflow/utils/validation.py`), and with `MLFLOW_TRUNCATE_LONG_VALUES`
behaviour the server silently **truncates** an over-long value instead of
rejecting it. Truncating a JSON array mid-entry leaves invalid JSON, so the
next `log_table` call died parsing it — at exactly char 8000.

The arithmetic pins the epoch: each tag entry is ~58 chars
(`{"path": "val_per_class_ap_step136.json", "type": "table"}` + comma),
and 8000 / 58 ≈ **137** entries. A latent time bomb armed at run start,
guaranteed to detonate at epoch ~137 of any sufficiently long run.

## Consequences

- The `mlflow.loggedArtifacts` tag of run `cc54f94e…` is permanently
  corrupted server-side: any future `log_table` call against that run fails
  immediately. Metrics, params and plain artifacts still work.
- Epoch 137's val result (0.4885, which would have been a new best) was lost:
  the crash happened *before* the best-checkpoint block in `run_pipeline()`.
- `last.pt` was never written — at the time it was only saved *after* the
  training loop — so the only surviving checkpoint is `best.pt` from epoch
  136 (1-indexed; `epoch=135` 0-indexed inside the checkpoint,
  mAP50_95=0.4859).

## Fixes (all in `scripts/training/yolov5s/`)

1. **`evaluation.py` — no more `log_table`.** The per-class AP table is now
   written as a plain JSON file and uploaded with `mlflow.log_artifact()`
   under the `per_class_ap/` artifact path. `log_artifact` never touches the
   bookkeeping tag, so there is no growth and no 8000-char ceiling. The whole
   block is additionally wrapped in `try/except` — a logging hiccup must
   never again abort a multi-day training run. (Trade-off: the tables no
   longer appear in the MLflow UI's "Tables" evaluation view, only as
   artifacts — acceptable, they are diagnostics.)
2. **`training_pipeline.py` — `last.pt` is refreshed every epoch**, not only
   after the loop, so any future crash leaves an up-to-date resume point.
3. **`training_pipeline.py` + `run_training_pipeline.py` — resume support.**
   New `--resume-from <ckpt>` CLI flag; `TrainingPipeline` restores model
   weights, EMA copy + update count, optimizer, scheduler (incl. plateau
   state/LR), AMP scaler, `best_metric`, early-stop patience counter, and
   continues at `epoch + 1` under a **new** run dir and a **new** MLflow run
   (the old run's tag corruption makes reusing it unattractive anyway).
   `global_step` is reconstructed as `start_epoch × len(dl_train)` so step
   curves line up. The previous `best.pt` is copied into the new run dir so
   the final test eval works even if the resumed run never improves.
   The checkpoint path is logged as the `resume_from` MLflow param.
4. **`Makefile` — `YOLOV5S_ARGS` pass-through** for the `yolov5s-train`
   target.

### Known approximation on resume

Checkpoints store the deployable (EMA) weights only. On resume, both the raw
model and the EMA copy restart from the EMA weights; the raw pre-EMA weights
are unrecoverable. This is the standard practice trade-off and is benign for
fine-tuning continuation.

## Resuming the interrupted run

```bash
make yolov5s-train \
  YOLOV5S_ARGS="--resume-from scripts/training/yolov5s/model_exports/yolov5s-20260629-235646/best.pt"
```

This continues at epoch 137/200 (retraining the crashed epoch) with the
inherited best mAP50_95=0.4859 and a reset-at-crash patience counter.

## Lessons

- Treat experiment-tracking calls as untrusted I/O inside long training
  loops: anything non-essential gets a `try/except`.
- Per-epoch `log_table` calls (or anything that appends to MLflow run tags)
  have a hard hidden budget of ~8000 chars — avoid per-epoch unique entries.
- Checkpoint for *resume* (every epoch), not just for *export* (end of run).
