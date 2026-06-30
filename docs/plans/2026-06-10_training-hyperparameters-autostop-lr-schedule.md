# YOLOv5s Training: Hyperparameter Review, Auto-Stop & LR-Schedule Analysis

**Date:** 2026-06-10
**Status:** Implemented 2026-06-11 — recommendations #1–#6 of the TL;DR are live
(early stopping + `ReduceLROnPlateau` + single `SELECTION_METRIC` + raised ceiling;
EMA; AMP; dead warmup constants removed; lr-logging step-semantics fixed). The
optional `lr0` 1e-3→2e-3 sweep (#7) and the AdamW-betas fix remain open.
**Scope:** `scripts/training/yolov5s/` (custom training loop, post-augmentation)
**Triggered by:** augmentation now wired (see
`docs/progress_notes/2026-06-09_contamination-flagging-and-augmentation-implementation.md`);
need (a) a sanity pass on hyperparameters, (b) an auto-stop so runs end when they
stop improving, (c) a verdict on whether the cosine schedule is the right
LR-control mechanism or whether a more dynamic method fits better.

**Guiding principle (from the request):** prefer the *simpler, more reliable*
option over the SOTA one. Maintenance cost and bug surface are first-class
criteria. Everything below is weighed on "impact vs. complexity vs. risk", not on
"what gets the last 0.5 mAP".

---

## 0. TL;DR — recommendation

| # | Change | Impact | Complexity | Risk | Verdict |
|---|--------|--------|-----------|------|---------|
| 1 | **Early stopping** (patience on val metric) + raise `EPOCH_COUNT` to a high ceiling | High (the explicit ask) | Low | Low | **Do it** |
| 2 | **Switch cosine → `ReduceLROnPlateau`** (metric-driven), keep a manual linear warmup | High (required for #1 to work) | Low–Med | Low | **Do it** |
| 3 | **Add EMA** (weight averaging) | High for YOLO quality/stability | Low (reuse yolov5 `ModelEMA`) | Low | **Recommended** |
| 4 | **Add AMP** (mixed precision) | High for *wall-clock* (≈1.5–2× faster) | Low | Low–Med | **Recommended** |
| 5 | Make the *best-checkpoint*, *plateau* and *early-stop* metric **the same** (`mAP50_95`) | Med (coherence) | Low | Low | **Do it** |
| 6 | Remove/wire the dead warmup constants (`WARMUP_MOMENTUM`, `WARMUP_BIAS_LR`) | Low (correctness/clarity) | Low | Low | **Do it** |
| 7 | Raise `lr0` slightly (1e-3 → ~2e-3) given the head trains from scratch | Low–Med | Low | Low | Optional / sweep |
| 8 | Resume-safety + LR logging step-semantics cleanup | Low | Low | Low | Nice-to-have |

The headline: the current cosine schedule is fine *in isolation* but **structurally
incompatible with the auto-stop requirement**. The cleanest, lowest-risk way to get
"train as long as it's improving, then stop" is **`ReduceLROnPlateau` + early
stopping, both keyed off the same validation metric**. That pairing is in the
PyTorch standard library, needs no horizon guess, and has almost no moving parts.

---

## 1. What the pipeline does today (ground truth)

Read from `constants.py`, `yolov5s_model.py`, `training_pipeline.py`,
`run_training_pipeline.py`:

- **Data:** train = 145,764 images / 178,924 anns; val = 12,545 / 18,445;
  test = 63,822 / 86,785. 225 classes (≈50 with zero train anns → permanently
  unpredictable, depresses macro-mAP — known, see prior plan §8).
  Batch 32 → **≈4,555 steps/epoch**; 50 epochs → ≈228k steps. Large dataset.
- **Optimizer:** SGD, `lr0=1e-3`, `momentum=0.937`, `weight_decay=5e-4`,
  `nesterov=True`. Three param groups (BN no-decay / conv decay / bias no-decay) —
  matches the YOLOv5 reference grouping. ✅
- **Scheduler:** hand-rolled `LambdaLR`: linear warmup over `WARMUP_EPOCHS=3`
  (`(epoch+1)/warmup`), then cosine from `lr0` → `lr0*LRF` (`LRF=0.01`) over the
  remaining epochs.
- **Loss gains:** `box=0.05, cls=0.5, obj=1.0`, `anchor_t=4.0`, `fl_gamma=0`,
  `label_smoothing=0` — YOLOv5 reference defaults. ✅
- **Loop:** fixed `EPOCH_COUNT=50`. Each epoch: train → eval val → `scheduler.step()`
  → save `best.pt` if `val mAP50` improved. After the loop: `last.pt`, final test eval.
- **No EMA, no AMP, no gradient clipping, no early stopping.**
- **Init:** COCO-pretrained backbone/neck loaded; the **detect head is
  shape-mismatched (80→225 classes) and silently dropped** → the head is trained
  **from scratch**. This matters for LR/warmup choices (see §2.3).

---

## 2. Hyperparameter sanity pass

### 2.1 What is sound and should be left alone
- **SGD + momentum 0.937 + wd 5e-4 + nesterov** — the canonical YOLOv5 recipe; no
  reason to deviate for a baseline. Changing the optimizer family is a bigger,
  riskier lever than the task needs.
- **Param-group split** (BN/conv/bias) is correct and matches the reference.
- **Loss gains** are reference defaults and appropriate; not worth tuning before a
  baseline exists.
- **Image size 640, batch 32** — standard, fits the proxy-hardware narrative.
- **Eval thresholds** (`conf=0.001, iou=0.6, max_det=300`) — these are the COCO-mAP
  evaluation conventions, correct for *measuring* (not for deployment). ✅

### 2.2 Latent issues / dead config (low risk to fix, worth fixing)
1. **`WARMUP_MOMENTUM=0.8` and `WARMUP_BIAS_LR=0.1` are never used.** The scheduler
   applies the same linear ramp to all param groups and never touches momentum.
   These constants get logged to MLflow as if they were active → **misleading run
   provenance**. Either implement the real YOLOv5 warmup (bias group starts at
   `warmup_bias_lr` and decays; momentum ramps from `warmup_momentum`→`momentum`)
   or delete the two constants. *Recommendation: delete them.* The classic
   per-iteration bias-LR/momentum warmup is extra moving parts for marginal gain on
   a fine-tune; a plain linear LR warmup is enough and far simpler.
2. **`train/lr` is logged with two different step semantics** — `step=global_step`
   inside the epoch (`training_pipeline.py:98`) and `step=epoch` after
   `scheduler.step()` (`:178`). Same metric key, mixed x-axis → the MLflow chart is
   garbled. Pick one (log lr only at step granularity, or use a separate
   `train/epoch_lr` key).
3. **AdamW branch is half-wired:** `betas=(MOMENTUM, 0.999)` feeds `0.937` as β₁
   (should be ~0.9) and `lr=1e-3` is high for AdamW fine-tuning. SGD is the default
   so this is dormant, but if anyone flips `OPTIMIZER="AdamW"` it will train poorly.
   Either fix the betas/lr coupling or document that the AdamW path is unsupported.
4. **No nominal-batch-size scaling.** Reference YOLOv5 scales `wd` by
   `batch/nbs(=64)` and ramps lr accordingly. We hard-code the reference values at
   batch 32. This is *acceptable* (values are within a sane band) but means the
   numbers aren't the "official" effective values — note it in the thesis methods,
   don't bother implementing the scaling.

### 2.3 The init detail that should shape LR/warmup choices
Because the **detect head is reinitialised from scratch** (80→225 mismatch dropped
in `yolov5s_model.py:49`), this run is a *hybrid*: a fine-tune of the
backbone/neck **plus** from-scratch training of the head. Implications:
- A pure-fine-tune `lr0` (very low) can starve the fresh head; a from-scratch `lr0`
  (0.01) can wreck the pretrained backbone. `1e-3` is a reasonable middle, but
  **slightly higher (≈2e-3) is worth a quick A/B** — the head has the most to learn
  and warmup protects the backbone early. Keep this as an optional sweep, not a
  baseline change.
- This is also why **warmup matters** and should be kept (don't drop it when
  switching schedulers).

### 2.4 Missing training-quality pieces (impact vs. complexity)
- **EMA (exponential moving average of weights) — recommended.** YOLOv5's published
  numbers rely on EMA; it materially stabilises val-mAP curves and usually adds a
  bit of final mAP, *and it makes early-stopping decisions less noisy* (the EMA mAP
  curve is smoother, so "no improvement" is more trustworthy). Complexity is low:
  reuse `yolov5.utils.torch_utils.ModelEMA` (already a dependency), update it after
  each `optimizer.step()`, and **evaluate / checkpoint the EMA weights**. Risk:
  must remember to save EMA (not raw) weights as `best.pt`, and EMA state isn't in
  the current checkpoint dict. Net: high value, low complexity, low-but-nonzero
  risk → **recommended, implement carefully**.
- **AMP (mixed precision) — recommended for wall-clock.** The task explicitly
  worries that "no one knows when training finishes". 145k images × up-to-100+
  epochs is the real cost driver. `torch.cuda.amp.autocast` + `GradScaler` is ~10
  lines, ~1.5–2× faster on the GPU, low risk. Only caveat: NaN-loss under fp16 is
  possible (GradScaler handles it; just don't also add manual grad-clip without
  unscaling first). → **recommended**.
- **Gradient clipping — skip.** Not part of the reference recipe; adds a knob with
  little benefit here. Only add if AMP shows instability.

### 2.5 Background: what EMA and AMP actually are
These two are referenced above as "recommended" bolt-ons. Neither changes the
training *logic*; they are standard add-ons that stock YOLO training includes by
default. Spelled out here so the recommendation is self-contained.

**EMA — Exponential Moving Average (of weights).** Instead of
evaluating/checkpointing the model's *current* weights, keep a second copy that is
a smoothed running average of the weights seen during training. After each
optimizer step:

```
ema_weight = decay * ema_weight + (1 - decay) * current_weight     # decay ≈ 0.9999
```

So the EMA weights track where the model has been over the last few thousand steps
rather than where the last noisy gradient landed.
- *Why it helps:* SGD bounces around the minimum step-to-step (augmentation adds
  noise); the average position is usually a flatter, better point than any single
  step. YOLOv5's published mAP relies on it — a small, near-free accuracy gain.
- *Why it matters for the auto-stop (§3):* the EMA val-mAP curve is much smoother
  than the raw one, so "has it stopped improving?" becomes a far more trustworthy
  signal — fewer false plateaus and fewer premature stops.
- *Cost/risk:* low. Reuse `yolov5.utils.torch_utils.ModelEMA` (already a
  dependency): build it, call `ema.update(model)` after each step, and
  **evaluate + checkpoint the EMA weights, not the raw ones**. The one gotcha is
  exactly that — `best.pt` must be the EMA copy, and EMA state must be added to the
  checkpoint dict.

**AMP — Automatic Mixed Precision.** By default PyTorch computes in 32-bit floats
(fp32). AMP runs most ops in 16-bit (fp16/bf16) automatically, keeping the few that
need full range (e.g. loss accumulation) in fp32:

```python
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    preds = model(imgs)
    loss = loss_fn(preds, targets)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

- *Why it helps:* 16-bit math is ~2× faster on the GPU and uses ~half the memory.
  This is a pure **wall-clock** win — it does **not** change what the model learns,
  only how fast each step runs. With 145k images × potentially 100+ epochs, it is
  the single biggest lever on the "no one knows when it finishes" worry.
- *Cost/risk:* low–moderate, ~10 lines. fp16 has a narrow numeric range and can
  under/overflow to NaN; `GradScaler` exists precisely to prevent that (scales the
  loss up before backprop, down after). Caveat: if manual gradient clipping is ever
  added, `scaler.unscale_()` must be called first — moot here since §2.4 recommends
  no clipping.

*One line each:* **EMA** = smoothed average of weights → slightly better, more
stable model + a cleaner early-stop signal (accuracy/stability). **AMP** = 16-bit
math → ~2× faster training, same result (speed). Both are independent of the core
scheduler/auto-stop change and each is testable on its own (see §8).

---

## 3. Auto-stop (early stopping)

### 3.1 Requirement
"Run as long as there are improvements, then stop." This is textbook **patience
-based early stopping** on a validation metric.

### 3.2 Design (simple, reliable)
- Track the **same metric used to pick `best.pt`** (see §5 — recommend `mAP50_95`).
- Keep `best_metric` and `epochs_since_improve`. On each val eval:
  - if `metric > best_metric + min_delta`: update best, save `best.pt`, reset counter;
  - else: increment counter.
- If `epochs_since_improve >= PATIENCE`: stop the loop, then run the final test eval
  on `best.pt` (not the last, possibly-overfit weights).
- **Raise `EPOCH_COUNT` to a high ceiling** (e.g. 150–200) so the *patience*, not the
  fixed count, is what ends training. The ceiling is just a safety cap.
- **`min_delta`** (e.g. 0.001 mAP) avoids stopping-resets on pure noise.

### 3.3 Patience value
YOLOv5's default `patience=100` assumes 300-epoch schedules; scaled to this dataset
and a metric-driven LR (below), **patience ≈ 15–25 epochs** is reasonable. With
`ReduceLROnPlateau`, set **early-stop patience > LR-reduction patience** (e.g. LR
patience 5, stop patience 15) so the run gets *two or three* LR drops to react to a
plateau before it gives up. This coupling is the crux of §4.

### 3.4 Complexity / risk
~15 lines in `TrainingPipeline.run_pipeline`, no new dependency, no new failure
mode beyond "stops too early on a noisy metric" — mitigated by `min_delta`, EMA
smoothing (§2.4), and a sane patience. **Low risk.**

---

## 4. LR control: is cosine the right mechanism?

### 4.1 The core problem
A **cosine schedule is defined over a fixed horizon** `EPOCH_COUNT`. It anneals
`lr0 → lr0*LRF` across exactly that many epochs. This is great when the horizon is
known — but it **does not compose with the auto-stop requirement**:

- If `EPOCH_COUNT` is set to a *high ceiling* (so early-stop decides the end), the
  cosine decays slowly; lr stays high; the model may hit the early-stop patience
  **before the schedule ever anneals into the fine, low-LR regime** where the last
  mAP gains come from. You stop with a half-finished schedule.
- If `EPOCH_COUNT` is set *short* (so the cosine finishes), then the fixed count —
  not the "still improving?" signal — governs the run length, which **defeats the
  point of early stopping**.

You can't have both a fixed-horizon schedule and a horizon-agnostic stop. So the
schedule should become **horizon-agnostic too**.

### 4.2 Options considered

| Method | Needs fixed horizon? | Composes w/ early-stop? | Stdlib? | Complexity | Notes |
|--------|:---:|:---:|:---:|---|---|
| Cosine (current `LambdaLR`) | **Yes** | ✗ | ✓ | Low | Best peak *if* horizon known; conflicts with auto-stop |
| **`ReduceLROnPlateau`** | **No** | **✓** | **✓** | **Low** | Drops lr ×factor when val metric plateaus; same signal as early-stop |
| OneCycle | Yes | ✗ | ✓ | Med | Aggressive; horizon-bound; wrong fit |
| Cosine warm restarts (SGDR) | Partial (cycle len) | ~ | ✓ | Med | More knobs, marginal gain, harder to reason about |
| Step/MultiStep decay | No (but fixed milestones) | ~ | ✓ | Low | Milestones are just a guessed horizon in disguise |

### 4.3 Recommendation: `ReduceLROnPlateau` + manual warmup
**`ReduceLROnPlateau` is the natural partner for "train until no improvement".** It
watches the **same validation metric** as early stopping; when the metric plateaus
for `lr_patience` epochs it multiplies lr by `factor` (e.g. 0.5) down to `min_lr`.
No horizon to guess. The LR machinery and the stop machinery read one signal, so
their behaviour is easy to reason about and to log.

Concretely:
1. **Keep a manual linear warmup** for the first `WARMUP_EPOCHS` (set lr directly on
   the optimizer per epoch; the head-from-scratch init in §2.3 needs it). Warmup
   and `ReduceLROnPlateau` don't chain cleanly via `SequentialLR` (the plateau
   scheduler needs a metric, which `SequentialLR` can't pass), so a tiny manual
   warmup branch is the *simplest* correct construction — far less fragile than
   forcing it into `SequentialLR`.
2. After warmup, each epoch call `plateau.step(val_metric)` (note: **metric**, not
   `.step()` — different signature from the current call site; the existing
   `self.scheduler.step()` in `training_pipeline.py:176` must change).
3. Suggested defaults: `mode="max"` (mAP is higher-better), `factor=0.5`,
   `patience=5`, `min_lr=lr0*LRF` (≈1e-5), `threshold≈min_delta`.
4. Early-stop patience (15–25) > plateau patience (5) so the model gets ~2–3 lr
   drops before the run ends.

**Why not just keep cosine with a fixed horizon and no early stop?** That's the
*other* legitimate answer and it's even simpler (zero code change). But it directly
contradicts the explicit requirement ("runs as long as there are improvements" +
"no one knows when it finishes"). Cosine-with-fixed-horizon means *committing to a
horizon up front* — exactly what the request rules out.

### 4.4 Honest trade-off
A well-tuned cosine over a *correctly guessed* horizon often reaches a slightly
higher peak than `ReduceLROnPlateau`, because its annealing is smooth and ends at a
very low lr. `ReduceLROnPlateau` is a hair less optimal at the very end and its
drops are reactive (a few wasted epochs detecting each plateau). **For this use
case the robustness of not needing a horizon outweighs that small peak
difference** — and it's the only option that satisfies the auto-stop requirement
without a horizon guess, at standard-library complexity. If, after a baseline, the
thesis wants the absolute best single number on a *known* good epoch budget, a
final cosine run over that now-known horizon is a cheap follow-up.

---

## 5. Make the three metrics one metric
Currently `best.pt` is chosen on **`mAP50`**. Early stopping and `ReduceLROnPlateau`
must watch *something*; if they watch a different metric than the checkpoint
selector, behaviour gets hard to reason about. **Use one metric for all three:**

- Recommend **`mAP50_95`** — it is the primary COCO metric and the number the
  thesis reports, so "best checkpoint" = "best reported result". It is also less
  saturated/jumpy than `mAP50` late in training.
- Counter-argument: `mAP50_95` can be very low/noisy in the *first* few epochs.
  Mitigations: warmup epochs are excluded from patience counting anyway, EMA
  smooths it, and `min_delta` filters noise. `mAP50` is an acceptable fallback if
  early curves look too noisy in practice.

Pick one, wire it into checkpoint-selection + plateau + early-stop identically.

---

## 6. Proposed constant changes (for when implementing — not done here)

```python
# ─── Scheduler ───
WARMUP_EPOCHS      = 3
LR_SCHEDULE        = "plateau"   # was implicit "cosine"
PLATEAU_FACTOR     = 0.5
PLATEAU_PATIENCE   = 5
PLATEAU_MIN_LR     = 1e-5        # ≈ lr0 * LRF
# LRF kept only if a cosine fallback is retained; otherwise remove
# WARMUP_MOMENTUM / WARMUP_BIAS_LR  → DELETE (dead config, §2.2)

# ─── Auto-stop ───
EPOCH_COUNT        = 200         # safety ceiling, NOT the expected length
EARLY_STOP         = True
EARLY_STOP_PATIENCE = 20
EARLY_STOP_MIN_DELTA = 0.001
SELECTION_METRIC   = "mAP50_95"  # best.pt + plateau + early-stop all use this

# ─── Training-quality (recommended) ───
USE_EMA            = True
USE_AMP            = True

# ─── Optional sweep ───
LEARNING_RATE      = 1e-3        # try 2e-3 (head trains from scratch, §2.3)
```

Touch-points if implemented:
- `yolov5s_model.py::model_scheduler` — replace `LambdaLR` with a warmup-aware
  `ReduceLROnPlateau` construction (or return both warmup fn + plateau scheduler).
- `training_pipeline.py::run_pipeline` — `scheduler.step(val_metric)`; early-stop
  loop break; EMA update + EMA-based eval/checkpoint; AMP autocast/scaler in
  `_train_one_epoch`; single-source `SELECTION_METRIC`.
- `training_pipeline._save_checkpoint` — include EMA state + `best_metric` +
  `epochs_since_improve` for resume-safety (§2.2 item, optional).
- `run_training_pipeline.py` — pass new constants through.

---

## 7. What I deliberately do NOT recommend
- **OneCycle / SGDR / cosine-warm-restarts** — more knobs, horizon-bound or
  cycle-bound, higher cognitive + bug cost; the robustness win over
  `ReduceLROnPlateau` is not worth it for a baseline.
- **Per-iteration bias-LR + momentum warmup** (full YOLOv5 warmup) — extra moving
  parts for marginal fine-tune gain; a plain linear LR warmup is enough.
- **Optimizer change / loss-gain tuning** — premature before a clean baseline exists.
- **Gradient clipping** — only if AMP proves unstable.

## 8. Suggested sequencing
1. **Early stopping + `ReduceLROnPlateau` + single metric + raise ceiling** (the
   core ask; one coherent change). Verify on a short `--smoke` run that the loop
   breaks, lr drops on plateau, and `best.pt` is the best-metric checkpoint.
2. **EMA**, then **AMP** — independent, each verifiable on its own; add after the
   control-flow change is proven so failures are easy to localise.
3. *(Optional, later)* `lr0` 1e-3 vs 2e-3 A/B once the harness is stable.

Each step is independently testable and independently revertible — which is the
whole point of keeping the mechanisms simple.
```
