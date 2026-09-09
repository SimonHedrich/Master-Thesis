# SpeciesNet fine-tune: freeze-fraction sweep finds a clear improvement at 0.75, disqualifies full fine-tune (0.0)

## Motivation

The existing SpeciesNet classifier fine-tune (§4.1, 2026-08-05/06) plateaus by epoch 4
(val `f1_macro=0.7645`) and is documented as a "genuine ceiling for this
LR/architecture/freeze-fraction setup, not noise." `FREEZE_PARAM_FRACTION=0.5` was never
actually tuned — the implementation plan
(`docs/plans/2026-06-30_yolo26-kd-and-teacher-finetune-implementation-plan.md` §2.2)
explicitly flagged it as an unresolved guess ("the user was asked... did not respond in
time"). This session ran a small, staged sweep on that one knob: cheap 5-epoch probes at
`{0.0, 0.75}` first (bracketing the existing 0.5), promoting only stable/competitive
candidates to a full run.

## Code change

`scripts/training/teacher_finetune/run_finetune.py` had no CLI override for freeze
fraction or epoch count (`teacher_model.py::speciesnet_model()` already accepted
`freeze_fraction` as a parameter — the gap was purely at the CLI entry point). Added
`--freeze-fraction FLOAT` and `--epochs INT` (both optional, default `None` = existing
behavior unchanged) to `run_finetune.py`, and a matching `--freeze-fraction` override to
`find_max_batch_size.py` (needed once the 0.0 probe hit an OOM — see below). Verified the
no-flags path reproduces the original run configuration exactly (same params logged to
MLflow, same `run_name` format); overridden runs get a `probe-`/`ff{X}-`/`bs{N}-` suffix
in their run name for traceability.

## Probe 1: freeze-fraction 0.0 (full fine-tune) — disqualified

First attempt (`BATCH_SIZE=32`, the default) OOM'd ~6 minutes into epoch 1 — not
immediately, but right as it hit the first AMP-overflow/non-finite-loss event. The
existing `find_max_batch_size.py` tool (previously used to validate the current
`BATCH_SIZE=32` default) said batch 19 should fit at `freeze_fraction=0.0`, but that tool
doesn't exercise the training loop's `GradScaler`, EMA shadow copy, or the finite-check
guard's snapshot/restore path — its number wasn't representative of real training memory
pressure once a hard batch triggers the guard's recovery logic. Retried at
`--batch-size 8`, which fit within VRAM but revealed the real problem: **catastrophic
divergence, not a memory issue**. 21,463/23,463 batches (91.5%) were already non-finite
in epoch 1, reaching 100% skip rate by epoch 2, with val `f1_macro` collapsing to
near-floor (0.018 → 0.003 → 0.0015) — essentially zero learning across all 5 probe
epochs. This is the expected, well-known failure mode of unfreezing a large pretrained
backbone at an LR (`ONE_CYCLE_MAX_LR=3e-4`) tuned for a much smaller trainable-parameter
regime — full fine-tuning needs a substantially lower LR, which this sweep deliberately
did not also vary (single-variable-per-run principle, to avoid confounding the
freeze-fraction signal). **Disqualified — no full run.**

## Probe 2: freeze-fraction 0.75 — promoted

Ran cleanly at the default `BATCH_SIZE=32` (fewer trainable params than even the 0.5
baseline). Zero non-finite batches across all 5 epochs, and beat the baseline's own
epoch-by-epoch val `f1_macro` at every checkpoint:

| Epoch | Baseline (ff=0.5) | Probe (ff=0.75) |
|---|---|---|
| 1 | 0.7143 | 0.7212 |
| 2 | 0.7433 | 0.7651 |
| 3 | 0.7569 | 0.7786 |
| 4 | 0.7645 | 0.7778 |
| 5 | — | 0.7728 (declining) |

Already exceeded the baseline's peak (0.7645) by epoch 3, with a clear peak-then-decline
shape suggesting the best epoch was close by. **Promoted to a full run.**

## Full run: freeze-fraction 0.75

Early-stopped at epoch 20 (patience=15, best at epoch 5 — 1-indexed training-loop
numbering; checkpoint metadata records it as epoch 4, a benign off-by-one between the two
logging points, not two different epochs). **Not perfectly stable throughout**: epoch 20
itself (the epoch that triggered early-stop) hit the same late-stage non-finite-batch
storm pattern documented in the original baseline and the disqualified 0.0 probe
(5,800+ skipped batches) — but this happened *after* patience had already exhausted and
`best.pt` was locked in from epoch 5, so it never touched the deployed checkpoint. This is
exactly the scenario the finite-check guard (§4.1's bug fix) was built to protect against,
working as intended.

**Result — clearly better on every comparison-gate metric:**

| Metric | Baseline (ff=0.5) | New (ff=0.75) | Δ |
|---|---|---|---|
| val f1_macro (best epoch) | 0.7645 | 0.7864 | +0.0219 |
| test f1_macro | 0.5390 | 0.5588 | +0.0198 |
| test accuracy_top1 | 0.6688 | 0.6870 | +0.0182 |

Checkpoint: `scripts/training/teacher_finetune/model_exports/teacher-finetune-ff0.75-20260908-075032/best.pt`
(gitignored, only on this A40, not yet rsynced to NAS per §1.3).

## Downstream refresh (per the "only if clearly better" policy)

Re-cached KD teacher soft labels (`cache_soft_labels.py --split {train,val}` against the
new checkpoint — 187,705 / 19,732 records, both matching the original counts exactly) and
re-ran the MD+SN ensemble prediction + scoring
(`predict_ensemble.py` → `run_evaluation.py`, output at
`scripts/training/megadet_speciesnet_ensemble/model_exports/finetuned-teacher-finetune-ff0.75-20260908-075032/eval/`).

Two transient environment issues along the way, both resolved without a code change:
MegaDetector's weight download hit a persistent `HTTP 504` from Zenodo on two consecutive
attempts inside `training-container` — turned out the weights were already cached on the
*host* (`~/.cache/torch/hub/checkpoints/md_v5a.0.0.pt`, presumably from an out-of-container
run), just not visible from inside the container's own ephemeral cache path (lost when the
container was recreated during the prior KD session's DNS fix). Copied the host's cached
file in via `docker cp` rather than re-fighting Zenodo.

**Ensemble result — improves across every band, not just the headline:**

| Metric | Old ensemble (ff=0.5) | New ensemble (ff=0.75) | Δ |
|---|---|---|---|
| mixed mAP | 0.549 | 0.567 | +0.018 |
| real mAP | 0.536 | 0.553 | +0.017 |
| Band A mAP (mixed / real) | 0.013 / 0.008 | 0.027 / 0.014 | ~2× |
| Band B mAP (mixed / real) | 0.630 / 0.574 | 0.654 / 0.599 | +0.024 / +0.025 |
| Band C mAP (mixed / real) | 0.634 / 0.605 | 0.653 / 0.631 | +0.019 / +0.026 |
| Band D mAP (mixed / real) | 0.823 / 0.801 | 0.840 / 0.822 | +0.017 / +0.021 |

Unlike the original pretrained→fine-tuned transition (§4.1), which traded away Band A
performance for gains on B/C/D (`docs/2026-09-07_model-comparison-teacher-and-students.md`
finding 3), this 0.5→0.75 change is a **uniform improvement across all four bands
simultaneously** — a cleaner win, not a trade-off.

**Important caveat on the Band A numbers, discovered independently in this session's
timeframe (TODO.md §4.7):** `teacher_finetune`, like every other training pipeline in this
campaign, hardcodes its training data source to `data/real/annotations_{train,val}.json`
only (`constants.py`) — it never loads `data/synthetic/annotations_{train,val}.json`,
where all ~8,000 Band-A synthetic training images actually live. **Both the ff=0.5 and
ff=0.75 SpeciesNet checkpoints have had literal zero training exposure to all 50 Band-A
species.** The Band A improvement reported above is real (better calibration/less
aggressive drift from the pretrained model's own zero-shot Band-A competence, likely
because a higher freeze fraction retains more of the pretrained backbone) — but it is
**not** evidence that freeze-fraction improves long-tail generalization from more data,
since no additional Band-A data was involved on either side of this comparison. All Band A
numbers in this note (and in the pre-existing model-comparison doc) should be read with
that caveat until §4.7 is fixed and the affected models retrained.

## Next steps

- **TODO.md §4.1** updated in place with this result.
- **TODO.md §4.7** updated to include `teacher_finetune` in its list of affected
  pipelines (previously only named YOLOv5s/YOLO26n) — same root cause, same fix needed
  (wire `data/synthetic/annotations_{train,val}.json` into the dataloader construction).
- The new checkpoint's numbers should be reflected in
  `docs/2026-09-07_model-comparison-teacher-and-students.md` (§3.1/TL;DR) — that file is
  someone else's in-progress work in this same checkout as of this session, so left
  untouched here rather than editing concurrently; flagging for whoever owns it next.
- Per repo convention, if a `(T,α)` KD grid or any other downstream re-run happens later,
  it should now use this new checkpoint's soft-label cache (already refreshed) rather than
  the old one.
