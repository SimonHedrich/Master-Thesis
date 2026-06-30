# Progress Note — Evaluation Suite Implementation (orchestrated)

**Date:** 2026-06-11
**Implements:** `docs/plans/2026-06-10_model-evaluation-strategy.md` via
`docs/plans/2026-06-11_evaluation-script-implementation.md`

## What was built

A modular, standalone evaluation suite at
`scripts/training/yolov5s/eval_suite/` that scores a saved YOLOv5s checkpoint
across the strategy doc's axes (granularity × test-domain × training-band) and
emits the Tier 1/2/3 report (Markdown + CSV + JSON + MLflow). It runs
independently of training and also as an optional post-training hook
(`run_training_pipeline.py --full-eval`).

| File | Role |
|------|------|
| `eval_suite/predict.py` | checkpoint + COCO annotations → cached COCO predictions JSON |
| `eval_suite/scoring.py` | remap+filter+torchmetrics COCO-12; band/domain/confusion/bootstrap helpers |
| `eval_suite/grouping.py` | fine/coarse/detect remaps + class→band map |
| `eval_suite/report.py` | assemble Tier 1/2/3 → Markdown/CSV/JSON/MLflow |
| `eval_suite/run_evaluation.py` | CLI + `evaluate_checkpoint()` programmatic entrypoint |
| `scripts/dataset_quality/16-build_lookalike_groups.py` | builds `reports/lookalike_groups.csv` (coarse grouping) |

## Orchestration

Three Sonnet subagents built `grouping`, `predict`, and `scoring` in parallel
against frozen data contracts (cached-predictions JSON schema; GT-index shape;
`remap` dicts). The orchestrator wrote `report.py`, `run_evaluation.py`, and the
training hook, then ran a critical senior review + bug-fix pass.

## Design highlights (predict once, score many ways)

- Inference runs once per **component domain** (real, synthetic), cached to a
  COCO predictions JSON; the **`mixed`** domain is the two prediction sets
  concatenated with synthetic `image_id`s offset past the real max — no extra
  inference. Every granularity/band/domain slice is a cheap CPU remap+filter.
- **Granularity = label remap**, not new metric code: `fine` (identity),
  `coarse` (genus rollup + curated look-alike overrides from
  `reports/lookalike_groups.csv`), `detect` (all→1), applied to preds *and* GT.
- Coordinate un-letterboxing reuses `evaluation.py`'s exact transform.

## Validation & bug fixes (senior review)

- **Equivalence check:** on a 400-image representative sample the suite's fine
  mAP50_95 / mAP50 matched the trusted training-time `evaluate()` to 4 dp
  (0.3897 / 0.4483 both) — confirms the predictions→COCO-id remap and GT
  coordinate handling are consistent with the established evaluator.
- **Trap avoided:** an initial smoke test on the *first* 60 images gave fine
  mAP = 0.000 (those records are all aardvark/aardwolf — the hardest band-A
  classes). Cross-checking against the training log (test mAP50_95 = 0.259)
  showed this was sample bias, not a bug. `--limit` now takes a *representative
  random* sample.
- **Fixes applied to `report.py`:** torchmetrics' `-1` "not-applicable" sentinel
  (empty area splits, absent classes) was poisoning the macro-mean (produced a
  nonsensical `-0.983`); now excluded from means and rendered `—`. MLflow logging
  skips non-finite values.
- **Full smoke (1500 random images):** all tiers render coherently — granularity
  ladder nests correctly (detect 0.75 ≥ coarse 0.38 ≥ fine 0.34); band grid is
  monotonic A<B<C<D; within-group confusion surfaces the expected look-alikes
  (asian↔african elephant, european↔american bison, dingo↔domestic dog, donkey↔horse).

## Manual step left for the user

- **Review `reports/lookalike_groups.csv`** (strategy doc open question #1). It is
  genus-level taxonomic rollup (216 classes) + 3 frozen curated override groups
  (elephant; lynx/bobcat/caracal/lynx; hyaena) = 173 coarse groups, 28 of them
  multi-member. Genus rollup merges some clusters that are visually *distinct*
  within a genus (e.g. `panthera`: lion/tiger/leopard/jaguar/snow leopard;
  `canis` includes domestic dog). Confirm or refine these before relying on the
  coarse-granularity numbers; rebuild with `16-build_lookalike_groups.py`.
- **TIDE** error-typing (strategy §4.2) is left as a future add-on: `tidecv` is
  not installed, so it is skipped. The predictions JSON is already in the COCO
  format TIDE consumes.
