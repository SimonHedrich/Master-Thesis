# eval_suite scoring performance fix: benchmark, hybrid design, and validation

Implements TODO.md §2.1, following the go/no-go plan in
`docs/plans/2026-07-22_eval-suite-scoring-performance-investigation.md`.

## Benchmark

Wrote `scripts/training/yolov5s/eval_suite/benchmarks/bench_scoring_engines.py`
and ran it on `gpu-server` (RTX 3060, 31GB RAM) using synthetic predictions
matching the NMS-free yolo26n model's density (100 detection slots/image,
spread across the full 225-class universe) over the real GT image universe
(`data/real/annotations_test.json`, 63,802 images).

| Engine | 20k images | 63.8k images (real) | 75k images (mixed) |
|---|---|---|---|
| `torchmetrics` (current per-class loop) | 494.5s, 2.3GB | not run (~26-35min extrapolated) | — |
| `pycocotools` | stalls badly | not run | — |
| `faster_coco_eval`, single call (drop the per-class loop) | 43.3s, 7.7GB | 155.6s, **23.2GB** | 186.5s, **27.0GB** |
| `faster_coco_eval`, per-class loop (hybrid) | 92.4s, 2.5GB | ~5min (extrap.), ~5GB (extrap.) | — |

The naive single-call design (drop the per-class stratification entirely, one
`faster_coco_eval` call per report section — the design originally proposed
in the investigation doc) is 11-15x faster than today, but peaks at 23-27GB
RSS at real/mixed scale — only ~4GB of headroom on this 31GB host, before
accounting for the rest of the report pipeline's resident memory (multiple
domains' GT indices and prediction lists held at once across report.py's
~15 sequential scoring calls). Running it risked recreating the exact OOM
this task exists to fix.

**Decision: hybrid design.** Kept the existing per-class loop structure
(already proven memory-safe — each class only touches its own relevant
images) and swapped only the inner per-class engine from
`torchmetrics.MeanAveragePrecision` to `faster_coco_eval.COCOeval_faster`.
~5x speedup (494.5s → 92.4s at 20k-image scale) with memory essentially
unchanged from today (2.5GB vs. 2.3GB) — turns the ~5-hour full report into
an estimated 20-40 minutes, without the single-call design's memory risk.

## Implementation

- New `scripts/training/yolov5s/eval_suite/_fce_backend.py`: per-class
  GT/DT-dict adapter + `faster_coco_eval` call, restricted to each class's
  `relevant_ids` exactly like the code it replaces.
- `scoring.py::score()`: the per-class loop body now calls
  `_fce_backend.score_class()` instead of building `torchmetrics` tensors;
  the loop structure, capping logic, and aggregation (macro-mean over
  GT-bearing classes, `-1.0` sentinel for predictions-only classes) are
  unchanged. `report.py`, `predict.py`, `grouping.py`, and both
  `run_evaluation.py` entrypoints needed zero changes (confirmed: `score()`'s
  signature and output schema are exactly preserved).
- Added `faster-coco-eval>=1.7.2` to `pyproject.toml`/`uv.lock` (`uv add`) and
  rebuilt the training Docker image (`make build`).
- Fixed a latent, previously-unexercised misconfiguration:
  `[tool.pytest.ini_options]` had `pythonpath = ["scripts"]`, inconsistent
  with every module in the repo importing via `scripts.*` absolute paths
  (per CLAUDE.md's own documented convention) — no test file existed before
  this task to surface the mismatch. Changed to `pythonpath = ["."]`.

## Validation

**Parity test** (new — first test file in the repo):
`scripts/training/yolov5s/eval_suite/tests/test_scoring_engine_parity.py`, a
synthetic 500-image/20-class case with deliberately varied per-class hit
rates (perfect matches, all-miss, sub-threshold-IoU partial matches, a
shared image exceeding the max-detections cap, predictions-only classes).
The new implementation matches a frozen pre-rewrite snapshot to float
tolerance, plus 3 hand-verified per-class anchors. `uv run pytest
scripts/training/yolov5s/eval_suite/tests/` passes.

**Real-data smoke test**: ran the full `report.build_full_report` pipeline
(all Tier 1-3 sections) on a 300-image/domain subset of real cached
predictions (`yolov5s-20260714-010652`'s `predictions_real.json`/
`predictions_synth.json`, since no yolo26n checkpoint is present on this
machine to run fresh inference — see below), before and after the rewrite,
diffed via the new `scripts/training/yolov5s/eval_suite/tests/diff_smoke_reports.py`.

This surfaced small but real differences (order 1e-4 to 1e-2 in per-class
AP, smaller in aggregates) — **not** float noise, but also **not** a bug.
Root-caused via a direct three-way comparison on one flagged class: calling
`pycocotools.COCOeval` directly and calling `faster_coco_eval.COCOeval_faster`
on identical GT/DT data produced **bit-identical** results (0.5675247524752475
both), while `torchmetrics.MeanAveragePrecision(backend="pycocotools")` on
the exact same data produced a different value (0.5605940818786621) —
matching the *old* `scoring.py`'s output. **`torchmetrics`'s own
`pycocotools`-backend wrapper does not reproduce calling `pycocotools`
directly**, on real (non-degenerate, multi-detection, partial-recall) data —
a pre-existing, previously-undetected discrepancy in the old implementation
(no test had ever cross-checked it against vanilla `pycocotools`), not
something introduced by this rewrite. The new implementation's numbers are
the ones that agree with the industry-standard reference implementation.

Practical effect: headline mAP numbers will shift by a few thousandths
(e.g. `map` +0.0003 in one 300-image mixed-domain sample) when the full
suite is re-run — an accuracy *correction*, in the direction of the true
COCO standard, not a regression.

## Final re-run: wall-clock confirmed, headline numbers did NOT reproduce — and why

`yolo26n-20260715-010031/best.pt` wasn't present on this machine (checkpoints
are gitignored). Rsynced it from the A40 (`ubuntu@ics-server` — the tailscale
peer name; the Makefile's `ICS_HOST` value `thesis.taile550ef.ts.net` is
stale/unresolvable, the node's own hostname is still `thesis` but its
tailscale peer name changed), checksum-verified identical
(`5d941fe2d5e8df83b6c7ca457874769f`), then ran:

```
uv run python -m scripts.training.yolo26n.eval_suite.run_evaluation \
  --run-dir scripts/training/yolo26n/model_exports/yolo26n-20260715-010031 --mlflow
```

**Wall-clock: 67.8 minutes** (11:04:30 → 12:12:16), down from ~5 hours —
confirms the performance fix. Memory plateaued at a stable ~19.5GB RSS (no
climb, no OOM, `journalctl -k` clean) — confirms the hybrid design's memory
safety holds at real production scale, not just in the isolated benchmark.

**But the headline numbers did not reproduce**: `map` came back 0.451 mixed /
0.410 real, vs. the previously-published 0.523 / 0.481 — a ~7-point drop,
far larger than the documented torchmetrics-vs-pycocotools gap above.
Root-caused via elimination, each checked directly against the A40:

1. **Not the scoring rewrite.** Took the exact fresh predictions this run
   generated and scored them with both the new (`faster_coco_eval`) and the
   *old* (pre-rewrite, `torchmetrics`) engine, side by side: `map` = 0.41048
   (new) vs. 0.41047 (old) — they agree. Whatever's producing the lower
   number, both engines see it identically in the input data.
2. **Not the checkpoint.** `md5sum` identical on both machines.
3. **Not the images.** Full inaturalist corpus (169,715 local vs. 169,511 A40
   files — the dominant source for bands B/C/D) diffed by exact set
   membership: 204 extra local files, **zero** missing; a 15-file sample from
   the worst-affected species (wildebeest, impala, buffalo, gemsbok,
   blackbuck, springbok, hartebeest) hashed identically on both machines.
4. **Not dependency drift.** `torch`/`torchvision`/`ultralytics` pinned
   versions unchanged across every `uv.lock` commit back to 2026-06-06.
5. **It's the annotation file.** `data/real/annotations_test.json` itself
   has a different checksum on each machine. Pulled the A40's copy and
   diffed: same 225 categories, image *count* close (63,865 vs. local's
   63,802) but **annotation count is not** — local has **92,094**
   annotations (~1.44 boxes/image) against the A40's **63,796** (~1.0
   box/image, essentially always exactly one). The local file's `info.date_created`
   is `2026-06-09`; the A40's is `2026-05-26`. `2026-06-09` is exactly the
   date of the multi-animal-per-image contamination-flagging work
   (`docs/progress_notes/2026-06-09_contamination-flagging-and-augmentation-implementation.md`)
   — the A40's copy of this test-set annotation file predates that work and
   was never resynced, so it's missing the secondary GT boxes in multi-animal
   images. Exactly matches the observed pattern: Band A (sparse, mostly
   single-animal images) barely moved; Bands B/C/D (where multi-animal scenes
   are common) dropped hardest, concentrated in gregarious/herd species
   (wildebeest, impala, buffalo, gemsbok, springbok, hartebeest — all
   commonly photographed in groups).

**Implication**: the previously-published 0.523/0.481 was computed against a
stale, single-box-per-image test set that under-counts ground truth (missed
secondary animals were never scored as false negatives). The new
0.451/0.410 is scored against the current, more complete multi-box
annotations and is the methodologically correct number *for this test set
version* — but this is a **test-set version mismatch between machines**, not
a property of either scoring engine, and not something TODO §2.1 set out to
fix. Flagging for a decision: which annotation file is authoritative
going forward, and whether the A40's `data/real/annotations_test.json` (and
possibly other `data/*` files) need a resync per the two-machine sync
protocol (TODO.md §1.3) before further cross-machine eval comparisons are
trusted.
