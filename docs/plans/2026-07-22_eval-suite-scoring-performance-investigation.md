# eval_suite scoring engine: why it takes ~5 hours, and how to fix it

## Context

The full `scripts.training.yolo26n.eval_suite.run_evaluation` run against `yolo26n-20260715-010031/best.pt` completed successfully (see `docs/progress_notes/2026-07-21_...` session context) but took **~5 hours**, which is only tolerable as a one-off. The thesis will need to re-evaluate multiple checkpoints (teacher fine-tune, KD student, ablations) against this same eval suite, and a 5-hour turnaround per run is a real bottleneck for iteration speed. This doc investigates why, and lays out a plan to fix it — **investigation and planning only, no implementation yet.**

### How the scoring engine got here

The shared `scripts/training/yolov5s/eval_suite/scoring.py::score()` (used by both the yolov5s and yolo26n eval suites) originally called `torchmetrics.MeanAveragePrecision` once per call with all 225 classes' data loaded simultaneously. That single-call, all-classes-at-once design is what OOM-killed the original evaluation run earlier in this session (the process was killed by the Linux kernel OOM-killer while computing Tier 1's `headline_mixed`). It was rewritten to score one class at a time — building, updating, and calling `.compute()` on a fresh `MeanAveragePrecision` instance per class, restricted to only the images relevant to that class — which bounds peak memory (validated bit-identical against the original implementation on a synthetic reference case with known non-degenerate per-class hit rates). That fixed the OOM. It is also why the run now takes ~5 hours: **it trades memory for time.**

## What's actually slow (profiled this session)

Instrumented timing on synthetic test cases matching the real data's density (~100 predictions/image, spread across 225 classes — verified this matches `predictions_real.json`'s actual per-class touch rate: median ~29% of images touched per class) isolated the cost:

- The `relevant_ids` image-scan per class (finding which images touch a given class) is cheap — ~0.02s/class at 1,500 images, ~0.3s/class at 20,000 images. A minor contributor (~3-4 min extrapolated at real 75k-image scale).
- **The dominant cost is `torchmetrics.MeanAveragePrecision`'s own per-class object construction + `update()` + `compute()`.** At 20,000-image scale this took 2.2-3.9s *per class*, scaling with that class's relevant-image count. Because predictions are densely spread (median class touches ~18,000-43,000 of ~63,865-75,115 images in the real data, *not* a small subset), each of the 225 per-class calls is itself substantial work. Extrapolating this to real scale (~4x more images than the 20k test) matches what was actually observed: ~35-45 minutes for a single full-scope "fine" (225-class) call, and Tier 1 alone (which makes two such calls, `headline_mixed` + `headline_real`) took ~68 minutes.
- **Batching multiple classes per call doesn't help.** Because predictions/GT are densely spread rather than clustered by class, even a modest batch of ~15 classes would touch nearly 100% of images (union of relevant images across classes), re-approaching the exact memory blowup the per-class rewrite was designed to avoid, for only a partial reduction in per-call overhead.
- **A quick benchmark of `pycocotools.COCOeval`** (already a project dependency) on the same 20,000-image/225-class synthetic case did not clearly resolve this either — it stalled for 2+ minutes on `evaluate()` before being killed/orphaned. Likely cause: pycocotools's per-(image, category) loop is Python-level (only the box IoU math itself is C-accelerated); with ~80-90 distinct categories touched per image at this density, that's ~1.6-1.8M Python-level loop iterations even at 20k images — worse at real (75k-image) scale.
- **`faster-coco-eval`** (PyPI package, actively maintained C++/Cython drop-in replacement for pycocotools) exists specifically to fix this known pycocotools bottleneck, by moving the *entire* per-image-per-category loop into compiled code rather than just the IoU step. Not yet installed or benchmarked against our specific dense-prediction access pattern — the most promising unverified lead.

## An important dead end — do not do this

**Filtering out low-confidence predictions before scoring looks like a free 10-20x speedup and is not.** The NMS-free yolo26n model always fills all 100 detection slots/image regardless of confidence: 98% of the 6.4M cached real predictions have score < 0.05 (median score is 0; p90 is 0.0017). Thresholding those out cuts scoring time dramatically (124s → 11s → 6s at thresholds 0/0.01/0.05 on a 5,000-image test) — but it also **changes the reported mAP substantially**: 0.616 → 0.564 → 0.526 on the same subset. Some hard/rare classes' only correct hits are low-confidence detections; dropping them below threshold caps the achievable recall for that class and lowers its AP (COCO-style AP integrates the full precision-recall curve; capping recall early is scored as if precision were zero beyond that point). **This is not a valid optimization — it silently changes the number being reported, not just the runtime.** Any future speed work must preserve the full, unfiltered prediction set.

## Plan

1. **Benchmark `faster-coco-eval` first, as a go/no-go check**, before writing any integration code against it:
   - Install it in a scratch environment and re-run a `pycocotools`-style benchmark (build COCO GT/DT structures from `gt_index`/`predictions`, run `evaluate`/`accumulate`/`summarize`) using its C++ backend, on a synthetic case matching real density (225 classes, ~100 preds/image) at increasing scale up to ~75,000 images, and ideally directly on a large slice of the real cached `predictions_real.json`.
   - Compare wall-clock and peak RSS against both the current per-class `torchmetrics` approach and vanilla `pycocotools`.
   - **Decision point:** if it completes the full 75k-image/225-class case in a reasonable time (target: single-digit minutes, not hours) without excessive memory, proceed to step 2a. If it also struggles with this access pattern, proceed to step 2b.

2. **(2a) If `faster-coco-eval` works:** replace `scoring.score()`'s internals to build a COCO-style GT/DT structure from the existing `gt_index`/`predictions` inputs (same input contract — no caller changes needed in `report.py`) and run evaluation as a single multi-class call per `(gt, preds, image_ids, remap)`, likely dropping the per-class stratification loop entirely (back to one call, like the pre-OOM code), since a C++-backed engine should not exhibit `torchmetrics`'s memory-blowup pathology either. Map its output (per-category AP/AR across IoU/size/max-det bins) onto the existing flat output dict schema (`map`, `map_50`, ..., `map_per_class`, `n_images`, `n_dets`, `n_gt`) so `report.py` needs no changes.

3. **(2b) If it doesn't:** write a hand-rolled vectorized (numpy/torch) COCO-style AP calculator that computes AP for all classes in one pass using array/broadcast operations (grouped cumulative-sum style ops per class over pre-sorted arrays) instead of instantiating 225 separate `MeanAveragePrecision` objects. Higher effort and validation risk than 2a — pursue only if 2a is a dead end.

4. **Validate whichever engine is chosen against the current, already-validated implementation:**
   - Re-run the synthetic correctness test case with known non-degenerate per-class hit rates and confirm per-class AP matches to float precision, aggregate `map`/`map_50` etc. match within float32 tolerance.
   - Re-run the 300-image `--limit 300 --no-cache` smoke test end-to-end and diff the resulting `evaluation_report.md`/`.json` against the current version's output for material changes beyond expected floating-point noise.

5. **Re-run the full evaluation** (`PYTHONPATH=. uv run -m scripts.training.yolo26n.eval_suite.run_evaluation --run-dir scripts/training/yolo26n/model_exports/yolo26n-20260715-010031 --mlflow`) and confirm both a large wall-clock improvement and headline numbers matching the already-completed report (mAP=0.523 mixed / 0.481 real) to within float precision.

## Non-goals / explicitly rejected

- **Do not** add a confidence threshold before scoring to cut prediction volume — validated this session to materially change mAP, not just speed (see "An important dead end" above).
- **Do not** batch multiple classes together per `MeanAveragePrecision` call within the current torchmetrics-based approach as a speed compromise — profiled this session: because predictions are densely spread (median class touches ~29% of images), even a modest class batch re-approaches full-dataset memory usage, undermining the original OOM fix for little speed gain.

## Verification (once implemented)

- **Correctness:** per-class AP bit-identical (or float32-tolerance identical) to the current validated implementation on the synthetic reference case; small-scale smoke-test report numbers match.
- **Speed:** full 75k-image evaluation completes in a small fraction of ~5 hours (target: minutes, not hours), with `free -h` / `journalctl -k` showing no memory pressure or OOM.
