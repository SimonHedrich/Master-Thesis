# yolo26n post-training eval: OOM crash diagnosis, memory fix, and completed results

## Incident

A `run_training_pipeline.py --full-eval --resume-from .../yolo26n-20260713-225715/last.pt` invocation was reported as "crashed." Investigation (tmux scrollback, `/proc` forensics, `journalctl -k`) found the actual sequence of events was different and better than a training crash:

- **Training had already finished successfully.** The long-running job (resumed weeks earlier, writing to `scripts/training/yolo26n/model_exports/yolo26n-20260715-010031/`) trained to **epoch 179/200**, where early stopping fired (no `mAP50_95` improvement for 20 epochs). `best.pt` is epoch 158, training-time val `mAP50_95 = 0.6421`.
- Immediately after, the mandatory post-training `--full-eval` step began. ~5 minutes in, the Linux kernel **OOM-killer** killed the process (`journalctl -k`: `Out of memory: Killed process ... anon-rss:4331444kB ... shmem-rss:8937580kB`) — a hard `SIGKILL`, no Python traceback, which is why it looked like a silent hang rather than a crash.
- Three orphaned `pt_data_worker` DataLoader-worker children survived the kill, reparented to PID 1, holding ~15GB of leaked GPU memory and a deleted-but-open log file handle.
- A second `--resume-from .../yolo26n-20260713-225715/last.pt` command (using a **stale**, ~epoch-50-60 checkpoint from days earlier) was typed in a tmux pane but never actually submitted — running it would have discarded ~120 epochs of already-completed training. It was cleared rather than run.

**Conclusion:** nothing needed to resume for training — it was done. What was missing was just the evaluation report on the existing `best.pt`.

## Root cause of the OOM

`scripts/training/yolov5s/eval_suite/scoring.py::score()` (shared by both the yolov5s and yolo26n eval suites) called `torchmetrics.MeanAveragePrecision` **once, with all 225 classes' data for the full test set loaded simultaneously** — for the "headline mixed" cell that's 75,115 images (63,865 real + 11,250 synthetic) × 225 classes × ~7.5M cached detections. RSS climbed past 30GB on the very first such call before it was killed again on a second attempt.

## Fix implemented

1. **Chunked `metric.update()` calls** in `score()` (build/feed per-image tensors in batches of 2,000 images instead of one giant list) — removed a double-buffering peak spike, but the underlying multi-class-at-once design still grew memory linearly with no plateau; insufficient alone (thrashed even with 48GB of added swap).
2. **48GB swap file added** (`/swapfile`, persisted in `/etc/fstab`) as a standing safety net — helped absorb overflow but the run still ballooned toward thrashing (swap use climbing ~2GB/min with no sign of plateauing after 21 minutes) and was killed rather than left to grind indefinitely.
3. **Rewrote `score()` to score one class at a time** (the actual fix): for each class, restrict to the (much smaller) subset of images touching that class, run an independent `MeanAveragePrecision` instance, and macro-average the per-class results myself — reproducing torchmetrics' own "exclude no-GT classes from the mean" convention. This bounds peak memory to one class's data at a time instead of all 225 simultaneously.
   - **Validated correctness** against the original (memory-unsafe) implementation on a synthetic test case engineered to have non-degenerate per-class hit rates (true positives, false positives, a GT-only class, and a predictions-only class): per-class AP matched **bit-identically** across all 15 test classes; aggregate `map`/`map_50`/etc. matched to ~7 significant figures (residual difference is ordinary float32 summation-order noise).
4. **Skipped a redundant recomputation in `report.py`**: Tier 2.1's `granularity_scores()` call was independently recomputing the exact same (mixed, all-images, fine-remap) score already produced by Tier 1's `headline_mixed` — added a `precomputed` parameter so Tier 2.1 reuses it instead of paying for a second full-scope pass.
5. **Smoke-tested end-to-end** at small scale (`--limit 300 --no-cache`) before committing to the full run, to catch integration issues cheaply.

All four changes are in `scripts/training/yolov5s/eval_suite/scoring.py` and `report.py` (shared by both eval suites).

## Results

The full evaluation (`scripts.training.yolo26n.eval_suite.run_evaluation --run-dir .../yolo26n-20260715-010031 --mlflow`) completed successfully — memory stayed flat and healthy throughout (25-35GB available at every checkpoint, confirmed via repeated `free -h`/`journalctl -k` polling), no OOM.

**It took ~5 hours** (08:13→13:20). That is safe but slow, and is documented as its own follow-on problem in `docs/plans/2026-07-22_eval-suite-scoring-performance-investigation.md` (per-class stratification trades memory for time — not yet fixed; confidence-threshold filtering was investigated as a shortcut and explicitly **rejected**, since it changes the reported mAP, not just the runtime).

**Headline numbers** (`scripts/training/yolo26n/model_exports/yolo26n-20260715-010031/eval_best/evaluation_report.md`):

| Metric | Mixed (headline) | Real (breakout) |
|---|---|---|
| mAP | 0.523 | 0.481 |
| mAP50 | 0.574 | 0.541 |
| mAP75 | 0.549 | 0.511 |

Other notable findings in the report:
- **Public-comparison analog** (class-agnostic `mAP_detect`): mixed 0.812 / real 0.778 — vs. fine-grained 225-way mAP of 0.523/0.481, a Δ_coarse (cross-group cost) of 0.276 and Δ_fine (look-alike cost) of only 0.013 — most of the difficulty is telling *which* of 225 species it is, not detecting *that* something is there.
- **Band A (lowest-data classes) is very weak**: mAP_fine 0.019 (mixed) / 0.011 (real), vs. 0.788/0.741 for Band D.
- **Real-vs-synthetic domain shift** (mean Δ = -0.214, fine granularity) is large enough to flag against the strategy doc's watchdog criterion (`docs/plans/2026-06-10_model-evaluation-strategy.md`) — worth revisiting whether the mixed default should be reconsidered for this checkpoint.
- **Elephant (0.514) and hyaena (0.402) look-alike groups** have the highest within-group fine-grained confusion rates.

MLflow scalar/artifact logging failed at the very end (filesystem-backend maintenance-mode warning, unrelated to the memory fix) but all report/CSV/JSON files were written to disk successfully regardless.
