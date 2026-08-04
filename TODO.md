# TODO

Working task list for the remaining thesis work, ordered roughly by dependency.
Items marked **(gap)** were not in the original task list but surfaced from an
audit of `docs/` and `scripts/training/` — inserted where they logically block
or extend the adjacent item, not appended at the end.

Every task now has a stable `<section>.<item>` number (for cross-referencing
from commits, progress notes, and the phase schedule in §1) and a machine tag
showing where it should run:

- **[A40]** — needs this machine specifically (24GB VRAM, but see §1.1 caveat
  on shared-tenancy risk).
- **[3060]** — fits the RTX 3060 (12GB) and should default there so it doesn't
  queue behind A40 work.
- **[Either]** — no GPU required, or light enough that whichever machine is
  free first should take it.
- **[Human]** — not a compute task (rating, writing, a decision to confirm).

## 1. Two-machine execution plan (A40 + RTX 3060)

### 1.1 Machine roles

The A40 (24GB, this machine, hostname `thesis` / Makefile's `ICS_HOST`) is a
**vGPU instance and has shown shared-tenancy contention before** (other
tenants observed consuming 16–22GB, forcing a batch-size drop to 4 during the
teacher-finetune smoke test — `docs/plans/2026-06-30_yolo26-kd-and-teacher-finetune-implementation-plan.md`
§ "smoke test"). Treat CUDA OOM on this box as possibly contention, not
necessarily a script bug. Disk is also at 95% (29GB free) — check headroom
before queuing the largest generation cells.

The RTX 3060 (12GB) is confirmed already set up with the repo and dataset —
likely the Makefile's `REMOTE_HOST` (`gpu.local` / tailscale `gpu-server`);
**confirm the exact host alias before scripting any rsync commands against
it**, since this wasn't verified directly, only inferred from naming.

Split rationale (from measured VRAM/timing data already in the repo):

- §3.1's six remaining generation cells are **A40-only**. Only
  `realvisxl-lightning` comfortably fits 12GB; `sd35m` needs ~16.6GB; the
  other four already require `enable_model_cpu_offload()` *even on the 24GB
  A40* (`docs/synthetic-model-comparison/04_local-models-and-output-parameters.md`
  §7) — they would thrash or fail outright on a 3060.
- Every other compute task below (§2.1, §4.1–4.4) is documented at 7–10GB
  (`docs/2026-04-29_gpu_training_options.md`'s VRAM table) — trivial for the
  3060, and there are *many* such runs (multiple seeds, multiple cells, a KD
  hyperparameter sweep). A dedicated second GPU removes all of it from behind
  the generation queue.
- §2.2 (`eval_suite` scoring) is **CPU/RAM-bound, not GPU-bound** — no
  `.cuda()`/`.to(device)` call anywhere in `scoring.py`, and the one OOM
  incident on record was the kernel OOM-killer on system RAM, not
  `CUDA_OUT_OF_MEMORY`. It can run on either box without occupying a GPU slot
  at all.

### 1.2 Phased schedule

| Phase | A40 | RTX 3060 | Gate to next phase |
|---|---|---|---|
| **A** (start now, parallel) | §3.1 generation queue, cell 1 (`realvisxl-lightning`) | §2.2 `eval_suite` scoring fix | none — both start immediately |
| **B** (rolling per finished cell) | §3.1 generation queue, next cell | §3.2 labeling pipeline → §3.3 YOLO26n training, per cell as it lands | each cell needs its images rsynced off the A40 before §3.2 can run on the 3060 (§1.3) |
| **C** (after §2.2 lands) | free once generation queue ends | §4.1 SpeciesNet head fine-tune, §4.2 KD Phase-0 baselines, §4.3 YOLOv5s retrain, §4.4 YOLO26n KD | §4.5 needs all of §4.1–4.4's eval reports |
| **D** | — | — | §5, §6, §7 — decisions and writing, not gated on a GPU |

**Checkpoint discipline for §3.1** (per standing preference — don't run a
multi-model GPU batch queue unattended): pause after each cell completes and
check in before starting the next, in cheapest-first order
(`realvisxl-lightning` → `sd35m` → `flux2-klein-9b` → `sd35-large` →
`qwen-image` → `hidream-i1`) so a shortened campaign still bought the most
model diversity per hour spent. `qwen-image` was dropped mid-queue (user
decision 2026-07-30, confirmed via its `compressed`-regime smoke test —
doc `13` §9); `hidream-i1` ran last and completed the queue — §3.1.

### 1.3 Git sync protocol

`origin` is `git@github.com:SimonHedrich/Master-Thesis.git`; both machines
push/pull against it (already the pattern — recent history has merge commits
from exactly this kind of two-machine workflow). Two categories of state need
different sync mechanisms:

**Tracked in git** (code, docs, and — notably — eval reports/logs/metrics,
which are *not* gitignored even though model weights are):
- `git pull` at the start of any work session on either machine, before
  editing anything.
- Commit + push promptly after each discrete unit of work — a cell's
  benchmark entry, an `eval_suite` dev milestone, a training run's report —
  rather than batching into one large end-of-day commit. Shortens the window
  where the two machines' trees can diverge.
- **Shared status files are the actual conflict risk**: `TODO.md`,
  `docs/README.md`, `docs/synthetic-model-comparison/README.md`. Both
  machines' sessions will want to update these as work lands. Whichever
  machine finishes a unit of work commits + pushes its doc update
  immediately; the other machine's session `git pull`s before touching the
  same file. Avoid two uncommitted edits to the same status doc sitting on
  two machines at once — prose merge conflicts are the annoying kind.
- Per-run artifacts (`model_exports/<run>/*.log`, `eval_best/*.json|.md|.csv`)
  are timestamped/uniquely named per run, so cross-machine conflicts on those
  specifically are structurally unlikely even without careful coordination.

**Not in git** (`data/*`, `mlruns/*`, `*.pt`/`*.pth`/`*.onnx` are all
gitignored) — sync via the Makefile's existing rsync targets instead:
- After each §3.1 generation cell finishes on the A40, rsync its images
  before §3.2's labeling step can run on the 3060 — extend the existing
  `sync-ics`/`sync-ics-data` pattern (or push straight to NAS
  `data-server` and have the 3060 pull from there) rather than inventing a
  new transfer path.
- Trained checkpoints from §4.1–4.4 (all on the 3060) should get rsynced to
  the NAS backup (`data-server`, RAID 1, always-on) right after each run
  completes — they're gitignored, so the 3060 is their only copy otherwise.
  §4.5's comparison synthesis itself only needs the (git-tracked) eval
  reports, so this is a durability step, not a blocker for §4.5.

## 2. Infrastructure prerequisite

- [x] **2.1 [Either] (gap) Fix `eval_suite` scoring performance.** Fixed
      2026-07-31 on `gpu-server`: benchmarked `faster-coco-eval` per the
      investigation's go/no-go plan, found a naive single-call replacement
      is 11-15x faster but peaks at 23-27GB RSS (too tight on this 31GB
      host), and shipped a hybrid instead — kept the memory-safe per-class
      loop, swapped only the inner engine (`_fce_backend.py`) — ~5x speedup,
      no memory-risk increase. Validated via a new parity test
      (`eval_suite/tests/`), a real-data smoke-test diff, and a full re-run
      against `yolo26n-20260715-010031`: **~5h → 67.8min confirmed**, memory
      stable ~19.5GB throughout, no OOM. See
      `docs/progress_notes/2026-07-30_eval-suite-scoring-perf-fix.md` for
      full results.
      **Headline numbers did not reproduce (0.451/0.410 vs. the published
      0.523/0.481) — root-caused to a data issue, not this fix**: old and
      new scoring engines agree exactly on identical predictions (ruling out
      the rewrite); the A40's `data/real/annotations_test.json` predates the
      2026-06-09 multi-animal-per-image contamination-flagging work (single
      box/image, `date_created: 2026-05-26`) while this machine's copy
      postdates it (~1.44 boxes/image, `date_created: 2026-06-09`) — a stale
      `data/*` file on the A40 that was never resynced per §1.3. **Needs a
      decision**: which annotation-file version is authoritative, and an
      A40 `data/` resync before trusting any further cross-machine eval
      comparison (also affects the yolov5s headline numbers if that
      checkpoint was evaluated against the same stale file — not yet
      checked).

## 3. Synthetic-model-comparison experiment

- [x] **3.1 [A40] Generate remaining local-model `maxlen` cells:**
      all six non-dropped cells are now complete — `sd35-large-turbo`,
      `realvisxl-lightning` (actual: 0.31h inference, 0.93s/image,
      1200/1200, 0 failures), `sd35m` (actual: 7.98h inference,
      23.94s/image — above the ~5h estimate, 1200/1200, 0 failures),
      `flux2-klein-9b` (actual: 10.40h inference, 31.19s/image — under the
      ~16-22h estimate, 1200/1200, 0 failures; **quality note:** `kinkajou`
      renders show a consistent genet/civet-like ringed tail rather than
      the real species' plain tail — a per-class model-accuracy signal for
      §3.4/3.5, not a pipeline failure), `sd35-large` (actual: 18.11h
      inference, 54.34s/image — within the ~19-20h estimate, 1200/1200, 0
      failures; its own `kinkajou` renders are correct, confirming the
      tail-confusion above is specific to `flux2-klein-9b`), and
      `hidream-i1` (actual: 47.97h inference, 143.92s/image — above the
      ~43h estimate, 1200/1200, 0 failures; its `kinkajou` renders are also
      correct) — doc `13` §6. Run in this cheapest-first order with a
      check-in between each cell (§1.2) — not as one unattended queue.
      `hidream-i1` first needed `1i-generate_images_local_maxlen.py`
      extended with its loader/tier support (previously only in `1g`, the
      `compressed`-regime script), including fixing a copy-paste bug where
      its generator function would otherwise have hardcoded
      `max_sequence_length=128` instead of the 512 its maxlen tier
      assignment requires — doc `13` §9. `qwen-image` was dropped, not
      generated, due to confirmed NF4-quantization graininess (doc `13`
      §9) — its `enable_model_cpu_offload()` requirement is therefore now
      moot for this task.
- [x] **3.2 [3060] (gap) Run the labeling pipeline**
      (`scripts/synthetic_model_comparison/2-run_megadetector.py` through
      `5-export_coco.py`) on each generated cell — blocks the training step
      below. Stage 2 (MegaDetector) done directly on the A40 (GPU was idle,
      cheap enough not to wait for a 3060 handoff) for all six completed
      `maxlen` cells: `realvisxl-lightning`, `sd35m`, `flux2-klein-9b`,
      `sd35-large`, `sd35-large-turbo`, `hidream-i1` — 1,200/1,200 images
      each, 0 missing. Required adding `"maxlen"` to `2-run_megadetector.py`'s
      `--prompt-regime` choices (only had `full`/`compressed`, a gap from
      before doc `13` introduced the `maxlen` regime). See
      `docs/synthetic-model-comparison/README.md` for the per-cell
      `n_significant` breakdown. **2026-08-04: stage 5 (COCO export) run
      for all five cells** using `5-export_coco.py`'s documented best-effort
      fallback (MegaDetector's own boxes, no human review) — all five
      exported 1,200/1,200 images, 0 skipped. **Stages 3/4 (triage review,
      bbox labeling) still have not run on any cell** — these exports are
      explicitly provisional/not thesis-final until they do (§3.4). Needed
      the same `"maxlen"` argparse-choices fix in `3-single_detect_review.py`,
      `4-bbox_labeling_server.py`, and `5-export_coco.py` too. **`hidream-i1`
      exported the same way (2026-08-04, on the A40):** 1,200 images, 1,198
      annotated / 2 skipped (its two `n_significant == 0` images from the
      MegaDetector table above), 1,229 boxes total — also provisional, same
      caveat. `annotations.json` is gitignored (`data/*`), so it exists only
      on the A40 for now — needs an rsync per §1.3 before the 3060 can run
      §3.3 training on this cell.
- [x] **3.3 [3060] Train yolo26n on each comparison dataset**
      (`scripts/synthetic_model_comparison/training/`), ideally multiple runs
      per dataset for averaged metrics — code exists, never run end-to-end on
      real cell data yet. Lightweight enough to run continuously on the 3060
      between §3.2 arrivals, in parallel with the A40's ongoing §3.1 queue.
      **2026-08-04: run end-to-end on all five cells, 2 seeds (42/43) each,
      `--full-eval` against the full real test set** — results provisional,
      not thesis-final (built on §3.2's un-reviewed best-effort exports; will
      need re-running once §3.4 lands). Avg real-test map: `sd35-large`
      0.055, `sd35m` 0.054, `sd35-large-turbo` 0.043, `flux2-klein-9b` 0.037,
      `realvisxl-lightning` 0.024 — see `docs/synthetic-model-comparison/README.md`
      2026-08-04 update for the full per-seed table. All in the same
      low-map range as the historic incumbent-generator direct-FT run
      (0.064), as expected for ~960-image synthetic fine-tunes.
      **Two real bugs found and fixed along the way** (both also existed in
      the main, non-comparison training pipeline — see
      `scripts/training/yolov5s/training_pipeline.py`, imported by both main
      pipelines, and this package's own copy): (1) no gradient clipping
      anywhere in the training step, causing NaN divergence during LR
      warmup on real data — fixed via `clip_grad_norm_(max_norm=10.0)`,
      matching Ultralytics' own trainer; (2) the post-training full-eval
      hook hangs indefinitely (7+ hrs, 0% GPU/CPU util, no error) because it
      forks a fresh `num_workers=8` DataLoader deep into an already-CUDA-
      active process — fixed by forcing `num_workers=0` for that call only.
- [ ] **3.4 [Human] (gap) Blind multi-rater qualitative rubric**
      (`docs/synthetic-model-comparison/06_evaluation-methodology.md`) — the
      human-rating axis, distinct from automatic proxies/downstream mAP; not
      yet executed.
- [ ] **3.5 [Human] (gap) Final model-comparison writeup/decision** for the
      supervising professor once ratings + downstream mAP are in — the actual
      deliverable this whole subdirectory exists to produce.

## 4. Core model training campaign

- [ ] **4.1 [3060] Fine-tune SpeciesNet's classifier head** (in the MD+SN
      ensemble) on the 225-class taxonomy and evaluate — scaffolding complete
      (`scripts/training/teacher_finetune/`,
      `scripts/training/megadet_speciesnet_ensemble/`), not yet actually run.
      54M-param EfficientNetV2-M head, comfortably fits 12GB without the
      contention the shared A40 has shown. Note: no `make speciesnet-*`
      target exists yet despite being documented — run via the manual `uv
      run python -m ...` commands in the script docstrings, or add the
      Makefile targets first.
- [ ] **4.2 [3060] (gap) KD ladder Phase 0 — zero-shot baselines** (untrained
      teacher/student) — needed as the floor for the Phase 4 comparison
      table; confirm whether these are already logged anywhere before
      assuming they need a fresh run. Inference-only, cheap.
- [ ] **4.3 [3060] Retrain YOLOv5s** with the new anchor/loss-autoscaling
      implementation (`autoanchor.py` fix from
      `docs/progress_notes/2026-07-16_yolov5s-underperformance-hyp-scaling-fix.md`)
      — fix is implemented, full retrain not yet done.
- [ ] **4.4 [3060, or A40 once §3.1 frees it] Train YOLO26n with knowledge
      distillation**, MD+SN ensemble as teacher (Phase 3 of
      `docs/plans/2026-06-30_knowledge-distillation-and-teacher-finetuning-strategy.md`).
      Not VRAM-bound either way — take whichever GPU is idle first.
- [ ] **4.5 [Either, no GPU] (gap) KD ladder Phase 4 — final comparison
      synthesis**: assemble direct-FT vs. teacher-FT vs. KD results into the
      comparison the strategy doc's experimental ladder is building toward.
      Pure analysis over the (git-tracked) eval reports from §4.1–4.4 —
      needs §2.1 fixed first so those evals were affordable to produce.

## 5. Deployment / embedded pipeline

- [ ] **5.1 [Human — decision, then TBD hardware] (gap) Quantization-aware
      training (QAT).** Named in `CLAUDE.md`'s pipeline description but
      explicitly out of scope in the KD strategy doc. Confirm this is a
      deliberate deferral (and when it picks back up), not a dropped step.
- [ ] **5.2 [RPi5 / QCS605 target hardware, not A40/3060] (gap) Export +
      on-device benchmarking** on the RPi 5 proxy (and/or QCS605 target).
      Nothing in the repo yet exports or times a model on real target
      hardware — this is the thesis's core "real-time inference on embedded
      hardware" claim and currently has no artifact behind it. Neither GPU
      machine is the target here; this needs the actual proxy/target device.

## 6. Scope decisions to confirm (not tasks, but open questions)

- [ ] **6.1 [Human — decision] (gap)** NanoDet-Plus-m / PicoDet-S were
      smoke-tested early
      (`docs/progress_notes/2026-04-24_training-setup-and-model-smoke-test.md`)
      but all current work has converged on YOLO26n only. Confirm this
      narrowing is intentional so the thesis states it as a decision rather
      than a silently dropped student model.

## 7. Writing

- [ ] **7.1 [Human] (gap) Thesis manuscript.** Everything under `docs/` is
      working notes/plans, not thesis text — this needs to be synthesized
      separately once the experimental results above land.
