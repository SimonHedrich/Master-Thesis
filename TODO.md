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

The RTX 3060 (12GB) is confirmed already set up with the repo and dataset.
**Host alias confirmed live (2026-08-04):** the Makefile's `REMOTE_HOST`
(`gpu.local`) doesn't resolve from the A40 — use the Tailscale MagicDNS
name `gpu-server.taile550ef.ts.net` instead (same pattern as `ICS_HOST`).
Login user is `debian`, not `ubuntu` (the A40's own `ICS_HOST` user) —
`ssh-copy-id` was needed first, key auth wasn't already in place from the
A40 side. Verified via `nvidia-smi` (RTX 3060, 12288MiB) and an existing
`~/Master-Thesis` checkout on connection.

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
      caveat. `annotations.json`, `index.jsonl`, and all 1,200 images
      (`data/*`, gitignored) rsynced to `gpu-server` (the 3060), checksums
      and counts confirmed matching on both ends (1,200/1,200 images,
      1.5GB) — the 3060 can now run §3.3 training on this cell whenever
      it picks it up. Found and fixed a stale partial copy already sitting
      on `gpu-server` (21 images, a 2-record `index.jsonl` — from an
      earlier, unrelated attempt) before syncing the authoritative version
      over it. Note: `gpu-server`'s login user is `debian`, not
      `ubuntu` as the A40's own `ICS_HOST` uses — confirmed live, since
      TODO.md previously only inferred the host alias, not the user.
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

- [x] **4.1 [3060] Fine-tune SpeciesNet's classifier head** (in the MD+SN
      ensemble) on the 225-class taxonomy and evaluate — run 2026-08-05/06,
      on the A40 (not the 3060 as originally tagged; scaffolding fit
      comfortably even on the shared/contended box). Added the missing
      `make speciesnet-build/start/stop/shell/finetune` targets
      (`Makefile`) — `speciesnet-start` also needed
      `--dns 100.100.100.100 --dns 192.168.178.2` since containers on this
      host can't otherwise resolve the Tailscale-hosted MLflow server.
      First resynced this machine's `data/real/annotations_*.json` from
      `gpu-server` — it had the pre-contamination-flagging copy §2.1
      flagged as stale, resolving that open question for the A40 as a side
      effect.

      **Two real bugs found by actually running training, not by
      inspection** (both fixed in `scripts/training/teacher_finetune/`):
      `ONE_CYCLE_MAX_LR=1e-3` (copied by convention from the detector
      pipelines' "10x base LR" rule) reliably diverges for this
      classifier/freeze-fraction/AMP combo — confirmed via isolated
      fixed-LR probes (gradient norms already `inf` within ~10 steps, NaN
      loss by ~27); lowered to `3e-4` (`constants.py`), empirically stable.
      Missing gradient clipping (`training_pipeline.py`) — same class of
      bug as the sibling detector pipelines' §3.3 fix, added
      `unscale_`+`clip_grad_norm_`. Even at the corrected LR, occasional
      hard batches under AMP still overflow and permanently poison
      BatchNorm running stats and, via the unconditional EMA update, the
      EMA copy used for eval/checkpointing — verified directly (one full
      run's `last.pt` came out with 43.5M non-finite values after a single
      unhandled batch). Added a finite-check guard that snapshots
      BatchNorm buffers before each forward pass and restores them if that
      batch's loss/logits are non-finite, skipping backward/EMA/scheduler
      for it entirely — verified clean (zero non-finite values in
      `best.pt`/`last.pt`) across three full runs, including two that hit
      sustained near-100%-batch-failure episodes and had to be killed.

      **Result, confirmed reproducible across two independent full runs**
      (byte-identical epoch-by-epoch losses/val-metrics through epoch 5,
      identical final test numbers): `best.pt` at epoch 4
      (`val f1_macro=0.7645`) — training past that point oscillates and
      eventually destabilizes rather than improving, a genuine ceiling for
      this LR/architecture/freeze-fraction setup, not noise. Final test-set
      eval (92,094 samples): `accuracy_top1=0.6688`, `f1_macro=0.5390`,
      `f1_micro=0.6688` (per-source: 98.5% coco_humans, 82.6% images_cv,
      68.1% inaturalist, 64.3% gbif, 59.4% wikimedia, 43.0% openimages) —
      per `teacher_finetune/README.md`'s documented ceiling caveat, macro
      metrics are capped ~4.9 points below 100% by the 11/225 classes with
      no matching SpeciesNet leaf class. Checkpoint only exists on this A40
      machine so far (gitignored, not yet rsynced to the NAS backup per
      §1.3's durability step) — `scripts/training/teacher_finetune/model_exports/teacher-finetune-20260806-131233/best.pt`.
      **2026-08-13: downstream steps for §4.4 done.** Cached teacher soft
      labels for both splits (`cache_soft_labels.py --split train`:
      187,705 records; `--split val`: 19,732 records — both match their
      annotation counts exactly) — `data/real/teacher_soft_labels_{train,val}.jsonl`.
      Re-ran `predict_ensemble.py --checkpoint best.pt` for the fine-tuned
      MD+SN ensemble predictions (`megadet_speciesnet_ensemble/model_exports/finetuned-teacher-finetune-20260806-131233/`)
      — still running as of this writing (real+synth test sets through the
      full MD→SN pipeline, ~8h estimated); not yet scored against ground
      truth. NAS backup of `best.pt` still not done (§1.3's durability
      step), by request.
- [ ] **4.2 [3060] (gap) KD ladder Phase 0 — zero-shot baselines** (untrained
      teacher/student) — needed as the floor for the Phase 4 comparison
      table. **Teacher zero-shot is already done and current**: the
      off-the-shelf MD+SN eval at
      `scripts/training/megadet_speciesnet_ensemble/model_exports/pretrained/eval/evaluation_report.md`
      (mixed mAP 0.487 / real 0.445) was verified 2026-08-13 to already be
      scored against the *current* (post-contamination-flagging)
      `annotations_test.json`, not stale — no rerun needed.
      **Student zero-shot attempted 2026-08-13, failed**: `uv run -m
      scripts.training.yolo26n.eval_suite.run_evaluation --checkpoint
      weights/yolo26n.pt` (raw COCO weights, 225-class head untrained —
      the standard reading of "zero-shot student") crashed near the end of
      the real+synth pass with `FileNotFoundError` on
      `data/blanks/images/blank_211.jpg` — see new gap **4.6** below. Not
      yet retried.
- [ ] **4.3 [3060] Retrain YOLOv5s** with the new anchor/loss-autoscaling
      implementation (`autoanchor.py` fix from
      `docs/progress_notes/2026-07-16_yolov5s-underperformance-hyp-scaling-fix.md`)
      — **running as of 2026-08-13**, dispatched to `gpu-server` (3060) per
      the machine tag, fresh run (no `--resume-from`, confirmed via
      `hyp_cls_effective=1.40625` in the run log — the fix is active).
      `val mAP50_95` climbing steadily each epoch (0.05 → 0.10 → 0.14 →
      0.16 by epoch 4), no instability. Multi-day job (the last comparable
      full run took ~6.5 days elapsed) — check
      `docker exec training-container cat /tmp/yolov5s_retrain.log` on
      gpu-server for current status.
- [ ] **4.4 [3060, or A40 once §3.1 frees it] Train YOLO26n with knowledge
      distillation**, MD+SN ensemble as teacher (Phase 3 of
      `docs/plans/2026-06-30_knowledge-distillation-and-teacher-finetuning-strategy.md`).
      **2026-08-13: `--kd` wiring validated for the first time** —
      `smoke_test_kd_loss.py` and `run_training_pipeline.py --kd --smoke`
      both pass end-to-end (forward → KD-loss blend → backward →
      checkpoint → eval) against the real cached teacher soft labels from
      §4.1. **Full run (`--kd --full-eval`, default `T=4/α=0.5` point)
      attempted, failed** at epoch 1 (~28min in) on the same
      `data/blanks/` `FileNotFoundError` as §4.2 — see gap **4.6**. Not yet
      retried; the 4-point `(T,α)` hyperparameter grid the strategy doc's
      §3.5 calls for is deferred until a single run completes (scope
      decision: one default-point run is the §4.4 bar, not the full grid).
- [ ] **4.5 [Either, no GPU] (gap) KD ladder Phase 4 — final comparison
      synthesis**: assemble direct-FT vs. teacher-FT vs. KD results into the
      comparison the strategy doc's experimental ladder is building toward.
      Pure analysis over the (git-tracked) eval reports from §4.1–4.4 —
      needs §2.1 fixed first so those evals were affordable to produce.
- [ ] **4.6 [Either] (gap) `data/blanks/` (negative/no-object training
      images) is incomplete on at least two machines** — discovered
      2026-08-13 when it broke both §4.2 and §4.4. This A40 has 170 images
      under `data/blanks/images/`, `gpu-server` has 174, but annotations
      reference indices up to at least `blank_216` — neither machine has a
      complete set, and the NAS backup (`data-server`) isn't SSH-trusted
      from the A40 yet (`ssh-copy-id` not done, per §1.3's prerequisite
      note). Blocks §4.2's student zero-shot eval and §4.4's KD training
      (and, unverified, may also affect any other pipeline that pulls
      negative examples from this directory) until the authoritative
      source/count is located and resynced.

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
