# yolo26n KD full training run: narrow blanks-gap root cause, DNS/OOM environment fixes, completed run

**Status: complete — training finished, full eval suite ran, results below.**

## Plan / what we're about to do

TODO.md §4.4 (KD training, MD+SN ensemble as teacher) has been blocked since 2026-08-13,
when the one prior full-run attempt (`--kd --full-eval`, default point `T=4/α=0.5`)
crashed ~28 minutes into epoch 1 on `FileNotFoundError: cv2.imread returned None for:
/app/data/blanks/images/blank_149.jpg`. §4.6 framed this as a large, unresolved gap
("A40 has 170, gpu-server has 174, neither complete").

Re-investigated this session — the actual gap is narrow:

- `data/real/annotations_{train,val,test}.json` reference exactly **174** unique
  `blank_N.jpg` filenames. This A40 had 170/174 — missing exactly `blank_149.jpg`,
  `blank_157.jpg`, `blank_210.jpg`, `blank_211.jpg` (the same file that also crashed
  §4.2's student zero-shot eval attempt).
- These are curated, human-reviewed true-negative photos (a Label Studio export
  correcting classifier false positives to "blank") — not regeneratable by any script in
  the repo; the only fix is sourcing the actual missing files.
- `gpu-server` (the 3060) has the complete 174/174 set. User ran `ssh-copy-id` to
  `debian@gpu-server.taile550ef.ts.net` from this A40 to enable key auth, then the 4
  files were rsynced over directly:
  `rsync -avh debian@gpu-server.taile550ef.ts.net:~/Master-Thesis/data/blanks/images/blank_{149,157,210,211}.jpg data/blanks/images/`.
  `data/blanks/images/` now has 174/174 files on this machine. No code change was needed
  (the dataset loader — `scripts/training/yolov5s/dataset.py::CocoYoloDataset._load_raw`,
  shared by yolov5s and, via `scripts/training/yolo26n/kd_dataset.py`'s
  `KDCocoYoloDataset` composition, by yolo26n/KD too — still hard-crashes on any missing
  file; we're avoiding that path entirely by having the real files instead of working
  around their absence).

Plan: validate with the two existing smoke tests
(`smoke_test_kd_loss`, `run_training_pipeline.py --kd --smoke`), then launch the full run
(`--kd --full-eval`, default point `T=4, α=0.5`, COCO-pretrained init — isolates the KD
signal from Phase-1 fine-tune weights, per the strategy doc), monitor to completion, and
update TODO.md §4.4/§4.6 with the outcome.

## Two environment bugs found while validating (unrelated to the blanks gap)

1. **MLflow DNS resolution failure inside `training-container`.** The plain `make run`
   target starts the container without the Tailscale MagicDNS servers, so
   `mlflow.set_experiment()` couldn't resolve `hetzner.taile550ef.ts.net` and crashed at
   startup, before touching any data. `speciesnet-start` already solved the identical
   problem for the SpeciesNet container via `--dns 100.100.100.100 --dns 192.168.178.2`
   (Makefile, `speciesnet-start` target) — the plain `run` target never got the same
   fix. Worked around for this session by restarting `training-container` manually with
   the same `--dns` flags; the `run` target itself is still missing them (a real gap
   worth porting the fix into, not fixed in this note — flagging for a follow-up).
2. **Shared-tenancy GPU OOM**, same class of issue TODO.md §1.1 already documents: a
   retry of `--kd --smoke` (after the DNS fix) hit `CUDA OutOfMemoryError` inside
   `TaskAlignedAssigner` — other vGPU tenant processes were holding ~21.5GB of the
   23.78GB slice at that moment (420MB free), unrelated to our own batch size or the
   blanks fix. Resolved by retrying once `nvidia-smi` showed the GPU idle again (0/24576
   MiB) — not a code bug, matches the exact pattern already seen on 2026-08-13.

## Command run

```bash
# smoke validation (inside training-container, restarted with --dns flags):
PYTHONPATH=. uv run -m scripts.training.yolo26n.smoke_test_kd_loss
PYTHONPATH=. uv run -m scripts.training.yolo26n.run_training_pipeline --kd --smoke

# full run (pending smoke-test confirmation):
PYTHONPATH=. uv run -m scripts.training.yolo26n.run_training_pipeline --kd --full-eval
```

## What happened / Results

Both smoke tests passed cleanly with the completed 174-file blanks set and the `--dns`
fix — `smoke_test_kd_loss` (4/4 checks) and `--kd --smoke` (1 epoch on val split, train →
val eval → test eval, no crash). The full run was then launched
(`run_name=yolo26n-kd-20260825-164250`, `experiment=yolo26n-wildlife225`,
MLflow run `0598b43c70054526898cc45c84aa05d6`) and ran to completion:

- **Training**: started 2026-08-25 16:43 UTC, early-stopped 2026-08-29 18:36 UTC at
  epoch 182 (`epochs_since_improve=20/20`, the configured patience). Best checkpoint at
  epoch 161, val `mAP50_95=0.6553`. `train_batches=2277`, `val_batches=196`,
  `test_batches=997`. One transient MLflow `log-metric` connection timeout around
  epoch 8→9 (network blip to `hetzner.taile550ef.ts.net`), self-recovered — no data lost,
  training unaffected.
- **Post-training test eval** (real-only, training pipeline's own quick pass):
  `test mAP50=0.5470 mAP50_95=0.4789`.
- **Full eval suite** (`--full-eval`, granularity × band × domain report against
  `best.pt`) ran 2026-08-29 21:07 → 2026-08-30 03:26 UTC (~6h19m) and completed without
  error, writing `evaluation_report.{md,json}`, `eval_per_class.csv`,
  `eval_band_grid.csv`, `eval_confusion_pairs.csv` under
  `scripts/training/yolo26n/model_exports/yolo26n-kd-20260825-164250/evaluation/`.
- Total wall-clock, launch to finished report: **~4.45 days**.

**Tier 1 headline** (mixed = primary metric per `CLAUDE.md`'s evaluation-strategy
convention; real = the always-reported breakout):

| Metric | mixed (headline) | real (breakout) |
|--------|------------------|------------------|
| mAP | 0.510 | 0.479 |
| mAP50 | 0.570 | 0.547 |
| mAP75 | 0.538 | 0.509 |

Statistical hygiene note from the report: mixed mAP rises to 0.530 excluding the 9
test-limited (<30 real img) classes; count-weighted (micro) mixed mAP is 0.564.

**Comparison against existing baselines** (same eval suite, same test set):

| Model | mixed mAP | real mAP |
|-------|-----------|----------|
| Teacher, pretrained (zero-shot MD+SN, §4.2) | 0.487 | 0.445 |
| Teacher, fine-tuned (§4.1) | 0.549 | 0.536 |
| Student, direct FT (Phase 1, no teacher, `2026-07-22_yolo26n-eval-oom-crash-diagnosis-and-fix.md`) | 0.523 | 0.481 |
| **Student, KD (this run, Phase 3, T=4/α=0.5)** | **0.510** | **0.479** |

**This is the headline finding of this run: at this single hyperparameter point, KD did
not outperform direct fine-tuning.** It's essentially on par — marginally lower on the
mixed headline (-0.013) and statistically indistinguishable on the real breakout
(-0.002) versus the Phase 1 direct-FT baseline. This is one point in the strategy doc's
4-point `(T,α)` grid, not the full sweep — a single negative/neutral result here doesn't
close the question, but it means the default point alone doesn't demonstrate a KD
advantage for this task.

**Domain-shift watchdog** (Tier 2.3, per `docs/plans/2026-06-10_model-evaluation-strategy.md`):
mean real−synthetic Δ (fine granularity) = **-0.238**, ranging from -0.142 (Band A) to
-0.390 (Band C) — a fairly large, systematic gap. Per the watchdog policy this is worth
flagging for whoever reviews the default mixed/real evaluation axes next, though it's
consistent with the KD run's own band-grid numbers (Tier 2.2) and not a new anomaly
specific to this checkpoint.

Checkpoint: `scripts/training/yolo26n/model_exports/yolo26n-kd-20260825-164250/best.pt`
(gitignored, exists only on this A40 — not yet rsynced to the NAS backup, per §1.3's
durability step).

## Next steps

- **§4.5 (Phase 4 comparison synthesis)** can now proceed — all of §4.1–4.4's eval
  reports exist.
- **§4.6 (blanks gap)** is resolved on this A40 (174/174) via the `gpu-server` rsync
  above. The NAS-side durability sync (`ssh-copy-id` to `data-server`) is still a
  separate, open item per §1.3 — not required for this task, but worth doing so the
  complete set has a backup beyond `gpu-server`.
- **The `(T,α)` hyperparameter grid** (`T∈{4,8} × α∈{0.5,0.7}`) remains deferred, per the
  strategy doc's own scope — this run only covers the default point. Given the result
  above, the grid is now more clearly worth running before drawing a final KD-vs-direct-FT
  conclusion, rather than treating this single point as representative.
- **Two environment gaps found this session, not yet fixed**: the plain `make run`
  target is missing the `--dns` flags `speciesnet-start` already has (§ above) — worth
  porting over so this doesn't need a manual workaround next time; and the MLflow
  transient-timeout resilience (self-recovered here, but a retry/backoff review might be
  worth it if it recurs more persistently in a future run).
