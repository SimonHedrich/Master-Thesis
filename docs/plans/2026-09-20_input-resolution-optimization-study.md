# Input-Resolution Optimization Study — the final training run

**Date:** 2026-09-20
**Status:** In progress — see the Progress log at the end of this document.
**Feeds:** `thesis/manuscript/chapters/4-Results.tex` §`sec:results_resolution_tradeoff` (new),
§`sec:results_embedded_benchmark` (figure repoint), `5-Discussion_and_Conclusion.tex` §Further Work
**Related:** `docs/2026-03-10_object-detection-models-for-embedded-systems.md` (the candidate
universe), `docs/2026-09-18_embedded-benchmark-results.md` (the measured baseline this study
extends), `docs/plans/2026-09-17_on-device-benchmarking-plan.md` (the harness reused here)

---

## 0. Context

Three days of wall-clock remain before all numbers must be frozen. Four models are already
trained: YOLO26n (direct fine-tune, and KD — the post-Band-A-fix KD run is still executing on
`gpu-server`), YOLOv5s, and the fine-tuned MegaDetector+SpeciesNet teacher.

The question this document answers: **can any other model from
`docs/2026-03-10_object-detection-models-for-embedded-systems.md` be (1) fine-tuned in the
remaining time and (2) contribute a research aspect the thesis does not already have?**

**Answer: no other architecture clears both bars.** The binding constraint is not engineering
effort — it is that no 640 px detector can complete the 200-epoch protocol on this dataset in
72 h. What *does* fit, and supplies the aspect the thesis is currently missing entirely, is a
**reduced-input-resolution YOLO26n (320 px)**.

### 0.1 The time arithmetic that disqualifies every architecture candidate

Measured epoch times on the full 155,808-image training set, read from the run logs:

| Run | Setting | s/epoch (train) | Wall clock |
|---|---|---:|---|
| `yolo26n-bs32-20260910-212812` | 640 px, effective bs 32 | 2,024–4,896 | 193 epochs / **8.9 d** |
| `yolo26n-kd-20260825-164250` | 640 px, bs 64 (fastest ever observed) | 1,180–1,220 | 182 epochs / **4.45 d** |
| `yolov5s-20260909-230900` | 640 px, bs 32 | 1,666–1,994 | 200 epochs / **4.96 d** |

200 epochs in 72 h requires **≤ ~1,150 s/epoch including the per-epoch validation pass**. The
fastest 640 px epoch ever recorded on this hardware is ~1,200 s of *training alone*. Therefore
no new 640 px model — YOLO11n, YOLOv12n, DAMO-YOLO-T, EfficientDet-Lite0, anything — can finish
the standard protocol in the window.

This matters because of the acceptance rule set for this run: at the 72 h mark the validation
curve is inspected and the run is kept only if it has substantially plateaued. A 640 px run
would sit at roughly epoch 50–120 of 200 with the OneCycle LR still near its maximum and
nowhere near annealed — i.e. it would be discarded, and the three days would yield nothing.

### 0.2 Candidate-by-candidate verdict

| Candidate | (1) Trainable in 3 d? | (2) New research aspect? | Verdict |
|---|---|---|---|
| **YOLO11n / YOLOv12n** | **No** — 640 px ⇒ 5–9 GPU-days. The engineering itself is cheap: a `DetectionModel(cfg="yolo11.yaml")` swap, and `KDv8DetectionLoss` in `scripts/training/yolo26n/kd_loss.py` already exists for single-head detectors, with the NMS eval path already present in `yolov5s/evaluation.py` | Yes — would replicate the KD-vs-FT result on a single-head architecture, testing whether YOLO26n's negative KD result is an artefact of `KD_APPLY_TO="one2one"` diluting the teacher blend across E2ELoss's dual heads. But that needs **two** matched runs (FT + KD) = 10–18 GPU-days | **Reject** — the strongest scientific complement, but two 640 px runs. Record as Further Work |
| **NanoDet-Plus-m** | **No** — not on PyPI (verified); external repo pinned to pytorch-lightning <2.0 against this project's Python 3.13 / torch 2.11. Port + dataset adapter + eval adapter before a single epoch runs | Yes — the only shortlisted model with a published sub-30 ms ARM figure (19.77 ms via NCNN) | **Reject** — already recorded as "no pipeline in this repo at all" in `docs/synthetic-model-comparison/11_detector-architecture-selection.md` |
| **PicoDet-S** | **No** — `paddlepaddle` + `paddledet` *do* resolve on Python 3.13 (verified), but it is an entirely separate training framework, config system and export path | Yes — fastest recorded ARM CPU inference (4.8 ms @ 320 px) | **Reject** — framework risk far exceeds the window |
| **EfficientDet-Lite0** | Marginal — `effdet` resolves and is PyTorch/timm-native and COCO-native, but it is still a new training loop at 640 px-class cost | **Weak** — 25.7 COCO mAP; duplicates the "legacy stable baseline" role YOLOv5s already fills | **Reject** |
| **RT-DETRv2 / any DETR** | **No** — DETRs need long schedules to converge | No — the source document already rejects transformers for the Hexagon 685 (no LayerNorm/MHA instruction support) | **Reject** |
| **MobileNetV3-SSD** | Marginal | No — "extremely low relative accuracy", superseded by anchor-free methods | **Reject** |
| **YOLO26n @ 320 px** | **Yes** — 4× fewer FLOPs per image; projected ~700 s/epoch ⇒ 200 epochs in ~40 h. Zero new architecture code: `IMAGE_SIZE` is a single constant accessed attribute-style (`constants.IMAGE_SIZE`) everywhere, and `scripts/benchmark/constants.py` already carries per-model `image_size` entries (640/480/1280) | **Yes** — §0.3 | **Recommended — this document** |

### 0.3 Why input resolution is the aspect that is actually missing

The thesis is titled around *optimizing* detection models for real-time embedded inference, and
as of today it contains **no successful optimization result**:

- KD changed latency not at all (401 ms direct-FT vs 419 ms KD `W_infer` on the Pi 400) and did
  not beat the direct-FT baseline on accuracy.
- QAT was never carried out (`TODO.md` §5.1) and is itself named "the single largest lever on
  the headline result".
- `docs/2026-09-18_embedded-benchmark-results.md`: **nothing meets ≤30 ms** — YOLO26n is ~12×
  over budget.

Input resolution is the one classical optimization lever that is both affordable in the
remaining window and measurable end-to-end on hardware that is currently reachable. It converts
"nothing is close to the target" into a measured accuracy-vs-latency curve with a defensible
operating point — which is what an embedded-optimization thesis needs in place of a single
failed point estimate. It is also a properly *controlled* experiment: same architecture, same
data, same 200-epoch protocol, one variable changed.

The study has two arms:

- **Arm 1 — no retraining.** The existing 640 px checkpoint evaluated and benchmarked at
  320/416/512/640. Cheap, and it is a complete deliverable on its own.
- **Arm 2 — resolution-native training.** YOLO26n trained from COCO at 320 px under the
  standard protocol, showing how much of Arm 1's accuracy loss native training recovers.

Arm 1 is executed first precisely so that a thesis result exists even if Arm 2 is discarded at
the 72 h gate.

---

## Phase A — this document

Write this plan to `docs/plans/`, add it to the `docs/README.md` index, commit to `main`.
The Progress log at the end is the single place execution state is recorded: update it at the
end of every phase, and append any deviation from the plan (a go/no-go branch taken, an abort,
a changed batch size) as a dated bullet beneath it.

## Phase 0 — Arm 1, the guaranteed deliverable (~3 h, no training)

Produces the accuracy-vs-latency curve from the **existing**
`scripts/training/yolo26n/model_exports/yolo26n-bs32-20260910-212812/best.pt`.

1. Add `--image-size` to `scripts/training/yolo26n/eval_suite/run_evaluation.py` — an argparse
   flag feeding the existing `image_size` parameter (line 99). It is already threaded through
   to `predict.py`, and `predict.py` writes `image_size` into its cache manifest, so resolution
   variants will not collide in the prediction cache.
2. Evaluate the 640 px checkpoint at inference resolutions **320 / 416 / 512**, with
   `--limit 8000` per domain to hold each pass to roughly 20 minutes. Re-score **640 at the
   same `--limit`** so all four points share an identical subsample (the published full-test
   figures of 0.599 mixed / 0.529 real stay the headline; this subsample is only for the curve).
3. On the Pi 400: add 320/416/512 entries for `yolo26n-direct` to
   `scripts/benchmark/constants.py`, re-run `scripts/benchmark/1-export_models.py`, then
   `scripts/benchmark/pi/run_all.sh`. This runs entirely off the A40, so it does not contend
   with Phase 2. ONNX Runtime only — multi-threaded TorchScript SIGILLs on YOLO26n on this core
   (`reports/embedded_benchmark/torchscript_sigill_finding.md`).

## Phase 1 — timing probe and go/no-go (~1 h)

4. Add `--image-size` to `scripts/training/yolo26n/run_training_pipeline.py`, implemented as
   `constants.IMAGE_SIZE = args.image_size` at the top of `main()`. Verified safe: every read is
   attribute-style (`yolo26n_model.py:90`, `evaluation.py:89-92`,
   `run_training_pipeline.py:120/127/142`) and happens inside functions, so the mutation
   propagates. Log the effective value in the run config dump.
5. Run `smoke_test_loss_and_decode`, then `run_training_pipeline --smoke --image-size 320`.
6. Raise `NUM_WORKERS` 8 → 16 for this run. The A40 box has 24 cores and is idle; the historical
   4× spread in epoch times is co-tenancy, and these runs are partly dataloader-bound. This
   changes throughput only, not the optimization math.
7. Time 2 real epochs. **Go/no-go rule, applied without further consultation:**
   - projected 200-epoch wall clock **≤ 60 h** → proceed at `--batch-size 32`, matching the
     640 px baseline exactly so that resolution is the only changed variable;
   - **60–70 h** → switch to `--batch-size 64`. `scripts/training/yolo26n/README.md`'s
     comparability contract explicitly exempts `BATCH_SIZE` and requires only that it be
     logged, and the Aug-25 KD run used 64, so there is precedent;
   - **> 70 h** → abort Arm 2, keep Phase 0 as the deliverable, and report that outcome.

## Phase 2 — the run (~40–60 h)

8. `EPOCH_COUNT=200`, early-stop patience 20, `SEED=42`, all `AUG_*` values and eval thresholds
   untouched. Run directory `yolo26n-320-<timestamp>`.

   ```bash
   nohup docker exec training-container env PYTHONPATH=. \
     uv run -m scripts.training.yolo26n.run_training_pipeline \
     --image-size 320 --batch-size 32 &
   ```

9. **Do not touch `gpu-server`.** The post-fix KD run (`yolo26n-kd-bs16-20260916-101612`) is
   still executing there and is the blocking item for §4.3.1.

## Phase 3 — the 72 h decision gate

10. Inspect the validation `mAP50_95` curve. Keep the run if it has plateaued — the existing
    runs' best epochs land at 161–194 of 200, so late convergence is the norm here — and
    discard it if it is still climbing steeply.
11. If kept: `eval_suite.run_evaluation --run-dir <dir> --image-size 320` on the full test set
    (1–5 h), then export and benchmark on the Pi exactly as in Phase 0 step 3, producing the
    resolution-native arm of the curve.

## Phase 4 — write-up

12. New subsection under §4.3 of `thesis/manuscript/chapters/4-Results.tex`
    (`sec:results_resolution_tradeoff`): the curve, both arms, and the resulting
    best-achievable operating point against the 30 ms / 500 MB target.
13. Record in §5 Further Work: the YOLO11n KD replication (single-head vs E2ELoss dual-head
    `KD_APPLY_TO="one2one"` dilution) and NanoDet-Plus-m / PicoDet-S, citing §0.1's arithmetic
    as the reason they were not run.

## Incidental fix folded in (~10 min)

14. `scripts/benchmark/5-report.py`'s `MAP_SOURCES` (lines 46–51) points `yolo26n-direct` at the
    **pre-Band-A-fix** report `yolo26n-20260715-010031` (0.523 / 0.481) instead of
    `yolo26n-bs32-20260910-212812` (0.599 / 0.529). The published
    `embedded_latency_vs_map.png` therefore plots stale accuracy against fresh latency and is
    0.076 mAP too low on its YOLO26n point. Repoint and regenerate — the new resolution curve
    lands in that same figure.

## Verification

- This document exists, is linked from `docs/README.md`, and its Progress log reflects the last
  completed phase at every point during execution.
- `smoke_test_loss_and_decode` and `--smoke --image-size 320` pass before the real run starts.
- The run's startup config dump logs `IMAGE_SIZE = 320` and `batch_size_effective`.
- `train_batches` equals **4,869** at batch size 32 (155,808 / 32) — confirms the Band-A merged
  dataset is loaded, not the real-only one.
- Phase 0's 640 px `--limit 8000` score lands near the published 0.599 mixed / 0.529 real; a
  large gap means the subsample or the new flag is wrong.
- Pi parity via `scripts/benchmark/3-bench_parity.py`: host-vs-device detection match stays in
  the 98.9–99.5 % band established for the 640 px models.

## Progress log

| Phase | Status | Date | Notes / artefacts |
|---|---|---|---|
| A — plan document | ✅ done | 2026-09-20 | this file; indexed in `docs/README.md`; commit `13b1ea4` |
| — code plumbing | ✅ done | 2026-09-20 | `--image-size` on training + eval suite, 320/416/512 export specs, 2 latent bugs fixed; commit `296e0e9` |
| 0 — Arm 1 accuracy (GPU) | ✅ done | 2026-09-20 | mixed mAP 0.6134 / 0.5799 / 0.5379 / 0.4609 at 640/512/416/320 |
| 0 — Arm 1 latency (Pi 400) | ✅ done | 2026-09-20 | 30 new cells, gate PASS ×3; W_e2e 423 / 283 / 180 / **109 ms** at 4 threads |
| 0 — Arm 1 report | ✅ done | 2026-09-20 | `scripts/benchmark/6-resolution_report.py` → `reports/resolution_study/` |
| 1 — probe + go/no-go | ✅ done | 2026-09-20 | bs32 measured 124 h → **fail**; re-specced to bs128 + lr×2 + eval-every-2 → **49 h, go** |
| 2 — 200-epoch run @ 320 px | 🔄 running | 2026-09-20 | `yolo26n-res320-bs128-20260920-165602`, started 16:56 UTC; **1,012 s/epoch over 5 completed epochs → ETA Sep 23 ~09:00**; full 200 epochs confirmed by the user |
| 3 — 72 h gate | ⬜ not started | | keep / discard + reason |
| 4 — write-up | ⬜ not started | | §4.3 subsection, figure |
| Incidental — `MAP_SOURCES` fix | ✅ done | 2026-09-20 | repointed to 0.599/0.529; `embedded_latency_vs_map.png` regenerated |

### Deviations from the plan

- **2026-09-20 18:35 — timing re-measured from completed epochs: 1,012 s/epoch.** Both
  earlier figures (511 s, then 852 s) came from step-rate windows sampled mid-epoch,
  and both read optimistic because the page cache was warm over images the previous
  run had just read. The five completed epochs are 959.9 / 1038.6 / 997.3 / 1001.0 /
  1062.7 s, mean **1,012 s**. Projection: 56.2 h train + ~7.8 h validation ≈ **64 h**,
  finishing Sep 23 ~09:00 against a Sep 23 13:00 deadline. **Lesson for the rest of
  this run: quote timings only from completed epochs, never from a step window.**
- **2026-09-20 — Arm 2 needs no new on-device benchmark.** Latency depends on
  architecture and input shape only, never on weight values (`scripts/benchmark/README.md`
  makes this explicit, and it is why the campaign could benchmark an
  architecture-only YOLOv5s export). The resolution-native 320 px model has the same
  architecture and the same 320×320 input as the `yolo26n-direct-res320` export
  already measured, so its W_infer/W_e2e are the cells already in
  `latency_summary.csv` (102.5 / 109.0 ms at 4 threads). Arm 2's remaining cost is
  therefore the full test eval alone (~1–2 h at 320 px) plus regenerating
  `6-resolution_report.py` — not the ~3–4 h a fresh Pi campaign would take.
- **2026-09-20 — decision (user): run the full 200 epochs rather than restart.** The
  alternatives offered were a 160-epoch restart (~17 h margin; a complete annealed
  OneCycle landing where the 640 px baseline actually peaked — best @ epoch 172, KD
  best @ 161) and a bs256 restart keeping 200 epochs (~1.4× on the training half, at
  the cost of LR scaling to √8 ≈ 2.8× against documented AMP instability). The user
  chose the full 200 epochs, accepting ~4 h of margin. Arm 1 is already complete and
  committed, so a late failure costs Arm 2 only, not the study.

- **2026-09-20 — corrected projection: ~55.6 h, not 36.7 h.** The 36.7 h figure was
  extrapolated from a 0.42 s/step sample taken over 50 steps immediately after the
  previous run was killed, with the page cache still warm over the same images.
  Steady state is **0.70 s/step → 852 s/epoch**, measured over epoch 1 (959.9 s
  including startup) and mid-epoch-2:

  | | extrapolated | measured |
  |---|---|---|
  | train / epoch | 511 s | **852 s** |
  | train, 200 epochs | 28.4 h | **47.3 h** |
  | validation, 40 passes | 8.3 h | 8.3 h |
  | **total** | **36.7 h** | **≈55.6 h** |

  The `--eval-schedule` saving is unaffected (validation cost does not depend on step
  rate); only the baseline it was subtracted from was wrong. ETA ~Sep 23 01:00
  against a Sep 23 13:00 deadline — ~12 h of slack for the full-test eval, export,
  Pi benchmark and write-up. Remaining lever if more margin is needed: GPU
  utilisation is still only ~50 % at bs128, so bs256 would likely buy another
  1.3–1.5× on the training half, at the cost of a third restart and pushing LR
  scaling to √8 ≈ 2.8× — declined for now as the run already fits.

- **2026-09-20 — non-uniform validation schedule (`--eval-schedule 0:20,100:5,150:2`),
  replacing the uniform every-2.** Under OneCycleLR the peak LR anneals toward zero,
  so essentially all late improvement — the part that decides `best.pt` and whether
  the curve has plateaued — lands in the final third. A uniform interval spends most
  of the validation budget at epochs from which no checkpoint will ever be selected.
  The schedule validates at epochs 20/40/60/80/100, then every 5th to 150, then every
  2nd to 200: **40 evaluations instead of 100**, 8.3 h of scoring instead of 20.9 h.

  | | uniform every-2 | scheduled |
  |---|---|---|
  | evaluations | 100 | **40** |
  | validation time | 20.9 h | **8.3 h** |
  | projected total | 49.2 h | **≈36.7 h** |

  Verified before restarting: the schedule yields exactly 40 evaluations with the
  final epoch always included, and both pre-existing paths (no schedule, and a
  uniform `eval_every`) are bit-identical to before. Known cost: a divergence
  between two evaluations is visible only in the train loss until the next one —
  acceptable, since the train loss is logged every `LOG_EVERY_N_STEPS` steps
  regardless. Also note `best.pt` does not exist until epoch 20; `last.pt` is still
  written every epoch, so a crash before then is still resumable.

- **2026-09-20 — the go/no-go rule failed at bs32, and the fix was not the one the
  rule anticipated.** Measured epoch 1 of `yolo26n-res320-bs32-20260920-160041`:
  **1,476.7 s train + 751 s validate = 37.1 min/epoch → 124 h for 200 epochs**,
  against ~69 h remaining. Phase 1's rule said 60–70 h → raise the batch size; this
  was far past even that, and batch size alone could not have fixed it:

  - **The validation pass is the hidden fixed cost.** It is dominated by
    single-threaded 225-class mAP scoring — one process at 106 % CPU with the
    machine 93 % idle and no IO wait — and it costs the *same* ~12.5 min at 320 px
    as the 13–16 min the 640 px run spent. Resolution does not touch it. Across 200
    epochs that is **43 h of pure scoring**, independent of every other knob.
  - **Training was latency-bound, not compute- or data-bound.** At bs32: GPU
    14–27 %, whole machine 21 % CPU, 48 dataloader workers averaging 8 % each, zero
    IO wait — *nothing* saturated. A 2.71M-param model fed 32 images per step
    spends its time in per-step Python and kernel-launch overhead. This is the
    regime where a larger batch pays, and the 640 px runs already showed it
    (bs64 was 1.7× more efficient per image than bs32).

  Re-specced and restarted with three changes, at the user's explicit preference
  for shorter wall clock over an exact batch-size match:

  | | before | after |
  |---|---|---|
  | batch size | 32 | **128** (2.3 → 7.6 GB of 24 GB) |
  | `ONE_CYCLE_MAX_LR` | 0.01 | **0.02** |
  | validation | every epoch | **every 2nd epoch** |
  | throughput | 105 img/s | **305 img/s (2.9×)** |
  | projected 200 epochs | 124 h | **≈49 h** |

  **Why the LR moved with the batch:** the pipeline has no gradient accumulation and
  does not scale LR with batch size, so 4× the batch at an unchanged peak LR means
  4× fewer optimizer steps at the same step size — it would undertrain. Scaled by
  the conservative √ rule (×2) rather than linear (×4), because the teacher run has
  a documented history of AMP instability at high LR. With `eval_every=2`,
  `EARLY_STOP_PATIENCE=20` now counts **evaluations**, i.e. 40 epochs — more
  tolerant, not less.

  **Comparability impact:** the 640 px baseline used effective batch 32, so batch
  size is now a second changed variable alongside resolution. This is an accepted
  cost — `scripts/training/yolo26n/README.md`'s comparability contract explicitly
  exempts `BATCH_SIZE` and requires only that it be logged, and the Aug-25 KD run
  already used 64 against the same baseline. The study's reported axis is latency,
  which is weight-independent and wholly unaffected. It must be stated in the
  write-up.
- **2026-09-20 — corrected a stale comment in `scripts/training/yolo26n/constants.py`**
  claiming the applied LR schedule comes from `yolov5s.constants`. It does not —
  `run_training_pipeline.py` passes yolo26n's own values explicitly. Had the comment
  been right, `--lr-scale` would have been a silent no-op that still logged the
  scaled value to MLflow.

- **2026-09-20 — Arm 1 results.** One 640-px-trained checkpoint, four inference
  resolutions, fixed subsample, Pi 400 @ 4 threads:

  | Input | GFLOPs | W_e2e (ms) | QCS605 est. | RSS (MB) | mixed mAP | real mAP | Δ real |
  |---:|---:|---:|---:|---:|---:|---:|---:|
  | 640 | 6.81 | 423.5 | 360–381 | 284 | 0.6134 | 0.5483 | — |
  | 512 | 4.33 | 283.1 | 241–255 | 260 | 0.5799 | 0.5044 | −8.0 % |
  | 416 | 2.85 | 179.8 | 153–162 | 241 | 0.5379 | 0.4529 | −17.4 % |
  | 320 | 1.68 | 109.0 | 93–98 | 226 | 0.4609 | 0.3541 | −35.4 % |

  Latency tracks (res/640)² almost exactly, confirming the forward pass is
  compute-bound on this core. Reducing input 640→320 buys a **3.9× latency
  reduction** and moves the QCS605 projection from ~12× over the 30 ms target to
  ~3×; memory was never the binding constraint. Without retraining it costs 35 %
  of real-only mAP — which is the upper bound Arm 2 exists to beat.
- **2026-09-20 — Phase 1 merged into Phase 2's first epochs.** Rather than a separate
  2-epoch probe run followed by a fresh start, the real 200-epoch run was started and
  its own first epochs are the timing measurement. The run is resumable and abortable,
  so a separate probe would only have cost ~40 min of the budget. The go/no-go
  thresholds are applied unchanged.

- **2026-09-20 — Phase 0 step 2: proportional subsample instead of `--limit 8000`.**
  `--limit N` applies the same N to *both* domains, which would have shifted the mixed
  real:synth image ratio from the full test set's 85:15 to 50:50 and made the `mixed`
  aggregate incomparable to the published headline. Replaced with two fixed-seed (42)
  standalone annotation files built once and reused unchanged at every resolution:
  `data/_resolution_sweep/annotations_test_real_8k.json` (8,000 of 63,802 real) and
  `annotations_test_synth_prop.json` (1,411 of 11,250 synthetic), preserving the ratio.
- **2026-09-20 — the subsample carries a small constant optimistic offset.** The 640 px
  control scores **0.6134 mixed / 0.5483 real** on it, against the published full-test
  **0.5989 / 0.5292** (+0.015 / +0.019). Expected for 8,000-image per-class AP. It is
  constant across resolutions and therefore cancels in the curve, but the curve's
  absolute values must **not** be quoted alongside full-test numbers.
- **2026-09-20 — the resolution curve gets its own figure**, not
  `embedded_latency_vs_map.png` as Phase 4 step 12 originally implied. That figure's
  points are full-test mAP; mixing subsample mAP into it would be inconsistent. The
  incidental `MAP_SOURCES` fix still lands in it, unchanged.
- **2026-09-20 — two latent bugs found and fixed while wiring the study** (commit
  `296e0e9`), both of which would have produced wrong numbers rather than an error:
  `_run_full_evaluation` did not pass `image_size` (a 320 px model would have been
  scored at 640 after training), and `4-score_parity.py` built its host reference at
  the training `IMAGE_SIZE` rather than the model spec's (any reduced-resolution
  export would have been reported as a parity failure that was an artefact).
