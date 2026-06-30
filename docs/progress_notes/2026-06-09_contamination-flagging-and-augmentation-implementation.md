# Implementation Log: Contamination Flagging + Data Augmentation

**Date:** 2026-06-09
**Role:** Orchestrator (Claude Code) driving subagents step-by-step.
**Implements:**
- `docs/plans/2026-06-09_flag-cross-species-contamination-multi-box.md` (Plan 1)
- `docs/plans/2026-06-07_data-augmentation-strategy.md` (Plan 2)

This file is a **persistent, append-only log**. For each step the orchestrator
writes the **instruction block** *before* dispatching the subagent, and the
**result block** *after* it completes — so the work is resumable after any error.

---

## Strategy & Environment Findings

**Key unblock:** Plan 1 Step 0 was specced to require `Dockerfile.speciesnet`
(which does **not** exist in this repo — the Makefile has no `speciesnet-*`
targets, and `data/speciesnet_classes.json` holds only integer indices). The
orchestrator verified that:

- `resources/speciesnet_taxonomy_release.txt` (3538 lines) does **NOT** align
  with the classifier's class indices (2498 classes; only 1.3% match on
  high-confidence records) — so labels cannot be reconstructed from it.
- `pip install speciesnet` **installs cleanly on the host Python 3.13** and the
  classifier (EfficientNetV2-M, ~214 MB) downloads and loads **without Docker**.

→ The orchestrator generated `data/speciesnet_labels.json` (2498 labels) directly
on the host. **Both plans are therefore fully implementable and runnable on the
host with no Docker.** (`speciesnet` downgrades `yolov5` 7.0.13→7.0.11 on install;
the orchestrator restored `yolov5==7.0.13` afterwards. The augmentation imports
`augment_hsv` / `random_perspective` work on both.)

**Host capabilities confirmed:** `torch 2.11.0+cu130` (CUDA available), `cv2 4.11.0`,
`yolov5 7.0.13` (`augment_hsv`, `random_perspective` importable),
`scripts.training.yolov5s.dataset` importable.

**Caveat on `import` of script 7:** the module file is `7-filter_speciesnet.py`
(leading digit + hyphen) → not importable via `import`. Reuse its functions via
`importlib.util.spec_from_file_location`.

### Data shapes (verified)
- COCO ann: `{id, image_id, category_id, bbox:[x,y,w,h], area, iscrowd, source:"megadetector", conf}`.
  `category_id` 1..225; `image.file_name` starts with `data/...`; image carries
  `band` (A/B/C/D) and `source` and `split`.
- `speciesnet_results.jsonl` record: `{filepath, expected_common,
  speciesnet_detections:[{detection_idx, bbox_norm:[cx,cy,w,h]?, megadetector_conf,
  speciesnet_scores:{idx:score}, speciesnet_top1_idx, speciesnet_top1,
  speciesnet_top1_score, crop_size_px, speciesnet_skipped, skip_reason}], n_animal_detections}`.
- `data/speciesnet_labels.json`: `{"<idx>": "uuid;class;order;family;genus;species;common"}`.

### Step plan
- **Plan 1:** S0 dump labels (DONE by orchestrator) → S1 write `14-flag...` →
  S2 run+inspect → S3 visualizer flagged mode → S4 write `15-apply...` (+dry-run).
  S5 verification folded into S2/S4.
- **Plan 2:** P2-S1 constants + basic-set transforms + dataset wiring + assertion + smoke →
  P2-S2 Setup C compositing (mosaic/mixup/close-mosaic) + smoke.
- **REVIEW:** senior-dev pass over everything, fix directly.

Subagents use **sonnet** (token-saving); orchestrator (opus) holds full context
and writes detailed per-step instructions.

---

## P1-S0 — Dump SpeciesNet labels  ✅ DONE (orchestrator)

**Instruction:** Produce `data/speciesnet_labels.json` (idx→label) and a
reproducibility script reusing `load_speciesnet_labels()` logic.

**Result:**
- `data/speciesnet_labels.json` written: **2498 labels**, format
  `{"<idx>": "uuid;class;order;family;genus;species;common"}`.
  Alignment sanity check vs high-conf single-animal records: 62% exact-substring
  (rest are genus/family-level or non-animal top1 like "blank" — expected).
- `scripts/dataset_quality/0-dump_speciesnet_labels.py` written (host-runnable;
  documents the `pip install speciesnet` path and the yolov5 re-pin).
- **No Docker required.** Manual re-run command documented in the script header.

---

## P1-S1 — Write `14-flag_multi_animal_contamination.py`  ⏳ dispatched

**Instruction (to subagent, sonnet):** Create
`scripts/dataset_quality/14-flag_multi_animal_contamination.py` per Plan 1 §4–§5.
- Load `data/speciesnet_labels.json` (idx→label) directly (no Docker).
- Reuse `_compute_match_level`, `load_taxonomy`, `load_classes_225` from
  `7-filter_speciesnet.py` via `importlib` (numeric module name not importable).
- Per source, stream `data/<source>/speciesnet_results.jsonl`; select significant
  detections (`speciesnet_skipped==False`, `megadetector_conf>=MD_CONF` default 0.5);
  skip images with <2 significant detections.
- Per significant detection compute `match_level` via imported `_compute_match_level`.
- Tolerance band default `family`; CLI `--tolerance {genus,family,order}`.
- Confidence gating: flag only if `speciesnet_top1_score>=SN_SCORE` (0.3) AND outside
  band; low-score mismatches → separate "uncertain" list. `LOOKALIKE_GROUPS` empty.
- Outputs in `reports/`: `multi_animal_contamination.csv` (per flagged box),
  `multi_animal_contamination_review.json` (per flagged image),
  `multi_animal_contamination.md` (summary + projected per-class delta). No data writes.
- CLI: `--source {<each>,all}` default all, `--md-conf`, `--sn-score`, `--tolerance`.

**Result:** ✅ DONE. `scripts/dataset_quality/14-flag_multi_animal_contamination.py`
(530 lines). Reuses script 7 via importlib (registered in `sys.modules` before
`exec_module` — required on Py3.13 for its `@dataclass`). Functions:
`load_speciesnet_labels`, `_resolve_expected_class`, `_build_pred_taxonomy`,
`_is_lookalike`, `_evaluate_secondary_box`, `ContaminationStats`, `process_source`,
3 report writers. **openimages run:** 7,688 images → 2,035 with ≥2 significant
boxes → **568 flagged** (27.9% of multi-box; 1.8% of the 32,401 naive bound — band
working). 132 uncertain-only. Reports written (csv/json/md).
**Deviation to review:** non-taxonomic SpeciesNet top1 (blank/vehicle/"animal" —
empty taxonomy fields) return None and are skipped (treated as failed crops, not a
different animal). Real mammals outside the 225 set keep full taxonomy → match_level
`no_match` → still flag. Flagged senior-review to confirm this matches plan §10.

---

## P2-S1 — Augmentation: constants + basic set + wiring + smoke  ⏳ dispatched

**Instruction (to subagent, sonnet):** Implement Plan 2 §6/§7 basic set (setups A&B),
shared single code path.
- Add full Section-6 constant block to `constants.py` (replace old 3 `AUG_*`):
  basic block + compositing block (defaults = basic on, compositing off). Keep
  `as_dict()` working.
- In `transforms.py` replace NotImplementedError stubs with real basic-set ops
  reusing `yolov5.utils.augmentations.augment_hsv` + `random_perspective`
  (degrees=shear=perspective=0 → scale+translate only) + trivial hflip.
- Wire into `CocoYoloDataset.__getitem__` via `augment: bool`; targets stay ∈[0,1];
  drop degenerate boxes. Seed discipline: `worker_init_fn`+`generator` from SEED.
- Add `assert_distillation_safe()` (raises if any compositing flag truthy). Train
  dataset `augment=True`; val/test `augment=False` in run_training_pipeline.py.
- Smoke test on host (no Docker/MLflow): assert target coords ∈[0,1], boxes survive,
  HSV changes pixels. Run and paste output. Do NOT touch mosaic/mixup (that's P2-S2).

**Result:** ✅ DONE. Changed `constants.py` (full 15-const Section-6 block; `as_dict()`
unchanged, picks up all `AUG_*`), `transforms.py` (`augment_basic()` reusing
`augment_hsv` + `random_perspective` with degrees/shear/perspective=0; hflip;
`assert_distillation_safe()`), `dataset.py` (`augment` param on `CocoYoloDataset`;
per-sample deterministic `rng=np.random.default_rng(hash((SEED,idx)))`;
`make_worker_init_fn`; `Dataloader` accepts `worker_init_fn`/`generator`),
`run_training_pipeline.py` (train `augment=True`; val/test `augment=False`; train
loader gets `worker_init_fn`+seeded `generator`). New
`smoke_test_augmentation.py` — **all 6 checks PASS** (coords∈[0,1]; no NaN; tensor
shape/dtype; aug≠no-aug pixels; all-flags-off ≡ no-aug; multi-box boxes survive).
Verified: `random_perspective` wants abs-pixel `[cls,x1,y1,x2,y2]`; `AUG_SCALE=0.5`→
scale∈[0.5,1.5]; it uses py-`random` (seed via worker_init_fn). Compositing untouched.

---

## P1-S2 — Run flagging over all sources  ✅ DONE (orchestrator)

**Instruction:** `python scripts/dataset_quality/14-flag_multi_animal_contamination.py
--source all` (tolerance=family, md-conf=0.5, sn-score=0.3). Log →
`output/contamination_flag_all.log`. Confirm flagged ≪ 32,401.

**Result:** Ran clean (exit 0) over all 5 sources.
- Total classified 465,130; **64,835** images with ≥2 significant boxes;
  **12,241 flagged** (≥1 confident cross-band mismatch); 3,874 uncertain-only;
  26,435 consistent multi-box; 1,317 expected-class-not-in-225 (skipped).
- Flagged / naive-bound = **12,241 / 32,401 = 37.8%** — below the 32k bound the
  plan §9 warns against; band is doing real work but is on the looser side.
- By source: inaturalist 8,786, gbif 1,646, wikimedia 1,138, openimages 568,
  images_cv 103. Offending match-levels: order 7,691 / class 8,442 / no_match 5,580.
- Reports: csv 21,713 rows, review.json 12,241 images, md (top-30 + full per-class
  projected delta). Largest projected losses: eared seals 1,123, bighorn sheep 535,
  llama genus 511, domestic horse 454, african elephant 306.
- **For the user / senior review:** 12,241 is a manageable review list but the
  big-count classes (eared seals, bighorn sheep) merit a FiftyOne spot-check (P1-S3)
  to confirm precision before any apply step; per plan §10, tightening to
  `--tolerance genus` + `LOOKALIKE_GROUPS` is the lever if family proves too loose.
  No data was modified.

---

## P2-S2 — Setup C compositing (mosaic/mixup/close-mosaic)  ⏳ dispatched

**Instruction (to subagent, sonnet):** Implement Plan 2 §4.2/§6 Setup-C aggressive
arm on top of the basic set, gated on compositing flags (default off; A/B unaffected).
Mosaic(4-img)+MixUp+close-mosaic tail; copy-paste documented inert (box-only GT).
Reuse yolov5 primitives. `assert_distillation_safe()` still passes at defaults.
Smoke-test setup C (flags forced on) on host: targets∈[0,1], boxes survive.

**Result:** ✅ DONE (one brittle test assertion flagged for review). Changed
`transforms.py` (`build_mosaic` — yolov5 `load_mosaic` recipe: 2S canvas, paste 4
imgs, `random_perspective(border=(-S//2,-S//2))` crop; `mixup` Beta(32,32)),
`dataset.py` (`set_epoch`, `_compositing_active()` close-mosaic gate, `_load_raw`,
mosaic branch in `__getitem__` gated on `AUG_MOSAIC>0 and augment and
_compositing_active()`; post-mosaic basic pass uses scale=0/translate=0 to avoid
double random_perspective; mixup nested in mosaic branch; copy-paste inert + one-shot
`logger.warning`), `training_pipeline.py` (calls `ds_train.set_epoch(epoch,epochs)`
guarded by getattr/hasattr), `smoke_test_augmentation.py` (`--setup-c`).
**Defaults leave A/B byte-for-byte identical** (verified: basic smoke still PASS;
single-image `else` branch unchanged). `assert_distillation_safe()` passes at
defaults, raises when mosaic on.
**Issue for review:** the `--setup-c` smoke has a flaky assertion "tail mean boxes ≤
mosaic mean" (3.3 vs 3.1) — compares two *different* noisy random samples; the real
close-mosaic logic (`_compositing_active()=False` at tail) passes. Needs a
deterministic comparison instead. NOT a bug in the augmentation itself.

---

## P1-S3 + P1-S4 — Visualizer flagged mode + apply script  ✅ DONE

**Instruction (to subagent, sonnet):** (S3) Add `--flagged-review` mode to
`visualize_fiftyone.py` loading only flagged images (subset COCO) + `sn_prediction`
detections per box (label `pred_common [match_level/verdict]`). (S4) Write
`15-apply_contamination_decisions.py` per plan §7: decisions JSON (discard/edit/keep),
`--from-review` default derivation (edit if some boxes offend, discard if all),
box→annotation match by IoU>0.5 + closest conf, atomic per-split rewrite, per-class
band assertion (abort w/o `--force`), `--dry-run`.

**Result:** ✅ DONE.
- `visualize_fiftyone.py`: `--flagged-review`, `--review-json`, `--split-annotations`,
  `--dry-run`. `subset_coco_to_flagged()` writes a minimal COCO (only flagged imgs) to
  `scripts/evaluation/outputs/_flagged_review_<split>.json` (avoids loading 145k imgs).
  `build_sn_detections_for_sample()` converts `bbox_norm`[cx,cy,w,h]→FiftyOne
  [x_tl,y_tl,w,h]. Dry-run (50 imgs) passes, no app launched. Default mode unchanged.
- `15-apply_contamination_decisions.py`: `derive_decisions_from_review()` (prefer-edit),
  `_match_box_to_annotation()` (IoU>0.5 + closest conf; verified IoU=1.000 on test),
  `check_band_violations()` (tier lbs 0/100/500/1500), atomic `.tmp`+`os.replace`,
  aborts w/o `--force` on violation.
  **Dry-run (`--from-review --dry-run`):** 12,241 decisions (1 discard, 12,240 edit);
  would remove 11,219/284,154 anns (3.9%) — train 6,059, val 988, test 4,172.
  Combined band violations: `brown-throated sloth` 1,501→1,499 (Tier4 lb1500),
  `hippopotamus` 504→487 (Tier3 lb500) → **aborts without --force** (correct safety).
  Idempotent across 2 runs. **No data files modified.**

---

## REVIEW — Critical senior-developer audit + direct fixes  ⏳ dispatched

**Instruction (to subagent, sonnet, acting as critical senior dev):** Audit ALL
changes for correctness; FIX issues directly; keep data files untouched; re-run
smoke tests + dry-runs to prove green. Focus areas: (1) fix the flaky `--setup-c`
smoke assertion deterministically; (2) verify mosaic bbox math + post-mosaic
double-aug avoidance; (3) script 14 non-taxonomic skip vs plan §10; (4) script 15
IoU box→annotation matching + band logic; (5) RNG/seed reproducibility; (6) confirm
A/B defaults byte-identical.

**Result:** ✅ DONE — 3 real bugs found & fixed, all verification green.

1. **[HIGH] Flaky `--setup-c` smoke assertion** (`smoke_test_augmentation.py`):
   replaced the noisy "tail mean boxes ≤ mosaic mean" comparison with a deterministic
   monkey-patch counter asserting `build_mosaic` is called **0** times at the
   close-mosaic tail epoch. Now conclusive.
2. **[MEDIUM] Script 14 non-mammal guard** (`14-...py` ~L225): old guard
   `not genus and not class_` let **birds** (class_=aves, e.g. mallard/generic bird)
   and other non-mammals through → false `no_match` flags. Fixed to
   `if pred_tax["class_"] != "mammalia": return None`. Cross-species contamination is
   mammal-vs-mammal only. images_cv flags 103→95, no_match boxes 13→0. A genuine
   non-225 *mammal* still keeps full taxonomy → `no_match` → flagged (plan §10 ✓).
3. **[MEDIUM] PYTHONHASHSEED-dependent mixup rng** (`dataset.py` ~L202): `hash((SEED,
   idx,"mixup"))` is per-process salted (string in tuple) → broke §5.2 byte-identical
   requirement for Setup C. Fixed to arithmetic `(SEED*1_000_003 + idx*2 + 1) % 2**32`.
   Primary rng `hash((SEED,idx))` is int-only (CPython `hash(int)==int`, stable) — left.

**Confirmed correct (no change):** mosaic coord math (norm by S not 2S; border
crop; no double random_perspective; post-mosaic hflip), script 14 evaluates only
secondary boxes & `_compute_match_level` arg order, script 15 IoU bbox→ann match +
discard/edit + atomic write + band lower-bounds (0/100/500/1500), A/B defaults =
single-image path only.

### Final verification (orchestrator, post-fix)
- **Full report regenerated** (`--source all`): **11,354 flagged** (was 12,241 before
  bug-2 fix — bird/non-mammal FPs removed); 3,517 uncertain-only; 64,835 multi-box;
  35% of the 32,401 naive bound. Reports rewritten.
- `smoke_test_augmentation.py` → PASS; `--setup-c` → PASS; basic re-run → PASS.
- `15-... --from-review --dry-run` → 11,354 decisions (1 discard / 11,353 edit);
  would remove 10,468 anns; **aborts on band violations without `--force`** (correct).
  No files written.
- `git status --short data/real/` → empty (no data mutated).

---

## Final Status & Handoff to User

**Both plans implemented and verified on the host — no Docker was needed.**

### Plan 1 (contamination flagging) — what exists now
- `data/speciesnet_labels.json` (2498 labels) + `scripts/dataset_quality/0-dump_speciesnet_labels.py`.
- `scripts/dataset_quality/14-flag_multi_animal_contamination.py` — run; reports in
  `reports/multi_animal_contamination.{csv,md,_review.json}` (11,354 flagged images).
- `scripts/evaluation/visualize_fiftyone.py --flagged-review` — review UI mode.
- `scripts/dataset_quality/15-apply_contamination_decisions.py` — apply step (safe
  by default; needs `--from-review` or a human decisions file, and `--force` to cross
  band limits).

### Plan 2 (augmentation) — what exists now
- Basic set (setups A & B) live in `constants.py`/`transforms.py`/`dataset.py`, wired
  into the direct training entry point with seed discipline; `assert_distillation_safe()`
  ready for a future distillation entry point (none exists yet).
- Setup C compositing (mosaic/mixup/close-mosaic) implemented, gated off by default.
- `smoke_test_augmentation.py` (+`--setup-c`) both PASS.

### Manual steps remaining for the USER (no automation possible here)
1. **Human review of flagged images (Plan 1 §6):** launch the FiftyOne review UI on a
   machine with a display/browser:
   `python scripts/evaluation/visualize_fiftyone.py --flagged-review --split-annotations data/real/annotations_train.json`
   Tag images `discard` / `edit`, export decisions to
   `reports/multi_animal_contamination_decisions.json`. (Big-count classes — eared
   seals, bighorn sheep, llama genus — are worth checking first; if family tolerance
   looks too loose, re-run script 14 with `--tolerance genus`.)
2. **Apply decisions (Plan 1 §7):** `python scripts/dataset_quality/15-apply_contamination_decisions.py`
   (review the dry-run diff first; `--force` only if you accept the band deltas for
   `brown-throated sloth`/`hippopotamus` etc.).
3. **Real training runs for setups A/B/C (Plan 2 §7):** the augmentation is wired but
   actual MLflow training runs were not launched here (need GPU + MLflow server). Run
   `make yolov5s-train` (Setup B) and create the distillation entry point for Setup A.
4. **`pip` note:** installing `speciesnet` downgraded `yolov5` 7.0.13→7.0.11; the
   orchestrator restored `7.0.13`. If you re-run step 0, re-pin afterward:
   `pip install 'yolov5==7.0.13'`.
