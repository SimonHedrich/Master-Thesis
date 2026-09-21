# Results Chapter Reference Index

**Purpose.** `thesis/manuscript/chapters/4-Results.tex` is partially drafted — §4.1 (synthetic
generator comparison) and §4.3.2 (embedded runtime benchmark) are finished prose; §4.2 (all 5
subsections), §4.3.1 (KD vs. direct fine-tuning), and §4.4 (cross-cutting observations) are
comment-only `% TODO` stubs. Chapter 1 (Introduction) and Chapter 5 (Discussion/Conclusion) are
empty stubs, and the appendix is missing its class/band table and look-alike group listing.
This doc indexes where the evidence for each of those gaps already lives, so drafting doesn't
require re-deriving numbers or re-finding files. It sits alongside, not in place of:

- `TODO.md` — task tracker (what's done/open, machine-tagged)
- `docs/2026-09-07_model-comparison-teacher-and-students.md` — the synthesized four-model
  verdict (KD vs. direct-FT), the single most load-bearing doc for §4.2/§4.3.1

No numbers are restated here beyond what's needed to orient — always read the cited source
before writing prose from it, since this index will drift out of sync over time.

## 1. Thesis stub → primary source

| Thesis section | What's needed | Primary source(s) |
|---|---|---|
| §4.2 lead-in (line 191) | Framing paragraph for the regime-comparison section | `docs/plans/2026-06-10_model-evaluation-strategy.md` (mixed/real-only rationale, bands) |
| §4.2.1 Headline mixed/real mAP (line 197) | Tier-1 table, mixed + real-only breakout | `docs/2026-09-07_model-comparison-teacher-and-students.md` §1 TL;DR table + per-model `evaluation_report.md` |
| §4.2.2 Granularity gap decomposition (line 202) | detect / coarse / fine mAP gap table + TIDE error-type chart | `docs/2026-09-07...` §2 finding 4 (classification is the bottleneck) + per-model `evaluation_report.md` (detect-only mAP figures) |
| §4.2.3 Band × Granularity grid (line 209) | 4-band × {fine, coarse} × {mAP, mAP50} grid — core comparison | each model's `eval_band_grid.csv` (9 rows) + `docs/plans/2026-06-10_model-evaluation-strategy.md` §4/§6a/§6b/§7/§9 for band rationale |
| §4.2.4 Domain-shift delta (line 216) | Paired Δ_domain per class, per band (mixed vs. real vs. synthetic) | per-model `eval_per_class.csv` (226 rows) + `eval_band_grid.csv`; watchdog criterion defined in CLAUDE.md "Important Constraints" |
| §4.2.5 Look-alike within-group confusion (line 223) | Confusion table + block-diagonal matrix | each model's `eval_confusion_pairs.csv` (~135 rows) + `docs/plans/2026-06-11_lookalike-groups-review.md` (group definitions/rationale) |
| §4.3 lead-in (line 231) | Framing paragraph for the KD comparison section | `docs/2026-09-07...` §2 finding 2 |
| §4.3.1 Distillation vs. direct fine-tuning (line 240) | Setup A vs. B headline mAP comparison | `docs/2026-09-07...` §2 finding 2 (full metric/band table) + §3.3 (KD checkpoint provenance, crash note) |
| §4.3.2 Runtime benchmark | **Already drafted** — verify against source if numbers changed | `docs/2026-09-18_embedded-benchmark-results.md`, `reports/embedded_benchmark/` |
| *(new)* resolution tradeoff subsection (`sec:results_resolution_tradeoff`, referenced in plan doc, not yet in `4-Results.tex`) | Arm 1 (inference-resolution sweep) done; Arm 2 (resolution-native training) **in progress** — do not draft until Arm 2 finishes | `docs/plans/2026-09-20_input-resolution-optimization-study.md`, `reports/resolution_study/resolution_tradeoff.md` |
| §4.4 Cross-cutting observations (line 410) | Synthesis across all above | `docs/2026-09-07...` §2 findings 1 & 4, `docs/2026-09-18_embedded-benchmark-results.md`, resolution study doc, §6 below (open gaps as explicit scope statements) |
| Ch.1 Introduction — all 5 sections (empty stub) | Motivation, numbered RQs, environment (inovex), AI-tools disclosure, structure overview | CLAUDE.md "Core Research Question"; `docs/2026-03-09_thesis-overview.md`; `docs/2026-03-09_hardware-proxy-selection.md` (environment/target hardware) |
| Ch.5 Discussion/Conclusion — all 4 sections (empty stub) | Restate Ch.1 RQs, answer using Ch.4 evidence, limitations, further work | Blocked on Ch.1 (RQs) + Ch.4 being drafted first; limitations draw from §6 below |
| Appendix — full 225-class × band table (TODO at top of `appendix.tex`) | Class → band assignment table | `resources/GBIF_image_counts.csv`; band definitions in `docs/2026-09-07...` §4 |
| Appendix — full look-alike group listing (same TODO) | Group → member-species listing | `docs/plans/2026-06-11_lookalike-groups-review.md` |

## 2. Per-model final evaluation artifacts

All eval reports follow the same schema: `evaluation_report.{md,json}`, `eval_per_class.csv`
(226 rows incl. header), `eval_band_grid.csv` (9 rows), `eval_confusion_pairs.csv` (~135 rows).

| Model | Path | Headline mixed/real mAP (mAP50) | Status |
|---|---|---|---|
| Teacher, zero-shot pretrained | `scripts/training/megadet_speciesnet_ensemble/model_exports/pretrained/eval/` | 0.487 / 0.445 | Final — zero-shot baseline |
| Teacher, fine-tuned (ensemble) | `scripts/training/megadet_speciesnet_ensemble/model_exports/finetuned-teacher-finetune-ff0.75-bs64-20260909-231034/eval/` | 0.663 / 0.599 | **Final — best overall number, post-Band-A-fix** |
| Teacher checkpoint only (no eval here) | `scripts/training/teacher_finetune/model_exports/teacher-finetune-ff0.75-bs64-20260909-231034/` | (see ensemble eval above) | Final checkpoint; 13 sibling dirs are superseded probes/smoke tests |
| YOLOv5s | `scripts/training/yolov5s/model_exports/yolov5s-20260909-230900/eval_best/` | 0.496 / 0.407 | **Final, post-Band-A-fix.** 4 older sibling dirs superseded |
| YOLO26n direct-FT | `scripts/training/yolo26n/model_exports/yolo26n-bs32-20260910-212812/evaluation/` | 0.599 / 0.529 | **Final, post-Band-A-fix — thesis headline direct-FT number.** Also holds `resolution_sweep/{res320,res416,res512,res640}/`, i.e. resolution-study Arm 1's raw eval reports |
| YOLO26n KD | `scripts/training/yolo26n/model_exports/yolo26n-kd-bs16-20260916-101612/evaluation/` | 0.560 / 0.479 | **Final, post-Band-A-fix.** Checkpoint from epoch 185 (crashed run at epoch 187, Tailscale/MLflow outage) — provenance documented in `docs/2026-09-07...` §3.3. `-162339` sibling dir is an earlier interrupted attempt, not used |
| YOLO26n student, zero-shot | `scripts/training/yolo26n/model_exports/zero-shot-pretrained/` | — | **Not run** — empty `eval/` dir. Open gap, see §6 |
| YOLO26n res320 native training (Arm 2) | `scripts/training/yolo26n/model_exports/yolo26n-res320-bs128-20260920-165602/` | — | **In progress** as of 2026-09-21 (last logged ~epoch 82–186/200, ETA ~2026-09-23). `res320-bs32-...`, `res320-bs128-...164041`, and `smoke-res320-...` sibling dirs are abandoned/smoke attempts — not data |

## 3. Synthesis docs (already-written narrative — cite, don't re-derive)

- `docs/2026-09-07_model-comparison-teacher-and-students.md` — four-model verdict, KD vs.
  direct-FT finding, band definitions (§4), open gaps (§5)
- `docs/2026-09-18_embedded-benchmark-results.md` + `reports/embedded_benchmark/` — latency,
  thermal, parity on the Pi 400 proxy; also `torchscript_sigill_finding.md` and
  `ncnn_conversion_finding.md` for negative results worth a footnote
- `docs/plans/2026-09-20_input-resolution-optimization-study.md` + `reports/resolution_study/`
  — **in progress**. Arm 1 (inference-resolution sweep on the existing 640px checkpoint) done;
  Arm 2 (resolution-native training) still running; thesis write-up phase explicitly marked
  not started in the plan doc itself. Re-check status before drafting the new subsection

## 4. Methodology/provenance plans

For justifying evaluation design choices in Results prose and cross-referencing the Methods
chapter:

- `docs/plans/2026-06-10_model-evaluation-strategy.md` — mixed vs. real-only rationale, band
  definitions, domain-shift watchdog
- `docs/plans/2026-06-11_lookalike-groups-review.md` — look-alike group construction
- `docs/plans/2026-06-11_evaluation-script-implementation.md` — eval suite design
- `docs/plans/2026-06-30_knowledge-distillation-and-teacher-finetuning-strategy.md` — KD setup rationale
- `docs/plans/2026-07-22_eval-suite-scoring-performance-investigation.md` — scoring-perf fix (relevant if eval timing is discussed)
- `docs/plans/2026-09-17_on-device-benchmarking-plan.md` — embedded benchmark design

## 5. Not model-eval data (dataset construction, out of scope for Results)

`reports/wikimedia_categories/`, `reports/wikimedia_categories_filtered/`,
`reports/wikimedia_file_manifests/` — image-sourcing scaffolding for the Methods chapter's
data-sourcing section, not Results evidence.

## 6. Known open gaps to state explicitly (§4.4 / Discussion / Limitations)

From `TODO.md` §3.4–3.5, §4.2, §5.1, §6.1 and `docs/2026-09-07...` §5:

- **Student zero-shot baseline** never successfully run (crashed on the blanks gap, not retried)
- **KD hyperparameter grid** — only the default (T, α) point was run, no sweep
- **Multi-animal subset cut** not evaluated separately
- **QAT** — explicitly deferred as a scope decision (not a gap), flagged as "the single
  largest lever" left unexplored for the embedded target
- **Blind human rating rubric** for the synthetic generator comparison (TODO §3.4) not run
- **Resolution study Arm 2** incomplete as of this writing — do not present resolution-native
  training numbers until that run finishes and is evaluated

---
*Compiled 2026-09-21. Source data changes as new runs land — treat directory listings and
headline numbers here as a snapshot, and re-verify against the linked files before quoting
in thesis prose.*
