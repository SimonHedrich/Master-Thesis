# Synthetic-Generation Model Comparison

**Created:** 2026-07-14
**Owner:** Simon Hedrich

This subdirectory collects the planning and reference material for a **controlled
comparison of image-generation models** used to produce synthetic wildlife
training data.

## Why this exists (the short version)

The production synthetic dataset (~12,600 images, 76 rare classes) was generated
with a single model — `gemini-3.1-flash-image-preview` (Gemini "Nano Banana 2")
— chosen after informally judging that its images *looked* more realistic than
the local diffusion models'. The supervising professor flagged this as
**methodologically insufficient**: a thesis-grade model choice must be
**qualitatively evaluated** against alternatives under a stated, repeatable
criterion, even though the thesis's core question is *"can current image models
be used to generate synthetic training data?"* rather than *"which is best?"*.

The budget is largely spent, so we **cannot** regenerate the whole dataset per
model. Instead we run a **rigorous comparison on a deliberately chosen subset of
classes** — spanning rare/unusual species, species with robust real test sets,
and a fine-grained look-alike group (the three zebra species) — across a few API
and local generators, evaluated on qualitative, automatic, and downstream axes.

## Documents in this directory

| File | What it covers |
|------|----------------|
| [`00_motivation-and-research-question.md`](00_motivation-and-research-question.md) | The professor's critique, the scoped research question, budget constraint, scope of the comparison |
| [`01_experiment-design.md`](01_experiment-design.md) | The controlled protocol: what varies (generator, prompt regime) vs what's held fixed; tractable model grid; cost |
| [`02_class-selection.md`](02_class-selection.md) | The 12 proposed classes and rationale (rare vs robust-test vs zebra look-alike), with test-count/GBIF data and the central tension |
| [`03_api-models-landscape-and-pricing.md`](03_api-models-landscape-and-pricing.md) | OpenAI (multiple models!), Google (Nano Banana + Imagen), FLUX/Stability/Ideogram/Recraft/Firefly/Midjourney/fal/Replicate; pricing, dimensions, quality, batch, licensing |
| [`04_local-models-and-output-parameters.md`](04_local-models-and-output-parameters.md) | Local diffusion models, **prompt-length limits**, native output resolutions (see update note re: the superseded 500×500 downscale idea), throughput/cost, licensing |
| [`05_prompt-strategy-and-length-limits.md`](05_prompt-strategy-and-length-limits.md) | How to build fair prompts across models with wildly different limits; the ≤75-token compressed prompt; worked examples |
| [`06_evaluation-methodology.md`](06_evaluation-methodology.md) | The three evaluation axes (blind multi-rater rubric; automatic proxies; downstream real-test mAP); statistics; the final results table |
| [`07_open-questions-and-what-to-reconsider.md`](07_open-questions-and-what-to-reconsider.md) | Confounds, gaps, framing risks, and decisions to make before generating |
| [`08_classes-and-models-overview.md`](08_classes-and-models-overview.md) | Condensed summary: the 12 output classes, and the model grid with cost/image and full-vs-compressed prompt length |
| [`09_test-subset-build.md`](09_test-subset-build.md) | Build log for the materialized 12-class real test subset (`data/synthetic_model_comparison/`): the §4a expansion rule, live vs. doc counts, output layout, and the build script |
| [`10_train-subset-incumbent-selection.md`](10_train-subset-incumbent-selection.md) | Selecting the incumbent generator's synthetic train subset (7 of 12 classes) by reusing already-generated production images: the stratify-by-environment, greedy-diversify selection algorithm, live coverage results, output layout |
| [`11_detector-architecture-selection.md`](11_detector-architecture-selection.md) | Deciding the fixed Axis-C detector architecture: YOLO26n vs. YOLOv5s, why NanoDet/PicoDet are out of scope, the KD-strategy-doc precedent resolving simple-vs-heavy, capacity/floor-effect reasoning, a log-derived training-time estimate for this experiment's much smaller per-cell dataset, and the internal-val-split recommendation |
| [`12_additional-generator-cells-build-log.md`](12_additional-generator-cells-build-log.md) | Build log for the Nano Banana 2 Lite and gpt-image-2 (low/medium) `full`-regime cells (generic reuse scripts, resolution/token-limit/char-limit fixes, actual-vs-estimated cost), the shared `compressed` prompt regime, and the local-model tier's pipeline + smoke tests |
| [`13_local-model-roster-overhaul-and-maxlen-regime.md`](13_local-model-roster-overhaul-and-maxlen-regime.md) | Build log for the GPU-memory-offload fix (~1.8x speedup), the roster overhaul (3→7 models, with FLUX.2-dev researched and rejected, and HiDream-I1 initially rejected then unblocked and added — §10), full benchmarking of all seven models, the new `maxlen` prompt regime (256/512-token tiers, and why naively truncating the `full` prompt didn't work), and the first full 1,200-image production cell (`sd35-large-turbo`) |
| [`14_maxlen-cell-deep-comparison.md`](14_maxlen-cell-deep-comparison.md) | Deep comparison across all six `maxlen` cells (incl. `hidream-i1`): aggregated headline/per-class/confusion results, charts, cost-vs-quality analysis, and the reasoning behind the ranking |
| [`15_all-cells-comparison-with-api-incumbent.md`](15_all-cells-comparison-with-api-incumbent.md) | Extends `14` with the trained API incumbent (`gemini-3.1-flash-image-preview`, `full` regime): headline ranking, the `full`-vs-`maxlen` prompt-length confound, and why its mAP lead is concentrated in the zebra classes specifically |
| [`16_prompt-length-ablation.md`](16_prompt-length-ablation.md) | The controlled ablation doc `15` called for: `full` vs. `compressed` prompts, same model held fixed, across `gpt-image-2-low` and `gemini-3.1-flash-lite-image` — resolves the confound (prompt length cuts mAP ~40-50%, model-independent) and surfaces `gpt-image-2-low/full`'s downstream mAP as the best in the experiment so far |
| [`17_full-experiment-comparison.md`](17_full-experiment-comparison.md) | Consolidates docs `14`-`16` into one ranking of all 12 trained cells, plus a new finding only visible at full scale: the production incumbent is beaten by its own cheaper tier (`gemini-3.1-flash-lite-image`) under the identical prompt regime |
| [`scraped_sources/`](scraped_sources/) | Verbatim scrapes of the OpenAI and Gemini pricing pages (primary sources) |

## Key facts at a glance

- **Incumbent model:** `gemini-3.1-flash-image-preview` (Gemini 3.1 Flash Image /
  "Nano Banana 2"), aspect `4:3`, size `0.5K` (~512 px), ~€0.04/image.
- **"Other OpenAI models?" → Yes:** `gpt-image-2` (flagship), `gpt-image-1.5`,
  `gpt-image-1-mini` (last two EOL 2026-12-01). DALL·E removed from the API.
- **Prompt-length wall:** SDXL = **77 tokens (~55 words)**; FLUX.1-schnell / SD3.5
  = **256 T5 tokens (~180 words)**; FLUX.1-dev = **512 (~350 words)**; API models
  = effectively unlimited (Gemini 32k-token context; OpenAI 32k chars). The
  incumbent prompts were ~1,300 words — hence a mandatory *compressed* prompt
  regime for fair comparison.
- **Output size:** generate at each model's native ~1 MP 4:3 bucket and save
  as-is — no forced downscale. (An earlier version of this doc recommended
  downscaling to a fixed 500×500; that number matched no other convention in
  the repo — every other generator cell stores images at whatever resolution
  the model natively produces, and this experiment's training pipeline
  letterboxes every input to 640×640 regardless of source size. See
  [`04`](04_local-models-and-output-parameters.md) §3.)
- **Class subset (proposed 12):** 3 zebras (fine-grained); red fox / American
  black bear / lion (robust-test anchors); kinkajou / water deer / ringtail
  (rare **and** >100 test); saiga / aye-aye / pangolin (iconic rare, test-limited).
- **Local-model roster (current, 7 models):** `flux2-klein-9b`, `realvisxl-lightning`,
  `sd35m`, `sd35-large`, `sd35-large-turbo`, `qwen-image`, `hidream-i1` — see
  [`04`](04_local-models-and-output-parameters.md) §7-9 for the roster,
  measured benchmark numbers (~126 GPU-hours total for all seven 1,200-image
  cells), and the GPU-memory-offload finding (removing an unneeded
  `enable_model_cpu_offload()` call gave SD 3.5 Medium a ~1.8x speedup).
  FLUX.2-dev was evaluated and not included (too large). HiDream-I1 was
  initially blocked on a gated Llama dependency but that access was later
  granted, so it was added and benchmarked as the 7th model (doc
  [`13`](13_local-model-roster-overhaul-and-maxlen-regime.md) §10).

## Related existing docs

- `docs/2026-04-02_synthetic-image-generation-model-research.md` — prior local-model survey
- `docs/plans/2026-05-12_synthetic-image-generation-strategy.md` — production generation strategy & prompt template
- `docs/plans/2026-06-10_model-evaluation-strategy.md` — the thesis evaluation strategy (real-only anchor, domain-shift watchdog)
- `docs/plans/2026-06-11_lookalike-groups-review.md` — frozen look-alike groups (zebra, panthera_rosette, gazelle, …)
- `reports/class_split_counts.csv`, `reports/lookalike_groups_v2.csv`, `data/gbif/metadata/GBIF_image_counts_v1.csv` — the data behind class selection

## Status / next decisions

Open decisions are collected in [`07`](07_open-questions-and-what-to-reconsider.md)
§5 — euro cap, final class count, rater setup, detector-vs-classifier, and the
name-only prior-knowledge probe. Generation is well underway — see below.

The 12-class **real** test subset (§4a of `02`) has been materialized —
see [`09`](09_test-subset-build.md) — at
`data/synthetic_model_comparison/test/` (9,742 images, ~4.6 GB) with a
matching COCO json.

The **incumbent generator's** synthetic train subset is complete for all 12
classes (1,200/1,200 images) — see [`10`](10_train-subset-incumbent-selection.md).

**Three more `full`-regime cells are complete** — Nano Banana 2 Lite
(`gemini-3.1-flash-lite-image`) and both `gpt-image-2` quality tiers (`low`,
`medium`), 1,200/1,200 images each. **The shared `compressed` prompt regime
(all 12 classes) is built.** See
[`12`](12_additional-generator-cells-build-log.md) for the generic
reuse-script pattern, the resolution/token-limit/char-limit issues found
and fixed along the way, and actual-vs-estimated API costs.

**The local-model tier is now benchmarked on all seven models** (roster
above) — every model works end-to-end, per-image timing measured, and one
real bug found only by actually running each new model (FLUX.2-klein-9B
needed `Flux2KleinPipeline`, not `Flux2Pipeline`; Qwen-Image needed
custom on-the-fly NF4 quantization since no pre-quantized checkpoint
exists). `hidream-i1` was added last, after its gated Llama dependency was
unblocked, and is the slowest model in the roster at 129.85s/image even
after NF4-quantizing its two largest components. See
[`04`](04_local-models-and-output-parameters.md) §8 for the full results
table (~126 GPU-hours to generate all seven full 1,200-image cells) and
the GPU-offload finding.

**A second prompt regime, `maxlen`, now exists alongside `compressed`** —
built by `1h-generate_prompts_maxlen.py` / `1i-generate_images_local_maxlen.py`,
sized to each model's *real* text-encoder capacity (256/512 tokens instead
of the fairness-motivated 75) for the actual production dataset, keeping
the `compressed` cells untouched for a possible later ablation against the
best proprietary model. See [`05`](05_prompt-strategy-and-length-limits.md)
§7 for the design (and why the first, naive "truncate the full prompt"
idea didn't work).

**`sd35-large-turbo`'s full 1,200-image `maxlen` cell is complete** —
1,200/1,200 images, 0 failures, 8h16min total (24.8s/image average). It was
chosen as the fastest of the four newly-added models, to validate the new
script/prompt design at production scale before committing the much larger
GPU-hour investment the other models would need. See
[`13`](13_local-model-roster-overhaul-and-maxlen-regime.md) §6 for the full
results. `realvisxl-lightning`, `sd35m`, `flux2-klein-9b`, `sd35-large`,
and `hidream-i1` have since completed the same way — `hidream-i1` last,
at 143.92s/image (47.97h total), the slowest cell in the roster.
**This completes generation for the entire non-dropped roster.**

**`qwen-image`'s full `maxlen` cell was dropped, not deferred** — its
`compressed`-regime 5-image smoke test showed clearly visible
quantization-artifact graininess (stippled noise in flat regions — sky,
grass), the same issue first flagged in doc [`04`](04_local-models-and-output-parameters.md)
§8. Since `qwen-image` was never a headline model (kept mainly for its
Apache-2.0 license) and the artifact is structural to the on-the-fly NF4
quantization this box's VRAM requires, the ~34h production run was skipped
rather than generated and discarded later. See
[`13`](13_local-model-roster-overhaul-and-maxlen-regime.md) §9.

Still open: the fifth API cell (Nano Banana Pro) and the `compressed`-regime
ablation pair (incumbent + gpt-image-2 low). Local-model `maxlen`
generation is otherwise done — see above.

The **fixed Axis-C detector architecture is decided: YOLO26n**, not YOLOv5s
— see [`11`](11_detector-architecture-selection.md) for the full rationale
(matches the thesis's actual embedded-deployment target per the KD strategy
doc; training time is not a differentiator at this experiment's much smaller
per-cell data scale).

**Labeling and training code now exists** (per generator × prompt-regime
cell, not gated on every cell existing first):
`scripts/synthetic_model_comparison/2-run_megadetector.py` through
`5-export_coco.py` adapt the production MegaDetector → triage-review →
bbox-labeling → COCO-export chain for this experiment's per-cell layout, and
`scripts/synthetic_model_comparison/training/` is a self-contained YOLO26n
pipeline (copied and adapted from `scripts/training/yolo26n/`; see its own
README) that trains on one cell's labeled images and evaluates on the fixed
real test set.

**Stage 2 (MegaDetector) has now run on all six completed `maxlen`
cells** — `realvisxl-lightning`, `sd35m`, `flux2-klein-9b`, `sd35-large`,
`sd35-large-turbo`, `hidream-i1`, all 1,200/1,200 images, 0 missing.
`2-run_megadetector.py`'s `--prompt-regime` choices didn't yet include
`maxlen` (added when the regime was introduced in doc `13` after this
script was written) — fixed as part of the first five cells' run;
`hidream-i1`'s run needed no further script changes. Per-cell
`n_significant` distribution (share of images with 0 / 1 / ≥2 detections
≥0.5 conf):

| Cell | 0 | 1 | ≥2 |
|---|---|---|---|
| `realvisxl-lightning` | 0.0% | 97.1% | 2.9% |
| `sd35m` | 0.3% | 98.2% | 1.5% |
| `flux2-klein-9b` | 0.2% | 93.6% | 6.2% |
| `sd35-large` | 0.1% | 98.8% | 1.2% |
| `sd35-large-turbo` | 0.1% | 97.1% | 2.8% |
| `hidream-i1` | 0.2% | 97.3% | 2.5% |

Stages 3–5 (triage review, bbox labeling, COCO export) are still pending —
those are human-in-the-loop review steps, not yet run on any cell. A first
training pass is the natural next step once a cell clears them.

**Update 2026-08-04 — best-effort export + first full §3.3 training pass
on all five `maxlen` cells, results provisional, not thesis-final.**
Stages 3/4 (human triage/bbox review) still have not run on any cell — the
numbers below use `5-export_coco.py`'s documented best-effort fallback
(MegaDetector's own boxes, no human review), which the training package's
own README already flags as *"useful for pipeline verification, but full
review is still required before a cell's numbers are thesis-final."*
**These cells will need re-export and re-training once §3.4 lands.** All
five cells exported cleanly at 1,200/1,200 images, 0 skipped (every image
had at least one MD detection above 0.5 conf, so nothing fell through to
the zero-detection SKIP path). `--prompt-regime maxlen` also needed adding
to `3-single_detect_review.py`, `4-bbox_labeling_server.py`,
`5-export_coco.py`, and `training/run_training_pipeline.py`'s argparse
choices (the same gap `2-run_megadetector.py` already had fixed).

**`hidream-i1` exported the same way, 2026-08-04, on the A40**: 1,200
images, 1,198 annotated / 2 skipped (its two `n_significant == 0` images
from the table above fell through to the zero-detection SKIP path this
time), 1,229 boxes total. Same provisional caveat as the other five —
stages 3/4 have not run on this cell either. `annotations.json` lives
under `data/synthetic_model_comparison/train/hidream-i1/maxlen/`, which is
gitignored — rsynced straight to `gpu-server` (the 3060) right after
export, along with the full `index.jsonl` and all 1,200 images (1.5GB),
so §3.3 training can pick this cell up there the same way it did for the
other five. Counts (1,200/1,200) and spot-checked md5sums matched on both
ends. A stale partial copy was already sitting on `gpu-server` (21 images,
a 2-record `index.jsonl`, from an unrelated earlier attempt) — removed and
replaced with the authoritative version rather than merged with it. See
TODO.md §1.1 for the now-confirmed host alias/login user
(`gpu-server.taile550ef.ts.net`, user `debian`) — previously only
inferred, not verified.

Running §3.3 end-to-end on real data for the first time surfaced two
latent bugs in the shared training loop (`scripts/training/yolov5s/training_pipeline.py`,
imported by both the main YOLOv5s and YOLO26n pipelines, and duplicated
into this package's own `training/training_pipeline.py`) — both now fixed
in all three copies:

1. **No gradient clipping anywhere in the training step** — training
   diverged to NaN partway through the 3-epoch LR warmup on 2 of 5 cells
   tested before the fix (confirmed not data-corruption: bbox geometry was
   clean, and the historic incumbent-generator run had trained fine, just
   with a lower initial loss). Fixed by adding
   `scaler.unscale_` + `clip_grad_norm_(max_norm=10.0)` before
   `scaler.step()`, matching Ultralytics' own trainer's clipping exactly.
2. **The post-training full-eval hook hangs indefinitely** — it builds a
   fresh `num_workers=8` DataLoader deep into a process that has already
   been driving CUDA for the whole training run (unlike `dl_train`/`dl_val`/
   `dl_test`, which fork their workers once near process start, before any
   CUDA activity). Forking new workers at that point hung at 0% GPU/CPU
   utilization for 7+ hours in practice with no error. Fixed by forcing
   `num_workers=0` for that specific call; the *standalone* post-hoc eval
   entrypoint (`eval_suite/run_evaluation.py --run-dir ...`, run as its own
   fresh process) was unaffected and still defaults to
   `constants.NUM_WORKERS`.

Real-test results, 2 seeds (42/43) per cell, `--full-eval` on the fixed full
9,742-image real test set:

| Cell | seed 42 map / map_50 | seed 43 map / map_50 | avg map |
|---|---|---|---|
| `realvisxl-lightning` | 0.023 / 0.049 | 0.024 / 0.052 | 0.024 |
| `sd35m` | 0.054 / 0.101 | 0.053 / 0.100 | 0.054 |
| `flux2-klein-9b` | 0.037 / 0.079 | 0.036 / 0.075 | 0.037 |
| `sd35-large` | 0.056 / 0.100 | 0.054 / 0.094 | 0.055 |
| `sd35-large-turbo` | 0.047 / 0.093 | 0.039 / 0.088 | 0.043 |

`sd35-large` and `sd35m` currently rank highest, `realvisxl-lightning`
lowest — all in the same low-map range expected for a direct fine-tune on
~960 synthetic training images (12 classes, one 80/20 internal split),
consistent with the historic incumbent-generator run's own real-test map
of 0.064 on the same test set. Per-run logs, eval reports, and per-class/
confusion CSVs are at `scripts/synthetic_model_comparison/training/model_exports/yolo26n-<cell>-maxlen-seed<N>-<timestamp>/`.
**Do not treat this ranking as final** — it reflects MegaDetector's
best-effort boxes only, not the reviewed labels §3.4 will produce.

**Update 2026-08-05 — `hidream-i1` trained (6th and final `maxlen` cell),
plus a deep cross-cell comparison.** `hidream-i1` was trained the same way
as the other five (2 seeds, `--full-eval`), completing the `maxlen` grid:
seed 42 map=0.028, seed 43 map=0.031, avg map=0.029 — second-to-last in the
grid despite being by far the most expensive cell to generate (~43.3h vs.
`sd35-large`'s 18.11h). Both runs completed cleanly, no recurrence of the
gradient-divergence or eval-hang bugs fixed 2026-08-04.

A new script, `scripts/synthetic_model_comparison/6-compare_maxlen_cells.py`,
now aggregates all six cells' `evaluation_report.json` files (mean/std
across seeds) into `reports/model_comparison_maxlen_downstream_map.csv` and
`reports/model_comparison_maxlen_per_class_ap.csv`, plus four charts — no
such cross-cell aggregation existed before this (the table above was
assembled by hand). Full ranking, per-class/confusion breakdowns, the
cost-vs-quality analysis, and a specific check of whether
`flux2-klein-9b`'s documented kinkajou-tail rendering bug shows up as
measurable AP loss (it doesn't — both `flux2-klein-9b` and `sd35-large`
land at ~0.006 AP, indistinguishable within the Band-A noise floor) are
written up in [`14_maxlen-cell-deep-comparison.md`](14_maxlen-cell-deep-comparison.md).
Headline: the SD3.5 family wins (`sd35-large` ≈ `sd35m` > `sd35-large-turbo`),
generation cost does not predict downstream quality (`hidream-i1`, the most
expensive, and `realvisxl-lightning`, the cheapest, are the two weakest),
and zebra-group confusion tracks overall detector quality rather than
behaving as an independent failure mode. Same caveat as above: still
provisional pending §3.4.

**Same day — a second comparison including the API incumbent.** Per
follow-up request, extended the comparison to also include
`gemini-3.1-flash-image-preview` (the only API-model cell trained so far;
`gemini-3.1-flash-lite-image` and both `gpt-image-2` tiers aren't labeled or
trained yet, so were left out rather than triggering a new labeling
campaign). New script `scripts/synthetic_model_comparison/7-compare_all_cells.py`
and doc [`15`](15_all-cells-comparison-with-api-incumbent.md). Headline:
the incumbent tops the 7-cell ranking (0.064 mAP) but **this isn't a
controlled comparison** — it used the unabridged `full` prompt regime
(~1,300 words) vs. the local cells' length-capped `maxlen` regime
(256/512 tokens), and has only 1 seed vs. 2. Per-class, its lead turns out
to be concentrated almost entirely in the three zebra classes (especially
`grevy's zebra`, a 3.6x margin over the best local cell) while it's
mid-pack-or-below on every other class — and its zebra confusion rate
isn't actually the lowest in the grid, suggesting the advantage is better
zebra *detection*, not better fine-species *discrimination*. Doc 14's
six-cell grid remains the fair, apples-to-apples comparison.

**2026-08-07 — the controlled ablation, plus every remaining API cell
labeled and trained.** Per follow-up request: fixed both API generation
scripts (`1e-generate_images_openai.py`, `1d-generate_images_new_generator.py`)
to support `--prompt-regime compressed` (both had a stale
guard/hardcoded-`"full"` blocker — see doc `16` for the root cause and
fix), generated `gpt-image-2-low/compressed` (1,200/1,200 images, ~$3.60
estimated) and `gemini-3.1-flash-lite-image/compressed` (1,200/1,200,
flexible Google/Gemini budget), then labeled (best-effort MegaDetector
export) and trained (2 seeds, `--full-eval`) **five** cells total: those
two plus the three previously-generated-but-never-trained `full`-regime
cells (`gpt-image-2-low/full`, `gemini-3.1-flash-lite-image/full`,
`gpt-image-2-medium/full`). New script
`scripts/synthetic_model_comparison/8-compare_prompt_regime_ablation.py`
and doc [`16`](16_prompt-length-ablation.md).

**Headline: prompt length is a real, large, model-independent effect** —
compressing the prompt cuts downstream mAP by ~42% (`gpt-image-2-low`:
0.130→0.075) and ~49% (`gemini-3.1-flash-lite-image`: 0.094→0.048), and
zebra-group confusion gets *worse* under `compressed` for both models too
(unlike doc 15's incumbent case, where confusion didn't track the mAP
lead). This resolves doc 15's confound: the gemini incumbent's edge over
the local `maxlen` cells is much more plausibly its longer prompt than the
model itself. **Bonus finding**: `gpt-image-2-low/full` (0.130) and
`gpt-image-2-medium/full` (0.132) both clearly beat every other cell in
the experiment — over 2x the best local `maxlen` cell and ~2x the gemini
incumbent. Same provisional caveat as every comparison so far.

**2026-08-08 — consolidated into one full-experiment comparison.** New
script `scripts/synthetic_model_comparison/9-compare_all_models.py` and
doc [`17`](17_full-experiment-comparison.md) bring all 12 trained cells
(the 6 local `maxlen` cells + 6 API cells: the incumbent, both
`gpt-image-2` tiers, and `gemini-3.1-flash-lite-image`, each `full` and/or
`compressed`) into one ranking, ties docs 14-16's findings together, and
surfaces a new result only visible with everything side by side: comparing
the two `full`-regime gemini cells directly (a genuinely single-variable
comparison neither doc 15 nor 16 made) shows **`gemini-3.1-flash-lite-image`
(0.094 mAP) clearly beats the production incumbent
`gemini-3.1-flash-image-preview` (0.064 mAP)** on nearly every class and
with lower zebra confusion too. Combined with `gpt-image-2-low` tying
`gpt-image-2-medium` at a third of the cost, the pattern holds on both API
families: **the cheaper tier is not the worse one.** The top 5 of 12 cells
are all API models — every local diffusion cell trails even the weakest
API cell. Same provisional caveat as every comparison so far.
