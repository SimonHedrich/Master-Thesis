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
results. `realvisxl-lightning`, `sd35m`, `flux2-klein-9b`, and `sd35-large`
have since completed the same way.

**`qwen-image`'s full `maxlen` cell was dropped, not deferred** — its
`compressed`-regime 5-image smoke test showed clearly visible
quantization-artifact graininess (stippled noise in flat regions — sky,
grass), the same issue first flagged in doc [`04`](04_local-models-and-output-parameters.md)
§8. Since `qwen-image` was never a headline model (kept mainly for its
Apache-2.0 license) and the artifact is structural to the on-the-fly NF4
quantization this box's VRAM requires, the ~34h production run was skipped
rather than generated and discarded later. See
[`13`](13_local-model-roster-overhaul-and-maxlen-regime.md) §9.
`hidream-i1` (added and benchmarked later) still has no `maxlen` tier built
for it — separate, not-yet-started status.

Still open: the fifth API cell (Nano Banana Pro), the `compressed`-regime
ablation pair (incumbent + gpt-image-2 low), and `hidream-i1`'s full
`maxlen` cell (~43h, plus porting its loader into `1i` first — see
[`13`](13_local-model-roster-overhaul-and-maxlen-regime.md) §9).

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

**Stage 2 (MegaDetector) has now run on all five completed `maxlen`
cells** — `realvisxl-lightning`, `sd35m`, `flux2-klein-9b`, `sd35-large`,
`sd35-large-turbo`, all 1,200/1,200 images, 0 missing. `2-run_megadetector.py`'s
`--prompt-regime` choices didn't yet include `maxlen` (added when the regime
was introduced in doc `13` after this script was written) — fixed as part
of this run. Per-cell `n_significant` distribution (share of images with
0 / 1 / ≥2 detections ≥0.5 conf):

| Cell | 0 | 1 | ≥2 |
|---|---|---|---|
| `realvisxl-lightning` | 0.0% | 97.1% | 2.9% |
| `sd35m` | 0.3% | 98.2% | 1.5% |
| `flux2-klein-9b` | 0.2% | 93.6% | 6.2% |
| `sd35-large` | 0.1% | 98.8% | 1.2% |
| `sd35-large-turbo` | 0.1% | 97.1% | 2.8% |

Stages 3–5 (triage review, bbox labeling, COCO export) are still pending —
those are human-in-the-loop review steps, not yet run on any cell. A first
training pass is the natural next step once a cell clears them.
