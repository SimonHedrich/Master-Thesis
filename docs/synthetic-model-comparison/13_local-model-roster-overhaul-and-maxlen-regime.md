# Local-Model Roster Overhaul, GPU-Memory Optimization, and the `maxlen` Prompt Regime

**Date:** 2026-07-30
**Status:** roster expanded from 3 to 7 models, all benchmarked (including
HiDream-I1, initially rejected on a gated-repo access check and later
unblocked and adopted once real access was confirmed); a second prompt
regime (`maxlen`) built alongside `compressed`; first full 1,200-image
production cell (`sd35-large-turbo`) complete
**Depends on:** [`04_local-models-and-output-parameters.md`](04_local-models-and-output-parameters.md),
[`05_prompt-strategy-and-length-limits.md`](05_prompt-strategy-and-length-limits.md),
[`12_additional-generator-cells-build-log.md`](12_additional-generator-cells-build-log.md)

---

## 1. What this materializes

Doc `12` §5 left the local-model tier at "pipeline built and smoke-tested,
3 models, 2 images each — full cells not yet run." This doc covers
everything built since: a GPU-memory-utilization fix found by actually
profiling the smoke test, a roster overhaul (3 models → 7, with FLUX.2-dev
researched and rejected, and HiDream-I1 initially rejected then
subsequently unblocked and re-adopted once real repo access was confirmed),
full benchmarking of all seven models, a second prompt regime built to use
each model's real capacity instead of the fairness-motivated `compressed`
prompt, and the first full 1,200-image production cell. The GPU-memory
fix (~1.8x speedup, §2), the `FLUX.2-dev` rejection (too large, §3), and
the `maxlen` prompt regime (2,400 prompts, all 12 classes, §5) are each
one-time, roster-wide pieces of work rather than per-model status; per-model
progress (benchmarked vs. full `maxlen` cell generated) is tracked in the
benchmark table in §4.

## 2. GPU-memory-utilization investigation and fix

While benchmarking the original 3-model roster, GPU memory utilization sat
between 23-39% of this GPU's 23.8GB VRAM. Investigating why: every model in
`1g-generate_images_local.py` called `enable_model_cpu_offload()`, a blanket
default inherited from the RTX-3060-12GB-targeting production script this
was copied from. Offload deliberately parks most weights on system RAM,
moving only the active submodule to GPU per denoising step — which reads as
low average VRAM use and costs real per-step PCIe transfer time on a card
that doesn't need the VRAM savings.

**Fix:** removed `enable_model_cpu_offload()` from `_load_sd35m` (SD 3.5
Medium), replacing it with a direct `.to("cuda")` — its 2.5B transformer +
T5-XXL/CLIP encoders fit comfortably in ~16.6GB, well under 23.8GB.
RealVisXL (never offloaded to begin with) was already the fastest model
benchmarked, consistent with this being the real lever. **Measured result:
14.84s/image vs. the previous 27.33s/image with offload — a ~1.8x
speedup**, confirmed by re-running the benchmark, not just theorized.

This fix does **not** generalize to any of the other models in the roster —
all of them structurally exceed 23.8GB before activations even without
offload, so there's no headroom for them to reclaim; they keep
`enable_model_cpu_offload()` regardless.

## 3. Roster changes: research trail and final picks

The user asked to replace FLUX.1-schnell with "the current best" FLUX
model and evaluate SD 3.5 Large/Large-Turbo. Research (HF API file-size
queries, `inspect.signature()` on installed pipeline classes, targeted web
fetches — all read-only, no guessing) went through several corrections:

- **FLUX.2-dev (32B params)** — evaluated first. Its pre-quantized 4-bit
  community checkpoint (`diffusers/FLUX.2-dev-bnb-4bit`) still totals ~34GB
  of weights, more than 23.8GB VRAM even 4-bit, mandatory
  `enable_model_cpu_offload()`, real OOM risk. **Not adopted.**
- **FLUX.2-klein-9B** — proposed as a lighter alternative. A first search
  suggested it "fits 24GB at FP16, no offload needed"; a direct fetch of
  its actual model card contradicted this — its 9B transformer (~18GB
  bf16) plus its 8B "Qwen3" text encoder (~16GB bf16) also total ~34GB, and
  offload is "essential" per BFL's own docs. **Adopted anyway** — its real
  advantage over dev isn't a smaller footprint, it's speed: step-distilled
  to 4 steps (vs. dev's 28-50), so far fewer offload swap-cycles per image.
  Also required a different pipeline class than expected
  (`Flux2KleinPipeline`, not the generic `Flux2Pipeline`) — caught by
  inspecting the actual model card before writing the loader, not assumed.
- **SD 3.5 Large / Large-Turbo** — added as requested. Both 8B-transformer
  variants exceed 23.8GB combined with their shared T5-XXL/CLIP encoders
  (~27.6GB), so both need offload, unlike Medium.
- **Qwen-Image (20.4B DiT + 8.3B Qwen2.5-VL encoder, Apache-2.0)** and
  **HiDream-I1 (17B DiT, MIT-licensed)** — proposed by the user as further
  current-generation candidates after more research. Qwen-Image adopted:
  no official or `diffusers`-org pre-quantized checkpoint exists (unlike
  FLUX.2-dev's), so this quantizes it at load time from the **official**
  repo using the same `BitsAndBytesConfig` NF4 pattern already proven for
  FLUX.1-schnell. HiDream-I1 was **initially rejected**: it needs Meta's
  gated `meta-llama/Llama-3.1-8B-Instruct` as a 4th text encoder (not
  bundled in HiDream's own repo), and this account's HF token got a live
  403 on it — confirmed via an actual download attempt, not assumed from
  the "auto"-gated `model_info()` check that (misleadingly) succeeded. An
  ungated community mirror exists but has reported pipeline-compatibility
  issues; the user chose to skip it rather than debug a third-party
  substitution.

  **Re-checked later the same session** with the identical real-access
  test — `hf_hub_download(..., filename="config.json")` against the live
  repo, not `model_info()`, whose `gated` field reports "manual" whether or
  not real access exists. **It succeeded.** Access had been granted since
  the first check, so HiDream-I1 was adopted as the roster's 7th model. See
  §4 for the load-time preparation (disk/RAM checks, NF4 quantization,
  pipeline-shape verification) and the resulting benchmark numbers.

**Final roster (7 models):** `flux2-klein-9b`, `realvisxl-lightning`,
`sd35m`, `sd35-large`, `sd35-large-turbo`, `qwen-image`, `hidream-i1`. See
doc `04` §7 for the full table (license, param count, offload status per
model) and §6 for the updated licensing implications (Qwen-Image and
HiDream-I1 are the roster's two most permissive licenses — Apache-2.0 and
MIT respectively — though HiDream-I1's is a composite with Meta's gated
Llama 3.1 Community License for its 4th text encoder).

## 4. Benchmark results (all 7 models)

Measured via `1g-generate_images_local.py --benchmark N` (5-10 images/model,
`compressed` regime — see `reports/model_comparison_local_generation_benchmark.csv`).
Full table and the GPU-offload finding are in doc `04` §8; summary:

| Generator | s/image (steady-state) | Est. hours for 1,200 | Benchmarked | Full `maxlen` cell |
|---|---|---|---|---|
| `realvisxl-lightning` | 1.14 | 0.38 | ✅ | ❌ |
| `sd35m` (offload removed) | 14.84 | 4.95 | ✅ | ❌ |
| `sd35-large-turbo` | ~25.5 | 8.5 | ✅ | ✅ — §6 |
| `flux2-klein-9b` | ~48 | 16.0 | ✅ | ❌ |
| `sd35-large` | ~57.8 | 19.3 | ✅ | ❌ |
| `qwen-image` | ~101.4 | 33.8 | ✅ | ❌ |
| `hidream-i1` | 129.85 | 43.28 | ✅ | ❌ |

**Total ≈ 126 GPU-hours (~5.3 days continuous)** for all seven full cells
back to back — itself a thesis-relevant finding given doc `04` §5's
original throughput-gap argument, now with real numbers on real
current-generation models rather than an RTX-3060-era FLUX.1-dev estimate.
`hidream-i1` is the slowest model in the roster, ahead of `qwen-image`'s
101.4s/image — plausibly the combined cost of 50 non-distilled steps
(matching Qwen-Image) plus offloading four separate text-encoder components
(vs. Qwen-Image's one) in and out of GPU memory per image, despite
NF4-quantizing its two largest pieces.

Two quality notes surfaced during benchmarking:
- **Qwen-Image's NF4-quantized output showed visible quantization-artifact
  graininess** (blocky texture noise, most visible in grass/fur) compared
  to the other models' clean output — a real speed/quality tradeoff of the
  custom on-the-fly quantization approach, not necessarily representative
  of Qwen-Image at full precision.
- **HiDream-I1's NF4-quantized output showed no comparable artifacts**, in
  either of its two spot-checked images (a smoke test and a benchmark
  image) — clean and anatomically coherent despite quantizing its two
  largest components (transformer, Llama text encoder) the same way
  Qwen-Image's single quantized component was.

Three gated-repo license-acceptance / access gaps were also found and
resolved during benchmarking (not obvious from `HfApi.model_info()`
metadata checks alone, which can succeed even without real download
access — verified this the hard way each time):
- `stabilityai/stable-diffusion-3.5-large-turbo` needed a separate manual
  license acceptance on huggingface.co even though Medium and Large were
  already approved for this account — gating is per-repo, not
  per-organization.
- `meta-llama/Llama-3.1-8B-Instruct` (needed by HiDream-I1 as its 4th text
  encoder) needs manual per-repo approval from Meta specifically, unlike
  the "auto" gates on every Stability/BFL repo used here. This account's
  access was initially denied (a live 403 — see §3), then granted later the
  same session, reconfirmed with a real
  `hf_hub_download(..., filename="config.json")` call against the live
  repo rather than trusting `model_info()`'s `gated` field.

**Bringing HiDream-I1 online — preparation before benchmarking:** its
footprint made this the most resource-constrained model onboarded so far,
so disk and RAM were checked before writing any loader code:
- **Disk:** only 62GB free at the time. `HiDream-ai/HiDream-I1-Full`
  bundles its transformer + CLIP/OpenCLIP/T5-XXL text encoders + VAE,
  totaling ~47.2GB (`model_info(files_metadata=True)`); the
  Llama-3.1-8B-Instruct safetensors shards add ~16GB more — ~63GB combined,
  more than what was free. Cleared
  `stabilityai/stable-diffusion-3.5-large-turbo`'s HF cache (~26GB, its
  compressed-benchmark and full `maxlen` cell both already complete — see
  §6) to get to ~88GB free, the same "clear an already-benchmarked model's
  cache" move already established as practice here.
- **RAM:** `free -h` showed 47GB total / 41GB available. HiDream-I1's full
  bf16 footprint (transformer ~34.7GB + Llama ~16GB + T5-XXL ~9.5GB +
  CLIP/OpenCLIP ~3.3GB ≈ 63.5GB) doesn't fit in that — and
  `enable_model_cpu_offload()` loads the *entire* pipeline onto the CPU
  before moving submodules to GPU, so a naive bf16 load risked heavy
  swapping or an OOM before generation even started, not just a slow run.
  **Mitigation:** quantized the transformer and the Llama text encoder to
  NF4 at load time — the exact `BitsAndBytesConfig` pattern already proven
  for `qwen-image`, and the same rationale (a model too large to load
  safely on this machine otherwise). This cuts resident footprint to
  roughly transformer 8.7GB + Llama 4GB + T5-XXL 9.5GB (kept bf16, bundled)
  + CLIP/OpenCLIP 3.3GB ≈ 25.5GB — comfortably inside RAM, and still over
  the 23.8GB VRAM ceiling, so `enable_model_cpu_offload()` stays mandatory.

Confirmed the pipeline's actual shape against the installed `diffusers`
(0.39.0) source rather than assuming: `HiDreamImagePipeline` takes
`text_encoder_4`/`tokenizer_4` (Llama) as separately-constructed arguments,
same shape as `_load_qwen_image`'s separate transformer/text_encoder
construction; its own docstring example loads Llama with
`output_hidden_states=True` and calls the pipeline with
`num_inference_steps=50`, `guidance_scale=5.0` as its stated defaults, and
accepts a plain string `negative_prompt` (unlike the Flux family in this
same script). Added `_load_hidream_i1()`/`_generate_hidream_i1()` to
`1g-generate_images_local.py` following that shape, at the same
(1152, 864) 4:3 resolution bucket as the rest of the roster.

**Smoke-tested first** (`--generator hidream-i1 --classes lion --limit 1`)
before committing to the full benchmark — pipeline load took 811s (a cold
download of ~63GB), generation itself took 128.0s, and the output image was
spot-checked visually: clean, anatomically coherent, no visible
quantization artifacts. GPU memory returned to 0MiB used after pipeline
unload — no leak.

**Then ran the real benchmark** (`--generator hidream-i1 --benchmark 5`,
`compressed` regime, same as every other model): weights were already
cached from the smoke test, so pipeline load dropped to well under a
minute. **Result: 5/5 images generated, 0 errors** — appended to
`reports/model_comparison_local_generation_benchmark.csv`. A second
spot-checked image (`plains_zebra`) showed the same clean, artifact-free
quality as the smoke test.

## 5. The `maxlen` prompt regime

The `compressed` regime (doc `05` §3) deliberately gives every local model
the *same* ≤75-token prompt, for a fairness ablation against prompt length
as a confound. The user's follow-on ask reframed the goal for the actual
production dataset: use each model's *real* capacity instead of leaving it
on the table — SD 3.5's T5 branch holds 256 tokens; FLUX.2-klein-9B and
Qwen-Image's Qwen-family encoders hold 512, but `compressed` only ever
supplies ~40-57.

**First approach tried and rejected:** truncate the existing `full`-regime
prompt (~1,300-word incumbent-style template, already built for all 12
classes) down to each tier's budget. Inspecting an actual full prompt file
showed this doesn't work — its own *structural* sections (SCENE
SPECIFICATION + PHOTOGRAPHY STYLE + CRITICAL REQUIREMENTS, i.e. everything
that isn't a free-text Wikipedia excerpt) already total ~464 words (~600+
T5 tokens) on their own, more than the entire 256-token budget before any
species description is added.

**What was built instead** (`scripts/synthetic_model_comparison/1h-generate_prompts_maxlen.py`):
the simpler, already-proven `build_prompt()` shape from
`scripts/synthetic/2-generate_synthetic_images_local.py` — doc `05` §1's
"path A" tier — extended to actually *fill* each budget (not just clear
it), with genuine per-image scene variation (pose/environment sentences)
pulled directly from the canonical `train/gemini-3.1-flash-image-preview/full/index.jsonl`
(1,200 already-parsed records) rather than a 5-item pose rotation. One real
bug found and fixed along the way: 6 of the 12 classes (kinkajou, water
deer, ringtail, saiga, aye-aye, pangolin family) keep non-contiguous
indices from a 200-image pool (per doc `10`'s stratify-diversify
selection), so the script iterates whatever indices a class actually has
rather than assuming `range(1, 101)` — caught by an actual `sys.exit` on
class aye-aye during testing, not anticipated in advance.

**Result:** 2,400 prompts (12 classes × 2 tiers × 100 images) at
`data/synthetic_model_comparison/train/prompts_maxlen_{256,512}/`, plus
`reports/model_comparison_maxlen_prompt_metadata.jsonl`. Verified token
counts: max 245/245 (256 tier) and 495/495 (512 tier) — every class
comfortably at or under budget, including both extremes (aye-aye's short
Wikipedia article vs. red fox's very long one). Full design rationale in
doc `05` §7.

`1i-generate_images_local_maxlen.py` consumes this (plus the unchanged
`compressed` prompt for `realvisxl-lightning`, already optimal at 77
tokens) and writes to a new `<generator>/maxlen/` cell, sibling to
`<generator>/compressed/` — the `compressed` cells are untouched, staying
available for a possible later ablation against the best proprietary API
model.

## 6. First full production cell: `sd35-large-turbo`

Per the user's explicit scoping ("do part A and B but then only run
sd35-large-turbo for now"), only this one model's full 1,200-image `maxlen`
cell was generated this session — chosen as the fastest of the four newly
added models, to validate the new script and prompt design at real
production scale before committing the much larger GPU-hour investment the
other four would need.

**Result: 1,200/1,200 images, 0 failures.** Total wall-clock 8h 16min
(29,783s inference-only), averaging **24.8s/image** — close to but
slightly above the small-sample benchmark's 25.5s estimate, with a mild,
real rate variation observed across classes (saiga/aye-aye/pangolin family
ran ~29-31s/image vs. plains zebra/grevy's zebra's ~23-24s, plausibly
correlated with those classes' maxlen prompts sitting closer to the
256-token ceiling). Visually spot-checked images from `plains_zebra`,
`lion`, and `pangolin_family` for basic sanity — all show clean,
anatomically coherent subjects with the diagnostic features and per-image
scene variation the `maxlen` prompt was designed to carry (e.g. the
pangolin's scaled body rendered accurately amid rocky terrain).

## 7. Output layout (new since doc `12`)

```
data/synthetic_model_comparison/train/
├── prompts_maxlen_256/<class_slug>/<NNN>.txt   # new — §5
├── prompts_maxlen_512/<class_slug>/<NNN>.txt   # new — §5
├── flux2-klein-9b/compressed/                   # benchmark only (10 images) — §4
├── sd35-large/compressed/                       # benchmark only (5 images) — §4
├── sd35-large-turbo/
│   ├── compressed/                              # benchmark only (5 images) — §4
│   └── maxlen/                                  # DONE — 1,200/1,200 images — §6
├── qwen-image/compressed/                       # benchmark only (5 images) — §4
├── hidream-i1/compressed/                       # benchmark only (5 images) — §4
└── ... (flux-schnell/, realvisxl-lightning/compressed/, sd35m/compressed/ from doc 12, unchanged)
```

New reports: `reports/model_comparison_local_generation_benchmark.csv`,
`reports/model_comparison_maxlen_prompt_metadata.jsonl`,
`reports/model_comparison_local_generation_timing_maxlen.csv`,
`reports/model_comparison_local_generation_benchmark_maxlen.csv`.

## 8. Verification performed

For the `maxlen` prompt build: dry-run first, then a 2-class test
(aye-aye/lion — shortest/longest Wikipedia source) before the full 12-class
run; confirmed all 2,400 prompts land at or under their tier budget via the
tier's own tokenizer. For `1i`: smoke-tested with `--limit 2` before the
full run. For the full `sd35-large-turbo` cell: confirmed 1,200/1,200
`index.jsonl` records, 1,200/1,200 files on disk, zero errors in the run
log, and visual spot-checks across three classes. For `hidream-i1`'s
onboarding: verified real (not metadata-only) repo access via a live
`hf_hub_download()` before writing any loader code, checked disk/RAM
headroom against the model's actual bundled-file sizes before pulling
~63GB of weights, smoke-tested a single image before the full 5-image
benchmark, and confirmed GPU memory returned to 0MiB after pipeline
unload. Disk was actively monitored throughout (each new model's HF cache
is 13-54GB; cleared already-benchmarked models' caches between runs to stay
within this machine's ~500GB disk).

## 9. Not built here

- **The other six models' full `maxlen` cells** (`realvisxl-lightning`,
  `sd35m`, `flux2-klein-9b`, `sd35-large`, `qwen-image`, `hidream-i1`) —
  each a separate multi-hour-to-multi-day GPU commitment (up to ~43h for
  `hidream-i1`, the roster's slowest model), left for a follow-up
  go-ahead per model, not run automatically.
- **Batching multiple prompts per pipeline call** and **`torch.compile`** —
  both documented as real throughput follow-ups in doc `04` §4/§8, given
  the GPU headroom found in §2, but not implemented — not necessary to
  answer what was actually asked this session.
- Everything doc `12` §8 already listed as not built (Nano Banana Pro, the
  `compressed`-regime ablation pair, labeling/COCO export on new cells).
