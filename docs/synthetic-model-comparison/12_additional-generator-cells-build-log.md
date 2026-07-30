# Additional Generator Cells: Nano Banana 2 Lite, gpt-image-2, and the Local-Model Tier

**Date:** 2026-07-29
**Status:** 4 more generator cells complete (1,200 images each); local-model tier's
pipeline is built and smoke-tested but its three full cells are not yet generated
**Depends on:** [`01_experiment-design.md`](01_experiment-design.md),
[`05_prompt-strategy-and-length-limits.md`](05_prompt-strategy-and-length-limits.md),
[`10_train-subset-incumbent-selection.md`](10_train-subset-incumbent-selection.md)

---

## 1. What this materializes

`10_train-subset-incumbent-selection.md` closed out the incumbent cell
(`gemini-3.1-flash-image-preview`/`full`, 1,200/1,200 images) and flagged that
every other cell in the 01-design's model grid was still open. This doc
covers everything built since:

| Cell | Regime | Images | Status |
|---|---|---|---|
| `gemini-3.1-flash-image-preview` (incumbent) | `full` | 1,200/1,200 | Done — see doc `10` |
| `gemini-3.1-flash-lite-image` (Nano Banana 2 Lite) | `full` | 1,200/1,200 | **Done — §2** |
| `gpt-image-2-low` | `full` | 1,200/1,200 | **Done — §3** |
| `gpt-image-2-medium` | `full` | 1,200/1,200 | **Done — §3** |
| Nano Banana Pro | `full` | 0 | Not started |
| `flux-schnell` | `compressed` | 2 (smoke test) | Pipeline built — §5 |
| `realvisxl-lightning` | `compressed` | 2 (smoke test) | Pipeline built — §5 |
| `sd35m` | `compressed` | 2 (smoke test) | Pipeline built — §5 |
| `gemini-3.1-flash-image-preview` / `gpt-image-2-low` | `compressed` (ablation, doc `05` §2) | 0 | Not started — prompts exist (§4), not yet generated |

The `full`-regime prompt text (`data/synthetic_model_comparison/train/prompts_full/`)
was already complete for all 12 classes before this doc's work started (doc
`10` finished it) — so the two `full`-regime cells below (§2, §3) needed
**no new prompt authoring**, only a new generic image-generation script per
API vendor. The local-model tier needed a new prompt regime first (§4)
since it can't read `full`-length prompts at all.

## 2. Nano Banana 2 Lite (`gemini-3.1-flash-lite-image`)

`scripts/synthetic_model_comparison/1d-generate_images_new_generator.py` — a
generic script, not lite-specific: it reads an existing cell's `index.jsonl`
(the incumbent's, by default) for per-image class/band/shot_type/distance/
lighting/occlusion/pose/environment metadata, reads the prompt from that
record's `dest_prompt_file` (the shared `prompts_full/` location doc `10` §5
specifically designed for reuse), and generates a new generator's images via
the **Gemini Batch API** (`client.batches.create`, `-50%` vs. direct calls).
This generalizes to any future Gemini-family cell, not just this one.

**Resolution:** doc `03`'s pricing table lists `0.5K, 1K` as both supported
for this model, but the installed `google-genai` SDK's `ImageConfig.image_size`
docstring only documents `1K`/`2K`/`4K` (no `0.5K`) — the incumbent's `0.5K`
clearly works in practice, but that's not proof the Lite model accepts it
too. Rather than guess, this was checked empirically: a 3-image batch smoke
test at `1K` succeeded outright, so `1K` was used for the full cell (no
`0.5K` variant was tested since `1K` worked on the first try).

**Result:** 1,200/1,200 images (100/class × 12), 0 failures, single batch
job (no chunking needed — Gemini's batch token limits didn't bind at this
prompt-length/image-count combination, unlike OpenAI's, see §3).

## 3. gpt-image-2 (`low` and `medium` quality tiers)

`scripts/synthetic_model_comparison/1e-generate_images_openai.py` — same
reuse pattern as `1d` (reads the incumbent's `index.jsonl` + `dest_prompt_file`),
adapted for the **OpenAI Batch API** (`/v1/images/generations` endpoint,
confirmed supported via the installed `openai` SDK's `batches.create`
docstring). Quality tier is a distinct axis from the model name per doc `01`,
so each tier gets its own cell directory: `gpt-image-2-low/full/` and
`gpt-image-2-medium/full/`. Size fixed to `1024x768` (4:3, matching every
other cell) — confirmed valid via a live test call (gpt-image-2's actual
constraint is divisibility by 16 + a total-pixel range, not a fixed preset
list like Gemini's).

Two real per-model constraints surfaced during generation, both now handled
automatically by the script:

- **1,000,000 enqueued-token org cap.** A same-size submission to the
  ~1,300-word (avg. ~2,600-token) `full`-regime prompts failed instantly
  with `token_limit_exceeded` — the org-wide cap for `gpt-image-2` batches.
  Fix: `chunk_by_token_budget()` greedily packs pending requests into
  ≤700,000-token batches (via `tiktoken`, `o200k_base` encoding as a proxy
  for the model's own tokenizer) and `--mode submit` only ever sends the
  next chunk that fits; re-running the same command drives the cell to
  completion over several sequential batches (5 for `low`, 5 for `medium`).
- **32,000-character hard prompt limit.** Every single `red_fox` prompt
  (all 100) is ~33.3-33.5k characters — its Wikipedia article is unusually
  long — so all 100 failed with `string_above_max_length` the first time
  they were submitted. Per doc `05` §2's own framing ("`full` = each model
  at its *maximum usable* prompt"), trimming just enough to fit is
  consistent with the regime's definition, not a departure from it. Fix:
  `truncate_for_openai()` trims only the free-text span between
  `SPECIES DESCRIPTION:` and `SCENE SPECIFICATION:` (the Wikipedia-sourced
  description/behavior text) down to fit under 32,000 chars, leaving the
  scene-spec and critical-requirements tail — the actually
  generation-critical instructions — fully intact. This only changes the
  in-memory request body sent to OpenAI; the shared `prompts_full/red_fox/`
  files on disk (and every other model's copy of the same prompt) are
  untouched.

**Cost reality check.** doc `03`'s flat reference price for `low` quality
(~$0.006/image at 1024×1024, direct) implies ~$0.003/image batch-discounted
— but the actual observed cost for the `low` cell was **$10.11** (~1,200
images ⇒ ~$0.0084/image), about 2.8× the naive estimate. Reason: OpenAI
bills prompt text as input tokens separately from output-image tokens, and
this experiment's long full-regime prompts (~2,600 tokens average) add a
roughly fixed ~$0.0065/image (batch rate) that the flat per-image reference
figure — evidently calibrated against a much shorter prompt — doesn't
capture. This input-token cost is *fixed* regardless of quality tier (same
prompt submitted to `low` and `medium`), so it matters far less at `medium`
quality, where the (quality-scaling) output-token cost dominates — the
naive ~$31.80 estimate for `medium` should hold up much better than `low`'s
did. **Anyone estimating a future OpenAI cell's cost should use the actual
observed low-tier rate (~$0.0084/image direct-equivalent) as the floor, not
doc `03`'s flat reference table, for any cell using these long full-regime
prompts.**

**Result:** both cells 1,200/1,200 images, 0 failures after the two fixes
above (3 requests failed on the very first `low` submission before the
char-limit fix landed; those 3 succeeded on retry with the fix in place, no
extra cost — OpenAI doesn't bill failed/rejected requests).

## 4. Compressed prompt regime (all 12 classes)

`scripts/synthetic_model_comparison/1f-generate_prompts_compressed.py` — the
mandatory shared prerequisite for the local-model tier (SDXL-family models
truncate silently at 77 CLIP tokens; doc `05` requires every local + the two
`compressed`-ablation cells to read the *identical* short prompt). Modeled
on `1b-generate_prompts_fresh.py`'s structure, but self-contained and
LLM-free: the ≤75-token prompt is built entirely from data already cached
in `reports/synthetic_scene_profiles.json` (diagnostic features) plus a
short hand-written per-class habitat phrase and a small rotating pose list
— there's no token budget left for `full`'s six-axis scene grid, so per doc
`05` §3 rule 2 only pose rotates across a class's 100 images, not habitat.

Prompt shape (doc `05` §3): `{class name}, {1-3 features}, {pose}, {habitat},
wildlife photograph, telephoto, photorealistic, full body`. Token count is
verified per-image against the actual SDXL tokenizer
(`openai/clip-vit-large-patch14`), with a trim-and-recheck loop (habitat,
then pose, then features down to 2/1) if a rendered prompt comes in over
budget.

**Result:** 1,200 prompts (100/class × 12) at
`data/synthetic_model_comparison/train/prompts_compressed/`, plus
`reports/model_comparison_compressed_prompt_metadata.jsonl`. Max observed
length: 57 tokens (well under the 75-token ceiling).

## 5. Local-model tier (pipeline built, smoke-tested; full cells not yet run)

`scripts/synthetic_model_comparison/1g-generate_images_local.py` — copies
(not imports) the three `diffusers` pipeline loaders from
`scripts/synthetic/2-generate_synthetic_images_local.py` (FLUX.1-schnell
NF4-quantized, RealVisXL V5.0 + SDXL-Lightning 4-step fusion, SD 3.5 Medium
bf16), reading this experiment's compressed-prompt metadata (§4) instead of
production's, and writing into this experiment's
`train/<generator>/compressed/{images/,index.jsonl}` layout. Resolution is
fixed per generator to that model's native ~1MP 4:3 bucket and saved as-is
— **no forced downscale** (an earlier draft of this doc's parent README
recommended a fixed 500×500; that convention was superseded — see doc `04`
§3 — since it matched nothing else in the repo and this experiment's
training pipeline letterboxes every input to 640×640 regardless of source
size anyway).

**Smoke-tested, not yet run at scale.** All three models generated 2 lion
images each successfully on this machine's GPU (an NVIDIA A40 vGPU
instance — notably not the RTX 3060 12GB the original local-model survey,
doc `04`, assumed):

| Generator | Seconds/image (excl. one-time pipeline load) |
|---|---|
| `realvisxl-lightning` | ~1.05-1.54s |
| `flux-schnell` | ~8.86-10.34s |
| `sd35m` | ~27.5-46.73s |

All three are far faster than doc `04` §5's RTX-3060-based throughput
estimate — the A40 has both more VRAM and more compute. The full
1,200-image/model cells (a multi-hour unattended GPU job per model) were
**deliberately not launched** — that's a real GPU-time commitment left for
the user's call, not something to run unattended by default.

Practical notes for whoever runs the full cells:
- The three models' cached weights (RealVisXL ~13GB, SD 3.5 Medium ~16GB,
  FLUX.1-schnell ~32GB — all larger than doc `04`'s ~25GB estimate) don't
  fit in this disk's free space (~40-53GB) simultaneously. Generate one
  model's cell at a time (`--generator <name>`, not `--generator all`) and
  clear `~/.cache/huggingface/hub/` of the previous model if space runs
  low before the next download.
- Local-diffusion dependencies (`diffusers`, `transformers`, `accelerate`,
  `bitsandbytes`, `sentencepiece`, `protobuf`, `peft`) are now uv-managed in
  the root `pyproject.toml` — no separate ad-hoc `pip install` needed.
- An `HF_TOKEN` in `scripts/synthetic/.env` with access to the gated
  `stabilityai/stable-diffusion-3.5-medium` repo is required (already
  present).
- Per-image seeds are `--seed + global_index`, with `global_index` derived
  from `reports/model_comparison_classes.csv`'s fixed row order, so seeds
  stay stable regardless of which `--classes` subset a given run covers —
  local models are the only fully seed-reproducible tier (doc `06`).

## 6. Output layout

```
data/synthetic_model_comparison/train/
├── gemini-3.1-flash-image-preview/full/    # incumbent — done (doc 10)
├── gemini-3.1-flash-lite-image/full/        # done (§2) — images/, index.jsonl
├── gpt-image-2-low/full/                    # done (§3) — images/, index.jsonl,
│                                             #   openai_batch_{input,output,state,error}*
├── gpt-image-2-medium/full/                 # done (§3) — same layout as low/
├── prompts_full/                            # shared full-regime prompts (doc 10 §5)
├── prompts_compressed/<class_slug>/<NNN>.txt  # shared compressed-regime prompts (§4)
├── flux-schnell/compressed/                 # smoke-tested only (§5) — 2 images
├── realvisxl-lightning/compressed/          # smoke-tested only (§5) — 2 images
└── sd35m/compressed/                        # smoke-tested only (§5) — 2 images
```

## 7. Verification performed

For each of the three newly-completed `full`-regime cells (Lite, gpt-image-2
low, gpt-image-2 medium): confirmed exactly 100 images/class across all 12
classes (1,200/cell), zero duplicate `filename`s in `index.jsonl`, and every
`file_name` path resolves to a real file on disk. Spot-checked 2-3 images
per cell visually for species-diagnostic accuracy and prompt-scene fidelity
(lion, red fox, plains zebra, kinkajou). For the compressed-prompt regime,
spot-checked token counts against the tokenizer's own count and confirmed
the max (57) stays under the 75-token ceiling.

## 8. Not built here

- **Nano Banana Pro** (`full` regime) — the fifth and last API cell in doc
  `01`'s grid.
- **The `compressed`-regime ablation pair** (incumbent + gpt-image-2 low,
  per doc `05` §2) — prompts already exist (§4), just not yet submitted to
  either model.
- **The three local models' full 1,200-image cells** — pipeline is ready
  (§5); running them is a multi-hour-per-model GPU commitment left for the
  user to schedule.
- **Labeling / COCO export** on any of these new cells — deliberately
  deferred until every generator cell is ready to be labeled together (doc
  `10` §6).
