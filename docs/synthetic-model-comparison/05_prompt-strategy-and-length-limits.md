# Prompt Strategy and Length Limits

**Date:** 2026-07-14
**Status:** How to build fair prompts across models with very different limits
**Depends on:** [`04_local-models`](04_local-models-and-output-parameters.md),
[`01_experiment-design`](01_experiment-design.md)

---

## 1. The problem in one table

The incumbent images were conditioned on prompts far longer than any local model
can read. A fair comparison therefore cannot just "use the same prompt" — the
same text means totally different things to a 77-token encoder and a
32k-token one.

| Prompt asset in the repo | Length | Who can use it |
|--------------------------|--------|----------------|
| Production `PROMPT_TEMPLATE` (path B, `1-generate_image_list.py`) | ~1,300 words (train) / ~2,700 words (test prompts) | Gemini, OpenAI only |
| Short `build_prompt()` + `STYLE_SUFFIX` (path A, single-class scripts) | ~90–130 words (~130–180 tok) | all API; FLUX/SD3.5 (fits 256 T5); **still too long for SDXL 77-tok** |
| **New `compressed` prompt (this doc §3)** | **≤55 words / ≤75 CLIP tokens** | **every model, including SDXL** |

## 2. Two prompt regimes (the fairness design)

Per [`01`](01_experiment-design.md) §4, run two regimes so model quality and
prompt capacity are separable:

- **`full`** — each model at its *maximum usable* prompt:
  - Gemini / OpenAI: the full production template (~1,300 words).
  - FLUX.1-dev: ≤512 T5 tokens (~350 words) — a trimmed template.
  - FLUX.1-schnell / SD3.5: ≤256 T5 tokens (~180 words).
  - SDXL/RealVisXL: ≤75 CLIP tokens (the compressed prompt is already its max).
- **`compressed`** — the **identical** ≤75-token prompt (§3). Generated for
  *every local* model, but — as a cost decision — for only **two of the five API
  models: Nano Banana 2 (incumbent) and gpt-image-2 low**. These are the
  cheapest ablation sites (the incumbent's `full` images already exist; a
  gpt-image-2-low cell costs ~$7) and they span the capability range, so the
  cheap-vs-strong interaction is measurable. Full rationale in
  [`01`](01_experiment-design.md) §4b. This is the apples-to-apples cell.

The comparison of `full` vs `compressed` **within the API models** (on the two
ablation models above) measures how much the long Wikipedia prompt actually
buys. If the gain is small, the local models' prompt handicap is largely moot —
an important, non-obvious result.

## 3. Constructing the `compressed` prompt (≤75 CLIP tokens)

> **Update (2026-07-29):** built for all 12 classes × 100 images via
> `scripts/synthetic_model_comparison/1f-generate_prompts_compressed.py` —
> `data/synthetic_model_comparison/train/prompts_compressed/` +
> `reports/model_comparison_compressed_prompt_metadata.jsonl`. Verified
> against the SDXL CLIP-L tokenizer (`openai/clip-vit-large-patch14`); max
> observed length across all 1,200 prompts is 57 tokens, comfortably under
> budget. Habitat is fixed per class (one hand-written short phrase, in the
> style of the worked examples below) rather than rotated — only *pose*
> rotates across a class's 100 images, per rule 2 below; diagnostic features
> are pulled automatically from `reports/synthetic_scene_profiles.json` per
> rule 3.

CLIP ignores everything after ~75 tokens and weights early tokens most, so the
compressed prompt must be **dense, front-loaded, and ordered by importance**.
Recommended slot structure (drop trailing slots first if over budget):

```
[species common name], [1–3 diagnostic visual features], [pose], [habitat],
wildlife photograph, telephoto, photorealistic, full body
```

Worked examples (each ≤ ~50 words):

- **Grévy's zebra** — `Grevy's zebra, narrow dense black-and-white stripes,
  white belly, large rounded ears, standing on arid grassland, wildlife
  photograph, telephoto, photorealistic, full body`
- **saiga** — `saiga antelope, bloated humped downturned nose, pale tan coat,
  ringed amber horns (male), standing on dry steppe, wildlife photograph,
  telephoto, photorealistic, full body`
- **kinkajou** — `kinkajou, small golden-brown arboreal mammal, long prehensile
  tail, round ears, large dark eyes, on a rainforest branch at night, wildlife
  photograph, telephoto, photorealistic`

Construction rules:
1. **Diagnostic features first** — the markers that separate look-alikes (stripe
   type for zebras, nose for saiga) must appear in the first ~30 tokens.
2. **One species, one scene** — no six-axis scene grid (that lived in the long
   template); pick a single pose+habitat per image and vary it across the set by
   rotating through a small fixed list (reuse the existing scene axes as a
   lookup, but render only a few words each).
3. **Derive features automatically** from the existing `reports/synthetic_scene_profiles.json`
   "key diagnostic features" field (already LLM-extracted) — truncate to 3.
4. **Verify token count** with the model's own tokenizer before generating;
   log the token length so truncation is auditable.

## 4. Prompt weighting / chunking for SDXL (optional)

If SDXL underperforms purely due to the 77-token wall, optionally test **Compel
/ sd_embed** chunking + weighting as a *documented variant* (not the default), to
report whether the community workaround recovers quality. Keep it out of the main
comparison cell so the headline SDXL number reflects vanilla capability, but note
the workaround result.

## 5. Keep the semantic content matched

Across regimes and models, the *same species facts* must be expressed — only the
verbosity changes. Build the compressed prompt by **summarising the same scene
profile** used for the long prompt, not by writing a different description. This
keeps the contrast about **how much text the model can use**, not **what the text
says**.

## 6. Negative prompts

The local script already defines a `NEGATIVE_PROMPT` (SDXL/SD3 support it; FLUX
largely ignores CFG-based negatives at guidance 0). API models have no negative
prompt. To stay fair, treat negatives as a **model-native affordance**: use each
model's standard negative-prompt setup (or none), and document it — do not try to
force parity where the mechanism doesn't exist.

## 7. The `maxlen` regime — a third regime for production, not comparison

**Added 2026-07-30.** `compressed` (§3) answers a *comparison* question: does
prompt-length handicap explain a local model's gap vs. the incumbent? That's
still useful, and the `compressed` cells stay as-is for a possible later
ablation against the best proprietary model. But for the actual **production**
1,200-image/model dataset, giving every model the same ≤75-token prompt wastes
most local models' real capacity — SD 3.5's T5 branch holds 256 tokens;
FLUX.2-klein-9B and Qwen-Image's Qwen-family encoders hold 512. `maxlen` (built
by `scripts/synthetic_model_comparison/1h-generate_prompts_maxlen.py`, consumed
by `1i-generate_images_local_maxlen.py`) uses each model's real budget instead:

| Tier | Generators | Token budget (target) |
|---|---|---|
| 75 (unchanged) | `realvisxl-lightning` | reuses the existing `compressed` prompt verbatim — already fills this tier (§3) |
| 256 | `sd35m`, `sd35-large`, `sd35-large-turbo` | ~230-245 tokens (T5-XXL) |
| 512 | `flux2-klein-9b`, `qwen-image` | ~480-495 tokens (Qwen-family encoders) |

### 7a. Why truncating the `full` prompt doesn't work

The first approach tried: take the existing `full`-regime prompt (already
built for all 12 classes, ~1,300-word incumbent-style template) and truncate
it to fit each tier. Inspecting an actual full prompt file
(`data/synthetic_model_comparison/train/prompts_full/lion/001.txt`) showed
this isn't feasible — its own **structural** sections (SCENE SPECIFICATION +
PHOTOGRAPHY STYLE + CRITICAL REQUIREMENTS, i.e. everything that *isn't* a
free-text Wikipedia excerpt) already total ~464 words (~600+ T5 tokens) on
their own, more than the entire 256-token budget before a single word of
species description is added. Free-text section length also varies wildly by
class (696 words for aye-aye's *entire* full prompt vs. 5,438 for red fox's),
so "keep structure, trim only the free text" simply doesn't fit regardless of
how aggressively the free text is cut.

### 7b. What was built instead

The simpler, already-proven `build_prompt()` template shape from
`scripts/synthetic/2-generate_synthetic_images_local.py` — this is exactly
§1's "path A" tier (~90-130 words), which the table above already noted fits
FLUX/SD3.5's 256-token budget with room to spare. `1h` extends that same
shape to actually **fill** each budget (not just clear it) and adds genuine
per-image scene variation:

```
Realistic wildlife photograph of a/an {class} ({scientific name}).
Species characteristics: {1-3 diagnostic features}. {description excerpt,
accumulated sentence-by-sentence up to the tier budget}. {pose sentence}
{environment sentence}. {STYLE_SUFFIX}. Exactly one {class} in frame,
diagnostic features clearly visible, no other individuals of the same
species.
```

- **Description excerpt**: parsed once per class from the *existing*
  `train/prompts_full/<slug>/001.txt`'s `SPECIES DESCRIPTION:` section
  (same content for all 100 images of a class), then greedily accumulated
  sentence-by-sentence until the next sentence would exceed the tier's token
  budget — the inverse of `1f`'s trim-down `fit_to_token_budget`, building
  up instead.
- **Pose / environment**: real per-image values, pulled directly from the
  canonical `train/gemini-3.1-flash-image-preview/full/index.jsonl` (1,200
  already-parsed records, one per class × image) rather than re-deriving the
  production `SHOT_SCHEDULE` — this gives genuine per-image angle/lighting/
  behavior variation the `compressed` regime's simple 5-item pose rotation
  doesn't have, at no extra engineering cost.
- **Index numbers aren't always contiguous 1-100**: the 6 Bucket-3 classes
  (kinkajou, water deer, ringtail, saiga, aye-aye, pangolin family) keep
  their original index out of a 200-image pool (per
  `10_train-subset-incumbent-selection.md`'s stratify-diversify selection),
  so `1h` iterates whatever 100 indices the canonical index.jsonl actually
  has for a class rather than assuming `range(1, 101)`.

Token verification: `T5TokenizerFast` loaded from SD 3.5 Medium's
`tokenizer_3` subfolder for the 256 tier; `Qwen/Qwen-Image`'s own tokenizer
for the 512 tier (FLUX.2-klein-9B's "Qwen3" encoder and Qwen-Image's
Qwen2.5-VL encoder are both Qwen-family BPE — close enough for budgeting
given the safety margin below each nominal ceiling). Measured across all
2,400 generated prompts: max 245/245 (256 tier) and 495/495 (512 tier) —
comfortably at or under budget for every class, including the two extremes
(aye-aye's short Wikipedia article naturally produces shorter prompts;
red fox's very long one still fits since the tier budget, not the source
length, is what's being filled).

Output: `data/synthetic_model_comparison/train/prompts_maxlen_256/<slug>/`,
`.../prompts_maxlen_512/<slug>/`,
`reports/model_comparison_maxlen_prompt_metadata.jsonl`. Generated images
land in a new `<generator>/maxlen/` cell (sibling to `<generator>/compressed/`,
which stays untouched).
