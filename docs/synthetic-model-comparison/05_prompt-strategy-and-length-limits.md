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
