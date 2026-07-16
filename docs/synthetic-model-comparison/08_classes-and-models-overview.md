# Overview: Output Classes and Model Grid

**Date:** 2026-07-16
**Status:** Condensed summary of [`02_class-selection.md`](02_class-selection.md),
[`01_experiment-design.md`](01_experiment-design.md) §6, and
[`05_prompt-strategy-and-length-limits.md`](05_prompt-strategy-and-length-limits.md)

---

## 1. Output classes (12)

| # | Class | Bucket | Why included |
|---|-------|--------|---------------|
| 1 | plains zebra | Look-alike + robust-test anchor | broad + shadow stripes; 2,075 real test images in this experiment (expanded — see [`02`](02_class-selection.md#4a-test-set-expansion-for-band-bcd-classes-this-experiment-only)) |
| 2 | Grévy's zebra | Look-alike | narrow dense stripes, white belly — the rare zebra |
| 3 | mountain zebra | Look-alike | gridiron rump pattern + dewlap |
| 4 | red fox | Robust-test anchor | ubiquitous, in pretraining — upper-baseline sanity check |
| 5 | American black bear | Robust-test anchor | large, common, well-represented |
| 6 | lion | Robust-test anchor | iconic — quality ceiling check |
| 7 | kinkajou | Rare + robust test | de-confounds rarity vs. test-set size |
| 8 | water deer | Rare + robust test | distinctive tusks, no antlers |
| 9 | ringtail | Rare + robust test | distinctive banded tail |
| 10 | saiga | Rare, test-limited | bizarre proboscis — textbook "not in pretraining" case |
| 11 | aye-aye | Rare, test-limited | extreme morphology stress test |
| 12 | pangolin family | Rare, test-limited | scaled body texture — very rare |

All 12 already have synthetic-test and real train/test images in the existing
split; only new **synthetic-train** images need generating per model.

## 2. Model grid: cost and prompt length

"Full" = the production ~1,300-word Wikipedia-description template (API models
only — their context windows are effectively unlimited: Gemini 32k tokens,
OpenAI 32k chars). "Compressed" = the identical ≤75-token / ≤55-word prompt
built for every model, including the 77-token-limited SDXL family.

| Model | Tier | Regime(s) run | ~Cost/image | Full prompt | Compressed prompt |
|-------|------|----------------|-------------|--------------|---------------------|
| Nano Banana 2 (`gemini-3.1-flash-image`) — **incumbent** | Google | full + compressed | $0.045–$0.15 | ~1,300 words | ≤75 tokens / ≤55 words |
| gpt-image-2 low | OpenAI, price floor | full + compressed | $0.006 | ~1,300 words | ≤75 tokens / ≤55 words |
| gpt-image-2 medium | OpenAI, mid-tier | full only | $0.053 | ~1,300 words | — |
| Nano Banana 2 Lite (`gemini-3.1-flash-lite-image`) | Google, price floor | full only | ~$0.034 | ~1,300 words | — |
| Nano Banana Pro (`gemini-3-pro-image`) | Google, ceiling | full only | $0.13–$0.24 | ~1,300 words | — |
| FLUX.1-schnell | Local (Apache-2.0) | compressed only | GPU time only | not run (256-token T5 ceiling) | ≤75 tokens / ≤55 words |
| RealVisXL V5.0 + SDXL-Lightning | Local (SDXL) | compressed only | GPU time only | not run (77-token hard ceiling = compressed) | ≤75 tokens / ≤55 words |
| SD 3.5 Medium *(optional)* | Local, no-quant baseline | compressed only | GPU time only | not run (256-token T5 ceiling) | ≤75 tokens / ≤55 words |

Notes:
- Only the incumbent and gpt-image-2 low get **both** regimes — the cheapest
  pair spanning the capability range, used to isolate the prompt-length effect
  from raw model quality (rationale: [`01`](01_experiment-design.md) §4b).
- Local models never see the full prompt: their hard ceilings (SDXL 77 tokens;
  FLUX.1-schnell/SD3.5-Medium 256 T5 tokens) are all shorter than the
  compressed prompt would need to be, so compressed *is* their effective max.
- API "full" prompts are token-billed and vary slightly by resolution/quality
  tier; figures above are the representative per-image price at the
  resolution used in production (~0.5K–1K, `4:3`). Batch APIs (Gemini, OpenAI)
  give −50% but are not reflected in the table.
