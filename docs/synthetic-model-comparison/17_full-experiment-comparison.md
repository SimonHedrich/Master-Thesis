# Full comparison: all 12 trained cells + the prompt-length ablation, explained

**Created:** 2026-08-08
**Owner:** Simon Hedrich

## Scope and status

This doc consolidates everything docs [`14`](14_maxlen-cell-deep-comparison.md),
[`15`](15_all-cells-comparison-with-api-incumbent.md), and
[`16`](16_prompt-length-ablation.md) each covered separately into one place:
**every cell trained so far** (12 total — 6 local `maxlen` cells, the API
incumbent, and the 5 cells from the prompt-length ablation), one ranking,
and one explanation that ties the three earlier docs' findings together
— plus a new finding that only shows up once all 12 cells sit side by side
(see "The incumbent vs. its own cheaper tier" below).

**Same provisional caveat as every comparison in this experiment**: built
on `5-export_coco.py`'s best-effort MegaDetector export, not the
human-reviewed labels stages 3/4 would produce (TODO.md §3.2/§3.4,
deliberately deferred). Treat every ranking here as the best comparison
currently obtainable, not a final verdict.

## The headline ranking — all 12 cells

![All 12 cells ranked by mAP](../../reports/model_comparison_full_headline_map.png)

| Rank | Cell | Category | Seeds | mAP | Zebra confusion |
|---|---|---|---|---|---|
| 1 | `gpt-image-2-low` / full | API, full | 3 | 0.134 ± 0.014 | 0.253 |
| 2 | `gpt-image-2-medium` / full | API, full | 3 | 0.126 ± 0.011 | 0.228 |
| 3 | `gemini-3.1-flash-image-preview` / full (incumbent) | API, full | 3 | 0.105 ± 0.036 | 0.339 |
| 4 | `gemini-3.1-flash-lite-image` / full | API, full | 3 | 0.095 ± 0.006 | 0.218 |
| 5 | `gpt-image-2-low` / compressed | API, compressed | 3 | 0.074 ± 0.002 | 0.457 |
| 6 | `sd35-large` / maxlen | local | 2 | 0.055 ± 0.002 | 0.318 |
| 7 | `sd35m` / maxlen | local | 2 | 0.054 ± 0.001 | 0.472 |
| 8 | `gemini-3.1-flash-lite-image` / compressed | API, compressed | 3 | 0.049 ± 0.003 | 0.405 |
| 9 | `sd35-large-turbo` / maxlen | local | 2 | 0.043 ± 0.005 | 0.343 |
| 10 | `flux2-klein-9b` / maxlen | local | 2 | 0.037 ± 0.001 | 0.380 |
| 11 | `hidream-i1` / maxlen | local | 2 | 0.029 ± 0.002 | 0.683 |
| 12 | `realvisxl-lightning` / maxlen | local | 2 | 0.023 ± 0.001 | 0.628 |

Full metric vectors and per-class tables:
`reports/model_comparison_full_downstream_map.csv` /
`_per_class_ap.csv`.

**The top 5 of 12 cells are all API models — every local diffusion model
trails behind even the weakest API cell** (`gpt-image-2-low`/compressed,
rank 5, still beats every local cell). That single fact reframes the whole
experiment's local-vs-API question doc `01`/`13` set out to answer, subject
to the caveats below (every API cell is now at 3 seeds as of 2026-08-27 —
see the Update log — but local cells remain at 2; no `compressed`
counterpart for every local model; and still-provisional labels).

## Three questions this experiment set out to answer — now all resolved together

**1. Which local diffusion model is best, and does generation cost predict
quality?** (doc 14) The SD3.5 family wins the local grid (`sd35-large` ≈
`sd35m` > `sd35-large-turbo`), and generation cost does *not* predict
quality — the most expensive local model (`hidream-i1`, ~43h/cell) and the
cheapest (`realvisxl-lightning`, ~0.3h/cell) are the two weakest performers.

**2. Is the API incumbent's edge over the local cells about the model or
the prompt?** (doc 15 asked; doc 16 answered) Doc 15 found the incumbent's
lead concentrated in the zebra classes and flagged the comparison as
confounded — model *and* prompt regime differed at once. Doc 16's
controlled ablation (same model, only prompt length varies) showed
compressing the prompt cuts mAP by **45-48%** for both `gpt-image-2-low`
and `gemini-3.1-flash-lite-image` (both now at 3 seeds per regime) — a
large, model-independent effect. So:
**yes, it's substantially about the prompt.** The full-vs-compressed gap
(rank 1 → rank 5, rank 4 → rank 8 in the table above) is bigger than most
of the model-to-model gaps in this whole ranking.

**3. New, from this consolidated view — the incumbent vs. its own cheaper
tier, and a lesson about single-seed conclusions.** Neither doc 15 nor 16
directly compared `gemini-3.1-flash-image-preview` against
`gemini-3.1-flash-lite-image` *under the same prompt regime* — doc 15
compared the incumbent to a different prompt regime (`maxlen`), and doc
16's ablation only ever varied prompt length within one model at a time.
Sitting both `full`-regime gemini cells side by side here, for the first
time, gives a genuinely single-variable comparison: same prompt, different
model.

**With only 1 incumbent seed (as of 2026-08-08), the result looked
decisive**: Lite at 0.094 clearly beat the incumbent's single-seed 0.064.
**Two more incumbent seeds (seed43, seed44 — added 2026-08-26) overturned
that conclusion, and then partially reversed it again**: the three
incumbent seeds scored 0.064 / 0.116 / 0.134 — a genuinely wide spread —
pulling the 3-seed mean to **0.105 ± 0.036**, now *above* Lite's
0.095 ± 0.006 (also now at 3 seeds, and much tighter) rather than below
it. Given the incumbent's std is still ~6x wider than Lite's, the two
remain statistically indistinguishable — the honest reading is "roughly
tied, incumbent's point estimate currently higher," not "either one
clearly wins." **The original "Lite clearly beats the incumbent"
conclusion does not survive more seeds**, and the direction of the (still
noisy) gap has now flipped once already — a clear demonstration of why a
single training seed can't be trusted for a close-call comparison like
this one. Per-class, the picture is now mixed rather than one-sided: the
incumbent leads on `plains zebra` (0.280 vs. 0.256), `mountain zebra`
(0.194 vs. 0.111), and `grevy's zebra` (0.191 vs. 0.159), while Lite still
leads on `red fox` (0.201 vs. 0.173) and `lion` (0.074 vs. 0.054) —
consistent with two similar-strength models rather than one dominating
the other.

The same "cheaper tier does not mean worse" pattern shows up on the OpenAI
side too, and even more starkly on cost: `gpt-image-2-low/full` ($10.11 for
1,200 images) and `gpt-image-2-medium/full` ($29.36 — **2.9x the price**)
land statistically indistinguishable at 3 seeds each (0.134 ± 0.014 vs.
0.126 ± 0.011 — a 0.008 gap against a combined std of ~0.018, well within
noise), with `low`'s point estimate now *ahead* of `medium`'s for a third
of the price. Paying 3x more for the higher quality tier bought nothing
measurable here.

## Per-class breakdown

![Per-class AP heatmap, all 12 cells](../../reports/model_comparison_full_per_class_heatmap.png)

Full table: `reports/model_comparison_full_per_class_ap.csv`. The same
patterns from docs 14-16 hold at this larger scale:

- **Band D (data-rich) classes carry the ranking** — `american black bear`
  alone ranges from 0.072 (`realvisxl-lightning`) to 0.326
  (`gpt-image-2-low/full`), a 4.5x spread across the full 12-cell grid.
- **Band A (rare) classes stay in a noise floor for every cell** — every
  generator, every regime, `aye-aye`/`kinkajou`/`ringtail`/`saiga` sit
  under 0.05 AP regardless of how well that same cell does on Band D. This
  remains a resolution limit of the downstream-mAP axis, not a generator
  signal — axis A (the qualitative rubric, §3.4) is still the only way to
  compare generators on these classes, and it's still deferred.

## Confusion: the same detector-quality axis, now visible across all 12 cells

![Confusion, all 12 cells](../../reports/model_comparison_full_confusion.png)

Ranked in the same mAP order as the headline chart, zebra-group confusion
tracks mAP rank closely at both ends — the top 3 cells (`gpt-image-2-medium/full`,
`gpt-image-2-low/full`, `gemini-3.1-flash-lite-image/full`) have the 3
lowest confusion rates (0.21-0.25), and the bottom 2 (`hidream-i1`,
`realvisxl-lightning`) have the 2 highest (0.63-0.68). It isn't perfectly
monotonic in the middle (e.g. `sd35m` has middling mAP but the single
highest confusion rate outside the bottom two, 0.472) — but the broad
pattern first noted in doc 14 (confusion is downstream of general detector
quality, not an independent failure mode) still holds at full scale.

(`ursus` group confusion is excluded from this metric and chart: `american
black bear` is the only member of that look-alike group present in this
12-class subset, so the rate is structurally 0.000 for every cell by
construction — there is no other class it could be confused with, so it
carries no signal about generator quality. See doc 14 for the original
observation.)

## What this means, provisionally

If the goal is the best downstream detector, the evidence assembled across
docs 14-16 and this synthesis points the same direction on every axis
tested so far:

1. **Prompt detail matters more than which API model you pick** — the
   full-vs-compressed gap is larger than most model-to-model gaps in this
   ranking (§2 above).
2. **Cheaper tiers are not worse** — `gpt-image-2-low` ties
   `gpt-image-2-medium` at ~1/3 the cost, and `gemini-3.1-flash-lite-image`
   beats the incumbent it's supposed to be a lighter version of.
3. **`gpt-image-2` with the full prompt is the strongest synthetic-data
   source found in this experiment**, on either quality tier, and it's
   also comparatively inexpensive at the `low` tier ($10.11/1,200 images
   measured).
4. Local diffusion models, even the best of them, trail every API `full`
   cell tested — though the local grid intentionally used the
   length-capped `maxlen` regime throughout (doc 13), so this isn't a
   clean prompt-length-controlled comparison for the local side; it's
   consistent with, not proof of, finding #1 generalizing to local models.

## Limitations

- **Provisional labels** — same caveat as every comparison in this
  experiment (§3.4 deferred).
- **API seed counts now uniform at 3 (as of 2026-08-27), local cells still
  at 2** — the incumbent's original 1-seed estimate has been fully caught
  up (see the Update log for the full trajectory: 0.064 → 0.090 → 0.105 as
  seeds 43/44 landed). The incumbent's std (0.036 across 3 seeds) is still
  ~6x wider than any other API cell's even after matching everyone else's
  seed count — that variance looks like a genuine property of this cell,
  not an artifact of too few seeds, so treat its rank 3 position as the
  least stable ranking in this doc regardless. Local cells were not
  re-seeded as part of this round — still 2 seeds each, per doc 14.
- **Not a full factorial design** — only 2 of the 6+ local models and 2 of
  the ~4 API models have both `full` and `compressed` cells; the local
  grid has no `full`-regime cells at all (by design, per doc 13) and the
  API side has no `maxlen` cells. The prompt-length effect is demonstrated
  on 2 models, not all of them.
- **Real-only axis only** — per `docs/plans/2026-06-10_model-evaluation-strategy.md`,
  this experiment has no `mixed` test set defined, so (consistent with
  docs 14-16) everything here is the real-only axis.

## Update log

This doc is re-aggregated and edited after each new training seed lands
(script 9 regenerates the CSVs/charts in place; the prose below is updated
by hand to match). Terse, dated, newest last:

- **2026-08-08** — initial version. 12 cells, incumbent at 1 seed.
- **2026-08-26** — incumbent's `seed43` added (2 seeds). Mean map moved
  0.064 → 0.090 ± 0.037 (rank 5 → 4), now statistically indistinguishable
  from `gemini-3.1-flash-lite-image`/full (0.094 ± 0.009) given the
  incumbent's much wider variance. §3's "Lite clearly beats the incumbent"
  finding downgraded accordingly.
- **2026-08-26 (same day)** — incumbent's `seed44` added (3 seeds; hit and
  recovered from an orphaned/hung DataLoader worker mid-run — training
  itself completed fine, only the in-process test eval hung, worked around
  via the standalone `eval_suite/run_evaluation.py --run-dir` entrypoint on
  the already-saved `best.pt`). Third seed scored 0.134, pulling the mean
  to 0.105 ± 0.036 (rank 4 → 3) — now *above* Lite rather than below it,
  though still within noise. §3 updated to reflect the now-mixed per-class
  picture. A third seed (`seed44`) is being added to the remaining API
  cells next.
- **2026-08-26 (same day)** — `gpt-image-2-low/full`'s `seed44` added
  (3 seeds). Mean map moved 0.130 → 0.134 ± 0.014 (rank 2 → 1), edging
  ahead of `gpt-image-2-medium/full` (still 2 seeds, 0.132 ± 0.008) —
  reinforces rather than changes the "cheaper tier is not worse" finding.
- **2026-08-26 (same day)** — `gpt-image-2-medium/full`'s `seed44` added
  (3 seeds). Mean map moved 0.132 → 0.126 ± 0.011 — still statistically
  tied with `gpt-image-2-low/full`, still the same finding (paying 3x more
  bought nothing measurable).
- **2026-08-26 (same day)** — `gemini-3.1-flash-lite-image/full`'s `seed44`
  added (3 seeds). Mean map barely moved (0.094 → 0.095) but variance
  tightened sharply (std 0.009 → 0.006) — this cell's estimate was already
  stable; the incumbent's remains the noisy one. Every `full`-regime API
  cell now has 3 seeds; only the two `compressed` cells (2 seeds) remain.
- **2026-08-26 (same day)** — `gpt-image-2-low/compressed`'s `seed44`
  added (3 seeds). Mean map barely moved (0.075 → 0.074, std stayed tight
  at 0.002) — this cell's ranking (5th) and the prompt-length-ablation
  finding in §2 (45-49% drop vs. `full`) both hold. Zebra confusion ticked
  up slightly (0.414 → 0.457).
- **2026-08-27** — `gemini-3.1-flash-lite-image/compressed`'s `seed44`
  added (3 seeds), the last of the 7 planned runs. Mean map barely moved
  (0.048 → 0.049). **Every API cell now has 3 seeds** (up from a mix of 1
  and 2 at the start of this batch); the ablation percentage in §2
  refreshed to 45-48%. One run in this batch (the incumbent's `seed44`)
  hit an orphaned/deadlocked DataLoader worker mid-eval — training itself
  finished cleanly, only the eval step hung; recovered via the standalone
  `eval_suite/run_evaluation.py --run-dir` entrypoint on the already-saved
  checkpoint rather than re-training. Final state: the incumbent's 3-seed
  estimate (0.105 ± 0.036) is noticeably noisier than every other API
  cell's, which now looks like a real property of that cell rather than an
  artifact of too few seeds — see Limitations. Local `maxlen` cells remain
  at 2 seeds; this batch only re-seeded the API side.

## Reproducing this analysis

```
uv run python scripts/synthetic_model_comparison/9-compare_all_models.py
```

Aggregates all 12 cells' `evaluation_report.json` files (mean/std across
available seeds), writing `reports/model_comparison_full_downstream_map.csv`,
`reports/model_comparison_full_per_class_ap.csv`, and the three charts
above. No combined cost chart: local cells cost GPU-hours, API cells cost
$/image — see `6-compare_maxlen_cells.py`'s cost-vs-map chart (local only)
and doc 16 (API full-vs-compressed) for the cost angle within each axis
separately.
