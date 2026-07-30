# Selecting the Incumbent Synthetic Train Subset

**Date:** 2026-07-16
**Status:** Done — 7 of 12 classes selected and copied; the remaining 5
(fresh generation) are covered in the follow-up note in §6 below.
**Depends on:** [`01_experiment-design.md`](01_experiment-design.md) §5,
[`02_class-selection.md`](02_class-selection.md),
[`09_test-subset-build.md`](09_test-subset-build.md)

---

## 1. What this materializes

`01_experiment-design.md` §5 fixes **100 synthetic images/class** as the
train-set control for every generator cell. The **incumbent cell** (Nano
Banana 2 / `gemini-3.1-flash-image-preview`, `full` prompt regime) doesn't
need fresh generation for 7 of the 12 frozen classes — production already
generated images with that exact model and prompt template:

- **Grévy's zebra** (Bucket 1, Band B): exactly 100 images already exist —
  the fixed control count. Reused as-is, no selection needed.
- **The 6 Bucket-3 classes** (kinkajou, water deer, ringtail, saiga,
  aye-aye, pangolin family; Band A): 200 images each exist — production's
  full Band-A synthetic pool (160 train + 40 val). Double the needed count,
  so 100 of 200 had to be **selected**, chosen to preserve variety in pose,
  camera position, and environment despite halving the pool.

**Out of scope / follow-up:** the other 5 classes in the 12-class set
(plains zebra, mountain zebra, red fox, American black bear, lion) have
**no** existing incumbent synthetic images — Band D classes used real
images in production, not synthetic. These need fresh generation before
their incumbent-cell train images exist at all.

**Labeling is deliberately deferred.** MegaDetector will run once, later,
over every generator cell's images together (per
[`01_experiment-design.md`](01_experiment-design.md) §3's pipeline). This
step only selects and copies image files plus a provenance manifest — it
does not touch `data/synthetic/annotations_*.json` and produces no COCO json.

## 2. What the source data contains

Each `data/synthetic/prompts/<class>/<NNN>.txt` file is the exact prompt
used to generate `data/synthetic/images/band_a|band_b/<class>/<band-prefix>_<class>_<NNN>.png`
(1:1 correspondence via the numeric index; the basename is deterministically
`<band-prefix>_<class-slug>_<NNN>.png`, e.g. `a_kinkajou_001.png`,
`b_grevy_s_zebra_026.png`). Every prompt's `SCENE SPECIFICATION` block has
two free-text lines, extracted by regex:

- `Animal pose / behavior: ...`
- `Environment / background: ...`

For every one of the 6 Bucket-3 classes, the 200 images decompose into
**exactly 8 distinct environment strings × 25 images each**, and **20–23
distinct poses per 25-image environment group** (mostly singleton, a few
repeated twice). `data/synthetic/annotations_{train,val}.json` additionally
carry `shot_type`, `distance`, `lighting`, `occlusion` categorical tags per
image (production metadata; camera angle/position and lighting/occlusion
proxies). All 200 (100 for Grévy's zebra) images per class are accounted for
— no quarantined or failed generations in this pool (train+val counts sum
exactly to 200/100 per class, verified against the annotation files).

## 3. Selection algorithm (Bucket-3 classes only)

For each of the 6 classes:

1. Join the 200 prompt files (parsed pose + environment text) with the 200
   image records (`shot_type`, `distance`, `lighting`, `occlusion`,
   original production `train`/`val` split) via the shared numeric index.
2. **Stratify by environment** (8 groups of 25 images). Allocate a quota per
   group so quotas sum to 100: sort the 8 environment strings ascending,
   give the first 4 a quota of **13** and the remaining 4 a quota of **12**
   (100/8 = 12.5, largest-remainder rounding, deterministic tie-break by
   sorted string). This guarantees every environment stays represented at
   the same (or near-same) proportion after halving — no environment is
   dropped or over/under-weighted.
3. **Within each environment group, greedily maximize diversity**: track a
   "covered" set for each of 5 fields (`pose`, `shot_type`, `distance`,
   `lighting`, `occlusion`). Repeatedly pick the still-unpicked image whose
   field values add the most **new** values to those covered sets, ties
   broken by lowest original index (fully deterministic, no randomness
   anywhere, reproducible on rerun). Because each environment group has
   20–23 distinct poses against a quota of only 12–13, nearly every picked
   image ends up contributing a genuinely new pose, while the same greedy
   score spreads shot_type/distance/lighting/occlusion coverage as a side
   effect.
4. Concatenate the 8 groups' picks → 100 images for that class.

Grévy's zebra needs no algorithm — all 100 existing images are reused.

## 4. Live results

Run via `scripts/synthetic_model_comparison/1-select_train_subset_incumbent.py`
on 2026-07-16:

| Class | Band | Pool | Selected | Environments (of 8) | Distinct poses selected (of available) |
|---|---|---|---|---|---|
| grevy's zebra | B | 100 | 100 | 8/8 | 41/41 |
| kinkajou | A | 200 | 100 | 8/8 | 39/45 |
| water deer | A | 200 | 100 | 8/8 | 39/45 |
| ringtail | A | 200 | 100 | 8/8 | 39/45 |
| saiga | A | 200 | 100 | 8/8 | 40/45 |
| aye-aye | A | 200 | 100 | 8/8 | 40/45 |
| pangolin family | A | 200 | 100 | 8/8 | 54/65 |

**Total: 700 images, 362 MB.** Every environment survives the halving
(8/8 for every class); 87–90% of distinct poses survive for the 5 classes
with 45 available, and 83% for pangolin family (54/65) — the algorithm's
target of maximizing pose coverage under a per-environment quota well below
the available pose count is met. Full per-class numbers are frozen in
`reports/model_comparison_train_incumbent_selection.csv`.

Verification performed:
- File counts on disk match exactly: 700 total, 100 per class-directory.
- `index.jsonl` has 700 lines, all `file_name`s unique (no duplicate
  selections, no collisions — the script's collision guard never fired).
- 5 random spot-checks: each selected file exists, opens correctly via
  PIL, and its `index.jsonl` record's `pose`/`environment` text is verified
  to appear verbatim in its source prompt `.txt` file's scene-spec block.

## 5. Output layout

```
data/synthetic_model_comparison/
├── test/                                          # existing, see 09_test-subset-build.md
└── train/
    ├── gemini-3.1-flash-image-preview/            # incumbent model slug
    │   └── full/                                  # prompt regime
    │       ├── images/
    │       │   ├── grevy_s_zebra/                 # 100 images
    │       │   ├── kinkajou/                       # 100 images
    │       │   ├── water_deer/                     # 100 images
    │       │   ├── ringtail/                       # 100 images
    │       │   ├── saiga/                          # 100 images
    │       │   ├── aye_aye/                        # 100 images
    │       │   └── pangolin_family/                # 100 images
    │       └── index.jsonl                         # provenance manifest, one record/image
    └── prompts_full/                               # prompt text for the "full" regime,
        ├── grevy_s_zebra/<NNN>.txt                 #   shared across every generator that
        ├── kinkajou/<NNN>.txt                       #   runs it (only the model consuming
        ├── water_deer/<NNN>.txt                     #   the prompt varies, not the prompt
        ├── ringtail/<NNN>.txt                       #   itself) -- lives outside any single
        ├── saiga/<NNN>.txt                           #   generator's subdirectory
        ├── aye_aye/<NNN>.txt
        └── pangolin_family/<NNN>.txt
```

`prompts_full/<class_slug>/<NNN>.txt` is a verbatim copy of the exact prompt
used to generate the paired image (`data/synthetic/prompts/<class>/<NNN>.txt`),
copied only for the selected images (100/class), not the full 200-image
source pool for the Bucket-3 classes. The numeric stem matches the image's
index (e.g. `.../full/images/kinkajou/a_kinkajou_074.png` ↔
`prompts_full/kinkajou/074.txt`), so the pairing is visible on disk without
opening `index.jsonl`. It sits one level up from any single generator's
`<model>/<regime>/` subdirectory because prompt text for a given regime is
identical across generators — only the model consuming it differs — so a
future second `full`-regime generator cell reuses the same `prompts_full/`
rather than duplicating it.

`index.jsonl` mirrors `data/synthetic/index.jsonl`'s schema (`class`, `band`,
`shot_type`, `distance`, `lighting`, `occlusion`) plus the two new free-text
fields (`pose`, `environment`), the original production `source_split`
(`train`/`val` — provenance only, not functionally used here), the source
`prompt_file` path, the copied `dest_prompt_file` path (under
`train/prompts_full/`), and the copied `file_name`. Directory names follow
the existing `data/gbif/images/<slug>/`-style
convention (spaces → underscores).

Sibling `train/<other-generator>/<regime>/` subdirs will hold each future
cell's synthetic images once generated — those cells generate exactly 100
images/class directly (no pre-existing oversized pool), so this
stratify-then-greedy selection step is specific to reusing the incumbent's
legacy production images and won't be needed again.

## 6. Not built here (deliberately out of scope)

- **The other 5 classes** (plains zebra, mountain zebra, red fox, American
  black bear, lion): had no incumbent synthetic images. These have since
  been freshly generated (100/class, full prompt regime) via
  `scripts/synthetic_model_comparison/1b-generate_prompts_fresh.py` +
  `1c-generate_images_fresh.py` (Gemini Batch API), completing the
  incumbent cell at 1,200/1,200 images. Prompts reuse this experiment's
  cached LLM scene profiles (`reports/synthetic_scene_profiles.json`) and
  the same `BAND_B_SCHEDULE`-derived 100-shot schedule as production, so
  the shot-diversity method matches the 7 reused classes exactly.
- **Labeling / COCO annotations.** Per an explicit scope decision, bounding
  boxes are not carried over from the production annotation files, and
  MegaDetector is not re-run here. Labeling happens once, later, across all
  generator cells together.
- **A combined mixed real+synthetic training view.** Out of scope for
  training data by definition — per `CLAUDE.md` and
  `docs/plans/2026-06-10_model-evaluation-strategy.md`, synthetic images are
  train-only and never part of the evaluation set.
