# Building the 12-Class Real Test Subset

**Date:** 2026-07-16
**Status:** Done — subset built and verified
**Depends on:** [`02_class-selection.md`](02_class-selection.md) §4a,
[`06_evaluation-methodology.md`](06_evaluation-methodology.md)

---

## 1. What this materializes

`02_class-selection.md` §4a defines a real test set specific to this
experiment, but only as a *rule* applied to `data/real/annotations_*.json` —
nothing had been copied out into its own artifact yet. This note tracks
turning that rule into a physical, frozen dataset:

- **Band A classes** (kinkajou, water deer, ringtail, saiga, aye-aye,
  pangolin family): the unmodified base `test` split only.
- **Band B/D classes** (Grévy's zebra, mountain zebra, plains zebra, red fox,
  American black bear, lion): `train + val + test` folded together, since
  this experiment never trains on real images, so the reserved train/val
  pools are otherwise idle.

Scope is **real images only**. The synthetic test images for these 12
classes already exist at `data/synthetic/images/test/<class>/` (50/class,
part of the project-wide balanced 225×50 synthetic test set) and are not
duplicated here — a combined "mixed" (real+synthetic) COCO json, if needed
later for the Axis C headline metric, is left as a separate follow-on step
(would need image-id offsetting per
`docs/plans/2026-06-10_model-evaluation-strategy.md`).

## 2. Live counts vs. the doc's table

Counts were recomputed directly from `data/real/annotations_{train,val,test}.json`
(distinct `image_id`s per category, verified with zero multi-species-image
overlap) rather than taken from `02_class-selection.md`'s table, per
CLAUDE.md's "run your own numbers" instruction and that doc's own stated
data-source policy. `annotations_test.json` has an mtime of 2026-05-31 —
before the class-selection doc was written (2026-07-14) — so this is not a
case of the dataset changing since the doc was drafted; the doc's own count
appears to have been off for several Band A classes.

| Class | Band | Splits pulled | Doc's count (§4a/Bucket 3) | Live count |
|---|---|---|---|---|
| plains zebra | D | train+val+test | 2,075 | 2,079 |
| grevy's zebra | B | train+val+test | 224 | 224 |
| mountain zebra | D | train+val+test | 467 | 467 |
| red fox | D | train+val+test | 2,149 | 2,150 |
| american black bear | D | train+val+test | 2,097 | 2,097 |
| lion | D | train+val+test | 2,097 | 2,097 |
| kinkajou | A | test only | 125 | 160 |
| water deer | A | test only | 133 | 151 |
| ringtail | A | test only | 123 | 186 |
| saiga | A | test only | 47 | 50 |
| aye-aye | A | test only | 29 | 29 |
| pangolin family | A | test only | 45 | 52 |

**Total: 9,742 images, ≈4.8 GB** (sum of `os.path.getsize` over every
resolved `file_name`). Band B/D deltas are small (0–3 images) and don't
change any conclusion in `02_class-selection.md`. Band A deltas are larger
(up to +63 for ringtail) but only make the rare-species real-test pool more
robust than the doc assumed — they don't undermine the doc's "test-limited"
framing for saiga/aye-aye/pangolin, which remain small in absolute terms.

## 3. Output layout

```
data/synthetic_model_comparison/
└── test/
    ├── images/
    │   ├── plains_zebra/
    │   ├── grevy's_zebra/
    │   ├── mountain_zebra/
    │   ├── red_fox/
    │   ├── american_black_bear/
    │   ├── lion/
    │   ├── kinkajou/
    │   ├── water_deer/
    │   ├── ringtail/
    │   ├── saiga/
    │   ├── aye-aye/
    │   └── pangolin_family/
    └── annotations_test.json
```

The test set lives under its own `test/` subdir (rather than directly under
`data/synthetic_model_comparison/`) so that per-generator, per-prompt-regime
synthetic **train** sets can later be added as sibling subdirs of the same
parent, e.g. `data/synthetic_model_comparison/train/<generator>/<regime>/`.

Class-slug directory names mirror the existing `data/gbif/images/<slug>/`
convention (spaces→underscores; apostrophes/hyphens kept as-is). Original
filenames are preserved on copy. `annotations_test.json` is standard COCO
(`info`/`licenses`/`categories`/`images`/`annotations`); category ids/names
are kept identical to the master 225-class taxonomy (not renumbered), so the
file stays compatible with `reports/lookalike_groups_v2.csv` and any
downstream eval tooling keyed on those ids (e.g. the zebra look-alike group,
id 173). Each image's original `band`/`source`/`split`/`quality_score`
fields are preserved as provenance — `split` in particular records which of
the base train/val/test pools an image came from, even though all of them
now serve as "test" for this experiment.

## 4. How it's built

`scripts/synthetic_model_comparison/0-build_test_subset.py`
(`uv run python scripts/synthetic_model_comparison/0-build_test_subset.py`):

1. Loads `data/real/annotations_{train,val,test}.json`.
2. For each of the 12 classes, selects the contributing split(s) per the
   band rule above and collects matching image + annotation records.
3. Copies each image file to
   `data/synthetic_model_comparison/test/images/<slug>/`, erroring out
   rather than silently overwriting on any unexpected filename collision
   across the merged splits.
4. Reassigns fresh sequential `image`/`annotation` ids across the combined
   set (the source train/val/test files each restart ids at 1, so they'd
   collide if concatenated as-is); category ids are left untouched.
5. Writes `data/synthetic_model_comparison/test/annotations_test.json` and
   the frozen class list `reports/model_comparison_classes.csv` (the freeze
   file `02_class-selection.md` §7 recommends before generating).

Also updated as part of this work: `data/STRUCTURE.md` and `docs/README.md`
(new directory/doc references).

## 5. Status

Run on 2026-07-16. Actual output matched the projected counts in §2 exactly
(no further deviation — the projections were themselves computed live from
the same source files).

- **9,742 images copied, 4.6 GB on disk** (`du -sh data/synthetic_model_comparison/`;
  differs slightly from the 4.8 GB estimated in §2, which summed raw
  `os.path.getsize` before filesystem block rounding/`shutil.copy2` metadata
  handling — not a discrepancy in which files were selected).
- `data/synthetic_model_comparison/test/annotations_test.json` loads cleanly
  via `pycocotools.coco.COCO`; `len(images) == len(annotations) == 9742`,
  `len(categories) == 12`, and every category id matches the master
  225-class taxonomy id (e.g. `plains zebra` → 166, `lion` → 132, `grevy's
  zebra` → 107).
- Per-class file counts on disk (`ls data/synthetic_model_comparison/test/images/<slug>/ | wc -l`)
  match the JSON's per-category image counts exactly for all 12 classes.
- Spot-checked 5 random images: files exist, `PIL`-read dimensions match the
  JSON's `width`/`height`, and each annotation's `bbox` lies within
  `[0, width] × [0, height]`.
- No filename collisions occurred across the merged train/val/test splits
  for Band B/D classes (the script's collision guard never fired).
- `reports/model_comparison_classes.csv` written with the frozen
  `class,band,exp_test_count` list.

**Update (same day):** the output was moved from
`data/synthetic_model_comparison/{images/,annotations_test.json}` to
`data/synthetic_model_comparison/test/{images/,annotations_test.json}` (the
script was updated and rerun rather than hand-patching paths), to leave room
for sibling `train/<generator>/<regime>/` synthetic-train subdirs under the
same parent once generation starts.

**Not built here (deliberately out of scope):** a combined "mixed"
real+synthetic COCO json. The synthetic test images for these 12 classes
already exist at `data/synthetic/images/test/<class>/` (50/class); merging
them with this real subset for the Axis C "mixed" headline metric is left as
a separate follow-on step, since it requires image/annotation-id offsetting
per `docs/plans/2026-06-10_model-evaluation-strategy.md` and wasn't part of
this request.
