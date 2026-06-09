# Flag Cross-Species Contamination in Multi-Box Images

**Date:** 2026-06-09
**Status:** Plan only — no implementation yet
**Relates to:** `docs/plans/2026-06-08_fix-incomplete-gt-annotations.md` (the multi-box fix that created this problem), `scripts/dataset_quality/6-classify_speciesnet.py`, `scripts/dataset_quality/7-filter_speciesnet.py`, `scripts/evaluation/visualize_fiftyone.py`

---

## 1. Problem Statement

The GT-annotation fix (`2026-06-08`) made every confident MegaDetector detection
(conf ≥ 0.5) a ground-truth box, so multi-animal images now carry one box per
animal. The COCO JSONs now look correct on the box count axis:

| Split | Images | Annotations | Max boxes / image |
|---|---:|---:|---:|
| train | 145,764 | 178,924 | 28 |
| val   | 12,545  | 18,445  | 22 |
| test  | 63,822  | 86,785  | 28 |

But this exposed a **label-correctness** problem that the single-box export hid.
**Every box on an image inherits that image's single category** (the directory's
species, `expected_common`). When an image filed under species *X* actually
contains *X* **plus a different animal Y**, the box drawn around *Y* is now a
ground-truth annotation that says "*X*". That is a mislabeled training/eval box.

This was harmless when only the (presumably-correct) primary box was exported;
it becomes a real data-quality defect now that secondary boxes are emitted.

### Constraints from the user

- **No re-labeling by hand at scale** — 225 classes, tens of thousands of
  multi-box images. The flagging must be automatic.
- **Tolerance for look-alikes.** SpeciesNet is not fine-grained-accurate. An
  African elephant boxed inside an Asian-elephant image is **fine** (same family
  *Elephantidae*, different genus); different zebra species are **fine** (same
  genus *Equus*). Only a *genuinely different* animal (a zebra in an elephant
  image) should be flagged.
- **Keep the dataset stable.** The hope is that few images need removal, so class
  counts stay within their current bands (no need to source replacement images;
  small per-class deviations are acceptable).
- The output is a **review list** (new or existing file); a human then discards
  contaminated images or edits their boxes.

---

## 2. Key Enabler — SpeciesNet Already Classified *Every* Box

The infrastructure to solve this already exists and has **already been run on all
detections, not just the primary one**:

- `scripts/dataset_quality/6-classify_speciesnet.py` runs SpeciesNet on **every**
  MegaDetector crop in `detections[]` and writes
  `data/{source}/speciesnet_results.jsonl`. Each record has a
  `speciesnet_detections[]` array with one entry per `detection_idx`, carrying
  `bbox_norm`, `megadetector_conf`, `speciesnet_top1_idx`, `speciesnet_top1_score`.
- `scripts/dataset_quality/7-filter_speciesnet.py` already contains the exact
  taxonomic-tolerance machinery we need:
  - `load_speciesnet_labels()` — resolves `top1_idx` → `uuid;class;order;family;genus;species;common`.
  - `load_taxonomy()` + `load_classes_225()` — expected-class taxonomy lookup.
  - `_compute_match_level()` — returns one of
    `species → genus → family → order → class → no_match` for a prediction
    relative to an expected class.

Script 7 only ever evaluated the **primary** detection (`detection_idx == 0`,
which set the image's pass/fail and label). **The secondary detections were
classified and stored but never checked against the image's class.** Those stored
classifications are exactly what we need — no GPU re-run required for the flagging
logic itself (only a one-time label-string resolution, see §4.1).

### Linkage is trivial

COCO `file_name` is byte-identical to the `filepath` key in both
`filter_results.jsonl` and `speciesnet_results.jsonl`
(e.g. `data/openimages/images/sun_bear/oi_sun_bear_00002.jpg`). Detections are
ordered consistently by `detection_idx` across the two files. So:

```
COCO image.file_name  ==  speciesnet_results.filepath
COCO annotation.conf  ==  detections[i].conf  (carried through the multi-box fix)
```

The image's expected class = `expected_common` = COCO category name.

---

## 3. Why the Tolerance Band Is Essential — Scale Estimate

Counting images with ≥ 2 confident detections (MegaDetector conf ≥ 0.5,
SpeciesNet top-1 score ≥ 0.3), comparing each secondary box's SpeciesNet **class
index** to the primary's:

```
total images classified                                  465,130
images with >=2 confident detections                      57,205
  all secondary boxes SAME species-idx as primary          24,804
  >=1 secondary box with a DIFFERENT species-idx            32,401   (56.6%)
```

If we flagged on "different SpeciesNet index" we would queue **32,401 images** for
review — that destabilizes the dataset and is exactly the manual burden the user
wants to avoid. The overwhelming majority of those 32k are **genus/family
siblings** (different *index*, same animal-kind): the zebra-species and
elephant-genus cases. Collapsing predictions to a **taxonomic match level** and
forgiving species/genus/family matches is what reduces 32k to a manageable,
genuinely-contaminated subset. This is the core design decision.

---

## 4. Design

A new script `scripts/dataset_quality/14-flag_multi_animal_contamination.py`
(numbered after the existing pipeline; 12 = splits, 13 = cleanup).

### 4.1 Resolve SpeciesNet labels once (Docker), then run flagging anywhere

`data/speciesnet_classes.json` currently stores only integer indices
(`[0,1,2,...]`) — useless for taxonomy. The label strings live only inside the
SpeciesNet classifier (script 7 loads them via Docker).

**Step 0 (one-time, in `Dockerfile.speciesnet`):** dump the resolved
`idx → "uuid;class;order;family;genus;species;common"` map to
`data/speciesnet_labels.json`. Reuse `load_speciesnet_labels()` from script 7.
After this, the flagging script needs **no GPU and no Docker** and is cheap to
re-run with different thresholds — matching the script-6/script-7 philosophy
("script 7 can be re-run cheaply without touching script 6's output").

### 4.2 Per-box evaluation

For each source's `speciesnet_results.jsonl`, for each image:

1. Select **significant detections**: `speciesnet_skipped == False`,
   `megadetector_conf ≥ MD_CONF` (default 0.5, matches the GT export threshold
   `CONF_SIG`), so the set of evaluated boxes equals the set of exported GT boxes.
2. If fewer than 2 significant detections → skip (single-box images are unchanged
   and already passed the primary filter).
3. Resolve the image's expected class via `load_classes_225()` /
   `load_taxonomy()` (reuse script 7).
4. For **each** significant detection, build its predicted taxonomy from
   `speciesnet_top1_idx` and compute `match_level` against the expected class with
   the **existing `_compute_match_level()`** (imported, not reimplemented).

### 4.3 Tolerance band → flag rule

Map the match level to a verdict, encoding the user's examples directly:

| `match_level` of a secondary box | Meaning | Verdict |
|---|---|---|
| `species` | same species | **consistent** |
| `genus`   | e.g. different zebra species (*Equus*) | **consistent** |
| `family`  | e.g. African vs Asian elephant (*Elephantidae*) | **consistent** (default) |
| `order`   | different family, same order (zebra in elephant image: both… no — different order) | **flag** |
| `class`   | different order, both mammals | **flag** |
| `no_match`| not even the same class | **flag** |

So the **tolerance band = {species, genus, family}**; `{order, class, no_match}`
flags the image. `family` is the loosest tolerated level *because that is exactly
where the user's "looks similar" examples live* (elephants), and within-family
confusion is SpeciesNet's dominant error mode. The threshold is a CLI flag
(`--tolerance {genus,family,order}`, default `family`) so it can be tightened if
review shows family is too loose for some groups.

**Confidence gating to avoid false flags.** Only a *confident* mismatch flags the
image: a secondary box flags **only if** its `speciesnet_top1_score ≥ SN_SCORE`
(default 0.3, matching script 7's `--sn-score`) **and** its match level is outside
the band. A secondary box with a low SpeciesNet score is *uncertain*, not
*contradictory* — it goes to a separate low-priority "uncertain" list rather than
auto-flagging, keeping the review queue tight and the dataset stable.

**Optional cross-family allow-list.** A small curated set of
known-look-alike groups that are *not* the same family but that SpeciesNet (and
humans) confuse, kept in the script as a constant
(`LOOKALIKE_GROUPS: list[set[str]]`, keyed by family or genus). If both
expected and predicted fall in the same group, treat as consistent regardless of
match level. Start empty; populate only if review surfaces a recurring
false-positive pair. (The elephant case is already covered by family tolerance,
so this is a safety valve, not a day-one requirement.)

### 4.4 Flag granularity

A flag is **per image** (the unit of review), but each flagged record lists the
**offending box(es)**: `detection_idx`, `bbox`, predicted common name + score,
and match level. This lets the reviewer either delete the whole image or **delete
just the offending box** (the user's "edit the bounding boxes" option).

---

## 5. Outputs

Written to `reports/` alongside the existing review artifacts
(`manual_review_queue.{csv,md}`, `speciesnet_filter.md`):

1. `reports/multi_animal_contamination.csv` — one row per flagged **box**:
   `filepath, expected_class, detection_idx, bbox_norm, megadetector_conf,
   pred_common, pred_scientific, pred_top1_score, match_level, verdict`.
2. `reports/multi_animal_contamination_review.json` — one entry per flagged
   **image** (filepath → list of offending boxes + all-box summary). This is the
   machine-readable artifact the apply step (§7) and the FiftyOne review (§6)
   consume.
3. `reports/multi_animal_contamination.md` — human summary: counts by source, by
   split, by expected class, by match level; the top contaminated classes; and
   the **projected per-class image-count delta** if every flagged image were
   removed — so the user can confirm classes stay in-band *before* touching the
   data.

The script writes **no changes to any data file** (stats/report only), exactly
like scripts 7-phase-1 and 9.

---

## 6. Manual Review Workflow

The review surface should show, per image, the GT boxes **and** what SpeciesNet
thinks each box is. `scripts/evaluation/visualize_fiftyone.py` is the natural fit
and already loads COCO GT into FiftyOne:

1. Add a small mode/flag to the visualizer (or a sibling script) that loads
   **only the flagged images** from `multi_animal_contamination_review.json`,
   and attaches the per-box SpeciesNet prediction as a second detections field
   (`sn_prediction`) so each box shows expected-label vs predicted-species
   side-by-side.
2. The reviewer uses FiftyOne tags:
   - tag `discard` → remove the whole image, or
   - tag `edit` → box(es) need fixing (drop the offending box).
3. Export the tagged decisions back to a decisions file
   (`reports/multi_animal_contamination_decisions.json`).

This reuses the existing visualization stack rather than standing up new review
UI. (The existing `10-review_server.py` / `11-batch_review_server.py` are an
alternative if per-box tagging in FiftyOne proves awkward.)

---

## 7. Applying Decisions

A separate, explicit apply step
`scripts/dataset_quality/15-apply_contamination_decisions.py` consumes
`multi_animal_contamination_decisions.json` and rewrites the three COCO JSONs:

- `discard` image → drop the image entry **and** all its annotations.
- `edit` image → drop only the listed offending annotation id(s), keep the image
  and its correct boxes.
- Idempotent and atomic (write `.tmp`, then replace), mirroring script 7's write
  discipline. Prints a before/after per-class count diff and asserts no class
  leaves its tier band (fail loudly if it would, so the user decides).

Kept separate from flagging so the destructive step is deliberate and re-runnable.

---

## 8. Implementation Sequence

| Step | Action | Env |
|---|---|---|
| 0 | Dump `idx → label` map to `data/speciesnet_labels.json` (reuse `load_speciesnet_labels()`) | Docker (one-time) |
| 1 | Write `14-flag_multi_animal_contamination.py`; import `_compute_match_level`, taxonomy + 225-class loaders from script 7 | no Docker |
| 2 | Run flagging (stats only) → inspect `reports/multi_animal_contamination.md`; tune `--tolerance` / `--sn-score`; confirm projected per-class deltas stay in-band | no Docker |
| 3 | Add flagged-images mode to `visualize_fiftyone.py`; manually review; export decisions | no Docker |
| 4 | Write + run `15-apply_contamination_decisions.py` to patch the COCO JSONs | no Docker |
| 5 | Re-verify box distribution and re-launch FiftyOne to spot-check | no Docker |

Steps 0–2 are non-destructive and answer the central question — *how many images
are actually contaminated* — before any data is touched.

---

## 9. Verification

- After step 2: report shows flagged-image count ≪ 32,401 (the naive upper
  bound). If it is anywhere near 32k, the tolerance band is mis-set — investigate
  before reviewing.
- After step 4: per-class counts before/after; assert every class stays within
  its current band (Band A / Tier thresholds from
  `reports/class_distribution.csv`). Print any class with > N images removed for
  the user to eyeball.
- Re-run the box-distribution check from the `2026-06-08` plan to confirm the
  multi-box structure is intact (only contaminated boxes/images removed).
- Spot-check a sample of flagged images in FiftyOne to confirm the offending box
  really contains a different animal (precision of the flag), and a sample of
  *non*-flagged multi-box images to confirm genuine contamination is not slipping
  through (recall).

---

## 10. Open Questions

- **`family` vs `genus` tolerance default.** `family` is chosen so the elephant
  example passes. If review shows family is too loose for some pairs
  (e.g. domestic cat vs lion, both *Felidae*), tighten to `genus` globally and
  move the genuine look-alikes (elephants, etc.) into `LOOKALIKE_GROUPS`.
- **Low-SpeciesNet-confidence secondary boxes.** Default: do *not* auto-flag
  (uncertain ≠ wrong); list separately. Revisit if spot-checks show real
  contamination hiding in the low-confidence bucket.
- **Boxes SpeciesNet maps outside the 225-class set but to a real mammal.** A
  secondary box could be a valid different mammal that simply is not one of our
  225 classes — still contamination for *this* image's label. The match-level
  logic handles this via `no_match`; confirm it flags rather than silently passes.
- **Edit-vs-discard default.** When only one secondary box is contaminated and
  the rest are correct, prefer `edit` (drop the one box) to preserve the image and
  keep class counts stable — but leave the final call to the human reviewer.
```
