# Automatic Image Quality Filtering

Script: `scripts/filter_dataset_quality.py`

Filters unusable images from all dataset sources before YOLO training assembly.
Each source produces `filter_results.jsonl` — one JSON line per on-disk image
with a pass/fail decision and a YOLO-format bounding box where available.

---

## Design rationale

The dataset comes from five sources with very different noise characteristics.
A single global filter would either over-reject clean sources or under-reject noisy ones.
The pipeline is therefore **source-aware** and **staged**: cheap checks run first,
expensive GPU inference only runs on images that survived the cheap checks.

### Why not a cloud VLM (Gemini / GPT-4o-mini)?

| Concern | Detail |
|---|---|
| Rate limits | Even paid Gemini 1.5 Flash tops out at ~4 000 req/min → >1 h for 250 k images |
| Cost unpredictability | Estimated ~$25 total, but re-runs cost the same and image-size pricing varies |
| Privacy / ToS | External APIs may not be used to generate training data for competing models |
| Network bandwidth | Uploading 200 k+ images is slow and brittle |
| Overkill | MegaDetector already answers the core question: *is there a real animal here?* |

### Why not a VLM on every image?

Florence-2 at ~100 ms/image × 250 k images = ~7 hours.
The majority of bad images (corrupted files, wrong format, blur, tiny resolution)
don't need semantic understanding — they should be caught by free checks before
the GPU is touched.

---

## Pipeline stages

Run the stages in order for each source.  Each stage is safe to re-run; it skips
images whose entry already contains a `stages_done` record for that stage.

```
metadata → heuristics → megadetector → [vlm, Wikimedia only] → report
```

### Stage 0 — metadata (no image I/O, no GPU)

Reads source-specific metadata files.  Emits `filter_results.jsonl` with one
entry per image **currently on disk**.

| Source | Filter criteria | Bbox source |
|---|---|---|
| `gbif` | `prediction_source == "classifier"`, `prediction_score ≥ 0.6`, ≥1 animal detection with `conf ≥ 0.5` and bbox area ≥ 1 % | MegaDetector v4 detections already in `SNPredictions_all.json` |
| `inaturalist` | `quality_grade == "research"` (community-verified by ≥2 observers) | None — added by megadetector stage |
| `wikimedia` | `min(width, height) ≥ 320 px` and `max/min ≤ 4.0` and `mime ∈ {jpeg, png}` — read from `metadata.csv`, no file open needed | None — added by megadetector stage |
| `lila_bc` | All on-disk images pass (camera-trap data, trusted source) | COCO-format ground-truth bboxes in `filtered_images_225.json` where `has_bbox == true`; remainder via megadetector |
| `openimages` | All on-disk images pass (professionally annotated) | From `metadata_catalog.csv` bbox column (`xmin,ymin,xmax,ymax` → converted to YOLO) |

iNaturalist note: `quality_grade` is resolved by streaming `condensed/photos.csv`
(6.8 M rows, tab-separated) to map `photo_id → observation_uuid`, then streaming
`condensed/observations.csv` (4 M rows) to map `observation_uuid → quality_grade`.
Only rows matching images on disk are kept in memory.

### Stage 1 — heuristics (CPU only)

Processes images that passed the metadata stage and have not yet been heuristic-checked.

| Check | Threshold | Rejection reason |
|---|---|---|
| Corruption / truncation | `PIL.Image.verify()` + `load()` | Unreadable or truncated file |
| Resolution | short side ≥ 320 px | Too small for detection training |
| Aspect ratio | long/short ≤ 4.0 | Extreme panorama or portrait strip |
| Grayscale | std of per-channel means ≥ 10 | Effectively monochrome |
| Blur | Laplacian variance of 512 px grayscale thumbnail ≥ 100 | Motion blur or out-of-focus |

Resolution and aspect ratio are already checked in the metadata stage for Wikimedia
(from `metadata.csv`); the heuristics stage re-checks on the actual pixel data and
also covers the remaining sources.

### Stage 2 — megadetector (GPU)

Uses **MegaDetector v5** via `pytorchwildlife` to detect animals in images that
survived heuristics and do not yet have a bbox.

```
pip install pytorchwildlife
```

The model weights (~600 MB) download automatically on first use.

**Skipped for** sources that already have reliable bboxes: `gbif` (MegaDetector
already run by SpeciesNet pipeline) and `lila_bc` entries with `has_bbox == true`.

Acceptance criteria:
- ≥1 animal detection (class 0 in PytorchWildlife's 0-indexed scheme) with `conf ≥ threshold`
- Bbox covers ≥ 1 % of the image area (rejects extreme close-ups of fur/texture)

Default threshold: `0.6`.  Tune with `--conf` flag; see Verification below for
guidance on choosing a threshold.

What MegaDetector catches beyond the heuristics:
- Drawings and illustrations (MegaDetector does not detect non-photographic animals)
- Taxidermy specimens (static, unnatural posture → low or no detection)
- Extreme close-ups with no full-body view
- Landscape / habitat photos with no visible animal

### Stage 3 — vlm, Wikimedia only (GPU, optional)

**Florence-2-large** (`microsoft/Florence-2-large`) runs a `<DETAILED_CAPTION>` task
on Wikimedia images that **passed heuristics but failed MegaDetector**.
Goal: rescue real photographs that MegaDetector missed (unusual lighting, heavy
occlusion, underwater, etc.).

If the caption contains none of the rejection keywords below, the image is
**rescued** (marked `passed = true`) so it can enter the YOLO assembly step.
If a keyword is found, the metadata-stage rejection is confirmed.

Rejection keywords: `drawing`, `illustration`, `painting`, `sketch`, `watercolor`,
`sculpture`, `figurine`, `statue`, `diagram`, `taxidermy`, `stuffed animal`,
`museum specimen`, `plush`, `rendering`, `3d model`, `cartoon`

```
pip install transformers timm einops
```

Expected scope: ~10–20 % of Wikimedia images reach Stage 3 (those that pass
heuristics but have no confident MegaDetector detection).  At ~150 ms/image on
an RTX 3060, that is 2–4 hours for the full Wikimedia corpus.  If the rescue
rate turns out to be < 5 %, skip this stage.

---

## Output format

Each source writes `filter_results.jsonl` next to its data directory:

| Source | Path |
|---|---|
| gbif | `resources/GBIFImages/filter_results.jsonl` |
| inaturalist | `data/inaturalist/filter_results.jsonl` |
| wikimedia | `data/wikimedia/filter_results.jsonl` |
| lila_bc | `data/lila_bc/filter_results.jsonl` |
| openimages | `data/openimages/filter_results.jsonl` |

Line format:

```json
{
  "filepath":    "data/wikimedia/images/cheetah/File_001.jpg",
  "passed":      true,
  "stage_failed": null,
  "reason":      null,
  "bbox":        [0.512, 0.438, 0.340, 0.612],
  "bbox_conf":   0.91,
  "stages_done": ["metadata", "heuristics", "megadetector"]
}
```

- `bbox` is in **YOLO normalised format**: `[x_center, y_center, width, height]`
- `bbox_conf` is `1.0` for ground-truth annotations (LILA BC, Open Images) and the
  detector confidence for MegaDetector-derived bboxes
- `stages_done` tracks which stages have been applied; stages skip entries already
  in their list, making all stages **safely re-runnable and resumable**

---

## Usage

```bash
# Process a single source end-to-end
python scripts/filter_dataset_quality.py metadata     --source wikimedia
python scripts/filter_dataset_quality.py heuristics   --source wikimedia
python scripts/filter_dataset_quality.py megadetector --source wikimedia --batch-size 16 --conf 0.6
python scripts/filter_dataset_quality.py vlm          --source wikimedia   # optional
python scripts/filter_dataset_quality.py report       --source all

# Makefile shortcuts (set SOURCE= to target a single source)
make filter-metadata    SOURCE=wikimedia
make filter-heuristics  SOURCE=wikimedia
make filter-megadetector SOURCE=wikimedia
make filter-report
```

---

## Standalone Caption Generation

`scripts/dataset_quality/4-generate_captions.py` generates a `<DETAILED_CAPTION>`
description for every image that survived the filter pipeline (`passed = true`).
Captions are written back into `filter_results.jsonl` as a new `caption` field and are
intended for downstream use in training data analysis, retrieval, and KD soft-label
pipelines.

**Model:** `microsoft/Florence-2-base` — **Batch size:** 6 — **Image size:** 768 × 768 px — **Decoding:** greedy (`num_beams=1`)

The script is **resumable**: entries that already have a `caption` field are skipped.
Results are flushed to disk every 500 images so an interrupted run loses at most one
flush interval of work.

```bash
# Single source
python scripts/dataset_quality/4-generate_captions.py --source wikimedia

# All sources in sequence (model loaded once)
python scripts/dataset_quality/4-generate_captions.py --source all
```

Additional `filter_results.jsonl` fields written by this script:

| Field | Type | Meaning |
|-------|------|---------|
| `caption` | string | Florence-2 detailed caption for passed images |
| `caption_error` | string | Error message when an image could not be opened |

Note: `vlm_caption` (written by Stage 3 of `1-filter_dataset_quality.py`) is a separate
field on *failed* Wikimedia borderline entries used for the filter rescue decision.
Do not confuse it with `caption`.

Requirements:
```
pip install transformers==4.48.3 timm einops pillow tqdm torch
```
(transformers 5.x breaks Florence-2 remote code)

---

## Caption-Based LLM Evaluation

`scripts/dataset_quality/5-evaluate_captions.py` applies an LLM to the captions
generated by Script 4, making a binary pass/fail decision against three criteria
that MegaDetector and heuristics cannot reliably detect from pixel data alone:

1. **Real, living animal** — rejects paintings, drawings, figurines, taxidermy,
   fossils, and footprints
2. **Quality photograph** — rejects camera-trap, night-vision, IR, and thermal images
3. **Whole animal visible** — rejects head-only close-ups and isolated body parts

Rejected images get `passed=false`, `stage_failed="caption_eval"`, and the LLM's
one-sentence reason — identical to how the megadetector and vlm stages work. All
evaluated entries receive a `caption_eval` field regardless of outcome.

### Backends

| Backend | Model | Est. runtime (543 k images) | Notes |
|---------|-------|-----------------------------|-------|
| `vllm` (default) | `Qwen/Qwen2.5-7B-Instruct-AWQ` | 12–18 h | Overnight; best quality |
| `vllm --model Qwen/Qwen2.5-3B-Instruct-AWQ` | Qwen2.5-3B | 8–10 h | Faster, marginal quality loss |
| `openrouter` | `qwen/qwen-2.5-7b-instruct` | 30–90 min | ~$5 for 543 k; needs API key |

Structured output is enforced via `GuidedDecodingParams(json=schema)` (vLLM/XGrammar)
or `response_format={"type": "json_object"}` (OpenRouter), eliminating JSON parse errors.
Evaluation failures (network errors, model load issues) are soft-passed with a
`"eval error — soft pass"` reason so infrastructure problems never silently reject good images.

### Usage

```bash
# vLLM, all sources — recommended overnight run:
python scripts/dataset_quality/5-evaluate_captions.py --source all

# Faster 3B model:
python scripts/dataset_quality/5-evaluate_captions.py --source all \
    --model Qwen/Qwen2.5-3B-Instruct-AWQ

# OpenRouter — set OPENROUTER_API_KEY in .env:
python scripts/dataset_quality/5-evaluate_captions.py --source all \
    --backend openrouter --concurrency 60

# Re-run evaluation from scratch:
python scripts/dataset_quality/5-evaluate_captions.py --source wikimedia --force
```

### Additional `filter_results.jsonl` fields written by this script

| Field | Type | Written on | Meaning |
|-------|------|-----------|---------|
| `caption_eval` | `{"pass": bool, "reason": str}` | all evaluated entries | LLM pass/fail decision |
| `stages_done` | list | all evaluated entries | `"caption_eval"` appended |
| `passed` | bool | LLM-rejected entries | set to `false` |
| `stage_failed` | str | LLM-rejected entries | `"caption_eval"` |
| `reason` | str | LLM-rejected entries | LLM's one-sentence rejection reason |

Requirements:
```
vllm backend:       pip install vllm
openrouter backend: pip install aiohttp python-dotenv
```

---

## SpeciesNet Classification

`scripts/dataset_quality/6-classify_speciesnet.py` runs SpeciesNet's EfficientNetV2-M
classifier on every MegaDetector animal crop from images that completed the `caption_eval`
stage (`passed=true` and `"caption_eval"` in `stages_done`).

**Purpose: pure data capture.** No filtering decisions are made here. Script 7
(`7-filter_speciesnet.py`) handles threshold-based pass/fail logic and can be re-run
with different thresholds without touching the expensive GPU inference.

### What it does

For each qualifying image:
1. Opens the image (PIL) to obtain pixel dimensions
2. For each animal detection in `filter_results.jsonl["detections"]`:
   - Skips detections below the MegaDetector confidence floor (default: 0.1)
   - Skips detections whose crop dimensions are smaller than `--min-crop` (default: 32 px)
   - Otherwise runs SpeciesNet's EfficientNetV2-M classifier directly on the crop
     (bypasses SpeciesNet's internal MegaDetector stage using the cached bbox)
3. Saves per-detection scores and top-1 prediction to `data/{source}/speciesnet_results.jsonl`

Image loading and crop preprocessing run in background threads (default: 8 workers) while
the main thread dispatches GPU batches (default: 128 crops per forward pass), giving
near-full GPU utilisation on a multi-core machine.

### Outputs

| File | Description |
|---|---|
| `data/{source}/speciesnet_results.jsonl` | One record per image; per-detection scores and top-1 |
| `data/speciesnet_classes.json` | Ordered SpeciesNet class indices (written once) |

Per-image record structure:
```json
{
  "filepath": "data/gbif/images/aardvark/img.jpg",
  "expected_common": "aardvark",
  "speciesnet_detections": [
    {
      "detection_idx": 0,
      "bbox_norm": [0.916, 0.358, 0.168, 0.348],
      "megadetector_conf": 0.937,
      "speciesnet_scores": {"1291": 0.872, "1787": 0.040, ...},
      "speciesnet_top1_idx": 1291,
      "speciesnet_top1": 1291,
      "speciesnet_top1_score": 0.872,
      "crop_size_px": [2264, 1189],
      "speciesnet_skipped": false,
      "skip_reason": null,
      "inference_ms": 76.3
    }
  ],
  "n_animal_detections": 1,
  "inference_total_ms": 91.4
}
```

Skipped detections (low confidence, crop too small, preprocess failure) are recorded with
`speciesnet_skipped: true` and a `skip_reason` string so the full detection history is
preserved for analysis.

### Key design choices

- **Compact integer indices instead of label strings.** Storing the full
  `"uuid;mammalia;carnivora;…"` string for every detection across 400 k+ images would
  add tens of GB. Instead, `speciesnet_scores` stores a sparse dict
  `{str(class_idx): probability}` and `speciesnet_top1_idx` stores the integer index.
  Script 7 resolves these indices to label strings by loading the SpeciesNet classifier
  at startup.
- **Sparse score storage.** Only probabilities ≥ `--min-score` (default: 0.01) are
  retained — on average ~4 classes per detection. The top-1 result is always stored
  separately. This gives ~600× smaller output vs. the full 2498-element vector while
  preserving enough mass for the 225-class probability projection in Script 7.
- **Geofencing disabled.** `SpeciesNet(geofence=False)` returns raw classifier
  probabilities with no geographic post-processing, consistent with the project's
  geo-filtering policy (applied post-hoc at inference, not during data curation).
- **Person/vehicle detections not available.** `filter_results.jsonl["detections"]`
  stores only animal detections (MegaDetector category `"1"`) — earlier pipeline stages
  dropped person and vehicle detections. Human co-occurrence tracking (`has_human`,
  `n_person_detections` as described in the strategy doc) is therefore not populated.
- **Resumable.** Already-classified images are tracked by filepath; interrupted runs
  continue from where they left off. Results are flushed every 100 images (`FLUSH_EVERY`).
- **OpenImages limitation.** OpenImages entries have no MegaDetector detections in
  `filter_results.jsonl` (pre-annotated bboxes were stored differently in the metadata
  stage). Script 6 produced 7,405 zero-detection records for OpenImages; these are
  expected and handled by Script 7 as `"no_animal_detection"` failures.

Full design rationale: `docs/plans/2026-04-30_speciesnet-classification-strategy.md`

### Classification results (as of 2026-05)

| Source | Records | Classified | Notes |
|---|---|---|---|
| gbif | 39,388 | 39,388 | All classified |
| inaturalist | 400,067 | 400,067 | 5,887 with ≥1 skipped detection |
| wikimedia | 12,486 | 12,486 | 5 with ≥1 skipped detection |
| images_cv | 5,501 | 5,501 | 9 with ≥1 skipped detection |
| openimages | 7,499 | 94 | 7,405 have no MegaDetector detections |

### Usage

```bash
# Single source (run inside Dockerfile.speciesnet)
python scripts/dataset_quality/6-classify_speciesnet.py --source gbif

# All sources in sequence
python scripts/dataset_quality/6-classify_speciesnet.py --source all

# Force re-run from scratch
python scripts/dataset_quality/6-classify_speciesnet.py --source inaturalist --force

# Re-encode existing full-vector files to sparse format (no GPU needed):
python scripts/dataset_quality/6-classify_speciesnet.py --migrate-scores all
```

Requirements: run inside `Dockerfile.speciesnet` (Python 3.11, `speciesnet` package).
```bash
make speciesnet-build   # build the image once
make speciesnet-start   # opens a bash shell inside the container
```

---

## SpeciesNet Filtering and 225-Class Mapping

`scripts/dataset_quality/7-filter_speciesnet.py` reads the raw classification output from
Script 6, applies the filtering rules from the strategy document, and optionally merges
results back into `filter_results.jsonl`.

**Two-phase design:**
- **Phase 1 (default):** statistics-only dry run — computes all pass/fail decisions and
  prints per-source and per-class summaries. No files are written.
- **Phase 2 (`--write`):** after validating the statistics look sensible, merges a
  `speciesnet_eval` block into `filter_results.jsonl` and updates `passed`/`stage_failed`.

### What it does

At startup, loads three reference tables once:

| Table | Source | Purpose |
|---|---|---|
| `idx_to_label` | SpeciesNet classifier (`clf.labels`) | Resolve integer indices → `"uuid;mammalia;…"` strings |
| `sn_taxonomy` | `resources/speciesnet_taxonomy_release.txt` | Look up family/order/class for expected species |
| `class225` | `reports/classes_225.csv` | Map predicted genus/species/family → 225-class index |

For each record in `speciesnet_results.jsonl`:

1. **Failure gates** (in order):
   - No animal detections → `"no_animal_detection"`
   - Primary detection skipped (crop too small) → `"primary_crop_too_small"`
   - Primary MegaDetector confidence < 0.5 → `"low_megadetector_confidence"`
   - SpeciesNet top-1 score < 0.3 → `"low_speciesnet_confidence"`

2. **Hierarchical match level** — compares the SpeciesNet top-1 prediction against the
   directory label at each taxonomic rank:

   | `match_level` | Condition | Default action |
   |---|---|---|
   | `species` | genus + species match (or best possible for genus/family-level classes) | PASS |
   | `genus` | same genus, different species | PASS |
   | `family` | same family, score < 0.5 | PASS (uncertain, trust directory label) |
   | `family` | same family, score ≥ 0.5 | FAIL `"family_mismatch_high_confidence"` |
   | `order` | same order | FAIL |
   | `class` | both Mammalia | FAIL |
   | `no_match` | otherwise | FAIL |

   The family-match confidence threshold encodes the reasoning from the strategy doc:
   a high-confidence family-level mismatch means SpeciesNet is *sure* it sees a different
   genus — that is not a labelling ambiguity but a different animal.

3. **225-class probability vector** — projects the sparse `speciesnet_scores` dict
   onto a 225-element float vector. Lookup priority per SpeciesNet class:
   species → genus → family (the 12 family-level classes in `classes_225.csv` such as
   `cricetidae`, `otariidae`, `sciuridae` are handled via a dedicated `family_to_225`
   lookup). `prob_225_sum` is the total probability mass mapped to any 225 class and
   serves as an out-of-distribution diagnostic (≈ 0.0 means the image is entirely
   outside the 225-class universe).

4. **Multi-animal flag.** `multi_animal: true` when `n_animal_detections > 1`. These
   images are not filtered — they provide richer soft labels for knowledge distillation
   because the teacher classifies all visible animals, not just the labeled subject.
   See strategy doc §Issue 2 for the KD research rationale.

### Label resolution: why Script 7 loads SpeciesNet

Script 6 stores compact integer class indices rather than full label strings to keep
`speciesnet_results.jsonl` file sizes manageable (the 400 k-record inaturalist file is
~306 MB; storing full strings would multiply this). Script 7 therefore needs to resolve
those indices back to `"uuid;mammalia;carnivora;…"` strings. The only authoritative
mapping is the SpeciesNet classifier's internal label list, accessed at startup via
`dict(clf.labels)`. This means Script 7 must also run inside `Dockerfile.speciesnet`.

### Output fields added to `filter_results.jsonl` (Phase 2, `--write`)

```json
{
  "speciesnet_eval": {
    "pass": true,
    "reason": null,
    "primary_detection": {
      "detection_idx": 0,
      "megadetector_conf": 0.937,
      "speciesnet_top1_idx": 1291,
      "speciesnet_top1_label": "uuid;mammalia;tubulidentata;orycteropodidae;orycteropus;afer;aardvark",
      "speciesnet_top1_scientific": "orycteropus afer",
      "speciesnet_top1_common": "aardvark",
      "speciesnet_top1_score": 0.872,
      "match_level": "species",
      "matched_class_225_common": "orycteropus afer",
      "matched_class_225_idx": 0,
      "probs_225": [0.872, 0.0, ...],
      "prob_225_sum": 0.89
    },
    "n_animal_detections": 2,
    "multi_animal": true
  }
}
```

For rejected images, the top-level `passed`, `stage_failed`, `reason`, and `stages_done`
fields are also updated, matching the convention used by earlier pipeline stages.

### Usage

```bash
# Phase 1: statistics only — run inside Dockerfile.speciesnet
python scripts/dataset_quality/7-filter_speciesnet.py --source gbif
python scripts/dataset_quality/7-filter_speciesnet.py --source all

# Phase 2: write results after validating statistics
python scripts/dataset_quality/7-filter_speciesnet.py --source all --write

# Adjust thresholds (all have defaults matching the strategy doc):
python scripts/dataset_quality/7-filter_speciesnet.py --source gbif \
    --md-conf 0.4 --sn-score 0.25 --family-fail-thresh 0.6
```

Requirements: run inside `Dockerfile.speciesnet` (same container as Script 6).

### Threshold rationale

| Parameter | Default | Reasoning |
|---|---|---|
| `--md-conf` | 0.5 | Below 0.5, MegaDetector is uncertain enough that the crop may not be a clean animal view. The lower 0.1 floor in Script 6 was for capture completeness; 0.5 is the filter threshold. |
| `--sn-score` | 0.3 | A top-1 score below 0.3 provides little taxonomic signal — the classifier is effectively uniform over the top candidates. |
| `--family-fail-thresh` | 0.5 | At ≥ 0.5, SpeciesNet is confident enough in a *different genus* that the family-level mismatch is more likely a labelling error than visual ambiguity between genera. |

All thresholds can be adjusted without re-running Script 6. The `match_level` field in
the output preserves the raw taxonomic comparison so thresholds can be revised post-hoc
by re-running Script 7 alone.

---

## Verification

1. **Calibrate on GBIF** — GBIF already has MegaDetector results in `SNPredictions_all.json`.
   Run the metadata stage and compare rejected images against entries with
   `prediction_source == "blank"` in the JSON; rejection sets should largely overlap.

2. **Threshold sensitivity for MegaDetector** — before committing to `--conf 0.6`,
   run on a 1 000-image Wikimedia sample with thresholds 0.4 / 0.5 / 0.6 / 0.7 and
   inspect 20–30 rejected images per threshold to find the operating point that
   discards drawings without losing real photographs.

3. **Spot-check Wikimedia rejections** — after the megadetector stage, randomly
   sample 50 failed Wikimedia images and view them.  Unexpected real-photo rejections
   indicate the threshold is too aggressive; adjust `--conf` or run the VLM stage.

4. **Per-class coverage** — after all stages, run `report --source all` and check
   that no target class drops to 0 images.  A class going to 0 after filtering
   signals a systematic issue (e.g., all images for that class were scraped from
   non-photographic categories).
