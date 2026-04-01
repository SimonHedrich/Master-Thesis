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
| openimages | `data/supplementary_openimages/filter_results.jsonl` |

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
