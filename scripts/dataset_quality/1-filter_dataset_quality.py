"""Filter dataset images for quality across multiple sources.

Implements a staged funnel (run each in order):

  metadata      — source-specific metadata pre-filtering; produces the initial
                  filter_results.jsonl for all images currently on disk
  heuristics    — fast per-image checks: corruption, resolution, blur, grayscale
  megadetector  — MegaDetector v5 inference for animal detection + bbox generation
  vlm           — Florence-2 semantic rescue for Wikimedia borderline cases only
  report        — print per-stage rejection statistics

Each stage reads data/{source}/filter_results.jsonl, updates entries that
survived all previous stages, and writes the file back.  Stages track progress
via a ``stages_done`` list per entry so they can be safely re-run or resumed.

Output line format (one JSON object per line):
  {
    "filepath":    "relative/path/from/repo/root.jpg",
    "passed":      true,
    "stage_failed": null,
    "reason":      null,
    "detections":  [{"bbox": [xc, yc, w, h], "conf": 0.87}, ...],  # YOLO normalized, sorted by conf desc; [] if none
    "stages_done": ["metadata", "heuristics"]
  }

Source overview:
  gbif        — data/gbif/images/  (MegaDetector bboxes in SNPredictions_all.json)
  inaturalist — data/inaturalist/images/       (quality_grade in observations.csv)
  wikimedia   — data/wikimedia/images/         (dimensions/mime in metadata.csv)
  openimages  — data/openimages/images/               (bboxes in metadata_catalog.csv)
  images_cv   — data/images_cv/images/         (manually curated CV images; all pass metadata)

Usage:
    python scripts/dataset_quality/1-filter_dataset_quality.py metadata    --source wikimedia
    python scripts/dataset_quality/1-filter_dataset_quality.py heuristics  --source wikimedia
    python scripts/dataset_quality/1-filter_dataset_quality.py megadetector --source wikimedia [--batch-size 16] [--conf 0.6]
    python scripts/dataset_quality/1-filter_dataset_quality.py vlm         --source wikimedia
    python scripts/dataset_quality/1-filter_dataset_quality.py report      --source all

Requirements (base):
    pip install pillow opencv-python-headless tqdm numpy

Additional per stage:
    megadetector: pip install pytorchwildlife
    vlm:          pip install transformers timm einops
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile, UnidentifiedImageError
from tqdm import tqdm

# Allow Pillow to attempt loading—we want controlled failure, not silent truncation
ImageFile.LOAD_TRUNCATED_IMAGES = False

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# ── Per-source paths ──────────────────────────────────────────────────────────

GBIF_IMAGES_DIR       = REPO_ROOT / "data" / "gbif" / "images"
GBIF_PREDICTIONS_JSON = REPO_ROOT / "data" / "gbif" / "metadata" / "SNPredictions_all.json"

INAT_IMAGES_DIR       = REPO_ROOT / "data" / "inaturalist" / "images"
INAT_PHOTOS_CSV       = REPO_ROOT / "data" / "inaturalist" / "metadata" / "photos.csv"
INAT_OBS_CSV          = REPO_ROOT / "data" / "inaturalist" / "metadata" / "observations.csv"

WIKI_IMAGES_DIR       = REPO_ROOT / "data" / "wikimedia" / "images"
WIKI_METADATA_CSV     = REPO_ROOT / "data" / "wikimedia" / "metadata.csv"

OI_DIR                = REPO_ROOT / "data" / "openimages"
OI_IMAGES_DIR         = OI_DIR / "images"
OI_CATALOG_CSV        = OI_DIR / "metadata_catalog.csv"

CV_IMAGES_DIR         = REPO_ROOT / "data" / "images_cv" / "images"

# filter_results.jsonl lives next to the images dir for each source
RESULTS_PATHS = {
    "gbif":        REPO_ROOT / "data" / "gbif" / "filter_results.jsonl",
    "inaturalist": REPO_ROOT / "data" / "inaturalist" / "filter_results.jsonl",
    "wikimedia":   REPO_ROOT / "data" / "wikimedia" / "filter_results.jsonl",
    "openimages":  REPO_ROOT / "data" / "openimages" / "filter_results.jsonl",
    "images_cv":   REPO_ROOT / "data" / "images_cv" / "filter_results.jsonl",
}

# ── Thresholds ────────────────────────────────────────────────────────────────

MIN_RESOLUTION     = 256    # pixels — shorter side (iNat medium landscape shots land at ~281px)
MAX_ASPECT_RATIO   = 4.0    # long / short side
BLUR_THRESHOLD     = 100.0  # Laplacian variance; below this → blurry
GRAYSCALE_STDEV    = 10.0   # std of per-channel means; below this → grayscale

GBIF_MIN_SCORE             = 0.6    # SpeciesNet prediction_score minimum
GBIF_ANIMAL_CONF           = 0.5    # Min confidence for an image to pass (primary detection)
GBIF_ANIMAL_CONF_SECONDARY = 0.2    # Min confidence for secondary detections to be recorded
GBIF_MIN_BBOX_AREA         = 0.01   # Minimum animal bbox fractional area (1 %)

MD_CONF_PASS       = 0.5    # Image passes if any detection ≥ this threshold
MD_CONF_SECONDARY  = 0.2    # Lower bound for secondary detections recorded alongside a passing image
MD_BBOX_MIN_AREA   = 0.01   # Minimum animal bbox fractional area after MD inference

INAT_ACCEPTED_GRADES = {"research", "needs_id"}  # needs_id = real photo, community ID pending

VALID_MIMES = {"image/jpeg", "image/png"}

VLM_REJECT_KEYWORDS = {
    "drawing", "illustration", "painting", "sketch", "watercolor", "artwork",
    "sculpture", "figurine", "statue", "diagram", "taxidermy", "stuffed animal",
    "museum specimen", "plush", "rendering", "3d model", "cartoon",
}

# ── JSONL utilities ───────────────────────────────────────────────────────────

def load_results(jsonl_path: Path) -> list:
    if not jsonl_path.exists():
        return []
    entries = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def save_results(jsonl_path: Path, entries: list) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


# ── Bbox conversion (all output YOLO: [x_center, y_center, w, h] normalised) ──

def megadetector_to_yolo(bbox: list) -> list:
    """MegaDetector / COCO [xmin, ymin, w, h] → YOLO [xc, yc, w, h]."""
    xmin, ymin, w, h = bbox
    return [xmin + w / 2.0, ymin + h / 2.0, w, h]


def xyxy_to_yolo(xmin: float, ymin: float, xmax: float, ymax: float) -> list:
    """Normalised [xmin, ymin, xmax, ymax] → YOLO [xc, yc, w, h]."""
    w = xmax - xmin
    h = ymax - ymin
    return [(xmin + xmax) / 2.0, (ymin + ymax) / 2.0, w, h]


def catalog_bbox_to_yolo(bbox_str: str):
    """'xmin,ymin,xmax,ymax' string (Open Images / catalog) → YOLO list or None."""
    if not bbox_str or not bbox_str.strip():
        return None
    try:
        parts = [float(x) for x in bbox_str.split(",")]
        if len(parts) != 4:
            return None
        return xyxy_to_yolo(*parts)
    except ValueError:
        return None


def bbox_area(bbox: list) -> float:
    """Fractional area of a YOLO [xc, yc, w, h] bbox."""
    return bbox[2] * bbox[3]


# ── Entry constructors ────────────────────────────────────────────────────────

def _pass_entry(filepath: str, *, detections=None, stages_done=None,
                bbox=None, bbox_conf=None) -> dict:
    entry = {
        "filepath":    filepath,
        "passed":      True,
        "stage_failed": None,
        "reason":      None,
        "detections":  detections if detections is not None else [],
        "stages_done": stages_done or [],
    }
    if bbox is not None:
        entry["bbox"]      = bbox
        entry["bbox_conf"] = bbox_conf
    return entry


def _fail_entry(filepath: str, stage: str, reason: str) -> dict:
    return {
        "filepath":    filepath,
        "passed":      False,
        "stage_failed": stage,
        "reason":      reason,
        "detections":  [],
        "stages_done": [stage],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Stage 0: metadata
# ══════════════════════════════════════════════════════════════════════════════

def cmd_metadata(args):
    if args.source == "all":
        for source in RESULTS_PATHS:
            a = argparse.Namespace(**vars(args)); a.source = source
            cmd_metadata(a)
        return
    source = args.source
    dispatch = {
        "gbif":        _meta_gbif,
        "inaturalist": _meta_inaturalist,
        "wikimedia":   _meta_wikimedia,
        "openimages":  _meta_openimages,
        "images_cv":   _meta_images_cv,
    }
    entries = dispatch[source]()
    path = RESULTS_PATHS[source]
    save_results(path, entries)
    passed  = sum(1 for e in entries if e["passed"])
    skipped = len(entries) - passed
    print(f"[metadata] {source}: {len(entries)} images on disk — "
          f"{passed} passed, {skipped} filtered  →  {path}")


# ── GBIF ──────────────────────────────────────────────────────────────────────

def _meta_gbif() -> list:
    print("Loading SNPredictions_all.json …")
    with open(GBIF_PREDICTIONS_JSON, encoding="utf-8") as f:
        data = json.load(f)
    predictions = data["predictions"]

    # Images are stored in per-species subdirectories; build a filename→rel-path index
    print("Indexing GBIF images on disk …")
    filename_to_rel: dict[str, str] = {}
    for img in GBIF_IMAGES_DIR.rglob("*"):
        if img.is_file():
            filename_to_rel[img.name] = img.relative_to(REPO_ROOT).as_posix()

    entries = []
    for pred in tqdm(predictions, desc="GBIF metadata", unit=" images"):
        rel = filename_to_rel.get(pred["filepath"])
        if rel is None:
            continue   # not downloaded

        src = pred.get("prediction_source", "")
        if src != "classifier":
            entries.append(_fail_entry(rel, "metadata",
                f"prediction_source '{src}' (need 'classifier')"))
            continue

        score = pred.get("prediction_score", 0.0)
        if score < GBIF_MIN_SCORE:
            entries.append(_fail_entry(rel, "metadata",
                f"prediction_score {score:.3f} < {GBIF_MIN_SCORE}"))
            continue

        # Collect all animal dets above secondary threshold, sorted by confidence desc
        all_animal_dets = sorted(
            [d for d in (pred.get("detections") or [])
             if d.get("label") == "animal" and d.get("conf", 0.0) >= GBIF_ANIMAL_CONF_SECONDARY],
            key=lambda d: d["conf"], reverse=True,
        )
        if not all_animal_dets or all_animal_dets[0]["conf"] < GBIF_ANIMAL_CONF:
            entries.append(_fail_entry(rel, "metadata", "no animal detection"))
            continue

        primary_bbox = megadetector_to_yolo(all_animal_dets[0]["bbox"])
        if bbox_area(primary_bbox) < GBIF_MIN_BBOX_AREA:
            entries.append(_fail_entry(rel, "metadata",
                f"animal bbox area {bbox_area(primary_bbox):.4f} < {GBIF_MIN_BBOX_AREA}"))
            continue

        # Build final detections list: convert all to YOLO, drop those with tiny area
        detections = []
        for d in all_animal_dets:
            b = megadetector_to_yolo(d["bbox"])
            if bbox_area(b) >= GBIF_MIN_BBOX_AREA:
                detections.append({"bbox": b, "conf": d["conf"]})

        entries.append(_pass_entry(rel, detections=detections, stages_done=["metadata"]))

    # Fallback: images were renamed after the initial metadata run.
    # SNPredictions_all.json still uses original filenames so no matches are found.
    # Create pass entries for all on-disk images; MegaDetector provides bboxes later.
    if not entries and filename_to_rel:
        print(f"  WARNING: {len(filename_to_rel):,} GBIF images on disk but none matched "
              "SNPredictions_all.json.\n"
              "  Images were likely renamed after the initial metadata stage ran.\n"
              "  Creating pass entries for all on-disk images "
              "(SpeciesNet prediction filters skipped).")
        for rel in sorted(filename_to_rel.values()):
            entries.append(_pass_entry(rel, stages_done=["metadata"]))

    return entries


# ── iNaturalist ───────────────────────────────────────────────────────────────

def _meta_inaturalist() -> list:
    # Images were originally saved as inat_<photo_id>.jpg but were later renamed
    # to inat_<species>_<seq>.jpg by scripts/rename_dataset_images.py, which
    # destroyed the photo_id→filename link needed to look up quality grades in
    # photos.csv/observations.csv.  Pass all on-disk images; MegaDetector filters
    # bad images in the megadetector stage.
    print("Scanning iNaturalist images …")
    entries = []
    for cls_dir in sorted(INAT_IMAGES_DIR.iterdir()):
        if not cls_dir.is_dir():
            continue
        for img in sorted(cls_dir.iterdir()):
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            rel = img.relative_to(REPO_ROOT).as_posix()
            entries.append(_pass_entry(rel, stages_done=["metadata"]))
    print(f"  Found {len(entries):,} images on disk (quality_grade check skipped — "
          "photo_id→filename mapping lost after rename)")
    return entries


# ── Wikimedia ─────────────────────────────────────────────────────────────────

def _meta_wikimedia() -> list:
    if not WIKI_METADATA_CSV.exists():
        print(f"WARNING: {WIKI_METADATA_CSV} not found — run download_wikimedia_images.py first")
        return []

    entries = []
    with open(WIKI_METADATA_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Wikimedia metadata"):
            label_dir = row.get("label_dir", "")
            filename  = row.get("filename", "")
            if not label_dir or not filename:
                continue
            rel = f"data/wikimedia/images/{label_dir}/{filename}"
            if not (REPO_ROOT / rel).exists():
                continue

            # Dimension check (no file I/O — already in metadata)
            try:
                w, h = int(row["width"]), int(row["height"])
            except (ValueError, KeyError):
                entries.append(_fail_entry(rel, "metadata", "missing/invalid dimensions"))
                continue

            if min(w, h) < MIN_RESOLUTION:
                entries.append(_fail_entry(rel, "metadata",
                    f"resolution {w}×{h} below {MIN_RESOLUTION}px"))
                continue

            if max(w, h) / min(w, h) > MAX_ASPECT_RATIO:
                entries.append(_fail_entry(rel, "metadata",
                    f"aspect ratio {max(w,h)/min(w,h):.1f} > {MAX_ASPECT_RATIO}"))
                continue

            mime = row.get("mime", "")
            if mime not in VALID_MIMES:
                entries.append(_fail_entry(rel, "metadata",
                    f"unsupported mime: {mime!r}"))
                continue

            entries.append(_pass_entry(rel, stages_done=["metadata"]))
    return entries


# ── Open Images ───────────────────────────────────────────────────────────────

def _meta_openimages() -> list:
    if not OI_CATALOG_CSV.exists():
        print(f"WARNING: {OI_CATALOG_CSV} not found — run download_supplementary.py first")
        return []

    entries = []
    with open(OI_CATALOG_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Open Images metadata"):
            if row.get("source") != "openimages":
                continue
            label    = row["label"]
            filename = row["filename"]
            label_dir = label.replace(" ", "_")
            rel = f"data/openimages/images/{label_dir}/{filename}"
            if not (REPO_ROOT / rel).exists():
                continue

            bbox = catalog_bbox_to_yolo(row.get("bbox", ""))
            # Discard implausibly tiny bboxes (let MegaDetector provide a better one)
            if bbox is not None and bbox_area(bbox) < GBIF_MIN_BBOX_AREA:
                bbox = None
            entries.append(_pass_entry(rel, bbox=bbox,
                                       bbox_conf=1.0 if bbox else None,
                                       stages_done=["metadata"]))
    return entries


# ── images_cv ─────────────────────────────────────────────────────────────────

def _meta_images_cv() -> list:
    """All images in images_cv are manually curated; pass all of them through."""
    if not CV_IMAGES_DIR.exists():
        print(f"WARNING: {CV_IMAGES_DIR} not found")
        return []

    entries = []
    for img in sorted(CV_IMAGES_DIR.rglob("*")):
        if not img.is_file():
            continue
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        rel = img.relative_to(REPO_ROOT).as_posix()
        entries.append(_pass_entry(rel, stages_done=["metadata"]))

    print(f"images_cv: {len(entries)} images found, all passed metadata")
    return entries


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1: heuristics
# ══════════════════════════════════════════════════════════════════════════════

def cmd_heuristics(args):
    if args.source == "all":
        for source in RESULTS_PATHS:
            a = argparse.Namespace(**vars(args)); a.source = source
            cmd_heuristics(a)
        return
    path = RESULTS_PATHS[args.source]
    entries = load_results(path)
    if not entries:
        print(f"No entries found at {path} — run metadata stage first")
        return

    if args.force:
        reset_count = 0
        for e in entries:
            if "heuristics" not in e.get("stages_done", []):
                continue
            e["stages_done"] = [s for s in e["stages_done"] if s != "heuristics"]
            if e.get("stage_failed") == "heuristics":
                e["passed"]       = True
                e["stage_failed"] = None
                e["reason"]       = None
            reset_count += 1
        print(f"[heuristics] --force: reset {reset_count:,} entries, re-running …")

    pending = [e for e in entries
               if e["passed"] and "heuristics" not in e.get("stages_done", [])]
    print(f"Checking {len(pending):,} images (of {len(entries):,} total) …")

    workers = min(args.workers, len(pending)) if pending else 1
    paths   = [str(REPO_ROOT / e["filepath"]) for e in pending]

    results: dict[str, tuple[bool, str | None]] = {}
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers) as pool:
        for path_str, ok, reason in tqdm(
            pool.imap_unordered(_check_image_worker, paths, chunksize=8),
            total=len(paths),
            desc="heuristics",
        ):
            results[path_str] = (ok, reason)

    failed = 0
    for entry in pending:
        ok, reason = results[str(REPO_ROOT / entry["filepath"])]
        entry.setdefault("stages_done", []).append("heuristics")
        if not ok:
            entry["passed"]       = False
            entry["stage_failed"] = "heuristics"
            entry["reason"]       = reason
            failed += 1

    save_results(path, entries)
    print(f"[heuristics] {args.source}: {failed:,} failed, {len(pending)-failed:,} passed")


def _check_image_worker(path_str: str) -> tuple[str, bool, str | None]:
    """Multiprocessing worker: run heuristic checks and return (path_str, passed, reason)."""
    ok, reason = _check_image(Path(path_str))
    return path_str, ok, reason


def _check_image(path: Path) -> tuple[bool, str | None]:
    """Run all heuristic checks. Returns (passed, reason_or_None)."""
    import cv2

    # 1. Corruption / truncation — single open + full decode
    try:
        img_pil = Image.open(path)
        img_pil.load()
    except (UnidentifiedImageError, Exception) as e:
        return False, f"corrupt/unreadable: {e}"

    w, h = img_pil.size

    # 2. Resolution
    if min(w, h) < MIN_RESOLUTION:
        return False, f"too small: {w}×{h} (min {MIN_RESOLUTION}px on short side)"

    # 3. Aspect ratio
    ratio = max(w, h) / min(w, h)
    if ratio > MAX_ASPECT_RATIO:
        return False, f"extreme aspect ratio: {ratio:.2f} > {MAX_ASPECT_RATIO}"

    # 4. Grayscale detection — mean HSV saturation (robust to muted earthy tones)
    img_rgb = img_pil.convert("RGB")
    arr_u8  = np.array(img_rgb, dtype=np.uint8)
    arr_bgr = arr_u8[:, :, ::-1]   # RGB → BGR in-place view (no copy)
    hsv     = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2HSV)
    mean_sat = float(hsv[:, :, 1].mean())
    if mean_sat < 15.0:
        return False, f"effectively grayscale (mean HSV saturation={mean_sat:.1f} < 15.0)"

    # 5. Blur — Laplacian variance on a 512×512 grayscale thumbnail
    thumb   = img_rgb.convert("L").resize((512, 512), Image.LANCZOS)
    gray    = np.array(thumb, dtype=np.float64)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if lap_var < BLUR_THRESHOLD:
        return False, f"blurry (Laplacian var={lap_var:.1f} < {BLUR_THRESHOLD})"

    return True, None


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2: MegaDetector
# ══════════════════════════════════════════════════════════════════════════════

def cmd_megadetector(args):
    if args.source == "all":
        for source in RESULTS_PATHS:
            a = argparse.Namespace(**vars(args)); a.source = source
            cmd_megadetector(a)
        return
    path = RESULTS_PATHS[args.source]
    entries = load_results(path)
    if not entries:
        print(f"No entries found at {path} — run metadata + heuristics stages first")
        return

    if args.force:
        reset_count = 0
        for e in entries:
            if "megadetector" not in e.get("stages_done", []):
                continue
            e["stages_done"] = [s for s in e["stages_done"] if s != "megadetector"]
            if e.get("stage_failed") == "megadetector":
                # Restore to the state after heuristics passed
                e["passed"]       = True
                e["stage_failed"] = None
                e["reason"]       = None
            else:
                # Entry had a MD-assigned bbox — clear it so it gets re-inferred
                e["bbox"]      = None
                e["bbox_conf"] = None
            reset_count += 1
        print(f"[megadetector] --force: reset {reset_count:,} entries, re-running …")

    # Process images that: passed so far, haven't had MegaDetector run, and lack a bbox
    pending = [e for e in entries
               if e["passed"]
               and "megadetector" not in e.get("stages_done", [])
               and e.get("bbox") is None]

    if not pending:
        print("Nothing to process — all surviving images already have bboxes or were "
              "already run through MegaDetector.")
        return

    print(f"Running MegaDetector v5 on {len(pending):,} images …")

    try:
        import torch
        from torch.utils.data import Dataset, DataLoader
        from PytorchWildlife.models import detection as pw_detection
        from yolov5.utils.general import non_max_suppression, scale_boxes
    except ImportError as exc:
        print(f"ERROR: missing dependency: {exc}\n"
              "  Run: pip install pytorchwildlife")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    model = pw_detection.MegaDetectorV5(device=device, pretrained=True)
    model.model.eval()

    conf_pass      = args.conf            # image must have ≥ 1 animal detection ≥ this
    conf_secondary = args.conf_secondary  # NMS lower bound; all detections above this are recorded
    batch_size  = args.batch_size
    num_workers = args.num_workers

    # Index entries for fast lookup
    entry_map = {e["filepath"]: e for e in pending}
    rel_paths  = list(entry_map)
    abs_paths  = [str(REPO_ROOT / p) for p in rel_paths]

    # Custom dataset: accepts a flat list of file paths rather than a directory,
    # bypassing pytorchwildlife's batch_image_detection which hardcodes num_workers=0
    # and expects a directory string (causing silent fallback to per-image inference).
    class _FileListDataset(Dataset):
        def __init__(self, paths, transform):
            self.paths = paths
            self.transform = transform

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            p = self.paths[idx]
            img = Image.open(p).convert("RGB")
            size = torch.tensor(img.size[::-1])  # (H, W)
            if self.transform:
                img = self.transform(img)
            return img, p, size

    dataset = _FileListDataset(abs_paths, model.transform)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=(num_workers > 0),
        shuffle=False,
        drop_last=False,
    )

    failed   = 0
    img_size = model.IMAGE_SIZE  # typically 1280 for MDv5
    save_interval = max(1, 500 // batch_size)  # flush to disk roughly every 500 images

    with torch.no_grad():
        for batch_idx, (imgs, batch_abs, sizes) in enumerate(
                tqdm(loader, desc="MegaDetector batches")):
            imgs = imgs.to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=(device == "cuda")):
                raw = model.model(imgs)[0]
            raw = raw.float().detach().cpu()
            preds = non_max_suppression(raw, conf_thres=conf_secondary)

            for i, pred in enumerate(preds):
                abs_p = batch_abs[i]
                rel_p = str(Path(abs_p).relative_to(REPO_ROOT))
                entry = entry_map[rel_p]
                entry.setdefault("stages_done", []).append("megadetector")

                if pred is None or len(pred) == 0:
                    entry["passed"]       = False
                    entry["stage_failed"] = "megadetector"
                    entry["reason"]       = f"no animal detected (conf ≥ {conf_pass})"
                    failed += 1
                    continue

                H, W = sizes[i].tolist()
                pred_np = pred.numpy().copy()
                pred_np[:, :4] = scale_boxes(
                    [img_size] * 2, pred_np[:, :4], (H, W)
                ).round()

                # All class 0 detections above secondary threshold, sorted desc by conf
                animal_dets = sorted(
                    [{"bbox": [float(x1/W), float(y1/H),
                                float((x2-x1)/W), float((y2-y1)/H)],
                      "conf": float(c)}
                     for x1, y1, x2, y2, c, cls in pred_np if int(cls) == 0],
                    key=lambda d: d["conf"], reverse=True,
                )

                # Image passes only if at least one detection clears the pass bar
                pass_dets = [d for d in animal_dets if d["conf"] >= conf_pass]
                if not pass_dets:
                    entry["passed"]       = False
                    entry["stage_failed"] = "megadetector"
                    entry["reason"]       = f"no animal detected (conf ≥ {conf_pass})"
                    failed += 1
                    continue

                best = pass_dets[0]  # highest-conf detection above pass threshold
                bbox = megadetector_to_yolo(best["bbox"])
                if bbox_area(bbox) < MD_BBOX_MIN_AREA:
                    entry["passed"]       = False
                    entry["stage_failed"] = "megadetector"
                    entry["reason"]       = (f"animal bbox area {bbox_area(bbox):.4f}"
                                             f" < {MD_BBOX_MIN_AREA}")
                    failed += 1
                    continue

                # Record all secondary detections with sufficient area
                detections = []
                for d in animal_dets:
                    b = megadetector_to_yolo(d["bbox"])
                    if bbox_area(b) >= MD_BBOX_MIN_AREA:
                        detections.append({"bbox": b, "conf": d["conf"]})

                entry["bbox"]       = bbox
                entry["bbox_conf"]  = best["conf"]
                entry["detections"] = detections

            if (batch_idx + 1) % save_interval == 0:
                save_results(path, entries)

    save_results(path, entries)
    passed_count = len(pending) - failed
    print(f"[megadetector] {args.source}: {failed:,} failed, {passed_count:,} passed")
    print(f"  Results written to: {path}")


def _parse_md_result(result, conf_thresh: float, img_path=None) -> list[dict]:
    """Extract animal detections from a PytorchWildlife result.

    Handles both the dict format (older API / batch output) and the
    Ultralytics Results object returned by newer PytorchWildlife versions.

    MegaDetector classes: 1 = animal, 2 = person, 3 = vehicle
    (PytorchWildlife may use 0-indexed: 0 = animal)

    img_path is required for the newer tuple detection format, where bounding
    boxes are returned in pixel coordinates and must be normalised.
    """
    dets = []

    if result is None:
        return dets

    # ── Dict format ──────────────────────────────────────────────────────────
    if isinstance(result, dict):
        raw = result.get("detections", [])
        # raw may itself be an Ultralytics Results object
        if hasattr(raw, "boxes"):
            return _parse_ultralytics_boxes(raw, conf_thresh)
        img_size = None  # (w, h) — loaded lazily if pixel-space tuples appear
        for d in raw:
            if isinstance(d, dict):
                cat = str(d.get("category", ""))
                lbl = str(d.get("label", ""))
                if cat == "1" or lbl == "animal":
                    conf = float(d.get("conf", 0.0))
                    if conf >= conf_thresh:
                        dets.append({"bbox": d["bbox"], "conf": conf})
            elif isinstance(d, (list, tuple)) and len(d) >= 4:
                # Newer PytorchWildlife batch_image_detection returns detections
                # as (xyxy_pixels, track_id, conf, cls_id, mask, extra) tuples.
                # Class 0 = animal (0-indexed).
                conf = float(d[2])
                cls  = int(d[3])
                if cls == 0 and conf >= conf_thresh:
                    if img_size is None:
                        if img_path is not None:
                            try:
                                with Image.open(img_path) as im:
                                    img_size = im.size  # (w, h)
                            except Exception:
                                img_size = (0, 0)
                        else:
                            img_size = (0, 0)
                    iw, ih = img_size
                    if iw > 0 and ih > 0:
                        x1, y1, x2, y2 = [float(v) for v in d[0][:4]]
                        # Convert pixel xyxy → normalised COCO [xmin, ymin, w, h]
                        dets.append({
                            "bbox": [x1 / iw, y1 / ih,
                                     (x2 - x1) / iw, (y2 - y1) / ih],
                            "conf": conf,
                        })
        return dets

    # ── Ultralytics Results object ────────────────────────────────────────────
    if hasattr(result, "boxes"):
        return _parse_ultralytics_boxes(result, conf_thresh)

    return dets


def _parse_ultralytics_boxes(result, conf_thresh: float) -> list[dict]:
    """Parse an Ultralytics Results object for animal detections."""
    dets = []
    try:
        boxes  = result.boxes
        xyxyn  = boxes.xyxyn.cpu().tolist()   # normalised xyxy
        confs  = boxes.conf.cpu().tolist()
        clss   = boxes.cls.cpu().tolist()
    except (AttributeError, Exception):
        return dets

    for (x1, y1, x2, y2), conf, cls in zip(xyxyn, confs, clss):
        # Class 0 = animal in PytorchWildlife 0-indexed scheme
        if int(cls) == 0 and float(conf) >= conf_thresh:
            w = x2 - x1
            h = y2 - y1
            # Store as COCO [xmin, ymin, w, h] so megadetector_to_yolo() converts it
            dets.append({"bbox": [x1, y1, w, h], "conf": float(conf)})
    return dets


# ══════════════════════════════════════════════════════════════════════════════
# Stage 3: VLM (Florence-2) — Wikimedia borderline rescue only
# ══════════════════════════════════════════════════════════════════════════════

def cmd_vlm(args):
    if args.source == "all":
        a = argparse.Namespace(**vars(args)); a.source = "wikimedia"
        cmd_vlm(a)
        return
    path = RESULTS_PATHS[args.source]
    entries = load_results(path)
    if not entries:
        print(f"No entries at {path}"); return

    if args.force:
        reset_count = 0
        for e in entries:
            if "vlm" not in e.get("stages_done", []):
                continue
            e["stages_done"] = [s for s in e["stages_done"] if s != "vlm"]
            if e.get("stage_failed") == "vlm" or e.get("passed") is True and "vlm_caption" in e:
                # Rescued image: revert to megadetector-failed state
                e["passed"]       = False
                e["stage_failed"] = "megadetector"
                e["reason"]       = "no animal detected (conf >= 0.6)"
            e.pop("vlm_caption", None)
            e.pop("vlm_error", None)
            reset_count += 1
        print(f"[vlm] --force: reset {reset_count:,} entries, re-running …")

    # Target: passed heuristics but FAILED MegaDetector (and not yet VLM-processed)
    borderline = [
        e for e in entries
        if not e["passed"]
        and e.get("stage_failed") == "megadetector"
        and "heuristics" in e.get("stages_done", [])
        and "vlm" not in e.get("stages_done", [])
    ]

    if not borderline:
        print("No borderline images to process"); return

    print(f"Running Florence-2 on {len(borderline):,} borderline Wikimedia images …")
    print("  Goal: rescue real photos that MegaDetector missed (unusual angles, occlusion, etc.)")

    try:
        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM
    except ImportError:
        print("ERROR: transformers not installed.\n"
              "  Run: pip install transformers timm einops")
        sys.exit(1)

    device     = "cuda" if torch.cuda.is_available() else "cpu"
    model_id   = "microsoft/Florence-2-large"
    print(f"  Loading {model_id} on {device} …")

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    vlm_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        attn_implementation="eager",
    ).to(device)
    vlm_model.eval()

    task  = "<DETAILED_CAPTION>"
    rescued = 0

    for entry in tqdm(borderline, desc="Florence-2"):
        entry.setdefault("stages_done", []).append("vlm")
        abs_path = REPO_ROOT / entry["filepath"]
        try:
            img    = Image.open(abs_path).convert("RGB")
            inputs = processor(text=task, images=img, return_tensors="pt").to(device)
            import torch as _torch
            with _torch.no_grad():
                tokens = vlm_model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=200,
                    do_sample=False,
                )
            caption = processor.batch_decode(tokens, skip_special_tokens=False)[0]
            cap_low = caption.lower()
            entry["vlm_caption"] = caption[:400]

            is_non_photo = any(kw in cap_low for kw in VLM_REJECT_KEYWORDS)
            if not is_non_photo:
                # Likely a real photograph — rescue it; MegaDetector will get another
                # chance once a lower threshold or broader model is applied downstream.
                entry["passed"]       = True
                entry["stage_failed"] = None
                entry["reason"]       = None
                rescued += 1
            else:
                matched = next(kw for kw in VLM_REJECT_KEYWORDS if kw in cap_low)
                entry["reason"] = (f"vlm confirmed non-photo ({matched!r}); "
                                   f"original: {entry['reason']}")
        except Exception as e:
            entry["vlm_error"] = str(e)

    save_results(path, entries)
    print(f"[vlm] {args.source}: rescued {rescued:,} / {len(borderline):,} borderline images")


# ══════════════════════════════════════════════════════════════════════════════
# Stage 4: report
# ══════════════════════════════════════════════════════════════════════════════

def cmd_report(args):
    sources = (list(RESULTS_PATHS) if args.source == "all" else [args.source])

    for source in sources:
        path = RESULTS_PATHS[source]
        if not path.exists():
            print(f"\n{source}: no filter_results.jsonl found (run metadata stage first)")
            continue

        entries = load_results(path)
        if not entries:
            print(f"\n{source}: empty filter_results.jsonl")
            continue

        total   = len(entries)
        passed  = sum(1 for e in entries if e["passed"])
        with_bbox = sum(1 for e in entries if e["passed"] and e.get("bbox"))

        stage_fail: dict[str, int] = {}
        for e in entries:
            if not e["passed"]:
                s = e.get("stage_failed", "unknown")
                stage_fail[s] = stage_fail.get(s, 0) + 1

        print(f"\n{'═'*52}")
        print(f"  Source : {source}")
        print(f"{'═'*52}")
        print(f"  Total images   : {total:>8,}")
        print(f"  Passed         : {passed:>8,}  ({100*passed/total:.1f}%)")
        print(f"  With bbox      : {with_bbox:>8,}")
        print(f"  Rejected:")
        for stage, count in sorted(stage_fail.items()):
            print(f"    {stage:<22} {count:>7,}")
        if not stage_fail:
            print("    (none)")

    print()


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    SOURCES = ["gbif", "inaturalist", "wikimedia", "openimages", "images_cv"]

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # metadata
    p = sub.add_parser("metadata",
                       help="Generate filter_results.jsonl from source metadata (no GPU)")
    p.add_argument("--source", choices=SOURCES + ["all"], required=True)
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing filter_results.jsonl (always a full rebuild)")
    p.set_defaults(func=cmd_metadata)

    # heuristics
    p = sub.add_parser("heuristics",
                       help="Run per-image quality checks: corruption, resolution, blur, grayscale")
    p.add_argument("--source", choices=SOURCES + ["all"], required=True)
    p.add_argument("--force", action="store_true",
                   help="Reset and re-run heuristics even for entries already processed")
    p.add_argument("--workers", type=int, default=min(8, __import__("os").cpu_count() or 1),
                   metavar="N", help="Parallel worker processes (default: min(8, cpu_count))")
    p.set_defaults(func=cmd_heuristics)

    # megadetector
    p = sub.add_parser("megadetector",
                       help="MegaDetector v5 inference for animal detection + bbox generation")
    p.add_argument("--source", choices=SOURCES + ["all"], required=True)
    p.add_argument("--batch-size", type=int, default=32, metavar="N",
                   help="Images per GPU batch (default: 32)")
    p.add_argument("--num-workers", type=int, default=4, metavar="N",
                   help="DataLoader CPU worker processes for image prefetching (default: 4)")
    p.add_argument("--conf", type=float, default=MD_CONF_PASS, metavar="T",
                   help=f"Minimum confidence for an image to pass (default: {MD_CONF_PASS})")
    p.add_argument("--conf-secondary", type=float, default=MD_CONF_SECONDARY, metavar="T",
                   help=f"Lower confidence for recording secondary detections (default: {MD_CONF_SECONDARY})")
    p.add_argument("--force", action="store_true",
                   help="Reset and re-run MegaDetector even for entries already processed")
    p.set_defaults(func=cmd_megadetector)

    # vlm
    p = sub.add_parser("vlm",
                       help="Florence-2 semantic rescue for Wikimedia borderline images")
    p.add_argument("--source", choices=["wikimedia", "all"], default="wikimedia",
                   help="Source to process; 'all' runs wikimedia only (default: wikimedia)")
    p.add_argument("--force", action="store_true",
                   help="Reset and re-run VLM even for entries already processed")
    p.set_defaults(func=cmd_vlm)

    # report
    p = sub.add_parser("report", help="Print per-stage filtering statistics")
    p.add_argument("--source", choices=SOURCES + ["all"], default="all")
    p.set_defaults(func=cmd_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
