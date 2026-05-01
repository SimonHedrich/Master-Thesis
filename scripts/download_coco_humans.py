"""
Download ~2000 human images from COCO 2017 with strict quality filters.

Selection filters (applied before any download):
  - Hard exclude: any image containing animal or vehicle annotations
  - Hard exclude: any image with iscrowd=1 person annotation
  - Hard exclude: more than MAX_PERSONS person annotations
  - Per-bbox: person must cover MIN_NORM_AREA–MAX_NORM_AREA of the image
  - Per-bbox: bbox must be ≥ EDGE_MARGIN from all four edges (not cropped)
  - Per-bbox: bbox height/width ≥ MIN_ASPECT (upright, not horizontal crop)
  - Soft rank: fewest indoor annotations first, then largest person area first

Post-download quality checks (mirrors 1-filter_dataset_quality.py heuristics):
  - Minimum resolution: shorter side ≥ MIN_SHORT_SIDE px
  - Blur: Laplacian variance ≥ BLUR_THRESHOLD
  - Grayscale/IR: mean HSV saturation ≥ MIN_SATURATION

Output:
    data/coco_humans/
    ├── metadata/           ← cached annotation ZIP + JSONs
    ├── images/human/       ← coco_{image_id}.jpg
    ├── metadata_catalog.csv
    └── download_progress.json

Usage:
    python scripts/download_coco_humans.py --reset   # fresh start
    python scripts/download_coco_humans.py --reset --target 100 --workers 4  # smoke test
    python scripts/download_coco_humans.py --target 2000  # resume / full run
"""

import argparse
import csv
import json
import os
import signal
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import cv2
import numpy as np
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _image_utils import save_as_jpg

# ── Paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
COCO_HUMANS_DIR = REPO_ROOT / "data" / "coco_humans"
METADATA_DIR = COCO_HUMANS_DIR / "metadata"
IMAGES_DIR = COCO_HUMANS_DIR / "images" / "human"
CATALOG_FILE = COCO_HUMANS_DIR / "metadata_catalog.csv"
PROGRESS_FILE = COCO_HUMANS_DIR / "download_progress.json"

ANNOTATIONS_ZIP_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
ANNOTATIONS_ZIP_PATH = METADATA_DIR / "annotations_trainval2017.zip"
TRAIN_IMAGE_BASE = "http://images.cocodataset.org/train2017"
VAL_IMAGE_BASE = "http://images.cocodataset.org/val2017"

USER_AGENT = (
    "MasterThesis-WildlifeDetection/1.0 "
    "(wildlife-detection-research) python-requests"
)

# ── Selection filter thresholds ───────────────────────────────────────────────

ANIMAL_SUPERCATEGORY = "animal"
VEHICLE_SUPERCATEGORY = "vehicle"
INDOOR_SUPERCATEGORIES = frozenset({"furniture", "appliance", "electronic", "indoor"})

MAX_PERSONS   = 3     # hard limit on total person annotations per image
MIN_NORM_AREA = 0.05  # person bbox must cover ≥ 5% of image area
MAX_NORM_AREA = 0.60  # person bbox must cover ≤ 60% of image area (no extreme close-ups)
EDGE_MARGIN   = 0.02  # bbox must be ≥ 2% from all four edges
MIN_ASPECT    = 0.5   # bbox height/width ≥ 0.5 (person roughly upright)

# ── Post-download quality thresholds (mirrors 1-filter_dataset_quality.py) ────

MIN_SHORT_SIDE = 256    # px
BLUR_THRESHOLD = 100.0  # Laplacian variance
MIN_SATURATION = 15.0   # mean HSV S-channel

# ── COCO license map ──────────────────────────────────────────────────────────

COCO_LICENSE_NAMES = {
    1: "CC-BY-NC-SA-2.0",
    2: "CC-BY-NC-2.0",
    3: "CC-BY-NC-ND-2.0",
    4: "CC-BY-2.0",
    5: "CC-BY-SA-2.0",
    6: "CC-BY-ND-2.0",
    7: "No known copyright restrictions",
    8: "US Government Work",
}

# ── Download defaults ─────────────────────────────────────────────────────────

DEFAULT_TARGET = 2000
DEFAULT_CANDIDATES = 3000   # larger pool since filters are stricter
DEFAULT_WORKERS = 8
FAILURE_BUFFER = 500
PROGRESS_SAVE_INTERVAL = 50


# ── Data ──────────────────────────────────────────────────────────────────────

@dataclass
class ImageCandidate:
    image_id: int
    split: str
    file_name: str
    license_id: int
    indoor_score: int      # count of indoor-supercategory anns (lower = more outdoor)
    max_norm_area: float   # largest qualifying person bbox area (higher = more prominent)
    person_bboxes: List[str] = field(default_factory=list)  # normalized "xmin,ymin,xmax,ymax"


# ── Annotation download ───────────────────────────────────────────────────────

def download_annotations():
    """Download and cache COCO 2017 annotation JSONs. Returns (train_path, val_path)."""
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    train_json = METADATA_DIR / "instances_train2017.json"
    val_json = METADATA_DIR / "instances_val2017.json"

    if train_json.exists() and val_json.exists():
        print("  Annotations already cached.")
        return train_json, val_json

    if not ANNOTATIONS_ZIP_PATH.exists():
        print(f"  Downloading annotation ZIP (~252 MB) …")
        resp = requests.get(
            ANNOTATIONS_ZIP_URL, stream=True, headers={"User-Agent": USER_AGENT}, timeout=120
        )
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(ANNOTATIONS_ZIP_PATH, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc="annotations.zip"
        ) as pbar:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                pbar.update(len(chunk))
    else:
        print("  Annotation ZIP already on disk, extracting …")

    with zipfile.ZipFile(ANNOTATIONS_ZIP_PATH) as zf:
        for member in zf.namelist():
            if "instances_" in member and member.endswith(".json"):
                dest = METADATA_DIR / Path(member).name
                if not dest.exists():
                    with zf.open(member) as src, open(dest, "wb") as dst:
                        dst.write(src.read())
                    print(f"    Extracted {dest.name}")

    if not train_json.exists() or not val_json.exists():
        raise RuntimeError("instances_train/val2017.json not found in ZIP.")
    return train_json, val_json


# ── Candidate building ────────────────────────────────────────────────────────

def build_candidates(train_path: Path, val_path: Path, pool_size: int) -> list:
    """Return sorted ImageCandidates: outdoor-first, most-prominent-person-first."""
    candidates = {}

    for split, json_path in [("train2017", train_path), ("val2017", val_path)]:
        print(f"  Scanning {json_path.name} …")
        with open(json_path) as f:
            data = json.load(f)

        # Category sets derived from supercategory — never hardcoded IDs
        person_cat_ids  = {c["id"] for c in data["categories"] if c["name"] == "person"}
        animal_cat_ids  = {c["id"] for c in data["categories"] if c["supercategory"] == ANIMAL_SUPERCATEGORY}
        vehicle_cat_ids = {c["id"] for c in data["categories"] if c["supercategory"] == VEHICLE_SUPERCATEGORY}
        indoor_cat_ids  = {c["id"] for c in data["categories"] if c["supercategory"] in INDOOR_SUPERCATEGORIES}

        img_has_animal:  set  = set()
        img_has_vehicle: set  = set()
        img_has_crowd:   set  = set()
        img_indoor_count: dict = {}
        img_person_anns:  dict = {}

        for ann in data["annotations"]:
            iid = ann["image_id"]
            cat = ann["category_id"]

            if cat in animal_cat_ids:
                img_has_animal.add(iid)
            elif cat in vehicle_cat_ids:
                img_has_vehicle.add(iid)
            elif cat in indoor_cat_ids:
                img_indoor_count[iid] = img_indoor_count.get(iid, 0) + 1
            elif cat in person_cat_ids:
                if ann.get("iscrowd", 0):
                    img_has_crowd.add(iid)
                img_person_anns.setdefault(iid, []).append(ann)

        img_meta = {img["id"]: img for img in data["images"]}

        for iid, person_anns in img_person_anns.items():
            # ── Image-level hard excludes ──
            if iid in img_has_animal or iid in img_has_vehicle or iid in img_has_crowd:
                continue
            if len(person_anns) > MAX_PERSONS:
                continue
            if iid not in img_meta:
                continue

            meta = img_meta[iid]
            w, h = meta.get("width", 0), meta.get("height", 0)
            if w <= 0 or h <= 0:
                continue

            # ── Per-bbox qualifying checks ──
            qualifying_bboxes = []
            for ann in person_anns:
                bx, by, bw, bh = ann["bbox"]
                norm_area = (bw / w) * (bh / h)
                if norm_area < MIN_NORM_AREA or norm_area > MAX_NORM_AREA:
                    continue

                xmin = bx / w
                ymin = by / h
                xmax = (bx + bw) / w
                ymax = (by + bh) / h

                if (xmin < EDGE_MARGIN or ymin < EDGE_MARGIN or
                        xmax > 1 - EDGE_MARGIN or ymax > 1 - EDGE_MARGIN):
                    continue

                # Upright aspect: bbox height (in image fraction) / bbox width ≥ MIN_ASPECT
                bbox_h_frac = bh / h
                bbox_w_frac = bw / w
                if bbox_w_frac > 0 and (bbox_h_frac / bbox_w_frac) < MIN_ASPECT:
                    continue

                xmin = max(0.0, round(xmin, 6))
                ymin = max(0.0, round(ymin, 6))
                xmax = min(1.0, round(xmax, 6))
                ymax = min(1.0, round(ymax, 6))
                qualifying_bboxes.append((norm_area, f"{xmin},{ymin},{xmax},{ymax}"))

            if not qualifying_bboxes:
                continue

            max_norm_area = max(a for a, _ in qualifying_bboxes)
            bbox_strs = [b for _, b in qualifying_bboxes]

            candidates[iid] = ImageCandidate(
                image_id=iid,
                split=split,
                file_name=meta["file_name"],
                license_id=meta.get("license", 0),
                indoor_score=img_indoor_count.get(iid, 0),
                max_norm_area=max_norm_area,
                person_bboxes=bbox_strs,
            )

        print(f"    {len(candidates)} candidates so far (after {split})")

    # Sort: outdoor first (indoor_score ASC), then most prominent person first (max_norm_area DESC)
    sorted_cands = sorted(candidates.values(), key=lambda c: (c.indoor_score, -c.max_norm_area))
    print(f"  Total candidates: {len(sorted_cands)}  (capped at {pool_size})")
    return sorted_cands[:pool_size]


# ── Post-download quality check ───────────────────────────────────────────────

def _quality_ok(dest_path: Path) -> tuple:
    """Return (True, '') or (False, reason). Mirrors heuristics in 1-filter_dataset_quality.py."""
    try:
        # Resolution
        bgr = cv2.imread(str(dest_path))
        if bgr is None:
            return False, "cv2 could not read image"
        h, w = bgr.shape[:2]
        if min(h, w) < MIN_SHORT_SIDE:
            return False, f"resolution {min(h,w)}px < {MIN_SHORT_SIDE}"

        # Blur (Laplacian variance on 512×512 grayscale thumbnail)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        thumb = cv2.resize(gray, (512, 512))
        lap_var = cv2.Laplacian(thumb, cv2.CV_64F).var()
        if lap_var < BLUR_THRESHOLD:
            return False, f"blurry (Laplacian var={lap_var:.1f} < {BLUR_THRESHOLD})"

        # Grayscale / IR (mean HSV saturation)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mean_sat = float(hsv[:, :, 1].mean())
        if mean_sat < MIN_SATURATION:
            return False, f"near-grayscale (mean sat={mean_sat:.1f} < {MIN_SATURATION})"

        return True, ""
    except Exception as e:
        return False, f"quality check error: {e}"


# ── Progress ──────────────────────────────────────────────────────────────────

def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"downloaded_ids": [], "failed_ids": [], "count": 0}


def save_progress(progress: dict):
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = PROGRESS_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(progress, f)
    os.replace(tmp, PROGRESS_FILE)


# ── Image download ────────────────────────────────────────────────────────────

def _image_url(candidate: ImageCandidate) -> str:
    base = TRAIN_IMAGE_BASE if candidate.split == "train2017" else VAL_IMAGE_BASE
    return f"{base}/{candidate.file_name}"


def download_image(url: str, dest_path: Path, timeout: int = 30) -> bool:
    if dest_path.exists():
        return True
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        if len(resp.content) < 100:
            return False
        save_as_jpg(resp.content, dest_path)
        return True
    except Exception:
        return False


# ── Catalog ───────────────────────────────────────────────────────────────────

def append_catalog_rows(rows):
    write_header = not CATALOG_FILE.exists() or CATALOG_FILE.stat().st_size == 0
    with open(CATALOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["source", "label", "filename", "license",
                             "scientific_name", "url", "search_term", "bbox"])
        for row in rows:
            writer.writerow(row)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download COCO human images with strict quality filters."
    )
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET,
                        help=f"Images to download (default {DEFAULT_TARGET})")
    parser.add_argument("--candidates", type=int, default=DEFAULT_CANDIDATES,
                        help=f"Candidate pool size (default {DEFAULT_CANDIDATES})")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Download threads (default {DEFAULT_WORKERS})")
    parser.add_argument("--reset", action="store_true",
                        help="Delete progress and catalog files and start fresh")
    args = parser.parse_args()

    if args.reset:
        for f in (PROGRESS_FILE, CATALOG_FILE):
            if f.exists():
                f.unlink()
                print(f"  Deleted {f.name}")

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    progress = load_progress()
    if progress["count"] >= args.target:
        print(f"Already downloaded {progress['count']} images — target reached.")
        return

    # Step 1: annotations
    print("Step 1/3: Annotations")
    train_path, val_path = download_annotations()

    # Step 2: candidates
    print("\nStep 2/3: Building candidate list")
    candidates = build_candidates(train_path, val_path, args.candidates)

    already_downloaded = set(str(i) for i in progress["downloaded_ids"])
    already_failed = set(str(i) for i in progress["failed_ids"])
    remaining = [
        c for c in candidates
        if str(c.image_id) not in already_downloaded and str(c.image_id) not in already_failed
    ]

    needed = args.target - progress["count"]
    to_process = remaining[: needed + FAILURE_BUFFER]
    print(f"  Need {needed} more. Submitting {len(to_process)} candidates.")

    if not to_process:
        print("No candidates remaining. Try increasing --candidates.")
        return

    # Signal handler
    def _flush(sig, frame):
        print("\nInterrupted — saving progress …")
        save_progress(progress)
        sys.exit(0)
    signal.signal(signal.SIGINT, _flush)
    signal.signal(signal.SIGTERM, _flush)

    # Step 3: download + quality check
    print(f"\nStep 3/3: Downloading (target={args.target}, workers={args.workers})")
    dirty = 0
    quality_rejected = 0

    def _worker(candidate: ImageCandidate):
        url = _image_url(candidate)
        dest = IMAGES_DIR / f"coco_{candidate.image_id}.jpg"
        if not download_image(url, dest):
            return False, "download_failed", candidate, url
        ok, reason = _quality_ok(dest)
        if not ok:
            dest.unlink(missing_ok=True)
            return False, reason, candidate, url
        return True, "", candidate, url

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_worker, c): c for c in to_process}
        with tqdm(total=len(futures), unit="img", desc="coco humans") as pbar:
            for future in as_completed(futures):
                ok, reason, cand, url = future.result()
                if ok:
                    progress["downloaded_ids"].append(str(cand.image_id))
                    progress["count"] += 1
                    lic = COCO_LICENSE_NAMES.get(cand.license_id, f"license-{cand.license_id}")
                    rows = [
                        ("coco_humans", "human", f"coco_{cand.image_id}.jpg",
                         lic, "homo sapiens", url, "person", bbox)
                        for bbox in cand.person_bboxes
                    ]
                    append_catalog_rows(rows)
                else:
                    progress["failed_ids"].append(str(cand.image_id))
                    if reason != "download_failed":
                        quality_rejected += 1

                dirty += 1
                if dirty >= PROGRESS_SAVE_INTERVAL:
                    save_progress(progress)
                    dirty = 0

                pbar.set_postfix(ok=progress["count"], qrej=quality_rejected,
                                 fail=len(progress["failed_ids"]))
                pbar.update(1)

                if progress["count"] >= args.target:
                    break

    save_progress(progress)
    print(
        f"\nDone."
        f"\n  Downloaded:        {progress['count']}"
        f"\n  Quality rejected:  {quality_rejected}"
        f"\n  Download failed:   {len(progress['failed_ids']) - quality_rejected}"
        f"\n  Images:   {IMAGES_DIR}"
        f"\n  Catalog:  {CATALOG_FILE}"
    )


if __name__ == "__main__":
    main()
