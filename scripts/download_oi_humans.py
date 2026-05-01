"""
Download human images from Open Images V7 with strict quality filters.

Selects images where:
  - At least one "person" bbox covers >= MIN_PERSON_AREA of the image (person is a main subject)
  - At most MAX_PERSONS person bboxes (no crowds)
  - No vehicle annotations (car, bus, truck, motorcycle, bicycle, van, taxi, train, boat, airplane)
  - No group-of or depiction annotations on the qualifying person bbox
Sorted by largest person bbox area (most prominent subjects first).

Output (consistent with existing Open Images structure):
    data/openimages/images/human/      ← oi_{image_id}.jpg
    data/openimages/metadata_catalog.csv  ← appended (same schema as existing rows)
    data/openimages/download_progress_human.json

Does NOT touch filter_results.jsonl or any other existing pipeline state.

Usage:
    python scripts/download_oi_humans.py
    python scripts/download_oi_humans.py --target 500 --workers 4   # quick test
"""

import argparse
import csv
import json
import os
import signal
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _image_utils import save_as_jpg

# ── Paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
OI_DIR = REPO_ROOT / "data" / "openimages"
METADATA_DIR = OI_DIR / "metadata" / "openimages"
IMAGES_DIR = OI_DIR / "images" / "human"
CATALOG_FILE = OI_DIR / "metadata_catalog.csv"
PROGRESS_FILE = OI_DIR / "download_progress_human.json"

OI_IMAGE_BASE = "https://s3.amazonaws.com/open-images-dataset/train"

USER_AGENT = (
    "MasterThesis-WildlifeDetection/1.0 "
    "(wildlife-detection-research) python-requests"
)

# ── OI metadata URLs (only needed if files are missing) ───────────────────────

OI_CLASS_DESC_URL = "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv"
OI_BBOX_URL = "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv"

# ── MID constants (looked up from class-descriptions.csv) ─────────────────────

# All person-related labels treated as "person" for filtering
PERSON_DISPLAY_NAMES = {"person", "man", "woman", "boy", "girl", "child", "human"}

# Any of these in an image → hard exclude
VEHICLE_DISPLAY_NAMES = {
    "car", "bus", "truck", "motorcycle", "bicycle",
    "van", "taxi", "train", "boat", "airplane", "vehicle",
}

# ── Filter thresholds ─────────────────────────────────────────────────────────

MIN_PERSON_AREA = 0.05   # person bbox must cover >= 5 % of image area
MAX_PERSONS = 2          # at most 2 person bboxes per image (no crowds)
FAILURE_BUFFER = 500

DEFAULT_TARGET = 2000
DEFAULT_CANDIDATES = 2500
DEFAULT_WORKERS = 8
PROGRESS_SAVE_INTERVAL = 50


# ── Metadata helpers ──────────────────────────────────────────────────────────

def _download_if_missing(url: str, dest: Path):
    if dest.exists():
        return
    print(f"  Downloading {dest.name} …")
    resp = requests.get(url, stream=True, headers={"User-Agent": USER_AGENT}, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as pbar:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            pbar.update(len(chunk))


def load_mids(class_desc_path: Path) -> tuple[set, set]:
    """Return (person_mids, vehicle_mids) by matching display names."""
    person_mids, vehicle_mids = set(), set()
    with open(class_desc_path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            mid, name = row[0].strip(), row[1].strip().lower()
            if name in PERSON_DISPLAY_NAMES:
                person_mids.add(mid)
            elif name in VEHICLE_DISPLAY_NAMES:
                vehicle_mids.add(mid)
    print(f"  Person MIDs: {person_mids}")
    print(f"  Vehicle MIDs ({len(vehicle_mids)}): {vehicle_mids}")
    return person_mids, vehicle_mids


# ── Candidate building ────────────────────────────────────────────────────────

def build_candidates(bbox_path: Path, person_mids: set, vehicle_mids: set, pool_size: int) -> list:
    """
    Stream bbox CSV once and return sorted candidate list.

    Each candidate: {"image_id": str, "max_area": float, "bboxes": [bbox_str, ...]}
    Sorted by max_area descending (most prominent person first).
    """
    print(f"  Streaming {bbox_path.name} …")

    # Per-image accumulators
    person_bboxes: dict = defaultdict(list)   # image_id -> list of (area, bbox_str, is_group, is_depiction)
    has_vehicle: set = set()

    row_count = 0
    all_mids = person_mids | vehicle_mids

    with open(bbox_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_count += 1
            mid = row["LabelName"]
            if mid not in all_mids:
                if row_count % 5_000_000 == 0:
                    print(f"    … {row_count:,} rows")
                continue

            iid = row["ImageID"]

            if mid in vehicle_mids:
                has_vehicle.add(iid)
                continue

            # Person annotation
            try:
                xmin = float(row["XMin"])
                xmax = float(row["XMax"])
                ymin = float(row["YMin"])
                ymax = float(row["YMax"])
            except (ValueError, KeyError):
                continue

            area = (xmax - xmin) * (ymax - ymin)
            is_group = row.get("IsGroupOf", "0").strip() == "1"
            is_depiction = row.get("IsDepiction", "0").strip() == "1"
            bbox_str = f"{xmin},{ymin},{xmax},{ymax}"

            person_bboxes[iid].append((area, bbox_str, is_group, is_depiction))

            if row_count % 5_000_000 == 0:
                print(f"    … {row_count:,} rows, {len(person_bboxes):,} images with persons")

    print(f"  {row_count:,} rows scanned")
    print(f"  {len(person_bboxes):,} images with person annotations")
    print(f"  {len(has_vehicle):,} images have vehicle annotations (will be excluded)")

    # Filter and score
    candidates = []
    for iid, anns in person_bboxes.items():
        if iid in has_vehicle:
            continue

        # Qualifying bboxes: area >= threshold, not a group-of, not a depiction
        qualifying = [
            (area, bbox_str)
            for area, bbox_str, is_group, is_depiction in anns
            if area >= MIN_PERSON_AREA and not is_group and not is_depiction
        ]
        if not qualifying:
            continue

        # Total person bbox count (including small ones) must not exceed MAX_PERSONS
        if len(anns) > MAX_PERSONS:
            continue

        max_area = max(a for a, _ in qualifying)
        bboxes = [b for _, b in qualifying]
        candidates.append({"image_id": iid, "max_area": max_area, "bboxes": bboxes})

    print(f"  {len(candidates):,} candidates after filtering")

    # Sort by largest person area (most prominent subject first)
    candidates.sort(key=lambda c: c["max_area"], reverse=True)
    return candidates[:pool_size]


# ── Progress ──────────────────────────────────────────────────────────────────

def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"downloaded_ids": [], "failed_ids": [], "count": 0}


def save_progress(progress: dict):
    tmp = PROGRESS_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(progress, f)
    os.replace(tmp, PROGRESS_FILE)


# ── Catalog ───────────────────────────────────────────────────────────────────

def append_catalog_rows(rows: list):
    write_header = not CATALOG_FILE.exists() or CATALOG_FILE.stat().st_size == 0
    with open(CATALOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["source", "label", "filename", "license",
                             "scientific_name", "url", "search_term", "bbox"])
        for row in rows:
            writer.writerow(row)


# ── Download ──────────────────────────────────────────────────────────────────

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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download Open Images human images (strict filters).")
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET,
                        help=f"Images to download (default {DEFAULT_TARGET})")
    parser.add_argument("--candidates", type=int, default=DEFAULT_CANDIDATES,
                        help=f"Candidate pool size (default {DEFAULT_CANDIDATES})")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Download threads (default {DEFAULT_WORKERS})")
    args = parser.parse_args()

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    progress = load_progress()
    if progress["count"] >= args.target:
        print(f"Already downloaded {progress['count']} images — target reached.")
        return

    # Ensure metadata files exist
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    _download_if_missing(OI_CLASS_DESC_URL, METADATA_DIR / "class-descriptions.csv")
    _download_if_missing(OI_BBOX_URL, METADATA_DIR / "train-annotations-bbox.csv")

    # Signal handler
    def _flush(sig, frame):
        print("\nInterrupted — saving progress …")
        save_progress(progress)
        sys.exit(0)
    signal.signal(signal.SIGINT, _flush)
    signal.signal(signal.SIGTERM, _flush)

    # Step 1: MIDs
    print("Step 1/3: Load class MIDs")
    person_mids, vehicle_mids = load_mids(METADATA_DIR / "class-descriptions.csv")

    # Step 2: Candidates
    print("\nStep 2/3: Build candidate list")
    candidates = build_candidates(
        METADATA_DIR / "train-annotations-bbox.csv",
        person_mids, vehicle_mids, args.candidates,
    )

    already_downloaded = set(progress["downloaded_ids"])
    already_failed = set(progress["failed_ids"])
    remaining = [
        c for c in candidates
        if c["image_id"] not in already_downloaded and c["image_id"] not in already_failed
    ]

    needed = args.target - progress["count"]
    to_process = remaining[: needed + FAILURE_BUFFER]
    print(f"\n  Need {needed} more. Submitting {len(to_process)} candidates.")

    if not to_process:
        print("No candidates remaining.")
        return

    # Step 3: Download
    print(f"\nStep 3/3: Downloading (target={args.target}, workers={args.workers})")
    dirty = 0

    def _worker(candidate: dict):
        iid = candidate["image_id"]
        url = f"{OI_IMAGE_BASE}/{iid}.jpg"
        dest = IMAGES_DIR / f"oi_{iid}.jpg"
        ok = download_image(url, dest)
        return ok, candidate, url

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_worker, c): c for c in to_process}
        with tqdm(total=len(futures), unit="img", desc="OI humans") as pbar:
            for future in as_completed(futures):
                ok, cand, url = future.result()
                iid = cand["image_id"]
                if ok:
                    progress["downloaded_ids"].append(iid)
                    progress["count"] += 1
                    rows = [
                        ("openimages", "human", f"oi_{iid}.jpg", "CC-BY-2.0",
                         "homo sapiens", url, "person", bbox)
                        for bbox in cand["bboxes"]
                    ]
                    append_catalog_rows(rows)
                else:
                    progress["failed_ids"].append(iid)

                dirty += 1
                if dirty >= PROGRESS_SAVE_INTERVAL:
                    save_progress(progress)
                    dirty = 0

                pbar.set_postfix(ok=progress["count"], fail=len(progress["failed_ids"]))
                pbar.update(1)

                if progress["count"] >= args.target:
                    break

    save_progress(progress)
    print(
        f"\nDone. Downloaded: {progress['count']}  Failed: {len(progress['failed_ids'])}"
        f"\nImages:  {IMAGES_DIR}"
        f"\nCatalog: {CATALOG_FILE}"
    )


if __name__ == "__main__":
    main()
