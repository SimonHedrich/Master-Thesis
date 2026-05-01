"""
Download iNaturalist Homo sapiens images for the human class.

Operates independently of 1-download_inaturalist.py. Reuses the metadata CSVs
already on disk (taxa.csv, observations.csv, photos.csv) but maintains its own
progress file and does NOT touch filtered_images_225.json, filter_results.jsonl,
attribution_manifest.csv, or any other existing pipeline state.

iNaturalist human observations are ideal for this use case: nature enthusiasts
photograph people they encounter in the field, so images are outdoor, person-
centred, and taken at natural-setting distances — unlike COCO where humans are
incidental background elements in urban scenes.

Output:
    data/inaturalist/images/human/    ← inat_{photo_id}.jpg
    data/inaturalist/download_progress_human.json
    data/inaturalist/attribution_manifest_human.csv  (CC-BY only)

Usage:
    python scripts/inaturalist/2-download_inat_humans.py
    python scripts/inaturalist/2-download_inat_humans.py --target 500           # quick test
    python scripts/inaturalist/2-download_inat_humans.py --size large           # 1024 px
    python scripts/inaturalist/2-download_inat_humans.py --research-grade-only  # stricter quality
"""

import argparse
import csv
import json
import os
import random
import signal
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
INAT_DIR = REPO_ROOT / "data" / "inaturalist"
METADATA_DIR = INAT_DIR / "metadata"
IMAGES_DIR = INAT_DIR / "images" / "human"
PROGRESS_FILE = INAT_DIR / "download_progress_human.json"
ATTRIBUTION_FILE = INAT_DIR / "attribution_manifest_human.csv"

# ── iNaturalist config ────────────────────────────────────────────────────────

S3_BASE_URL = "https://inaturalist-open-data.s3.amazonaws.com"
IMAGE_SIZES = {"original": 2048, "large": 1024, "medium": 500, "small": 240}
SAFE_LICENSES = {"cc0", "cc-by"}

HOMO_SAPIENS_NAME = "Homo sapiens"

USER_AGENT = (
    "MasterThesis-WildlifeDetection/1.0 "
    "(wildlife-detection-research) python-requests"
)

DEFAULT_TARGET = 2000
DEFAULT_WORKERS = 8
DEFAULT_SIZE = "large"   # 1024 px — better quality than COCO medium-res images
RANDOM_SEED = 42
PROGRESS_SAVE_INTERVAL = 50


# ── Metadata helpers ──────────────────────────────────────────────────────────

def find_homo_sapiens_taxon_ids(taxa_path: Path) -> set:
    """Return all taxon_ids that are Homo sapiens or its subspecies."""
    print("  Scanning taxa.csv for Homo sapiens …")
    hs_id = None
    all_taxa = {}

    with open(taxa_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            tid = row["taxon_id"]
            name = row.get("name", "").strip()
            rank = row.get("rank", "").strip()
            ancestry = row.get("ancestry", "")
            active = row.get("active", "t")
            all_taxa[tid] = {"name": name, "rank": rank, "ancestry": ancestry, "active": active}
            if name.lower() == HOMO_SAPIENS_NAME.lower() and rank == "species":
                hs_id = tid
                print(f"    Found Homo sapiens: taxon_id={tid}")

    if hs_id is None:
        raise RuntimeError(
            "Could not find 'Homo sapiens' (rank=species) in taxa.csv. "
            "Re-run `1-download_inaturalist.py metadata` to refresh."
        )

    # Include Homo sapiens itself and any subspecies (ancestry contains hs_id)
    target_ids = {hs_id}
    for tid, t in all_taxa.items():
        if hs_id in t["ancestry"].split("/") and t["active"] == "t":
            target_ids.add(tid)

    print(f"    Target taxon_ids: {len(target_ids)} (species + subspecies)")
    return target_ids


def collect_observation_uuids(obs_path: Path, taxon_ids: set, research_grade_only: bool) -> set:
    """Stream observations.csv and return UUIDs matching our taxon_ids."""
    print("  Streaming observations.csv …")
    matched = set()
    total = 0

    with open(obs_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            total += 1
            if row.get("taxon_id") in taxon_ids:
                if research_grade_only and row.get("quality_grade") != "research":
                    continue
                matched.add(row["observation_uuid"])
            if total % 10_000_000 == 0:
                print(f"    … {total:,} rows scanned, {len(matched):,} matched")

    print(f"    {total:,} observations scanned → {len(matched):,} Homo sapiens observations")
    return matched


def collect_photos(photos_path: Path, obs_uuids: set, size: str) -> list:
    """Stream photos.csv and return photo records for matched observations."""
    print("  Streaming photos.csv …")
    photos = []
    total = 0

    with open(photos_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            total += 1
            if row.get("observation_uuid") not in obs_uuids:
                if total % 50_000_000 == 0:
                    print(f"    … {total:,} rows scanned, {len(photos):,} kept")
                continue

            license_str = (row.get("license") or "").strip().lower()
            if license_str not in SAFE_LICENSES:
                continue

            photo_id = row.get("photo_id", "")
            ext = row.get("extension", "jpg") or "jpg"
            photos.append({
                "photo_id": photo_id,
                "observation_uuid": row["observation_uuid"],
                "license": license_str,
                "extension": ext,
                "url": f"{S3_BASE_URL}/photos/{photo_id}/{size}.{ext}",
            })

            if total % 50_000_000 == 0:
                print(f"    … {total:,} rows scanned, {len(photos):,} kept")

    print(f"    {total:,} photos scanned → {len(photos):,} CC0/CC-BY Homo sapiens photos")
    return photos


# ── Download ──────────────────────────────────────────────────────────────────

def download_image(url: str, dest_path: Path, timeout: int = 30) -> bool:
    if dest_path.exists():
        return True
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        if len(resp.content) < 500:
            return False
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest_path.with_suffix(".tmp")
        try:
            tmp.write_bytes(resp.content)
            tmp.rename(dest_path)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        return True
    except Exception:
        return False


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


# ── Attribution ───────────────────────────────────────────────────────────────

def append_attribution(entries: list):
    """Append CC-BY photo records to the human-specific attribution file."""
    if not entries:
        return
    write_header = not ATTRIBUTION_FILE.exists() or ATTRIBUTION_FILE.stat().st_size == 0
    with open(ATTRIBUTION_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["photo_id", "observation_uuid", "license", "image_url"])
        for e in entries:
            writer.writerow([e["photo_id"], e["observation_uuid"], e["license"], e["url"]])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download iNaturalist Homo sapiens images (standalone, non-destructive)."
    )
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET,
                        help=f"Images to download (default {DEFAULT_TARGET})")
    parser.add_argument("--size", choices=list(IMAGE_SIZES.keys()), default=DEFAULT_SIZE,
                        help=f"Image size (default {DEFAULT_SIZE} = {IMAGE_SIZES[DEFAULT_SIZE]}px)")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Download threads (default {DEFAULT_WORKERS})")
    parser.add_argument("--research-grade-only", action="store_true",
                        help="Only include research-grade observations (higher quality, fewer results)")
    args = parser.parse_args()

    # Validate metadata files exist
    for fname in ("taxa.csv", "observations.csv", "photos.csv"):
        p = METADATA_DIR / fname
        if not p.exists():
            print(f"ERROR: {p} not found.")
            print("  Run: python scripts/inaturalist/1-download_inaturalist.py metadata")
            sys.exit(1)

    progress = load_progress()
    if progress["count"] >= args.target:
        print(f"Already downloaded {progress['count']} images — target reached.")
        return

    already_downloaded = set(progress["downloaded_ids"])
    already_failed = set(progress["failed_ids"])

    # Signal handler: flush on interrupt
    def _flush(sig, frame):
        print("\nInterrupted — saving progress …")
        save_progress(progress)
        sys.exit(0)

    signal.signal(signal.SIGINT, _flush)
    signal.signal(signal.SIGTERM, _flush)

    # ── Step 1: Find taxon IDs ────────────────────────────────────────────
    print("\nStep 1/3: Locate Homo sapiens in taxonomy")
    taxon_ids = find_homo_sapiens_taxon_ids(METADATA_DIR / "taxa.csv")

    # ── Step 2: Collect candidate photos ─────────────────────────────────
    print("\nStep 2/3: Collect candidate photos")
    obs_uuids = collect_observation_uuids(
        METADATA_DIR / "observations.csv", taxon_ids, args.research_grade_only
    )
    all_photos = collect_photos(METADATA_DIR / "photos.csv", obs_uuids, args.size)

    # Shuffle for diversity, then exclude already-processed IDs
    random.seed(RANDOM_SEED)
    random.shuffle(all_photos)
    candidates = [
        p for p in all_photos
        if p["photo_id"] not in already_downloaded and p["photo_id"] not in already_failed
    ]

    needed = args.target - progress["count"]
    print(f"\n  Need {needed} more. {len(candidates)} unseen candidates available.")

    if not candidates:
        print("No candidates remaining. The dataset may be exhausted for this target size.")
        return

    # ── Step 3: Download ──────────────────────────────────────────────────
    print(f"\nStep 3/3: Downloading (target={args.target}, size={args.size}, workers={args.workers})")
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    to_process = candidates[:needed + 500]  # buffer for failures
    dirty = 0
    attribution_buffer = []

    def _worker(photo: dict):
        ext = photo["extension"]
        dest = IMAGES_DIR / f"inat_{photo['photo_id']}.{ext}"
        ok = download_image(photo["url"], dest)
        return ok, photo

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_worker, p): p for p in to_process}
        with tqdm(total=len(futures), unit="img", desc="inat humans") as pbar:
            for future in as_completed(futures):
                ok, photo = future.result()
                if ok:
                    progress["downloaded_ids"].append(photo["photo_id"])
                    progress["count"] += 1
                    if photo["license"] == "cc-by":
                        attribution_buffer.append(photo)
                else:
                    progress["failed_ids"].append(photo["photo_id"])

                dirty += 1
                if dirty >= PROGRESS_SAVE_INTERVAL:
                    save_progress(progress)
                    append_attribution(attribution_buffer)
                    attribution_buffer.clear()
                    dirty = 0

                pbar.set_postfix(ok=progress["count"], fail=len(progress["failed_ids"]))
                pbar.update(1)

                if progress["count"] >= args.target:
                    break

    save_progress(progress)
    append_attribution(attribution_buffer)

    print(
        f"\nDone. Downloaded: {progress['count']}  Failed: {len(progress['failed_ids'])}"
        f"\nImages:  {IMAGES_DIR}"
    )
    if ATTRIBUTION_FILE.exists():
        print(f"CC-BY attribution: {ATTRIBUTION_FILE}")


if __name__ == "__main__":
    main()
