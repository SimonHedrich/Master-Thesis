"""Download Wikimedia Commons images from file-list manifests.

Reads .jsonl manifests from reports/wikimedia_file_manifests/ (produced by
scrape_wikimedia_file_list.py), fetches full image metadata via the Wikimedia
API in batches of 50, filters by license, downloads images, and appends rows
to data/wikimedia/metadata.csv.

Images are saved to:
    data/wikimedia/images/{label_dir}/{original_filename}

Original Wikimedia filenames are preserved (spaces replaced with underscores).

Resume behaviour (default): files that already exist on disk are skipped.
The metadata.csv is opened in append mode; already-downloaded images will not
get duplicate rows as long as the file exists when the script starts.

Usage:
    python scripts/download_wikimedia_images.py
    python scripts/download_wikimedia_images.py --rate-limit 0.5 --min-width 400
    python scripts/download_wikimedia_images.py --manifest-dir reports/wikimedia_file_manifests

Requirements:
    pip install requests tqdm
"""

import argparse
import csv
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_supplementary import RateLimiter, USER_AGENT, WIKI_API, WIKI_SAFE_LICENSES

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_DIR = REPO_ROOT / "reports" / "wikimedia_file_manifests"
OUTPUT_DIR = REPO_ROOT / "data" / "wikimedia"
IMAGES_DIR = OUTPUT_DIR / "images"
METADATA_CSV = OUTPUT_DIR / "metadata.csv"

METADATA_FIELDS = [
    "filename", "title", "url", "description_url",
    "label", "scientific_name", "genus", "species",
    "wikimedia_category", "label_dir",
    "width", "height", "mime", "size_bytes",
    "upload_timestamp", "uploader",
    "license_short", "license_url",
    "artist", "image_description", "date_taken",
    "gps_lat", "gps_lon",
]


# ── API helpers ───────────────────────────────────────────────────────────────

def _make_session():
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    return session


def _api_get(session, params, rate_limiter, max_retries=8):
    """Rate-limited GET with exponential back-off on 429/503."""
    params.setdefault("maxlag", 5)
    rate_limiter.wait()
    for attempt in range(max_retries):
        try:
            resp = session.get(WIKI_API, params=params, timeout=30)
            if resp.status_code in (429, 503):
                retry_after = resp.headers.get("Retry-After")
                wait = int(retry_after) if retry_after and retry_after.isdigit() else min(2 ** (attempt + 1), 60)
                tqdm.write(f"  Rate limited ({resp.status_code}), waiting {wait}s…")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError:
            time.sleep(min(2 ** (attempt + 1), 60))
        except Exception as e:
            tqdm.write(f"  API error: {e}")
            break
    return None


# ── Manifest loading ──────────────────────────────────────────────────────────

def load_manifests(manifest_dir: Path):
    """Load all .jsonl manifests. Returns list of record dicts.

    Deduplicates by title globally: the first label that claims a title wins.
    """
    records = []
    seen_titles = set()

    for jsonl_path in sorted(manifest_dir.glob("*.jsonl")):
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                title = rec.get("title", "")
                if title and title not in seen_titles:
                    seen_titles.add(title)
                    records.append(rec)

    return records


# ── Title → filename ──────────────────────────────────────────────────────────

def title_to_filename(title: str) -> str:
    """Convert 'File:Foo bar.jpg' → 'Foo_bar.jpg'."""
    name = title.removeprefix("File:")
    return name.replace(" ", "_")


# ── imageinfo fetch ───────────────────────────────────────────────────────────

def fetch_imageinfo_batch(session, titles: list[str], rate_limiter: RateLimiter):
    """Fetch imageinfo + extmetadata for up to 50 file titles.

    Returns dict: title → info_dict (or None if unavailable/filtered).
    """
    params = {
        "action": "query",
        "titles": "|".join(titles),
        "prop": "imageinfo",
        "iiprop": "url|extmetadata|size|mime|timestamp|user|canonicaltitle",
        "format": "json",
    }
    data = _api_get(session, params, rate_limiter)
    if data is None:
        return {}

    results = {}
    pages = data.get("query", {}).get("pages", {})
    for page_id, page in pages.items():
        if int(page_id) < 0:
            continue
        title = page.get("title", "")
        ii_list = page.get("imageinfo", [])
        if not ii_list:
            continue
        info = ii_list[0]
        results[title] = info

    return results


def extract_extmetadata(ext: dict, key: str) -> str:
    """Safely extract a string value from extmetadata."""
    return ext.get(key, {}).get("value", "") or ""


def sanitize_csv_field(value: str) -> str:
    """Remove newlines and escape commas so the value is safe in a CSV cell."""
    value = value.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    value = value.replace(",", "\\,")
    return value.strip()


# ── Image download ────────────────────────────────────────────────────────────

def download_image(url: str, dest: Path, timeout: int = 60) -> bool:
    """Download a single image. Returns True on success."""
    if dest.exists():
        return True
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        if len(resp.content) < 100:
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Write to .tmp first to avoid partial files on interruption
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        tmp.write_bytes(resp.content)
        tmp.rename(dest)
        return True
    except Exception:
        return False


# ── CSV writer ────────────────────────────────────────────────────────────────

class MetadataCatalog:
    """Append-mode CSV writer for image metadata."""

    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self._write_header = not csv_path.exists()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(csv_path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._f, fieldnames=METADATA_FIELDS, extrasaction="ignore")
        if self._write_header:
            self._writer.writeheader()

    def append(self, row: dict):
        self._writer.writerow(row)
        self._f.flush()

    def close(self):
        self._f.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download Wikimedia images and save full metadata",
    )
    parser.add_argument("--manifest-dir", default=str(MANIFEST_DIR),
                        help="Directory with .jsonl manifests from scrape_wikimedia_file_list.py")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR),
                        help="Root output directory (images/ and metadata.csv go here)")
    parser.add_argument("--rate-limit", type=float, default=0.0,
                        help="Minimum seconds between API calls (default: 0.5)")
    parser.add_argument("--min-width", type=int, default=150,
                        help="Skip images narrower than this (default: 300 px)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel download threads (default: 4, set to 1 to disable)")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Ignore the failed-titles log and retry previously failed downloads")
    args = parser.parse_args()

    manifest_dir = Path(args.manifest_dir)
    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    metadata_csv = output_dir / "metadata.csv"
    failed_log = output_dir / "failed_titles.txt"

    # Load all manifests
    print("Loading manifests…", flush=True)
    all_records = load_manifests(manifest_dir)
    if not all_records:
        print(f"No records found in {manifest_dir}. Run scrape_wikimedia_file_list.py first.")
        sys.exit(1)
    print(f"  {len(all_records):,} unique file titles across all labels")

    # Load previously failed filenames so we don't waste API calls on them
    failed_filenames: set[str] = set()
    if not args.retry_failed and failed_log.exists():
        failed_filenames = {line.strip() for line in failed_log.read_text(encoding="utf-8").splitlines() if line.strip()}
        if failed_filenames:
            print(f"  {len(failed_filenames):,} previously failed titles skipped (use --retry-failed to retry)")

    # Determine which files still need downloading
    pending = []
    already_done = 0
    skipped_failed = 0
    for rec in all_records:
        filename = title_to_filename(rec["title"])
        dest = images_dir / rec["label_dir"] / filename
        if dest.exists():
            already_done += 1
        elif filename in failed_filenames:
            skipped_failed += 1
        else:
            pending.append(rec)

    print(f"  {already_done:,} already downloaded, {skipped_failed:,} skipped (failed), {len(pending):,} pending")

    if not pending:
        print("Nothing to download.")
        return

    session = _make_session()
    rate_limiter = RateLimiter(min_interval=args.rate_limit)
    catalog = MetadataCatalog(metadata_csv)
    catalog_lock = threading.Lock()

    skipped_license = 0
    skipped_size = 0
    downloaded = 0
    failed = 0

    def _norm_title(t: str) -> str:
        return t.replace("_", " ")

    def _download_one(dest: Path, url: str, row: dict) -> bool:
        ok = download_image(url, dest)
        with catalog_lock:
            if ok:
                catalog.append(row)
            else:
                with open(failed_log, "a", encoding="utf-8") as f:
                    f.write(row["filename"] + "\n")
        return ok

    workers = max(1, args.workers)
    parallel = workers > 1
    mode_str = f"parallel, {workers} workers" if parallel else "sequential"

    # Process in batches of 50 (Wikimedia API limit for multi-title queries).
    # Each batch fetches metadata, downloads images, and writes to CSV before
    # moving on — so progress is saved continuously and restarts skip already-
    # downloaded files without redoing any API work.
    batch_size = 50
    batches = [pending[i:i + batch_size] for i in range(0, len(pending), batch_size)]

    print(f"Fetching metadata and downloading ({mode_str})…", flush=True)
    with tqdm(total=len(pending), desc="Progress", unit="file") as pbar:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for batch in batches:
                titles = [rec["title"] for rec in batch]
                rec_by_title = {_norm_title(rec["title"]): rec for rec in batch}

                info_map = fetch_imageinfo_batch(session, titles, rate_limiter)

                # Build download queue for this batch
                batch_queue: list[tuple[Path, str, dict]] = []
                for title, info in info_map.items():
                    rec = rec_by_title.get(_norm_title(title))
                    if rec is None:
                        pbar.update(1)
                        continue

                    mime = info.get("mime", "")
                    if mime not in ("image/jpeg", "image/png", "image/webp", "image/tiff", "image/gif"):
                        pbar.update(1)
                        continue

                    width = info.get("width", 0)
                    if width < args.min_width:
                        skipped_size += 1
                        pbar.update(1)
                        continue

                    ext = info.get("extmetadata", {})
                    lic_short = extract_extmetadata(ext, "LicenseShortName").strip()
                    if not lic_short or lic_short.lower() not in WIKI_SAFE_LICENSES:
                        skipped_license += 1
                        pbar.update(1)
                        continue

                    url = info.get("url", "")
                    if not url:
                        pbar.update(1)
                        continue

                    filename = title_to_filename(title)
                    dest = images_dir / rec["label_dir"] / filename

                    if dest.exists():
                        pbar.update(1)
                        continue

                    row = {
                        "filename": filename,
                        "title": title,
                        "url": url,
                        "description_url": info.get("descriptionurl", ""),
                        "label": rec["label"],
                        "scientific_name": rec["scientific"],
                        "genus": rec["genus"],
                        "species": rec["species"],
                        "wikimedia_category": rec["category"],
                        "label_dir": rec["label_dir"],
                        "width": width,
                        "height": info.get("height", ""),
                        "mime": mime,
                        "size_bytes": info.get("size", ""),
                        "upload_timestamp": info.get("timestamp", ""),
                        "uploader": info.get("user", ""),
                        "license_short": lic_short,
                        "license_url": extract_extmetadata(ext, "LicenseUrl"),
                        "artist": sanitize_csv_field(extract_extmetadata(ext, "Artist")),
                        "image_description": sanitize_csv_field(extract_extmetadata(ext, "ImageDescription")),
                        "date_taken": sanitize_csv_field(extract_extmetadata(ext, "DateTimeOriginal")),
                        "gps_lat": extract_extmetadata(ext, "GPSLatitude"),
                        "gps_lon": extract_extmetadata(ext, "GPSLongitude"),
                    }
                    batch_queue.append((dest, url, row))

                # Titles not returned by the API
                returned_titles = set(info_map.keys())
                for rec in batch:
                    if rec["title"] not in returned_titles:
                        pbar.update(1)

                # Log all batch titles that weren't queued (not returned by API,
                # wrong MIME, too small, bad license, empty URL) so they're
                # skipped on re-runs instead of re-fetched every time.
                queued_filenames = {row["filename"] for _, _, row in batch_queue}
                with open(failed_log, "a", encoding="utf-8") as flog:
                    for rec in batch:
                        filename = title_to_filename(rec["title"])
                        dest = images_dir / rec["label_dir"] / filename
                        if filename not in queued_filenames and not dest.exists():
                            flog.write(filename + "\n")

                # Download this batch (parallel or sequential), then continue
                if parallel:
                    futures = {
                        executor.submit(_download_one, dest, url, row): (dest, url)
                        for dest, url, row in batch_queue
                    }
                    for future in as_completed(futures):
                        try:
                            ok = future.result()
                        except Exception:
                            ok = False
                        if ok:
                            downloaded += 1
                        else:
                            failed += 1
                        pbar.update(1)
                else:
                    for dest, url, row in batch_queue:
                        ok = _download_one(dest, url, row)
                        if ok:
                            downloaded += 1
                        else:
                            failed += 1
                        pbar.update(1)

    catalog.close()

    print(f"\nDone.")
    print(f"  Downloaded : {downloaded:,}")
    print(f"  License-filtered : {skipped_license:,}")
    print(f"  Too small (<{args.min_width}px) : {skipped_size:,}")
    print(f"  Failed : {failed:,} (logged to {failed_log})")
    print(f"  Images  : {images_dir}/")
    print(f"  Metadata: {metadata_csv}")


if __name__ == "__main__":
    main()
