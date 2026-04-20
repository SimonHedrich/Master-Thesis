"""Enumerate file titles from filtered Wikimedia Commons category trees.

Reads .txt files from reports/wikimedia_categories_filtered/, calls the
Wikimedia API to list all files in each category, and writes one .jsonl
manifest per label to reports/wikimedia_file_manifests/.

Each manifest line is a JSON record:
    {"title": "File:Foo.jpg", "category": "Category:Panthera_leo",
     "label": "lion", "scientific": "Panthera leo",
     "family": "Felidae", "genus": "Panthera", "species": "leo"}

By default the script resumes: labels whose .jsonl already exists are skipped.
Use --force to re-scrape everything.

Usage:
    python scripts/wikimedia/3-scrape_wikimedia_file_list.py
    python scripts/wikimedia/3-scrape_wikimedia_file_list.py --rate-limit 0.3
    python scripts/wikimedia/3-scrape_wikimedia_file_list.py --force

Requirements:
    pip install requests tqdm
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from download_supplementary import RateLimiter, USER_AGENT, WIKI_API

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FILTERED_DIR = REPO_ROOT / "reports" / "wikimedia_categories_filtered"
OUTPUT_DIR = REPO_ROOT / "reports" / "wikimedia_file_manifests"

_CATEGORY_RE = re.compile(r"^( *)Category:(.+?)\s+\((\d+) files?\)\s*$")


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
                print(f"  Rate limited ({resp.status_code}), waiting {wait}s…", flush=True)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError:
            time.sleep(min(2 ** (attempt + 1), 60))
        except Exception as e:
            print(f"  API error: {e}", flush=True)
            break
    return None


# ── Filtered .txt parser ──────────────────────────────────────────────────────

def parse_filtered_txt(path: Path):
    """Parse a filtered category .txt file.

    Returns:
        header: dict with keys label, scientific (from first comment line)
        categories: list of "Category:..." strings with file_count > 0
    """
    header = {"label": path.stem, "scientific": ""}
    categories = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            # Header line: "# Lion | Panthera leo"
            if line.startswith("#"):
                parts = line.lstrip("# ").strip().split("|")
                header["label"] = parts[0].strip().lower()
                header["scientific"] = parts[1].strip() if len(parts) > 1 else ""
                continue

            m = _CATEGORY_RE.match(line.rstrip("\n"))
            if m:
                file_count = int(m.group(3))
                if file_count > 0:
                    categories.append(f"Category:{m.group(2).strip()}")

    return header, categories


def _parse_label_fields(scientific: str):
    """Split 'Genus species' into genus/species; return family as empty (not in txt)."""
    parts = scientific.split()
    genus = parts[0] if parts else ""
    species = parts[1] if len(parts) > 1 else ""
    return genus, species


# ── Category file enumeration ─────────────────────────────────────────────────

def enumerate_files_in_category(session, category: str, rate_limiter, label_info: dict):
    """Paginate through all files in a Wikimedia category.

    Yields dicts ready to be serialised as JSONL records.
    """
    params = {
        "action": "query",
        "generator": "categorymembers",
        "gcmtitle": category,
        "gcmtype": "file",
        "gcmlimit": 500,
        "format": "json",
    }

    while True:
        data = _api_get(session, params, rate_limiter)
        if data is None:
            print(f"  Failed to fetch files for {category}", flush=True)
            break

        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            title = page.get("title", "")
            if title.startswith("File:"):
                yield {
                    "title": title,
                    "category": category,
                    "label": label_info["label"],
                    "scientific": label_info["scientific"],
                    "genus": label_info["genus"],
                    "species": label_info["species"],
                    "label_dir": label_info["label_dir"],
                }

        cont = data.get("continue", {})
        if "gcmcontinue" not in cont:
            break
        params["gcmcontinue"] = cont["gcmcontinue"]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Enumerate Wikimedia file titles from filtered category trees",
    )
    parser.add_argument("--filtered-dir", default=str(FILTERED_DIR),
                        help="Directory with filtered .txt category files")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR),
                        help="Output directory for .jsonl manifests")
    parser.add_argument("--rate-limit", type=float, default=0.2,
                        help="Minimum seconds between API calls (default: 0.2)")
    parser.add_argument("--force", action="store_true",
                        help="Re-scrape labels that already have a manifest")
    args = parser.parse_args()

    filtered_dir = Path(args.filtered_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(filtered_dir.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in {filtered_dir}")
        sys.exit(1)

    print(f"Found {len(txt_files)} filtered category files")

    session = _make_session()
    rate_limiter = RateLimiter(min_interval=args.rate_limit)

    total_files = 0
    skipped = 0

    for txt_path in tqdm(txt_files, desc="Labels", unit="label"):
        out_path = output_dir / (txt_path.stem + ".jsonl")

        if out_path.exists() and not args.force:
            skipped += 1
            continue

        header, categories = parse_filtered_txt(txt_path)
        if not categories:
            # Write empty manifest so this label is considered done
            out_path.write_text("")
            continue

        genus, species = _parse_label_fields(header["scientific"])
        label_info = {
            "label": header["label"],
            "scientific": header["scientific"],
            "genus": genus,
            "species": species,
            "label_dir": txt_path.stem,  # filesystem-safe name from filename
        }

        # Enumerate files, deduplicating by title within this label
        seen_titles = set()
        records = []

        for cat in categories:
            for rec in enumerate_files_in_category(session, cat, rate_limiter, label_info):
                if rec["title"] not in seen_titles:
                    seen_titles.add(rec["title"])
                    records.append(rec)

        # Write manifest atomically via temp file
        tmp_path = out_path.with_suffix(".jsonl.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        tmp_path.rename(out_path)

        total_files += len(records)
        tqdm.write(f"  {txt_path.stem}: {len(categories)} categories → {len(records)} files")

    print(f"\nDone. {total_files:,} file titles enumerated across {len(txt_files) - skipped} labels.")
    if skipped:
        print(f"Skipped {skipped} labels (manifest already exists; use --force to re-scrape).")
    print(f"Manifests: {output_dir}/")


if __name__ == "__main__":
    main()
