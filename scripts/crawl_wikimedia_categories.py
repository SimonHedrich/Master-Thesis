"""Crawl Wikimedia Commons category hierarchies for the 225 target labels.

Outputs one .txt file per label into reports/wikimedia_categories/, showing the
full category tree (3 levels deep by default) with file counts. The user can then
review and remove categories that aren't useful for training (anatomy, art, maps, etc.)
before using the curated files to download images.

Usage:
    python scripts/crawl_wikimedia_categories.py
    python scripts/crawl_wikimedia_categories.py --max-depth 2 --rate-limit 0.3
    python scripts/crawl_wikimedia_categories.py --resume

Requirements:
    pip install requests tqdm
"""

import argparse
import sys
import time
from pathlib import Path

import requests

# Import shared utilities from download_supplementary
sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_supplementary import (
    FAMILY_SPECIES_MAP,
    GENUS_SPECIES_MAP,
    LABELS_225,
    REPO_ROOT,
    USER_AGENT,
    WIKI_API,
    WIKIMEDIA_CATEGORY_OVERRIDES,
    RateLimiter,
    load_family_species_mapping,
    load_genus_species_mapping,
    load_target_labels,
    sanitize_dirname,
)

OUTPUT_DIR = REPO_ROOT / "reports" / "wikimedia_categories"


def _make_session():
    """Create a requests.Session with proper headers for Wikimedia."""
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    return session


def _api_get(session, params, rate_limiter, max_retries=8):
    """Make a rate-limited API request with retry logic for 429/maxlag/errors.

    Returns parsed JSON or None on failure.
    """
    params.setdefault("maxlag", 5)
    rate_limiter.wait()
    for attempt in range(max_retries):
        try:
            resp = session.get(WIKI_API, params=params, timeout=30)
            if resp.status_code == 429 or resp.status_code == 503:
                retry_after = resp.headers.get("Retry-After")
                wait = int(retry_after) if retry_after and retry_after.isdigit() else min(2 ** (attempt + 1), 60)
                print(f"    Rate limited ({resp.status_code}), waiting {wait}s...", flush=True)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError:
            wait = min(2 ** (attempt + 1), 60)
            time.sleep(wait)
            continue
        except Exception as e:
            print(f"    API error: {e}", flush=True)
            break
    return None


def get_category_info(session, category, rate_limiter):
    """Get file count for a category using prop=categoryinfo (single lightweight call).

    Returns file count (int) or 0 on failure.
    """
    params = {
        "action": "query",
        "titles": category,
        "prop": "categoryinfo",
        "format": "json",
    }
    data = _api_get(session, params, rate_limiter)
    if data is None:
        return 0
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        ci = page.get("categoryinfo", {})
        return ci.get("files", 0)
    return 0


def get_subcategories_with_counts(session, category, rate_limiter, max_pages=20):
    """Fetch subcategories AND their file counts in a single request using generators.

    Uses generator=categorymembers + prop=categoryinfo to get subcategories
    and their file counts in one API call per page.

    Returns list of (subcat_title, file_count) tuples.
    """
    results = []
    params = {
        "action": "query",
        "generator": "categorymembers",
        "gcmtitle": category,
        "gcmtype": "subcat",
        "gcmlimit": 500,
        "prop": "categoryinfo",
        "format": "json",
    }

    for _ in range(max_pages):
        data = _api_get(session, params, rate_limiter)
        if data is None:
            print(f"    Failed after retries: {category} (subcats+info)", flush=True)
            break

        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            title = page.get("title", "")
            ci = page.get("categoryinfo", {})
            file_count = ci.get("files", 0)
            results.append((title, file_count))

        # Check for continuation
        cont = data.get("continue")
        if cont and "gcmcontinue" in cont:
            params["gcmcontinue"] = cont["gcmcontinue"]
        else:
            break

    return results


def crawl_category_tree(session, category, rate_limiter, max_depth=3, current_depth=0,
                        visited=None, max_categories=200):
    """Recursively crawl a category tree.

    Returns list of (category_title, file_count, depth) tuples, in tree order.
    Uses combined generator queries: 1 API call per node instead of 2.
    """
    if visited is None:
        visited = set()

    # Avoid cycles and cap total categories
    if category in visited or len(visited) >= max_categories:
        return []
    visited.add(category)

    if current_depth < max_depth:
        # Get subcategories AND their file counts in one call
        subcats_with_counts = get_subcategories_with_counts(session, category, rate_limiter)
        # Get file count for the current category
        file_count = get_category_info(session, category, rate_limiter)
        results = [(category, file_count, current_depth)]

        for subcat_title, _ in subcats_with_counts:
            if len(visited) >= max_categories:
                break
            sub_results = crawl_category_tree(
                session, subcat_title, rate_limiter, max_depth, current_depth + 1,
                visited, max_categories,
            )
            results.extend(sub_results)
    else:
        # Leaf node: just get file count
        file_count = get_category_info(session, category, rate_limiter)
        results = [(category, file_count, current_depth)]

    return results


def build_root_categories(labels, genus_map, family_map=None):
    """Build root Wikimedia categories to crawl for each label.

    Returns dict: {common_name: {"categories": [...], "scientific": str, "genus": str, "species": str}}
    """
    if family_map is None:
        family_map = {}

    result = {}
    for entry in labels:
        cn = entry["common_name"]
        family = entry.get("family", "")
        genus = entry["genus"]
        species = entry["species"]
        categories = []

        # Check manual override first (colloquial/common-name categories on Commons)
        override_cats = WIKIMEDIA_CATEGORY_OVERRIDES.get(cn)
        if override_cats:
            categories = list(override_cats)

        if genus and species:
            # Species-level: Category:Genus_species (spaces → underscores for subspecies trinomials)
            cat = f"Category:{genus.capitalize()}_{species.lower().replace(' ', '_')}"
            if cat not in categories:
                categories.append(cat)
        elif genus:
            # Genus-level: Category:Genus
            categories.append(f"Category:{genus.capitalize()}")
        elif family and not genus and not species:
            # Family-level: Category:Family
            categories.append(f"Category:{family.capitalize()}")

        # For genus-level labels, also add species from genus_species_mapping.csv
        if cn in genus_map:
            for sp_row in genus_map[cn]:
                sci = sp_row["species_scientific"].strip()
                if sci:
                    parts = sci.split()
                    if len(parts) >= 2:
                        cat = f"Category:{parts[0].capitalize()}_{parts[1].lower()}"
                        if cat not in categories:
                            categories.append(cat)
        # Fallback: match by genus scientific name
        elif genus and not species:
            for label_key, species_list in genus_map.items():
                if species_list and species_list[0]["genus_scientific"].lower() == genus.lower():
                    for sp_row in species_list:
                        sci = sp_row["species_scientific"].strip()
                        if sci:
                            parts = sci.split()
                            if len(parts) >= 2:
                                cat = f"Category:{parts[0].capitalize()}_{parts[1].lower()}"
                                if cat not in categories:
                                    categories.append(cat)

        # For family-level labels, add species from family_species_mapping.csv
        if family and not genus and not species and cn in family_map:
            for sp_row in family_map[cn]:
                sci = sp_row["species_scientific"].strip()
                if sci:
                    parts = sci.split()
                    if len(parts) >= 2:
                        cat = f"Category:{parts[0].capitalize()}_{parts[1].lower()}"
                        if cat not in categories:
                            categories.append(cat)

        sci_name = f"{genus.capitalize()} {species.lower()}".strip() if genus else cn
        result[cn] = {
            "categories": categories,
            "scientific": sci_name,
            "genus": genus,
            "species": species,
            "dir_name": sanitize_dirname(cn),
        }

    return result


def write_hierarchy_file(output_path, common_name, scientific_name, tree_results):
    """Write the category tree to an indented .txt file."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# {common_name} | {scientific_name}\n")
        for cat_title, file_count, depth in tree_results:
            indent = "  " * depth
            f.write(f"{indent}{cat_title}  ({file_count} files)\n")


def main():
    parser = argparse.ArgumentParser(
        description="Crawl Wikimedia Commons category hierarchies for target labels",
    )
    parser.add_argument("--labels", type=str, default=str(LABELS_225),
                        help="Path to labels file")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                        help="Output directory for hierarchy files")
    parser.add_argument("--rate-limit", type=float, default=0.1,
                        help="Seconds between API calls (default: 0.4)")
    parser.add_argument("--max-depth", type=int, default=2,
                        help="Max category depth to crawl (default: 2)")
    parser.add_argument("--max-categories", type=int, default=5000,
                        help="Max categories to crawl per label (default: 100)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip labels that already have output files")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rate_limiter = RateLimiter(min_interval=args.rate_limit)
    session = _make_session()

    # Load data
    labels = load_target_labels(args.labels)
    genus_map = load_genus_species_mapping(GENUS_SPECIES_MAP)
    family_map = load_family_species_mapping(FAMILY_SPECIES_MAP)
    root_cats = build_root_categories(labels, genus_map, family_map)
    print(f"Loaded {len(root_cats)} labels")

    total_categories = 0
    skipped = 0

    sorted_labels = sorted(root_cats.keys())
    for i, cn in enumerate(sorted_labels):
        info = root_cats[cn]
        out_file = output_dir / f"{info['dir_name']}.txt"

        if args.resume and out_file.exists():
            skipped += 1
            print(f"Skipped {info['dir_name']}")
            continue

        print(f"[{i+1}/{len(sorted_labels)}] {cn} ({len(info['categories'])} root categories)", flush=True)

        if not info["categories"]:
            # No categories to crawl — write empty file with header
            write_hierarchy_file(out_file, cn, info["scientific"], [])
            continue

        # Crawl all root categories for this label
        all_results = []
        visited = set()
        for cat in info["categories"]:
            tree = crawl_category_tree(
                session, cat, rate_limiter, max_depth=args.max_depth,
                current_depth=0, visited=visited,
                max_categories=args.max_categories,
            )
            all_results.extend(tree)

        write_hierarchy_file(out_file, cn, info["scientific"], all_results)
        total_categories += len(all_results)
        print(f"  -> {len(all_results)} categories found", flush=True)

    print(f"\nDone. Crawled {total_categories} categories across {len(root_cats) - skipped} labels.")
    if skipped:
        print(f"Skipped {skipped} labels (already had output files, --resume).")
    print(f"Output: {output_dir}/")


if __name__ == "__main__":
    main()
