"""Crawl Wikimedia Commons category hierarchies for the 225 target labels.

Outputs one .txt file per label into reports/wikimedia_categories/, showing the
full category tree (3 levels deep by default) with file counts. The user can then
review and remove categories that aren't useful for training (anatomy, art, maps, etc.)
before using the curated files to download images.

Usage:
    python scripts/crawl_wikimedia_categories.py
    python scripts/crawl_wikimedia_categories.py --max-depth 2 --rate-limit 1.0
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
    GENUS_SPECIES_MAP,
    LABELS_225,
    REPO_ROOT,
    USER_AGENT,
    WIKI_API,
    RateLimiter,
    load_genus_species_mapping,
    load_target_labels,
    sanitize_dirname,
)

OUTPUT_DIR = REPO_ROOT / "reports" / "wikimedia_categories"


def get_category_members(category, cmtype, rate_limiter, max_pages=20):
    """Fetch all members of a Wikimedia Commons category (paginated).

    Args:
        category: Category title, e.g. "Category:Orycteropus_afer"
        cmtype: "subcat" for subcategories, "file" for files
        rate_limiter: RateLimiter instance
        max_pages: Safety cap on pagination

    Returns:
        List of member title strings.
    """
    all_titles = []
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": category,
        "cmtype": cmtype,
        "cmlimit": 500,
        "format": "json",
    }

    for _ in range(max_pages):
        rate_limiter.wait()
        data = None
        for attempt in range(8):
            try:
                resp = requests.get(WIKI_API, params=params, timeout=30,
                                    headers={"User-Agent": USER_AGENT})
                if resp.status_code == 429:
                    wait = min(2 ** (attempt + 1), 60)
                    print(f"    Rate limited, waiting {wait}s...", flush=True)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                break
            except requests.exceptions.HTTPError:
                wait = min(2 ** (attempt + 1), 60)
                time.sleep(wait)
                continue
            except Exception as e:
                print(f"    API error for {category} ({cmtype}): {e}", flush=True)
                break
        if data is None:
            print(f"    Failed after retries: {category} ({cmtype})", flush=True)
            break

        members = data.get("query", {}).get("categorymembers", [])
        all_titles.extend(m["title"] for m in members)

        # Check for continuation
        cont = data.get("continue")
        if cont and "cmcontinue" in cont:
            params["cmcontinue"] = cont["cmcontinue"]
        else:
            break

    return all_titles


def crawl_category_tree(category, rate_limiter, max_depth=3, current_depth=0,
                        visited=None, max_categories=200):
    """Recursively crawl a category tree.

    Returns list of (category_title, file_count, depth) tuples, in tree order.
    Stops after visiting max_categories to avoid spending too long on huge trees.
    """
    if visited is None:
        visited = set()

    # Avoid cycles and cap total categories
    if category in visited or len(visited) >= max_categories:
        return []
    visited.add(category)

    # Count files at this level
    files = get_category_members(category, "file", rate_limiter)
    file_count = len(files)

    results = [(category, file_count, current_depth)]

    # Recurse into subcategories if not at max depth
    if current_depth < max_depth:
        subcats = get_category_members(category, "subcat", rate_limiter)
        for subcat in subcats:
            if len(visited) >= max_categories:
                break
            sub_results = crawl_category_tree(
                subcat, rate_limiter, max_depth, current_depth + 1,
                visited, max_categories,
            )
            results.extend(sub_results)

    return results


def build_root_categories(labels, genus_map):
    """Build root Wikimedia categories to crawl for each label.

    Returns dict: {common_name: {"categories": [...], "scientific": str, "genus": str, "species": str}}
    """
    result = {}
    for entry in labels:
        cn = entry["common_name"]
        genus = entry["genus"]
        species = entry["species"]
        categories = []

        if genus and species:
            # Species-level: Category:Genus_species
            cat = f"Category:{genus.capitalize()}_{species.lower()}"
            categories.append(cat)
        elif genus:
            # Genus-level: Category:Genus
            categories.append(f"Category:{genus.capitalize()}")

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
    parser.add_argument("--rate-limit", type=float, default=1.0,
                        help="Seconds between API calls (default: 1.0)")
    parser.add_argument("--max-depth", type=int, default=2,
                        help="Max category depth to crawl (default: 2)")
    parser.add_argument("--max-categories", type=int, default=100,
                        help="Max categories to crawl per label (default: 100)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip labels that already have output files")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rate_limiter = RateLimiter(min_interval=args.rate_limit)

    # Load data
    labels = load_target_labels(args.labels)
    genus_map = load_genus_species_mapping(GENUS_SPECIES_MAP)
    root_cats = build_root_categories(labels, genus_map)
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
                cat, rate_limiter, max_depth=args.max_depth,
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
