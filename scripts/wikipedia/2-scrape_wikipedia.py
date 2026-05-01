"""
Scrape Wikipedia pages for all entries in reports/wikipedia_urls.json.

For each entry:
  1. Resolve the guessed URL (follows redirects, tries scientific name fallback,
     then Wikipedia search as last resort).
  2. Fetch plain-text content via the MediaWiki action API.
  3. Save text to data/wikipedia/{scientific_name}.txt.

For genus / family entries:
  4. Query iNaturalist to find the top-5 most-observed species within that taxon,
     excluding species that are already listed as standalone entries in
     classes_225.csv (the overlap exclusion rule: e.g. ocelot must not appear
     under the leopardus genus).
  5. Resolve and save Wikipedia pages for those top species too.
  6. Populate the "top_species" list in the JSON.

Outputs:
  reports/wikipedia_urls.json  — updated with canonical URLs and top_species
  data/wikipedia/*.txt         — plain-text Wikipedia articles
  data/wikipedia/missing_pages.txt — pages that could not be found (for manual review)

Rate limits:
  Wikipedia action API: ~200 req/min  → 0.35 s sleep
  iNaturalist v1 API:  ~100 req/min  → 0.65 s sleep
"""
import json
import time
import urllib.parse
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
URLS_JSON = REPO_ROOT / "reports" / "wikipedia_urls.json"
CSV_PATH = REPO_ROOT / "reports" / "classes_225.csv"
GENUS_MAP_CSV = REPO_ROOT / "reports" / "genus_species_mapping.csv"
FAMILY_MAP_CSV = REPO_ROOT / "reports" / "family_species_mapping.csv"
OUT_DIR = REPO_ROOT / "data" / "wikipedia"

WIKI_ACTION = "https://en.wikipedia.org/w/api.php"
WIKI_BASE = "https://en.wikipedia.org/wiki/"
INAT_API = "https://api.inaturalist.org/v1"

HEADERS = {
    "User-Agent": (
        "master-thesis-wildlife-wikipedia/1.0 "
        "(simon.hedrich@inovex.de; academic research)"
    )
}
WIKI_SLEEP = 0.35  # seconds between Wikipedia requests
INAT_SLEEP = 0.65  # seconds between iNaturalist requests
TOP_N = 5          # max species per genus/family


# ---------------------------------------------------------------------------
# Wikipedia helpers
# ---------------------------------------------------------------------------

def _wiki_fetch(title: str) -> tuple[str | None, str | None]:
    """
    Fetch a Wikipedia article by title (redirects followed automatically).

    Returns (canonical_url, plain_text) or (None, None) if the page does not
    exist.  Uses the MediaWiki action API with prop=extracts|info so that one
    HTTP round-trip gives us both the text and the canonical URL.
    """
    params = {
        "action": "query",
        "prop": "extracts|info",
        "explaintext": "true",
        "inprop": "url",
        "titles": title,
        "format": "json",
        "redirects": "true",
    }
    try:
        r = requests.get(WIKI_ACTION, params=params, headers=HEADERS, timeout=30)
        r.raise_for_status()
    except requests.RequestException:
        return None, None
    finally:
        time.sleep(WIKI_SLEEP)

    pages = r.json().get("query", {}).get("pages", {})
    for page_id, page in pages.items():
        if page_id == "-1":
            return None, None
        resolved_title = page.get("title", title)
        canonical_url = page.get("fullurl") or (
            WIKI_BASE + resolved_title.replace(" ", "_")
        )
        text = page.get("extract") or ""
        return canonical_url, text

    return None, None


def _wiki_search(query: str) -> str | None:
    """Search Wikipedia and return the title of the first result, or None."""
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": "1",
        "srnamespace": "0",
        "format": "json",
    }
    try:
        r = requests.get(WIKI_ACTION, params=params, headers=HEADERS, timeout=30)
        r.raise_for_status()
    except requests.RequestException:
        return None
    finally:
        time.sleep(WIKI_SLEEP)

    results = r.json().get("query", {}).get("search", [])
    return results[0]["title"] if results else None


def resolve_and_fetch(entry: dict) -> tuple[str | None, str | None]:
    """
    Try up to three strategies to find a Wikipedia page for an entry.

    Returns (canonical_url, plain_text) or (None, None).
    """
    sci = entry["scientific_name"]
    common = entry["common_name"]
    guessed_title = entry["wikipedia_url"].split("/wiki/", 1)[-1]
    guessed_title = urllib.parse.unquote(guessed_title).replace("_", " ")

    # Attempt 1: guessed title (common name or scientific name)
    url, text = _wiki_fetch(guessed_title)
    if url:
        return url, text

    # Attempt 2: scientific name (capitalized, spaces → underscores)
    sci_title = sci.replace("_", " ").capitalize()
    if sci_title.lower() != guessed_title.lower():
        url, text = _wiki_fetch(sci_title)
        if url:
            return url, text

    # Attempt 3: Wikipedia full-text search
    found_title = _wiki_search(f"{common} {sci}")
    if found_title:
        url, text = _wiki_fetch(found_title)
        if url:
            return url, text

    return None, None


# ---------------------------------------------------------------------------
# iNaturalist helpers
# ---------------------------------------------------------------------------

def _inat_request(params: dict) -> dict:
    try:
        r = requests.get(f"{INAT_API}/taxa", params=params, headers=HEADERS, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return {}
    finally:
        time.sleep(INAT_SLEEP)


def _inat_taxon_id(name: str, rank: str) -> int | None:
    """Return the iNaturalist taxon ID for a name+rank, or None."""
    data = _inat_request({"q": name, "rank": rank, "per_page": "5"})
    for t in data.get("results", []):
        if t.get("name", "").lower() == name.lower():
            return t["id"]
    results = data.get("results", [])
    return results[0]["id"] if results else None


def get_top_species(
    sci_name: str,
    level: str,
    exclusion_set: set[str],
    genus_df: pd.DataFrame,
    family_df: pd.DataFrame,
) -> list[dict]:
    """
    Return up to TOP_N species for a genus or family entry, sorted by iNaturalist
    observation count.

    Strategy (same for genus and family):
      1. Resolve the taxon's iNaturalist ID.
      2. Query /v1/taxa?taxon_id={id}&rank=species — iNaturalist returns all
         species within that clade, sorted by observation count.
      3. Exclude species already listed as standalone entries in classes_225.csv.

    Fallback: mapping CSVs (genus_species_mapping / family_species_mapping),
    used when the iNaturalist query returns no results.
    """
    # --- iNaturalist primary path ---
    taxon_id = _inat_taxon_id(sci_name, level)
    if taxon_id:
        data = _inat_request({"taxon_id": taxon_id, "rank": "species", "per_page": "30"})
        candidates = [
            {
                "name": t["name"].lower(),
                "common_name": t.get("preferred_common_name", ""),
                "observations_count": t.get("observations_count", 0),
            }
            for t in data.get("results", [])
        ]
        candidates = [c for c in candidates if c["name"] not in exclusion_set]
        if candidates:
            return candidates[:TOP_N]

    # --- Fallback: mapping CSVs ---
    if level == "genus":
        mask = genus_df["genus_scientific"].str.lower() == sci_name.lower()
        rows = genus_df[mask].dropna(subset=["species_scientific"])
    else:
        mask = family_df["family_scientific"].str.lower() == sci_name.lower()
        rows = family_df[mask].dropna(subset=["species_scientific"])

    seen: set[str] = set()
    candidates = []
    for _, row in rows.iterrows():
        sp = str(row["species_scientific"]).lower()
        if sp not in seen and sp not in exclusion_set:
            seen.add(sp)
            candidates.append({
                "name": sp,
                "common_name": str(row.get("species_common_name", "") or ""),
                "observations_count": 0,
            })

    return candidates[:TOP_N]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    url_data: dict = json.loads(URLS_JSON.read_text())
    df_classes = pd.read_csv(CSV_PATH)
    genus_df = pd.read_csv(GENUS_MAP_CSV)
    family_df = pd.read_csv(FAMILY_MAP_CSV)

    # Species already listed as standalone entries — excluded from genus/family expansion
    exclusion_set: set[str] = {
        str(r["scientific_name"]).strip().lower()
        for _, r in df_classes[df_classes["level"] == "species"].iterrows()
    }

    missing: list[str] = []

    entries = list(url_data.items())
    pbar = tqdm(entries, desc="Wikipedia pages")

    for key, entry in pbar:
        pbar.set_postfix_str(entry["common_name"][:30])
        level = entry["level"]

        # --- Resolve and fetch main article ---
        canonical_url, text = resolve_and_fetch(entry)

        if canonical_url:
            entry["wikipedia_url"] = canonical_url
            out_file = OUT_DIR / entry["wikipedia_file"]
            out_file.write_text(text or "", encoding="utf-8")
        else:
            missing.append(
                f"{key} | {entry['common_name']} | {entry['wikipedia_url']}"
            )

        # --- Genus / family: find and fetch top species ---
        if level in ("genus", "family") and entry["top_species"] is not None:
            sci_name = entry["scientific_name"]
            top = get_top_species(
                sci_name, level, exclusion_set, genus_df, family_df
            )

            species_records = []
            for sp in top:
                sp_sci = sp["name"]
                sp_common = sp["common_name"]
                sp_key = sp_sci.replace(" ", "_")

                # Construct best-guess Wikipedia URL for species
                sp_title = sp_common.title() if sp_common else sp_sci.capitalize()
                sp_url = WIKI_BASE + sp_title.replace(" ", "_")

                sp_entry = {
                    "common_name": sp_common,
                    "scientific_name": sp_sci,
                    "level": "species",
                    "wikipedia_url": sp_url,
                    "wikipedia_file": f"{sp_key}.txt",
                    "top_species": None,
                }

                sp_canonical, sp_text = resolve_and_fetch(sp_entry)

                if sp_canonical:
                    sp_entry["wikipedia_url"] = sp_canonical
                    sp_file = OUT_DIR / sp_entry["wikipedia_file"]
                    sp_file.write_text(sp_text or "", encoding="utf-8")
                else:
                    missing.append(
                        f"{sp_key} | {sp_common or sp_sci} | {sp_url}"
                    )

                species_records.append(
                    {
                        "scientific_name": sp_sci,
                        "common_name": sp_common,
                        "wikipedia_url": sp_entry["wikipedia_url"],
                        "wikipedia_file": sp_entry["wikipedia_file"],
                        "inat_observations": sp["observations_count"],
                        "wikipedia_found": sp_canonical is not None,
                    }
                )

            entry["top_species"] = species_records

    # Write updated JSON
    URLS_JSON.write_text(json.dumps(url_data, indent=2, ensure_ascii=False) + "\n")
    print(f"\nUpdated {URLS_JSON}")

    # Write missing pages report
    missing_path = OUT_DIR / "missing_pages.txt"
    if missing:
        missing_path.write_text("\n".join(missing) + "\n", encoding="utf-8")
        print(f"Missing pages ({len(missing)}) → {missing_path}")
    else:
        missing_path.write_text("# No missing pages\n", encoding="utf-8")
        print("All pages found.")

    found = sum(1 for e in url_data.values() if (OUT_DIR / e["wikipedia_file"]).exists())
    print(f"Saved {found}/{len(url_data)} main articles to {OUT_DIR}/")


if __name__ == "__main__":
    main()
