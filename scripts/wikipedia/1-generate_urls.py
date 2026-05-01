"""
Generate best-guess Wikipedia URL seeds for all 225 wildlife classes.

Input:  reports/classes_225.csv
Output: reports/wikipedia_urls.json

URL strategy:
- species  → title-cased common name  (e.g. "african elephant" → African_elephant)
- genus    → capitalized scientific name  (e.g. "dasyprocta" → Dasyprocta)
- family   → capitalized scientific name  (e.g. "cricetidae" → Cricetidae)
"""
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = REPO_ROOT / "reports" / "classes_225.csv"
OUT_PATH = REPO_ROOT / "reports" / "wikipedia_urls.json"
WIKI_BASE = "https://en.wikipedia.org/wiki/"

# Suffixes appended to common names of genus/family entries (not part of article title)
_LEVEL_SUFFIXES = {" genus", " species", " family", " clade"}


def _wiki_url(title: str) -> str:
    return WIKI_BASE + title.replace(" ", "_")


def _common_to_title(common_name: str) -> str:
    """Title-case common name, stripping taxonomic level suffixes."""
    name = common_name
    for suffix in _LEVEL_SUFFIXES:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name.title()


def make_key(scientific_name: str) -> str:
    return scientific_name.replace(" ", "_")


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    result: dict = {}

    for _, row in df.iterrows():
        common = str(row["common_name"]).strip()
        sci = str(row["scientific_name"]).strip()
        level = str(row["level"]).strip()
        key = make_key(sci)

        if level == "species":
            url_title = _common_to_title(common)
        else:
            # Genus and family articles live under the scientific name
            url_title = sci.capitalize()

        result[key] = {
            "common_name": common,
            "scientific_name": sci,
            "level": level,
            "wikipedia_url": _wiki_url(url_title),
            "wikipedia_file": f"{key}.txt",
            # Populated by script 2 for genus/family; null means "not applicable"
            "top_species": None if level == "species" else [],
        }

    OUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {len(result)} entries → {OUT_PATH}")

    n_genus = sum(1 for v in result.values() if v["level"] == "genus")
    n_family = sum(1 for v in result.values() if v["level"] == "family")
    n_species = sum(1 for v in result.values() if v["level"] == "species")
    print(f"  species={n_species}  genus={n_genus}  family={n_family}")


if __name__ == "__main__":
    main()
