"""
Map LILA BC taxonomy entries to the 225-class target label set.

Produces reports/lila_matched_taxa_225.csv — analogous to
reports/inaturalist_matched_taxa_225.csv — with one row per
(dataset_name, query) LILA category entry that resolves to a
target class, recording how the match was made.

Matching priority (highest to lowest):
  1. species   — LILA 'species' field matches a species-level target
  2. genus     — LILA 'genus' field matches a genus-level target label
                 OR matches via genus_species_mapping (any species in that genus)
  3. family    — LILA 'family' field matches a family-level target label
                 OR matches via family_species_mapping
  4. common    — LILA 'query' (common name) matches a target common_name

Usage:
    python scripts/map_lila_taxonomy.py
"""

import csv
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

LILA_TAXONOMY = REPO / "data/lila_bc/metadata/lila-taxonomy-mapping_release.csv"
CLASS_COUNTS   = REPO / "reports/class_counts_225.csv"
GENUS_MAP      = REPO / "reports/genus_species_mapping.csv"
FAMILY_MAP     = REPO / "reports/family_species_mapping.csv"
OUTPUT         = REPO / "reports/lila_matched_taxa_225.csv"


# ── Load target label lookups ─────────────────────────────────────────────────

def load_targets():
    """Return lookup dicts keyed by lowercase scientific name / common name."""
    species_to_class = {}   # 'genus species' -> common_name
    genus_to_class   = {}   # 'genus'         -> common_name  (genus-level labels)
    family_to_class  = {}   # 'family'         -> common_name  (family-level labels)
    common_to_class  = {}   # 'common name'   -> common_name

    with open(CLASS_COUNTS) as f:
        for row in csv.DictReader(f):
            name  = row["common_name"].strip()
            sci   = row["scientific_name"].lower().strip()
            level = row["level"].strip()
            common_to_class[name.lower()] = name
            if level == "species":
                species_to_class[sci] = name
            elif level == "genus":
                genus_to_class[sci] = name      # sci IS the genus name
            elif level == "family":
                family_to_class[sci] = name     # sci IS the family name

    return species_to_class, genus_to_class, family_to_class, common_to_class


def load_genus_species_map():
    """Return genus_scientific (lower) -> genus_label for all genus-level entries."""
    m = {}
    with open(GENUS_MAP) as f:
        for row in csv.DictReader(f):
            g = row["genus_scientific"].lower().strip()
            if g:
                m[g] = row["genus_label"].strip()
    return m


def load_family_species_map():
    """Return family_scientific (lower) -> family_label for all family-level entries."""
    m = {}
    with open(FAMILY_MAP) as f:
        for row in csv.DictReader(f):
            fam = row["family_scientific"].lower().strip()
            if fam:
                m[fam] = row["family_label"].strip()
    return m


# ── Matching ──────────────────────────────────────────────────────────────────

def match_row(row, species_to_class, genus_to_class, family_to_class,
              common_to_class, genus_map, family_map):
    """Return (target_class, match_type) or (None, None)."""

    sci_species = row["species"].lower().strip()
    sci_genus   = row["genus"].lower().strip()
    sci_family  = row["family"].lower().strip()
    query       = row["query"].lower().strip()
    sci_name    = row["scientific_name"].lower().strip()

    # 1. Exact species match
    if sci_species and sci_species in species_to_class:
        return species_to_class[sci_species], "species"

    # Also try the full scientific_name field (covers subspecies rows where
    # 'species' col may be the subspecies string)
    if sci_name and sci_name in species_to_class:
        return species_to_class[sci_name], "species"

    # Also try the query field as a scientific name — WCS uses scientific names
    # as category labels (e.g. "canis mesomelas"), which may be older synonyms
    # that differ from the resolved scientific_name in the taxonomy CSV.
    if query and query in species_to_class:
        return species_to_class[query], "species_synonym"

    # 2. Genus-level target label (direct)
    if sci_genus and sci_genus in genus_to_class:
        return genus_to_class[sci_genus], "genus"

    # 3. Genus via genus_species_mapping (genus is a member of a genus-label)
    if sci_genus and sci_genus in genus_map:
        return genus_map[sci_genus], "genus_via_mapping"

    # 4. Family-level target label (direct)
    if sci_family and sci_family in family_to_class:
        return family_to_class[sci_family], "family"

    # 5. Family via family_species_mapping
    if sci_family and sci_family in family_map:
        return family_map[sci_family], "family_via_mapping"

    # 6. Common-name fallback (query string matches a target common name)
    if query and query in common_to_class:
        return common_to_class[query], "common_name"

    return None, None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    species_to_class, genus_to_class, family_to_class, common_to_class = load_targets()
    genus_map  = load_genus_species_map()
    family_map = load_family_species_map()

    print(f"Target lookups — species: {len(species_to_class)}, "
          f"genus: {len(genus_to_class)}, family: {len(family_to_class)}")
    print(f"Genus-species mapping entries: {len(genus_map)}")
    print(f"Family-species mapping entries: {len(family_map)}")

    results = []
    skipped = 0
    match_type_counts = defaultdict(int)

    with open(LILA_TAXONOMY) as f:
        reader = csv.DictReader(f)
        for row in reader:
            target_class, match_type = match_row(
                row,
                species_to_class, genus_to_class, family_to_class,
                common_to_class, genus_map, family_map,
            )
            if target_class is None:
                skipped += 1
                continue

            results.append({
                "dataset_name":    row["dataset_name"].strip(),
                "query":           row["query"].strip(),
                "scientific_name": row["scientific_name"].strip(),
                "rank":            row["taxonomy_level"].strip(),
                "target_class":    target_class,
                "match_type":      match_type,
            })
            match_type_counts[match_type] += 1

    # Deduplicate: same (scientific_name, target_class) across datasets is fine —
    # keep all rows since the per-dataset entry matters for the download script.
    # But deduplicate exact duplicates (same dataset + query appearing twice).
    seen = set()
    deduped = []
    for r in results:
        key = (r["dataset_name"], r["query"])
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    deduped.sort(key=lambda r: (r["target_class"], r["dataset_name"], r["query"]))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset_name", "query", "scientific_name", "rank",
                        "target_class", "match_type"],
        )
        writer.writeheader()
        writer.writerows(deduped)

    print(f"\nWrote {len(deduped)} rows to {OUTPUT}")
    print(f"Skipped (no match): {skipped}")
    print("\nMatch type breakdown:")
    for mt, cnt in sorted(match_type_counts.items(), key=lambda x: -x[1]):
        print(f"  {mt:<25} {cnt}")

    # Coverage report
    covered = set(r["target_class"] for r in deduped)
    with open(CLASS_COUNTS) as f:
        all_classes = {row["common_name"] for row in csv.DictReader(f)}
    uncovered = all_classes - covered
    print(f"\nTarget classes covered: {len(covered)} / {len(all_classes)}")
    if uncovered:
        print(f"Uncovered ({len(uncovered)}):")
        for name in sorted(uncovered):
            print(f"  - {name}")


if __name__ == "__main__":
    main()
