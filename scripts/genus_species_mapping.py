"""
Create mappings from genus- and family-level labels to their constituent species.

For the 37 genus-level and 12 family-level labels in the 225-class student model
label set, this script resolves the specific species within each group. This is
useful when searching for training images — searching by specific species name
yields far more results than searching by genus or family alone.

Data sources (in order of priority):
    1. SpeciesNet taxonomy (resources/speciesnet_taxonomy_release.txt) — local
    2. iNaturalist taxa.csv (data/inaturalist/metadata/taxa.csv) — local, ~186 MB
    3. iNaturalist API (optional) — adds observation counts per species

Usage:
    # Basic: SpeciesNet + iNaturalist taxa.csv (no network)
    python scripts/genus_species_mapping.py

    # With observation counts from iNaturalist API
    python scripts/genus_species_mapping.py --include-inat-counts

    # Custom output paths
    python scripts/genus_species_mapping.py --output reports/my_genus_mapping.csv
    python scripts/genus_species_mapping.py --family-output reports/my_family_mapping.csv
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    requests = None

# ── Paths ────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
LABELS_225 = REPO_ROOT / "resources" / "2026-03-19_student_model_labels.txt"
SPECIESNET_TAXONOMY = REPO_ROOT / "resources" / "speciesnet_taxonomy_release.txt"
INAT_TAXA = REPO_ROOT / "data" / "inaturalist" / "metadata" / "taxa.csv"
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "genus_species_mapping.csv"
DEFAULT_FAMILY_OUTPUT = REPO_ROOT / "reports" / "family_species_mapping.csv"

# ── Related Genera ───────────────────────────────────────────────────────────
# Some genus labels consolidate multiple taxonomic genera. Derived from the
# consolidation decisions in docs/species-label-selection.md.

RELATED_GENERA = {
    "tamias": ["neotamias", "eutamias"],          # chipmunks
    "presbytis": ["trachypithecus", "semnopithecus"],  # leaf monkeys / langurs
    "colobus": ["piliocolobus"],                  # colobus monkeys
    "cebus": ["sapajus"],                         # capuchins
    "saguinus": ["leontocebus"],                  # tamarins
    "eulemur": ["hapalemur", "prolemur", "propithecus", "microcebus"],  # lemurs
    "callithrix": ["mico"],                       # marmosets
    "cercocebus": ["lophocebus"],                 # mangabeys
    "cercopithecus": ["allochrocebus"],           # guenons
    "martes": ["pekania"],                        # martens (fisher = pekania)
    "mustela": ["putorius"],                      # weasels / polecats
}

# ── Label Loading ────────────────────────────────────────────────────────────


def load_genus_labels(label_path):
    """Load the label file and return genus-level entries.

    Returns:
        dict: {genus_name: common_name} for all genus-level labels.
    """
    genera = {}
    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(";")
            if len(parts) < 7:
                continue
            genus, species, common_name = parts[4], parts[5], parts[6]
            if genus and not species:
                genera[genus.lower()] = common_name
    return genera


def load_family_labels(label_path):
    """Load the label file and return family-level entries.

    Family-level entries have a family set but genus and species empty.

    Returns:
        dict: {family_name: common_name} for all family-level labels.
    """
    families = {}
    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(";")
            if len(parts) < 7:
                continue
            family, genus, species, common_name = parts[3], parts[4], parts[5], parts[6]
            if family and not genus and not species:
                families[family.lower()] = common_name
    return families


# ── SpeciesNet Taxonomy ──────────────────────────────────────────────────────


def query_speciesnet(target_genera):
    """Find species within target genera from the SpeciesNet taxonomy.

    Args:
        target_genera: set of genus names (lowercase) to search for.

    Returns:
        list of dicts with keys: genus, species, common_name.
    """
    results = []
    with open(SPECIESNET_TAXONOMY) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(";")
            if len(parts) < 7:
                continue
            genus, species, common_name = parts[4].lower(), parts[5], parts[6]
            if genus in target_genera and species:
                results.append({
                    "genus": genus,
                    "species": f"{genus} {species}".lower(),
                    "common_name": common_name,
                })
    return results


def query_speciesnet_families(target_families):
    """Find species within target families from the SpeciesNet taxonomy.

    Args:
        target_families: set of family names (lowercase) to search for.

    Returns:
        list of dicts with keys: family, genus, species, common_name.
    """
    results = []
    with open(SPECIESNET_TAXONOMY) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(";")
            if len(parts) < 7:
                continue
            family = parts[3].lower()
            genus, species, common_name = parts[4].lower(), parts[5], parts[6]
            if family in target_families and species:
                results.append({
                    "family": family,
                    "genus": genus,
                    "species": f"{genus} {species}".lower(),
                    "common_name": common_name,
                })
    return results


# ── iNaturalist taxa.csv ─────────────────────────────────────────────────────


def query_inat_taxa(target_genera):
    """Find species within target genera from iNaturalist taxa.csv.

    Parses the large taxa.csv to find genus taxon_ids, then finds all
    species-rank children whose ancestry includes that genus.

    Args:
        target_genera: set of genus names (lowercase) to search for.

    Returns:
        list of dicts with keys: genus, species, common_name, taxon_id.
    """
    if not INAT_TAXA.exists():
        print(f"  [skip] iNaturalist taxa.csv not found at {INAT_TAXA}")
        return []

    print(f"  Parsing {INAT_TAXA.name} (~186 MB)...")

    # Pass 1: find taxon_ids for target genera
    genus_taxon_ids = {}  # taxon_id -> genus_name
    all_rows = []

    with open(INAT_TAXA) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            name = row["name"].lower()
            rank = row["rank"]
            taxon_id = row["taxon_id"]
            active = row.get("active", "t")

            if active == "f":
                continue

            if rank == "genus" and name in target_genera:
                genus_taxon_ids[taxon_id] = name

            all_rows.append(row)

    print(f"  Found {len(genus_taxon_ids)} genus taxon_ids out of {len(target_genera)} target genera")

    # Pass 2: find species whose ancestry includes a target genus
    results = []
    for row in all_rows:
        if row["rank"] != "species":
            continue
        if row.get("active", "t") == "f":
            continue

        ancestry = row.get("ancestry", "")
        if not ancestry:
            continue

        ancestor_ids = ancestry.split("/")
        for aid in ancestor_ids:
            if aid in genus_taxon_ids:
                genus_name = genus_taxon_ids[aid]
                results.append({
                    "genus": genus_name,
                    "species": row["name"].lower(),
                    "common_name": "",  # taxa.csv doesn't have common names
                    "taxon_id": row["taxon_id"],
                })
                break

    print(f"  Found {len(results)} species across matched genera")
    return results


def query_inat_taxa_families(target_families):
    """Find species within target families from iNaturalist taxa.csv.

    Args:
        target_families: set of family names (lowercase) to search for.

    Returns:
        list of dicts with keys: family, genus, species, common_name, taxon_id.
    """
    if not INAT_TAXA.exists():
        print(f"  [skip] iNaturalist taxa.csv not found at {INAT_TAXA}")
        return []

    print(f"  Parsing {INAT_TAXA.name} (~186 MB)...")

    # Pass 1: find taxon_ids for target families, and genus taxon_ids for genus lookup
    family_taxon_ids = {}  # taxon_id -> family_name
    genus_taxon_ids = {}   # taxon_id -> genus_name
    all_rows = []

    with open(INAT_TAXA) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            name = row["name"].lower()
            rank = row["rank"]
            taxon_id = row["taxon_id"]
            active = row.get("active", "t")

            if active == "f":
                continue

            if rank == "family" and name in target_families:
                family_taxon_ids[taxon_id] = name

            if rank == "genus":
                genus_taxon_ids[taxon_id] = name

            all_rows.append(row)

    print(f"  Found {len(family_taxon_ids)} family taxon_ids out of {len(target_families)} target families")

    # Pass 2: find species whose ancestry includes a target family
    results = []
    for row in all_rows:
        if row["rank"] != "species":
            continue
        if row.get("active", "t") == "f":
            continue

        ancestry = row.get("ancestry", "")
        if not ancestry:
            continue

        ancestor_ids = ancestry.split("/")
        for aid in ancestor_ids:
            if aid in family_taxon_ids:
                family_name = family_taxon_ids[aid]
                # Resolve genus from the direct parent (last ancestor before species)
                genus_name = ""
                if ancestor_ids:
                    parent_id = ancestor_ids[-1]
                    genus_name = genus_taxon_ids.get(parent_id, "")
                results.append({
                    "family": family_name,
                    "genus": genus_name,
                    "species": row["name"].lower(),
                    "common_name": "",
                    "taxon_id": row["taxon_id"],
                })
                break

    print(f"  Found {len(results)} species across matched families")
    return results


# ── iNaturalist API (optional) ───────────────────────────────────────────────

INAT_API_BASE = "https://api.inaturalist.org/v1"
CACHE_PATH = REPO_ROOT / "reports" / ".genus_species_cache.json"


def load_cache():
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def query_inat_api(genus_name):
    """Query iNaturalist API for species within a genus, with observation counts.

    Returns:
        list of dicts with keys: species, common_name, observation_count.
    """
    if requests is None:
        print("  [skip] requests library not installed, cannot query iNat API")
        return []

    url = f"{INAT_API_BASE}/taxa"
    params = {
        "taxon_name": genus_name,
        "rank": "species",
        "per_page": 200,
        "order": "desc",
        "order_by": "observations_count",
    }

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    results = []
    for taxon in data.get("results", []):
        # Verify this taxon is actually in the target genus
        if taxon.get("rank") != "species":
            continue
        sci_name = taxon.get("name", "").lower()
        if not sci_name.startswith(genus_name):
            continue
        results.append({
            "species": sci_name,
            "common_name": (taxon.get("preferred_common_name") or "").lower(),
            "observation_count": taxon.get("observations_count", 0),
        })

    return results


def fetch_inat_counts(genera_list):
    """Fetch observation counts for all genera, with caching.

    Returns:
        dict: {species_name: {"common_name": str, "observation_count": int}}
    """
    cache = load_cache()
    species_data = {}

    for genus in genera_list:
        if genus in cache:
            for entry in cache[genus]:
                species_data[entry["species"]] = entry
            continue

        print(f"  Querying iNat API for {genus}...")
        try:
            results = query_inat_api(genus)
            cache[genus] = results
            for entry in results:
                species_data[entry["species"]] = entry
            time.sleep(1.1)  # Rate limit: ~60 req/min
        except Exception as e:
            print(f"  [warn] API query failed for {genus}: {e}")

    save_cache(cache)
    return species_data


# ── Merge & Output ───────────────────────────────────────────────────────────


def build_mapping(genera, include_inat_counts=False):
    """Build the genus-to-species mapping from all sources.

    Args:
        genera: dict {genus_name: common_name} from the label set.
        include_inat_counts: whether to query iNat API for observation counts.

    Returns:
        list of row dicts for the output CSV.
    """
    # Expand target genera to include related genera
    all_target_genera = set()
    genus_to_label = {}  # maps actual genus -> label genus
    for genus in genera:
        all_target_genera.add(genus)
        genus_to_label[genus] = genus
        for related in RELATED_GENERA.get(genus, []):
            all_target_genera.add(related)
            genus_to_label[related] = genus

    # Collect species from SpeciesNet
    print("Querying SpeciesNet taxonomy...")
    sn_results = query_speciesnet(all_target_genera)
    print(f"  Found {len(sn_results)} species entries")

    # Collect species from iNaturalist taxa.csv
    print("Querying iNaturalist taxa.csv...")
    inat_results = query_inat_taxa(all_target_genera)

    # Merge: use species name as key, prefer SpeciesNet common names
    species_map = {}  # (label_genus, species_name) -> row

    for entry in sn_results:
        label_genus = genus_to_label[entry["genus"]]
        key = (label_genus, entry["species"])
        species_map[key] = {
            "genus_label": genera[label_genus],
            "genus_scientific": label_genus,
            "species_scientific": entry["species"],
            "species_common_name": entry["common_name"],
            "source": "speciesnet",
            "inat_observation_count": "",
        }

    for entry in inat_results:
        label_genus = genus_to_label[entry["genus"]]
        key = (label_genus, entry["species"])
        if key in species_map:
            species_map[key]["source"] = "speciesnet+inat"
        else:
            species_map[key] = {
                "genus_label": genera[label_genus],
                "genus_scientific": label_genus,
                "species_scientific": entry["species"],
                "species_common_name": entry["common_name"],
                "source": "inat",
                "inat_observation_count": "",
            }

    # Optionally add iNaturalist observation counts
    if include_inat_counts:
        print("Fetching iNaturalist observation counts...")
        all_genera_for_api = list(all_target_genera)
        inat_counts = fetch_inat_counts(all_genera_for_api)

        for key, row in species_map.items():
            species_name = row["species_scientific"]
            if species_name in inat_counts:
                count_data = inat_counts[species_name]
                row["inat_observation_count"] = count_data.get("observation_count", "")
                # Fill in common name if missing
                if not row["species_common_name"] and count_data.get("common_name"):
                    row["species_common_name"] = count_data["common_name"]

    # Sort by genus label, then species name
    rows = sorted(species_map.values(), key=lambda r: (r["genus_scientific"], r["species_scientific"]))
    return rows


def write_output(rows, output_path):
    """Write the mapping to CSV."""
    fieldnames = [
        "genus_label",
        "genus_scientific",
        "species_scientific",
        "species_common_name",
        "source",
        "inat_observation_count",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows, genera):
    """Print a summary of species counts per genus."""
    counts = {}
    for row in rows:
        g = row["genus_scientific"]
        counts[g] = counts.get(g, 0) + 1

    print(f"\n{'Genus Label':<35} {'Scientific':<20} {'Species Count':>13}")
    print("-" * 70)
    for genus in sorted(genera.keys()):
        count = counts.get(genus, 0)
        marker = " ⚠" if count == 0 else ""
        print(f"{genera[genus]:<35} {genus:<20} {count:>13}{marker}")

    total = sum(counts.values())
    missing = [g for g in genera if counts.get(g, 0) == 0]
    print("-" * 70)
    print(f"{'TOTAL':<35} {len(genera)} genera{' ':>7} {total:>5} species")
    if missing:
        print(f"\n⚠ Genera with NO species found: {', '.join(missing)}")


# ── Family Mapping ────────────────────────────────────────────────────────────


def build_family_mapping(families):
    """Build the family-to-species mapping from all sources.

    Args:
        families: dict {family_name: common_name} from the label set.

    Returns:
        list of row dicts for the output CSV.
    """
    target_families = set(families.keys())

    # Collect species from SpeciesNet
    print("Querying SpeciesNet taxonomy for families...")
    sn_results = query_speciesnet_families(target_families)
    print(f"  Found {len(sn_results)} species entries")

    # Collect species from iNaturalist taxa.csv
    print("Querying iNaturalist taxa.csv for families...")
    inat_results = query_inat_taxa_families(target_families)

    # Merge: use (family, species) as key, prefer SpeciesNet common names
    species_map = {}

    for entry in sn_results:
        key = (entry["family"], entry["species"])
        species_map[key] = {
            "family_label": families[entry["family"]],
            "family_scientific": entry["family"],
            "genus_scientific": entry["genus"],
            "species_scientific": entry["species"],
            "species_common_name": entry["common_name"],
            "source": "speciesnet",
            "inat_observation_count": "",
        }

    for entry in inat_results:
        key = (entry["family"], entry["species"])
        if key in species_map:
            species_map[key]["source"] = "speciesnet+inat"
        else:
            species_map[key] = {
                "family_label": families[entry["family"]],
                "family_scientific": entry["family"],
                "genus_scientific": entry.get("genus", ""),
                "species_scientific": entry["species"],
                "species_common_name": entry["common_name"],
                "source": "inat",
                "inat_observation_count": "",
            }

    # Sort by family, then species
    rows = sorted(species_map.values(), key=lambda r: (r["family_scientific"], r["species_scientific"]))
    return rows


def write_family_output(rows, output_path):
    """Write the family mapping to CSV."""
    fieldnames = [
        "family_label",
        "family_scientific",
        "genus_scientific",
        "species_scientific",
        "species_common_name",
        "source",
        "inat_observation_count",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_family_summary(rows, families):
    """Print a summary of species counts per family."""
    counts = {}
    for row in rows:
        f = row["family_scientific"]
        counts[f] = counts.get(f, 0) + 1

    print(f"\n{'Family Label':<35} {'Scientific':<20} {'Species Count':>13}")
    print("-" * 70)
    for family in sorted(families.keys()):
        count = counts.get(family, 0)
        marker = " ⚠" if count == 0 else ""
        print(f"{families[family]:<35} {family:<20} {count:>13}{marker}")

    total = sum(counts.values())
    missing = [f for f in families if counts.get(f, 0) == 0]
    print("-" * 70)
    print(f"{'TOTAL':<35} {len(families)} families{' ':>5} {total:>5} species")
    if missing:
        print(f"\n⚠ Families with NO species found: {', '.join(missing)}")


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Map genus- and family-level labels to their constituent species.",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path for genus mapping (default: {DEFAULT_OUTPUT.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--family-output",
        type=Path,
        default=DEFAULT_FAMILY_OUTPUT,
        help=f"Output CSV path for family mapping (default: {DEFAULT_FAMILY_OUTPUT.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--include-inat-counts",
        action="store_true",
        help="Query iNaturalist API for observation counts per species (requires network)",
    )
    args = parser.parse_args()

    # ── Genus mapping ──
    print(f"Loading genus labels from {LABELS_225.name}...")
    genera = load_genus_labels(LABELS_225)
    print(f"  Found {len(genera)} genus-level labels")

    rows = build_mapping(genera, include_inat_counts=args.include_inat_counts)
    write_output(rows, args.output)
    print(f"\nWrote {len(rows)} rows to {args.output.relative_to(REPO_ROOT)}")
    print_summary(rows, genera)

    # ── Family mapping ──
    print(f"\n{'='*70}")
    print(f"Loading family labels from {LABELS_225.name}...")
    families = load_family_labels(LABELS_225)
    print(f"  Found {len(families)} family-level labels")

    family_rows = build_family_mapping(families)
    write_family_output(family_rows, args.family_output)
    print(f"\nWrote {len(family_rows)} rows to {args.family_output.relative_to(REPO_ROOT)}")
    print_family_summary(family_rows, families)


if __name__ == "__main__":
    main()
