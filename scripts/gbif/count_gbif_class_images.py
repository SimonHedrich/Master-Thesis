"""
Count pre-reduction GBIF image counts per classes_225 class.

Reads resources/SNPredictions_all-formatted.json (all 91,291 original GBIF
predictions) and maps each SpeciesNet prediction to a classes_225 class using
a three-level fallback: species → genus → family.

Usage:
    uv run python -m scripts.gbif.count_gbif_class_images

Output:
    reports/gbif_image_counts_all.csv  — per-class counts (all 225, incl. zeros)
"""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PREDICTIONS_PATH = REPO_ROOT / "resources" / "SNPredictions_all-formatted.json"
CLASSES_225_PATH = REPO_ROOT / "reports" / "classes_225.csv"
OUTPUT_PATH = REPO_ROOT / "reports" / "gbif_image_counts_all.csv"


def load_classes_225(path):
    """Return list of dicts and three lookup dicts for species/genus/family matching."""
    rows = []
    species_map = {}  # "genus species" -> row
    genus_map = {}    # "genus" -> row
    family_map = {}   # "family" -> row

    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
            sci = row["scientific_name"].strip().lower()
            level = row["level"]
            if level == "species":
                species_map[sci] = row
            elif level == "genus":
                genus_map[sci] = row      # sci is a single word (genus)
            elif level == "family":
                family_map[sci] = row     # sci is a single word (family)

    return rows, species_map, genus_map, family_map


def match_prediction(pred_str, species_map, genus_map, family_map):
    """Map a SpeciesNet prediction string to a classes_225 row.

    Prediction format: UUID;class;order;family;genus;species;common_name
    Returns the matched row dict or None.
    """
    parts = pred_str.split(";")
    if len(parts) < 7:
        return None

    family = parts[3].strip().lower()
    genus = parts[4].strip().lower()
    species = parts[5].strip().lower()

    if genus and species:
        key = f"{genus} {species}"
        if key in species_map:
            return species_map[key]

    if genus and genus in genus_map:
        return genus_map[genus]

    if family and family in family_map:
        return family_map[family]

    return None


def main():
    print("Loading classes_225 …")
    rows, species_map, genus_map, family_map = load_classes_225(CLASSES_225_PATH)

    print(f"Loading predictions from {PREDICTIONS_PATH} …")
    with open(PREDICTIONS_PATH) as f:
        data = json.load(f)
    predictions = data["predictions"]
    print(f"  {len(predictions):,} predictions loaded")

    counts = defaultdict(int)
    for row in rows:
        counts[row["common_name"]] = 0  # ensure all 225 classes appear

    unmatched_by_source = Counter()
    total_matched = 0

    for entry in predictions:
        pred_str = entry.get("prediction", "")
        matched = match_prediction(pred_str, species_map, genus_map, family_map)
        if matched:
            counts[matched["common_name"]] += 1
            total_matched += 1
        else:
            src = entry.get("prediction_source", "MISSING")
            unmatched_by_source[src] += 1

    total_unmatched = sum(unmatched_by_source.values())
    print(f"\nMatched:   {total_matched:,} / {len(predictions):,}")
    print(f"Unmatched: {total_unmatched:,}")
    print("Unmatched by source:")
    for src, n in unmatched_by_source.most_common():
        print(f"  {src}: {n:,}")

    # Build output rows with full class metadata
    class_meta = {row["common_name"]: row for row in rows}
    output_rows = sorted(
        [
            {
                "common_name": cn,
                "scientific_name": class_meta[cn]["scientific_name"],
                "level": class_meta[cn]["level"],
                "image_count": count,
            }
            for cn, count in counts.items()
        ],
        key=lambda r: -r["image_count"],
    )

    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["common_name", "scientific_name", "level", "image_count"]
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"\nWrote {OUTPUT_PATH}")
    print("\nTop 20 classes by image count:")
    for r in output_rows[:20]:
        print(f"  {r['image_count']:5d}  {r['common_name']}")
    zero_count = [r for r in output_rows if r["image_count"] == 0]
    print(f"\nClasses with 0 images ({len(zero_count)}):")
    for r in zero_count:
        print(f"  {r['common_name']}")


if __name__ == "__main__":
    main()
