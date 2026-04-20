"""
Analyze iNaturalist Open Data metadata to count available images per target class.

Streams the large metadata CSVs (taxa, observations, photos) and produces a
per-class breakdown of image counts by license type. This helps answer:
"How many training images can we get from iNaturalist for each of our 225 classes?"

Pipeline:
    1. Load taxa.csv (~1.6M rows) → find mammal subtree → match to 225 target labels
    2. Stream observations.csv (~233M rows) → map observation_uuid → class name
    3. Stream photos.csv (~413M rows) → count (class, license) pairs

Output: reports/inaturalist_class_image_counts.csv

Usage:
    python scripts/inaturalist/2-analyze_inaturalist_metadata.py
    python scripts/inaturalist/2-analyze_inaturalist_metadata.py --label-set 480
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
LABELS_225 = REPO_ROOT / "resources" / "2026-03-19_student_model_labels.txt"
LABELS_480 = REPO_ROOT / "resources" / "2026-03-20_student_model_labels_extended.txt"
METADATA_DIR = REPO_ROOT / "data" / "inaturalist" / "metadata"

# All license strings we expect to see in photos.csv
LICENSE_COLUMNS = ["cc0", "cc-by", "cc-by-nc", "cc-by-sa", "cc-by-nd", "cc-by-nc-sa", "cc-by-nc-nd"]
SAFE_LICENSES = {"cc0", "cc-by"}


# ── Label Loading ────────────────────────────────────────────────────────────


def load_target_labels(label_path):
    """Load label file and return lookup structures for matching.

    Label format: UUID;mammalia;order;family;genus;species;common_name
    """
    labels = {}
    species_lookup = {}
    genus_lookup = {}
    family_lookup = {}
    order_lookup = {}

    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(";")
            if len(parts) < 7:
                continue

            entry = {
                "uuid": parts[0],
                "order": parts[2],
                "family": parts[3],
                "genus": parts[4],
                "species": parts[5],
                "common_name": parts[6],
            }

            labels[parts[0]] = entry

            if parts[5]:  # species level
                species_lookup[f"{parts[4]} {parts[5]}".lower()] = entry
            elif parts[4]:  # genus level
                genus_lookup[parts[4].lower()] = entry
            elif parts[3]:  # family level
                family_lookup[parts[3].lower()] = entry
            elif parts[2]:  # order level
                order_lookup[parts[2].lower()] = entry

    return {
        "labels": labels,
        "species": species_lookup,
        "genus": genus_lookup,
        "family": family_lookup,
        "order": order_lookup,
    }


# ── Taxonomy Matching ────────────────────────────────────────────────────────


def match_taxon_to_label(taxon_name, taxon_rank, ancestry_names, target):
    """Match an iNaturalist taxon to a target label.

    Strategy (in order of preference):
        1. Exact species match
        2. Subspecies → parent species
        3. Genus match
        4. Family match
        5. Order match
    """
    name_lower = taxon_name.lower().strip()

    # 1. Species-level match
    if taxon_rank in ("species", "hybrid"):
        if name_lower in target["species"]:
            return target["species"][name_lower], "species"

    # 2. Subspecies → parent species (first two words)
    if taxon_rank == "subspecies":
        parts = name_lower.split()
        if len(parts) >= 2:
            parent = f"{parts[0]} {parts[1]}"
            if parent in target["species"]:
                return target["species"][parent], "subspecies"

    # 3. Genus match
    genus = name_lower.split()[0] if name_lower else ""
    if taxon_rank in ("species", "subspecies", "hybrid") and genus:
        if genus in target["genus"]:
            return target["genus"][genus], "genus"
    if taxon_rank == "genus" and name_lower in target["genus"]:
        return target["genus"][name_lower], "genus"

    # 4. Family match
    family = ancestry_names.get("family", "").lower()
    if taxon_rank == "family":
        family = name_lower
    if family and family in target["family"]:
        return target["family"][family], "family"

    # 5. Order match
    order = ancestry_names.get("order", "").lower()
    if taxon_rank == "order":
        order = name_lower
    if order and order in target["order"]:
        return target["order"][order], "order"

    return None, None


# ── Step 1: Load taxa and match to labels ────────────────────────────────────


def load_and_match_taxa(target, label_set):
    """Load taxa.csv, find mammal subtree, match to target labels.

    Returns:
        matched_taxon_ids: dict[int, str] — taxon_id → class common_name
    """
    taxa_path = METADATA_DIR / "taxa.csv"
    if not taxa_path.exists():
        print(f"ERROR: {taxa_path} not found. Run download_inaturalist.py metadata first.")
        sys.exit(1)

    print("── Step 1: Load taxa.csv and match to target labels ──")
    print("  Loading taxa.csv...")

    taxon_by_id = {}
    row_count = 0

    with open(taxa_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            row_count += 1
            tid = int(row["taxon_id"])
            taxon_by_id[tid] = {
                "name": row.get("name", ""),
                "rank": row.get("rank", ""),
                "ancestry": row.get("ancestry", ""),
                "active": row.get("active", "t"),
            }
            if row_count % 500_000 == 0:
                print(f"    ...{row_count:,} taxa loaded")

    print(f"  {row_count:,} taxa loaded")

    # Find Mammalia subtree
    mammalia_id = None
    for tid, t in taxon_by_id.items():
        if t["name"].lower() == "mammalia" and t["rank"] == "class":
            mammalia_id = tid
            break

    if mammalia_id is None:
        print("  ERROR: Could not find Mammalia in taxa.csv")
        sys.exit(1)

    print(f"  Mammalia taxon_id: {mammalia_id}")

    mammalia_str = str(mammalia_id)
    mammal_ids = set()
    for tid, t in taxon_by_id.items():
        if mammalia_str in t["ancestry"].split("/"):
            mammal_ids.add(tid)
        elif tid == mammalia_id:
            mammal_ids.add(tid)

    print(f"  {len(mammal_ids):,} mammal taxa found")

    # Resolve ancestry names for mammal taxa
    print("  Resolving ancestry names...")
    for tid in mammal_ids:
        t = taxon_by_id[tid]
        ancestry_names = {}
        for ancestor_id_str in t["ancestry"].split("/"):
            if not ancestor_id_str:
                continue
            try:
                ancestor_id = int(ancestor_id_str)
            except ValueError:
                continue
            ancestor = taxon_by_id.get(ancestor_id)
            if ancestor and ancestor["rank"]:
                ancestry_names[ancestor["rank"]] = ancestor["name"]
        if t["rank"]:
            ancestry_names[t["rank"]] = t["name"]
        t["ancestry_names"] = ancestry_names

    # Match mammal taxa to target labels
    print("  Matching taxa to target labels...")
    matched_taxon_ids = {}  # taxon_id -> common_name
    matched_details = []    # for saving to file

    for tid in mammal_ids:
        t = taxon_by_id[tid]
        if t["rank"] not in ("species", "subspecies", "genus", "hybrid"):
            continue
        entry, match_type = match_taxon_to_label(
            t["name"], t["rank"], t.get("ancestry_names", {}), target
        )
        if entry:
            matched_taxon_ids[tid] = entry["common_name"]
            matched_details.append({
                "taxon_id": tid,
                "scientific_name": t["name"],
                "rank": t["rank"],
                "target_class": entry["common_name"],
                "match_type": match_type,
            })

    # Coverage report
    matched_classes = set(matched_taxon_ids.values())
    all_classes = set(e["common_name"] for e in target["labels"].values())
    uncovered = sorted(all_classes - matched_classes)

    print(f"  {len(matched_taxon_ids):,} iNaturalist taxa matched")
    print(f"  Target classes covered: {len(matched_classes)} / {len(all_classes)}")
    if uncovered:
        print(f"  Uncovered ({len(uncovered)}): {', '.join(uncovered)}")

    # Save matched taxon_ids to file
    output_path = REPO_ROOT / "reports" / f"inaturalist_matched_taxa_{label_set}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matched_details.sort(key=lambda r: (r["target_class"], r["scientific_name"]))
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["taxon_id", "scientific_name", "rank", "target_class", "match_type"])
        writer.writeheader()
        writer.writerows(matched_details)
    print(f"  Saved {len(matched_details):,} matched taxa to {output_path.relative_to(REPO_ROOT)}")

    return matched_taxon_ids


# ── Step 2: Stream observations ──────────────────────────────────────────────


def stream_observations(matched_taxon_ids):
    """Stream observations.csv and map observation_uuid → class name.

    Returns:
        obs_to_class: dict[str, str] — observation_uuid → common_name
    """
    obs_path = METADATA_DIR / "observations.csv"
    if not obs_path.exists():
        print(f"ERROR: {obs_path} not found.")
        sys.exit(1)

    print("\n── Step 2: Stream observations.csv ──")
    matching_taxon_set = set(matched_taxon_ids.keys())
    obs_to_class = {}
    obs_count = 0

    with open(obs_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            obs_count += 1
            try:
                taxon_id = int(row["taxon_id"])
            except (ValueError, KeyError):
                continue

            if taxon_id in matching_taxon_set:
                obs_to_class[row["observation_uuid"]] = matched_taxon_ids[taxon_id]

            if obs_count % 10_000_000 == 0:
                print(f"    ...{obs_count:,} observations scanned, {len(obs_to_class):,} matched")

    print(f"  {obs_count:,} observations scanned")
    print(f"  {len(obs_to_class):,} observations match target taxa")
    return obs_to_class


# ── Step 3: Stream photos ────────────────────────────────────────────────────


def stream_photos(obs_to_class):
    """Stream photos.csv and count (class, license) pairs.

    Returns:
        class_license_counts: Counter[(common_name, license)]
    """
    photos_path = METADATA_DIR / "photos.csv"
    if not photos_path.exists():
        print(f"ERROR: {photos_path} not found.")
        sys.exit(1)

    print("\n── Step 3: Stream photos.csv ──")
    matching_obs_ids = set(obs_to_class.keys())
    class_license_counts = Counter()
    photo_count = 0
    matched_count = 0

    with open(photos_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            photo_count += 1
            obs_id = row.get("observation_uuid", "")

            if obs_id in matching_obs_ids:
                license_str = (row.get("license") or "").strip().lower()
                class_name = obs_to_class[obs_id]
                class_license_counts[(class_name, license_str)] += 1
                matched_count += 1

            if photo_count % 50_000_000 == 0:
                print(f"    ...{photo_count:,} photos scanned, {matched_count:,} matched")

    print(f"  {photo_count:,} photos scanned")
    print(f"  {matched_count:,} photos match target observations")
    return class_license_counts


# ── Report Generation ────────────────────────────────────────────────────────


def generate_report(target, class_license_counts, label_set):
    """Write per-class image counts CSV and print summary."""
    # Build per-class totals
    all_classes = {}
    for entry in target["labels"].values():
        cn = entry["common_name"]
        if entry["species"]:
            sci = f"{entry['genus']} {entry['species']}".lower()
            level = "species"
        elif entry["genus"]:
            sci = entry["genus"].lower()
            level = "genus"
        elif entry["family"]:
            sci = entry["family"].lower()
            level = "family"
        else:
            sci = entry["order"].lower()
            level = "order"
        all_classes[cn] = {"scientific_name": sci, "level": level}

    # Aggregate counts per class
    rows = []
    for cn, info in sorted(all_classes.items()):
        license_counts = {}
        for lic in LICENSE_COLUMNS:
            license_counts[lic] = class_license_counts.get((cn, lic), 0)

        # Count anything not in the known list as "other"
        known_total = sum(license_counts.values())
        all_total = sum(
            count for (c, _), count in class_license_counts.items() if c == cn
        )
        license_counts["other"] = all_total - known_total

        safe_total = license_counts["cc0"] + license_counts["cc-by"]

        rows.append({
            "common_name": cn,
            "scientific_name": info["scientific_name"],
            "level": info["level"],
            "total_images": all_total,
            "commercially_safe": safe_total,
            **{lic.replace("-", "_"): license_counts[lic] for lic in LICENSE_COLUMNS},
            "other": license_counts["other"],
        })

    # Sort by total_images descending
    rows.sort(key=lambda r: r["total_images"], reverse=True)

    # Write CSV
    output_path = REPO_ROOT / "reports" / f"inaturalist_class_image_counts_{label_set}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "common_name", "scientific_name", "level", "total_images", "commercially_safe",
        "cc0", "cc_by", "cc_by_nc", "cc_by_sa", "cc_by_nd", "cc_by_nc_sa", "cc_by_nc_nd", "other",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n── Report written to {output_path.relative_to(REPO_ROOT)} ──")

    # Print summary
    total_images = sum(r["total_images"] for r in rows)
    total_safe = sum(r["commercially_safe"] for r in rows)
    classes_with_images = sum(1 for r in rows if r["total_images"] > 0)
    classes_with_safe = sum(1 for r in rows if r["commercially_safe"] > 0)

    print(f"\n── Summary ──")
    print(f"  Total images across all classes:    {total_images:>12,}")
    print(f"  Commercially safe (CC0 + CC-BY):    {total_safe:>12,}")
    print(f"  Classes with any images:            {classes_with_images:>12} / {len(rows)}")
    print(f"  Classes with commercially safe:     {classes_with_safe:>12} / {len(rows)}")

    # Top 10
    print(f"\n  Top 10 classes by total images:")
    for r in rows[:10]:
        print(f"    {r['total_images']:>10,}  ({r['commercially_safe']:>8,} safe)  {r['common_name']}")

    # Bottom 10 (with images)
    with_images = [r for r in rows if r["total_images"] > 0]
    if with_images:
        print(f"\n  Bottom 10 classes (with images):")
        for r in with_images[-10:]:
            print(f"    {r['total_images']:>10,}  ({r['commercially_safe']:>8,} safe)  {r['common_name']}")

    # Classes with no images
    no_images = [r for r in rows if r["total_images"] == 0]
    if no_images:
        print(f"\n  Classes with NO images ({len(no_images)}):")
        for r in no_images:
            print(f"    - {r['common_name']} ({r['scientific_name']})")

    # License distribution overall
    print(f"\n  License distribution (all matched photos):")
    lic_totals = Counter()
    for (_, lic), count in class_license_counts.items():
        lic_totals[lic] += count
    for lic, count in lic_totals.most_common():
        safe = " [SAFE]" if lic in SAFE_LICENSES else ""
        print(f"    {lic or '(none)':>15}: {count:>12,}{safe}")


# ── Condense ─────────────────────────────────────────────────────────────────


CONDENSED_DIR = METADATA_DIR / "condensed"


def condense_taxa(matched_taxon_ids):
    """Write a condensed taxa.csv containing only matched mammal taxa."""
    taxa_path = METADATA_DIR / "taxa.csv"
    out_path = CONDENSED_DIR / "taxa.csv"

    print("\n── Condense taxa.csv ──")
    matched_set = set(str(tid) for tid in matched_taxon_ids)
    written = 0

    with open(taxa_path, newline="", encoding="utf-8") as fin, \
         open(out_path, "w", newline="", encoding="utf-8") as fout:
        reader = csv.DictReader(fin, delimiter="\t")
        fieldnames = reader.fieldnames + ["target_class"]
        writer = csv.DictWriter(fout, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in reader:
            tid_str = row["taxon_id"]
            if tid_str in matched_set:
                row["target_class"] = matched_taxon_ids[int(tid_str)]
                writer.writerow(row)
                written += 1

    print(f"  {written:,} taxa written to {out_path.relative_to(REPO_ROOT)}")
    print(f"  File size: {out_path.stat().st_size / 1e6:.1f} MB")


def condense_observations(matched_taxon_ids):
    """Write condensed observations.csv and return obs_to_class mapping.

    Returns:
        obs_to_class: dict[str, str] — observation_uuid → common_name
    """
    obs_path = METADATA_DIR / "observations.csv"
    out_path = CONDENSED_DIR / "observations.csv"

    print("\n── Condense observations.csv ──")
    matching_taxon_set = set(matched_taxon_ids.keys())
    obs_to_class = {}
    obs_count = 0
    written = 0

    with open(obs_path, newline="", encoding="utf-8") as fin, \
         open(out_path, "w", newline="", encoding="utf-8") as fout:
        reader = csv.DictReader(fin, delimiter="\t")
        fieldnames = reader.fieldnames + ["target_class"]
        writer = csv.DictWriter(fout, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        for row in reader:
            obs_count += 1
            try:
                taxon_id = int(row["taxon_id"])
            except (ValueError, KeyError):
                continue

            if taxon_id in matching_taxon_set:
                class_name = matched_taxon_ids[taxon_id]
                obs_to_class[row["observation_uuid"]] = class_name
                row["target_class"] = class_name
                writer.writerow(row)
                written += 1

            if obs_count % 10_000_000 == 0:
                print(f"    ...{obs_count:,} scanned, {written:,} written")

    print(f"  {obs_count:,} observations scanned")
    print(f"  {written:,} observations written to {out_path.relative_to(REPO_ROOT)}")
    print(f"  File size: {out_path.stat().st_size / 1e9:.2f} GB")
    return obs_to_class


def condense_photos(obs_to_class):
    """Write condensed photos.csv containing only photos for matched observations."""
    photos_path = METADATA_DIR / "photos.csv"
    out_path = CONDENSED_DIR / "photos.csv"

    print("\n── Condense photos.csv ──")
    matching_obs_ids = set(obs_to_class.keys())
    photo_count = 0
    written = 0

    with open(photos_path, newline="", encoding="utf-8") as fin, \
         open(out_path, "w", newline="", encoding="utf-8") as fout:
        reader = csv.DictReader(fin, delimiter="\t")
        fieldnames = reader.fieldnames + ["target_class"]
        writer = csv.DictWriter(fout, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        for row in reader:
            photo_count += 1
            obs_id = row.get("observation_uuid", "")

            if obs_id in matching_obs_ids:
                row["target_class"] = obs_to_class[obs_id]
                writer.writerow(row)
                written += 1

            if photo_count % 50_000_000 == 0:
                print(f"    ...{photo_count:,} scanned, {written:,} written")

    print(f"  {photo_count:,} photos scanned")
    print(f"  {written:,} photos written to {out_path.relative_to(REPO_ROOT)}")
    print(f"  File size: {out_path.stat().st_size / 1e9:.2f} GB")


def cmd_condense(target, label_set):
    """Create condensed metadata CSVs containing only rows for target classes."""
    CONDENSED_DIR.mkdir(parents=True, exist_ok=True)

    matched_taxon_ids = load_and_match_taxa(target, label_set)
    condense_taxa(matched_taxon_ids)
    obs_to_class = condense_observations(matched_taxon_ids)
    condense_photos(obs_to_class)

    print(f"\n── Done ──")
    print(f"  Condensed files in: {CONDENSED_DIR.relative_to(REPO_ROOT)}/")
    print(f"  All files include a 'target_class' column mapping rows to the 225 labels.")


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Analyze or condense iNaturalist metadata for target classes.",
    )
    parser.add_argument(
        "--label-set",
        choices=["225", "480"],
        default="225",
        help="Target label set (default: 225)",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("analyze", help="Count images per target class by license (default)")
    subparsers.add_parser("condense", help="Write filtered metadata CSVs with only target-class rows")

    args = parser.parse_args()

    # Check metadata exists
    for name in ["taxa.csv", "observations.csv", "photos.csv"]:
        if not (METADATA_DIR / name).exists():
            print(f"ERROR: {METADATA_DIR / name} not found.")
            print("  Run: python scripts/inaturalist/1-download_inaturalist.py metadata")
            sys.exit(1)

    label_path = LABELS_225 if args.label_set == "225" else LABELS_480
    target = load_target_labels(label_path)
    print(f"Loaded {len(target['labels'])} target labels from {label_path.name}\n")

    if args.command == "condense":
        cmd_condense(target, args.label_set)
    else:
        # Default: analyze
        matched_taxon_ids = load_and_match_taxa(target, args.label_set)
        obs_to_class = stream_observations(matched_taxon_ids)
        class_license_counts = stream_photos(obs_to_class)
        generate_report(target, class_license_counts, args.label_set)


if __name__ == "__main__":
    main()


"""
── Step 1: Load taxa.csv and match to target labels ──
  Loading taxa.csv...
    ...500,000 taxa loaded
    ...1,000,000 taxa loaded
    ...1,500,000 taxa loaded
  1,626,690 taxa loaded
  Mammalia taxon_id: 40151
  16,698 mammal taxa found
  Resolving ancestry names...
  Matching taxa to target labels...
  7,158 iNaturalist taxa matched
  Target classes covered: 221 / 225
  Uncovered (4): dingo, domestic goat, domestic pig, pinniped clade
  Saved 7,158 matched taxa to reports/inaturalist_matched_taxa_225.csv

── Step 2: Stream observations.csv ──
    ...10,000,000 observations scanned, 237,643 matched
    ...20,000,000 observations scanned, 422,879 matched
    ...30,000,000 observations scanned, 624,321 matched
    ...40,000,000 observations scanned, 788,927 matched
    ...50,000,000 observations scanned, 980,832 matched
    ...60,000,000 observations scanned, 1,112,908 matched
    ...70,000,000 observations scanned, 1,305,972 matched
    ...80,000,000 observations scanned, 1,452,075 matched
    ...90,000,000 observations scanned, 1,595,156 matched
    ...100,000,000 observations scanned, 1,801,897 matched
    ...110,000,000 observations scanned, 1,938,817 matched
    ...120,000,000 observations scanned, 2,072,142 matched
    ...130,000,000 observations scanned, 2,257,371 matched
    ...140,000,000 observations scanned, 2,442,147 matched
    ...150,000,000 observations scanned, 2,579,210 matched
    ...160,000,000 observations scanned, 2,720,019 matched
    ...170,000,000 observations scanned, 2,891,540 matched
    ...180,000,000 observations scanned, 3,114,562 matched
    ...190,000,000 observations scanned, 3,261,714 matched
    ...200,000,000 observations scanned, 3,403,693 matched
    ...210,000,000 observations scanned, 3,548,180 matched
    ...220,000,000 observations scanned, 3,708,941 matched
    ...230,000,000 observations scanned, 3,924,930 matched
  233,286,559 observations scanned
  4,003,990 observations match target taxa

── Step 3: Stream photos.csv ──
    ...50,000,000 photos scanned, 974,996 matched
    ...100,000,000 photos scanned, 1,762,676 matched
    ...150,000,000 photos scanned, 2,536,004 matched
    ...200,000,000 photos scanned, 3,318,085 matched
    ...250,000,000 photos scanned, 4,140,629 matched
    ...300,000,000 photos scanned, 4,876,678 matched
    ...350,000,000 photos scanned, 5,720,462 matched
    ...400,000,000 photos scanned, 6,508,627 matched
  413,168,476 photos scanned
  6,816,962 photos match target observations

── Report written to reports/inaturalist_class_image_counts_225.csv ──

── Summary ──
  Total images across all classes:       6,816,962
  Commercially safe (CC0 + CC-BY):         831,050
  Classes with any images:                     216 / 225
  Classes with commercially safe:              215 / 225

  Top 10 classes by total images:
       463,529  (  61,865 safe)  white-tailed deer
       402,406  (  47,196 safe)  eastern gray squirrel
       379,223  (  51,099 safe)  squirrel family
       233,505  (  29,216 safe)  mule deer
       222,123  (  23,459 safe)  northern raccoon
       187,721  (  20,071 safe)  coyote
       185,552  (  18,329 safe)  eastern cottontail
       176,062  (  16,646 safe)  eastern fox squirrel
       168,081  (  20,852 safe)  red fox
       149,953  (  20,415 safe)  domestic cat

  Bottom 10 classes (with images):
           501  (      84 safe)  kirk's dik-dik
           463  (      50 safe)  binturong
           440  (      68 safe)  malay tapir
           433  (      69 safe)  fossa
           268  (      91 safe)  clouded leopard
           236  (      39 safe)  aye-aye
           216  (      11 safe)  hog badger genus
           213  (      24 safe)  giant armadillo
           143  (      21 safe)  drill
             3  (       0 safe)  mouflon

  Classes with NO images (9):
    - aardwolf (proteles cristata)
    - american mink (neovison vison)
    - black-backed jackal (canis mesomelas)
    - dingo (canis lupus dingo)
    - domestic goat (capra aegagrus hircus)
    - domestic pig (sus scrofa scrofa)
    - human (homo sapiens)
    - pinniped clade (pinniped)
    - red-necked wallaby (macropus rufogriseus)

  License distribution (all matched photos):
           cc-by-nc:    5,566,509
              cc-by:      678,971 [SAFE]
        cc-by-nc-nd:      217,604
                cc0:      152,079 [SAFE]
        cc-by-nc-sa:      116,214
           cc-by-sa:       68,368
           cc-by-nd:       17,211
                 pd:            6
"""