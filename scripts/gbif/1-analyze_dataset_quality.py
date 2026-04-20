"""
Dataset Quality Analysis for the GBIF/SpeciesNet Wildlife Dataset.

Analyzes Danielle's dataset (SNPredictions_all.json + GBIFImages) against the
225-class and 480-class student model label sets. Produces a comprehensive
report covering:

  1. Image inventory (JSON vs. disk, missing files, orphan images)
  2. Prediction source breakdown (species-level vs. rollups vs. blanks)
  3. Per-class image counts mapped to both label sets
  4. Confidence distribution analysis
  5. Class coverage gaps (labels with zero or few images)
  6. Detection quality (bbox presence, multi-detection images)
  7. Image quality flags (corrupt files, extreme aspect ratios, tiny images)

Usage:
    python scripts/gbif/1-analyze_dataset_quality.py

Output:
    reports/dataset_quality_report.txt   — human-readable summary
    reports/class_counts_225.csv         — per-class counts for the 225-class list
    reports/class_counts_480.csv         — per-class counts for the 480-class list
"""

import csv
import json
import os
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
IMAGES_DIR = REPO_ROOT / "resources" / "GBIFImages" / "images"
LABELS_PATH = REPO_ROOT / "resources" / "SNPredictions_all.json"
LABELS_225 = REPO_ROOT / "resources" / "2026-03-19_student_model_labels.txt"
LABELS_480 = REPO_ROOT / "resources" / "2026-03-20_student_model_labels_extended.txt"
REPORTS_DIR = REPO_ROOT / "reports"


# ── Label Set Loading ────────────────────────────────────────────────────────


def load_label_set(path):
    """Load a label file and return a dict mapping UUID -> {uuid, class_, order,
    family, genus, species, common_name, level}."""
    labels = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(";")
            if len(parts) < 7:
                continue
            uuid = parts[0]
            genus = parts[4]
            species = parts[5]

            if species:
                level = "species"
            elif genus:
                level = "genus"
            elif parts[3]:
                level = "family"
            else:
                level = "higher"

            labels[uuid] = {
                "uuid": uuid,
                "class_": parts[1],
                "order": parts[2],
                "family": parts[3],
                "genus": genus,
                "species": species,
                "common_name": parts[6],
                "level": level,
                "full_line": line,
            }
    return labels


def build_lookup(label_set):
    """Build multiple lookup indices for matching predictions to labels.

    Returns a dict with keys:
      - uuid: {uuid -> label}
      - genus_species: {"genus species" -> label}  (species-level)
      - genus: {"genus" -> label}                   (genus-level fallback)
      - family: {"family" -> label}                 (family-level fallback)
    """
    idx = {
        "uuid": {},
        "genus_species": {},
        "genus": {},
        "family": {},
    }
    for label in label_set.values():
        idx["uuid"][label["uuid"]] = label
        if label["level"] == "species" and label["genus"] and label["species"]:
            key = f"{label['genus']} {label['species']}"
            idx["genus_species"][key] = label
        if label["level"] == "genus" and label["genus"]:
            idx["genus"][label["genus"]] = label
        if label["level"] == "family" and label["family"]:
            idx["family"][label["family"]] = label
    return idx


def match_prediction(prediction_str, lookup):
    """Try to match a SpeciesNet prediction string to a label in the set.

    Matching priority: UUID -> genus+species -> genus fallback -> family fallback.
    Returns (matched_label, match_type) or (None, None).
    """
    parts = prediction_str.split(";")
    if len(parts) < 7:
        return None, None

    pred_uuid = parts[0]
    pred_family = parts[3]
    pred_genus = parts[4]
    pred_species = parts[5]

    # 1. Exact UUID match
    if pred_uuid in lookup["uuid"]:
        return lookup["uuid"][pred_uuid], "uuid"

    # 2. Genus + species match
    if pred_genus and pred_species:
        key = f"{pred_genus} {pred_species}"
        if key in lookup["genus_species"]:
            return lookup["genus_species"][key], "genus_species"

    # 3. Genus fallback
    if pred_genus and pred_genus in lookup["genus"]:
        return lookup["genus"][pred_genus], "genus_fallback"

    # 4. Family fallback
    if pred_family and pred_family in lookup["family"]:
        return lookup["family"][pred_family], "family_fallback"

    return None, None


# ── Image Quality Checks ─────────────────────────────────────────────────────


def get_jpeg_dimensions(filepath):
    """Fast JPEG dimension extraction without PIL (reads only header bytes)."""
    try:
        with open(filepath, "rb") as f:
            f.seek(0)
            if f.read(2) != b"\xff\xd8":
                return None, None  # not a JPEG
            while True:
                marker = f.read(2)
                if len(marker) < 2:
                    return None, None
                if marker[0] != 0xFF:
                    return None, None
                if marker[1] == 0xC0 or marker[1] == 0xC2:  # SOF0 or SOF2
                    f.read(3)  # length + precision
                    h = struct.unpack(">H", f.read(2))[0]
                    w = struct.unpack(">H", f.read(2))[0]
                    return w, h
                else:
                    length = struct.unpack(">H", f.read(2))[0]
                    f.seek(length - 2, 1)
    except Exception:
        return None, None


# ── Main Analysis ─────────────────────────────────────────────────────────────


def analyze():
    print("=" * 70)
    print("DATASET QUALITY ANALYSIS")
    print("=" * 70)

    # ── Load data ────────────────────────────────────────────────────────
    print("\nLoading predictions...")
    with open(LABELS_PATH) as f:
        data = json.load(f)
    predictions = data["predictions"]
    print(f"  JSON entries: {len(predictions)}")

    print("Loading label sets...")
    labels_225 = load_label_set(LABELS_225)
    labels_480 = load_label_set(LABELS_480)
    print(f"  225-class set: {len(labels_225)} labels")
    print(f"  480-class set: {len(labels_480)} labels")

    lookup_225 = build_lookup(labels_225)
    lookup_480 = build_lookup(labels_480)

    # ── 1. Image Inventory ───────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("1. IMAGE INVENTORY")
    print("─" * 70)

    json_files = set(p["filepath"] for p in predictions)
    disk_files = set(
        f for f in os.listdir(IMAGES_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    in_json_not_disk = json_files - disk_files
    in_disk_not_json = disk_files - json_files
    usable = json_files & disk_files

    print(f"  Images in JSON:          {len(json_files):>7}")
    print(f"  Images on disk:          {len(disk_files):>7}")
    print(f"  Usable (in both):        {len(usable):>7}")
    print(f"  In JSON, missing on disk:{len(in_json_not_disk):>7}")
    print(f"  On disk, not in JSON:    {len(in_disk_not_json):>7}")

    # ── 2. Prediction Source Breakdown ───────────────────────────────────
    print("\n" + "─" * 70)
    print("2. PREDICTION SOURCE BREAKDOWN")
    print("─" * 70)

    source_counts = Counter()
    usable_preds = []
    for p in predictions:
        if p["filepath"] in usable:
            usable_preds.append(p)
            source_counts[p.get("prediction_source", "(empty)")] += 1

    print(f"  Total usable predictions: {len(usable_preds)}")
    for source, count in source_counts.most_common():
        pct = count / len(usable_preds) * 100
        print(f"    {source:<35} {count:>6}  ({pct:5.1f}%)")

    # Categorize
    species_level = [
        p for p in usable_preds if p.get("prediction_source") == "classifier"
    ]
    rollup = [
        p for p in usable_preds
        if p.get("prediction_source", "").startswith("classifier+rollup")
    ]
    detector_only = [
        p for p in usable_preds if p.get("prediction_source") == "detector"
    ]
    blanks = [
        p for p in usable_preds
        if "blank" in p.get("prediction", "").lower()
    ]

    print(f"\n  Summary:")
    print(f"    Species-level confident: {len(species_level):>6}")
    print(f"    Rolled up (genus/family/order/etc): {len(rollup):>6}")
    print(f"    Detector only (no classification):  {len(detector_only):>6}")
    print(f"    Blank frames:                       {len(blanks):>6}")

    # ── 3. Confidence Distribution ───────────────────────────────────────
    print("\n" + "─" * 70)
    print("3. CONFIDENCE DISTRIBUTION")
    print("─" * 70)

    conf_buckets = Counter()
    all_confs = []
    for p in usable_preds:
        score = p.get("prediction_score", 0)
        all_confs.append(score)
        if score >= 0.9:
            conf_buckets["≥0.9"] += 1
        elif score >= 0.7:
            conf_buckets["0.7–0.9"] += 1
        elif score >= 0.5:
            conf_buckets["0.5–0.7"] += 1
        elif score >= 0.3:
            conf_buckets["0.3–0.5"] += 1
        else:
            conf_buckets["<0.3"] += 1

    for bucket in ["≥0.9", "0.7–0.9", "0.5–0.7", "0.3–0.5", "<0.3"]:
        count = conf_buckets[bucket]
        pct = count / len(usable_preds) * 100
        print(f"    {bucket:<12} {count:>6}  ({pct:5.1f}%)")

    if all_confs:
        all_confs.sort()
        print(f"\n    Mean confidence:   {sum(all_confs)/len(all_confs):.3f}")
        print(f"    Median confidence: {all_confs[len(all_confs)//2]:.3f}")

    # ── 4. Detection Quality ─────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("4. DETECTION QUALITY")
    print("─" * 70)

    no_det = 0
    single_det = 0
    multi_det = 0
    det_confs = []

    for p in usable_preds:
        dets = p.get("detections", [])
        if len(dets) == 0:
            no_det += 1
        elif len(dets) == 1:
            single_det += 1
            det_confs.append(dets[0].get("conf", 0))
        else:
            multi_det += 1
            for d in dets:
                det_confs.append(d.get("conf", 0))

    print(f"    No detections:      {no_det:>6}")
    print(f"    Single detection:   {single_det:>6}")
    print(f"    Multiple detections:{multi_det:>6}")
    if det_confs:
        det_confs.sort()
        print(f"    Mean det confidence:   {sum(det_confs)/len(det_confs):.3f}")
        print(f"    Median det confidence: {det_confs[len(det_confs)//2]:.3f}")
        low_det = sum(1 for c in det_confs if c < 0.5)
        print(f"    Detections with conf < 0.5: {low_det}")

    # ── 5. Per-Class Mapping & Counts ────────────────────────────────────
    print("\n" + "─" * 70)
    print("5. PER-CLASS MAPPING TO LABEL SETS")
    print("─" * 70)

    def count_classes(preds_list, lookup, label_set):
        """Map predictions to label set and count per class."""
        class_counts = defaultdict(int)
        class_confs = defaultdict(list)
        unmatched = Counter()
        match_types = Counter()

        for p in preds_list:
            pred_str = p.get("prediction", "")
            matched, mtype = match_prediction(pred_str, lookup)
            if matched:
                key = matched["common_name"]
                class_counts[key] += 1
                class_confs[key].append(p.get("prediction_score", 0))
                match_types[mtype] += 1
            else:
                # Extract common name for unmatched
                parts = pred_str.split(";")
                name = parts[-1] if len(parts) >= 7 else pred_str
                unmatched[name] += 1

        return class_counts, class_confs, unmatched, match_types

    # Only use species-level + rollup predictions (skip blanks and detector-only)
    classifiable = [
        p for p in usable_preds
        if p.get("prediction_source", "") not in ("detector", "")
        and "blank" not in p.get("prediction", "").lower()
        and "no cv result" not in p.get("prediction", "").lower()
    ]
    print(f"  Classifiable predictions (excl. blanks/detector-only): {len(classifiable)}")

    for name, lookup, label_set, csv_name in [
        ("225-class", lookup_225, labels_225, "class_counts_225.csv"),
        ("480-class", lookup_480, labels_480, "class_counts_480.csv"),
    ]:
        counts, confs, unmatched, mtypes = count_classes(
            classifiable, lookup, label_set
        )

        total_matched = sum(counts.values())
        total_unmatched = sum(unmatched.values())

        print(f"\n  ── {name} set ──")
        print(f"    Matched to a label:   {total_matched:>6} ({total_matched/len(classifiable)*100:.1f}%)")
        print(f"    Unmatched:            {total_unmatched:>6} ({total_unmatched/len(classifiable)*100:.1f}%)")
        print(f"    Match types: {dict(mtypes.most_common())}")

        # Coverage stats
        labels_with_images = sum(1 for c in counts.values() if c > 0)
        labels_no_images = len(label_set) - labels_with_images
        labels_under_10 = sum(
            1 for lbl in label_set.values()
            if counts.get(lbl["common_name"], 0) < 10
        )
        labels_under_50 = sum(
            1 for lbl in label_set.values()
            if counts.get(lbl["common_name"], 0) < 50
        )
        labels_over_100 = sum(
            1 for lbl in label_set.values()
            if counts.get(lbl["common_name"], 0) >= 100
        )

        print(f"    Labels with ≥1 image:  {labels_with_images:>4} / {len(label_set)}")
        print(f"    Labels with 0 images:  {labels_no_images:>4}")
        print(f"    Labels with <10 images:{labels_under_10:>4}")
        print(f"    Labels with <50 images:{labels_under_50:>4}")
        print(f"    Labels with ≥100 images:{labels_over_100:>3}")

        # Top 10 unmatched
        if unmatched:
            print(f"    Top 10 unmatched predictions:")
            for un_name, un_count in unmatched.most_common(10):
                print(f"      {un_count:>5}  {un_name}")

        # Write CSV
        csv_path = REPORTS_DIR / csv_name
        with open(csv_path, "w", newline="") as csvf:
            writer = csv.writer(csvf)
            writer.writerow([
                "common_name", "scientific_name", "level", "image_count",
                "mean_confidence", "min_confidence", "median_confidence",
            ])
            for lbl in sorted(label_set.values(), key=lambda x: x["common_name"]):
                cn = lbl["common_name"]
                sci = (
                    f"{lbl['genus']} {lbl['species']}"
                    if lbl["species"]
                    else lbl["genus"] or lbl["family"] or ""
                )
                count = counts.get(cn, 0)
                c_list = confs.get(cn, [])
                mean_c = sum(c_list) / len(c_list) if c_list else 0
                min_c = min(c_list) if c_list else 0
                c_list.sort()
                med_c = c_list[len(c_list) // 2] if c_list else 0
                writer.writerow([cn, sci, lbl["level"], count, f"{mean_c:.3f}", f"{min_c:.3f}", f"{med_c:.3f}"])
        print(f"    → Wrote {csv_path}")

    # ── 6. Coverage Gaps (Labels with 0 images) ─────────────────────────
    print("\n" + "─" * 70)
    print("6. COVERAGE GAPS — 225-CLASS LABELS WITH 0 IMAGES")
    print("─" * 70)

    counts_225, _, _, _ = count_classes(classifiable, lookup_225, labels_225)
    zero_labels = [
        lbl for lbl in labels_225.values()
        if counts_225.get(lbl["common_name"], 0) == 0
    ]
    zero_labels.sort(key=lambda x: x["common_name"])
    print(f"  {len(zero_labels)} labels with zero images:")
    for lbl in zero_labels:
        sci = (
            f"{lbl['genus']} {lbl['species']}"
            if lbl["species"]
            else lbl["genus"] or lbl["family"] or ""
        )
        print(f"    [{lbl['level']:<7}] {lbl['common_name']:<35} ({sci})")

    # ── 7. Image Quality Spot Check ──────────────────────────────────────
    print("\n" + "─" * 70)
    print("7. IMAGE QUALITY SPOT CHECK")
    print("─" * 70)

    print("  Scanning image dimensions (JPEG headers only)...")
    tiny_images = []       # < 100px on either side
    huge_images = []       # > 5000px on either side
    extreme_ratio = []     # aspect ratio > 4:1 or < 1:4
    corrupt_images = []    # can't read dimensions
    file_sizes = []

    sample_files = list(usable)
    for i, fname in enumerate(sample_files):
        fpath = IMAGES_DIR / fname
        fsize = fpath.stat().st_size
        file_sizes.append(fsize)

        w, h = get_jpeg_dimensions(str(fpath))
        if w is None or h is None:
            corrupt_images.append(fname)
            continue
        if w < 100 or h < 100:
            tiny_images.append((fname, w, h))
        if w > 5000 or h > 5000:
            huge_images.append((fname, w, h))
        if w > 0 and h > 0:
            ratio = max(w / h, h / w)
            if ratio > 4.0:
                extreme_ratio.append((fname, w, h, ratio))

    file_sizes.sort()
    print(f"  Total images checked:       {len(sample_files)}")
    print(f"  Corrupt / unreadable:       {len(corrupt_images)}")
    print(f"  Tiny (<100px):              {len(tiny_images)}")
    print(f"  Very large (>5000px):       {len(huge_images)}")
    print(f"  Extreme aspect ratio (>4:1):{len(extreme_ratio)}")
    if file_sizes:
        print(f"  File size range: {file_sizes[0]/1024:.0f}KB – {file_sizes[-1]/1024:.0f}KB")
        print(f"  Median file size: {file_sizes[len(file_sizes)//2]/1024:.0f}KB")
        tiny_files = sum(1 for s in file_sizes if s < 10 * 1024)
        print(f"  Files < 10KB (likely corrupt/placeholder): {tiny_files}")

    if corrupt_images[:5]:
        print(f"  Sample corrupt: {corrupt_images[:5]}")
    if tiny_images[:5]:
        print(f"  Sample tiny: {tiny_images[:5]}")

    # ── 8. Filename-Inferred Species Distribution ────────────────────────
    print("\n" + "─" * 70)
    print("8. FILENAME-INFERRED SPECIES VS. PREDICTION AGREEMENT")
    print("─" * 70)
    print("  (Filenames follow 'genus_species_id.jpg' pattern)")

    mismatches = 0
    matches = 0
    checked = 0
    mismatch_examples = []

    for p in usable_preds[:10000]:  # sample first 10k for speed
        fname = p["filepath"]
        parts = fname.rsplit("_", 1)
        if len(parts) != 2:
            continue
        fname_species = parts[0].replace("_", " ")
        pred_str = p.get("prediction", "")
        pred_parts = pred_str.split(";")
        if len(pred_parts) >= 7:
            pred_genus = pred_parts[4]
            pred_species = pred_parts[5]
            pred_sci = f"{pred_genus} {pred_species}" if pred_species else pred_genus
        else:
            continue

        checked += 1
        if fname_species.lower() == pred_sci.lower():
            matches += 1
        else:
            # Check if prediction is a rollup that contains the filename species
            if fname_species.lower().startswith(pred_sci.lower()):
                matches += 1
            else:
                mismatches += 1
                if len(mismatch_examples) < 10:
                    mismatch_examples.append(
                        (fname, fname_species, pred_sci, p.get("prediction_score", 0))
                    )

    print(f"  Checked (first 10k): {checked}")
    print(f"  Filename ≈ prediction: {matches} ({matches/max(checked,1)*100:.1f}%)")
    print(f"  Filename ≠ prediction: {mismatches} ({mismatches/max(checked,1)*100:.1f}%)")
    if mismatch_examples:
        print(f"  Sample mismatches:")
        for fname, fn_sp, pred_sp, score in mismatch_examples[:5]:
            print(f"    {fname}: filename='{fn_sp}' vs pred='{pred_sp}' (conf={score:.2f})")

    # ── Write full report ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"Reports written to {REPORTS_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    REPORTS_DIR.mkdir(exist_ok=True)
    analyze()
