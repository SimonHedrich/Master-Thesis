"""SpeciesNet filter and 225-class mapping for dataset quality assessment.

Reads speciesnet_results.jsonl (produced by 6-classify_speciesnet.py) and applies
the filtering rules from docs/plans/2026-04-30_speciesnet-classification-strategy.md:

  - MegaDetector confidence floor on the primary detection
  - SpeciesNet confidence floor
  - Hierarchical taxonomic match level (species → genus → family → order → class → no_match)
  - Family match + high-confidence threshold → fail

Phase 1 (default): statistics-only dry run.
  Computes all filtering decisions and prints per-source and per-class summaries.
  No changes are written to any file.

Phase 2 (--write): extends Phase 1.
  Merges a speciesnet_eval block into each matched entry in filter_results.jsonl,
  updating passed / stage_failed / reason. Writes atomically via a .tmp file.

Must run inside Dockerfile.speciesnet (Python 3.11, speciesnet package) — the
SpeciesNet classifier is loaded at startup to resolve integer class indices to
'uuid;class;order;family;genus;species;common' label strings.

Usage:
    # Phase 1: print statistics only (safe, no writes)
    python scripts/dataset_quality/7-filter_speciesnet.py --source gbif
    python scripts/dataset_quality/7-filter_speciesnet.py --source all

    # Phase 2: write speciesnet_eval to filter_results.jsonl
    python scripts/dataset_quality/7-filter_speciesnet.py --source all --write

    # Adjust thresholds (defaults match strategy doc):
    python scripts/dataset_quality/7-filter_speciesnet.py --source gbif \\
        --md-conf 0.4 --sn-score 0.25 --family-fail-thresh 0.6

Output added to filter_results.jsonl in Phase 2 (per image):
    {
      "speciesnet_eval": {
        "pass": true,
        "reason": null,
        "primary_detection": {
          "detection_idx": 0,
          "megadetector_conf": 0.97,
          "speciesnet_top1_idx": 1787,
          "speciesnet_top1_label": "uuid;mammalia;carnivora;canidae;vulpes;vulpes;red fox",
          "speciesnet_top1_scientific": "vulpes vulpes",
          "speciesnet_top1_common": "red fox",
          "speciesnet_top1_score": 0.91,
          "match_level": "species",
          "matched_class_225_common": "red fox",
          "matched_class_225_idx": 0,
          "probs_225": [0.91, 0.0, ...],
          "prob_225_sum": 0.94
        },
        "n_animal_detections": 1,
        "multi_animal": false
      }
    }
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

SPECIESNET_RESULTS_PATHS = {
    "gbif":        REPO_ROOT / "data" / "gbif"        / "speciesnet_results.jsonl",
    "inaturalist": REPO_ROOT / "data" / "inaturalist" / "speciesnet_results.jsonl",
    "wikimedia":   REPO_ROOT / "data" / "wikimedia"   / "speciesnet_results.jsonl",
    "openimages":  REPO_ROOT / "data" / "openimages"  / "speciesnet_results.jsonl",
    "images_cv":   REPO_ROOT / "data" / "images_cv"   / "speciesnet_results.jsonl",
}

FILTER_RESULTS_PATHS = {
    source: REPO_ROOT / "data" / source / "filter_results.jsonl"
    for source in SPECIESNET_RESULTS_PATHS
}

TAXONOMY_PATH   = REPO_ROOT / "resources" / "speciesnet_taxonomy_release.txt"
CLASSES_225_PATH = REPO_ROOT / "reports"   / "classes_225.csv"

DEFAULT_MD_CONF          = 0.5
DEFAULT_SN_SCORE         = 0.3
DEFAULT_FAMILY_FAIL_THRESH = 0.5

MATCH_LEVELS = ["species", "genus", "family", "order", "class", "no_match"]

FAIL_REASONS_ORDER = [
    "no_animal_detection",
    "primary_crop_too_small",
    "low_megadetector_confidence",
    "low_speciesnet_confidence",
    "not_in_225_classes",
    "family_mismatch_high_confidence",
    "match_level_order",
    "match_level_class",
    "match_level_no_match",
]


# ── Environment check ─────────────────────────────────────────────────────────

def _check_environment() -> None:
    try:
        import speciesnet  # noqa: F401
    except ImportError:
        print(
            "ERROR: 'speciesnet' is not installed.\n"
            "This script must run inside Dockerfile.speciesnet (Python 3.11).\n"
            "  make speciesnet-build\n"
            "  make speciesnet-start",
            file=sys.stderr,
        )
        sys.exit(1)


# ── Label loading ─────────────────────────────────────────────────────────────

def load_speciesnet_labels() -> dict[int, str]:
    """Load SpeciesNet classifier labels as {int_idx: label_string}.

    Script 6 stored compact integer indices to keep speciesnet_results.jsonl small.
    This function resolves those indices to full 'uuid;class;order;...' strings
    by loading the SpeciesNet classifier.
    """
    from speciesnet import SpeciesNet, DEFAULT_MODEL

    print("Loading SpeciesNet EfficientNetV2-M for label lookup …")
    clf = SpeciesNet(DEFAULT_MODEL, components="classifier", geofence=False).classifier
    print(f"  Classifier loaded. Extracting labels …")

    for attr in ("class_names", "labels"):
        if not hasattr(clf, attr):
            continue
        raw = getattr(clf, attr)

        # Dict-like: {int_idx: label_string}
        if hasattr(raw, "items"):
            result: dict[int, str] = {int(k): str(v) for k, v in raw.items()}
            sample = next(iter(result.values()), "")
            if isinstance(sample, str) and ";" in sample:
                print(f"  Labels from clf.{attr} ({len(result)} classes). "
                      f"[0] = {sample[:70]}")
                return result

        # Sequence
        labels_list = list(raw)
        if labels_list and isinstance(labels_list[0], str) and ";" in labels_list[0]:
            result = {i: v for i, v in enumerate(labels_list)}
            print(f"  Labels from clf.{attr} ({len(result)} classes). "
                  f"[0] = {labels_list[0][:70]}")
            return result

    raise RuntimeError(
        "Could not extract string labels from SpeciesNet classifier.\n"
        "Tried clf.class_names and clf.labels — neither returned 'uuid;...' strings."
    )


# ── Taxonomy lookup ───────────────────────────────────────────────────────────

def load_taxonomy(path: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    """Parse speciesnet_taxonomy_release.txt.

    Format per line: UUID;class;order;family;genus;species;common

    Returns:
      by_genus_species — keyed by "{genus} {species}" (lowercased, both non-empty)
      by_genus         — keyed by genus (lowercased), only for entries with empty species
    """
    by_genus_species: dict[str, dict] = {}
    by_genus: dict[str, dict] = {}

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(";")
            if len(parts) < 7:
                continue
            _, class_, order, family, genus, species, common = parts[:7]
            class_  = class_.lower().strip()
            order   = order.lower().strip()
            family  = family.lower().strip()
            genus   = genus.lower().strip()
            species = species.lower().strip()
            entry = {
                "class_": class_,
                "order":  order,
                "family": family,
                "genus":  genus,
                "species": species,
                "common": common.strip(),
            }
            if genus and species:
                by_genus_species[f"{genus} {species}"] = entry
            elif genus:
                by_genus.setdefault(genus, entry)  # first entry wins

    return by_genus_species, by_genus


# ── 225-class lookup ──────────────────────────────────────────────────────────

def load_classes_225(
    path: Path,
) -> tuple[dict[str, dict], dict[str, int], dict[str, int], dict[str, int]]:
    """Parse classes_225.csv (common_name, scientific_name, level).

    Levels in the file: 'species', 'genus', or 'family'.

    Returns:
      by_common            — keyed by lowercased common_name
      genus_species_to_225 — "genus species" → 0-based row index (species-level rows)
      genus_to_225         — "genus"         → 0-based row index (genus-level rows only)
      family_to_225        — "family"        → 0-based row index (family-level rows only)
    """
    by_common: dict[str, dict] = {}
    genus_species_to_225: dict[str, int] = {}
    genus_to_225: dict[str, int] = {}
    family_to_225: dict[str, int] = {}

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx_225, row in enumerate(reader):
            common  = row["common_name"].strip().lower()
            sci     = row["scientific_name"].strip().lower()
            level   = row["level"].strip()
            by_common[common] = {"scientific_name": sci, "level": level, "idx_225": idx_225}

            parts = sci.split()
            if level == "species" and len(parts) >= 2:
                genus_species_to_225[f"{parts[0]} {' '.join(parts[1:])}"] = idx_225
            elif level == "genus" and parts:
                genus_to_225[parts[0]] = idx_225
            elif level == "family" and parts:
                family_to_225[parts[0]] = idx_225

    return by_common, genus_species_to_225, genus_to_225, family_to_225


# ── Match logic ───────────────────────────────────────────────────────────────

def _compute_match_level(
    pred: dict,
    exp_genus: str,
    exp_species: str,
    exp_family: str,
    exp_tax: dict,
    exp_level: str,
) -> str:
    """Return one of: species, genus, family, order, class, no_match.

    For genus-level and family-level expected classes the finest possible match
    is labelled 'species' since it represents the best achievable precision given
    the label granularity.
    """
    pg = pred["genus"]
    ps = pred["species"]

    if exp_level == "family":
        # Best achievable is a family match
        if pred["family"] == exp_family:
            return "species"
        if pred["order"] == exp_tax.get("order", "NONE"):
            return "order"
        if pred["class_"] == "mammalia" and exp_tax.get("class_") == "mammalia":
            return "class"
        return "no_match"

    if exp_level == "genus":
        # Best achievable is a genus match
        if pg == exp_genus:
            return "species"
        if pred["family"] == exp_tax.get("family", "NONE"):
            return "family"
        if pred["order"] == exp_tax.get("order", "NONE"):
            return "order"
        if pred["class_"] == "mammalia" and exp_tax.get("class_") == "mammalia":
            return "class"
        return "no_match"

    # Species-level expected class
    if pg == exp_genus and ps == exp_species:
        return "species"
    if pg == exp_genus:
        return "genus"
    if pred["family"] == exp_tax.get("family", "NONE"):
        return "family"
    if pred["order"] == exp_tax.get("order", "NONE"):
        return "order"
    if pred["class_"] == "mammalia" and exp_tax.get("class_") == "mammalia":
        return "class"
    return "no_match"


def _apply_match_rules(
    match_level: str,
    top1_score: float,
    family_fail_thresh: float,
) -> tuple[bool, str | None]:
    if match_level in ("species", "genus"):
        return True, None
    if match_level == "family":
        if top1_score >= family_fail_thresh:
            return False, "family_mismatch_high_confidence"
        return True, None
    if match_level == "order":
        return False, "match_level_order"
    if match_level == "class":
        return False, "match_level_class"
    return False, "match_level_no_match"


# ── 225-class probability vector ──────────────────────────────────────────────

def compute_probs_225(
    speciesnet_scores: dict[str, float] | list[float],
    idx_to_label: dict[int, str],
    genus_species_to_225: dict[str, int],
    genus_to_225: dict[str, int],
    family_to_225: dict[str, int],
) -> tuple[list[float], float]:
    """Project sparse SpeciesNet scores onto the 225-class probability vector.

    speciesnet_scores may be either:
      - a sparse dict {str(idx): score} (current format, min_score-filtered)
      - a full list of floats indexed by class (legacy pre-migration format)

    Lookup priority per SpeciesNet class: species → genus → family.
    Returns (probs_225, prob_225_sum). prob_225_sum < 1.0 means some probability
    mass belongs to species outside the 225-class set (expected for most images).
    prob_225_sum ≈ 0.0 means the image is entirely out-of-distribution.
    """
    probs = [0.0] * 225

    if isinstance(speciesnet_scores, list):
        items = ((str(i), s) for i, s in enumerate(speciesnet_scores) if s > 0)
    else:
        items = speciesnet_scores.items()

    for idx_str, score in items:
        label = idx_to_label.get(int(idx_str))
        if label is None:
            continue
        parts = label.split(";")
        if len(parts) < 6:
            continue
        family  = parts[3].lower().strip()
        genus   = parts[4].lower().strip()
        species = parts[5].lower().strip()

        cls225_idx = genus_species_to_225.get(f"{genus} {species}")
        if cls225_idx is None:
            cls225_idx = genus_to_225.get(genus)
        if cls225_idx is None:
            cls225_idx = family_to_225.get(family)
        if cls225_idx is not None:
            probs[cls225_idx] += score

    total = sum(probs)
    return probs, round(total, 6)


# ── Per-record evaluation ─────────────────────────────────────────────────────

def evaluate_record(
    rec: dict,
    idx_to_label: dict[int, str],
    tax_by_gs: dict[str, dict],
    tax_by_genus: dict[str, dict],
    class225_by_common: dict[str, dict],
    genus_species_to_225: dict[str, int],
    genus_to_225: dict[str, int],
    family_to_225: dict[str, int],
    md_conf: float,
    sn_score: float,
    family_fail_thresh: float,
) -> dict:
    """Evaluate one record from speciesnet_results.jsonl.

    Returns a dict with:
      pass, reason, match_level, primary, probs_225, prob_225_sum,
      n_animal_detections, multi_animal, expected_common, idx_225
    """
    n_animal = rec.get("n_animal_detections", 0)
    result: dict = {
        "pass":              False,
        "reason":            None,
        "match_level":       None,
        "primary":           None,
        "pred_label":        None,
        "exp_genus":         None,
        "exp_species":       None,
        "class225_entry":    None,
        "probs_225":         None,
        "prob_225_sum":      None,
        "n_animal_detections": n_animal,
        "multi_animal":      n_animal > 1,
        "expected_common":   rec.get("expected_common", ""),
    }

    if n_animal == 0:
        result["reason"] = "no_animal_detection"
        return result

    # Primary detection = detection_idx 0
    primary = next(
        (d for d in (rec.get("speciesnet_detections") or []) if d.get("detection_idx") == 0),
        None,
    )
    if primary is None:
        result["reason"] = "no_animal_detection"
        return result

    result["primary"] = primary

    if primary.get("speciesnet_skipped"):
        result["reason"] = "primary_crop_too_small"
        return result

    if primary.get("megadetector_conf", 0.0) < md_conf:
        result["reason"] = "low_megadetector_confidence"
        return result

    top1_score = primary.get("speciesnet_top1_score", 0.0)
    if top1_score < sn_score:
        result["reason"] = "low_speciesnet_confidence"
        return result

    # Expected taxonomy
    expected_norm = rec.get("expected_common", "").lower().replace("_", " ").strip()
    class225_entry = class225_by_common.get(expected_norm)
    if class225_entry is None:
        result["reason"] = "not_in_225_classes"
        return result

    result["class225_entry"] = class225_entry
    sci_parts = class225_entry["scientific_name"].split()
    exp_level   = class225_entry["level"]
    exp_genus   = sci_parts[0] if sci_parts else ""
    exp_species = " ".join(sci_parts[1:]) if len(sci_parts) > 1 else ""
    # For family-level entries, scientific_name IS the family name
    exp_family  = sci_parts[0] if exp_level == "family" and sci_parts else ""
    result["exp_genus"]   = exp_genus
    result["exp_species"] = exp_species

    exp_tax = (
        tax_by_gs.get(f"{exp_genus} {exp_species}")
        or tax_by_genus.get(exp_genus)
        or {}
    )

    # Predicted taxonomy
    top1_idx = primary.get("speciesnet_top1_idx")
    pred_label = idx_to_label.get(int(top1_idx)) if top1_idx is not None else None
    if pred_label is None:
        result["reason"] = "not_in_225_classes"
        return result

    result["pred_label"] = pred_label
    parts = pred_label.split(";")
    if len(parts) < 6:
        result["reason"] = "not_in_225_classes"
        return result

    pred_tax = {
        "class_": parts[1].lower().strip(),
        "order":  parts[2].lower().strip(),
        "family": parts[3].lower().strip(),
        "genus":  parts[4].lower().strip(),
        "species": parts[5].lower().strip(),
        "common": parts[6].strip() if len(parts) > 6 else "",
    }

    # Match level and pass/fail
    match_level = _compute_match_level(
        pred_tax, exp_genus, exp_species, exp_family, exp_tax, exp_level
    )
    result["match_level"] = match_level

    passed, reason = _apply_match_rules(match_level, top1_score, family_fail_thresh)
    result["pass"]   = passed
    result["reason"] = reason

    # 225-class probability vector
    speciesnet_scores = primary.get("speciesnet_scores") or {}
    probs_225, prob_225_sum = compute_probs_225(
        speciesnet_scores, idx_to_label, genus_species_to_225, genus_to_225, family_to_225
    )
    result["probs_225"]    = probs_225
    result["prob_225_sum"] = prob_225_sum

    return result


# ── Statistics accumulation ───────────────────────────────────────────────────

@dataclass
class Stats:
    n_total:        int = 0
    n_pass:         int = 0
    n_fail:         int = 0
    n_multi_animal: int = 0
    fail_reasons:   dict = field(default_factory=lambda: defaultdict(int))
    match_levels:   dict = field(default_factory=lambda: defaultdict(int))
    prob_225_sums:  list = field(default_factory=list)
    pre_filter_counts: dict = field(default_factory=lambda: defaultdict(int))
    per_class:      dict = field(default_factory=lambda: defaultdict(
        lambda: {"n_total": 0, "n_pass": 0, "fail_reasons": defaultdict(int),
                 "match_levels": defaultdict(int)}
    ))

    def record(self, ev: dict) -> None:
        self.n_total += 1
        cls_key = ev["expected_common"].lower().replace("_", " ").strip()
        cls_stats = self.per_class[cls_key]
        cls_stats["n_total"] += 1

        if ev["multi_animal"]:
            self.n_multi_animal += 1

        if ev["pass"]:
            self.n_pass += 1
            cls_stats["n_pass"] += 1
        else:
            self.n_fail += 1
            reason = ev["reason"] or "unknown"
            self.fail_reasons[reason] += 1
            cls_stats["fail_reasons"][reason] += 1

        if ev["match_level"]:
            self.match_levels[ev["match_level"]] += 1
            cls_stats["match_levels"][ev["match_level"]] += 1

        if ev["prob_225_sum"] is not None:
            self.prob_225_sums.append(ev["prob_225_sum"])


# ── Statistics printer ────────────────────────────────────────────────────────

def _pct(num: int, denom: int) -> str:
    if denom == 0:
        return "  n/a"
    return f"{100 * num / denom:5.1f}%"


def print_stats(source: str, stats: Stats) -> None:
    n = stats.n_total
    sep = "─" * 60

    print(f"\n{sep}")
    print(f"[{source}]  {n:,} records processed")
    print(f"  Pass: {stats.n_pass:>8,}  ({_pct(stats.n_pass, n)})")
    print(f"  Fail: {stats.n_fail:>8,}  ({_pct(stats.n_fail, n)})")

    if stats.fail_reasons:
        print(f"\n  Fail reasons:")
        for reason in FAIL_REASONS_ORDER:
            cnt = stats.fail_reasons.get(reason, 0)
            if cnt:
                print(f"    {reason:<40}  {cnt:>7,}  ({_pct(cnt, n)})")
        for reason, cnt in stats.fail_reasons.items():
            if reason not in FAIL_REASONS_ORDER:
                print(f"    {reason:<40}  {cnt:>7,}  ({_pct(cnt, n)})")

    n_classified = sum(stats.match_levels.values())
    if n_classified:
        print(f"\n  Match levels  ({n_classified:,} images with valid classification):")
        for lvl in MATCH_LEVELS:
            cnt = stats.match_levels.get(lvl, 0)
            if cnt or lvl in ("species", "no_match"):
                print(f"    {lvl:<12}  {cnt:>7,}  ({_pct(cnt, n_classified)})")

    print(f"\n  Multi-animal images: {stats.n_multi_animal:,}  ({_pct(stats.n_multi_animal, n)})")

    if stats.prob_225_sums:
        s = sorted(stats.prob_225_sums)
        p = lambda pct: s[max(0, int(len(s) * pct / 100) - 1)]
        print(f"\n  prob_225_sum  (probability mass mapped to any 225-class, "
              f"{len(s):,} images):")
        print(f"    mean={statistics.mean(s):.3f}  "
              f"median={statistics.median(s):.3f}  "
              f"p10={p(10):.3f}  p90={p(90):.3f}  "
              f"zeros={sum(1 for v in s if v == 0.0)}")

    # Per-class breakdown
    print(f"\n  Per-class breakdown  ({len(stats.per_class)} classes):")
    header = f"  {'Class':<35}  {'pre-filter':>10}  {'sn-input':>8}  {'pass':>6}  {'pass%':>6}  top fail reason"
    print(header)
    print("  " + "─" * (len(header) - 2))

    for cls_name in sorted(stats.per_class):
        cs = stats.per_class[cls_name]
        ct = cs["n_total"]
        cp = cs["n_pass"]
        pre_cnt = stats.pre_filter_counts.get(cls_name, 0)
        pct_str = _pct(cp, ct)
        if cs["fail_reasons"]:
            top_fail_reason, top_fail_cnt = max(cs["fail_reasons"].items(), key=lambda x: x[1])
            fail_str = f"{top_fail_reason}={top_fail_cnt}"
        else:
            fail_str = ""
        print(f"  {cls_name:<35}  {pre_cnt:>10,}  {ct:>8,}  {cp:>6,}  {pct_str}  {fail_str}")

    # Ranked list
    ranked = sorted(stats.per_class.items(), key=lambda x: x[1]["n_pass"], reverse=True)
    print(f"\n  Classes ranked by passed images (most → least):")
    for rank, (cls_name, cs) in enumerate(ranked, 1):
        print(f"    {rank:>3}. {cls_name:<35}  {cs['n_pass']:>6,}")

    print(sep)


# ── Markdown report ───────────────────────────────────────────────────────────

def _pct_md(num: int, denom: int) -> str:
    if denom == 0:
        return "n/a"
    return f"{100 * num / denom:.1f}%"


def format_stats_md(source: str, stats: Stats) -> str:
    """Return a markdown section string for one source."""
    lines: list[str] = []
    n = stats.n_total

    lines.append(f"## {source} — {n:,} records\n")
    lines.append("| | Count | % |")
    lines.append("|---|---:|---:|")
    lines.append(f"| Pass | {stats.n_pass:,} | {_pct_md(stats.n_pass, n)} |")
    lines.append(f"| Fail | {stats.n_fail:,} | {_pct_md(stats.n_fail, n)} |")
    lines.append("")

    if stats.fail_reasons:
        lines.append("### Fail Reasons\n")
        lines.append("| Reason | Count | % |")
        lines.append("|---|---:|---:|")
        for reason in FAIL_REASONS_ORDER:
            cnt = stats.fail_reasons.get(reason, 0)
            if cnt:
                lines.append(f"| {reason} | {cnt:,} | {_pct_md(cnt, n)} |")
        for reason, cnt in stats.fail_reasons.items():
            if reason not in FAIL_REASONS_ORDER:
                lines.append(f"| {reason} | {cnt:,} | {_pct_md(cnt, n)} |")
        lines.append("")

    n_classified = sum(stats.match_levels.values())
    if n_classified:
        lines.append(f"### Match Levels ({n_classified:,} images with valid classification)\n")
        lines.append("| Level | Count | % |")
        lines.append("|---|---:|---:|")
        for lvl in MATCH_LEVELS:
            cnt = stats.match_levels.get(lvl, 0)
            if cnt or lvl in ("species", "no_match"):
                lines.append(f"| {lvl} | {cnt:,} | {_pct_md(cnt, n_classified)} |")
        lines.append("")

    lines.append(
        f"**Multi-animal images:** {stats.n_multi_animal:,}"
        f" ({_pct_md(stats.n_multi_animal, n)})\n"
    )

    if stats.prob_225_sums:
        s = sorted(stats.prob_225_sums)
        p = lambda pct: s[max(0, int(len(s) * pct / 100) - 1)]
        lines.append(
            f"**prob\\_225\\_sum** ({len(s):,} images with valid classification):"
            f" mean={statistics.mean(s):.3f}"
            f"  median={statistics.median(s):.3f}"
            f"  p10={p(10):.3f}  p90={p(90):.3f}"
            f"  zeros={sum(1 for v in s if v == 0.0)}\n"
        )

    lines.append(f"### Per-Class Breakdown ({len(stats.per_class)} classes)\n")
    lines.append("| Class | Pre-filter | SN Input | Pass | Pass% | Top Fail Reason |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for cls_name in sorted(stats.per_class):
        cs = stats.per_class[cls_name]
        ct = cs["n_total"]
        cp = cs["n_pass"]
        pre_cnt = stats.pre_filter_counts.get(cls_name, 0)
        if cs["fail_reasons"]:
            top_reason, top_cnt = max(cs["fail_reasons"].items(), key=lambda x: x[1])
            fail_str = f"{top_reason}={top_cnt}"
        else:
            fail_str = ""
        lines.append(
            f"| {cls_name} | {pre_cnt:,} | {ct:,} | {cp:,} | {_pct_md(cp, ct)} | {fail_str} |"
        )
    lines.append("")

    ranked = sorted(stats.per_class.items(), key=lambda x: x[1]["n_pass"], reverse=True)
    lines.append("### Classes Ranked by Passed Images\n")
    lines.append("| Rank | Class | Passed |")
    lines.append("|---:|---|---:|")
    for rank, (cls_name, cs) in enumerate(ranked, 1):
        lines.append(f"| {rank} | {cls_name} | {cs['n_pass']:,} |")
    lines.append("")

    return "\n".join(lines)


def merge_stats(sources_stats: dict[str, Stats]) -> Stats:
    """Combine multiple per-source Stats into one aggregate Stats object."""
    merged = Stats()
    for st in sources_stats.values():
        merged.n_total        += st.n_total
        merged.n_pass         += st.n_pass
        merged.n_fail         += st.n_fail
        merged.n_multi_animal += st.n_multi_animal
        for reason, cnt in st.fail_reasons.items():
            merged.fail_reasons[reason] += cnt
        for lvl, cnt in st.match_levels.items():
            merged.match_levels[lvl] += cnt
        merged.prob_225_sums.extend(st.prob_225_sums)
        for cls_name, cnt in st.pre_filter_counts.items():
            merged.pre_filter_counts[cls_name] += cnt
        for cls_name, cs in st.per_class.items():
            mc = merged.per_class[cls_name]
            mc["n_total"] += cs["n_total"]
            mc["n_pass"]  += cs["n_pass"]
            for reason, cnt in cs["fail_reasons"].items():
                mc["fail_reasons"][reason] += cnt
            for lvl, cnt in cs["match_levels"].items():
                mc["match_levels"][lvl] += cnt
    return merged


def write_report(
    sources_stats: dict[str, Stats],
    args: argparse.Namespace,
) -> None:
    report_path = REPO_ROOT / "reports" / "speciesnet_filter.md"

    header = [
        "# SpeciesNet Filter Report\n",
        f"**Thresholds:** md_conf≥{args.md_conf}  "
        f"sn_score≥{args.sn_score}  "
        f"family_fail≥{args.family_fail_thresh}\n",
        f"**Mode:** {'WRITE' if args.write else 'STATS ONLY'}\n",
        "---\n",
    ]

    per_source = [format_stats_md(src, st) for src, st in sources_stats.items()]
    if len(sources_stats) > 1:
        combined = [format_stats_md("all sources combined", merge_stats(sources_stats))]
    else:
        combined = []

    report_path.write_text("\n".join(header + combined + per_source), encoding="utf-8")
    print(f"\nReport written → {report_path.relative_to(REPO_ROOT)}")


# ── Phase 2: build speciesnet_eval dict ───────────────────────────────────────

def _build_speciesnet_eval(ev: dict, class225_by_common: dict[str, dict]) -> dict:
    """Construct the speciesnet_eval dict to be merged into filter_results.jsonl."""
    primary = ev.get("primary")
    class225_entry = ev.get("class225_entry")
    pred_label = ev.get("pred_label")

    primary_det_out: dict | None = None
    if primary is not None and pred_label is not None and class225_entry is not None:
        parts = pred_label.split(";")
        pred_genus   = parts[4].strip() if len(parts) > 4 else ""
        pred_species = parts[5].strip() if len(parts) > 5 else ""
        pred_common  = parts[6].strip() if len(parts) > 6 else ""
        primary_det_out = {
            "detection_idx":           primary.get("detection_idx"),
            "megadetector_conf":       primary.get("megadetector_conf"),
            "speciesnet_top1_idx":     primary.get("speciesnet_top1_idx"),
            "speciesnet_top1_label":   pred_label,
            "speciesnet_top1_scientific": f"{pred_genus} {pred_species}".strip(),
            "speciesnet_top1_common":  pred_common,
            "speciesnet_top1_score":   primary.get("speciesnet_top1_score"),
            "match_level":             ev["match_level"],
            "matched_class_225_common": class225_entry["scientific_name"],
            "matched_class_225_idx":    class225_entry["idx_225"],
            "probs_225":               ev["probs_225"],
            "prob_225_sum":            ev["prob_225_sum"],
        }
    elif primary is not None and ev["reason"] in (
        "primary_crop_too_small",
        "low_megadetector_confidence",
        "low_speciesnet_confidence",
    ):
        primary_det_out = {
            "detection_idx":     primary.get("detection_idx"),
            "megadetector_conf": primary.get("megadetector_conf"),
            "speciesnet_top1_score": primary.get("speciesnet_top1_score"),
        }

    return {
        "pass":                ev["pass"],
        "reason":              ev["reason"],
        "primary_detection":   primary_det_out,
        "n_animal_detections": ev["n_animal_detections"],
        "multi_animal":        ev["multi_animal"],
    }


# ── Per-source processing ─────────────────────────────────────────────────────

def process_source(
    source: str,
    idx_to_label: dict[int, str],
    tax_by_gs: dict[str, dict],
    tax_by_genus: dict[str, dict],
    class225_by_common: dict[str, dict],
    genus_species_to_225: dict[str, int],
    genus_to_225: dict[str, int],
    family_to_225: dict[str, int],
    md_conf: float,
    sn_score: float,
    family_fail_thresh: float,
    write: bool,
) -> Stats | None:
    sn_path     = SPECIESNET_RESULTS_PATHS[source]
    filter_path = FILTER_RESULTS_PATHS[source]

    if not sn_path.exists():
        print(f"\n[{source}] speciesnet_results.jsonl not found — skipping.")
        return None

    stats = Stats()

    # Count raw images per class from filter_results.jsonl (before any stage filter)
    if filter_path.exists():
        with open(filter_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    fp = entry.get("filepath", "")
                    class_name = Path(fp).parent.name.lower().replace("_", " ")
                    stats.pre_filter_counts[class_name] += 1
                except Exception:
                    continue

    # Phase 1: stream speciesnet_results.jsonl, evaluate, accumulate stats
    eval_by_filepath: dict[str, dict] = {}

    with open(sn_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ev = evaluate_record(
                rec, idx_to_label, tax_by_gs, tax_by_genus,
                class225_by_common, genus_species_to_225, genus_to_225, family_to_225,
                md_conf, sn_score, family_fail_thresh,
            )
            stats.record(ev)
            if write:
                eval_by_filepath[rec["filepath"]] = ev

    print_stats(source, stats)

    if not write:
        return stats

    # Phase 2: update filter_results.jsonl
    if not filter_path.exists():
        print(f"[{source}] filter_results.jsonl not found — cannot write.")
        return

    tmp_path = filter_path.with_suffix(".jsonl.tmp")
    n_updated = 0
    n_already = 0

    with open(filter_path, encoding="utf-8") as f_in, \
         open(tmp_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            fp = entry.get("filepath", "")
            ev = eval_by_filepath.get(fp)

            if ev is None:
                # No speciesnet result for this entry — leave unchanged
                f_out.write(json.dumps(entry) + "\n")
                continue

            if "speciesnet_eval" in entry:
                n_already += 1
                f_out.write(json.dumps(entry) + "\n")
                continue

            # Merge speciesnet_eval
            speciesnet_eval = _build_speciesnet_eval(ev, class225_by_common)
            entry["speciesnet_eval"] = speciesnet_eval

            if not ev["pass"]:
                entry["passed"]       = False
                entry["stage_failed"] = "speciesnet"
                entry["reason"]       = ev["reason"]

            stages_done = entry.get("stages_done") or []
            if "speciesnet" not in stages_done:
                stages_done = list(stages_done) + ["speciesnet"]
            entry["stages_done"] = stages_done

            f_out.write(json.dumps(entry) + "\n")
            n_updated += 1

    tmp_path.replace(filter_path)

    msg_parts = [f"[{source}] wrote {n_updated:,} speciesnet_eval records"]
    if n_already:
        msg_parts.append(f"{n_already:,} already had speciesnet_eval (skipped)")
    print("  " + "; ".join(msg_parts))

    return stats


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        required=True,
        choices=list(SPECIESNET_RESULTS_PATHS.keys()) + ["all"],
        help="Dataset source to process, or 'all' for every source in sequence.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Phase 2: merge speciesnet_eval into filter_results.jsonl. "
             "Default (no --write) prints statistics only without any file writes.",
    )
    parser.add_argument(
        "--md-conf",
        type=float,
        default=DEFAULT_MD_CONF,
        metavar="CONF",
        help=f"MegaDetector confidence floor for primary detection "
             f"(default: {DEFAULT_MD_CONF}).",
    )
    parser.add_argument(
        "--sn-score",
        type=float,
        default=DEFAULT_SN_SCORE,
        metavar="PROB",
        help=f"SpeciesNet top-1 score floor (default: {DEFAULT_SN_SCORE}).",
    )
    parser.add_argument(
        "--family-fail-thresh",
        type=float,
        default=DEFAULT_FAMILY_FAIL_THRESH,
        metavar="PROB",
        help=f"Family-match + score >= this threshold → fail "
             f"(default: {DEFAULT_FAMILY_FAIL_THRESH}).",
    )
    args = parser.parse_args()

    _check_environment()

    # Load shared tables once
    idx_to_label = load_speciesnet_labels()

    print(f"Loading taxonomy from {TAXONOMY_PATH.name} …")
    tax_by_gs, tax_by_genus = load_taxonomy(TAXONOMY_PATH)
    print(f"  {len(tax_by_gs):,} species-level entries, "
          f"{len(tax_by_genus):,} genus-level entries.")

    print(f"Loading 225 classes from {CLASSES_225_PATH.name} …")
    class225_by_common, genus_species_to_225, genus_to_225, family_to_225 = (
        load_classes_225(CLASSES_225_PATH)
    )
    print(f"  {len(class225_by_common)} classes "
          f"({len(genus_species_to_225)} species-level, {len(genus_to_225)} genus-level, "
          f"{len(family_to_225)} family-level).")

    if args.write:
        print("\nMode: WRITE — will update filter_results.jsonl files.")
    else:
        print("\nMode: STATS ONLY — no files will be modified.")

    print(f"Thresholds: md_conf>={args.md_conf}  sn_score>={args.sn_score}  "
          f"family_fail>={args.family_fail_thresh}")

    sources = (
        list(SPECIESNET_RESULTS_PATHS.keys()) if args.source == "all" else [args.source]
    )

    sources_stats: dict[str, Stats] = {}
    for source in sources:
        stats = process_source(
            source,
            idx_to_label, tax_by_gs, tax_by_genus,
            class225_by_common, genus_species_to_225, genus_to_225, family_to_225,
            md_conf=args.md_conf,
            sn_score=args.sn_score,
            family_fail_thresh=args.family_fail_thresh,
            write=args.write,
        )
        if stats is not None:
            sources_stats[source] = stats

    if sources_stats:
        write_report(sources_stats, args)

    print("\nAll done.")


if __name__ == "__main__":
    main()
