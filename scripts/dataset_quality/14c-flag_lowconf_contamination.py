"""Flag cross-species contamination in low-confidence MegaDetector boxes.

Script 14 covered images with ≥2 significant detections at MD conf ≥ 0.5.
Script 15b subsequently added all detections at 0.1 ≤ conf < 0.5 as GT
annotations ("megadetector_lowconf" source).  Those boxes were never evaluated
for cross-species contamination.  This script closes that gap.

For every image NOT already covered by the ≥ 0.5 review pipeline, each
detection in the range [md_conf_lower, md_conf_upper) is evaluated against the
image's expected class using the same taxonomy machinery as script 14.  If a
detection is classified as a genuinely different mammal species — outside the
configured tolerance band and with sufficient SpeciesNet confidence — the image
is flagged for review.

Skipped images
--------------
Filepaths already present in ``reports/multi_animal_contamination_review.json``
are excluded: those images were visible in the ≥ 0.5 review queue where the
gray context boxes (10–50% conf) were shown to the reviewer.

Tolerance levels (--tolerance, default 'family'): same as script 14.

The script is NON-DESTRUCTIVE: it only writes report files.

Usage
-----
    # Flag all sources (default thresholds):
    python scripts/dataset_quality/14c-flag_lowconf_contamination.py --source all

    # Custom confidence window:
    python scripts/dataset_quality/14c-flag_lowconf_contamination.py \\
        --source all --md-conf-lower 0.15 --md-conf-upper 0.5

    # Tighter tolerance:
    python scripts/dataset_quality/14c-flag_lowconf_contamination.py \\
        --source all --tolerance genus

Outputs (written to reports/)
------------------------------
    lowconf_contamination.csv           — one row per flagged/uncertain box
    lowconf_contamination_review.json   — per-image dict (same schema as
                                          multi_animal_contamination_review.json;
                                          compatible with 14b --review-json)
    lowconf_contamination.md            — human summary

See also
--------
    scripts/dataset_quality/14-flag_multi_animal_contamination.py
        — ≥ 0.5 tier; provides the excluded-filepaths set and shared helpers
    scripts/dataset_quality/14b-review_contamination.py
        — manual review UI; pass --review-json reports/lowconf_contamination_review.json
    scripts/dataset_quality/15c-remove_lowconf_contamination.py
        — applies decisions to COCO JSONs (removes megadetector_lowconf annotations)
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# ── Load script 7's taxonomy machinery (no speciesnet import needed) ──────────

_s7_path = Path(__file__).resolve().parent / "7-filter_speciesnet.py"
_spec = importlib.util.spec_from_file_location("filter_speciesnet_s7_14c", _s7_path)
s7 = importlib.util.module_from_spec(_spec)
sys.modules["filter_speciesnet_s7_14c"] = s7
_spec.loader.exec_module(s7)

_compute_match_level    = s7._compute_match_level
_load_taxonomy          = s7.load_taxonomy
_load_classes_225       = s7.load_classes_225
TAXONOMY_PATH           = s7.TAXONOMY_PATH
CLASSES_225_PATH        = s7.CLASSES_225_PATH
SPECIESNET_RESULTS_PATHS = s7.SPECIESNET_RESULTS_PATHS
MATCH_LEVELS            = s7.MATCH_LEVELS

# ── Constants ─────────────────────────────────────────────────────────────────

SPECIESNET_LABELS_PATH = REPO_ROOT / "data" / "speciesnet_labels.json"
REPORTS_DIR            = REPO_ROOT / "reports"

# The ≥ 0.5 pipeline review JSON — filepaths here are excluded from this run.
HIGH_CONF_REVIEW_JSON  = REPORTS_DIR / "multi_animal_contamination_review.json"

DEFAULT_MD_CONF_LOWER = 0.1   # inclusive lower bound of low-conf tier
DEFAULT_MD_CONF_UPPER = 0.5   # exclusive upper bound (< 0.5 means not in ≥ 0.5 tier)
DEFAULT_SN_SCORE      = 0.3   # SpeciesNet top-1 score floor for "confident" flag

TOLERANCE_BANDS: dict[str, set[str]] = {
    "genus":  {"species", "genus"},
    "family": {"species", "genus", "family"},
    "order":  {"species", "genus", "family", "order"},
}

LOOKALIKE_GROUPS: list[set[str]] = []


# ── Load SpeciesNet labels ────────────────────────────────────────────────────

def load_speciesnet_labels(path: Path) -> dict[int, str]:
    with open(path, encoding="utf-8") as f:
        raw: dict[str, str] = json.load(f)
    return {int(k): v for k, v in raw.items()}


# ── Taxonomy helpers (copied from script 14 — same logic) ────────────────────

def _resolve_expected_class(
    expected_common: str,
    class225_by_common: dict[str, dict],
    tax_by_gs: dict[str, dict],
    tax_by_genus: dict[str, dict],
) -> tuple[str | None, str, str, str, dict, str]:
    expected_norm = expected_common.lower().replace("_", " ").strip()
    class225_entry = class225_by_common.get(expected_norm)
    if class225_entry is None:
        return "not_in_225", "", "", "", {}, ""

    sci_parts = class225_entry["scientific_name"].split()
    exp_level   = class225_entry["level"]
    exp_genus   = sci_parts[0] if sci_parts else ""
    exp_species = " ".join(sci_parts[1:]) if len(sci_parts) > 1 else ""
    exp_family  = sci_parts[0] if exp_level == "family" and sci_parts else ""

    exp_tax = (
        tax_by_gs.get(f"{exp_genus} {exp_species}")
        or tax_by_genus.get(exp_genus)
        or {}
    )

    return None, exp_genus, exp_species, exp_family, exp_tax, exp_level


def _build_pred_taxonomy(label: str) -> dict | None:
    parts = label.split(";")
    if len(parts) < 6:
        return None
    pred_tax = {
        "class_":  parts[1].lower().strip(),
        "order":   parts[2].lower().strip(),
        "family":  parts[3].lower().strip(),
        "genus":   parts[4].lower().strip(),
        "species": parts[5].lower().strip(),
        "common":  parts[6].strip() if len(parts) > 6 else "",
    }
    return pred_tax


def _is_lookalike(pred_tax: dict, exp_tax: dict) -> bool:
    if not LOOKALIKE_GROUPS:
        return False
    pred_identifiers = {pred_tax.get("family", ""), pred_tax.get("genus", "")} - {""}
    exp_identifiers  = {exp_tax.get("family", ""), exp_tax.get("genus", "")} - {""}
    for group in LOOKALIKE_GROUPS:
        if pred_identifiers & group and exp_identifiers & group:
            return True
    return False


# ── Box evaluation ────────────────────────────────────────────────────────────

def _evaluate_box(
    det: dict,
    exp_genus: str,
    exp_species: str,
    exp_family: str,
    exp_tax: dict,
    exp_level: str,
    idx_to_label: dict[int, str],
    tolerated: set[str],
    sn_score_thresh: float,
) -> dict | None:
    """Evaluate one low-conf detection against the image's expected class.

    Returns a result dict or None if the prediction cannot be resolved.
    """
    top1_idx = det.get("speciesnet_top1_idx")
    if top1_idx is None:
        return None

    label = idx_to_label.get(int(top1_idx))
    if label is None:
        return None

    pred_tax = _build_pred_taxonomy(label)
    if pred_tax is None:
        return None

    if pred_tax["class_"] != "mammalia":
        return None

    match_level = _compute_match_level(
        pred_tax, exp_genus, exp_species, exp_family, exp_tax, exp_level
    )

    if _is_lookalike(pred_tax, exp_tax):
        verdict = "consistent"
    elif match_level in tolerated:
        verdict = "consistent"
    else:
        top1_score = det.get("speciesnet_top1_score", 0.0)
        if top1_score >= sn_score_thresh:
            verdict = "flag"
        else:
            verdict = "uncertain"

    pred_genus      = pred_tax["genus"]
    pred_species    = pred_tax["species"]
    pred_common     = pred_tax["common"]
    pred_scientific = f"{pred_genus} {pred_species}".strip() if pred_genus else ""

    return {
        "detection_idx":      det.get("detection_idx"),
        "bbox_norm":          det.get("bbox_norm"),
        "megadetector_conf":  det.get("megadetector_conf"),
        "pred_common":        pred_common,
        "pred_scientific":    pred_scientific,
        "pred_top1_score":    det.get("speciesnet_top1_score", 0.0),
        "match_level":        match_level,
        "verdict":            verdict,
    }


# ── Statistics ────────────────────────────────────────────────────────────────

@dataclass
class LowConfStats:
    n_total:               int = 0   # total records scanned
    n_excluded:            int = 0   # records skipped (already in ≥ 0.5 review)
    n_with_lowconf:        int = 0   # images with ≥1 qualifying low-conf detection
    n_expected_not_in_225: int = 0   # images skipped: expected class not in 225
    n_flagged:             int = 0   # images with ≥1 'flag' verdict
    n_uncertain_only:      int = 0   # images with 'uncertain' but no 'flag'
    n_consistent_only:     int = 0   # images where all low-conf boxes are consistent

    box_verdicts:           dict[str, int] = field(default_factory=lambda: defaultdict(int))
    offending_match_levels: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    flagged_by_class:       dict[str, int] = field(default_factory=lambda: defaultdict(int))
    flagged_by_source:      dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def record_image(
        self,
        expected_class: str,
        source: str,
        offending: list[dict],
        all_results: list[dict],
    ) -> None:
        self.n_with_lowconf += 1
        verdicts = {b["verdict"] for b in offending}

        if "flag" in verdicts:
            self.n_flagged += 1
            self.flagged_by_class[expected_class] += 1
            self.flagged_by_source[source] += 1
        elif "uncertain" in verdicts:
            self.n_uncertain_only += 1
        else:
            self.n_consistent_only += 1

        for box in offending:
            self.box_verdicts[box["verdict"]] += 1
            self.offending_match_levels[box["match_level"]] += 1


# ── Utility ───────────────────────────────────────────────────────────────────

def _pct(num: int, denom: int) -> str:
    if denom == 0:
        return "  n/a"
    return f"{100 * num / denom:5.1f}%"


def _pct_md(num: int, denom: int) -> str:
    if denom == 0:
        return "n/a"
    return f"{100 * num / denom:.1f}%"


# ── Per-source processing ─────────────────────────────────────────────────────

def process_source(
    source: str,
    idx_to_label: dict[int, str],
    tax_by_gs: dict[str, dict],
    tax_by_genus: dict[str, dict],
    class225_by_common: dict[str, dict],
    md_conf_lower: float,
    md_conf_upper: float,
    sn_score: float,
    tolerated: set[str],
    stats: LowConfStats,
    csv_rows: list[dict],
    review_dict: dict[str, dict],
    excluded_fps: set[str],
) -> int:
    """Stream one source's speciesnet_results.jsonl; populate stats + outputs.

    Returns the number of records processed (0 if file missing).
    """
    sn_path = SPECIESNET_RESULTS_PATHS[source]
    if not sn_path.exists():
        print(f"[{source}] speciesnet_results.jsonl not found — skipping.")
        return 0

    n_records = 0

    with open(sn_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n_records += 1
            stats.n_total += 1

            filepath = rec.get("filepath", "")

            # Skip images already reviewed in the ≥ 0.5 pipeline
            if filepath in excluded_fps:
                stats.n_excluded += 1
                continue

            detections: list[dict] = rec.get("speciesnet_detections") or []

            # Low-conf tier: [md_conf_lower, md_conf_upper)
            lowconf = [
                d for d in detections
                if not d.get("speciesnet_skipped", False)
                and md_conf_lower <= d.get("megadetector_conf", 0.0) < md_conf_upper
            ]

            if not lowconf:
                continue

            expected_common = rec.get("expected_common", "")

            fail_reason, exp_genus, exp_species, exp_family, exp_tax, exp_level = (
                _resolve_expected_class(
                    expected_common, class225_by_common, tax_by_gs, tax_by_genus
                )
            )
            if fail_reason:
                stats.n_expected_not_in_225 += 1
                continue

            all_box_results: list[dict] = []
            offending: list[dict] = []

            for det in lowconf:
                result = _evaluate_box(
                    det,
                    exp_genus, exp_species, exp_family, exp_tax, exp_level,
                    idx_to_label,
                    tolerated,
                    sn_score,
                )
                if result is None:
                    continue
                all_box_results.append(result)
                if result["verdict"] in ("flag", "uncertain"):
                    offending.append(result)

            if all_box_results or offending:
                stats.record_image(
                    expected_common.lower().replace("_", " ").strip(),
                    source,
                    offending,
                    all_box_results,
                )
            elif lowconf:
                # All boxes were unresolvable (non-mammal or missing labels)
                stats.n_with_lowconf += 1

            if not offending:
                continue

            has_flag = any(b["verdict"] == "flag" for b in offending)

            # CSV: one row per offending box
            for box in offending:
                csv_rows.append({
                    "filepath":          filepath,
                    "source":            source,
                    "split":             "",
                    "expected_class":    expected_common,
                    "detection_idx":     box["detection_idx"],
                    "bbox_norm":         json.dumps(box["bbox_norm"]),
                    "megadetector_conf": box["megadetector_conf"],
                    "pred_common":       box["pred_common"],
                    "pred_scientific":   box["pred_scientific"],
                    "pred_top1_score":   box["pred_top1_score"],
                    "match_level":       box["match_level"],
                    "verdict":           box["verdict"],
                })

            # Review JSON: only images with ≥1 "flag" verdict
            if has_flag:
                review_dict[filepath] = {
                    "expected_class":      expected_common,
                    "source":              source,
                    "n_significant_boxes": len(lowconf),
                    "offending_boxes":     offending,
                    "all_boxes":           all_box_results,
                }

    print(f"  [{source}] {n_records:,} records — "
          f"{stats.n_flagged} flagged images so far "
          f"(+{stats.n_uncertain_only} uncertain-only)")
    return n_records


# ── Report writers ────────────────────────────────────────────────────────────

CSV_FIELDS = [
    "filepath", "source", "split", "expected_class", "detection_idx",
    "bbox_norm", "megadetector_conf", "pred_common", "pred_scientific",
    "pred_top1_score", "match_level", "verdict",
]


def write_csv(csv_rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"  CSV  → {path.relative_to(REPO_ROOT)}  ({len(csv_rows):,} rows)")


def write_review_json(review_dict: dict, path: Path) -> None:
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(review_dict, f, indent=2, ensure_ascii=False)
    tmp.replace(path)
    print(f"  JSON → {path.relative_to(REPO_ROOT)}  ({len(review_dict):,} images)")


def write_markdown_report(
    stats: LowConfStats,
    sources_processed: list[str],
    args: argparse.Namespace,
    path: Path,
) -> None:
    lines: list[str] = []

    lines.append("# Low-Confidence Box Contamination Report\n")
    lines.append(
        f"**Thresholds:** md_conf∈[{args.md_conf_lower}, {args.md_conf_upper})  "
        f"sn_score≥{args.sn_score}  "
        f"tolerance={args.tolerance}\n"
    )
    lines.append(f"**Sources:** {', '.join(sources_processed)}\n")
    lines.append(
        "> Images already present in `multi_animal_contamination_review.json` "
        "(the ≥ 0.5 pipeline) are excluded from this report.\n"
    )
    lines.append("---\n")

    lines.append("## Summary\n")
    lines.append("| Metric | Count |")
    lines.append("|---|---:|")
    lines.append(f"| Total records scanned | {stats.n_total:,} |")
    lines.append(f"| Excluded (already in ≥ 0.5 review) | {stats.n_excluded:,} |")
    lines.append(
        f"| Expected class not in 225 (skipped) | {stats.n_expected_not_in_225:,} |"
    )
    lines.append(
        f"| Images with ≥1 low-conf detection evaluated | {stats.n_with_lowconf:,} |"
    )
    lines.append(
        f"| **Flagged images** (≥1 confident mismatch) | **{stats.n_flagged:,}** |"
    )
    lines.append(
        f"| Uncertain-only images (low-confidence mismatch) | {stats.n_uncertain_only:,} |"
    )
    lines.append(f"| Consistent-only images | {stats.n_consistent_only:,} |")
    lines.append("")

    if stats.n_with_lowconf > 0:
        rate = 100 * stats.n_flagged / stats.n_with_lowconf
        lines.append(
            f"> **Flagging rate:** {stats.n_flagged:,} / {stats.n_with_lowconf:,} images "
            f"= {rate:.1f}%  (reference: ≥ 0.5 tier was ~7.5%)\n"
        )

    if stats.flagged_by_source:
        lines.append("## Breakdown by Source\n")
        lines.append("| Source | Flagged Images |")
        lines.append("|---|---:|")
        for src, cnt in sorted(stats.flagged_by_source.items(), key=lambda x: -x[1]):
            lines.append(f"| {src} | {cnt:,} |")
        lines.append("")

    if stats.offending_match_levels:
        lines.append("## Offending Boxes — Match Level Breakdown\n")
        lines.append("| Match Level | Count |")
        lines.append("|---|---:|")
        for lvl in MATCH_LEVELS:
            cnt = stats.offending_match_levels.get(lvl, 0)
            if cnt:
                lines.append(f"| {lvl} | {cnt:,} |")
        lines.append("")

    if stats.box_verdicts:
        lines.append("## Offending Box Verdict Breakdown\n")
        lines.append("| Verdict | Count |")
        lines.append("|---|---:|")
        for verdict, cnt in sorted(stats.box_verdicts.items(), key=lambda x: -x[1]):
            lines.append(f"| {verdict} | {cnt:,} |")
        lines.append("")

    if stats.flagged_by_class:
        top_n = 30
        ranked = sorted(stats.flagged_by_class.items(), key=lambda x: -x[1])
        lines.append(f"## Top {min(top_n, len(ranked))} Contaminated Classes\n")
        lines.append("| Rank | Class | Flagged Images |")
        lines.append("|---:|---|---:|")
        for rank, (cls_name, cnt) in enumerate(ranked[:top_n], 1):
            lines.append(f"| {rank} | {cls_name} | {cnt:,} |")
        lines.append("")

    report_text = "\n".join(lines)
    tmp = path.with_suffix(".md.tmp")
    tmp.write_text(report_text, encoding="utf-8")
    tmp.replace(path)
    print(f"  MD   → {path.relative_to(REPO_ROOT)}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        choices=list(SPECIESNET_RESULTS_PATHS.keys()) + ["all"],
        default="all",
        help="Dataset source to process (default: all).",
    )
    parser.add_argument(
        "--md-conf-lower",
        type=float,
        default=DEFAULT_MD_CONF_LOWER,
        metavar="CONF",
        help=f"Inclusive lower bound of low-conf tier (default: {DEFAULT_MD_CONF_LOWER}).",
    )
    parser.add_argument(
        "--md-conf-upper",
        type=float,
        default=DEFAULT_MD_CONF_UPPER,
        metavar="CONF",
        help=f"Exclusive upper bound of low-conf tier (default: {DEFAULT_MD_CONF_UPPER}).",
    )
    parser.add_argument(
        "--sn-score",
        type=float,
        default=DEFAULT_SN_SCORE,
        metavar="PROB",
        help=f"SpeciesNet top-1 score floor for confident flag (default: {DEFAULT_SN_SCORE}).",
    )
    parser.add_argument(
        "--tolerance",
        choices=["genus", "family", "order"],
        default="family",
        help="Tolerance band (default: family).",
    )
    parser.add_argument(
        "--high-conf-review-json",
        type=Path,
        default=HIGH_CONF_REVIEW_JSON,
        metavar="PATH",
        help=f"Review JSON from the ≥ 0.5 pipeline; filepaths here are excluded "
             f"(default: {HIGH_CONF_REVIEW_JSON.name}).",
    )
    args = parser.parse_args()

    if args.md_conf_lower >= args.md_conf_upper:
        print(
            f"ERROR: --md-conf-lower ({args.md_conf_lower}) must be < "
            f"--md-conf-upper ({args.md_conf_upper})",
            file=sys.stderr,
        )
        sys.exit(1)

    tolerated = TOLERANCE_BANDS[args.tolerance]

    # Load exclusion set from the ≥ 0.5 review pipeline
    excluded_fps: set[str] = set()
    if args.high_conf_review_json.exists():
        print(f"Loading exclusion set from {args.high_conf_review_json.name} …")
        with open(args.high_conf_review_json, encoding="utf-8") as f:
            hc_data: dict = json.load(f)
        excluded_fps = set(hc_data.keys())
        print(f"  {len(excluded_fps):,} filepaths excluded.")
    else:
        print(
            f"WARNING: {args.high_conf_review_json} not found; "
            "no filepaths will be excluded (all images scanned).",
            file=sys.stderr,
        )

    print("Loading SpeciesNet labels …")
    idx_to_label = load_speciesnet_labels(SPECIESNET_LABELS_PATH)
    print(f"  {len(idx_to_label):,} label entries.")

    print(f"Loading taxonomy from {TAXONOMY_PATH.name} …")
    tax_by_gs, tax_by_genus = _load_taxonomy(TAXONOMY_PATH)
    print(f"  {len(tax_by_gs):,} species-level entries, "
          f"{len(tax_by_genus):,} genus-level entries.")

    print(f"Loading 225 classes from {CLASSES_225_PATH.name} …")
    class225_by_common, _, _, _ = _load_classes_225(CLASSES_225_PATH)
    print(f"  {len(class225_by_common)} classes loaded.")

    print(
        f"\nThresholds: md_conf∈[{args.md_conf_lower}, {args.md_conf_upper})  "
        f"sn_score≥{args.sn_score}  tolerance={args.tolerance}  "
        f"(tolerated match levels: {sorted(tolerated)})"
    )
    print("Mode: STATS + REPORTS ONLY — no data files will be modified.\n")

    sources = (
        list(SPECIESNET_RESULTS_PATHS.keys()) if args.source == "all" else [args.source]
    )

    stats = LowConfStats()
    csv_rows: list[dict] = []
    review_dict: dict[str, dict] = {}
    sources_processed: list[str] = []

    for source in sources:
        sn_path = SPECIESNET_RESULTS_PATHS[source]
        if not sn_path.exists():
            print(f"[{source}] speciesnet_results.jsonl not found — skipping.")
            continue
        print(f"Processing {source} …")
        n = process_source(
            source,
            idx_to_label, tax_by_gs, tax_by_genus, class225_by_common,
            args.md_conf_lower, args.md_conf_upper, args.sn_score, tolerated,
            stats, csv_rows, review_dict,
            excluded_fps,
        )
        if n > 0:
            sources_processed.append(source)

    if not sources_processed:
        print("No sources had speciesnet_results.jsonl — nothing to report.")
        sys.exit(0)

    # ── Console summary ────────────────────────────────────────────────────────
    sep = "─" * 60
    print(f"\n{sep}")
    print("LOW-CONFIDENCE CONTAMINATION FLAGGING SUMMARY")
    print(sep)
    print(f"  Total records scanned:              {stats.n_total:>10,}")
    print(f"  Excluded (already in ≥ 0.5 review): {stats.n_excluded:>10,}")
    print(f"  Expected class not in 225:          {stats.n_expected_not_in_225:>10,}")
    print(f"  Images with low-conf boxes:         {stats.n_with_lowconf:>10,}")
    print(f"  Flagged images (≥1 confident flag): {stats.n_flagged:>10,}  "
          f"({_pct(stats.n_flagged, stats.n_with_lowconf)}  of images with lowconf)")
    print(f"  Uncertain-only images:              {stats.n_uncertain_only:>10,}  "
          f"({_pct(stats.n_uncertain_only, stats.n_with_lowconf)})")
    print(f"  Consistent-only images:             {stats.n_consistent_only:>10,}")

    if stats.flagged_by_source:
        print(f"\n  Flagged by source:")
        for src, cnt in sorted(stats.flagged_by_source.items(), key=lambda x: -x[1]):
            print(f"    {src:<20}  {cnt:>6,}")

    if stats.offending_match_levels:
        print(f"\n  Offending box match levels:")
        for lvl in MATCH_LEVELS:
            cnt = stats.offending_match_levels.get(lvl, 0)
            if cnt:
                print(f"    {lvl:<12}  {cnt:>6,}")

    if stats.flagged_by_class:
        top10 = sorted(stats.flagged_by_class.items(), key=lambda x: -x[1])[:10]
        print(f"\n  Top 10 contaminated classes (flagged images):")
        for rank, (cls_name, cnt) in enumerate(top10, 1):
            print(f"    {rank:>2}. {cls_name:<35}  {cnt:>5,}")

    print(sep)

    # ── Write reports ──────────────────────────────────────────────────────────
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    print("\nWriting reports …")
    write_csv(csv_rows, REPORTS_DIR / "lowconf_contamination.csv")
    write_review_json(review_dict, REPORTS_DIR / "lowconf_contamination_review.json")
    write_markdown_report(
        stats, sources_processed, args,
        REPORTS_DIR / "lowconf_contamination.md",
    )

    print(f"\nAll done. {stats.n_flagged:,} images flagged for review.")
    if stats.n_flagged > 0:
        print(
            f"\nNext step — review flagged images:\n"
            f"  python3 scripts/dataset_quality/14b-review_contamination.py \\\n"
            f"    --review-json reports/lowconf_contamination_review.json \\\n"
            f"    --decisions-jsonl reports/lowconf_review_decisions.jsonl \\\n"
            f"    --decisions-json reports/lowconf_contamination_decisions.json"
        )


if __name__ == "__main__":
    main()
