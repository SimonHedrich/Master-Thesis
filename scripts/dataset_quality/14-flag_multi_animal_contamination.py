"""Flag cross-species contamination in multi-box images.

For images with ≥2 significant MegaDetector detections (conf ≥ MD_CONF), each
secondary box (detection_idx != 0) is evaluated against the image's expected
class using SpeciesNet's stored per-box classifications.

If a secondary box is classified as a *genuinely different* animal — outside the
configured tolerance band — and the SpeciesNet score is confident enough, the
image is **flagged** for review.  A secondary box with a differing prediction but
low SpeciesNet confidence goes to a separate "uncertain" list rather than
auto-flagging, keeping the review queue tight.

Tolerance levels (--tolerance, default 'family'):
  genus   → tolerate species + genus differences (same genus is fine)
  family  → tolerate species + genus + family (default; covers African vs Asian
             elephant, different zebra species, etc.)
  order   → tolerate up through order level

The script is NON-DESTRUCTIVE: it only writes report files, no data is modified.

Usage:
    # Flag contamination in one source (default thresholds):
    uv run python scripts/dataset_quality/14-flag_multi_animal_contamination.py --source openimages

    # Run all sources:
    uv run python scripts/dataset_quality/14-flag_multi_animal_contamination.py --source all

    # Tighter tolerance (flag family-level differences too):
    uv run python scripts/dataset_quality/14-flag_multi_animal_contamination.py \\
        --source all --tolerance genus

    # Adjust confidence thresholds:
    uv run python scripts/dataset_quality/14-flag_multi_animal_contamination.py \\
        --source all --md-conf 0.6 --sn-score 0.35

Outputs (written to reports/):
    multi_animal_contamination.csv          — one row per flagged/uncertain box
    multi_animal_contamination_review.json  — per-image dict for downstream tools
    multi_animal_contamination.md           — human summary with projected class deltas

See also:
    docs/plans/2026-06-09_flag-cross-species-contamination-multi-box.md
        — design spec: problem statement, tolerance-band rationale, review workflow
    docs/progress_notes/2026-06-09_contamination-flagging-and-augmentation-implementation.md
        — implementation log: execution steps, bug fixes, final run results
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
# Script 7 has a digit+hyphen prefix so we cannot `import` it directly.

_s7_path = Path(__file__).resolve().parent / "7-filter_speciesnet.py"
_spec = importlib.util.spec_from_file_location("filter_speciesnet_s7", _s7_path)
s7 = importlib.util.module_from_spec(_spec)
# Register in sys.modules BEFORE exec so that @dataclass can resolve cls.__module__
# (Python 3.13+ requires sys.modules[cls.__module__] to exist when the class body runs).
sys.modules["filter_speciesnet_s7"] = s7
_spec.loader.exec_module(s7)  # safe: _check_environment() is only called in main()

# Aliases for readability
_compute_match_level = s7._compute_match_level
_load_taxonomy = s7.load_taxonomy
_load_classes_225 = s7.load_classes_225
TAXONOMY_PATH = s7.TAXONOMY_PATH
CLASSES_225_PATH = s7.CLASSES_225_PATH
SPECIESNET_RESULTS_PATHS = s7.SPECIESNET_RESULTS_PATHS
MATCH_LEVELS = s7.MATCH_LEVELS  # ["species","genus","family","order","class","no_match"]

# ── Constants ─────────────────────────────────────────────────────────────────

SPECIESNET_LABELS_PATH = REPO_ROOT / "data" / "speciesnet_labels.json"
REPORTS_DIR = REPO_ROOT / "reports"

DEFAULT_MD_CONF  = 0.5   # MegaDetector confidence floor (matches GT export threshold)
DEFAULT_SN_SCORE = 0.3   # SpeciesNet top-1 score floor for "confident" flag

# Tolerance-band mapping: flag if match_level is NOT in the tolerated set.
# "family" default: tolerate species/genus/family differences (same family = OK).
TOLERANCE_BANDS: dict[str, set[str]] = {
    "genus":  {"species", "genus"},
    "family": {"species", "genus", "family"},
    "order":  {"species", "genus", "family", "order"},
}

# Optional cross-family allow-list.  A secondary box whose expected AND predicted
# family/genus both fall in the same group is treated as consistent regardless of
# match_level.  Start empty — populate only if review surfaces recurring
# false-positive pairs (the elephant case is already handled by family tolerance).
LOOKALIKE_GROUPS: list[set[str]] = []


# ── Load SpeciesNet labels from pre-dumped JSON (no Docker needed) ────────────

def load_speciesnet_labels(path: Path) -> dict[int, str]:
    """Load {int_idx: 'uuid;class;order;family;genus;species;common'} from JSON.

    The file data/speciesnet_labels.json was produced by
    scripts/dataset_quality/0-dump_speciesnet_labels.py and contains the full
    label strings for all 2498 SpeciesNet classifier classes.
    """
    with open(path, encoding="utf-8") as f:
        raw: dict[str, str] = json.load(f)
    return {int(k): v for k, v in raw.items()}


# ── Taxonomy helpers ──────────────────────────────────────────────────────────

def _resolve_expected_class(
    expected_common: str,
    class225_by_common: dict[str, dict],
    tax_by_gs: dict[str, dict],
    tax_by_genus: dict[str, dict],
) -> tuple[str | None, str, str, str, dict, str]:
    """Resolve expected_common name to taxonomy components.

    Returns:
        (fail_reason, exp_genus, exp_species, exp_family, exp_tax, exp_level)
        fail_reason is None on success, or a string reason if unresolvable.
    """
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
    """Parse a 'uuid;class;order;family;genus;species;common' label string.

    Returns None if the label is malformed or represents a non-taxonomic entry
    (blank, animal, vehicle, etc.) that has no genus/species.
    """
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
    """Return True if expected and predicted taxa both belong to the same
    LOOKALIKE_GROUPS entry, in which case the box is treated as consistent.
    """
    if not LOOKALIKE_GROUPS:
        return False
    pred_identifiers = {
        pred_tax.get("family", ""),
        pred_tax.get("genus", ""),
    } - {""}
    exp_identifiers = {
        exp_tax.get("family", ""),
        exp_tax.get("genus", ""),
    } - {""}
    for group in LOOKALIKE_GROUPS:
        if pred_identifiers & group and exp_identifiers & group:
            return True
    return False


# ── Box evaluation ────────────────────────────────────────────────────────────

def _evaluate_secondary_box(
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
    """Evaluate one secondary detection against the image's expected class.

    Returns a result dict or None if the prediction cannot be resolved (e.g.
    SpeciesNet predicted "blank" or a non-mammalian entry with no taxonomy).

    Result fields:
        detection_idx, bbox_norm, megadetector_conf,
        pred_common, pred_scientific, pred_top1_score,
        match_level, verdict   ('consistent' | 'flag' | 'uncertain' | 'unresolvable')
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

    # Skip any prediction that is not a mammal.  Non-mammalian predictions
    # (blank/vehicle with class_='', birds with class_='aves', reptiles, insects,
    # etc.) cannot constitute real cross-species contamination in a mammal dataset
    # — they are almost certainly SpeciesNet misclassifications or failed crops.
    # Only mammalian predictions (class_='mammalia') are meaningful: they can
    # represent a genuinely different mammal species in the frame.
    if pred_tax["class_"] != "mammalia":
        return None

    match_level = _compute_match_level(
        pred_tax, exp_genus, exp_species, exp_family, exp_tax, exp_level
    )

    # Check LOOKALIKE_GROUPS before applying band
    if _is_lookalike(pred_tax, exp_tax):
        verdict = "consistent"
    elif match_level in tolerated:
        verdict = "consistent"
    else:
        # Outside tolerance band: flag only if confident
        top1_score = det.get("speciesnet_top1_score", 0.0)
        if top1_score >= sn_score_thresh:
            verdict = "flag"
        else:
            verdict = "uncertain"

    parts = label.split(";")
    pred_genus   = pred_tax["genus"]
    pred_species = pred_tax["species"]
    pred_common  = pred_tax["common"]
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
class ContaminationStats:
    """Accumulate counts while streaming speciesnet_results.jsonl."""

    n_total:               int = 0  # total images processed
    n_multi:               int = 0  # images with >=2 significant detections
    n_expected_not_in_225: int = 0  # images skipped: expected class not in 225
    n_flagged:             int = 0  # images with >=1 'flag' verdict
    n_uncertain_only:      int = 0  # images with 'uncertain' boxes but no 'flag'
    n_consistent_only:     int = 0  # multi-box images fully consistent

    # Box-level counts across secondary boxes only
    box_verdicts:   dict[str, int] = field(default_factory=lambda: defaultdict(int))
    match_levels:   dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Per-expected-class flagged image count  (for top-N report + delta table)
    flagged_by_class: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Per-source flagged image count
    flagged_by_source: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Match-level breakdown of offending (flag+uncertain) boxes
    offending_match_levels: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )

    def record_image(
        self,
        expected_class: str,
        source: str,
        offending: list[dict],
        all_secondary: list[dict],
    ) -> None:
        """Update counts for one multi-box image."""
        self.n_multi += 1
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

        for box in all_secondary:
            self.match_levels[box["match_level"]] += 1


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
    md_conf: float,
    sn_score: float,
    tolerated: set[str],
    stats: ContaminationStats,
    csv_rows: list[dict],
    review_dict: dict[str, dict],
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

            detections: list[dict] = rec.get("speciesnet_detections") or []

            # Significant = not skipped AND megadetector_conf >= threshold
            significant = [
                d for d in detections
                if not d.get("speciesnet_skipped", False)
                and d.get("megadetector_conf", 0.0) >= md_conf
            ]

            # Only multi-box images are interesting
            if len(significant) < 2:
                continue

            expected_common = rec.get("expected_common", "")
            filepath = rec.get("filepath", "")

            # Resolve expected class taxonomy
            fail_reason, exp_genus, exp_species, exp_family, exp_tax, exp_level = (
                _resolve_expected_class(
                    expected_common, class225_by_common, tax_by_gs, tax_by_genus
                )
            )
            if fail_reason:
                stats.n_expected_not_in_225 += 1
                continue  # can't evaluate without taxonomy reference

            # Primary box (detection_idx 0) is assumed correct — only check secondaries
            secondary = [d for d in significant if d.get("detection_idx", 0) != 0]

            all_box_results: list[dict] = []
            offending: list[dict] = []

            for det in secondary:
                result = _evaluate_secondary_box(
                    det,
                    exp_genus, exp_species, exp_family, exp_tax, exp_level,
                    idx_to_label,
                    tolerated,
                    sn_score,
                )
                if result is None:
                    # Unresolvable prediction (blank/non-taxonomic) — skip
                    continue
                all_box_results.append(result)
                if result["verdict"] in ("flag", "uncertain"):
                    offending.append(result)

            # Also include a "consistent" entry for the primary box in all_boxes
            primary_det = next(
                (d for d in significant if d.get("detection_idx", 0) == 0), None
            )
            primary_result: dict | None = None
            if primary_det is not None:
                primary_result = _evaluate_secondary_box(
                    primary_det,
                    exp_genus, exp_species, exp_family, exp_tax, exp_level,
                    idx_to_label,
                    tolerated,
                    sn_score,
                )
                if primary_result is not None:
                    # Primary is always treated as "consistent" (it set the label)
                    primary_result = dict(primary_result)
                    primary_result["verdict"] = "consistent"

            # Record statistics
            if all_box_results or offending:
                stats.record_image(
                    expected_common.lower().replace("_", " ").strip(),
                    source,
                    offending,
                    all_box_results,
                )
            else:
                # All secondary boxes unresolvable — still count as multi
                stats.n_multi += 1

            # Only emit output rows for images with >=1 offending box
            if not offending:
                continue

            has_flag = any(b["verdict"] == "flag" for b in offending)

            # CSV: one row per offending box (flag + uncertain)
            for box in offending:
                csv_rows.append({
                    "filepath":          filepath,
                    "source":            source,
                    "split":             "",  # not available in jsonl
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

            # Review JSON: only images with >=1 "flag" verdict
            if has_flag:
                all_boxes_for_review = []
                if primary_result is not None:
                    all_boxes_for_review.append(primary_result)
                all_boxes_for_review.extend(all_box_results)

                review_dict[filepath] = {
                    "expected_class":       expected_common,
                    "source":               source,
                    "n_significant_boxes":  len(significant),
                    "offending_boxes":      offending,
                    "all_boxes":            all_boxes_for_review,
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
    """Write flagged/uncertain boxes to CSV."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"  CSV  → {path.relative_to(REPO_ROOT)}  ({len(csv_rows):,} rows)")


def write_review_json(review_dict: dict, path: Path) -> None:
    """Write per-image review dict to JSON (pretty-printed, ~readable)."""
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(review_dict, f, indent=2, ensure_ascii=False)
    tmp.replace(path)
    print(f"  JSON → {path.relative_to(REPO_ROOT)}  ({len(review_dict):,} images)")


def write_markdown_report(
    stats: ContaminationStats,
    sources_processed: list[str],
    args: argparse.Namespace,
    path: Path,
) -> None:
    """Write human-readable summary to markdown."""
    lines: list[str] = []

    lines.append("# Multi-Animal Contamination Report\n")
    lines.append(
        f"**Thresholds:** md_conf≥{args.md_conf}  "
        f"sn_score≥{args.sn_score}  "
        f"tolerance={args.tolerance}\n"
    )
    lines.append(f"**Sources:** {', '.join(sources_processed)}\n")
    lines.append("---\n")

    # ── Headline counts ───────────────────────────────────────────────────────
    lines.append("## Summary\n")
    lines.append("| Metric | Count |")
    lines.append("|---|---:|")
    lines.append(f"| Total images classified | {stats.n_total:,} |")
    lines.append(f"| Images with ≥2 significant detections | {stats.n_multi:,} |")
    lines.append(
        f"| Expected class not in 225 (skipped) | {stats.n_expected_not_in_225:,} |"
    )
    lines.append(f"| **Flagged images** (≥1 confident mismatch) | **{stats.n_flagged:,}** |")
    lines.append(
        f"| Uncertain-only images (low-confidence mismatch) | {stats.n_uncertain_only:,} |"
    )
    lines.append(f"| Consistent multi-box images | {stats.n_consistent_only:,} |")
    lines.append("")

    naive_upper_bound = 32_401  # from plan §3
    lines.append(
        f"> **Note:** {stats.n_flagged:,} flagged images — "
        f"this should be FAR below the naive upper bound of {naive_upper_bound:,} "
        f"(images with any different SpeciesNet index).  "
    )
    if stats.n_flagged < naive_upper_bound * 0.5:
        ratio = naive_upper_bound / stats.n_flagged if stats.n_flagged else float("inf")
        lines.append(
            f"Ratio: 1:{ratio:.0f} — tolerance band is working as intended.\n"
        )
    else:
        lines.append(
            f"WARNING: flagged count is unexpectedly high — review tolerance setting.\n"
        )
    lines.append("")

    # ── By source ─────────────────────────────────────────────────────────────
    if stats.flagged_by_source:
        lines.append("## Breakdown by Source\n")
        lines.append("| Source | Flagged Images |")
        lines.append("|---|---:|")
        for src, cnt in sorted(stats.flagged_by_source.items(), key=lambda x: -x[1]):
            lines.append(f"| {src} | {cnt:,} |")
        lines.append("")

    # ── By match level of offending boxes ─────────────────────────────────────
    if stats.offending_match_levels:
        lines.append("## Offending Boxes — Match Level Breakdown\n")
        lines.append("| Match Level | Count |")
        lines.append("|---|---:|")
        for lvl in MATCH_LEVELS:
            cnt = stats.offending_match_levels.get(lvl, 0)
            if cnt:
                lines.append(f"| {lvl} | {cnt:,} |")
        lines.append("")

    # ── Verdict breakdown ─────────────────────────────────────────────────────
    if stats.box_verdicts:
        lines.append("## Offending Box Verdict Breakdown\n")
        lines.append("| Verdict | Count |")
        lines.append("|---|---:|")
        for verdict, cnt in sorted(stats.box_verdicts.items(), key=lambda x: -x[1]):
            lines.append(f"| {verdict} | {cnt:,} |")
        lines.append("")

    # ── Top contaminated classes ───────────────────────────────────────────────
    if stats.flagged_by_class:
        top_n = 30
        ranked = sorted(stats.flagged_by_class.items(), key=lambda x: -x[1])
        lines.append(f"## Top {min(top_n, len(ranked))} Contaminated Classes (by Flagged Images)\n")
        lines.append("| Rank | Class | Flagged Images |")
        lines.append("|---:|---|---:|")
        for rank, (cls_name, cnt) in enumerate(ranked[:top_n], 1):
            lines.append(f"| {rank} | {cls_name} | {cnt:,} |")
        lines.append("")

    # ── Projected per-class image-count delta ─────────────────────────────────
    if stats.flagged_by_class:
        lines.append("## Projected Per-Class Image-Count Delta\n")
        lines.append(
            "If every flagged image were removed, each class would lose at most "
            "this many images.  Actual loss may be lower (reviewer may choose to "
            "edit individual boxes rather than discard whole images).\n"
        )
        lines.append("| Class | Flagged Images (max loss) |")
        lines.append("|---|---:|")
        for cls_name, cnt in sorted(stats.flagged_by_class.items(), key=lambda x: -x[1]):
            lines.append(f"| {cls_name} | {cnt:,} |")
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
        help="Dataset source to process, or 'all' for every source in sequence "
             "(default: all).",
    )
    parser.add_argument(
        "--md-conf",
        type=float,
        default=DEFAULT_MD_CONF,
        metavar="CONF",
        help=f"MegaDetector confidence floor for significant detections "
             f"(default: {DEFAULT_MD_CONF}; matches GT export threshold).",
    )
    parser.add_argument(
        "--sn-score",
        type=float,
        default=DEFAULT_SN_SCORE,
        metavar="PROB",
        help=f"SpeciesNet top-1 score floor for confident mismatch flag "
             f"(default: {DEFAULT_SN_SCORE}).  Secondary boxes below this "
             "threshold go to the 'uncertain' list.",
    )
    parser.add_argument(
        "--tolerance",
        choices=["genus", "family", "order"],
        default="family",
        help="Tolerance band: how similar a secondary box must be to NOT flag. "
             "'family' (default): same genus or family is fine. "
             "'genus': only same-genus differences tolerated. "
             "'order': tolerate up through order level.",
    )
    args = parser.parse_args()

    tolerated = TOLERANCE_BANDS[args.tolerance]

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
        f"\nThresholds: md_conf≥{args.md_conf}  sn_score≥{args.sn_score}  "
        f"tolerance={args.tolerance}  "
        f"(tolerated match levels: {sorted(tolerated)})"
    )
    print("Mode: STATS + REPORTS ONLY — no data files will be modified.\n")

    sources = (
        list(SPECIESNET_RESULTS_PATHS.keys()) if args.source == "all" else [args.source]
    )

    stats = ContaminationStats()
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
            args.md_conf, args.sn_score, tolerated,
            stats, csv_rows, review_dict,
        )
        if n > 0:
            sources_processed.append(source)

    if not sources_processed:
        print("No sources had speciesnet_results.jsonl — nothing to report.")
        sys.exit(0)

    # ── Print console summary ──────────────────────────────────────────────────
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"CONTAMINATION FLAGGING SUMMARY")
    print(sep)
    print(f"  Total images classified:              {stats.n_total:>10,}")
    print(f"  Images with ≥2 significant boxes:     {stats.n_multi:>10,}  "
          f"({_pct(stats.n_multi, stats.n_total)})")
    print(f"  Expected class not in 225 (skipped):  "
          f"{stats.n_expected_not_in_225:>10,}")
    print(f"  Flagged images (≥1 confident flag):   {stats.n_flagged:>10,}  "
          f"({_pct(stats.n_flagged, stats.n_multi)}  of multi-box)")
    print(f"  Uncertain-only images:                {stats.n_uncertain_only:>10,}  "
          f"({_pct(stats.n_uncertain_only, stats.n_multi)}  of multi-box)")
    print(f"  Consistent multi-box images:          {stats.n_consistent_only:>10,}  "
          f"({_pct(stats.n_consistent_only, stats.n_multi)}  of multi-box)")

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

    naive_ub = 32_401
    print(f"\n  Naive upper bound (any different SpeciesNet idx): {naive_ub:,}")
    if stats.n_flagged > 0:
        print(f"  Actual flagged / naive upper bound: "
              f"{stats.n_flagged:,} / {naive_ub:,} "
              f"= {100 * stats.n_flagged / naive_ub:.1f}%")

    if stats.flagged_by_class:
        top10 = sorted(stats.flagged_by_class.items(), key=lambda x: -x[1])[:10]
        print(f"\n  Top 10 contaminated classes (flagged images):")
        for rank, (cls_name, cnt) in enumerate(top10, 1):
            print(f"    {rank:>2}. {cls_name:<35}  {cnt:>5,}")

    print(sep)

    # ── Write reports ──────────────────────────────────────────────────────────
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    print("\nWriting reports …")
    write_csv(csv_rows, REPORTS_DIR / "multi_animal_contamination.csv")
    write_review_json(review_dict, REPORTS_DIR / "multi_animal_contamination_review.json")
    write_markdown_report(
        stats, sources_processed, args,
        REPORTS_DIR / "multi_animal_contamination.md",
    )

    print(f"\nAll done. {stats.n_flagged:,} images flagged for review.")


if __name__ == "__main__":
    main()
