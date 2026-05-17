"""Trust-aware per-class image distribution report.

Reads filter_results.jsonl (quality stages 1–5) and speciesnet_results.jsonl
for all five sources, then produces a per-class CSV with:

  trusted_quality_pass   — images from iNat/GBIF/Wikimedia passing quality filter
  trusted_sn_pass        — subset also passing SpeciesNet
  trusted_sn_fail_count  — trusted quality-pass images failing SpeciesNet
  trusted_sn_fail_reason — most common SpeciesNet fail reason among those failures
  trusted_no_sn_result   — trusted quality-pass images with no SpeciesNet entry
  unverified_sn_pass     — images from OpenImages/images_cv passing SpeciesNet
  effective_pool         — trusted_quality_pass + unverified_sn_pass
  tier                   — 1/2/3/4 based on effective_pool thresholds (100/500/1500)

This is a read-only analytics script; it does not modify any JSONL files.
The output CSV replaces speciesnet_filter.md as the authoritative input for tier
assignment (docs/plans/2026-05-04_dataset-construction-action-plan.md §3).

Must run inside Dockerfile.speciesnet (Python 3.11, speciesnet package) — the
SpeciesNet classifier is needed to resolve integer class indices to label strings.

Usage:
    python scripts/dataset_quality/8-class_distribution_report.py
    python scripts/dataset_quality/8-class_distribution_report.py \\
        --md-conf 0.4 --sn-score 0.25 --family-fail-thresh 0.6
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

TRUSTED_SOURCES    = {"gbif", "inaturalist", "wikimedia"}
UNVERIFIED_SOURCES = {"openimages", "images_cv"}
ALL_SOURCES        = ["gbif", "inaturalist", "wikimedia", "openimages", "images_cv"]

_CLASSES_225_PATH = REPO_ROOT / "reports" / "classes_225.csv"

# Tier lower bounds: Tier 2 starts at 100, Tier 3 at 500, Tier 4 at 1500.
_TIER_BOUNDARIES = [100, 500, 1500]


def _tier(n: int) -> int:
    for i, boundary in enumerate(_TIER_BOUNDARIES, start=2):
        if n < boundary:
            return i - 1
    return 4


def _strip_apostrophe(name: str) -> str:
    return name.replace("'", "").replace("’", "")


def _build_canonical_lookup(classes_csv: Path) -> dict[str, str]:
    """Return {stripped_name: canonical_name} for class names that contain apostrophes.

    Used to merge ghost folder names (e.g. grevys_zebra) into their canonical
    counterpart (grevy's zebra) during per-source accumulation.
    """
    if not classes_csv.exists():
        return {}
    lookup: dict[str, str] = {}
    with open(classes_csv, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            name = row["common_name"].strip().lower()
            stripped = _strip_apostrophe(name)
            if stripped != name:
                lookup[stripped] = name
    return lookup


# ── Import helpers from script 7 ──────────────────────────────────────────────

def _load_script7():
    """Import 7-filter_speciesnet.py via importlib (numeric name blocks regular import)."""
    path = Path(__file__).parent / "7-filter_speciesnet.py"
    spec = importlib.util.spec_from_file_location("filter_speciesnet", path)
    mod  = importlib.util.module_from_spec(spec)
    # Register before exec_module so @dataclass can resolve cls.__module__ in sys.modules.
    sys.modules["filter_speciesnet"] = mod
    spec.loader.exec_module(mod)
    return mod


# ── Quality-stage pass check ──────────────────────────────────────────────────

def passed_quality_stages(entry: dict) -> bool:
    """True if the image cleared stages 1–5 regardless of SpeciesNet outcome.

    Resilient to both pre-write state (stage_failed never equals 'speciesnet')
    and post-write state (stage_failed == 'speciesnet' for sn-only failures).
    """
    sf = entry.get("stage_failed")
    return (sf is None and entry.get("passed", False)) or sf == "speciesnet"


# ── Build SpeciesNet eval index ───────────────────────────────────────────────

def build_sn_index(
    sn_path: Path,
    s7,
    idx_to_label,
    tax_by_gs,
    tax_by_genus,
    class225_by_common,
    genus_species_to_225,
    genus_to_225,
    family_to_225,
    md_conf: float,
    sn_score: float,
    family_fail_thresh: float,
) -> dict[str, tuple[bool, str | None]]:
    """Stream speciesnet_results.jsonl → {filepath: (pass, reason)}."""
    index: dict[str, tuple[bool, str | None]] = {}
    with open(sn_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ev  = s7.evaluate_record(
                rec,
                idx_to_label, tax_by_gs, tax_by_genus,
                class225_by_common, genus_species_to_225, genus_to_225, family_to_225,
                md_conf, sn_score, family_fail_thresh,
            )
            index[rec["filepath"]] = (ev["pass"], ev["reason"])
    return index


# ── Per-source accumulation ───────────────────────────────────────────────────

def process_source(
    source: str,
    sn_index: dict[str, tuple[bool, str | None]],
    per_class: dict,
    canonical_lookup: dict[str, str],
) -> None:
    """Stream filter_results.jsonl and accumulate per-class counts."""
    filter_path = REPO_ROOT / "data" / source / "filter_results.jsonl"
    if not filter_path.exists():
        print(f"  [{source}] filter_results.jsonl not found — skipping.")
        return

    trusted = source in TRUSTED_SOURCES

    with open(filter_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            cls = Path(entry.get("filepath", "")).parent.name.lower().replace("_", " ")
            cls = canonical_lookup.get(cls, cls)
            if not cls:
                continue

            d  = per_class[cls]
            fp = entry.get("filepath", "")

            if trusted:
                if not passed_quality_stages(entry):
                    continue
                d["trusted_quality_pass"] += 1
                result = sn_index.get(fp)
                if result is None:
                    d["trusted_no_sn_result"] += 1
                elif result[0]:
                    d["trusted_sn_pass"] += 1
                else:
                    d["trusted_sn_fail_count"] += 1
                    if result[1]:
                        d["trusted_sn_fail_reasons"][result[1]] += 1

            else:  # unverified source
                if not passed_quality_stages(entry):
                    continue
                result = sn_index.get(fp)
                if result is not None and result[0]:
                    d["unverified_sn_pass"] += 1


# ── Output helpers ────────────────────────────────────────────────────────────

def _top_reason(counter: Counter) -> str:
    if not counter:
        return ""
    return counter.most_common(1)[0][0]


def _build_rows(per_class: dict) -> list[dict]:
    rows = []
    for cls, d in sorted(per_class.items()):
        tqp  = d["trusted_quality_pass"]
        tsp  = d["trusted_sn_pass"]
        tfc  = d["trusted_sn_fail_count"]
        tnr  = d["trusted_no_sn_result"]
        usp  = d["unverified_sn_pass"]
        pool = tqp + usp
        rows.append({
            "class":                  cls,
            "trusted_quality_pass":   tqp,
            "trusted_sn_pass":        tsp,
            "trusted_sn_fail_count":  tfc,
            "trusted_sn_fail_reason": _top_reason(d["trusted_sn_fail_reasons"]),
            "trusted_no_sn_result":   tnr,
            "unverified_sn_pass":     usp,
            "effective_pool":         pool,
            "tier":                   _tier(pool),
        })
    rows.sort(key=lambda r: r["effective_pool"], reverse=True)
    return rows


def write_csv(rows: list[dict], out_path: Path) -> None:
    if not rows:
        print("No data to write.")
        return
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV  → {out_path.relative_to(REPO_ROOT)}")


def write_md(rows: list[dict], args: argparse.Namespace, out_path: Path) -> None:
    tier_summary: dict[int, dict] = {t: {"n": 0, "pool": 0} for t in (1, 2, 3, 4)}
    for r in rows:
        t = r["tier"]
        tier_summary[t]["n"]    += 1
        tier_summary[t]["pool"] += r["effective_pool"]

    lines: list[str] = [
        "# Trust-Aware Class Distribution Report\n",
        f"**Date:** {date.today()}  ",
        f"**Thresholds:** md_conf≥{args.md_conf}  "
        f"sn_score≥{args.sn_score}  "
        f"family_fail≥{args.family_fail_thresh}  ",
        "**Mode:** STATS ONLY — no JSONL files modified\n",
        "---\n",
        "## Tier Summary\n",
        "| Tier | Effective pool | Classes | Total effective pool |",
        "|---:|---|---:|---:|",
    ]
    ranges = ["< 100", "100–499", "500–1 499", "≥ 1 500"]
    for t in (1, 2, 3, 4):
        ts = tier_summary[t]
        lines.append(f"| {t} | {ranges[t-1]} | {ts['n']:,} | {ts['pool']:,} |")

    lines += [
        "",
        "## Per-Class Table\n",
        "Columns: `tq_pass` = trusted quality-pass · `tsn_pass` = trusted SN-pass · "
        "`tsn_fail` = trusted quality-pass but SN-fail · `t_no_sn` = no SN result · "
        "`uv_pass` = unverified SN-pass\n",
        "| Class | tq_pass | tsn_pass | tsn_fail | tsn_fail_reason | t_no_sn | uv_pass | eff_pool | tier |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['class']} "
            f"| {r['trusted_quality_pass']:,} "
            f"| {r['trusted_sn_pass']:,} "
            f"| {r['trusted_sn_fail_count']:,} "
            f"| {r['trusted_sn_fail_reason']} "
            f"| {r['trusted_no_sn_result']:,} "
            f"| {r['unverified_sn_pass']:,} "
            f"| {r['effective_pool']:,} "
            f"| {r['tier']} |"
        )
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"MD   → {out_path.relative_to(REPO_ROOT)}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--md-conf", type=float, default=0.5, metavar="CONF",
                        help="MegaDetector confidence floor (default: 0.5).")
    parser.add_argument("--sn-score", type=float, default=0.3, metavar="PROB",
                        help="SpeciesNet top-1 score floor (default: 0.3).")
    parser.add_argument("--family-fail-thresh", type=float, default=0.5, metavar="PROB",
                        help="Family-match + score ≥ this → fail (default: 0.5).")
    parser.add_argument("--output-csv", type=Path,
                        default=REPO_ROOT / "reports" / "class_distribution.csv",
                        metavar="PATH")
    parser.add_argument("--output-md", type=Path,
                        default=REPO_ROOT / "reports" / "class_distribution.md",
                        metavar="PATH")
    args = parser.parse_args()

    try:
        import speciesnet  # noqa: F401
    except ImportError:
        print(
            "ERROR: 'speciesnet' is not installed.\n"
            "This script must run inside Dockerfile.speciesnet:\n"
            "  make speciesnet-build && make speciesnet-start",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Loading script 7 helpers …")
    s7 = _load_script7()

    idx_to_label = s7.load_speciesnet_labels()

    print(f"Loading taxonomy from {s7.TAXONOMY_PATH.name} …")
    tax_by_gs, tax_by_genus = s7.load_taxonomy(s7.TAXONOMY_PATH)
    print(f"  {len(tax_by_gs):,} species entries, {len(tax_by_genus):,} genus entries.")

    print(f"Loading 225 classes from {s7.CLASSES_225_PATH.name} …")
    class225_by_common, gs225, g225, f225 = s7.load_classes_225(s7.CLASSES_225_PATH)
    print(f"  {len(class225_by_common)} classes loaded.")

    def _new_class_dict() -> dict:
        return {
            "trusted_quality_pass":    0,
            "trusted_sn_pass":         0,
            "trusted_sn_fail_count":   0,
            "trusted_sn_fail_reasons": Counter(),
            "trusted_no_sn_result":    0,
            "unverified_sn_pass":      0,
        }

    per_class: dict[str, dict] = defaultdict(_new_class_dict)

    canonical_lookup = _build_canonical_lookup(_CLASSES_225_PATH)
    if canonical_lookup:
        print(f"Loaded {len(canonical_lookup)} ghost-name → canonical mappings from classes_225.csv.")
    else:
        print("classes_225.csv not found — ghost-name normalization disabled.")

    for source in ALL_SOURCES:
        sn_path = REPO_ROOT / "data" / source / "speciesnet_results.jsonl"
        if not sn_path.exists():
            print(f"\n[{source}] speciesnet_results.jsonl not found — skipping.")
            continue

        print(f"\n[{source}] Indexing speciesnet_results.jsonl …")
        sn_index = build_sn_index(
            sn_path, s7,
            idx_to_label, tax_by_gs, tax_by_genus,
            class225_by_common, gs225, g225, f225,
            args.md_conf, args.sn_score, args.family_fail_thresh,
        )
        print(f"  {len(sn_index):,} records indexed.")

        print(f"[{source}] Counting quality-pass images from filter_results.jsonl …")
        process_source(source, sn_index, per_class, canonical_lookup)

    print(f"\nAggregating {len(per_class)} classes …")
    rows = _build_rows(per_class)

    write_csv(rows, args.output_csv)
    write_md(rows, args, args.output_md)
    print("\nDone.")


if __name__ == "__main__":
    main()
