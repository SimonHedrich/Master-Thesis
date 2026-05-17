"""Post-review class distribution report.

Reads reports/class_distribution.csv (output of 8-class_distribution_report.py)
and reports/review_decisions.jsonl, then produces a revised per-class table
that reflects the outcome of manual image review.

New columns added to the existing distribution:

  review_approved   — images approved during manual review
  review_declined   — images declined during manual review
  effective_trusted — valid trusted-source images after applying tier-specific rules:
                        Tier 1/2  → review_approved only (all were reviewed; SN skipped)
                        Tier 3    → trusted_sn_pass + review_approved; except when SN is an
                                    unreliable signal (tsp == 0, OR dominant fail reason is
                                    match_level_no_match / match_level_class with <20% pass rate,
                                    OR match_level_order with <15% pass rate) → use
                                    trusted_quality_pass directly, same as Tier 4.
                                    family_mismatch_high_confidence and low_speciesnet_confidence
                                    classes remain conservative (tsp + review_approved).
                        Tier 4    → trusted_quality_pass (all assumed valid; no review applied)
  effective_pool    — effective_trusted + unverified_sn_pass  (replaces old effective_pool)
  final_tier        — tier based on effective_pool (100 / 500 / 1500)

Undo semantics: decisions are processed in timestamp order; an 'undo' entry
cancels the most recent decision for the same filepath (stack-based).

This is a read-only analytics script; it does not modify any JSONL files.

Usage:
    python scripts/dataset_quality/9-class_distribution_with_reviews.py
    python scripts/dataset_quality/9-class_distribution_with_reviews.py \\
        --input-csv reports/class_distribution.csv \\
        --decisions reports/review_decisions.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

_TIER_BOUNDARIES = [100, 500, 1500]
_CLASSES_225_PATH = REPO_ROOT / "reports" / "classes_225.csv"

# SpeciesNet coverage-gap detection: if SN's dominant fail reason indicates the species
# is simply out-of-distribution (no taxonomy match at all), treat SN-fail as noise and
# fall back to trusted_quality_pass rather than the conservative tsp+app formula.
_COVERAGE_GAP_FAIL_REASONS_STRICT = {"match_level_no_match", "match_level_class"}
_COVERAGE_GAP_THRESHOLD_STRICT = 0.20   # <20% pass rate → SN has no useful signal
_COVERAGE_GAP_THRESHOLD_ORDER  = 0.15   # tighter for match_level_order (SN sees order)


def _tier(n: int) -> int:
    for i, boundary in enumerate(_TIER_BOUNDARIES, start=2):
        if n < boundary:
            return i - 1
    return 4


def _strip_apostrophe(name: str) -> str:
    return name.replace("'", "").replace("'", "")


def _build_canonical_lookup(classes_csv: Path) -> dict[str, str]:
    """Return {stripped_name: canonical_name} for names that contain apostrophes."""
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


def _class_from_filepath(fp: str, canonical_lookup: dict[str, str] | None = None) -> str:
    name = Path(fp).parent.name.lower().replace("_", " ")
    if canonical_lookup:
        name = canonical_lookup.get(name, name)
    return name


# ── SpeciesNet coverage-gap helper ───────────────────────────────────────────

def _is_sn_coverage_gap(tsp: int, tqp: int, fail_reason: str) -> bool:
    """True when SpeciesNet's fail signal is noise, not evidence of a bad image.

    This happens when SN was never trained on the species (coverage gap) and
    its predictions have no taxonomic relationship to the expected class.
    """
    if tsp == 0:
        return True
    if tqp == 0:
        return False
    rate = tsp / tqp
    if fail_reason in _COVERAGE_GAP_FAIL_REASONS_STRICT:
        return rate < _COVERAGE_GAP_THRESHOLD_STRICT
    if fail_reason == "match_level_order":
        return rate < _COVERAGE_GAP_THRESHOLD_ORDER
    return False


# ── Review decisions ──────────────────────────────────────────────────────────

def load_decisions(decisions_path: Path) -> tuple[Counter, Counter]:
    """Return (approved_per_class, declined_per_class) after resolving undos.

    Each filepath maintains a stack of decisions. 'undo' pops the stack.
    The final effective decision is the top of the stack.
    """
    canonical_lookup = _build_canonical_lookup(_CLASSES_225_PATH)

    stacks: dict[str, list[str]] = defaultdict(list)

    with open(decisions_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            fp  = rec.get("filepath", "")
            dec = rec.get("decision", "")
            if dec in ("approve", "decline"):
                stacks[fp].append(dec)
            elif dec == "undo" and stacks[fp]:
                stacks[fp].pop()

    approved: Counter = Counter()
    declined: Counter = Counter()
    for fp, stack in stacks.items():
        if not stack:
            continue
        cls = _class_from_filepath(fp, canonical_lookup)
        if stack[-1] == "approve":
            approved[cls] += 1
        else:
            declined[cls] += 1

    return approved, declined


# ── Base CSV ──────────────────────────────────────────────────────────────────

def load_base_csv(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ── Merge ─────────────────────────────────────────────────────────────────────

def build_rows(
    base_rows: list[dict],
    approved: Counter,
    declined: Counter,
) -> list[dict]:
    rows = []
    for r in base_rows:
        cls  = r["class"]
        tier = int(r["tier"])
        tqp  = int(r["trusted_quality_pass"])
        tsp  = int(r["trusted_sn_pass"])
        usp  = int(r["unverified_sn_pass"])
        app  = approved[cls]
        dec  = declined[cls]

        # Tier-aware effective trusted count per dataset construction strategy:
        #   Tier 1/2: SN skipped; all quality-pass images were manually reviewed.
        #             Only approved images are valid (review is the sole gate).
        #   Tier 3:   SN-pass images are valid without review.
        #             SN-fail images were queued for review; approved ones are added.
        #             Coverage-gap exception: if _is_sn_coverage_gap() is True (SN was
        #             never trained on this species), SN-fail is noise. Fall back to
        #             trusted_quality_pass directly, same as Tier 4.
        #             family_mismatch_high_confidence and low_speciesnet_confidence stay
        #             conservative (tsp + app) because SN has a meaningful signal there.
        #   Tier 4:   No manual review applied; all trusted quality-pass images assumed valid.
        if tier in (1, 2):
            eff_trusted = app
        elif tier == 3:
            fail_reason = r.get("trusted_sn_fail_reason", "")
            if _is_sn_coverage_gap(tsp, tqp, fail_reason):
                eff_trusted = tqp
            else:
                eff_trusted = tsp + app
        else:  # tier 4
            eff_trusted = tqp

        pool = eff_trusted + usp
        rows.append({
            **r,
            "review_approved":  app,
            "review_declined":  dec,
            "effective_trusted": eff_trusted,
            "effective_pool":   pool,
            "final_tier":       _tier(pool),
        })
    rows.sort(key=lambda x: x["effective_pool"], reverse=True)
    return rows


# ── Output ────────────────────────────────────────────────────────────────────

def write_csv(rows: list[dict], out_path: Path) -> None:
    if not rows:
        print("No data to write.")
        return
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV  → {out_path.relative_to(REPO_ROOT)}")


def write_md(
    rows: list[dict],
    approved: Counter,
    declined: Counter,
    args: argparse.Namespace,
    out_path: Path,
) -> None:
    total_app = sum(approved.values())
    total_dec = sum(declined.values())
    total_rev = total_app + total_dec

    # Tier summary: compare pre-review tier (from base CSV) vs final_tier
    pre_summary:   dict[int, dict] = {t: {"n": 0, "pool": 0} for t in (1, 2, 3, 4)}
    final_summary: dict[int, dict] = {t: {"n": 0, "pool": 0} for t in (1, 2, 3, 4)}
    tier_changes = 0
    for r in rows:
        pt = int(r["tier"])
        ft = int(r["final_tier"])
        pre_summary[pt]["n"]     += 1
        pre_summary[pt]["pool"]  += int(r["effective_pool"]) - int(r["review_approved"]) - int(r["unverified_sn_pass"]) + int(r["trusted_quality_pass"])
        final_summary[ft]["n"]    += 1
        final_summary[ft]["pool"] += int(r["effective_pool"])
        if pt != ft:
            tier_changes += 1

    lines: list[str] = [
        "# Post-Review Class Distribution Report\n",
        f"**Date:** {date.today()}  ",
        f"**Base CSV:** {args.input_csv}  ",
        f"**Decisions:** {args.decisions}  ",
        f"**Review stats:** {total_rev:,} images reviewed "
        f"({total_app:,} approved · {total_dec:,} declined)  ",
        f"**Tier changes after review:** {tier_changes} classes\n",
        "---\n",
        "## Filtering rules by tier\n",
        "| Tier | Trusted source gate | Unverified source gate |",
        "|---:|---|---|",
        "| 1 & 2 | Manual review (SN skipped); `review_approved` only | SpeciesNet pass |",
        "| 3 | SN-pass valid; coverage-gap classes (no_match/class <20% or order <15% pass rate) → `trusted_quality_pass`; otherwise `trusted_sn_pass + review_approved` | SpeciesNet pass |",
        "| 4 | All quality-pass assumed valid; `trusted_quality_pass` | SpeciesNet pass |",
        "",
        "## Tier Summary\n",
        "| Tier | Pool range | Pre-review classes | Post-review classes | Post-review pool |",
        "|---:|---|---:|---:|---:|",
    ]
    ranges = ["< 100", "100–499", "500–1 499", "≥ 1 500"]
    for t in (1, 2, 3, 4):
        pre   = pre_summary[t]
        final = final_summary[t]
        lines.append(
            f"| {t} | {ranges[t-1]} "
            f"| {pre['n']:,} "
            f"| {final['n']:,} | {final['pool']:,} |"
        )

    lines += [
        "",
        "## Per-Class Table\n",
        "Columns: `tq_pass` = trusted quality-pass · `tsn_pass` = trusted SN-pass · "
        "`uv_pass` = unverified SN-pass · `rev_app` = review approved · "
        "`rev_dec` = review declined · `eff_trusted` = effective trusted (tier-aware) · "
        "`eff_pool` = effective_trusted + uv_pass\n",
        "| Class | tq_pass | tsn_pass | uv_pass | tier "
        "| rev_app | rev_dec | eff_trusted | eff_pool | final_tier |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['class']} "
            f"| {r['trusted_quality_pass']:>} "
            f"| {r['trusted_sn_pass']:>} "
            f"| {r['unverified_sn_pass']:>} "
            f"| {r['tier']} "
            f"| {r['review_approved']:>} "
            f"| {r['review_declined']:>} "
            f"| {r['effective_trusted']:>} "
            f"| {r['effective_pool']:>} "
            f"| {r['final_tier']} |"
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
    parser.add_argument(
        "--input-csv", type=Path,
        default=REPO_ROOT / "reports" / "class_distribution.csv",
        metavar="PATH",
        help="Base class distribution CSV from script 8 (default: reports/class_distribution.csv).",
    )
    parser.add_argument(
        "--decisions", type=Path,
        default=REPO_ROOT / "reports" / "review_decisions.jsonl",
        metavar="PATH",
        help="Review decisions JSONL (default: reports/review_decisions.jsonl).",
    )
    parser.add_argument(
        "--output-csv", type=Path,
        default=REPO_ROOT / "reports" / "class_distribution_reviewed.csv",
        metavar="PATH",
    )
    parser.add_argument(
        "--output-md", type=Path,
        default=REPO_ROOT / "reports" / "class_distribution_reviewed.md",
        metavar="PATH",
    )
    args = parser.parse_args()

    if not args.input_csv.exists():
        print(
            f"ERROR: base CSV not found: {args.input_csv}\n"
            "Run scripts/dataset_quality/8-class_distribution_report.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not args.decisions.exists():
        print(f"ERROR: decisions file not found: {args.decisions}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading review decisions from {args.decisions.name} …")
    approved, declined = load_decisions(args.decisions)
    total = sum(approved.values()) + sum(declined.values())
    print(f"  {total:,} effective decisions "
          f"({sum(approved.values()):,} approved · {sum(declined.values()):,} declined) "
          f"across {len(approved) + len(declined)} classes.")

    print(f"Loading base CSV from {args.input_csv.name} …")
    base_rows = load_base_csv(args.input_csv)
    print(f"  {len(base_rows):,} classes.")

    print("Merging …")
    rows = build_rows(base_rows, approved, declined)

    write_csv(rows, args.output_csv)
    write_md(rows, approved, declined, args, args.output_md)
    print("\nDone.")


if __name__ == "__main__":
    main()
