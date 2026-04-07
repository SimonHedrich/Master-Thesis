"""
Analyse filtered image coverage per class across all four datasets.

Reads filter_results.jsonl from each dataset directory and cross-references
against the 225 target classes in reports/class_counts_225.csv to produce
a detailed coverage report.

Usage:
    python scripts/analyse_dataset_coverage.py
    python scripts/analyse_dataset_coverage.py --verbose   # show per-class failure reasons
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).parent.parent

DATASETS = {
    "gbif": REPO_ROOT / "data/gbif/filter_results.jsonl",
    "inaturalist": REPO_ROOT / "data/inaturalist/filter_results.jsonl",
    "wikimedia": REPO_ROOT / "data/wikimedia/filter_results.jsonl",
    "openimages": REPO_ROOT / "data/supplementary_openimages/filter_results.jsonl",
}

CLASS_COUNTS_CSV = REPO_ROOT / "reports/class_counts_225.csv"
OUTPUT_FULL = REPO_ROOT / "reports/coverage_analysis.csv"
OUTPUT_GAPS = REPO_ROOT / "reports/coverage_gaps.csv"
OUTPUT_MD   = REPO_ROOT / "reports/coverage_report.md"

# Coverage tiers based on raw passed image count
TIER_EXCELLENT = 1500
TIER_GOOD = 1000
TIER_MARGINAL = 500
TIER_LOW = 100
QUALITY_BUFFER = 0.80  # keep 80% after estimated quality loss

TARGET_IMAGES = 1500


def classify_tier(count: int) -> str:
    if count >= TIER_EXCELLENT:
        return "excellent"
    elif count >= TIER_GOOD:
        return "good"
    elif count >= TIER_MARGINAL:
        return "marginal"
    elif count >= TIER_LOW:
        return "low"
    else:
        return "critical"


TIER_ORDER = ["critical", "low", "marginal", "good", "excellent"]
TIER_EMOJI = {
    "excellent": "✓✓",
    "good": "✓",
    "marginal": "~",
    "low": "!",
    "critical": "!!",
}


def extract_class(filepath: str) -> str | None:
    """Extract class name from a filter_results filepath.

    Paths follow the pattern: data/{source}/images/{class_name}/{filename}
    Open Images uses underscores in directory names; normalise to spaces.
    """
    parts = Path(filepath).parts
    try:
        img_idx = parts.index("images")
        class_part = parts[img_idx + 1]
        return class_part.replace("_", " ")
    except (ValueError, IndexError):
        return None


def load_filter_results(path: Path, dataset_name: str) -> tuple[dict, dict, dict]:
    """Return (passed_counts, failed_counts, failure_reasons) dicts keyed by class name."""
    passed: dict[str, int] = defaultdict(int)
    failed: dict[str, int] = defaultdict(int)
    reasons: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    if not path.exists():
        print(f"  [WARN] {dataset_name}: file not found at {path}")
        return passed, failed, reasons

    total = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            cls = extract_class(entry.get("filepath", ""))
            if cls is None:
                continue

            total += 1
            if entry.get("passed"):
                passed[cls] += 1
            else:
                failed[cls] += 1
                reason = entry.get("reason") or entry.get("stage_failed") or "unknown"
                reasons[cls][reason] += 1

    print(f"  {dataset_name:12s}: {total:>8,} entries  "
          f"passed={sum(passed.values()):>7,}  failed={sum(failed.values()):>7,}")
    return dict(passed), dict(failed), {k: dict(v) for k, v in reasons.items()}


def top_reasons(reasons_dict: dict[str, int], n: int = 3) -> str:
    """Return top-n failure reasons as a compact string."""
    if not reasons_dict:
        return ""
    sorted_r = sorted(reasons_dict.items(), key=lambda x: x[1], reverse=True)
    return "; ".join(f"{r}({c})" for r, c in sorted_r[:n])


TIER_BADGE = {
    "excellent": "🟢 Excellent",
    "good":      "🟡 Good",
    "marginal":  "🟠 Marginal",
    "low":       "🔴 Low",
    "critical":  "⛔ Critical",
}


def write_md_report(
    df: pd.DataFrame,
    all_passed: dict,
    all_failed: dict,
    class_list: list[str],
) -> None:
    """Write a human-readable Markdown coverage report."""
    from datetime import date

    tier_counts = df["status"].value_counts()
    total_pass = int(df["total_pass"].sum())
    total_buffer = int(df["after_buffer"].sum())

    tier_labels = {
        "excellent": f"≥{TIER_EXCELLENT}",
        "good":      f"{TIER_GOOD}–{TIER_EXCELLENT - 1}",
        "marginal":  f"{TIER_MARGINAL}–{TIER_GOOD - 1}",
        "low":       f"{TIER_LOW}–{TIER_MARGINAL - 1}",
        "critical":  f"<{TIER_LOW}",
    }

    lines: list[str] = []

    # ── Header ────────────────────────────────────────────────────
    lines += [
        "# Dataset Coverage Report",
        "",
        f"Generated: {date.today().isoformat()}  ",
        f"Target classes: **{len(class_list)}**  ",
        f"Datasets: GBIF · iNaturalist · Wikimedia · Open Images  ",
        f"Quality buffer applied: **20%** (estimated post-filtering loss)",
        "",
    ]

    # ── Tier summary ──────────────────────────────────────────────
    lines += [
        "## Coverage Summary",
        "",
        f"Ultralytics guideline: **≥1,500 images per class** (≥1,200 after buffer).  ",
        f"Total passed images across all datasets: **{total_pass:,}**  ",
        f"Estimated usable after 20% buffer: **{total_buffer:,}**",
        "",
        "| Tier | Range | Classes |",
        "|------|-------|---------|",
    ]
    for tier in reversed(TIER_ORDER):  # best → worst
        badge = TIER_BADGE[tier]
        label = tier_labels[tier]
        count = tier_counts.get(tier, 0)
        lines.append(f"| {badge} | {label} imgs | {count} |")
    lines.append("")

    # ── Per-dataset summary ───────────────────────────────────────
    lines += [
        "## Per-Dataset Summary",
        "",
        "| Dataset | Passed | Failed | Total | Pass% | Classes covered |",
        "|---------|-------:|-------:|------:|------:|----------------:|",
    ]
    for ds in DATASETS:
        p = sum(all_passed[ds].values())
        f = sum(all_failed[ds].values())
        t = p + f
        pct = f"{p / t * 100:.1f}%" if t > 0 else "n/a"
        covered = sum(1 for cls in class_list if all_passed[ds].get(cls, 0) > 0)
        lines.append(f"| {ds} | {p:,} | {f:,} | {t:,} | {pct} | {covered} |")
    lines.append("")

    # ── Full class table sorted by total descending ───────────────
    lines += [
        "## All Classes by Coverage",
        "",
        "Sorted by total passed images (highest first).  ",
        "**Buffer** = total × 0.80 (estimated usable). **Gap** = images still needed to reach 1,200 usable.",
        "",
        "| # | Class | Scientific name | iNaturalist | GBIF | Open Images | Wikimedia | **Total** | Buffer | Gap | Status |",
        "|--:|-------|-----------------|------------:|-----:|------------:|----------:|----------:|-------:|----:|--------|",
    ]
    df_by_total = df.sort_values("total_pass", ascending=False).reset_index(drop=True)
    for i, row in df_by_total.iterrows():
        gap_str = f"{int(row['gap']):,}" if row["gap"] > 0 else "—"
        badge = TIER_BADGE[row["status"]]
        lines.append(
            f"| {i + 1} "
            f"| {row['common_name']} "
            f"| *{row['scientific_name']}* "
            f"| {int(row['inaturalist_pass']):,} "
            f"| {int(row['gbif_pass']):,} "
            f"| {int(row['openimages_pass']):,} "
            f"| {int(row['wikimedia_pass']):,} "
            f"| **{int(row['total_pass']):,}** "
            f"| {int(row['after_buffer']):,} "
            f"| {gap_str} "
            f"| {badge} |"
        )
    lines.append("")

    # ── Gap detail: critical classes ──────────────────────────────
    critical_df = df[df["status"] == "critical"].sort_values("total_pass")
    low_df      = df[df["status"] == "low"].sort_values("total_pass")

    lines += [
        "## Critical Classes — Urgent Supplementation Needed",
        "",
        f"**{len(critical_df)} classes** with fewer than {TIER_LOW} passed images.",
        "",
        "| Class | Scientific name | iNat | GBIF | OI | Wiki | Total | Gap |",
        "|-------|-----------------|-----:|-----:|---:|-----:|------:|----:|",
    ]
    for _, row in critical_df.iterrows():
        lines.append(
            f"| {row['common_name']} "
            f"| *{row['scientific_name']}* "
            f"| {int(row['inaturalist_pass']):,} "
            f"| {int(row['gbif_pass']):,} "
            f"| {int(row['openimages_pass']):,} "
            f"| {int(row['wikimedia_pass']):,} "
            f"| {int(row['total_pass']):,} "
            f"| {int(row['gap']):,} |"
        )
    lines.append("")

    lines += [
        "## Low Classes — Supplementation Recommended",
        "",
        f"**{len(low_df)} classes** with {TIER_LOW}–{TIER_MARGINAL - 1} passed images.",
        "",
        "| Class | Scientific name | iNat | GBIF | OI | Wiki | Total | Gap |",
        "|-------|-----------------|-----:|-----:|---:|-----:|------:|----:|",
    ]
    for _, row in low_df.iterrows():
        lines.append(
            f"| {row['common_name']} "
            f"| *{row['scientific_name']}* "
            f"| {int(row['inaturalist_pass']):,} "
            f"| {int(row['gbif_pass']):,} "
            f"| {int(row['openimages_pass']):,} "
            f"| {int(row['wikimedia_pass']):,} "
            f"| {int(row['total_pass']):,} "
            f"| {int(row['gap']):,} |"
        )
    lines.append("")

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main(verbose: bool = False) -> None:
    print("=" * 60)
    print("DATASET COVERAGE ANALYSIS")
    print("=" * 60)

    # ── 1. Load class list ────────────────────────────────────────
    classes_df = pd.read_csv(CLASS_COUNTS_CSV)
    classes_df.columns = classes_df.columns.str.strip()
    classes_df["common_name"] = classes_df["common_name"].str.strip().str.lower()
    class_list = classes_df["common_name"].tolist()
    print(f"\nTarget classes loaded: {len(class_list)}")

    # ── 2. Parse filter results ───────────────────────────────────
    print("\nReading filter_results.jsonl files:")
    all_passed: dict[str, dict[str, int]] = {}
    all_failed: dict[str, dict[str, int]] = {}
    all_reasons: dict[str, dict[str, dict[str, int]]] = {}

    for name, path in DATASETS.items():
        p, f, r = load_filter_results(path, name)
        all_passed[name] = p
        all_failed[name] = f
        all_reasons[name] = r

    # ── 3. Build coverage matrix ──────────────────────────────────
    rows = []
    for _, row in classes_df.iterrows():
        cls = row["common_name"]
        sci = row["scientific_name"]
        level = row["level"]

        ds_pass = {ds: all_passed[ds].get(cls, 0) for ds in DATASETS}
        ds_fail = {ds: all_failed[ds].get(cls, 0) for ds in DATASETS}

        total_pass = sum(ds_pass.values())
        total_all = total_pass + sum(ds_fail.values())
        after_buffer = round(total_pass * QUALITY_BUFFER)
        tier = classify_tier(total_pass)
        gap = max(0, TARGET_IMAGES - after_buffer)

        # Aggregate failure reasons for this class across all datasets
        combined_reasons: dict[str, int] = defaultdict(int)
        for ds in DATASETS:
            for reason, cnt in all_reasons[ds].get(cls, {}).items():
                combined_reasons[reason] += cnt

        rows.append({
            "common_name": cls,
            "scientific_name": sci,
            "level": level,
            "gbif_pass": ds_pass["gbif"],
            "inaturalist_pass": ds_pass["inaturalist"],
            "wikimedia_pass": ds_pass["wikimedia"],
            "openimages_pass": ds_pass["openimages"],
            "total_pass": total_pass,
            "total_imgs": total_all,
            "after_buffer": after_buffer,
            "status": tier,
            "gap": gap,
            "top_failure_reasons": top_reasons(dict(combined_reasons)),
        })

    df = pd.DataFrame(rows)

    # Sort worst-first for the full CSV
    df["_tier_ord"] = df["status"].map({t: i for i, t in enumerate(TIER_ORDER)})
    df_sorted = df.sort_values(["_tier_ord", "total_pass"]).drop(columns="_tier_ord")

    # ── 4. Save outputs ───────────────────────────────────────────
    df_sorted.to_csv(OUTPUT_FULL, index=False)
    print(f"\nFull report saved → {OUTPUT_FULL.relative_to(REPO_ROOT)}")

    gaps_df = df_sorted[df_sorted["status"].isin(["critical", "low"])].copy()
    gaps_df = gaps_df.sort_values("gap", ascending=False)
    gaps_df.to_csv(OUTPUT_GAPS, index=False)
    print(f"Gaps report saved → {OUTPUT_GAPS.relative_to(REPO_ROOT)}")

    write_md_report(df, all_passed, all_failed, class_list)
    print(f"Markdown report saved → {OUTPUT_MD.relative_to(REPO_ROOT)}")

    # ── 5. Console summary ────────────────────────────────────────
    tier_counts = df["status"].value_counts()

    print(f"\n{'=' * 60}")
    print("COVERAGE TIERS (based on raw passed image count)")
    print(f"  Target: ≥{TARGET_IMAGES} raw images → ≥{round(TARGET_IMAGES * QUALITY_BUFFER):,} usable after {int((1-QUALITY_BUFFER)*100)}% buffer")
    print(f"{'─' * 60}")
    tier_labels = {
        "excellent": f"≥{TIER_EXCELLENT}",
        "good":      f"{TIER_GOOD}–{TIER_EXCELLENT - 1}",
        "marginal":  f"{TIER_MARGINAL}–{TIER_GOOD - 1}",
        "low":       f"{TIER_LOW}–{TIER_MARGINAL - 1}",
        "critical":  f"<{TIER_LOW}",
    }
    for tier in TIER_ORDER:
        count = tier_counts.get(tier, 0)
        label = tier_labels[tier]
        marker = TIER_EMOJI[tier]
        print(f"  {marker} {tier.upper():8s} ({label:>9s} imgs): {count:>3d} classes")

    total_pass_all = df["total_pass"].sum()
    total_after_buffer = df["after_buffer"].sum()
    print(f"{'─' * 60}")
    print(f"  Total passed images (all datasets):  {total_pass_all:>8,}")
    print(f"  After 20% buffer estimate:           {total_after_buffer:>8,}")

    # ── 6. Per-dataset summary ────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("PER-DATASET SUMMARY")
    print(f"{'─' * 60}")
    print(f"  {'Dataset':12s} | {'Passed':>8s} | {'Failed':>8s} | {'Total':>8s} | {'Pass%':>6s} | Classes")
    print(f"  {'─'*12}-+-{'─'*8}-+-{'─'*8}-+-{'─'*8}-+-{'─'*6}-+-{'─'*7}")
    for ds in DATASETS:
        p = sum(all_passed[ds].values())
        f = sum(all_failed[ds].values())
        t = p + f
        pct = f"{p/t*100:.1f}%" if t > 0 else "n/a"
        covered = sum(1 for cls in class_list if all_passed[ds].get(cls, 0) > 0)
        print(f"  {ds:12s} | {p:>8,} | {f:>8,} | {t:>8,} | {pct:>6s} | {covered}")

    # ── 7. Critical & low class detail ───────────────────────────
    critical_df = df_sorted[df_sorted["status"] == "critical"].copy()
    low_df = df_sorted[df_sorted["status"] == "low"].copy()

    print(f"\n{'=' * 60}")
    print(f"CRITICAL CLASSES (<{TIER_LOW} imgs) — {len(critical_df)} total")
    print(f"{'─' * 60}")
    _print_class_table(critical_df, verbose)

    print(f"\n{'=' * 60}")
    print(f"LOW CLASSES ({TIER_LOW}–{TIER_MARGINAL - 1} imgs) — {len(low_df)} total")
    print(f"{'─' * 60}")
    _print_class_table(low_df, verbose)

    if verbose:
        print(f"\n{'=' * 60}")
        print("MARGINAL CLASSES (500–999 imgs)")
        print(f"{'─' * 60}")
        marginal_df = df_sorted[df_sorted["status"] == "marginal"].copy()
        _print_class_table(marginal_df, verbose=False)


def _print_class_table(subset: pd.DataFrame, verbose: bool) -> None:
    if subset.empty:
        print("  (none)")
        return
    header = f"  {'Class':<30s} {'Total':>6s} {'Buf':>5s} {'Gap':>5s}  {'iNat':>5s} {'GBIF':>5s} {'OI':>4s} {'Wiki':>4s}"
    print(header)
    print(f"  {'─'*30} {'─'*6} {'─'*5} {'─'*5}  {'─'*5} {'─'*5} {'─'*4} {'─'*4}")
    for _, row in subset.iterrows():
        gap_str = f"{int(row['gap']):,}" if row["gap"] > 0 else "ok"
        print(
            f"  {row['common_name']:<30s} "
            f"{int(row['total_pass']):>6,} "
            f"{int(row['after_buffer']):>5,} "
            f"{gap_str:>5s}  "
            f"{int(row['inaturalist_pass']):>5,} "
            f"{int(row['gbif_pass']):>5,} "
            f"{int(row['openimages_pass']):>4,} "
            f"{int(row['wikimedia_pass']):>4,}"
        )
        if verbose and row["top_failure_reasons"]:
            print(f"    └─ failures: {row['top_failure_reasons']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse dataset coverage per class.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show top failure reasons for critical/low classes")
    args = parser.parse_args()
    main(verbose=args.verbose)
