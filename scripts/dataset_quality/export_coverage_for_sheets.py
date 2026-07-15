"""
Export pre-computed chart data for the class coverage plot as a Google-Sheets-ready CSV.

Produces reports/class_image_counts_sheets.csv with columns:
  class, band, effective_pool, synth_disp, train_disp, test_disp, capped

Import that CSV into Google Sheets, select the class + three *_disp columns,
then Insert → Chart → Stacked column chart.

Colours to set manually:
  synth_disp  → #E07B39  (amber)
  train_disp  → #2C6E9E  (dark blue)
  test_disp   → #81C7D4  (teal)

Usage:
    uv run python -m scripts.dataset_quality.export_coverage_for_sheets
"""

import math
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
CSV_IN  = REPO_ROOT / "reports" / "class_distribution_reviewed.csv"
CSV_OUT = REPO_ROOT / "reports" / "class_image_counts_sheets.csv"

CAP = 500
EXCLUDE = {"human", "unmatched"}

BANDS = [
    ("A",   0,          150),
    ("B", 150,          250),
    ("C", 250,          400),
    ("D", 400, float("inf")),
]


def assign_band(pool: int) -> str:
    for label, lo, hi in BANDS:
        if lo <= pool < hi:
            return label
    return "D"


def compute_display(pool: int, band: str) -> tuple[float, float, float, bool]:
    """Return (test_disp, train_disp, synth_disp, capped) scaled to CAP."""
    if band == "A":
        test_r, train_r, synth = pool, 0, 200
    elif band == "B":
        test_r, train_r, synth = pool - 100, 100, 100
    elif band == "C":
        test_r, train_r, synth = pool - 200, 200, 0
    else:  # D
        test_r  = min(max(math.floor(pool * 0.2), 50), 500)
        train_r = min(pool - test_r, 1500)
        synth   = 0

    real_total = test_r + train_r
    capped = real_total > CAP
    scale = (CAP / real_total) if capped else 1.0
    return round(test_r * scale, 2), round(train_r * scale, 2), float(synth), capped


def main() -> None:
    df = pd.read_csv(CSV_IN)
    df = df[~df["class"].isin(EXCLUDE)].copy()

    df["band"] = df["effective_pool"].apply(lambda p: assign_band(int(p)))

    rows = [compute_display(int(r.effective_pool), r.band) for r in df.itertuples()]
    df["test_disp"], df["train_disp"], df["synth_disp"], df["capped"] = zip(*rows)

    df = df.sort_values("effective_pool", ascending=True).reset_index(drop=True)

    out = df[["class", "band", "effective_pool", "synth_disp", "train_disp", "test_disp", "capped"]]
    out.to_csv(CSV_OUT, index=False)

    band_counts = df["band"].value_counts().sort_index()
    print(f"Saved → {CSV_OUT.relative_to(REPO_ROOT)}  ({len(df)} classes: "
          + " / ".join(f"{k}={v}" for k, v in band_counts.items()) + ")")
    print()
    print("Google Sheets steps:")
    print("  1. File → Import → upload class_image_counts_sheets.csv")
    print("  2. Select columns: class, synth_disp, train_disp, test_disp")
    print("  3. Insert → Chart → Stacked column chart")
    print("  4. Set series colours:")
    print("       synth_disp  #E07B39  (amber)")
    print("       train_disp  #2C6E9E  (dark blue)")
    print("       test_disp   #81C7D4  (teal)")
    print("  5. Colour rows by band: A=#FFCCCC  B=#FFF0CC  C=#CCFFCC  D=#CCE5FF")


if __name__ == "__main__":
    main()
