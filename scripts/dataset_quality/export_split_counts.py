"""
Export actual (unscaled) train/test split counts per class for Google Sheets.

Produces reports/class_split_counts.csv with columns:
  class, band, effective_pool, train_real, train_synth, test_real

Formulas per band (from docs/plans/2026-05-04_dataset-construction-strategy.md §3):
  A  train_real=0,   train_synth=200, test_real=pool
  B  train_real=100, train_synth=100, test_real=pool-100
  C  train_real=200, train_synth=0,   test_real=pool-200
  D  test_real=min(max(floor(pool*0.2), 50), 500)
     train_real=min(pool-test_real, 1500)
     train_synth=0

Usage:
    uv run python -m scripts.dataset_quality.export_split_counts
"""

import math
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
CSV_IN  = REPO_ROOT / "reports" / "class_distribution_reviewed.csv"
CSV_OUT = REPO_ROOT / "reports" / "class_split_counts.csv"

EXCLUDE = {"human", "unmatched"}


def assign_band(pool: int) -> str:
    if pool < 150:
        return "A"
    if pool < 250:
        return "B"
    if pool < 400:
        return "C"
    return "D"


def compute_split(pool: int, band: str) -> tuple[int, int, int]:
    """Return (train_real, train_synth, test_real)."""
    if band == "A":
        return 0, 200, pool
    if band == "B":
        return 100, 100, pool - 100
    if band == "C":
        return 200, 0, pool - 200
    # Band D
    test_real  = min(max(math.floor(pool * 0.2), 50), 500)
    train_real = min(pool - test_real, 1500)
    return train_real, 0, test_real


def main() -> None:
    df = pd.read_csv(CSV_IN)
    df = df[~df["class"].isin(EXCLUDE)].copy()

    df["band"] = df["effective_pool"].apply(lambda p: assign_band(int(p)))

    splits = [compute_split(int(r.effective_pool), r.band) for r in df.itertuples()]
    df["train_real"], df["train_synth"], df["test_real"] = zip(*splits)

    df = df.sort_values("effective_pool", ascending=True).reset_index(drop=True)

    out = df[["class", "band", "effective_pool", "train_real", "train_synth", "test_real"]]
    out.to_csv(CSV_OUT, index=False)

    band_counts = df["band"].value_counts().sort_index()
    print(f"Saved → {CSV_OUT.relative_to(REPO_ROOT)}  ({len(df)} classes: "
          + " / ".join(f"{k}={v}" for k, v in band_counts.items()) + ")")

    # Sanity-check a few representative Band D rows
    print("\nBand D spot-check:")
    band_d = df[df["band"] == "D"][["class", "effective_pool", "train_real", "test_real"]]
    for _, row in band_d.sample(5, random_state=42).sort_values("effective_pool").iterrows():
        print(f"  {row['class']:30s}  pool={row['effective_pool']:6d}  "
              f"train={row['train_real']:5d}  test={row['test_real']:4d}")


if __name__ == "__main__":
    main()
