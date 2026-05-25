"""
Plot available images per class as a sorted, stacked bar chart.

Shows the training/test split and synthetic image additions per class,
colour-coded by training band (A–D) as defined in:
  docs/plans/2026-05-04_dataset-construction-strategy.md

Bands and their training regime:
  A (<150 real)   : 200 synthetic only; all real images go to the test set
  B (150–249 real): 100 real train + 100 synthetic; remainder to test
  C (250–399 real): 200 real train; remainder to test
  D (≥400 real)   : all-real, min(pool−test, 1500) train; max(20%,50)≤500 test

Real pools > 500 are scaled proportionally to fit the 500-unit display height.
A ▲ marker indicates a capped bar.

Usage:
    python scripts/dataset_quality/plot_class_coverage.py
"""

import math
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = REPO_ROOT / "reports" / "class_distribution_reviewed.csv"
OUT_PATH = REPO_ROOT / "reports" / "class_image_counts.png"

CAP = 500
REF_STEP = 50
EXCLUDE = {"human", "unmatched"}

# (label, pool_lo, pool_hi, background_color)
BANDS = [
    ("A",  0,           150,         "#FFCCCC"),
    ("B",  150,         250,         "#FFF0CC"),
    ("C",  250,         400,         "#CCFFCC"),
    ("D",  400,         float("inf"), "#CCE5FF"),
]
BAND_DESCRIPTIONS = {
    "A": "A  <150 real  •  200 synth train / all real → test",
    "B": "B  150–249    •  100 real + 100 synth train / rest → test",
    "C": "C  250–399    •  200 real train / rest → test",
    "D": "D  ≥400       •  real only, up to 1500 train",
}

COLOR_TEST  = "#81C7D4"   # teal  – real images reserved for test
COLOR_TRAIN = "#2C6E9E"   # dark blue – real images used for training
COLOR_SYNTH = "#E07B39"   # amber – synthetic images used for training


def _assign_band(pool: int) -> str:
    for label, lo, hi, _ in BANDS:
        if lo <= pool < hi:
            return label
    return "D"


def _compute_display(pool: int, band: str) -> tuple[float, float, float, bool]:
    """Return (test_disp, train_disp, synth_disp, is_capped) scaled to CAP."""
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
    return test_r * scale, train_r * scale, float(synth), capped


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    df = df[~df["class"].isin(EXCLUDE)].copy()
    df = df.rename(columns={"class": "common_name"})

    df["band"] = df["effective_pool"].apply(lambda p: _assign_band(int(p)))

    rows = [_compute_display(int(r.effective_pool), r.band) for r in df.itertuples()]
    df["test_disp"], df["train_disp"], df["synth_disp"], df["capped"] = zip(*rows)

    df = df.sort_values("effective_pool", ascending=True).reset_index(drop=True)
    n = len(df)

    fig_w = max(32, n * 0.19)
    fig, ax = plt.subplots(figsize=(fig_w, 10))

    # Band background shading — bands are contiguous after ascending sort
    for label, lo, hi, color in BANDS:
        mask = df["band"] == label
        if not mask.any():
            continue
        idxs = df[mask].index.tolist()
        x0, x1 = idxs[0] - 0.5, idxs[-1] + 0.5
        ax.axvspan(x0, x1, color=color, alpha=0.4, zorder=0)
        ax.text((x0 + x1) / 2, CAP + 22, f"Band {label}  ({len(idxs)})",
                ha="center", va="bottom", fontsize=8, fontweight="bold")
        if x0 > 0:
            ax.axvline(x0, color="#555555", linewidth=0.9, linestyle="--", alpha=0.6, zorder=1)

    xs = range(n)
    ax.bar(xs, df["synth_disp"],  color=COLOR_SYNTH, width=0.8, zorder=2)
    ax.bar(xs, df["train_disp"],  bottom=df["synth_disp"],
           color=COLOR_TRAIN, width=0.8, zorder=2)
    ax.bar(xs, df["test_disp"],   bottom=df["synth_disp"] + df["train_disp"],
           color=COLOR_TEST,  width=0.8, zorder=2)

    # Small triangle above bars whose real pool was scaled down
    for i, row in df.iterrows():
        if row["capped"]:
            top = row["synth_disp"] + row["train_disp"] + row["test_disp"]
            ax.plot(i, top + 5, "^", color="#888888", markersize=4, zorder=4, clip_on=False)

    for y in range(REF_STEP, CAP + 1, REF_STEP):
        ax.axhline(y, color="gray", linewidth=0.7, alpha=0.4, zorder=1)

    ax.set_xticks(range(n))
    ax.set_xticklabels(df["common_name"], rotation=90, fontsize=5.5)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(0, CAP + 55)
    ax.set_ylabel("Images")
    ax.set_title(
        "Images per class — training / test split and synthetic additions by band"
        " (sorted ascending by real pool)"
    )

    band_legend = [
        mpatches.Patch(facecolor=color, alpha=0.6, label=BAND_DESCRIPTIONS[label])
        for label, lo, hi, color in BANDS
    ]
    seg_legend = [
        mpatches.Patch(color=COLOR_TEST,  label="Real — test set"),
        mpatches.Patch(color=COLOR_TRAIN, label="Real — training"),
        mpatches.Patch(color=COLOR_SYNTH, label="Synthetic — training"),
        mpatches.Patch(facecolor="#cccccc", edgecolor="#888888",
                       label="▲ real pool scaled to 500"),
    ]
    l1 = ax.legend(handles=band_legend, loc="upper left",
                   fontsize=6.5, title="Training bands", title_fontsize=7, framealpha=0.9)
    ax.add_artist(l1)
    ax.legend(handles=seg_legend, loc="lower right", fontsize=6.5, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)

    band_counts = df["band"].value_counts().sort_index()
    print(
        f"Saved → {OUT_PATH.relative_to(REPO_ROOT)}  ({n} classes: "
        + " / ".join(f"{k}={v}" for k, v in band_counts.items())
        + ")"
    )


if __name__ == "__main__":
    main()
