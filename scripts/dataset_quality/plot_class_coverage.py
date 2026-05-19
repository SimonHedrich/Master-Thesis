"""
Plot available images per class as a sorted bar chart.

Reads reports/coverage_analysis.csv and produces reports/class_image_counts.png.
Classes with more than 500 validated images are capped at 500 for display and
rendered in amber so the y-axis stays readable.

Usage:
    python scripts/dataset_quality/plot_class_coverage.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = REPO_ROOT / "reports" / "coverage_analysis.csv"
OUT_PATH = REPO_ROOT / "reports" / "class_image_counts.png"

CAP = 500
STEP = 50
COLOR_NORMAL = "#4C8BB5"   # steelblue
COLOR_CAPPED = "#E07B39"   # amber


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    df = df.sort_values("total_pass", ascending=True).reset_index(drop=True)

    capped_mask = df["total_pass"] > CAP
    display = df["total_pass"].clip(upper=CAP)
    colors = [COLOR_CAPPED if c else COLOR_NORMAL for c in capped_mask]

    n = len(df)
    fig_w = max(30, n * 0.18)
    fig, ax = plt.subplots(figsize=(fig_w, 9))

    ax.bar(range(n), display, color=colors, width=0.8, zorder=2)

    for y in range(STEP, CAP + 1, STEP):
        ax.axhline(y, color="gray", linewidth=0.8, alpha=0.4, zorder=1)

    ax.set_xticks(range(n))
    ax.set_xticklabels(df["common_name"], rotation=90, fontsize=6)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(0, CAP + STEP * 0.6)
    ax.set_ylabel("Images (capped at 500)")
    ax.set_title("Available images per class (validated, sorted ascending)")

    n_capped = capped_mask.sum()
    if n_capped:
        from matplotlib.patches import Patch
        ax.legend(
            handles=[Patch(color=COLOR_CAPPED, label=f">{CAP} images — display capped ({n_capped} classes)")],
            loc="upper left",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"Saved → {OUT_PATH.relative_to(REPO_ROOT)}  ({n} classes)")


if __name__ == "__main__":
    main()
