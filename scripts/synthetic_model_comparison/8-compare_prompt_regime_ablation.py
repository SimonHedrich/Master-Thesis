"""Prompt-length ablation: full vs. compressed, same model held fixed.

Doc 15 flagged the API-incumbent comparison as confounded: gemini differed
from the local `maxlen` cells in two ways at once (model *and* prompt
length), so its mAP lead couldn't be attributed to either cause alone. This
script builds the real, controlled ablation — the same model, generated
under both the unabridged `full` prompt regime (~1,300 words) and the
length-capped `compressed` regime (~55-75 tokens) — for two models
(`gpt-image-2-low`, `gemini-3.1-flash-lite-image`), a 2x2 grid. A fifth,
bonus cell (`gpt-image-2-medium/full`) has no compressed counterpart and is
reported alongside but outside the controlled grid.

Usage:
    uv run python scripts/synthetic_model_comparison/8-compare_prompt_regime_ablation.py
"""
from __future__ import annotations

import csv
import json
import math
import statistics
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_EXPORTS = REPO_ROOT / "scripts" / "synthetic_model_comparison" / "training" / "model_exports"
REPORTS_DIR = REPO_ROOT / "reports"

# (generator, prompt_regime, in_core_grid) — the 2x2 ablation pair plus one
# bonus cell (gpt-image-2-medium/full has no compressed counterpart).
CELLS = [
    ("gpt-image-2-low", "full", True),
    ("gpt-image-2-low", "compressed", True),
    ("gemini-3.1-flash-lite-image", "full", True),
    ("gemini-3.1-flash-lite-image", "compressed", True),
    ("gpt-image-2-medium", "full", False),
]

HEADLINE_METRICS = [
    "map", "map_50", "map_75", "map_medium", "map_large",
    "mar_1", "mar_10", "mar_100", "mar_medium", "mar_large",
]
CONFUSION_GROUPS = ["zebra"]  # ursus dropped: this 12-class subset has only 1 ursus-group member
                              # (american black bear) so its confusion rate is structurally 0, not a finding

BLUE = "#2a78d6"
ORANGE = "#eb6834"
MUTED = "#898781"
GRID = "#e1e0d9"
INK_SECONDARY = "#52514e"
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95"]


def slugify(name: str) -> str:
    return name.replace(".", "-").replace("_", "-")


def _clean(v: float | None) -> float | None:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return float(v)


def find_reports(generator: str, prompt_regime: str) -> list[Path]:
    slug = slugify(generator)
    reports = []
    for run_dir in sorted(MODEL_EXPORTS.glob(f"yolo26n-{slug}-{prompt_regime}-seed*")):
        if "smoke" in run_dir.name:
            continue
        for subdir in ("evaluation", "eval_best"):
            report = run_dir / subdir / "evaluation_report.json"
            if report.exists():
                reports.append(report)
                break
    return reports


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def load_cell(generator: str, prompt_regime: str, in_core_grid: bool) -> dict | None:
    report_paths = find_reports(generator, prompt_regime)
    if not report_paths:
        print(f"WARNING: no evaluation reports found for {generator!r} ({prompt_regime}) — skipping")
        return None
    reports = [json.loads(p.read_text()) for p in report_paths]

    headline = {m: mean_std([_clean(r["headline"][m]) for r in reports]) for m in HEADLINE_METRICS}
    confusion_overall = mean_std([_clean(r["confusion"]["overall_confusion_rate"]) for r in reports])
    confusion_groups = {
        g: mean_std([_clean(r["confusion"]["by_group"][g]["confusion_rate"]) for r in reports])
        for g in CONFUSION_GROUPS
    }

    per_class: dict[str, dict] = {}
    for cls in reports[0]["per_class"]:
        name = cls["class_name"]
        aps = [_clean(next(c["ap"] for c in r["per_class"] if c["class_name"] == name)) for r in reports]
        per_class[name] = {
            "band": cls["band"],
            "real_test_images": cls["real_test_images"],
            "ap_mean_std": mean_std(aps),
        }

    return {
        "generator": generator,
        "prompt_regime": prompt_regime,
        "in_core_grid": in_core_grid,
        "n_seeds": len(reports),
        "headline": headline,
        "confusion_overall": confusion_overall,
        "confusion_groups": confusion_groups,
        "per_class": per_class,
    }


def write_headline_csv(cells: list[dict], path: Path) -> None:
    fieldnames = ["generator", "prompt_regime", "in_core_grid", "n_seeds"]
    for metric in HEADLINE_METRICS:
        fieldnames += [f"{metric}_mean", f"{metric}_std"]
    fieldnames += ["confusion_overall_mean", "confusion_overall_std"]
    for group in CONFUSION_GROUPS:
        fieldnames += [f"confusion_{group}_mean", f"confusion_{group}_std"]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cell in cells:
            row = {
                "generator": cell["generator"],
                "prompt_regime": cell["prompt_regime"],
                "in_core_grid": cell["in_core_grid"],
                "n_seeds": cell["n_seeds"],
            }
            for metric in HEADLINE_METRICS:
                m, s = cell["headline"][metric]
                row[f"{metric}_mean"], row[f"{metric}_std"] = m, s
            m, s = cell["confusion_overall"]
            row["confusion_overall_mean"], row["confusion_overall_std"] = m, s
            for group in CONFUSION_GROUPS:
                m, s = cell["confusion_groups"][group]
                row[f"confusion_{group}_mean"], row[f"confusion_{group}_std"] = m, s
            writer.writerow(row)
    print(f"wrote {path.relative_to(REPO_ROOT)} ({len(cells)} rows)")


def write_per_class_csv(cells: list[dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["generator", "prompt_regime", "class_name", "band", "real_test_images", "ap_mean", "ap_std"])
        for cell in cells:
            for name, info in cell["per_class"].items():
                m, s = info["ap_mean_std"]
                writer.writerow([cell["generator"], cell["prompt_regime"], name, info["band"], info["real_test_images"], m, s])
    n_rows = sum(len(c["per_class"]) for c in cells)
    print(f"wrote {path.relative_to(REPO_ROOT)} ({n_rows} rows)")


def plot_grouped_map(cells: list[dict], path: Path) -> None:
    by_key = {(c["generator"], c["prompt_regime"]): c for c in cells}
    models = ["gpt-image-2-low", "gemini-3.1-flash-lite-image"]
    xs = np.arange(len(models))
    width = 0.32

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for offset, regime, color in [(-width / 2, "full", BLUE), (width / 2, "compressed", ORANGE)]:
        means = [by_key[(m, regime)]["headline"]["map"][0] or 0.0 for m in models]
        stds = [by_key[(m, regime)]["headline"]["map"][1] or 0.0 for m in models]
        bars = ax.bar(xs + offset, means, width=width, yerr=stds, capsize=4, color=color, zorder=2,
                      label=regime, error_kw={"ecolor": INK_SECONDARY, "elinewidth": 1.2})
        for x, m, s in zip(xs + offset, means, stds):
            ax.text(x, m + s + 0.002, f"{m:.3f}", ha="center", va="bottom", fontsize=8, color=INK_SECONDARY)

    # Bonus cell: gpt-image-2-medium/full, shown as a muted reference marker
    if ("gpt-image-2-medium", "full") in by_key:
        bonus = by_key[("gpt-image-2-medium", "full")]
        bm = bonus["headline"]["map"][0] or 0.0
        ax.axhline(bm, color=MUTED, linewidth=1.2, linestyle="--", zorder=1)
        ax.text(len(models) - 0.5, bm + 0.002, f"gpt-image-2-medium/full: {bm:.3f} (bonus, no compressed pair)",
                ha="right", va="bottom", fontsize=8, color=MUTED)

    ax.set_xticks(xs)
    ax.set_xticklabels(models)
    ax.set_ylabel("Real-test mAP (fine, 12-way)")
    title = "Prompt-length ablation: full vs. compressed, same model held fixed (mean ± std over 2 seeds)"
    ax.set_title("\n".join(textwrap.wrap(title, 60)), fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.yaxis.grid(True, color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path.relative_to(REPO_ROOT)}")


def plot_per_class_heatmap(cells: list[dict], path: Path) -> None:
    by_key = {(c["generator"], c["prompt_regime"]): c for c in cells}
    col_order = [(g, r) for g, r, _ in CELLS if (g, r) in by_key]
    class_names = list(next(iter(by_key.values()))["per_class"].keys())
    ref = col_order[0]
    class_order = sorted(class_names, key=lambda n: (by_key[ref]["per_class"][n]["band"], n))

    data = np.array([
        [by_key[k]["per_class"][cls]["ap_mean_std"][0] or 0.0 for k in col_order]
        for cls in class_order
    ])

    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_BLUE)
    fig, ax = plt.subplots(figsize=(1.6 * len(col_order) + 3, 0.4 * len(class_order) + 2))
    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=0.0)

    col_labels = [f"{g}\n({r})" for g, r in col_order]
    ax.set_xticks(range(len(col_order)))
    ax.set_xticklabels(col_labels, fontsize=8)
    row_labels = [f"{cls}  ({by_key[ref]['per_class'][cls]['band']})" for cls in class_order]
    ax.set_yticks(range(len(class_order)))
    ax.set_yticklabels(row_labels)

    for i in range(len(class_order)):
        for j in range(len(col_order)):
            v = data[i, j]
            color = "white" if v > data.max() * 0.6 else "#0b0b0b"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8, color=color)

    ax.set_title("Per-class AP: prompt-length ablation — row label shows band A/B/D")
    fig.colorbar(im, ax=ax, label="AP", shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path.relative_to(REPO_ROOT)}")


def main() -> None:
    cells = [c for (g, r, core) in CELLS if (c := load_cell(g, r, core)) is not None]
    if not cells:
        raise SystemExit("no cells with evaluation reports found — nothing to aggregate")

    REPORTS_DIR.mkdir(exist_ok=True)
    write_headline_csv(cells, REPORTS_DIR / "model_comparison_prompt_ablation_downstream_map.csv")
    write_per_class_csv(cells, REPORTS_DIR / "model_comparison_prompt_ablation_per_class_ap.csv")

    plot_grouped_map(cells, REPORTS_DIR / "model_comparison_prompt_ablation_headline_map.png")
    plot_per_class_heatmap(cells, REPORTS_DIR / "model_comparison_prompt_ablation_per_class_heatmap.png")

    if len(cells) < len(CELLS):
        missing = [g for g, r, _ in CELLS if (g, r) not in {(c["generator"], c["prompt_regime"]) for c in cells}]
        print(f"NOTE: {len(cells)}/{len(CELLS)} cells aggregated — missing: {missing}")


if __name__ == "__main__":
    main()
