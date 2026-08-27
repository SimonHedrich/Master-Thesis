"""Master comparison: every trained cell in the synthetic-model-comparison
experiment, in one place — the 6 local `maxlen` cells, the API incumbent,
and the prompt-length-ablation cells (full + compressed, two models).

Consolidates what docs 14/15/16 each covered separately:
  14 — the 6 local `maxlen` cells (controlled: model varies, everything
       else fixed)
  15 — + the gemini incumbent (confounded: model AND prompt regime vary)
  16 — the controlled full-vs-compressed ablation (model fixed, prompt
       length varies) for gpt-image-2-low and gemini-3.1-flash-lite-image

This script aggregates all 12 cells trained so far into one ranking, one
per-class heatmap, and one confusion chart, colored by category (local
diffusion / API full-prompt / API compressed-prompt) so the ablation pairs
are visible in context rather than siloed in their own doc. No combined
cost chart: local cells cost GPU-hours, API cells cost $/image — those
units aren't comparable on one axis (see 6-compare_maxlen_cells.py's
cost-vs-map chart for the local-only version).

Usage:
    uv run python scripts/synthetic_model_comparison/9-compare_all_models.py
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

# (generator, prompt_regime, category) — category drives chart color, not aggregation.
CELLS = [
    ("realvisxl-lightning", "maxlen", "local"),
    ("sd35m", "maxlen", "local"),
    ("flux2-klein-9b", "maxlen", "local"),
    ("sd35-large-turbo", "maxlen", "local"),
    ("sd35-large", "maxlen", "local"),
    ("hidream-i1", "maxlen", "local"),
    ("gemini-3.1-flash-image-preview", "full", "api_full"),
    ("gpt-image-2-low", "full", "api_full"),
    ("gpt-image-2-medium", "full", "api_full"),
    ("gemini-3.1-flash-lite-image", "full", "api_full"),
    ("gpt-image-2-low", "compressed", "api_compressed"),
    ("gemini-3.1-flash-lite-image", "compressed", "api_compressed"),
]

HEADLINE_METRICS = [
    "map", "map_50", "map_75", "map_medium", "map_large",
    "mar_1", "mar_10", "mar_100", "mar_medium", "mar_large",
]
CONFUSION_GROUPS = ["zebra"]  # ursus dropped: this 12-class subset has only 1 ursus-group member
                              # (american black bear) so its confusion rate is structurally 0, not a finding

# Palette (docs/synthetic-model-comparison dataviz convention — validated categorical order).
BLUE = "#2a78d6"      # local diffusion
ORANGE = "#eb6834"    # API, full prompt
AQUA = "#1baf7a"       # API, compressed prompt
CATEGORY_COLOR = {"local": BLUE, "api_full": ORANGE, "api_compressed": AQUA}
CATEGORY_BASE_LABEL = {
    "local": "local diffusion (maxlen)",
    "api_full": "API, full prompt",
    "api_compressed": "API, compressed prompt",
}


def category_label(cells: list[dict], category: str) -> str:
    """Seed count varies per cell and changes as more seeds are trained —
    compute the range from the actual aggregated data instead of hardcoding it."""
    seeds = sorted({c["n_seeds"] for c in cells if c["category"] == category})
    if not seeds:
        seed_str = "0 seeds"
    elif len(seeds) == 1:
        seed_str = f"{seeds[0]} seed" + ("" if seeds[0] == 1 else "s")
    else:
        seed_str = f"{seeds[0]}-{seeds[-1]} seeds"
    return f"{CATEGORY_BASE_LABEL[category]}, {seed_str}"
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
    """Locate each seed's evaluation_report.json for a generator/regime cell.

    Normally under <run_dir>/evaluation/. realvisxl-lightning-maxlen-seed42
    hit the pre-fix post-training eval-DataLoader hang and was later
    re-evaluated standalone into <run_dir>/eval_best/ instead — fall back
    there when evaluation/ is empty.
    """
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


def load_cell(generator: str, prompt_regime: str, category: str) -> dict | None:
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
        "category": category,
        "n_seeds": len(reports),
        "headline": headline,
        "confusion_overall": confusion_overall,
        "confusion_groups": confusion_groups,
        "per_class": per_class,
    }


def label(cell: dict) -> str:
    if cell["category"] == "local":
        return cell["generator"]
    return f"{cell['generator']}\n({cell['prompt_regime']})"


def write_headline_csv(cells: list[dict], path: Path) -> None:
    fieldnames = ["generator", "prompt_regime", "category", "n_seeds"]
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
                "category": cell["category"],
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
        writer.writerow(["generator", "prompt_regime", "category", "class_name", "band", "real_test_images", "ap_mean", "ap_std"])
        for cell in cells:
            for name, info in cell["per_class"].items():
                m, s = info["ap_mean_std"]
                writer.writerow([cell["generator"], cell["prompt_regime"], cell["category"], name, info["band"], info["real_test_images"], m, s])
    n_rows = sum(len(c["per_class"]) for c in cells)
    print(f"wrote {path.relative_to(REPO_ROOT)} ({n_rows} rows)")


def plot_headline_map(cells: list[dict], path: Path) -> None:
    ranked = sorted(cells, key=lambda c: c["headline"]["map"][0] or 0.0, reverse=True)
    names = [label(c) for c in ranked]
    means = [c["headline"]["map"][0] or 0.0 for c in ranked]
    stds = [c["headline"]["map"][1] or 0.0 for c in ranked]
    colors = [CATEGORY_COLOR[c["category"]] for c in ranked]

    fig, ax = plt.subplots(figsize=(14, 6.5))
    xs = np.arange(len(names))
    ax.bar(xs, means, yerr=stds, capsize=4, color=colors, width=0.65, zorder=2,
           error_kw={"ecolor": INK_SECONDARY, "elinewidth": 1.2})
    for x, m, s in zip(xs, means, stds):
        ax.text(x, m + s + 0.002, f"{m:.3f}", ha="center", va="bottom", fontsize=8, color=INK_SECONDARY)

    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Real-test mAP (fine, 12-way)")
    title = "All 12 trained cells ranked by downstream YOLO26n mAP"
    ax.set_title(title, fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.yaxis.grid(True, color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    handles = [plt.matplotlib.patches.Patch(color=CATEGORY_COLOR[k], label=category_label(cells, k)) for k in ["local", "api_full", "api_compressed"]]
    ax.legend(handles=handles, frameon=False, loc="upper right")
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
    fig, ax = plt.subplots(figsize=(1.5 * len(col_order) + 3, 0.45 * len(class_order) + 2))
    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=0.0)

    col_labels = [label(by_key[k]) for k in col_order]
    ax.set_xticks(range(len(col_order)))
    ax.set_xticklabels(col_labels, fontsize=7, rotation=30, ha="right")
    row_labels = [f"{cls}  ({by_key[ref]['per_class'][cls]['band']})" for cls in class_order]
    ax.set_yticks(range(len(class_order)))
    ax.set_yticklabels(row_labels)

    for i in range(len(class_order)):
        for j in range(len(col_order)):
            v = data[i, j]
            color = "white" if v > data.max() * 0.6 else "#0b0b0b"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7, color=color)

    ax.set_title("Per-class AP — all 12 trained cells (row label shows band A/B/D)")
    fig.colorbar(im, ax=ax, label="AP", shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path.relative_to(REPO_ROOT)}")


def plot_confusion(cells: list[dict], path: Path) -> None:
    ranked = sorted(cells, key=lambda c: c["headline"]["map"][0] or 0.0, reverse=True)
    names = [label(c) for c in ranked]
    xs = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(14, 6.5))
    values = [(c["confusion_groups"]["zebra"][0] or 0.0) for c in ranked]
    ax.bar(xs, values, width=0.55, color=BLUE, zorder=2)

    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Within-group confusion rate")
    ax.set_title("Zebra-group confusion — all 12 cells, ranked by mAP (same order as headline chart)", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.yaxis.grid(True, color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path.relative_to(REPO_ROOT)}")


def main() -> None:
    cells = [c for (g, r, cat) in CELLS if (c := load_cell(g, r, cat)) is not None]
    if not cells:
        raise SystemExit("no cells with evaluation reports found — nothing to aggregate")

    REPORTS_DIR.mkdir(exist_ok=True)
    write_headline_csv(cells, REPORTS_DIR / "model_comparison_full_downstream_map.csv")
    write_per_class_csv(cells, REPORTS_DIR / "model_comparison_full_per_class_ap.csv")

    plot_headline_map(cells, REPORTS_DIR / "model_comparison_full_headline_map.png")
    plot_per_class_heatmap(cells, REPORTS_DIR / "model_comparison_full_per_class_heatmap.png")
    plot_confusion(cells, REPORTS_DIR / "model_comparison_full_confusion.png")

    if len(cells) < len(CELLS):
        missing = [g for g, r, _ in CELLS if (g, r) not in {(c["generator"], c["prompt_regime"]) for c in cells}]
        print(f"NOTE: {len(cells)}/{len(CELLS)} cells aggregated — missing: {missing}")


if __name__ == "__main__":
    main()
