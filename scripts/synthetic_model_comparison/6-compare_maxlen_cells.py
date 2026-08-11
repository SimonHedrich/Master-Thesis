"""Aggregate real-test eval results across the 6 local-model `maxlen` cells.

Each `training/run_training_pipeline.py --full-eval` run only emits its own
`evaluation/evaluation_report.json` (one cell, one seed). This script globs
every seed's report per generator, aggregates mean/std across seeds, and
writes a combined CSV table plus comparison charts — the cross-cell view
`docs/synthetic-model-comparison/README.md`'s per-seed table so far has only
assembled by hand.

Scope: the 6 local-diffusion `maxlen` cells (docs/synthetic-model-comparison/
13_local-model-roster-overhaul-and-maxlen-regime.md) — the apples-to-apples
grid this experiment's final design converged on. The API-model `full`/
`compressed` cells use a different pipeline/prompt-regime and aren't included.

Results are provisional: built on `5-export_coco.py`'s best-effort
MegaDetector export, not the human-reviewed labels stages 3/4 would produce
(TODO.md §3.2/§3.4) — see the emitted report's own caveat.

Usage:
    uv run python scripts/synthetic_model_comparison/6-compare_maxlen_cells.py
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

# Order matches the cheapest-first generation order (TODO.md §1.2 / §3.1).
GENERATORS = [
    "realvisxl-lightning",
    "sd35m",
    "flux2-klein-9b",
    "sd35-large-turbo",
    "sd35-large",
    "hidream-i1",
]

# Measured hours to generate the full 1,200-image maxlen cell (TODO.md §3.1,
# docs/synthetic-model-comparison/13 §6/§9). hidream-i1's ~43h is the
# planning estimate (129.85s/image x 1200), not yet timed end-to-end.
GENERATION_HOURS = {
    "realvisxl-lightning": 0.31,
    "sd35m": 7.98,
    "flux2-klein-9b": 10.40,
    "sd35-large-turbo": 8.27,
    "sd35-large": 18.11,
    "hidream-i1": 43.3,
}

HEADLINE_METRICS = [
    "map", "map_50", "map_75", "map_medium", "map_large",
    "mar_1", "mar_10", "mar_100", "mar_medium", "mar_large",
]
CONFUSION_GROUPS = ["zebra", "ursus"]  # only groups with any matched instances in this 12-class subset

# Palette (docs/synthetic-model-comparison dataviz convention — validated categorical order)
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


def find_reports(generator: str) -> list[Path]:
    """Locate each seed's evaluation_report.json for a generator.

    Normally under <run_dir>/evaluation/. One run
    (realvisxl-lightning-maxlen-seed42) hit the pre-fix post-training
    eval-DataLoader hang (docs/synthetic-model-comparison README, 2026-08-04
    update) and was later re-evaluated standalone into <run_dir>/eval_best/
    instead — fall back there when evaluation/ is empty.
    """
    slug = slugify(generator)
    reports = []
    for run_dir in sorted(MODEL_EXPORTS.glob(f"yolo26n-{slug}-maxlen-seed*")):
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


def load_cell(generator: str) -> dict | None:
    report_paths = find_reports(generator)
    if not report_paths:
        print(f"WARNING: no evaluation reports found for {generator!r} — skipping")
        return None
    reports = [json.loads(p.read_text()) for p in report_paths]

    headline = {}
    for metric in HEADLINE_METRICS:
        values = [_clean(r["headline"][metric]) for r in reports]
        headline[metric] = mean_std(values)

    confusion_overall = mean_std([_clean(r["confusion"]["overall_confusion_rate"]) for r in reports])
    confusion_groups = {}
    for group in CONFUSION_GROUPS:
        values = [_clean(r["confusion"]["by_group"][group]["confusion_rate"]) for r in reports]
        confusion_groups[group] = mean_std(values)

    per_class: dict[str, dict] = {}
    for cls in reports[0]["per_class"]:
        name = cls["class_name"]
        aps = [_clean(next(c["ap"] for c in r["per_class"] if c["class_name"] == name)) for r in reports]
        per_class[name] = {
            "band": cls["band"],
            "real_test_images": cls["real_test_images"],
            "test_limited": cls["test_limited"],
            "ap_mean_std": mean_std(aps),
        }

    return {
        "generator": generator,
        "n_seeds": len(reports),
        "generation_hours": GENERATION_HOURS.get(generator),
        "headline": headline,
        "confusion_overall": confusion_overall,
        "confusion_groups": confusion_groups,
        "per_class": per_class,
    }


def write_headline_csv(cells: list[dict], path: Path) -> None:
    fieldnames = ["generator", "n_seeds", "generation_hours"]
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
                "n_seeds": cell["n_seeds"],
                "generation_hours": cell["generation_hours"],
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
        writer.writerow(["generator", "class_name", "band", "real_test_images", "test_limited", "ap_mean", "ap_std"])
        for cell in cells:
            for name, info in cell["per_class"].items():
                m, s = info["ap_mean_std"]
                writer.writerow([cell["generator"], name, info["band"], info["real_test_images"], info["test_limited"], m, s])
    n_rows = sum(len(c["per_class"]) for c in cells)
    print(f"wrote {path.relative_to(REPO_ROOT)} ({n_rows} rows)")


def plot_headline_map(cells: list[dict], path: Path) -> None:
    ranked = sorted(cells, key=lambda c: c["headline"]["map"][0] or 0.0, reverse=True)
    names = [c["generator"] for c in ranked]
    means = [c["headline"]["map"][0] or 0.0 for c in ranked]
    stds = [c["headline"]["map"][1] or 0.0 for c in ranked]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    xs = np.arange(len(names))
    ax.bar(xs, means, yerr=stds, capsize=4, color=BLUE, width=0.6, zorder=2,
           error_kw={"ecolor": INK_SECONDARY, "elinewidth": 1.2})
    for x, m, s in zip(xs, means, stds):
        ax.text(x, m + s + 0.002, f"{m:.3f}", ha="center", va="bottom", fontsize=9, color=INK_SECONDARY)

    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Real-test mAP (fine, 12-way)")
    title = "Downstream YOLO26n mAP by synthetic-generator (maxlen cells, mean ± std over 2 seeds)"
    ax.set_title("\n".join(textwrap.wrap(title, 55)), fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.yaxis.grid(True, color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path.relative_to(REPO_ROOT)}")


def plot_per_class_heatmap(cells: list[dict], path: Path) -> None:
    by_generator = {c["generator"]: c for c in cells}
    generators = [g for g in GENERATORS if g in by_generator]
    class_names = list(next(iter(by_generator.values()))["per_class"].keys())
    class_order = sorted(class_names, key=lambda n: (by_generator[generators[0]]["per_class"][n]["band"], n))

    data = np.array([
        [by_generator[g]["per_class"][cls]["ap_mean_std"][0] or 0.0 for g in generators]
        for cls in class_order
    ])

    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_BLUE)
    fig, ax = plt.subplots(figsize=(1.3 * len(generators) + 3, 0.4 * len(class_order) + 2))
    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=0.0)

    ax.set_xticks(range(len(generators)))
    ax.set_xticklabels(generators, rotation=20, ha="right")
    row_labels = [f"{cls}  ({by_generator[generators[0]]['per_class'][cls]['band']})" for cls in class_order]
    ax.set_yticks(range(len(class_order)))
    ax.set_yticklabels(row_labels)

    for i in range(len(class_order)):
        for j in range(len(generators)):
            v = data[i, j]
            color = "white" if v > data.max() * 0.6 else "#0b0b0b"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8, color=color)

    ax.set_title("Per-class AP by generator (maxlen cells, mean over 2 seeds) — row label shows band A/B/D")
    fig.colorbar(im, ax=ax, label="AP", shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path.relative_to(REPO_ROOT)}")


def plot_confusion(cells: list[dict], path: Path) -> None:
    ranked = sorted(cells, key=lambda c: c["generator"])
    names = [c["generator"] for c in ranked]
    xs = np.arange(len(names))
    width = 0.32

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for offset, group, color in [(-width / 2, "zebra", BLUE), (width / 2, "ursus", ORANGE)]:
        values = [(c["confusion_groups"][group][0] or 0.0) for c in ranked]
        ax.bar(xs + offset, values, width=width, color=color, zorder=2, label=f"{group} group")

    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Within-group confusion rate")
    title = "Look-alike-group confusion by generator (only groups with matched instances in this class subset)"
    ax.set_title("\n".join(textwrap.wrap(title, 55)), fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.yaxis.grid(True, color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.text(0.5, 0.01,
             "ursus group: 0.000 across all cells — bears are never confused with each other in this class subset.",
             ha="center", fontsize=8, color=INK_SECONDARY)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path.relative_to(REPO_ROOT)}")


def plot_cost_vs_map(cells: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for c in cells:
        x = c["generation_hours"]
        y = c["headline"]["map"][0] or 0.0
        ax.scatter(x, y, s=64, color=BLUE, zorder=3)
        ax.annotate(c["generator"], (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8, color=INK_SECONDARY)

    ax.set_xlabel("Generation time for the 1,200-image cell (hours)")
    ax.set_ylabel("Real-test mAP")
    ax.set_title("Generation cost vs. downstream mAP (maxlen cells)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.grid(True, color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path.relative_to(REPO_ROOT)}")


def main() -> None:
    cells = [c for g in GENERATORS if (c := load_cell(g)) is not None]
    if not cells:
        raise SystemExit("no cells with evaluation reports found — nothing to aggregate")

    REPORTS_DIR.mkdir(exist_ok=True)
    write_headline_csv(cells, REPORTS_DIR / "model_comparison_maxlen_downstream_map.csv")
    write_per_class_csv(cells, REPORTS_DIR / "model_comparison_maxlen_per_class_ap.csv")

    plot_headline_map(cells, REPORTS_DIR / "model_comparison_maxlen_headline_map.png")
    plot_per_class_heatmap(cells, REPORTS_DIR / "model_comparison_maxlen_per_class_heatmap.png")
    plot_confusion(cells, REPORTS_DIR / "model_comparison_maxlen_confusion.png")
    plot_cost_vs_map(cells, REPORTS_DIR / "model_comparison_maxlen_cost_vs_map.png")

    if len(cells) < len(GENERATORS):
        missing = [g for g in GENERATORS if g not in {c["generator"] for c in cells}]
        print(f"NOTE: {len(cells)}/{len(GENERATORS)} cells aggregated — missing: {missing}")


if __name__ == "__main__":
    main()
