"""Aggregate real-test eval results across all trained generator cells,
including the API-model incumbent alongside the 6 local `maxlen` cells.

Extends `6-compare_maxlen_cells.py`'s aggregation to also cover
`gemini-3.1-flash-image-preview` (the production incumbent, `full` prompt
regime — a different pipeline than the local `maxlen` grid, see doc `13`).
Only the incumbent is included here: it's the only API-model cell that has
actually been trained so far. `gemini-3.1-flash-lite-image` and both
`gpt-image-2` tiers have generated images but haven't been through the
labeling pipeline (stages 2-5) or training yet — out of scope for this run.

Results are provisional: built on `5-export_coco.py`'s best-effort
MegaDetector export, not the human-reviewed labels stages 3/4 would produce
(TODO.md §3.2/§3.4) — see each report's own caveat. The incumbent cell also
differs methodologically from the 6 local cells: 1 seed only (not 2), and a
`full` prompt regime rather than `maxlen` — see the emitted doc for both
caveats spelled out.

Usage:
    uv run python scripts/synthetic_model_comparison/7-compare_all_cells.py
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

# (generator, prompt_regime, kind) — kind distinguishes local-diffusion cells
# (GPU-hours to generate) from API cells (billed per-image, no comparable
# GPU-hours figure).
CELLS = [
    ("realvisxl-lightning", "maxlen", "local"),
    ("sd35m", "maxlen", "local"),
    ("flux2-klein-9b", "maxlen", "local"),
    ("sd35-large-turbo", "maxlen", "local"),
    ("sd35-large", "maxlen", "local"),
    ("hidream-i1", "maxlen", "local"),
    ("gemini-3.1-flash-image-preview", "full", "api"),
]

# Measured hours to generate the full 1,200-image maxlen cell (TODO.md §3.1,
# docs/synthetic-model-comparison/13 §6/§9). hidream-i1's ~43h is the
# planning estimate (129.85s/image x 1200), not yet timed end-to-end.
# API cells have no comparable GPU-hours figure (billed ~€0.04/image instead)
# — omitted from GENERATION_HOURS and from the cost-vs-quality chart.
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
CONFUSION_GROUPS = ["zebra"]  # ursus dropped: this 12-class subset has only 1 ursus-group member
                              # (american black bear) so its confusion rate is structurally 0, not a finding

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


def find_reports(generator: str, prompt_regime: str) -> list[Path]:
    """Locate each seed's evaluation_report.json for a generator/regime cell.

    Normally under <run_dir>/evaluation/. One run
    (realvisxl-lightning-maxlen-seed42) hit the pre-fix post-training
    eval-DataLoader hang (docs/synthetic-model-comparison README, 2026-08-04
    update) and was later re-evaluated standalone into <run_dir>/eval_best/
    instead — fall back there when evaluation/ is empty.
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


def load_cell(generator: str, prompt_regime: str, kind: str) -> dict | None:
    report_paths = find_reports(generator, prompt_regime)
    if not report_paths:
        print(f"WARNING: no evaluation reports found for {generator!r} ({prompt_regime}) — skipping")
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
        "prompt_regime": prompt_regime,
        "kind": kind,
        "n_seeds": len(reports),
        "generation_hours": GENERATION_HOURS.get(generator),
        "headline": headline,
        "confusion_overall": confusion_overall,
        "confusion_groups": confusion_groups,
        "per_class": per_class,
    }


def write_headline_csv(cells: list[dict], path: Path) -> None:
    fieldnames = ["generator", "prompt_regime", "kind", "n_seeds", "generation_hours"]
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
                "kind": cell["kind"],
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
        writer.writerow(["generator", "prompt_regime", "class_name", "band", "real_test_images", "test_limited", "ap_mean", "ap_std"])
        for cell in cells:
            for name, info in cell["per_class"].items():
                m, s = info["ap_mean_std"]
                writer.writerow([cell["generator"], cell["prompt_regime"], name, info["band"], info["real_test_images"], info["test_limited"], m, s])
    n_rows = sum(len(c["per_class"]) for c in cells)
    print(f"wrote {path.relative_to(REPO_ROOT)} ({n_rows} rows)")


def plot_headline_map(cells: list[dict], path: Path) -> None:
    ranked = sorted(cells, key=lambda c: c["headline"]["map"][0] or 0.0, reverse=True)
    names = [c["generator"] for c in ranked]
    means = [c["headline"]["map"][0] or 0.0 for c in ranked]
    stds = [c["headline"]["map"][1] or 0.0 for c in ranked]
    colors = [ORANGE if c["kind"] == "api" else BLUE for c in ranked]

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    xs = np.arange(len(names))
    ax.bar(xs, means, yerr=stds, capsize=4, color=colors, width=0.6, zorder=2,
           error_kw={"ecolor": INK_SECONDARY, "elinewidth": 1.2})
    for x, m, s in zip(xs, means, stds):
        ax.text(x, m + s + 0.002, f"{m:.3f}", ha="center", va="bottom", fontsize=9, color=INK_SECONDARY)

    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Real-test mAP (fine, 12-way)")
    title = "Downstream YOLO26n mAP: 6 local maxlen cells (blue) + the API incumbent (orange, 1 seed, full regime)"
    ax.set_title("\n".join(textwrap.wrap(title, 60)), fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.yaxis.grid(True, color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    handles = [
        plt.matplotlib.patches.Patch(color=BLUE, label="local diffusion (maxlen, 2 seeds)"),
        plt.matplotlib.patches.Patch(color=ORANGE, label="API incumbent (full, 1 seed)"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path.relative_to(REPO_ROOT)}")


def plot_per_class_heatmap(cells: list[dict], path: Path) -> None:
    by_generator = {c["generator"]: c for c in cells}
    generators = [g for g, _, _ in CELLS if g in by_generator]
    class_names = list(next(iter(by_generator.values()))["per_class"].keys())
    class_order = sorted(class_names, key=lambda n: (by_generator[generators[0]]["per_class"][n]["band"], n))

    data = np.array([
        [by_generator[g]["per_class"][cls]["ap_mean_std"][0] or 0.0 for g in generators]
        for cls in class_order
    ])

    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_BLUE)
    fig, ax = plt.subplots(figsize=(1.3 * len(generators) + 3, 0.4 * len(class_order) + 2))
    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=0.0)

    col_labels = [f"{g}*" if by_generator[g]["kind"] == "api" else g for g in generators]
    ax.set_xticks(range(len(generators)))
    ax.set_xticklabels(col_labels, rotation=20, ha="right")
    row_labels = [f"{cls}  ({by_generator[generators[0]]['per_class'][cls]['band']})" for cls in class_order]
    ax.set_yticks(range(len(class_order)))
    ax.set_yticklabels(row_labels)

    for i in range(len(class_order)):
        for j in range(len(generators)):
            v = data[i, j]
            color = "white" if v > data.max() * 0.6 else "#0b0b0b"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8, color=color)

    ax.set_title("Per-class AP: local maxlen cells + API incumbent (*, full regime, 1 seed) — row label shows band A/B/D")
    fig.colorbar(im, ax=ax, label="AP", shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path.relative_to(REPO_ROOT)}")


def plot_confusion(cells: list[dict], path: Path) -> None:
    ranked = sorted(cells, key=lambda c: c["generator"])
    names = [c["generator"] for c in ranked]
    xs = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    values = [(c["confusion_groups"]["zebra"][0] or 0.0) for c in ranked]
    ax.bar(xs, values, width=0.6, color=BLUE, zorder=2)

    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Within-group confusion rate")
    title = "Zebra-group confusion, local cells + API incumbent (only look-alike group with >1 member in this class subset)"
    ax.set_title("\n".join(textwrap.wrap(title, 60)), fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.yaxis.grid(True, color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path.relative_to(REPO_ROOT)}")


def main() -> None:
    cells = [c for (g, regime, kind) in CELLS if (c := load_cell(g, regime, kind)) is not None]
    if not cells:
        raise SystemExit("no cells with evaluation reports found — nothing to aggregate")

    REPORTS_DIR.mkdir(exist_ok=True)
    write_headline_csv(cells, REPORTS_DIR / "model_comparison_all_downstream_map.csv")
    write_per_class_csv(cells, REPORTS_DIR / "model_comparison_all_per_class_ap.csv")

    plot_headline_map(cells, REPORTS_DIR / "model_comparison_all_headline_map.png")
    plot_per_class_heatmap(cells, REPORTS_DIR / "model_comparison_all_per_class_heatmap.png")
    plot_confusion(cells, REPORTS_DIR / "model_comparison_all_confusion.png")
    # No cost-vs-quality chart here: API cells are billed per-image (€), not
    # GPU-hours — the two cost units aren't comparable on one axis. See
    # 6-compare_maxlen_cells.py's plot_cost_vs_map for the local-only version.

    if len(cells) < len(CELLS):
        missing = [g for g, _, _ in CELLS if g not in {c["generator"] for c in cells}]
        print(f"NOTE: {len(cells)}/{len(CELLS)} cells aggregated — missing: {missing}")


if __name__ == "__main__":
    main()
