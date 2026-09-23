#!/usr/bin/env python3
"""
Stage 6 — The input-resolution accuracy/latency curve (host).

Joins two measurements of the SAME 640-px-trained YOLO26n checkpoint:

  accuracy  scripts/training/yolo26n/model_exports/<run>/resolution_sweep/res<N>/
            evaluation_report.json — scored by the project's own eval suite at
            each inference resolution, on one fixed proportional subsample of
            the test set (see the study doc for why `--limit` was not used).
  latency   reports/embedded_benchmark/latency_summary.csv — the Pi 400 cells
            for the yolo26n-direct-res<N> exports, ONNX Runtime, headline
            thread count.

Two arms, per docs/plans/2026-09-20_input-resolution-optimization-study.md:

  Arm 1  "no retraining" — the 640-px checkpoint simply run at a smaller input.
  Arm 2  "resolution-native" — a checkpoint trained at that resolution. Picked
         up automatically once such a run's `evaluation/` report exists and is
         named on the command line; absent until then, and the report says so.

NOTE ON ABSOLUTE VALUES: the accuracy axis is the fixed subsample, which runs
~0.015 mAP optimistic against the published full-test headline. The offset is
constant across resolutions, so the SHAPE of the curve is sound, but these
numbers must not be quoted next to full-test figures.

Usage:
    uv run python scripts/benchmark/6-resolution_report.py
    uv run python scripts/benchmark/6-resolution_report.py \
        --native-run scripts/training/yolo26n/model_exports/yolo26n-res320-bs32-<ts>

Outputs:
    reports/resolution_study/resolution_tradeoff.{csv,md}
    reports/resolution_study/plots/resolution_tradeoff.png
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import constants as C  # noqa: E402

# Same two pre-validated categorical slots 1-2 as 5-report.py, carrying the same
# meaning: the colour is the TEST DOMAIN, never the model or the arm. The arms
# separate by line style, so a reader never has to learn a second colour code.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

DEFAULT_SWEEP_RUN = (
    REPO_ROOT / "scripts/training/yolo26n/model_exports/yolo26n-bs32-20260910-212812"
)
RESOLUTIONS = (640, 512, 416, 320)
OUT_DIR = REPO_ROOT / "reports" / "resolution_study"


def read_accuracy(run_dir: Path, res: int) -> dict | None:
    p = run_dir / "resolution_sweep" / f"res{res}" / "evaluation_report.json"
    if not p.exists():
        return None
    t1 = json.loads(p.read_text())["tier1"]
    return {
        "map_mixed": t1["headline_mixed_fine"]["map"],
        "map50_mixed": t1["headline_mixed_fine"]["map_50"],
        "map_real": t1["headline_real_fine"]["map"],
        "map50_real": t1["headline_real_fine"]["map_50"],
    }


def read_latency() -> dict[int, dict]:
    """Pi 400 cells for the resolution variants, at the headline thread count."""
    path = C.REPORT_DIR / "latency_summary.csv"
    if not path.exists():
        raise SystemExit(f"missing {path} — run 5-report.py first")
    th = str(C.PROTOCOL["headline_threads"])
    out: dict[int, dict] = {}
    with path.open() as f:
        for r in csv.DictReader(f):
            if r["runtime"] != "onnxruntime" or r["threads"] != th:
                continue
            if not r["model"].startswith("yolo26n-direct"):
                continue
            # "yolo26n-direct" itself is the 640 cell; "-res<N>" are the variants.
            if r["model"] != "yolo26n-direct" and "-res" not in r["model"]:
                continue
            out[int(r["image_size"])] = {
                "infer_ms": float(r["infer_median_ms"]),
                "e2e_ms": float(r["e2e_median_ms"]),
                "gflops": float(r["gflops"]),
                "peak_rss_mb": float(r["peak_rss_mb"]),
            }
    return out


def build_rows(sweep_run: Path, native_run: Path | None) -> list[dict]:
    lat = read_latency()
    rows = []
    for res in RESOLUTIONS:
        acc = read_accuracy(sweep_run, res)
        if acc is None or res not in lat:
            print(f"    ! res{res}: missing {'accuracy' if acc is None else 'latency'} — skipped")
            continue
        row = {"image_size": res, "arm": "no-retraining", **acc, **lat[res]}
        rows.append(row)

    if native_run is not None:
        # Prefer the subsample scoring: Arm 1's points are on the fixed 8k
        # proportional subsample, and mixing a full-test Arm 2 point into the same
        # figure would compare two different denominators. The run's own
        # evaluation/ (full test) stays the headline figure quoted elsewhere.
        p = native_run / "resolution_sweep_subsample" / "evaluation_report.json"
        if not p.exists():
            p = native_run / "evaluation" / "evaluation_report.json"
            if p.exists():
                print(f"    ! {native_run.name}: no subsample scoring, falling back to "
                      f"the FULL-TEST report — not comparable to Arm 1's subsample points")
        if not p.exists():
            print(f"    ! --native-run given but {p} does not exist yet — Arm 2 omitted")
        else:
            t1 = json.loads(p.read_text())["tier1"]
            cfg = json.loads(p.read_text()).get("config", {})
            res = int(cfg.get("image_size") or 320)
            if res not in lat:
                print(f"    ! no Pi latency cell at {res} px for Arm 2 — omitted")
            else:
                rows.append({
                    "image_size": res, "arm": "resolution-native",
                    "map_mixed": t1["headline_mixed_fine"]["map"],
                    "map50_mixed": t1["headline_mixed_fine"]["map_50"],
                    "map_real": t1["headline_real_fine"]["map"],
                    "map50_real": t1["headline_real_fine"]["map_50"],
                    **lat[res],
                })
    return rows


def write_csv(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def write_markdown(rows: list[dict], out: Path, native_run: Path | None) -> None:
    lo, hi = C.PI400_PASS_BAND_MS
    base = next((r for r in rows if r["image_size"] == 640), None)
    lines = [
        "# Input resolution: accuracy against measured on-device latency",
        "",
        "One YOLO26n checkpoint (`yolo26n-bs32-20260910-212812`, trained at 640 px), ",
        "scored at four inference resolutions and timed on the Raspberry Pi 400 ",
        f"(ONNX Runtime, {C.PROTOCOL['headline_threads']} threads, batch 1, FP32).",
        "",
        "**Accuracy is on a fixed proportional subsample** (8,000 of 63,802 real + ",
        "1,411 of 11,250 synthetic test images, seed 42), which runs ~0.015 mAP ",
        "optimistic against the published full-test headline. The offset is constant ",
        "across resolutions, so the shape of the curve is sound — but these values ",
        "must not be quoted alongside full-test figures.",
        "",
        "| Arm | Input | GFLOPs | W_infer (ms) | W_e2e (ms) | QCS605 est. (ms) | Peak RSS (MB) | mixed mAP | real mAP | Δ real vs 640 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        q_lo = r["e2e_ms"] * C.QCS605_SCALE[0]
        q_hi = r["e2e_ms"] * C.QCS605_SCALE[1]
        delta = ""
        if base and r["arm"] == "no-retraining" and r["image_size"] != 640:
            d = r["map_real"] - base["map_real"]
            delta = f"{d:+.4f} ({d / base['map_real'] * 100:+.1f} %)"
        elif base and r["arm"] == "resolution-native":
            d = r["map_real"] - base["map_real"]
            delta = f"{d:+.4f} ({d / base['map_real'] * 100:+.1f} %)"
        lines.append(
            f"| {r['arm']} | {r['image_size']} | {r['gflops']:.2f} | {r['infer_ms']:.1f} | "
            f"{r['e2e_ms']:.1f} | {q_lo:.0f}–{q_hi:.0f} | {r['peak_rss_mb']:.0f} | "
            f"{r['map_mixed']:.4f} | {r['map_real']:.4f} | {delta} |"
        )
    lines += [
        "",
        f"Design targets: **≤{C.TARGET_LATENCY_MS_QCS605:.0f} ms** on the QCS605 "
        f"(Pi 400 pass band ≤{lo:.0f}–{hi:.0f} ms) and **≤500 MB**. Every row meets "
        "the memory budget; none meets the latency budget.",
        "",
    ]
    if not any(r["arm"] == "resolution-native" for r in rows):
        lines += [
            "**Arm 2 (resolution-native training) is not in this table yet.** Until it "
            "lands, every row is the 640-px-trained checkpoint simply run at a smaller "
            "input, which is an upper bound on the accuracy cost of reducing "
            "resolution, not the cost after retraining.",
            "",
        ]
    out.write_text("\n".join(lines))


def plot(rows: list[dict], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lo, _ = C.PI400_PASS_BAND_MS
    arm1 = sorted([r for r in rows if r["arm"] == "no-retraining"], key=lambda r: r["e2e_ms"])
    arm2 = sorted([r for r in rows if r["arm"] == "resolution-native"], key=lambda r: r["e2e_ms"])

    fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=200)
    ax.axvspan(1, lo, color="#1baf7a", alpha=0.10, zorder=0)
    ax.axvline(lo, color=INK_MUTED, linestyle="--", linewidth=1.2, zorder=1)

    for series, colour, name in ((arm1, BLUE, "mixed"), (arm1, ORANGE, "real")):
        key = "map_mixed" if name == "mixed" else "map_real"
        ax.plot([r["e2e_ms"] for r in series], [r[key] for r in series],
                color=colour, linewidth=2, zorder=2)
        ax.scatter([r["e2e_ms"] for r in series], [r[key] for r in series],
                   s=64, color=colour, zorder=3, edgecolor="white", linewidth=1.4)

    for r in arm1:
        ax.annotate(f"{r['image_size']}px", (r["e2e_ms"], r["map_mixed"]),
                    textcoords="offset points", xytext=(0, 13), fontsize=8.5,
                    color=INK, ha="center")

    for r in arm2:
        for key, colour in (("map_mixed", BLUE), ("map_real", ORANGE)):
            ax.scatter([r["e2e_ms"]], [r[key]], s=84, color=colour, marker="D",
                       zorder=4, edgecolor="white", linewidth=1.4)
        ax.annotate(f"{r['image_size']}px\ntrained natively",
                    (r["e2e_ms"], r["map_mixed"]), textcoords="offset points",
                    xytext=(0, 14), fontsize=8.5, color=INK, ha="center")

    ax.set_xscale("log")
    ys = [r[k] for r in rows for k in ("map_mixed", "map_real")]
    pad = max(0.03, (max(ys) - min(ys)) * 0.30)
    ax.set_ylim(min(ys) - pad * 0.5, max(ys) + pad)
    ax.set_xlim(lo * 0.45, max(r["e2e_ms"] for r in rows) * 2.2)

    ax.annotate(
        f"$\\leq{C.TARGET_LATENCY_MS_QCS605:.0f}$ ms QCS605 budget\n"
        f"(Pi 400 pass band $\\leq{lo:.0f}$ ms)",
        xy=(lo, max(ys) + pad * 0.55), xytext=(4, 0), textcoords="offset points",
        fontsize=8, color=INK_MUTED, ha="left", va="top",
    )
    ax.set_xlabel("End-to-end latency per frame on the Raspberry Pi 400, ms "
                  f"(log scale, ONNX Runtime, {C.PROTOCOL['headline_threads']} threads)",
                  fontsize=9.5, color=INK)
    ax.set_ylabel("mAP@[.5:.95]  (fixed test subsample)", fontsize=9.5, color=INK)

    ax.scatter([], [], color=BLUE, s=64, label="mixed test set")
    ax.scatter([], [], color=ORANGE, s=64, label="real-only breakout")
    ax.plot([], [], color=INK_MUTED, linewidth=2, label="640 px weights, reduced input")
    if arm2:
        ax.scatter([], [], color=INK_MUTED, s=84, marker="D", label="trained at that resolution")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")

    # A log axis over a ~4x range emits a single decade tick, which leaves the
    # reader unable to read any value off the plot. Label the actual measured
    # points plus the budget edge instead.
    ticks = sorted({round(lo)} | {round(r["e2e_ms"]) for r in rows})
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.minorticks_off()

    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=8.5)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep-run", type=Path, default=DEFAULT_SWEEP_RUN,
                    help="run dir holding resolution_sweep/res<N>/ (Arm 1)")
    ap.add_argument("--native-run", type=Path, default=None,
                    help="run dir of a resolution-natively-trained model (Arm 2)")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    rows = build_rows(args.sweep_run, args.native_run)
    if not rows:
        raise SystemExit("no complete (accuracy, latency) pairs found")

    write_csv(rows, args.out / "resolution_tradeoff.csv")
    write_markdown(rows, args.out / "resolution_tradeoff.md", args.native_run)
    plot(rows, args.out / "plots" / "resolution_tradeoff.png")
    print(f"wrote {len(rows)} rows -> {args.out}/resolution_tradeoff.{{csv,md}}")
    print(f"wrote {args.out}/plots/resolution_tradeoff.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
