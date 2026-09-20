#!/usr/bin/env python3
"""
Stage 6 — Join the device measurements into the reported tables and plots (host).

Reads the raw JSONL the Pi produced, aggregates the three invocations per cell,
applies the thermal validity gate, translates to the QCS605, and writes
everything `4-Results.tex` §`sec:results_ax_visio_benchmark` needs.

Every number it emits carries a path back to a raw artifact, per the
`thesis-writing` skill's claims rules: `latency_raw.jsonl` keeps every
individual per-iteration timing, so any table cell can be recomputed.

Usage:
    uv run python scripts/benchmark/5-report.py
    uv run python scripts/benchmark/5-report.py --no-plots

Outputs (reports/embedded_benchmark/):
    latency_summary.csv, thermal_summary.csv, embedded_benchmark.md,
    plots/embedded_latency_vs_map.png, plots/embedded_latency_by_runtime.png
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import constants as C  # noqa: E402

# Documented, pre-validated categorical slots 1-2 from the dataviz reference
# palette (light surface, adjacent pairlist: worst CVD dE 9.1). Used in fixed
# order and never cycled.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

# Where each model's accuracy comes from. Latency is weight-independent, so a
# model can have a valid latency row and no mAP (an unevaluated checkpoint);
# such models are simply absent from the accuracy-vs-latency plot.
MAP_SOURCES = {
    # Post-Band-A-fix run (mixed 0.599 / real 0.529). The previous entry here
    # pointed at the pre-fix run yolo26n-20260715-010031 (0.523 / 0.481), which
    # plotted stale accuracy against fresh latency — 0.076 mAP too low.
    "yolo26n-direct": "scripts/training/yolo26n/model_exports/yolo26n-bs32-20260910-212812/evaluation/evaluation_report.json",
    # Still the PRE-Band-A-fix KD run — the only completed KD evaluation. Repoint
    # once the post-fix KD run (yolo26n-kd-bs16-20260916-101612, on gpu-server)
    # finishes and its evaluation report lands here.
    "yolo26n-kd": "scripts/training/yolo26n/model_exports/yolo26n-kd-20260825-164250/evaluation/evaluation_report.json",
    "yolov5s": "scripts/training/yolov5s/model_exports/yolov5s-20260909-230900/eval_best/evaluation_report.json",
}


def load_records(path: Path) -> list[dict]:
    recs = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def load_map(model: str) -> dict | None:
    rel = MAP_SOURCES.get(model)
    if not rel:
        return None
    p = REPO_ROOT / rel
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    t1 = d.get("tier1", {})
    return {
        "map_mixed": t1.get("headline_mixed_fine", {}).get("map"),
        "map_real": t1.get("headline_real_fine", {}).get("map"),
        "source": rel,
        "checkpoint": d.get("checkpoint"),
    }


def aggregate(recs: list[dict]) -> list[dict]:
    """Pool the independent invocations of each (model, runtime, threads) cell."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in recs:
        if r.get("sustained"):
            continue
        groups[(r["model"], r["runtime"], r["threads"])].append(r)

    rows = []
    for (model, runtime, threads), rs in sorted(groups.items()):
        w = [r["windows"] for r in rs]
        def med(window: str, field: str = "median_ms") -> float | None:
            vals = [x[window][field] for x in w if window in x]
            return statistics.median(vals) if vals else None

        infer_medians = [x["infer"]["median_ms"] for x in w if "infer" in x]
        valid = [r for r in rs if r["gate"].get("valid")]
        rows.append(
            {
                "model": model,
                "label": rs[0].get("label"),
                "runtime": runtime,
                "runtime_version": rs[0].get("runtime_version"),
                "threads": threads,
                "invocations": len(rs),
                "invocations_valid": len(valid),
                "iters_per_invocation": rs[0]["protocol"]["iters"],
                "infer_median_ms": med("infer"),
                "infer_iqr_ms": med("infer", "iqr_ms"),
                "infer_p95_ms": med("infer", "p95_ms"),
                "infer_min_ms": med("infer", "min_ms"),
                "infer_spread_across_invocations_ms": (
                    round(max(infer_medians) - min(infer_medians), 3)
                    if len(infer_medians) > 1
                    else 0.0
                ),
                "pre_median_ms": med("pre_median"),
                "pre_axvisio_ms": med("pre_axvisio"),
                "post_median_ms": med("post"),
                "e2e_median_ms": med("e2e"),
                "e2e_iqr_ms": med("e2e", "iqr_ms"),
                "fps_infer": round(1000.0 / med("infer"), 3) if med("infer") else None,
                "fps_e2e": round(1000.0 / med("e2e"), 3) if med("e2e") else None,
                "peak_rss_mb": round(statistics.median(r["peak_rss_mb"] for r in rs), 1),
                "model_load_ms": round(statistics.median(r["model_load_ms"] for r in rs), 1),
                "artifact_mb": round((rs[0].get("artifact_bytes") or 0) / 1e6, 2),
                "params_m": round((rs[0].get("params") or 0) / 1e6, 3),
                "gflops": rs[0].get("gflops"),
                "image_size": rs[0].get("image_size"),
                "decode": rs[0].get("decode"),
                "weights_source": rs[0].get("weights_source"),
                "parity_valid": rs[0].get("parity_valid"),
                "all_invocations_valid": len(valid) == len(rs),
                "temp_max_c": max(r["monitor"].get("temp_max_c", 0) for r in rs),
                "freq_median_khz": statistics.median(
                    r["monitor"].get("freq_median_khz", -1) for r in rs
                ),
            }
        )
    return rows


def thermal_rows(recs: list[dict]) -> list[dict]:
    out = []
    for r in recs:
        m, g = r["monitor"], r["gate"]
        row = {
            "cell": r["cell"],
            "model": r["model"],
            "runtime": r["runtime"],
            "threads": r["threads"],
            "invocation": r["invocation"],
            "sustained": r.get("sustained", False),
            "iters": r["protocol"]["iters"],
            "temp_max_c": m.get("temp_max_c"),
            "temp_median_c": m.get("temp_median_c"),
            "freq_median_khz": m.get("freq_median_khz"),
            "freq_min_khz": m.get("freq_min_khz"),
            "throttled_bits": m.get("throttled_bits"),
            "peak_rss_mb": r["peak_rss_mb"],
            "valid": g.get("valid"),
            "reasons": "; ".join(g.get("reasons", [])),
        }
        s = r["windows"].get("infer", {}).get("sustained")
        if s and r["protocol"]["iters"] >= 20:
            row["sustained_drift_pct"] = s.get("drift_pct")
            row["sustained_first_median_ms"] = s.get("first_median_ms")
            row["sustained_last_median_ms"] = s.get("last_median_ms")
        out.append(row)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    cols: list[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)



def parse_failures(log_dir: Path) -> list[dict]:
    """Recover crashed cells from the driver logs.

    A cell that dies takes its record with it, so the JSONL simply lacks it and
    an aggregated table would show a silent blank. The driver logs the model /
    runtime / threads immediately before launching each cell and logs the exit
    code after, so the pairing is recoverable — and a crash is a result about
    the runtime, not an absence of data.
    """
    import re

    failures: list[dict] = []
    for log in sorted(log_dir.glob("*.log")):
        current = None
        for line in log.read_text(errors="replace").splitlines():
            m = re.search(
                r"^\[(?:driver|cell)\]\s+\d\d:\d\d:\d\d\s+(\S+)\s*/\s*(\S+)\s*/\s*"
                r"(\d+)t\s*/\s*inv(\d+)",
                line,
            )
            if m:
                current = {
                    "model": m.group(1),
                    "runtime": m.group(2),
                    "threads": int(m.group(3)),
                    "invocation": int(m.group(4)),
                    # Phase C re-ran these cells with a mitigation flag. Those
                    # failures say something different from the Phase A ones
                    # and must not be pooled with them.
                    "mitigation": "--no-mkldnn" if "--no-mkldnn" in line else "none",
                }
                continue
            f = re.search(r"cell FAILED \(rc=(\d+)\)|^\[cell\] FAILED rc=(\d+)", line)
            if f and current:
                rc = int(f.group(1) or f.group(2))
                sig = rc - 128 if rc > 128 else None
                failures.append(
                    {
                        **current,
                        "exit_code": rc,
                        "signal": sig,
                        "diagnosis": (
                            "SIGILL — illegal instruction"
                            if sig == 4
                            else (f"signal {sig}" if sig else "non-zero exit")
                        ),
                        "log": log.name,
                    }
                )
                current = None
    return failures


# ─── plots ────────────────────────────────────────────────────────────────────

def _ensemble_point(rows: list[dict]) -> tuple[float, float, float] | None:
    """Composed MegaDetector + SpeciesNet per-frame cost.

    The ensemble is never run as a pipeline on the device: its per-frame cost is
    t_MD(1280) + k * t_SN(480), where k is the mean animal crops per image taken
    from the ensemble's own cached predictions. Labelled a derived estimate
    wherever it appears.
    """
    th = C.PROTOCOL["headline_threads"]
    md = next((r for r in rows if r["model"] == "megadetector" and r["threads"] == th
               and r["runtime"] == "onnxruntime"), None)
    sn = next((r for r in rows if r["model"] == "speciesnet" and r["threads"] == th
               and r["runtime"] == "onnxruntime"), None)
    if not md or not sn or not md["e2e_median_ms"] or not sn["infer_median_ms"]:
        return None
    k = _mean_crops_per_image()
    return md["e2e_median_ms"] + k * sn["infer_median_ms"], k, md["e2e_median_ms"]


def _mean_crops_per_image(default: float = 1.0) -> float:
    pred = (REPO_ROOT / "scripts/training/megadet_speciesnet_ensemble/model_exports"
            / "finetuned-teacher-finetune-ff0.75-bs64-20260909-231034/predictions_real.json")
    if not pred.exists():
        return default
    try:
        d = json.loads(pred.read_text())
        preds = d.get("predictions", [])
        imgs = {p["image_id"] for p in preds}
        return max(1.0, len(preds) / max(1, len(imgs)))
    except Exception:  # noqa: BLE001
        return default


def plot_pareto(rows: list[dict], out: Path) -> bool:
    """Accuracy against measured per-frame latency, with the budget marked.

    Identity is carried by direct labels rather than hue: with a handful of
    points a categorical palette cannot clear the all-pairs separation floors,
    and labels are strictly more legible in print. The two colours separate the
    two *test domains* (mixed and real-only), which the evaluation strategy
    requires to be reported together — they are not a model palette.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    th = C.PROTOCOL["headline_threads"]
    pts = []
    for r in rows:
        if r["threads"] != th or r["runtime"] != "onnxruntime":
            continue
        acc = load_map(r["model"])
        if not acc or acc["map_mixed"] is None or not r["e2e_median_ms"]:
            continue
        note = "" if r["parity_valid"] else "*"
        pts.append((r["e2e_median_ms"], acc["map_mixed"], acc["map_real"], r["label"] + note))

    ens = _ensemble_point(rows)
    if ens:
        ens_ms, k, _ = ens
        ens_map = load_map_ensemble()
        if ens_map:
            pts.append((ens_ms, ens_map["map_mixed"], ens_map["map_real"],
                        f"MD+SpeciesNet ensemble\n(derived, $k={k:.1f}$ crops)"))
    if not pts:
        return False
    pts.sort(key=lambda t: t[0])

    fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=200)
    lo, hi = C.PI400_PASS_BAND_MS

    # Budget band first, so marks sit above it.
    ax.axvspan(1, lo, color="#1baf7a", alpha=0.10, zorder=0)
    ax.axvline(lo, color=INK_MUTED, linestyle="--", linewidth=1.2, zorder=1)

    for x, m_mixed, m_real, label in pts:
        # Tie each model's two domain readings together so the pair reads as
        # one entity rather than two unrelated points.
        ax.plot([x, x], [m_real, m_mixed], color=GRID, linewidth=1.2, zorder=2)
        ax.scatter([x], [m_mixed], s=64, color=BLUE, zorder=3,
                   edgecolor="white", linewidth=1.4)
        ax.scatter([x], [m_real], s=64, color=ORANGE, marker="s", zorder=3,
                   edgecolor="white", linewidth=1.4)
        ax.annotate(label, (x, m_mixed), textcoords="offset points", xytext=(0, 13),
                    fontsize=8.5, color=INK, ha="center")

    ax.set_xscale("log")
    ax.set_xlim(lo * 0.45, max(p[0] for p in pts) * 2.6)
    ymin = min(min(p[1], p[2]) for p in pts)
    ymax = max(max(p[1], p[2]) for p in pts)
    pad = max(0.03, (ymax - ymin) * 0.35)
    ax.set_ylim(ymin - pad * 0.5, ymax + pad)

    ax.annotate(
        f"$\\leq{C.TARGET_LATENCY_MS_QCS605:.0f}$ ms QCS605 budget\n"
        f"(Pi 400 pass band $\\leq{lo:.0f}$ ms)",
        xy=(lo, ymax + pad * 0.55), xytext=(4, 0), textcoords="offset points",
        fontsize=8, color=INK_MUTED, ha="left", va="top",
    )
    ax.set_xlabel("End-to-end latency per frame on the Raspberry Pi 400, ms "
                  "(log scale, ONNX Runtime, 4 threads)", fontsize=9.5, color=INK)
    ax.set_ylabel("mAP@[.5:.95]", fontsize=9.5, color=INK)
    ax.scatter([], [], color=BLUE, s=64, label="mixed test set")
    ax.scatter([], [], color=ORANGE, s=64, marker="s", label="real-only breakout")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left", ncol=2)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=8.5)
    if any(p[3].endswith("*") for p in pts):
        fig.text(0.01, -0.02,
                 "* accuracy from a separately evaluated checkpoint of the same "
                 "architecture; latency is weight-independent.",
                 fontsize=7.5, color=INK_MUTED)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return True


def load_map_ensemble() -> dict | None:
    p = (REPO_ROOT / "scripts/training/megadet_speciesnet_ensemble/model_exports"
         / "finetuned-teacher-finetune-ff0.75-bs64-20260909-231034/eval/evaluation_report.json")
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    t1 = d.get("tier1", {})
    return {
        "map_mixed": t1.get("headline_mixed_fine", {}).get("map"),
        "map_real": t1.get("headline_real_fine", {}).get("map"),
    }


def plot_by_runtime(rows: list[dict], out: Path, failures=None) -> bool:
    """Per-runtime inference latency at the headline thread count.

    Linear axis with a zero baseline: a bar encodes magnitude by length, and a
    log axis destroys that proportionality. Runtimes that crashed are drawn as
    an explicit annotation rather than an absent bar, so a gap is never read as
    missing data.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    th = C.PROTOCOL["headline_threads"]
    models = []
    for r in rows:
        if r["threads"] == th and r["model"] not in models:
            models.append(r["model"])
    if not models:
        return False
    runtimes = ["onnxruntime", "torchscript"]
    colors = {"onnxruntime": BLUE, "torchscript": ORANGE}
    lookup = {(r["model"], r["runtime"]): r for r in rows if r["threads"] == th}
    crashed = {
        (f["model"], f["runtime"])
        for f in (failures or [])
        if f["threads"] == th and f.get("mitigation", "none") == "none"
    }

    fig, ax = plt.subplots(figsize=(7.8, 4.4), dpi=200)
    x = np.arange(len(models))
    width = 0.30
    top = max((r["infer_median_ms"] or 0) for r in rows if r["threads"] == th)
    for i, rt in enumerate(runtimes):
        vals, errs = [], []
        for m in models:
            r = lookup.get((m, rt))
            vals.append(r["infer_median_ms"] if r else np.nan)
            errs.append(r["infer_iqr_ms"] if r else 0.0)
        off = (i - 0.5) * (width + 0.03)
        bars = ax.bar(x + off, vals, width, label=rt, color=colors[rt], zorder=3)
        ax.errorbar(x + off, vals, yerr=errs, fmt="none", ecolor=INK_MUTED,
                    elinewidth=1, capsize=3, zorder=4)
        for j, (b, v) in enumerate(zip(bars, vals)):
            if v == v:
                ax.annotate(f"{v:,.0f}", (b.get_x() + b.get_width() / 2, v),
                            textcoords="offset points", xytext=(0, 4), ha="center",
                            fontsize=8, color=INK)
            elif (models[j], rt) in crashed:
                ax.annotate("SIGILL", (x[j] + off, top * 0.03), rotation=90,
                            ha="center", va="bottom", fontsize=7.5, color=INK_MUTED)

    ax.set_xticks(x)
    ax.set_xticklabels([lookup.get((m, "onnxruntime"), {}).get("label", m) for m in models],
                       fontsize=9, color=INK)
    ax.set_ylabel(f"Inference latency $W_{{\\mathrm{{infer}}}}$, ms ({th} threads)",
                  fontsize=9.5, color=INK)
    ax.set_ylim(0, top * 1.18)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=8.5)
    if crashed:
        fig.text(0.01, -0.02,
                 "SIGILL: the cell aborted with an illegal instruction; see "
                 "failed_cells.csv.", fontsize=7.5, color=INK_MUTED)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return True


# ─── LaTeX ────────────────────────────────────────────────────────────────────

def write_latex(rows, out: Path) -> None:
    """Emit a ready-to-paste LaTeX table.

    The manuscript's convention is hardcoded `tabular` blocks, not generated
    `\\input{}` files, so this is written to `reports/` to be pasted rather
    than included — keeping the convention while removing the transcription
    risk of retyping two dozen numbers by hand.
    """
    th = C.PROTOCOL["headline_threads"]
    head = [r for r in rows if r["threads"] == th and r["runtime"] == "onnxruntime"]
    head.sort(key=lambda r: r["infer_median_ms"] or 0)
    lo, _ = C.PI400_PASS_BAND_MS
    L = [
        "% Generated by scripts/benchmark/5-report.py — paste into 4-Results.tex.",
        "% Source artifact: reports/embedded_benchmark/latency_summary.csv",
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\caption[Runtime benchmark on the Raspberry Pi 400]{Measured runtime of every "
        "evaluated model on the \\textit{Raspberry Pi 400} proxy: ONNX Runtime, "
        f"{th} threads, batch size $1$, full precision, \\texttt{{performance}} governor. "
        "$W_{\\mathrm{infer}}$ is the forward pass alone (the \\textit{MLPerf Tiny} "
        "measurement scope); $W_{\\mathrm{e2e}}$ adds JPEG decode, letterboxing and "
        "post-processing. Median of three independent process invocations, interquartile "
        "range in parentheses.}",
        "  \\label{tab:embedded-latency}",
        "  \\begin{tabular}{lrrrrrr}",
        "    \\toprule",
        "    Model & Params & GFLOPs & $W_{\\mathrm{infer}}$ & $W_{\\mathrm{e2e}}$ & "
        "FPS & Peak RSS \\\\",
        "     & (M) & & (ms) & (ms) & & (MB) \\\\",
        "    \\midrule",
    ]
    for r in head:
        label = (r["label"] or r["model"]).replace("&", "\\&")
        infer = (f"${r['infer_median_ms']:,.0f}$ ({r['infer_iqr_ms']:,.0f})"
                 if r["infer_median_ms"] else "--")
        e2e = f"${r['e2e_median_ms']:,.0f}$" if r["e2e_median_ms"] else "--"
        fps = f"${r['fps_e2e']:,.2f}$" if r["fps_e2e"] else "--"
        L.append(
            f"    \\textit{{{label}}} & ${r['params_m']:,.2f}$ & ${r['gflops']:,.1f}$ & "
            f"{infer} & {e2e} & {fps} & ${r['peak_rss_mb']:,.0f}$ \\\\"
        )
    L += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
        "",
        f"% Pass band: a Pi 400 measurement at or below {lo:.0f} ms implies "
        f"<= {C.TARGET_LATENCY_MS_QCS605:.0f} ms on the QCS605 CPU",
        f"% (QCS605 ~ Pi400 x {C.QCS605_SCALE[0]:.2f}-{C.QCS605_SCALE[1]:.2f}; "
        "docs/2026-03-09_hardware-proxy-selection.md). CPU path only.",
    ]
    out.write_text("\n".join(L) + "\n")


# ─── markdown ─────────────────────────────────────────────────────────────────

def fmt(v, nd=1, dash="—"):
    return dash if v is None else f"{v:,.{nd}f}"


def write_markdown(rows, therm, parity, device, out: Path, failures=None) -> None:
    th = C.PROTOCOL["headline_threads"]
    lo, hi = C.PI400_PASS_BAND_MS
    head = [r for r in rows if r["threads"] == th and r["runtime"] == "onnxruntime"]
    L = []
    a = L.append

    a("# Embedded benchmark — Raspberry Pi 400 as QCS605 proxy\n")
    a("Generated by `scripts/benchmark/5-report.py`. Every figure here is "
      "recomputable from `latency_raw.jsonl`, which keeps each individual "
      "per-iteration timing.\n")

    a("## Device\n")
    if device:
        for k, v in device.items():
            a(f"- **{k}**: {v}")
    a("")

    a("## Protocol\n")
    a(f"- Batch size {C.PROTOCOL['batch_size']}; core affinity pinned with `taskset`; "
      f"thread count passed explicitly to each runtime; `performance` governor.")
    a(f"- {C.PROTOCOL['invocations']} independent process invocations per cell; "
      f"the headline is the median of the pooled runs, and the spread across "
      f"invocations is reported so process-level variance stays visible.")
    a(f"- Iteration counts adapt to a per-window wall-clock budget (nominal "
      f"{C.PROTOCOL['warmup_iters']} warm-up / {C.PROTOCOL['measure_iters']} measured); "
      f"the realised `n` is in `latency_summary.csv`.")
    a(f"- Deployment regime: conf {C.REGIMES['deployment']['conf_thres']}, "
      f"IoU {C.REGIMES['deployment']['iou_thres']}, max_det "
      f"{C.REGIMES['deployment']['max_det']}.")
    a("- `W_infer` is the MLPerf-Tiny-scoped window (forward call only). "
      "`W_e2e` adds decode, letterbox and post-processing — it is the number "
      "that answers the per-frame budget question.\n")

    a(f"## Headline — ONNX Runtime, {th} threads\n")
    a("| Model | Params (M) | GFLOPs | Artifact (MB) | W_infer (ms) | IQR | W_e2e (ms) | FPS (e2e) | Peak RSS (MB) |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in head:
        a(f"| {r['label']} | {fmt(r['params_m'], 2)} | {fmt(r['gflops'], 1)} | "
          f"{fmt(r['artifact_mb'], 1)} | **{fmt(r['infer_median_ms'])}** | "
          f"{fmt(r['infer_iqr_ms'])} | {fmt(r['e2e_median_ms'])} | "
          f"{fmt(r['fps_e2e'], 2)} | {fmt(r['peak_rss_mb'])} |")
    a("")

    a("## Verdict against the design targets\n")
    a(f"Targets are **project design estimates, not vendor-certified figures**: "
      f"≤{C.TARGET_LATENCY_MS_QCS605:.0f} ms per frame at 640×640 and "
      f"≤{C.TARGET_PEAK_RSS_MB:.0f} MB, from "
      f"`docs/2026-03-10_object-detection-models-for-embedded-systems.md` §1.\n")
    a(f"`docs/2026-03-09_hardware-proxy-selection.md` rates the Pi 4/400 at "
      f"−10 % CPU versus the QCS605, so QCS605 ≈ Pi 400 × "
      f"{C.QCS605_SCALE[0]:.2f}–{C.QCS605_SCALE[1]:.2f}, i.e. a Pi 400 measurement "
      f"of **≤{lo:.0f}–{hi:.0f} ms** would imply ≤{C.TARGET_LATENCY_MS_QCS605:.0f} ms "
      f"on the QCS605 CPU. **This applies to the CPU path only** — the Pi 400 has "
      f"no analogue to the Hexagon 685 DSP and its VideoCore VI is far weaker "
      f"than the Adreno 615.\n")
    a("| Model | W_e2e Pi 400 (ms) | Implied QCS605 CPU (ms) | ≤30 ms? | Peak RSS (MB) | ≤500 MB? |")
    a("|---|---:|---:|:--:|---:|:--:|")
    for r in head:
        e2e = r["e2e_median_ms"]
        if e2e is None:
            continue
        q = f"{e2e * C.QCS605_SCALE[0]:,.0f}–{e2e * C.QCS605_SCALE[1]:,.0f}"
        ok = "yes" if e2e <= lo else "**no**"
        rss_ok = "yes" if r["peak_rss_mb"] <= C.TARGET_PEAK_RSS_MB else "**no**"
        a(f"| {r['label']} | {fmt(e2e)} | {q} | {ok} | {fmt(r['peak_rss_mb'])} | {rss_ok} |")
    a("")

    a("## Latency decomposition (ONNX Runtime, "
      f"{th} threads)\n")
    a("| Model | W_pre median img (ms) | W_pre AX Visio 4192×3120 (ms) | W_infer (ms) | W_post (ms) | W_e2e (ms) |")
    a("|---|---:|---:|---:|---:|---:|")
    for r in head:
        a(f"| {r['label']} | {fmt(r['pre_median_ms'], 2)} | {fmt(r['pre_axvisio_ms'], 1)} | "
          f"{fmt(r['infer_median_ms'])} | {fmt(r['post_median_ms'], 2)} | "
          f"{fmt(r['e2e_median_ms'])} |")
    a("")

    a("## Thread scaling\n")
    a("The 2-thread column is the more transferable figure: the QCS605 has 2 big "
      "A75 cores plus 6 little A55s, so a 4×A72 number and a 2-thread number "
      "bracket it from different directions.\n")
    a("| Model | Runtime | 1 thread | 2 threads | 4 threads | 1→4 speedup |")
    a("|---|---|---:|---:|---:|---:|")
    by = {(r["model"], r["runtime"], r["threads"]): r for r in rows}
    seen = []
    for r in rows:
        key = (r["model"], r["runtime"])
        if key in seen:
            continue
        seen.append(key)
        v = {t: by.get((r["model"], r["runtime"], t), {}).get("infer_median_ms") for t in (1, 2, 4)}
        sp = f"{v[1] / v[4]:.2f}×" if v[1] and v[4] else "—"
        a(f"| {r['label']} | {r['runtime']} | {fmt(v[1])} | {fmt(v[2])} | {fmt(v[4])} | {sp} |")
    a("")

    a("## Thermal validity\n")
    invalid = [t for t in therm if t["valid"] is False]
    a(f"- Cells measured: **{len(therm)}**; failing the validity gate: "
      f"**{len(invalid)}**.")
    if therm:
        a(f"- Peak temperature across the whole sweep: "
          f"**{max((t['temp_max_c'] or 0) for t in therm):.1f} °C** "
          f"(gate trips at {C.THERMAL_GATE['max_temp_c']:.0f} °C).")
    for t in therm:
        if t.get("sustained_drift_pct") is not None and (t.get("iters") or 0) >= 20:
            a(f"- Sustained run, {t['model']}: {t['sustained_drift_pct']:+.2f} % drift "
              f"over {t['iters']} iterations "
              f"({t['sustained_first_median_ms']:,.1f} → {t['sustained_last_median_ms']:,.1f} ms), "
              f"peak {t['temp_max_c']:.1f} °C.")
    if invalid:
        a("\n  Invalid cells (numbers retained but flagged):")
        for t in invalid:
            a(f"  - `{t['cell']}` — {t['reasons']}")
    a("")

    if parity:
        a("## Parity — did the model survive export and a different ISA?\n")
        a("A numerical-equivalence check, not a re-evaluation. T3 interprets only "
          "the *delta*: a 600-image subset cannot produce a meaningful absolute "
          "225-class mAP and is not used as one.\n")
        a("| Model | Side | T1 max abs | T2 match rate | T3 subset mAP | Δ mAP |")
        a("|---|---|---:|---:|---:|---:|")
        for p in parity:
            if p.get("skipped"):
                a(f"| {p['model']} | — | _skipped: {p['skipped']}_ | | | |")
                continue
            for c in p["comparisons"]:
                t1 = c.get("t1", {})
                t1s = f"{t1['max_abs']:.2e}" if t1.get("available") else "—"
                a(f"| {p['model']} | {c['side']} | {t1s} | "
                  f"{c['t2']['match_rate'] * 100:.2f} % | {c['map']:.4f} | "
                  f"{c['map_delta']:+.4f} |")
        a("")

    if failures:
        a("## Cells that crashed\n")
        a("A crashed cell is a result about the runtime, not missing data. Recovered "
          "from the driver logs; full list in `failed_cells.csv`.\n")
        for mitigation, title, note in (
            ("none", "Under the default configuration",
             "These cells have no measurement. Every one is TorchScript on YOLO26n: "
             "the same runtime completed normally on YOLOv5s, SpeciesNet and "
             "MegaDetector at the same thread counts, so the fault is specific to "
             "that architecture's graph rather than to multi-threading as such."),
            ("--no-mkldnn", "Under the `--no-mkldnn` mitigation attempt",
             "Disabling PyTorch's oneDNN backend was tried as a fix and made things "
             "strictly worse: with it, **every** model's multi-threaded TorchScript "
             "cell aborts, including the three that succeed with oneDNN enabled. "
             "The mitigation was therefore rejected and the Phase A numbers stand."),
        ):
            fs_all = [f for f in failures if f.get("mitigation", "none") == mitigation]
            if not fs_all:
                continue
            a(f"**{title}.** {note}\n")
            by_rt: dict[tuple, list[dict]] = defaultdict(list)
            for f in fs_all:
                by_rt[(f["model"], f["runtime"], f["threads"], f["diagnosis"])].append(f)
            a("| Model | Runtime | Threads | Cells | Exit | Diagnosis |")
            a("|---|---|---:|---:|---:|---|")
            for (model, rt, th, diag), fs in sorted(by_rt.items()):
                a(f"| {model} | {rt} | {th} | {len(fs)} | {fs[0]['exit_code']} | {diag} |")
            a("")

    a("## Non-claims\n")
    a("- **FP32, unoptimized.** No PTQ, QAT or pruning was applied. ONNX Runtime "
      "applies its own graph optimizations at session creation (`ORT_ENABLE_ALL`), "
      "which is recorded per cell; the shipped ONNX graph is the unmodified export "
      "(onnxslim was run for inspection only and is not shipped).")
    a("- **A bounded measurement campaign on one device**, not a benchmarking study.")
    a("- **No power or thermal-envelope measurement.** The Pi has no instrumented "
      "rail, and no power budget for the AX Visio is documented anywhere in this repo.")
    a("- **The QCS605's accelerated path is entirely unproxied.** The Hexagon 685 "
      "DSP (~2.1 TOPS) has no analogue on this device. The translation rule above "
      "covers the CPU path only.")
    a("- **No GPU number.** See `ncnn_conversion_finding.md`: the device's Vulkan "
      "stack works (V3D 4.2.14.0, Vulkan 1.3.354, Mesa 26.2.2) and the `ncnn` "
      "wheel ships with Vulkan compiled in, but `pnnx` crashes converting both "
      "detectors, so no NCNN model could be produced.")
    a("- Models exported without trained weights are valid for latency only and "
      "are excluded from parity; `latency_summary.csv` carries `weights_source` "
      "and `parity_valid` per row.")
    out.write_text("\n".join(L) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=C.REPORT_DIR / "pi_results" / "latency_raw.jsonl")
    ap.add_argument("--parity", type=Path, default=C.REPORT_DIR / "parity_summary.json")
    ap.add_argument("--device", type=Path, default=C.REPORT_DIR / "device_profile.json")
    ap.add_argument("--out", type=Path, default=C.REPORT_DIR)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    if not args.raw.exists():
        print(f"no raw results at {args.raw} — run the sweep on the Pi first", file=sys.stderr)
        return 2
    recs = load_records(args.raw)
    print(f"{len(recs)} raw cell records")

    rows = aggregate(recs)
    therm = thermal_rows(recs)
    parity = json.loads(args.parity.read_text()) if args.parity.exists() else []
    failures = parse_failures(args.raw.parent)
    if failures:
        print(f"{len(failures)} crashed cell(s) recovered from the driver logs")
    device = json.loads(args.device.read_text()) if args.device.exists() else {}

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "latency_summary.csv", rows)
    write_csv(args.out / "thermal_summary.csv", therm)
    if failures:
        write_csv(args.out / "failed_cells.csv", failures)
    write_markdown(rows, therm, parity, device, args.out / "embedded_benchmark.md",
                   failures=failures)
    write_latex(rows, args.out / "embedded_latency_table.tex")
    print(f"wrote latency_summary.csv ({len(rows)} cells), thermal_summary.csv, "
          f"embedded_benchmark.md, embedded_latency_table.tex")

    if not args.no_plots:
        p1 = args.out / "plots" / "embedded_latency_vs_map.png"
        p2 = args.out / "plots" / "embedded_latency_by_runtime.png"
        if plot_pareto(rows, p1):
            print(f"wrote {p1.relative_to(REPO_ROOT)}")
        else:
            print("  ! Pareto plot skipped (no model has both a latency row and an mAP)")
        if plot_by_runtime(rows, p2, failures):
            print(f"wrote {p2.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
