"""Assemble the Tier 1/2/3 evaluation tables (strategy doc §9) and emit them.

This module is the *reporting glue*: it takes the cached predictions + GT indices
(real / synthetic / mixed) and the label remaps, drives :mod:`scoring` to produce
every slice the strategy doc asks for, and writes the result to Markdown + CSV +
JSON (and, optionally, MLflow).

Nothing here runs the model — it consumes what :mod:`predict` and :mod:`scoring`
provide. See ``docs/plans/2026-06-10_model-evaluation-strategy.md``.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np

from scripts.training.yolov5s.eval_suite import scoring

logger = logging.getLogger(__name__)

# The COCO-12 metric vector, in report order.
COCO12 = [
    "map", "map_50", "map_75",
    "map_small", "map_medium", "map_large",
    "mar_1", "mar_10", "mar_100",
    "mar_small", "mar_medium", "mar_large",
]
BANDS = ["A", "B", "C", "D"]
TEST_LIMITED_THRESHOLD = 30  # <30 real test images → flagged (strategy §8)


# ───────────────────────────────────────────────────────────────────────────────
# Small helpers
# ───────────────────────────────────────────────────────────────────────────────

def _fmt(v: float | None) -> str:
    """Format a metric for Markdown: 3 dp, or '—' for NaN/None."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:.3f}"


def _na(v: float) -> float:
    """Map torchmetrics' -1 'not-applicable' sentinel (e.g. an empty area split)
    to NaN so it renders as '—'. Applied ONLY to absolute metrics, never deltas
    (which may legitimately be negative)."""
    return float("nan") if (isinstance(v, (int, float)) and v == -1.0) else v


def _coco12(result: dict) -> dict[str, float]:
    """Extract the COCO-12 vector from a score() result, sentinel-cleaned."""
    return {k: _na(result[k]) for k in COCO12}


def real_images_per_class(real_gt: dict) -> dict[int, int]:
    """Count distinct real test images that contain each fine category (strategy §8).

    Each image is counted once per category it carries a GT box for.
    """
    counts: dict[int, int] = {}
    for iid, anns in real_gt["anns"].items():
        cats_here = {a["category_id"] for a in anns}
        for c in cats_here:
            counts[c] = counts.get(c, 0) + 1
    return counts


def _is_valid_ap(ap: float) -> bool:
    """True for a real AP. torchmetrics uses -1 as a 'class not present' sentinel
    and NaN for degenerate cells — both must be excluded from any mean."""
    return not (isinstance(ap, float) and math.isnan(ap)) and ap >= 0.0


def macro_mean(per_class: dict[int, float], exclude: set[int] | None = None) -> float:
    """Unweighted mean of per-class AP (COCO macro), optionally excluding classes.

    Skips torchmetrics' -1 'not-present' sentinel and NaN cells so they do not
    poison the mean (which would otherwise drag it negative)."""
    exclude = exclude or set()
    vals = [
        ap for cid, ap in per_class.items()
        if cid not in exclude and _is_valid_ap(ap)
    ]
    return float(np.mean(vals)) if vals else float("nan")


def count_weighted_mean(per_class: dict[int, float], gt_counts: dict[int, int]) -> float:
    """GT-count-weighted mean of per-class AP (strategy §8 sensitivity check)."""
    num = 0.0
    den = 0.0
    for cid, ap in per_class.items():
        if not _is_valid_ap(ap):
            continue
        w = gt_counts.get(cid, 0)
        num += ap * w
        den += w
    return num / den if den > 0 else float("nan")


def gt_counts_per_class(gt_index: dict, image_ids: set[int] | None = None) -> dict[int, int]:
    """Number of GT boxes per fine category over the given image set."""
    counts: dict[int, int] = {}
    ids = image_ids if image_ids is not None else set(gt_index["images"].keys())
    for iid in ids:
        for a in gt_index["anns"].get(iid, []):
            counts[a["category_id"]] = counts.get(a["category_id"], 0) + 1
    return counts


def granularity_scores(
    gt: dict,
    preds: list[dict],
    remaps: dict[str, dict[int, int]],
    image_ids: set[int] | None,
    max_det: int,
    class_metrics: bool = False,
    precomputed: dict[str, dict] | None = None,
) -> dict[str, dict]:
    """Score the same (gt, preds, image set) at detect / coarse / fine granularity.

    *precomputed* lets a caller supply an already-computed result for a level
    (e.g. Tier 1's headline_mixed is the exact same (gt, preds, all-images,
    fine-remap) computation as this function's "fine" level) so it isn't
    scored a second time — each of these calls stratifies per class and is
    expensive at full test-set scale.
    """
    out: dict[str, dict] = dict(precomputed or {})
    for level in ("detect", "coarse", "fine"):
        if level in out:
            continue
        out[level] = scoring.score(
            gt, preds, image_ids=image_ids, remap=remaps[level],
            max_det=max_det, class_metrics=class_metrics,
        )
    return out


# ───────────────────────────────────────────────────────────────────────────────
# Report assembly
# ───────────────────────────────────────────────────────────────────────────────

def build_full_report(
    *,
    real_gt: dict,
    real_preds: list[dict],
    synth_gt: dict | None,
    synth_preds: list[dict] | None,
    mixed_gt: dict,
    mixed_preds: list[dict],
    remaps: dict[str, dict[int, int]],
    band_by_id: dict[int, str],
    lookalike_gids: list[int],
    group_labels: dict[int, str],
    cat_id_to_name: dict[int, str],
    max_det: int,
    bootstrap_n: int = 0,
    checkpoint: str = "",
) -> dict:
    """Compute every tier of the evaluation. Returns a nested, JSON-serialisable dict.

    ``synth_gt``/``synth_preds`` may be ``None`` — in that case the mixed domain
    equals real, the synthetic-only and domain-shift sections are skipped, and a
    note is recorded.
    """
    has_synth = synth_gt is not None and synth_preds is not None
    report: dict = {
        "checkpoint": checkpoint,
        "config": {"max_det": max_det, "bootstrap_n": bootstrap_n, "has_synthetic": has_synth},
        "notes": [],
    }
    if not has_synth:
        report["notes"].append(
            "Synthetic test set was not supplied; 'mixed' == 'real', and the "
            "domain-shift / synthetic-only sections are omitted."
        )

    real_img_counts = real_images_per_class(real_gt)
    test_limited = {
        cid for cid, n in real_img_counts.items() if n < TEST_LIMITED_THRESHOLD
    }
    report["test_limited_classes"] = {
        cat_id_to_name.get(cid, str(cid)): real_img_counts[cid]
        for cid in sorted(test_limited)
    }

    # ── TIER 1 — Headline ──────────────────────────────────────────────────────
    logger.info("Tier 1: headline (fine/mixed/all) + real breakout + detect analog")
    headline_mixed = scoring.score(mixed_gt, mixed_preds, remap=remaps["fine"],
                                   max_det=max_det, class_metrics=True)
    headline_real = scoring.score(real_gt, real_preds, remap=remaps["fine"],
                                  max_det=max_det, class_metrics=True)
    detect_mixed = scoring.score(mixed_gt, mixed_preds, remap=remaps["detect"],
                                 max_det=max_det, class_metrics=False)
    detect_real = scoring.score(real_gt, real_preds, remap=remaps["detect"],
                                max_det=max_det, class_metrics=False)

    mixed_gt_counts = gt_counts_per_class(mixed_gt)
    real_gt_counts = gt_counts_per_class(real_gt)

    report["tier1"] = {
        "headline_mixed_fine": _coco12(headline_mixed),
        "headline_real_fine": _coco12(headline_real),
        "detect_analog": {"mixed_map": detect_mixed["map"], "mixed_map_50": detect_mixed["map_50"],
                          "real_map": detect_real["map"], "real_map_50": detect_real["map_50"]},
        "headline_mixed_map_excl_test_limited": macro_mean(
            headline_mixed["map_per_class"], exclude=test_limited),
        "headline_mixed_map_count_weighted": count_weighted_mean(
            headline_mixed["map_per_class"], mixed_gt_counts),
        "headline_real_map_count_weighted": count_weighted_mean(
            headline_real["map_per_class"], real_gt_counts),
        "n_test_limited_classes": len(test_limited),
    }

    if bootstrap_n > 0:
        logger.info("Tier 1: bootstrap CI on headline mixed mAP (n=%d)", bootstrap_n)
        report["tier1"]["headline_mixed_map_ci"] = scoring.bootstrap_ci(
            mixed_gt, mixed_preds, image_ids=set(mixed_gt["images"].keys()),
            remap=remaps["fine"], max_det=max_det, n_boot=bootstrap_n, metric_key="map")

    # ── TIER 2.1 — Granularity gap decomposition (mixed, all) ──────────────────
    logger.info("Tier 2.1: granularity gap (detect/coarse/fine, mixed, all)")
    gran = granularity_scores(mixed_gt, mixed_preds, remaps, image_ids=None,
                              max_det=max_det, class_metrics=False,
                              precomputed={"fine": headline_mixed})
    d_map, c_map, f_map = gran["detect"]["map"], gran["coarse"]["map"], gran["fine"]["map"]
    d_50, c_50, f_50 = gran["detect"]["map_50"], gran["coarse"]["map_50"], gran["fine"]["map_50"]
    report["tier2_granularity"] = {
        "detect": {"map": d_map, "map_50": d_50},
        "coarse": {"map": c_map, "map_50": c_50},
        "fine": {"map": f_map, "map_50": f_50},
        "delta_coarse_map": d_map - c_map,   # cost of coarse ID (cross-group)
        "delta_fine_map": c_map - f_map,     # cost of fine look-alike ID
        "delta_coarse_map_50": d_50 - c_50,
        "delta_fine_map_50": c_50 - f_50,
    }

    # ── TIER 2.2 — Band × granularity grid (mixed + real breakout) ─────────────
    logger.info("Tier 2.2: band × granularity grid (mixed + real)")
    band_grid: dict[str, dict] = {}
    for domain_name, gt, preds in (("mixed", mixed_gt, mixed_preds), ("real", real_gt, real_preds)):
        band_grid[domain_name] = {}
        for band in BANDS:
            ids = scoring.filter_image_ids_by_band(gt, {band})
            if not ids:
                band_grid[domain_name][band] = None
                continue
            fine_s = scoring.score(gt, preds, image_ids=ids, remap=remaps["fine"],
                                   max_det=max_det, class_metrics=False)
            coarse_s = scoring.score(gt, preds, image_ids=ids, remap=remaps["coarse"],
                                     max_det=max_det, class_metrics=False)
            band_grid[domain_name][band] = {
                "n_images": len(ids),
                "fine_map": fine_s["map"], "fine_map_50": fine_s["map_50"],
                "coarse_map": coarse_s["map"], "coarse_map_50": coarse_s["map_50"],
            }
    report["tier2_band_grid"] = band_grid

    # ── TIER 2.3 — Domain-shift delta + within-group confusion ─────────────────
    if has_synth:
        logger.info("Tier 2.3: domain-shift delta (fine + coarse) + within-group confusion")
        delta_fine = scoring.domain_shift_delta(
            real_gt, real_preds, synth_gt, synth_preds, remap=remaps["fine"],
            max_det=max_det, class_to_band=band_by_id)
        delta_coarse = scoring.domain_shift_delta(
            real_gt, real_preds, synth_gt, synth_preds, remap=remaps["coarse"],
            max_det=max_det, class_to_band=None)
        report["tier2_domain_shift"] = {
            "fine_mean_delta": delta_fine["mean_delta"],
            "fine_by_band": delta_fine["by_band"],
            "coarse_mean_delta": delta_coarse["mean_delta"],
            "fine_per_class": delta_fine["per_class"],
        }

    # within-group confusion runs on the mixed set (uses fine labels + coarse remap)
    confusion = scoring.within_group_confusion(
        mixed_gt, mixed_preds, coarse_remap=remaps["coarse"],
        lookalike_group_ids=lookalike_gids, image_ids=None)
    report["tier2_within_group_confusion"] = {
        "overall_confusion_rate": confusion["overall_confusion_rate"],
        "by_group": {
            group_labels.get(gid, str(gid)): stats
            for gid, stats in confusion["by_group"].items()
        },
        "pairs": [
            {"true": cat_id_to_name.get(t, str(t)), "pred": cat_id_to_name.get(p, str(p)), "count": n}
            for (t, p), n in sorted(confusion["pairs"].items(), key=lambda kv: -kv[1])
        ],
    }

    # ── TIER 3 — Appendix artifacts ────────────────────────────────────────────
    logger.info("Tier 3: per-class table + per-band COCO-12")
    per_class_rows = []
    for cid in sorted(cat_id_to_name):
        per_class_rows.append({
            "coco_id": cid,
            "class_name": cat_id_to_name[cid],
            "band": band_by_id.get(cid, "?"),
            "real_test_images": real_img_counts.get(cid, 0),
            "test_limited": cid in test_limited,
            "ap_mixed_fine": _na(headline_mixed["map_per_class"].get(cid, float("nan"))),
            "ap_real_fine": _na(headline_real["map_per_class"].get(cid, float("nan"))),
        })
    report["tier3_per_class"] = per_class_rows

    # full COCO-12 per band cell (mixed domain)
    band_coco12: dict[str, dict] = {}
    for band in BANDS:
        ids = scoring.filter_image_ids_by_band(mixed_gt, {band})
        if not ids:
            band_coco12[band] = None
            continue
        s = scoring.score(mixed_gt, mixed_preds, image_ids=ids, remap=remaps["fine"],
                          max_det=max_det, class_metrics=False)
        band_coco12[band] = _coco12(s)
    report["tier3_band_coco12"] = band_coco12

    return report


# ───────────────────────────────────────────────────────────────────────────────
# Writers
# ───────────────────────────────────────────────────────────────────────────────

def write_markdown(report: dict, path: Path) -> None:
    """Render Tiers 1–2 as a human-readable Markdown report."""
    L: list[str] = []
    L.append("# Model Evaluation Report\n")
    L.append(f"**Checkpoint:** `{report.get('checkpoint','')}`  ")
    cfg = report["config"]
    L.append(f"**max_det:** {cfg['max_det']} · **synthetic:** {cfg['has_synthetic']}\n")
    for note in report.get("notes", []):
        L.append(f"> ⚠️ {note}\n")

    # Tier 1
    t1 = report["tier1"]
    L.append("\n## Tier 1 — Headline\n")
    L.append("Default cell `G=fine · D=mixed · B=all` (cross-model ranking number), "
             "with the `D=real` breakout (public-comparison anchor) alongside.\n")
    L.append("| Metric | mixed (headline) | real (breakout) |")
    L.append("|--------|------------------|-----------------|")
    for k in COCO12:
        L.append(f"| {k} | {_fmt(t1['headline_mixed_fine'][k])} | {_fmt(t1['headline_real_fine'][k])} |")
    da = t1["detect_analog"]
    L.append("\n**Public-comparison analog** — class-agnostic `mAP_detect`: "
             f"mixed {_fmt(da['mixed_map'])} (mAP50 {_fmt(da['mixed_map_50'])}), "
             f"real {_fmt(da['real_map'])} (mAP50 {_fmt(da['real_map_50'])}).\n")
    L.append(f"\n**Statistical hygiene** — headline mixed mAP "
             f"{_fmt(t1['headline_mixed_fine']['map'])} "
             f"→ {_fmt(t1['headline_mixed_map_excl_test_limited'])} excluding the "
             f"{t1['n_test_limited_classes']} test-limited (<{TEST_LIMITED_THRESHOLD} real img) classes. "
             f"Count-weighted (micro) mixed mAP: {_fmt(t1['headline_mixed_map_count_weighted'])}.")
    if "headline_mixed_map_ci" in t1:
        ci = t1["headline_mixed_map_ci"]
        L.append(f"\n**Bootstrap 95% CI** (mixed mAP): {_fmt(ci['mean'])} "
                 f"[{_fmt(ci['lo'])}, {_fmt(ci['hi'])}].")

    # Tier 2.1
    g = report["tier2_granularity"]
    L.append("\n## Tier 2.1 — Granularity gap decomposition (mixed, all classes)\n")
    L.append("| Level | mAP | mAP50 |")
    L.append("|-------|-----|-------|")
    L.append(f"| detect (localisation only) | {_fmt(g['detect']['map'])} | {_fmt(g['detect']['map_50'])} |")
    L.append(f"| coarse (look-alikes merged) | {_fmt(g['coarse']['map'])} | {_fmt(g['coarse']['map_50'])} |")
    L.append(f"| fine (full 225-way) | {_fmt(g['fine']['map'])} | {_fmt(g['fine']['map_50'])} |")
    L.append(f"\nΔ_coarse (cross-group cost) = {_fmt(g['delta_coarse_map'])} · "
             f"Δ_fine (look-alike cost) = {_fmt(g['delta_fine_map'])} (mAP).")

    # Tier 2.2
    L.append("\n## Tier 2.2 — Band × granularity grid\n")
    for domain_name in ("mixed", "real"):
        L.append(f"\n**Domain: {domain_name}**\n")
        L.append("| Band | n_img | mAP_fine | mAP50_fine | mAP_coarse | mAP50_coarse |")
        L.append("|------|-------|----------|------------|------------|--------------|")
        for band in BANDS:
            cell = report["tier2_band_grid"][domain_name].get(band)
            if cell is None:
                L.append(f"| {band} | 0 | — | — | — | — |")
            else:
                L.append(f"| {band} | {cell['n_images']} | {_fmt(cell['fine_map'])} | "
                         f"{_fmt(cell['fine_map_50'])} | {_fmt(cell['coarse_map'])} | "
                         f"{_fmt(cell['coarse_map_50'])} |")

    # Tier 2.3
    if "tier2_domain_shift" in report:
        ds = report["tier2_domain_shift"]
        L.append("\n## Tier 2.3 — Domain shift (real − synthetic), fine granularity\n")
        L.append(f"Mean paired Δ (fine): {_fmt(ds['fine_mean_delta'])} · "
                 f"coarse: {_fmt(ds['coarse_mean_delta'])}\n")
        L.append("| Band | mean Δ (real − synth), fine |")
        L.append("|------|------------------------------|")
        for band in BANDS:
            L.append(f"| {band} | {_fmt(ds['fine_by_band'].get(band))} |")
        L.append("\n> Watchdog (strategy §3.1): a large/systematic real−synth gap is the "
                 "signal to revise the `mixed` default.")

    wgc = report["tier2_within_group_confusion"]
    L.append("\n## Tier 2.3b — Within look-alike group confusion\n")
    L.append(f"Overall within-group fine-confusion rate: {_fmt(wgc['overall_confusion_rate'])}\n")
    L.append("| Look-alike group | matched | confused | rate |")
    L.append("|------------------|---------|----------|------|")
    for label, st in sorted(wgc["by_group"].items(), key=lambda kv: -(kv[1]["matched"])):
        L.append(f"| {label} | {st['matched']} | {st['confused']} | {_fmt(st['confusion_rate'])} |")

    L.append("\n---\n*Per-class (225-row) table, per-band COCO-12 vectors and the "
             "confusion pairs are emitted as CSV/JSON artifacts alongside this file.*\n")

    path.write_text("\n".join(L), encoding="utf-8")
    logger.info("wrote markdown report: %s", path)


def write_csv_artifacts(report: dict, out_dir: Path) -> None:
    """Emit the Tier 3 machine-readable CSVs."""
    import csv

    # per-class table
    pc_path = out_dir / "eval_per_class.csv"
    with pc_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "coco_id", "class_name", "band", "real_test_images", "test_limited",
            "ap_mixed_fine", "ap_real_fine"])
        w.writeheader()
        for row in report["tier3_per_class"]:
            w.writerow(row)

    # band grid
    bg_path = out_dir / "eval_band_grid.csv"
    with bg_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["domain", "band", "n_images", "fine_map", "fine_map_50",
                    "coarse_map", "coarse_map_50"])
        for domain_name in ("mixed", "real"):
            for band in BANDS:
                cell = report["tier2_band_grid"][domain_name].get(band)
                if cell is None:
                    w.writerow([domain_name, band, 0, "", "", "", ""])
                else:
                    w.writerow([domain_name, band, cell["n_images"], cell["fine_map"],
                                cell["fine_map_50"], cell["coarse_map"], cell["coarse_map_50"]])

    # confusion pairs
    cp_path = out_dir / "eval_confusion_pairs.csv"
    with cp_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["true_species", "pred_species", "count"])
        for row in report["tier2_within_group_confusion"]["pairs"]:
            w.writerow([row["true"], row["pred"], row["count"]])

    logger.info("wrote CSV artifacts: %s, %s, %s", pc_path, bg_path, cp_path)


def write_json(report: dict, path: Path) -> None:
    """Dump the entire report dict (all tiers) as JSON."""
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info("wrote JSON report: %s", path)


def log_to_mlflow(report: dict, prefix: str = "eval") -> None:
    """Log headline scalars as MLflow metrics and the report files as artifacts.

    Safe to call only when an MLflow run is active; the caller decides.
    """
    import mlflow

    def _log(key: str, v) -> None:
        """Log only finite values — MLflow rejects/garbles NaN/inf, and a single
        bad value should not abort the whole logging pass."""
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            return
        mlflow.log_metric(key, float(v))

    t1 = report["tier1"]
    for k in COCO12:
        _log(f"{prefix}/mixed_fine/{k}", t1["headline_mixed_fine"][k])
        _log(f"{prefix}/real_fine/{k}", t1["headline_real_fine"][k])
    da = t1["detect_analog"]
    _log(f"{prefix}/detect/mixed_map", da["mixed_map"])
    _log(f"{prefix}/detect/real_map", da["real_map"])
    _log(f"{prefix}/mixed_fine/map_excl_test_limited", t1["headline_mixed_map_excl_test_limited"])
    _log(f"{prefix}/mixed_fine/map_count_weighted", t1["headline_mixed_map_count_weighted"])
    g = report["tier2_granularity"]
    _log(f"{prefix}/gran/delta_coarse_map", g["delta_coarse_map"])
    _log(f"{prefix}/gran/delta_fine_map", g["delta_fine_map"])
    wgc = report["tier2_within_group_confusion"]
    _log(f"{prefix}/confusion/overall_rate", wgc["overall_confusion_rate"])
    logger.info("logged eval scalars to MLflow under '%s/'", prefix)


def emit_all(report: dict, out_dir: Path, log_mlflow: bool = False) -> dict[str, Path]:
    """Write markdown + CSVs + JSON to *out_dir*; optionally log to MLflow.

    Returns a dict of the written paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    md = out_dir / "evaluation_report.md"
    js = out_dir / "evaluation_report.json"
    write_markdown(report, md)
    write_csv_artifacts(report, out_dir)
    write_json(report, js)
    if log_mlflow:
        try:
            log_to_mlflow(report)
            import mlflow
            for f in [md, js, out_dir / "eval_per_class.csv",
                      out_dir / "eval_band_grid.csv", out_dir / "eval_confusion_pairs.csv"]:
                mlflow.log_artifact(str(f), artifact_path="evaluation")
        except Exception as exc:  # pragma: no cover - mlflow optional
            logger.warning("MLflow logging failed (%s); files still written locally", exc)
    return {"markdown": md, "json": js}
