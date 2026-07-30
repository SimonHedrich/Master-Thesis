"""Assemble the simplified evaluation report and emit it.

This experiment's Axis C ask (docs/synthetic-model-comparison/
06_evaluation-methodology.md §4) is narrower than the production
scripts/training/yolov5s suite this module was copied from: just headline
real-test mAP, a per-class AP table (the rare-species readout), and the
zebra-style within-look-alike-group confusion matrix. Dropped relative to the
production version: the mixed/real-domain duplication (moot here — this
experiment's test set is real-only, see run_evaluation.py), the class-agnostic
"detect" granularity, the granularity-gap decomposition, the band×granularity
grid, and the domain-shift-delta section. None of those answer a question
this experiment (comparing image generators, not detector architectures)
actually asks.

Nothing here runs the model — it consumes the cached predictions + GT index
from :mod:`predict`/:mod:`scoring` and the label remaps from :mod:`grouping`.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path

from scripts.synthetic_model_comparison.training.eval_suite import scoring

logger = logging.getLogger(__name__)

# The COCO-12 metric vector, in report order.
COCO12 = [
    "map", "map_50", "map_75",
    "map_small", "map_medium", "map_large",
    "mar_1", "mar_10", "mar_100",
    "mar_small", "mar_medium", "mar_large",
]
TEST_LIMITED_THRESHOLD = 30  # <30 real test images → flagged (rare-species readout, strategy §8)


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
    to NaN so it renders as '—'."""
    return float("nan") if (isinstance(v, (int, float)) and v == -1.0) else v


def _coco12(result: dict) -> dict[str, float]:
    """Extract the COCO-12 vector from a score() result, sentinel-cleaned."""
    return {k: _na(result[k]) for k in COCO12}


def real_images_per_class(gt: dict) -> dict[int, int]:
    """Count distinct real test images that contain each fine category.

    Each image is counted once per category it carries a GT box for.
    """
    counts: dict[int, int] = {}
    for iid, anns in gt["anns"].items():
        for c in {a["category_id"] for a in anns}:
            counts[c] = counts.get(c, 0) + 1
    return counts


# ───────────────────────────────────────────────────────────────────────────────
# Report assembly
# ───────────────────────────────────────────────────────────────────────────────

def build_report(
    *,
    gt: dict,
    preds: list[dict],
    fine_remap: dict[int, int],
    coarse_remap: dict[int, int],
    band_by_id: dict[int, str],
    lookalike_gids: list[int],
    group_labels: dict[int, str],
    cat_id_to_name: dict[int, str],
    max_det: int,
    bootstrap_n: int = 0,
    checkpoint: str = "",
) -> dict:
    """Compute the headline mAP, per-class AP table, and within-group confusion.

    Returns a nested, JSON-serialisable dict with keys ``headline``,
    (optionally) ``headline_map_ci``, ``per_class``, and ``confusion``.
    """
    report: dict = {
        "checkpoint": checkpoint,
        "config": {"max_det": max_det, "bootstrap_n": bootstrap_n},
    }

    # ── Headline — real-test mAP (fine, 12-way) ────────────────────────────────
    logger.info("headline: real-test mAP (fine)")
    headline = scoring.score(gt, preds, remap=fine_remap, max_det=max_det, class_metrics=True)
    report["headline"] = _coco12(headline)

    if bootstrap_n > 0:
        logger.info("headline: bootstrap CI on mAP (n=%d)", bootstrap_n)
        report["headline_map_ci"] = scoring.bootstrap_ci(
            gt, preds, image_ids=set(gt["images"].keys()), remap=fine_remap,
            max_det=max_det, n_boot=bootstrap_n, metric_key="map")

    # ── Per-class AP table (the rare-species readout) ──────────────────────────
    logger.info("per-class AP table")
    img_counts = real_images_per_class(gt)
    per_class_rows = []
    for cid in sorted(cat_id_to_name):
        n_img = img_counts.get(cid, 0)
        per_class_rows.append({
            "coco_id": cid,
            "class_name": cat_id_to_name[cid],
            "band": band_by_id.get(cid, "?"),
            "real_test_images": n_img,
            "test_limited": n_img < TEST_LIMITED_THRESHOLD,
            "ap": _na(headline["map_per_class"].get(cid, float("nan"))),
        })
    report["per_class"] = per_class_rows

    # ── Within look-alike group confusion (the zebra test) ─────────────────────
    logger.info("within-look-alike-group confusion")
    confusion = scoring.within_group_confusion(
        gt, preds, coarse_remap=coarse_remap,
        lookalike_group_ids=lookalike_gids, image_ids=None)
    report["confusion"] = {
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

    return report


# ───────────────────────────────────────────────────────────────────────────────
# Writers
# ───────────────────────────────────────────────────────────────────────────────

def write_markdown(report: dict, path: Path) -> None:
    """Render the report as human-readable Markdown."""
    L: list[str] = []
    L.append("# Model Evaluation Report\n")
    L.append(f"**Checkpoint:** `{report.get('checkpoint','')}`  ")
    cfg = report["config"]
    L.append(f"**max_det:** {cfg['max_det']}\n")

    # Headline
    h = report["headline"]
    L.append("\n## Headline — real-test mAP (fine, 12-way)\n")
    L.append("| Metric | value |")
    L.append("|--------|-------|")
    for k in COCO12:
        L.append(f"| {k} | {_fmt(h[k])} |")
    if "headline_map_ci" in report:
        ci = report["headline_map_ci"]
        L.append(f"\n**Bootstrap 95% CI** (mAP): {_fmt(ci['mean'])} [{_fmt(ci['lo'])}, {_fmt(ci['hi'])}].")

    # Per-class
    L.append("\n## Per-class AP\n")
    L.append("| Class | Band | Real test images | Test-limited | AP |")
    L.append("|-------|------|-------------------|--------------|----|")
    for row in report["per_class"]:
        L.append(
            f"| {row['class_name']} | {row['band']} | {row['real_test_images']} | "
            f"{'yes' if row['test_limited'] else ''} | {_fmt(row['ap'])} |"
        )
    L.append(
        f"\n> Classes flagged test-limited have <{TEST_LIMITED_THRESHOLD} real test images — "
        "lean on Axes A/B (qualitative rubric, teacher-recognition proxy) for those "
        "(`06_evaluation-methodology.md`)."
    )

    # Confusion
    c = report["confusion"]
    L.append("\n## Within look-alike group confusion\n")
    L.append(f"Overall within-group fine-confusion rate: {_fmt(c['overall_confusion_rate'])}\n")
    L.append("| Look-alike group | matched | confused | rate |")
    L.append("|------------------|---------|----------|------|")
    for label, st in sorted(c["by_group"].items(), key=lambda kv: -(kv[1]["matched"])):
        L.append(f"| {label} | {st['matched']} | {st['confused']} | {_fmt(st['confusion_rate'])} |")

    L.append(
        "\n---\n*The per-class table and confusion pairs are emitted as CSV/JSON "
        "artifacts alongside this file.*\n"
    )

    path.write_text("\n".join(L), encoding="utf-8")
    logger.info("wrote markdown report: %s", path)


def write_csv_artifacts(report: dict, out_dir: Path) -> None:
    """Emit the machine-readable CSVs."""
    import csv

    # per-class table
    pc_path = out_dir / "eval_per_class.csv"
    with pc_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "coco_id", "class_name", "band", "real_test_images", "test_limited", "ap"])
        w.writeheader()
        for row in report["per_class"]:
            w.writerow(row)

    # confusion pairs
    cp_path = out_dir / "eval_confusion_pairs.csv"
    with cp_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["true_species", "pred_species", "count"])
        for row in report["confusion"]["pairs"]:
            w.writerow([row["true"], row["pred"], row["count"]])

    logger.info("wrote CSV artifacts: %s, %s", pc_path, cp_path)


def write_json(report: dict, path: Path) -> None:
    """Dump the entire report dict as JSON."""
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

    h = report["headline"]
    for k in COCO12:
        _log(f"{prefix}/{k}", h[k])
    c = report["confusion"]
    _log(f"{prefix}/confusion/overall_rate", c["overall_confusion_rate"])
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
            for f in [md, js, out_dir / "eval_per_class.csv", out_dir / "eval_confusion_pairs.csv"]:
                mlflow.log_artifact(str(f), artifact_path="evaluation")
        except Exception as exc:  # pragma: no cover - mlflow optional
            logger.warning("MLflow logging failed (%s); files still written locally", exc)
    return {"markdown": md, "json": js}
