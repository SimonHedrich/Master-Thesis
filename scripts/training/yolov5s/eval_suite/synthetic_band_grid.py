"""Band × granularity grid on the synthetic test set alone.

:mod:`report` scores the band grid (Tier 2.2) on the mixed and real domains only,
because the evaluation strategy never judges a model on synthetic images alone.
This script fills in the synthetic-only grid from a run's cached
``predictions_synth.json`` with the same per-band image subsets and scoring
call, so the figures are directly comparable to ``eval_band_grid.csv``. It
writes ``eval_band_grid_synthetic.csv`` next to the run's evaluation report and
prints the rows, followed by one ``all`` row over the whole synthetic test set.
``--predictions`` points at a ``predictions_synth.json`` that does not sit in the
evaluation directory (the ensemble runs keep it one level up).

Usage
-----
    uv run python -m scripts.training.yolov5s.eval_suite.synthetic_band_grid \
        --eval-dir scripts/training/yolo26n/model_exports/<run_name>/evaluation \
        [--eval-dir ...] [--predictions <path>/predictions_synth.json ...]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

from scripts.training.yolov5s.eval_suite import grouping, scoring
from scripts.training.yolov5s.eval_suite.report import BANDS
from scripts.training.yolov5s.eval_suite.run_evaluation import DEFAULT_SYNTH_ANN

logger = logging.getLogger(__name__)

FIELDS = ["domain", "band", "n_images", "fine_map", "fine_map_50", "coarse_map", "coarse_map_50"]


def synthetic_band_grid(predictions: Path, synth_ann: Path, max_det: int = 100) -> list[dict]:
    """Score cached synthetic predictions per band, then over the whole synthetic test set."""
    synth_gt = scoring.build_gt_index(synth_ann)
    with open(predictions) as f:
        preds = json.load(f)["predictions"]
    cat_ids = sorted(synth_gt["cats"].keys())
    remaps = {"fine": grouping.identity_remap(cat_ids), "coarse": grouping.load_coarse_remap()}
    rows = []
    for band in BANDS + ["all"]:
        ids = (set(synth_gt["images"].keys()) if band == "all"
               else scoring.filter_image_ids_by_band(synth_gt, {band}))
        if not ids:
            continue
        fine_s = scoring.score(synth_gt, preds, image_ids=ids, remap=remaps["fine"],
                               max_det=max_det, class_metrics=False)
        coarse_s = scoring.score(synth_gt, preds, image_ids=ids, remap=remaps["coarse"],
                                 max_det=max_det, class_metrics=False)
        rows.append({
            "domain": "synthetic", "band": band, "n_images": len(ids),
            "fine_map": fine_s["map"], "fine_map_50": fine_s["map_50"],
            "coarse_map": coarse_s["map"], "coarse_map_50": coarse_s["map_50"],
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", action="append", required=True, type=Path,
                    help="evaluation directory to write the CSV into (repeatable)")
    ap.add_argument("--predictions", action="append", type=Path, default=None,
                    help="predictions_synth.json per --eval-dir, in the same order "
                         "(default: <eval-dir>/predictions_synth.json)")
    ap.add_argument("--synth-ann", type=Path, default=DEFAULT_SYNTH_ANN)
    ap.add_argument("--max-det", type=int, default=100)
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    preds = args.predictions or [d / "predictions_synth.json" for d in args.eval_dir]
    if len(preds) != len(args.eval_dir):
        ap.error("--predictions must be given once per --eval-dir")
    for eval_dir, pred_path in zip(args.eval_dir, preds):
        rows = synthetic_band_grid(pred_path, args.synth_ann, args.max_det)
        out = eval_dir / "eval_band_grid_synthetic.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"== {eval_dir}")
        for r in rows:
            print(f"  {r['band']:<3} n={r['n_images']:5d}  fine {r['fine_map']:.3f} / {r['fine_map_50']:.3f}"
                  f"  coarse {r['coarse_map']:.3f} / {r['coarse_map_50']:.3f}")
        print(f"  -> {out}")


if __name__ == "__main__":
    main()
