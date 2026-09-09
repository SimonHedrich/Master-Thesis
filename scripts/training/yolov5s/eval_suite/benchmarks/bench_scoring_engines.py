"""Go/no-go benchmark for `scoring.py`'s scoring engine (TODO.md 2.1).

Compares wall-clock and peak RSS of three ways to compute COCO-style mAP at
increasing image-count scale: the current per-class `torchmetrics` loop
(`scoring.score()`), vanilla `pycocotools.COCOeval`, and `faster_coco_eval`
(not a project dependency yet — install into the dev venv before running,
e.g. `uv pip install faster-coco-eval`; this script does not add it to
pyproject.toml/uv.lock).

Uses *synthetic* predictions rather than a cached predictions JSON, generated
to match the density profile documented in
`docs/plans/2026-07-22_eval-suite-scoring-performance-investigation.md`
(the NMS-free yolo26n model fills all 100 detection slots/image regardless of
confidence — cached real predictions_real.json files from other checkpoints
that went through NMS are ~15-60x sparser and would not exercise the actual
bottleneck). Predictions are generated over the real GT image/category
universe (`data/real/annotations_test.json`, 63,802 images / 225 categories)
so scale numbers are directly comparable to a real evaluation run.

Each engine runs in its own forked subprocess so a stalled/hung engine (this
is expected of vanilla pycocotools per the investigation doc) can be killed
on a timeout without aborting the whole benchmark matrix, and so peak RSS
(self-reported via `resource.getrusage`) reflects only that engine's run.

Run command
-----------
    uv run python -m scripts.training.yolov5s.eval_suite.benchmarks.bench_scoring_engines
    uv run python -m scripts.training.yolov5s.eval_suite.benchmarks.bench_scoring_engines \
        --scales 500,2000,5000,20000,63802 --engines faster_coco_eval

Go/no-go: faster_coco_eval PASSES if the largest scale run completes in under
10 minutes with peak RSS well under ~20GB (leaving headroom under the ~30GB
level that OOM-killed the original all-classes-at-once design), and its
map/map_50 at a shared moderate scale matches the torchmetrics baseline.
"""
from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import random
import resource
import time
from pathlib import Path

from scripts.training.yolov5s import constants
from scripts.training.yolov5s.eval_suite import scoring

logger = logging.getLogger(__name__)

DEFAULT_REAL_ANN = constants.ANNOTATIONS_TEST
DEFAULT_MAX_DET = constants.EVAL_MAX_DET
DEFAULT_SCALES = "500,2000,5000,20000,63802"
# Engines known to blow up well past these scales — capped so a single bad
# engine can't stall the whole matrix; override via --max-scale-<engine>.
DEFAULT_ENGINE_SCALE_CAPS = {
    "pycocotools": 5_000,
    "torchmetrics": 20_000,
    "faster_coco_eval": None,  # this is the one being tested at full scale
}


# ---------------------------------------------------------------------------
# Synthetic dense-prediction generator
# ---------------------------------------------------------------------------

def _subsample_image_ids(all_ids: list[int], n: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return sorted(rng.sample(all_ids, min(n, len(all_ids))))


def _make_dense_predictions(
    gt_index: dict, image_ids: list[int], dets_per_image: int = 100, seed: int = 42,
) -> list[dict]:
    """Synthesize predictions filling every detection slot/image, spread
    across the full category universe — matching the NMS-free model's
    density profile (see module docstring)."""
    rng = random.Random(seed)
    cat_ids = sorted(gt_index["cats"].keys())
    images = gt_index["images"]
    preds: list[dict] = []
    for iid in image_ids:
        info = images[iid]
        w = info.get("width") or 640
        h = info.get("height") or 640
        for _ in range(dets_per_image):
            cid = rng.choice(cat_ids)
            bw = rng.uniform(10, max(w * 0.5, 11))
            bh = rng.uniform(10, max(h * 0.5, 11))
            x = rng.uniform(0, max(w - bw, 0))
            y = rng.uniform(0, max(h - bh, 0))
            preds.append(
                {"image_id": iid, "category_id": cid, "bbox": [x, y, bw, bh], "score": rng.random()}
            )
    return preds


def _gt_index_to_coco_dict(gt_index: dict, image_ids: list[int]) -> dict:
    """Build a raw COCO-format dict (pycocotools/faster_coco_eval input)
    restricted to *image_ids*, manufacturing sequential annotation ids that
    `gt_index`'s internal representation doesn't carry."""
    images = [
        {"id": iid, "width": gt_index["images"][iid].get("width", 0) or 640,
         "height": gt_index["images"][iid].get("height", 0) or 640,
         "file_name": gt_index["images"][iid].get("file_name", "")}
        for iid in image_ids
    ]
    categories = [{"id": cid, "name": name} for cid, name in sorted(gt_index["cats"].items())]
    annotations = []
    next_id = 1
    for iid in image_ids:
        for ann in gt_index["anns"].get(iid, []):
            annotations.append({
                "id": next_id, "image_id": iid, "category_id": ann["category_id"],
                "bbox": ann["bbox"], "area": ann["area"], "iscrowd": 0,
            })
            next_id += 1
    return {"images": images, "annotations": annotations, "categories": categories}


# ---------------------------------------------------------------------------
# Per-engine run functions (imported lazily inside the subprocess)
# ---------------------------------------------------------------------------

def _run_torchmetrics(gt_index: dict, image_ids: list[int], preds: list[dict], max_det: int) -> dict:
    result = scoring.score(
        gt_index, preds, image_ids=set(image_ids), remap=None, max_det=max_det, class_metrics=True,
    )
    return {"map": result["map"], "map_50": result["map_50"]}


def _run_pycocotools(gt_dict: dict, preds: list[dict], max_det: int) -> dict:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO()
    coco_gt.dataset = gt_dict
    coco_gt.createIndex()
    coco_dt = coco_gt.loadRes(preds)
    ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
    ev.params.maxDets = [1, 10, max_det]
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return {"map": float(ev.stats[0]), "map_50": float(ev.stats[1])}


def _run_faster_coco_eval(gt_dict: dict, preds: list[dict], max_det: int) -> dict:
    import faster_coco_eval as fce

    coco_gt = fce.COCO(gt_dict)
    coco_dt = coco_gt.loadRes(preds)
    ev = fce.COCOeval_faster(coco_gt, coco_dt, iouType="bbox")
    ev.params.maxDets = [1, 10, max_det]
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return {"map": float(ev.stats[0]), "map_50": float(ev.stats[1])}


_ENGINE_FUNCS = {
    "torchmetrics": _run_torchmetrics,
    "pycocotools": _run_pycocotools,
    "faster_coco_eval": _run_faster_coco_eval,
}


# ---------------------------------------------------------------------------
# Subprocess isolation + timeout
# ---------------------------------------------------------------------------

def _subprocess_entry(engine: str, gt_index: dict, gt_dict: dict | None, image_ids: list[int],
                       preds: list[dict], max_det: int, queue: mp.Queue) -> None:
    try:
        t0 = time.perf_counter()
        if engine == "torchmetrics":
            metrics = _run_torchmetrics(gt_index, image_ids, preds, max_det)
        else:
            metrics = _ENGINE_FUNCS[engine](gt_dict, preds, max_det)
        elapsed = time.perf_counter() - t0
        peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        queue.put({"ok": True, "elapsed_s": elapsed, "peak_rss_mb": peak_rss_mb, **metrics})
    except Exception as e:  # noqa: BLE001 — report any failure back to the parent
        queue.put({"ok": False, "error": repr(e)})


def _run_with_timeout(engine: str, gt_index: dict, gt_dict: dict | None, image_ids: list[int],
                       preds: list[dict], max_det: int, timeout_s: float) -> dict:
    ctx = mp.get_context("fork")
    queue: mp.Queue = ctx.Queue()
    proc = ctx.Process(
        target=_subprocess_entry,
        args=(engine, gt_index, gt_dict, image_ids, preds, max_det, queue),
    )
    proc.start()
    proc.join(timeout_s)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        return {"ok": False, "error": f"TIMEOUT after {timeout_s:.0f}s"}
    if not queue.empty():
        return queue.get()
    return {"ok": False, "error": f"no result (exit code {proc.exitcode})"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--real-ann", type=Path, default=DEFAULT_REAL_ANN)
    p.add_argument("--scales", default=DEFAULT_SCALES, help="comma-separated image counts")
    p.add_argument("--engines", default="torchmetrics,pycocotools,faster_coco_eval",
                   help="comma-separated subset of torchmetrics,pycocotools,faster_coco_eval")
    p.add_argument("--dets-per-image", type=int, default=100)
    p.add_argument("--max-det", type=int, default=DEFAULT_MAX_DET)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--timeout", type=float, default=600.0, help="per-engine-run kill timeout, seconds")
    p.add_argument("--max-scale-torchmetrics", type=int, default=DEFAULT_ENGINE_SCALE_CAPS["torchmetrics"])
    p.add_argument("--max-scale-pycocotools", type=int, default=DEFAULT_ENGINE_SCALE_CAPS["pycocotools"])
    args = p.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    scales = sorted(int(s) for s in args.scales.split(","))
    engines = [e.strip() for e in args.engines.split(",")]
    caps = {
        "torchmetrics": args.max_scale_torchmetrics,
        "pycocotools": args.max_scale_pycocotools,
        "faster_coco_eval": None,
    }

    print(f"loading GT index from {args.real_ann} ...")
    gt_index = scoring.build_gt_index(args.real_ann)
    all_image_ids = sorted(gt_index["images"].keys())
    print(f"real GT universe: {len(all_image_ids)} images, {len(gt_index['cats'])} categories")

    rows = []
    for scale in scales:
        if scale > len(all_image_ids):
            print(f"skipping scale={scale} — exceeds GT universe size ({len(all_image_ids)})")
            continue
        image_ids = _subsample_image_ids(all_image_ids, scale, args.seed)
        preds = _make_dense_predictions(gt_index, image_ids, args.dets_per_image, args.seed)
        print(f"\n=== scale={scale} images ({len(preds)} predictions) ===")

        gt_dict = None
        if any(e in ("pycocotools", "faster_coco_eval") for e in engines):
            gt_dict = _gt_index_to_coco_dict(gt_index, image_ids)

        for engine in engines:
            cap = caps.get(engine)
            if cap is not None and scale > cap:
                print(f"  {engine:>17}: skipped (scale {scale} > cap {cap})")
                continue
            result = _run_with_timeout(
                engine, gt_index, gt_dict, image_ids, preds, args.max_det, args.timeout,
            )
            if result.get("ok"):
                print(
                    f"  {engine:>17}: {result['elapsed_s']:8.1f}s  "
                    f"peak_rss={result['peak_rss_mb']:8.0f}MB  "
                    f"map={result['map']:.4f}  map_50={result['map_50']:.4f}"
                )
            else:
                print(f"  {engine:>17}: FAILED — {result.get('error')}")
            rows.append({"scale": scale, "engine": engine, **result})

    # ── go/no-go verdict ───────────────────────────────────────────────────
    fce_rows = [r for r in rows if r["engine"] == "faster_coco_eval" and r.get("ok")]
    print("\n=== go/no-go ===")
    if not fce_rows:
        print("NO-GO: faster_coco_eval produced no successful runs.")
        return
    largest = max(fce_rows, key=lambda r: r["scale"])
    pass_time = largest["elapsed_s"] < 600
    pass_rss = largest["peak_rss_mb"] < 20_000
    print(
        f"largest faster_coco_eval run: scale={largest['scale']} "
        f"elapsed={largest['elapsed_s']:.1f}s peak_rss={largest['peak_rss_mb']:.0f}MB"
    )
    print(f"  under 10 min? {pass_time}   under 20GB RSS? {pass_rss}")
    tm_rows = [r for r in rows if r["engine"] == "torchmetrics" and r.get("ok")]
    if tm_rows:
        shared_scale = min(
            {r["scale"] for r in tm_rows} & {r["scale"] for r in fce_rows}, default=None,
        )
        if shared_scale is not None:
            tm = next(r for r in tm_rows if r["scale"] == shared_scale)
            fc = next(r for r in fce_rows if r["scale"] == shared_scale)
            close = abs(tm["map"] - fc["map"]) < 1e-2
            print(
                f"  cross-engine map @ scale={shared_scale}: torchmetrics={tm['map']:.4f} "
                f"faster_coco_eval={fc['map']:.4f}  close? {close}"
            )
    verdict = "GO" if (pass_time and pass_rss) else "NO-GO"
    print(f"\nVERDICT: {verdict}")


if __name__ == "__main__":
    main()
