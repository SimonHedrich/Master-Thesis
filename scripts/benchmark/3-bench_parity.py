#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=2.0.0",
#   "opencv-python-headless>=4.10.0",
#   "onnxruntime>=1.20.0",
#   "torch>=2.4.0",
# ]
#
# # Pin torch to the CPU wheel index. Without this uv resolves the default
# # index, which drags the whole nvidia-cu13 stack onto an aarch64 SBC that
# # has no CUDA device -- gigabytes of download over a slow link, and a
# # slower torch import for no benefit.
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
#
# [tool.uv.sources]
# torch = [{ index = "pytorch-cpu" }]
# ///
"""
Stage 5 — Correctness parity: emit predictions over the fixed subset. RUNS ON THE PI.

Predictions should be identical across hardware, so this is a *numerical
equivalence* check rather than a re-evaluation: it answers "did the model
survive export and a different instruction set", not "how good is the model".

Output follows the frozen predictions-JSON contract defined by
`scripts/training/yolov5s/eval_suite/predict.py`, so the host can score it with
the project's own scorer and no scoring ever runs on the device — exactly the
predict-once / score-separately split the eval suite was built around.

Also writes the T1 tensor-level check: the exported graph's raw output on the
fixed reference input, compared against `reference_output_0.npy` (the host
PyTorch reference that shipped with the export).

Refuses to run against an architecture-only export, whose weights are
meaningless — those exports are valid for latency and nothing else.

Usage (on the Pi):
    uv run --script 3-bench_parity.py --model yolo26n-direct --runtime onnxruntime
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# Reuse the benchmark script's preprocessing, runtimes and decode so the two
# stages can never drift apart (a parity check run against a *different*
# preprocessing path would be measuring the wrong thing).
_spec = importlib.util.spec_from_file_location("bench", HERE / "2-bench_latency.py")
_bench = importlib.util.module_from_spec(_spec)
sys.modules["bench"] = _bench
_spec.loader.exec_module(_bench)

REGIMES = _bench.REGIMES
RUNNERS = _bench.RUNNERS


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--runtime", required=True, choices=sorted(RUNNERS))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--no-mkldnn", action="store_true",
                    help="see 2-bench_latency.py; required for multi-threaded TorchScript here")
    ap.add_argument("--regime", default="eval", choices=sorted(REGIMES))
    ap.add_argument("--exports", type=Path, default=HERE / "exports")
    ap.add_argument("--subset", type=Path, default=HERE / "benchmark_subset")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=0, help="debug: stop after N images")
    ap.add_argument(
        "--allow-architecture-only",
        action="store_true",
        help="run even though the export carries no trained weights (debug only)",
    )
    args = ap.parse_args()

    model_dir = args.exports / args.model
    manifest = json.loads((model_dir / "manifest.json").read_text())
    if not manifest.get("parity_valid") and not args.allow_architecture_only:
        print(
            f"REFUSING: {args.model} was exported architecture-only "
            f"(weights_source={manifest.get('weights_source')}). Its latency numbers "
            f"are valid; its predictions are not. Pass --allow-architecture-only to override.",
            file=sys.stderr,
        )
        return 2

    imgsz = manifest["image_size"]
    layout = manifest.get("input_layout", "nchw")
    decode = manifest["decode"]
    regime = REGIMES[args.regime]

    if decode == "classifier":
        print(f"REFUSING: {args.model} is a classifier — it emits no boxes to score.", file=sys.stderr)
        return 2

    out_path = args.out or (
        HERE / "results" / f"predictions_{args.model}_{args.runtime}_pi.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ann_path = args.subset / "annotations_subset.json"
    ann = json.loads(ann_path.read_text())
    images = ann["images"]
    if args.limit:
        images = images[: args.limit]

    # COCO ids -> YOLO indices exactly as scripts/training/yolov5s/dataset.py
    # builds them: sort categories by id, enumerate. We need the inverse.
    sorted_cats = sorted(ann["categories"], key=lambda c: c["id"])
    yolo_to_cat = {i: c["id"] for i, c in enumerate(sorted_cats)}

    runner = RUNNERS[args.runtime](model_dir, args.threads, no_mkldnn=args.no_mkldnn)
    print(f"=== parity {args.model} / {args.runtime} / {len(images)} images ===")
    print(f"    regime={args.regime} {regime}")

    # ── T1: tensor-level check against the shipped host reference ────────────
    t1: dict = {"available": False}
    ref_in = model_dir / "reference_input.npy"
    ref_out = model_dir / "reference_output_0.npy"
    if ref_in.exists() and ref_out.exists():
        x = np.load(ref_in)
        ref = np.load(ref_out)
        got = runner(x)
        if got.shape == ref.shape:
            d = np.abs(ref.astype(np.float64) - got.astype(np.float64))
            scale = float(np.abs(ref.astype(np.float64)).max())
            t1 = {
                "available": True,
                "max_abs": float(d.max()),
                "mean_abs": float(d.mean()),
                "ref_absmax": scale,
                "bound": 1e-5 + 1e-5 * scale,
                "passed": bool(d.max() <= 1e-5 + 1e-5 * scale),
            }
        else:
            t1 = {"available": True, "shape_mismatch": [list(ref.shape), list(got.shape)],
                  "passed": False}
        print(f"    T1 tensor: max_abs={t1.get('max_abs', float('nan')):.3e} "
              f"bound={t1.get('bound', float('nan')):.3e} "
              f"[{'PASS' if t1.get('passed') else 'FAIL'}]")

    # ── predictions over the subset ──────────────────────────────────────────
    import cv2

    predictions: list[dict] = []
    t0 = time.time()
    for n, rec in enumerate(images, 1):
        path = args.subset / rec["file_name"]
        raw = cv2.imread(str(path))
        if raw is None:
            print(f"    ! unreadable: {path}", file=sys.stderr)
            continue
        h0, w0 = raw.shape[:2]
        img, r, pad = _bench.letterbox(raw, imgsz)
        x = _bench.to_tensor(img, layout)
        out = runner(x)
        # apply_conf=False: match eval_suite/predict.py's contract exactly
        # (it emits every in-graph top-k row for the NMS-free head).
        dets = _bench.postprocess(out, decode, regime, r, pad, w0, h0,
                                  apply_conf=(decode != "end2end"))
        for det in dets:
            x1, y1, x2, y2, score, cls = det.tolist()
            cat = yolo_to_cat.get(int(cls))
            if cat is None:
                continue
            predictions.append(
                {
                    "image_id": rec["id"],
                    "category_id": cat,
                    "bbox": [round(x1, 3), round(y1, 3), round(x2 - x1, 3), round(y2 - y1, 3)],
                    "score": round(float(score), 5),
                }
            )
        if n % 50 == 0:
            rate = n / (time.time() - t0)
            print(f"    {n}/{len(images)}  {rate:.2f} img/s  {len(predictions)} dets", flush=True)

    elapsed = time.time() - t0
    payload = {
        # Frozen contract — see eval_suite/predict.py's docstring.
        "checkpoint": manifest.get("weights_source"),
        "annotations": str(ann_path),
        "eval": {
            "conf_thres": regime["conf_thres"],
            "iou_thres": regime["iou_thres"],
            "max_det": regime["max_det"],
            "image_size": imgsz,
        },
        "num_images": len(images),
        "predictions": predictions,
        # Extra provenance; the scorer ignores unknown top-level keys.
        "device": {
            "runtime": args.runtime,
            "runtime_version": runner.version,
            "threads": args.threads,
            "node": __import__("platform").node(),
            "machine": __import__("platform").machine(),
        },
        "t1_tensor_check": t1,
        "wall_seconds": round(elapsed, 1),
        "images_per_second": round(len(images) / elapsed, 3),
    }
    out_path.write_text(json.dumps(payload))
    print(
        f"    wrote {out_path.name}: {len(predictions)} predictions over "
        f"{len(images)} images in {elapsed / 60:.1f} min ({len(images) / elapsed:.2f} img/s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
