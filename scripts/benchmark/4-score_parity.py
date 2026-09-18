#!/usr/bin/env python3
"""
Stage 5b — Score the Pi's predictions against a host reference (host).

Generates the host reference with the project's OWN inference path
(`eval_suite/predict.py::run_inference`), so the reference is the same code
that produced every mAP in the thesis rather than a reimplementation, and
compares it against the predictions `3-bench_parity.py` emitted on the device.

Three tiers, increasing tolerance:

  T1 tensor     raw graph output vs the shipped PyTorch reference. Computed on
                the device by 3-bench_parity.py; read back from its JSON here.
  T2 detection  per-image detection matching: same category, IoU >= 0.99,
                |dscore| <= 0.01.
  T3 metric     subset mAP from each side's predictions, via the project's own
                scorer. Only the DELTA is interpreted — a 600-image subset
                cannot produce a meaningful absolute 225-class mAP and is not
                used as one.

The reference is generated twice (GPU and host CPU) so a delta can be
attributed to *device / ISA* versus *runtime* rather than lumped together:
host-GPU vs host-CPU is the fp32 accumulation-order baseline that any
cross-hardware comparison has to be read against.

Usage:
    uv run python scripts/benchmark/4-score_parity.py
    uv run python scripts/benchmark/4-score_parity.py --models yolo26n-direct
    uv run python scripts/benchmark/4-score_parity.py --skip-reference   # reuse cached
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import constants as C  # noqa: E402
from scripts.training.yolov5s.eval_suite import scoring  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

REPO_ANN = C.SUBSET_DIR / "annotations_subset_repo.json"


def make_repo_relative_annotations() -> Path:
    """Rewrite the subset annotation file with repo-root-relative `file_name`s.

    `eval_suite`'s dataset resolves `file_name` against IMAGE_ROOT == REPO_ROOT,
    while the shipped subset uses paths relative to its own directory (so it is
    self-contained on the device). This bridges the two without duplicating
    any images.
    """
    data = json.loads((C.SUBSET_DIR / "annotations_subset.json").read_text())
    prefix = C.SUBSET_DIR.relative_to(REPO_ROOT)
    for img in data["images"]:
        if not img["file_name"].startswith("data/"):
            img["file_name"] = str(prefix / img["file_name"])
    REPO_ANN.write_text(json.dumps(data))
    return REPO_ANN


def host_reference(model: str, device: str, out_path: Path) -> Path:
    """Run the project's own inference path over the subset.

    Uses the frozen snapshot `1-export_models.py` left in the export directory,
    not the live checkpoint: a training run still in flight rewrites `best.pt`,
    and a reference built from different weights than the device is running
    would show up as a hardware discrepancy that is not one.
    """
    spec = C.MODELS[model]
    frozen = C.EXPORT_DIR / model / "source_checkpoint.pt"
    if frozen.exists():
        spec = {**spec, "checkpoint": frozen}
    else:
        print(f"    ! no frozen snapshot for {model}; using the live checkpoint")
    family = spec["family"]
    if family == "yolo26n":
        from scripts.training.yolo26n.eval_suite import predict as P
        import scripts.training.yolo26n.constants as MC
    elif family == "yolov5s":
        from scripts.training.yolov5s.eval_suite import predict as P
        import scripts.training.yolov5s.constants as MC
    else:
        raise ValueError(f"no detector reference path for family {family}")

    P.run_inference(
        checkpoint_path=spec["checkpoint"],
        annotations_path=REPO_ANN,
        output_path=out_path,
        device=torch.device(device),
        conf_thres=MC.EVAL_CONF_THRES,
        iou_thres=MC.EVAL_IOU_THRES,
        max_det=MC.EVAL_MAX_DET,
        image_size=MC.IMAGE_SIZE,
        batch_size=8 if device == "cpu" else 16,
        num_workers=2,
        cache=True,
    )
    return out_path


# ─── T2: detection-level matching ─────────────────────────────────────────────

def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a, b are (N,4)/(M,4) in xywh; returns (N,M) IoU."""
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    ax1, ay1 = a[:, 0], a[:, 1]
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx1, by1 = b[:, 0], b[:, 1]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    ix1 = np.maximum(ax1[:, None], bx1[None, :])
    iy1 = np.maximum(ay1[:, None], by1[None, :])
    ix2 = np.minimum(ax2[:, None], bx2[None, :])
    iy2 = np.minimum(ay2[:, None], by2[None, :])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = ((ax2 - ax1) * (ay2 - ay1))[:, None]
    area_b = ((bx2 - bx1) * (by2 - by1))[None, :]
    return inter / (area_a + area_b - inter + 1e-12)


def t2_match(ref: list[dict], got: list[dict]) -> dict:
    by_img_ref: dict[int, list[dict]] = {}
    by_img_got: dict[int, list[dict]] = {}
    for p in ref:
        by_img_ref.setdefault(p["image_id"], []).append(p)
    for p in got:
        by_img_got.setdefault(p["image_id"], []).append(p)

    matched = total = 0
    score_deltas: list[float] = []
    for iid, rlist in by_img_ref.items():
        glist = by_img_got.get(iid, [])
        total += len(rlist)
        if not glist:
            continue
        rb = np.array([p["bbox"] for p in rlist], dtype=np.float64)
        gb = np.array([p["bbox"] for p in glist], dtype=np.float64)
        rc = np.array([p["category_id"] for p in rlist])
        gc = np.array([p["category_id"] for p in glist])
        rs = np.array([p["score"] for p in rlist])
        gs = np.array([p["score"] for p in glist])
        iou = _iou_matrix(rb, gb)
        same_cat = rc[:, None] == gc[None, :]
        close_score = np.abs(rs[:, None] - gs[None, :]) <= C.PARITY["t2_max_score_delta"]
        ok = (iou >= C.PARITY["t2_iou"]) & same_cat & close_score
        used = set()
        for i in range(len(rlist)):
            cand = np.where(ok[i])[0]
            cand = [j for j in cand if j not in used]
            if cand:
                j = max(cand, key=lambda j: iou[i, j])
                used.add(j)
                matched += 1
                score_deltas.append(abs(rs[i] - gs[j]))
    rate = matched / total if total else 0.0
    return {
        "ref_detections": total,
        "pi_detections": len(got),
        "matched": matched,
        "match_rate": round(rate, 6),
        "max_score_delta": round(max(score_deltas), 6) if score_deltas else 0.0,
        "passed": bool(rate >= C.PARITY["t2_min_match_rate"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default="all")
    ap.add_argument("--pi-results", type=Path, default=C.REPORT_DIR / "pi_results")
    ap.add_argument("--out", type=Path, default=C.REPORT_DIR / "parity_summary.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--skip-reference", action="store_true")
    args = ap.parse_args()

    names = list(C.PARITY_MODELS) if args.models == "all" else args.models.split(",")
    make_repo_relative_annotations()
    gt = scoring.build_gt_index(REPO_ANN)
    C.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ref_dir = C.REPORT_DIR / "host_reference"
    ref_dir.mkdir(parents=True, exist_ok=True)

    summary: list[dict] = []
    for name in names:
        manifest_path = C.EXPORT_DIR / name / "manifest.json"
        if not manifest_path.exists():
            print(f"skip {name}: not exported")
            continue
        manifest = json.loads(manifest_path.read_text())
        if not manifest.get("parity_valid"):
            print(
                f"skip {name}: architecture-only export "
                f"({manifest.get('weights_source')}) — latency valid, parity not applicable"
            )
            summary.append(
                {"model": name, "skipped": "architecture-only export",
                 "weights_source": manifest.get("weights_source")}
            )
            continue

        print(f"\n=== {name} ===")
        refs: dict[str, list[dict]] = {}
        for dev in ([args.device, "cpu"] if args.device != "cpu" else ["cpu"]):
            p = ref_dir / f"{name}_host_{dev}.json"
            if not (args.skip_reference and p.exists()):
                print(f"    host reference on {dev} ...")
                host_reference(name, dev, p)
            refs[f"host_{dev}"] = json.loads(p.read_text())["predictions"]
            print(f"    host_{dev}: {len(refs[f'host_{dev}'])} predictions")

        base_key = f"host_{args.device}"
        base = refs[base_key]
        base_map = scoring.score(gt, base, max_det=100, class_metrics=False)["map"]
        print(f"    reference subset mAP ({base_key}) = {base_map:.4f}")

        rows = []
        # host CPU vs host GPU — the fp32 accumulation-order baseline.
        if "host_cpu" in refs and base_key != "host_cpu":
            t2 = t2_match(base, refs["host_cpu"])
            m = scoring.score(gt, refs["host_cpu"], max_det=100, class_metrics=False)["map"]
            rows.append({"side": "host_cpu", "kind": "baseline (same ISA family, x86)",
                         "t2": t2, "map": m, "map_delta": m - base_map})

        for pj in sorted(args.pi_results.glob(f"predictions_{name}_*_pi.json")):
            payload = json.loads(pj.read_text())
            rt = payload.get("device", {}).get("runtime", pj.stem)
            preds = payload["predictions"]
            t2 = t2_match(base, preds)
            m = scoring.score(gt, preds, max_det=100, class_metrics=False)["map"]
            rows.append(
                {
                    "side": f"pi_{rt}",
                    "kind": "device (aarch64 Cortex-A72)",
                    "t1": payload.get("t1_tensor_check", {}),
                    "t2": t2,
                    "map": m,
                    "map_delta": m - base_map,
                    "images_per_second": payload.get("images_per_second"),
                }
            )

        for r in rows:
            t3_pass = abs(r["map_delta"]) <= C.PARITY["t3_max_map_delta"]
            r["t3_passed"] = bool(t3_pass)
            t1 = r.get("t1", {})
            t1s = (
                f"T1 {'PASS' if t1.get('passed') else 'FAIL'} ({t1.get('max_abs', float('nan')):.2e})"
                if t1.get("available")
                else "T1 n/a"
            )
            print(
                f"    {r['side']:<20} {t1s:<22} "
                f"T2 {r['t2']['match_rate'] * 100:6.2f}% "
                f"{'PASS' if r['t2']['passed'] else 'FAIL'}   "
                f"mAP {r['map']:.4f} (Δ{r['map_delta']:+.4f}) "
                f"{'PASS' if t3_pass else 'FAIL'}"
            )

        summary.append(
            {"model": name, "reference": base_key, "reference_map": base_map,
             "tolerances": C.PARITY, "comparisons": rows}
        )

    args.out.write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
