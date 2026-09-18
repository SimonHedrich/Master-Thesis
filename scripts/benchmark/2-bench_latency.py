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
Stage 3 — The latency benchmark. RUNS ON THE RASPBERRY PI.

A PEP 723 standalone script: it takes no dependency on this repository's model
code and never imports `constants.py`, because it has to run on a machine where
the repo is not checked out and where `pyproject.toml`'s `pytorch-cu130` torch
pin would resolve to unusable CUDA wheels. It consumes only the artifacts that
`1-export_models.py` produced (`model.onnx`, `model.torchscript`,
`manifest.json`) plus the fixed image subset.

One process = one cell = one (model, runtime, threads, invocation). The driver
`pi/run_all.sh` invokes it three times per cell, which is what makes
"3 independent process invocations" mean something: allocator state, thread
pools and page cache all start fresh.

Four separately-timed windows (collapsing them is what makes most published
embedded latency numbers incomparable):

  W_infer  runtime forward call only, on an already-preprocessed tensor
           resident in memory. This is the MLPerf-Tiny-scoped headline the
           thesis's Literature Review 2.3.2 commits to.
  W_pre    JPEG decode -> letterbox -> BGR2RGB -> /255 -> CHW -> contiguous,
           timed both on a median-sized test image and on a native
           4192x3120 AX Visio still.
  W_post   NMS + rescale (YOLOv5s, MegaDetector), rescale only (YOLO26n,
           whose top-k is in-graph), or softmax+top-k (SpeciesNet).
  W_e2e    all three in sequence as one wall-clock loop, so allocator and
           cache effects are captured rather than assumed away.

A monitor process samples temperature / frequency / throttle bits for the
duration and the result is gated on them: a cell measured while throttled is
not a measurement of the model.

Usage (on the Pi):
    uv run --script 2-bench_latency.py --model yolo26n-direct --runtime onnxruntime \
        --threads 4 --invocation 1
    uv run --script 2-bench_latency.py --model yolo26n-direct --runtime onnxruntime \
        --threads 4 --sustained
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
LETTERBOX_COLOR = (114, 114, 114)

# Mirrors scripts/benchmark/constants.py::PROTOCOL / REGIMES / THERMAL_GATE.
# Duplicated rather than imported (this script must run with no repo present);
# every value is echoed into the output record so divergence is detectable.
PROTOCOL = {"warmup_iters": 20, "measure_iters": 100, "sustained_iters": 600, "batch_size": 1}
REGIMES = {
    "deployment": {"conf_thres": 0.25, "iou_thres": 0.45, "max_det": 100},
    "eval": {"conf_thres": 0.001, "iou_thres": 0.6, "max_det": 100},
}
THERMAL_GATE = {
    "throttled_mask": 0x7,
    "min_median_freq_khz": 1_750_000,
    "max_temp_c": 80.0,
}


# ─── preprocessing (byte-identical to scripts/training/yolov5s/transforms.py) ──

def letterbox(img: np.ndarray, new_shape: int) -> tuple[np.ndarray, float, tuple[float, float]]:
    h0, w0 = img.shape[:2]
    r = min(new_shape / h0, new_shape / w0)
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
    dw = (new_shape - new_unpad[0]) / 2
    dh = (new_shape - new_unpad[1]) / 2
    if (w0, h0) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=LETTERBOX_COLOR
    )
    return img, r, (dw * 2, dh * 2)


def to_tensor(img: np.ndarray, layout: str) -> np.ndarray:
    """HWC uint8 BGR -> float32 RGB in [0,1], NCHW or NHWC."""
    img = img[:, :, ::-1]
    if layout == "nhwc":
        return np.ascontiguousarray(img)[None].astype(np.float32) / 255.0
    img = np.ascontiguousarray(img.transpose(2, 0, 1))
    return img[None].astype(np.float32) / 255.0


def preprocess(path: Path, size: int, layout: str) -> tuple[np.ndarray, float, tuple[float, float]]:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"cv2.imread returned None for {path}")
    img, r, pad = letterbox(img, size)
    return to_tensor(img, layout), r, pad


# ─── post-processing ──────────────────────────────────────────────────────────

def _xywh2xyxy(x: np.ndarray) -> np.ndarray:
    y = np.empty_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def _nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> list[int]:
    """Greedy IoU NMS, same semantics as torchvision.ops.nms."""
    order = scores.argsort()[::-1]
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    keep: list[int] = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-12)
        order = rest[iou <= iou_thres]
    return keep


def nms_yolov5(pred: np.ndarray, conf_thres: float, iou_thres: float, max_det: int) -> np.ndarray:
    """Replicates yolov5.utils.general.non_max_suppression at its defaults
    (multi_label=False, agnostic=False, max_nms=30000), which is how
    `eval_suite/predict.py` calls it.

    Input  (N, 5+nc) xywh + objectness + class scores, letterboxed pixel space.
    Output (M, 6)    xyxy + conf + class index.
    """
    x = pred[pred[:, 4] > conf_thres]
    if not x.shape[0]:
        return np.zeros((0, 6), dtype=np.float32)
    x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
    box = _xywh2xyxy(x[:, :4])
    j = x[:, 5:].argmax(1)
    conf = x[np.arange(x.shape[0]), 5 + j]
    keep_mask = conf > conf_thres
    box, conf, j = box[keep_mask], conf[keep_mask], j[keep_mask]
    if not box.shape[0]:
        return np.zeros((0, 6), dtype=np.float32)
    if box.shape[0] > 30000:  # max_nms
        idx = conf.argsort()[::-1][:30000]
        box, conf, j = box[idx], conf[idx], j[idx]
    offsets = j.astype(np.float32) * 7680.0  # class offset, agnostic=False
    keep = _nms_numpy(box + offsets[:, None], conf, iou_thres)[:max_det]
    return np.concatenate(
        [box[keep], conf[keep, None], j[keep, None].astype(np.float32)], axis=1
    ).astype(np.float32)


def unletterbox(boxes: np.ndarray, r: float, pad: tuple[float, float], w0: int, h0: int) -> np.ndarray:
    """Letterboxed xyxy -> original-image xyxy. Same transform as
    `eval_suite/predict.py`, which copies it verbatim from evaluation.py."""
    dw, dh = pad
    b = boxes.copy()
    b[:, [0, 2]] = (b[:, [0, 2]] - dw / 2) / r
    b[:, [1, 3]] = (b[:, [1, 3]] - dh / 2) / r
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, w0)
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, h0)
    return b


def postprocess(
    out: np.ndarray,
    decode: str,
    regime: dict,
    r: float,
    pad,
    w0: int,
    h0: int,
    apply_conf: bool = True,
):
    """Decode raw model output to original-image detections.

    *apply_conf* exists because the two use sites want different contracts for
    the NMS-free head. A deployment would drop low-confidence boxes, so the
    latency path filters (realistic per-frame cost). The parity path must NOT:
    `scripts/training/yolo26n/eval_suite/predict.py` emits every one of the
    in-graph top-k rows regardless of score ("NMS-free: conf/iou thresholds
    unused"), and a parity run that filtered would be comparing two different
    contracts and reporting the difference as a hardware discrepancy.
    """
    if decode == "end2end":
        # YOLO26: (1, max_det, 6) = xyxy + score + cls, already top-k ranked
        # and deduplicated in-graph. Post-processing is a rescale only.
        d = out[0]
        if apply_conf:
            d = d[d[:, 4] > regime["conf_thres"]]
        if not d.shape[0]:
            return np.zeros((0, 6), dtype=np.float32)
        boxes = unletterbox(d[:, :4], r, pad, w0, h0)
        return np.concatenate([boxes, d[:, 4:6]], axis=1)
    if decode == "nms":
        d = nms_yolov5(out[0], regime["conf_thres"], regime["iou_thres"], regime["max_det"])
        if not d.shape[0]:
            return d
        boxes = unletterbox(d[:, :4], r, pad, w0, h0)
        return np.concatenate([boxes, d[:, 4:6]], axis=1)
    if decode == "classifier":
        logits = out[0]
        e = np.exp(logits - logits.max())
        probs = e / e.sum()
        k = min(5, probs.shape[0])
        top = np.argpartition(-probs, k - 1)[:k]
        return np.stack([top.astype(np.float32), probs[top]], axis=1)
    raise ValueError(f"unknown decode: {decode}")


# ─── runtimes ─────────────────────────────────────────────────────────────────

class OnnxRuntimeRunner:
    name = "onnxruntime"

    def __init__(self, model_dir: Path, threads: int, no_mkldnn: bool = False) -> None:
        del no_mkldnn  # ORT has its own dispatch; the flag is TorchScript-only
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(
            str(model_dir / "model.onnx"), so, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.sess.get_inputs()[0].name
        self.version = ort.__version__
        # ORT applies its own graph optimizations at session creation; recorded
        # so the "unoptimized model" claim in the thesis stays precise.
        self.opt_level = "ORT_ENABLE_ALL"

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.sess.run(None, {self.input_name: x})[0]


class TorchScriptRunner:
    name = "torchscript"

    def __init__(self, model_dir: Path, threads: int, no_mkldnn: bool = False) -> None:
        import torch

        # On this device PyTorch's MULTI-THREADED CPU path dies with SIGILL
        # (rc 132) while the single-threaded path is fine. The Cortex-A72 is
        # ARMv8.0: no `asimddp`, no `asimdhp`. The oneDNN/ACL backend that
        # torch dispatches multi-threaded convolutions to appears to select a
        # kernel assuming those extensions. Disabling mkldnn forces the
        # reference/NNPACK path, which is ISA-safe.
        if no_mkldnn:
            torch.backends.mkldnn.enabled = False
        self.mkldnn_enabled = bool(torch.backends.mkldnn.enabled)

        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        torch.set_grad_enabled(False)
        self.torch = torch
        self.module = torch.jit.load(str(model_dir / "model.torchscript"), map_location="cpu")
        self.module.eval()
        self.version = torch.__version__
        self.opt_level = (
            f"jit.freeze (applied at export); mkldnn={'on' if self.mkldnn_enabled else 'off'}"
        )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        with self.torch.no_grad():
            return self.module(self.torch.from_numpy(x)).numpy()


RUNNERS = {"onnxruntime": OnnxRuntimeRunner, "torchscript": TorchScriptRunner}


# ─── timing ───────────────────────────────────────────────────────────────────

def time_loop(fn, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        fn()
    out: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1000.0)
    return out


def plan_iters(probe_ms: float, budget_s: float, nominal: int) -> tuple[int, int]:
    """Choose (warmup, iters) so one cell fits a wall-clock budget.

    The protocol's nominal 20 warm-up / 100 measured is affordable for the
    students but not for MegaDetector, which is 831 GFLOPs at 1280x1280 — a
    fixed 120 iterations there would run for hours per cell and the sweep
    would never finish. So iterations shrink with per-iteration cost, down to
    a floor that still supports a median, and the realised `n` is recorded in
    every record so under-powered cells are visible rather than implied.
    """
    if probe_ms <= 0:
        return 3, nominal
    floor = 10 if probe_ms < 2_000 else (5 if probe_ms < 20_000 else 3)
    iters = int(budget_s * 1000.0 / probe_ms)
    iters = max(floor, min(nominal, iters))
    warmup = max(2, min(20, int(0.2 * iters)))
    return warmup, iters


def summarize(samples: list[float]) -> dict:
    s = sorted(samples)
    n = len(s)
    q1 = s[n // 4]
    q3 = s[(3 * n) // 4]
    return {
        "n": n,
        "median_ms": round(statistics.median(s), 4),
        "mean_ms": round(statistics.fmean(s), 4),
        "min_ms": round(s[0], 4),
        "p95_ms": round(s[min(n - 1, int(0.95 * n))], 4),
        "iqr_ms": round(q3 - q1, 4),
        "q1_ms": round(q1, 4),
        "q3_ms": round(q3, 4),
        "stdev_ms": round(statistics.pstdev(s), 4) if n > 1 else 0.0,
    }


# ─── monitor plumbing ─────────────────────────────────────────────────────────

def start_monitor(out_csv: Path, pid: int) -> subprocess.Popen | None:
    script = HERE / "pi" / "monitor.py"
    if not script.exists():
        script = HERE / "monitor.py"  # flat layout on the device
    if not script.exists():
        print("  ! monitor.py not found — cell will be ungated", file=sys.stderr)
        return None
    return subprocess.Popen(
        ["uv", "run", "--script", str(script), "--out", str(out_csv), "--pid", str(pid)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def stop_monitor(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def monitor_summary(csv_path: Path) -> dict:
    if not csv_path.exists():
        return {"available": False}
    temps: list[float] = []
    freqs: list[int] = []
    throttles: list[int] = []
    hwm: list[int] = []
    with csv_path.open() as f:
        header = f.readline().strip().split(",")
        fi = [i for i, c in enumerate(header) if c.startswith("freq")]
        ti, thi, hi = header.index("temp_c"), header.index("throttled"), header.index("vmhwm_kb")
        for line in f:
            p = line.strip().split(",")
            if len(p) != len(header):
                continue
            try:
                temps.append(float(p[ti]))
                freqs.extend(int(p[i]) for i in fi if int(p[i]) > 0)
                throttles.append(int(p[thi]))
                if int(p[hi]) > 0:
                    hwm.append(int(p[hi]))
            except ValueError:
                continue
    if not temps:
        return {"available": False}
    bits = 0
    for t in throttles:
        if t > 0:
            bits |= t
    return {
        "available": True,
        "samples": len(temps),
        "temp_max_c": max(temps),
        "temp_median_c": statistics.median(temps),
        "freq_median_khz": int(statistics.median(freqs)) if freqs else -1,
        "freq_min_khz": min(freqs) if freqs else -1,
        "throttled_bits": bits,
        "vmhwm_max_kb": max(hwm) if hwm else -1,
    }


def apply_gate(mon: dict) -> dict:
    """The validity gate. A cell measured while throttled is not a measurement
    of the model, so it is flagged rather than silently kept."""
    if not mon.get("available"):
        return {"valid": None, "reasons": ["no monitor data"]}
    reasons = []
    if mon["throttled_bits"] & THERMAL_GATE["throttled_mask"]:
        reasons.append(f"throttled bits 0x{mon['throttled_bits']:x}")
    if 0 < mon["freq_median_khz"] < THERMAL_GATE["min_median_freq_khz"]:
        reasons.append(f"median freq {mon['freq_median_khz']} kHz below floor")
    if mon["temp_max_c"] >= THERMAL_GATE["max_temp_c"]:
        reasons.append(f"peak temp {mon['temp_max_c']} C")
    return {"valid": not reasons, "reasons": reasons}


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--runtime", required=True, choices=sorted(RUNNERS))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--invocation", type=int, default=1)
    ap.add_argument("--regime", default="deployment", choices=sorted(REGIMES))
    ap.add_argument("--exports", type=Path, default=HERE / "exports")
    ap.add_argument("--subset", type=Path, default=HERE / "benchmark_subset")
    ap.add_argument("--out", type=Path, default=HERE / "results" / "latency_raw.jsonl")
    ap.add_argument("--monitor-dir", type=Path, default=HERE / "results" / "monitor")
    ap.add_argument("--warmup", type=int, default=PROTOCOL["warmup_iters"])
    ap.add_argument("--iters", type=int, default=PROTOCOL["measure_iters"])
    ap.add_argument(
        "--budget-s",
        type=float,
        default=45.0,
        help="wall-clock budget per timed window; iteration counts shrink to fit "
        "(see plan_iters). 0 disables adaptive sizing and uses --iters verbatim.",
    )
    ap.add_argument(
        "--no-mkldnn",
        action="store_true",
        help="disable PyTorch's mkldnn/oneDNN backend. Needed on this device: the "
        "multi-threaded TorchScript path otherwise dies with SIGILL on the "
        "ARMv8.0 Cortex-A72. Recorded in the output record.",
    )
    ap.add_argument(
        "--sustained",
        action="store_true",
        help="thermal-drift run: sustained_iters of W_infer, reporting first-100 "
        "vs last-100 medians and the peak temperature reached",
    )
    args = ap.parse_args()

    model_dir = args.exports / args.model
    manifest = json.loads((model_dir / "manifest.json").read_text())
    imgsz = manifest["image_size"]
    layout = manifest.get("input_layout", "nchw")
    decode = manifest["decode"]
    regime = REGIMES[args.regime]

    cell = f"{args.model}__{args.runtime}__t{args.threads}__inv{args.invocation}"
    if args.sustained:
        cell += "__sustained"
    print(f"=== {cell} ===")

    args.monitor_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    mon_csv = args.monitor_dir / f"monitor_{cell}.csv"

    # ── fixed inputs ─────────────────────────────────────────────────────────
    median_img = args.subset / "fixed" / "median_real.jpg"
    ax_img = args.subset / "fixed" / "ax_visio.jpeg"
    raw_median = cv2.imread(str(median_img))
    h0, w0 = raw_median.shape[:2]
    x, r, pad = preprocess(median_img, imgsz, layout)
    print(f"    input {x.shape} from {w0}x{h0}  layout={layout}  decode={decode}")

    t_load0 = time.perf_counter()
    runner = RUNNERS[args.runtime](model_dir, args.threads, no_mkldnn=args.no_mkldnn)
    load_ms = (time.perf_counter() - t_load0) * 1000.0
    print(f"    {args.runtime} {runner.version} loaded in {load_ms:.0f} ms")

    monitor = start_monitor(mon_csv, os.getpid())
    windows: dict[str, dict] = {}
    try:
        # W_infer — the MLPerf-scoped headline.
        nominal = PROTOCOL["sustained_iters"] if args.sustained else args.iters
        t_probe = time.perf_counter()
        raw_out = runner(x)
        probe_ms = (time.perf_counter() - t_probe) * 1000.0
        if args.budget_s > 0:
            warmup, iters = plan_iters(
                probe_ms, args.budget_s * (4 if args.sustained else 1), nominal
            )
        else:
            warmup, iters = args.warmup, nominal
        print(f"    probe {probe_ms:.1f} ms -> warmup={warmup} iters={iters}")
        samples = time_loop(lambda: runner(x), warmup, iters)
        windows["infer"] = summarize(samples)
        print(f"    W_infer   {windows['infer']['median_ms']:9.2f} ms "
              f"(IQR {windows['infer']['iqr_ms']:.2f})")

        if args.sustained and len(samples) >= 20:
            # Below ~20 samples the first/last windows overlap and the drift
            # figure is 0 by construction rather than by measurement.
            k = max(5, len(samples) // 6)
            first, last = samples[:k], samples[-k:]
            windows["infer"]["sustained"] = {
                "window_iters": k,
                "first_median_ms": round(statistics.median(first), 4),
                "last_median_ms": round(statistics.median(last), 4),
                "drift_pct": round(
                    100.0 * (statistics.median(last) - statistics.median(first))
                    / max(statistics.median(first), 1e-9),
                    2,
                ),
            }
            print(f"    drift     {windows['infer']['sustained']['drift_pct']:+.2f}% "
                  f"over {iters} iters")
        else:
            # W_pre — decode + letterbox, on a representative image and on a
            # native AX Visio still (the real device's decode cost).
            windows["pre_median"] = summarize(
                time_loop(lambda: preprocess(median_img, imgsz, layout), 5, min(50, iters * 2, 50))
            )
            print(f"    W_pre     {windows['pre_median']['median_ms']:9.2f} ms  "
                  f"({w0}x{h0})")
            if ax_img.exists():
                ax_raw = cv2.imread(str(ax_img))
                windows["pre_axvisio"] = summarize(
                    time_loop(lambda: preprocess(ax_img, imgsz, layout), 3, 20)
                )
                windows["pre_axvisio"]["source_shape"] = list(ax_raw.shape[:2])
                print(f"    W_pre_ax  {windows['pre_axvisio']['median_ms']:9.2f} ms  "
                      f"({ax_raw.shape[1]}x{ax_raw.shape[0]} AX Visio still)")

            # W_post — on a captured raw output, so the decode cost is isolated
            # from the forward pass.
            windows["post"] = summarize(
                time_loop(
                    lambda: postprocess(raw_out, decode, regime, r, pad, w0, h0),
                    5,
                    50,
                )
            )
            n_det = len(postprocess(raw_out, decode, regime, r, pad, w0, h0))
            windows["post"]["detections"] = int(n_det)
            print(f"    W_post    {windows['post']['median_ms']:9.2f} ms  "
                  f"({n_det} detections @ conf {regime['conf_thres']})")

            # W_e2e — one wall-clock loop, not the sum of the medians.
            def _e2e() -> None:
                xx, rr, pp = preprocess(median_img, imgsz, layout)
                oo = runner(xx)
                postprocess(oo, decode, regime, rr, pp, w0, h0)

            windows["e2e"] = summarize(time_loop(_e2e, max(2, warmup // 4), max(3, iters // 2)))
            print(f"    W_e2e     {windows['e2e']['median_ms']:9.2f} ms")
    finally:
        stop_monitor(monitor)

    mon = monitor_summary(mon_csv)
    gate = apply_gate(mon)
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    record = {
        "cell": cell,
        "model": args.model,
        "label": manifest.get("label"),
        "runtime": args.runtime,
        "runtime_version": runner.version,
        "runtime_opt": runner.opt_level,
        "no_mkldnn": args.no_mkldnn,
        "threads": args.threads,
        "invocation": args.invocation,
        "regime": args.regime,
        "regime_params": regime,
        "sustained": args.sustained,
        "image_size": imgsz,
        "input_shape": manifest["input_shape"],
        "decode": decode,
        "params": manifest.get("params"),
        "gflops": manifest.get("gflops"),
        "weights_source": manifest.get("weights_source"),
        "parity_valid": manifest.get("parity_valid"),
        "artifact_bytes": manifest.get("artifacts", {})
        .get("onnx" if args.runtime == "onnxruntime" else "torchscript", {})
        .get("bytes"),
        "model_load_ms": round(load_ms, 1),
        "windows": windows,
        "peak_rss_mb": round(peak_rss_mb, 1),
        "monitor": mon,
        "gate": gate,
        "protocol": {
            **PROTOCOL,
            "warmup": warmup,
            "iters": iters,
            "nominal_iters": nominal,
            "budget_s": args.budget_s,
            "probe_ms": round(probe_ms, 2),
            "adaptive": args.budget_s > 0,
        },
        "host": {
            "node": platform.node(),
            "machine": platform.machine(),
            "kernel": platform.release(),
            "python": platform.python_version(),
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with args.out.open("a") as f:
        f.write(json.dumps(record) + "\n")

    verdict = "VALID" if gate["valid"] else f"INVALID ({'; '.join(gate['reasons'])})"
    print(f"    peak RSS {peak_rss_mb:.0f} MB   temp max {mon.get('temp_max_c', '?')} C   [{verdict}]")
    return 0 if gate["valid"] is not False else 3


if __name__ == "__main__":
    raise SystemExit(main())
