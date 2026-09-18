"""Single source of truth for the on-device benchmark: paths, model registry, protocol.

Imported by the host-side scripts in this package (`0-`, `1-`, `4-`, `5-`).
The Pi-side scripts (`2-`, `3-`, `pi/monitor.py`) deliberately do NOT import
this module — they are PEP 723 standalone scripts that must run on a machine
where this repository is not checked out.  The protocol constants they need are
duplicated into their own headers and cross-referenced back here; the
`PROTOCOL` dict below is written into every artifact so any divergence is
detectable after the fact rather than silent.

See `docs/plans/2026-09-17_on-device-benchmarking-plan.md`.
"""
from __future__ import annotations

from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[2]

ANNOTATIONS_TEST_REAL = REPO_ROOT / "data" / "real" / "annotations_test.json"
ANNOTATIONS_TEST_SYNTH = REPO_ROOT / "data" / "synthetic" / "annotations_test.json"

# Stage 2 output — the fixed, seeded benchmark subset.
SUBSET_DIR = REPO_ROOT / "data" / "benchmark_subset"
SUBSET_ANN = SUBSET_DIR / "annotations_subset.json"
SUBSET_MANIFEST = SUBSET_DIR / "subset_manifest.json"

# Stage 1 output — deployable artifacts, gitignored.
EXPORT_DIR = REPO_ROOT / "scripts" / "benchmark" / "exports"

# Stage 6 output — the reported numbers.
REPORT_DIR = REPO_ROOT / "reports" / "embedded_benchmark"

# Where everything lands on the device (see README.md).
PI_HOST = "debian@sh-rpi-400.taile550ef.ts.net"
PI_ROOT = "~/benchmark"

# ─── Dataset ──────────────────────────────────────────────────────────────────

NUM_CLASSES = 225
SEED = 42  # matches the evaluation strategy's statistical-hygiene rules

# Subset composition — plan §4.4.
SUBSET_PER_BAND = 120  # bands A/B/C/D
SUBSET_NEGATIVES = 20
SUBSET_SYNTHETIC = 100
SUBSET_BANDS = ("A", "B", "C", "D")

# ─── Model registry ───────────────────────────────────────────────────────────
#
# `checkpoint` is the preferred source. When it is absent from this machine the
# exporter falls back to an architecture-only export (COCO-pretrained backbone,
# freshly-initialised 225-class head) and records
# `weights_source = "architecture-only"` in the manifest. That export is VALID
# FOR LATENCY (which depends on architecture and input shape, never on the
# values in the tensors) and INVALID FOR PARITY. `3-bench_parity.py` refuses to
# run against it.

MODELS: dict[str, dict] = {
    "yolo26n-direct": {
        "family": "yolo26n",
        "checkpoint": REPO_ROOT
        / "scripts/training/yolo26n/model_exports/yolo26n-bs32-20260910-212812/best.pt",
        "image_size": 640,
        "num_classes": NUM_CLASSES,
        "formats": ("torchscript", "onnx"),
        # YOLO26's Detect head is end2end=True: score-ranked top-k runs INSIDE
        # the graph, so post-processing is a coordinate rescale only.
        "decode": "end2end",
        "label": "YOLO26n direct-FT",
    },
    "yolo26n-kd": {
        "family": "yolo26n",
        "checkpoint": REPO_ROOT
        / "scripts/training/yolo26n/model_exports/yolo26n-kd-20260825-164250/best.pt",
        "image_size": 640,
        "num_classes": NUM_CLASSES,
        "formats": ("torchscript", "onnx"),
        "decode": "end2end",
        "label": "YOLO26n KD",
    },
    "yolov5s": {
        "family": "yolov5s",
        "checkpoint": REPO_ROOT
        / "scripts/training/yolov5s/model_exports/yolov5s-20260909-230900/best.pt",
        "image_size": 640,
        "num_classes": NUM_CLASSES,
        "formats": ("torchscript", "onnx"),
        # YOLOv5's NMS is external and is timed separately as W_post.
        "decode": "nms",
        "label": "YOLOv5s",
    },
    "speciesnet": {
        "family": "speciesnet",
        "checkpoint": REPO_ROOT
        / "scripts/training/teacher_finetune/model_exports"
        / "teacher-finetune-ff0.75-bs64-20260909-231034/best.pt",
        "image_size": 480,
        "num_classes": 2498,  # native SpeciesNet leaf taxonomy
        # SpeciesNet's classifier is an onnx2torch-converted TensorFlow graph:
        # it takes NHWC and transposes to NCHW as its first op. Verified
        # directly — (1, 480, 480, 3) -> (1, 2498).
        "input_layout": "nhwc",
        "formats": ("torchscript", "onnx"),
        "decode": "classifier",
        "label": "SpeciesNet classifier",
    },
    "megadetector": {
        "family": "megadetector",
        "checkpoint": Path.home() / ".cache/torch/hub/checkpoints/md_v5a.0.0.pt",
        "image_size": 1280,
        "num_classes": 3,  # animal / person / vehicle
        "formats": ("torchscript", "onnx"),
        "decode": "nms",
        "label": "MegaDetector v5a",
    },
}

# Models whose predictions can be scored against the 225-class subset GT.
# The two teacher components are timed but not scored: SpeciesNet is a
# classifier (no boxes) and MegaDetector is class-agnostic (3 classes, not 225).
PARITY_MODELS = ("yolo26n-direct", "yolo26n-kd", "yolov5s")

# ─── Measurement protocol ─────────────────────────────────────────────────────
#
# Plan §4.5. Sourced from docs/2026-03-12_research-and-experimentation-plan.md
# §4.2, extended where that protocol is silent.

PROTOCOL = {
    "batch_size": 1,  # embedded inference is sequential
    "warmup_iters": 20,
    "measure_iters": 100,
    "invocations": 3,  # independent processes per cell
    "thread_counts": (1, 2, 4),
    "headline_threads": 4,
    "sustained_iters": 600,  # thermal-drift run, 4 threads, ONNX Runtime
    "governor": "performance",
}

# Confidence regimes. `deployment` produces the headline latency and the
# 30 ms verdict; `eval` mirrors constants.EVAL_* and is used for the parity
# run only. Never mixed — the eval regime's NMS cost over 225 classes at
# conf 0.001 is pathological on an A72 and would misrepresent deployment cost.
REGIMES = {
    "deployment": {"conf_thres": 0.25, "iou_thres": 0.45, "max_det": 100},
    "eval": {"conf_thres": 0.001, "iou_thres": 0.6, "max_det": 100},
}

# Validity gate — a cell is invalid and re-run once after a cooldown if any
# of these trips. Plan §4.5.
THERMAL_GATE = {
    "throttled_mask": 0x7,  # under-voltage | arm-freq-capped | currently throttled
    "min_median_freq_khz": 1_750_000,
    "max_temp_c": 80.0,
    "cooldown_s": 120,
}

# ─── Targets (project design estimates, NOT vendor-certified) ─────────────────
#
# docs/2026-03-10_object-detection-models-for-embedded-systems.md §1.

TARGET_LATENCY_MS_QCS605 = 30.0
TARGET_PEAK_RSS_MB = 500.0

# docs/2026-03-09_hardware-proxy-selection.md rates the Pi 4/400 at -10% CPU
# vs. the QCS605 (the doc's text widens that to 10-15%), so the QCS605 is
# expected to take 0.85-0.90x the Pi 400's wall-clock time. CPU PATH ONLY:
# the Pi 400 has no analogue to the Hexagon 685 DSP and its VideoCore VI is
# far weaker than the Adreno 615.
QCS605_SCALE = (0.85, 0.90)
PI400_PASS_BAND_MS = (
    TARGET_LATENCY_MS_QCS605 / QCS605_SCALE[1],  # ~33.3 ms
    TARGET_LATENCY_MS_QCS605 / QCS605_SCALE[0],  # ~35.3 ms
)

# ─── Parity tolerances ────────────────────────────────────────────────────────
# Plan §4.7.

# Agreement is checked scale-aware, as `max_abs <= atol + rtol * max|ref|`,
# NOT against a bare absolute tolerance. The detectors' output tensors mix
# pixel coordinates (0-1280) with scores (0-1) in one array, so a fixed atol
# is meaningless: YOLO26n's ONNX export reproduces PyTorch to max_abs 1.5e-4
# on coordinates of magnitude ~640, i.e. a relative error of ~3e-7 — pure
# fp32 accumulation-order noise, which a bare 1e-4 atol would have flagged as
# an export bug. A genuine export bug moves outputs by O(1) or more and is
# caught comfortably by the scaled bound.
PARITY = {
    "t1_atol": 1e-5,
    "t1_rtol": 1e-5,
    "t2_min_match_rate": 0.99,
    "t2_iou": 0.99,
    "t2_max_score_delta": 0.01,
    "t3_max_map_delta": 0.002,
    "export_gate_atol": 1e-5,  # host-side gate, before anything ships
    "export_gate_rtol": 1e-5,
}

# NCNN (and with it the only GPU compute path on this device) is unavailable:
# pnnx crashes converting both detectors. Full evidence and disposition in
# reports/embedded_benchmark/ncnn_conversion_finding.md.
NCNN_STATUS = "unavailable: pnnx pass_level2 crash (see ncnn_conversion_finding.md)"
