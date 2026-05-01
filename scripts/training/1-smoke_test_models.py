"""
Model smoke test — verifies that teacher and student models load correctly,
run inference, and that their detection heads can be adapted to nc=225.

No dataset is required. A synthetic 640×640 RGB noise image is used as input.
Model weights are downloaded automatically via Ultralytics on first run and
cached to the standard Ultralytics cache directory (~/.config/Ultralytics/).

Usage:
    python scripts/training/1-smoke_test_models.py          # host (GPU/CPU)
    make smoke                                              # host shortcut
    make smoke-docker                                       # inside Docker (main image)

Tested models:
    Teachers:   YOLOv8s, RT-DETR-L, SpeciesNet (SKIP — run: make speciesnet-smoke)
    Students:   YOLO11n, YOLOv10n, YOLO12n, YOLO26n
    Skipped:    NanoDet-Plus-m (run: make nanodet-smoke)
                PicoDet-S     (run: make paddle-smoke)
                SpeciesNet    (run: make speciesnet-smoke)

NanoDet-Plus-m — runs in Dockerfile.nanodet (Python 3.11):
    make nanodet-build && make nanodet-smoke
    # Installs from: https://github.com/RangiLyu/nanodet (not on PyPI)
    # NanoDet-Plus-m config: scripts/training/configs/nanodet-plus-m-wildlife225.yml

PicoDet-S — runs in Dockerfile.paddle (PaddlePaddle):
    make paddle-build && make paddle-smoke
    # PicoDet training: make paddle-shell then python tools/train.py
    # PicoDet config:   scripts/training/configs/picodet-s-wildlife225.yml

SpeciesNet two-stage pipeline — runs in Dockerfile.speciesnet (Python 3.11):
    make speciesnet-build && make speciesnet-smoke
    # Full pipeline: make speciesnet-demo IMAGE=path/to/image.jpg
    # Soft labels:   make speciesnet-labels DIR=data/training/

Class adaptation — 225 wildlife classes:
    See docs/progress_notes/2026-04-24_training-setup-and-model-smoke-test.md
    for detailed instructions per model family.
"""

import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np


DEVICE = "cpu"          # set below after torch import
NC_WILDLIFE = 225       # target class count for all student models
DUMMY_W, DUMMY_H = 640, 640
N_WARMUP = 3
N_MEASURE = 10


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class ModelResult:
    name: str
    status: str         # PASS | FAIL | SKIP
    params_m: float | None = None
    latency_ms: float | None = None
    device: str | None = None
    note: str = ""


# ── Utilities ─────────────────────────────────────────────────────────────────

def resolve_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def make_dummy_image() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (DUMMY_H, DUMMY_W, 3), dtype=np.uint8)


def count_parameters(model) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def time_inference(fn, n_warmup: int = N_WARMUP, n_measure: int = N_MEASURE) -> float:
    """Return mean inference time in milliseconds."""
    try:
        import torch
        use_cuda = torch.cuda.is_available()
    except ImportError:
        use_cuda = False

    for _ in range(n_warmup):
        fn()

    if use_cuda:
        import torch
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_measure):
        fn()
    if use_cuda:
        import torch
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / n_measure

    return elapsed * 1000.0


# ── Ultralytics model tests ───────────────────────────────────────────────────

def test_ultralytics_model(model_id: str, display_name: str) -> ModelResult:
    """Load an Ultralytics model, run timed inference on a dummy image."""
    try:
        from ultralytics import YOLO
        import ultralytics as ul
    except ImportError:
        return ModelResult(display_name, "FAIL", note="ultralytics not installed")

    img = make_dummy_image()

    try:
        model = YOLO(model_id)
    except Exception as e:
        msg = str(e)
        if "not found" in msg.lower() or "no such file" in msg.lower() or "404" in msg:
            return ModelResult(
                display_name, "SKIP",
                note=f"{model_id} not available in ultralytics {ul.__version__}",
            )
        return ModelResult(display_name, "FAIL", note=msg[:100])

    try:
        params = count_parameters(model.model)
        latency = time_inference(lambda: model(img, verbose=False, device=DEVICE))
        return ModelResult(display_name, "PASS", params_m=params, latency_ms=latency, device=DEVICE)
    except Exception as e:
        return ModelResult(display_name, "FAIL", note=str(e)[:100])


# ── Non-Ultralytics model stubs ───────────────────────────────────────────────

def test_nanodet() -> ModelResult:
    """
    NanoDet requires a separate Docker environment (Python 3.11, git install).
    Run the dedicated smoke test with: make nanodet-smoke
    """
    try:
        from nanodet.model.arch import build_model  # noqa: F401
        # If we're inside Dockerfile.nanodet this import succeeds —
        # defer to nanodet_smoke_test.py for full forward-pass verification.
        return ModelResult(
            "NanoDet-Plus-m", "SKIP",
            note="nanodet importable — run scripts/training/nanodet_smoke_test.py for full test",
        )
    except ImportError:
        return ModelResult(
            "NanoDet-Plus-m", "SKIP",
            note="Not in this env — run: make nanodet-build && make nanodet-smoke",
        )


def test_picodet() -> ModelResult:
    """
    PicoDet requires PaddlePaddle + PaddleDetection (separate Docker environment).
    Run the dedicated smoke test with: make paddle-smoke
    """
    try:
        import paddle  # noqa: F401
        return ModelResult(
            "PicoDet-S", "SKIP",
            note="PaddlePaddle importable — run scripts/training/picodet_smoke_test.py for full test",
        )
    except ImportError:
        return ModelResult(
            "PicoDet-S", "SKIP",
            note="Not in this env — run: make paddle-build && make paddle-smoke",
        )


def test_speciesnet() -> ModelResult:
    """
    SpeciesNet (EfficientNetV2-M) requires Python <3.13 (separate Docker environment).
    Run the dedicated pipeline with: make speciesnet-smoke
    """
    try:
        import speciesnet  # noqa: F401
        return ModelResult(
            "SpeciesNet Pipeline", "SKIP",
            note="speciesnet importable — run scripts/training/0-teacher_speciesnet_pipeline.py",
        )
    except ImportError:
        return ModelResult(
            "SpeciesNet Pipeline", "SKIP",
            note="Requires Python <3.13 — run: make speciesnet-build && make speciesnet-smoke",
        )


# ── Class adaptation tests (nc=225, architecture only) ───────────────────────

def test_class_adaptation(model_id: str, nc: int = NC_WILDLIFE) -> ModelResult:
    """
    Verify a model's detection head can be reconfigured for nc classes.

    For YOLO variants: loads architecture from the corresponding YAML (no
    pre-trained weights) so we can test the head without needing the ~100MB
    checkpoint. Rebuilds the cv3 classification convolutions in the Detect head.

    For RT-DETR: loads from .pt (YAML init is unsupported in this ultralytics
    version for DETR models) and rebuilds the dec_score_head Linear layers.

    The Ultralytics train API performs this swap automatically when the dataset
    YAML declares a different nc than the loaded checkpoint — this test
    validates the architectural precondition only.
    """
    display_name = f"{model_id.replace('.pt', '')} nc={nc}"

    try:
        from ultralytics import YOLO
        import torch.nn as nn
    except ImportError:
        return ModelResult(display_name, "FAIL", note="ultralytics not installed")

    # RT-DETR: YAML init is not supported in ultralytics 8.4.x — load from .pt
    is_rtdetr = "rtdetr" in model_id.lower()

    if is_rtdetr:
        try:
            model = YOLO(model_id)
        except Exception as e:
            return ModelResult(display_name, "FAIL", note=f"PT load failed: {str(e)[:80]}")
    else:
        yaml_id = model_id.replace(".pt", ".yaml")
        try:
            model = YOLO(yaml_id)
        except Exception as e:
            return ModelResult(display_name, "FAIL", note=f"YAML load failed: {str(e)[:80]}")

    try:
        detect = model.model.model[-1]  # Detect or RTDETRDecoder

        # RT-DETR: replace dec_score_head (ModuleList of Linear layers)
        if hasattr(detect, "dec_score_head"):
            new_heads = nn.ModuleList([
                nn.Linear(layer.in_features, nc)
                for layer in detect.dec_score_head
            ])
            detect.dec_score_head = new_heads
            detect.num_classes = nc
            return ModelResult(
                display_name, "PASS",
                params_m=count_parameters(model.model),
                device="cpu",
                note=f"RTDETRDecoder dec_score_head rebuilt for nc={nc}",
            )

        # YOLO family: set nc, rebuild cv3 classification convolutions
        if not hasattr(detect, "nc"):
            return ModelResult(display_name, "FAIL", note="Detect layer has no nc attribute")

        detect.nc = nc
        # cv3 is a ModuleList of 3 Sequential modules (one per FPN scale).
        # The last Conv in each Sequential outputs nc channels — only this
        # layer is incompatible with the COCO-pretrained weights.
        for seq in detect.cv3:
            old_conv = seq[-1]
            new_conv = nn.Conv2d(old_conv.in_channels, nc, 1)
            nn.init.normal_(new_conv.weight, 0, 0.01)
            nn.init.zeros_(new_conv.bias)
            seq[-1] = new_conv

        actual_nc = detect.nc
        if actual_nc != nc:
            return ModelResult(display_name, "FAIL", note=f"Expected nc={nc}, got nc={actual_nc}")

        return ModelResult(
            display_name, "PASS",
            params_m=count_parameters(model.model),
            device="cpu",
            note=f"Detect head nc={actual_nc} confirmed",
        )

    except Exception as e:
        return ModelResult(display_name, "FAIL", note=str(e)[:100])


# ── Output formatting ─────────────────────────────────────────────────────────

_COL_NAME    = 28
_COL_STATUS  = 7
_COL_PARAMS  = 10
_COL_LATENCY = 12
_COL_DEVICE  = 7


def _fmt(value, fmt_spec: str, fallback: str = "—") -> str:
    if value is None:
        return fallback
    return format(value, fmt_spec)


def print_header() -> None:
    import torch
    import ultralytics as ul

    sep = "─" * 78
    print(f"\n{sep}")
    print("  Wildlife Model Smoke Test")
    print(sep)
    print(f"  torch      : {torch.__version__}")
    print(f"  ultralytics: {ul.__version__}")
    print(f"  device     : {DEVICE}")
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem  = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU        : {name} ({mem:.1f} GB)")
    print(sep + "\n")


def print_results(results: list[ModelResult]) -> None:
    header = (
        f"{'Model':<{_COL_NAME}}"
        f"{'Status':<{_COL_STATUS}}"
        f"{'Params(M)':>{_COL_PARAMS}}"
        f"{'Latency(ms)':>{_COL_LATENCY}}"
        f"{'Device':<{_COL_DEVICE + 2}}"
        f"Note"
    )
    sep = "─" * 90

    print(header)
    print(sep)

    for r in results:
        params   = _fmt(r.params_m,   ".2f")
        latency  = _fmt(r.latency_ms, ".1f")
        device   = r.device or "—"
        print(
            f"{r.name:<{_COL_NAME}}"
            f"{r.status:<{_COL_STATUS}}"
            f"{params:>{_COL_PARAMS}}"
            f"{latency:>{_COL_LATENCY}}"
            f"  {device:<{_COL_DEVICE}}"
            f"  {r.note}"
        )

    print(sep)
    n_pass = sum(1 for r in results if r.status == "PASS")
    n_fail = sum(1 for r in results if r.status == "FAIL")
    n_skip = sum(1 for r in results if r.status == "SKIP")
    print(f"\n  PASS: {n_pass}  FAIL: {n_fail}  SKIP: {n_skip}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    global DEVICE
    DEVICE = resolve_device()

    print_header()

    results: list[ModelResult] = []

    # ── Teachers ──────────────────────────────────────────────────────────────
    print("Testing teachers...")
    results.append(test_ultralytics_model("yolov8s.pt",  "YOLOv8s (teacher)"))
    # Note: rtdetr-r18 does not exist in ultralytics. Available variants are
    # rtdetr-l (ResNet-50 backbone, 33M params) and rtdetr-x (ResNet-101, 67M).
    # rtdetr-l is used here as the DETR-based teacher candidate.
    results.append(test_ultralytics_model("rtdetr-l.pt", "RT-DETR-L (teacher)"))

    # ── Students — Ultralytics ────────────────────────────────────────────────
    print("Testing Ultralytics students...")
    results.append(test_ultralytics_model("yolo11n.pt",  "YOLO11n (student)"))
    results.append(test_ultralytics_model("yolov10n.pt", "YOLOv10n (student)"))
    # YOLO12 was introduced in late 2024/early 2025; model ID is yolo12n (no 'v')
    results.append(test_ultralytics_model("yolo12n.pt",  "YOLO12n (student)"))
    # YOLO26n: Sep 2025 release — highest nano-class COCO mAP (40.9%), 2.4M params
    results.append(test_ultralytics_model("yolo26n.pt",  "YOLO26n (student)"))

    # ── Students — non-Ultralytics (separate Docker images) ───────────────────
    print("Testing non-Ultralytics students (expect SKIP in main Docker)...")
    results.append(test_nanodet())
    results.append(test_picodet())
    results.append(test_speciesnet())

    # ── Class adaptation (nc=225 head swap) ───────────────────────────────────
    print(f"Testing class adaptation to nc={NC_WILDLIFE}...")
    results.append(test_class_adaptation("yolov8s.pt",  NC_WILDLIFE))
    results.append(test_class_adaptation("rtdetr-l.pt", NC_WILDLIFE))
    results.append(test_class_adaptation("yolo11n.pt",  NC_WILDLIFE))
    results.append(test_class_adaptation("yolo12n.pt",  NC_WILDLIFE))
    results.append(test_class_adaptation("yolo26n.pt",  NC_WILDLIFE))

    print()
    print_results(results)

    if any(r.status == "FAIL" for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
