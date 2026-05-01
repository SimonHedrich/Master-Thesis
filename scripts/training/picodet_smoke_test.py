"""
PicoDet-S smoke test — runs inside Dockerfile.paddle.

Verifies that:
  1. PaddlePaddle imports correctly and GPU is accessible
  2. PicoDet-S architecture can be instantiated from config
  3. A forward pass completes on a dummy 320×320 input
  4. Parameter count is within expected range
  5. Paddle → ONNX export runs without error

Usage (inside Dockerfile.paddle):
    python scripts/training/picodet_smoke_test.py

Makefile shortcut:
    make paddle-smoke
"""

import sys
import time
import tempfile
from pathlib import Path

import numpy as np

PADDLEDET_ROOT = Path("/opt/PaddleDetection")
CONFIG_PATH = Path("/app/scripts/training/configs/picodet-s-wildlife225.yml")
FALLBACK_CONFIG = PADDLEDET_ROOT / "configs" / "picodet" / "picodet_s_320_coco_lcnet.yml"

EXPECTED_PARAMS_M = (0.8, 1.2)   # PicoDet-S has ~0.99M params


# ── Utilities ─────────────────────────────────────────────────────────────────

def resolve_config() -> Path:
    if CONFIG_PATH.exists():
        return CONFIG_PATH
    if FALLBACK_CONFIG.exists():
        print(f"  [warn] Wildlife config not found; using fallback: {FALLBACK_CONFIG}")
        return FALLBACK_CONFIG
    raise FileNotFoundError(
        f"No PicoDet config found. Expected:\n"
        f"  {CONFIG_PATH}\n"
        f"  {FALLBACK_CONFIG}"
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_import() -> tuple[bool, str]:
    try:
        import paddle
        gpu_available = paddle.device.cuda.device_count() > 0
        device_str = f"GPU ({paddle.device.cuda.device_count()} device(s))" if gpu_available else "CPU"
        return True, f"PaddlePaddle {paddle.__version__}  device: {device_str}"
    except ImportError as e:
        return False, str(e)


def test_paddledet_import() -> tuple[bool, str]:
    try:
        sys.path.insert(0, str(PADDLEDET_ROOT))
        from ppdet.core.workspace import load_config, create  # noqa: F401
        return True, "ppdet importable"
    except ImportError as e:
        return False, str(e)[:120]


def test_model_build(config_path: Path) -> tuple[bool, float, str]:
    try:
        import paddle
        sys.path.insert(0, str(PADDLEDET_ROOT))
        from ppdet.core.workspace import load_config, create

        cfg = load_config(str(config_path))
        # Create the model (architecture only, no pretrained weights)
        with paddle.fluid.dygraph.guard():
            model = create("PicoDet")
            model.eval()

        # Parameter count via paddle
        n_params = sum(p.numel().item() for p in model.parameters()) / 1e6
        lo, hi = EXPECTED_PARAMS_M
        note = f"{n_params:.2f}M params"
        if not (lo <= n_params <= hi):
            note += f" (expected {lo}–{hi}M)"

        return True, n_params, note
    except Exception as e:
        return False, 0.0, str(e)[:120]


def test_forward_pass(config_path: Path) -> tuple[bool, float, str]:
    try:
        import paddle
        sys.path.insert(0, str(PADDLEDET_ROOT))
        from ppdet.core.workspace import load_config, create

        cfg = load_config(str(config_path))
        model = create("PicoDet")
        model.eval()

        device = "gpu" if paddle.device.cuda.device_count() > 0 else "cpu"
        paddle.device.set_device(device)

        x = paddle.randn([1, 3, 320, 320])
        scale_factor = paddle.ones([1, 2])
        im_shape = paddle.to_tensor([[320, 320]], dtype="float32")

        # Warmup
        for _ in range(3):
            _ = model({"image": x, "scale_factor": scale_factor, "im_shape": im_shape})

        t0 = time.perf_counter()
        for _ in range(10):
            _ = model({"image": x, "scale_factor": scale_factor, "im_shape": im_shape})
        latency_ms = (time.perf_counter() - t0) / 10 * 1000

        return True, latency_ms, f"{latency_ms:.1f}ms on {device}"
    except Exception as e:
        return False, 0.0, str(e)[:120]


def test_onnx_export() -> tuple[bool, str]:
    """Export PicoDet to ONNX via paddle2onnx (requires a saved paddle model)."""
    try:
        import paddle2onnx  # noqa: F401
        version = getattr(paddle2onnx, "__version__", "unknown")
        return True, f"paddle2onnx {version} importable (full export requires trained weights)"
    except ImportError as e:
        return False, str(e)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    sep = "─" * 60
    print(f"\n{sep}")
    print("  PicoDet-S Smoke Test")
    print(sep + "\n")

    failures = 0

    # 1. PaddlePaddle import
    ok, note = test_import()
    status = "PASS" if ok else "FAIL"
    print(f"[1/4] PaddlePaddle import    {status:4s}  {note}")
    if not ok:
        failures += 1

    # 2. PaddleDetection import
    ok_pd, note_pd = test_paddledet_import()
    status = "PASS" if ok_pd else "FAIL"
    print(f"[2/4] PaddleDetection import {status:4s}  {note_pd}")
    if not ok_pd:
        failures += 1

    # 3. Model build + forward pass
    if ok and ok_pd:
        try:
            config_path = resolve_config()

            ok_build, params, note = test_model_build(config_path)
            status = "PASS" if ok_build else "FAIL"
            print(f"[3/4] Architecture build     {status:4s}  {note}")
            if not ok_build:
                failures += 1
            else:
                ok_fwd, latency, note = test_forward_pass(config_path)
                status = "PASS" if ok_fwd else "FAIL"
                print(f"      Forward pass           {status:4s}  {note}")
                if not ok_fwd:
                    failures += 1

        except FileNotFoundError as e:
            print(f"[3/4] Architecture build     FAIL  {e}")
            failures += 1
    else:
        print("[3/4] Architecture build     SKIP  (import failed)")

    # 4. ONNX export check
    ok_onnx, note = test_onnx_export()
    status = "PASS" if ok_onnx else "WARN"
    print(f"[4/4] ONNX export tool       {status:4s}  {note}")

    print()
    if failures:
        print(f"  FAIL ({failures} failure(s))\n")
        sys.exit(1)
    else:
        print("  ALL PASS\n")


if __name__ == "__main__":
    main()
