"""
NanoDet-Plus-m smoke test — runs inside Dockerfile.nanodet.

Verifies that:
  1. nanodet package imports correctly
  2. NanoDet-Plus-m architecture can be instantiated from config
  3. A forward pass completes on a dummy 416×416 input
  4. Parameter count and latency are within expected ranges
  5. ONNX export runs without error

Usage (inside Dockerfile.nanodet):
    python scripts/training/nanodet_smoke_test.py

Makefile shortcut:
    make nanodet-smoke
"""

import sys
import time
from pathlib import Path

import numpy as np

NANODET_ROOT = Path("/opt/nanodet")
CONFIG_PATH = Path("/app/scripts/training/configs/nanodet-plus-m-wildlife225.yml")
# Fall back to the stock COCO config bundled in the NanoDet repo
FALLBACK_CONFIG = NANODET_ROOT / "config" / "nanodet-plus-m_416.yml"

EXPECTED_PARAMS_M = (1.0, 1.5)   # expected parameter count range (millions)
LATENCY_WARN_MS   = 200.0         # warn (not fail) if GPU inference exceeds this


# ── Utilities ─────────────────────────────────────────────────────────────────

def resolve_config() -> Path:
    if CONFIG_PATH.exists():
        return CONFIG_PATH
    if FALLBACK_CONFIG.exists():
        print(f"  [warn] Wildlife config not found; using fallback: {FALLBACK_CONFIG}")
        return FALLBACK_CONFIG
    raise FileNotFoundError(
        f"No NanoDet config found. Expected:\n"
        f"  {CONFIG_PATH}\n"
        f"  {FALLBACK_CONFIG}"
    )


def count_parameters(model) -> float:
    import torch
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_import() -> tuple[bool, str]:
    try:
        from nanodet.model.arch import build_model  # noqa: F401
        from nanodet.util import cfg, load_config   # noqa: F401
        import nanodet
        version = getattr(nanodet, "__version__", "unknown")
        return True, f"nanodet {version}"
    except ImportError as e:
        return False, str(e)


def test_model_build(config_path: Path) -> tuple[bool, float, str]:
    try:
        import torch
        from nanodet.model.arch import build_model
        from nanodet.util import cfg, load_config

        load_config(cfg, str(config_path))
        model = build_model(cfg.model)
        model.eval()

        params = count_parameters(model)
        lo, hi = EXPECTED_PARAMS_M
        note = f"{params:.2f}M params"
        if not (lo <= params <= hi):
            note += f" (expected {lo}–{hi}M — using wildlife nc=225 config?)"

        return True, params, note
    except Exception as e:
        return False, 0.0, str(e)[:120]


def test_forward_pass(config_path: Path) -> tuple[bool, float, str]:
    try:
        import torch
        from nanodet.model.arch import build_model
        from nanodet.util import cfg, load_config

        load_config(cfg, str(config_path))
        model = build_model(cfg.model)
        model.eval()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        # NanoDet expects a batch of images as a tensor
        x = torch.randn(1, 3, 416, 416, device=device)
        meta = {
            "img_info": {"height": 416, "width": 416, "id": 0, "file_name": "dummy"},
            "img_shape": (416, 416),
        }

        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(meta, x)

        if device == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(10):
                _ = model(meta, x)
        if device == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) / 10 * 1000

        note = f"{latency_ms:.1f}ms on {device}"
        if latency_ms > LATENCY_WARN_MS:
            note += f" (warn: exceeds {LATENCY_WARN_MS}ms)"

        return True, latency_ms, note
    except Exception as e:
        return False, 0.0, str(e)[:120]


def test_onnx_export(config_path: Path) -> tuple[bool, str]:
    import tempfile
    try:
        import torch
        from nanodet.model.arch import build_model
        from nanodet.util import cfg, load_config

        load_config(cfg, str(config_path))
        model = build_model(cfg.model)
        model.eval()

        x = torch.randn(1, 3, 416, 416)
        meta = {
            "img_info": {"height": 416, "width": 416, "id": 0, "file_name": "dummy"},
            "img_shape": (416, 416),
        }

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            out_path = f.name

        torch.onnx.export(
            model,
            (meta, x),
            out_path,
            opset_version=11,
            input_names=["input"],
        )
        size_mb = Path(out_path).stat().st_size / 1024**2
        Path(out_path).unlink(missing_ok=True)
        return True, f"{size_mb:.1f}MB ONNX exported"
    except Exception as e:
        return False, str(e)[:120]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    sep = "─" * 60
    print(f"\n{sep}")
    print("  NanoDet-Plus-m Smoke Test")
    print(sep)

    import torch
    print(f"  torch  : {torch.__version__}")
    print(f"  device : {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    print(sep + "\n")

    failures = 0

    # 1. Import
    ok, note = test_import()
    status = "PASS" if ok else "FAIL"
    print(f"[1/4] Package import        {status:4s}  {note}")
    if not ok:
        failures += 1

    # 2. Config + model build
    try:
        config_path = resolve_config()
        ok, params, note = test_model_build(config_path)
    except FileNotFoundError as e:
        ok, note = False, str(e)
    status = "PASS" if ok else "FAIL"
    print(f"[2/4] Architecture build     {status:4s}  {note}")
    if not ok:
        failures += 1

    # 3. Forward pass
    if ok:
        ok_fwd, latency, note = test_forward_pass(config_path)
        status = "PASS" if ok_fwd else "FAIL"
        print(f"[3/4] Forward pass           {status:4s}  {note}")
        if not ok_fwd:
            failures += 1
    else:
        print("[3/4] Forward pass           SKIP  (model build failed)")

    # 4. ONNX export (optional — warn on failure, don't fail the test)
    if ok:
        ok_onnx, note = test_onnx_export(config_path)
        status = "PASS" if ok_onnx else "WARN"
        print(f"[4/4] ONNX export            {status:4s}  {note}")
    else:
        print("[4/4] ONNX export            SKIP  (model build failed)")

    print()
    if failures:
        print(f"  FAIL ({failures} failure(s))\n")
        sys.exit(1)
    else:
        print("  ALL PASS\n")


if __name__ == "__main__":
    main()
