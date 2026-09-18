#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=2.0.0",
#   "onnx>=1.17.0",
#   "onnxruntime>=1.20.0",
#   "onnxslim>=0.1.50",
#   "pnnx>=20250000",
# ]
# ///
"""
Stage 1b — Export verification gate + NCNN conversion (host, isolated env).

Runs in its own PEP 723 environment rather than the project venv, for the same
reason `scripts/literature/` does: `pyproject.toml` pins torch to the
`pytorch-cu130` index and this host may be training, so adding onnxruntime /
onnxslim / pnnx to the shared lockfile would be both unnecessary and risky.
This script never imports torch or any repo code — it works purely off the
artifacts `1-export_models.py` left behind.

Three jobs:

1. **The export gate.** Load `reference_input.npy` / `reference_output_0.npy`
   and check the exported ONNX reproduces the PyTorch reference within
   `1e-4`. A failure here is an export bug, not a device finding, and must be
   caught before anything ships to the Pi.
2. **onnxslim inspection.** Report whether graph-slimming would change the
   node count and the outputs. The *shipped* artifact stays the unmodified
   export — the thesis reports "a single runtime measurement of the
   unoptimized, full-precision model" (Literature Review 2.3.2), and shipping
   a slimmed graph would muddy that. Pass `--ship-slim` to override.
3. **NCNN conversion** from TorchScript via pnnx. Best-effort: NCNN is a third
   runtime for the matrix, not a prerequisite, and a conversion failure is
   recorded as a portability finding rather than aborting the run.

Usage:
    uv run --script scripts/benchmark/1b-convert_and_verify.py
    uv run --script scripts/benchmark/1b-convert_and_verify.py --models yolo26n-direct
    uv run --script scripts/benchmark/1b-convert_and_verify.py --no-ncnn

Outputs (per model dir under scripts/benchmark/exports/):
    model.ncnn.param, model.ncnn.bin, gate.json, manifest.json (updated)
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# Scale-aware: max_abs <= ATOL + RTOL * max|ref|. See constants.PARITY for
# why a bare absolute tolerance is wrong for these outputs.
GATE_ATOL = 1e-5
GATE_RTOL = 1e-5
EXPORT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "benchmark" / "exports"


def _delta(a: np.ndarray, b: np.ndarray) -> dict:
    if a.shape != b.shape:
        return {"shape_mismatch": [list(a.shape), list(b.shape)]}
    af = a.astype(np.float64)
    d = np.abs(af - b.astype(np.float64))
    scale = float(np.abs(af).max())
    bound = GATE_ATOL + GATE_RTOL * scale
    return {
        "max_abs": float(d.max()),
        "mean_abs": float(d.mean()),
        "ref_absmax": scale,
        "bound": bound,
        "passed": bool(d.max() <= bound),
    }


def _run_onnx(path: Path, x: np.ndarray, threads: int = 4) -> np.ndarray:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    return sess.run(None, {name: x})[0]


def _node_count(path: Path) -> int:
    import onnx

    return len(onnx.load(str(path)).graph.node)


def _convert_ncnn(model_dir: Path, input_shape: list[int]) -> dict:
    """TorchScript -> NCNN via pnnx. Best-effort; failures are recorded."""
    ts = model_dir / "model.torchscript"
    if not ts.exists():
        return {"ok": False, "reason": "no TorchScript artifact to convert from"}

    shape = "[" + ",".join(str(d) for d in input_shape) + "]"
    # pnnx writes a pile of intermediates next to its input; run it in a temp
    # dir and lift out only the two files ncnn actually loads.
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        local = tmpdir / "model.pt"
        shutil.copy2(ts, local)
        cmd = ["pnnx", str(local), f"inputshape={shape}", "fp16=0", "optlevel=0"]
        t0 = time.time()
        try:
            proc = subprocess.run(
                cmd, cwd=tmpdir, capture_output=True, text=True, timeout=3600
            )
        except FileNotFoundError:
            return {"ok": False, "reason": "pnnx executable not found"}
        except subprocess.TimeoutExpired:
            return {"ok": False, "reason": "pnnx timed out after 3600s"}

        param = tmpdir / "model.ncnn.param"
        binf = tmpdir / "model.ncnn.bin"
        if not (param.exists() and binf.exists()):
            tail = (proc.stderr or proc.stdout or "")[-800:]
            return {
                "ok": False,
                "reason": f"pnnx produced no .ncnn.param/.bin (exit {proc.returncode})",
                "log_tail": tail,
            }
        shutil.copy2(param, model_dir / "model.ncnn.param")
        shutil.copy2(binf, model_dir / "model.ncnn.bin")
        return {
            "ok": True,
            "seconds": round(time.time() - t0, 1),
            "param_bytes": (model_dir / "model.ncnn.param").stat().st_size,
            "bin_bytes": (model_dir / "model.ncnn.bin").stat().st_size,
        }


def process(model_dir: Path, *, do_ncnn: bool, ship_slim: bool) -> dict:
    name = model_dir.name
    print(f"\n=== {name} ===")
    manifest = json.loads((model_dir / "manifest.json").read_text())

    x = np.load(model_dir / "reference_input.npy")
    ref = np.load(model_dir / "reference_output_0.npy")
    onnx_path = model_dir / "model.onnx"
    print(f"    reference input {x.shape}  output {ref.shape}")

    result: dict = {"model": name, "gate_atol": GATE_ATOL, "gate_rtol": GATE_RTOL}

    # ── 1. the gate ──────────────────────────────────────────────────────────
    t0 = time.time()
    got = _run_onnx(onnx_path, x)
    d = _delta(ref, got)
    passed = d.get("passed", False)
    result["onnx"] = {**d, "seconds": round(time.time() - t0, 2)}
    status = "PASS" if passed else "FAIL"
    print(
        f"    gate  ONNX vs PyTorch: max_abs={d.get('max_abs'):.3e} "
        f"(bound {d.get('bound', float('nan')):.3e}, |ref|max={d.get('ref_absmax', 0):.1f})  [{status}]"
    )

    # ── 2. onnxslim inspection (not shipped by default) ──────────────────────
    try:
        import onnxslim

        slim_path = model_dir / "model.slim.onnx"
        n_before = _node_count(onnx_path)
        onnxslim.slim(str(onnx_path), str(slim_path))
        n_after = _node_count(slim_path)
        slim_out = _run_onnx(slim_path, x)
        sd = _delta(ref, slim_out)
        result["slim"] = {
            "nodes_before": n_before,
            "nodes_after": n_after,
            "bytes": slim_path.stat().st_size,
            **sd,
            "shipped": bool(ship_slim),
        }
        print(
            f"    slim  nodes {n_before} -> {n_after}, "
            f"max_abs={sd.get('max_abs'):.3e}  (shipped={ship_slim})"
        )
        if ship_slim and sd.get("passed", False):
            shutil.copy2(slim_path, onnx_path)
            print("    slim graph promoted to model.onnx")
    except Exception as exc:  # noqa: BLE001 — informational step only
        result["slim"] = {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}
        print(f"    slim  skipped ({type(exc).__name__})")

    # ── 3. NCNN ──────────────────────────────────────────────────────────────
    if do_ncnn and "ncnn" in _formats(name):
        print("    converting to NCNN via pnnx ...")
        result["ncnn"] = _convert_ncnn(model_dir, manifest["input_shape"])
        if result["ncnn"]["ok"]:
            mb = result["ncnn"]["bin_bytes"] / 1e6
            print(f"    ncnn  model.ncnn.param + .bin ({mb:.1f} MB)")
        else:
            print(f"    ncnn  NOT AVAILABLE — {result['ncnn']['reason']}")
    else:
        result["ncnn"] = {"ok": False, "reason": "not requested for this model"}

    (model_dir / "gate.json").write_text(json.dumps(result, indent=2))

    manifest["gate"] = result
    if result["ncnn"].get("ok"):
        manifest.setdefault("artifacts", {})["ncnn"] = {
            "path": "model.ncnn.param",
            "bytes": result["ncnn"]["param_bytes"] + result["ncnn"]["bin_bytes"],
        }
    (model_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return result


def _formats(name: str) -> tuple[str, ...]:
    # Mirrors constants.MODELS[...]["formats"] without importing the project
    # module (this script runs in an isolated env with no repo on sys.path).
    return {
        "yolo26n-direct": ("torchscript", "onnx", "ncnn"),
        "yolo26n-kd": ("torchscript", "onnx", "ncnn"),
        "yolov5s": ("torchscript", "onnx", "ncnn"),  # ncnn attempted only under --ncnn
        "speciesnet": ("torchscript", "onnx"),
        "megadetector": ("torchscript", "onnx"),
    }.get(name, ("torchscript", "onnx"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default="all")
    ap.add_argument(
        "--ncnn",
        action="store_true",
        help="attempt NCNN conversion via pnnx. Off by default: pnnx crashes on "
        "both detectors (reports/embedded_benchmark/ncnn_conversion_finding.md). "
        "Kept so the finding stays reproducible.",
    )
    ap.add_argument(
        "--ship-slim",
        action="store_true",
        help="promote the onnxslim-optimized graph to model.onnx (default: keep "
        "the unmodified export, so the benchmark measures the unoptimized model)",
    )
    ap.add_argument("--exports", type=Path, default=EXPORT_DIR)
    args = ap.parse_args()

    dirs = sorted(d for d in args.exports.iterdir() if (d / "manifest.json").exists())
    if args.models != "all":
        wanted = set(args.models.split(","))
        dirs = [d for d in dirs if d.name in wanted]
    if not dirs:
        print(f"no exported models under {args.exports} — run 1-export_models.py first")
        return 2

    results = [process(d, do_ncnn=args.ncnn, ship_slim=args.ship_slim) for d in dirs]

    print("\n=== gate summary ===")
    failed = []
    for r in results:
        ok = r["onnx"]["passed"]
        ncnn = "ncnn:yes" if r["ncnn"].get("ok") else "ncnn:no "
        print(
            f"  {r['model']:<18} onnx max_abs={r['onnx'].get('max_abs', float('nan')):.3e} "
            f"{'PASS' if ok else 'FAIL'}  {ncnn}"
        )
        if not ok:
            failed.append(r["model"])
    if failed:
        print(f"\n  GATE FAILED for {failed} — do not ship these to the device")
        return 1
    print("\n  all exported graphs reproduce the PyTorch reference within "
          f"{GATE_ATOL:g} + {GATE_RTOL:g}*|ref|max")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
