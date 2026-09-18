#!/usr/bin/env python3
"""
Stage 1 — Export trained checkpoints to deployable artifacts (host, CPU only).

Reconstructs each architecture with this repo's own factories and checkpoint
loaders (the training pipelines bypass Ultralytics' high-level Model/Trainer
API, so the checkpoints are bare state_dicts and `YOLO("best.pt").export()`
does NOT work off them), then writes TorchScript and ONNX next to a reference
input/output pair.

The reference pair is what makes the export gate possible without installing
onnxruntime into the project environment: `1b-convert_and_verify.py` is a
standalone PEP 723 script that loads `reference_input.npy` /
`reference_output_*.npy` and checks the exported graphs against them in an
isolated environment.

Runs on CPU deliberately: ONNX export must trace the CPU path, and this host
may be training.

Usage:
    uv run python scripts/benchmark/1-export_models.py
    uv run python scripts/benchmark/1-export_models.py --models yolo26n-direct,yolov5s
    uv run python scripts/benchmark/1-export_models.py --models megadetector --no-torchscript

Outputs:
    scripts/benchmark/exports/<model>/{model.onnx,model.torchscript,
                                       reference_input.npy,reference_output_0.npy,
                                       manifest.json}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import constants as C  # noqa: E402

OPSET = 17


# ─── helpers ──────────────────────────────────────────────────────────────────

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class InferenceHead(nn.Module):
    """Return only the inference tensor from a detector's eval-mode output.

    Every model here returns either a bare tensor or a ``(inference_out,
    train_out)`` tuple in eval mode; ONNX and TorchScript both want a single
    deterministic output, and the train-mode side outputs are never used
    downstream (see `eval_suite/predict.py`, which also takes ``raw[0]``).
    """

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.inner(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out


def _reference_input(image_size: int, layout: str = "nchw") -> torch.Tensor:
    """The fixed median real test image, letterboxed to *image_size*.

    A real image rather than random noise, so the traced graph is exercised
    over realistic value ranges and the parity tolerances mean something.
    Falls back to a seeded pseudo-random tensor if the subset has not been
    built yet.

    *layout* is ``"nchw"`` for every model except SpeciesNet, whose classifier
    is an onnx2torch-converted TensorFlow graph that takes NHWC and transposes
    internally (see `constants.MODELS["speciesnet"]`).
    """
    fixed = C.SUBSET_DIR / "fixed" / "median_real.jpg"
    if not fixed.exists():
        print(f"  ! {fixed} missing — using a seeded random reference input")
        g = torch.Generator().manual_seed(C.SEED)
        t = torch.rand(1, 3, image_size, image_size, generator=g)
    else:
        import cv2

        from scripts.training.yolov5s import transforms

        img = cv2.imread(str(fixed))
        img, _, _ = transforms.letterbox(img, new_shape=image_size)
        t = transforms.to_tensor(img).unsqueeze(0)

    if layout == "nhwc":
        t = t.permute(0, 2, 3, 1).contiguous()
    return t


# ─── per-family model construction ────────────────────────────────────────────

def _build_yolo26n(spec: dict) -> tuple[nn.Module, str]:
    from scripts.training.yolo26n.yolo26n_model import yolo26n_model

    model, _ = yolo26n_model(
        num_classes=spec["num_classes"], weights=None, device=torch.device("cpu")
    )
    src = _load_state_dict_into(model, spec["checkpoint"])
    return model, src


def _build_yolov5s(spec: dict) -> tuple[nn.Module, str]:
    from scripts.training.yolov5s.yolov5s_model import yolov5s_model

    model, _ = yolov5s_model(
        num_classes=spec["num_classes"], weights=None, device=torch.device("cpu")
    )
    src = _load_state_dict_into(model, spec["checkpoint"])
    return model, src


def _build_speciesnet(spec: dict) -> tuple[nn.Module, str]:
    from speciesnet import DEFAULT_MODEL, SpeciesNet

    sn = SpeciesNet(DEFAULT_MODEL, components="classifier", geofence=False)
    model = sn.classifier.model
    src = _load_state_dict_into(model, spec["checkpoint"])
    return model.to("cpu"), src


def _build_megadetector(spec: dict) -> tuple[nn.Module, str]:
    from PytorchWildlife.models import detection as pw_detection

    md = pw_detection.MegaDetectorV5(device="cpu", pretrained=True)
    # MegaDetector's weights come with the wrapper; there is no separate
    # fine-tuned checkpoint in this project (it is never fine-tuned — see
    # scripts/training/megadet_speciesnet_ensemble/README.md).
    return md.model.float(), "pretrained (PytorchWildlife MegaDetectorV5)"


BUILDERS = {
    "yolo26n": _build_yolo26n,
    "yolov5s": _build_yolov5s,
    "speciesnet": _build_speciesnet,
    "megadetector": _build_megadetector,
}


def _snapshot_checkpoint(checkpoint: Path, out_dir: Path) -> tuple[Path, str] | tuple[None, None]:
    """Copy the source checkpoint into the export dir and return (path, sha256).

    A training run that is still in flight rewrites `best.pt` whenever
    validation improves. Exporting from the live file and then generating the
    host parity reference from it minutes later can silently use two different
    sets of weights — which reads as a hardware discrepancy in the parity
    check and is not one. Freezing a snapshot makes the export, the reference
    and the device all provably the same weights, and the recorded hash makes
    a mismatch detectable rather than mysterious.
    """
    import shutil

    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        return None, None
    out_dir.mkdir(parents=True, exist_ok=True)
    frozen = out_dir / "source_checkpoint.pt"
    shutil.copy2(checkpoint, frozen)
    return frozen, _sha256(frozen)


def _load_state_dict_into(model: nn.Module, checkpoint: Path) -> str:
    """Load *checkpoint* into *model*, or report an architecture-only fallback.

    Mirrors the strict=False / shape-filtered loading in
    `eval_suite/predict.py::load_checkpoint`, including its two accepted
    checkpoint layouts (training dict with a "model" key, or a raw state_dict).
    """
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        print(f"  ! checkpoint absent: {checkpoint}")
        print("    → architecture-only export (VALID for latency, INVALID for parity)")
        return "architecture-only"

    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "model" in raw:
        candidate = raw["model"]
        state = candidate.state_dict() if isinstance(candidate, nn.Module) else candidate
    else:
        state = raw.state_dict() if isinstance(raw, nn.Module) else raw

    own = model.state_dict()
    matched = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    model.load_state_dict(matched, strict=False)
    print(f"    loaded {len(matched)}/{len(own)} keys from {checkpoint.name}")
    if len(matched) < len(own):
        print(f"    ! {len(own) - len(matched)} keys skipped (missing / shape-mismatched)")
    return str(checkpoint.relative_to(REPO_ROOT)) if checkpoint.is_relative_to(REPO_ROOT) else str(checkpoint)


# ─── FLOPs ────────────────────────────────────────────────────────────────────

def _count_flops(model: nn.Module, example: torch.Tensor) -> float | None:
    """GFLOPs for one forward pass, via torch's built-in flop counter.

    Uses `torch.utils.flop_counter` rather than `thop` so no dependency is
    added to the project environment. Returns None if it is unavailable or
    raises (some traced heads confuse the dispatcher).
    """
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        return None
    try:
        counter = FlopCounterMode(display=False)
        with counter, torch.no_grad():
            model(example)
        # FlopCounterMode counts MACs*2 already.
        return counter.get_total_flops() / 1e9
    except Exception as exc:  # noqa: BLE001 — diagnostic only, never fatal
        print(f"    ! FLOPs count failed ({type(exc).__name__}: {exc})")
        return None


# ─── export ───────────────────────────────────────────────────────────────────

def export_one(name: str, spec: dict, out_root: Path, *, torchscript: bool, onnx: bool) -> dict:
    print(f"\n=== {name} ({spec['label']}) ===")
    t0 = time.time()

    out_dir = out_root / name
    frozen, ckpt_sha = _snapshot_checkpoint(spec["checkpoint"], out_dir)
    if frozen is not None:
        # Build from the frozen copy, never from the live training file.
        spec = {**spec, "checkpoint": frozen}
    model, weights_source = BUILDERS[spec["family"]](spec)
    model = InferenceHead(model).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    imgsz = spec["image_size"]
    layout = spec.get("input_layout", "nchw")
    example = _reference_input(imgsz, layout)
    print(f"    input {tuple(example.shape)}  weights={weights_source}")

    with torch.no_grad():
        ref_out = model(example)
    ref_np = ref_out.detach().cpu().numpy()
    print(f"    output {ref_np.shape} dtype={ref_np.dtype}")

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "reference_input.npy", example.numpy())
    np.save(out_dir / "reference_output_0.npy", ref_np)

    n_params = sum(p.numel() for p in model.parameters())
    gflops = _count_flops(model, example)

    artifacts: dict[str, dict] = {}

    if torchscript:
        ts_path = out_dir / "model.torchscript"
        print("    tracing TorchScript ...")
        with torch.no_grad():
            traced = torch.jit.trace(model, example, strict=False)
            traced = torch.jit.freeze(traced.eval())
        traced.save(str(ts_path))
        artifacts["torchscript"] = {
            "path": ts_path.name,
            "bytes": ts_path.stat().st_size,
            "sha256": _sha256(ts_path),
        }
        print(f"    wrote {ts_path.name} ({ts_path.stat().st_size / 1e6:.1f} MB)")

    if onnx:
        onnx_path = out_dir / "model.onnx"
        print(f"    exporting ONNX (opset {OPSET}, static batch-1) ...")
        with torch.no_grad():
            torch.onnx.export(
                model,
                example,
                str(onnx_path),
                input_names=["images"],
                output_names=["output0"],
                opset_version=OPSET,
                do_constant_folding=True,
                dynamo=False,  # static graph; the legacy exporter is the stable path here
            )
        artifacts["onnx"] = {
            "path": onnx_path.name,
            "bytes": onnx_path.stat().st_size,
            "sha256": _sha256(onnx_path),
        }
        print(f"    wrote {onnx_path.name} ({onnx_path.stat().st_size / 1e6:.1f} MB)")

    manifest = {
        "model": name,
        "label": spec["label"],
        "family": spec["family"],
        "decode": spec["decode"],
        "weights_source": (
            str(Path(spec["checkpoint"]).name) if frozen is not None else weights_source
        ),
        "source_checkpoint": str(C.MODELS[name]["checkpoint"]),
        "source_checkpoint_sha256": ckpt_sha,
        "frozen_snapshot": "source_checkpoint.pt" if frozen is not None else None,
        "parity_valid": weights_source != "architecture-only",
        "input_shape": list(example.shape),
        "input_layout": layout,
        "image_size": imgsz,
        "num_classes": spec["num_classes"],
        "output_shape": list(ref_np.shape),
        "params": n_params,
        "gflops": gflops,
        "opset": OPSET,
        "artifacts": artifacts,
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "exporter": {
            "torch": torch.__version__,
            "python": platform.python_version(),
            "host": platform.node(),
        },
        "export_seconds": round(time.time() - t0, 1),
    }
    with (out_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    print(
        f"    params={n_params / 1e6:.2f}M"
        + (f"  {gflops:.1f} GFLOPs" if gflops else "")
        + f"  ({manifest['export_seconds']}s)"
    )
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--models",
        default="all",
        help="comma-separated model keys from constants.MODELS, or 'all'",
    )
    ap.add_argument("--out", type=Path, default=C.EXPORT_DIR)
    ap.add_argument("--no-torchscript", action="store_true")
    ap.add_argument("--no-onnx", action="store_true")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    # This host may be training; do not contend for every core.
    torch.set_num_threads(min(4, torch.get_num_threads()))

    names = list(C.MODELS) if args.models == "all" else args.models.split(",")
    unknown = [n for n in names if n not in C.MODELS]
    if unknown:
        print(f"unknown model key(s): {unknown}", file=sys.stderr)
        print(f"available: {list(C.MODELS)}", file=sys.stderr)
        return 2

    manifests = []
    failed = []
    for name in names:
        try:
            manifests.append(
                export_one(
                    name,
                    C.MODELS[name],
                    args.out,
                    torchscript=not args.no_torchscript,
                    onnx=not args.no_onnx,
                )
            )
        except Exception as exc:  # noqa: BLE001 — one model failing must not abort the rest
            import traceback

            print(f"  !! {name} FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            failed.append(name)

    print("\n=== summary ===")
    for m in manifests:
        formats = "+".join(m["artifacts"])
        flag = "" if m["parity_valid"] else "  [architecture-only: latency valid, parity invalid]"
        print(f"  {m['model']:<18} {m['params'] / 1e6:6.2f}M  {formats:<20}{flag}")
    if failed:
        print(f"  FAILED: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
