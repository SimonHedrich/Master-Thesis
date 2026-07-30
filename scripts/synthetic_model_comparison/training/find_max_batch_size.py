"""Find the maximum batch size for YOLO26n training on the current device.

The script doubles the batch size each iteration and runs a single forward +
backward training step with synthetic data.  When CUDA OOM is hit, the last
successful batch size is reported.

Set constants.BATCH_SIZE to the largest power-of-two reported here before a
real (non-smoke) training run.

Run instructions
----------------
From the repository root (dependencies are managed by uv):

    PYTHONPATH=. uv run -m scripts.synthetic_model_comparison.training.find_max_batch_size

With options:

    PYTHONPATH=. uv run -m scripts.synthetic_model_comparison.training.find_max_batch_size \\
        --start 8 --image-size 320 --device cuda

CPU smoke-test (OOM cannot occur; use --max to limit iterations):

    PYTHONPATH=. uv run -m scripts.synthetic_model_comparison.training.find_max_batch_size \\
        --device cpu --max 8
"""
from __future__ import annotations

import argparse
import gc
import sys

import torch

import scripts.synthetic_model_comparison.training.constants as constants
from scripts.synthetic_model_comparison.training.loss import Yolo26Loss
from scripts.synthetic_model_comparison.training.yolo26n_model import model_optimizer, yolo26n_model


def _synthetic_batch(
    batch_size: int, image_size: int, num_classes: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    imgs = torch.rand(batch_size, 3, image_size, image_size, device=device)
    # Two dummy annotations per image: [batch_idx, cls, cx, cy, w, h]
    rows = []
    for i in range(batch_size):
        rows.append([float(i), 0.0, 0.5, 0.5, 0.4, 0.4])
        rows.append([float(i), float(num_classes - 1), 0.2, 0.3, 0.1, 0.15])
    targets = torch.tensor(rows, dtype=torch.float32, device=device)
    return imgs, targets


def _try_batch_size(
    batch_size: int,
    image_size: int,
    num_classes: int,
    weights_path,
    device: torch.device,
) -> bool:
    """Return True if the batch size fits, False on OOM."""
    model, _ = yolo26n_model(num_classes, weights_path, device)
    optimizer = model_optimizer(model)
    loss_fn = Yolo26Loss(model)

    try:
        model.train()
        imgs, targets = _synthetic_batch(batch_size, image_size, num_classes, device)
        preds = model(imgs)
        total, _ = loss_fn(preds, targets)
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()
        return True
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower():
            return False
        raise
    finally:
        del model, optimizer, loss_fn
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()


def _vram_used_gb(device: torch.device) -> str:
    if device.type != "cuda":
        return "n/a"
    return f"{torch.cuda.memory_allocated(device) / 1e9:.2f} GB"


def main() -> None:
    parser = argparse.ArgumentParser(description="Find max YOLO26n training batch size.")
    parser.add_argument("--start", type=int, default=1, help="Starting batch size (default: 1)")
    parser.add_argument("--image-size", type=int, default=constants.IMAGE_SIZE, help=f"Image size (default: {constants.IMAGE_SIZE})")
    parser.add_argument("--device", type=str, default=None, help="Device string, e.g. 'cuda' or 'cpu' (default: auto)")
    parser.add_argument("--max", type=int, default=4096, help="Safety cap on batch size (default: 4096)")
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights = constants.PRETRAINED_WEIGHTS if constants.PRETRAINED_WEIGHTS.exists() else None

    print(f"\nProbing max batch size on {device} (image_size={args.image_size}, nc={constants.NUM_CLASSES})\n")

    last_good: int | None = None
    first_bad: int | None = None
    batch_size = args.start

    while batch_size <= args.max:
        ok = _try_batch_size(batch_size, args.image_size, constants.NUM_CLASSES, weights, device)
        if ok:
            mem = _vram_used_gb(device)
            print(f"batch {batch_size:>5}: OK    {mem}")
            last_good = batch_size
            batch_size *= 2
        else:
            print(f"batch {batch_size:>5}: OOM")
            first_bad = batch_size
            break

    print()
    if last_good is None:
        print("No batch size succeeded — check your GPU memory and model setup.")
        sys.exit(1)

    if first_bad is None:
        print(
            f"Max batch size: {last_good} "
            f"(reached --max cap of {args.max} without OOM; true max may be higher)"
        )
        return

    lo, hi = last_good, first_bad
    while hi - lo > 1:
        mid = (lo + hi) // 2
        ok = _try_batch_size(mid, args.image_size, constants.NUM_CLASSES, weights, device)
        status = f"OK    {_vram_used_gb(device)}" if ok else "OOM"
        print(f"batch {mid:>5}: {status}  (binary search)")
        if ok:
            lo = mid
        else:
            hi = mid

    print(f"\nMax batch size: {lo}")


if __name__ == "__main__":
    main()
