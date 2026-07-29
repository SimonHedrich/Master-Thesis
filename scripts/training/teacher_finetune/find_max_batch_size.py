"""Find the maximum batch size for SpeciesNet classifier fine-tuning on the
current device.

Same doubling-until-OOM structure as `yolov5s/find_max_batch_size.py`, but
probes **training** (forward+backward) memory on the classifier at
`constants.IMAGE_SIZE`, not the inference-only batch size
`docs/2026-04-29_gpu_training_options.md` already estimated for SpeciesNet
soft-label generation (this fine-tuning run also needs gradients/optimizer
state, which that inference-only estimate did not account for).

The synthetic batch is built by running `preprocess_fn` once on a random noise
image (via `speciesnet_model()`'s real preprocessing path) and stacking that
one preprocessed array `batch_size` times — this sidesteps needing to know
whether SpeciesNet's internal array layout is NCHW or NHWC, since we never
construct the tensor shape by hand.

Run instructions
-----------------
Inside the SpeciesNet Docker container (same image used for fine-tuning):

    make speciesnet-start
    python -m scripts.training.teacher_finetune.find_max_batch_size

With options:

    python -m scripts.training.teacher_finetune.find_max_batch_size \\
        --start 8 --device cuda

CPU smoke-test (OOM cannot occur; use --max to limit iterations):

    python -m scripts.training.teacher_finetune.find_max_batch_size \\
        --device cpu --max 8
"""
from __future__ import annotations

import argparse
import gc
import sys

import numpy as np
import torch
from PIL import Image

import scripts.training.teacher_finetune.constants as constants
from scripts.training.teacher_finetune.loss import GroupedCrossEntropyLoss
from scripts.training.teacher_finetune.taxonomy import build_group_table
from scripts.training.teacher_finetune.teacher_model import model_optimizer, speciesnet_model


def _synthetic_arr(preprocess_fn) -> np.ndarray:
    noise = Image.fromarray(
        (np.random.rand(512, 512, 3) * 255).astype(np.uint8), mode="RGB"
    )
    arr = preprocess_fn(noise, [0.0, 0.0, 1.0, 1.0])
    if arr is None:
        raise RuntimeError("preprocess_fn returned None for a full-image synthetic crop")
    return arr


def _try_batch_size(
    batch_size: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: GroupedCrossEntropyLoss,
    one_arr: np.ndarray,
    device: torch.device,
) -> bool:
    """Return True if the batch size fits, False on OOM."""
    try:
        model.train()
        batch = np.stack([one_arr] * batch_size, axis=0)
        arrs = torch.from_numpy(batch).to(device)
        labels = torch.randint(0, constants.NUM_CLASSES_225, (batch_size,), device=device)

        logits = model(arrs)
        loss = loss_fn(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return True
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower():
            return False
        raise
    finally:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()


def _vram_used_gb(device: torch.device) -> str:
    if device.type != "cuda":
        return "n/a"
    return f"{torch.cuda.memory_allocated(device) / 1e9:.2f} GB"


def main() -> None:
    parser = argparse.ArgumentParser(description="Find max teacher_finetune training batch size.")
    parser.add_argument("--start", type=int, default=1, help="Starting batch size (default: 1)")
    parser.add_argument("--device", type=str, default=None, help="Device string, e.g. 'cuda' or 'cpu' (default: auto)")
    parser.add_argument("--max", type=int, default=1024, help="Safety cap on batch size (default: 1024)")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nProbing max batch size on {device} (image_size={constants.IMAGE_SIZE})\n")

    model, preprocess_fn, _labels = speciesnet_model(device)
    optimizer = model_optimizer(model)
    group_table, _levels = build_group_table()
    loss_fn = GroupedCrossEntropyLoss(group_table)
    one_arr = _synthetic_arr(preprocess_fn)

    last_good: int | None = None
    batch_size = args.start

    while batch_size <= args.max:
        ok = _try_batch_size(batch_size, model, optimizer, loss_fn, one_arr, device)
        if ok:
            mem = _vram_used_gb(device)
            print(f"batch {batch_size:>5}: OK    {mem}")
            last_good = batch_size
            batch_size *= 2
        else:
            print(f"batch {batch_size:>5}: OOM")
            break

    print()
    if last_good is None:
        print("No batch size succeeded — check your GPU memory and model setup.")
        sys.exit(1)
    else:
        print(f"Max batch size: {last_good}")


if __name__ == "__main__":
    main()
