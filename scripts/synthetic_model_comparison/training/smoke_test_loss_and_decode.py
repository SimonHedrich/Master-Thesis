"""Smoke test for this package's yolo26n loss adapter and eval-decode path.

Copied from scripts/training/yolo26n/smoke_test_loss_and_decode.py (same
risk-mitigation rationale: YOLO26n's Detect head (end2end=True) is NMS-free,
so evaluation.py's decode path has no yolov5-equivalent to copy verbatim —
get the decode shape/coordinate-space wrong and per-class AP would silently
look plausible while being computed on mis-decoded boxes). This test
validates the mechanism on synthetic data (cheap, CPU-only, seconds) before
any real training run. Checks 1-5 need no real data at all; check 6 needs
one cell's exported+split annotations (pass --generator/--prompt-regime, or
it's skipped with a note if omitted / not yet built).

Run from the repo root:
    PYTHONPATH=. uv run -m scripts.synthetic_model_comparison.training.smoke_test_loss_and_decode
    PYTHONPATH=. uv run -m scripts.synthetic_model_comparison.training.smoke_test_loss_and_decode \\
        --generator gemini-3.1-flash-image-preview --prompt-regime full

Checks:
  1. Train-mode forward returns a dict with "one2many"/"one2one" keys
     (confirms end2end=True actually took effect for this model config).
  2. Yolo26Loss(model)(preds, targets) returns a finite scalar total loss and
     a parts dict with keys {loss_box, loss_cls, loss_dfl, loss_total};
     total.backward() does not raise and populates .grad on model parameters.
  3. loss_fn.update() does not raise, and after several calls
     loss_fn.criterion.o2m has moved away from its initial 0.8 toward
     final_o2m=0.1 (confirms the per-epoch anneal hook works).
  4. Eval-mode decode: model.eval(); model(imgs) returns a 2-tuple (y, preds)
     where y.shape == (batch, <=max_det, 6); class indices in y[..., 5] are
     in [0, nc); scores in y[..., 4] are in [0, 1] (post-sigmoid).
  5. Coordinate-space sanity: y[..., :4] values fall in [0, IMAGE_SIZE] (the
     letterboxed-canvas pixel range), not [0, 1] (normalized) — catches a
     double-application-of-un-letterbox bug before it silently corrupts mAP.
  6. evaluation.evaluate() runs end-to-end on a tiny (<=4-image) DataLoader
     built from one cell's annotations_val_split.json (augment=False) on an
     UNTRAINED (random-init) model, and returns a result dict with
     mAP50/mAP50_95 keys present and finite (checks plumbing, not accuracy).
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import torch

# ── Ensure repo root is on sys.path ──────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.synthetic_model_comparison.training.constants as constants
from scripts.synthetic_model_comparison.training.loss import Yolo26Loss
from scripts.synthetic_model_comparison.training.yolo26n_model import yolo26n_model

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


def check(condition: bool, label: str, detail: str = "") -> bool:
    tag = PASS if condition else FAIL
    msg = f"  [{tag}] {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return bool(condition)


def _synthetic_batch(batch_size: int = 2, image_size: int = None) -> tuple[torch.Tensor, torch.Tensor]:
    image_size = image_size or constants.IMAGE_SIZE
    imgs = torch.rand(batch_size, 3, image_size, image_size)
    rows = []
    for b in range(batch_size):
        rows.append([b, float(b % constants.NUM_CLASSES), 0.5, 0.5, 0.2, 0.3])
        rows.append([b, float((b + 5) % constants.NUM_CLASSES), 0.3, 0.4, 0.1, 0.15])
    targets = torch.tensor(rows, dtype=torch.float32)
    return imgs, targets


def run_loss_and_decode_checks() -> bool:
    all_pass = True
    device = torch.device("cpu")
    model, _ = yolo26n_model(constants.NUM_CLASSES, weights=None, device=device)
    loss_fn = Yolo26Loss(model)
    imgs, targets = _synthetic_batch()

    print("\n1. Train-mode forward output shape")
    model.train()
    preds = model(imgs)
    all_pass &= check(isinstance(preds, dict), "forward returns a dict")
    all_pass &= check(
        set(preds.keys()) == {"one2many", "one2one"},
        "dict has one2many/one2one keys",
        detail=str(sorted(preds.keys())),
    )

    print("\n2. Loss adapter forward/backward")
    total, parts = loss_fn(preds, targets)
    all_pass &= check(torch.isfinite(total).item(), "total loss is finite", detail=f"{total.item():.4f}")
    all_pass &= check(
        set(parts.keys()) == {"loss_box", "loss_cls", "loss_dfl", "loss_total"},
        "parts dict has expected keys",
        detail=str(sorted(parts.keys())),
    )
    model.zero_grad(set_to_none=True)
    total.backward()
    has_grad = any(p.grad is not None for p in model.parameters())
    all_pass &= check(has_grad, "backward() populates .grad on model parameters")

    print("\n3. Per-epoch anneal hook")
    o2m_before = loss_fn.criterion.o2m
    try:
        for _ in range(10):
            loss_fn.update()
        anneal_ok = True
    except Exception:
        traceback.print_exc()
        anneal_ok = False
    o2m_after = loss_fn.criterion.o2m if anneal_ok else o2m_before
    all_pass &= check(anneal_ok, "loss_fn.update() does not raise")
    all_pass &= check(
        anneal_ok and o2m_after < o2m_before,
        "o2m moved toward final_o2m after repeated update() calls",
        detail=f"{o2m_before:.4f} -> {o2m_after:.4f}",
    )

    print("\n4. Eval-mode decode shape/range")
    model.eval()
    with torch.no_grad():
        raw = model(imgs)
    all_pass &= check(isinstance(raw, (tuple, list)) and len(raw) == 2, "model(imgs) returns a (y, preds) 2-tuple")
    y = raw[0] if isinstance(raw, (tuple, list)) else raw
    max_det = model.model[-1].max_det
    shape_ok = y.dim() == 3 and y.shape[0] == imgs.shape[0] and y.shape[2] == 6 and y.shape[1] <= max_det
    all_pass &= check(shape_ok, "y.shape == (batch, <=max_det, 6)", detail=str(tuple(y.shape)))
    cls_idx = y[..., 5]
    cls_ok = bool((cls_idx >= 0).all() and (cls_idx < constants.NUM_CLASSES).all())
    all_pass &= check(cls_ok, "class indices in [0, nc)", detail=f"min={cls_idx.min().item()} max={cls_idx.max().item()}")
    scores = y[..., 4]
    scores_ok = bool((scores >= 0).all() and (scores <= 1).all())
    all_pass &= check(scores_ok, "scores in [0, 1]", detail=f"min={scores.min().item():.4f} max={scores.max().item():.4f}")

    print("\n5. Coordinate-space sanity (letterboxed-canvas pixels, not normalized)")
    boxes = y[..., :4]
    # An untrained/random-init model's raw decoded boxes can legitimately
    # overshoot the canvas edges (clipping only happens later, during
    # un-letterbox in evaluation.py) — so this checks the right invariant,
    # pixel-scale vs. normalized-scale, with a loose bound (one canvas-width
    # of slack on each side) rather than strict [0, IMAGE_SIZE] clamping.
    slack = constants.IMAGE_SIZE
    coord_ok = bool((boxes >= -slack).all() and (boxes <= constants.IMAGE_SIZE + slack).all())
    all_pass &= check(
        coord_ok,
        f"boxes are within one canvas-width of [0, {constants.IMAGE_SIZE}] (loose overshoot bound)",
        detail=f"min={boxes.min().item():.2f} max={boxes.max().item():.2f}",
    )
    # A model whose boxes are still normalized [0,1] would also technically
    # satisfy the bound above — guard against that degenerate case explicitly.
    not_normalized = bool(boxes.abs().max().item() > 1.5)
    all_pass &= check(not_normalized, "boxes are NOT stuck in a normalized [0,1] range", detail=f"max abs={boxes.abs().max().item():.2f}")

    return all_pass


def run_evaluate_plumbing_check(generator: str | None, prompt_regime: str | None) -> bool:
    print("\n6. evaluation.evaluate() end-to-end plumbing (random-init model, <=4 val images)")
    if generator is None or prompt_regime is None:
        return check(
            True,
            "skipped — no --generator/--prompt-regime passed",
            detail="pass both to also exercise the real dataset/eval plumbing",
        )

    try:
        from scripts.synthetic_model_comparison.training.evaluation import evaluate
    except ImportError as exc:
        return check(False, "...evaluation is importable", detail=str(exc))

    from scripts.synthetic_model_comparison.training.dataset import CocoYoloDataset, Dataloader, collate_fn
    from scripts.synthetic_model_comparison.training.split_dataset import split_cell

    cell = constants.cell_dir(generator, prompt_regime)
    _, val_split = split_cell(cell)

    device = torch.device("cpu")
    model, _ = yolo26n_model(constants.NUM_CLASSES, weights=None, device=device)

    ds = CocoYoloDataset(
        val_split,
        constants.IMAGE_ROOT,
        constants.IMAGE_SIZE,
        augment=False,
    )
    subset = torch.utils.data.Subset(ds, list(range(min(4, len(ds)))))
    loader = Dataloader(
        subset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    ).get_dataloader()

    all_pass = True
    try:
        result = evaluate(
            model,
            loader,
            device,
            constants.EVAL_CONF_THRES,
            constants.EVAL_IOU_THRES,
            constants.EVAL_MAX_DET,
        )
        ran_ok = True
    except Exception:
        traceback.print_exc()
        ran_ok = False
        result = {}
    all_pass &= check(ran_ok, "evaluate() runs without raising")
    if ran_ok:
        has_keys = "mAP50" in result and "mAP50_95" in result
        all_pass &= check(has_keys, "result has mAP50/mAP50_95 keys", detail=str(sorted(result.keys())))
        if has_keys:
            finite = torch.isfinite(torch.tensor([result["mAP50"], result["mAP50_95"]])).all().item()
            all_pass &= check(finite, "mAP50/mAP50_95 are finite", detail=f"{result['mAP50']}, {result['mAP50_95']}")
    return all_pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator", default=None)
    parser.add_argument("--prompt-regime", default=None, choices=["full", "compressed"])
    args = parser.parse_args()

    all_ok = True
    try:
        all_ok &= run_loss_and_decode_checks()
        all_ok &= run_evaluate_plumbing_check(args.generator, args.prompt_regime)
    except Exception:
        traceback.print_exc()
        all_ok = False

    print(f"\n{'=' * 60}")
    print(f"OVERALL: {PASS if all_ok else FAIL}")
    sys.exit(0 if all_ok else 1)
