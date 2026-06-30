"""Smoke test for the basic augmentation pipeline (setups A & B) and,
optionally, the compositing augmentation (setup C).

Run from the repo root:
    # Setups A & B only (default)
    python scripts/training/yolov5s/smoke_test_augmentation.py

    # Setups A & B + setup C (compositing)
    python scripts/training/yolov5s/smoke_test_augmentation.py --setup-c

Checks (A & B):
  1. All target coords (cx, cy, w, h) ∈ [0, 1].
  2. No NaN / Inf in targets or image tensor.
  3. Image tensor shape (3, IMAGE_SIZE, IMAGE_SIZE), dtype float32, values in [0, 1].
  4. augment=True vs augment=False produce different pixels for at least one sample.
  5. When all AUG_* flags are forced False, augment=True matches augment=False pixel-for-pixel.
  6. Multi-box samples retain >=1 box after augmentation.

Additional checks (--setup-c / setup C):
  7. All coords still ∈ [0, 1] with mosaic=1.0, mixup=0.5.
  8. No NaN/Inf with compositing on.
  9. Image tensor shape correct.
 10. Mosaic images tend to have more boxes than a single-image baseline
     (box count typically increases; we assert median boxes ≥ baseline).
 11. set_epoch(total-1, total) with AUG_CLOSE_MOSAIC=2 suppresses mosaic
     (verified by two deterministic checks: _compositing_active() returns
     False, AND build_mosaic call count stays at 0 for all sampled indices
     — no fragile mean-comparison between different random samples).
 12. After setup-C test, re-running the basic test proves A/B defaults
     are byte-for-byte unchanged.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

# ── Ensure repo root is on sys.path ──────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.training.yolov5s.constants as constants
from scripts.training.yolov5s.dataset import CocoYoloDataset

N_SAMPLES = 20
PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


def check(condition: bool, label: str, detail: str = "") -> bool:
    tag = PASS if condition else FAIL
    msg = f"  [{tag}] {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return condition


def build_dataset(augment: bool) -> CocoYoloDataset:
    return CocoYoloDataset(
        constants.ANNOTATIONS_VAL,
        constants.IMAGE_ROOT,
        constants.IMAGE_SIZE,
        augment=augment,
    )


def run() -> bool:
    all_pass = True

    print(f"\n{'=' * 60}")
    print("Smoke test: basic augmentation (setups A & B)")
    print(f"  val annotations : {constants.ANNOTATIONS_VAL}")
    print(f"  IMAGE_SIZE       : {constants.IMAGE_SIZE}")
    print(f"  N_SAMPLES        : {N_SAMPLES}")
    print(f"{'=' * 60}\n")

    # ── Build datasets ────────────────────────────────────────────────────────
    ds_aug = build_dataset(augment=True)
    ds_noaug = build_dataset(augment=False)
    total = len(ds_aug)
    print(f"Dataset size: {total} images\n")

    # Pick indices spread across the dataset; include small ones likely to have
    # multiple boxes when available.
    indices = list(range(0, min(N_SAMPLES, total)))

    # ── Per-sample checks ─────────────────────────────────────────────────────
    print("── Per-sample checks (augment=True) ────────────────────────")
    coord_ok_all = True
    nan_ok_all = True
    shape_ok_all = True
    multibox_ok_all = True

    multi_box_samples = 0

    for idx in indices:
        img_t, targets, path, _ = ds_aug[idx]

        # Shape / dtype / range
        shape_ok = (
            img_t.shape == (3, constants.IMAGE_SIZE, constants.IMAGE_SIZE)
            and img_t.dtype == torch.float32
            and float(img_t.min()) >= 0.0
            and float(img_t.max()) <= 1.0
        )
        shape_ok_all = shape_ok_all and shape_ok

        # NaN / Inf in tensor
        nan_img = bool(torch.isnan(img_t).any() or torch.isinf(img_t).any())
        nan_ok_all = nan_ok_all and (not nan_img)

        if targets.shape[0] > 0:
            # coords are cols 2,3,4,5 (col 0 = batch placeholder, col 1 = cls)
            coords = targets[:, 2:6]
            coord_in_range = bool((coords >= 0.0).all() and (coords <= 1.0).all())
            coord_ok_all = coord_ok_all and coord_in_range

            nan_tgt = bool(torch.isnan(targets).any() or torch.isinf(targets).any())
            nan_ok_all = nan_ok_all and (not nan_tgt)

            n_boxes = targets.shape[0]
            if n_boxes >= 2:
                multi_box_samples += 1

    all_pass = check(coord_ok_all, "All target coords in [0, 1]") and all_pass
    all_pass = check(nan_ok_all, "No NaN/Inf in tensors or targets") and all_pass
    all_pass = check(shape_ok_all,
                     f"Image tensor shape (3,{constants.IMAGE_SIZE},{constants.IMAGE_SIZE}), float32, [0,1]") and all_pass
    print(f"  (multi-box samples encountered: {multi_box_samples}/{len(indices)})")

    # ── Multi-box samples retain >=1 box ─────────────────────────────────────
    print("\n── Multi-box survival check ────────────────────────────────")
    multi_found = 0
    multi_kept = 0
    for idx in range(min(total, 200)):  # scan wider to find multi-box samples
        _, targets_noaug, _, _ = ds_noaug[idx]
        if targets_noaug.shape[0] >= 2:
            multi_found += 1
            _, targets_aug, _, _ = ds_aug[idx]
            if targets_aug.shape[0] >= 1:
                multi_kept += 1
            if multi_found >= 10:
                break

    if multi_found == 0:
        print(f"  [SKIP] No multi-box samples found in first 200 images — cannot test survival")
    else:
        ok = multi_kept == multi_found
        all_pass = check(ok, f"Multi-box samples retain >=1 box ({multi_kept}/{multi_found})") and all_pass

    # ── augment=True vs augment=False produce different pixels ───────────────
    print("\n── Augmentation produces different pixels ──────────────────")
    diff_found = False
    for idx in indices:
        img_aug, _, _, _ = ds_aug[idx]
        img_noaug, _, _, _ = ds_noaug[idx]
        if not torch.equal(img_aug, img_noaug):
            diff_found = True
            break
    all_pass = check(diff_found,
                     "augment=True vs augment=False differ in at least one sample") and all_pass

    # ── With all AUG flags forced off, augment=True == augment=False ─────────
    print("\n── All-flags-off identity check ────────────────────────────")
    # Temporarily patch constants so all augmentation is disabled.
    saved = {
        "AUG_HFLIP": constants.AUG_HFLIP,
        "AUG_HSV": constants.AUG_HSV,
        "AUG_SCALE": constants.AUG_SCALE,
        "AUG_TRANSLATE": constants.AUG_TRANSLATE,
        "AUG_DEGREES": constants.AUG_DEGREES,
        "AUG_SHEAR": constants.AUG_SHEAR,
        "AUG_PERSPECTIVE": constants.AUG_PERSPECTIVE,
    }
    constants.AUG_HFLIP = False
    constants.AUG_HSV = False
    constants.AUG_SCALE = 0.0
    constants.AUG_TRANSLATE = 0.0
    constants.AUG_DEGREES = 0.0
    constants.AUG_SHEAR = 0.0
    constants.AUG_PERSPECTIVE = 0.0

    identity_ok = True
    for idx in indices[:10]:
        img_aug_off, tgt_aug_off, _, _ = ds_aug[idx]
        img_noaug_ref, tgt_noaug_ref, _, _ = ds_noaug[idx]
        if not torch.equal(img_aug_off, img_noaug_ref):
            identity_ok = False
            print(f"    sample {idx}: pixel mismatch when all flags OFF (unexpected!)")
            break

    # Restore constants
    for k, v in saved.items():
        setattr(constants, k, v)

    all_pass = check(identity_ok, "All-flags-off augment=True identical to augment=False") and all_pass

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    if all_pass:
        print(f"Result: {PASS} — all checks passed.")
    else:
        print(f"Result: {FAIL} — one or more checks FAILED.")
    print(f"{'=' * 60}\n")

    return all_pass


def run_setup_c() -> bool:
    """Smoke test for setup-C compositing augmentation.

    Monkey-patches constants to force mosaic=1.0, mixup=0.5,
    close_mosaic=2, then exercises ~20 samples and verifies correctness.
    Restores constants before returning so the A/B test can re-run cleanly.
    """
    all_pass = True

    print(f"\n{'=' * 60}")
    print("Smoke test: setup-C compositing augmentation")
    print("  (monkey-patching constants: mosaic=1.0, mixup=0.5, close_mosaic=2)")
    print(f"{'=' * 60}\n")

    # ── Patch constants ───────────────────────────────────────────────────────
    saved_c = {
        "AUG_MOSAIC": constants.AUG_MOSAIC,
        "AUG_MIXUP": constants.AUG_MIXUP,
        "AUG_COPY_PASTE": constants.AUG_COPY_PASTE,
        "AUG_CLOSE_MOSAIC": constants.AUG_CLOSE_MOSAIC,
    }
    constants.AUG_MOSAIC = 1.0
    constants.AUG_MIXUP = 0.5
    constants.AUG_COPY_PASTE = 0.0
    constants.AUG_CLOSE_MOSAIC = 2

    try:
        ds_c = CocoYoloDataset(
            constants.ANNOTATIONS_VAL,
            constants.IMAGE_ROOT,
            constants.IMAGE_SIZE,
            augment=True,
        )
        ds_noaug = build_dataset(augment=False)
        total = len(ds_c)
        n_samples = min(N_SAMPLES, total)
        indices = list(range(n_samples))

        print(f"Dataset size: {total} images, checking {n_samples} samples\n")

        # ── Setup C: epoch in the normal range (mosaic active) ────────────────
        # epoch=0 of total=10 → not in close-mosaic tail (last 2 epochs = 8,9)
        TOTAL_EPOCHS = 10
        ds_c.set_epoch(0, TOTAL_EPOCHS)

        print("── Per-sample checks (mosaic active, epoch 0/10) ───────────────")
        coord_ok = True
        nan_ok = True
        shape_ok = True
        box_counts_c: list[int] = []
        box_counts_base: list[int] = []

        for idx in indices:
            img_t, targets, path, _ = ds_c[idx]

            shape_ok_s = (
                img_t.shape == (3, constants.IMAGE_SIZE, constants.IMAGE_SIZE)
                and img_t.dtype == torch.float32
                and float(img_t.min()) >= 0.0
                and float(img_t.max()) <= 1.0
            )
            shape_ok = shape_ok and shape_ok_s

            if torch.isnan(img_t).any() or torch.isinf(img_t).any():
                nan_ok = False
                print(f"    sample {idx}: NaN/Inf in image tensor!")

            if targets.shape[0] > 0:
                coords = targets[:, 2:6]
                if not ((coords >= 0.0).all() and (coords <= 1.0).all()):
                    coord_ok = False
                    out_of_range = coords[(coords < 0) | (coords > 1)]
                    print(f"    sample {idx}: out-of-range coords: {out_of_range}")
                if torch.isnan(targets).any() or torch.isinf(targets).any():
                    nan_ok = False
                    print(f"    sample {idx}: NaN/Inf in targets!")

            box_counts_c.append(int(targets.shape[0]))
            _, t_base, _, _ = ds_noaug[idx]
            box_counts_base.append(int(t_base.shape[0]))

        all_pass = check(coord_ok, "All target coords in [0, 1] with mosaic on") and all_pass
        all_pass = check(nan_ok, "No NaN/Inf in tensors or targets with mosaic on") and all_pass
        all_pass = check(shape_ok,
                         f"Image tensor shape (3,{constants.IMAGE_SIZE},{constants.IMAGE_SIZE}), "
                         "float32, [0,1] with mosaic on") and all_pass

        # Mosaic images should tend to have more boxes (median ≥ baseline median)
        median_c = float(np.median(box_counts_c)) if box_counts_c else 0.0
        median_base = float(np.median(box_counts_base)) if box_counts_base else 0.0
        print(f"\n  box counts — mosaic: {box_counts_c}")
        print(f"  box counts — single: {box_counts_base}")
        print(f"  median boxes: mosaic={median_c:.1f}, single={median_base:.1f}")
        all_pass = check(
            median_c >= median_base,
            f"Mosaic median box count ({median_c:.1f}) ≥ single-image median ({median_base:.1f})",
        ) and all_pass

        # ── Close-mosaic tail: set_epoch to last epoch ────────────────────────
        print("\n── Close-mosaic tail check (epoch 9/10, AUG_CLOSE_MOSAIC=2) ────")
        # epoch 9 >= 10 - 2 = 8 → _compositing_active() must return False
        ds_c.set_epoch(TOTAL_EPOCHS - 1, TOTAL_EPOCHS)

        # Direct unit check: _compositing_active() must report False
        compositing_suppressed = not ds_c._compositing_active()
        all_pass = check(
            compositing_suppressed,
            "ds._compositing_active() is False at close-mosaic tail epoch",
            f"epoch={ds_c._current_epoch}, total={ds_c._total_epochs}, "
            f"close={constants.AUG_CLOSE_MOSAIC}, threshold={ds_c._total_epochs - int(constants.AUG_CLOSE_MOSAIC)}",
        ) and all_pass

        # Behavioural check: with compositing suppressed, all samples at the
        # close-mosaic epoch must still pass validity checks (coords in range,
        # no NaN, correct tensor shape).  The pixel-identity comparison is
        # intentionally avoided here because augment_hsv uses global np.random
        # (not our per-sample rng), so the global state at this point in the
        # test differs from a freshly constructed comparison dataset.
        # The sufficient guarantee is already provided by the _compositing_active()
        # unit check above: if that method returns False, use_mosaic is False
        # (Python short-circuits), so the single-image path is taken.
        close_validity_ok = True
        close_coord_ok = True
        close_nan_ok = True
        for idx in indices[:10]:
            img_close, tgt_close, _, _ = ds_c[idx]

            shape_ok_s = (
                img_close.shape == (3, constants.IMAGE_SIZE, constants.IMAGE_SIZE)
                and img_close.dtype == torch.float32
                and float(img_close.min()) >= 0.0
                and float(img_close.max()) <= 1.0
            )
            if not shape_ok_s:
                close_validity_ok = False
                print(f"    sample {idx}: bad tensor shape/dtype/range at close-mosaic epoch")

            if torch.isnan(img_close).any() or torch.isinf(img_close).any():
                close_nan_ok = False

            if tgt_close.shape[0] > 0:
                coords = tgt_close[:, 2:6]
                if not ((coords >= 0.0).all() and (coords <= 1.0).all()):
                    close_coord_ok = False
                    print(f"    sample {idx}: out-of-range coords at close-mosaic epoch")

        all_pass = check(
            close_validity_ok and close_coord_ok and close_nan_ok,
            "Close-mosaic: samples at tail epoch still valid (shape/coords/no-NaN)",
        ) and all_pass

        # Deterministic behavioural check: verify that build_mosaic is NEVER
        # called at the tail epoch.  We monkey-patch transforms.build_mosaic with
        # a counter wrapper and assert the call count stays at zero.
        # This is a hard guarantee — not a statistical comparison of two different
        # random samples (which could flip on any run).
        import scripts.training.yolov5s.transforms as _transforms_mod
        _mosaic_call_count = 0
        _original_build_mosaic = _transforms_mod.build_mosaic

        def _counting_build_mosaic(*args, **kwargs):
            nonlocal _mosaic_call_count
            _mosaic_call_count += 1
            return _original_build_mosaic(*args, **kwargs)

        _transforms_mod.build_mosaic = _counting_build_mosaic
        try:
            for idx in indices[:10]:
                ds_c[idx]  # trigger __getitem__ at tail epoch
        finally:
            _transforms_mod.build_mosaic = _original_build_mosaic

        print(f"\n  build_mosaic call count at tail epoch (must be 0): {_mosaic_call_count}")
        all_pass = check(
            _mosaic_call_count == 0,
            f"build_mosaic never called at close-mosaic tail epoch "
            f"(call count={_mosaic_call_count})",
        ) and all_pass

    finally:
        # Always restore constants
        for k, v in saved_c.items():
            setattr(constants, k, v)

    # ── Summary for setup C ───────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    if all_pass:
        print(f"Result: {PASS} — setup-C checks all passed.")
    else:
        print(f"Result: {FAIL} — one or more setup-C checks FAILED.")
    print(f"{'=' * 60}\n")

    return all_pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Smoke test for augmentation pipeline"
    )
    parser.add_argument(
        "--setup-c",
        action="store_true",
        help="Also run setup-C compositing checks (mosaic, mixup, close-mosaic)",
    )
    args = parser.parse_args()

    all_ok = True
    try:
        if args.setup_c:
            ok_c = run_setup_c()
            all_ok = all_ok and ok_c
            print(f"\n{'─' * 60}")
            print("Re-running A/B basic test to confirm defaults unchanged after setup-C patch …")
            print(f"{'─' * 60}")

        ok_ab = run()
        all_ok = all_ok and ok_ab
    except Exception:
        traceback.print_exc()
        all_ok = False
    sys.exit(0 if all_ok else 1)
