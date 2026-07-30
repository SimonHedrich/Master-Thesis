"""Image preprocessing for the YOLOv5s pipeline.

`letterbox` and `to_tensor` handle inference-time preprocessing.
`augment_basic` implements the single-image, distillation-safe basic
augmentation set (setups A & B); it is gated by the `AUG_*` flags in
`constants.py` and called from `CocoYoloDataset.__getitem__` when
`augment=True`.

Setup-C compositing augmentations:
  - `build_mosaic`  : 4-image 2×2 stitch → random_perspective crop → IMAGE_SIZE.
  - `mixup`         : alpha-blend two same-size images, concatenate their labels.
  - Copy-paste      : NO-OP on box-only GT (needs instance masks). If
                      AUG_COPY_PASTE > 0 a single warning is emitted; no
                      effect on data.

`assert_distillation_safe()` guards against accidental enabling in setup A.
"""
from __future__ import annotations

import logging

import numpy as np
import cv2
import torch

from yolov5.utils.augmentations import augment_hsv, random_perspective

logger = logging.getLogger(__name__)


def letterbox(
    img: np.ndarray,
    new_shape: int | tuple[int, int] = 640,
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
    """Resize an image to `new_shape` preserving aspect ratio, padding with `color`.

    Returns
    -------
    img : np.ndarray
        Letterboxed image, HWC BGR uint8, shape (new_h, new_w, 3).
    ratio : (r_w, r_h)
        Per-axis scale factor applied before padding (both equal for
        aspect-preserving resize, but returned per-axis to match the
        standard YOLOv5 signature so callers can multiply directly).
    pad : (dw, dh)
        Total padding width / height in pixels (split evenly across the two sides).
    """
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    h0, w0 = img.shape[:2]
    r = min(new_shape[0] / h0, new_shape[1] / w0)
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    if (w0, h0) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return img, (r, r), (dw * 2, dh * 2)


def to_tensor(img: np.ndarray) -> torch.Tensor:
    """HWC uint8 BGR (OpenCV default) → CHW float32 RGB in [0, 1]."""
    img = img[:, :, ::-1]  # BGR → RGB
    img = np.ascontiguousarray(img.transpose(2, 0, 1))  # HWC → CHW
    return torch.from_numpy(img).float() / 255.0


# ─── Basic augmentation — setups A & B ───────────────────────────────────────


def augment_basic(
    img: np.ndarray,
    labels: np.ndarray,
    *,
    hflip: bool,
    hsv: bool,
    hsv_h: float,
    hsv_s: float,
    hsv_v: float,
    scale: float,
    translate: float,
    degrees: float,
    shear: float,
    perspective: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply single-image, distillation-safe basic augmentation.

    img    : HWC BGR uint8 (already letterboxed to a square IMAGE_SIZE).
    labels : (N,5) float array [cls, cx, cy, w, h] NORMALISED to [0,1] (YOLO format).
    rng    : np.random.Generator for deterministic per-sample randomness.
    Returns (img, labels) with labels still normalised [0,1], degenerate boxes dropped.

    Transform order: scale/translate (geometry) → hflip → hsv (photometric last).

    Coordinate convention for random_perspective:
        targets in  : (N,5) [cls, x1, y1, x2, y2] ABSOLUTE pixels
        targets out : (N,5) [cls, x1, y1, x2, y2] ABSOLUTE pixels (clipped,
                      filtered by box_candidates with area_thr=0.10)
    The scale parameter maps to random_perspective's `scale` argument:
        final_scale ~ Uniform(1 - scale, 1 + scale)
    """
    h, w = img.shape[:2]  # both equal IMAGE_SIZE after letterbox
    n = len(labels)

    # ── 1. Scale + translate (geometry) ───────────────────────────────────────
    if scale != 0.0 or translate != 0.0:
        if n > 0:
            # normalised xywh → absolute xyxy  (shape N×5: [cls, x1,y1,x2,y2])
            targets_abs = np.zeros((n, 5), dtype=np.float32)
            targets_abs[:, 0] = labels[:, 0]  # cls
            cx_px = labels[:, 1] * w
            cy_px = labels[:, 2] * h
            bw_px = labels[:, 3] * w
            bh_px = labels[:, 4] * h
            targets_abs[:, 1] = cx_px - bw_px / 2  # x1
            targets_abs[:, 2] = cy_px - bh_px / 2  # y1
            targets_abs[:, 3] = cx_px + bw_px / 2  # x2
            targets_abs[:, 4] = cy_px + bh_px / 2  # y2
        else:
            targets_abs = np.zeros((0, 5), dtype=np.float32)

        img, targets_abs = random_perspective(
            img,
            targets=targets_abs,
            degrees=degrees,
            translate=translate,
            scale=scale,
            shear=shear,
            perspective=perspective,
            border=(0, 0),
        )

        # absolute xyxy → normalised xywh; clip to [0,1]
        if len(targets_abs) > 0:
            x1 = targets_abs[:, 1] / w
            y1 = targets_abs[:, 2] / h
            x2 = targets_abs[:, 3] / w
            y2 = targets_abs[:, 4] / h
            x1 = np.clip(x1, 0.0, 1.0)
            y1 = np.clip(y1, 0.0, 1.0)
            x2 = np.clip(x2, 0.0, 1.0)
            y2 = np.clip(y2, 0.0, 1.0)
            cx_n = (x1 + x2) / 2
            cy_n = (y1 + y2) / 2
            bw_n = x2 - x1
            bh_n = y2 - y1
            labels = np.stack(
                [targets_abs[:, 0], cx_n, cy_n, bw_n, bh_n], axis=1
            ).astype(np.float32)
            # drop degenerate boxes
            keep = (labels[:, 3] > 0) & (labels[:, 4] > 0)
            labels = labels[keep]
        else:
            labels = np.zeros((0, 5), dtype=np.float32)

    # ── 2. Horizontal flip ────────────────────────────────────────────────────
    if hflip and rng.random() < 0.5:
        img = np.ascontiguousarray(np.fliplr(img))
        if len(labels) > 0:
            labels[:, 1] = 1.0 - labels[:, 1]  # cx → 1 - cx

    # ── 3. HSV jitter (photometric, no geometry) ──────────────────────────────
    if hsv:
        # augment_hsv uses global np.random internally (not our rng); that is
        # acceptable — worker_init_fn seeds numpy per worker for reproducibility.
        augment_hsv(img, hgain=hsv_h, sgain=hsv_s, vgain=hsv_v)  # mutates in place

    return img, labels


# ─── Setup-C compositing augmentations ───────────────────────────────────────
# These are multi-image operations and are INCOMPATIBLE with distillation
# (no valid teacher view of the composite).  They are only active when the
# corresponding AUG_* constants are > 0 AND self.augment is True in the
# dataset.  With default constants (all 0.0 / 0) these functions are never
# called, so setups A & B are byte-for-byte unaffected.


def build_mosaic(
    samples: list[tuple[np.ndarray, np.ndarray]],
    image_size: int,
    rng: np.random.Generator,
    degrees: float = 0.0,
    translate: float = 0.1,
    scale: float = 0.5,
    shear: float = 0.0,
    perspective: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble a 4-image mosaic and crop back to `image_size`.

    This mirrors yolov5's ``LoadImagesAndLabels.load_mosaic`` but is
    self-contained and uses our ``np.random.Generator`` for reproducibility.

    Algorithm
    ---------
    1. Allocate a (2*S, 2*S) canvas filled with the YOLOv5 pad colour (114).
    2. Choose a random mosaic centre (xc, yc) in [S/2, 3*S/2] × [S/2, 3*S/2]
       so all four quadrants are non-empty.
    3. Resize each of the 4 source images to (S, S) and paste into its quadrant
       using the YOLOv5 offset arithmetic (no letterbox — mosaic resizes itself).
    4. Convert each image's normalised [cls,cx,cy,w,h] labels to absolute
       [cls,x1,y1,x2,y2] in the (2*S) canvas coordinate frame.
    5. Call ``random_perspective`` with ``border=(-S//2, -S//2)`` to crop the
       canvas back to an (S, S) output.  The geometric jitter (scale, translate)
       is folded in here so the caller must NOT re-apply it.
    6. Convert the output absolute [cls,x1,y1,x2,y2] back to normalised xywh.

    Parameters
    ----------
    samples : list of (img, labels)
        Exactly 4 entries.  Each ``img`` is HWC BGR uint8 at any resolution;
        ``labels`` is (N,5) float32 [cls,cx,cy,w,h] normalised to [0,1].
    image_size : int
        Target output side length S (e.g. 640).
    rng : np.random.Generator
        Source of randomness (per-sample seeded in the dataset).
    degrees, translate, scale, shear, perspective :
        Passed straight to ``random_perspective``; defaults match the
        constants.py basic-set values (rotation/shear/perspective OFF).

    Returns
    -------
    img_out : (S, S, 3) uint8 BGR
    labels_out : (M, 5) float32 [cls, cx, cy, w, h] normalised to [0, 1]
        Degenerate boxes are already removed by random_perspective's
        box_candidates filter (area_thr=0.10).
    """
    assert len(samples) == 4, "build_mosaic requires exactly 4 (img, labels) pairs"
    s = image_size

    # ── 1. Mosaic centre in [S/2, 3S/2] ─────────────────────────────────────
    # Using rng.integers keeps this reproducible with the per-sample seed.
    xc = int(rng.integers(s // 2, 3 * s // 2))
    yc = int(rng.integers(s // 2, 3 * s // 2))

    # ── 2. Allocate 2S × 2S canvas ───────────────────────────────────────────
    canvas = np.full((2 * s, 2 * s, 3), 114, dtype=np.uint8)

    labels4: list[np.ndarray] = []  # accumulate absolute [cls,x1,y1,x2,y2]

    for i, (img, labels) in enumerate(samples):
        # Resize source image to S×S (mosaic manages its own scale)
        img = cv2.resize(img, (s, s), interpolation=cv2.INTER_LINEAR)
        h, w = img.shape[:2]  # both == s after resize

        # YOLOv5 quadrant arithmetic (mirrors load_mosaic exactly)
        if i == 0:  # top-left quadrant
            x1a, y1a = max(xc - w, 0), max(yc - h, 0)
            x2a, y2a = xc, yc
            x1b = w - (x2a - x1a)  # source crop start x
            y1b = h - (y2a - y1a)  # source crop start y
            x2b, y2b = w, h
        elif i == 1:  # top-right
            x1a, y1a = xc, max(yc - h, 0)
            x2a, y2a = min(xc + w, 2 * s), yc
            x1b, y1b = 0, h - (y2a - y1a)
            x2b, y2b = min(w, x2a - x1a), h
        elif i == 2:  # bottom-left
            x1a, y1a = max(xc - w, 0), yc
            x2a, y2a = xc, min(2 * s, yc + h)
            x1b, y1b = w - (x2a - x1a), 0
            x2b, y2b = w, min(y2a - y1a, h)
        else:  # bottom-right  (i == 3)
            x1a, y1a = xc, yc
            x2a, y2a = min(xc + w, 2 * s), min(2 * s, yc + h)
            x1b, y1b = 0, 0
            x2b, y2b = min(w, x2a - x1a), min(y2a - y1a, h)

        canvas[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]

        # Offset from source-image origin to canvas paste position
        pad_x = x1a - x1b
        pad_y = y1a - y1b

        # Convert normalised [cls,cx,cy,w,h] → absolute [cls,x1,y1,x2,y2]
        # in the 2S canvas frame, using the quadrant offset.
        if len(labels) > 0:
            abs_boxes = np.zeros((len(labels), 5), dtype=np.float32)
            abs_boxes[:, 0] = labels[:, 0]  # cls
            cx_px = labels[:, 1] * w
            cy_px = labels[:, 2] * h
            bw_px = labels[:, 3] * w
            bh_px = labels[:, 4] * h
            abs_boxes[:, 1] = cx_px - bw_px / 2 + pad_x  # x1 in canvas
            abs_boxes[:, 2] = cy_px - bh_px / 2 + pad_y  # y1
            abs_boxes[:, 3] = cx_px + bw_px / 2 + pad_x  # x2
            abs_boxes[:, 4] = cy_px + bh_px / 2 + pad_y  # y2
            labels4.append(abs_boxes)

    # ── 3. Clip label coords to the 2S canvas before calling random_perspective
    if labels4:
        targets4 = np.concatenate(labels4, axis=0)
        targets4[:, 1:] = np.clip(targets4[:, 1:], 0, 2 * s)
    else:
        targets4 = np.zeros((0, 5), dtype=np.float32)

    # ── 4. random_perspective crops 2S→S and applies geometric jitter ─────────
    # border=(-s//2, -s//2) shrinks the output by S in each dimension.
    img_out, targets_out = random_perspective(
        canvas,
        targets=targets4,
        degrees=degrees,
        translate=translate,
        scale=scale,
        shear=shear,
        perspective=perspective,
        border=(-s // 2, -s // 2),
    )
    # img_out is now (S, S, 3); targets_out is [cls, x1, y1, x2, y2] absolute in S frame

    # ── 5. Absolute [cls,x1,y1,x2,y2] → normalised [cls,cx,cy,w,h] ──────────
    if len(targets_out) > 0:
        x1 = targets_out[:, 1] / s
        y1 = targets_out[:, 2] / s
        x2 = targets_out[:, 3] / s
        y2 = targets_out[:, 4] / s
        x1 = np.clip(x1, 0.0, 1.0)
        y1 = np.clip(y1, 0.0, 1.0)
        x2 = np.clip(x2, 0.0, 1.0)
        y2 = np.clip(y2, 0.0, 1.0)
        cx_n = (x1 + x2) / 2
        cy_n = (y1 + y2) / 2
        bw_n = x2 - x1
        bh_n = y2 - y1
        labels_out = np.stack(
            [targets_out[:, 0], cx_n, cy_n, bw_n, bh_n], axis=1
        ).astype(np.float32)
        keep = (labels_out[:, 3] > 0) & (labels_out[:, 4] > 0)
        labels_out = labels_out[keep]
    else:
        labels_out = np.zeros((0, 5), dtype=np.float32)

    return img_out, labels_out


def mixup(
    img1: np.ndarray,
    labels1: np.ndarray,
    img2: np.ndarray,
    labels2: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Blend two same-size images and concatenate their label sets.

    Following the yolov5 / YOLOX recipe:
      - lam ~ Beta(32, 32)  (tightly concentrated around 0.5)
      - img  = img1 * lam + img2 * (1 - lam)  (float blend, cast back to uint8)
      - labels = concat(labels1, labels2)       (both label sets kept, no drop)

    Both images must already be (S, S, 3) uint8 BGR at the same resolution.
    Labels are (N,5) float32 [cls, cx, cy, w, h] normalised to [0,1].
    """
    assert img1.shape == img2.shape, (
        f"mixup: both images must be the same shape; got {img1.shape} vs {img2.shape}"
    )
    # Beta(32,32) is very concentrated → lam typically 0.45–0.55
    lam = float(rng.beta(32.0, 32.0))
    img_blend = (img1.astype(np.float32) * lam + img2.astype(np.float32) * (1.0 - lam))
    img_out = np.clip(img_blend, 0, 255).astype(np.uint8)

    if len(labels1) > 0 and len(labels2) > 0:
        labels_out = np.concatenate([labels1, labels2], axis=0)
    elif len(labels1) > 0:
        labels_out = labels1.copy()
    elif len(labels2) > 0:
        labels_out = labels2.copy()
    else:
        labels_out = np.zeros((0, 5), dtype=np.float32)

    return img_out, labels_out


# ─── Distillation-safety guard ────────────────────────────────────────────────
# NOTE: there is currently no distillation entry point.
#       A future distillation training script MUST call assert_distillation_safe()
#       at startup before building dataloaders.


def assert_distillation_safe() -> None:
    """Raise if any multi-image compositing augmentation is enabled.

    Compositing has no valid teacher view (Plan §5.4). A future distillation
    entry point must call this at startup.
    """
    import scripts.synthetic_model_comparison.training.constants as c

    bad = {
        n: getattr(c, n)
        for n in ("AUG_MOSAIC", "AUG_MIXUP", "AUG_COPY_PASTE", "AUG_CLOSE_MOSAIC")
        if getattr(c, n)
    }
    if bad:
        raise RuntimeError(
            f"Compositing augmentation must be OFF in distillation: {bad}"
        )
