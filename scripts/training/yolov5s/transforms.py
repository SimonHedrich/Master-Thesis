"""Image preprocessing for the YOLOv5s pipeline.

Only `letterbox` + `to_tensor` are active. Augmentation stubs (`mosaic`,
`hsv`, `random_hflip`) are kept as placeholders so future runs can flip
them on by toggling the `AUG_*` flags in `constants.py` without
touching this module's signature.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch


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


# ─── Augmentation stubs (disabled by default; gated by constants.AUG_*) ──────


def mosaic(*args, **kwargs):  # pragma: no cover — placeholder
    raise NotImplementedError("Mosaic augmentation is disabled in this baseline.")


def hsv(*args, **kwargs):  # pragma: no cover — placeholder
    raise NotImplementedError("HSV augmentation is disabled in this baseline.")


def random_hflip(*args, **kwargs):  # pragma: no cover — placeholder
    raise NotImplementedError("Random hflip is disabled in this baseline.")
