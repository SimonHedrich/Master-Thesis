"""Shared image conversion utilities for download scripts."""

import io
from pathlib import Path

from PIL import Image


def save_as_jpg(image_bytes: bytes, dest_path: Path, quality: int = 92) -> None:
    """Convert any image format to JPEG and save to dest_path.

    For animated GIFs, takes the first frame. For images with transparency
    (RGBA, LA, palette with alpha), composites onto a white background.
    Raises on failure so callers can treat it as a download error.

    Args:
        image_bytes: Raw image bytes in any Pillow-supported format.
        dest_path:   Destination path — must have a .jpg extension.
        quality:     JPEG quality (1–95). Default 92 gives visually lossless
                     results for photographs while achieving ~3–5× size
                     reduction versus PNG.
    """
    img = Image.open(io.BytesIO(image_bytes))

    # For animated GIFs, use the first frame only
    try:
        img.seek(0)
    except EOFError:
        pass

    # Composite transparent images onto a white background
    if img.mode in ("RGBA", "LA", "PA"):
        img_rgba = img.convert("RGBA")
        bg = Image.new("RGB", img_rgba.size, (255, 255, 255))
        bg.paste(img_rgba, mask=img_rgba.split()[3])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest_path.with_suffix(".jpg.tmp")
    try:
        img.save(tmp, "JPEG", quality=quality, optimize=True)
        tmp.rename(dest_path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
