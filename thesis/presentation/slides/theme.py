"""Look and feel of the final talk: the inovex template rebuilt for manim.

Colours, fonts, the gradient background, and the building blocks every slide uses
(titles, pills, rounded photos, placeholders, bars). Imported by talk.py.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from manim import (
    BOLD,
    DOWN,
    LEFT,
    ORIGIN,
    RIGHT,
    UP,
    Arc,
    Arrow,
    Circle,
    CurvedArrow,
    DashedVMobject,
    Group,
    ImageMobject,
    Line,
    RoundedRectangle,
    Text,
    VGroup,
    config,
)
from PIL import Image, ImageDraw, ImageOps

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CACHE = HERE / "build" / "cache"
FONTS = HERE / "assets" / "fonts"

# --- inovex theme colours (from the interim deck's theme1.xml) -----------------
NAVY = "#061B59"
BLUE = "#2C5DFF"
GREEN = "#7DF381"  # the one highlight colour: the bar or point a slide is about
SKY = "#0AAFE8"  # second categorical colour (API generators, real images)
CYAN = "#41E8E0"
TEAL = "#12A3A3"
MIST = "#DFE5F0"  # neutral marks, muted text
WHITE = "#FFFFFF"
RED = "#FF6B6B"  # only for the ✗ verdict marks

# --- video encoding ---------------------------------------------------------------
# manim hard-codes libx264 at crf 23, which is tuned for motion and blurs fine text and
# coloured edges on static slides. manim-slides only remuxes the partial movie files when
# it joins them, so the crf chosen here is what reaches the screen. Lower = sharper and
# larger; 12 is visually lossless for slide content.
CRF = 12


def _sharpen_encoding(crf: int = CRF) -> None:
    import manim.scene.scene_file_writer as sfw

    real_av = sfw.av

    class _Container:
        def __init__(self, inner):
            self._inner = inner

        def add_stream(self, codec, *args, options=None, **kwargs):
            if options and "crf" in options:
                options = {**options, "crf": str(crf)}
            return self._inner.add_stream(codec, *args, options=options, **kwargs)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    class _AV:
        def __getattr__(self, name):
            return getattr(real_av, name)

        def open(self, *args, **kwargs):
            container = real_av.open(*args, **kwargs)
            writing = kwargs.get("mode") == "w" or (len(args) > 1 and args[1] == "w")
            return _Container(container) if writing else container

    if not isinstance(sfw.av, _AV):
        sfw.av = _AV()


_sharpen_encoding()

FONT = "Atkinson Hyperlegible"
MONO = "Menlo"

FRAME_W = config.frame_width
FRAME_H = config.frame_height
LEFT_X = -FRAME_W / 2 + 0.6  # left content edge
TITLE_Y = 2.75


# --- text ----------------------------------------------------------------------
def txt(s: str, size: float = 28, color: str = WHITE, bold: bool = False, **kw) -> Text:
    return Text(s, font=FONT, font_size=size, color=color, weight=BOLD if bold else "NORMAL", **kw)


def title(s: str) -> Text:
    t = txt(s, 40)
    t.move_to([LEFT_X, TITLE_Y, 0], aligned_edge=LEFT)
    return t


def pill(s: str, size: float = 22, fill: str = WHITE, color: str = NAVY, bold: bool = False) -> VGroup:
    """White rounded label, the interim deck's signature element."""
    t = txt(s, size, color, bold)
    h = t.height + 0.22
    box = RoundedRectangle(corner_radius=h / 2, width=t.width + 0.42, height=h)
    box.set_fill(fill, 1).set_stroke(width=0)
    t.move_to(box)
    return VGroup(box, t)


def big_number(s: str, caption: str | None = None, size: float = 96, color: str = WHITE) -> VGroup:
    n = txt(s, size, color, bold=True)
    if caption is None:
        return VGroup(n)
    c = txt(caption, 26, MIST).next_to(n, DOWN, buff=0.15)
    return VGroup(n, c)


# --- images --------------------------------------------------------------------
def _cached(src: Path, aspect: float | None, radius: float, max_px: int, top: bool = False) -> Path:
    key = hashlib.md5(f"{src}|{src.stat().st_mtime}|{aspect}|{radius}|{max_px}|{top}".encode()).hexdigest()[:16]
    out = CACHE / f"{key}.png"
    if out.exists():
        return out
    CACHE.mkdir(parents=True, exist_ok=True)
    im = ImageOps.exif_transpose(Image.open(src)).convert("RGBA")
    if aspect is not None:  # centre-crop to the requested width/height ratio
        w, h = im.size
        if w / h > aspect:
            nw = int(h * aspect)
            im = im.crop(((w - nw) // 2, 0, (w - nw) // 2 + nw, h))
        else:
            nh = int(w / aspect)
            y0 = 0 if top else (h - nh) // 2
            im = im.crop((0, y0, w, y0 + nh))
    im.thumbnail((max_px, max_px), Image.LANCZOS)
    if radius > 0:
        r = int(min(im.size) * radius)
        mask = Image.new("L", im.size, 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, im.size[0] - 1, im.size[1] - 1), r, fill=255)
        im.putalpha(Image.composite(im.getchannel("A"), mask, mask))
    im.save(out)
    return out


def _pixels_needed(src: Path, height: float | None, width: float | None, aspect: float | None) -> int:
    """Longest side, in source pixels, that the photo occupies at the current render resolution.

    Sizing from config.pixel_width keeps photos sharp at 4K without bloating 1080p drafts.
    """
    if aspect is None:
        with Image.open(src) as im:
            w, h = im.size
            if im.getexif().get(0x0112) in (5, 6, 7, 8):  # EXIF rotation swaps the sides
                w, h = h, w
        aspect = w / h
    disp_w = width if width is not None else (height or 2.0) * aspect
    disp_h = disp_w / aspect
    scale = config.pixel_width / FRAME_W  # pixels per scene unit
    return int(min(4096, max(256, max(disp_w, disp_h) * scale * 1.1)))


def photo(
    src: Path | str,
    height: float | None = None,
    width: float | None = None,
    aspect: float | None = None,
    radius: float = 0.05,
    max_px: int | None = None,
    top: bool = False,
):
    """A rounded-corner photo, or a labelled placeholder if the file is missing.

    aspect centre-crops to width/height; top=True keeps the top instead (screenshots).
    """
    src = Path(src)
    if not src.exists():
        a = aspect or 4 / 3
        h = height or (width / a if width else 2.0)
        return placeholder(src.name, h * a, h)
    if max_px is None:
        max_px = _pixels_needed(src, height, width, aspect)
    m = ImageMobject(str(_cached(src, aspect, radius, max_px, top)))
    if height is not None:
        m.scale_to_fit_height(height)
    elif width is not None:
        m.scale_to_fit_width(width)
    return m


def placeholder(label: str, w: float, h: float) -> VGroup:
    """Dashed box that marks an asset still to be supplied (see README)."""
    box = RoundedRectangle(corner_radius=0.12, width=w, height=h).set_stroke(MIST, 2, 0.8)
    dashed = DashedVMobject(box, num_dashes=40)
    t = txt("missing:\n" + label, 16, MIST)
    if t.width > w * 0.9:
        t.scale_to_fit_width(w * 0.9)
    t.move_to(box)
    return VGroup(dashed, t)


def background() -> ImageMobject:
    """Diagonal navy-to-blue gradient, darker than the template so charts keep contrast."""
    out = CACHE / "background.png"
    if not out.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        w, h = 1920, 1080
        x = np.linspace(0, 1, w)[None, :]
        y = np.linspace(0, 1, h)[:, None]
        t = np.clip(0.55 * x + 0.75 * y - 0.15, 0, 1) ** 1.3
        a = np.array([6, 27, 89], float)  # NAVY
        b = np.array([30, 70, 214], float)  # softened BLUE
        rgb = (a[None, None, :] * (1 - t[..., None]) + b[None, None, :] * t[..., None]).astype(np.uint8)
        Image.fromarray(rgb, "RGB").save(out)
    m = ImageMobject(str(out))
    m.stretch_to_fit_width(FRAME_W).stretch_to_fit_height(FRAME_H).move_to(ORIGIN)
    return m


# --- arrows: one stroke width and one tip size on every slide ------------------
ARROW_STROKE = 4
ARROW_TIP = 0.22


def arrow(start, end, color: str = WHITE, buff: float = 0.12) -> Arrow:
    """Straight arrow whose width and tip never scale with its length."""
    return Arrow(
        start, end, buff=buff, color=color, stroke_width=ARROW_STROKE, tip_length=ARROW_TIP,
        max_tip_length_to_length_ratio=1.0, max_stroke_width_to_length_ratio=1000,
    )


def curved_arrow(start, end, angle: float, color: str = WHITE) -> CurvedArrow:
    a = CurvedArrow(start, end, angle=angle, color=color, stroke_width=ARROW_STROKE)
    a.pop_tips()
    a.add_tip(tip_length=ARROW_TIP, tip_width=ARROW_TIP)
    return a.set_stroke(width=ARROW_STROKE)


def pptx_box(x: float, y: float, w: float, h: float) -> tuple[np.ndarray, float, float]:
    """Interim-deck geometry (inches on a 10 x 5.625 page, top-left origin) to manim."""
    s = FRAME_W / 10
    centre = np.array([(x + w / 2) * s - FRAME_W / 2, FRAME_H / 2 - (y + h / 2) * s, 0])
    return centre, w * s, h * s


# --- small drawn icons ---------------------------------------------------------
def stopwatch(r: float = 0.8, color: str = WHITE) -> VGroup:
    ring = Circle(radius=r).set_stroke(color, 6)
    knob = RoundedRectangle(corner_radius=0.05, width=r * 0.35, height=r * 0.22).set_fill(color, 1).set_stroke(width=0)
    knob.next_to(ring, UP, buff=0.04)
    hand = Line(ORIGIN, UP * r * 0.7).set_stroke(color, 6)
    hand.rotate(-0.9, about_point=ORIGIN).shift(ring.get_center())
    hub = Circle(radius=0.06).set_fill(color, 1).set_stroke(width=0).move_to(ring)
    return VGroup(ring, knob, hand, hub)


def check_mark(size: float = 0.5, color: str = GREEN) -> VGroup:
    a = Line([-0.5, 0, 0], [-0.15, -0.38, 0])
    b = Line([-0.15, -0.38, 0], [0.5, 0.4, 0])
    return VGroup(a, b).set_stroke(color, 10).scale(size)


def cross_mark(size: float = 0.5, color: str = RED) -> VGroup:
    a = Line([-0.4, -0.4, 0], [0.4, 0.4, 0])
    b = Line([-0.4, 0.4, 0], [0.4, -0.4, 0])
    return VGroup(a, b).set_stroke(color, 10).scale(size)


def tilde_mark(size: float = 0.5, color: str = "#F1C232") -> VGroup:
    """'Partly' verdict: a wave."""
    w = Arc(radius=0.25, start_angle=np.pi, angle=-np.pi).shift(LEFT * 0.25)
    w2 = Arc(radius=0.25, start_angle=np.pi, angle=np.pi).shift(RIGHT * 0.25)
    return VGroup(w, w2).set_stroke(color, 10).scale(size)


def card(w: float, h: float, fill: str = WHITE, opacity: float = 1.0, radius: float = 0.15) -> RoundedRectangle:
    return RoundedRectangle(corner_radius=radius, width=w, height=h).set_fill(fill, opacity).set_stroke(width=0)


# --- bars ----------------------------------------------------------------------
def hbar(value: float, vmax: float, length: float, h: float = 0.36, color: str = MIST, opacity: float = 1.0):
    """Horizontal bar with a rounded data end, anchored at its left edge."""
    w = max(length * value / vmax, 0.02)
    r = min(h / 2, w / 2)
    return RoundedRectangle(corner_radius=r, width=w, height=h).set_fill(color, opacity).set_stroke(width=0)


def vbar(value: float, vmax: float, length: float, w: float = 0.8, color: str = MIST, opacity: float = 1.0):
    h = max(length * value / vmax, 0.02)
    r = min(w / 6, h / 2)
    return RoundedRectangle(corner_radius=r, width=w, height=h).set_fill(color, opacity).set_stroke(width=0)


def collage(paths: list[Path], cols: int, cell_w: float, aspect: float = 4 / 3, gap: float = 0.08) -> Group:
    """Uniform grid of rounded photos."""
    tiles = Group()
    cell_h = cell_w / aspect
    for i, p in enumerate(paths):
        m = photo(p, width=cell_w, aspect=aspect, radius=0.06)
        r, c = divmod(i, cols)
        m.move_to([c * (cell_w + gap), -r * (cell_h + gap), 0])
        tiles.add(m)
    return tiles.move_to(ORIGIN)


def mosaic(paths: list[Path], cols: int, tile_px: int | None = None, aspect: float = 4 / 3, gap_px: int = 8) -> Path:
    """Pre-render many thumbnails into one image (much faster than one mobject each).

    tile_px defaults to the tile's on-screen width when the sheet spans the frame.
    """
    if tile_px is None:
        tile_px = max(160, config.pixel_width // cols)
    key = hashlib.md5("|".join(map(str, paths)).encode() + f"{cols}{tile_px}{aspect}".encode()).hexdigest()[:16]
    out = CACHE / f"mosaic_{key}.png"
    if out.exists():
        return out
    CACHE.mkdir(parents=True, exist_ok=True)
    th = int(tile_px / aspect)
    rows = (len(paths) + cols - 1) // cols
    sheet = Image.new("RGBA", (cols * (tile_px + gap_px), rows * (th + gap_px)), (0, 0, 0, 0))
    for i, p in enumerate(paths):
        im = ImageOps.exif_transpose(Image.open(p)).convert("RGBA")
        im = ImageOps.fit(im, (tile_px, th), Image.LANCZOS)
        mask = Image.new("L", im.size, 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, tile_px - 1, th - 1), 14, fill=255)
        r, c = divmod(i, cols)
        sheet.paste(im, (c * (tile_px + gap_px), r * (th + gap_px)), mask)
    sheet.save(out)
    return out

