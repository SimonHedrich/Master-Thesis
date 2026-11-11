# /// script
# requires-python = ">=3.11"
# dependencies = ["pillow>=10", "numpy>=1.26"]
# ///
"""Hero (featured) image candidates for the inovex blog article.

The inovex blog's featured images share one template: a 1500x880 isometric illustration
on a radial gradient from inovex blue #2D5DFF (centre) to navy #061B59 (edges, reached
at ~625 px from the centre), white / light-blue objects, lime-green accents, no text,
one central group with wide empty margins. These candidates rebuild that template with
vector shapes drawn here plus the thesis' own AI-generated animal images, shown as
isometric photo cards (no real photos, no product photos, no logos).

Writes 1600x939 JPEGs (the blog's 1500:880 aspect) to thesis/blog/figures/hero/:
  hero_a.jpg  photo wall: the twelve generated species as an upright card wall on a
              floating platform, four with lime detection boxes
  hero_b.jpg  pipeline: a stack of generated cards -> a chip on a platform -> one
              binturong card with a detection box
  hero_c.jpg  eyepiece: the binocular's two-circle field of view around a generated
              binturong with a detection box, small generated cards floating around

Everything important stays inside the centre ~1260x800 px, so the 720x520 listing
thumbnail and the ~1.91:1 social preview crops keep the subject.

PEP 723 script rather than the repo-root uv project: that project is the Linux/CUDA
training environment, and these images are composed natively on the Mac.

Run (from the repo root):
    uv run --script thesis/blog/figures/hero/make_hero.py
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[4]
GEN = ROOT / "thesis/presentation/images/generators"
TWELVE = GEN / "twelve"
OUT = Path(__file__).resolve().parent

W, H = 1600, 939          # output size (1500:880 aspect)
SS = 2                    # supersampling factor
CW, CH = W * SS, H * SS   # canvas size

BLUE = (0x2D, 0x5D, 0xFF)
NAVY = (0x06, 0x1B, 0x59)
LIME = (0xC0, 0xEF, 0x59)
WHITE = (255, 255, 255)
PALE = (0xE4, 0xEA, 0xFF)       # card / object side faces
PERI = (0x83, 0x9E, 0xFF)       # platform top (as in the blog's disc platforms)
PERI_DARK = (0x4A, 0x6C, 0xF0)  # platform side
GLOW = (0x85, 0xD0, 0xF2)       # thin connection lines

COS30, SIN30 = math.cos(math.radians(30)), 0.5

# Detection boxes in image fractions (x0, y0, x1, y1), set by eye per source image.
BOXES = {
    "american_black_bear": (0.10, 0.30, 0.63, 0.75),
    "grevy_s_zebra": (0.18, 0.28, 0.84, 0.81),
    "kinkajou": (0.37, 0.09, 0.67, 0.92),
    "red_fox": (0.05, 0.36, 0.79, 0.74),
    "binturong_gemini_0001": (0.05, 0.10, 0.83, 0.82),
}


# --------------------------------------------------------------------------- helpers
def s(v: float) -> float:
    """Design units (output px) -> canvas px."""
    return v * SS


def background() -> Image.Image:
    """Radial gradient #2D5DFF -> #061B59, linear in distance, radius 625/1500 of W."""
    yy, xx = np.mgrid[0:CH, 0:CW].astype(np.float32)
    d = np.hypot(xx - CW / 2, yy - CH / 2) / (CW * 625 / 1500)
    t = np.clip(d, 0, 1)[..., None]
    rgb = np.array(BLUE, np.float32) * (1 - t) + np.array(NAVY, np.float32) * t
    return Image.fromarray(rgb.astype(np.uint8), "RGB").convert("RGBA")


def load(path: Path, box_key: str | None = None, border: float = 0.035) -> Image.Image:
    """Load an image, optionally draw a lime detection box, add a white card border."""
    im = Image.open(path).convert("RGB")
    im.thumbnail((1400, 1400), Image.LANCZOS)
    w, h = im.size
    if box_key:
        x0, y0, x1, y1 = BOXES[box_key]
        d = ImageDraw.Draw(im)
        lw = max(6, int(w * 0.02))
        d.rectangle([x0 * w, y0 * h, x1 * w, y1 * h], outline=LIME, width=lw)
        # label tab without text, as on a detector's output
        d.rectangle([x0 * w, y0 * h - w * 0.05, x0 * w + w * 0.22, y0 * h], fill=LIME)
    b = int(w * border)
    card = Image.new("RGB", (w + 2 * b, h + 2 * b), WHITE)
    card.paste(im, (b, b))
    return card


def paste_affine(canvas: Image.Image, img: Image.Image, p0, u, v) -> None:
    """Paste img so that its pixel (a, b) lands at p0 + a*u' + b*v' (u, v span the
    full width / height in canvas px)."""
    w, h = img.size
    ux, uy = u[0] / w, u[1] / w
    vx, vy = v[0] / h, v[1] / h
    det = ux * vy - vx * uy
    ia, ib, ic, id_ = vy / det, -vx / det, -uy / det, ux / det
    ox, oy = p0
    coeffs = (ia, ib, -(ia * ox + ib * oy), ic, id_, -(ic * ox + id_ * oy))
    rgba = img.convert("RGBA")
    layer = rgba.transform(canvas.size, Image.AFFINE, coeffs, resample=Image.BICUBIC)
    canvas.alpha_composite(layer)


def poly(canvas: Image.Image, pts, fill, blur: float = 0) -> None:
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    ImageDraw.Draw(layer).polygon([tuple(p) for p in pts], fill=fill)
    if blur:
        layer = layer.filter(ImageFilter.GaussianBlur(blur))
    canvas.alpha_composite(layer)


def iso(cx, cy, x, y, z=0.0):
    """Isometric projection, design units -> canvas px. x along the right-down axis,
    y along the left-down axis, z up."""
    return (s(cx + (x - y) * COS30), s(cy + (x + y) * SIN30 - z))


def upright_card(canvas, img, origin, width, height, depth=6.0, shadow=True):
    """An upright photo card in the plane spanned by the iso x axis and z.
    origin = (cx, cy) design coords of the card's bottom-left corner on the floor."""
    ox, oy = origin
    u = (s(width * COS30), s(width * SIN30))
    v = (0.0, s(height))
    top_left = (s(ox), s(oy - height))
    if shadow:
        # soft shadow cast to the lower left on the floor
        f = [(s(ox), s(oy)), (s(ox) + u[0], s(oy) + u[1]),
             (s(ox) + u[0] - s(40), s(oy) + u[1] + s(23)), (s(ox) - s(40), s(oy) + s(23))]
        poly(canvas, f, (2, 10, 40, 110), blur=s(10))
    # card thickness: a pale side face on the left edge
    d = s(depth)
    side = [top_left, (top_left[0] - d * COS30, top_left[1] + d * SIN30),
            (top_left[0] - d * COS30, top_left[1] + d * SIN30 + v[1]),
            (top_left[0], top_left[1] + v[1])]
    poly(canvas, side, PALE + (255,))
    topf = [top_left, (top_left[0] + u[0], top_left[1] + u[1]),
            (top_left[0] + u[0] - d * COS30, top_left[1] + u[1] + d * SIN30),
            (top_left[0] - d * COS30, top_left[1] + d * SIN30)]
    poly(canvas, topf, WHITE + (255,))
    paste_affine(canvas, img, top_left, u, v)


def disc(canvas, cx, cy, rx, thick, top=PERI, side=PERI_DARK, glow=True):
    """Floating isometric disc platform (ellipse rx : rx/sqrt(3))."""
    ry = rx * math.tan(math.radians(30))
    if glow:
        g = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        ImageDraw.Draw(g).ellipse([s(cx - rx * 1.25), s(cy - ry * 1.1), s(cx + rx * 1.25),
                                   s(cy + ry * 1.5 + thick)], fill=(80, 130, 255, 90))
        canvas.alpha_composite(g.filter(ImageFilter.GaussianBlur(s(40))))
    d = ImageDraw.Draw(canvas)
    d.ellipse([s(cx - rx), s(cy - ry + thick), s(cx + rx), s(cy + ry + thick)], fill=side)
    d.rectangle([s(cx - rx), s(cy), s(cx + rx), s(cy + thick)], fill=side)
    d.ellipse([s(cx - rx), s(cy - ry), s(cx + rx), s(cy + ry)], fill=top)


def dotted_line(canvas, pts, width=2.0, dots=True, color=GLOW):
    d = ImageDraw.Draw(canvas)
    sp = [(s(x), s(y)) for x, y in pts]
    d.line(sp, fill=color + (200,), width=int(s(width)), joint="curve")
    if dots:
        for x, y in sp[1:-1] + [sp[-1]]:
            r = s(4.5)
            d.ellipse([x - r, y - r, x + r, y + r], fill=color + (255,))


def sparkle_dots(canvas, pts, r=3.0, color=GLOW):
    g = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(g)
    for x, y in pts:
        d.ellipse([s(x - r * 2.5), s(y - r * 2.5), s(x + r * 2.5), s(y + r * 2.5)],
                  fill=color + (90,))
    canvas.alpha_composite(g.filter(ImageFilter.GaussianBlur(s(4))))
    d = ImageDraw.Draw(canvas)
    for x, y in pts:
        d.ellipse([s(x - r), s(y - r), s(x + r), s(y + r)], fill=color + (255,))


def chip(canvas, cx, cy, size, h=16.0):
    """Isometric microchip: a flat prism with pins, a bright inner die."""
    a = size / 2
    # pins along the two visible lower edges
    d = ImageDraw.Draw(canvas)
    n = 6
    for i in range(n):
        t = -a + (i + 0.5) * size / n
        for (x0, y0, x1, y1) in [(a, t, a + 14, t), (t, a, t, a + 14)]:
            p, q = iso(cx, cy, x0, y0, 2), iso(cx, cy, x1, y1, 2)
            d.line([p, q], fill=WHITE, width=int(s(4)))
    top = [iso(cx, cy, -a, -a, h), iso(cx, cy, a, -a, h), iso(cx, cy, a, a, h), iso(cx, cy, -a, a, h)]
    right = [iso(cx, cy, a, -a, h), iso(cx, cy, a, a, h), iso(cx, cy, a, a, 0), iso(cx, cy, a, -a, 0)]
    left = [iso(cx, cy, -a, a, h), iso(cx, cy, a, a, h), iso(cx, cy, a, a, 0), iso(cx, cy, -a, a, 0)]
    poly(canvas, right, (0xC4, 0xD2, 0xFF, 255))
    poly(canvas, left, (0x9C, 0xB2, 0xFF, 255))
    poly(canvas, top, WHITE + (255,))
    b = a * 0.58
    die = [iso(cx, cy, -b, -b, h), iso(cx, cy, b, -b, h), iso(cx, cy, b, b, h), iso(cx, cy, -b, b, h)]
    g = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    ImageDraw.Draw(g).polygon(die, fill=BLUE + (255,))
    canvas.alpha_composite(g.filter(ImageFilter.GaussianBlur(s(10))))
    poly(canvas, die, BLUE + (255,))
    c = a * 0.3
    core = [iso(cx, cy, -c, -c, h), iso(cx, cy, c, -c, h), iso(cx, cy, c, c, h), iso(cx, cy, -c, c, h)]
    poly(canvas, core, LIME + (255,))


def finish(canvas: Image.Image, name: str) -> None:
    out = canvas.convert("RGB").resize((W, H), Image.LANCZOS)
    path = OUT / name
    out.save(path, quality=88, optimize=True, progressive=True)
    print(f"{path.relative_to(ROOT)}  {out.size[0]}x{out.size[1]}  {path.stat().st_size // 1024} KB")


# --------------------------------------------------------------------------- candidates
def hero_a() -> None:
    """Upright 4x3 wall of the twelve generated species on a floating platform."""
    c = background()
    cx, cy = 800, 610
    disc(c, cx, cy, 390, 24)
    names = ["american_black_bear", "aye_aye", "grevy_s_zebra", "kinkajou",
             "lion", "mountain_zebra", "pangolin_family", "plains_zebra",
             "red_fox", "ringtail", "saiga", "water_deer"]
    boxed = {"american_black_bear", "kinkajou", "grevy_s_zebra", "red_fox"}
    cw, ch, gap = 150.0, 112.5, 14.0
    cols, rows = 4, 3
    wall_w = cols * cw + (cols - 1) * gap
    # bottom-left of the wall on the floor, so that the wall is centred on the disc
    base_x = cx - wall_w * COS30 / 2
    base_y = cy - wall_w * SIN30 / 2 + 10
    lift = 14.0
    # back-to-front: draw top row first is irrelevant (same plane); draw left to right
    for r in range(rows):
        for k in range(cols):
            n = names[r * cols + k]
            img = load(TWELVE / f"{n}.png", n if n in boxed else None)
            along = k * (cw + gap)
            ox = base_x + along * COS30
            oy = base_y + along * SIN30 - lift - (rows - 1 - r) * (ch + gap)
            upright_card(c, img, (ox, oy), cw, ch, shadow=(r == rows - 1))
    # connection lines and dots, as in the blog's data-flow heroes
    dotted_line(c, [(cx + 395, cy + 30), (cx + 500, cy - 30), (cx + 590, cy + 22)])
    dotted_line(c, [(cx - 395, cy + 30), (cx - 500, cy - 30), (cx - 590, cy + 22)])
    sparkle_dots(c, [(330, 250), (1290, 230), (1200, 150), (420, 170), (1380, 420), (250, 440)])
    finish(c, "hero_a.jpg")


def hero_b() -> None:
    """Stack of generated cards -> chip -> one detected binturong card."""
    c = background()
    # platforms
    disc(c, 470, 560, 210, 20)
    disc(c, 830, 420, 120, 16)
    disc(c, 1150, 590, 220, 20)
    # flow lines (drawn first, objects sit on top)
    dotted_line(c, [(600, 505), (700, 448), (745, 473)], width=2.5)
    dotted_line(c, [(915, 473), (965, 448), (1020, 480)], width=2.5)
    # left: a deck of generated cards, back to front along the iso y axis
    stack = ["water_deer", "saiga", "ringtail", "pangolin_family", "lion", "plains_zebra"]
    for i, n in enumerate(stack):
        off = (len(stack) - 1 - i) * 18.0
        upright_card(c, load(TWELVE / f"{n}.png"), (300 + off * COS30, 560 - off * SIN30),
                     240, 180, shadow=(i == len(stack) - 1))
    # centre: the chip
    chip(c, 830, 412, 110)
    # right: the detection result
    img = load(GEN / "binturong/binturong_gemini_0001.png", "binturong_gemini_0001")
    upright_card(c, img, (985, 565), 330, 180)
    sparkle_dots(c, [(300, 200), (1320, 220), (1230, 160), (1400, 470), (220, 470), (830, 190), (800, 800)])
    finish(c, "hero_b.jpg")


def hero_c() -> None:
    """Binocular field of view around a generated binturong with a detection box."""
    c = background()
    cx, cy, r = 800, 455, 255
    # field of view: two overlapping circles, the image clipped to their union
    src = Image.open(GEN / "binturong/binturong_gemini_0001.png").convert("RGB")
    fw, fh = int(s(2 * r + r * 0.9)), int(s(2 * r))
    sw, sh = src.size
    scale = max(fw / sw, fh / sh)
    src = src.resize((int(sw * scale), int(sh * scale)), Image.LANCZOS)
    ox = (src.width - fw) // 2
    src = src.crop((ox, 0, ox + fw, fh))
    # detection box in the field-of-view frame
    x0, y0, x1, y1 = BOXES["binturong_gemini_0001"]
    bx = lambda f: (f * sw * scale - ox)  # noqa: E731
    by = lambda f: (f * sh * scale)       # noqa: E731
    dd = ImageDraw.Draw(src)
    lw = int(s(5))
    bb = [max(bx(x0), s(60)), max(by(y0) + s(40), s(70)), min(bx(x1), fw - s(60)), min(by(y1), fh - s(55))]
    dd.rectangle(bb, outline=LIME, width=lw)
    dd.rectangle([bb[0], bb[1] - s(22), bb[0] + s(120), bb[1]], fill=LIME)
    mask = Image.new("L", (fw, fh), 0)
    md = ImageDraw.Draw(mask)
    off = r * 0.9
    md.ellipse([0, 0, s(2 * r), s(2 * r)], fill=255)
    md.ellipse([s(off), 0, s(off + 2 * r), s(2 * r)], fill=255)
    left = int(s(cx - r - off / 2))
    top = int(s(cy - r))
    # thin outer ring around the union of both circles, then glow, white rim, image
    ring = ImageDraw.Draw(c)
    for k in (0, 1):
        ex = cx - r - off / 2 + k * off
        ring.ellipse([s(ex - 34), s(cy - r - 34), s(ex + 2 * r + 34), s(cy + r + 34)],
                     outline=GLOW + (255,), width=int(s(2)))
    inner = Image.new("RGBA", c.size, (0, 0, 0, 0))
    idr = ImageDraw.Draw(inner)
    for k in (0, 1):
        ex = cx - r - off / 2 + k * off
        idr.ellipse([s(ex - 32), s(cy - r - 32), s(ex + 2 * r + 32), s(cy + r + 32)], fill=(0, 0, 0, 255))
    # keep only the union's outline: erase ring pixels inside the union of the inner discs
    arr = np.array(c)
    bg = np.array(background())
    m = np.array(inner)[..., 3] > 0
    arr[m] = bg[m]
    c = Image.fromarray(arr, "RGBA")
    g2 = Image.new("RGBA", c.size, (0, 0, 0, 0))
    gd2 = ImageDraw.Draw(g2)
    for k in (0, 1):
        ex = cx - r - off / 2 + k * off
        gd2.ellipse([s(ex - 30), s(cy - r - 30), s(ex + 2 * r + 30), s(cy + r + 30)], fill=(120, 170, 255, 120))
    c.alpha_composite(g2.filter(ImageFilter.GaussianBlur(s(35))))
    rim = Image.new("RGBA", c.size, (0, 0, 0, 0))
    rd = ImageDraw.Draw(rim)
    for k in (0, 1):
        ex = cx - r - off / 2 + k * off
        rd.ellipse([s(ex - 12), s(cy - r - 12), s(ex + 2 * r + 12), s(cy + r + 12)], fill=WHITE + (255,))
    c.alpha_composite(rim)
    c.paste(src, (left, top), mask)
    # floating generated cards around the view, tilted isometrically
    cards = [("red_fox", (225, 330)), ("lion", (1255, 250)), ("plains_zebra", (1200, 700)),
             ("saiga", (260, 690))]
    for n, (x, y) in cards:
        upright_card(c, load(TWELVE / f"{n}.png"), (x, y), 150, 112, shadow=False)
    dotted_line(c, [(350, 300), (395, 330), (420, 380)])
    dotted_line(c, [(1250, 245), (1205, 280), (1185, 330)])
    sparkle_dots(c, [(160, 520), (1440, 470), (700, 110), (930, 850), (470, 840), (1100, 90)])
    finish(c, "hero_c.jpg")


if __name__ == "__main__":
    hero_a()
    hero_b()
    hero_c()
