# /// script
# requires-python = ">=3.11"
# dependencies = ["pillow>=10"]
# ///
"""Photo composites for the inovex blog article: one image per figure.

Each figure of the article is a single image, so that a WordPress gallery cannot reflow
it and the labels stay with their photos. This script combines the source images in
thesis/blog/images/figNN_*/ into one labelled JPEG per figure, in the light look of the
charts (plots/make_plots.py): white page, Atkinson Hyperlegible, navy ink, and the
article's colour code — navy pill = real photo, green pill = AI-generated image.

Writes 1600 px wide JPEGs to thesis/blog/figures/ (the paths draft.md and media.md
reference) and a copy into the figure's own folder under thesis/blog/images/:
  rejects.jpg       Fig. 2  four rejected images, each with the species folder it came from
  binturong.jpg     Fig. 3  3 x 5 grid: real photos, three local generators, Gemini
  lion_prompts.jpg  Fig. 5  one lion per prompt length, above its prompt as a text sheet
  kinkajou.jpg      Fig. 6  2 x 2: real photo, FLUX.2 klein (ringed tail circled), SD 3.5 L, HiDream
  elephants.jpg     Fig. 10 Asian vs African elephant, confusion arrows, training-image squares

rejects.jpg and binturong.jpg contain real photos whose licence is not cleared yet; they
are gitignored like their sources. elephants.jpg uses the generated elephants by default
(pass --real-elephants for the real photos, whose licence is also not cleared).

PEP 723 script rather than the repo-root uv project: that project is the Linux/CUDA
training environment, and these images are composed natively on the Mac.

Run (from the repo root):
    uv run --script thesis/blog/figures/make_figures.py [--real-elephants]
"""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

HERE = Path(__file__).resolve().parent
BLOG = HERE.parent
IMAGES = BLOG / "images"
FONTS = BLOG.parent / "presentation" / "slides" / "assets" / "fonts"
PROMPTS = BLOG.parent / "presentation" / "slides" / "assets" / "prompts"

SS = 2          # supersampling factor: draw at 2x, downscale once at the end
W = 1600        # output width
PAD = 40        # outer margin
GAP = 16        # gap between photos
RADIUS = 14     # photo corner radius

# The charts' light theme (plots/make_plots.py)
BG = (255, 255, 255)
INK = (0x06, 0x1B, 0x59)
MUTED = (0x5B, 0x66, 0x80)
GRID = (0xE3, 0xE7, 0xEF)
GENERATED = (0x17, 0xA3, 0x4A)
REAL = INK
AMBER = (0xE0, 0xA1, 0x00)


def font(size: float, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "AtkinsonHyperlegible-Bold.ttf" if bold else "AtkinsonHyperlegible-Regular.ttf"
    return ImageFont.truetype(str(FONTS / name), round(size * SS))


class Canvas:
    """A white page drawn at SS x scale; all coordinates are given in output pixels."""

    def __init__(self, h: int):
        self.h = h
        self.im = Image.new("RGB", (W * SS, h * SS), BG)
        self.d = ImageDraw.Draw(self.im)

    @staticmethod
    def s(*v: float) -> list[int]:
        return [round(x * SS) for x in v]

    def photo(self, src: Image.Image, x: float, y: float, w: float, h: float,
              focus: tuple[float, float] = (0.5, 0.5), border: bool = False) -> None:
        """Crop `src` to w:h around `focus` (image fractions), paste with rounded corners."""
        tile = ImageOps.fit(src.convert("RGB"), (round(w * SS), round(h * SS)),
                            Image.LANCZOS, centering=focus)
        mask = Image.new("L", tile.size, 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, *tile.size], RADIUS * SS, fill=255)
        self.im.paste(tile, self.s(x, y), mask)
        if border:
            self.d.rounded_rectangle(self.s(x, y, x + w, y + h), RADIUS * SS, outline=GRID,
                                     width=2 * SS)

    def text(self, x: float, y: float, s: str, size: float, colour=INK, bold=False,
             anchor: str = "la") -> None:
        self.d.text(self.s(x, y), s, font=font(size, bold), fill=colour, anchor=anchor)

    def pill(self, x: float, y: float, s: str, fill, size: float = 22, anchor: str = "l") -> float:
        """A rounded label; (x, y) is its left-centre (anchor l), centre (m) or right-centre (r)."""
        f = font(size, bold=True)
        tw = f.getlength(s) / SS
        ph, px = size * 1.55, size * 0.6
        pw = tw + 2 * px
        x0 = {"l": x, "m": x - pw / 2, "r": x - pw}[anchor]
        self.d.rounded_rectangle(self.s(x0, y - ph / 2, x0 + pw, y + ph / 2), round(ph / 2 * SS),
                                 fill=fill)
        self.d.text(self.s(x0 + pw / 2, y), s, font=f, fill=(255, 255, 255), anchor="mm")
        return pw

    def save(self, name: str, folder: str) -> None:
        out = self.im.resize((W, self.h), Image.LANCZOS)
        path = HERE / f"{name}.jpg"
        out.save(path, quality=90, optimize=True, progressive=True)
        shutil.copy(path, IMAGES / folder / f"{name}.jpg")
        print(f"wrote {path.relative_to(BLOG.parents[1])} ({W}x{self.h})")


def load(folder: str, name: str) -> Image.Image:
    im = Image.open(IMAGES / folder / name)
    if im.mode in ("RGBA", "LA", "P"):  # flatten transparency onto white
        im = im.convert("RGBA")
        flat = Image.new("RGB", im.size, BG)
        flat.paste(im, mask=im.split()[3])
        return flat
    return im.convert("RGB")


# ---------------------------------------------------------------------------------------
# Fig. 2: rejected images
# ---------------------------------------------------------------------------------------

def rejects() -> None:
    items = [  # (file, species folder, what it actually shows)
        ("reject_tutankhamun_cheetah.jpg", "cheetah", "a tomb painting"),
        ("reject_elephant_mosaic.jpg", "African elephant", "a floor mosaic"),
        ("reject_aardvark_mask.jpg", "aardvark", "a carved mask"),
        ("reject_buffalo_range_map.png", "African buffalo", "a range map"),
    ]
    ims = [load("fig02_rejects", f) for f, _, _ in items]
    # One row, common height, natural aspect ratios (nothing cropped).
    aspects = [im.width / im.height for im in ims]
    ph = (W - 2 * PAD - GAP * (len(ims) - 1)) / sum(aspects)
    top, label_h = PAD, 92
    c = Canvas(round(top + ph + label_h + 14))
    x = PAD
    for im, a, (f, folder, shows) in zip(ims, aspects, items):
        pw = ph * a
        c.photo(im, x, top, pw, ph, border=f.endswith(".png"))
        c.text(x, top + ph + 18, "filed under", 19, MUTED)
        c.text(x, top + ph + 42, f"“{folder}”", 24, INK, bold=True)
        c.text(x, top + ph + 74, shows, 19, MUTED)
        x += pw + GAP
    c.save("rejects", "fig02_rejects")


# ---------------------------------------------------------------------------------------
# Fig. 3: binturong, real against four generators
# ---------------------------------------------------------------------------------------

def binturong() -> None:
    cols = [  # (label, file stem, real?, crop focus per image)
        ("Real photo", "binturong_real", True, ["1", "2", "3"]),
        ("RealVisXL", "binturong_realvisxl-lightning", False, ["0001", "0003", "0005"]),
        ("SD 3.5 Medium", "binturong_sd35m", False, ["0001", "0002", "0003"]),
        ("FLUX.1 schnell", "binturong_flux-schnell", False, ["0001", "0004", "0014"]),
        ("Gemini", "binturong_gemini", False, ["0001", "0004", "0007"]),
    ]
    focus = {"binturong_real_1": (0.6, 0.5), "binturong_real_2": (0.4, 0.5),
             "binturong_real_3": (0.55, 0.5), "binturong_gemini_0001": (0.4, 0.5),
             "binturong_gemini_0004": (0.4, 0.5)}
    cw = (W - 2 * PAD - GAP * 4) / 5
    ch = cw * 3 / 4
    group_y, pill_y, top = PAD + 12, PAD + 64, PAD + 96
    c = Canvas(round(top + 3 * ch + 2 * GAP + PAD))

    # Group headers: which generators run locally, which is an API model.
    def bracket(c0: int, c1: int, s: str, colour) -> None:
        x0, x1 = PAD + c0 * (cw + GAP), PAD + c1 * (cw + GAP) + cw
        c.text((x0 + x1) / 2, group_y, s, 20, colour, anchor="mm")
        tw = font(20).getlength(s) / SS
        for a, b in ((x0 + 4, (x0 + x1) / 2 - tw / 2 - 12), ((x0 + x1) / 2 + tw / 2 + 12, x1 - 4)):
            c.d.line(c.s(a, group_y, b, group_y), fill=GRID, width=2 * SS)
        for xe in (x0 + 4, x1 - 4):
            c.d.line(c.s(xe, group_y, xe, group_y + 10), fill=GRID, width=2 * SS)

    bracket(1, 3, "runs locally", MUTED)
    bracket(4, 4, "API model", MUTED)

    for i, (label, stem, is_real, ids) in enumerate(cols):
        x = PAD + i * (cw + GAP)
        c.pill(x + cw / 2, pill_y, label, REAL if is_real else GENERATED, 21, anchor="m")
        for j, k in enumerate(ids):
            name = f"{stem}_{k}"
            c.photo(load("fig03_binturong", f"{name}.jpg"), x, top + j * (ch + GAP), cw, ch,
                    focus=focus.get(name, (0.5, 0.45)))
    c.save("binturong", "fig03_binturong")


# ---------------------------------------------------------------------------------------
# Fig. 5: one lion, four prompt lengths
# ---------------------------------------------------------------------------------------

def wrap(text: str, f: ImageFont.FreeTypeFont, width_px: float) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        cand = f"{line} {word}".strip()
        if f.getlength(cand) <= width_px or not line:
            line = cand
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def lion_prompts() -> None:
    cols = [  # (image, generator, prompt file, limit, words)
        ("prompt_full_gemini.jpg", "Gemini", "lion_full.txt", "full prompt", 4642),
        ("prompt_maxlen512_flux2-klein.jpg", "FLUX.2 klein", "lion_maxlen512.txt", "≤ 512 tokens", 380),
        ("prompt_maxlen256_sd35m.jpg", "SD 3.5 Medium", "lion_maxlen256.txt", "≤ 256 tokens", 149),
        ("prompt_compressed_realvisxl.jpg", "RealVisXL", "lion_compressed.txt", "≤ 75 tokens", 20),
    ]
    cw = (W - 2 * PAD - GAP * 3) / 4
    ch = cw * 3 / 4
    top = PAD + 22
    meta_y = top + ch + 34
    sheet_y = meta_y + 26
    pad_in, fs, lh = 12, 7.2, 9.4               # sheet padding, prompt font size, line height
    f = font(fs)
    sheets = [wrap((PROMPTS / p).read_text(), f, (cw - 2 * pad_in) * SS) for _, _, p, _, _ in cols]
    max_lines = len(sheets[1]) + 4             # the full prompt is cut a little below FLUX's
    sheet_hs = [2 * pad_in + min(len(s), max_lines) * lh for s in sheets]
    c = Canvas(round(sheet_y + max(sheet_hs) + 36 + PAD))

    for i, ((img, gen, _, limit, words), lines, sh) in enumerate(zip(cols, sheets, sheet_hs)):
        x = PAD + i * (cw + GAP)
        c.photo(load("fig05_lion_prompts", img), x, top, cw, ch)
        c.pill(x + 12, top + 12 + 16, gen, GENERATED, 19)
        c.text(x, meta_y, f"{words:,} words", 24, INK, bold=True, anchor="ls")
        tw = font(24, True).getlength(f"{words:,} words") / SS
        c.text(x + tw + 12, meta_y, limit, 19, MUTED, anchor="ls")
        # The prompt as a sheet of text, its height proportional to its length.
        c.d.rounded_rectangle(c.s(x, sheet_y, x + cw, sheet_y + sh), 8 * SS,
                              fill=(0xF5, 0xF7, 0xFB), outline=GRID, width=SS)
        cut = len(lines) > max_lines
        for k, line in enumerate(lines[:max_lines]):
            fade = max(0.0, (k - (max_lines - 8)) / 8) if cut else 0.0
            col = tuple(round(m + (245 - m) * fade) for m in MUTED)
            c.text(x + pad_in, sheet_y + pad_in + k * lh, line, fs, col)
        if cut:
            more = len(lines) - max_lines
            c.text(x + cw / 2, sheet_y + sh + 22, f"… and {more:,} more lines", 18, MUTED,
                   anchor="mm")
    c.text(PAD, PAD - 6, "One prompt for a lion, cut to what each generator can read", 22, INK,
           bold=True, anchor="lt")
    c.save("lion_prompts", "fig05_lion_prompts")


# ---------------------------------------------------------------------------------------
# Fig. 6: kinkajou
# ---------------------------------------------------------------------------------------

def kinkajou() -> None:
    cells = [  # (file, label, real?, focus)
        ("kinkajou_real.jpg", "Real photo", True, (0.5, 0.5)),
        ("kinkajou_flux2-klein-9b.jpg", "FLUX.2 klein", False, (0.5, 0.5)),
        ("kinkajou_sd35-large.jpg", "SD 3.5 Large", False, (0.5, 0.5)),
        ("kinkajou_hidream-i1.jpg", "HiDream-I1", False, (0.5, 0.5)),
    ]
    cw = (W - 2 * PAD - GAP) / 2
    ch = cw * 3 / 4
    top = PAD
    c = Canvas(round(top + 2 * ch + GAP + PAD))
    for i, (f, label, is_real, foc) in enumerate(cells):
        x = PAD + (i % 2) * (cw + GAP)
        y = top + (i // 2) * (ch + GAP)
        c.photo(load("fig06_kinkajou", f), x, y, cw, ch, focus=foc)
        c.pill(x + 18, y + 18 + 20, label, REAL if is_real else GENERATED, 24)
        if f.startswith("kinkajou_flux"):
            # Ring around the ringed tail (lower left of the FLUX image), with a label.
            ex, ey, rx, ry = x + 0.33 * cw, y + 0.86 * ch, 0.15 * cw, 0.12 * ch
            c.d.ellipse(c.s(ex - rx, ey - ry, ex + rx, ey + ry), outline=AMBER, width=6 * SS)
            lx, ly = ex + rx + 16, ey - 6
            label_txt = "ringed tail"
            tw = font(24, True).getlength(label_txt) / SS
            c.d.rounded_rectangle(c.s(lx - 10, ly - 22, lx + tw + 10, ly + 20), 10 * SS, fill=AMBER)
            c.text(lx, ly, label_txt, 24, (255, 255, 255), bold=True, anchor="lm")
    c.save("kinkajou", "fig06_kinkajou")


# ---------------------------------------------------------------------------------------
# Fig. 10: elephants
# ---------------------------------------------------------------------------------------

def arrow(c: Canvas, x0: float, x1: float, y: float, width: float, colour) -> None:
    """Horizontal arrow from x0 to x1 (either direction) with a triangular head."""
    d = 1 if x1 > x0 else -1
    head = max(16, width * 2.2)
    c.d.line(c.s(x0, y, x1 - d * head * 0.9, y), fill=colour, width=round(width * SS))
    c.d.polygon([tuple(c.s(x1, y)), tuple(c.s(x1 - d * head, y - head * 0.62)),
                 tuple(c.s(x1 - d * head, y + head * 0.62))], fill=colour)


def squares(c: Canvas, x: float, y: float, real: int, gen: int, size: float = 22) -> None:
    for k in range(real + gen):
        xx = x + k * (size + 6)
        c.d.rounded_rectangle(c.s(xx, y, xx + size, y + size), 4 * SS,
                              fill=REAL if k < real else GENERATED)


def elephants(real_photos: bool) -> None:
    pre = "" if real_photos else "synthetic_"
    mid = 300                                  # space for the arrows between the photos
    pw = (W - 2 * PAD - mid) / 2
    ph = pw * 3 / 4
    top = PAD
    lab_y = top + ph + 34
    c = Canvas(round(lab_y + 150 + PAD))
    tag, tag_col = ("Real photo", REAL) if real_photos else ("AI-generated", GENERATED)
    sides = [  # (x, file, name, real training images, generated training images, caption)
        (PAD, "asian_elephant.jpg", "Asian elephant", 1, 1, "100 real + 100 generated"),
        (W - PAD - pw, "african_elephant.jpg", "African elephant", 15, 0, "1,500 real"),
    ]
    for x, f, name, n_real, n_gen, cap in sides:
        c.photo(load("fig10_elephants", pre + f), x, top, pw, ph)
        c.pill(x + 16, top + 16 + 17, tag, tag_col, 20)
        c.text(x, lab_y, name, 30, INK, bold=True, anchor="lm")
        c.text(x, lab_y + 40, f"{cap} training images", 21, MUTED, anchor="lm")
        squares(c, x, lab_y + 70, n_real, n_gen)

    # Confusion arrows between the photos.
    ax0, ax1 = PAD + pw + 28, W - PAD - pw - 28
    cx = (ax0 + ax1) / 2
    y1, y2 = top + ph * 0.36, top + ph * 0.70
    arrow(c, ax0, ax1, y1, 22, INK)
    c.text(cx, y1 - 34, "79 %", 46, INK, bold=True, anchor="ms")
    c.text(cx, y1 + 46, "of Asian elephants", 18, MUTED, anchor="mm")
    c.text(cx, y1 + 68, "named African", 18, MUTED, anchor="mm")
    arrow(c, ax1, ax0, y2, 3, MUTED)
    c.text(cx, y2 - 24, "0.2 %", 30, MUTED, bold=True, anchor="ms")
    c.text(cx, y2 + 32, "the other way", 18, MUTED, anchor="mm")

    # Key for the squares, right-aligned under the African elephant.
    ky = lab_y + 120
    kx = W - PAD
    for s, colour in (("generated", GENERATED), ("real", REAL)):
        tw = font(18).getlength(s) / SS
        c.text(kx, ky + 11, s, 18, MUTED, anchor="rm")
        kx -= tw + 8 + 18
        c.d.rounded_rectangle(c.s(kx, ky + 2, kx + 18, ky + 20), 4 * SS, fill=colour)
        kx -= 22
    c.text(kx, ky + 11, "one square = 100 training images:", 18, MUTED, anchor="rm")
    c.save("elephants", "fig10_elephants")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--real-elephants", action="store_true",
                    help="use the real elephant photos (licence not cleared) for Fig. 10")
    args = ap.parse_args()
    rejects()
    binturong()
    lion_prompts()
    kinkajou()
    elephants(args.real_elephants)


if __name__ == "__main__":
    main()
