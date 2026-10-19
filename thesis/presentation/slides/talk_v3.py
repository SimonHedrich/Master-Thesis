"""Final thesis presentation, version 3: talk_v2 cut down towards 20 minutes.

Same six acts as talk_v2.py. Changes: Swarovski Optik logo and the BirdID clip on the
product slide instead of a BirdID slide, the model ensemble folded into the MammalID
slide (with YOLOv5 as the detector in the AX Visio), the prompt and the production-set wall moved before the
zebras, a twelve-species slide before the generator ranking, no rare-animals slide, and
a summary with six named steps. Scene names end in 3 so v2's slide files stay intact.
Images come from thesis/presentation/images/ (built by collect_images.py); numbers from
data.py. talk_v2.py keeps the previous version.

Render (from thesis/presentation/slides/; the summary needs the earlier scenes rendered
first in the same run, because it reuses their snapshots):
    uv run manim render -qk --frame_rate 30 talk_v3.py UseCase3 Data3 Generative3 Bands3 Models3 Summary3
Present:
    uv run python present.py -F UseCase3 Data3 Generative3 Bands3 Models3 Summary3
"""

from __future__ import annotations

import csv
import hashlib
import subprocess
import textwrap

import numpy as np
from manim import (
    DL,
    Animation,
    DOWN,
    DR,
    LEFT,
    ORIGIN,
    RIGHT,
    UL,
    UP,
    UR,
    Circle,
    Create,
    DashedLine,
    Dot,
    FadeIn,
    FadeOut,
    GrowFromCenter,
    GrowFromEdge,
    Group,
    ImageMobject,
    LaggedStart,
    Line,
    Rectangle,
    Restore,
    RoundedRectangle,
    Square,
    SurroundingRectangle,
    Text,
    Transform,
    VGroup,
    linear,
    register_font,
)
from manim_slides import Slide
from PIL import Image, ImageDraw, ImageFont

import data as D
from talk import axis_label, icon, labelled, prompt_card, tile, tiles
from theme import (
    BLUE,
    CACHE,
    FONTS,
    FRAME_H,
    FRAME_W,
    GREEN,
    HERE,
    LEFT_X,
    MIST,
    NAVY,
    RED,
    REPO,
    SKY,
    TEAL,
    WHITE,
    arrow,
    background,
    big_number,
    card,
    collage,
    curved_arrow,
    hbar,
    photo,
    pill,
    pptx_box,
    stopwatch,
    title,
    txt,
    vbar,
)

IMG = HERE.parent / "images"
SNAPS = CACHE / "snaps_v3"
PROMPTS = HERE / "assets" / "prompts"
# One colour per image source (images-per-class chart), in data.SOURCES order.
SOURCE_COLORS = [SKY, "#FFD45C", "#FF8A3D", MIST, RED]
# Synthetic test images (final-dataset chart and its legend).
SYNTH_TEST = "#FFD45C"
# Real test images and the results measured on them: pink, which stands out on the blue
# background better than sky blue and is not used for anything else.
REAL_TEST = "#FF79C6"


# =================================================================================
# Shared pieces
# =================================================================================
class Talk3(Slide):
    """Background + logo on every frame; snap() stores frames for the summary."""

    def construct(self):
        with (
            register_font(str(FONTS / "AtkinsonHyperlegible-Regular.ttf")),
            register_font(str(FONTS / "AtkinsonHyperlegible-Bold.ttf")),
        ):
            self.bg = background()
            self.logo = photo(IMG / "deck/inovex_logo.png", height=0.5, radius=0).to_corner(UL, buff=0.25)
            self.add(self.bg, self.logo)
            self.build()

    def build(self):  # pragma: no cover - overridden
        raise NotImplementedError

    def content(self):
        return [m for m in self.mobjects if m is not self.bg and m is not self.logo]

    def clear(self, run_time: float = 0.45):
        old = self.content()
        if old:
            self.play(*[FadeOut(m) for m in old], run_time=run_time)

    def page(self):
        self.next_slide()
        self.clear()

    def play(self, *args, **kwargs):
        """Animations without their own run time take 0.7 s instead of manim's 1 s: shorter
        pauses, e.g. between a title and the content that follows it."""
        # (.animate builders answer every getattr, so only real Animations are asked for theirs)
        if "run_time" not in kwargs and all(a.run_time == 1.0 for a in args if isinstance(a, Animation)):
            kwargs["run_time"] = 0.7
        super().play(*args, **kwargs)

    def snap(self, name: str):
        """Save the current frame; the summary scene shows it again."""
        SNAPS.mkdir(parents=True, exist_ok=True)
        Image.fromarray(self.renderer.get_frame()).convert("RGB").save(SNAPS / f"{name}.png")



def video_frames(src, width_px: int, radius: float = 0.04) -> list:
    """Decode a clip into rounded-corner RGBA frames (cached), at its own frame rate."""
    key = hashlib.md5(f"{src}|{src.stat().st_mtime}|{width_px}|{radius}".encode()).hexdigest()[:12]
    out = CACHE / f"video_{key}"
    if not out.exists():
        tmp = CACHE / f"video_{key}_tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(src), "-vf", f"scale={width_px}:-2", str(tmp / "%04d.png")],
            check=True,
        )
        for f in sorted(tmp.glob("*.png")):
            im = Image.open(f).convert("RGBA")
            mask = Image.new("L", im.size, 0)
            ImageDraw.Draw(mask).rounded_rectangle((0, 0, im.size[0] - 1, im.size[1] - 1), int(min(im.size) * radius), fill=255)
            im.putalpha(mask)
            im.save(f)
        tmp.rename(out)
    return sorted(out.glob("*.png"))


class PlayVideo(Animation):
    """Swap an ImageMobject's pixels through the clip's frames; it ends on the last one."""

    def __init__(self, mobject: ImageMobject, frames: list, fps: float = 25, **kw):
        self.frames, self.shown = frames, -1
        super().__init__(mobject, run_time=len(frames) / fps, rate_func=linear, **kw)

    def interpolate_mobject(self, alpha: float) -> None:
        i = min(len(self.frames) - 1, int(alpha * len(self.frames)))
        if i != self.shown:
            self.mobject.pixel_array = np.array(Image.open(self.frames[i]).convert("RGBA"))
            self.shown = i


def copyright_note(s: str = "© SWAROVSKI OPTIK") -> Text:
    return txt(s, 13, MIST)


def disc(inner, r: float = 0.95) -> Group:
    c = Circle(radius=r).set_fill(TEAL, 0.35).set_stroke(MIST, 2)
    return Group(c, inner.move_to(c))


def prompt_text_card(name: str, width: float, max_height: float) -> ImageMobject:
    """A prompt typeset at one fixed size, so card length shows prompt length."""
    out = CACHE / f"prompt_{name}_v2.png"
    px_per_unit = 300
    if not out.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        text = (PROMPTS / f"lion_{name}.txt").read_text()
        lines = []
        for para in text.splitlines():
            lines += textwrap.wrap(para, 92) or [""]
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 15)
        lh, pad = 19, 22
        w = int(width * px_per_unit)
        max_lines = int((max_height * px_per_unit - 2 * pad) / lh)
        cut = len(lines) > max_lines
        lines = lines[:max_lines]
        h = 2 * pad + lh * len(lines)
        im = Image.new("RGBA", (w, h), (255, 255, 255, 0))
        d = ImageDraw.Draw(im)
        d.rounded_rectangle((0, 0, w - 1, h - 1), 26, fill=(255, 255, 255, 255))
        for i, line in enumerate(lines):
            d.text((pad, pad + i * lh), line, fill=(40, 50, 90), font=font)
        if cut:  # fade card and text out to transparent: the prompt keeps going
            fade = 8 * lh
            px = np.array(im)
            ramp = np.linspace(1.0, 0.0, fade)[:, None]
            px[h - fade :, :, 3] = (px[h - fade :, :, 3] * ramp).astype(np.uint8)
            im = Image.fromarray(px)
        im.save(out)
    m = ImageMobject(str(out))
    return m.scale_to_fit_width(width)


# =================================================================================
# Act 1 — The use case: BirdID, MammalID, and how to improve it
# =================================================================================
class UseCase3(Talk3):
    def build(self):
        # 1 — Title
        tag = pill("Master Thesis", 20)
        head = txt("Real-Time Object Recognition\nwith Neural Networks\non Embedded Platforms", 50, bold=True, line_spacing=0.9)
        head.move_to([LEFT_X, 0.9, 0], aligned_edge=LEFT)
        tag.next_to(head, UP, buff=0.35).align_to(head, LEFT)
        face = photo(IMG / "deck/portrait.png", height=1.35, radius=0.5)
        name = txt("Simon Hedrich", 28)
        who = Group(face, name).arrange(RIGHT, buff=0.35).move_to([LEFT_X, -2.6, 0], aligned_edge=LEFT)
        self.play(FadeIn(tag, shift=UP * 0.2), FadeIn(head, shift=UP * 0.2), FadeIn(who), run_time=1.0)
        self.page()

        # 2 — The product
        shop = photo(IMG / "deck/shop_ax_visio.png", width=12.4, radius=0.02).move_to(DOWN * 0.25)
        # the Swarovski Optik logo over the web page's own logo, top left
        so = photo(IMG / "deck/swarovski_optik_logo.png", height=1.35, radius=0.04)
        so.move_to(shop.get_corner(UL) + np.array([0.3, -0.55, 0]))
        note = copyright_note().next_to(shop, DOWN, buff=0.1).align_to(shop, LEFT)
        self.play(FadeIn(shop, scale=0.96), FadeIn(so), FadeIn(note))
        self.next_slide()
        # the BirdID clip, bottom right over the page; it stays on its last frame
        clip = IMG / "deck/ax_visio_birdid.mp4"
        frames = video_frames(clip, 1280)
        vid = ImageMobject(str(frames[0])).scale_to_fit_width(6.6)
        vid.move_to([FRAME_W / 2 - 0.3, -FRAME_H / 2 + 0.3, 0], aligned_edge=DR)
        self.play(FadeIn(vid, scale=0.9), run_time=0.4)
        self.play(PlayVideo(vid, frames))
        self.page()

        # 3 — MammalID: the AX Visio (the mode is pointed at by hand), then the model
        # ensemble behind the mode in one go (over SpeciesNet's blog post), then YOLOv5 as
        # the detector in the device
        self.play(FadeIn(title("MammalID")))
        ax = photo(IMG / "deck/ax_visio_top.jpg", height=3.6, radius=0.04).move_to([-4.6, -0.2, 0])
        note = copyright_note().next_to(ax, DOWN, buff=0.1).align_to(ax, LEFT)
        self.play(FadeIn(ax), FadeIn(note))
        self.next_slide()
        # the blog post on top, detection -> classification -> result in one line below it
        blog = photo(IMG / "usecase/speciesnet_blog.png", width=5.8, aspect=5.8 / 3.0, top=True, radius=0.03).move_to([3.0, 1.85, 0])
        row_y = -1.55
        frame = photo(IMG / "usecase/ocelot.jpg", height=2.0).move_to([-1.0, row_y, 0])
        W, H = frame.width, frame.height
        x0, y0 = frame.get_left()[0], frame.get_top()[1]
        bx = D.OCELOT_BBOX
        bbox = Rectangle(width=(bx[2] - bx[0]) / 800 * W, height=(bx[3] - bx[1]) / 600 * H).set_stroke(GREEN, 5)
        bbox.move_to([x0 + (bx[0] + bx[2]) / 2 / 800 * W, y0 - (bx[1] + bx[3]) / 2 / 600 * H, 0])
        md = pill("MegaDetector · 94 %", 20).next_to(frame, DOWN, buff=0.2)
        crop = photo(IMG / "usecase/ocelot_crop.jpg", height=1.6).move_to([1.75, row_y, 0])
        sn = pill("SpeciesNet", 20).next_to(crop, DOWN, buff=0.2)
        out = pill("Ocelot · 98.8 %", 24, fill=GREEN, bold=True).move_to([0, row_y, 0]).align_to([6.75, 0, 0], RIGHT)
        a1 = arrow(bbox.get_right(), crop.get_left())
        a2 = arrow(crop.get_right(), out.get_left())
        self.play(FadeIn(blog, shift=UP * 0.2), run_time=0.6)
        self.play(
            FadeIn(frame), Create(bbox), FadeIn(md), Create(a1), FadeIn(crop, shift=RIGHT * 0.3), FadeIn(sn),
            Create(a2), FadeIn(out, scale=1.1), run_time=1.2,
        )
        self.next_slide()
        # in the AX Visio a YOLOv5 model is the detector: MegaDetector greys out into the background
        yolo = pill("YOLOv5 · in the AX Visio", 20, fill=GREEN, bold=True).next_to(md, DOWN, buff=0.25)
        link = curved_arrow(ax.get_bottom() + RIGHT * 1.2 + DOWN * 0.1, yolo.get_left() + LEFT * 0.1, angle=0.6, color=GREEN)
        self.play(md.animate.set_opacity(0.35))
        self.play(Create(link), FadeIn(yolo, shift=UP * 0.15))
        self.page()

        # 4 — Three challenges (what each one means is said, not written)
        self.play(FadeIn(title("Three challenges")))
        items = [
            (icon(IMG / "deck/icon_image.png", 0.95), "Data"),
            (icon(IMG / "deck/icon_chip.png", 1.1), "Hardware"),
            (stopwatch(0.45), "Speed"),
        ]
        cols = []
        for inner, head_s in items:
            g = disc(inner, 1.1)
            h_ = txt(head_s, 32, bold=True).next_to(g, DOWN, buff=0.35)
            cols.append(Group(g, h_))
        Group(*cols).arrange(RIGHT, buff=1.6, aligned_edge=UP).move_to(DOWN * 0.3)
        for g in cols:
            self.play(FadeIn(g, shift=UP * 0.2), run_time=0.6)
        self.next_slide()


# =================================================================================
# Act 2 — Data: which classes, which sources, which images
# =================================================================================
class Data3(Talk3):
    def build(self):
        # 5 — SpeciesNet's taxonomy
        tree = photo(IMG / "data/taxonomy_hierarchy.png", height=5.2).move_to([-2.6, -0.4, 0])
        n = big_number("~2,500", "species SpeciesNet knows").move_to([3.7, 0.3, 0])
        self.play(FadeIn(title("SpeciesNet")), FadeIn(tree))
        self.play(FadeIn(n, shift=UP * 0.2))
        self.page()

        # 6 — Hedgehogs: how fine to distinguish
        hedgehogs = sorted((IMG / "data/hedgehogs").glob("*.*"))
        grid = collage(hedgehogs, cols=6, cell_w=2.05).move_to(DOWN * 0.35)
        lbl = pill("17 species  →  1 class", 34, fill=GREEN, bold=True).move_to(grid).shift(DOWN * 0.3)
        self.play(LaggedStart(*[FadeIn(t, scale=0.9) for t in grid], lag_ratio=0.06), run_time=1.6)
        self.next_slide()
        self.play(FadeIn(lbl, scale=1.15))
        self.page()

        # 7 — 225 custom classes: every name, grouped by order (shown, not meant to be read)
        self.play(FadeIn(title("225 Custom Classes")))
        lines, heads_ = [], set()
        for order, names in D.CLASS_TABLE.items():
            heads_.add(len(lines))
            lines.append(txt(order, 14, GREEN, bold=True))
            lines += [txt(n_, 12, MIST) for n_ in names]
        box_w, box_h = FRAME_W - 1.2, 5.6

        def columns_of(n_cols):
            per = -(-len(lines) // n_cols)
            groups, k = [], 0
            while k < len(lines):
                end = min(k + per, len(lines))
                if end < len(lines) and end - 1 in heads_:  # never leave an order name alone at a column's foot
                    end -= 1
                groups.append(lines[k:end])
                k = end
            return VGroup(*[
                VGroup(*g).copy().arrange(DOWN, aligned_edge=LEFT, buff=0.07) for g in groups
            ]).arrange(RIGHT, aligned_edge=UP, buff=0.3)

        # the column count that lets the names come out largest in the free area
        table = max((columns_of(n_) for n_ in range(5, 13)), key=lambda t: min(box_w / t.width, box_h / t.height))
        table.scale(min(box_w / table.width, box_h / table.height)).move_to(DOWN * 0.45)
        self.play(LaggedStart(*[FadeIn(col, shift=UP * 0.15) for col in table], lag_ratio=0.15), run_time=2.2)
        self.page()

        # 8 — Images per class: GBIF, then four more sources, each in its own colour
        self.play(FadeIn(title("Images per class")))
        counts = D.gbif_counts()
        chart_w, chart_h = 12.4, 4.6
        base_y, left = -3.1, -6.0
        top = np.log10(50000)

        def y_of(v):
            return chart_h * np.log10(v + 1) / top

        grid = VGroup()
        for v in (1, 10, 100, 1000, 10000):
            yy = base_y + y_of(v)
            ln = Line([left, yy, 0], [left + chart_w, yy, 0]).set_stroke(MIST, 1, 0.25)
            grid.add(ln, axis_label(f"{v:,}").next_to(ln, LEFT, buff=0.12))
        bw = chart_w / len(counts)
        bars = VGroup()
        for i, v in enumerate(counts):
            b = Rectangle(width=bw * 0.7, height=max(y_of(v), 0.025)).set_fill(SKY, 1).set_stroke(width=0)
            b.move_to([left + (i + 0.5) * bw, base_y, 0], aligned_edge=DOWN)
            bars.add(b)
        line100 = DashedLine([left, base_y + y_of(100), 0], [left + chart_w, base_y + y_of(100), 0]).set_stroke(GREEN, 3)
        self.play(FadeIn(grid))
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.01), run_time=2.0)
        self.play(Create(line100))
        p1 = pill("GBIF: 114 classes  < 100 images", 24).move_to([-1.6, 1.5, 0])
        p2 = pill("30 classes  without a single image", 24).next_to(p1, DOWN, buff=0.2).align_to(p1, LEFT)
        self.play(FadeIn(p1, shift=UP * 0.2))
        self.play(FadeIn(p2, shift=UP * 0.2))
        self.next_slide()
        # Usable images per source, stacked on the log axis (GBIF at the bottom), first in
        # the order of usable GBIF images, then every class column moves to its rank by the total.
        stack = D.source_stack_gbif_order()
        layers = [VGroup() for _ in D.SOURCES]
        for i, row in enumerate(stack):
            cum = 0
            for k, v in enumerate(row):
                lo, hi = y_of(cum), y_of(cum + v)
                cum += v
                r = Rectangle(width=bw * 0.7, height=max(hi - lo, 0.001)).set_stroke(width=0)
                r.set_fill(SOURCE_COLORS[k], 1 if v else 0)
                layers[k].add(r.move_to([left + (i + 0.5) * bw, base_y + lo, 0], aligned_edge=DOWN))
        totals = [sum(c) for c in zip(*stack)]
        legend = VGroup(*[
            pill(f"{name} · {tot:,}", 18, fill=SOURCE_COLORS[k]) for k, ((name, _), tot) in enumerate(zip(D.SOURCES, totals))
        ]).arrange(RIGHT, buff=0.15).move_to([0, 2.05, 0]).align_to([left, 0, 0], LEFT)
        lic = pill("commercially usable licences only", 20).move_to([left, 1.2, 0], aligned_edge=LEFT)
        p3 = pill("29 classes  still < 100", 24, fill=GREEN, bold=True).next_to(lic, DOWN, buff=0.2).align_to(lic, LEFT)
        self.play(FadeOut(p1), FadeOut(p2), FadeIn(legend[0]), FadeIn(lic))
        self.play(Transform(bars, layers[0]), run_time=1.2)
        for k in range(1, len(D.SOURCES)):
            self.play(FadeIn(legend[k], shift=UP * 0.1), LaggedStart(*[GrowFromEdge(r, DOWN) for r in layers[k]], lag_ratio=0.004), run_time=1.0)
        self.next_slide()
        rank = {i: r for r, i in enumerate(sorted(range(len(stack)), key=lambda i: (sum(stack[i]), i)))}
        columns_ = [VGroup(bars[i], *[layers[k][i] for k in range(1, len(D.SOURCES))]) for i in range(len(stack))]
        self.play(*[c.animate.set_x(left + (rank[i] + 0.5) * bw) for i, c in enumerate(columns_)], run_time=2.2)
        self.play(FadeIn(p3, scale=1.1))
        self.snap("images_per_class")
        self.page()

        # 9 — Data quality: not every "animal" is an animal (photos shown uncropped)
        rejects = [photo(p, height=2.25, radius=0.04) for p in sorted((IMG / "data").glob("reject_*"))]
        rows_ = Group(Group(*rejects[:4]).arrange(RIGHT, buff=0.15), Group(*rejects[4:]).arrange(RIGHT, buff=0.15))
        for r in rows_:
            if r.width > FRAME_W - 1.0:
                r.scale_to_fit_width(FRAME_W - 1.0)
        rej = rows_.arrange(DOWN, buff=0.15).move_to(DOWN * 0.3)
        self.play(FadeIn(title("Data Quality")))
        self.play(LaggedStart(*[FadeIn(t, scale=0.9) for r in rej for t in r], lag_ratio=0.12), run_time=1.6)
        self.next_slide()
        stages = ["Metadata", "Heuristics", "MegaDetector", "VLM", "SpeciesNet", "Manual review"]
        chips = VGroup(*[pill(s, 21) for s in stages]).arrange(RIGHT, buff=0.45).move_to(DOWN * 3.0)
        arrs = VGroup(*[arrow(a.get_right(), b.get_left(), buff=0.08) for a, b in zip(chips, chips[1:])])
        bbox_l = txt("= bbox label", 16, GREEN).next_to(chips[2], DOWN, buff=0.1)  # MegaDetector's boxes are the labels
        self.play(rej.animate.scale(0.82).shift(UP * 0.45))
        self.play(LaggedStart(*[FadeIn(c, shift=RIGHT * 0.2) for c in chips], lag_ratio=0.2), Create(arrs), run_time=1.5)
        self.play(FadeIn(bbox_l, shift=UP * 0.1))
        self.next_slide()


# =================================================================================
# Act 3 — Generated images: which generator, which prompt
# =================================================================================
class Generative3(Talk3):
    def build(self):
        # 10 — Local generators vs Gemini (binturong)
        B = IMG / "generators/binturong"
        cols = [
            ("Real", [B / f"binturong_real_{i}.jpg" for i in (1, 2, 3)], SKY),
            ("RealVisXL", [B / f"binturong_realvisxl-lightning_000{i}.png" for i in (1, 3, 5)], WHITE),
            ("SD 3.5 Medium", [B / f"binturong_sd35m_000{i}.png" for i in (1, 2, 3)], WHITE),
            ("FLUX.1 schnell", [B / f"binturong_flux-schnell_{i}.png" for i in ("0001", "0004", "0014")], WHITE),
            ("Gemini", [B / f"binturong_gemini_000{i}.png" for i in (1, 4, 7)], GREEN),
        ]
        columns = Group()
        for name, paths, color in cols:
            col = Group(*[photo(p, width=1.95, aspect=4 / 3) for p in paths]).arrange(DOWN, buff=0.1)
            columns.add(Group(col, pill(name, 20, fill=color).next_to(col, DOWN, buff=0.2)))
        columns.arrange(RIGHT, buff=0.4).move_to(DOWN * 0.6)
        local = txt("local models", 22, MIST).next_to(Group(*columns[1:4]), UP, buff=0.12)
        self.play(FadeIn(title("Synthetic images? – Binturong")), FadeIn(columns[0]))
        self.next_slide()
        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.2) for c in columns[1:4]], lag_ratio=0.25), FadeIn(local), run_time=1.5)
        self.next_slide()
        ring = SurroundingRectangle(columns[4][0], buff=0.08, corner_radius=0.15).set_stroke(GREEN, 6)
        self.play(FadeIn(columns[4], shift=LEFT * 0.3))
        self.play(Create(ring))
        self.snap("local_vs_gemini")
        self.page()

        # 11 — Prompt: the full prompt first, then ever shorter ones
        self.play(FadeIn(title("Prompt")))
        regimes = [
            ("full", "prompt_full_gemini.jpg", "Gemini", "full · 4,642 words"),
            ("maxlen512", "prompt_maxlen512_flux2-klein.jpg", "FLUX.2 klein", "≤ 512 tokens · 380 words"),
            ("maxlen256", "prompt_maxlen256_sd35m.jpg", "SD 3.5 Medium", "≤ 256 tokens · 149 words"),
            ("compressed", "lion_realvisxl.jpg", "RealVisXL", "≤ 75 tokens · 20 words"),
        ]
        cw = 3.0
        xs = [-4.95 + i * 3.3 for i in range(4)]
        cols_ = []
        for x, (name, img, gen, label) in zip(xs, regimes):
            im = labelled(photo(IMG / "generators" / img, width=cw, aspect=4 / 3), gen, DL, 16).move_to([x, 1.15, 0])
            tag = pill(label, 17, fill=GREEN if name == "full" else WHITE, bold=name == "full").move_to([x, -0.25, 0])
            pc = prompt_text_card(name, cw, 3.3)
            pc.move_to([x, -0.6, 0], aligned_edge=UP)
            cols_.append((im, tag, pc))
        im, tag, pc = cols_[0]
        self.play(FadeIn(im), FadeIn(tag), GrowFromEdge(pc, UP), run_time=0.9)
        self.next_slide()
        for im, tag, pc in cols_[1:]:
            self.play(FadeIn(im), FadeIn(tag), GrowFromEdge(pc, UP), run_time=0.8)
        self.page()

        # 12 — The production set (no caption: the pictures speak)
        sheet = photo(IMG / "generators/grevys_synthetic_wall.png", width=FRAME_W - 0.6, radius=0).move_to(DOWN * 0.25)
        self.play(FadeIn(sheet, scale=1.05), run_time=1.2)
        self.page()

        # 13 — Three zebras: details decide; the generator's name comes in on the click
        zs = Group()
        for path, name in (("zebra_grevys_gemini.png", "Grévy's zebra"), ("zebra_mountain_gemini.png", "Mountain zebra"), ("zebra_plains_gemini.png", "Plains zebra")):
            ph = photo(IMG / "generators" / path, width=4.1, aspect=4 / 3)
            zs.add(Group(ph, pill(name, 22).next_to(ph, DOWN, buff=0.2)))
        zs.arrange(RIGHT, buff=0.3).move_to(DOWN * 0.2)
        self.play(LaggedStart(*[FadeIn(z, shift=UP * 0.2) for z in zs], lag_ratio=0.3))
        self.next_slide()
        self.play(FadeIn(title("Gemini 3.1 Flash Image"), shift=UP * 0.15))
        self.page()

        # 14 — The twelve species of the generator comparison (one gpt-image-2 low picture each)
        head = title("Training Images Comparison")
        self.play(FadeIn(head))
        T = IMG / "generators/twelve"
        animals = Group()
        for folder, name in D.COMPARISON_CLASSES:
            ph = photo(T / f"{folder}.png", width=2.15, aspect=4 / 3)
            animals.add(Group(ph, pill(name, 15).move_to(ph.get_bottom() + UP * 0.05)))  # centred, slightly on the picture
        animals.arrange_in_grid(rows=2, cols=6, buff=(0.12, 0.35)).move_to(DOWN * 0.5)
        each = pill("100 images each", 26, fill=GREEN, bold=True).next_to(animals, DOWN, buff=0.4)
        self.play(LaggedStart(*[FadeIn(a, scale=0.95) for a in animals], lag_ratio=0.08), run_time=1.4)
        self.play(FadeIn(each, scale=1.1))
        self.next_slide()
        self.play(FadeOut(animals), FadeOut(each))

        # 15 — Training images comparison: detector accuracy per generator, cost per image and
        # prompt length, the same lion from six generators; then the two compressed-prompt
        # cells drop into the ranking.
        vmax, L, bars_x = 0.15, 2.9, -3.0
        x_val, x_cost, x_prompt = bars_x + L + 0.75, bars_x + L + 1.3, bars_x + L + 2.35

        def slot_y(k):  # twelve slots: ten full-prompt cells now, two compressed ones later
            return 1.75 - k * 0.42

        def gen_row(name, mean, std, kind, y):
            local = kind == "local"
            fg = GREEN if kind == "compressed" else (MIST if local else WHITE)
            bold = kind == "compressed"
            color = {"api": SKY, "pick": SKY, "local": MIST, "compressed": GREEN}[kind]
            lab = txt(name, 16, fg, bold=bold).next_to([bars_x - 0.2, y, 0], LEFT, buff=0)
            bar = hbar(mean, vmax, L, h=0.26, color=color, opacity=0.7 if local else 1).move_to([bars_x, y, 0], aligned_edge=LEFT)
            err = Line([bars_x + L * (mean - std) / vmax, y, 0], [bars_x + L * (mean + std) / vmax, y, 0]).set_stroke(WHITE, 2, 0.6)
            val = txt(D.pct(mean), 16, fg, bold=bold).move_to([x_val, y, 0])
            cost_s = D.GENERATOR_COST_COMPRESSED.get(name) or D.GENERATOR_COST[name]
            cost = txt(cost_s, 16, fg, bold=bold).move_to([x_cost, y, 0], aligned_edge=LEFT)
            prm = txt(D.GENERATOR_PROMPT[name], 16, fg, bold=bold).move_to([x_prompt, y, 0], aligned_edge=LEFT)
            return VGroup(lab, bar, err, val, cost, prm)

        rows = VGroup(*[gen_row(name, mean, std, kind, slot_y(k)) for k, (name, mean, std, kind) in enumerate(D.GENERATORS)])
        hy = slot_y(0) + 0.5
        heads = VGroup(
            txt("mAP", 15, MIST).move_to([x_val, hy, 0]),
            txt("per image", 15, MIST).move_to([x_cost, hy, 0], aligned_edge=LEFT),
            txt("prompt\ntokens", 15, MIST, line_spacing=0.8).move_to([x_prompt, hy + 0.08, 0], aligned_edge=LEFT),
        )
        foot = txt("per image: API = batch list price · local = seconds on one NVIDIA A40", 14, MIST).to_corner(DL, buff=0.3)
        legend = VGroup(pill("API", 16, fill=SKY), pill("local", 16, fill=MIST)).arrange(RIGHT, buff=0.15)
        legend.move_to([x_prompt + 0.9, foot.get_y(), 0], aligned_edge=RIGHT)
        # The same lion prompt, one picture per full-prompt cell, in ranking order.
        G = IMG / "generators"
        lion_file = {
            "gpt-image-2 low": "lion_gpt-image-2-low.jpg", "gpt-image-2 medium": "lion_gpt-image-2-medium.jpg",
            "Gemini Flash": "lion_gemini.jpg", "Gemini Flash-Lite": "lion_gemini-flash-lite.jpg",
            "SD 3.5 Large": "lion_sd35-large.jpg", "SD 3.5 Medium": "lion_sd35m.jpg",
            "SD 3.5 Large Turbo": "lion_sd35-large-turbo.jpg", "FLUX.2 klein": "lion_flux2-klein.jpg",
            "HiDream-I1": "lion_hidream-i1.jpg", "RealVisXL Lightning": "lion_realvisxl.jpg",
        }
        lions = Group()
        for name, *_x in D.GENERATORS:
            ph = photo(G / lion_file[name], width=1.6, aspect=4 / 3)
            lions.add(Group(ph, pill(name, 11).move_to(ph.get_corner(DL), aligned_edge=DL).shift(UR * 0.05)))
        lions.arrange_in_grid(rows=5, cols=2, buff=(0.08, 0.08)).move_to([5.25, -0.55, 0])
        self.play(FadeIn(legend), LaggedStart(*[FadeIn(r[0]) for r in rows], lag_ratio=0.05))
        self.play(LaggedStart(*[GrowFromEdge(r[1], LEFT) for r in rows], lag_ratio=0.08), run_time=1.6)
        self.play(*[FadeIn(r[2]) for r in rows], *[FadeIn(r[3]) for r in rows], FadeIn(heads[0]))
        self.play(
            LaggedStart(*[FadeIn(VGroup(r[4], r[5]), shift=LEFT * 0.1) for r in rows], lag_ratio=0.05),
            FadeIn(heads[1]), FadeIn(heads[2]), FadeIn(foot),
        )
        self.play(LaggedStart(*[FadeIn(t, shift=LEFT * 0.15) for t in lions], lag_ratio=0.12), run_time=1.4)
        self.next_slide()
        # The compressed prompt (≤ 75 tokens): rows below each insert point slide down.
        order = [name for name, *_ in D.GENERATORS]
        for name, *_rest, after in D.GENERATORS_COMPRESSED:
            order.insert(order.index(after) + 1, name)
        moves = [r.animate.set_y(slot_y(order.index(name))) for r, (name, *_x) in zip(rows, D.GENERATORS)]
        self.play(*moves, run_time=1.0)
        new = VGroup(*[gen_row(name, mean, std, "compressed", slot_y(order.index(name))) for name, mean, std, _a in D.GENERATORS_COMPRESSED])
        # their full-prompt originals light up too, so the audience sees which two were compressed
        idx = {name: k for k, (name, *_x) in enumerate(D.GENERATORS)}
        lit = [rows[idx[name.split(" · ")[0]]] for name, *_x in D.GENERATORS_COMPRESSED]
        self.play(
            *[FadeIn(r[0], shift=RIGHT * 0.2) for r in new], *[GrowFromEdge(r[1], LEFT) for r in new],
            *[Transform(r[k], r[k].copy().set_color(GREEN)) for r in lit for k in (0, 3)],
            *[Create(SurroundingRectangle(r[0], buff=0.06, corner_radius=0.08).set_stroke(GREEN, 2)) for r in lit],
            run_time=1.0,
        )
        self.play(*[FadeIn(VGroup(*r[2:])) for r in new])
        self.snap("ranking")
        self.next_slide()


# =================================================================================
# Act 4 — Supplementing the classes: bands A–D and what they show
# =================================================================================
class Bands3(Talk3):
    def build(self):
        # 16 — Final dataset, built up: real images → bands → synthetic train → synthetic test
        head = title("Final Dataset")
        self.play(FadeIn(head))
        with open(REPO / "reports/class_split_counts.csv") as f:
            rows = list(csv.DictReader(f))
        order = {"A": 0, "B": 1, "C": 2, "D": 3}
        rows.sort(key=lambda r: (order[r["band"]], int(r["effective_pool"])))
        # linear axis cut at 800 images so bands A–C read large; band D columns (up to
        # 1,500 real train + 500 real test + 50 synthetic test = 2,050) are clipped at the top
        cap, chart_h, left, width, base = 800, 4.0, -5.6, 12.1, -2.9
        bw = width / len(rows)

        def stack(i, parts):
            """One class column: a rectangle per (count, colour), stacked bottom to top, clipped at the cap."""
            y, out = base, []
            for v, color in parts:
                h = max(0.0, min(v / cap * chart_h, base + chart_h - y))
                r = Rectangle(width=bw * 0.78, height=max(h, 0.001)).set_stroke(width=0).set_fill(color, 1 if h > 0 else 0)
                out.append(r.move_to([left + (i + 0.5) * bw, y, 0], aligned_edge=DOWN))
                y += h
            return out

        cols, after = [], []
        for i, r in enumerate(rows):
            tr, ts, te = int(r["train_real"]), int(r["train_synth"]), int(r["test_real"])
            cols.append(VGroup(*stack(i, [(0, GREEN), (tr, MIST), (te, MIST), (0, SYNTH_TEST)])))
            after.append(stack(i, [(ts, GREEN), (tr, MIST), (te, REAL_TEST), (50, SYNTH_TEST)]))  # 50 synthetic test images per class
        grid, img_labels = VGroup(), VGroup()
        for v in (200, 400, 600, 800):
            yy = base + v / cap * chart_h
            ln = Line([left, yy, 0], [left + width, yy, 0]).set_stroke(MIST, 1, 0.2)
            grid.add(ln)
            img_labels.add(axis_label(f"{v:,}").next_to(ln, LEFT, buff=0.1))
        seps, labs, spans, i0 = VGroup(), VGroup(), {}, 0
        for b in "ABCD":
            n_b = sum(1 for r in rows if r["band"] == b)
            spans[b] = (left + i0 * bw, left + (i0 + n_b) * bw)
            labs.add(txt(b, 26, bold=True).move_to([sum(spans[b]) / 2, base - 0.35, 0]))
            if i0:
                seps.add(DashedLine([left + i0 * bw, base, 0], [left + i0 * bw, base + chart_h, 0]).set_stroke(WHITE, 2, 0.6))
            i0 += n_b
        def legend():
            leg = {
                "real": pill("real", 17, fill=MIST),
                "real train": pill("real train", 17, fill=MIST),
                "synthetic train": pill("synthetic train", 17, fill=GREEN),
                "real test": pill("real test", 17, fill=REAL_TEST),
                "synthetic test": pill("synthetic test", 17, fill=SYNTH_TEST),
            }
            VGroup(*[leg[k] for k in ("real train", "synthetic train", "real test", "synthetic test")]).arrange(RIGHT, buff=0.18).to_corner(UR, buff=0.4).shift(DOWN * 1.25)
            leg["real"].move_to(leg["real train"], aligned_edge=LEFT)
            return leg

        leg = legend()
        # step 1: the real images per class
        self.play(FadeIn(grid), FadeIn(img_labels), FadeIn(leg["real"]))
        self.play(LaggedStart(*[GrowFromEdge(VGroup(c[1], c[2]), DOWN) for c in cols], lag_ratio=0.006), run_time=2.2)
        self.next_slide()
        # step 2: the bands
        self.play(*[Create(s) for s in seps], FadeIn(labs))
        self.next_slide()
        # step 3: synthetic training images push the real test images up; test gets its colour
        self.add(*[c[0] for c in cols])
        self.play(
            *[Transform(c[k], after[i][k]) for i, c in enumerate(cols) for k in range(3)],
            Transform(leg["real"], leg["real train"]), FadeIn(leg["synthetic train"]), FadeIn(leg["real test"]),
            run_time=2.0,
        )
        self.next_slide()
        # step 4: 50 synthetic test images per class
        tests = VGroup(*[a[3] for a in after])
        self.play(FadeIn(leg["synthetic test"]), LaggedStart(*[GrowFromEdge(t, DOWN) for t in tests], lag_ratio=0.004), run_time=1.6)
        self.next_slide()

        # step 5: only the real test images count: they stay lit, and their legend pill
        # becomes the label of the results that follow
        bars_all = VGroup(*cols, tests)
        bars_all.save_state()
        not_real_test = VGroup(*[c[k] for c in cols for k in (0, 1)], tests)
        note_pill = pill("real test photos", 17, fill=REAL_TEST)
        note = VGroup(txt("YOLO26n · mAP on", 18, MIST), note_pill).arrange(RIGHT, buff=0.15).to_corner(UR, buff=0.4).shift(DOWN * 1.25)
        self.play(
            not_real_test.animate.set_opacity(0.15),
            *[FadeOut(leg[k]) for k in ("real", "synthetic train", "synthetic test")],
            leg["real test"].animate.scale(1.15),
        )
        self.play(Transform(leg["real test"], note_pill), FadeIn(note[0], shift=LEFT * 0.2))
        self.next_slide()

        # 17 — Results on top of the same chart: the dataset greys out, the bands stay
        map_labels = VGroup(*[axis_label(f"{v / cap * 100:.0f} %").move_to(l, aligned_edge=RIGHT) for v, l in zip((200, 400, 600, 800), img_labels)])
        count_labels = img_labels.copy()
        self.play(
            Transform(head, title("Recognition Results")),
            bars_all.animate.set_fill(MIST, 0.16),
            Transform(img_labels, map_labels),
        )
        # band mAP = mean of the per-class AP shown as dots on the next click
        per_class = D.per_class_real_ap()
        band_map = {b: float(np.mean([v for _, v in items])) for b, items in per_class.items()}
        assert all(abs(band_map[b] - D.BAND_REAL_TEST[b]) < 5e-4 for b in "ABCD")
        res, vals = VGroup(), VGroup()
        for b, v in band_map.items():
            x_lo, x_hi = spans[b]
            r = Rectangle(width=x_hi - x_lo - 0.16, height=v * chart_h).set_fill(REAL_TEST, 0.38).set_stroke(REAL_TEST, 3)
            res.add(r.move_to([(x_lo + x_hi) / 2, base, 0], aligned_edge=DOWN))
            vals.add(txt(D.pct(v), 22, bold=True).next_to(r, UP, buff=0.12))
        self.play(LaggedStart(*[GrowFromEdge(r, DOWN) for r in res], lag_ratio=0.25), run_time=1.8)
        self.play(FadeIn(vals))
        self.snap("band_results")
        self.next_slide()
        # Per-class AP as dots, each at its own column (the bar top is the band mean)
        pc = {name: v for items in per_class.values() for name, v in items}
        dots, special = VGroup(), {}
        for i, r in enumerate(rows):
            v = pc.get(r["class"])
            if v is None:
                continue
            d = Dot(radius=0.05, color=WHITE, fill_opacity=0.85).move_to([left + (i + 0.5) * bw, base + v * chart_h, 0])
            if r["class"] in ("striped hyaena", "mangabeys genus"):
                special[r["class"]] = d
            dots.add(d)
        self.play(res.animate.set_fill(REAL_TEST, 0.18), FadeOut(vals))
        self.play(LaggedStart(*[GrowFromCenter(d) for d in dots], lag_ratio=0.004), run_time=1.6)
        self.next_slide()
        def example(path, label):  # label centred on the picture's bottom edge
            ph = photo(path, height=1.05, aspect=4 / 3)
            return Group(ph, pill(label, 12).move_to(ph.get_bottom()))

        hy = example(IMG / "bands/synthetic_striped_hyaena.png", f"hyaena · AP {D.pct(pc['striped hyaena'], 0)}")
        mg = example(IMG / "bands/synthetic_mangabeys.png", f"mangabeys · AP {D.pct(pc['mangabeys genus'], 0)}")
        # inside the chart, in the empty upper half of band A
        pair = Group(mg, hy).arrange(RIGHT, buff=0.12)  # same left-to-right order as their dots
        pair.move_to([spans["A"][0], base + chart_h + 0.06, 0], aligned_edge=DL)  # above the chart, over band A
        foot = txt("both trained on\nsynthetic images only", 15, MIST).next_to(pair, RIGHT, buff=0.25)
        hl = [special[k].copy().set_color(GREEN).scale(2.0) for k in ("striped hyaena", "mangabeys genus")]
        l1 = Line(hl[0].get_center(), hy.get_bottom(), buff=0.1).set_stroke(GREEN, 2)
        l2 = Line(hl[1].get_center(), mg.get_bottom(), buff=0.1).set_stroke(GREEN, 2)
        self.play(*[FadeIn(h, scale=1.5) for h in hl])
        self.play(FadeIn(pair), FadeIn(foot), Create(l1), Create(l2))
        self.next_slide()

        # 18 — Test data. First the dataset chart once more with the synthetic test images
        # lit; then the two test legend pills move onto the zebra rows as their labels.
        morphed = leg["real test"]  # the "real test photos" pill of the results note
        leg = legend()
        self.play(
            FadeOut(res), FadeOut(dots), *[FadeOut(h) for h in hl], FadeOut(pair), FadeOut(foot), FadeOut(l1), FadeOut(l2),
            FadeOut(note[0]), FadeOut(morphed),
            Restore(bars_all), Transform(img_labels, count_labels), Transform(head, title("Test Data")),
            *[FadeIn(leg[k]) for k in ("real train", "synthetic train", "real test", "synthetic test")],
        )
        not_synth_test = VGroup(*[c[k] for c in cols for k in (0, 1, 2)])
        self.play(not_synth_test.animate.set_opacity(0.15), leg["synthetic test"].animate.scale(1.1, about_edge=RIGHT))
        self.wait(0.6)
        G = IMG / "generators"
        syn = Group(*[photo(G / f, height=2.05, aspect=4 / 3) for f in ("zebra_grevys_gemini.png", "zebra_mountain_gemini.png", "zebra_plains_gemini.png")]).arrange(RIGHT, buff=0.15)
        real = Group(*[photo(IMG / "bands" / f, height=2.05, aspect=4 / 3) for f in ("real_test_grevys_zebra.jpg", "real_test_mountain_zebra.jpg", "real_test_plains_zebra.jpg")]).arrange(RIGHT, buff=0.15)
        sl = pill("synthetic test images", 20, fill=SYNTH_TEST)
        sl.move_to(syn.get_bottom() + UP * 0.08).align_to(syn, LEFT).shift(RIGHT * 0.12)  # slightly on the picture
        rl = pill("real test images", 20, fill=REAL_TEST)
        rl.move_to(real.get_bottom() + UP * 0.08).align_to(real, LEFT).shift(RIGHT * 0.12)
        block = Group(Group(syn, sl), Group(real, rl)).arrange(DOWN, buff=0.35).move_to(DOWN * 0.4)
        chart = VGroup(grid, img_labels, seps, labs, bars_all)
        leg["synthetic test"].set_z_index(5)
        leg["real test"].set_z_index(5)
        self.play(
            FadeOut(chart), FadeOut(leg["real train"]), FadeOut(leg["synthetic train"]),
            Transform(leg["synthetic test"], sl), Transform(leg["real test"], rl),
            run_time=1.2,
        )
        self.play(LaggedStart(*[FadeIn(t, shift=UP * 0.15) for t in (*syn, *real)], lag_ratio=0.12), run_time=1.2)
        # hand over to the block's own labels, drawn on top of their photos
        self.remove(leg["synthetic test"], leg["real test"])
        self.add(sl.set_z_index(5), rl.set_z_index(5))
        self.next_slide()
        self.play(block.animate.scale(0.75).to_edge(LEFT, buff=0.4).set_y(-0.3))
        L, base = 4.6, -2.6
        bars = VGroup()
        for k, b in enumerate("ABCD"):
            x = 0.9 + k * 1.65
            s = vbar(D.BAND_SYNTH_TEST[b], 1.0, L, w=0.6, color=SYNTH_TEST).move_to([x - 0.33, base, 0], aligned_edge=DOWN)
            r = vbar(D.BAND_REAL_TEST[b], 1.0, L, w=0.6, color=REAL_TEST).move_to([x + 0.33, base, 0], aligned_edge=DOWN)
            sv = txt(D.pct(D.BAND_SYNTH_TEST[b], 0), 16, MIST).next_to(s, UP, buff=0.08)
            rv = txt(D.pct(D.BAND_REAL_TEST[b], 0), 16).next_to(r, UP, buff=0.08)
            lab = VGroup(txt(b, 28, bold=True), band_glyph(b)).arrange(RIGHT, buff=0.14).move_to([x, base - 0.45, 0])
            bars.add(VGroup(s, r, sv, rv, lab))
        leg2 = VGroup(pill("synthetic test · mAP", 18, fill=SYNTH_TEST), pill("real test · mAP", 18, fill=REAL_TEST)).arrange(RIGHT, buff=0.2)
        leg2.next_to(bars, UP, buff=0.35).align_to(bars, LEFT).shift(UP * 0.4)
        self.play(FadeIn(leg2), *[FadeIn(g[4]) for g in bars])
        self.play(LaggedStart(*[a for g in bars for a in (GrowFromEdge(g[0], DOWN), GrowFromEdge(g[1], DOWN))], lag_ratio=0.1), run_time=1.6)
        self.play(*[FadeIn(g[2]) for g in bars], *[FadeIn(g[3]) for g in bars])
        self.next_slide()


# =================================================================================
# Act 5 — Models: accuracy, size and speed on embedded hardware
# =================================================================================
def band_glyph(band: str, size: float = 0.17) -> VGroup:
    """Training-set composition of a band as stacked squares, bottom to top like the dataset
    chart: green = 100 synthetic, white = 100 real images (A: 200 synthetic, B: 100 + 100,
    C: 200 real, D: many real, the top square fading out because the stack goes on)."""
    fills = {"A": [(GREEN, 1)] * 2, "B": [(GREEN, 1), (WHITE, 1)], "C": [(WHITE, 1)] * 2, "D": [(WHITE, 1)] * 2 + [(WHITE, 0.3)]}[band]
    return VGroup(*[Square(side_length=size).set_fill(f, o).set_stroke(width=0) for f, o in fills]).arrange(UP, buff=0.04)


def band_mix(species: str) -> str:
    """Band and training mix of a species, e.g. "B (100/100)" = 100 synthetic / 100 real train images."""
    n_syn, n_real = D.SPECIES_TRAIN[species]
    return f"{D.SPECIES_BAND[species]} ({n_syn:,}/{n_real:,})"


MODEL_ORDER = ["Ensemble, off the shelf", "Ensemble, fine-tuned", "YOLOv5s", "YOLO26n"]
# names on the slide (data.py keeps the names of the thesis tables)
MODEL_LABEL = {"Ensemble, off the shelf": "MD+SN, off the shelf", "Ensemble, fine-tuned": "MD+SN, SN fine-tuned"}


def accuracy_chart(order: list[str], extra: bool = False) -> dict:
    """Species-mAP bars with thin detection-mAP bars above, plus size/time pills per model.

    extra=True appends the distilled YOLO26n as a fifth row (the knowledge-distillation slide).
    """
    models = {name: (m, p, t) for name, m, p, t in D.MODELS}
    detect = dict(D.MODELS_DETECT)
    entries = [(name, *models[name], detect[name]) for name in order]
    if extra:
        name, m, p, t = D.MODEL_DISTILLED
        entries.append((name, m, p, t, D.MODEL_DISTILLED_DETECT))
    L, vmax, x0 = 5.0, 1.0, -2.4
    step = 0.95 if extra else 1.15
    rows, side, dots = VGroup(), VGroup(), VGroup()
    colors = {"Ensemble, off the shelf": BLUE, "Ensemble, fine-tuned": BLUE, "YOLOv5s": MIST}
    for k, (name, m, p, t, det) in enumerate(entries):
        y = 1.15 - k * step
        hi = name == "YOLO26n"
        color = GREEN if hi else MIST
        lab = txt(MODEL_LABEL.get(name, name), 24, bold=hi or name == D.MODEL_DISTILLED[0]).move_to([x0 - 0.3, y, 0], aligned_edge=RIGHT)
        bar = hbar(m, vmax, L, h=0.42, color=color, opacity=1 if hi else 0.75).move_to([x0, y - 0.13, 0], aligned_edge=LEFT)
        val = txt(D.pct(m), 24, bold=True).next_to(bar, RIGHT, buff=0.2)
        dbar = hbar(det, vmax, L, h=0.18, color=WHITE, opacity=0.4).move_to([x0, y + 0.25, 0], aligned_edge=LEFT)
        dval = txt(D.pct(det), 17, MIST).next_to(dbar, RIGHT, buff=0.2)
        rows.add(VGroup(lab, bar, val, dbar, dval))
        sec = f"{t:.0f} s" if t >= 10 else f"{t:.2f} s"
        side.add(pill(f"{p:g} M · {sec}", 20, fill=GREEN if hi else WHITE).move_to([5.6, y - 0.07, 0]))
        # the candidate circles, small: area still proportional to the parameter count
        c_ = colors.get(name, GREEN)
        dot = Circle(radius=0.32 * np.sqrt(p / 193)).set_fill(c_, 0.45 if c_ == BLUE else 0.9).set_stroke(MIST if c_ == BLUE else c_, 2)
        dots.add(dot.move_to([4.05, y - 0.07, 0]))
    # legend = a sample of each bar (thick species bar, thin translucent detection bar), so it reads like the chart
    leg = VGroup(
        VGroup(hbar(1, 1, 0.6, h=0.42, color=MIST, opacity=0.75), txt("species mAP", 18)).arrange(RIGHT, buff=0.15),
        VGroup(hbar(1, 1, 0.6, h=0.18, color=WHITE, opacity=0.4), txt("detection mAP (any animal)", 18, MIST)).arrange(RIGHT, buff=0.15),
    )
    leg.arrange(RIGHT, buff=0.5).move_to([LEFT_X, 2.05, 0], aligned_edge=LEFT)
    watch = stopwatch(0.32).next_to(side, UP, buff=0.25)
    return {"leg": leg, "rows": rows, "side": side, "watch": watch, "dots": dots}


class Models3(Talk3):
    def build(self):
        # 19 — Three candidates
        head = title("Three candidates")
        self.play(FadeIn(head))
        R = 2.45
        specs = [("MegaDetector + SpeciesNet", 193, "193 M", -2.6, BLUE), ("YOLOv5s", 7.63, "7.6 M", 1.7, MIST), ("YOLO26n", 2.71, "2.7 M", 3.7, GREEN)]
        circles = VGroup()
        for name, p, label, x, color in specs:
            r = R * np.sqrt(p / 193)
            c = Circle(radius=r).set_fill(color, 0.45 if color == BLUE else 0.9).set_stroke(MIST if color == BLUE else color, 2)
            c.move_to([x, -0.6, 0]).align_to([0, -3.0, 0], DOWN)
            n = txt(label, 30 if p > 100 else 24, bold=True)
            n.move_to(c) if p > 100 else n.next_to(c, UP, buff=0.15)
            lab = pill(name, 20).move_to([x, -3.4, 0])
            circles.add(VGroup(c, n, lab))
        self.play(LaggedStart(*[GrowFromCenter(c[0]) for c in circles], lag_ratio=0.4), run_time=1.8)
        self.play(*[FadeIn(c[1]) for c in circles], *[FadeIn(c[2]) for c in circles])
        self.next_slide()

        # 20 — Accuracy (species mAP), then detection mAP, size and time per frame. The
        # candidate circles shrink into a column next to their rows (the ensemble twice).
        chart = accuracy_chart(MODEL_ORDER)
        leg, rows, side, watch = chart["leg"], chart["rows"], chart["side"], chart["watch"]
        ens2 = circles[0][0].copy()
        self.add(ens2)
        sources = [circles[0][0], ens2, circles[1][0], circles[2][0]]
        self.play(
            Transform(head, title("Accuracy on real photos (mAP)")),
            *[FadeOut(c[1]) for c in circles], *[FadeOut(c[2]) for c in circles],
            *[Transform(src, dot) for src, dot in zip(sources, chart["dots"])],
            run_time=1.4,
        )
        self.play(FadeIn(leg[0]), LaggedStart(*[FadeIn(r[0]) for r in rows], lag_ratio=0.1))
        self.play(LaggedStart(*[GrowFromEdge(r[1], LEFT) for r in rows], lag_ratio=0.15), run_time=1.5)
        self.play(*[FadeIn(r[2]) for r in rows])
        self.next_slide()
        self.play(FadeIn(leg[1]), LaggedStart(*[GrowFromEdge(r[3], LEFT) for r in rows], lag_ratio=0.15), run_time=1.3)
        self.play(*[FadeIn(r[4]) for r in rows])
        self.next_slide()
        foot = txt("time per frame: Raspberry Pi 400 CPU (proxy for the QCS605), full precision", 16, MIST).to_corner(DL, buff=0.4)
        self.play(FadeIn(watch), LaggedStart(*[FadeIn(s_, shift=LEFT * 0.2) for s_ in side], lag_ratio=0.15), FadeIn(foot))
        self.next_slide()
        m = {name: v for name, v, _, _ in D.MODELS}
        gap = round((m["Ensemble, fine-tuned"] - m["YOLO26n"]) * 100)
        call = pill(f"83× faster · 71× smaller · −{gap} pp species mAP", 28, fill=GREEN, bold=True).move_to([1.0, -3.3, 0])
        self.play(FadeOut(foot), FadeIn(call, scale=1.1))
        self.snap("models")
        self.page()

        # 21 — Similar species: the zebras, then how often each is named as another; then the elephants
        self.play(FadeIn(title("Similar Species")))
        conf = {(t, p_): n_ / m_ for t, p_, n_, m_ in D.SPECIES_CONFUSION_REAL}
        zeb = Group()
        for name, f in (("Grévy's zebra", "real_test_grevys_zebra.jpg"), ("Mountain zebra", "real_test_mountain_zebra.jpg"), ("Plains zebra", "real_test_plains_zebra.jpg")):
            ph = photo(IMG / "bands" / f, height=2.3, aspect=4 / 3)
            zeb.add(Group(ph, pill(f"{name} · {band_mix(name)} · AP {D.pct(D.ZEBRA_AP[name])}", 15).next_to(ph, DOWN, buff=0.18)))
        zeb.arrange(RIGHT, buff=0.75).move_to(DOWN * 0.55)
        idx = {"Grévy's zebra": 0, "Mountain zebra": 1, "Plains zebra": 2}
        links = VGroup()
        for (src, dst), share in conf.items():
            if src not in idx:
                continue
            i, j = idx[src], idx[dst]
            p_i, p_j = zeb[i][0], zeb[j][0]
            if abs(i - j) == 2:  # Grévy's -> plains, over the top
                a_ = curved_arrow(p_i.get_top() + UP * 0.1, p_j.get_top() + UP * 0.1, angle=-0.55, color=GREEN)
                lab = pill(D.pct(share), 24, fill=GREEN, bold=True).next_to(a_, UP, buff=0.04)
            else:  # neighbours, left to right
                y_ = p_i.get_y() + 0.35
                a_ = arrow([p_i.get_right()[0], y_, 0], [p_j.get_left()[0], y_, 0], buff=0.08)
                lab = txt(D.pct(share), 18, bold=True).next_to(a_, UP, buff=0.06)
            links.add(VGroup(a_, lab))
        cfoot = txt("share of a species' found animals named as the other species · YOLO26n, real test photos", 14, MIST).to_corner(DL, buff=0.3)
        self.play(LaggedStart(*[FadeIn(z, shift=UP * 0.15) for z in zeb], lag_ratio=0.2), run_time=1.2)
        self.next_slide()
        def group_rate(group):  # the look-alike group's confusion rate, top right
            n_conf, n_match = D.WITHIN_GROUP_CONFUSION_REAL_GROUPS[group]
            return pill(f"{group}: {D.pct(n_conf / n_match)} confused", 22).move_to([FRAME_W / 2 - 0.4, title("x").get_y(), 0], aligned_edge=RIGHT)

        rate = group_rate("zebras")
        self.play(LaggedStart(*[Create(l_[0]) for l_ in links], lag_ratio=0.2), *[FadeIn(l_[1]) for l_ in links], FadeIn(cfoot), FadeIn(rate), run_time=1.4)
        self.next_slide()
        asian = labelled(photo(IMG / "models/synthetic_asian_elephant.jpg", height=3.0, aspect=4 / 3), f"Asian elephant · {band_mix('Asian elephant')}", DL, 20).move_to([-3.2, -0.9, 0])
        african = labelled(photo(IMG / "models/synthetic_african_elephant.jpg", height=3.0, aspect=4 / 3), f"African elephant · {band_mix('African elephant')}", DR, 20).move_to([3.6, -0.9, 0])
        fwd = curved_arrow(asian.get_top() + RIGHT * 0.6 + UP * 0.1, african.get_top() + LEFT * 0.6 + UP * 0.1, angle=-0.9, color=GREEN)
        rev = curved_arrow(african.get_bottom() + LEFT * 0.6 + DOWN * 0.1, asian.get_bottom() + RIGHT * 0.6 + DOWN * 0.1, angle=-0.6, color=MIST)
        fl = pill(D.pct(conf[("Asian elephant", "African elephant")]), 30, fill=GREEN, bold=True).next_to(fwd, UP, buff=0.05)
        rl = pill(D.pct(conf[("African elephant", "Asian elephant")]), 22).next_to(rev, DOWN, buff=0.02)
        syn = txt("pictures: synthetic test images", 14, MIST).to_corner(DR, buff=0.3)
        self.play(FadeOut(zeb), FadeOut(links), Transform(rate, group_rate("elephants")))
        self.play(FadeIn(asian), FadeIn(african), FadeIn(syn))
        self.play(Create(fwd), FadeIn(fl))
        self.play(Create(rev), FadeIn(rl))
        self.page()

        # 22 — Knowledge distillation: can the big model teach the small one? The accuracy chart again, plus the distilled student
        self.play(FadeIn(title("Knowledge Distillation")))
        chart = accuracy_chart(MODEL_ORDER, extra=True)
        leg, rows, side, watch = chart["leg"], chart["rows"], chart["side"], chart["watch"]
        dots = chart["dots"]
        self.play(FadeIn(leg), FadeIn(rows[:-1]), FadeIn(side[:-1]), FadeIn(watch), FadeIn(dots[:-1]))
        self.next_slide()
        tuned = [rows[MODEL_ORDER.index(n_)][0] for n_ in ("YOLOv5s", "YOLO26n")]
        self.play(
            FadeIn(rows[-1][0], shift=RIGHT * 0.2), GrowFromEdge(rows[-1][1], LEFT), GrowFromEdge(rows[-1][3], LEFT), FadeIn(side[-1]), FadeIn(dots[-1]),
            *[Transform(l_, txt(f"{l_.text}, fine-tuned", 24, bold=n_ == "YOLO26n").move_to(l_, aligned_edge=RIGHT)) for l_, n_ in zip(tuned, ("YOLOv5s", "YOLO26n"))],
        )
        self.play(FadeIn(rows[-1][2]), FadeIn(rows[-1][4]))
        self.next_slide()
        d_gap = round((m["YOLO26n"] - D.MODEL_DISTILLED[1]) * 100)
        verdict = pill(f"direct training wins: +{d_gap} pp species mAP", 26, fill=GREEN, bold=True).move_to([1.0, -3.45, 0])
        self.play(FadeIn(verdict, shift=DOWN * 0.2))
        self.next_slide()


# =================================================================================
# Act 6 — Summary: one picture from every step, in order
# =================================================================================
# (snapshot, or None for the AX Visio photo itself; caption)
SUMMARY = [
    (None, "MammalID"),
    ("images_per_class", "Data Gathering"),
    ("local_vs_gemini", "Synthetic Image Generation"),
    ("ranking", "Generator Comparison"),
    ("band_results", "Training Results"),
    ("models", "Model Comparison"),
]


def _content_crop(path):
    """Snapshot without its top band (logo and title), so the content fills the tile."""
    out = SNAPS / f"crop_{path.name}"
    if path.exists() and (not out.exists() or out.stat().st_mtime < path.stat().st_mtime):
        with Image.open(path) as im:
            w, h = im.size
            im.crop((0, int(h * 0.2), w, h)).save(out)
    return out


class Summary3(Talk3):
    def build(self):
        # 23 — Summary
        self.play(FadeIn(title("Summary")))
        tw, aspect = 4.0, 16 / 9 / 0.8  # snapshots minus their top 20 % (logo and title)
        cells = Group()
        for k, (name, caption) in enumerate(SUMMARY, 1):
            frame = RoundedRectangle(corner_radius=0.08, width=tw + 0.04, height=tw / aspect + 0.04).set_stroke(MIST, 1.5, 0.7)
            if name is None:  # the device itself, whole, centred in the tile
                m = photo(IMG / "deck/ax_visio_top.jpg", height=tw / aspect - 0.1, radius=0.05)
            else:
                m = photo(_content_crop(SNAPS / f"{name}.png"), width=tw, aspect=aspect, radius=0.05)
            cap = pill(f"{k}. {caption}", 17, bold=True)
            cells.add(Group(frame, m.move_to(frame), cap.next_to(frame, DOWN, buff=0.1)))
        cells.arrange_in_grid(rows=2, cols=3, buff=(0.35, 0.35)).move_to(DOWN * 0.45)
        # the tiles come in one after another by themselves
        self.play(LaggedStart(*[FadeIn(cell, scale=0.9) for cell in cells], lag_ratio=0.7), run_time=3.6)
        self.next_slide()
