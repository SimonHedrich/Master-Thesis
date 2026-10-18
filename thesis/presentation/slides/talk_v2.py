"""Final thesis presentation, version 2: built up step by step, from use case to result.

Six acts, one scene each: the use case and its challenges, the data, generated images,
supplementing the classes (bands), the models on embedded hardware, and a summary that
replays one picture from every act. Images come from thesis/presentation/images/ (built
by collect_images.py); numbers from data.py. talk.py keeps the first version.

Render (from thesis/presentation/slides/; the summary needs the earlier scenes rendered
first in the same run, because it reuses their snapshots):
    uv run manim render -qk --frame_rate 30 talk_v2.py UseCase DataV2 Generative Bands2 ModelsV2 Summary
Present:
    uv run python present.py -F UseCase DataV2 Generative Bands2 ModelsV2 Summary
"""

from __future__ import annotations

import csv
import textwrap

import numpy as np
from manim import (
    DL,
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
    RoundedRectangle,
    SurroundingRectangle,
    Transform,
    VGroup,
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
SNAPS = CACHE / "snaps"
PROMPTS = HERE / "assets" / "prompts"


# =================================================================================
# Shared pieces
# =================================================================================
class Talk2(Slide):
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

    def snap(self, name: str):
        """Save the current frame; the summary scene shows it again."""
        SNAPS.mkdir(parents=True, exist_ok=True)
        Image.fromarray(self.renderer.get_frame()).convert("RGB").save(SNAPS / f"{name}.png")

    @staticmethod
    def nda():
        return pill("NDA", 16).to_corner(UR, buff=0.3)


def disc(inner, r: float = 0.95) -> Group:
    c = Circle(radius=r).set_fill(TEAL, 0.35).set_stroke(MIST, 2)
    return Group(c, inner.move_to(c))


def prompt_text_card(name: str, width: float, max_height: float) -> ImageMobject:
    """A prompt typeset at one fixed size, so card length shows prompt length."""
    out = CACHE / f"prompt_{name}.png"
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
        if cut:  # fade the last lines out: the prompt keeps going
            fade = 8 * lh
            for k in range(fade):
                a = int(255 * k / fade)
                d.line([(0, h - fade + k), (w, h - fade + k)], fill=(255, 255, 255, a))
        im.save(out)
    m = ImageMobject(str(out))
    return m.scale_to_fit_width(width)


# =================================================================================
# Act 1 — The use case: BirdID, MammalID, and how to improve it
# =================================================================================
class UseCase(Talk2):
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
        self.play(FadeIn(shop, scale=0.96))
        self.page()

        # 3 — BirdID: optics and display meet in the eyepiece
        self.play(FadeIn(title("BirdID")), FadeIn(self.nda()))
        bird = photo(IMG / "usecase/barn_swallow.png", height=1.7).move_to([-4.6, 0.95, 0])
        bird_l = pill("bird", 20).next_to(bird, LEFT, buff=0.15).align_to(bird, UP)
        cam = photo(IMG / "nda/ax_visio.jpg", height=1.9).move_to([-0.9, 1.0, 0])
        cam_l = pill("camera", 20).next_to(cam, DOWN, buff=0.15)
        nn = Group(card(1.6, 1.6), photo(IMG / "deck/icon_neural_net.png", height=1.3, radius=0)).move_to([3.4, 1.0, 0])
        nn_l = pill("neural network", 20).next_to(nn, DOWN, buff=0.15)
        overlay = photo(IMG / "nda/overlay_display.png", height=1.55).move_to([3.4, -2.2, 0])
        ov_l = pill("overlay display", 20).next_to(overlay, DOWN, buff=0.15)
        view = photo(IMG / "nda/eyepiece_view.jpg", height=2.5, aspect=1.0, radius=0.5).move_to([-4.6, -1.95, 0])
        view_l = pill("view through the eyepiece", 20).move_to([-1.0, -2.75, 0])
        result = pill("Barn swallow", 24, fill=GREEN, bold=True).move_to(view.get_bottom() + UP * 0.25)

        a_cam = arrow(bird.get_right(), cam.get_left())
        a_nn = arrow(cam.get_right(), nn.get_left())
        a_ov = arrow(nn_l.get_bottom(), overlay.get_top())
        a_disp = arrow(overlay.get_left(), view.get_right())
        a_opt = arrow(bird.get_bottom(), view.get_top(), buff=0.1)
        opt_l = txt("optics", 18, MIST).next_to(a_opt, LEFT, buff=0.15)
        disp_l = txt("display", 18, MIST).next_to(a_disp, UP, buff=0.08)

        self.play(FadeIn(bird), FadeIn(bird_l))
        self.play(Create(a_cam), FadeIn(cam), FadeIn(cam_l))
        self.next_slide()
        self.play(Create(a_nn), FadeIn(nn), FadeIn(nn_l))
        self.play(Create(a_ov), FadeIn(overlay), FadeIn(ov_l))
        self.next_slide()
        self.play(FadeIn(view), Create(a_opt), FadeIn(opt_l))
        self.play(Create(a_disp), FadeIn(disp_l), FadeIn(view_l))
        self.play(FadeIn(result, scale=1.1))
        self.page()

        # 4 — MammalID, laid out as in the interim deck (slide 6)
        self.play(FadeIn(title("MammalID")), FadeIn(self.nda()))
        c, w, h = pptx_box(0.47, 1.73, 2.99, 2.1)
        dial = photo(IMG / "nda/mode_dial.png", width=w, aspect=w / h).move_to(c)
        c, w, h = pptx_box(2.06, 1.83, 0.38, 0.35)
        box = RoundedRectangle(corner_radius=0.06, width=w, height=h).set_stroke(GREEN, 5).move_to(c)
        c, w, h = pptx_box(3.93, 0.21, 4.41, 4.41)
        blog = photo(IMG / "usecase/speciesnet_blog.png", width=w, aspect=1.0, top=True, radius=0.03).move_to(c)
        c, w, h = pptx_box(6.74, 2.97, 2.93, 2.2)
        trap = photo(IMG / "usecase/camera_trap_armadillo.jpg", width=w, aspect=w / h).move_to(c)
        c, w, h = pptx_box(2.56, 4.25, 1.66, 0.49)
        md = pill("MegaDetector", 22).move_to(c)
        self.play(FadeIn(dial))
        self.play(Create(box))
        self.next_slide()
        self.play(FadeIn(blog, shift=UP * 0.2))
        self.play(FadeIn(trap, shift=LEFT * 0.2), FadeIn(md))
        self.snap("mammalid")
        self.page()

        # 5 — How MammalID works: detector + classifier
        head = title("Two models, one answer")
        frame = photo(IMG / "usecase/ocelot.jpg", height=3.6).move_to([-4.1, -0.4, 0])
        W, H = frame.width, frame.height
        x0, y0 = frame.get_left()[0], frame.get_top()[1]
        bx = D.OCELOT_BBOX
        bbox = Rectangle(width=(bx[2] - bx[0]) / 800 * W, height=(bx[3] - bx[1]) / 600 * H).set_stroke(GREEN, 5)
        bbox.move_to([x0 + (bx[0] + bx[2]) / 2 / 800 * W, y0 - (bx[1] + bx[3]) / 2 / 600 * H, 0])
        md = pill("MegaDetector · 94 %", 20).next_to(bbox, UP, buff=0.12).align_to(bbox, LEFT)
        crop = photo(IMG / "usecase/ocelot_crop.jpg", height=2.3).move_to([1.4, 0.3, 0])
        sn = pill("SpeciesNet", 22).next_to(crop, DOWN, buff=0.25)
        out = pill("Ocelot · 98.8 %", 28, fill=GREEN, bold=True).move_to([4.75, 0.3, 0])
        a1 = arrow(bbox.get_right(), crop.get_left())
        a2 = arrow(crop.get_right(), out.get_left())
        size = pill("≈ 193 M parameters", 24).move_to([1.4, -2.5, 0])
        self.play(FadeIn(head), FadeIn(frame))
        self.next_slide()
        self.play(Create(bbox), FadeIn(md))
        self.play(Create(a1), FadeIn(crop, shift=RIGHT * 0.3), FadeIn(sn))
        self.play(Create(a2), FadeIn(out, scale=1.1))
        self.next_slide()
        self.play(FadeIn(size, shift=UP * 0.2))
        self.page()

        # 6 — Built for camera traps, used through binoculars
        self.play(FadeIn(title("Built for camera traps")))
        trap = labelled(photo(IMG / "usecase/camera_trap_armadillo.jpg", width=5.6, aspect=4 / 3), "SpeciesNet: trained on camera traps", DL, 20)
        rhino = labelled(photo(IMG / "usecase/rhino.jpg", width=5.6, aspect=4 / 3), "binocular: daylight, distance", DL, 20)
        pair = Group(trap, rhino).arrange(RIGHT, buff=0.6).move_to(DOWN * 0.35)
        self.play(FadeIn(trap))
        self.play(FadeIn(rhino))
        self.next_slide()
        miss = pill("rhinos: 40 of 1,510 photos recognised", 22, fill=GREEN, bold=True)
        miss.next_to(rhino[0], UP, buff=-0.6).align_to(rhino[0], RIGHT).shift(LEFT * 0.15)
        self.play(FadeIn(miss, shift=DOWN * 0.15))
        self.next_slide()
        q = pill("Better with a model built for this use case?", 28, bold=True).to_edge(DOWN, buff=0.35)
        self.play(FadeIn(q, shift=UP * 0.2))
        self.page()

        # 7 — Three challenges
        self.play(FadeIn(title("Three challenges")))
        items = [
            (icon(IMG / "deck/icon_image.png", 0.95), "Data", "225 mammals,\nmany rarely photographed"),
            (icon(IMG / "deck/icon_chip.png", 1.1), "Hardware", "a mobile chip,\noffline"),
            (stopwatch(0.45), "Speed", "an answer while the\nanimal is still in view"),
        ]
        cols = []
        for inner, head_s, sub in items:
            g = disc(inner, 1.1)
            h_ = txt(head_s, 32, bold=True).next_to(g, DOWN, buff=0.35)
            s_ = txt(sub, 20, MIST).next_to(h_, DOWN, buff=0.15)
            cols.append(Group(g, h_, s_))
        Group(*cols).arrange(RIGHT, buff=1.6, aligned_edge=UP).move_to(DOWN * 0.3)
        for g in cols:
            self.play(FadeIn(g, shift=UP * 0.2), run_time=0.6)
        self.snap("challenges")
        self.next_slide()


# =================================================================================
# Act 2 — Data: which classes, which sources, which images
# =================================================================================
class DataV2(Talk2):
    def build(self):
        # 8 — Which animals?
        tree = photo(IMG / "data/taxonomy_hierarchy.png", height=5.2).move_to([-2.6, -0.4, 0])
        n = big_number("~2,500", "species SpeciesNet knows").move_to([3.7, 0.3, 0])
        self.play(FadeIn(title("Which animals?")), FadeIn(tree))
        self.play(FadeIn(n, shift=UP * 0.2))
        self.next_slide()
        m = big_number("Mammals", "seen through binoculars", size=64).move_to(n)
        self.play(Transform(n, m))
        self.page()

        # 9 — Hedgehogs: how fine to distinguish
        hedgehogs = sorted((IMG / "data/hedgehogs").glob("*.*"))
        grid = collage(hedgehogs, cols=6, cell_w=2.05).move_to(DOWN * 0.35)
        lbl = pill("17 species  →  1 class", 34, fill=GREEN, bold=True).move_to(grid).shift(DOWN * 0.3)
        self.play(LaggedStart(*[FadeIn(t, scale=0.9) for t in grid], lag_ratio=0.06), run_time=1.6)
        self.next_slide()
        self.play(FadeIn(lbl, scale=1.15))
        self.page()

        # 10 — 225 classes by order
        self.play(FadeIn(title("225 classes")))
        clusters = VGroup()
        golden = np.pi * (3 - np.sqrt(5))
        for name, k in D.ORDERS.items():
            dots = VGroup(*[
                Dot(radius=0.055, color=MIST).move_to([0.115 * np.sqrt(i + 0.5) * np.cos(i * golden), 0.115 * np.sqrt(i + 0.5) * np.sin(i * golden), 0])
                for i in range(k)
            ])
            lab = VGroup(txt(name, 18, MIST), txt(str(k), 26, bold=True)).arrange(DOWN, buff=0.08)
            clusters.add(VGroup(dots, lab))
        clusters.arrange(RIGHT, buff=0.45).move_to(DOWN * 0.2)
        for c in clusters:
            c[1].next_to(c[0], DOWN, buff=0.3)
            c[1].align_to([0, -1.45, 0], UP)
        self.play(LaggedStart(*[LaggedStart(*[GrowFromCenter(d) for d in c[0]], lag_ratio=0.02) for c in clusters], lag_ratio=0.15), run_time=2.5)
        self.play(LaggedStart(*[FadeIn(c[1]) for c in clusters], lag_ratio=0.1))
        self.snap("classes")
        self.page()

        # 11 — Images per class: GBIF, then four more sources
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
        totals = D.source_counts()
        new = VGroup()
        for i, (_, tot) in enumerate(totals):
            b = Rectangle(width=bw * 0.7, height=max(y_of(tot), 0.025)).set_fill(MIST, 1).set_stroke(width=0)
            b.move_to([left + (i + 0.5) * bw, base_y, 0], aligned_edge=DOWN)
            new.add(b)
        src = pill("+ iNaturalist · Wikimedia · OpenImages · images.cv", 22).move_to(p1).align_to(p1, LEFT)
        lic = pill("commercially usable licences only", 20).next_to(src, DOWN, buff=0.2).align_to(src, LEFT)
        p3 = pill("29 classes  still < 100", 24, fill=GREEN, bold=True).next_to(lic, DOWN, buff=0.2).align_to(src, LEFT)
        self.play(FadeOut(p1), FadeOut(p2), FadeIn(src), FadeIn(lic))
        self.play(Transform(bars, new), run_time=1.8)
        self.play(FadeIn(p3, scale=1.1))
        self.snap("images_per_class")
        self.page()

        # 12 — Not every "animal" is an animal
        rejects = sorted((IMG / "data").glob("reject_*"))
        rej = collage(rejects, cols=4, cell_w=2.85).move_to(DOWN * 0.3)
        self.play(FadeIn(title("Is this an animal?")))
        self.play(LaggedStart(*[FadeIn(t, scale=0.9) for t in rej], lag_ratio=0.12), run_time=1.6)
        self.next_slide()
        stages = ["Metadata", "Heuristics", "MegaDetector", "VLM check", "SpeciesNet"]
        chips = VGroup(*[pill(s, 22) for s in stages]).arrange(RIGHT, buff=0.6).move_to(DOWN * 3.15)
        arrs = VGroup(*[arrow(a.get_right(), b.get_left(), buff=0.08) for a, b in zip(chips, chips[1:])])
        self.play(rej.animate.scale(0.82).shift(UP * 0.45))
        self.play(LaggedStart(*[FadeIn(c, shift=RIGHT * 0.2) for c in chips], lag_ratio=0.2), Create(arrs), run_time=1.5)
        self.snap("is_animal")
        self.next_slide()
        hl = pill("SpeciesNet", 22, fill=GREEN, bold=True).move_to(chips[-1])
        note = pill("the species labels come from SpeciesNet", 22, fill=GREEN).next_to(chips[-1], UP, buff=0.3).align_to(chips[-1], RIGHT)
        self.play(Transform(chips[-1], hl), FadeIn(note, shift=UP * 0.2))
        self.next_slide()


# =================================================================================
# Act 3 — Generated images: which generator, which prompt
# =================================================================================
class Generative(Talk2):
    def build(self):
        # 13 — Image generation keeps getting better
        self.play(FadeIn(title("Image generation keeps getting better")))
        steps = [
            ("2022", "Stable Diffusion"),
            ("2023", "SDXL"),
            ("2024", "SD 3 · FLUX.1"),
            ("2025", "FLUX.2\nHiDream\nQwen-Image"),
            ("now", "Gemini 3.1\nFlash Image\ngpt-image-2"),
        ]
        xs = np.linspace(-5.5, 5.3, len(steps))
        line = Line([-6.2, -0.2, 0], [6.2, -0.2, 0]).set_stroke(MIST, 3, 0.7)
        marks = []
        for x, (year, models) in zip(xs, steps):
            d = Dot(radius=0.13, color=GREEN if year == "now" else WHITE).move_to([x, -0.2, 0])
            y = txt(year, 30, bold=True).next_to(d, UP, buff=0.3)
            m = txt(models, 20, MIST if year != "now" else WHITE).next_to(d, DOWN, buff=0.35)
            marks.append(VGroup(d, y, m))
        self.play(Create(line))
        self.play(LaggedStart(*[FadeIn(mk, shift=UP * 0.15) for mk in marks], lag_ratio=0.3), run_time=2.2)
        self.page()

        # 14 — Local generators vs Gemini (binturong)
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
        columns.arrange(RIGHT, buff=0.4).move_to(DOWN * 0.35)
        local = txt("local models", 22, MIST).next_to(Group(*columns[1:4]), UP, buff=0.12)
        self.play(FadeIn(columns[0]))
        self.next_slide()
        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.2) for c in columns[1:4]], lag_ratio=0.25), FadeIn(local), run_time=1.5)
        self.next_slide()
        ring = SurroundingRectangle(columns[4][0], buff=0.08, corner_radius=0.15).set_stroke(GREEN, 6)
        self.play(FadeIn(columns[4], shift=LEFT * 0.3))
        self.play(Create(ring))
        self.snap("local_vs_gemini")
        self.page()

        # 15 — Three zebras: details decide
        zs = Group()
        for path, name in (("zebra_grevys_gemini.png", "Grévy's zebra"), ("zebra_mountain_gemini.png", "Mountain zebra"), ("zebra_plains_gemini.png", "Plains zebra")):
            zs.add(labelled(photo(IMG / "generators" / path, width=4.1, aspect=4 / 3), name, DL, 22))
        zs.arrange(RIGHT, buff=0.3).move_to(DOWN * 0.2)
        self.play(FadeIn(title("Three zebras, generated")), LaggedStart(*[FadeIn(z, shift=UP * 0.2) for z in zs], lag_ratio=0.3))
        self.page()

        # 16 — The production prompt
        cardm = prompt_card(6.6).move_to([-2.4, -0.35, 0])
        self.play(FadeIn(cardm, shift=UP * 0.4))
        self.next_slide()
        keys = ["narrow vertical stripes", "white belly", "large rounded ears", "brown muzzle", "pose · habitat · light · distance"]
        kp = VGroup(*[pill(k, 22, fill=GREEN if i < 4 else WHITE) for i, k in enumerate(keys)]).arrange(DOWN, buff=0.22, aligned_edge=LEFT)
        kp.move_to([4.4, 0.3, 0])
        words = pill("≈ 1,300 words", 26, bold=True).next_to(kp, DOWN, buff=0.6).align_to(kp, LEFT)
        self.play(LaggedStart(*[FadeIn(k, shift=LEFT * 0.3) for k in kp], lag_ratio=0.2))
        self.play(FadeIn(words))
        self.page()

        # 17 — Prompt regimes: same lion, four prompt lengths
        self.play(FadeIn(title("How much does the generator read?")))
        regimes = [
            ("compressed", "prompt_compressed_sd35m.jpg", "SD 3.5 M", "≤ 75 tokens · 20 words"),
            ("maxlen256", "prompt_maxlen256_sd35m.jpg", "SD 3.5 M", "≤ 256 tokens · 149 words"),
            ("maxlen512", "prompt_maxlen512_flux2-klein.jpg", "FLUX.2 klein", "≤ 512 tokens · 380 words"),
            ("full", "prompt_full_gemini.jpg", "Gemini", "full · 4,642 words"),
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
        for im, tag, pc in cols_:
            self.play(FadeIn(im), FadeIn(tag), GrowFromEdge(pc, UP), run_time=0.8)
        self.snap("prompt_regimes")
        self.page()

        # 18 — The experiment: same prompt, eight sources
        lineup = {
            "Real photo": "lion_real.jpg", "SD 3.5 Large": "lion_sd35-large.jpg", "SD 3.5 Medium": "lion_sd35m.jpg",
            "FLUX.2 klein": "lion_flux2-klein.jpg", "HiDream-I1": "lion_hidream-i1.jpg", "RealVisXL": "lion_realvisxl.jpg",
            "Gemini Flash": "lion_gemini.jpg", "gpt-image-2": "lion_gpt-image-2-low.jpg",
        }
        tiles_ = Group(*[labelled(photo(IMG / "generators" / f, width=3.05, aspect=4 / 3), n, DL, 18) for n, f in lineup.items()])
        tiles_.arrange_in_grid(rows=2, cols=4, buff=0.2).move_to(DOWN * 0.3)
        self.play(FadeIn(title("Which generator helps the detector?")), FadeIn(tiles_[0]))
        self.play(LaggedStart(*[FadeIn(t, shift=UP * 0.15) for t in tiles_[1:]], lag_ratio=0.15), run_time=1.8)
        self.next_slide()
        exp = pill("12 setups · 1,200 images each · same detector · scored on 9,742 real photos", 20).to_edge(DOWN, buff=0.25)
        self.play(FadeIn(exp, shift=UP * 0.2))
        self.page()

        # 19 — Ranking
        self.play(FadeIn(title("Which generator helps the detector?")))
        rows = Group()
        vmax, L, bars_x = 0.15, 5.6, -3.3
        for name, mean, std, kind in D.GENERATORS:
            color = {"api": SKY, "pick": GREEN, "local": MIST}[kind]
            lab = txt(name, 20, WHITE if kind != "local" else MIST)
            bar = hbar(mean, vmax, L, h=0.34, color=color, opacity=1 if kind != "local" else 0.7)
            rows.add(VGroup(lab, bar))
        rows.arrange(DOWN, buff=0.14).move_to([0, -0.45, 0])
        for row, (name, mean, std, kind) in zip(rows, D.GENERATORS):
            row[0].next_to([bars_x - 0.25, row.get_y(), 0], LEFT, buff=0)
            row[1].move_to([bars_x, row.get_y(), 0], aligned_edge=LEFT)
            err = Line([bars_x + L * (mean - std) / vmax, row.get_y(), 0], [bars_x + L * (mean + std) / vmax, row.get_y(), 0]).set_stroke(WHITE, 2, 0.6)
            val = txt(f"{mean:.3f}", 18, MIST).move_to([bars_x + L + 0.55, row.get_y(), 0])
            row.add(err, val)
        legend = VGroup(pill("API", 18, fill=SKY), pill("local", 18, fill=MIST)).arrange(RIGHT, buff=0.2).to_corner(UR, buff=0.4).shift(DOWN * 0.55)
        self.play(FadeIn(legend), LaggedStart(*[FadeIn(r[0]) for r in rows], lag_ratio=0.05))
        self.play(LaggedStart(*[GrowFromEdge(r[1], LEFT) for r in rows], lag_ratio=0.08), run_time=1.6)
        self.play(*[FadeIn(r[2]) for r in rows], *[FadeIn(r[3]) for r in rows])
        self.next_slide()
        pick = pill("production choice: #3", 20, fill=GREEN, bold=True).next_to(rows[2][3], RIGHT, buff=0.25)
        self.play(FadeIn(pick, shift=LEFT * 0.2))
        self.next_slide()
        cheap = pill("#1 = cheapest setting", 20, bold=True).next_to(rows[0][3], RIGHT, buff=0.25)
        self.play(FadeIn(cheap, shift=LEFT * 0.2))
        self.snap("ranking")
        self.page()

        # 20 — Shorter prompt, worse data
        self.play(FadeIn(title("Shorter prompt, worse data")))
        groups = VGroup()
        L2, vmax2, x0 = 6.0, 0.15, -3.6
        for j, (name, f, c) in enumerate(D.PROMPT_ABLATION):
            y = 1.0 - j * 2.5
            lab = txt(name, 26, bold=True).move_to([x0, y + 0.65, 0], aligned_edge=LEFT)
            bf = hbar(f, vmax2, L2, h=0.45, color=SKY).move_to([x0, y, 0], aligned_edge=LEFT)
            bc = hbar(c, vmax2, L2, h=0.45, color=MIST, opacity=0.75).move_to([x0, y - 0.62, 0], aligned_edge=LEFT)
            vf = txt(f"{f:.3f}  full prompt", 20, MIST).next_to(bf, RIGHT, buff=0.2)
            vc = txt(f"{c:.3f}  compressed", 20, MIST).next_to(bc, RIGHT, buff=0.2)
            loss = pill(f"−{round((1 - c / f) * 100)} %", 28, fill=GREEN, bold=True).move_to([5.6, y - 0.3, 0])
            groups.add(VGroup(lab, bf, vf, bc, vc, loss))
        self.play(*[FadeIn(g[0]) for g in groups], *[GrowFromEdge(g[1], LEFT) for g in groups], *[FadeIn(g[2]) for g in groups])
        self.next_slide()
        self.play(*[GrowFromEdge(g[3], LEFT) for g in groups], *[FadeIn(g[4]) for g in groups])
        self.play(*[FadeIn(g[5], scale=1.2) for g in groups])
        self.page()

        # 21 — Rare animals are a blind spot
        classes, cols2, grid = D.generator_heatmap()
        cw2, ch2 = 0.46, 0.36
        hm = VGroup()
        x0, y0 = -4.3, 0.85
        for i, row in enumerate(grid):
            for j, v in enumerate(row):
                t = min(1.0, np.sqrt(v / 0.33))
                cell = Rectangle(width=cw2 - 0.04, height=ch2 - 0.04).set_stroke(width=0)
                hm.add(cell.set_fill(GREEN, 0.06 + 0.94 * t).move_to([x0 + j * cw2, y0 - i * ch2, 0]))
        names = VGroup(*[txt(c.replace(" family", ""), 15, MIST).move_to([x0 - 0.35, y0 - i * ch2, 0], aligned_edge=RIGHT) for i, c in enumerate(classes)])
        heads = VGroup(*[txt(c, 13, MIST).rotate(np.pi / 3).move_to([x0 + j * cw2, y0 + 0.6, 0], aligned_edge=DL).shift(LEFT * 0.1) for j, c in enumerate(cols2)])
        n_rare = 6
        rare_box = SurroundingRectangle(VGroup(*hm[: n_rare * len(cols2)]), buff=0.06, corner_radius=0.08).set_stroke(GREEN, 4)
        rare_l = pill("rare animals: AP < 0.07 for every generator", 20, fill=GREEN, bold=True).move_to([0, -3.55, 0]).align_to(names, LEFT)
        self.play(FadeIn(title("Rare animals are a blind spot")))
        self.play(FadeIn(names), FadeIn(heads), LaggedStart(*[FadeIn(c) for c in hm], lag_ratio=0.004), run_time=1.5)
        self.next_slide()
        self.play(Create(rare_box), FadeIn(rare_l))
        self.next_slide()
        real = labelled(photo(IMG / "generators/saiga_real.jpg", width=3.2, aspect=4 / 3), "real saiga", DL, 18).move_to([4.9, 1.0, 0])
        fakes = Group(*[photo(IMG / "generators" / f, width=1.15, aspect=4 / 3) for f in ("saiga_sd35-large.jpg", "saiga_flux2-klein.jpg", "saiga_hidream-i1.jpg")]).arrange(RIGHT, buff=0.1)
        fakes.next_to(real, DOWN, buff=0.35)
        fl = pill("generated (local models)", 18).next_to(fakes, DOWN, buff=0.15)
        nose = pill("no trunk-like nose", 20, fill=GREEN, bold=True).next_to(fl, DOWN, buff=0.15)
        self.play(FadeIn(real))
        self.play(LaggedStart(*[FadeIn(f, shift=UP * 0.2) for f in fakes], lag_ratio=0.2), FadeIn(fl))
        self.play(FadeIn(nose))
        self.snap("blind_spot")
        self.next_slide()


# =================================================================================
# Act 4 — Supplementing the classes: bands A–D and what they show
# =================================================================================
class Bands2(Talk2):
    def build(self):
        # 22 — The production set
        sheet = photo(IMG / "generators/grevys_synthetic_wall.png", width=FRAME_W - 0.6, radius=0).move_to(DOWN * 0.25)
        n = pill("23,850 synthetic images", 44, bold=True).move_to(sheet)
        self.play(FadeIn(sheet, scale=1.05), run_time=1.2)
        self.play(FadeIn(n, scale=1.2))
        self.page()

        # 23 — Same budget, different mix
        self.play(FadeIn(title("Same budget, different mix")))
        comps = {"A": (20, 0, 5), "B": (10, 10, 5), "C": (0, 20, 5), "D": (0, 150, 10)}
        cols = VGroup()
        for b, (ns, nr, per) in comps.items():
            t = tiles(ns, nr, per)
            lab = VGroup(txt(b, 44, bold=True), txt(f"{D.BAND_CLASSES[b]} classes", 18, MIST)).arrange(DOWN, buff=0.05)
            cols.add(VGroup(t, lab))
        cols.arrange(RIGHT, buff=1.1, aligned_edge=DOWN).move_to(DOWN * 0.5)
        for c in cols:
            c[1].next_to(c[0], DOWN, buff=0.3)
        bottom = min(c[0].get_bottom()[1] for c in cols)
        for c in cols:
            c.shift(UP * (bottom - c[0].get_bottom()[1]))
        legend = VGroup(
            VGroup(tile("synth"), txt("synthetic", 20)).arrange(RIGHT, buff=0.15),
            VGroup(tile("real"), txt("real", 20)).arrange(RIGHT, buff=0.15),
            txt("1 square = 10 training images", 18, MIST),
        ).arrange(RIGHT, buff=0.5).to_corner(UR, buff=0.4).shift(DOWN * 1.25)
        self.play(FadeIn(legend))
        self.play(LaggedStart(*[LaggedStart(*[FadeIn(s) for s in c[0]], lag_ratio=0.01) for c in cols], lag_ratio=0.3), *[FadeIn(c[1]) for c in cols], run_time=2.2)
        self.next_slide()
        abc = VGroup(*[c[0] for c in cols[:3]])
        brace = Line(abc.get_corner(UL) + UP * 0.35, abc.get_corner(UR) + UP * 0.35).set_stroke(GREEN, 4)
        bl = pill("200 images each", 22, fill=GREEN, bold=True).next_to(brace, UP, buff=0.15)
        self.play(Create(brace), FadeIn(bl))
        self.page()

        # 24 — Every class: training and test images
        self.play(FadeIn(title("Every class, train and test")))
        with open(REPO / "reports/class_split_counts.csv") as f:
            rows = list(csv.DictReader(f))
        order = {"A": 0, "B": 1, "C": 2, "D": 3}
        rows.sort(key=lambda r: (order[r["band"]], int(r["effective_pool"])))
        cap, chart_h, left, width, base = 1000, 4.0, -5.6, 12.1, -2.9
        bw = width / len(rows)
        segs = VGroup()
        for i, r in enumerate(rows):
            parts = [
                (int(r["train_real"]), MIST, 0.95),
                (int(r["train_synth"]), GREEN, 1.0),
                (int(r["test_real"]), SKY, 1.0),
                (50, GREEN, 0.4),  # balanced synthetic test set: 50 per class
            ]
            y = base
            col = VGroup()
            for v, color, op in parts:
                v_clip = min(v, max(0, cap - (y - base) / chart_h * cap))
                if v_clip <= 0:
                    continue
                hgt = v_clip / cap * chart_h
                col.add(Rectangle(width=bw * 0.78, height=hgt).set_fill(color, op).set_stroke(width=0).move_to([left + (i + 0.5) * bw, y, 0], aligned_edge=DOWN))
                y += hgt
            segs.add(col)
        grid = VGroup()
        for v in (200, 400, 600, 800, 1000):
            yy = base + v / cap * chart_h
            ln = Line([left, yy, 0], [left + width, yy, 0]).set_stroke(MIST, 1, 0.2)
            grid.add(ln, axis_label(f"{v}" if v < 1000 else "≥ 1,000").next_to(ln, LEFT, buff=0.1))
        seps, labs, i0 = VGroup(), VGroup(), 0
        for b in "ABCD":
            n_b = sum(1 for r in rows if r["band"] == b)
            x_mid = left + (i0 + n_b / 2) * bw
            labs.add(txt(b, 26, bold=True).move_to([x_mid, base - 0.35, 0]))
            if i0:
                seps.add(DashedLine([left + i0 * bw, base, 0], [left + i0 * bw, base + chart_h, 0]).set_stroke(WHITE, 2, 0.6))
            i0 += n_b
        legend = VGroup(
            pill("real train", 17, fill=MIST), pill("synthetic train", 17, fill=GREEN),
            pill("real test", 17, fill=SKY), pill("synthetic test", 17, fill="#B9F5BB"),
        ).arrange(RIGHT, buff=0.18).to_corner(UR, buff=0.4).shift(DOWN * 1.25)
        self.play(FadeIn(grid), FadeIn(legend), FadeIn(labs))
        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.1) for c in segs], lag_ratio=0.006), run_time=2.2)
        self.play(*[Create(s) for s in seps])
        self.snap("band_design")
        self.page()

        # 25 — Results per band
        self.play(FadeIn(title("Half synthetic is fine. All synthetic is not.")))
        comps_small = {"A": (20, 0), "B": (10, 10), "C": (0, 20), "D": (0, 20)}
        L, base = 4.1, -2.3
        xs = {"A": -4.5, "B": -1.5, "C": 1.5, "D": 4.5}
        group = {}
        for b, v in D.BAND_REAL.items():
            color = GREEN if b in "BC" else MIST
            bar = vbar(v, 0.8, L, w=1.5, color=color, opacity=1 if b != "D" else 0.3).move_to([xs[b], base, 0], aligned_edge=DOWN)
            if b == "D":
                bar.set_stroke(MIST, 2, 0.8)
            val = txt(f"{v:.2f}", 30, bold=True).next_to(bar, UP, buff=0.15)
            ns, nr = comps_small[b]
            strip = tiles(ns, nr, 10, s=0.12, gap=0.03).next_to(bar, DOWN, buff=0.25)
            lab = txt(b, 30, bold=True).next_to(strip, DOWN, buff=0.15)
            group[b] = VGroup(bar, val, strip, lab)
        ref = txt("reference:\nup to 1,500 real", 16, MIST).next_to(group["D"][0], RIGHT, buff=0.15)
        note = txt("YOLO26n · mAP on real photos", 18, MIST).to_corner(UR, buff=0.4).shift(DOWN * 1.25)
        self.play(*[FadeIn(g[2]) for g in group.values()], *[FadeIn(g[3]) for g in group.values()], FadeIn(note))
        self.play(LaggedStart(*[GrowFromEdge(group[b][0], DOWN) for b in "ABCD"], lag_ratio=0.25), run_time=1.8)
        self.play(*[FadeIn(g[1]) for g in group.values()], FadeIn(ref))
        self.next_slide()
        eq = pill("=", 36, fill=GREEN, bold=True).move_to([(xs["B"] + xs["C"]) / 2, group["B"][1].get_y(), 0])
        half = pill("½", 36, bold=True).next_to(group["A"][1], UP, buff=0.2)
        self.play(FadeIn(eq, scale=1.3))
        self.play(FadeIn(half, scale=1.3))
        self.snap("band_results")
        self.page()

        # 26 — Not every animal is the same
        self.play(FadeIn(title("Not every animal is the same")))
        pc = D.per_class_real_ap()
        y0, H = -3.0, 5.0
        bx = {"A": -5.2, "B": -3.2, "C": -1.2, "D": 0.8}
        grid = VGroup()
        for v in (0, 0.25, 0.5, 0.75, 1.0):
            yy = y0 + H * v
            ln = Line([-6.0, yy, 0], [1.7, yy, 0]).set_stroke(MIST, 1, 0.22)
            grid.add(ln, axis_label(f"{v:.2f}").next_to(ln, LEFT, buff=0.1))
        rng = np.random.default_rng(7)
        dots, meds, labs, special = VGroup(), VGroup(), VGroup(), {}
        for b, items in pc.items():
            for name, v in items:
                d = Dot(radius=0.055, color=MIST, fill_opacity=0.75).move_to([bx[b] + rng.uniform(-0.55, 0.55), y0 + H * v, 0])
                if b == "A" and name in ("striped hyaena", "mangabeys genus"):
                    d.set_color(GREEN).scale(2.2).set_fill(opacity=1)
                    special[name] = d
                dots.add(d)
            med = float(np.median([v for _, v in items]))
            meds.add(Line([bx[b] - 0.7, y0 + H * med, 0], [bx[b] + 0.7, y0 + H * med, 0]).set_stroke(WHITE, 5))
            labs.add(txt(b, 28, bold=True).move_to([bx[b], y0 - 0.45, 0]))
        self.play(FadeIn(grid), FadeIn(labs))
        self.play(LaggedStart(*[GrowFromCenter(d) for d in dots], lag_ratio=0.004), run_time=1.6)
        self.play(*[Create(m) for m in meds])
        self.next_slide()
        hy = labelled(photo(IMG / "bands/synthetic_striped_hyaena.png", width=2.9, aspect=4 / 3), "striped hyaena · 0.48", DL, 17).move_to([4.6, 1.2, 0])
        mg = labelled(photo(IMG / "bands/synthetic_mangabeys.png", width=2.9, aspect=4 / 3), "mangabeys · 0.01", DL, 17).move_to([4.6, -1.6, 0])
        note = txt("both trained on synthetic images only", 18, MIST).next_to(mg, DOWN, buff=0.2)
        l1 = Line(special["striped hyaena"].get_center(), hy.get_left(), buff=0.12).set_stroke(GREEN, 2)
        l2 = Line(special["mangabeys genus"].get_center(), mg.get_left(), buff=0.12).set_stroke(GREEN, 2)
        self.play(FadeIn(hy), Create(l1))
        self.play(FadeIn(mg), Create(l2), FadeIn(note))
        self.page()

        # 27 — Was the test fair?
        syn = labelled(photo(IMG / "bands/synthetic_test_plains_zebra.png", height=3.6, aspect=4 / 3), "synthetic test image", DL, 20)
        real = labelled(photo(IMG / "bands/real_test_zebra_herd.jpg", height=3.6, aspect=4 / 3), "real test image", DL, 20)
        pair = Group(syn, real).arrange(RIGHT, buff=0.4).move_to(DOWN * 0.3)
        self.play(FadeIn(title("Was the test fair?")), FadeIn(pair[0]))
        self.play(FadeIn(pair[1]))
        self.next_slide()
        self.play(pair.animate.scale(0.55).to_edge(LEFT, buff=0.5).set_y(0.95))
        self.play(pair[1].animate.next_to(pair[0], DOWN, buff=0.25))
        L, base = 4.6, -2.6
        bars = VGroup()
        for k, b in enumerate("ABCD"):
            x = 0.6 + k * 1.75
            s = vbar(D.BAND_SYNTH_TEST[b], 1.0, L, w=0.62, color=MIST, opacity=0.6).move_to([x - 0.34, base, 0], aligned_edge=DOWN)
            r = vbar(D.BAND_REAL_TEST[b], 1.0, L, w=0.62, color=GREEN).move_to([x + 0.34, base, 0], aligned_edge=DOWN)
            sv = txt(f"{D.BAND_SYNTH_TEST[b]:.2f}", 17, MIST).next_to(s, UP, buff=0.08)
            rv = txt(f"{D.BAND_REAL_TEST[b]:.2f}", 17).next_to(r, UP, buff=0.08)
            lab = txt(b, 28, bold=True).move_to([x, base - 0.4, 0])
            bars.add(VGroup(s, r, sv, rv, lab))
        leg = VGroup(pill("synthetic test", 18, fill=MIST), pill("real test", 18, fill=GREEN)).arrange(RIGHT, buff=0.2)
        leg.next_to(bars, UP, buff=0.35).align_to(bars, LEFT).shift(UP * 0.4)
        self.play(FadeIn(leg), *[FadeIn(g[4]) for g in bars])
        self.play(LaggedStart(*[a for g in bars for a in (GrowFromEdge(g[0], DOWN), GrowFromEdge(g[1], DOWN))], lag_ratio=0.1), run_time=1.6)
        self.play(*[FadeIn(g[2]) for g in bars], *[FadeIn(g[3]) for g in bars])
        self.next_slide()
        rule = pill("so: every number reported on real photos", 24, fill=GREEN, bold=True).to_edge(DOWN, buff=0.2).shift(RIGHT * 2.4)
        self.play(FadeIn(rule, shift=UP * 0.2))
        self.next_slide()


# =================================================================================
# Act 5 — Models: accuracy, size and speed on embedded hardware
# =================================================================================
class ModelsV2(Talk2):
    def build(self):
        # 28 — Three candidates
        self.play(FadeIn(title("Three candidates")))
        R = 2.45
        specs = [("MegaDetector + SpeciesNet", 193, "193 M", -2.6, BLUE), ("YOLOv5s", 7.63, "7.6 M", 2.7, MIST), ("YOLO26n", 2.71, "2.7 M", 5.0, GREEN)]
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
        tag = pill("in the binocular today", 18).next_to(circles[1][1], UP, buff=0.15)
        self.play(FadeIn(tag))
        self.page()

        # 29 — Accuracy, size and time per frame
        self.play(FadeIn(title("Accuracy on real photos")))
        L, vmax, x0 = 6.0, 0.7, -2.4
        rows = VGroup()
        for k, (name, m, p, t) in enumerate(D.MODELS):
            y = 1.3 - k * 1.15
            color = GREEN if name == "YOLO26n" else MIST
            lab = txt(name, 24, bold=name == "YOLO26n").move_to([x0 - 0.3, y, 0], aligned_edge=RIGHT)
            bar = hbar(m, vmax, L, h=0.55, color=color, opacity=1 if color == GREEN else 0.75).move_to([x0, y, 0], aligned_edge=LEFT)
            val = txt(f"{m:.2f}" + ("*" if k == 0 else ""), 26, bold=True).next_to(bar, RIGHT, buff=0.2)
            rows.add(VGroup(lab, bar, val))
        self.play(LaggedStart(*[FadeIn(r[0]) for r in rows], lag_ratio=0.1))
        self.play(LaggedStart(*[GrowFromEdge(r[1], LEFT) for r in rows], lag_ratio=0.15), run_time=1.5)
        self.play(*[FadeIn(r[2]) for r in rows])
        star = txt("* upper bound: MegaDetector drew the ground-truth boxes", 16, MIST).to_corner(DL, buff=0.4)
        self.play(FadeIn(star))
        self.next_slide()
        side = VGroup()
        for r, (name, m, p, t) in zip(rows, D.MODELS):
            sec = f"{t:.0f} s" if t >= 10 else f"{t:.2f} s"
            side.add(pill(f"{p:g} M · {sec}", 20, fill=GREEN if name == "YOLO26n" else WHITE).move_to([5.6, r.get_y(), 0]))
        watch = stopwatch(0.32).next_to(side, UP, buff=0.25)
        foot = txt("time per frame: Raspberry Pi 400 CPU (proxy for the QCS605), full precision", 16, MIST).next_to(star, UP, buff=0.08).align_to(star, LEFT)
        self.play(FadeIn(watch), LaggedStart(*[FadeIn(s, shift=LEFT * 0.2) for s in side], lag_ratio=0.15), FadeIn(foot))
        self.next_slide()
        call = pill("83× faster · 71× smaller · −0.07 mAP", 28, fill=GREEN, bold=True).move_to([1.0, -3.0, 0])
        self.play(FadeOut(star), FadeOut(foot), FadeIn(call, scale=1.1))
        self.snap("models")
        self.page()

        # 30 — Finding vs naming
        self.play(FadeIn(title("Finding is easy. Naming is hard.")))
        a = big_number(f"{D.DETECT_REAL:.2f}", "finds the animal", size=110)
        b = big_number(f"{D.FINE_REAL:.2f}", "names the species", size=110)
        ab = VGroup(a, b).arrange(RIGHT, buff=2.2).move_to(DOWN * 0.2)
        ar = arrow(a[0].get_right(), b[0].get_left(), color=GREEN, buff=0.35)
        self.play(FadeIn(a, shift=UP * 0.2))
        self.play(Create(ar), FadeIn(b, shift=UP * 0.2))
        self.next_slide()
        self.play(VGroup(ab, ar).animate.scale(0.4).to_corner(UR, buff=0.4).shift(DOWN * 0.15))
        asian = labelled(photo(IMG / "models/asian_elephant.jpg", height=3.3, aspect=1.0), "Asian elephant", DL, 20).move_to([-3.0, -1.0, 0])
        african = labelled(photo(IMG / "models/african_elephant.jpg", height=3.3, aspect=1.0), "African elephant", DR, 20).move_to([4.0, -1.0, 0])
        fwd = curved_arrow(asian.get_top() + RIGHT * 0.6 + UP * 0.1, african.get_top() + LEFT * 0.6 + UP * 0.1, angle=-0.9, color=GREEN)
        rev = curved_arrow(african.get_bottom() + LEFT * 0.6 + DOWN * 0.1, asian.get_bottom() + RIGHT * 0.6 + DOWN * 0.1, angle=-0.6, color=MIST)
        fl = pill(f"{D.ELEPHANT_FWD:,}×", 30, fill=GREEN, bold=True).next_to(fwd, UP, buff=0.05)
        rl = pill(f"{D.ELEPHANT_REV}×", 22).next_to(rev, DOWN, buff=0.02)
        self.play(FadeIn(asian), FadeIn(african))
        self.play(Create(fwd), FadeIn(fl))
        self.play(Create(rev), FadeIn(rl))
        self.page()

        # 31 — Can the big model teach the small one?
        self.play(FadeIn(title("Can a big model teach a small one?")))
        teacher = pill("teacher · 193 M", 20)
        student = pill("student · 2.7 M", 20, fill=GREEN)
        diag = VGroup(teacher, student).arrange(RIGHT, buff=1.6).to_corner(UR, buff=0.4).shift(DOWN * 1.3)
        da = arrow(teacher.get_right(), student.get_left(), buff=0.1)
        dl = txt("distillation", 16, MIST).next_to(da, UP, buff=0.06)
        self.play(FadeIn(diag), Create(da), FadeIn(dl))
        L, base = 4.3, -2.6
        bars = VGroup()
        for k, bnd in enumerate("ABCD"):
            x = -3.9 + k * 2.6
            kd, direct = D.KD_BANDS[bnd]
            s = vbar(kd, 0.8, L, w=0.9, color=MIST, opacity=0.6).move_to([x - 0.5, base, 0], aligned_edge=DOWN)
            r = vbar(direct, 0.8, L, w=0.9, color=GREEN).move_to([x + 0.5, base, 0], aligned_edge=DOWN)
            sv = txt(f"{kd:.2f}", 20, MIST).next_to(s, UP, buff=0.1)
            rv = txt(f"{direct:.2f}", 20, bold=True).next_to(r, UP, buff=0.1)
            lab = txt(bnd, 30, bold=True).move_to([x, base - 0.45, 0])
            bars.add(VGroup(s, r, sv, rv, lab))
        leg = VGroup(pill("with teacher", 20, fill=MIST), pill("direct training", 20, fill=GREEN)).arrange(RIGHT, buff=0.25)
        leg.move_to([LEFT_X, 1.75, 0], aligned_edge=LEFT)
        self.play(FadeIn(leg), *[FadeIn(g[4]) for g in bars])
        self.play(LaggedStart(*[GrowFromEdge(g[0], DOWN) for g in bars], lag_ratio=0.1), run_time=1.2)
        self.play(*[FadeIn(g[2]) for g in bars])
        self.next_slide()
        self.play(LaggedStart(*[GrowFromEdge(g[1], DOWN) for g in bars], lag_ratio=0.1), run_time=1.2)
        self.play(*[FadeIn(g[3]) for g in bars])
        self.next_slide()
        verdict = pill("direct training wins in every band", 26, fill=GREEN, bold=True).move_to([0, -3.6, 0])
        self.play(FadeIn(verdict, shift=DOWN * 0.2))
        self.next_slide()


# =================================================================================
# Act 6 — Summary: one picture from every step, in order
# =================================================================================
SUMMARY = [
    ("mammalid", "MammalID today"),
    ("challenges", "Data · hardware · speed"),
    ("classes", "225 classes"),
    ("images_per_class", "Too few images"),
    ("is_animal", "Filtering"),
    ("local_vs_gemini", "Local vs. Gemini"),
    ("prompt_regimes", "Prompt length"),
    ("ranking", "Generator ranking"),
    ("blind_spot", "Rare-animal blind spot"),
    ("band_design", "Bands A–D"),
    ("band_results", "Half synthetic is fine"),
    ("models", "Small, fast, close"),
]


def _content_crop(path):
    """Snapshot without its top band (logo and title), so the content fills the tile."""
    out = SNAPS / f"crop_{path.name}"
    if path.exists() and (not out.exists() or out.stat().st_mtime < path.stat().st_mtime):
        with Image.open(path) as im:
            w, h = im.size
            im.crop((0, int(h * 0.2), w, h)).save(out)
    return out


class Summary(Talk2):
    def build(self):
        # 32 — Summary
        self.play(FadeIn(title("Summary")))
        tw, aspect = 2.85, 16 / 9 / 0.8  # snapshots minus their top 20 % (logo and title)
        cells = Group()
        for name, caption in SUMMARY:
            m = photo(_content_crop(SNAPS / f"{name}.png"), width=tw, aspect=aspect, radius=0.05)
            frame = RoundedRectangle(corner_radius=0.08, width=tw + 0.04, height=tw / aspect + 0.04).set_stroke(MIST, 1.5, 0.7)
            cap = pill(caption, 14, bold=True)
            cells.add(Group(frame, m.move_to(frame), cap.next_to(frame, DOWN, buff=0.1)))
        cells.arrange_in_grid(rows=3, cols=4, buff=(0.3, 0.2)).move_to(DOWN * 0.7)
        for k, cell in enumerate(cells):  # one click per picture, so each gets its sentence
            if k:
                self.next_slide()
            self.play(FadeIn(cell, scale=0.9), run_time=0.45)
        self.next_slide()
        self.clear()

        # 33 — Thank you
        thanks = txt("Thank you", 80, bold=True).move_to(UP * 0.5)
        qs = pill("Questions?", 34, fill=GREEN, bold=True).next_to(thanks, DOWN, buff=0.5)
        self.play(FadeIn(thanks, shift=UP * 0.2))
        self.play(FadeIn(qs, scale=1.1))
        self.next_slide()
