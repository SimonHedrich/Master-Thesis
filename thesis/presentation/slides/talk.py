"""Final thesis presentation, built with manim + manim-slides.

Seven scenes, one per act of final-talk-storyline.md; the results acts follow
results-slides-strategy.md. Each `self.next_slide()` is one click in the presenter.

Render (from thesis/presentation/slides/):
    uv run manim render -qk --frame_rate 30 talk.py Intro Problem Data Generators Bands Models Close
Present:
    uv run python present.py -F Intro Problem Data Generators Bands Models Close
Export a self-contained HTML deck:
    uv run manim-slides convert --offline Intro Problem Data Generators Bands Models Close build/talk.html
"""

from __future__ import annotations

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
    Arrow,
    Circle,
    Create,
    CurvedArrow,
    DashedLine,
    Dot,
    FadeIn,
    FadeOut,
    GrowFromEdge,
    GrowFromCenter,
    Group,
    ImageMobject,
    LaggedStart,
    Line,
    Rectangle,
    RoundedRectangle,
    Square,
    SurroundingRectangle,
    Transform,
    ValueTracker,
    VGroup,
    always_redraw,
    linear,
    register_font,
    smooth,
)
from manim_slides import Slide
from PIL import Image, ImageDraw, ImageFont

import data as D
from theme import (
    BLUE,
    CACHE,
    FONTS,
    FRAME_W,
    GREEN,
    LEFT_X,
    MIST,
    NAVY,
    SKY,
    TEAL,
    WHITE,
    background,
    big_number,
    card,
    check_mark,
    collage,
    cross_mark,
    hbar,
    mosaic,
    photo,
    pill,
    stopwatch,
    tilde_mark,
    title,
    txt,
    vbar,
)


# =================================================================================
# Shared pieces
# =================================================================================
class TalkSlide(Slide):
    """Background + logo on every frame; subclasses implement build()."""

    def construct(self):
        with (
            register_font(str(FONTS / "AtkinsonHyperlegible-Regular.ttf")),
            register_font(str(FONTS / "AtkinsonHyperlegible-Bold.ttf")),
        ):
            self.bg = background()
            self.logo = photo(D.LOGO, height=0.5, radius=0).to_corner(UL, buff=0.25)
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
        """Close the current slide: wait for the click, then fade its content out."""
        self.next_slide()
        self.clear()

    @staticmethod
    def nda():
        return pill("NDA", 16).to_corner(UR, buff=0.3)


def icon(path, height: float, color: str = WHITE):
    """Recolour a line icon (transparent PNG) to one colour."""
    out = CACHE / f"icon_{path.stem}_{color.strip('#')}.png"
    if not out.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        im = Image.open(path).convert("RGBA")
        solid = Image.new("RGBA", im.size, color)
        solid.putalpha(im.getchannel("A"))
        solid.save(out)
    return ImageMobject(str(out)).scale_to_fit_height(height)


def tile(kind: str, s: float = 0.22) -> Square:
    """One square of a training-set composition: synthetic (green outline) or real (filled)."""
    sq = Square(s)
    if kind == "synth":
        return sq.set_stroke(GREEN, 2.5).set_fill(GREEN, 0.12)
    return sq.set_stroke(width=0).set_fill(MIST, 0.95)


def tiles(n_synth: int, n_real: int, per_row: int, s: float = 0.22, gap: float = 0.05) -> VGroup:
    g = VGroup(*[tile("synth", s) for _ in range(n_synth)], *[tile("real", s) for _ in range(n_real)])
    g.arrange_in_grid(cols=per_row, buff=gap, flow_order="rd")
    return g


def prompt_card(height: float):
    """The full Grévy's zebra prompt, typeset tiny on a white card (pre-rendered with PIL)."""
    out = CACHE / "prompt_card.png"
    if not out.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        lines = []
        for para in D.PROMPT.read_text().splitlines():
            lines += textwrap.wrap(para, 96) or [""]
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 22)
        w, lh = 1240, 30
        im = Image.new("RGBA", (w, 60 + lh * len(lines)), (255, 255, 255, 0))
        d = ImageDraw.Draw(im)
        d.rounded_rectangle((0, 0, w - 1, im.size[1] - 1), 36, fill=(255, 255, 255, 255))
        for i, line in enumerate(lines):
            bold = line.isupper() or line.rstrip(":").isupper()
            d.text((40, 30 + i * lh), line, fill=(6, 27, 89) if bold else (60, 70, 100), font=font)
        im.save(out)
    return ImageMobject(str(out)).scale_to_fit_height(height)


def text_lines(n: int, width: float, color: str = NAVY) -> VGroup:
    """Abstract lines of text inside a card (for prompt-length comparisons)."""
    rng = np.random.default_rng(n)
    rows = VGroup()
    for i in range(n):
        w = width * (0.55 + 0.45 * rng.random()) if i % 7 else width * 0.35
        rows.add(Line(ORIGIN, RIGHT * w).set_stroke(color, 3, 0.55))
    rows.arrange(DOWN, buff=0.09, aligned_edge=LEFT)
    return rows


def labelled(m, label: str, corner=DL, size: float = 18):
    """Photo with a pill sitting on its corner, as in the interim deck."""
    p = pill(label, size)
    p.move_to(m.get_corner(corner)).shift(-corner * np.array([0.0, 0.0, 0]))
    p.align_to(m, LEFT if corner[0] < 0 else RIGHT).shift(RIGHT * 0.12 if corner[0] < 0 else LEFT * 0.12)
    p.align_to(m, DOWN if corner[1] < 0 else UP).shift(UP * 0.12 if corner[1] < 0 else DOWN * 0.12)
    return Group(m, p)


def plan_diagram() -> dict:
    """The interim plan: teacher ensemble distils into a small student."""
    data = photo(D.OCELOT, height=1.3)
    data_lbl = txt("unlabeled\nphotos", 20, MIST).next_to(data, DOWN, buff=0.15)
    data_g = Group(data, data_lbl).move_to([-5.2, -0.2, 0])

    md = pill("MegaDetector", 22)
    sn = pill("SpeciesNet", 22)
    pair = VGroup(md, sn).arrange(RIGHT, buff=0.7)
    a = Arrow(md.get_right(), sn.get_left(), buff=0.08, stroke_width=3, color=WHITE, max_tip_length_to_length_ratio=0.3)
    t_box = RoundedRectangle(corner_radius=0.25, width=6.0, height=1.5).set_stroke(MIST, 2).set_fill(WHITE, 0.06)
    teacher = VGroup(t_box, pair, a).move_to([0.3, 1.2, 0])
    pair.move_to(t_box)
    a.put_start_and_end_on(md.get_right() + RIGHT * 0.08, sn.get_left() + LEFT * 0.08)
    t_lbl = txt("teacher · 193 M parameters", 18, MIST).next_to(t_box, UP, buff=0.1).align_to(t_box, LEFT)

    s_box = RoundedRectangle(corner_radius=0.25, width=6.0, height=1.5).set_stroke(MIST, 2).set_fill(WHITE, 0.06)
    s_box.move_to([0.3, -1.6, 0])
    s_in = pill("small detector", 22, fill=GREEN).move_to(s_box)
    student = VGroup(s_box, s_in)
    s_lbl = txt("student · ~3 M parameters", 18, MIST).next_to(s_box, DOWN, buff=0.1).align_to(s_box, LEFT)

    to_t = Arrow(data.get_right(), t_box.get_left(), buff=0.15, stroke_width=3, color=WHITE)
    to_s = Arrow(data.get_right(), s_box.get_left(), buff=0.15, stroke_width=3, color=WHITE)
    distil = CurvedArrow(t_box.get_right() + DOWN * 0.3, s_box.get_right() + UP * 0.3, angle=-1.6, color=GREEN, stroke_width=5)
    d_lbl = pill("distillation", 20, fill=GREEN).next_to(distil, RIGHT, buff=0.15)
    return dict(
        data=data_g, teacher=Group(teacher, t_lbl), student=Group(student, s_lbl),
        to_t=to_t, to_s=to_s, distil=VGroup(distil, d_lbl),
    )


def four_questions() -> list[Group]:
    """Icons for the four research questions."""
    r = 0.95

    def disc(inner):
        c = Circle(radius=r).set_fill(TEAL, 0.35).set_stroke(MIST, 2)
        return Group(c, inner.move_to(c))

    brain = icon(D.ICON_BRAIN, 1.1)
    image = icon(D.ICON_IMAGE, 0.95)
    mix = tiles(2, 2, 2, s=0.38, gap=0.1)
    chip = icon(D.ICON_CHIP, 1.1)
    labels = ["Teacher → student?", "Which generator?", "Real vs. synthetic?", "Fast enough?"]
    out = []
    for inner, lbl in zip([brain, image, mix, chip], labels):
        g = disc(inner)
        t = txt(lbl, 24).next_to(g, DOWN, buff=0.3)
        out.append(Group(g, t))
    Group(*out).arrange(RIGHT, buff=0.9).move_to(DOWN * 0.2)
    return out


def axis_label(s: str, size: float = 16):
    return txt(s, size, MIST)


# =================================================================================
# Act 0: Hook
# =================================================================================
class Intro(TalkSlide):
    def build(self):
        # 1 — Title
        tag = pill("Master Thesis", 20)
        head = txt("Real-Time Object Recognition\nwith Neural Networks\non Embedded Platforms", 50, bold=True, line_spacing=0.9)
        head.move_to([LEFT_X, 0.9, 0], aligned_edge=LEFT)
        tag.next_to(head, UP, buff=0.35).align_to(head, LEFT)
        face = photo(D.PORTRAIT, height=1.35, radius=0.5)
        name = txt("Simon Hedrich", 28)
        who = Group(face, name).arrange(RIGHT, buff=0.35).move_to([LEFT_X, -2.6, 0], aligned_edge=LEFT)
        self.play(FadeIn(tag, shift=UP * 0.2), FadeIn(head, shift=UP * 0.2), FadeIn(who), run_time=1.0)
        self.page()

        # 2 — The product
        shop = photo(D.SHOP, width=12.4, radius=0.02)
        shop.move_to(DOWN * 0.25)
        self.play(FadeIn(shop, scale=0.96))
        self.next_slide()
        w, h = shop.width, shop.height
        x0, y0 = shop.get_left()[0], shop.get_top()[1]
        ring = RoundedRectangle(corner_radius=0.12, width=w * 0.13, height=h * 0.09).set_stroke(GREEN, 6)
        ring.move_to([x0 + w * 0.567, y0 - h * 0.885, 0])
        price = pill("4.600 €", 34, bold=True).next_to(ring, UP, buff=0.25)
        self.play(Create(ring), FadeIn(price, shift=UP * 0.2))
        self.page()

        # 3 — BirdID: the binocular already names birds
        nda = self.nda()
        dev = photo(D.AX_VISIO, height=2.5).move_to([-4.55, -0.3, 0])
        view = photo(D.EYEPIECE, height=2.5, aspect=0.75).move_to([-0.95, -0.3, 0])
        nn = Group(card(1.7, 1.7), photo(D.NN_ICON, height=1.4, radius=0)).move_to([1.6, -0.3, 0])
        bird = photo(D.SWALLOW, height=1.7).move_to([4.85, -0.3, 0])
        lbl = pill("Barn swallow", 24, fill=GREEN).next_to(bird, DOWN, buff=0.25)
        head = title("BirdID")
        arrows = [Arrow(a.get_right(), b.get_left(), buff=0.15, stroke_width=4, color=WHITE) for a, b in ((dev, view), (view, nn), (nn, bird))]
        self.play(FadeIn(head), FadeIn(nda), FadeIn(dev))
        self.play(LaggedStart(*[a for pair in zip([Create(x) for x in arrows], [FadeIn(view), FadeIn(nn), FadeIn(bird)]) for a in pair], lag_ratio=0.35), run_time=2.2)
        self.play(FadeIn(lbl, shift=UP * 0.2))
        self.page()

        # 4 — MammalID?
        dial = photo(D.DIAL, height=5.2).move_to(DOWN * 0.4)
        q = pill("MammalID?", 40, bold=True).next_to(dial, RIGHT, buff=-1.0).shift(UP * 1.6)
        self.play(FadeIn(title("Turn the dial …")), FadeIn(self.nda()), FadeIn(dial))
        self.next_slide()
        self.play(FadeIn(q, scale=1.2))
        self.next_slide()


# =================================================================================
# Act 1: The obvious solution, and the plan
# =================================================================================
class Problem(TalkSlide):
    def build(self):
        # 5 — The ensemble that already works
        head = title("Two models, one answer")
        frame = photo(D.OCELOT, height=3.6).move_to([-4.1, -0.4, 0])
        W, H = frame.width, frame.height
        x0, y0 = frame.get_left()[0], frame.get_top()[1]
        bx = D.OCELOT_BBOX
        box = Rectangle(width=(bx[2] - bx[0]) / 800 * W, height=(bx[3] - bx[1]) / 600 * H).set_stroke(GREEN, 5)
        box.move_to([x0 + (bx[0] + bx[2]) / 2 / 800 * W, y0 - (bx[1] + bx[3]) / 2 / 600 * H, 0])
        md = pill("MegaDetector · 94 %", 20).next_to(box, UP, buff=0.12).align_to(box, LEFT)
        crop = photo(D.OCELOT_CROP, height=2.3).move_to([1.4, 0.3, 0])
        sn = pill("SpeciesNet", 22).next_to(crop, DOWN, buff=0.25)
        out = pill("Ocelot · 98.8 %", 28, fill=GREEN, bold=True).move_to([4.75, 0.3, 0])
        a1 = Arrow(box.get_right(), crop.get_left(), buff=0.15, stroke_width=4, color=WHITE)
        a2 = Arrow(crop.get_right(), out.get_left(), buff=0.15, stroke_width=4, color=WHITE)
        self.play(FadeIn(head), FadeIn(frame))
        self.next_slide()
        self.play(Create(box), FadeIn(md))
        self.play(Create(a1), FadeIn(crop, shift=RIGHT * 0.3), FadeIn(sn))
        self.play(Create(a2), FadeIn(out, scale=1.1))
        self.next_slide()

        # 6 — Plant the question: how fast on the binocular?
        size = pill("≈ 193 M parameters", 24).move_to([1.4, -2.5, 0])
        watch = stopwatch(0.7).move_to([4.75, -2.3, 0])
        q = txt("?", 72, bold=True).move_to(watch[0])
        self.play(FadeIn(size, shift=UP * 0.2))
        self.play(Create(watch[0]), FadeIn(watch[1]), FadeIn(q, scale=1.3))
        self.page()

        # 7 — The preliminary plan, built in three steps
        p = plan_diagram()
        self.play(FadeIn(title("The plan")), FadeIn(p["data"]), Create(p["to_t"]), FadeIn(p["teacher"]))
        self.next_slide()
        self.play(Create(p["to_s"]), FadeIn(p["student"]))
        self.next_slide()
        self.play(Create(p["distil"][0]), FadeIn(p["distil"][1]))
        self.page()

        # 8 — Four questions
        qs = four_questions()
        self.play(FadeIn(title("Four questions")))
        for g in qs:
            self.play(FadeIn(g, shift=UP * 0.2), run_time=0.6)
        self.next_slide()


# =================================================================================
# Act 2: The data that doesn't exist
# =================================================================================
class Data(TalkSlide):
    def build(self):
        # 9 — Taxonomy
        tree = photo(D.HIERARCHY, height=5.2).move_to([-2.6, -0.4, 0])
        n = big_number("~2,500", "species SpeciesNet knows").move_to([3.7, 0.3, 0])
        self.play(FadeIn(title("Which animals?")), FadeIn(tree))
        self.play(FadeIn(n, shift=UP * 0.2))
        self.next_slide()
        m = big_number("Mammals", "seen through binoculars", size=64).move_to(n)
        self.play(Transform(n, m))
        self.page()

        # 10 — Hedgehogs: some species can't be told apart
        grid = collage(D.HEDGEHOGS, cols=6, cell_w=2.05).move_to(DOWN * 0.35)
        lbl = pill("17 species  →  1 class", 34, fill=GREEN, bold=True).move_to(grid).shift(DOWN * 0.3)
        self.play(LaggedStart(*[FadeIn(t, scale=0.9) for t in grid], lag_ratio=0.06), run_time=1.6)
        self.next_slide()
        self.play(FadeIn(lbl, scale=1.15))
        self.page()

        # 11 — 225 classes, by order
        self.play(FadeIn(title("225 classes")))
        clusters = VGroup()
        golden = np.pi * (3 - np.sqrt(5))
        for name, k in D.ORDERS.items():
            dots = VGroup(*[
                Dot(radius=0.055, color=MIST).move_to([0.115 * np.sqrt(i + 0.5) * np.cos(i * golden), 0.115 * np.sqrt(i + 0.5) * np.sin(i * golden), 0])
                for i in range(k)
            ])
            lab = VGroup(txt(name, 18, MIST), txt(str(k), 26, bold=True)).arrange(DOWN, buff=0.08)
            lab.next_to(dots, DOWN, buff=0.3)
            clusters.add(VGroup(dots, lab))
        clusters.arrange(RIGHT, buff=0.45).move_to(DOWN * 0.2)
        for c in clusters:  # labels on one baseline under the biggest cluster
            c[1].align_to([0, -1.45, 0], UP)
        self.play(LaggedStart(*[LaggedStart(*[GrowFromCenter(d) for d in c[0]], lag_ratio=0.02) for c in clusters], lag_ratio=0.15), run_time=2.5)
        self.play(LaggedStart(*[FadeIn(c[1]) for c in clusters], lag_ratio=0.1))
        self.page()

        # 12 — Images per class: GBIF, then four more sources
        self.play(FadeIn(title("Images per class")))
        counts = D.gbif_counts()
        chart_w, chart_h = 12.4, 4.6
        base_y, left = -3.1, -6.0
        top = np.log10(50000)  # the all-source totals reach ~31k (squirrel family)

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
        p1 = pill("114 classes  < 100 images", 24).move_to([-2.0, 1.5, 0])
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
        p3 = pill("29 classes  still < 100", 24, fill=GREEN, bold=True).next_to(src, DOWN, buff=0.2).align_to(src, LEFT)
        self.play(FadeOut(p1), FadeOut(p2), FadeIn(src))
        self.play(Transform(bars, new), run_time=1.8)
        self.play(FadeIn(p3, scale=1.1))
        self.page()

        # 13 — Not every "animal" is an animal; the filtering pipeline
        rej = collage(D.reject_paths(), cols=4, cell_w=2.85).move_to(DOWN * 0.3)
        self.play(FadeIn(title("Is this an animal?")))
        self.play(LaggedStart(*[FadeIn(t, scale=0.9) for t in rej], lag_ratio=0.12), run_time=1.6)
        self.next_slide()
        stages = ["Metadata", "Heuristics", "MegaDetector", "VLM check", "SpeciesNet"]
        chips = VGroup(*[pill(s, 22) for s in stages]).arrange(RIGHT, buff=0.55).move_to(DOWN * 3.15)
        arrs = VGroup(*[Arrow(a.get_right(), b.get_left(), buff=0.06, stroke_width=3, color=WHITE, max_tip_length_to_length_ratio=0.35) for a, b in zip(chips, chips[1:])])
        self.play(rej.animate.scale(0.82).shift(UP * 0.45))
        self.play(LaggedStart(*[FadeIn(c, shift=RIGHT * 0.2) for c in chips], lag_ratio=0.2), Create(arrs), run_time=1.5)
        self.next_slide()
        hl = pill("SpeciesNet", 22, fill=GREEN, bold=True).move_to(chips[-1])
        note = pill("the labels come from SpeciesNet", 22, fill=GREEN).next_to(chips[-1], UP, buff=0.3).align_to(chips[-1], RIGHT)
        self.play(Transform(chips[-1], hl), FadeIn(note, shift=UP * 0.2))
        self.next_slide()


# =================================================================================
# Act 3: Faking the missing data, and which generator to trust (results block 1)
# =================================================================================
class Generators(TalkSlide):
    def build(self):
        # 14 — Binturong: real vs four generators
        cols = [
            ("Real", D.BINTURONG_REAL),
            ("Gemini Flash", [D.BINTURONG / f"binturong_gemini_000{i}.png" for i in (1, 4, 7)]),
            ("RealVisXL", [D.BINTURONG / f"binturong_realvisxl-lightning_000{i}.png" for i in (1, 3, 5)]),
            ("SD 3.5 Medium", [D.BINTURONG / f"binturong_sd35m_000{i}.png" for i in (1, 2, 3)]),
            ("FLUX.1 schnell", [D.BINTURONG / f"binturong_flux-schnell_{i}.png" for i in ("0001", "0004", "0014")]),
        ]
        columns = Group()
        for name, paths in cols:
            col = Group(*[photo(p, width=1.95, aspect=4 / 3) for p in paths]).arrange(DOWN, buff=0.1)
            lab = pill(name, 20, fill=SKY if name == "Real" else WHITE)
            columns.add(Group(col, lab.next_to(col, DOWN, buff=0.2)))
        columns.arrange(RIGHT, buff=0.4).move_to(DOWN * 0.35)
        self.play(FadeIn(columns[0]))
        self.next_slide()
        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.2) for c in columns[1:]], lag_ratio=0.25), run_time=1.6)
        self.next_slide()
        ring = SurroundingRectangle(columns[1][0], buff=0.08, corner_radius=0.15).set_stroke(GREEN, 6)
        why = pill("looked best", 20, fill=GREEN, bold=True).move_to(columns[1][1])
        self.play(Create(ring), Transform(columns[1][1], why))
        self.page()

        # 15 — Three zebras: the difference is in the stripes
        zs = Group()
        for name, p in D.ZEBRAS_GEMINI.items():
            zs.add(labelled(photo(p, width=4.1, aspect=4 / 3), name, DL, 22))
        zs.arrange(RIGHT, buff=0.3).move_to(DOWN * 0.2)
        self.play(FadeIn(title("Three zebras")), LaggedStart(*[FadeIn(z, shift=UP * 0.2) for z in zs], lag_ratio=0.3))
        self.page()

        # 16 — The prompt
        cardm = prompt_card(6.6).move_to([-2.4, -0.35, 0])
        self.play(FadeIn(cardm, shift=UP * 0.4))
        self.next_slide()
        keys = ["narrow vertical stripes", "white belly", "large rounded ears", "brown muzzle", "pose · habitat · light · distance"]
        kp = VGroup(*[pill(k, 22, fill=GREEN if i < 4 else WHITE) for i, k in enumerate(keys)]).arrange(DOWN, buff=0.22, aligned_edge=LEFT)
        kp.move_to([4.4, 0.3, 0])
        tokens = pill(f"≈ {D.FULL_PROMPT_TOKENS:,} tokens", 26, bold=True).next_to(kp, DOWN, buff=0.6).align_to(kp, LEFT)
        self.play(LaggedStart(*[FadeIn(k, shift=LEFT * 0.3) for k in kp], lag_ratio=0.2))
        self.play(FadeIn(tokens))
        self.page()

        # 17 — Thousands of synthetic zebras
        sheet = ImageMobject(str(mosaic(D.GREVY_SYNTH[:96], cols=12))).scale_to_fit_width(FRAME_W - 0.6).move_to(DOWN * 0.25)
        n = pill("23,850 synthetic images", 44, bold=True).move_to(sheet)
        self.play(FadeIn(sheet, scale=1.05), run_time=1.2)
        self.play(FadeIn(n, scale=1.2))
        self.page()

        # 18 — R1 lineup: same prompt, eight sources
        tiles_ = Group()
        for name, p in D.LINEUP.items():
            m = photo(p, width=3.05, aspect=4 / 3)
            tiles_.add(labelled(m, name, DL, 18) if name != "Real photo" else labelled(m, name, DL, 18))
        tiles_.arrange_in_grid(rows=2, cols=4, buff=0.2).move_to(DOWN * 0.3)
        head = title("Same prompt, eight sources")
        self.play(FadeIn(head), FadeIn(tiles_[0]))
        self.play(LaggedStart(*[FadeIn(t, shift=UP * 0.15) for t in tiles_[1:]], lag_ratio=0.15), run_time=1.8)
        exp = pill("12 setups · 1,200 images each · same detector · scored on 9,742 real photos", 20)
        exp.to_edge(DOWN, buff=0.25)
        self.next_slide()
        self.play(FadeIn(exp, shift=UP * 0.2))
        self.page()

        # 19 — R2 ranking
        self.play(FadeIn(title("Which generator helps the detector?")))
        rows = Group()
        vmax, L = 0.15, 5.6
        for name, mean, std, kind in D.GENERATORS:
            color = {"api": SKY, "pick": GREEN, "local": MIST}[kind]
            lab = txt(name, 20, WHITE if kind != "local" else MIST)
            bar = hbar(mean, vmax, L, h=0.34, color=color, opacity=1 if kind != "local" else 0.7)
            row = VGroup(lab, bar)
            rows.add(row)
        rows.arrange(DOWN, buff=0.14).move_to([0, -0.45, 0])
        bars_x = -3.3
        for row, (name, mean, std, kind) in zip(rows, D.GENERATORS):
            row[0].next_to([bars_x - 0.25, row.get_y(), 0], LEFT, buff=0)
            row[1].move_to([bars_x, row.get_y(), 0], aligned_edge=LEFT)
            end = bars_x + L * mean / vmax
            err = Line([bars_x + L * (mean - std) / vmax, row.get_y(), 0], [bars_x + L * (mean + std) / vmax, row.get_y(), 0]).set_stroke(WHITE, 2, 0.6)
            val = txt(f"{mean:.3f}", 18, MIST).move_to([bars_x + L + 0.55, row.get_y(), 0])
            row.add(err, val)
        legend = VGroup(pill("API", 18, fill=SKY), pill("local", 18, fill=MIST)).arrange(RIGHT, buff=0.2).to_corner(UR, buff=0.4).shift(DOWN * 0.55)
        self.play(FadeIn(legend), LaggedStart(*[FadeIn(r[0]) for r in rows], lag_ratio=0.05))
        self.play(LaggedStart(*[GrowFromEdge(r[1], LEFT) for r in rows], lag_ratio=0.08), run_time=1.6)
        self.play(*[FadeIn(r[2]) for r in rows], *[FadeIn(r[3]) for r in rows])
        self.next_slide()
        pick = pill("my pick: #3", 20, fill=GREEN, bold=True).next_to(rows[2][3], RIGHT, buff=0.25)
        self.play(FadeIn(pick, shift=LEFT * 0.2))
        self.next_slide()
        cheap = pill("#1 = cheapest setting", 20, bold=True).next_to(rows[0][3], RIGHT, buff=0.25)
        self.play(FadeIn(cheap, shift=LEFT * 0.2))
        self.page()

        # 20 — R3 prompt length
        self.play(FadeIn(title("Shorter prompt, worse data")))
        full = VGroup(card(2.2, 5.0), text_lines(36, 1.7)).move_to([-5.3, -0.5, 0])
        full[1].move_to(full[0])
        short = VGroup(card(2.2, 0.75), text_lines(3, 1.7)).move_to([-2.8, 1.62, 0])
        short[1].move_to(short[0])
        ft = txt(f"≈ {D.FULL_PROMPT_TOKENS:,} tokens", 18, MIST).next_to(full, DOWN, buff=0.15)
        st = txt(f"≈ {D.COMPRESSED_PROMPT_TOKENS} tokens", 18, MIST).next_to(short, DOWN, buff=0.15)
        self.play(FadeIn(full), FadeIn(ft), FadeIn(short), FadeIn(st))
        groups = VGroup()
        L2, vmax2, x0 = 4.0, 0.15, 0.3
        for j, (name, f, c) in enumerate(D.PROMPT_ABLATION):
            y = 1.1 - j * 2.55
            lab = txt(name, 24, bold=True).move_to([x0, y + 0.6, 0], aligned_edge=LEFT)
            bf = hbar(f, vmax2, L2, h=0.42, color=SKY).move_to([x0, y, 0], aligned_edge=LEFT)
            bc = hbar(c, vmax2, L2, h=0.42, color=MIST, opacity=0.75).move_to([x0, y - 0.6, 0], aligned_edge=LEFT)
            vf = txt(f"{f:.3f}  full", 18, MIST).next_to(bf, RIGHT, buff=0.2)
            vc = txt(f"{c:.3f}  short", 18, MIST).next_to(bc, RIGHT, buff=0.2)
            loss = pill(f"−{round((1 - c / f) * 100)} %", 26, fill=GREEN, bold=True).move_to([x0 + L2 + 2.05, y - 0.3, 0])
            groups.add(VGroup(lab, bf, vf, bc, vc, loss))
        self.play(*[FadeIn(g[0]) for g in groups], *[GrowFromEdge(g[1], LEFT) for g in groups], *[FadeIn(g[2]) for g in groups])
        self.next_slide()
        self.play(*[GrowFromEdge(g[3], LEFT) for g in groups], *[FadeIn(g[4]) for g in groups])
        self.play(*[FadeIn(g[5], scale=1.2) for g in groups])
        self.page()

        # 21 — One seed can lie
        self.play(FadeIn(title("One run can lie")))
        lo, hi, x_l, x_r, y = 0.05, 0.12, -5.5, 5.5, -0.6

        def x_of(v):
            return x_l + (v - lo) / (hi - lo) * (x_r - x_l)

        axis = Line([x_l, y, 0], [x_r, y, 0]).set_stroke(MIST, 2, 0.6)
        ticks = VGroup()
        for v in (0.06, 0.08, 0.10, 0.12):
            ticks.add(Line([x_of(v), y - 0.08, 0], [x_of(v), y + 0.08, 0]).set_stroke(MIST, 2, 0.6), axis_label(f"{v:.2f}").move_to([x_of(v), y - 0.4, 0]))
        sib = DashedLine([x_of(D.SEED_SIBLING), y - 0.1, 0], [x_of(D.SEED_SIBLING), y + 2.4, 0]).set_stroke(MIST, 3)
        sib_l = pill("cheaper sibling", 20).next_to(sib, UP, buff=0.12)
        dot = Dot(radius=0.2, color=GREEN).move_to([x_of(D.SEED_STEPS[0]), y + 0.9, 0])
        lab = pill("my pick · 1 run", 22, fill=GREEN, bold=True).next_to(dot, DOWN, buff=0.25)
        self.play(Create(axis), FadeIn(ticks), Create(sib), FadeIn(sib_l))
        self.play(GrowFromCenter(dot), FadeIn(lab))
        for k, v in enumerate(D.SEED_STEPS[1:], start=2):
            self.next_slide()
            new_lab = pill(f"my pick · {k} runs", 22, fill=GREEN, bold=True).next_to([x_of(v), y + 0.9, 0], DOWN, buff=0.45)
            self.play(dot.animate.move_to([x_of(v), y + 0.9, 0]), Transform(lab, new_lab), run_time=1.2)
        self.page()

        # 22 — R4 the blind spot for rare animals
        classes, cols_, grid = D.generator_heatmap()
        cw, chh = 0.46, 0.36
        hm = VGroup()
        x0, y0 = -4.3, 0.85
        for i, row in enumerate(grid):
            for j, v in enumerate(row):
                t = min(1.0, np.sqrt(v / 0.33))
                cell = Rectangle(width=cw - 0.04, height=chh - 0.04).set_stroke(width=0)
                cell.set_fill(GREEN, 0.06 + 0.94 * t).move_to([x0 + j * cw, y0 - i * chh, 0])
                hm.add(cell)
        names = VGroup(*[txt(c.replace(" family", ""), 15, MIST).move_to([x0 - 0.35, y0 - i * chh, 0], aligned_edge=RIGHT) for i, c in enumerate(classes)])
        heads = VGroup(*[txt(c, 13, MIST).rotate(np.pi / 3).move_to([x0 + j * cw, y0 + 0.6, 0], aligned_edge=DL).shift(LEFT * 0.1) for j, c in enumerate(cols_)])
        n_rare = sum(1 for c in classes if c in {"aye-aye", "kinkajou", "pangolin family", "ringtail", "saiga", "water deer"})
        rare_box = SurroundingRectangle(VGroup(*hm[: n_rare * len(cols_)]), buff=0.06, corner_radius=0.08).set_stroke(GREEN, 4)
        rare_l = pill("rare animals: AP < 0.07 for every generator", 20, fill=GREEN, bold=True).move_to([0, -3.55, 0]).align_to(names, LEFT)
        self.play(FadeIn(title("Rare animals are a blind spot")))
        self.play(FadeIn(names), FadeIn(heads), LaggedStart(*[FadeIn(c) for c in hm], lag_ratio=0.004), run_time=1.5)
        self.next_slide()
        self.play(Create(rare_box), FadeIn(rare_l))
        self.next_slide()
        real = labelled(photo(D.SAIGA_REAL, width=3.2, aspect=4 / 3), "real saiga", DL, 18).move_to([4.9, 1.0, 0])
        fakes = Group(*[photo(p, width=1.15, aspect=4 / 3) for p in D.SAIGA_SYNTH.values()]).arrange(RIGHT, buff=0.1)
        fakes.next_to(real, DOWN, buff=0.35)
        fl = pill("generated (local models)", 18).next_to(fakes, DOWN, buff=0.15)
        nose = pill("no trunk-like nose", 20, fill=GREEN, bold=True).next_to(fl, DOWN, buff=0.15)
        self.play(FadeIn(real))
        self.play(LaggedStart(*[FadeIn(f, shift=UP * 0.2) for f in fakes], lag_ratio=0.2), FadeIn(fl))
        self.play(FadeIn(nose))
        self.next_slide()


# =================================================================================
# Act 4: Bands: what can synthetic images do? (results block 2)
# =================================================================================
class Bands(TalkSlide):
    def build(self):
        # 23 — R5 the experiment
        self.play(FadeIn(title("Same budget, different mix")))
        comps = {"A": (20, 0, 5), "B": (10, 10, 5), "C": (0, 20, 5), "D": (0, 150, 10)}
        cols = VGroup()
        for b, (ns, nr, per) in comps.items():
            t = tiles(ns, nr, per)
            name = txt(b, 44, bold=True)
            sub = txt(f"{D.BAND_CLASSES[b]} classes", 18, MIST)
            lab = VGroup(name, sub).arrange(DOWN, buff=0.05)
            cols.add(VGroup(t, lab))
        cols.arrange(RIGHT, buff=1.1, aligned_edge=DOWN).move_to(DOWN * 0.5)
        for c in cols:
            c[1].next_to(c[0], DOWN, buff=0.3)
        bottom = min(c[0].get_bottom()[1] for c in cols)
        for c in cols:  # align tiles on a common baseline
            dy = bottom - c[0].get_bottom()[1]
            c.shift(UP * dy)
        legend = VGroup(
            VGroup(tile("synth"), txt("synthetic", 20)).arrange(RIGHT, buff=0.15),
            VGroup(tile("real"), txt("real", 20)).arrange(RIGHT, buff=0.15),
            txt("1 square = 10 images", 18, MIST),
        ).arrange(RIGHT, buff=0.5).to_corner(UR, buff=0.4).shift(DOWN * 0.5)
        self.play(FadeIn(legend))
        self.play(LaggedStart(*[LaggedStart(*[FadeIn(s) for s in c[0]], lag_ratio=0.01) for c in cols], lag_ratio=0.3), *[FadeIn(c[1]) for c in cols], run_time=2.2)
        self.next_slide()
        abc = VGroup(*[c[0] for c in cols[:3]])
        brace = Line(abc.get_corner(UL) + UP * 0.35, abc.get_corner(UR) + UP * 0.35).set_stroke(GREEN, 4)
        bl = pill("200 images each", 22, fill=GREEN, bold=True).next_to(brace, UP, buff=0.15)
        self.play(Create(brace), FadeIn(bl))
        self.page()

        # 24 — The zero: a bug before a result
        cap = txt("Band A, first trained models", 26, MIST).move_to(UP * 1.5)
        val = ValueTracker(D.FIRST_RUN_BAND_A)
        num = always_redraw(lambda: txt(f"{val.get_value():.3f}", 150, bold=True).move_to(DOWN * 0.1))
        self.play(FadeIn(cap), FadeIn(num))
        self.next_slide()
        bug = pill("the data loader never read the synthetic images", 26, fill=GREEN, bold=True).move_to(DOWN * 2.2)
        self.play(FadeIn(bug, shift=UP * 0.2))
        self.next_slide()
        cap2 = txt("Band A after the fix  (YOLO26n, real test)", 26, MIST).move_to(cap)
        self.play(FadeOut(bug), Transform(cap, cap2))
        self.play(val.animate.set_value(D.BAND_REAL["A"]), run_time=1.6, rate_func=smooth)
        self.wait(0.5)
        self.page()

        # 25 — R6 half fake is fine, all fake is not
        self.play(FadeIn(title("Half synthetic is fine. All synthetic is not.")))
        comps_small = {"A": (20, 0), "B": (10, 10), "C": (0, 20), "D": (0, 20)}
        L, base = 4.1, -2.3
        xs = {"A": -4.5, "B": -1.5, "C": 1.5, "D": 4.5}
        group = {}
        for b, v in D.BAND_REAL.items():
            color = GREEN if b in "BC" else MIST
            bar = vbar(v, 0.8, L, w=1.5, color=color, opacity=1 if b != "D" else 0.3)
            bar.move_to([xs[b], base, 0], aligned_edge=DOWN)
            if b == "D":
                bar.set_stroke(MIST, 2, 0.8)
            val_ = txt(f"{v:.2f}", 30, bold=True).next_to(bar, UP, buff=0.15)
            ns, nr = comps_small[b]
            strip = tiles(ns, nr, 10, s=0.12, gap=0.03).next_to(bar, DOWN, buff=0.25)
            lab = txt(b, 30, bold=True).next_to(strip, DOWN, buff=0.15)
            group[b] = VGroup(bar, val_, strip, lab)
        ref = txt("reference:\nup to 1,500 real", 16, MIST).next_to(group["D"][0], RIGHT, buff=0.15)
        self.play(*[FadeIn(g[2]) for g in group.values()], *[FadeIn(g[3]) for g in group.values()])
        self.play(LaggedStart(*[GrowFromEdge(group[b][0], DOWN) for b in "ABCD"], lag_ratio=0.25), run_time=1.8)
        self.play(*[FadeIn(g[1]) for g in group.values()], FadeIn(ref))
        self.next_slide()
        eq = pill("=", 36, fill=GREEN, bold=True).move_to([(xs["B"] + xs["C"]) / 2, group["B"][1].get_y(), 0])
        half = pill("½", 36, bold=True).next_to(group["A"][1], UP, buff=0.2)
        self.play(FadeIn(eq, scale=1.3))
        self.play(FadeIn(half, scale=1.3))
        self.page()

        # 26 — R7 not every animal is the same
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
        hy = labelled(photo(D.HYAENA_SYNTH, width=2.9, aspect=4 / 3), "striped hyaena · 0.48", DL, 17).move_to([4.6, 1.2, 0])
        mg = labelled(photo(D.mangabey_synth(), width=2.9, aspect=4 / 3), "mangabeys · 0.01", DL, 17).move_to([4.6, -1.6, 0])
        note = txt("both trained on synthetic images only", 18, MIST).next_to(mg, DOWN, buff=0.2)
        l1 = Line(special["striped hyaena"].get_center(), hy.get_left(), buff=0.12).set_stroke(GREEN, 2)
        l2 = Line(special["mangabeys genus"].get_center(), mg.get_left(), buff=0.12).set_stroke(GREEN, 2)
        self.play(FadeIn(hy), Create(l1))
        self.play(FadeIn(mg), Create(l2), FadeIn(note))
        self.page()

        # 27 — R8 the synthetic test set was too easy
        syn = labelled(photo(D.SYNTH_TEST_ZEBRA, height=3.6, aspect=4 / 3), "synthetic test image", DL, 20)
        real = labelled(photo(D.REAL_HERD, height=3.6, aspect=4 / 3), "real test image", DL, 20)
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
        self.play(LaggedStart(*[AnimationPair for g in bars for AnimationPair in (GrowFromEdge(g[0], DOWN), GrowFromEdge(g[1], DOWN))], lag_ratio=0.1), run_time=1.6)
        self.play(*[FadeIn(g[2]) for g in bars], *[FadeIn(g[3]) for g in bars])
        self.next_slide()
        rule = pill("from here on: real photos only", 24, fill=GREEN, bold=True).to_edge(DOWN, buff=0.2).shift(RIGHT * 2.6)
        self.play(FadeIn(rule, shift=UP * 0.2))
        self.next_slide()


# =================================================================================
# Act 5: Which model? (results block 3, with the embedded side note)
# =================================================================================
class Models(TalkSlide):
    def build(self):
        # 28 — R9 contenders by size
        self.play(FadeIn(title("Three candidates")))
        R = 2.45
        specs = [("MegaDetector + SpeciesNet", 193, "193 M", -2.6, BLUE), ("YOLOv5s", 7.63, "7.6 M", 2.7, MIST), ("YOLO26n", 2.71, "2.7 M", 5.0, GREEN)]
        circles = VGroup()
        for name, p, label, x, color in specs:
            r = R * np.sqrt(p / 193)
            c = Circle(radius=r).set_fill(color, 0.45 if color == BLUE else 0.9).set_stroke(MIST if color == BLUE else color, 2)
            c.move_to([x, -0.6 + (r - R) * 0, 0]).align_to([0, -3.0, 0], DOWN)
            n = txt(label, 30 if p > 100 else 24, bold=True)
            n.move_to(c) if p > 100 else n.next_to(c, UP, buff=0.15)
            lab = pill(name, 20).next_to(c, DOWN, buff=0.2)
            if p < 100:
                lab.move_to([x, -3.0 - 0.4, 0])
            circles.add(VGroup(c, n, lab))
        circles[0][2].move_to([-2.6, -3.4, 0])
        self.play(LaggedStart(*[GrowFromCenter(c[0]) for c in circles], lag_ratio=0.4), run_time=1.8)
        self.play(*[FadeIn(c[1]) for c in circles], *[FadeIn(c[2]) for c in circles])
        self.next_slide()
        tag = pill("in the binocular today", 18).next_to(circles[1][1], UP, buff=0.15)
        self.play(FadeIn(tag))
        self.page()

        # 29 — R10 accuracy, size and time per frame in one picture
        self.play(FadeIn(title("Accuracy on real photos")))
        L, vmax, x0 = 6.0, 0.7, -2.4
        rows = VGroup()
        for k, (name, m, p, t) in enumerate(D.MODELS):
            y = 1.3 - k * 1.15
            color = GREEN if name == "YOLO26n" else MIST
            lab = txt(name, 24, bold=name == "YOLO26n").move_to([x0 - 0.3, y, 0], aligned_edge=RIGHT)
            bar = hbar(m, vmax, L, h=0.55, color=color, opacity=1 if color == GREEN else 0.75).move_to([x0, y, 0], aligned_edge=LEFT)
            val = txt(f"{m:.2f}", 26, bold=True).next_to(bar, RIGHT, buff=0.2)
            rows.add(VGroup(lab, bar, val))
        self.play(LaggedStart(*[FadeIn(r[0]) for r in rows], lag_ratio=0.1))
        self.play(LaggedStart(*[GrowFromEdge(r[1], LEFT) for r in rows], lag_ratio=0.15), run_time=1.5)
        self.play(*[FadeIn(r[2]) for r in rows])
        star = txt("* upper bound: MegaDetector drew the ground-truth boxes", 16, MIST).to_corner(DL, buff=0.4)
        rows[0][2].become(txt(f"{D.MODELS[0][1]:.2f}*", 26, bold=True).move_to(rows[0][2], aligned_edge=LEFT))
        self.play(FadeIn(star))
        self.next_slide()
        side = VGroup()
        for r, (name, m, p, t) in zip(rows, D.MODELS):
            sec = f"{t:.0f} s" if t >= 10 else f"{t:.2f} s"
            sp = pill(f"{p:g} M · {sec}", 20, fill=GREEN if name == "YOLO26n" else WHITE).move_to([5.6, r.get_y(), 0])
            side.add(sp)
        watch = stopwatch(0.32).next_to(side, UP, buff=0.25)
        self.play(FadeIn(watch), LaggedStart(*[FadeIn(s, shift=LEFT * 0.2) for s in side], lag_ratio=0.15))
        foot = txt("time per frame: Raspberry Pi 400 CPU, full precision, no quantization", 16, MIST).next_to(star, UP, buff=0.08).align_to(star, LEFT)
        self.play(FadeIn(foot))
        self.next_slide()
        call = pill("83× faster · 71× smaller · −0.07 mAP", 28, fill=GREEN, bold=True).move_to([1.0, -3.0, 0])
        self.play(FadeOut(star), FadeOut(foot), FadeIn(call, scale=1.1))
        self.page()

        # 30 — R11 finding vs naming
        self.play(FadeIn(title("Finding is easy. Naming is hard.")))
        a = big_number(f"{D.DETECT_REAL:.2f}", "finds the animal", size=110)
        b = big_number(f"{D.FINE_REAL:.2f}", "names the species", size=110)
        ab = VGroup(a, b).arrange(RIGHT, buff=2.2).move_to(DOWN * 0.2)
        ar = Arrow(a[0].get_right(), b[0].get_left(), buff=0.35, stroke_width=6, color=GREEN)
        self.play(FadeIn(a, shift=UP * 0.2))
        self.play(Create(ar), FadeIn(b, shift=UP * 0.2))
        self.next_slide()
        self.play(VGroup(ab, ar).animate.scale(0.4).to_corner(UR, buff=0.4).shift(DOWN * 0.15))
        asian = labelled(photo(D.ASIAN_ELEPHANT, height=3.3, aspect=1.0), "Asian elephant", DL, 20).move_to([-3.0, -1.0, 0])
        african = labelled(photo(D.AFRICAN_ELEPHANT, height=3.3, aspect=1.0), "African elephant", DR, 20).move_to([4.0, -1.0, 0])
        fwd = CurvedArrow(asian.get_top() + RIGHT * 0.6 + UP * 0.1, african.get_top() + LEFT * 0.6 + UP * 0.1, angle=-0.9, color=GREEN, stroke_width=14)
        rev = CurvedArrow(african.get_bottom() + LEFT * 0.6 + DOWN * 0.1, asian.get_bottom() + RIGHT * 0.6 + DOWN * 0.1, angle=-0.6, color=MIST, stroke_width=2)
        fl = pill(f"{D.ELEPHANT_FWD:,}×", 30, fill=GREEN, bold=True).next_to(fwd, UP, buff=0.05)
        rl = pill(f"{D.ELEPHANT_REV}×", 22).next_to(rev, DOWN, buff=0.02)
        self.play(FadeIn(asian), FadeIn(african))
        self.play(Create(fwd), FadeIn(fl))
        self.play(Create(rev), FadeIn(rl))
        self.page()

        # 31 — R12 distillation
        self.play(FadeIn(title("Did the teacher help?")))
        L, base = 4.6, -2.6
        bars = VGroup()
        for k, b in enumerate("ABCD"):
            x = -3.9 + k * 2.6
            kd, direct = D.KD_BANDS[b]
            s = vbar(kd, 0.8, L, w=0.9, color=MIST, opacity=0.6).move_to([x - 0.5, base, 0], aligned_edge=DOWN)
            r = vbar(direct, 0.8, L, w=0.9, color=GREEN).move_to([x + 0.5, base, 0], aligned_edge=DOWN)
            sv = txt(f"{kd:.2f}", 20, MIST).next_to(s, UP, buff=0.1)
            rv = txt(f"{direct:.2f}", 20, bold=True).next_to(r, UP, buff=0.1)
            lab = txt(b, 30, bold=True).move_to([x, base - 0.45, 0])
            bars.add(VGroup(s, r, sv, rv, lab))
        leg = VGroup(pill("with teacher (distilled)", 20, fill=MIST), pill("direct training", 20, fill=GREEN)).arrange(RIGHT, buff=0.25)
        leg.to_corner(UR, buff=0.4).shift(DOWN * 0.55)
        self.play(FadeIn(leg), *[FadeIn(g[4]) for g in bars])
        self.play(LaggedStart(*[GrowFromEdge(g[0], DOWN) for g in bars], lag_ratio=0.1), run_time=1.2)
        self.play(*[FadeIn(g[2]) for g in bars])
        self.next_slide()
        self.play(LaggedStart(*[GrowFromEdge(g[1], DOWN) for g in bars], lag_ratio=0.1), run_time=1.2)
        self.play(*[FadeIn(g[3]) for g in bars])
        self.next_slide()
        verdict = pill("direct training wins in every band", 26, fill=GREEN, bold=True).move_to(UP * 2.0)
        self.play(FadeIn(verdict, shift=DOWN * 0.2))
        self.next_slide()


# =================================================================================
# Act 6: Close
# =================================================================================
class Close(TalkSlide):
    def build(self):
        # 32 — Plan vs reality
        self.play(FadeIn(title("Four questions, four answers")))
        qs = four_questions()
        for g in qs:
            g.shift(UP * 0.45)
        self.play(*[FadeIn(g) for g in qs])
        verdicts = [
            (cross_mark(), "no gain"),
            (check_mark(), "prompt > model"),
            (tilde_mark(), "half, not all"),
            (tilde_mark(), "not yet real-time"),
        ]
        for g, (mark, s) in zip(qs, verdicts):
            self.next_slide()
            mark.next_to(g, DOWN, buff=0.35)
            p = pill(s, 22, bold=True).next_to(mark, DOWN, buff=0.25)
            self.play(Create(mark), FadeIn(p, shift=UP * 0.15), run_time=0.8)
        self.page()

        # 33 — Which one is real?
        real = photo(D.ZEBRA_REAL_FINAL, height=4.6, aspect=4 / 3)
        fake = photo(D.ZEBRAS_GEMINI["Plains zebra"], height=4.6, aspect=4 / 3)
        pair = Group(fake, real).arrange(RIGHT, buff=0.4).move_to(DOWN * 0.3)
        self.play(FadeIn(title("Which one is real?")), FadeIn(pair))
        self.next_slide()
        lf = pill("generated", 24, fill=GREEN, bold=True).next_to(fake, DOWN, buff=0.2)
        lr = pill("real", 24, bold=True).next_to(real, DOWN, buff=0.2)
        self.play(FadeIn(lf, shift=UP * 0.2), FadeIn(lr, shift=UP * 0.2))
        self.page()

        # 34 — Thank you
        thanks = txt("Thank you", 80, bold=True).move_to(UP * 0.5)
        qs_ = pill("Questions?", 34, fill=GREEN, bold=True).next_to(thanks, DOWN, buff=0.5)
        self.play(FadeIn(thanks, shift=UP * 0.2))
        self.play(FadeIn(qs_, scale=1.1))
        self.next_slide()
