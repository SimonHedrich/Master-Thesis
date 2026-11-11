# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.9"]
# ///
"""Static charts for the blog article, drawn for an 800 px WordPress content column.

Writes 1600 px wide PNGs (2x for retina) to thesis/blog/plots/out/:
  ranking.png          generator ranking with the two short-prompt cells (Figure 4)
  ranking_dark.png     the same chart in the talk's dark look, to compare the two styles
  bands.png            real-photo mAP per training band (poster frame of Clip 2)
  synthetic_vs_real.png  generated vs real test set per band (Figure 7, chart half)
  latency_vs_map.png   accuracy against time per frame (Figure 8)

Every number is copied from thesis/manuscript/chapters/4-Results.tex (table labels in the
comments); the slides' data.py holds the same values but imports manim, so it is not
imported here. The tables hold mAP as a fraction (0.588); the charts, like the article,
show it in percent (58.8 %) and differences in percentage points.

PEP 723 script rather than the repo-root uv project: that project is the Linux/CUDA
training environment, and these charts are drawn natively on the Mac next to the slides.

Run (from the repo root):
    uv run --script thesis/blog/plots/make_plots.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Rectangle

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
FONTS = HERE.parents[1] / "presentation" / "slides" / "assets" / "fonts"

for f in FONTS.glob("*.ttf"):
    font_manager.fontManager.addfont(str(f))

# One colour code for the whole article: green = generated, navy = real photos.
LIGHT = dict(
    bg="#FFFFFF", ink="#061B59", muted="#5B6680", grid="#E3E7EF",
    api="#2C5DFF", local="#8A94A8", generated="#17A34A", real="#061B59",
    synth_test="#E0A100", real_test="#D6348A", highlight="#17A34A",
)
DARK = dict(
    bg="#0B2A8F", ink="#FFFFFF", muted="#C5CEE6", grid="#2A49B5",
    api="#0AAFE8", local="#DFE5F0", generated="#7DF381", real="#FFFFFF",
    synth_test="#FFD24D", real_test="#FF7AC6", highlight="#7DF381",
)

W_IN, DPI = 8.0, 200  # 1600 px wide
BASE = 13


def setup(t: dict) -> None:
    plt.rcParams.update({
        "font.family": "Atkinson Hyperlegible", "font.size": BASE,
        "figure.facecolor": t["bg"], "axes.facecolor": t["bg"], "savefig.facecolor": t["bg"],
        "text.color": t["ink"], "axes.labelcolor": t["muted"],
        "xtick.color": t["muted"], "ytick.color": t["muted"],
        "axes.edgecolor": t["grid"], "axes.spines.top": False, "axes.spines.right": False,
    })


def title(fig, t: dict, head: str, sub: str) -> None:
    fig.text(0.04, 0.955, head, fontsize=BASE + 6, fontweight="bold", va="top", color=t["ink"])
    fig.text(0.04, 0.955 - 0.42 / fig.get_figheight(), sub, fontsize=BASE, va="top", color=t["muted"])


def pct(v: float, decimals: int = 1) -> str:
    """mAP fraction as the article writes it: 0.588 -> '58.8 %'."""
    return f"{v * 100:.{decimals}f} %"


def save(fig, name: str) -> None:
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / name, dpi=DPI)
    plt.close(fig)
    print("wrote", OUT / name)


# --- Figure 4: generator ranking (tab:generator-ranking, data.GENERATOR_PROMPT) -----------
# (label, mAP, std, tier, prompt tokens); tier "short" = API model with the <= 75-token prompt
RANKING = [
    ("gpt-image-2 (low)", 0.134, 0.014, "api", "~5,900"),
    ("gpt-image-2 (medium)", 0.126, 0.011, "api", "~5,900"),
    ("Gemini 3.1 Flash Image", 0.105, 0.036, "api", "~5,900"),
    ("Gemini 3.1 Flash-Lite", 0.095, 0.006, "api", "~5,900"),
    ("gpt-image-2 (low), short prompt", 0.074, 0.002, "short", "≤ 75"),
    ("SD 3.5 Large", 0.055, 0.002, "local", "≤ 256"),
    ("SD 3.5 Medium", 0.054, 0.001, "local", "≤ 256"),
    ("Gemini Flash-Lite, short prompt", 0.049, 0.003, "short", "≤ 75"),
    ("SD 3.5 Large Turbo", 0.043, 0.005, "local", "≤ 256"),
    ("FLUX.2 klein", 0.037, 0.001, "local", "≤ 512"),
    ("HiDream-I1", 0.029, 0.002, "local", "≤ 512"),
    ("RealVisXL Lightning", 0.023, 0.001, "local", "≤ 75"),
]


def ranking(t: dict, name: str) -> None:
    setup(t)
    fig = plt.figure(figsize=(W_IN, 7.0))
    ax = fig.add_axes((0.36, 0.09, 0.40, 0.70))
    title(fig, t, "Which generator teaches a detector best?",
          "mAP in percent on 9,742 real photos after training on 1,200 generated images")
    n = len(RANKING)
    for i, (label, m, sd, tier, tok) in enumerate(RANKING):
        y = n - 1 - i
        colour = t["local"] if tier == "local" else t["api"]
        bar = ax.barh(y, m, height=0.56, color=colour, zorder=3)[0]
        if tier == "short":  # same model, less prompt: same hue, hatched
            bar.set_hatch("////")
            bar.set_edgecolor(t["bg"])
            bar.set_linewidth(0)
        ax.errorbar(m, y, xerr=sd, color=t["ink"], lw=1.2, capsize=0, zorder=4)
        ax.text(m + sd + 0.004, y, pct(m), va="center", fontsize=BASE - 1, color=t["ink"])
        ax.text(-0.004, y, label, va="center", ha="right", fontsize=BASE - 1, color=t["ink"],
                fontweight="bold" if tier == "short" else "normal")
        ax.text(0.222, y, tok, va="center", ha="right", fontsize=BASE - 1, color=t["muted"], clip_on=False)
    ax.text(0.222, n - 0.25, "prompt\ntokens", va="bottom", ha="right", fontsize=BASE - 2, color=t["muted"])
    ax.set_xlim(0, 0.16)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks([])
    ax.set_xticks([0, 0.05, 0.10, 0.15])
    ax.set_xticklabels(["0 %", "5 %", "10 %", "15 %"])
    ax.xaxis.grid(True, color=t["grid"], lw=1, zorder=0)
    ax.spines["left"].set_visible(False)
    ax.tick_params(length=0)
    # legend
    lx, ly = 0.04, 0.035
    for colour, hatch, text, dx in [(t["api"], None, "API model", 0.0), (t["api"], "////", "API model, short prompt", 0.17),
                                    (t["local"], None, "runs locally", 0.47)]:
        r = Rectangle((lx + dx, ly - 0.008), 0.03, 0.02, transform=fig.transFigure, facecolor=colour,
                      hatch=hatch, edgecolor=t["bg"], lw=0)
        fig.patches.append(r)
        fig.text(lx + dx + 0.04, ly, text, va="center", fontsize=BASE - 2, color=t["muted"])
    save(fig, name)


# --- Clip 2 poster: mAP per band on real photos (tab:band-grid-real, YOLO26n) -------------
BANDS = [  # (band, classes, real mAP, synthetic-test mAP, n generated squares, n real squares, caption)
    ("A", 50, 0.300, 0.991, 2, 0, "200 generated"),
    ("B", 26, 0.588, 0.993, 1, 1, "100 real +\n100 generated"),
    ("C", 26, 0.591, 0.933, 0, 2, "200 real"),
    ("D", 122, 0.742, 0.869, 0, 15, "up to\n1,500 real"),
]


def squares(ax, t: dict, x: float, y: float, n_gen: int, n_real: int, s: float = 0.085) -> None:
    """Training images per class as squares of 100: green generated, navy real."""
    total = n_gen + n_real
    per_row = 5 if total > 2 else 2
    x0 = x - (min(total, per_row) * s * 1.25 - s * 0.25) / 2
    for k in range(total):
        col, row = k % per_row, k // per_row
        colour = t["generated"] if k < n_gen else t["real"]
        ax.add_patch(Rectangle((x0 + col * s * 1.25, y - row * s * 0.36), s, s * 0.28,
                               facecolor=colour, lw=0, clip_on=False,
                               transform=ax.get_xaxis_transform()))


def bands(t: dict) -> None:
    setup(t)
    fig = plt.figure(figsize=(W_IN, 6.1))
    ax = fig.add_axes((0.10, 0.375, 0.86, 0.43))
    title(fig, t, "Half generated works. All generated does not.",
          "YOLO26n, mAP in percent on real test photos, by how each class was trained")
    for i, (b, n, real, _, g, r, cap) in enumerate(BANDS):
        ax.bar(i, real, width=0.5, color=t["real"], zorder=3)
        ax.text(i, real + 0.02, pct(real), ha="center", fontsize=BASE + 2, fontweight="bold")
        ax.text(i, -0.07, f"Band {b}", ha="center", va="top", fontsize=BASE + 1, fontweight="bold",
                transform=ax.get_xaxis_transform())
        ax.text(i, -0.18, f"{n} classes", ha="center", va="top", fontsize=BASE - 1, color=t["muted"],
                transform=ax.get_xaxis_transform())
        squares(ax, t, i, -0.30 if b == "D" else -0.36, g, r)
        ax.text(i, -0.50, cap, ha="center", va="top", fontsize=BASE - 1,
                color=t["ink"], transform=ax.get_xaxis_transform())
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(0, 0.85)
    ax.set_xticks([])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0 %", "20 %", "40 %", "60 %", "80 %"])
    ax.yaxis.grid(True, color=t["grid"], lw=1, zorder=0)
    ax.spines["left"].set_visible(False)
    ax.tick_params(length=0)
    fig.text(0.04, 0.03, "one square = 100 training images per class:", fontsize=BASE - 2, color=t["muted"], va="center")
    for dx, colour, text in [(0.44, t["generated"], "generated"), (0.60, t["real"], "real photo")]:
        fig.patches.append(Rectangle((dx, 0.022), 0.022, 0.016, transform=fig.transFigure, facecolor=colour, lw=0))
        fig.text(dx + 0.03, 0.03, text, fontsize=BASE - 2, color=t["muted"], va="center")
    save(fig, "bands.png")


# --- Figure 7: generated vs real test set (tab:kd-vs-direct-bands, column B) --------------
def synthetic_vs_real(t: dict) -> None:
    setup(t)
    fig = plt.figure(figsize=(W_IN, 5.6))
    ax = fig.add_axes((0.10, 0.20, 0.86, 0.56))
    title(fig, t, "The generated test set flatters every band",
          "YOLO26n, mAP in percent of the same classes on generated and on real test images")
    w = 0.3
    for i, (b, _, real, syn, *_rest) in enumerate(BANDS):
        ax.bar(i - w / 2 - 0.012, syn, width=w, color=t["synth_test"], zorder=3)
        ax.bar(i + w / 2 + 0.012, real, width=w, color=t["real_test"], zorder=3)
        ax.text(i - w / 2 - 0.012, syn + 0.02, pct(syn, 0), ha="center", fontsize=BASE)
        ax.text(i + w / 2 + 0.012, real + 0.02, pct(real, 0), ha="center", fontsize=BASE, fontweight="bold")
        ax.text(i, -0.06, f"Band {b}", ha="center", va="top", fontsize=BASE + 1, fontweight="bold",
                transform=ax.get_xaxis_transform())
        ax.text(i, -0.15, BANDS[i][6].replace("\n", " "), ha="center", va="top", fontsize=BASE - 2,
                color=t["muted"], transform=ax.get_xaxis_transform())
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(0, 1.12)
    ax.set_xticks([])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0 %", "25 %", "50 %", "75 %", "100 %"])
    ax.yaxis.grid(True, color=t["grid"], lw=1, zorder=0)
    ax.spines["left"].set_visible(False)
    ax.tick_params(length=0)
    for dx, colour, text in [(0.04, t["synth_test"], "generated test images"), (0.36, t["real_test"], "real test photos")]:
        fig.patches.append(Rectangle((dx, 0.030), 0.022, 0.018, transform=fig.transFigure, facecolor=colour, lw=0))
        fig.text(dx + 0.03, 0.039, text, fontsize=BASE - 1, color=t["muted"], va="center")
    save(fig, "synthetic_vs_real.png")


# --- Figure 8: accuracy vs time per frame (tab:headline-cross-model, tab:embedded-latency) -
MODELS = [  # (label, seconds per frame, real mAP, params M, kind, label offset in points)
    ("YOLO26n\n2.7 M parameters", 0.423, 0.529, 2.71, "ours", (12, -22)),
    ("YOLOv5s\n7.6 M", 0.949, 0.407, 7.63, "other", (14, -8)),
    ("MegaDetector + SpeciesNet,\nfine-tuned · 193 M", 35.0, 0.599, 193, "other", (-16, 14)),
    ("MegaDetector + SpeciesNet,\noff the shelf", 35.0, 0.445, 193, "other", (-16, -6)),
]


def latency_vs_map(t: dict) -> None:
    setup(t)
    fig = plt.figure(figsize=(W_IN, 5.8))
    ax = fig.add_axes((0.11, 0.15, 0.85, 0.62))
    title(fig, t, "83× faster for 7 percentage points of mAP",
          "Species mAP on real photos against time per frame on a Raspberry Pi 400 (CPU, FP32)")
    ax.set_xscale("log")
    for label, s, m, p, kind, (dx, dy) in MODELS:
        colour = t["highlight"] if kind == "ours" else t["local"]
        ax.scatter(s, m, s=140, color=colour, edgecolor=t["bg"], linewidth=2, zorder=4)
        ax.annotate(label, (s, m), xytext=(dx, dy), textcoords="offset points",
                    ha="left" if dx > 0 else "right", va="center", fontsize=BASE - 1,
                    fontweight="bold" if kind == "ours" else "normal", color=t["ink"])
    ax.annotate("", xy=(0.47, 0.537), xytext=(29, 0.597),
                arrowprops=dict(arrowstyle="->", color=t["muted"], lw=1.4, connectionstyle="arc3,rad=0.12"), zorder=2)
    ax.set_xlim(0.2, 80)
    ax.set_ylim(0.35, 0.66)
    ax.set_xticks([0.3, 1, 3, 10, 30])
    ax.set_xticklabels(["0.3 s", "1 s", "3 s", "10 s", "30 s"])
    ax.minorticks_off()
    ax.set_yticks([0.4, 0.5, 0.6])
    ax.set_yticklabels(["40 %", "50 %", "60 %"])
    ax.set_xlabel("time per frame (log scale)")
    ax.set_ylabel("mAP on real photos")
    ax.grid(True, color=t["grid"], lw=1, zorder=0)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(length=0)
    save(fig, "latency_vs_map.png")


if __name__ == "__main__":
    ranking(LIGHT, "ranking.png")
    ranking(DARK, "ranking_dark.png")
    bands(LIGHT)
    synthetic_vs_real(LIGHT)
    latency_vs_map(LIGHT)
