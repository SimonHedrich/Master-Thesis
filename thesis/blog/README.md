# Blog Article

The thesis is presented to a wider audience as an article on the
[inovex blog](https://www.inovex.de/en/blog/). It is the sequel to the bachelor-thesis article
[Enhancing Object Detection with Synthetic Data](https://www.inovex.de/de/blog/enhancing-object-detection-with-synthetic-data/)
(06/2025).

## Goals

- **Audience:** technically literate practitioners and decision makers, including potential
  inovex customers. They should come away thinking "these people know how to get computer
  vision onto real hardware with too little data", with no sales copy needed to get there.
- **Interesting first.** The article tells the story of a real problem (225 mammal species
  on a binocular chip, rare species without photos) and its transferable lessons. inovex is
  mentioned once in the framing. The site template adds the CTA boxes.
- **Visual.** Photos, blog-styled charts, and short silent manim clips wherever motion
  explains a mechanism better than a still image.

## Fixed decisions

- **Language:** English, as all recent inovex expert and thesis posts are.
- **Angle:** synthetic data first, edge deployment second.
- **Partner:** Swarovski Optik and the AX Visio may be named. The production-style baseline
  is described generically ("the fastest route to integration was a fine-tuned YOLOv5s
  detecting a single 'animal' class, handing crops to SpeciesNet"). The article never says
  "Swarovski uses X, which they fine-tuned". No material from
  `thesis/presentation/images/nda/` and no BirdID/MammalID internals.
- **Facts that differ from early plans:**
  - The benchmark ran on a **Raspberry Pi 400**, not a Pi 5.
  - All models ran at FP32 on the CPU. **Quantisation and the Hexagon DSP are outlook only.**
  - Knowledge distillation **lost** to direct training.
- **Numbers** come from `thesis/presentation/slides/data.py` (each one with its source) and
  `thesis/manuscript/chapters/4-Results.tex`, nothing from memory.

## Files

| File | Description |
|------|-------------|
| `style-analysis.md` | Analysis of 9 inovex blog posts (length, structure, tone, visuals, CTAs) and the style rules derived from it for this article. |
| `outline.md` | Section-by-section outline of the article, with each claim and visual mapped to its source in the repo. |
| `draft.md` | The article, first full draft (about 2,550 words), with `[Figure N]` / `[Clip N]` placeholders and a table at the end that traces every number to its source. |
| `visuals.md` | Visual concept: what WordPress allows, the design rules (one colour code, light figures, three clips), each visual with format, source, effort and status, and the open points on rights and video hosting. |
| `plots/make_plots.py` | Draws the article's static charts (generator ranking, bands, generated vs real test set, accuracy vs latency) as 1600 px PNGs into `plots/out/`. Run: `uv run --script thesis/blog/plots/make_plots.py`. |
| `figures/hero/` | Three candidates for the featured image (`hero_a.jpg` photo wall, `hero_b.jpg` images → chip → detection, `hero_c.jpg` eyepiece view), 1600×939, built from generated animal images only in the blog's house style (radial blue-to-navy gradient, isometric objects, lime accents). Run: `uv run --script thesis/blog/figures/hero/make_hero.py`. |

Planned next: the photo collages and composites of `visuals.md`, `clips/` (its own uv
project with the three manim clips, reusing `thesis/presentation/slides/theme.py` and
`data.py`), and `figures/` (final media for submission).

## Status

- 2026-11-04: Directory created. Style analysis and outline written; draft not started.
- 2026-11-11: First full draft and visual concept written; first versions of the four
  charts rendered. The band figures in the draft follow the thesis's per-band table
  throughout (Band A 0.99 on generated test images against 0.30 on real ones), not the
  outline's 0.931 against 0.202. Open: rights for the product photo and the real photos,
  the video route on the blog, light or dark figures.
