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

Planned next: `draft.md` (the article), `plots/` (matplotlib re-styles of the thesis
charts), `clips/` (its own uv project with the manim clips, reusing
`thesis/presentation/slides/theme.py` and `data.py`), and `figures/` (final media for
submission).

## Status

- 2026-11-04: Directory created. Style analysis and outline written; draft not started.
