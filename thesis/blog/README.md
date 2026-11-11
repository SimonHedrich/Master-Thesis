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
- **Visual.** Photos, photo composites and blog-styled charts. Static images only
  (decided 2026-11-11; the manim clips of the first plan were dropped).

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
| `writing-rules.md` | **Read before editing the draft.** The review of draft v1 (claims stronger than their evidence, slogan headings, invented drama), the rules derived from it and a checklist for finding the same problems. |
| `outline.md` | Section-by-section outline of the article, with each claim and visual mapped to its source in the repo. |
| `draft.md` | The article, second draft (about 3,900 words, written to `writing-rules.md`), with `[Figure N: file]` placeholders and a table at the end that traces every number to its source. |
| `media.md` | The ten images in article order with file, status and the WordPress fields: title, alternative text and caption. |
| `visuals.md` | Visual concept: what WordPress allows, the design rules (one colour code, light figures), each visual with format, source and effort, and the open points on rights. The clips it planned were dropped on 2026-11-11. |
| `plots/make_plots.py` | Draws the article's static charts (generator ranking, bands, generated vs real test set, accuracy vs latency) as 1600 px PNGs into `plots/out/`. Run: `uv run --script thesis/blog/plots/make_plots.py`. |
| `figures/hero/` | Three candidates for the featured image (`hero_a.jpg` photo wall, `hero_b.jpg` images → chip → detection, `hero_c.jpg` eyepiece view), 1600×939, built from generated animal images only in the blog's house style (radial blue-to-navy gradient, isometric objects, lime accents). Run: `uv run --script thesis/blog/figures/hero/make_hero.py`. |
| `figures/make_figures.py` | Combines the source photos into one labelled image per figure (navy pill = real photo, green pill = AI-generated): `rejects.jpg`, `binturong.jpg`, `lion_prompts.jpg`, `kinkajou.jpg`, `elephants.jpg`, 1600 px wide, into `figures/` and into each figure's folder under `images/`. The two composites with uncleared real photos (rejects, binturong) are gitignored. Run: `uv run --script thesis/blog/figures/make_figures.py`. |
| `images/` | Every image of the article in one place, one folder per figure (`fig01_hero` … `fig10_elephants`): the charts and hero candidates as built, and the five photo composites next to their source photos. The generated binturong PNGs are stored as 1024 px JPEGs. Real photos without cleared licence (`fig02_rejects/`, the real binturongs and elephants) are local only (gitignored). |

Every figure is now one image: charts as PNG, figures with photos as JPEG.

## Status

- 2026-11-04: Directory created. Style analysis and outline written; draft not started.
- 2026-11-11: First full draft and visual concept written; first versions of the four
  charts rendered. The band figures in the draft follow the thesis's per-band table
  throughout (Band A 99 % on generated test images against 30 % on real ones), not the
  outline's 93.1 % against 20.2 %. Open: rights for the product photo and the real photos,
  the video route on the blog, light or dark figures.
- 2026-11-11 (later): Light figures chosen. Clips dropped in favour of the static charts;
  `media.md` added with title, alt text and caption per image. Three hero candidates built.
- 2026-11-11 (v2): Draft revised after review. Claims brought in line with their evidence
  (generator choice, price, prompt against model, rare-class limit of the comparison, band
  and test-set confounds), slogan headings replaced, TL;DR and takeaways rewritten to match,
  links added on first mention. The production generator is now described as the easier
  setup, with gpt-image-2 about a quarter ahead in the comparison. The review and the
  rules are recorded in `writing-rules.md`. The title stays "No Photos of the Binturong?":
  a question to the reader, not a statement about the dataset.
- 2026-11-11 (later): mAP is given in percent throughout (text, charts, captions, alt
  texts), differences in percentage points. The TL;DR now opens with a short context
  paragraph before the key findings. Both are recorded as rules 11 and 12 in
  `writing-rules.md`.
- 2026-11-11 (later): The five photo composites built by `figures/make_figures.py`. Fig. 5
  uses the RealVisXL lion for the 75-token prompt (as in the talk), Fig. 10 the generated
  elephants, so its caption needs no attribution.
