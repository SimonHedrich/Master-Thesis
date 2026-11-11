# Blog Article: Visual Concept

> **Update 2026-11-11: no clips.** The article uses static images only. Clip C1 (pipeline
> against single detector) is dropped, C2 is replaced by the chart `plots/out/bands.png`,
> and C3 by `plots/out/latency_vs_map.png`. The clip rows and the section "Why three clips
> and not more" below are kept as the record of the earlier plan. The final figure list,
> with the numbering used in `draft.md` and the WordPress title, alt text and caption of
> each image, is in `media.md`.

What the article shows, in which format, and where each piece comes from. Written on
2026-11-11 together with `draft.md`; the `[Figure N]` and `[Clip N]` markers in the draft
refer to the table below.

## What WordPress allows, and what follows from it

| Constraint | Consequence |
|---|---|
| The content column is about 800 px wide, and about 340 px on a phone. | Nothing from the 16:9 talk can go in as it is: its labels are 1–2 % of the frame height and become unreadable. Every visual is laid out for the column, with text at least 3 % of the image width. Export at 1600 px wide (2×). |
| Images, galleries and self-hosted or embedded video. No custom JavaScript, usually no SVG upload. | No hover, no before/after sliders, no scrollytelling. Annotations are drawn into the image; comparisons sit side by side inside one image. |
| A gallery block reflows by screen size. | Collages are composed as **one** image, so that "real, then three generators" keeps its order and its labels on every device. |
| Video: the core video block can autoplay, loop and mute an MP4, which then behaves like a GIF at a tenth of the size. Whether the inovex editors allow self-hosted MP4 is the one open point (the Volleyball post has five videos, so some route exists). | Every clip is delivered three ways: `.mp4` (H.264, yuv420p, 1600×1000, ≤ 12 s, ≤ 2 MB, no sound), a `.webp`/`.gif` fallback at 800 px, and a poster `.png`. |
| Feeds, newsletters, link previews and print show no video. | **The last frame of every clip is a complete figure** and is used as the poster. A reader who never sees the motion loses nothing but the motion. |
| Captions and alt text are plain text fields. | Each visual has a one-sentence caption that states what to see, an alt text, and for real photos the CC attribution. Generated images are labelled as generated in the caption. |
| The featured image is cropped to about 1200×630 for cards and social previews. | The hero is designed for that crop, with nothing important near the edges. |

## Design rules

1. **One colour code from the first figure to the last.** Green means generated, navy means
   real photo. On test data: amber for the generated test set, pink for the real one. Blue
   for API models, grey for local ones. These are the talk's colours, darkened for a white page.
2. **Light figures on the white page**, not the talk's dark blue. `plots/out/ranking.png`
   and `ranking_dark.png` show the same chart both ways. The light one reads as part of the
   article, the dark one as a pasted slide, and six dark blocks in a row would dominate the
   page. The clips use the same light look.
3. **Photos carry the story, charts carry the evidence.** Each findings section opens with
   animals and gets at most one chart.
4. **Motion only where something changes.** Three clips, each showing a process: a frame
   passing through networks, a dataset being filled, a race. Charts with a single state stay
   static.
5. **A headline inside each chart** that states the finding, so the image still makes sense
   when it is shared without the article.

## The visuals

| # | Section | Visual | Format | Source and effort | Status |
|---|---|---|---|---|---|
| 1 | Hook | **Hero: "none of these animals exist".** A wall of generated animals (one per class, 6×3) with the AX Visio cut out in front, or the plain product photo as the fallback. | photo composite, 1600×840 | `presentation/images/generators/twelve/`, `grevys_synthetic_wall.png`, synthetic set on the GPU server; press photo `K22_AX_Visio_…jpg`. Half a day. | to build; **needs Swarovski's OK for the photo** |
| C1 | Two problems | **Pipeline vs single detector.** A frame with three zebras. Top lane: detector draws boxes, three crops queue at the classifier one by one. Bottom lane: one network, boxes with names at once. | clip, 8 s loop | new manim scene; the talk's `UseCase3` click 4–5 (MegaDetector → SpeciesNet → "ocelot") is the model for the top lane. One day. | to build |
| 2 | Collecting | **What a species folder contains.** Four rejected images: Tutankhamun "cheetah", tiger mascot, aardvark mask, buffalo range map. | collage, one image | `presentation/images/data/reject_*`; layout as on talk slide `Data3_08`. One hour plus licence check. | **check the four Wikimedia licences**, attribution goes into the caption |
| 3 | Generators | **Binturong: real, three local generators, Gemini.** The article's title picture. 3 rows × 5 columns, column labels as pills. | collage, one image | `presentation/images/generators/binturong/`; talk slide `Generative3_03` recomposed for the column. One hour. | to build; the three real photos need attribution |
| 4 | Generators | **Generator ranking** with the two short-prompt cells hatched and the prompt-token column. | chart | `plots/make_plots.py` | **first version rendered** (`plots/out/ranking.png`) |
| 5 | Generators | **One lion, four prompt lengths.** The four images above their prompts as shrinking sheets of text (4,642 words → 20 words). | composite, one image | `figures/images/generator_comparison/appendix/lion_001_*`, `slides/assets/prompts/`; talk slide `Generative3_05`. Two hours. | to build |
| 6 | Generators | **Kinkajou: real, FLUX.2 klein (ringed tail, circled), SD 3.5 Large, HiDream.** | 4-up, one image | `figures/images/generator_comparison/kinkajou_*.jpg`. One hour. | to build; real photo needs attribution |
| C2 | Bands | **The dataset, band by band.** 225 bars sorted by real photos → the four bands are cut → green fills A completely and B by half → the bars fold into the four result columns. | clip, 12 s loop | the talk's `Bands3` clicks 1–6 are exactly this sequence. Re-rendered in the light theme with larger labels, or cut from the existing 4K render with `loop_clip.py` as a quick first version. Half a day to a day. | poster frame rendered (`plots/out/bands.png`) |
| 7 | Test set | **Generated vs real test images.** Left: three generated zebras above three real ones (backlight, bushes, colour cast). Right: paired bars per band. | composite, one image | `presentation/images/bands/`, chart from `make_plots.py`; talk slide `Bands3_11`. Two hours. | chart half rendered (`plots/out/synthetic_vs_real.png`) |
| C3 | Making it small | **The race.** Three progress bars start together and run in real time: YOLO26n is done after 0.42 s, the two-stage pipeline after 5.4 s, and when the clip ends after 8 s the ensemble bar stands at a quarter with "35 s". | clip, 10 s loop | new manim scene, small. Half a day. | to build |
| 8 | Making it small | **Accuracy against time per frame**, log axis, four models. | chart | `plots/make_plots.py` | **first version rendered** (`plots/out/latency_vs_map.png`) |
| 9 | Look-alikes | **Elephants.** Asian and African elephant side by side, a thick arrow "79 %" one way, a thin one "0.2 %" back, training images under each (100 + 100 vs 1,500). | composite, one image | talk slide `Models3_10`; `presentation/images/models/`. One hour. | to build; the talk used generated elephants, real ones need attribution |

The model comparison and the band definition stay as real HTML tables in the article. They
are short, they reflow on a phone, and they can be read by a screen reader.

Optional, if the article should be richer still:

- **Three zebras, all generated** (`presentation/images/generators/zebra_*_gemini.png`) next
  to the prompt paragraph, to show how fine the details are that the long prompt carries.
- **Hyena and mangabeys** (`presentation/images/bands/synthetic_*.png`) next to the Band A
  paragraph: the best and the worst generated-only class.
- **"Real or generated?"** A 2×4 quiz grid at the very top, answers in the caption. It is
  the most engaging opener, but it competes with the hero and adds eight attributions.

## Why three clips and not more

The talk has 47 animated states, and most would work as clips. Each additional video costs
loading time and attention, and the article has to work without any of them. The three that
are planned are the ones where the still image is clearly worse:

- **C1** explains why two networks are slow (the classifier runs per animal), which a
  static diagram only asserts.
- **C2** is the experiment design. Readers who understand how the bars were filled
  understand the result without reading the table.
- **C3** makes "13× and 83×" something you wait for. It is the clip most likely to be
  shared on its own.

The ranking chart was a candidate (bars re-sorting when the short prompt comes in). It stays
static, because hatched bars show the same thing without asking the reader to wait.

## Reuse of the talk

- **Directly reusable:** `theme.py` building blocks (pills, rounded photos, bars),
  `data.py` (numbers), `collect_images.py` (downscaled images), `loop_clip.py` (cuts click
  states out of a rendered scene and crossfades the end into the start), and the image
  selection of slides `Generative3_03`, `_05`, `Data3_08`, `Bands3_11`, `Models3_10`.
- **Not reusable as rendered:** the 16:9 frames with title, logo and small labels. The
  scenes need a blog variant: 16:10 frame, light background, no title bar, text about
  twice as large, at most one idea per clip.
- **Numbers differ on purpose.** The talk's band slides show the per-class mean over the
  full test set (20.2 / 50.6 / 53.9 / 66.4 %, generated test 93 / 97 / 86 / 87 %). The
  article uses the thesis's per-band table throughout (0.300 / 0.588 / 0.591 / 0.742,
  generated test 0.99 / 0.99 / 0.93 / 0.87), so that the B-against-C comparison is the one
  from the thesis. A clip cut straight from the talk render would show the other set.

## Open points before production

1. **Rights:** the AX Visio photo and the wording about the product (Swarovski); the
   Wikimedia rejects and every real animal photo (licence and attribution per image).
2. **Video route:** ask the blog editors whether self-hosted MP4 with autoplay/loop is
   possible. If not, the clips go in as animated WebP, or as posters linking to YouTube.
3. **Hero:** pick one of the three candidates in `figures/hero/`. The blog's featured
   images are 1500×880 isometric vector illustrations on a radial gradient from #2D5DFF to
   #061B59 with lime accents, without text or logos; the candidates follow that, with
   generated animal images as the one photographic element.

Decided on 2026-11-11: light figures (`ranking_dark.png` stays only as the comparison).
