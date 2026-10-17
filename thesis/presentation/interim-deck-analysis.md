# Interim Deck — Visual and Structural Analysis

An analysis of `interim-presentation.pptx` (29 slides) and `interim-presentation-images/`,
made by reading the slide XML, the speaker notes, and approximate renders of every slide.
It complements the content-level descriptions in `interim-presentation-slides.md`.

## Design system (inovex template)

| Element | Value |
|---------|-------|
| Canvas | 16:9, 10" × 5.625" (Google Slides export) |
| Background | Navy-to-blue diagonal gradient, darkest top-left (`061B59` → `2C5DFF`) |
| Logo | inovex wordmark, top left, on every slide |
| Title | White, top left at ~1" from the top, short (2–4 words); on most slides it is the only text |
| Font | Atkinson Hyperlegible throughout; Roboto Mono for the prompt slide |
| Theme accents | `2C5DFF` blue · `7DF381` green · `0AAFE8` sky · `41E8E0` cyan · `12A3A3` teal · `DFE5F0` light grey |
| Labels | White rounded pills (`roundRect`, 123 of them across the deck): model names, class names, single facts |
| Images | Rounded corners, often overlapping in loose collages (hedgehogs, zebras, binturongs, summary) |
| Charts | Google Sheets exports in white cards, default blue (`#4285F4`) bars |
| Marker | "NDA" tag top right on the Swarovski-internal slides (4–6) |

What already works and should carry over:
- **Photo collages as the main visual** (hedgehogs, binturongs, zebras). They feel alive
  and need no explanation.
- **Pills instead of text blocks.** A model name or a number sits on the image it refers to.
- **The step-by-step build** (slides 10 → 11 → 12) adds one element per slide, so the
  audience follows the diagram as it grows.
- **The summary slide reuses earlier slides as thumbnails.** It is a visual recap with no new text.

What is weaker and should change for the final talk:
- **Charts are the least designed element.** Google Sheets defaults (thin blue bars,
  225 rotated class names, small fonts) sit in white boxes and don't match the dark,
  photo-heavy style. In the final talk the results are the centre of the story, so the
  charts need the most redesign: fewer bars, larger labels, one highlight colour.
- **Tables (9, 15, 24, 28) are unreadable at presentation distance.** Slide 15 is a
  225-row table and slide 28 shows two tables of percentages. Every table should become a
  chart or one large number.
- **Text-only slides.** Slide 26 ("Fiftyone") and slide 8's labels carry no image;
  slide 27's COCO definition block is dense mono text.
- **Placeholder text on the title slide.** "Manche Knacksätze brauchen einfach etwas mehr
  Platz. Here you go!" is still in the template text box. It is likely hidden behind the
  title card, but remove it.

## Speaker notes (story beats already written)

The notes (German/English mix) contain the spoken transitions, which follow an
"→ next slide" chaining pattern. Content worth reusing:

| Slide | Note |
|-------|------|
| 6 | SpeciesNet was trained on camera-trap images and works well on crops |
| 7 | Swarovski wants better accuracy and a model for the hardware; provides ~60k unlabeled GBIF nature photos |
| 9 | COCO pretraining: person, car → dog, cat, teddy bear → toothbrush, baseball bat (classes irrelevant to wildlife) |
| 10–12 | "No exact ground truth anyway", which motivated using the ensemble as teacher |
| 13 | SpeciesNet covers ~2,500 species incl. birds and reptiles; Swarovski needs mammals only, outdoor-observable, sometimes family/genus is enough; the Swarovski PO had ~480 classes |
| 17 | Wikipedia: family/genus → all species; "this won't work for proper training" |
| 18–19 | Long prompt built from Wikipedia description + habitat/behaviour; Gemini accepts an almost unlimited prompt |
| 20 | Prompt variants generated via a Qwen model on OpenRouter for €5; different poses and environments per species |
| 22 | Ideally >1,000 images per class (not possible); at least 200 per class; 50 for testing |
| 24 | Thanks to the CTO office (funding) |
| 25 | YOLOv5s is what the binocular currently runs (detection only); Swarovski holds a licence and performance fits |

## Image assets

`interim-presentation-images/` (~250 MB with the pptx; not committed) holds:

| Asset | Use in the final talk |
|-------|-----------------------|
| `Binturong/` — 3 each from Gemini, RealVisXL, SD 3.5 M, FLUX schnell | Generator lineup (it predates the 12-cell comparison, so it lacks gpt-image-2) |
| `grevy_s_zebra/` — 100 Gemini Grévy's zebras | Collages; "200 synthetic images" visual for Band A |
| `Gemini_Generated_*zebra*.png`, `t_*_zebra_*.png` | Three-zebra slide |
| `Hedgehogs/` — 17 real photos | Hedgehog collage |
| `orinoquia_camera_trap*.jpg`, `detector_example.jpg` | Ensemble diagram (ocelot) |
| `microchip.png`, `brain-circuit.png`, `hierarchy.png` | Icons |
| MLflow, AI Studio spend, SpeciesNet blog screenshots | Process evidence |

Additional assets for the results part, already in the repo:
- `thesis/manuscript/figures/images/generator_comparison/` — real vs generated lion, plains
  zebra, saiga, and kinkajou for the six local generators, plus Gemini lion samples.
- `thesis/manuscript/figures/plots/` — generator ranking, prompt ablation, per-class
  heatmap, cost vs mAP, latency vs mAP.
