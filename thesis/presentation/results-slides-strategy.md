# Results Slides — Strategy

How to present the results in the final talk: four blocks, about 13 slides and 8 minutes. The
blocks follow the order of the work: generators → bands → models, with the embedded
benchmark as a side note on the model slide. Every number is taken from the manuscript
(Chapter 4) or the evaluation reports, and the source is given in the table at the end.

## Ground rules for every results slide

- **One message per slide**, said in one sentence. The slide shows the evidence for that
  sentence and nothing else.
- **Real-only numbers.** The mixed test set flatters every model (see R8), so the talk uses
  the real test set throughout. Slide R8 explains this once.
- **Redraw the charts.** Don't paste the thesis plots. Use the deck's style: navy
  background, white axis text, Atkinson Hyperlegible, at most 6 bars, labels at ≥ 14 pt.
  - One **highlight colour** for the bar the sentence is about: green `7DF381`. Everything
    else is light grey `DFE5F0` at reduced opacity.
  - API vs local can use sky `0AAFE8` vs teal `12A3A3` where a split is needed.
- **Numbers as pills.** The one number the sentence needs sits in a white pill next to its
  bar, like the model-name pills in the interim deck.
- **Two decimals, mAP on a 0–1 scale**, or round to a ratio ("half", "83×") when the ratio is
  the point.
- **Single runs.** No model has confidence intervals. Never say "better" about a gap below ~0.03
  (the seed anecdote on R3 makes this point once).

---

## Block 1: Which generator? (4 slides, ~3 min)

**Question that opens the block:** *"I picked Gemini because its binturongs looked best.
Was that a good reason?"* This follows straight on from the interim binturong slide.

### R1: The lineup

- **Visual:** one row of six images of the **same animal from the same prompt**: real photo
  on the left, then five generators. A model-name pill on each. No title, or just the animal's name.
- **Recommended animal: lion** (`figures/images/generator_comparison/appendix/lion_*`). Everyone
  knows what a lion looks like, so the differences (HiDream's yellow cast, FLUX's
  posture) are visible without explanation.
- **Optional second and third rows:** plains zebra (stripes) and saiga (see R4). Three
  rows of six is the maximum before it turns into a wall.
- **Missing asset:** no gpt-image-2 sample exists locally. The training images live in
  `data/synthetic_model_comparison/train/` on the GPU server. Pull one lion and one saiga
  from the `gpt-image-2-low-full` cell, because it is the winner and must be in the lineup.
- **Spoken:** "12 generator setups, same 12 animals, 1,200 images each, same detector,
  scored on 9,742 real photos."

### R2: The ranking

- **Visual:** the redrawn `model_comparison_full_headline_map.png`, simplified:
  - Show the 10 full-prompt and length-capped cells; leave the 2 compressed cells for R3.
  - Colour API bars and local bars differently, as in the original.
  - Mark **Gemini (my pick)** with a pill and give its bar the green highlight.
  - Keep the error bars thin and light. The professor will look for them, and Gemini's huge
    bar is a story beat.
- **Message:** "My pick came 3rd. The winner was the *cheapest* setting of gpt-image-2. No
  local model reaches any API model."
- **Spoken, optional:** price doesn't predict quality. gpt-image-2 low and medium are tied
  at a 2.9× price difference, and HiDream, the most expensive local model at ~48 GPU-hours, ranks 11th of 12.

### R3: The prompt matters more than the model

- **Visual:** left, the interim prompt slide shrunk to a tall column of tiny text (the
  full prompt) next to a short stub (the compressed prompt). Right, two pairs of bars:
  - gpt-image-2 low: **0.134 → 0.074**
  - Gemini Flash-Lite: **0.095 → 0.049**
  - A **−45 %** / **−48 %** pill on each pair.
- **Message:** "Cut the prompt and you lose half the accuracy, more than the gap between
  most generators."
- **Spoken aside (scientific humility):** "With one seed, my pick lost to its cheaper
  sibling, 0.064 vs 0.094. With three seeds it won. Every gap below ~0.03 on the next
  slides could flip the same way."

### R4: The blind spot for rare animals

- **Visual:** split slide.
  - Left: the per-class heatmap reduced to two rows of colour, *rare classes* (Band A,
    all pale, AP < 0.07 for every generator) vs *common classes* (dark).
  - Right: **real saiga vs generated saigas.** The real saiga has its trunk-like nose. In
    the local-model samples, none do; they draw a generic gazelle. (Check the gpt-image-2
    and Gemini saigas before making the claim for all generators.)
  - Alternative to the saiga: **kinkajou**, a real one next to FLUX.2's ringed-tail,
    spotted "civet". The thesis documents this case (`fig:kinkajou-generator-samples`).
- **Message:** "For the animals we needed synthetic images for, the generators don't know
  the animal, and our metric can't tell them apart."
- This is the honest bridge to Block 2: the ranking holds for common animals, and the rare
  ones are exactly what the bands test next.

---

## Block 2: Bands: what can synthetic images do? (4 slides, ~3 min)

**Question:** *"So can a class learn from synthetic images at all?"*

### R5: The experiment

- **Visual:** four columns of image tiles, each tile a real or synthetic thumbnail:
  - A: 200 synthetic
  - B: 100 real + 100 synthetic
  - C: 200 real
  - D: up to 1,500 real
  - Synthetic tiles carry a thin green outline. D is drawn wider and greyed as "reference".
  - This replaces the interim stacked-bar chart and table (slides 22–23). It shows the
    design, and the 225-bar chart can go to the backup slides.
- **Message:** "A, B and C get the same 200 images; only the mix differs."
- **Spoken caveat:** the bands contain different species, so the comparison is across
  species, not within them.

### R6: Half fake is fine, all fake is not

- **Visual:** three tall bars (YOLO26n, real test):
  - **A 0.30 · B 0.59 · C 0.59**
  - D 0.74 as a thin grey reference line
  - Under each bar, its mini tile strip from R5
  - B and C are both highlighted, and a "=" pill between them
  - A carries a "½" pill
- **Message:** "Swap half of a small real set for synthetic images: no loss. Use only
  synthetic: half the accuracy."
- **Spoken:** the same holds for the fine-tuned ensemble (B 0.638, C 0.639). YOLOv5s is the
  exception (0.34 vs 0.39). At the coarse level B even beats C slightly.
- **Takeaway line** (said, or one pill on the slide): *synthetic data stretches a real
  dataset; it doesn't replace it.*

### R7: Not every animal is the same

- **Visual:** a strip plot. One dot per class, four columns A–D, y = real-test AP. Band
  medians are marked:
  - A **0.21** · B **0.51** · C **0.54** · D **0.67**
  - Two photos pinned to Band A's extremes: **striped hyaena 0.48** (top) and **domestic pig
    0.00** (bottom)
- **Message:** "Some animals learn well from synthetic images alone, others not at all."
- **Spoken interpretation (a hypothesis, label it as one):** distinctive, uniform-looking
  species do best (hyaena stripes, maned wolf, bongo). Classes that are visually diverse
  in reality fail: domestic pig (breeds), genus- and clade-level classes (mangabeys,
  pinnipeds). A generator draws one idealised version of a class, and a diverse class needs many versions.
- Note: these per-class values come from scoring each class on the full real test set
  (`tier3_per_class`). They are not the band-subset mAP of R6, so don't put both kinds of number on one slide.

### R8: The test was too easy

- **Visual:** left, one synthetic test image (one centred animal, clean background) next
  to one real test image (several animals, partly hidden, bad light). Right, paired bars
  per band, synthetic test vs real test (YOLO26n):
  - A **0.93 vs 0.20**
  - B 0.97 vs 0.51
  - C 0.86 vs 0.54
  - D **0.87 vs 0.66**
- **Message:** "Every model scores higher on synthetic test images, even classes that never
  saw a synthetic training image. So everything in this talk is measured on real photos."
- **Spoken:** about 0.2 of the gap is pure difficulty. The off-the-shelf ensemble never saw
  a synthetic image and still gains 0.2–0.27. The rest is the training domain.
- This slide earns the professor's trust. It shows the evaluation was questioned, not just run.

---

## Block 3: Which model? (4 slides + 1 backup, ~2.5 min)

**Question:** *"Does the small model have to be bad?"*

### R9: The contenders

- **Visual:** circles scaled by parameter count (area ∝ parameters), each with a model-name
  pill:
  - MegaDetector + SpeciesNet ensemble **193 M**
  - YOLOv5s **7.6 M** (the model currently in the binocular)
  - YOLO26n **2.7 M**
  - The ensemble circle fills most of the slide, and YOLO26n is a dot.
- **Message:** "Three candidates, and one is 70× bigger than the smallest."

### R10: Accuracy and speed, in one picture (includes the embedded side note)

- **Visual:** horizontal bars of real-test mAP. Each bar carries two small grey pills on its
  right: **size** and **time per frame on the Raspberry Pi**.

  | Model | mAP (real) | Size | Time per frame |
  |-------|-----------|------|----------------|
  | Ensemble, fine-tuned | 0.60 | 193 M | ~35 s |
  | Ensemble, off the shelf | 0.45 | 193 M | ~35 s |
  | YOLOv5s | 0.41 | 7.6 M | 0.95 s |
  | **YOLO26n** | **0.53** | **2.7 M** | **0.42 s** |

  - Highlight YOLO26n.
  - Optional stopwatch icon for the latency pills, as the payoff for Act 1's stopwatch question.
- **Message:** "YOLO26n beats the off-the-shelf ensemble, is within 0.07 of the fine-tuned
  one, and is 71× smaller and 83× faster."
- **Wording check:** "similar performance to the SpeciesNet ensemble" is true against the
  *off-the-shelf* ensemble (YOLO26n is better there: 0.53 vs 0.45). It is not true against the
  *fine-tuned* ensemble (0.60), which is itself an upper bound: MegaDetector drew the
  ground-truth boxes it is scored against. Say "close to", not "as good as".
- **Spoken side note:** all latency is CPU-only on a Pi 400, full precision, no quantization.
  The Pi is ~10–15 % slower than the target CPU. Real-time on the DSP is the next step.
  YOLO26n also beats YOLOv5s on every axis (smaller, faster, more accurate). That is two model
  generations of progress, without any tuning.

### R11: Finding vs naming

- **Visual:** a three-step ladder:
  - *found the animal* **0.78**
  - *right species* **0.53**
  - Next to it, the elephant pair: an Asian elephant photo with an arrow to an African
    elephant photo, labelled **1,545×**, and a thin reverse arrow labelled **9×**.
- **Message:** "The small model almost always finds the animal. Naming it is where accuracy is lost."
- **Spoken:** most naming errors cross look-alike groups, not within them. A coarse fallback
  ("elephant" instead of the wrong species) is the obvious next step.

### R12: The plan that didn't work: distillation

- **Visual:** paired bars per band (real test), distilled (grey) vs direct (green). Direct wins in all four:
  - A 0.24 / 0.30
  - B 0.57 / 0.59
  - C 0.57 / 0.59
  - D 0.71 / 0.74
  - Optionally, a small thumbnail of the interim plan diagram (slide 12) with a ✗.
- **Message:** "Learning from the big model made the small one *worse*, most of all where we
  hoped it would help."
- **Spoken why:** the labels were produced by SpeciesNet, the teacher itself, so its soft
  labels add almost nothing. A species has one right answer, so there is little "dark
  knowledge" to pass on. Only one setting was tested, and the batch sizes differed.

### Backup B1: Accuracy vs latency scatter

- `embedded_latency_vs_map.png`, redrawn with real-only points and a log-scale x axis. For
  questions about the benchmark. Not shown in the main flow, since the benchmark is a side measurement.

---

## Pacing

| Block | Slides | Time |
|-------|--------|------|
| 1 Generators | R1–R4 | 3:00 |
| 2 Bands | R5–R8 | 3:00 |
| 3 Models (+ embedded) | R9–R12 | 2:30 |
| **Total** | **12** | **~8:30** |

That leaves ~11:30 for intro, problem, data, and conclusion. If time is short, drop R7 (per-class
spread) or R11 (finding vs naming) first. R8 and R10 should stay.

## Number sources

| Number | Source |
|--------|--------|
| Generator ranking, seeds, prompt ablation | `tab:generator-ranking`, `tab:prompt-length-ablation` (Ch. 4); `reports/model_comparison_*_downstream_map.csv` |
| Rare-class AP < 0.07 | `sec:results_per_class_structure`; `reports/model_comparison_full_per_class_ap.csv` |
| Band grid A/B/C/D (real) | `tab:band-grid-real` |
| Per-class AP, band medians, hyaena / pig | `scripts/training/yolo26n/model_exports/yolo26n-bs32-20260910-212812/evaluation/evaluation_report.json` → `tier3_per_class` |
| Synthetic vs real per band | `tab:domain-shift-delta` |
| Headline mAP, params | `tab:headline-cross-model` |
| Granularity ladder (real detect 0.78) | same report → `tier1.detect_analog.real_map` |
| Elephant confusion | `sec:results_lookalike_confusion` |
| Distilled vs direct per band | `tab:kd-vs-direct-bands` |
| Latency, 83× / 71× | `tab:embedded-latency`, `sec:results_embedded_benchmark`; `reports/embedded_benchmark/` |
