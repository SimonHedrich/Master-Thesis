# Final Talk — Storyline

How the 29 interim slides (`interim-presentation-slides.md`) map onto a story for the
20-minute final talk. The numbers are taken from the manuscript (Chapter 4 and 5). Every
accuracy figure here is **real-only** unless it is marked as mixed. That is a deliberate
choice for the talk; see "Pitfalls" below.

## The story in one sentence

*A €4,600 binocular already recognizes birds. Making it recognize 225 mammals means
building a dataset that doesn't exist, faking the parts that are missing, and finding out
that most of the original plan was wrong, and that the answer was simpler than the plan.*

The interim talk ended with **"here is the plan."** The final talk is **"here is what
happened to the plan."** That gap carries the story. Each act opens with a question and
ends with an answer, and the answer is often a surprise.

## Arc and time budget

| Act | Driving question | Time | Slides |
|-----|------------------|------|--------|
| 0 Hook | Why can't this binocular recognize a fox? | 1:30 | ~4 |
| 1 The obvious solution | There's a model that works. Why not just use it? | 2:00 | ~4 |
| 2 The data that doesn't exist | Where do you get images of 225 species? | 3:30 | ~6 |
| 3 Faking the missing data | Can a generator fill the gap, and which one? | 4:00 | ~7 |
| 4 Training and the honest test | Does it work, and are we measuring it right? | 4:00 | ~6 |
| 5 The bet that failed | Does the big teacher help the small student? | 1:30 | ~2 |
| 6 Back on the device | Is it fast enough? | 2:00 | ~3 |
| 7 Close | What did we learn about the plan? | 1:30 | ~2 |
| | | **20:00** | **~34** |

About 35 seconds per slide. That works because nearly every slide is a single image
with one spoken point. If the time runs over, cut Act 3c (cost) and Act 4c (elephants) first.

---

## Act 0: Hook (interim 1–5)

| Slide | Source | Visual | Spoken beat |
|-------|--------|--------|-------------|
| Title | interim 1 | Unchanged | — |
| The product | interim 2 | Web-shop screenshot, €4,600 | "This is a €4,600 pair of binoculars." |
| Bird ID | interim 5 | Barn swallow → network → "BirdID" | "Point it at a bird and it tells you the species. Offline, on the device." |
| The dial | interim 6 (left half only) | Mode dial, mammal icon highlighted | "Turn the dial to mammals and ... nothing like that exists yet. That's the job." |

- **Cut interim 3** (the red X). It made sense as a joke in the interim talk; here the dial does that job.
- **Interim 4** (spec sheet) becomes a single overlay later, in Act 6, where the QCS605
  actually matters. One chip photo plus "Snapdragon-class, 2018" is enough.

## Act 1: The obvious solution (interim 6, 7, 9–12)

**Question:** *MegaDetector + SpeciesNet already do this. Why not ship them?*

| Slide | Source | Visual | Spoken beat |
|-------|--------|--------|-------------|
| The ensemble | interim 7 | Box → crop → "ocelot" | "Two models. One finds the animal, one names it. ~193 M parameters." |
| Plant the question | **new** | Same diagram, a stopwatch with "?" over it | "How long does that take on a binocular? Hold that thought." (answered in Act 6) |
| The plan | interim 10–12 merged into **one build slide** | Teacher (big) → distillation → student (small) | "So the plan: let the big model teach a small one." |
| Questions | interim 8 → **redrawn** | 4 icons: teacher→student, generator, mixed real+synthetic bar, chip | Name the four RQs in one breath each. No text. |

- **Cut interim 9** (the model list). The NanoDet/PicoDet lineup was never run (the
  student became YOLO26n). Showing it would invite the question "what happened to them?" Save
  that for Q&A or the Act 7 "plan vs reality" slide.
- Interim 6's detection examples (elephant, armadillo) can serve as the background of the ensemble slide.

## Act 2: The data that doesn't exist (interim 13–17, 22–23)

**Question:** *There is no dataset for this. How do you build one?*

| Slide | Source | Visual | Spoken beat |
|-------|--------|--------|-------------|
| 2,500 species | interim 13 | Taxonomy ladder | "~2,500 mammal species. Which ones?" |
| Hedgehog family | interim 14 | Hedgehog collage | "Some you can't tell apart, so some classes are a genus or family, not a species." |
| 225 classes | interim 15 | Order grouping as icons/silhouettes instead of a list | "225 classes, chosen by what users actually see." |
| Not enough images | interim 16 + 17 as **one animated chart** | GBIF bars, then the other sources stack on top | "114 classes under 100 images. Add four more sources: still 29." |
| **What the internet calls an animal** | **new** | Grid of rejected images: taxidermy, plush toys, drawings, museum skulls | "And many of the images aren't animals at all." Then the funnel: metadata → heuristics → MegaDetector → VLM → SpeciesNet. |
| Labels from a model | **new**, small | SpeciesNet in the pipeline, highlighted | Plant the seed: "the labels come from SpeciesNet." This pays off in Act 5. |

- The rejected-images grid is the best new visual in the talk: it is funny, and it is
  real. Pull examples from the curation review queue (`reports/manual_review_queue.*`).
- The "labels come from SpeciesNet" seed is what makes the distillation failure make sense later.

## Act 3: Faking the missing data (interim 18–21, 24 + new)

**Question:** *If the photos don't exist, can we generate them? With which generator?*

| Slide | Source | Visual | Spoken beat |
|-------|--------|--------|-------------|
| Binturong lineup | interim 18 | Real vs 4 generators | "I picked the one that looked best by eye." |
| Three zebras | interim 19 | Grévy's / mountain / plains | "The difference is in the stripes. The generator must get that right." |
| The prompt | interim 20 → **visualized** | Prompt as a wall of tiny text, key phrases highlighted | "So the prompt carries the field-guide description." |
| Zebra collage | interim 21 | Synthetic savannas | "23,850 images like this." |
| **Was it the right pick?** | **new**: `model_comparison_full_headline_map.png` | 12-cell ranking, production generator highlighted at #3 | "I tested it properly afterwards. My pick came 3rd of 12." |
| **Prompt > generator** | **new**: `model_comparison_prompt_ablation_headline_map.png` | Full vs compressed prompt bars | "Halving the prompt halves accuracy: −45 %. Bigger than the gap between most generators." |
| **The wrong animal** | **new**: kinkajou real vs FLUX.2 | Ringed tail vs plain tail | "This generator draws the wrong animal, and the metric doesn't notice." |

- Optional extras, if there is time: the cost-vs-mAP scatter ("the most expensive local
  model came 11th"), and interim 24's bill. **Fix the cost figure:** the interim talk said
  €647.29 billed, the manuscript says ~$785 at list price. Use one and say which it is.
- The **seed reversal** (0.064 → 0.090 → 0.105 with more seeds) is a strong one-liner
  about scientific humility. Use it here or in Act 7.

## Act 4: Training and the honest test (interim 22–23, 25–28 + new)

**Question:** *Does the detector learn, and can we trust the number?*

| Slide | Source | Visual | Spoken beat |
|-------|--------|--------|-------------|
| Bands | interim 22 + 23 → **one diagram** | 4 stacked bars: A = all synthetic, B = half/half, C = all real, D = lots of real | "Same budget for A, B, C. Only the mix differs. That's a controlled experiment." |
| **The zero** | interim 25 + **new** | MLflow curves, then a Band A bar at exactly 0.000 | "First results: Band A scored exactly zero. The data loader never read the synthetic images." |
| How we score | interim 27 → **replaced** | Ladder: *found it* → *right group* → *right species* (detect / coarse / fine) | "Three levels: did it find the animal, the look-alike group, the species?" |
| **Half fake is fine** | **new**: B vs C vs A bars (YOLO26n, real) | A 0.30 · B 0.59 · C 0.59 | "Half synthetic = all real. All synthetic = half the accuracy. Synthetic data stretches real data; it doesn't replace it." |
| **The test was too easy** | **new**: synthetic test image (one centred animal) next to a real one (three animals, partly hidden) | "Every model scores higher on synthetic test images, by up to 0.7. So in this talk, every number is real-only." |
| **Finding vs naming** | **new**: ladder with numbers (found 0.78 → named 0.53, real) + elephant pair | "It finds the animal. It struggles with the name. Asian → African elephant 1,545 times; the reverse 9 times." |

- **Cut interim 26** (the single word "FiftyOne"). It only fits if you do a live demo. A
  FiftyOne screenshot of the elephant confusions would replace it well.
- **Cut interim 27** (the COCO metric list). The professor knows it; everyone else doesn't
  need it. The ladder replaces it.
- **Interim 28** (65.1 % detection mAP) is outdated. The final numbers replace it.
- "The test was too easy" is the most honest moment in the talk, and the CV professor will
  respect it. It also explains why the evaluation strategy you committed to changed during the work.

## Act 5: The bet that failed

**Question:** *The whole plan was distillation. Did it work?*

| Slide | Source | Visual | Spoken beat |
|-------|--------|--------|-------------|
| Plan revisited | interim 12 (callback) | Same teacher→student diagram | "Remember the plan?" |
| **No** | **new** | Two bars per band, distilled vs direct; direct wins in every band | "Distillation lost everywhere: 0.48 vs 0.53. It lost most where we hoped it would help most, in Band A." |

Spoken *why*, no slide needed. The teacher's soft labels mostly repeat the hard labels,
because those labels *came from the teacher* (the seed from Act 2). And a species has one
correct answer, so there's little "dark knowledge" to transfer. Say what remains untested:
one configuration, one pairing, different batch sizes.

## Act 6: Back on the device (interim 4 + new)

**Question:** *The stopwatch from Act 1. How fast is it?*

| Slide | Source | Visual | Spoken beat |
|-------|--------|--------|-------------|
| The proxy | interim 4 (trimmed) + Pi 400 photo | QCS605 ↔ Pi 400 | "The real chip's toolchain is from 2018, so I used a Raspberry Pi as a stand-in, ~10–15 % slower." |
| **The answer** | **new**: stopwatch payoff | 35 s vs 0.42 s | "The ensemble: 35 seconds per frame. The animal is gone. YOLO26n: 0.42 s, 83× faster." |
| Accuracy vs speed | `embedded_latency_vs_map.png` | The scatter | "And it only costs 0.07 mAP. The new nano model beats the older, bigger YOLOv5s on every axis." |

- The newer generation beating the older one, with no tuning, is a good aside for the
  engineering colleagues: *use the latest model before you optimize the old one.*
- Be explicit: full precision, CPU only, no quantization. Real-time is the next step, not a claim.

## Act 7: Close (interim 29)

| Slide | Source | Visual | Spoken beat |
|-------|--------|--------|-------------|
| **Plan vs reality** | **new** | Interim plan diagram with each part ticked or crossed: distillation ✗, synthetic data ~ (half), small model ✓, real-time ~ (not yet) | The three takeaways: synthetic data stretches, it doesn't replace · prompt > generator · naming, not finding, is the hard part. |
| Summary | interim 29, refreshed | Collage + a generated zebra next to a real one | "A few years ago no generator could draw these stripes. The numbers will age; the protocol can be rerun." |

---

## Mapping at a glance

| Interim | Fate | Where |
|---------|------|-------|
| 1 | keep | Act 0 |
| 2 | keep | Act 0 |
| 3 | cut | dial replaces it |
| 4 | move, trim | Act 6 |
| 5 | keep | Act 0 |
| 6 | split | dial in Act 0, examples in Act 1 |
| 7 | keep | Act 1 |
| 8 | redraw as icons | Act 1 |
| 9 | cut | Q&A / Act 7 |
| 10–12 | merge into one build | Act 1, callback in Act 5 |
| 13–15 | keep (15 as icons) | Act 2 |
| 16–17 | merge | Act 2 |
| 18–21 | keep (20 visualized) | Act 3 |
| 22–23 | merge into one diagram | Act 4 |
| 24 | optional, fix figure | Act 3 |
| 25 | keep, add the "zero" | Act 4 |
| 26 | cut (or replace with a screenshot) | Act 4 |
| 27 | replace with the ladder | Act 4 |
| 28 | replace with final numbers | Act 4 |
| 29 | refresh | Act 7 |

## New slides needed

| Slide | Material |
|-------|----------|
| Rejected-images grid | `reports/manual_review_queue.*`, curation review decisions |
| Generator ranking | `thesis/manuscript/figures/plots/model_comparison_full_headline_map.png` |
| Prompt ablation | `.../plots/model_comparison_prompt_ablation_headline_map.png` |
| Kinkajou | `thesis/manuscript/figures/images/generator_comparison/kinkajou_*.jpg` |
| Band A zero | first-generation eval results (pre-loader-fix) |
| Bands B/C/A bars | Table `tab:band-grid-real` (Chapter 4) |
| Synthetic vs real test image | test-set samples |
| Granularity ladder + elephants | `tab:granularity-gaps`, `tab:lookalike-group-confusion-excerpt` |
| Distilled vs direct per band | `tab:kd-vs-direct-bands` |
| Stopwatch payoff | `tab:embedded-latency` |
| Latency vs mAP | `.../plots/embedded_latency_vs_map.png` |
| Plan vs reality | interim plan diagram + overlays |

The thesis plots are made for a printed page. For slides, redraw the key ones with larger
fonts, fewer labels, and one highlighted bar or point.

## Pitfalls to decide on

- **Mixed vs real numbers.** CLAUDE.md makes the mixed test set the headline. Chapter 4 shows
  it inflates every figure, and Chapter 5 reports real-only. This storyline uses real-only
  and turns the switch into a story beat ("the test was too easy"). If you keep mixed
  figures anywhere, label them every time.
- **Single runs.** No confidence intervals. The seed-reversal anecdote covers this honestly
  in one sentence. Don't present 0.003 differences as findings.
- **Teacher numbers are upper bounds.** MegaDetector produced the ground-truth boxes. That's
  worth one spoken sentence when the teacher's accuracy appears in Act 6.
- **Proxy hardware.** The manuscript uses a Raspberry Pi 400. Check that nothing (e.g. old
  slides) still says Pi 5.
