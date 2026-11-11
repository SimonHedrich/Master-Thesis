# Blog Article: Outline

Target: about 2,500 words, 8–10 static visuals, 2–3 silent manim clips. English. Style
rules: `style-analysis.md` §"Rules for this article". Every number below was checked against
`thesis/presentation/slides/data.py` (abbreviated `data`) and
`thesis/manuscript/chapters/4-Results.tex` (abbreviated `R`) on 2026-11-04.

**Working title:** *No Photos of the Binturong? Training a Wildlife Detector for a Binocular
with AI-Generated Images*
**Alternative:** *225 Species, One Tiny Chip: Synthetic Data and Edge AI for Wildlife
Detection*

---

## 0. TL;DR: Key findings (4 bullets)

1. Prompt detail matters more than the generator. Shortening the prompt costs 45–48 % of
   the accuracy (R l.58–59). The 2.9× more expensive gpt-image-2 tier does no better than the
   cheap one.
2. Synthetic images can replace **half** of a small real set (Band B 58.8 % vs Band C
   59.1 % real mAP), but not **all** of it (Band A 30.0 %) (`data.BAND_REAL`).
3. Synthetic test images flatter the score: Band A scores 93.1 % on synthetic images vs
   20.2 % on real ones (`data.BAND_SYNTH_TEST`, `BAND_REAL_TEST`).
4. A 2.7M-parameter YOLO26n is about 13× faster than a detector + classifier pipeline and
   83× faster than the large teacher. It costs 7 percentage points of mAP on real photos (R l.454, l.467).

## 1. Hook (no heading)

- **The moment:** a hiker sees a brown shape at the edge of the forest through the
  binoculars and wonders "what *is* that?"
- **The product:** the Swarovski Optik AX Visio has a camera and an AI chip that answers
  that question for birds. For mammals, it would need to tell **225 species** apart on a
  Qualcomm QCS605.
- **Framing (one sentence):** my Master's thesis, carried out at inovex together with
  Swarovski Optik, asked how to build that detector when the chip is tiny and photos of many
  species are scarce.
- **Visual:** AX Visio press photo. Candidates: the press image
  `thesis/presentation/K22_AX_Visio_10_Backside_Top_13_*.jpg`, or
  `manuscript/figures/images/ax_visio/`. Confirm with Swarovski which one may be published.

## 2. Two problems: rare animals and a small chip

- **The data gap:** 50 of 225 species have fewer than 150 usable real photos. Examples:
  binturong, saiga, kinkajou.
- **The baseline:** the fastest route to integration was a fine-tuned YOLOv5s that detects a
  single "animal" class and hands each crop to SpeciesNet for classification. Two networks
  run in sequence, so the cost grows with every animal in the frame.
- **The question:** can **one** small detector do both jobs, trained partly on generated
  images?
- **Clip 1 `PipelineVsNano`:** a frame through two-stage (detect → crop → classify) vs
  single-stage, with latency bars. *Could also sit in §6; decide during drafting.*
- **Wording rule:** describe this pipeline only generically (README "Partner"). Never write
  "Swarovski uses …".

## 3. Collecting real photos, and what you throw away

- **The funnel:** about 580k candidates from iNaturalist, GBIF, Wikimedia, Open Images and
  stock imagery. 465,242 pass the license filter. The final split is 155,808 train /
  15,063 val / 63,802 real test images.
- **Anecdotes:** a tiger mascot, a Tutankhamun cheetah, a museum aardvark mask. Scraped
  "species" folders contain a lot of things that are not animals.
- **Visual:** a 3–4 tile collage of rejected images, from `presentation/images/data/`. Check
  the CC licenses.
- Keep this section short (~200 words). It shows care for the data without being about the
  data pipeline.

## 4. Which image generator? Twelve candidates compared

- **The setup:** 12 generator × prompt combinations, 1,200 images each. A YOLO26n is trained
  on each and scored on **9,742 real photos**. The measure is how useful the images are for
  training, not how good they look.
- **The ranking** (`data` l.233ff):
  - gpt-image-2 (low) with the full prompt: 13.4 %.
  - Gemini 3.1 Flash Image, the production pick: 10.5 %, rank 3.
  - Best local model, SD 3.5 Large: 5.5 %, rank 6.
  - HiDream: about 48 GPU-hours, rank 11.
- **The kinkajou story:** FLUX.2-klein draws a ringed tail, which belongs to a different
  animal. Plausible-looking images can teach the wrong features.
- **The prompt:** −45 % / −48 % accuracy with the short prompt (R l.58–59).
- **Price:** it doesn't predict quality.
- **Visuals:**
  - kinkajou 4-up (`manuscript/figures/images/generator_comparison/kinkajou_*.jpg`);
  - ranking bar chart, re-styled (`plots/`);
  - lion per prompt regime (`generator_comparison/appendix/`), or Clip 3.
- **Cost:** the whole synthetic set (23,850 images) cost about $785.

## 5. How much synthetic data is too much? The band experiment

- **The bands:** classes are grouped by how many real photos they have:
  - A (50): synthetic only;
  - B (26): 100 real + 100 synthetic;
  - C (26): 200 real;
  - D (122): data-rich.
- **Results** (YOLO26n, real mAP): B 58.8 % ≈ C 59.1 % → half synthetic is as good as all
  real. A 30.0 % → synthetic alone isn't enough, though still better than nothing (the first
  model scored 0 % on Band A). D 74.2 %.
- **Clip 2 `BandsDataset`:** 225 bars sorted by real images. Real counts grow in, then
  synthetic fills A completely and half of B, then the mAP per band appears.

## 6. Don't trust the synthetic test set

- **The gap:** synthetic test images score 20–70 percentage points higher than real ones. Band A:
  93.1 % synthetic vs 20.2 % real.
- **How we report:** the mixed headline is always shown next to the real-only breakout.
- **The lesson:** if you generate your test data, you measure your generator, not your model.
- **Visual:** paired bars per band (synthetic vs real test), re-styled.

## 7. Making it small: one nano detector instead of two networks

- **Candidates:**
  - Teacher MegaDetector + SpeciesNet (≈193M params);
  - YOLOv5s (7.63M);
  - YOLO26n (2.71M).
- **Accuracy on real photos:**
  - teacher, fine-tuned: 59.9 % (an upper bound: its detector produced the boxes);
  - YOLO26n: 52.9 %;
  - YOLOv5s: 40.7 %.
- **Speed** on a Raspberry Pi 400 as CPU proxy (ONNX Runtime, FP32, 4 threads, full frame
  time):
  - YOLO26n: 423 ms (2.36 FPS, 284 MB);
  - YOLOv5s: 949 ms;
  - MegaDetector alone: 31 s.
  - Composed estimates: teacher pipeline ≈35 s, YOLOv5s + SpeciesNet ≈5.4 s (R l.452).
  - Result: **83× and 13× faster**.
- **A side note:** decoding a native 4192×3120 AX Visio frame alone costs 256–275 ms
  (R l.467). Pre-processing matters on the edge.
- **Distillation, a negative result:** distilling from the teacher scored 47.9 % vs 52.9 %
  for direct training. Caveats: one setting, single runs. Report it plainly.
- **Visual:** accuracy-vs-latency scatter, log-x (re-styled
  `manuscript/figures/plots/embedded_latency_vs_map.png`). **Clip 1** if it isn't used in §2.

## 8. Where it still fails: look-alikes

- **Detection vs naming:** finding the animal works (class-agnostic mAP 78.2 % on real
  photos, R l.380). Naming the species is the hard part.
- **Within look-alike groups:**
  - 19.3 % of correctly grouped detections get the wrong species (teacher 12.9 %).
  - Elephants: 50.8 %. 1,545 Asian elephants labelled African vs 9 the other way
    (`data.ELEPHANT_FWD/REV`).
  - Zebras are comparatively easy at 9.5 %.
- **Visual:** Asian vs African elephant pair (`presentation/images/models/`, check
  licenses), or the zebra trio (`manuscript/figures/images/zebra_species/`).

## 9. What's next

- INT8 quantisation and the Hexagon DSP of the QCS605. These were **not** measured, all
  results are FP32 on the CPU.
- Better generators keep arriving. The comparison protocol (train on generated images,
  score on real photos) is what carries over.

## 10. Conclusion + Takeaways for your next project

**Conclusion:** 3–4 sentences.

**Takeaways (you-form):**
1. Judge a generator by what it teaches the model on real photos, not by how the images look.
2. Spend effort on the prompt before spending money on the generator.
3. Use synthetic data to fill gaps, not to replace real data. Plan for a ceiling.
4. Never evaluate on synthetic data alone. Always report a real-only breakout.
5. On edge hardware, count every network and every pre-processing step, not just FLOPs.

**Related links:**
- the bachelor article;
- [Tracking on the Edge](https://www.inovex.de/de/blog/tracking-on-the-edge-a-case-study-on-real-time-object-tracking/);
- [Android Snapdragon Integration](https://www.inovex.de/de/blog/android-snapdragon-integration-tensor-interpretation-with-kotlin/);
- optionally inovex's computer vision / AI offering page.

---

## Visual inventory

| # | Visual | Type | Source | Status |
|---|---|---|---|---|
| 1 | AX Visio product photo | photo | press image (presentation/) | confirm rights |
| 2 | Rejected scraped images | collage | `presentation/images/data/` | check CC |
| 3 | Kinkajou: real vs 3 generators | 4-up | `manuscript/figures/images/generator_comparison/` | exists |
| 4 | Generator ranking | bar chart | `reports/model_comparison_*` CSVs → `plots/` | to re-style |
| 5 | Lion per prompt regime | 4-up / clip | `generator_comparison/appendix/` | exists |
| 6 | Bands dataset build-up | **clip** | `data.py` | to build |
| 7 | Synthetic vs real test per band | paired bars | `data.BAND_*_TEST` | to plot |
| 8 | Pipeline vs nano detector | **clip** | `data.MODELS` | to build |
| 9 | Accuracy vs latency | scatter | `scripts/benchmark/5-report.py` data | to re-style |
| 10 | Elephant look-alikes | photo pair | `presentation/images/models/` | check CC |
