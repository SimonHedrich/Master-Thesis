# Progress Notes – 24.04.2026

## Synthetic Data Strategy & Knowledge Distillation Experimental Design

**Context:** Coverage analysis is now complete but not final — the Wikimedia Commons download is still running, so tier assignments will shift. These notes treat the current numbers as a working baseline.

---

## 1. Current Coverage Landscape

The filtered dataset has **225 classes** distributed across five tiers:

| Tier | Range | Classes | Notes |
|------|-------|---------|-------|
| Excellent | ≥1,500 usable | 83 | Deep enough for full fine-tuning |
| Good | 1,000–1,499 | 21 | Near threshold; usable |
| Marginal | 500–999 | 30 | Under-represented; benefits from supplementation |
| Low | 100–499 | 58 | Limited; candidates for capped real training |
| Critical | <100 | 33 | Near-zero real data; primary synthetic target |

Combined, 91 classes (Low + Critical) have fewer than 500 usable images. This is roughly 40% of the label set and constitutes the central data problem for the thesis.

Two things worth noting about the current numbers before treating them as definitive:

- **Wikimedia is still running.** Several critical-tier species have substantial Wikimedia presence (aardwolf has 30 Wikimedia images already; giant panda 38). The download may promote some species to Low or even Marginal.
- **The 20% quality buffer is an estimate.** The actual post-MegaDetector yield could be better or worse class by class. African civet and brown hyaena suffer from grayscale detection failures, which suggests nighttime/trail-cam image pollution — their effective usable counts are probably worse than stated.

---

## 2. Proposed Three-Tier Experimental Framework

The coverage landscape maps cleanly to a three-condition experimental design that gives the thesis multiple levels of analysis:

### Tier A — Data-Rich Training (83 classes, Excellent)
All real images available, capped at 1,500 per class for training consistency. These serve as the **baseline and upper bound** for both the KD experiment and the object detection model overall.

### Tier B — Parity Real Training (subset of Low/Marginal, ~40–60 classes)
Real images capped to exactly the same count as the synthetic images generated for Tier C (recommendation: **150 images per class**). The rest of the available real images are moved to a held-out validation/test pool.

This creates a controlled comparison: 150 real images vs 150 synthetic images for classes of broadly comparable visual complexity. The difference in detection AP between Tier B and Tier C classes *at the same training count* isolates the effect of image modality (real vs synthetic) from data quantity.

### Tier C — Synthetic-Only Training, Real Evaluation (subset of Critical, ~15–25 classes)
**All available real images are held out exclusively for evaluation.** Training uses only synthetically generated images. This is the most novel and highest-risk experimental condition. It tests whether a detector can learn a meaningful visual representation for a species it has never seen a real photograph of during training.

The subset is not the entire Critical tier. Several critical classes are unsuitable for this experiment:
- **"human" (Homo sapiens):** 2 real images, and ethical/privacy considerations make generating synthetic human images for training inappropriate in this context. Recommend dropping this class from the model entirely or using it only as a background rejection class.
- **Dingo, bongo, mouflon:** Fewer than 5 real images means no meaningful evaluation set. Unless more Wikimedia images arrive, these should either be dropped or merged with a parent taxon (e.g., dingo → domestic dog genus).
- **Pinniped clade:** A family-level catch-all with grayscale detection issues; drop or treat as a test of the family-fallback inference layer rather than a detection class.

**Recommended Tier C candidates** — visually distinctive species with 15–90 real images (enough for a 10–30 image evaluation set):

| Class | Real images (current) | Eval set size (est.) | Reason for inclusion |
|-------|----------------------:|--------------------:|----------------------|
| Snow leopard | 16 | ~8 | Iconic; distinctive rosette pattern; zero GBIF coverage reflects genuine rarity |
| Aye-aye | 24 | ~12 | Most morphologically distinctive primate; completely absent from OI/LILA BC |
| Wolverine | 34 | ~17 | Mustelid with distinctive markings; most iNat images fail quality filters (too small) |
| Saiga | 36 | ~18 | Globally distinctive inflated nose; near-threatened; genuinely data-scarce |
| Aardwolf | 42 | ~21 | Small striped hyaenid; hard to photograph; classification challenge vs spotted hyaena |
| Giant panda | 48 | ~24 | Iconic; black-and-white pattern extremely distinctive; most existing images are zoo captive → domain gap |
| Pangolin | 51 | ~26 | Scaled mammal; unmistakable morphology; Wikimedia contributes 11 images |
| African civet | 57 | ~23 | Vivid pattern; most images are grayscale camera-trap failures |
| Aardvark | 59 | ~30 | Unmistakable silhouette; nearly absent from Western datasets |
| Brown hyaena | 74 | ~30 | Shaggy coat clearly differentiates from spotted/striped hyaena; important confusion test |
| Black-backed jackal | 85 | ~34 | Distinctive saddle marking; useful because the "good" jackal classes (golden jackal) provide a comparison |
| Striped hyaena | 89 | ~36 | Decent eval set; tests hyaena family within-class separation |

That is 12 solid candidates with evaluation sets of ≥8 images. Add 3–5 more from the 90–100 range if additional Wikimedia images do not materialize after the download completes.

---

## 3. Budget Analysis for Synthetic Image Generation

At $0.05–$0.07 per image (FLUX.1-dev API or equivalent commercial provider), the $300 budget yields:

| Scenario | Images/class | Classes covered | Total images | Total cost |
|----------|:-----------:|:---------------:|:------------:|:----------:|
| Conservative (120 imgs) | 120 | 25 | 3,000 | $180–$210 |
| Target (150 imgs) | 150 | 20 | 3,000 | $180–$210 |
| Generous (200 imgs) | 200 | 15 | 3,000 | $180–$210 |
| Full $300 spend at 150 imgs | 150 | ~28 | 4,286 | $300 |

**Recommendation: generate 150 images per class, targeting the 12–15 Tier C candidates, and set the Tier B cap at 150 images per class as well.** This spends approximately $180–$220 and leaves a $80–$120 buffer for regeneration of low-quality batches or for adding more classes after a first quality check.

**Do not attempt to fill the entire Critical tier synthetically.** 33 classes × 150 images = 4,950 images × $0.06 = ~$297 leaves almost nothing for quality failures. With a novel generation approach on rare species, you should expect a 20–30% rejection rate for images that fail a post-hoc quality check (animal barely visible, wrong species rendered, bizarre artifacts). Budget for this.

---

## 4. Synthetic Image Generation Strategy

The `docs/synthetic-image-generation-model-research.md` has already characterized the available models on RTX 3060 12 GB. **FLUX.1-dev** (or **SDXL 1.0 + RealVisXL V5.0**) are the recommended base models. For Tier C species with 15+ real images, a species-specific **LoRA fine-tune** (DreamBooth-style, 4–8 bit NF4 quantised) will meaningfully improve morphological fidelity. For species with <15 real images, LoRA training is unreliable — fall back to well-crafted text prompts only.

### Prompt design principles

A diverse batch of 150 images requires systematic variation across four dimensions to prevent the model from learning spurious correlations between the species and a stereotyped background:

| Dimension | Variants to use |
|-----------|----------------|
| Habitat | Deciduous forest, conifer forest, savanna grassland, scrubland, rocky terrain, riverbank, snow-covered mountain |
| Viewpoint | Profile, 3/4 view, frontal, rear-3/4, partial occlusion by vegetation |
| Time of day | Golden hour, overcast midday, open shade, dusk (avoid pure nighttime — matches the AX Visio's color daylight operational envelope) |
| Distance | Close portrait (head+shoulder), three-quarter body, full body in habitat |

With 7 habitats × 4 viewpoints × 3 lighting = 84 unique prompt combinations, generating 2 images per combination (with different seeds) yields 168 images. Reject the weakest 18 in the quality check → 150 final.

### Per-species quality check

After generation, run each synthetic batch through a zero-shot check:
1. **MegaDetector v5** — discard any image where no animal is detected at confidence ≥ 0.5
2. **BioCLIP 2 zero-shot classification** — score whether the generated image is classified as the intended species at ≥ top-3. Images where BioCLIP strongly disagrees with the intended class are suspicious and should be inspected manually.

---

## 5. Knowledge Distillation — Thesis-Specific Considerations

The KD research overview in `docs/knowledge_distillation_research_overview.md` establishes the general framework. The synthetic data experiment introduces several additional tensions worth thinking through before designing the training pipeline.

### 5.1 The "soft label quality" problem for synthetic-trained classes

Standard KD relies on the teacher's soft class probabilities carrying meaningful semantic proximity information (the "dark knowledge"). For a class where the teacher was trained exclusively on synthetic images, the reliability of those soft labels is in question.

- A teacher that has only seen generated snow leopard images may learn a "snow leopard" representation that is biased toward the compositional shortcuts of the generative model (e.g., always depicted on mountain rocks, always shown in profile) rather than the genuine real-world visual manifold.
- When the teacher then assigns soft labels during student training, it may encode these generative biases as seemingly confident probabilities, misleading the student about real inter-class similarity.

**Hypothesis to test:** For Tier C classes, the KD-trained student will *underperform* the directly fine-tuned student on real evaluation images, because the teacher's soft labels are biased by synthetic generation artifacts. If this hypothesis holds, it becomes a concrete, novel finding about the limits of KD under synthetic data scarcity.

Conversely, there is a competing hypothesis: the teacher, having learned robust shared representations across the 83 data-rich Tier A classes, may assign soft labels to a Tier C class that reference the correct visual neighborhood (leopards when it sees snow leopard images, regardless of their synthetic nature) — effectively donating learned visual structure to the data-starved class. This would be evidence that KD provides additional benefit precisely in data-scarce conditions.

Testing both hypotheses requires running four training conditions per Tier:

| Condition | Teacher data | Student training |
|-----------|-------------|-----------------|
| Baseline-Direct | — | Direct fine-tune, real images (Tier A/B only) |
| Synth-Direct | — | Direct fine-tune, synthetic images (Tier C) |
| KD-Real | Fine-tuned on real | KD from teacher, student learns from soft labels |
| KD-Synth | Fine-tuned on synth | KD from teacher trained on synth, student from synth |

Comparing Synth-Direct vs KD-Synth for Tier C reveals whether KD adds value when both teacher and student are trained on synthetic data. Comparing KD-Real (Tier A/B) vs KD-Synth (Tier C) reveals whether the quality of the teacher's training data affects the downstream distillation gain.

### 5.2 Teacher-assistant cascade for capacity gap mitigation

The KD overview flags the capacity gap as a serious risk when the teacher is much larger than the student. With YOLOv12-L or RT-DETR as teacher and YOLO-nano or PicoDet as student, the parameter count ratio can exceed 20:1. In the synthetic data scenario this is compounded: the teacher has not fully converged on synthetic Tier C classes because the data is limited.

Mitigation options ranked by simplicity:
1. **Use the same YOLO family** (YOLOv12-L teacher → YOLOv12-n student) to allow direct neck feature matching without adapter projections.
2. **Use a medium-sized teacher-assistant** (YOLOv12-m) trained on the same data before distilling into nano. The intermediate model bridges the capacity gap and typically outperforms single-step distillation in low-data scenarios.
3. **Decoupled Knowledge Distillation (DKD):** separates target-class and non-target-class distillation signals. Particularly useful for long-tail scenarios because it prevents the model from over-optimizing the dominant classes' soft labels at the expense of rare ones.

### 5.3 Distillation loss choice for detection

For object detection specifically:
- **Feature-based FGD (Fine-Grained Distillation)** is the strongest method from the literature. It separates foreground from background before computing the mimicry loss, which directly addresses the problem that synthetic images may have unusual foreground/background ratios.
- For the Tier C experiment, the foreground-background separation is especially important because synthetic images tend to have centered, well-lit subjects — cleaner than the real-world distribution. This may produce unrealistically strong foreground signals in the teacher's feature maps.

### 5.4 Evaluation on a tiny real test set

33 critical classes with <100 real images total; reserving 30% for eval gives some classes only 5–8 test images. Per-class AP at this scale is meaningless and should not be reported as a primary result metric.

Recommended evaluation approach:
- **Tier-level aggregate mAP50** (macro-averaged within each tier) is the primary metric
- **Pairwise Wilcoxon signed-rank test** across tiers for KD vs direct fine-tune comparisons (non-parametric, tolerates small sample sizes)
- **Per-class AP bootstrapped 95% CI** (1,000 bootstrap resamples from test images) for the specific Tier C candidates being highlighted in the thesis narrative
- Plot the **AP vs training image count** curve for all classes — this will produce the clearest single figure showing the "synthetic data penalty" and the "KD benefit" as a function of data availability

---

## 6. Open Questions and Recommended Next Steps

### Immediate (before generating synth data)

1. **Wait for Wikimedia download to complete** and re-run the coverage analysis. Several Critical-tier species may get promoted; finalize the Tier C candidate list after seeing the final numbers.

2. **Decide on the class reduction question.** 225 classes is likely too many for a nano-scale student. Consider:
   - Merging subspecies (black/common wildebeest → wildebeest; mountain/Grevy's/plains zebra → zebra if Tier distinctions aren't needed)
   - Dropping irrecoverable classes: human, bongo (<3 images post-final-download), mouflon, dingo
   - Targeting 150–175 classes for the actual student model; keep 225 as the "full label" teacher configuration

3. **Resolve the "human" class.** It currently has 2 passing images and fails both animal detection and quality filters. Either drop it or designate it explicitly as an out-of-distribution "background rejection" class rather than a detection class.

### Synthetic data generation pipeline

4. **Select the base generative model** and test generation quality on 5 pilot species (one per visual type: feline, canid, primate, ungulate, atypical morphology like aardvark/pangolin). Only commit the full budget after inspecting pilot quality.

5. **For LoRA fine-tuning candidates** (15+ real images): run DreamBooth on FLUX.1-dev or SDXL with the available real images as reference set. Validate that the LoRA-guided generation actually improves species fidelity over text-only, since for some charismatic species (giant panda) the base model may already be excellent without LoRA.

6. **Add a "synth domain discrepancy" measurement** to the quality check pipeline: after generating each batch, run BioCLIP 2 on both the synthetic batch and the held-out real images and compare the feature-space distributions (Fréchet distance or MMD). A large distribution gap is a warning that the synthetic images are not close to the real visual manifold and KD benefits will be limited.

### Training pipeline design

7. **Establish the training ladder** before touching the data:
   - Stage 1: Train teacher (YOLOv12-L) on Tier A + B real images + Tier C synth images → produces the teacher checkpoint
   - Stage 2: Fine-tune a teacher-assistant (YOLOv12-m) using KD from Stage 1 teacher
   - Stage 3: Distill nano student from teacher-assistant
   - Compare Stage 3 KD student vs a direct fine-tune of nano student (no teacher) on the same data

8. **Quantization-aware training (QAT)** should be applied after KD, not before. The thesis plans INT8 deployment on the QCS605. Run QAT as the final step in the ladder and measure the QAT penalty separately from the KD/synth penalties — otherwise the attribution gets muddled.

### What makes this thesis contribution distinct

The combination of (a) KD under extreme data scarcity, (b) controlled synthetic-vs-real comparison at matched count, and (c) a tiered class structure with embedded deployment constraints is not well-covered in the current literature. The strongest narrative arc is probably:

> "We show that Knowledge Distillation provides consistent mAP improvement across data-rich classes but exhibits diminishing returns—and in some cases regression—for classes trained exclusively on synthetic data, suggesting that the quality of the teacher's representations is bounded by the fidelity of its training images. This finding motivates a hybrid strategy: use KD aggressively where real data exists, and invest budget in higher-fidelity synthetic image generation for data-scarce classes rather than assuming KD alone can compensate for low-quality synthetic training data."

This is a falsifiable, practically motivated claim that the experimental design above is capable of testing.

---

## Summary

| Experimental Tier | Training data | KD? | Eval data |
|-------------------|--------------|-----|-----------|
| A — Data-rich (83 classes) | ≤1,500 real images/class | Yes + No | Real held-out |
| B — Parity real (40–60 classes) | 150 real images/class | Yes + No | Real (remaining) |
| C — Synth-only (12–15 classes) | 150 synthetic images/class | Yes + No | All real images |

**Budget:** $180–$220 for 12–15 Tier C classes at 150 images/class. Reserve $80 buffer for quality rejection and regeneration.

**Next decision point:** finalize Tier C candidate list after Wikimedia download completes and re-run coverage analysis.
