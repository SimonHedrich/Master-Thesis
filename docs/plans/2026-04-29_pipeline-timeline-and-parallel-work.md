# Pipeline Timeline & Parallel Work Plan
*2026-04-29*

## Original Query

Script 4 (`generate_captions.py`) is currently running. What is the estimated total runtime for the remaining GPU-bound pipeline steps, and what can be worked on in parallel while the GPU is occupied?

---

## Situation

The dataset quality pipeline runs sequentially on the RTX 3060 12 GB:

1. **Step 4** — Florence-2 generates detailed captions for all images (currently running)
2. **Step 5** — LLM filters captions (real animal, quality photo, whole animal visible)
3. **SpeciesNet validation** — runs SpeciesNet on the highest-confidence MegaDetector crop per image to check if labels match
4. **Supplementation planning** — decide how many synthetic images to generate per class
5. **Training** — distillation pipeline begins

**Dataset snapshot**: 531,178 images across 5 sources (iNat 458k, GBIF 51k, Wikimedia 15k, OI 11k, images_cv 7k), targeting 225 mammal species classes.

---

## Analysis

### Step 4 — Florence-2 captions (`generate_captions.py`)

- Model: `microsoft/Florence-2-base`, batch size 6, float16, `num_beams=1`
- Observed throughput: **10.3 images/sec** (1.72 batches/sec on iNat)
- Status at time of query: **93% complete** on iNaturalist, GBIF already done
- **ETA: ~27 minutes**

### Step 5 — Caption evaluation (`evaluate_captions.py`)

Two backends available:

| Backend | Time | Cost | GPU impact |
|---------|------|------|-----------|
| vLLM local (`Qwen2.5-7B-Instruct-AWQ`) | 12–18 hours | free | Blocks GPU |
| OpenRouter API (`qwen/qwen-2.5-7b-instruct`) | 30–90 min | ~$5 | GPU free |

The vLLM option blocks the GPU for more than half a day before SpeciesNet can start. At ~$5 for 531k images, OpenRouter is the clear choice unless running fully air-gapped.

### SpeciesNet label validation

**Goal**: Run SpeciesNet on the highest-confidence MegaDetector crop per image and compare top prediction vs. expected label. This reveals mislabeled or taxonomically ambiguous images before training.

**Key finding**: MegaDetector already ran during Step 1 quality filtering. Bounding boxes (`bbox_norm`, `conf`) are stored in `filter_results.jsonl`. The current `0-teacher_speciesnet_pipeline.py` would re-run MegaDetector unnecessarily.

| Mode | Time estimate |
|------|--------------|
| Re-run MegaDetector + SpeciesNet | ~21–28 hours |
| Reuse cached bboxes (needs code change) | ~10–15 hours |

Adding a `--use-cached-bboxes` flag to read detections from `filter_results.jsonl` saves ~10 hours.

---

## Timeline Comparison

| Milestone | Option A (OpenRouter + cached bboxes) | Option B (vLLM + re-run MD) |
|-----------|--------------------------------------|----------------------------|
| Step 4 done | now + 27 min | same |
| Step 5 done | **now + ~2 hours** | now + ~14 hours |
| SpeciesNet done | **~12–17 hours total** (tonight) | ~35–42 hours total |
| Ready for training | **tomorrow morning** | day after tomorrow |

**Recommendation**: OpenRouter for step 5 + modify SpeciesNet pipeline to reuse cached bboxes. Total cost: ~$5. Time saved: ~20–25 hours.

---

## Synthetic Supplementation Cost Analysis

### Class distribution

| Tier | Classes | Typical count |
|------|---------|--------------|
| Critical (<100 imgs) | 21 | ~30 avg |
| Low (100–499) | 50 | ~300 avg |
| Marginal (500–999) | 31 | ~700 avg |
| Good/Excellent (≥1000) | 123 | — |

Most-critical: Mouflon (9), Human (14), Domestic pig (15), various tropical/rare species.

### Cost scenarios (at ~$0.10/image)

| Strategy | Synthetic images | Estimated cost |
|----------|-----------------|---------------|
| Critical only → 100 imgs each | ~1,500 | ~$150 |
| Critical only → 200 imgs each | ~3,570 | ~$357 |
| Critical only → 500 imgs each | ~9,870 | ~$987 |
| Critical + Low → 300 imgs each | ~11,370 | ~$1,137 |

**Recommendation**: Supplement **critical tier only to ~200–300 images per class** (~$350–$460 total).

Rationale:
- Synthetic data is an experimental variable in the thesis, not a substitute for real data — a smaller, targeted supplement tells a cleaner story
- Class imbalance in the low tier is handled by loss weighting (standard practice, no extra cost)
- **Wait for SpeciesNet validation before committing**: some "critical" classes may have few images due to taxonomy mapping issues (e.g., Mouflon with 9 images), not genuine scarcity — supplementing them would be wasted spend

---

## Next Steps

### While GPU is occupied (start immediately)

| Task | Time | Blocker |
|------|------|---------|
| Modify `0-teacher_speciesnet_pipeline.py` to add `--use-cached-bboxes` | 1–2 h | none |
| Run `0-generate_scenario_prompts.py` for 21 critical classes | ~30 min | none (API only) |
| Design supplementation budget table (which classes, target count) | 1–2 h | none (use coverage_gaps.csv) |
| Review/finalize training configs (`nanodet-plus-m-wildlife225.yml`, `picodet-s-wildlife225.yml`) | 1–2 h | none |
| Write thesis dataset section | several hours | none |

### After Step 5 completes

- Review rejection rates (expect 15–25% filtered) and reason distribution
- Start SpeciesNet validation immediately

### After SpeciesNet validation completes

- Analyze mismatch rate per class — classes with >30% mismatch likely have labeling issues, not just scarcity
- Finalize supplementation list (remove classes that turn out to be mislabeled rather than rare)
- Run synthetic image generation for confirmed gaps
- Run `2-analyse_dataset_coverage.py` to verify final class distribution
- Begin training

---

## Critical Files

| File | Purpose |
|------|---------|
| `scripts/dataset_quality/5-evaluate_captions.py` | Run with `--backend openrouter` |
| `scripts/training/0-teacher_speciesnet_pipeline.py` | Add `--use-cached-bboxes` flag |
| `reports/coverage_gaps.csv` | Supplementation planning input |
| `scripts/synthetic/0-generate_scenario_prompts.py` | Run now for critical classes |
| `scripts/training/configs/nanodet-plus-m-wildlife225.yml` | Review before training |
| `scripts/training/configs/picodet-s-wildlife225.yml` | Review before training |
