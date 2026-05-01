# SpeciesNet Classification and Filtering Strategy

**Date:** 2026-04-30  
**Context:** Pipeline step 6–7, following Florence-2 caption generation (step 4, in progress) and LLM caption evaluation (step 5, complete).

---

## Background

After steps 4–5, each image in `filter_results.jsonl` has:
- `passed: true/false` (MegaDetector + heuristics + caption LLM)
- `bbox` / `bbox_conf` from MegaDetector (highest-confidence detection)
- `caption` (Florence-2)
- `caption_eval.pass` (LLM: real animal, quality photo, whole animal visible)

The remaining question is whether the **image label** (from directory structure, derived from iNat/GBIF metadata) is actually correct. SpeciesNet is the tool to answer this, but using it well requires thinking through four non-trivial issues.

---

## Issue 1: SpeciesNet May Predict the Wrong Species

SpeciesNet is a visual classifier trained on ~3,537 species. It has no access to collection metadata, geolocation, or behaviour — only pixels. It will sometimes confuse visually similar species that are taxonomically close (e.g., jaguar vs. leopard, grey wolf vs. coyote, ring-tailed lemur vs. mongoose lemur).

### Why binary match/no-match is wrong

A binary "does SpeciesNet's top-1 match the directory label?" is both too strict and too loose:
- **Too strict**: SpeciesNet predicts *Panthera pardus* (leopard) for an image labeled *Panthera onca* (jaguar) → fail, even though the image clearly shows a big spotted cat. The species-level confusion may be a legitimate visual ambiguity, not an error in the directory label.
- **Too loose**: SpeciesNet predicts something completely unrelated with low confidence → trivial pass, but the image might be mislabeled or corrupted.

### Recommended approach: Hierarchical Taxonomic Match Level

SpeciesNet classes have the format:  
`UUID;class;order;family;genus;species;common_name`  
Example: `5a9b1344...;mammalia;carnivora;felidae;panthera;onca;jaguar`

Compare the SpeciesNet top-1 prediction against the expected label at each taxonomic level:

| Match Level | Definition | Interpretation | Default Action |
|---|---|---|---|
| `species` | genus + species both match | Confirmed | PASS |
| `genus` | same genus, different species | Visually ambiguous ID, likely correct animal | PASS (flagged) |
| `family` | same family, different genus | Probably wrong, but visually similar clade | Conditional (see below) |
| `order` | same order, different family | Likely wrong image or mislabel | FAIL |
| `class` | both Mammalia, different order | Almost certainly wrong | FAIL |
| `no_match` | not even Mammalia, or blank detection | Definitely wrong | FAIL |

**Family match with confidence interaction:**  
- SpeciesNet uncertain (score < 0.5) + family match → trust the directory label, **PASS**  
- SpeciesNet confident (score ≥ 0.5) + family match → SpeciesNet is sure it's a different genus, **FAIL**

This rule is biologically motivated: high SpeciesNet confidence at family level means the animal genuinely resembles a different genus's species (e.g., predicted a mustelid when the label says viverrid). That is not species-level ambiguity — it is a different animal.

### Caveat: Visually similar species that span genus boundaries

The family match + confidence threshold rule has a known blind spot: some species pairs look nearly identical visually but are taxonomically separated at genus level. The leopard (*Panthera pardus*) and the leopard cat (*Prionailurus bengalensis*) are both Felidae but different genera, and both share a spotted coat pattern that is a dominant visual signal. SpeciesNet's accuracy on such pairs is unclear — a high-confidence family-level mismatch may reflect genuine visual ambiguity rather than a directory mislabel.

This contrasts with closely related species such as different zebra species (e.g., *Equus quagga* vs. *Equus zebra*), which fall into the genus match tier and are treated more leniently (PASS with flag). That leniency is appropriate given their visual similarity. But the asymmetry means the current rules may be too strict for cross-genus visual look-alikes and too lenient for within-genus species that are actually visually distinctive.

The `match_level` field in the output preserves the raw information, so thresholds can be revisited post-hoc. A recommended calibration step: pull a random sample of family-level mismatches for that are high-confidence and manually inspect whether SpeciesNet's prediction or the directory label is more plausible. If a substantial fraction of family mismatches turn out to be genuine visual ambiguity (as opposed to mislabels), the family match + confidence ≥ 0.5 = FAIL rule should be relaxed or supplemented with a per-family-pair allowlist.

### No LLM needed for this

The hierarchical match is automatic, interpretable, and requires no LLM call. The `match_level` field in the output lets human reviewers adjust thresholds post-hoc without rerunning SpeciesNet.

---

## Issue 2: Multiple Animals in an Image

### 2a: Multiple non-human animals

When an image contains the labeled species as the main subject plus other animals in the background, should those images be filtered?

**For standard supervised learning (hard labels):**  
Unlabeled animal instances are false negatives. The detector is penalised for correctly detecting them, which degrades training signal and harms mAP on evaluation sets.

**For knowledge distillation (soft labels from teacher):**  
This is where the KD approach gains a genuine advantage. The teacher pipeline (MegaDetector + SpeciesNet) detects and classifies *all* animals in the image, not just the labeled one. The student learns from these richer soft labels. An image with one labeled zebra and two background wildebeest provides three sets of teacher supervision; hard-label training provides only one. This is an intrinsic property of soft-label KD that direct fine-tuning cannot replicate.

**Recommendation:** Do NOT filter multi-animal images. Track them with `multi_animal: true` and `n_animal_detections: N`. This preserves a valuable ablation: compare KD performance on single-animal vs. multi-animal subsets to quantify whether the extra teacher supervision actually helps generalisation. This is a testable and publishable result.

### 2b: Humans in images

Wildlife images routinely contain incidental humans (researchers, tourists, rangers). MegaDetector category `"2"` = person. The dedicated `data/coco_humans` dataset covers the person class for training.

**Deployment scenario motivation:** The AX Visio binoculars are used in the field. A user may be observing an animal that is standing next to another person. The desired model output in this scenario is the animal species — not a person detection. A model that fires on the human instead of the mammal is a use-case failure, even if the detection is technically correct. Mixed animal+human images are therefore training signal for the correct behaviour, not noise to be removed. Filtering them would understate person co-occurrence in the training distribution and produce a model that is not trained on the hardest real-world cases.

**Deliberate asymmetry in training priority:** The model should be slightly penalised for preferring a person detection over an animal detection. This is an intentional design choice, not a quality issue. Implementation options:

- **Class weight (simplest):** Assign a lower loss weight to the person class in classification, e.g. 0.3× relative to animal-species classes. The model learns to be reluctant to output a person detection when an animal is present.
- **KD-level downweighting:** During distillation, suppress or reduce the weight on the teacher's person-class soft labels relative to its animal-class predictions. More elegant but requires care not to distort animal-class probability distributions.
- **NMS post-processing:** Apply a stricter confidence threshold for person boxes in the final output — person detections are suppressed more aggressively than animal boxes.

The class-weight approach is the most transparent and reproducible; it should be the default.

**Evaluation implications:** This design choice has measurable consequences for standard benchmarking:
- COCO-style mAP counts person detections as true positives. A model trained to suppress person detections will score lower on person-class mAP.
- **Mitigation:** Report animal-class mAP separately from full mAP (excluding person from ground truth in the animal-only variant). This makes the trade-off explicit and defensible.
- A targeted evaluation — species detection accuracy in images where humans are co-present — directly measures deployment performance and should be reported as a primary metric alongside standard mAP.

The trade-off is accepted: lower person-class mAP in exchange for better animal detection priority in the realistic use case. This is a product decision, not a data quality error.

**Recommendation:** Do NOT filter mixed images by default. Track with `has_human: true` and `n_person_detections: N`. Allow opt-in filtering via config for ablation studies. Pair with a class weight or KD weight scheme that encodes the deployment priority: animal species first, person class deprioritised.

**One filter that IS correct:** If the highest-confidence *animal* detection (category `"1"`) cannot be found — meaning MegaDetector's top detection overall was a person or vehicle — the image should fail. The primary detection must be an animal, not a bystander.

---

## Issue 3: Mapping SpeciesNet Taxonomy to the 225 Classes

### The scope mismatch

SpeciesNet covers ~3,537 species globally. The 225-class target covers non-bird mammals only. Most SpeciesNet predictions will not map to any of the 225 classes. This is expected.

### The 225-class taxonomy types

`reports/classes_225.csv` has `common_name, scientific_name, level` where level is:
- `species`: e.g., `red fox, vulpes vulpes, species` — match on genus+species
- `genus`: e.g., `ateles species, ateles, genus` — match on genus only; aggregate scores across all species within the genus

### Mapping algorithm

**Pre-compute a lookup table from classes_225.csv:**
- For each species-level row: key = `"genus species"` (lowercase) → class index
- For each genus-level row: key = `"genus"` (lowercase, first word of scientific_name) → class index

**For each SpeciesNet prediction string** (per detection):
1. Parse: split by `;`, extract genus at position [4] and species at position [5]
2. Try species match: `genus + " " + species` → look up species table
3. If no match, try genus match: `genus` → look up genus table
4. If no match: record `"no_match"` with the raw SpeciesNet taxon

**For the 225-class probability vector (soft labels):**
- Initialise a zero vector of length 225
- For each (SpeciesNet class, score) pair: add score to the corresponding 225-class slot
- Genus-level 225 classes naturally accumulate scores from all matching SpeciesNet species
- Record `prob_225_sum` = total probability mass mapped to any 225 class (diagnostic: 0.0 means entirely out-of-distribution)

### Disable SpeciesNet geo-filtering

SpeciesNet has a geo-aware prediction mode. Per the project's design principle (geo-filtering is post-hoc, applied after inference, not at model input), use classifier-only output (`prediction_source == "classifier"`). Geo-filtering during classification would conflate location priors with visual evidence.

### Reuse existing code

`scripts/training/0-teacher_speciesnet_pipeline.py` already implements 225-class probability projection in its `soft_label` mode. The step-7 filter script should import and reuse that mapping logic rather than reimplementing it.

---

## Issue 4: Which Detections to Analyse

### Classify all animal detections, not just the top one

Running SpeciesNet only on the highest-confidence detection loses information about secondary animals. Instead, classify ALL MegaDetector animal detections (category `"1"`) above a minimum confidence threshold (default: `0.1`, MegaDetector's recommended floor). Sort detections by descending megadetector confidence; label them `detection_idx: 0, 1, 2, ...`.

Person and vehicle detections (categories `"2"` and `"3"`) are NOT passed to SpeciesNet. Record their bboxes and confidence, but do not classify them.

### Primary detection for match checking

For label verification in step 7, use `detection_idx: 0` — the highest-confidence *animal* detection (category `"1"`). Do not use the globally highest-confidence detection, which may be a person.

### Minimum crop size guard

SpeciesNet performs poorly on very small crops. If a crop's absolute pixel dimensions fall below a minimum (e.g., 32×32 px), skip SpeciesNet for that detection and record `speciesnet_skipped: true` with reason `"crop_too_small"`.

---

## Two-Script Design

### Script 6: `scripts/dataset_quality/6-classify_speciesnet.py`

**Purpose:** Pure data capture. Run SpeciesNet on MegaDetector crops and record raw output. No filtering logic, no 225-class mapping.

**Input:** `filter_results.jsonl` entries where `passed=true` AND `"caption_eval"` in `stages_done`

**Key design choices:**
- Read MegaDetector bboxes from the existing `detections` field — do NOT re-run MegaDetector
- Save the **full** SpeciesNet class list and score array (all ~3,537 entries), not just top-N — required for correct 225-class probability projection in step 7
- Use raw SpeciesNet class string format (`UUID;taxon;hierarchy`)
- Separate output file (`speciesnet_results.jsonl`), not merged into filter_results yet
- Resumable: skip entries already present in output

**Output schema per image in `speciesnet_results.jsonl`:**
```json
{
  "filepath": "data/inaturalist/images/red fox/img_001.jpg",
  "expected_common": "red fox",
  "expected_scientific": "vulpes vulpes",
  "speciesnet_detections": [
    {
      "detection_idx": 0,
      "bbox_norm": [0.12, 0.18, 0.54, 0.82],
      "megadetector_conf": 0.97,
      "megadetector_category": "1",
      "speciesnet_classes": [
        "uuid;mammalia;carnivora;canidae;vulpes;vulpes;red fox",
        "uuid;mammalia;carnivora;canidae;vulpes;corsac;corsac fox",
        "..."
      ],
      "speciesnet_scores": [0.91, 0.02, "..."],
      "speciesnet_top1": "uuid;mammalia;carnivora;canidae;vulpes;vulpes;red fox",
      "speciesnet_top1_score": 0.91,
      "speciesnet_prediction_source": "classifier",
      "crop_size_px": [340, 264],
      "speciesnet_skipped": false,
      "inference_ms": 45.2
    }
  ],
  "non_animal_detections": [
    {
      "detection_idx": 1,
      "megadetector_category": "2",
      "megadetector_conf": 0.72,
      "bbox_norm": [0.0, 0.05, 0.15, 0.95]
    }
  ],
  "n_animal_detections": 1,
  "n_person_detections": 1,
  "n_vehicle_detections": 0,
  "inference_total_ms": 91.4
}
```

---

### Script 7: `scripts/dataset_quality/7-filter_speciesnet.py`

**Purpose:** Post-processing and filtering decisions. CPU-only. Maps raw SpeciesNet output to 225 classes, computes match levels, applies configurable thresholds, updates `filter_results.jsonl`.

**Input:**
- `speciesnet_results.jsonl` (from step 6)
- `filter_results.jsonl` (to update in-place)
- `reports/classes_225.csv`
- Config file or CLI flags for thresholds

**Config parameters (with defaults):**
```yaml
megadetector_min_conf: 0.5         # below this, primary detection is unreliable
speciesnet_min_score: 0.3          # below this, classification is uninformative
family_match_fail_threshold: 0.5   # family match + score >= this = fail
fail_on_match_levels:              # these always fail
  - no_match
  - class
  - order
filter_multi_animal: false         # set true to exclude multi-animal images
filter_has_human: false            # set true to exclude mixed animal+human images
```

**Per-image processing:**
1. Find primary animal detection: `detection_idx=0` from `speciesnet_detections`
2. If `megadetector_conf < megadetector_min_conf` → fail with `"low_megadetector_confidence"`
3. If `speciesnet_skipped=true` for primary detection → fail with `"primary_crop_too_small"`
4. If `speciesnet_top1_score < speciesnet_min_score` → fail with `"low_speciesnet_confidence"`
5. Compute `match_level` by comparing SpeciesNet top-1 taxonomy against expected label taxonomy
6. Apply match_level rules (table in Issue 1)
7. Compute 225-class probability vector (reuse logic from `0-teacher_speciesnet_pipeline.py`)
8. Apply optional `filter_multi_animal` / `filter_has_human` checks

**Output fields added to each entry in `filter_results.jsonl`:**
```json
{
  "speciesnet_eval": {
    "pass": true,
    "reason": null,
    "primary_detection": {
      "detection_idx": 0,
      "megadetector_conf": 0.97,
      "speciesnet_top1_scientific": "vulpes vulpes",
      "speciesnet_top1_common": "red fox",
      "speciesnet_top1_score": 0.91,
      "match_level": "species",
      "matched_class_225_common": "red fox",
      "matched_class_225_idx": 42,
      "probs_225": [0.0, "...", 0.91, "..."],
      "prob_225_sum": 0.94
    },
    "n_animal_detections": 1,
    "n_person_detections": 1,
    "multi_animal": false,
    "has_human": true
  }
}
```

For rejected images, also update top-level fields:
- `passed: false`
- `stage_failed: "speciesnet"`
- `reason: "<reason string>"`
- Append `"speciesnet"` to `stages_done`

---

## Summary of Recommendations

| Question | Recommendation |
|---|---|
| How to handle SpeciesNet wrong-species predictions? | Hierarchical taxonomic match level; genus = pass with flag; family = conditional on confidence; order/class/no_match = fail |
| Should multi-animal images be filtered? | No — track with `multi_animal` flag. KD research advantage: teacher supervises all animals |
| Should human-in-image be filtered? | No — track with `has_human` flag; these images train the model's animal-priority behaviour; opt-in filter for ablation |
| How to handle the model detecting humans instead of animals? | Slight person-class penalty via class weight (0.3×) or KD downweighting; report animal-only mAP alongside full mAP |
| What if no animal detection at all? | Fail — no animal detection means the image is unsuitable regardless of label |
| How to map SpeciesNet → 225 classes? | Lookup table on genus+species (species-level) or genus (genus-level); sum scores for genus-level classes |
| Which detections to classify? | All animal detections above megadetector_conf 0.1; skip crops < 32×32 px |
| Which detection drives the label match? | Highest-confidence animal detection (detection_idx=0, category="1") |
| Save full SpeciesNet scores or top-N? | Full scores array — required for correct 225-class probability projection |
| When to apply 225-class mapping? | Step 7 only; step 6 saves raw SpeciesNet strings |
| Use SpeciesNet geo-filtering? | No — classifier-only output, per project geo-filtering policy |

---

## Notes on Research Value

### Visually similar cross-genus species

The calibration step for family-level mismatches (manually inspecting a sample of high-confidence family-match failures) is worth doing before finalising the filter thresholds. Pairs like leopard / leopard cat, or ocelot / margay, are likely to appear in the 225-class set and represent genuine visual difficulty that SpeciesNet may not resolve correctly at genus level. Documenting which pairs cause the most confusion adds species-identification difficulty context to the thesis.

### Human detection asymmetry

The deliberate person-class downweighting is a product design decision embedded in the training pipeline. It should be framed explicitly in the thesis:

> "The deployment context (AX Visio binoculars, field use, humans frequently co-present with animals) motivates a class-weight asymmetry: the person class is assigned a lower training loss weight than mammal species. This encodes the use-case priority — correct animal species identification is the objective; person detection is an incidental capability. The trade-off is lower person-class mAP in standard benchmarks, which is acceptable and reported transparently alongside animal-only mAP."

### Multi-animal KD advantage

The multi-animal property of KD deserves explicit framing in the thesis:

> "Unlike standard supervised training where unlabeled animal instances constitute false negatives and degrade the training signal, the knowledge distillation approach inherently benefits from multi-animal images: the teacher model provides bounding box and classification supervision for all visible animals, not only the labeled subject. This property — that soft-label KD inherits the teacher's detection completeness — may contribute to improved generalisation on complex natural scenes compared to direct fine-tuning."

This is testable: compare KD performance on single-animal vs. multi-animal subsets of the training data. The `multi_animal` flag in the output makes this split trivial to reproduce.
