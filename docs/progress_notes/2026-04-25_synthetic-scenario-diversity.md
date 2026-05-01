# Progress Notes – 25.04.2026

## Synthetic Image Diversity: Scenario Pre-Generation Strategy

**Context:** Initial synthetic image batches are visually repetitive — the Gemini model produces images with consistent posture, position, and background per species because all N calls for a species use the same static prompt. This note documents the problem, a prompt-length feasibility check, and the chosen solution.

---

## 1. The Repetition Problem

The current `build_prompt()` function in `scripts/synthetic/1-generate_synthetic_images.py` constructs the same text for every image of a species:

```
Generate a realistic wildlife photograph of a {class_name}.
The animal has these characteristics: {description[:600]}.
Professional wildlife photograph. Telephoto lens, sharp focus... [style suffix]
```

No scenario context is injected. The model defaults to whatever "canonical" pose it associates with the species — typically a lateral view in a generic habitat at neutral lighting. The result is a training set with low intra-class variation in pose, viewpoint, and background despite covering many different species.

This is a real problem for object detection training: models trained on low-diversity synthetic data tend to overfit to canonical poses and fail on real-world viewpoint variation.

---

## 2. Gemini Prompt Length: Feasibility of Rich Prompts

The script uses `google/gemini-3.1-flash-image-preview` via OpenRouter — this is Gemini's **native image generation** capability (not the separate Imagen API). Prompt length limits by service:

| Service | Character limit | Token limit | Notes |
|---------|:--------------:|:-----------:|-------|
| **Gemini native image gen** | **32,000** | – | The model used in this project |
| Gemini Imagen API | ~2,000 | 480 | Separate, more restricted product |
| Midjourney | 6,000 | – | Effective limit ~40–60 words |
| DALL-E 3 | 1,000 | ~256 | GPT-4 rewrites before generation |
| Stable Diffusion | ~380 | **77** | Hard CLIP encoder constraint |

The 32,000-character budget confirms that adding rich scene descriptions (~200–400 characters) to prompts is completely feasible with Gemini and should be fully attended to by the model.

---

## 3. Why Generic Scenario Matrices Fall Short

The experimental design (`2026-04-24_synthetic-data-kd-experimental-design.md`) already targets 7 habitats × 4 viewpoints × 3 lighting = 84 unique prompt combinations. The issue is that a generic matrix generates ecologically nonsensical combinations:

- "Aye-aye (nocturnal arboreal primate) grazing in open savanna at noon"
- "Pangolin swimming in alpine lake"
- "Saiga antelope perched on a rainforest canopy branch"

Applying the generic matrix without per-species filtering would either require substantial manual curation or produce training images that actively harm generalisation (the model could learn to associate strange habitats with certain species).

---

## 4. Solutions Considered

### Option A — Generic matrix (no LLM)
Pre-enumerate 84 scenario strings from (habitat, viewpoint, lighting) permutations. Apply identically to all species.
- **Pro:** Zero additional API cost; trivial to implement.
- **Con:** Ecologically nonsensical for many combinations. Manual per-species filtering is more effort than the LLM approach and less thorough.

### Option B — LLM pre-generation of per-species scenario bank ✓ **Recommended**
A one-time batch script calls a cheap text LLM (e.g. Gemini 2.5 Flash text via OpenRouter) for each target species. The LLM generates N ecologically realistic, visually diverse scenario descriptions specific to that species. Results are stored in `reports/animal_scenario_prompts.json`. The image generation script cycles through these at runtime.

- **Pro:** Animal-specific; fully automated; one-time cost negligible (~15 text API calls for Tier C); reproducible and auditable; scales to all 225 classes if needed later.
- **Con:** Adds one setup step and one data file.

### Option C — Per-image on-the-fly scenario generation
One text LLM call precedes each image generation call.
- **Pro:** Maximum freshness.
- **Con:** 2× API calls and latency per image; harder to audit or reproduce.

### Option D — Ecological guild templates
Manually classify species into guilds (arboreal, fossorial, aquatic, cursorial) and write guild-specific scenario lists.
- **Pro:** Reusable.
- **Con:** Manual classification effort; still too coarse for species-specific realism.

---

## 5. Recommended Solution: Option B

**Why:** The LLM already "knows" each species' ecology, habitat, activity period, and typical behaviours. A single API call asking for 90 diverse scenarios for "snow leopard (Panthera uncia)" will naturally produce rocky alpine settings, crepuscular lighting, camouflaged poses, and stalking behaviour — without any manual curation. For 12–15 Tier C species, total text API cost is a few cents.

**Implementation:** Two scripts; `0-generate_scenario_prompts.py` runs once, producing `reports/animal_scenario_prompts.json`. The modified `1-generate_synthetic_images.py` accepts `--scenarios-file` and cycles through the pre-generated scenarios per image. See the script files for full details.

**LLM prompt strategy for scenario generation:**

The prompt instructs the model to vary across five dimensions simultaneously:
1. **Behaviour** — foraging, resting, alert, moving, nursing young, vocalising
2. **Substrate/location** — species-appropriate microhabitat (burrow entrance, rocky ledge, reed bed, canopy)
3. **Viewpoint** — frontal, lateral, three-quarter, overhead, from below
4. **Lighting** — golden hour, overcast, dappled forest light, moonlit, harsh midday
5. **Season/weather** — dry season, after rainfall, snow, leaf-off winter

Requesting 90 scenarios (6 more than the 84 needed) gives a small rejection buffer for any implausible outputs after review.

---

## 6. Next Steps

1. Run `scripts/synthetic/0-generate_scenario_prompts.py` for the 12 Tier C species.
2. Spot-check generated scenarios for ecological plausibility.
3. Run `1-generate_synthetic_images.py` with `--scenarios-file` for 2–3 trial species; visually verify diversity.
4. If quality is confirmed, run full Tier C batch (168 images × 12–15 species).
