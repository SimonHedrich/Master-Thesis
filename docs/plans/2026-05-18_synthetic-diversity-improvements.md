# Synthetic Image Diversity Improvements

**Date:** 2026-05-18  
**Status:** Implemented  
**Triggered by:** Review of 50 preview images per class revealed limited pose/orientation diversity

---

## Problem

A review of the first 50 generated images per class (via `--preview 50`) revealed three compounding issues:

1. **Pose/activity monotony** — Animals are predominantly walking or standing alert. Resting, sleeping, lying-down, grooming, digging, climbing, and other passive or species-specific behaviors are nearly absent.

2. **Camera-facing bias** — In almost all images the animal faces toward the camera. Behaviors where the animal looks away, has its back turned, or is absorbed in an activity without awareness of the observer are missing.

3. **Repeated near-identical pairs** — Within each shot group the same 5 LLM-generated behaviors cycle deterministically. With 25 slots cycling 5 descriptions, each behavior appears 5 times; with a `--preview 50` that samples every 4th image, users see 4-5 shots per behavior — nearly identical except for the background environment.

None of these were caused by the shot schedule (angle/distance breakdown is sound). All three trace to the **scene profile generation prompt** being under-specified.

---

## Root Cause

### Profile generation prompt (`build_profile_request` in `1-generate_image_list.py`)

The LLM is asked for "5 one-sentence behaviors per shot type" with no diversity constraints. It defaults to the mental model of classic wildlife photography: alertness, mid-stride locomotion, face toward camera.

Confirmed by inspection of cached profiles (`reports/synthetic_scene_profiles.json`):

**Aardwolf `eye_level` behaviors:**
- standing alert in the middle of a dirt track during the early evening  
- pauses mid-stride, body angled slightly away, observing the photographer  
- A young aardwolf rests on its haunches amidst scattered twigs  
- The subject trots along a game trail  
- stands quietly near the entrance of a disused aardvark burrow  

→ 2 walking, 2 standing, 1 sitting; no lying down; no sleeping; no foraging

**Patas monkey `eye_level` behavior #3:**
> "A primate pauses mid-stride, turning its head to look directly at the camera"

This is the camera-facing bias rendered explicit.

### Secondary: angle descriptions don't prohibit camera-facing

The `eye_level` description says nothing about head orientation. Combined with CRITICAL REQUIREMENT #2 ("diagnostic features must be clearly visible"), the image model defaults to the orientation that maximises feature visibility — face toward camera.

### Enabler: all prompt files already existed

`1-generate_image_list.py` skips `.txt` files that already exist (no `--force`). Since Stage 2 had already run for all 76 classes before any image was generated, re-running the script after fixing the profile prompt would have had zero effect. All existing `.txt` prompt files must be deleted to allow regeneration.

---

## Changes Made

### 1. `scripts/synthetic/1-generate_image_list.py` — `build_profile_request`

Added explicit per-category diversity requirements to the LLM prompt:
- ≥1 behavior per category where the animal is resting, lying, or sleeping
- ≥1 behavior per category showing active species-typical foraging/feeding/activity
- ≥1 behavior per non-frontal category where the animal is NOT facing the camera
- Behaviors must span the full activity spectrum (passive rest → alert → active)
- No more than one "walking" or "standing alert" behavior per category

Guild-specific additions:
| Guild | Required behavior |
|-------|-------------------|
| arboreal | ≥1 tree-climbing, branch-perching, or canopy-foraging |
| semi_aquatic | ≥1 wading, swimming, or water-entering |
| fossorial | ≥1 digging, burrowing, or nose-to-ground sniffing |
| primate | ≥1 social/grooming AND ≥1 climbing or leaping |
| large_grazing | ≥1 grazing/browsing AND ≥1 lying-down-to-ruminate |

### 2. `scripts/synthetic/1-generate_image_list.py` — `ANGLE_DESCRIPTIONS`

- **`eye_level`**: appended note that head orientation is NOT fixed to frontal
- **`side_profile`**: appended note that animal gaze is directed forward/sideways, not toward camera

### 3. `scripts/synthetic/1-generate_image_list.py` — `PROMPT_TEMPLATE`

Added CRITICAL REQUIREMENT #7:
> "The animal's body and head orientation must match the angle specification and behavior description above exactly. Do not reorient the animal toward the camera to improve species feature visibility; instead render the diagnostic features from the specified angle."

---

## Procedure

```bash
# 1. Clear profile cache (forces Stage 1 LLM regeneration for all 76 classes)
echo '{}' > reports/synthetic_scene_profiles.json

# 2. Delete all existing prompt text files (Stage 2 would skip them otherwise)
find data/synthetic/prompts -name "*.txt" -delete

# 3. Regenerate profiles and prompts
python scripts/synthetic/1-generate_image_list.py

# 4. Generate remaining images (already-generated images are skipped automatically)
python scripts/synthetic/2-generate_images.py
```

The 50 already-generated image files in `data/synthetic/images/` are **not affected**.
`2-generate_images.py` checks for output file existence before calling the API and skips any that already exist.

---

## What Was NOT Changed

- Shot schedule structure (angle/distance breakdown remains the same)
- Image counts per class (Band A: 200, Band B: 94)
- `2-generate_images.py`
- Any generated image files
