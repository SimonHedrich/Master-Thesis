#!/usr/bin/env python3
"""
Stage 1b — Generate fresh prompts for the 5 incumbent-cell classes with no
existing synthetic images

Materializes the 100-prompt-per-class "full" regime prompt set for the 5
classes in the model-comparison experiment
(docs/synthetic-model-comparison/01_experiment-design.md,
docs/synthetic-model-comparison/10_train-subset-incumbent-selection.md §6)
that have no existing incumbent (gemini-3.1-flash-image-preview) synthetic
images to reuse: plains zebra, mountain zebra, red fox, american black
bear, lion (all Band D — production used real photos for these, not
synthetic generation).

Adapted from scripts/synthetic/1-generate_image_list.py's Stage 2 (prompt
assembly), copied here rather than imported so this experiment's code stays
independent of the production pipeline. Two simplifications versus the
production script, both because every class here is species-level (no
genus/family fan-out):
  - No representative-species cycling / genus article prefix logic.
  - LLM scene-profile generation (Stage 1 there) is skipped entirely: all 5
    species already have a cached profile in
    reports/synthetic_scene_profiles.json, so no OPENROUTER_API_KEY is
    needed here.

The shot schedule is copied from production's BAND_B_SCHEDULE (100 shots,
val+train, full angle/distance/lighting/occlusion variety) — it already
sums to exactly the 100 images/class this experiment fixes as the train-set
control (01_experiment-design.md §5), so no new schedule needed.

Wikipedia article text (data/wikipedia/*.txt) is not checked into the repo
(data/* is gitignored) and is fetched on demand via the public MediaWiki
action API, using the canonical URLs already resolved in
reports/wikipedia_urls.json — cached to data/wikipedia/ (the same shared
location scripts/synthetic/ reads from; this is reference data, not
experiment-specific code, so it is not duplicated).

Usage:
    uv run python scripts/synthetic_model_comparison/1b-generate_prompts_fresh.py
    uv run python scripts/synthetic_model_comparison/1b-generate_prompts_fresh.py --dry-run
    uv run python scripts/synthetic_model_comparison/1b-generate_prompts_fresh.py --classes "lion,red fox"
    uv run python scripts/synthetic_model_comparison/1b-generate_prompts_fresh.py --force

Outputs:
    data/wikipedia/<wikipedia_file>.txt                                  (fetched on demand)
    data/synthetic_model_comparison/train/prompts_full/<class_slug>/<NNN>.txt
    reports/model_comparison_fresh_prompt_metadata.jsonl

Requirements:
    pip install requests
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
WIKI_URLS_JSON = REPO_ROOT / "reports" / "wikipedia_urls.json"
WIKI_DIR = REPO_ROOT / "data" / "wikipedia"
PROFILES_JSON = REPO_ROOT / "reports" / "synthetic_scene_profiles.json"
TRAIN_ROOT = REPO_ROOT / "data" / "synthetic_model_comparison" / "train"
PROMPTS_OUT_DIR = TRAIN_ROOT / "prompts_full"
METADATA_OUT_PATH = REPO_ROOT / "reports" / "model_comparison_fresh_prompt_metadata.jsonl"

TARGET_COUNT_PER_CLASS = 100

# The 5 classes with no existing incumbent synthetic images (all Band D).
CLASSES: dict[str, str] = {
    "plains zebra": "plains_zebra",
    "mountain zebra": "mountain_zebra",
    "red fox": "red_fox",
    "american black bear": "american_black_bear",
    "lion": "lion",
}
BAND = "D"

# ---------------------------------------------------------------------------
# Wikipedia fetch (minimal subset of scripts/wikipedia/2-scrape_wikipedia.py —
# the canonical URL is already resolved for these 5 species in
# wikipedia_urls.json, so no search-fallback logic is needed here).
# ---------------------------------------------------------------------------

WIKI_ACTION = "https://en.wikipedia.org/w/api.php"
WIKI_HEADERS = {
    "User-Agent": (
        "master-thesis-wildlife-synthetic-model-comparison/1.0 "
        "(simon.hedrich@inovex.de; academic research)"
    )
}
WIKI_SLEEP = 0.35


def fetch_wikipedia_text(title: str) -> Optional[str]:
    params = {
        "action": "query",
        "prop": "extracts|info",
        "explaintext": "true",
        "inprop": "url",
        "titles": title,
        "format": "json",
        "redirects": "true",
    }
    try:
        r = requests.get(WIKI_ACTION, params=params, headers=WIKI_HEADERS, timeout=30)
        r.raise_for_status()
    except requests.RequestException as exc:
        print(f"  Wikipedia fetch error for '{title}': {exc}")
        return None
    finally:
        time.sleep(WIKI_SLEEP)

    pages = r.json().get("query", {}).get("pages", {})
    for page_id, page in pages.items():
        if page_id == "-1":
            return None
        return page.get("extract") or ""
    return None


def ensure_wikipedia_text(wiki_entry: dict) -> str:
    """Return the cached article text for wiki_entry, fetching it if missing."""
    wiki_file = wiki_entry["wikipedia_file"]
    path = WIKI_DIR / wiki_file
    if path.exists():
        return path.read_text(encoding="utf-8")

    title = wiki_entry["wikipedia_url"].split("/wiki/", 1)[-1].replace("_", " ")
    print(f"  Fetching Wikipedia article '{title}' -> {path.relative_to(REPO_ROOT)}")
    text = fetch_wikipedia_text(title)
    if text is None:
        sys.exit(f"Error: could not fetch Wikipedia article '{title}' for caching to {path}")

    WIKI_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text

# ---------------------------------------------------------------------------
# Wikipedia section extraction (copied verbatim from
# scripts/synthetic/1-generate_image_list.py)
# ---------------------------------------------------------------------------

DESCRIPTION_ALIASES = frozenset({
    "description", "characteristics", "physical description",
    "physical characteristics", "appearance", "morphology",
    "anatomy", "physical appearance",
})
BEHAVIOR_ALIASES = frozenset({
    "behavior", "behaviour", "ecology", "ecology and behaviour",
    "behaviour and ecology", "ecology and behavior", "biology and behaviour",
    "biology and behavior", "habits", "ecology and biology",
    "feeding and behavior", "feeding and behaviour",
})
HABITAT_ALIASES = frozenset({
    "distribution and habitat", "habitat and distribution", "habitat and range",
    "habitat", "distribution", "range", "range and habitat",
    "range and distribution", "range and ecology",
})


def _section_depth(header: str) -> int:
    m = re.match(r"^(=+)", header)
    return len(m.group(1)) if m else 0


def extract_sections(text: str) -> tuple[str, str, str]:
    """Return (description_text, behavior_text, habitat_text) from a Wikipedia .txt article."""
    parts = re.split(r"^(==+[^=\n]+==+)", text, flags=re.MULTILINE)
    lead = parts[0].strip()

    sections: list[tuple[str, int, str]] = []
    for i in range(1, len(parts) - 1, 2):
        header_raw = parts[i].strip()
        content = parts[i + 1] if i + 1 < len(parts) else ""
        depth = _section_depth(header_raw)
        name = re.sub(r"^=+\s*|\s*=+$", "", header_raw).strip()
        sections.append((name, depth, content))

    def collect(aliases: frozenset) -> str:
        result = []
        collecting = False
        base_depth = 0
        for name, depth, content in sections:
            if not collecting:
                if name.lower() in aliases:
                    collecting = True
                    base_depth = depth
                    result.append(content)
            else:
                if depth <= base_depth:
                    break
                result.append(content)
        return "\n".join(result).strip()

    desc = collect(DESCRIPTION_ALIASES) or lead[:800]
    behavior = collect(BEHAVIOR_ALIASES)
    habitat = collect(HABITAT_ALIASES)
    return desc, behavior, habitat

# ---------------------------------------------------------------------------
# Lookup tables (copied verbatim from scripts/synthetic/1-generate_image_list.py)
# ---------------------------------------------------------------------------

ANGLE_DESCRIPTIONS = {
    "eye_level": (
        "Camera at the same height as the animal's mid-body, horizontal perspective. "
        "The animal's head orientation is NOT fixed to frontal; it may face in any "
        "direction (away from camera, in profile, or toward camera) as determined by "
        "the behavior description."
    ),
    "low": (
        "Camera below the animal, looking slightly upward at approximately 20–30° from the ground"
    ),
    "high": (
        "Camera from an elevated natural vantage point (hillside, rock outcrop, vehicle roof), "
        "oblique downward angle; full body remains visible"
    ),
    "side_profile": (
        "Pure lateral view, camera perpendicular to the animal's long axis. "
        "The animal's gaze is directed forward along its path or to the side — "
        "not turned toward the camera."
    ),
    "three_quarter_front": (
        "Approximately 45° front-diagonal view, between full-frontal and side profile"
    ),
    "three_quarter_rear": (
        "Approximately 45° rear-diagonal view, between full-rear and side profile"
    ),
    "head_on": "Full frontal, direct face-to-camera view",
    "rear": (
        "Whole animal facing away or walking away from camera; full body visible from rear aspect"
    ),
    "closeup_head": "Tight frame on head and face; head fills approximately 60–70% of the frame",
}

DISTANCE_DESCRIPTIONS = {
    "close": "Very close — animal fills approximately 50–70% of frame (telephoto at ~20–80 m)",
    "medium": "Standard field view — animal fills approximately 20–40% of frame (~80–250 m telephoto)",
    "far": "Animal at distance — fills less than 10% of frame (>250 m); habitat context dominant",
    "medium-close": "Between medium and close — animal fills approximately 35–50% of frame",
    "medium-far": "Between medium and far — animal fills approximately 15–25% of frame",
    "close-medium": "Prominent in frame — animal fills approximately 40–55% of frame (binocular portrait)",
    "varies": "Distance appropriate for the scene; natural field distance for this shot type",
}

LIGHTING_POOL = ["golden_hour", "overcast", "midday", "dappled", "backlit"]
VAL_LIGHTING_POOL = ["overcast", "golden_hour"]
LIGHTING_DESCRIPTIONS = {
    "golden_hour": "Warm directional golden-hour light, long shadows, rich warm tones",
    "overcast": "Soft diffuse overcast light, no hard shadows, even exposure across the scene",
    "midday": "Harsh overhead midday sunlight, strong shadows",
    "dappled": "Intermittent dappled light through canopy or foliage",
    "backlit": "Sun behind the animal, rim lighting effect, partial silhouette",
}

OCCLUSION_DESCRIPTIONS = {
    "none": "Animal fully visible, no obstruction",
    "partial_vegetation": (
        "20–40% of the body occluded by grasses, branches, or leaves; "
        "animal clearly identifiable throughout"
    ),
    "semi_submerged": "Lower body in water; upper body, back, and head visible above surface",
}

VAL_BEHAVIOR_DESCRIPTIONS = {
    "standing_alert": (
        "{name} standing fully upright and alert, head raised, ears forward, "
        "scanning the surroundings with calm attentiveness"
    ),
    "walking": (
        "{name} walking forward at a natural unhurried pace, mid-stride, "
        "relaxed natural gait"
    ),
    "eating_foraging": (
        "{name} actively foraging and feeding naturally on species-appropriate food "
        "in its primary habitat"
    ),
    "resting": (
        "{name} resting comfortably, lying down or sitting, in a fully relaxed posture"
    ),
    "looking_at_camera": (
        "{name} standing or sitting calmly, looking directly at the camera "
        "with quiet awareness"
    ),
}

VARIES_ANGLES = [
    "eye_level", "low", "high", "side_profile",
    "three_quarter_front", "head_on", "rear", "three_quarter_rear",
    "low", "three_quarter_front",
]
SPECIES_SPECIFIC_ANGLES = [
    "eye_level", "low", "high", "side_profile", "head_on", "three_quarter_front",
]

# ---------------------------------------------------------------------------
# Shot schedule — copied from production's BAND_B_SCHEDULE (sums to exactly
# 100, which is this experiment's fixed train-set-per-class control).
# ---------------------------------------------------------------------------

@dataclass
class ShotGroup:
    shot_type: str
    distance: str
    count: int
    split: str
    occlusion: str = "none"
    val_behavior_code: Optional[str] = None


SHOT_SCHEDULE: list[ShotGroup] = [
    # Val (images 001–015 and 074–078) — 20 images per class, 20% of 100
    ShotGroup("eye_level",           "medium",     15, "val"),   # 001–015
    # Train (images 016–073, 079–100)
    ShotGroup("low",                 "medium",     12, "train"), # 016–027
    ShotGroup("high",                "medium",     12, "train"), # 028–039
    ShotGroup("head_on",             "medium",     12, "train"), # 040–051
    ShotGroup("rear",                "medium",     12, "train"), # 052–063
    ShotGroup("side_profile",        "medium-far", 10, "train"), # 064–073
    ShotGroup("three_quarter_front", "medium",      5, "val"),   # 074–078
    ShotGroup("three_quarter_front", "medium",      3, "train"), # 079–081
    ShotGroup("eye_level",           "far",         8, "train"), # 082–089
    ShotGroup("varies",              "medium",      5, "train", occlusion="partial_vegetation"),
    ShotGroup("species_specific",    "varies",      6, "train"), # 095–100
]
assert sum(g.count for g in SHOT_SCHEDULE) == TARGET_COUNT_PER_CLASS

# ---------------------------------------------------------------------------
# Prompt template (copied verbatim from scripts/synthetic/1-generate_image_list.py)
# ---------------------------------------------------------------------------

PHOTOGRAPHY_STYLE_BOKEH = (
    "Telephoto lens (400–600 mm equivalent), natural shallow depth of field "
    "with background softly blurred, authentic field conditions."
)

PHOTOGRAPHY_STYLE_NO_BOKEH = (
    "Wildlife observation through optical binoculars (8–10× magnification), "
    "full-scene sharp focus, authentic field conditions. "
    "No background blur — the animal is integrated into its environment, "
    "not isolated against a softened background."
)

PROMPT_TEMPLATE = """\
TASK: Generate a single photorealistic wildlife photograph for use in a computer vision object detection training dataset. The photograph must look like an authentic field photograph taken by a wildlife photographer or camera trap.

SUBJECT: {subject_line}

SPECIES DESCRIPTION:
{desc_text}

NATURAL BEHAVIOR AND HABITAT:
{behavior_text}
{habitat_text}

SCENE SPECIFICATION:
- Camera angle / perspective: {angle_description}
- Distance from subject: {distance_description}
- Animal pose / behavior: {behavior_description}
- Environment / background: {environment_description}
- Lighting conditions: {lighting_description}
- Occlusion / visibility: {occlusion_description}
- Special focus: {focus_note}

PHOTOGRAPHY STYLE: Documentary wildlife photography, comparable in quality and style to BBC Natural World or National Geographic. {photography_style} No studio lighting, no artificial backgrounds, no text overlays, no watermarks, no borders, no frames, no human-made objects unless contextually relevant to the scene.

CRITICAL REQUIREMENTS:
1. The {subject_name} must be the unmistakable primary subject of the photograph.
2. The following species-diagnostic features must be clearly visible and anatomically correct: {key_diagnostic_features}
3. The animal's body proportions, coloration, and posture must match the species description above exactly.
4. The scene must be ecologically plausible — the environment, weather, and behavior must all occur naturally for this species.
5. The photograph must be suitable for training an automated species identification system.
6. The animal must be fully visible within the frame — no part of its body, head, or limbs should be cut off at the image edges.
7. The animal's body and head orientation must match the angle specification and behavior description above exactly. Do not reorient the animal toward the camera to improve species feature visibility; instead render the diagnostic features from the specified angle.
8. Exactly one individual {subject_name} must appear in the photograph. No other individual of the same species should be clearly visible in the frame. Other species may appear incidentally in the distant background.\
"""


def resolve_angle(shot_type: str, slot_idx: int) -> str:
    if shot_type == "varies":
        return VARIES_ANGLES[slot_idx % len(VARIES_ANGLES)]
    if shot_type == "species_specific":
        return SPECIES_SPECIFIC_ANGLES[slot_idx % len(SPECIES_SPECIFIC_ANGLES)]
    return shot_type


def build_prompt(
    *,
    subject_line: str,
    subject_name: str,
    desc_text: str,
    behavior_text: str,
    habitat_text: str,
    angle_code: str,
    distance_code: str,
    lighting_code: str,
    occlusion_code: str,
    behavior_description: str,
    environment_description: str,
    focus_note: str,
    key_diagnostic_features: str,
) -> str:
    return PROMPT_TEMPLATE.format(
        subject_line=subject_line,
        subject_name=subject_name,
        desc_text=desc_text.strip() or f"[Species description not available for {subject_name}]",
        behavior_text=behavior_text.strip(),
        habitat_text=habitat_text.strip(),
        angle_description=ANGLE_DESCRIPTIONS.get(angle_code, angle_code),
        distance_description=DISTANCE_DESCRIPTIONS.get(distance_code, distance_code),
        behavior_description=behavior_description,
        environment_description=environment_description,
        lighting_description=LIGHTING_DESCRIPTIONS.get(lighting_code, lighting_code),
        occlusion_description=OCCLUSION_DESCRIPTIONS.get(occlusion_code, occlusion_code),
        focus_note=focus_note,
        key_diagnostic_features=key_diagnostic_features,
        photography_style=PHOTOGRAPHY_STYLE_NO_BOKEH,
    )

# ---------------------------------------------------------------------------
# Per-class prompt generation
# ---------------------------------------------------------------------------

def build_class_records(
    class_name: str,
    slug: str,
    wiki_entry: dict,
    profile: dict,
) -> list[dict]:
    """Build the 100 prompt texts + metadata records for one class."""
    scientific = wiki_entry["scientific_name"]
    subject_line = f"{class_name} ({scientific})"
    subject_name = class_name

    text = ensure_wikipedia_text(wiki_entry)
    desc_text, behavior_text, habitat_text = extract_sections(text)

    key_features = profile.get("key_diagnostic_features", class_name)
    environments = profile.get("environments") or ["natural habitat"]
    behaviors = profile.get("behaviors", {})
    focus_notes = profile.get("focus_notes", {})

    records: list[dict] = []
    global_idx = 0  # 0-based image counter within this class

    for group in SHOT_SCHEDULE:
        for slot_idx in range(group.count):
            image_num = global_idx + 1
            angle_code = resolve_angle(group.shot_type, slot_idx)

            if group.val_behavior_code is not None:
                lighting_code = VAL_LIGHTING_POOL[global_idx % len(VAL_LIGHTING_POOL)]
            else:
                lighting_code = LIGHTING_POOL[global_idx % len(LIGHTING_POOL)]

            environment_description = environments[global_idx % len(environments)]

            if group.val_behavior_code is not None:
                behavior_description = VAL_BEHAVIOR_DESCRIPTIONS[
                    group.val_behavior_code
                ].format(name=subject_name)
            else:
                behavior_key = group.shot_type
                if group.shot_type == "varies":
                    behavior_key = angle_code
                if group.shot_type == "species_specific":
                    behavior_key = "species_specific"
                b_pool = behaviors.get(behavior_key) or behaviors.get("eye_level") or [subject_name]
                behavior_description = b_pool[slot_idx % len(b_pool)]

            if group.shot_type == "closeup_head":
                focus_note = focus_notes.get("closeup_head", focus_notes.get("default", ""))
            else:
                focus_note = focus_notes.get("default", "")

            occlusion_code = group.occlusion

            prompt_text = build_prompt(
                subject_line=subject_line,
                subject_name=subject_name,
                desc_text=desc_text,
                behavior_text=behavior_text,
                habitat_text=habitat_text,
                angle_code=angle_code,
                distance_code=group.distance,
                lighting_code=lighting_code,
                occlusion_code=occlusion_code,
                behavior_description=behavior_description,
                environment_description=environment_description,
                focus_note=focus_note,
                key_diagnostic_features=key_features,
            )

            prompt_rel_path = PROMPTS_OUT_DIR / slug / f"{image_num:03d}.txt"
            records.append({
                "class": class_name,
                "band": BAND,
                "index": image_num,
                "filename": f"{BAND.lower()}_{slug}_{image_num:03d}.png",
                "shot_type": angle_code,
                "distance": group.distance,
                "lighting": lighting_code,
                "occlusion": occlusion_code,
                "pose": behavior_description,
                "environment": environment_description,
                "source_split": group.split,
                "prompt_file": str(prompt_rel_path.relative_to(REPO_ROOT)),
                "prompt_text": prompt_text,
            })
            global_idx += 1

    assert global_idx == TARGET_COUNT_PER_CLASS
    return records


def write_class_prompts(slug: str, records: list[dict], dry_run: bool) -> None:
    if dry_run:
        return
    class_dir = PROMPTS_OUT_DIR / slug
    class_dir.mkdir(parents=True, exist_ok=True)
    for rec in records:
        (REPO_ROOT / rec["prompt_file"]).write_text(rec["prompt_text"], encoding="utf-8")


def load_existing_metadata() -> list[dict]:
    if not METADATA_OUT_PATH.exists():
        return []
    records = []
    with open(METADATA_OUT_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_metadata(records: list[dict]) -> None:
    METADATA_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METADATA_OUT_PATH, "w", encoding="utf-8") as f:
        for rec in records:
            # prompt_text is not part of the sidecar schema (it lives in the .txt file).
            out = {k: v for k, v in rec.items() if k != "prompt_text"}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--classes",
        default=None,
        help="Comma-separated subset of class common names to process (default: all 5).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate prompt files even if 100 already exist for a class.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print the plan without fetching Wikipedia text or writing files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    classes = CLASSES
    if args.classes:
        requested = {c.strip().lower() for c in args.classes.split(",")}
        classes = {name: slug for name, slug in CLASSES.items() if name.lower() in requested}
        not_found = requested - {name.lower() for name in classes}
        if not_found:
            print(f"Warning: class(es) not found: {', '.join(sorted(not_found))}")
        if not classes:
            sys.exit("No matching classes found.")

    with open(WIKI_URLS_JSON, encoding="utf-8") as f:
        wiki_urls: dict = json.load(f)
    wiki_by_common_name = {
        v.get("common_name", "").strip().lower(): v for v in wiki_urls.values()
    }

    with open(PROFILES_JSON, encoding="utf-8") as f:
        scene_profiles: dict = json.load(f)

    existing_metadata = load_existing_metadata()
    metadata_by_class: dict[str, list[dict]] = {}
    for rec in existing_metadata:
        metadata_by_class.setdefault(rec["class"], []).append(rec)

    print(f"{'class':22s} {'slug':22s} {'shots':>6s}")
    all_new_records: dict[str, list[dict]] = {}

    for class_name, slug in classes.items():
        class_dir = PROMPTS_OUT_DIR / slug
        existing_count = len(list(class_dir.glob("*.txt"))) if class_dir.exists() else 0
        if not args.force and existing_count >= TARGET_COUNT_PER_CLASS:
            print(f"{class_name:22s} {slug:22s} {existing_count:6d}  (already complete, skipping)")
            continue

        wiki_entry = wiki_by_common_name.get(class_name)
        if wiki_entry is None:
            sys.exit(f"Error: '{class_name}' not found in {WIKI_URLS_JSON}")
        profile = scene_profiles.get(class_name)
        if profile is None:
            sys.exit(f"Error: '{class_name}' has no cached scene profile in {PROFILES_JSON}")

        if args.dry_run:
            print(f"{class_name:22s} {slug:22s} {TARGET_COUNT_PER_CLASS:6d}  (dry-run, not built)")
            continue

        records = build_class_records(class_name, slug, wiki_entry, profile)
        write_class_prompts(slug, records, dry_run=args.dry_run)
        all_new_records[class_name] = records
        print(f"{class_name:22s} {slug:22s} {len(records):6d}")

    if args.dry_run:
        print("\n--dry-run: no Wikipedia fetches, no files written.")
        return

    if not all_new_records:
        print("\nNothing to do.")
        return

    for class_name, records in all_new_records.items():
        metadata_by_class[class_name] = [
            {k: v for k, v in rec.items() if k != "prompt_text"} for rec in records
        ]

    combined = [rec for recs in metadata_by_class.values() for rec in recs]
    write_metadata(combined)

    total_new = sum(len(r) for r in all_new_records.values())
    print(f"\nWrote {total_new} prompt files across {len(all_new_records)} class(es).")
    print(f"Wrote {len(combined)} total records to {METADATA_OUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
