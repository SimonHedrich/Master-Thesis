"""
Generate per-image prompt files and index.jsonl for the synthetic wildlife image dataset.

Pipeline:
  Stage 1 — LLM (OpenRouter) generates a structured scene profile per species, cached to
             reports/synthetic_scene_profiles.json. Skipped on re-run unless --force.
  Stage 2 — Deterministically expands the shot schedule into prompt .txt files and writes
             data/synthetic/index.jsonl in a single final pass.

Outputs:
    data/synthetic/prompts/{class_slug}/{nnn:03d}.txt  — one prompt file per image
    data/synthetic/index.jsonl                         — metadata index (no prompt text)
    reports/synthetic_scene_profiles.json              — cached LLM scene profiles

Usage:
    # Full run (requires OPENROUTER_API_KEY in .env or environment):
    uv run python scripts/synthetic/1-generate_image_list.py

    # Smoke-test two classes without API key:
    uv run python scripts/synthetic/1-generate_image_list.py --classes walrus,kinkajou --skip-llm

    # Regenerate prompts for one class:
    uv run python scripts/synthetic/1-generate_image_list.py --classes aardvark --force

Requirements:
    pip install requests python-dotenv
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from math import ceil
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DESCRIPTIONS_CSV = PROJECT_ROOT / "reports" / "animal_descriptions.csv"
WIKI_URLS_JSON = PROJECT_ROOT / "reports" / "wikipedia_urls.json"
WIKI_DIR = PROJECT_ROOT / "data" / "wikipedia"
PROFILES_JSON = PROJECT_ROOT / "reports" / "synthetic_scene_profiles.json"
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"
INDEX_JSONL = SYNTHETIC_DIR / "index.jsonl"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemini-3.1-flash-lite"
MAX_TRIES = 3
RETRY_DELAY = 5

# ---------------------------------------------------------------------------
# Wikipedia section aliases
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

# ---------------------------------------------------------------------------
# Lookup tables
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

# Angle codes to cycle through for "varies" shot groups (partial vegetation)
VARIES_ANGLES = [
    "eye_level", "low", "high", "side_profile",
    "three_quarter_front", "head_on", "rear", "three_quarter_rear",
    "low", "three_quarter_front",
]
# Angle codes for Band B species_specific group
SPECIES_SPECIFIC_ANGLES = [
    "eye_level", "low", "high", "side_profile", "head_on", "three_quarter_front",
]

PINNIPED_OVERRIDE = [
    {"scientific_name": "zalophus californianus", "common_name": "California Sea Lion",  "wikipedia_file": "zalophus_californianus.txt"},
    {"scientific_name": "arctocephalus pusillus",  "common_name": "Cape Fur Seal",        "wikipedia_file": "arctocephalus_pusillus.txt"},
    {"scientific_name": "mirounga leonina",        "common_name": "Southern Elephant Seal","wikipedia_file": "mirounga_leonina.txt"},
    {"scientific_name": "mirounga angustirostris", "common_name": "Northern Elephant Seal","wikipedia_file": "mirounga_angustirostris.txt"},
    {"scientific_name": "phoca vitulina",          "common_name": "Harbour Seal",          "wikipedia_file": "phoca_vitulina.txt"},
    {"scientific_name": "halichoerus grypus",      "common_name": "Grey Seal",             "wikipedia_file": "halichoerus_grypus.txt"},
]

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ClassConfig:
    common_name: str    # matches animal_descriptions.csv and wikipedia_urls.json
    band: str           # "A" or "B"
    guilds: list[str]   # informational; used in fallback profiles


@dataclass
class ShotGroup:
    shot_type: str
    distance: str
    count: int
    split: str
    occlusion: str = "none"
    val_behavior_code: Optional[str] = None  # set for val groups

# ---------------------------------------------------------------------------
# Shot schedules
# ---------------------------------------------------------------------------

BAND_A_SCHEDULE: list[ShotGroup] = [
    # Val (images 001–040)
    ShotGroup("eye_level",           "medium", 8, "val", val_behavior_code="standing_alert"),
    ShotGroup("eye_level",           "medium", 8, "val", val_behavior_code="walking"),
    ShotGroup("three_quarter_front", "medium", 8, "val", val_behavior_code="eating_foraging"),
    ShotGroup("eye_level",           "medium", 8, "val", val_behavior_code="resting"),
    ShotGroup("three_quarter_front", "medium", 8, "val", val_behavior_code="looking_at_camera"),
    # Train (images 041–200)
    ShotGroup("eye_level",           "medium",       25, "train"),
    ShotGroup("low",                 "medium-close", 20, "train"),
    ShotGroup("high",                "medium",       15, "train"),
    ShotGroup("head_on",             "medium-close", 15, "train"),
    ShotGroup("rear",                "medium",       15, "train"),
    ShotGroup("side_profile",        "medium-far",   15, "train"),
    ShotGroup("three_quarter_rear",  "medium",       10, "train"),
    ShotGroup("three_quarter_front", "medium",       10, "train"),
    ShotGroup("eye_level",           "far",          10, "train"),
    ShotGroup("eye_level",           "close-medium", 15, "train"),
    ShotGroup("varies",              "medium",       10, "train", occlusion="partial_vegetation"),
]

BAND_B_SCHEDULE: list[ShotGroup] = [
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

# ---------------------------------------------------------------------------
# Class configuration (76 entries)
# ---------------------------------------------------------------------------

BAND_A_CLASSES: list[ClassConfig] = [
    ClassConfig("walrus",                    "A", ["fully_aquatic"]),
    ClassConfig("old world porcupine family","A", ["fossorial", "terrestrial"]),
    ClassConfig("raccoon dog",               "A", ["terrestrial", "fossorial"]),
    ClassConfig("callicebus genus",          "A", ["arboreal", "primate"]),
    ClassConfig("wild cat",                  "A", ["terrestrial"]),
    ClassConfig("black-backed jackal",       "A", ["arid_savanna"]),
    ClassConfig("ringtail",                  "A", ["arboreal", "fossorial"]),
    ClassConfig("kinkajou",                  "A", ["arboreal"]),
    ClassConfig("genet genus",               "A", ["arboreal", "terrestrial"]),
    ClassConfig("leopardus species",         "A", ["terrestrial"]),
    ClassConfig("water deer",                "A", ["large_grazing"]),
    ClassConfig("eurasian badger",           "A", ["fossorial"]),
    ClassConfig("nine-banded armadillo",     "A", ["fossorial"]),
    ClassConfig("sloth bear",                "A", ["arboreal", "terrestrial"]),
    ClassConfig("yak",                       "A", ["large_grazing", "cold_climate"]),
    ClassConfig("fisher",                    "A", ["arboreal", "terrestrial"]),
    ClassConfig("striped hyaena",            "A", ["arid_savanna"]),
    ClassConfig("asiatic black bear",        "A", ["arboreal", "terrestrial"]),
    ClassConfig("leopard cat",               "A", ["terrestrial"]),
    ClassConfig("cephalophus species",       "A", ["large_grazing"]),
    ClassConfig("ocelot",                    "A", ["terrestrial"]),
    ClassConfig("domestic water buffalo",    "A", ["large_grazing"]),
    ClassConfig("sun bear",                  "A", ["arboreal", "terrestrial"]),
    ClassConfig("asiatic wild ass",          "A", ["large_grazing"]),
    ClassConfig("maned wolf",                "A", ["arid_savanna"]),
    ClassConfig("honey badger",              "A", ["fossorial", "arid_savanna"]),
    ClassConfig("fossa",                     "A", ["arboreal", "terrestrial"]),
    ClassConfig("brown hyaena",              "A", ["arid_savanna"]),
    ClassConfig("red brocket",               "A", ["large_grazing"]),
    ClassConfig("pinniped clade",            "A", ["fully_aquatic"]),
    ClassConfig("saiga",                     "A", ["large_grazing"]),
    ClassConfig("wolverine",                 "A", ["cold_climate"]),
    ClassConfig("pangolin family",           "A", ["fossorial", "terrestrial"]),
    ClassConfig("mangabeys genus",           "A", ["arboreal", "primate"]),
    ClassConfig("red river hog",             "A", ["large_grazing"]),
    ClassConfig("aardwolf",                  "A", ["fossorial", "arid_savanna"]),
    ClassConfig("bongo",                     "A", ["large_grazing"]),
    ClassConfig("binturong",                 "A", ["arboreal"]),
    ClassConfig("aardvark",                  "A", ["fossorial"]),
    ClassConfig("spilogale species",         "A", ["fossorial"]),
    ClassConfig("red-necked wallaby",        "A", ["large_grazing"]),
    ClassConfig("clouded leopard",           "A", ["arboreal"]),
    ClassConfig("malay tapir",               "A", ["semi_aquatic"]),
    ClassConfig("aye-aye",                   "A", ["arboreal", "primate"]),
    ClassConfig("drill",                     "A", ["arboreal", "primate"]),
    ClassConfig("domestic pig",              "A", ["large_grazing"]),
    ClassConfig("giant armadillo",           "A", ["fossorial"]),
    ClassConfig("hog badger genus",          "A", ["fossorial"]),
    ClassConfig("african civet",             "A", ["terrestrial"]),
    ClassConfig("mouflon",                   "A", ["large_grazing"]),
]

BAND_B_CLASSES: list[ClassConfig] = [
    ClassConfig("canada lynx",      "B", ["terrestrial"]),
    ClassConfig("spectacled bear",  "B", ["arboreal", "terrestrial"]),
    ClassConfig("caracal",          "B", ["terrestrial"]),
    ClassConfig("eurasian lynx",    "B", ["terrestrial"]),
    ClassConfig("black wildebeest", "B", ["large_grazing"]),
    ClassConfig("giant panda",      "B", ["arboreal", "terrestrial"]),
    ClassConfig("serval",           "B", ["terrestrial"]),
    ClassConfig("patas monkey",     "B", ["arboreal", "primate"]),
    ClassConfig("american mink",    "B", ["semi_aquatic"]),
    ClassConfig("gerenuk",          "B", ["large_grazing"]),
    ClassConfig("dhole",            "B", ["arid_savanna"]),
    ClassConfig("bat-eared fox",    "B", ["arid_savanna"]),
    ClassConfig("baird's tapir",    "B", ["semi_aquatic"]),
    ClassConfig("grevy's zebra",    "B", ["large_grazing"]),
    ClassConfig("asian elephant",   "B", ["large_grazing"]),
    ClassConfig("kirk's dik-dik",   "B", ["large_grazing"]),
    ClassConfig("american badger",  "B", ["fossorial"]),
    ClassConfig("chimpanzee",       "B", ["arboreal", "primate"]),
    ClassConfig("african wild dog", "B", ["arid_savanna"]),
    ClassConfig("glaucomys species","B", ["arboreal"]),
    ClassConfig("common wombat",    "B", ["fossorial"]),
    ClassConfig("european bison",   "B", ["large_grazing"]),
    ClassConfig("lowland tapir",    "B", ["semi_aquatic"]),
    ClassConfig("tayra",            "B", ["arboreal", "terrestrial"]),
    ClassConfig("eurasian otter",   "B", ["semi_aquatic"]),
    ClassConfig("springbok",        "B", ["large_grazing"]),
]

ALL_CLASSES = BAND_A_CLASSES + BAND_B_CLASSES

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_animal_descriptions(csv_path: Path) -> dict[str, dict]:
    rows = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[row["common_name"].strip().lower()] = row
    return rows


def load_wikipedia_urls(json_path: Path) -> dict:
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def build_common_name_index(wiki_urls: dict) -> dict[str, tuple[str, dict]]:
    """Returns dict[lower(common_name) → (wiki_key, wiki_entry)]."""
    idx = {}
    for key, entry in wiki_urls.items():
        cn = entry.get("common_name", "").strip().lower()
        if cn:
            idx[cn] = (key, entry)
    return idx

# ---------------------------------------------------------------------------
# Wikipedia section extraction
# ---------------------------------------------------------------------------

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


def load_wiki_sections(wikipedia_file: str) -> tuple[str, str, str]:
    path = WIKI_DIR / wikipedia_file
    if not path.exists():
        return "", "", ""
    return extract_sections(path.read_text(encoding="utf-8"))

# ---------------------------------------------------------------------------
# Representative species for genus/family classes
# ---------------------------------------------------------------------------

def get_representative_species(wiki_entry: dict, class_common_name: str) -> list[dict]:
    top = wiki_entry.get("top_species") or []
    if not top:
        if class_common_name.lower() == "pinniped clade":
            return PINNIPED_OVERRIDE
        # Species-level: single entry
        return [{"scientific_name": wiki_entry["scientific_name"],
                 "common_name":     wiki_entry["common_name"],
                 "wikipedia_file":  wiki_entry.get("wikipedia_file", "")}]
    return top

# ---------------------------------------------------------------------------
# LLM scene profile generation (Stage 1)
# ---------------------------------------------------------------------------

PROFILE_SYSTEM_PROMPT = (
    "You are an expert wildlife biologist and nature photographer specialising in computer "
    "vision training data. You write ecologically precise, anatomically accurate scene "
    "descriptions for AI image generation of wildlife photographs."
)

REQUIRED_BEHAVIOR_KEYS = {
    "closeup_head", "eye_level", "low", "high", "head_on", "rear",
    "side_profile", "three_quarter_front", "three_quarter_rear",
    "partial_vegetation", "species_specific",
}


def build_profile_request(common_name: str, scientific_name: str,
                          condensed: str, characteristics: str) -> str:
    return (
        f"Species: {common_name} ({scientific_name})\n"
        f"Summary: {condensed}\n"
        f"Physical characteristics: {characteristics}\n\n"
        f"Generate a JSON object with EXACTLY this structure. "
        f"No markdown fences, no explanation, no extra keys.\n\n"
        "{\n"
        '  "key_diagnostic_features": "<3–5 comma-separated species-diagnostic visual features>",\n'
        '  "environments": [\n'
        '    "<1–2 sentence description of specific habitat variant with substrate, vegetation, weather>",\n'
        "    ... 8 items total, each ecologically plausible and distinct\n"
        "  ],\n"
        '  "behaviors": {\n'
        '    "closeup_head":        ["<1-sentence behavior for close head/face shot>", ... 5 items],\n'
        '    "eye_level":           ["<1-sentence for eye-level medium shot>", ... 5 items],\n'
        '    "low":                 ["<1-sentence for low-angle shot>", ... 5 items],\n'
        '    "high":                ["<1-sentence for elevated-vantage shot>", ... 5 items],\n'
        '    "head_on":             ["<1-sentence for direct frontal shot>", ... 5 items],\n'
        '    "rear":                ["<1-sentence for whole-body rear shot>", ... 5 items],\n'
        '    "side_profile":        ["<1-sentence for lateral full-body shot>", ... 5 items],\n'
        '    "three_quarter_front": ["<1-sentence for 3/4 front shot>", ... 5 items],\n'
        '    "three_quarter_rear":  ["<1-sentence for 3/4 rear shot>", ... 5 items],\n'
        '    "partial_vegetation":  ["<1-sentence for partially obscured shot>", ... 5 items],\n'
        '    "species_specific":    ["<1-sentence for unique behavior/habitat this species exhibits>", ... 6 items]\n'
        "  },\n"
        '  "focus_notes": {\n'
        '    "closeup_head": "<2–3 sentences: which facial/head features must be rendered correctly>",\n'
        '    "default":      "<2–3 sentences: which whole-body features must be anatomically correct>"\n'
        "  }\n"
        "}\n\n"
        f"ALL behaviors and environments must be ecologically plausible for {common_name}. "
        "Do not place it in habitats or behaviors it would never exhibit.\n\n"
        "DIVERSITY REQUIREMENTS — apply to every behavior category:\n"
        "- Include at least 1 behavior where the animal is resting, lying down, or sleeping.\n"
        "- Include at least 1 behavior showing active foraging, feeding, digging, or a "
        "species-typical activity (not just locomotion).\n"
        "- For all categories EXCEPT head_on and closeup_head: include at least 1 behavior "
        "where the animal is NOT facing the camera — body turned away, head averted, or "
        "absorbed in an activity with no awareness of the observer.\n"
        "- Span the full activity spectrum from passive rest to vigorous movement.\n"
        "- Do NOT repeat 'walking' or 'standing alert' more than once per category.\n\n"
        "SINGLE-ANIMAL CONSTRAINT — applies to every behavior description:\n"
        "- Every behavior must describe a single individual animal only.\n"
        "- Social or group contexts are allowed but must be written from the "
        "perspective of the one subject (e.g. 'approaches a companion off-frame', "
        "'rests at the edge of a group that is out of frame', 'grooms itself after "
        "a social encounter'). Never describe two or more animals simultaneously "
        "visible and prominent in the same shot.\n\n"
        "GUILD-SPECIFIC REQUIREMENTS — include where ecologically appropriate:\n"
        "- Arboreal species: at least 1 behavior of climbing, perching on a branch, or "
        "foraging in a tree canopy.\n"
        "- Semi-aquatic species: at least 1 behavior of wading, swimming, or entering water.\n"
        "- Fossorial species: at least 1 behavior of digging, emerging from a burrow, or "
        "sniffing the ground intensely.\n"
        "- Primate species: at least 1 social or grooming behavior AND at least 1 climbing "
        "or branch-leaping behavior.\n"
        "- Large grazing species: at least 1 grazing or browsing behavior AND at least 1 "
        "lying-down-to-ruminate or resting behavior."
    )


def validate_profile(profile: dict) -> bool:
    if not isinstance(profile, dict):
        return False
    for key in ("key_diagnostic_features", "environments", "behaviors", "focus_notes"):
        if key not in profile:
            return False
    if len(profile["environments"]) < 4:
        return False
    behaviors = profile["behaviors"]
    for key in REQUIRED_BEHAVIOR_KEYS:
        if key not in behaviors or not isinstance(behaviors[key], list) or len(behaviors[key]) < 1:
            return False
    if "closeup_head" not in profile["focus_notes"] or "default" not in profile["focus_notes"]:
        return False
    return True


def generate_scene_profile(
    common_name: str,
    scientific_name: str,
    condensed: str,
    characteristics: str,
    api_key: str,
    model: str,
) -> Optional[dict]:
    request_text = build_profile_request(common_name, scientific_name, condensed, characteristics)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": PROFILE_SYSTEM_PROMPT},
            {"role": "user",   "content": request_text},
        ],
    }

    for attempt in range(1, MAX_TRIES + 1):
        try:
            r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=90)
        except requests.RequestException as exc:
            print(f"  network error: {exc}")
            if attempt < MAX_TRIES:
                time.sleep(RETRY_DELAY)
            continue

        if r.status_code != 200:
            print(f"  HTTP {r.status_code}: {r.text[:200]}")
            if attempt < MAX_TRIES:
                time.sleep(RETRY_DELAY)
            continue

        content = r.json()["choices"][0]["message"]["content"].strip()
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        try:
            profile = json.loads(content)
            if validate_profile(profile):
                return profile
            print(f"  profile shape invalid, retrying ({attempt}/{MAX_TRIES})")
        except json.JSONDecodeError as exc:
            print(f"  JSON parse error: {exc} — retrying ({attempt}/{MAX_TRIES})")

        time.sleep(RETRY_DELAY)

    return None


def static_fallback_profile(common_name: str, short_desc: str, guilds: list[str]) -> dict:
    guild_env = {
        "arboreal":      "tropical forest canopy, dense vegetation",
        "semi_aquatic":  "riverbank and shallow water, riparian vegetation",
        "fully_aquatic": "ocean shore, rocky coast, sea water",
        "fossorial":     "open grassland with burrow entrances, loose soil",
        "cold_climate":  "alpine meadow, snow-covered terrain",
        "arid_savanna":  "dry African savanna, sparse acacia scrub",
        "large_grazing": "open grassland, mixed savanna terrain",
        "primate":       "tropical forest interior, dappled light",
        "terrestrial":   "temperate woodland, forest edge",
    }
    env_hint = guild_env.get(guilds[0] if guilds else "terrestrial", "natural habitat")
    environments = [f"{common_name} in {env_hint}, variant {i+1}" for i in range(8)]

    generic_behaviors: dict[str, list[str]] = {}
    for key in REQUIRED_BEHAVIOR_KEYS:
        count = 6 if key == "species_specific" else 5
        generic_behaviors[key] = [
            f"{common_name} {key.replace('_', ' ')} in its natural habitat" for _ in range(count)
        ]

    return {
        "key_diagnostic_features": short_desc,
        "environments": environments,
        "behaviors": generic_behaviors,
        "focus_notes": {
            "closeup_head": f"Render the face and head of {common_name} accurately: {short_desc}",
            "default": f"Render the full body of {common_name} accurately: {short_desc}",
        },
    }


def run_stage1(
    classes: list[ClassConfig],
    descriptions: dict[str, dict],
    api_key: str,
    model: str,
    skip_llm: bool,
) -> dict[str, dict]:
    profiles: dict[str, dict] = {}
    if PROFILES_JSON.exists():
        with open(PROFILES_JSON, encoding="utf-8") as f:
            profiles = json.load(f)
        print(f"Loaded {len(profiles)} existing profiles from {PROFILES_JSON}")

    for i, cls in enumerate(classes, 1):
        name = cls.common_name
        if name in profiles:
            print(f"[{i}/{len(classes)}] {name} — already in profiles, skipping")
            continue

        desc_row = descriptions.get(name.lower())
        if desc_row is None:
            print(f"[{i}/{len(classes)}] {name} — not found in descriptions CSV, using fallback")
            profiles[name] = static_fallback_profile(name, name, cls.guilds)
            continue

        condensed = desc_row.get("condensed_description", "").strip()
        characteristics = desc_row.get("wikipedia_characteristics", "").strip()
        scientific = desc_row.get("scientific_name", name).strip()
        short_desc = desc_row.get("very_short_description", condensed).strip()

        if skip_llm:
            print(f"[{i}/{len(classes)}] {name} — --skip-llm, using fallback")
            profiles[name] = static_fallback_profile(name, short_desc, cls.guilds)
            continue

        print(f"[{i}/{len(classes)}] {name} ({scientific}) ...", end=" ", flush=True)
        profile = generate_scene_profile(name, scientific, condensed, characteristics, api_key, model)

        if profile is None:
            print(f"FAILED — using fallback")
            profiles[name] = static_fallback_profile(name, short_desc, cls.guilds)
        else:
            print("ok")
            profiles[name] = profile

        PROFILES_JSON.parent.mkdir(parents=True, exist_ok=True)
        with open(PROFILES_JSON, "w", encoding="utf-8") as f:
            json.dump(profiles, f, indent=2, ensure_ascii=False)

        if i < len(classes):
            time.sleep(1)

    return profiles

# ---------------------------------------------------------------------------
# Prompt assembly (Stage 2)
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
    bokeh: bool = False,
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
        photography_style=PHOTOGRAPHY_STYLE_BOKEH if bokeh else PHOTOGRAPHY_STYLE_NO_BOKEH,
    )


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def run_stage2(
    classes: list[ClassConfig],
    descriptions: dict[str, dict],
    cn_index: dict[str, tuple],
    scene_profiles: dict[str, dict],
    force: bool,
) -> None:
    prompts_dir = SYNTHETIC_DIR / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    for cls in classes:
        slug = slugify(cls.common_name)
        class_dir = prompts_dir / slug
        schedule = BAND_A_SCHEDULE if cls.band == "A" else BAND_B_SCHEDULE
        total = sum(g.count for g in schedule)

        existing = len(list(class_dir.glob("*.txt"))) if class_dir.exists() else 0
        if not force and existing >= total:
            print(f"  {cls.common_name} — {existing} prompt files already exist, skipping")
            continue

        class_dir.mkdir(parents=True, exist_ok=True)

        # Resolve wiki entry and representative species
        wiki_key, wiki_entry = cn_index.get(cls.common_name.lower(), (None, None))
        if wiki_entry is None:
            print(f"  WARNING: {cls.common_name} not found in wikipedia_urls.json")
            wiki_entry = {"scientific_name": cls.common_name, "common_name": cls.common_name,
                          "wikipedia_file": "", "level": "species", "top_species": None}

        rep_species = get_representative_species(wiki_entry, cls.common_name)
        is_multi = len(rep_species) > 1

        # Load genus article sections (for genus/family classes)
        genus_desc_prefix = ""
        if is_multi and wiki_entry.get("wikipedia_file"):
            g_desc, _, _ = load_wiki_sections(wiki_entry["wikipedia_file"])
            if g_desc:
                genus_desc_prefix = g_desc[:600] + "\n\n"

        profile = scene_profiles.get(cls.common_name)
        if profile is None:
            desc_row = descriptions.get(cls.common_name.lower(), {})
            short = desc_row.get("very_short_description", cls.common_name)
            profile = static_fallback_profile(cls.common_name, short, cls.guilds)

        key_features = profile.get("key_diagnostic_features", cls.common_name)
        environments = profile.get("environments") or ["natural habitat"]
        behaviors = profile.get("behaviors", {})
        focus_notes = profile.get("focus_notes", {})

        global_idx = 0  # 0-based image counter within this class

        for group in schedule:
            for slot_idx in range(group.count):
                image_num = global_idx + 1
                fname = f"{image_num:03d}.txt"
                txt_path = class_dir / fname

                if not force and txt_path.exists():
                    global_idx += 1
                    continue

                # Resolve angle (handles "varies" and "species_specific")
                angle_code = resolve_angle(group.shot_type, slot_idx)

                # Select representative species for this image
                sp = rep_species[global_idx % len(rep_species)]
                sp_scientific = sp["scientific_name"]
                sp_common = sp.get("common_name", sp_scientific)
                sp_wiki_file = sp.get("wikipedia_file", "")

                # Subject line
                if is_multi:
                    subject_line = (
                        f"{sp_common} ({sp_scientific}), "
                        f"a member of the {cls.common_name} group"
                    )
                    subject_name = sp_common
                else:
                    subject_line = f"{sp_common} ({sp_scientific})"
                    subject_name = sp_common

                # Wikipedia sections
                s_desc, s_behavior, s_habitat = load_wiki_sections(sp_wiki_file)
                desc_text = genus_desc_prefix + s_desc
                behavior_text = s_behavior
                habitat_text = s_habitat

                # Lighting
                if group.val_behavior_code is not None:
                    lighting_code = VAL_LIGHTING_POOL[global_idx % len(VAL_LIGHTING_POOL)]
                else:
                    lighting_code = LIGHTING_POOL[global_idx % len(LIGHTING_POOL)]

                # Environment (cycle through all 8)
                environment_description = environments[global_idx % len(environments)]

                # Behavior description
                if group.val_behavior_code is not None:
                    behavior_description = VAL_BEHAVIOR_DESCRIPTIONS[
                        group.val_behavior_code
                    ].format(name=subject_name)
                else:
                    behavior_key = group.shot_type
                    if group.shot_type in ("varies",):
                        behavior_key = angle_code
                    if group.shot_type == "species_specific":
                        behavior_key = "species_specific"
                    b_pool = behaviors.get(behavior_key) or behaviors.get("eye_level") or [subject_name]
                    behavior_description = b_pool[slot_idx % len(b_pool)]

                # Focus note
                if group.shot_type == "closeup_head":
                    focus_note = focus_notes.get("closeup_head", focus_notes.get("default", ""))
                else:
                    focus_note = focus_notes.get("default", "")

                # Occlusion
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

                txt_path.write_text(prompt_text, encoding="utf-8")
                global_idx += 1

        print(f"  {cls.common_name} — wrote {global_idx} prompt files")

# ---------------------------------------------------------------------------
# Index writing (final pass)
# ---------------------------------------------------------------------------

def write_index(
    classes: list[ClassConfig],
    cn_index: dict[str, tuple],
    descriptions: dict[str, dict],
) -> None:
    prompts_dir = SYNTHETIC_DIR / "prompts"
    records = []

    for cls in classes:
        slug = slugify(cls.common_name)
        class_dir = prompts_dir / slug
        if not class_dir.exists():
            continue

        schedule = BAND_A_SCHEDULE if cls.band == "A" else BAND_B_SCHEDULE

        wiki_key, wiki_entry = cn_index.get(cls.common_name.lower(), (None, None))
        if wiki_entry is None:
            wiki_entry = {"scientific_name": cls.common_name, "common_name": cls.common_name,
                          "wikipedia_file": "", "level": "species", "top_species": None}

        rep_species = get_representative_species(wiki_entry, cls.common_name)
        desc_row = descriptions.get(cls.common_name.lower(), {})
        scientific_class = desc_row.get("scientific_name", wiki_entry.get("scientific_name", ""))

        global_idx = 0
        for group in schedule:
            for slot_idx in range(group.count):
                image_num = global_idx + 1
                fname_base = f"{image_num:03d}"
                txt_file = class_dir / f"{fname_base}.txt"
                if not txt_file.exists():
                    global_idx += 1
                    continue

                sp = rep_species[global_idx % len(rep_species)]
                angle_code = resolve_angle(group.shot_type, slot_idx)

                if group.val_behavior_code is not None:
                    lighting_code = VAL_LIGHTING_POOL[global_idx % len(VAL_LIGHTING_POOL)]
                else:
                    lighting_code = LIGHTING_POOL[global_idx % len(LIGHTING_POOL)]

                band_lower = cls.band.lower()
                image_filename = f"{band_lower}_{slug}_{image_num:03d}.png"
                prompt_file = f"prompts/{slug}/{fname_base}.txt"

                records.append({
                    "filename":    image_filename,
                    "class":       cls.common_name,
                    "scientific":  sp["scientific_name"] if len(rep_species) > 1 else scientific_class,
                    "band":        cls.band,
                    "split":       group.split,
                    "shot_type":   angle_code,
                    "distance":    group.distance,
                    "lighting":    lighting_code,
                    "occlusion":   group.occlusion,
                    "prompt_file": prompt_file,
                    "bokeh":       False,
                    "status":      "pending",
                })
                global_idx += 1

    INDEX_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX_JSONL, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(records)} records to {INDEX_JSONL}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic image prompt files and index.jsonl."
    )
    parser.add_argument(
        "--classes",
        default=None,
        help="Comma-separated subset of class common names to process (default: all 76).",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM stage and use static fallback scene profiles.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing .txt prompt files.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenRouter model for scene profile generation (default: {DEFAULT_MODEL}).",
    )
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key and not args.skip_llm:
        sys.exit(
            "Error: OPENROUTER_API_KEY not set. "
            "Add it to .env or use --skip-llm for testing without an API key."
        )

    classes = ALL_CLASSES
    if args.classes:
        requested = {c.strip().lower() for c in args.classes.split(",")}
        classes = [c for c in ALL_CLASSES if c.common_name.lower() in requested]
        not_found = requested - {c.common_name.lower() for c in classes}
        if not_found:
            print(f"Warning: class(es) not found in config: {', '.join(sorted(not_found))}")
        if not classes:
            sys.exit("No matching classes found.")

    print(f"Classes : {len(classes)}")
    print(f"Model   : {args.model}")
    print(f"Skip LLM: {args.skip_llm}")
    print(f"Force   : {args.force}\n")

    descriptions = load_animal_descriptions(DESCRIPTIONS_CSV)
    wiki_urls = load_wikipedia_urls(WIKI_URLS_JSON)
    cn_index = build_common_name_index(wiki_urls)

    # Stage 1 — scene profiles
    print("=== Stage 1: LLM scene profiles ===")
    scene_profiles = run_stage1(classes, descriptions, api_key, args.model, args.skip_llm)

    # Stage 2 — prompt file expansion
    print("\n=== Stage 2: Prompt file generation ===")
    run_stage2(classes, descriptions, cn_index, scene_profiles, args.force)

    # Final — rebuild index.jsonl
    print("\n=== Writing index.jsonl ===")
    write_index(classes, cn_index, descriptions)

    print("Done.")


if __name__ == "__main__":
    main()
