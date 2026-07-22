"""
Generate per-image prompt files and test_index.jsonl for the synthetic test set.

Covers all 225 classes × 50 images = 11,250 test images using a fixed prototypical
shot schedule (the same 5-behavior design as the Band A val set).  Unlike the training
script this uses species-adaptive behavior descriptions so arboreal, aquatic, and
fossorial species are rendered in ecologically appropriate postures.

Pipeline:
  Stage 1 — LLM (OpenRouter) generates a structured scene profile per species, cached
             to reports/synthetic_scene_profiles.json.  The 76 Band A/B profiles that
             already exist are reused; only the ~149 new C/D classes trigger LLM calls.
  Stage 2 — Deterministically expands the test shot schedule into prompt .txt files and
             writes data/synthetic/test_index.jsonl.

Outputs:
    data/synthetic/test_prompts/{class_slug}/{nnn:03d}.txt
    data/synthetic/test_index.jsonl
    reports/synthetic_scene_profiles.json   (extended in-place)

Usage:
    # Full run (requires OPENROUTER_API_KEY in .env or environment):
    uv run python scripts/synthetic/1-generate_test_image_list.py

    # Smoke-test two classes without API key:
    uv run python scripts/synthetic/1-generate_test_image_list.py --classes walrus,kinkajou --skip-llm

    # Regenerate prompts for one class:
    uv run python scripts/synthetic/1-generate_test_image_list.py --classes aardvark --force

Requirements:
    pip install requests python-dotenv
"""

import argparse
import csv
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DESCRIPTIONS_CSV = PROJECT_ROOT / "reports" / "animal_descriptions.csv"
WIKI_URLS_JSON   = PROJECT_ROOT / "reports" / "wikipedia_urls.json"
INAT_COUNTS_CSV  = PROJECT_ROOT / "reports" / "inaturalist_class_image_counts_225.csv"
WIKI_DIR         = PROJECT_ROOT / "data" / "wikipedia"
PROFILES_JSON    = PROJECT_ROOT / "reports" / "synthetic_scene_profiles.json"
SYNTHETIC_DIR    = PROJECT_ROOT / "data" / "synthetic"
TEST_PROMPTS_DIR = SYNTHETIC_DIR / "test_prompts"
TEST_INDEX_JSONL = SYNTHETIC_DIR / "test_index.jsonl"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL  = "google/gemini-3.1-flash-lite"
MAX_TRIES      = 3
RETRY_DELAY    = 5

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
    "close":        "Very close — animal fills approximately 50–70% of frame (telephoto at ~20–80 m)",
    "medium":       "Standard field view — animal fills approximately 20–40% of frame (~80–250 m telephoto)",
    "far":          "Animal at distance — fills less than 10% of frame (>250 m); habitat context dominant",
    "medium-close": "Between medium and close — animal fills approximately 35–50% of frame",
    "medium-far":   "Between medium and far — animal fills approximately 15–25% of frame",
    "close-medium": "Prominent in frame — animal fills approximately 40–55% of frame (binocular portrait)",
    "varies":       "Distance appropriate for the scene; natural field distance for this shot type",
}

VAL_LIGHTING_POOL = ["overcast", "golden_hour"]
LIGHTING_POOL = ["golden_hour", "overcast", "midday", "dappled", "backlit"]
LIGHTING_DESCRIPTIONS = {
    "golden_hour": "Warm directional golden-hour light, long shadows, rich warm tones",
    "overcast":    "Soft diffuse overcast light, no hard shadows, even exposure across the scene",
    "midday":      "Harsh overhead midday sunlight, strong shadows",
    "dappled":     "Intermittent dappled light through canopy or foliage",
    "backlit":     "Sun behind the animal, rim lighting effect, partial silhouette",
}

OCCLUSION_DESCRIPTIONS = {
    "none": "Animal fully visible, no obstruction",
    "partial_vegetation": (
        "20–40% of the body occluded by grasses, branches, or leaves; "
        "animal clearly identifiable throughout"
    ),
    "semi_submerged": "Lower body in water; upper body, back, and head visible above surface",
}

# Species-adaptive test behavior descriptions.
# Phrased to allow ecologically appropriate postures for every guild:
# arboreal species can perch/hang, aquatic species can swim/wade, fossorial
# species can emerge from burrows, etc.  The image generator picks the
# posture that fits the species description and environment.
TEST_BEHAVIOR_DESCRIPTIONS = {
    "standing_alert": (
        "{name} stationary and alert, holding a posture natural for this species — "
        "standing on the ground, perching on a branch, crouching on a rock, or "
        "hanging from vegetation as appropriate — scanning its surroundings with "
        "calm attentiveness"
    ),
    "walking": (
        "{name} moving through its environment at an unhurried, natural pace — "
        "walking, climbing, swimming, wading, or brachating as appropriate for "
        "this species — mid-motion, relaxed"
    ),
    "eating_foraging": (
        "{name} actively foraging or feeding in the posture typical for this species: "
        "grazing on the ground, plucking fruit from a branch, digging at a burrow, "
        "fishing at a water's edge, or any other ecologically appropriate feeding mode"
    ),
    "resting": (
        "{name} at rest in a posture natural for this species — lying down, "
        "crouching, curled up, perched on a branch, hanging, or floating — "
        "fully relaxed and undisturbed"
    ),
    "looking_at_camera": (
        "{name} in a natural resting or alert position appropriate for this species, "
        "oriented toward the camera with quiet awareness"
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

PINNIPED_OVERRIDE = [
    {"scientific_name": "zalophus californianus", "common_name": "California Sea Lion",   "wikipedia_file": "zalophus_californianus.txt"},
    {"scientific_name": "arctocephalus pusillus",  "common_name": "Cape Fur Seal",         "wikipedia_file": "arctocephalus_pusillus.txt"},
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
    common_name: str
    band: str           # "A", "B", "C", or "D"
    guilds: list[str]   # used only for static fallback profiles


@dataclass
class ShotGroup:
    shot_type: str
    distance: str
    count: int
    split: str
    occlusion: str = "none"
    val_behavior_code: Optional[str] = None

# ---------------------------------------------------------------------------
# Test shot schedule — fixed for all 225 classes
# ---------------------------------------------------------------------------

TEST_SCHEDULE: list[ShotGroup] = [
    ShotGroup("eye_level",           "medium", 10, "test", val_behavior_code="standing_alert"),
    ShotGroup("eye_level",           "medium", 10, "test", val_behavior_code="walking"),
    ShotGroup("three_quarter_front", "medium", 10, "test", val_behavior_code="eating_foraging"),
    ShotGroup("eye_level",           "medium", 10, "test", val_behavior_code="resting"),
    ShotGroup("three_quarter_front", "medium", 10, "test", val_behavior_code="looking_at_camera"),
]

# ---------------------------------------------------------------------------
# Band A / B class names (for band lookup)
# Kept as plain name sets — the full ClassConfig lists live in the training
# script; here we only need common names to assign A/B to the 76 known classes.
# ---------------------------------------------------------------------------

BAND_A_NAMES: frozenset[str] = frozenset({
    "walrus", "old world porcupine family", "raccoon dog", "callicebus genus",
    "wild cat", "black-backed jackal", "ringtail", "kinkajou", "genet genus",
    "leopardus species", "water deer", "eurasian badger", "nine-banded armadillo",
    "sloth bear", "yak", "fisher", "striped hyaena", "asiatic black bear",
    "leopard cat", "cephalophus species", "ocelot", "domestic water buffalo",
    "sun bear", "asiatic wild ass", "maned wolf", "honey badger", "fossa",
    "brown hyaena", "red brocket", "pinniped clade", "saiga", "wolverine",
    "pangolin family", "mangabeys genus", "red river hog", "aardwolf", "bongo",
    "binturong", "aardvark", "spilogale species", "red-necked wallaby",
    "clouded leopard", "malay tapir", "aye-aye", "drill", "domestic pig",
    "giant armadillo", "hog badger genus", "african civet", "mouflon",
})

BAND_B_NAMES: frozenset[str] = frozenset({
    "canada lynx", "spectacled bear", "caracal", "eurasian lynx",
    "black wildebeest", "giant panda", "serval", "patas monkey",
    "american mink", "gerenuk", "dhole", "bat-eared fox", "baird's tapir",
    "grevy's zebra", "asian elephant", "kirk's dik-dik", "american badger",
    "chimpanzee", "african wild dog", "glaucomys species", "common wombat",
    "european bison", "lowland tapir", "tayra", "eurasian otter", "springbok",
})

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


def load_inat_counts(csv_path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            counts[row["common_name"].strip().lower()] = int(row["total_images"])
    return counts


def build_common_name_index(wiki_urls: dict) -> dict[str, tuple[str, dict]]:
    idx = {}
    for key, entry in wiki_urls.items():
        cn = entry.get("common_name", "").strip().lower()
        if cn:
            idx[cn] = (key, entry)
    return idx


def load_all_classes(wiki_urls: dict, inat_counts: dict[str, int]) -> list[ClassConfig]:
    """Load all 225 classes and assign bands A–D.

    A/B are fixed (same 76 classes as the training script).
    Of the remaining 149, the 26 with the lowest iNaturalist counts become
    Band C; the rest are Band D.
    """
    non_ab: list[tuple[str, int]] = []
    for entry in wiki_urls.values():
        cn = entry["common_name"]
        if cn.lower() not in BAND_A_NAMES and cn.lower() not in BAND_B_NAMES:
            non_ab.append((cn, inat_counts.get(cn.lower(), 0)))
    non_ab.sort(key=lambda x: x[1])
    band_c = frozenset(n.lower() for n, _ in non_ab[:26])

    result: list[ClassConfig] = []
    for entry in wiki_urls.values():
        cn = entry["common_name"]
        lo = cn.lower()
        if lo in BAND_A_NAMES:
            band = "A"
        elif lo in BAND_B_NAMES:
            band = "B"
        elif lo in band_c:
            band = "C"
        else:
            band = "D"
        result.append(ClassConfig(common_name=cn, band=band, guilds=[]))
    return result

# ---------------------------------------------------------------------------
# Wikipedia section extraction
# ---------------------------------------------------------------------------

def _section_depth(header: str) -> int:
    m = re.match(r"^(=+)", header)
    return len(m.group(1)) if m else 0


def extract_sections(text: str) -> tuple[str, str, str]:
    parts = re.split(r"^(==+[^=\n]+==+)", text, flags=re.MULTILINE)
    lead = parts[0].strip()

    sections: list[tuple[str, int, str]] = []
    for i in range(1, len(parts) - 1, 2):
        header_raw = parts[i].strip()
        content    = parts[i + 1] if i + 1 < len(parts) else ""
        depth      = _section_depth(header_raw)
        name       = re.sub(r"^=+\s*|\s*=+$", "", header_raw).strip()
        sections.append((name, depth, content))

    def collect(aliases: frozenset) -> str:
        result: list[str] = []
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

    desc     = collect(DESCRIPTION_ALIASES) or lead[:800]
    behavior = collect(BEHAVIOR_ALIASES)
    habitat  = collect(HABITAT_ALIASES)
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
        "'rests at the edge of a group that is out of frame'). Never describe two or "
        "more animals simultaneously visible and prominent in the same shot.\n\n"
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
            "default":      f"Render the full body of {common_name} accurately: {short_desc}",
        },
    }


def _stage1_worker(
    cls: ClassConfig,
    descriptions: dict[str, dict],
    api_key: str,
    model: str,
    skip_llm: bool,
) -> tuple[str, dict]:
    """Generate (or fall back to) a scene profile for one class. Thread-safe."""
    name     = cls.common_name
    desc_row = descriptions.get(name.lower())

    if desc_row is None:
        return name, static_fallback_profile(name, name, cls.guilds)

    condensed       = desc_row.get("condensed_description", "").strip()
    characteristics = desc_row.get("wikipedia_characteristics", "").strip()
    scientific      = desc_row.get("scientific_name", name).strip()
    short_desc      = desc_row.get("very_short_description", condensed).strip()

    if skip_llm:
        return name, static_fallback_profile(name, short_desc, cls.guilds)

    profile = generate_scene_profile(name, scientific, condensed, characteristics, api_key, model)
    if profile is None:
        return name, static_fallback_profile(name, short_desc, cls.guilds)
    return name, profile


def run_stage1(
    classes: list[ClassConfig],
    descriptions: dict[str, dict],
    api_key: str,
    model: str,
    skip_llm: bool,
    workers: int = 4,
) -> dict[str, dict]:
    profiles: dict[str, dict] = {}
    if PROFILES_JSON.exists():
        with open(PROFILES_JSON, encoding="utf-8") as f:
            profiles = json.load(f)
        print(f"Loaded {len(profiles)} existing profiles from {PROFILES_JSON}")

    pending = [cls for cls in classes if cls.common_name not in profiles]
    skipped = len(classes) - len(pending)
    if skipped:
        print(f"{skipped} classes already have profiles, skipping")
    if not pending:
        return profiles

    print(f"Generating profiles for {len(pending)} classes with {workers} workers ...")
    lock     = threading.Lock()
    done     = 0
    total    = len(pending)

    PROFILES_JSON.parent.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_stage1_worker, cls, descriptions, api_key, model, skip_llm): cls
            for cls in pending
        }
        for future in as_completed(futures):
            cls = futures[future]
            try:
                name, profile = future.result()
            except Exception as exc:
                name = cls.common_name
                desc_row  = descriptions.get(name.lower(), {})
                short     = desc_row.get("very_short_description", name)
                profile   = static_fallback_profile(name, short, cls.guilds)
                print(f"  {name} — exception: {exc}, using fallback")

            with lock:
                done += 1
                profiles[name] = profile
                status = "fallback" if profile.get("environments", [""])[0].endswith("variant 1") else "ok"
                print(f"  [{done}/{total}] {name} — {status}")
                with open(PROFILES_JSON, "w", encoding="utf-8") as f:
                    json.dump(profiles, f, indent=2, ensure_ascii=False)

    return profiles

# ---------------------------------------------------------------------------
# Prompt assembly (Stage 2)
# ---------------------------------------------------------------------------

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


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _stage2_worker(
    cls: ClassConfig,
    descriptions: dict[str, dict],
    cn_index: dict[str, tuple],
    scene_profiles: dict[str, dict],
    force: bool,
) -> tuple[str, int]:
    """Write prompt files for one class. Returns (common_name, files_written). Thread-safe."""
    slug      = slugify(cls.common_name)
    class_dir = TEST_PROMPTS_DIR / slug
    total_per_class = sum(g.count for g in TEST_SCHEDULE)

    existing = len(list(class_dir.glob("*.txt"))) if class_dir.exists() else 0
    if not force and existing >= total_per_class:
        return cls.common_name, 0

    class_dir.mkdir(parents=True, exist_ok=True)

    wiki_key, wiki_entry = cn_index.get(cls.common_name.lower(), (None, None))
    if wiki_entry is None:
        wiki_entry = {"scientific_name": cls.common_name, "common_name": cls.common_name,
                      "wikipedia_file": "", "level": "species", "top_species": None}

    rep_species = get_representative_species(wiki_entry, cls.common_name)
    is_multi    = len(rep_species) > 1

    genus_desc_prefix = ""
    if is_multi and wiki_entry.get("wikipedia_file"):
        g_desc, _, _ = load_wiki_sections(wiki_entry["wikipedia_file"])
        if g_desc:
            genus_desc_prefix = g_desc[:600] + "\n\n"

    profile = scene_profiles.get(cls.common_name)
    if profile is None:
        desc_row = descriptions.get(cls.common_name.lower(), {})
        short    = desc_row.get("very_short_description", cls.common_name)
        profile  = static_fallback_profile(cls.common_name, short, cls.guilds)

    key_features = profile.get("key_diagnostic_features", cls.common_name)
    environments = profile.get("environments") or ["natural habitat"]
    focus_notes  = profile.get("focus_notes", {})

    global_idx = 0
    written    = 0

    for group in TEST_SCHEDULE:
        for slot_idx in range(group.count):
            image_num = global_idx + 1
            txt_path  = class_dir / f"{image_num:03d}.txt"

            if not force and txt_path.exists():
                global_idx += 1
                continue

            angle_code    = resolve_angle(group.shot_type, slot_idx)
            sp            = rep_species[global_idx % len(rep_species)]
            sp_scientific = sp["scientific_name"]
            sp_common     = sp.get("common_name", sp_scientific)
            sp_wiki_file  = sp.get("wikipedia_file", "")

            if is_multi:
                subject_line = (
                    f"{sp_common} ({sp_scientific}), "
                    f"a member of the {cls.common_name} group"
                )
                subject_name = sp_common
            else:
                subject_line = f"{sp_common} ({sp_scientific})"
                subject_name = sp_common

            s_desc, s_behavior, s_habitat = load_wiki_sections(sp_wiki_file)
            desc_text     = genus_desc_prefix + s_desc
            behavior_text = s_behavior
            habitat_text  = s_habitat

            lighting_code           = VAL_LIGHTING_POOL[global_idx % len(VAL_LIGHTING_POOL)]
            environment_description = environments[global_idx % len(environments)]
            behavior_description    = TEST_BEHAVIOR_DESCRIPTIONS[
                group.val_behavior_code
            ].format(name=subject_name)
            focus_note = focus_notes.get("default", "")

            prompt_text = build_prompt(
                subject_line=subject_line,
                subject_name=subject_name,
                desc_text=desc_text,
                behavior_text=behavior_text,
                habitat_text=habitat_text,
                angle_code=angle_code,
                distance_code=group.distance,
                lighting_code=lighting_code,
                occlusion_code=group.occlusion,
                behavior_description=behavior_description,
                environment_description=environment_description,
                focus_note=focus_note,
                key_diagnostic_features=key_features,
            )

            txt_path.write_text(prompt_text, encoding="utf-8")
            global_idx += 1
            written    += 1

    return cls.common_name, written


def run_stage2_test(
    classes: list[ClassConfig],
    descriptions: dict[str, dict],
    cn_index: dict[str, tuple],
    scene_profiles: dict[str, dict],
    force: bool,
    workers: int = 8,
) -> None:
    TEST_PROMPTS_DIR.mkdir(parents=True, exist_ok=True)

    done  = 0
    total = len(classes)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_stage2_worker, cls, descriptions, cn_index, scene_profiles, force): cls
            for cls in classes
        }
        for future in as_completed(futures):
            cls = futures[future]
            try:
                name, written = future.result()
            except Exception as exc:
                name    = cls.common_name
                written = -1
                print(f"  ERROR {name}: {exc}")
            done += 1
            if written == 0:
                print(f"  [{done}/{total}] {name} — already complete, skipped")
            elif written < 0:
                pass
            else:
                print(f"  [{done}/{total}] {name} — wrote {written} prompt files")

# ---------------------------------------------------------------------------
# Index writing
# ---------------------------------------------------------------------------

def write_test_index(
    classes: list[ClassConfig],
    cn_index: dict[str, tuple],
    descriptions: dict[str, dict],
) -> None:
    records: list[dict] = []

    for cls in classes:
        slug      = slugify(cls.common_name)
        class_dir = TEST_PROMPTS_DIR / slug
        if not class_dir.exists():
            continue

        wiki_key, wiki_entry = cn_index.get(cls.common_name.lower(), (None, None))
        if wiki_entry is None:
            wiki_entry = {"scientific_name": cls.common_name, "common_name": cls.common_name,
                          "wikipedia_file": "", "level": "species", "top_species": None}

        rep_species    = get_representative_species(wiki_entry, cls.common_name)
        desc_row       = descriptions.get(cls.common_name.lower(), {})
        scientific_cls = desc_row.get("scientific_name", wiki_entry.get("scientific_name", ""))

        global_idx = 0
        for group in TEST_SCHEDULE:
            for slot_idx in range(group.count):
                image_num  = global_idx + 1
                fname_base = f"{image_num:03d}"
                txt_file   = class_dir / f"{fname_base}.txt"
                if not txt_file.exists():
                    global_idx += 1
                    continue

                sp         = rep_species[global_idx % len(rep_species)]
                angle_code = resolve_angle(group.shot_type, slot_idx)
                lighting_code = VAL_LIGHTING_POOL[global_idx % len(VAL_LIGHTING_POOL)]

                records.append({
                    "filename":    f"t_{slug}_{image_num:03d}.png",
                    "class":       cls.common_name,
                    "scientific":  sp["scientific_name"] if len(rep_species) > 1 else scientific_cls,
                    "band":        cls.band,
                    "split":       "test",
                    "shot_type":   angle_code,
                    "distance":    group.distance,
                    "lighting":    lighting_code,
                    "occlusion":   group.occlusion,
                    "behavior":    group.val_behavior_code,
                    "prompt_file": f"test_prompts/{slug}/{fname_base}.txt",
                    "bokeh":       False,
                    "status":      "pending",
                })
                global_idx += 1

    TEST_INDEX_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(TEST_INDEX_JSONL, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(records)} records to {TEST_INDEX_JSONL}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic test-set prompt files and test_index.jsonl (225 classes × 50 images)."
    )
    parser.add_argument(
        "--classes",
        default=None,
        help="Comma-separated subset of class common names to process (default: all 225).",
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
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers for Stage 1 (LLM calls) and Stage 2 (prompt writing). Default: 8.",
    )
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key and not args.skip_llm:
        sys.exit(
            "Error: OPENROUTER_API_KEY not set. "
            "Add it to .env or use --skip-llm for testing without an API key."
        )

    wiki_urls   = load_wikipedia_urls(WIKI_URLS_JSON)
    inat_counts = load_inat_counts(INAT_COUNTS_CSV)
    all_classes = load_all_classes(wiki_urls, inat_counts)

    classes = all_classes
    if args.classes:
        requested = {c.strip().lower() for c in args.classes.split(",")}
        classes   = [c for c in all_classes if c.common_name.lower() in requested]
        not_found = requested - {c.common_name.lower() for c in classes}
        if not_found:
            print(f"Warning: class(es) not found: {', '.join(sorted(not_found))}")
        if not classes:
            sys.exit("No matching classes found.")

    band_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    for c in classes:
        band_counts[c.band] += 1

    total_images = len(classes) * sum(g.count for g in TEST_SCHEDULE)
    print(f"Classes : {len(classes)}  (A={band_counts['A']} B={band_counts['B']} C={band_counts['C']} D={band_counts['D']})")
    print(f"Images  : {total_images} ({len(classes)} × {sum(g.count for g in TEST_SCHEDULE)})")
    print(f"Workers : {args.workers}")
    print(f"Model   : {args.model}")
    print(f"Skip LLM: {args.skip_llm}")
    print(f"Force   : {args.force}\n")

    descriptions = load_animal_descriptions(DESCRIPTIONS_CSV)
    cn_index     = build_common_name_index(wiki_urls)

    print("=== Stage 1: LLM scene profiles ===")
    scene_profiles = run_stage1(classes, descriptions, api_key, args.model, args.skip_llm, args.workers)

    print("\n=== Stage 2: Prompt file generation ===")
    run_stage2_test(classes, descriptions, cn_index, scene_profiles, args.force, args.workers)

    print("\n=== Writing test_index.jsonl ===")
    write_test_index(classes, cn_index, descriptions)

    print("Done.")


if __name__ == "__main__":
    main()
