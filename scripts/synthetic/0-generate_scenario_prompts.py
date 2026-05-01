"""
Pre-generate ecologically realistic scenario descriptions for synthetic image prompts.

For each target species, calls a cheap text LLM to produce N diverse scene descriptions
(varying behaviour, substrate, viewpoint, lighting, season). Results are stored in a JSON
file that the image generation script reads at runtime to inject a different scenario into
every image prompt.

Usage:
    # Generate 90 scenarios for two Tier C species:
    python scripts/synthetic/0-generate_scenario_prompts.py \
        --species "snow leopard,aye-aye" \
        --n-scenarios 90

    # Generate for all species in the CSV:
    python scripts/synthetic/0-generate_scenario_prompts.py --species all

Output:
    reports/animal_scenario_prompts.json
    {
      "snow leopard": ["Snow leopard crouched on a rocky ledge ...", ...],
      "aye-aye": ["Aye-aye clinging to a vertical tree trunk ...", ...],
      ...
    }

Requirements:
    pip install requests python-dotenv
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# A cheap text-only model — fast, low cost, strong ecology knowledge.
DEFAULT_TEXT_MODEL = "google/gemini-2.5-flash-preview"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

PROJECT_ROOT = Path(__file__).parent.parent.parent
DESCRIPTIONS_CSV = PROJECT_ROOT / "reports" / "animal_descriptions.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "animal_scenario_prompts.json"

MAX_TRIES = 3
RETRY_DELAY = 5  # seconds between retries on failure

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SCENARIO_SYSTEM_PROMPT = (
    "You are an expert wildlife biologist and nature photographer. "
    "You know where each animal species lives, what it eats, when it is active, "
    "and how it moves and behaves in the wild."
)


def build_scenario_request(common_name: str, scientific_name: str, description: str, n: int) -> str:
    return (
        f"Generate {n} diverse, ecologically realistic scene descriptions for photographing "
        f"a {common_name} ({scientific_name}) in the wild.\n\n"
        f"Animal description: {description.strip()}\n\n"
        f"Each description must be 1–2 sentences and must vary across ALL of these five dimensions:\n"
        f"1. Behaviour — foraging, resting, alert/watching, moving, nursing young, vocalising\n"
        f"2. Substrate/location — specific microhabitat natural for this species "
        f"(e.g. 'perched on a mossy granite boulder', 'wading through papyrus reed beds')\n"
        f"3. Viewpoint angle — frontal, lateral, three-quarter, overhead, from slightly below\n"
        f"4. Lighting — golden hour, overcast midday, dappled forest light, moonlit, harsh midday sun\n"
        f"5. Season or weather — dry season dust, after rainfall, snow, leaf-off winter, heavy canopy\n\n"
        f"All scenes MUST be ecologically plausible for this specific species. "
        f"Do not place the animal in habitats it would never occupy.\n\n"
        f"Return ONLY a JSON array of {n} strings. "
        f"No keys, no explanation, no markdown fences, no numbering."
    )


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def generate_scenarios(
    common_name: str,
    scientific_name: str,
    description: str,
    n: int,
    api_key: str,
    model: str,
) -> list[str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SCENARIO_SYSTEM_PROMPT},
            {"role": "user", "content": build_scenario_request(common_name, scientific_name, description, n)},
        ],
    }

    for attempt in range(1, MAX_TRIES + 1):
        try:
            r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
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

        # Strip optional markdown fences the model may add despite instructions.
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        try:
            scenarios = json.loads(content)
            if isinstance(scenarios, list) and all(isinstance(s, str) for s in scenarios):
                return scenarios
            print(f"  unexpected JSON shape, retrying ({attempt}/{MAX_TRIES})")
        except json.JSONDecodeError as exc:
            print(f"  JSON parse error: {exc} — retrying ({attempt}/{MAX_TRIES})")

        time.sleep(RETRY_DELAY)

    return []


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_descriptions(csv_path: Path) -> list[dict]:
    import csv
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-generate per-species scenario descriptions for synthetic image prompts."
    )
    parser.add_argument(
        "--species",
        default="all",
        help=(
            'Comma-separated list of common names (e.g. "snow leopard,aye-aye"), '
            'or "all" to process every row in the descriptions CSV.'
        ),
    )
    parser.add_argument(
        "--n-scenarios",
        type=int,
        default=90,
        help="Number of scenario descriptions to generate per species (default: 90).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_TEXT_MODEL,
        help=f"OpenRouter model ID for text generation (default: {DEFAULT_TEXT_MODEL}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSON file path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--descriptions-csv",
        type=Path,
        default=DESCRIPTIONS_CSV,
        help=f"Input CSV with animal descriptions (default: {DESCRIPTIONS_CSV}).",
    )
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        sys.exit(
            "Error: OPENROUTER_API_KEY is not set. "
            "Add your key to the .env file in the project root."
        )

    rows = load_descriptions(args.descriptions_csv)

    if args.species.strip().lower() == "all":
        targets = rows
    else:
        requested = {s.strip().lower() for s in args.species.split(",")}
        targets = [r for r in rows if r["common_name"].strip().lower() in requested]
        found = {r["common_name"].strip().lower() for r in targets}
        missing = requested - found
        if missing:
            print(f"Warning: species not found in CSV: {', '.join(sorted(missing))}")

    if not targets:
        sys.exit("No matching species found. Check --species and the CSV path.")

    # Load existing output to allow incremental runs (skip already-done species).
    existing: dict[str, list[str]] = {}
    if args.output.exists():
        with open(args.output, encoding="utf-8") as f:
            existing = json.load(f)
        print(f"Loaded {len(existing)} existing entries from {args.output}")

    print(f"Model    : {args.model}")
    print(f"Scenarios: {args.n_scenarios} per species")
    print(f"Output   : {args.output}")
    print(f"Species  : {len(targets)} to process\n")

    results = dict(existing)

    for i, row in enumerate(targets, 1):
        name = row["common_name"].strip()
        sci = row["scientific_name"].strip()
        # Use the condensed description for the scenario prompt — detailed enough but concise.
        desc = row.get("condensed_description", row.get("wikipedia_characteristics", "")).strip()

        if name in results:
            print(f"[{i}/{len(targets)}] {name} — already in output, skipping")
            continue

        print(f"[{i}/{len(targets)}] {name} ({sci}) ...", end=" ", flush=True)

        scenarios = generate_scenarios(name, sci, desc, args.n_scenarios, api_key, args.model)

        if not scenarios:
            print("FAILED — no valid scenarios generated")
            continue

        results[name] = scenarios
        print(f"{len(scenarios)} scenarios generated")

        # Write incrementally after each species so a mid-run failure doesn't lose progress.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        if i < len(targets):
            time.sleep(1)

    print(f"\nDone. {len(results)} species written to {args.output}")


if __name__ == "__main__":
    main()
