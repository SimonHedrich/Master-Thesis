"""
Generate synthetic wildlife images via the Gemini API directly (google/gemini-3.1-flash-image-preview).

Images are saved to: data/synthetic/<class_name>/

Usage:
python scripts/synthetic/1-generate_synthetic_images_gemini.py \
--class-name "binturong" \
--description "The binturong is A large, heavily built viverrid with long, coarse, dark black hair, tufted ears, a prehensile tail, and a somewhat bear-like face with white whiskers." \
--n-images 5

Requirements:
    pip install google-genai pillow python-dotenv
"""

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = "gemini-3.1-flash-image-preview"
MODEL_KEYWORD = "gemini"
OUTPUT_BASE = Path(__file__).parent.parent.parent / "data" / "synthetic"
MAX_TRIES = 3

DEFAULT_ASPECT_RATIO = "4:3"

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

STYLE_SUFFIX = (
    "Professional wildlife photograph. Telephoto lens, sharp focus on the animal, "
    "natural lighting, photorealistic, high resolution. Full body of the animal visible, "
    "entire animal from head to tail fits within the frame. Natural habitat background. No text, no watermarks."
)


def build_prompt(class_name: str, description: str, scenario: str | None = None) -> str:
    description_excerpt = description.strip()[:400]
    scene = f" Scene: {scenario}." if scenario else ""
    return (
        f"Generate a realistic wildlife photograph of a {class_name}. "
        f"The animal has these characteristics: {description_excerpt}.{scene} "
        f"{STYLE_SUFFIX}"
    )


# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------

def generate_image(prompt: str, client: genai.Client) -> bytes | None:
    """Call the Gemini API and return image bytes, or None on failure."""
    config = types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"])

    for attempt in range(1, MAX_TRIES + 1):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=[prompt],
                config=config,
            )
        except Exception as exc:
            print(f"  API error: {exc}")
            return None

        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                img = part.as_image()
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                return buf.getvalue()

        print(f"  no image in response, retrying ({attempt}/{MAX_TRIES}) ...", end=" ", flush=True)
        time.sleep(2)

    return None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def load_scenarios(scenarios_file: Path, class_name: str) -> list[str]:
    with open(scenarios_file, encoding="utf-8") as f:
        data = json.load(f)
    if class_name in data:
        return data[class_name]
    lower = class_name.lower()
    for key, value in data.items():
        if key.lower() == lower:
            return value
    return []


def generate_images(
    class_name: str,
    description: str,
    n_images: int = 5,
    output_base: Path = OUTPUT_BASE,
    aspect_ratio: str = DEFAULT_ASPECT_RATIO,
    scenarios_file: Path | None = None,
) -> None:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        sys.exit(
            "Error: GEMINI_API_KEY is not set. "
            "Add your key to the .env file in the project root."
        )

    client = genai.Client(api_key=api_key)

    scenarios: list[str] = []
    if scenarios_file is not None:
        scenarios = load_scenarios(scenarios_file, class_name)
        if not scenarios:
            print(f"Warning: no scenarios found for '{class_name}' in {scenarios_file}. "
                  "Falling back to static prompt.")

    output_dir = output_base / class_name
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_scenario = scenarios[0] if scenarios else None
    sample_prompt = build_prompt(class_name, description, sample_scenario)
    print(f"Model    : {MODEL}")
    print(f"Output   : {output_dir}")
    print(f"Scenarios: {len(scenarios)} loaded" if scenarios else "Scenarios: none (static prompt)")
    print(f"Prompt   : {sample_prompt[:120]}...")
    print()

    saved = 0

    for i in range(n_images):
        print(f"Generating image {i + 1}/{n_images} ...", end=" ", flush=True)

        scenario = scenarios[i % len(scenarios)] if scenarios else None
        prompt = build_prompt(class_name, description, scenario)
        img_bytes = generate_image(prompt, client)

        if img_bytes is None:
            print("SKIPPED")
            continue

        existing = sorted(output_dir.glob(f"{class_name}_{MODEL_KEYWORD}_*.png"))
        next_idx = len(existing) + 1
        img_path = output_dir / f"{class_name}_{MODEL_KEYWORD}_{next_idx:04d}.png"
        Image.open(io.BytesIO(img_bytes)).save(img_path, "PNG")
        saved += 1
        print(f"saved → {img_path.name}")

        if i < n_images - 1:
            time.sleep(1)

    print(f"\nDone. {saved}/{n_images} images saved to {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic wildlife images via the Gemini API."
    )
    parser.add_argument("--class-name", required=True, help="Species/class name (e.g. binturong)")
    parser.add_argument("--description", required=True, help="Morphological description text")
    parser.add_argument("--n-images", type=int, default=5, help="Number of images to generate (default: 5)")
    parser.add_argument(
        "--aspect-ratio",
        default=DEFAULT_ASPECT_RATIO,
        help=f"Preferred aspect ratio hint included in prompt (default: {DEFAULT_ASPECT_RATIO}). "
             "Examples: 1:1, 4:3, 3:2, 16:9",
    )
    parser.add_argument(
        "--scenarios-file",
        type=Path,
        default=None,
        help=(
            "Path to animal_scenario_prompts.json produced by 0-generate_scenario_prompts.py. "
            "When provided, each image is generated with a different scene description cycled "
            "from the pre-generated list for this species."
        ),
    )
    args = parser.parse_args()

    generate_images(
        class_name=args.class_name,
        description=args.description,
        n_images=args.n_images,
        aspect_ratio=args.aspect_ratio,
        scenarios_file=args.scenarios_file,
    )


if __name__ == "__main__":
    main()
