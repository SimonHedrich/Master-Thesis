"""
Generate synthetic wildlife images via OpenRouter (google/gemini-2.5-flash-image-preview).

Images are saved to: data/synthetic/<class_name>/

Usage:
python scripts/generate_synthetic_images.py \
--class-name "binturong" \
--description "The binturong is A large, heavily built viverrid with long, coarse, dark black hair, tufted ears, a prehensile tail, and a somewhat bear-like face with white whiskers." \
--n-images 5

Requirements:
    pip install requests pillow python-dotenv
"""

import argparse
import base64
import io
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = "google/gemini-3.1-flash-image-preview"
MODEL_KEYWORD = "gemini"  # Short label used in output filenames.
OUTPUT_BASE = Path(__file__).parent.parent / "data" / "synthetic"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_TRIES = 3

# Image resolution config (native OpenRouter image_config object).
# Aspect ratio options: "1:1", "3:2", "2:3", "4:3", "3:4", "16:9", "9:16", "21:9", etc.
# Image size options:   "0.5K" (~512px), "1K" (standard), "2K" (higher), "4K" (highest, costs more)
# For synthetic training data the images are resized to the model's input size (320–416 px for
# YOLO-nano) during training, so 0.5K (~512 px) is the minimum that keeps meaningful detail
# while minimising cost per image.  The flash model supports 0.5K; the pro model does not.
DEFAULT_ASPECT_RATIO = "4:3"
DEFAULT_IMAGE_SIZE = "0.5K"

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

STYLE_SUFFIX = (
    "Professional wildlife photograph. Telephoto lens, sharp focus on the animal, "
    "natural lighting, photorealistic, high resolution. The animal is the main subject "
    "filling most of the frame. Natural habitat background. No text, no watermarks."
)


def build_prompt(class_name: str, description: str) -> str:
    description_excerpt = description.strip()[:600]
    return (
        f"Generate a realistic wildlife photograph of a {class_name}. "
        f"The animal has these characteristics: {description_excerpt}. "
        f"{STYLE_SUFFIX}"
    )


# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------

def _fetch_cost(generation_id: str, headers: dict) -> float:
    """Query the OpenRouter generation stats endpoint to get the actual cost in USD."""
    # The stats endpoint may take a moment to populate after the generation completes.
    time.sleep(1)
    try:
        r = requests.get(
            "https://openrouter.ai/api/v1/generation",
            headers=headers,
            params={"id": generation_id},
            timeout=15,
        )
        if r.ok:
            return float(r.json().get("data", {}).get("total_cost", 0))
    except Exception:
        pass
    return 0.0


def generate_image(
    prompt: str,
    api_key: str,
    aspect_ratio: str = DEFAULT_ASPECT_RATIO,
    image_size: str = DEFAULT_IMAGE_SIZE,
) -> tuple[bytes | None, float]:
    """Call OpenRouter and return (image_bytes, cost_usd). Returns (None, 0) on failure."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "modalities": ["image", "text"],
        "image_config": {
            "aspect_ratio": aspect_ratio,
            "image_size": image_size,
        },
    }

    for attempt in range(1, MAX_TRIES + 1):
        response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120)

        if response.status_code != 200:
            print(f"FAILED (HTTP {response.status_code}): {response.text[:200]}")
            return None, 0.0

        result = response.json()
        generation_id = result.get("id")
        cost = _fetch_cost(generation_id, headers) if generation_id else 0.0

        images = (
            result.get("choices", [{}])[0]
            .get("message", {})
            .get("images", [])
        )
        if images:
            data_url = images[0]["image_url"]["url"]
            # Data URL format: "data:<mime>;base64,<data>"
            b64 = data_url.split(",")[1] if "," in data_url else data_url
            return base64.b64decode(b64), cost

        print(f"  no image in response, retrying ({attempt}/{MAX_TRIES}) ...", end=" ", flush=True)
        time.sleep(2)

    return None, 0.0


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def generate_images(
    class_name: str,
    description: str,
    n_images: int = 5,
    output_base: Path = OUTPUT_BASE,
    aspect_ratio: str = DEFAULT_ASPECT_RATIO,
    image_size: str = DEFAULT_IMAGE_SIZE,
) -> None:
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        sys.exit(
            "Error: OPENROUTER_API_KEY is not set. "
            "Add your key to the .env file in the project root."
        )

    output_dir = output_base / class_name
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt = build_prompt(class_name, description)
    print(f"Model  : {MODEL}")
    print(f"Output : {output_dir}")
    print(f"Config : aspect_ratio={aspect_ratio}  image_size={image_size}")
    print(f"Prompt : {prompt[:120]}...")
    print()

    total_cost = 0.0
    saved = 0

    for i in range(n_images):
        print(f"Generating image {i + 1}/{n_images} ...", end=" ", flush=True)

        img_bytes, request_cost = generate_image(prompt, api_key, aspect_ratio, image_size)
        total_cost += request_cost

        if img_bytes is None:
            print("SKIPPED")
            continue

        # Convert to PNG via PIL (normalises whatever format Gemini returns).
        image = Image.open(io.BytesIO(img_bytes))
        existing = sorted(output_dir.glob(f"{class_name}_{MODEL_KEYWORD}_*.png"))
        next_idx = len(existing) + 1
        img_path = output_dir / f"{class_name}_{MODEL_KEYWORD}_{next_idx:04d}.png"
        image.save(img_path, "PNG")
        saved += 1
        print(f"saved → {img_path.name}  (${request_cost:.4f})")

        if i < n_images - 1:
            time.sleep(1)

    print(f"\nDone. {saved}/{n_images} images saved to {output_dir}")
    print(f"Total cost : ${total_cost:.4f}")
    if saved > 1:
        print(f"Cost/image : ${total_cost / saved:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic wildlife images via OpenRouter (Gemini)."
    )
    parser.add_argument("--class-name", required=True, help="Species/class name (e.g. binturong)")
    parser.add_argument("--description", required=True, help="Morphological description text")
    parser.add_argument("--n-images", type=int, default=5, help="Number of images to generate (default: 5)")
    parser.add_argument(
        "--aspect-ratio",
        default=DEFAULT_ASPECT_RATIO,
        help=f"Image aspect ratio passed to OpenRouter image_config (default: {DEFAULT_ASPECT_RATIO}). "
             "Examples: 1:1, 4:3, 3:2, 16:9, 9:16",
    )
    parser.add_argument(
        "--image-size",
        default=DEFAULT_IMAGE_SIZE,
        choices=["0.5K", "1K", "2K", "4K"],
        help=f"Resolution tier passed to OpenRouter image_config (default: {DEFAULT_IMAGE_SIZE}). "
             "4K costs more output tokens.",
    )
    args = parser.parse_args()

    generate_images(
        class_name=args.class_name,
        description=args.description,
        n_images=args.n_images,
        aspect_ratio=args.aspect_ratio,
        image_size=args.image_size,
    )


if __name__ == "__main__":
    main()
