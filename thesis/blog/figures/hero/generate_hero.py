# /// script
# requires-python = ">=3.11"
# dependencies = ["google-genai>=1.0", "python-dotenv>=1.0", "pillow>=10"]
# ///
"""Generate hero (featured) image candidates for the inovex blog article with Gemini.

The hero has to show the article's subject at a glance: a binocular looks at a rare
animal, the animal is detected (lime box), and the image of the animal is AI-generated
(it assembles out of pixels). Each concept below is one way to draw that as an isometric
illustration in the blog's house style (radial gradient #2D5DFF -> #061B59, white and
periwinkle objects, lime accents, no text). The binocular is a generic one, not the
AX Visio, so no product rights are involved.

Uses gemini-3.1-flash-image, the generator of the thesis' synthetic set, and the
same key (GEMINI_API_KEY in scripts/synthetic/.env, which exists on the GPU server only).

Run on the GPU server, from the repo root:
    uv run --script thesis/blog/figures/hero/generate_hero.py --concept sight --n 2
    (concepts: see CONCEPTS; --ref <image> passes a reference image of the animal;
     --env <file> and --out <dir> override the key file and the output folder)

Writes <out>/<concept>_<k>.png (raw model output, 16:9, not kept in the repo).

The chosen raw image is then cropped to the blog's 1500:880 aspect (centre crop, the
sides lose about 2 %) and saved as a 1600x939 JPEG; this step needs no key and runs
anywhere:
    uv run --script thesis/blog/figures/hero/generate_hero.py --finish raw/cards_3.png --name hero.jpg

Chosen on 2026-11-11: hero.jpg = concept "cards", hero_alt.jpg = concept "sight".
"""

from __future__ import annotations

import argparse
import os
from io import BytesIO
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[4]
MODEL = "gemini-3.1-flash-image"  # "gemini-3.1-flash-image-preview" when the synthetic set was made
W, H = 1600, 939                  # output size (the blog's 1500:880 aspect)

STYLE = (
    "Flat isometric vector illustration, the header image of a modern technology blog. "
    "Background: a smooth radial gradient, bright royal blue (#2D5DFF) in the centre fading "
    "to deep navy (#061B59) towards the edges, empty except for a few tiny glowing "
    "light-blue dots. Objects are white and pale periwinkle blue with soft flat shading; "
    "the only accent colour is lime green (#C0EF59). Clean geometric shapes, no outlines. "
    "Absolutely no text, letters, numbers, logos or watermarks anywhere. One central group "
    "of objects, with wide empty blue margins on all four sides. Wide 16:9 landscape format."
)
BINTURONG = (
    "a binturong (a bearcat: a stocky, shaggy, black-furred tree mammal with small rounded "
    "tufted ears, white whiskers and a very long, thick, bushy tail)"
)

CONCEPTS = {
    # binocular -> cone of sight -> detected animal that is half pixels
    "sight": (
        f"{STYLE} Scene: on the left, a large handheld pair of white binoculars of a generic "
        "design hovers just above a round floating periwinkle platform, with no stand, "
        "tripod or pedestal, and points to the right. A translucent light-blue cone of sight "
        f"leads from its two front lenses to the right, where {BINTURONG} walks along a mossy "
        "branch above a second round platform. A lime-green object-detection bounding box "
        "with a small blank lime label tab on its top-left corner frames the whole animal, "
        "tail included. The animal is drawn as a naturalistic illustration in its real "
        "colours. Its head and front half are complete; from the middle of its body "
        "backwards, the body and the tail break up into many small floating square pixels "
        "that are still flying into place, with a few lime four-pointed sparkles, which "
        "shows that the animal is an AI-generated image."
    ),
    # generated photos become the animal the binocular sees
    "cards": (
        f"{STYLE} Scene: on the right, a large handheld pair of white binoculars of a generic "
        "design hovers just above a round floating periwinkle platform, with no stand or "
        "pedestal. Its two big front lenses point to the left, straight at the animal, and a "
        "translucent light-blue cone of sight leads from the lenses to the animal. On the "
        f"left, on a larger round platform, {BINTURONG} stands on a mossy branch, drawn as a "
        "naturalistic illustration in its real colours, framed by a lime-green "
        "object-detection bounding box with a small blank lime label tab. Behind and above "
        "the animal, five white-bordered photo cards float in an arc; each shows the same "
        "animal in a different pose and each card is partly still made of small square "
        "pixels, with a lime four-pointed sparkle next to it, which shows that the photos "
        "are AI-generated."
    ),
    # the view through the binocular, with the binocular itself in the picture
    "view": (
        f"{STYLE} Scene: in the lower centre, a large pair of white binoculars of a generic "
        "design is seen from behind and slightly above, as if the viewer is about to look "
        "through it. Above and in front of it floats one large rounded rectangular screen, "
        "tilted isometrically, that shows what the binoculars see: a green jungle with "
        f"{BINTURONG} on a branch, framed by a lime-green object-detection bounding box with "
        "a small blank lime label tab. The left third of the screen is a finished, "
        "naturalistic picture; towards the right the picture breaks up into small square "
        "pixels that are still flying into place, with a few lime four-pointed sparkles, "
        "which shows that the picture is AI-generated."
    ),
}


def finish(raw: Path, name: str) -> None:
    """Centre-crop a raw image to W:H and save it as the final JPEG next to this script."""
    im = Image.open(raw).convert("RGB")
    w, h = im.size
    cw = min(w, round(h * W / H))
    ch = min(h, round(cw * H / W))
    im = im.crop(((w - cw) // 2, (h - ch) // 2, (w - cw) // 2 + cw, (h - ch) // 2 + ch))
    path = Path(__file__).resolve().parent / name
    im.resize((W, H), Image.LANCZOS).save(path, quality=90, optimize=True, progressive=True)
    print(f"{path}  {W}x{H}  {path.stat().st_size // 1024} KB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", choices=sorted(CONCEPTS))
    ap.add_argument("--finish", type=Path, help="raw image to crop into the final JPEG")
    ap.add_argument("--name", default="hero.jpg", help="file name for --finish")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--start", type=int, default=1, help="index of the first output file")
    ap.add_argument("--size", default="2K", choices=["1K", "2K", "4K"])
    ap.add_argument("--ref", type=Path, help="optional reference image of the animal")
    ap.add_argument("--extra", default="", help="text appended to the concept prompt")
    ap.add_argument("--env", type=Path, default=ROOT / "scripts/synthetic/.env")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "raw")
    args = ap.parse_args()

    if args.finish:
        finish(args.finish, args.name)
        return
    if not args.concept:
        ap.error("give --concept or --finish")

    from dotenv import load_dotenv
    from google import genai
    from google.genai import types

    load_dotenv(args.env)
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    prompt = CONCEPTS[args.concept] + (" " + args.extra if args.extra else "")
    contents: list = [prompt]
    if args.ref:
        contents.append(Image.open(args.ref))
        contents[0] += " The attached photo shows what the animal looks like."
    config = types.GenerateContentConfig(
        response_modalities=["IMAGE", "TEXT"],
        image_config=types.ImageConfig(aspect_ratio="16:9", image_size=args.size),
    )
    args.out.mkdir(parents=True, exist_ok=True)
    for k in range(args.start, args.start + args.n):
        resp = client.models.generate_content(model=MODEL, contents=contents, config=config)
        parts = resp.candidates[0].content.parts if resp.candidates else []
        data = next((p.inline_data.data for p in parts if p.inline_data), None)
        if data is None:
            print(f"{args.concept}_{k}: no image returned")
            continue
        im = Image.open(BytesIO(data))
        path = args.out / f"{args.concept}_{k}.png"
        im.save(path)
        print(f"{path}  {im.size[0]}x{im.size[1]}")


if __name__ == "__main__":
    main()
