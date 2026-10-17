"""Extract the images embedded in the interim deck into assets/interim/.

The interim pptx is gitignored (size, NDA), so its pictures are extracted on demand
rather than committed. Every picture, including those inside groups, is written as
slideNN_KK.<ext>; slides.py refers to the ones it needs by that name.

Run (from thesis/presentation/slides/):
    uv run python extract_assets.py
"""

from pathlib import Path

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

HERE = Path(__file__).resolve().parent
DECK = HERE.parent / "interim-presentation.pptx"
OUT = HERE / "assets" / "interim"


def pictures(shapes):
    for shape in shapes:
        if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            yield from pictures(shape.shapes)
        elif shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
            yield shape


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    deck = Presentation(str(DECK))
    count = 0
    for n, slide in enumerate(deck.slides, 1):
        for k, pic in enumerate(pictures(slide.shapes), 1):
            (OUT / f"slide{n:02d}_{k:02d}.{pic.image.ext}").write_bytes(pic.image.blob)
            count += 1
        # Layout pictures (logo) live on the layout, not the slide.
        if n == 2:
            for k, pic in enumerate(pictures(slide.slide_layout.shapes), 1):
                (OUT / f"layout_{k:02d}.{pic.image.ext}").write_bytes(pic.image.blob)
    print(f"wrote {count} images to {OUT}")


if __name__ == "__main__":
    main()
