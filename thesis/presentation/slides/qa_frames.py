"""Grab the last frame of every rendered slide into build/qa/ and a contact sheet.

For checking layouts without clicking through the presenter.

Run (from thesis/presentation/slides/, after rendering):
    uv run python qa_frames.py Intro Problem ...
"""

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
OUT = HERE / "build" / "qa"


def main(scenes: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for scene in scenes:
        cfg = json.loads((HERE / "slides" / f"{scene}.json").read_text())
        frames = []
        for i, s in enumerate(cfg["slides"], 1):
            png = OUT / f"{scene}_{i:02d}.png"
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-sseof", "-0.5", "-i", str(HERE / s["file"]), "-update", "1", str(png)],
                check=True,
            )
            frames.append(png)
        w, h = 640, 360
        cols = 3
        sheet = Image.new("RGB", (w * cols, h * ((len(frames) + cols - 1) // cols)), "gray")
        for k, f in enumerate(frames):
            im = Image.open(f).convert("RGB").resize((w, h))
            ImageDraw.Draw(im).text((6, 4), f.stem, fill="red")
            sheet.paste(im, ((k % cols) * w, (k // cols) * h))
        sheet.save(OUT / f"sheet_{scene}.jpg", quality=88)
        print(OUT / f"sheet_{scene}.jpg", len(frames), "slides")


if __name__ == "__main__":
    main(sys.argv[1:])
