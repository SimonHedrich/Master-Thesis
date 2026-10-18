"""Export rendered slides as one scrollable HTML page, slides stacked under each other.

Every slide state (one click in the presenter) becomes a card that shows its final
frame, so the page can be skimmed like a handout. When a card scrolls into view, its
animation plays once and stops on the final frame; clicking a card replays it.
With --static the page uses the final frames only (no videos, much smaller).

The output folder is self-contained: copy it as a whole and open index.html.

Run (from thesis/presentation/slides/, after rendering the scenes):
    uv run python export_scroll.py UseCase DataV2 Generative Bands2 ModelsV2 Summary
    uv run python export_scroll.py --static UseCase DataV2 Generative Bands2 ModelsV2 Summary
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent

# Section headings shown above the first card of each scene.
ACTS = {
    "UseCase": "Use case", "DataV2": "Data", "Generative": "Generated images",
    "Bands2": "Supplementing the classes", "ModelsV2": "Models on embedded hardware", "Summary": "Summary",
    "Intro": "Hook", "Problem": "Problem and plan", "Data": "Data", "Generators": "Generators",
    "Bands": "Bands", "Models": "Models", "Close": "Close",
}

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{ --bg: #050f33; --card: #0b1d5c; --ink: #e8edf7; --muted: #9fb0d6; --accent: #7df381; }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: var(--bg); color: var(--ink);
         font: 16px/1.5 "Atkinson Hyperlegible", system-ui, -apple-system, sans-serif; }}
  main {{ max-width: 1400px; margin: 0 auto; padding: 32px 16px 96px; }}
  header {{ margin-bottom: 28px; }}
  h1 {{ font-size: 28px; margin: 0 0 4px; }}
  header p {{ margin: 0; color: var(--muted); }}
  h2 {{ font-size: 20px; margin: 48px 0 14px; color: var(--accent); scroll-margin-top: 60px; }}
  figure {{ margin: 0 0 22px; }}
  .frame {{ position: relative; aspect-ratio: 16 / 9; background: var(--card); border-radius: 10px; overflow: hidden;
           box-shadow: 0 6px 24px rgb(0 0 0 / 0.35); }}
  .frame img, .frame video {{ display: block; width: 100%; height: 100%; object-fit: contain; }}
  .frame video {{ cursor: pointer; }}
  figcaption {{ margin-top: 6px; color: var(--muted); font-size: 13px; }}
  nav {{ position: sticky; top: 0; z-index: 5; display: flex; gap: 6px; flex-wrap: wrap; justify-content: center;
         padding: 10px 16px; background: rgb(5 15 51 / 0.92); backdrop-filter: blur(6px); }}
  nav a {{ background: rgb(255 255 255 / 0.1); color: var(--ink); text-decoration: none; padding: 4px 10px;
          border-radius: 999px; font-size: 13px; }}
  nav a:hover {{ background: rgb(255 255 255 / 0.22); }}
</style>
</head>
<body>
<nav>{nav}</nav>
<main>
<header>
  <h1>{title}</h1>
  <p>{subtitle}</p>
</header>
{body}
</main>
<script>
  // Play each animation once when it scrolls into view; it then rests on its last frame.
  const seen = new WeakSet();
  const io = new IntersectionObserver((entries) => {{
    for (const e of entries) {{
      const v = e.target;
      if (e.isIntersecting && e.intersectionRatio > 0.6 && !seen.has(v)) {{
        seen.add(v); v.currentTime = 0; v.play().catch(() => {{}});
      }}
    }}
  }}, {{ threshold: [0, 0.6, 1] }});
  document.querySelectorAll("video").forEach((v) => {{
    io.observe(v);
    v.addEventListener("click", () => {{ v.currentTime = 0; v.play(); }});
  }});
</script>
</body>
</html>
"""


def last_frame(video: Path, out: Path, width: int) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        png = Path(tmp) / "f.png"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-sseof", "-0.5", "-i", str(video), "-update", "1", str(png)],
            check=True,
        )
        im = Image.open(png).convert("RGB")
        if im.width > width:
            im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
        im.save(out, quality=88)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenes", nargs="+")
    ap.add_argument("--static", action="store_true", help="final frames only, no videos")
    ap.add_argument("--out", type=Path, default=HERE / "build" / "scroll")
    ap.add_argument("--width", type=int, default=1920, help="width of the still frames in pixels")
    ap.add_argument("--title", default="Real-Time Object Recognition on Embedded Platforms")
    args = ap.parse_args()

    out = args.out
    if out.exists():
        shutil.rmtree(out)
    (out / "frames").mkdir(parents=True)
    if not args.static:
        (out / "videos").mkdir()

    sections, nav, n = [], [], 0
    for scene in args.scenes:
        cfg = json.loads((HERE / "slides" / f"{scene}.json").read_text())
        act = ACTS.get(scene, scene)
        anchor = scene.lower()
        nav.append(f'<a href="#{anchor}">{html.escape(act)}</a>')
        cards = [f'<h2 id="{anchor}">{html.escape(act)}</h2>']
        for k, slide in enumerate(cfg["slides"], 1):
            n += 1
            src = HERE / slide["file"]
            still = f"frames/{n:03d}.jpg"
            last_frame(src, out / still, args.width)
            if args.static:
                media = f'<img src="{still}" alt="Slide {n}" loading="lazy">'
            else:
                clip = f"videos/{n:03d}.mp4"
                shutil.copyfile(src, out / clip)
                media = f'<video src="{clip}" poster="{still}" muted playsinline preload="none"></video>'
            cards.append(f'<figure><div class="frame">{media}</div><figcaption>{n} · {html.escape(act)} · {k}/{len(cfg["slides"])}</figcaption></figure>')
        sections.append("\n".join(cards))
        print(f"{scene}: {len(cfg['slides'])} slides")

    subtitle = "Scroll through all slides. Each animation plays once as it comes into view; click to replay." if not args.static else "All slides, final state of every step."
    (out / "index.html").write_text(PAGE.format(title=html.escape(args.title), subtitle=subtitle, body="\n".join(sections), nav="".join(nav)))
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    print(f"{n} slides → {out / 'index.html'} ({size:.0f} MB)")


if __name__ == "__main__":
    main()
