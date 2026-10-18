# Final Talk Slides (manim)

> **Fallback if the presenter misbehaves (wrong screen size, Qt issues, venue machine):**
> present the deck as an offline HTML page in a browser, which always scales the video to
> the screen.
>
> ```
> uv run manim-slides convert --offline UseCase DataV2 Generative Bands2 ModelsV2 Summary build/talk_v2.html
> ```
>
> Open `build/talk_v2.html` in the browser and press `F` for full screen (← / → to navigate).
> `--offline` also stores reveal.js locally, so no internet is needed. To present from
> another machine, copy `talk_v2.html` **and** the `talk_v2_assets/` folder next to it (~200 MB
> at 4K).

The 20-minute final presentation, built as animations with
[manim](https://www.manim.community/) and presented with
[manim-slides](https://manim-slides.eertmans.be/). The story follows
`../final-talk-storyline.md`, and the results part follows `../results-slides-strategy.md`.

This is a separate uv project, not part of the repo-root environment. The root project is
the Linux/CUDA training container, while manim renders natively on the Mac (Cairo, Pango,
ffmpeg) and the presenter needs a Qt window.

## Two versions

- **`talk_v2.py` (current):** the reworked talk. It builds up step by step: use case →
  data → generated images → bands → models on the hardware → a summary that replays one
  picture from every step. It has six scenes (`UseCase`, `DataV2`, `Generative`, `Bands2`,
  `ModelsV2`, `Summary`) and 33 slides. Images come from `../images/` (run
  `collect_images.py` first). The speaker script is `../final-talk-script-v2.md`.
- **`talk.py`:** the first version, told chronologically (plan → what happened). Kept as is.

```
uv run python extract_assets.py && uv run python collect_images.py
uv run manim render -qk --frame_rate 30 talk_v2.py UseCase DataV2 Generative Bands2 ModelsV2 Summary
uv run python present.py -F UseCase DataV2 Generative Bands2 ModelsV2 Summary
uv run manim-slides convert --offline UseCase DataV2 Generative Bands2 ModelsV2 Summary build/talk_v2.html
```

Render all six v2 scenes in one command, in this order. `Summary` reuses snapshots that
the earlier scenes save while rendering (`build/cache/snaps/`).

## Files

| File | Role |
|------|------|
| `talk.py` | The slides: seven scenes (`Intro`, `Problem`, `Data`, `Generators`, `Bands`, `Models`, `Close`), one per act, 34 slides in total |
| `theme.py` | inovex look: colours, Atkinson Hyperlegible, gradient background, pills, rounded photos, bars, drawn icons |
| `data.py` | Every number and asset path, each number with its thesis/report source |
| `talk_v2.py` | The reworked talk (see above) |
| `collect_images.py` | Builds `../images/` (gitignored): copies and downscales every picture `talk_v2.py` uses, Swarovski-internal ones under `nda/` |
| `assets/prompts/` | The four lion prompts (compressed, maxlen 256/512, full) from the manuscript appendix, for the prompt-length slide |
| `extract_assets.py` | Pulls the pictures out of the interim pptx into `assets/interim/` |
| `present.py` | Starts the presenter with a fix for 4K renders: manim-slides forces its window to at least the video resolution, so a 4K deck showed only the top-left quarter on a 1080p screen. Takes all `manim-slides present` options |
| `qa_frames.py` | Writes the last frame of every slide to `build/qa/` plus contact sheets, for layout review |
| `assets/fonts/` | Atkinson Hyperlegible (SIL Open Font License) |
| `assets/prompt_grevys_zebra.txt` | The full generation prompt shown on the prompt slide |

## Setup

From this directory (`thesis/presentation/slides/`):

```
uv sync
uv run python extract_assets.py       # needs ../interim-presentation.pptx (local only)
```

Inputs that are not in git:
- `../interim-presentation.pptx` and `../interim-presentation-images/`. They are gitignored
  because of size and the NDA slides.
- `data/` at the repo root (inaturalist, wikimedia, synthetic images), for the photos.
- **Missing:** `assets/extra/lion_gpt-image-2-low.jpg`, one lion from the
  `gpt-image-2-low-full` cell (`data/synthetic_model_comparison/train/` on the GPU
  server). Until it exists, the lineup slide shows a dashed placeholder.

## Render, present, export

```
uv run manim render -qk --frame_rate 30 talk.py Intro Problem Data Generators Bands Models Close
uv run python present.py -F Intro Problem Data Generators Bands Models Close
uv run manim-slides convert --offline Intro Problem Data Generators Bands Models Close build/talk.html
```

- **Render at 4K for the real thing** (`-qk`, about 10 minutes for the whole talk at 30 fps).
  A 1080p video shown full screen on a 1440p or Retina display gets upscaled and looks
  soft. 4K scales *down* cleanly to any projector or monitor.
- `theme.py` lowers manim's hard-coded x264 `crf 23` to `crf 12` (`CRF` constant).
  Otherwise the compression blurs fine text and coloured edges on the static slides.
- Photos are resized from the render resolution, so a 4K render gets 4K-sharp photos and
  a `-ql` draft stays fast.
- Present with `present.py`, not `manim-slides present`. Stock manim-slides 5.7 sets the
  presenter's minimum size to the video resolution, so a 4K render overflows any smaller
  screen. The launcher removes that minimum, so the video scales to the window.
- `-F` starts the presenter full screen. Use `--screen N` to choose the display. If the
  window isn't full screen, Qt scales the video and it looks soft again.
- Render with plain `manim render`, not `manim-slides render`. The `Slide` base class writes
  the slide files either way, and `manim-slides render` reads `-qk` as `-q -k` and fails.
- Use `-ql` for a fast draft.
- To iterate on one act, render only that scene.
- The presenter advances with → / space and goes back with ←.
- `convert` can also write `--to pptx` (slides become embedded videos), which is useful as a
  fallback when the venue machine can't run Qt.
- Review layouts without presenting: `uv run python qa_frames.py Intro Problem …`, then
  open `build/qa/sheet_<Scene>.jpg`.

## Version pins

- `manim<0.20`: manim-slides 5.7 crashes on manim 0.22 (`write_to_movie`).
- `pillow<12`: manim 0.19 calls `Image.fromarray(mode=…)`, which Pillow 12 removed.
