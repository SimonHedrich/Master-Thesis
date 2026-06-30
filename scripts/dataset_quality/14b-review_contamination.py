#!/usr/bin/env python3
"""Step 14b — Contamination Review Server

Review flagged multi-animal images and confirm or reject each contamination flag
before the destructive apply step (script 15).

Shows a grid of images for the same expected class.  Each card displays all
MegaDetector bounding boxes colour-coded:
  green  — correctly identified (verdict = 'consistent')
  yellow — MegaDetector conf > 50%, misclassified by SpeciesNet
           ('flag', 'uncertain', or non-mammalian prediction)
  gray   — MegaDetector conf 10–50% (supplementary low-confidence boxes)

Click a card to confirm the flag (real contamination).
Leave it unclicked to reject it (false positive — image kept as-is).
Space commits the batch; Z undoes the last batch.

Usage:
    cd /home/debian/Master-Thesis
    python3 scripts/dataset_quality/14b-review_contamination.py [--port 8082]

Access via browser at http://<tailscale-ip>:8082

Keyboard shortcuts:  Space Commit   B Bbox   Z Undo   ? Help

Outputs:
    reports/contamination_review_decisions.jsonl  — append-only session log (resume source)
    reports/multi_animal_contamination_decisions.json — per-image decisions for script 15
        Each entry: {"decision": "edit"|"discard"|"keep", "drop_detection_idx": [...]}

See also:
    docs/plans/2026-06-09_flag-cross-species-contamination-multi-box.md
        — design spec: problem statement, tolerance-band rationale, review workflow
    docs/progress_notes/2026-06-09_contamination-flagging-and-augmentation-implementation.md
        — implementation log: execution steps, bug fixes, final run results
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

# ── Paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT       = Path(__file__).resolve().parent.parent.parent
REVIEW_JSON     = REPO_ROOT / "reports" / "multi_animal_contamination_review.json"
DECISIONS_JSONL = REPO_ROOT / "reports" / "contamination_review_decisions.jsonl"
DECISIONS_JSON  = REPO_ROOT / "reports" / "multi_animal_contamination_decisions.json"

MAX_UNDO_BATCHES = 10
DISPLAY_MD_CONF  = 0.1   # minimum MegaDetector conf for supplementary bbox display
SOURCES = ["inaturalist", "gbif", "wikimedia", "openimages", "images_cv"]

SOURCE_DISPLAY = {
    "inaturalist": "iNaturalist",
    "gbif":        "GBIF",
    "wikimedia":   "Wikimedia",
    "openimages":  "OpenImages",
    "images_cv":   "ImagesCV",
}

# ── Global state ──────────────────────────────────────────────────────────────

_state: dict = {}
app = FastAPI(title="Contamination Review")

# ── Data helpers ──────────────────────────────────────────────────────────────

def _load_review_data() -> dict[str, dict]:
    """Load review JSON, return only entries that have ≥1 box with verdict 'flag'."""
    print(f"Loading {REVIEW_JSON.name} ...", end=" ", flush=True)
    data: dict = json.loads(REVIEW_JSON.read_text(encoding="utf-8"))
    n_total = sum(
        1 for entry in data.values()
        if any(b.get("verdict") == "flag" for b in entry.get("offending_boxes", []))
    )
    flagged = {
        fp: entry
        for fp, entry in data.items()
        if any(b.get("verdict") == "flag" for b in entry.get("offending_boxes", []))
        and (REPO_ROOT / fp).is_file()
    }
    n_missing = n_total - len(flagged)
    print(f"{len(flagged):,} flagged images on disk ({n_missing:,} missing, skipped)", flush=True)
    return flagged


def _load_decisions() -> dict[str, str]:
    """Replay JSONL → {filepath: 'confirm'|'reject'}.  Undo entries erase the entry."""
    if not DECISIONS_JSONL.exists():
        return {}
    last: dict[str, str] = {}
    with open(DECISIONS_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            fp, decision = entry["filepath"], entry["decision"]
            if decision == "undo":
                last.pop(fp, None)
            else:
                last[fp] = decision
    return last


def _load_raw_detections(review_fps: set[str]) -> dict[str, list[dict]]:
    """Scan speciesnet_results.jsonl for all sources, return {filepath: [detections]}.

    Only includes detections with megadetector_conf >= DISPLAY_MD_CONF and a valid
    bbox_norm.  Used to supplement all_boxes with non-mammalian and low-conf boxes
    that script 14 excluded.
    """
    result: dict[str, list[dict]] = {}
    for source in SOURCES:
        jsonl = REPO_ROOT / "data" / source / "speciesnet_results.jsonl"
        if not jsonl.exists():
            continue
        print(f"  Loading detections {source} ...", end=" ", flush=True)
        n = 0
        with open(jsonl, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                fp = rec.get("filepath", "")
                if fp not in review_fps:
                    continue
                dets = [
                    {
                        "detection_idx":     d["detection_idx"],
                        "bbox_norm":         d["bbox_norm"],
                        "megadetector_conf": d.get("megadetector_conf", 0.0),
                    }
                    for d in (rec.get("speciesnet_detections") or [])
                    if d.get("bbox_norm") and d.get("megadetector_conf", 0.0) >= DISPLAY_MD_CONF
                ]
                if dets:
                    result[fp] = dets
                    n += 1
        print(f"{n} images", flush=True)
    return result


def _merge_boxes(fp: str, all_boxes: list[dict]) -> list[dict]:
    """Append supplementary boxes from raw detections not already in all_boxes."""
    present_idx = {b["detection_idx"] for b in all_boxes}
    extra = []
    for d in _state.get("raw_detections", {}).get(fp, []):
        if d["detection_idx"] in present_idx:
            continue
        verdict = "extra_high" if d["megadetector_conf"] > 0.5 else "extra_low"
        extra.append({
            "detection_idx":     d["detection_idx"],
            "bbox_norm":         d["bbox_norm"],
            "megadetector_conf": d["megadetector_conf"],
            "verdict":           verdict,
        })
    return all_boxes + extra


def _generate_decisions_json(
    review_data: dict[str, dict],
    decisions: dict[str, str],
) -> None:
    """Write multi_animal_contamination_decisions.json for script 15.

    confirm → edit (drop flagged box indices) or discard (if all significant boxes flagged)
    reject  → keep
    """
    out: dict[str, dict] = {}
    for fp, decision in decisions.items():
        if decision == "reject":
            out[fp] = {"decision": "keep"}
        elif decision == "confirm":
            entry = review_data.get(fp, {})
            flagged_idx = [
                b["detection_idx"]
                for b in entry.get("offending_boxes", [])
                if b.get("verdict") == "flag" and b.get("detection_idx") is not None
            ]
            if not flagged_idx:
                out[fp] = {"decision": "keep"}
                continue
            n_sig = entry.get("n_significant_boxes", len(flagged_idx) + 1)
            if len(flagged_idx) >= n_sig:
                out[fp] = {"decision": "discard"}
            else:
                out[fp] = {"decision": "edit", "drop_detection_idx": flagged_idx}
    DECISIONS_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"  Wrote {len(out):,} decisions → {DECISIONS_JSON.name}", flush=True)


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup() -> None:
    print("\n── Contamination Review Tool ──────────────────────────────────────", flush=True)

    if not REVIEW_JSON.exists():
        raise RuntimeError(
            f"Review JSON not found: {REVIEW_JSON}\n"
            "Run script 14 first to generate it."
        )

    review_data    = _load_review_data()
    raw_detections = _load_raw_detections(set(review_data.keys()))
    all_decisions  = _load_decisions()
    decided_fps    = set(all_decisions.keys())

    # Total per-class counts (across all sessions, for header display)
    class_totals: dict[str, int] = defaultdict(int)
    for entry in review_data.values():
        class_totals[entry["expected_class"]] += 1

    # Build undecided item lists grouped by class
    by_class: dict[str, list[dict]] = defaultdict(list)
    for fp, entry in review_data.items():
        if fp in decided_fps:
            continue
        by_class[entry["expected_class"]].append({
            "filepath":            fp,
            "expected_class":      entry["expected_class"],
            "source":              entry["source"],
            "n_significant_boxes": entry["n_significant_boxes"],
            "all_boxes":           entry["all_boxes"],
        })

    for items in by_class.values():
        items.sort(key=lambda x: x["filepath"])

    # Most-contaminated class first
    class_order = sorted(
        by_class.keys(),
        key=lambda c: (-class_totals[c], c),
    )

    total_items = len(review_data)
    total_done  = len(decided_fps)

    _state.update({
        "review_data":    review_data,
        "raw_detections": raw_detections,
        "by_class":      dict(by_class),
        "class_order":   class_order,
        "class_idx":     0,
        "within_cursor": 0,
        "class_totals":  dict(class_totals),
        "total_items":   total_items,
        "total_done":    total_done,
        "undo_stack":    [],
    })

    remaining = total_items - total_done
    print(
        f"Ready — {remaining:,} remaining across {len(class_order)} classes, "
        f"{total_done:,} already decided\n",
        flush=True,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _current_class_and_items() -> tuple[str | None, list[dict]]:
    order  = _state["class_order"]
    idx    = _state["class_idx"]
    cursor = _state["within_cursor"]
    while idx < len(order):
        cls   = order[idx]
        items = _state["by_class"].get(cls, [])
        if cursor < len(items):
            return cls, items[cursor:]
        idx   += 1
        cursor = 0
    _state["class_idx"]     = idx
    _state["within_cursor"] = 0
    return None, []


# ── API ───────────────────────────────────────────────────────────────────────

@app.get("/")
async def root() -> HTMLResponse:
    return HTMLResponse(content=HTML_PAGE)


@app.get("/api/batch")
async def api_batch(n: int = 20) -> JSONResponse:
    n = max(1, min(n, 200))
    cls, remaining = _current_class_and_items()
    if cls is None:
        return JSONResponse({"done": True})

    batch           = remaining[:n]
    cls_display     = cls.replace("_", " ")
    all_class_items = _state["by_class"].get(cls, [])

    return JSONResponse({
        "class":              cls_display,
        "class_total":        len(all_class_items),
        "class_flagged_total": _state["class_totals"].get(cls, 0),
        "class_cursor":       _state["within_cursor"],
        "images": [
            {
                "filepath":            it["filepath"],
                "filename":            Path(it["filepath"]).name,
                "source":              it["source"],
                "source_display":      SOURCE_DISPLAY.get(it["source"], it["source"]),
                "n_significant_boxes": it["n_significant_boxes"],
                "all_boxes":           _merge_boxes(it["filepath"], it["all_boxes"]),
            }
            for it in batch
        ],
        "total_done":  _state["total_done"],
        "total_items": _state["total_items"],
        "can_undo":    bool(_state["undo_stack"]),
    })


class CommitBody(BaseModel):
    decisions: list[dict]


@app.post("/api/commit")
async def api_commit(body: CommitBody) -> JSONResponse:
    if not body.decisions:
        raise HTTPException(status_code=400, detail="empty decisions")

    cls, _ = _current_class_and_items()
    if cls is None:
        return JSONResponse({"done": True})

    ts = datetime.now(timezone.utc).isoformat()
    with open(DECISIONS_JSONL, "a", encoding="utf-8") as f:
        for d in body.decisions:
            if d.get("decision") not in ("confirm", "reject"):
                raise HTTPException(status_code=400, detail=f"bad decision: {d}")
            f.write(json.dumps({"filepath": d["filepath"], "decision": d["decision"], "ts": ts}) + "\n")

    all_decisions = _load_decisions()
    _generate_decisions_json(_state["review_data"], all_decisions)

    _state["undo_stack"].append({
        "class_idx":     _state["class_idx"],
        "within_cursor": _state["within_cursor"],
        "decisions":     list(body.decisions),
    })
    if len(_state["undo_stack"]) > MAX_UNDO_BATCHES:
        _state["undo_stack"].pop(0)

    _state["within_cursor"] += len(body.decisions)
    _state["total_done"]    += len(body.decisions)

    order = _state["class_order"]
    idx   = _state["class_idx"]
    while idx < len(order):
        cls_items = _state["by_class"].get(order[idx], [])
        if _state["within_cursor"] < len(cls_items):
            break
        idx += 1
        _state["within_cursor"] = 0
    _state["class_idx"] = idx

    return JSONResponse({"done": idx >= len(order)})


@app.post("/api/undo")
async def api_undo() -> JSONResponse:
    if not _state["undo_stack"]:
        raise HTTPException(status_code=400, detail="nothing to undo")
    entry = _state["undo_stack"].pop()
    ts = datetime.now(timezone.utc).isoformat()
    with open(DECISIONS_JSONL, "a", encoding="utf-8") as f:
        for d in entry["decisions"]:
            f.write(json.dumps({"filepath": d["filepath"], "decision": "undo", "ts": ts}) + "\n")
    _state["total_done"]    -= len(entry["decisions"])
    _state["class_idx"]      = entry["class_idx"]
    _state["within_cursor"]  = entry["within_cursor"]

    all_decisions = _load_decisions()
    _generate_decisions_json(_state["review_data"], all_decisions)

    return JSONResponse({"ok": True})


@app.get("/image")
async def serve_image(path: str) -> FileResponse:
    full = (REPO_ROOT / path).resolve()
    repo = REPO_ROOT.resolve()
    if not str(full).startswith(str(repo)):
        raise HTTPException(status_code=403, detail="Forbidden")
    if full.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
        raise HTTPException(status_code=403, detail="Forbidden")
    if not full.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(full, media_type="image/jpeg")


# ── Embedded single-page app ──────────────────────────────────────────────────

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Contamination Review</title>
<style>
:root {
  --bg:       #111;
  --surface:  #1c1c1c;
  --surface2: #252525;
  --border:   #2e2e2e;
  --text:     #ddd;
  --dim:      #777;
  --accent:   #00e07a;
  --danger:   #ff4d4d;
  --warn:     #ffaa00;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: var(--bg);
  color: var(--text);
  font-family: system-ui, -apple-system, sans-serif;
  height: 100vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  user-select: none;
}

/* ── Header ── */
#header {
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  padding: 8px 16px;
  display: flex;
  align-items: center;
  gap: 14px;
  flex-shrink: 0;
}
#logo { font-weight: 700; font-size: 14px; color: var(--warn); white-space: nowrap; letter-spacing: 0.03em; }
#progress-track { flex: 1; height: 6px; background: var(--border); border-radius: 3px; overflow: hidden; }
#progress-fill  { height: 100%; background: var(--accent); transition: width 0.4s ease; width: 0; }
#progress-text  { font-size: 12px; color: var(--dim); white-space: nowrap; font-variant-numeric: tabular-nums; }
#help-btn {
  background: none; border: 1px solid var(--border); color: var(--dim);
  border-radius: 50%; width: 24px; height: 24px; cursor: pointer; font-size: 13px;
  display: flex; align-items: center; justify-content: center; flex-shrink: 0;
}
#help-btn:hover { color: var(--text); border-color: var(--dim); }

/* ── Class bar ── */
#classbar {
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  padding: 8px 16px;
  display: flex;
  align-items: center;
  gap: 12px;
  flex-shrink: 0;
  flex-wrap: wrap;
  min-height: 40px;
}
#class-name  { font-size: 17px; font-weight: 700; letter-spacing: 0.04em; }
#class-count { font-size: 12px; color: var(--warn); }
#class-progress { margin-left: auto; font-size: 12px; color: var(--dim); white-space: nowrap; }
#class-progress strong { color: var(--text); }

/* ── Toolbar ── */
#toolbar {
  background: var(--surface2);
  border-bottom: 1px solid var(--border);
  padding: 7px 16px;
  display: flex;
  align-items: center;
  gap: 14px;
  flex-shrink: 0;
  flex-wrap: wrap;
}
button { cursor: pointer; border: none; border-radius: 5px; font-size: 13px; font-weight: 600; transition: background 0.12s, opacity 0.1s; }
button:disabled { opacity: 0.3; cursor: default; }
#btn-commit {
  background: #3d1f0f; color: var(--warn); padding: 8px 22px; margin-left: auto;
}
#btn-commit:hover:not(:disabled) { background: #52290e; }
#btn-commit.flash { background: var(--warn); color: #000; }
#btn-undo  { background: var(--surface); color: var(--dim); padding: 8px 14px; border: 1px solid var(--border); }
#btn-undo:hover:not(:disabled) { color: var(--text); }
#btn-bbox  { background: var(--surface); color: var(--dim); padding: 8px 12px; border: 1px solid var(--border); font-size: 12px; }
#btn-bbox:hover { color: var(--text); }
.slider-group { display: flex; align-items: center; gap: 7px; font-size: 12px; color: var(--dim); }
.slider-group label { white-space: nowrap; }
.slider-group input[type=range] { width: 90px; accent-color: var(--accent); }
.slider-group .val { color: var(--text); font-variant-numeric: tabular-nums; min-width: 2ch; text-align: right; }
#confirm-count { font-size: 12px; color: var(--danger); }

/* ── Grid ── */
#grid-wrap { flex: 1; overflow-y: auto; padding: 12px; }
#grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 8px;
}
.card {
  position: relative;
  border: 2px solid var(--accent);
  border-radius: 6px;
  overflow: hidden;
  cursor: pointer;
  background: var(--surface);
  transition: border-color 0.12s;
}
.card.confirmed { border-color: var(--danger); }
.card img {
  display: block;
  width: 100%;
  aspect-ratio: 4/3;
  object-fit: cover;
}
.card canvas {
  position: absolute;
  top: 0; left: 0;
  width: 100%; height: 100%;
  pointer-events: none;
}
.card .confirm-overlay {
  display: none;
  position: absolute;
  inset: 0;
  background: rgba(255, 77, 77, 0.35);
  align-items: center;
  justify-content: center;
  font-size: 42px;
  color: #fff;
  font-weight: 900;
  text-shadow: 0 2px 8px rgba(0,0,0,0.7);
}
.card.confirmed .confirm-overlay { display: flex; }
.card .caption {
  padding: 3px 6px;
  font-size: 10px;
  color: var(--dim);
  background: rgba(0,0,0,0.55);
  position: absolute;
  bottom: 0; left: 0; right: 0;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

/* ── Overlays ── */
.overlay {
  display: none; position: fixed; inset: 0;
  background: rgba(0,0,0,0.82); z-index: 50;
  align-items: center; justify-content: center;
}
.overlay.open { display: flex; }
#help-box {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 10px; padding: 28px 32px; width: 420px; max-width: 90vw;
}
#help-box h2 { color: var(--warn); margin-bottom: 18px; font-size: 16px; }
.hrow { display: flex; justify-content: space-between; align-items: center; padding: 7px 0; border-bottom: 1px solid var(--border); font-size: 13px; }
.hrow:last-child { border-bottom: none; }
.hrow kbd { background: var(--surface2); border: 1px solid var(--border); padding: 1px 7px; border-radius: 4px; font-family: monospace; font-size: 12px; }
.help-close-btn { margin-top: 18px; width: 100%; padding: 8px; background: var(--surface2); color: var(--dim); border-radius: 5px; }
.help-close-btn:hover { color: var(--text); }
#lightbox { cursor: zoom-out; }
#lb-img { max-width: 96vw; max-height: 96vh; object-fit: contain; border-radius: 4px; }

/* ── Legend ── */
#legend {
  display: flex; gap: 14px; align-items: center;
  font-size: 11px; color: var(--dim); flex-shrink: 0;
}
.leg-dot { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 4px; vertical-align: middle; }

/* ── Done / loading ── */
#done-screen {
  display: none; flex: 1; flex-direction: column;
  align-items: center; justify-content: center;
  text-align: center; gap: 16px; padding: 40px;
}
#done-screen h1 { color: var(--accent); font-size: 28px; }
#done-screen p  { color: var(--dim); font-size: 15px; max-width: 500px; }
#loader-screen {
  display: none; flex: 1;
  align-items: center; justify-content: center;
  color: var(--dim); font-size: 14px;
}
</style>
</head>
<body>

<div id="header">
  <span id="logo">Contamination Review</span>
  <div id="progress-track"><div id="progress-fill"></div></div>
  <span id="progress-text">0 / 0</span>
  <div id="legend">
    <span><span class="leg-dot" style="background:#00e07a"></span>correct</span>
    <span><span class="leg-dot" style="background:#ffaa00"></span>misclassified (&gt;50%)</span>
    <span><span class="leg-dot" style="background:#555"></span>low-conf (10–50%)</span>
  </div>
  <button id="help-btn" title="Help [?]">?</button>
</div>

<div id="classbar">
  <span id="class-name">Loading...</span>
  <span id="class-count"></span>
  <span id="class-progress"></span>
</div>

<div id="toolbar">
  <button id="btn-undo" disabled>&#8617; Undo [Z]</button>
  <button id="btn-bbox">Hide BBox [B]</button>
  <div class="slider-group">
    <label>Columns</label>
    <input type="range" id="sl-cols" min="2" max="8" value="4" step="1">
    <span class="val" id="sl-cols-val">4</span>
  </div>
  <div class="slider-group">
    <label>Batch</label>
    <input type="range" id="sl-batch" min="4" max="64" value="20" step="2">
    <span class="val" id="sl-batch-val">20</span>
  </div>
  <span id="confirm-count"></span>
  <button id="btn-commit">&#9888; Confirm Contamination [Space]</button>
</div>

<div id="grid-wrap">
  <div id="grid"></div>
</div>

<div id="loader-screen">Loading images...</div>
<div id="done-screen">
  <h1>&#10003; Review Complete</h1>
  <p>All flagged contamination images have been decided.</p>
  <p style="margin-top:8px">Decisions saved to <code>reports/multi_animal_contamination_decisions.json</code></p>
  <p style="margin-top:4px;font-size:13px">Run script 15 to apply the decisions.</p>
</div>

<div id="help-overlay" class="overlay">
  <div id="help-box">
    <h2>Contamination Review — Help</h2>
    <div class="hrow"><span>Commit batch</span><span><kbd>Space</kbd></span></div>
    <div class="hrow"><span>Undo last batch</span><span><kbd>Z</kbd></span></div>
    <div class="hrow"><span>Toggle bounding boxes</span><span><kbd>B</kbd></span></div>
    <div class="hrow"><span>Fullscreen image</span><span><kbd>dbl-click image</kbd></span></div>
    <div class="hrow"><span>Close overlay</span><span><kbd>Escape</kbd></span></div>
    <div class="hrow"><span>This help</span><span><kbd>?</kbd></span></div>
    <div style="margin-top:14px;font-size:12px;color:var(--dim);line-height:1.6">
      <strong style="color:var(--text)">Click</strong> a card to mark it as
      <span style="color:var(--danger)">contaminated</span> (confirms the flag — offending box will be removed).<br>
      Leave it unclicked to treat it as a <span style="color:var(--accent)">false positive</span> (image kept as-is).<br><br>
      Box colours: <span style="color:#00e07a">■</span> correct &nbsp;
                   <span style="color:#ffaa00">■</span> misclassified (&gt;50% conf) &nbsp;
                   <span style="color:#555">■</span> low-conf (10–50%)
    </div>
    <button class="help-close-btn" onclick="closeHelp()">Close</button>
  </div>
</div>

<div id="lightbox" class="overlay" onclick="closeLightbox()">
  <img id="lb-img" alt="Fullscreen" />
</div>

<script>
'use strict';

// ── State ─────────────────────────────────────────────────────────────────────
let batch     = [];
let confirmed = new Set();  // filepaths marked as contaminated
let showBbox  = true;
let busy      = false;
let batchN    = 20;

// ── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('btn-commit').addEventListener('click', commitBatch);
  document.getElementById('btn-undo').addEventListener('click', undoBatch);
  document.getElementById('btn-bbox').addEventListener('click', toggleBbox);
  document.getElementById('help-btn').addEventListener('click', openHelp);

  const slCols  = document.getElementById('sl-cols');
  const slBatch = document.getElementById('sl-batch');
  slCols.addEventListener('input', () => {
    document.getElementById('sl-cols-val').textContent = slCols.value;
    document.getElementById('grid').style.gridTemplateColumns = `repeat(${slCols.value}, 1fr)`;
  });
  slBatch.addEventListener('input', () => {
    batchN = parseInt(slBatch.value, 10);
    document.getElementById('sl-batch-val').textContent = slBatch.value;
  });

  document.addEventListener('keydown', onKey);
  fetchBatch();
});

// ── Fetch batch ───────────────────────────────────────────────────────────────
async function fetchBatch() {
  if (busy) return;
  busy = true;
  showLoader(true);
  confirmed.clear();
  try {
    const resp = await fetch(`/api/batch?n=${batchN}`);
    if (!resp.ok) throw new Error(resp.statusText);
    const data = await resp.json();
    if (data.done) { showDone(); return; }
    batch = data.images;
    renderMeta(data);
    renderGrid();
  } catch (e) {
    console.error('fetchBatch error:', e);
  } finally {
    busy = false;
    showLoader(false);
  }
}

// ── Render meta ───────────────────────────────────────────────────────────────
function renderMeta(data) {
  setText('class-name', data.class.toUpperCase());
  setText('class-count', `${(data.class_flagged_total || 0).toLocaleString()} flagged`);

  const clsDone = data.class_cursor || 0;
  const clsTot  = data.class_total  || 0;
  document.getElementById('class-progress').innerHTML =
    `class <strong>${clsDone}</strong> / <strong>${clsTot}</strong> remaining`;

  const pct = data.total_items > 0 ? (data.total_done / data.total_items * 100) : 0;
  document.getElementById('progress-fill').style.width = pct.toFixed(2) + '%';
  setText('progress-text',
    `${data.total_done.toLocaleString()} / ${data.total_items.toLocaleString()}  (${pct.toFixed(1)}%)`);

  document.getElementById('btn-undo').disabled = !data.can_undo;
  updateConfirmCount();
}

// ── Render grid ───────────────────────────────────────────────────────────────
function renderGrid() {
  const grid = document.getElementById('grid');
  grid.innerHTML = '';
  for (const img of batch) grid.appendChild(makeCard(img));
  updateConfirmCount();
}

function makeCard(imgData) {
  const card = document.createElement('div');
  card.className  = 'card';
  card.dataset.fp = imgData.filepath;

  const img = document.createElement('img');
  img.alt     = imgData.filename;
  img.loading = 'lazy';

  const canvas = document.createElement('canvas');

  const overlay = document.createElement('div');
  overlay.className   = 'confirm-overlay';
  overlay.textContent = '✕';

  const caption = document.createElement('div');
  caption.className   = 'caption';
  caption.textContent = `${imgData.source_display} · ${imgData.filename}`;

  card.appendChild(img);
  card.appendChild(canvas);
  card.appendChild(overlay);
  card.appendChild(caption);

  img.onload = () => { if (showBbox) drawBboxOnCard(card, imgData); };
  img.src = `/image?path=${encodeURIComponent(imgData.filepath)}`;

  let downTime = 0;
  card.addEventListener('mousedown', () => { downTime = Date.now(); });
  card.addEventListener('click', (e) => {
    if (Date.now() - downTime > 400) return;
    e.stopPropagation();
    toggleCard(card, imgData.filepath);
  });
  img.addEventListener('dblclick', (e) => {
    e.stopPropagation();
    openLightbox(imgData.filepath);
  });
  card.addEventListener('contextmenu', (e) => {
    e.preventDefault();
    openLightbox(imgData.filepath);
  });

  return card;
}

function toggleCard(card, fp) {
  if (confirmed.has(fp)) {
    confirmed.delete(fp);
    card.classList.remove('confirmed');
  } else {
    confirmed.add(fp);
    card.classList.add('confirmed');
  }
  updateConfirmCount();
}

function updateConfirmCount() {
  const el = document.getElementById('confirm-count');
  el.textContent = confirmed.size > 0
    ? `${confirmed.size} marked as contaminated`
    : '';
}

// ── BBox drawing ──────────────────────────────────────────────────────────────
// bbox_norm format: [cx, cy, w, h] normalised (same as MegaDetector output)
// Color scheme:
//   consistent  → green   (#00e07a) — correctly identified
//   flag / uncertain / extra_high → yellow (#ffaa00) — misclassified, conf > 50%
//   extra_low   → gray    (#555)    — low-conf det, 10–50%
function drawBboxOnCard(card, imgData) {
  const boxes = imgData.all_boxes;
  if (!boxes || boxes.length === 0) return;

  const img    = card.querySelector('img');
  const canvas = card.querySelector('canvas');

  const natW = img.naturalWidth;
  const natH = img.naturalHeight;
  if (!natW || !natH) return;

  // Set canvas to card display dimensions, not natural image dimensions.
  // This is required because the <img> uses object-fit:cover (uniform scale +
  // center-crop) while a canvas sized to natW×natH would be CSS-stretched
  // non-uniformly — producing a different coordinate mapping for every
  // non-4:3 image.
  const cardW = card.clientWidth;
  const cardH = card.clientHeight;
  if (!cardW || !cardH) return;

  canvas.width  = cardW;
  canvas.height = cardH;

  // Reproduce object-fit:cover: uniform scale so both dims are covered, then
  // subtract the half-crop to get the top-left origin of the visible area.
  const scale   = Math.max(cardW / natW, cardH / natH);
  const offsetX = (natW * scale - cardW) / 2;
  const offsetY = (natH * scale - cardH) / 2;

  const ctx = canvas.getContext('2d');
  const W   = cardW;
  const H   = cardH;
  const lw  = Math.max(2, W / 300);
  const fs  = Math.max(10, W / 60);

  ctx.clearRect(0, 0, W, H);

  for (const box of boxes) {
    if (!box.bbox_norm || box.bbox_norm.length < 4) continue;
    const [xc, yc, wn, hn] = box.bbox_norm;
    // Map normalised coords → canvas pixels via the cover transform
    const bx = (xc - wn / 2) * natW * scale - offsetX;
    const by = (yc - hn / 2) * natH * scale - offsetY;
    const bw = wn * natW * scale;
    const bh = hn * natH * scale;

    const verdict = box.verdict || 'consistent';
    const isYellow = verdict === 'flag' || verdict === 'uncertain' || verdict === 'extra_high';
    const color = verdict === 'consistent' ? '#00e07a'
                : isYellow                 ? '#ffaa00'
                : '#555';

    ctx.strokeStyle = color;
    ctx.lineWidth   = verdict === 'extra_low' ? lw * 0.7 : lw;
    ctx.strokeRect(bx, by, bw, bh);

    // Label for yellow boxes only
    if (isYellow) {
      const score  = ((box.pred_top1_score || box.megadetector_conf || 0) * 100).toFixed(0);
      const name   = box.pred_common || '';
      const suffix = verdict === 'uncertain' ? '?' : '';
      const label  = name ? `${name} ${score}%${suffix}` : `det ${score}%`;
      ctx.font = `bold ${fs}px monospace`;
      const tw = ctx.measureText(label).width;
      const lx = Math.min(Math.max(bx, 0), W - tw - 10);
      const ly = by > fs + 8 ? by - 5 : by + bh + fs + 5;

      ctx.fillStyle = 'rgba(180,120,0,0.9)';
      ctx.fillRect(lx - 3, ly - fs - 1, tw + 10, fs + 6);
      ctx.fillStyle = '#fff';
      ctx.fillText(label, lx + 2, ly);
    }
  }
}

function clearAllBboxes() {
  document.querySelectorAll('.card canvas').forEach(c => {
    c.getContext('2d').clearRect(0, 0, c.width, c.height);
  });
}

function redrawAllBboxes() {
  document.querySelectorAll('.card').forEach((card, i) => {
    if (batch[i]) drawBboxOnCard(card, batch[i]);
  });
}

function toggleBbox() {
  showBbox = !showBbox;
  document.getElementById('btn-bbox').textContent = showBbox ? 'Hide BBox [B]' : 'Show BBox [B]';
  if (showBbox) redrawAllBboxes(); else clearAllBboxes();
}

// ── Commit / Undo ─────────────────────────────────────────────────────────────
async function commitBatch() {
  if (busy || batch.length === 0) return;

  const decisions = batch.map(img => ({
    filepath: img.filepath,
    decision: confirmed.has(img.filepath) ? 'confirm' : 'reject',
  }));

  const btn = document.getElementById('btn-commit');
  btn.classList.add('flash');
  setTimeout(() => btn.classList.remove('flash'), 200);

  busy = true;
  try {
    const resp = await fetch('/api/commit', {
      method:  'POST',
      headers: {'Content-Type': 'application/json'},
      body:    JSON.stringify({decisions}),
    });
    const data = await resp.json();
    if (data.done) { showDone(); return; }
    await fetchBatchUnlocked();
  } finally {
    busy = false;
  }
}

async function undoBatch() {
  if (busy) return;
  busy = true;
  try {
    const resp = await fetch('/api/undo', {method: 'POST'});
    if (resp.ok) await fetchBatchUnlocked();
  } finally {
    busy = false;
  }
}

async function fetchBatchUnlocked() {
  confirmed.clear();
  const resp = await fetch(`/api/batch?n=${batchN}`);
  if (!resp.ok) throw new Error(resp.statusText);
  const data = await resp.json();
  if (data.done) { showDone(); return; }
  batch = data.images;
  renderMeta(data);
  renderGrid();
}

// ── Overlays ──────────────────────────────────────────────────────────────────
function openLightbox(path) {
  document.getElementById('lb-img').src = `/image?path=${encodeURIComponent(path)}`;
  document.getElementById('lightbox').classList.add('open');
}
function closeLightbox() { document.getElementById('lightbox').classList.remove('open'); }
function openHelp()  { document.getElementById('help-overlay').classList.add('open'); }
function closeHelp() { document.getElementById('help-overlay').classList.remove('open'); }

function showDone() {
  document.getElementById('classbar').style.display     = 'none';
  document.getElementById('toolbar').style.display      = 'none';
  document.getElementById('grid-wrap').style.display    = 'none';
  document.getElementById('loader-screen').style.display = 'none';
  document.getElementById('done-screen').style.display  = 'flex';
}

function showLoader(show) {
  document.getElementById('grid-wrap').style.display     = show ? 'none' : '';
  document.getElementById('loader-screen').style.display = show ? 'flex' : 'none';
}

function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

// ── Keyboard ──────────────────────────────────────────────────────────────────
function onKey(e) {
  if (e.target.tagName === 'INPUT' || e.ctrlKey || e.metaKey || e.altKey) return;

  if (e.key === 'Escape') { closeLightbox(); closeHelp(); return; }

  const lb   = document.getElementById('lightbox');
  const help = document.getElementById('help-overlay');
  if (lb.classList.contains('open'))   { closeLightbox(); return; }
  if (help.classList.contains('open')) { closeHelp();     return; }

  switch (e.key) {
    case ' ':         e.preventDefault(); commitBatch(); break;
    case 'z': case 'Z': e.preventDefault(); undoBatch();  break;
    case 'b': case 'B': toggleBbox();                      break;
    case '?':           openHelp();                        break;
  }
}
</script>
</body>
</html>"""


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Contamination Review Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8082, help="Port (default: 8082)")
    parser.add_argument(
        "--review-json",
        type=Path,
        default=None,
        help="Path to review JSON (default: reports/multi_animal_contamination_review.json). "
             "Pass reports/lowconf_contamination_review.json to review the low-conf tier.",
    )
    parser.add_argument(
        "--decisions-jsonl",
        type=Path,
        default=None,
        help="Path for append-only session log (default: reports/contamination_review_decisions.jsonl).",
    )
    parser.add_argument(
        "--decisions-json",
        type=Path,
        default=None,
        help="Path for per-image decisions JSON consumed by the apply script "
             "(default: reports/multi_animal_contamination_decisions.json). "
             "Use reports/lowconf_contamination_decisions.json for the low-conf tier.",
    )
    args = parser.parse_args()

    # Override module-level path constants when the caller supplies custom paths.
    if args.review_json is not None:
        REVIEW_JSON = args.review_json
    if args.decisions_jsonl is not None:
        DECISIONS_JSONL = args.decisions_jsonl
    if args.decisions_json is not None:
        DECISIONS_JSON = args.decisions_json

    print(f"Starting contamination review server — open http://<tailscale-ip>:{args.port} in your browser")
    print(f"  review-json:      {REVIEW_JSON}")
    print(f"  decisions-jsonl:  {DECISIONS_JSONL}")
    print(f"  decisions-json:   {DECISIONS_JSON}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
