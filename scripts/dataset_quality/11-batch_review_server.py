#!/usr/bin/env python3
"""
Step 3b Batch Review Server — Wildlife Dataset Quality Check

Shows a grid of images from the same class. Click to mark for decline,
Space to commit the batch (unmarked = approve, marked = decline).

Usage:
    cd /home/debian/Master-Thesis
    python3 scripts/dataset_quality/11-batch_review_server.py [--port 8081] [--host 0.0.0.0]

Access via browser at http://<tailscale-ip>:8081

Keyboard shortcuts:  Space Commit   B Bbox   Z Undo   ? Help
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

# ── Paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT        = Path(__file__).resolve().parent.parent.parent
REVIEW_QUEUE_CSV = REPO_ROOT / "reports" / "manual_review_queue.csv"
DECISIONS_FILE   = REPO_ROOT / "reports" / "review_decisions.jsonl"
CACHE_FILE       = REPO_ROOT / "reports" / "review_index.json"
DATA_DIR         = REPO_ROOT / "data"

TRUSTED_SOURCES = ["inaturalist", "gbif", "wikimedia"]
SOURCE_DISPLAY  = {
    "inaturalist": "iNaturalist",
    "gbif":        "GBIF",
    "wikimedia":   "Wikimedia",
    "openimages":  "OpenImages",
    "images_cv":   "ImagesCV",
    "coco_humans": "COCO",
}
PRIORITY_ORDER = {"P1 HIGH": 0, "P2 MED": 1, "P3 LOW": 2}
MAX_UNDO_BATCHES = 10

# ── Global state ──────────────────────────────────────────────────────────────

_state: dict = {}
app = FastAPI(title="Wildlife Batch Review")

# ── Data helpers (identical to 10-review_server.py) ───────────────────────────

def _load_class_info() -> dict[str, dict]:
    info: dict[str, dict] = {}
    with open(REVIEW_QUEUE_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            info[row["class"]] = {
                "tier":                 int(row["tier"]),
                "effective_pool":       int(row["effective_pool"]),
                "trusted_quality_pass": int(row["trusted_quality_pass"]),
                "tsn_fail_reason":      row["tsn_fail_reason"],
                "review_priority":      row["review_priority"],
                "review_notes":         row["review_notes"],
            }
    return info


def _source_mtimes() -> dict[str, float]:
    return {
        src: (DATA_DIR / src / "filter_results.jsonl").stat().st_mtime
        for src in TRUSTED_SOURCES
        if (DATA_DIR / src / "filter_results.jsonl").exists()
    }


def _scan_trusted_sources(queue_classes: set[str]) -> list[dict]:
    items: list[dict] = []
    for source in TRUSTED_SOURCES:
        jsonl = DATA_DIR / source / "filter_results.jsonl"
        if not jsonl.exists():
            continue
        print(f"  Scanning {source} ...", end=" ", flush=True)
        n = 0
        with open(jsonl, encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                if not entry.get("passed"):
                    continue
                parts = entry["filepath"].split("/")
                if len(parts) < 4:
                    continue
                cls_dir = parts[3]
                cls = cls_dir.replace("_", " ")
                if cls not in queue_classes:
                    continue
                items.append({
                    "filepath":   entry["filepath"],
                    "class":      cls,
                    "class_dir":  cls_dir,
                    "source":     source,
                    "bbox":       entry.get("bbox"),
                    "bbox_conf":  entry.get("bbox_conf"),
                    "detections": [
                        {"bbox": d["bbox"], "conf": d.get("conf", 0)}
                        for d in entry.get("detections", [])
                        if d.get("bbox")
                    ],
                })
                n += 1
        print(f"{n} images", flush=True)
    return items


def _load_or_build_cache(class_info: dict) -> list[dict]:
    mtimes = _source_mtimes()
    if CACHE_FILE.exists():
        try:
            cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
            if cache.get("mtimes") == mtimes:
                items = cache["items"]
                print(f"Loaded {len(items)} images from cache.", flush=True)
                return items
        except Exception:
            pass
    print("Building review index (first run, ~20 s) ...", flush=True)
    items = _scan_trusted_sources(set(class_info.keys()))
    try:
        CACHE_FILE.write_text(
            json.dumps({"mtimes": mtimes, "items": items}), encoding="utf-8"
        )
        print(f"Cache saved ({len(items)} images).", flush=True)
    except Exception as exc:
        print(f"Warning: cache not saved: {exc}", flush=True)
    return items


def _load_decisions() -> set[str]:
    if not DECISIONS_FILE.exists():
        return set()
    last: dict[str, str] = {}
    with open(DECISIONS_FILE, encoding="utf-8") as f:
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
    return set(last.keys())


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup() -> None:
    print("\n── Wildlife Batch Review Tool ────────────────────────", flush=True)
    class_info = _load_class_info()
    all_items  = _load_or_build_cache(class_info)
    decided    = _load_decisions()

    undecided = [it for it in all_items if it["filepath"] not in decided]

    # Group by class
    by_class: dict[str, list[dict]] = defaultdict(list)
    for item in undecided:
        by_class[item["class"]].append(item)

    # Sort within each class by filepath for determinism
    for items in by_class.values():
        items.sort(key=lambda x: x["filepath"])

    # Priority-sort classes
    def class_sort_key(cls: str) -> tuple:
        info = class_info.get(cls, {})
        return (
            PRIORITY_ORDER.get(info.get("review_priority", "P3 LOW"), 2),
            info.get("trusted_quality_pass", 999_999),
            cls,
        )

    class_order = sorted(by_class.keys(), key=class_sort_key)

    total_done  = len(decided)
    total_items = len(all_items)

    _state.update({
        "by_class":      dict(by_class),
        "class_order":   class_order,
        "class_idx":     0,
        "within_cursor": 0,
        "class_info":    class_info,
        "total_items":   total_items,
        "total_done":    total_done,
        "undo_stack":    [],
    })
    print(
        f"Ready — {len(undecided)} remaining across {len(class_order)} classes, "
        f"{total_done} already decided\n",
        flush=True,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _current_class_and_items() -> tuple[str | None, list[dict]]:
    """Return (current_class, remaining_items) advancing past exhausted classes."""
    order  = _state["class_order"]
    idx    = _state["class_idx"]
    cursor = _state["within_cursor"]
    while idx < len(order):
        cls   = order[idx]
        items = _state["by_class"].get(cls, [])
        if cursor < len(items):
            return cls, items[cursor:]
        # This class is exhausted — advance
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

    batch = remaining[:n]
    info  = _state["class_info"].get(cls, {})
    all_class_items = _state["by_class"].get(cls, [])

    return JSONResponse({
        "class":         cls,
        "class_info":    info,
        "images": [
            {
                "filepath":      it["filepath"],
                "filename":      Path(it["filepath"]).name,
                "source":        it["source"],
                "source_display": SOURCE_DISPLAY.get(it["source"], it["source"]),
                "bbox":          it["bbox"],
                "bbox_conf":     it["bbox_conf"],
                "detections":    it["detections"],
            }
            for it in batch
        ],
        "class_cursor":  _state["within_cursor"],
        "class_total":   len(all_class_items),
        "total_done":    _state["total_done"],
        "total_items":   _state["total_items"],
        "can_undo":      bool(_state["undo_stack"]),
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
    with open(DECISIONS_FILE, "a", encoding="utf-8") as f:
        for d in body.decisions:
            if d.get("decision") not in ("approve", "decline"):
                raise HTTPException(status_code=400, detail=f"bad decision: {d}")
            f.write(json.dumps({"filepath": d["filepath"], "decision": d["decision"], "ts": ts}) + "\n")

    # Save undo entry before advancing
    _state["undo_stack"].append({
        "class_idx":     _state["class_idx"],
        "within_cursor": _state["within_cursor"],
        "decisions":     list(body.decisions),
    })
    if len(_state["undo_stack"]) > MAX_UNDO_BATCHES:
        _state["undo_stack"].pop(0)

    _state["within_cursor"] += len(body.decisions)
    _state["total_done"]    += len(body.decisions)

    # Advance class if exhausted
    order  = _state["class_order"]
    idx    = _state["class_idx"]
    while idx < len(order):
        cls_items = _state["by_class"].get(order[idx], [])
        if _state["within_cursor"] < len(cls_items):
            break
        idx += 1
        _state["within_cursor"] = 0
    _state["class_idx"] = idx

    done = idx >= len(order)
    return JSONResponse({"done": done})


@app.post("/api/undo")
async def api_undo() -> JSONResponse:
    if not _state["undo_stack"]:
        raise HTTPException(status_code=400, detail="nothing to undo")
    entry = _state["undo_stack"].pop()
    ts = datetime.now(timezone.utc).isoformat()
    with open(DECISIONS_FILE, "a", encoding="utf-8") as f:
        for d in entry["decisions"]:
            f.write(json.dumps({"filepath": d["filepath"], "decision": "undo", "ts": ts}) + "\n")
    _state["total_done"]    -= len(entry["decisions"])
    _state["class_idx"]      = entry["class_idx"]
    _state["within_cursor"]  = entry["within_cursor"]
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
<title>Wildlife Batch Review</title>
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
#logo { font-weight: 700; font-size: 14px; color: var(--accent); white-space: nowrap; letter-spacing: 0.03em; }
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
#class-name { font-size: 17px; font-weight: 700; letter-spacing: 0.04em; }
.badge {
  font-size: 10px; font-weight: 700; padding: 2px 7px;
  border-radius: 3px; letter-spacing: 0.06em; white-space: nowrap;
}
.p1 { background: #6a1212; color: #ff8080; }
.p2 { background: #5a3a00; color: #ffb800; }
.p3 { background: #0e3d25; color: #4ec98a; }
#review-note { font-size: 12px; color: var(--dim); max-width: 420px; }
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
  background: #0f3d22; color: var(--accent); padding: 8px 22px; margin-left: auto;
}
#btn-commit:hover:not(:disabled) { background: #145230; }
#btn-commit.flash { background: var(--accent); color: #000; }
#btn-undo { background: var(--surface); color: var(--dim); padding: 8px 14px; border: 1px solid var(--border); }
#btn-undo:hover:not(:disabled) { color: var(--text); }
#btn-bbox { background: var(--surface); color: var(--dim); padding: 8px 12px; border: 1px solid var(--border); font-size: 12px; }
#btn-bbox:hover { color: var(--text); }
.slider-group { display: flex; align-items: center; gap: 7px; font-size: 12px; color: var(--dim); }
.slider-group label { white-space: nowrap; }
.slider-group input[type=range] { width: 90px; accent-color: var(--accent); }
.slider-group .val { color: var(--text); font-variant-numeric: tabular-nums; min-width: 2ch; text-align: right; }
#decline-count { font-size: 12px; color: var(--danger); }

/* ── Grid ── */
#grid-wrap {
  flex: 1;
  overflow-y: auto;
  padding: 12px;
}
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
.card.declined {
  border-color: var(--danger);
}
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
.card .decline-overlay {
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
.card.declined .decline-overlay { display: flex; }
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
  border-radius: 10px; padding: 28px 32px; width: 380px; max-width: 90vw;
}
#help-box h2 { color: var(--accent); margin-bottom: 18px; font-size: 16px; }
.hrow { display: flex; justify-content: space-between; align-items: center; padding: 7px 0; border-bottom: 1px solid var(--border); font-size: 13px; }
.hrow:last-child { border-bottom: none; }
.hrow kbd { background: var(--surface2); border: 1px solid var(--border); padding: 1px 7px; border-radius: 4px; font-family: monospace; font-size: 12px; }
.help-close-btn { margin-top: 18px; width: 100%; padding: 8px; background: var(--surface2); color: var(--dim); border-radius: 5px; }
.help-close-btn:hover { color: var(--text); }
#lightbox { cursor: zoom-out; }
#lb-img { max-width: 96vw; max-height: 96vh; object-fit: contain; border-radius: 4px; }

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
  <span id="logo">Wildlife Batch Review</span>
  <div id="progress-track"><div id="progress-fill"></div></div>
  <span id="progress-text">0 / 0</span>
  <button id="help-btn" title="Help [?]">?</button>
</div>

<div id="classbar">
  <span id="class-name">Loading...</span>
  <span id="priority-badge" class="badge"></span>
  <span id="review-note"></span>
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
  <span id="decline-count"></span>
  <button id="btn-commit">&#10003; Commit Batch [Space]</button>
</div>

<div id="grid-wrap">
  <div id="grid"></div>
</div>

<div id="loader-screen">Loading images...</div>
<div id="done-screen">
  <h1>&#10003; Review Complete</h1>
  <p>All images in the manual review queue have been decided.</p>
  <p style="margin-top:8px">Results saved to <code>reports/review_decisions.jsonl</code></p>
</div>

<div id="help-overlay" class="overlay">
  <div id="help-box">
    <h2>Keyboard Shortcuts</h2>
    <div class="hrow"><span>Commit batch</span><span><kbd>Space</kbd></span></div>
    <div class="hrow"><span>Undo last batch</span><span><kbd>Z</kbd></span></div>
    <div class="hrow"><span>Toggle bounding boxes</span><span><kbd>B</kbd></span></div>
    <div class="hrow"><span>Fullscreen image</span><span><kbd>click image</kbd></span></div>
    <div class="hrow"><span>Close overlay</span><span><kbd>Escape</kbd></span></div>
    <div class="hrow"><span>This help</span><span><kbd>?</kbd></span></div>
    <div style="margin-top:14px;font-size:12px;color:var(--dim)">
      Click a card to toggle it for <span style="color:var(--danger)">decline</span>.
      Unselected cards are <span style="color:var(--accent)">approved</span> on commit.
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
let batch      = [];   // current batch of image objects from server
let declined   = new Set();  // filepaths marked for decline
let showBbox   = true;
let busy       = false;
let batchN     = 20;
let canUndo    = false;

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
  declined.clear();
  try {
    const resp = await fetch(`/api/batch?n=${batchN}`);
    if (!resp.ok) throw new Error(resp.statusText);
    const data = await resp.json();
    if (data.done) { showDone(); return; }
    batch = data.images;
    canUndo = data.can_undo;
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
  const info = data.class_info || {};
  setText('review-note', info.review_notes || '');

  const badge = document.getElementById('priority-badge');
  const pri   = info.review_priority || '';
  badge.textContent = pri;
  badge.className   = 'badge ' + (pri === 'P1 HIGH' ? 'p1' : pri === 'P2 MED' ? 'p2' : 'p3');

  const clsDone = data.class_cursor || 0;
  const clsTot  = data.class_total  || 0;
  document.getElementById('class-progress').innerHTML =
    `class <strong>${clsDone}</strong> / <strong>${clsTot}</strong>`;

  const pct = data.total_items > 0 ? (data.total_done / data.total_items * 100) : 0;
  document.getElementById('progress-fill').style.width = pct.toFixed(2) + '%';
  setText('progress-text', `${data.total_done.toLocaleString()} / ${data.total_items.toLocaleString()}  (${pct.toFixed(1)}%)`);

  document.getElementById('btn-undo').disabled = !data.can_undo;
  updateDeclineCount();
}

// ── Render grid ───────────────────────────────────────────────────────────────
function renderGrid() {
  const grid = document.getElementById('grid');
  grid.innerHTML = '';
  for (const img of batch) {
    grid.appendChild(makeCard(img));
  }
  updateDeclineCount();
}

function makeCard(imgData) {
  const card = document.createElement('div');
  card.className   = 'card';
  card.dataset.fp  = imgData.filepath;

  const img = document.createElement('img');
  img.alt     = imgData.filename;
  img.loading = 'lazy';

  const canvas = document.createElement('canvas');

  const overlay = document.createElement('div');
  overlay.className   = 'decline-overlay';
  overlay.textContent = '✕';

  const caption = document.createElement('div');
  caption.className   = 'caption';
  caption.textContent = `${imgData.source_display} · ${imgData.filename}`;

  card.appendChild(img);
  card.appendChild(canvas);
  card.appendChild(overlay);
  card.appendChild(caption);

  // Load image then draw bbox
  img.onload = () => {
    if (showBbox) drawBboxOnCard(card, imgData);
  };
  img.src = `/image?path=${encodeURIComponent(imgData.filepath)}`;

  // Click: toggle decline (long-press-safe: use mousedown to avoid drag confusion)
  let downTime = 0;
  card.addEventListener('mousedown', () => { downTime = Date.now(); });
  card.addEventListener('click', (e) => {
    if (Date.now() - downTime > 400) return;  // ignore drag
    e.stopPropagation();
    toggleCard(card, imgData.filepath);
  });

  // Right-click / middle-click → lightbox
  card.addEventListener('contextmenu', (e) => {
    e.preventDefault();
    openLightbox(imgData.filepath);
  });
  img.addEventListener('dblclick', (e) => {
    e.stopPropagation();
    openLightbox(imgData.filepath);
  });

  return card;
}

function toggleCard(card, fp) {
  if (declined.has(fp)) {
    declined.delete(fp);
    card.classList.remove('declined');
  } else {
    declined.add(fp);
    card.classList.add('declined');
  }
  updateDeclineCount();
}

function updateDeclineCount() {
  const el = document.getElementById('decline-count');
  if (declined.size > 0) {
    el.textContent = `${declined.size} marked for decline`;
  } else {
    el.textContent = '';
  }
}

// ── BBox drawing ──────────────────────────────────────────────────────────────
function drawBboxOnCard(card, imgData) {
  const dets = imgData.detections;
  if (!dets || dets.length === 0) return;

  const img    = card.querySelector('img');
  const canvas = card.querySelector('canvas');

  canvas.width  = img.naturalWidth;
  canvas.height = img.naturalHeight;

  const ctx = canvas.getContext('2d');
  const W   = canvas.width;
  const H   = canvas.height;
  const lw  = Math.max(2, W / 300);
  const fs  = Math.max(10, W / 60);

  ctx.clearRect(0, 0, W, H);
  for (const det of dets) {
    if (!det.bbox || det.bbox.length < 4) continue;
    const [xc, yc, wn, hn] = det.bbox;
    const w = wn * W, h = hn * H;
    const x = (xc - wn / 2) * W, y = (yc - hn / 2) * H;

    ctx.strokeStyle = '#00e07a';
    ctx.lineWidth   = lw;
    ctx.strokeRect(x, y, w, h);

    const label = (det.conf * 100).toFixed(0) + '%';
    ctx.font     = `bold ${fs}px monospace`;
    const tw     = ctx.measureText(label).width;
    const lx     = Math.min(x, W - tw - 10);
    const ly     = y > fs + 8 ? y - 5 : y + h + fs + 5;

    ctx.fillStyle = 'rgba(0,224,122,0.88)';
    ctx.fillRect(lx - 3, ly - fs - 1, tw + 10, fs + 6);
    ctx.fillStyle = '#000';
    ctx.fillText(label, lx + 2, ly);
  }
}

function clearAllBboxes() {
  document.querySelectorAll('.card canvas').forEach(c => {
    const ctx = c.getContext('2d');
    ctx.clearRect(0, 0, c.width, c.height);
  });
}

function redrawAllBboxes() {
  const cards = document.querySelectorAll('.card');
  cards.forEach((card, i) => {
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
    decision: declined.has(img.filepath) ? 'decline' : 'approve',
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
  declined.clear();
  const resp = await fetch(`/api/batch?n=${batchN}`);
  if (!resp.ok) throw new Error(resp.statusText);
  const data = await resp.json();
  if (data.done) { showDone(); return; }
  batch = data.images;
  canUndo = data.can_undo;
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
  document.getElementById('classbar').style.display    = 'none';
  document.getElementById('toolbar').style.display     = 'none';
  document.getElementById('grid-wrap').style.display   = 'none';
  document.getElementById('loader-screen').style.display = 'none';
  document.getElementById('done-screen').style.display = 'flex';
}

function showLoader(show) {
  document.getElementById('grid-wrap').style.display    = show ? 'none' : '';
  document.getElementById('loader-screen').style.display = show ? 'flex' : 'none';
}

function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

// ── Keyboard ──────────────────────────────────────────────────────────────────
function onKey(e) {
  if (e.target.tagName === 'INPUT' || e.ctrlKey || e.metaKey || e.altKey) return;

  if (e.key === 'Escape') {
    closeLightbox();
    closeHelp();
    return;
  }

  const lb   = document.getElementById('lightbox');
  const help = document.getElementById('help-overlay');
  if (lb.classList.contains('open'))   { closeLightbox(); return; }
  if (help.classList.contains('open')) { closeHelp();     return; }

  switch (e.key) {
    case ' ':
      e.preventDefault();
      commitBatch();
      break;
    case 'z': case 'Z':
      e.preventDefault();
      undoBatch();
      break;
    case 'b': case 'B':
      toggleBbox();
      break;
    case '?':
      openHelp();
      break;
  }
}
</script>
</body>
</html>"""


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wildlife Image Batch Review Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8081, help="Port (default: 8081)")
    args = parser.parse_args()
    print(f"Starting batch review server — open http://<tailscale-ip>:{args.port} in your browser")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
