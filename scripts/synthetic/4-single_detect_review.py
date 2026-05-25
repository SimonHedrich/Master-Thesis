#!/usr/bin/env python3
"""
Stage 4 — Single-Detection Triage Review Server

Reviews synthetic images where MegaDetector found exactly one significant animal.
Click to flag as MULTI (more animals present than MD detected).
Unclicked cards are treated as SINGLE on commit.

Usage:
    cd /home/debian/Master-Thesis
    python3 scripts/synthetic/4-single_detect_review.py [--port 8082]

Access via browser at http://<tailscale-ip>:8082

Keyboard shortcuts:  Space Commit   B Bbox   Z Undo   ? Help
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

REPO_ROOT     = Path(__file__).resolve().parent.parent.parent
MD_DETECTIONS = REPO_ROOT / "data" / "synthetic" / "md_detections.jsonl"
FLAGS_FILE    = REPO_ROOT / "data" / "synthetic" / "single_detect_flags.jsonl"

MAX_UNDO_BATCHES = 10

# ── Global state ──────────────────────────────────────────────────────────────

_state: dict = {}
app = FastAPI(title="Synthetic Triage Review")

# ── Data helpers ──────────────────────────────────────────────────────────────

def _load_items() -> list[dict]:
    items = []
    with open(MD_DETECTIONS, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("n_significant") == 1:
                items.append(entry)
    items.sort(key=lambda x: (x["class"], x["filepath"]))
    return items


def _load_decisions() -> dict[str, str]:
    if not FLAGS_FILE.exists():
        return {}
    last: dict[str, str] = {}
    with open(FLAGS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry    = json.loads(line)
            fp       = entry.get("filepath", "")
            decision = entry.get("decision", "")
            if decision == "undo":
                last.pop(fp, None)
            elif decision in ("single", "multi"):
                last[fp] = decision
    return last


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup() -> None:
    force = _state.get("force", False)
    print("\n── Synthetic Single-Detection Triage ────────────────────", flush=True)
    print(f"  Loading from {MD_DETECTIONS.name} ...", end=" ", flush=True)
    all_items = _load_items()
    print(f"{len(all_items)} items with n_significant==1", flush=True)

    decided = _load_decisions()
    print(f"  {len(decided)} already decided", flush=True)

    undecided = all_items if force else [it for it in all_items if it["filepath"] not in decided]

    by_class: dict[str, list[dict]] = defaultdict(list)
    for item in undecided:
        by_class[item["class"]].append(item)

    class_order = sorted(by_class.keys())

    _state.update({
        "by_class":      dict(by_class),
        "class_order":   class_order,
        "class_idx":     0,
        "within_cursor": 0,
        "total_items":   len(all_items),
        "total_done":    len(decided),
        "undo_stack":    [],
        "decided":       decided,
    })
    print(
        f"Ready — {len(undecided)} remaining across {len(class_order)} species\n",
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
    all_class_items = _state["by_class"].get(cls, [])

    return JSONResponse({
        "class":         cls,
        "batch": [
            {
                "filepath":   it["filepath"],
                "class":      it["class"],
                "detections": it["detections"],
            }
            for it in batch
        ],
        "class_cursor": _state["within_cursor"],
        "class_total":  len(all_class_items),
        "total_done":   _state["total_done"],
        "total_items":  _state["total_items"],
        "can_undo":     bool(_state["undo_stack"]),
    })


class CommitBody(BaseModel):
    decisions: dict[str, str]


@app.post("/api/commit")
async def api_commit(body: CommitBody) -> JSONResponse:
    if not body.decisions:
        raise HTTPException(status_code=400, detail="empty decisions")

    cls, _ = _current_class_and_items()
    if cls is None:
        return JSONResponse({"done": True})

    for fp, dec in body.decisions.items():
        if dec not in ("single", "multi"):
            raise HTTPException(status_code=400, detail=f"invalid decision '{dec}' for {fp}")

    ts  = datetime.now(timezone.utc).isoformat()
    fps = list(body.decisions.keys())

    with open(FLAGS_FILE, "a", encoding="utf-8") as f:
        for fp, dec in body.decisions.items():
            f.write(json.dumps({"filepath": fp, "decision": dec, "ts": ts}) + "\n")

    _state["decided"].update(body.decisions)
    _state["undo_stack"].append({
        "class_idx":     _state["class_idx"],
        "within_cursor": _state["within_cursor"],
        "filepaths":     fps,
    })
    if len(_state["undo_stack"]) > MAX_UNDO_BATCHES:
        _state["undo_stack"].pop(0)

    _state["within_cursor"] += len(body.decisions)
    _state["total_done"]    += len(body.decisions)

    order = _state["class_order"]
    idx   = _state["class_idx"]
    while idx < len(order):
        if _state["within_cursor"] < len(_state["by_class"].get(order[idx], [])):
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
    ts    = datetime.now(timezone.utc).isoformat()

    with open(FLAGS_FILE, "a", encoding="utf-8") as f:
        for fp in entry["filepaths"]:
            f.write(json.dumps({"filepath": fp, "decision": "undo", "ts": ts}) + "\n")

    for fp in entry["filepaths"]:
        _state["decided"].pop(fp, None)

    _state["total_done"]    -= len(entry["filepaths"])
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
<title>Synthetic Triage Review</title>
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
  min-height: 40px;
}
#class-name { font-size: 17px; font-weight: 700; letter-spacing: 0.04em; }
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
#btn-commit { background: #0f3d22; color: var(--accent); padding: 8px 22px; margin-left: auto; }
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
#multi-count { font-size: 12px; color: var(--danger); }

/* ── Grid ── */
#grid-wrap { flex: 1; overflow-y: auto; padding: 12px; }
#grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }

.card {
  position: relative;
  border: 2px solid var(--accent);
  border-radius: 6px;
  overflow: hidden;
  cursor: pointer;
  background: var(--surface);
  transition: border-color 0.12s;
}
.card.multi { border-color: var(--danger); }
.card img {
  display: block;
  width: 100%;
  aspect-ratio: 1 / 1;
  object-fit: cover;
}
.card canvas {
  position: absolute;
  top: 0; left: 0;
  width: 100%; height: 100%;
  pointer-events: none;
}
.card .multi-overlay {
  display: none;
  position: absolute;
  inset: 0;
  background: rgba(255, 77, 77, 0.30);
  align-items: flex-start;
  justify-content: flex-end;
  padding: 6px;
}
.card.multi .multi-overlay { display: flex; }
.multi-badge {
  background: var(--danger);
  color: #fff;
  font-size: 10px;
  font-weight: 900;
  padding: 3px 8px;
  border-radius: 3px;
  letter-spacing: 0.1em;
}
.card .caption {
  padding: 3px 6px;
  font-size: 10px;
  color: var(--dim);
  background: rgba(0,0,0,0.6);
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
  <span id="logo">Synthetic Triage Review</span>
  <div id="progress-track"><div id="progress-fill"></div></div>
  <span id="progress-text">0 / 0</span>
  <button id="help-btn" title="Help [?]">?</button>
</div>

<div id="classbar">
  <span id="class-name">Loading...</span>
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
  <span id="multi-count"></span>
  <button id="btn-commit">&#10003; Commit Batch [Space]</button>
</div>

<div id="grid-wrap">
  <div id="grid"></div>
</div>

<div id="loader-screen">Loading images...</div>
<div id="done-screen">
  <h1>&#10003; Triage Complete</h1>
  <p>All single-detection images have been reviewed.</p>
  <p style="margin-top:8px">Results saved to <code>data/synthetic/single_detect_flags.jsonl</code></p>
</div>

<div id="help-overlay" class="overlay">
  <div id="help-box">
    <h2>Keyboard Shortcuts</h2>
    <div class="hrow"><span>Commit batch</span><span><kbd>Space</kbd></span></div>
    <div class="hrow"><span>Undo last batch</span><span><kbd>Z</kbd></span></div>
    <div class="hrow"><span>Toggle bounding boxes</span><span><kbd>B</kbd></span></div>
    <div class="hrow"><span>Fullscreen image</span><span><kbd>right-click / dbl-click</kbd></span></div>
    <div class="hrow"><span>Close overlay</span><span><kbd>Escape</kbd></span></div>
    <div class="hrow"><span>This help</span><span><kbd>?</kbd></span></div>
    <div style="margin-top:14px;font-size:12px;color:var(--dim)">
      Click a card to flag it as <span style="color:var(--danger);font-weight:700">MULTI</span>
      (image contains more than one animal).<br>
      Unclicked cards are committed as <span style="color:var(--accent);font-weight:700">SINGLE</span>.
    </div>
    <button class="help-close-btn" onclick="closeHelp()">Close</button>
  </div>
</div>

<div id="lightbox" class="overlay" onclick="closeLightbox()">
  <img id="lb-img" alt="Fullscreen" />
</div>

<script>
'use strict';

let batch    = [];
let multi    = new Set();
let showBbox = true;
let busy     = false;
let batchN   = 20;

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

async function fetchBatch() {
  if (busy) return;
  busy = true;
  showLoader(true);
  multi.clear();
  try {
    const resp = await fetch(`/api/batch?n=${batchN}`);
    if (!resp.ok) throw new Error(resp.statusText);
    const data = await resp.json();
    if (data.done) { showDone(); return; }
    batch = data.batch;
    renderMeta(data);
    renderGrid();
  } catch (e) {
    console.error('fetchBatch error:', e);
  } finally {
    busy = false;
    showLoader(false);
  }
}

function renderMeta(data) {
  setText('class-name', data.class.toUpperCase());

  const clsDone = data.class_cursor || 0;
  const clsTot  = data.class_total  || 0;
  document.getElementById('class-progress').innerHTML =
    `species <strong>${clsDone}</strong> / <strong>${clsTot}</strong>`;

  const pct = data.total_items > 0 ? (data.total_done / data.total_items * 100) : 0;
  document.getElementById('progress-fill').style.width = pct.toFixed(2) + '%';
  setText('progress-text',
    `${data.total_done.toLocaleString()} / ${data.total_items.toLocaleString()}  (${pct.toFixed(1)}%)`);

  document.getElementById('btn-undo').disabled = !data.can_undo;
  updateMultiCount();
}

function renderGrid() {
  const grid = document.getElementById('grid');
  grid.innerHTML = '';
  for (const img of batch) {
    grid.appendChild(makeCard(img));
  }
  updateMultiCount();
}

function makeCard(imgData) {
  const card = document.createElement('div');
  card.className  = 'card';
  card.dataset.fp = imgData.filepath;

  const img = document.createElement('img');
  img.alt     = imgData.filepath.split('/').pop();
  img.loading = 'lazy';

  const canvas = document.createElement('canvas');

  const overlay = document.createElement('div');
  overlay.className = 'multi-overlay';
  const badge = document.createElement('span');
  badge.className   = 'multi-badge';
  badge.textContent = 'MULTI';
  overlay.appendChild(badge);

  const caption = document.createElement('div');
  caption.className = 'caption';
  const conf = imgData.detections && imgData.detections[0]
    ? ' · ' + (imgData.detections[0].conf * 100).toFixed(0) + '%'
    : '';
  caption.textContent = imgData.class + conf;

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
  if (multi.has(fp)) {
    multi.delete(fp);
    card.classList.remove('multi');
  } else {
    multi.add(fp);
    card.classList.add('multi');
  }
  updateMultiCount();
}

function updateMultiCount() {
  const el = document.getElementById('multi-count');
  el.textContent = multi.size > 0 ? `${multi.size} flagged as multi` : '';
}

function drawBboxOnCard(card, imgData) {
  const dets = imgData.detections;
  if (!dets || dets.length === 0) return;

  const img    = card.querySelector('img');
  const canvas = card.querySelector('canvas');
  canvas.width  = img.naturalWidth;
  canvas.height = img.naturalHeight;

  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const lw = Math.max(2, W / 300);
  const fs = Math.max(10, W / 60);

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
    ctx.font = `bold ${fs}px monospace`;
    const tw = ctx.measureText(label).width;
    const lx = Math.min(x, W - tw - 10);
    const ly = y > fs + 8 ? y - 5 : y + h + fs + 5;
    ctx.fillStyle = 'rgba(0,224,122,0.88)';
    ctx.fillRect(lx - 3, ly - fs - 1, tw + 10, fs + 6);
    ctx.fillStyle = '#000';
    ctx.fillText(label, lx + 2, ly);
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

async function commitBatch() {
  if (busy || batch.length === 0) return;

  const decisions = {};
  for (const img of batch) {
    decisions[img.filepath] = multi.has(img.filepath) ? 'multi' : 'single';
  }

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
  multi.clear();
  const resp = await fetch(`/api/batch?n=${batchN}`);
  if (!resp.ok) throw new Error(resp.statusText);
  const data = await resp.json();
  if (data.done) { showDone(); return; }
  batch = data.batch;
  renderMeta(data);
  renderGrid();
}

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
  document.getElementById('grid-wrap').style.display     = show ? 'none' : '';
  document.getElementById('loader-screen').style.display = show ? 'flex' : 'none';
}

function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

function onKey(e) {
  if (e.target.tagName === 'INPUT' || e.ctrlKey || e.metaKey || e.altKey) return;
  if (e.key === 'Escape') { closeLightbox(); closeHelp(); return; }
  const lb   = document.getElementById('lightbox');
  const help = document.getElementById('help-overlay');
  if (lb.classList.contains('open'))   { closeLightbox(); return; }
  if (help.classList.contains('open')) { closeHelp();     return; }
  switch (e.key) {
    case ' ':           e.preventDefault(); commitBatch(); break;
    case 'z': case 'Z': e.preventDefault(); undoBatch();   break;
    case 'b': case 'B': toggleBbox();                      break;
    case '?':           openHelp();                        break;
  }
}
</script>
</body>
</html>"""


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic Image Single-Detection Triage Server")
    parser.add_argument("--host",  default="0.0.0.0")
    parser.add_argument("--port",  type=int, default=8082)
    parser.add_argument("--force", action="store_true", help="Re-review already-decided images")
    args = parser.parse_args()
    if args.force:
        _state["force"] = True
    print(f"Starting triage server — open http://<tailscale-ip>:{args.port} in your browser")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
