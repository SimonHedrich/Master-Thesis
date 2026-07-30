#!/usr/bin/env python3
"""
Stage 4 — Multi-animal bbox labeling server (per generator-comparison cell)

Adapted from scripts/synthetic/5-bbox_labeling_server.py for this
experiment's per-cell layout — pass --generator/--prompt-regime to pick which
cell's md_detections.jsonl / single_detect_flags.jsonl / manual_labels.jsonl
to serve. No "_test" queue source — this experiment has no synthetic test
data (its test set is real-only, built by 0-build_test_subset.py).

Interactive per-image bbox editor where MD bboxes are pre-loaded as a
starting point. The user can add, remove, or adjust bboxes, then Save & Next
or Skip.

Queue sources (within the selected cell):
  - md_detections.jsonl       where n_significant >= 2 or == 0
  - single_detect_flags.jsonl where decision == "multi" (Stage 3)

Output: <cell_dir>/manual_labels.jsonl

Usage:
    uv run python scripts/synthetic_model_comparison/4-bbox_labeling_server.py \\
        --generator gemini-3.1-flash-image-preview --prompt-regime full [--port 8083]

Access via browser at http://<tailscale-ip>:8083

Keyboard shortcuts:  S Save&Next   N Skip   Escape Deselect   Del Delete box
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_ROOT = REPO_ROOT / "data" / "synthetic_model_comparison" / "train"

_state: dict = {}
app = FastAPI(title="Synthetic Model Comparison — Bbox Labeling")


# ── Data helpers ───────────────────────────────────────────────────────────────

def _load_labels(labels_file: Path) -> dict[str, dict]:
    """Read manual_labels.jsonl; last entry per filepath wins."""
    if not labels_file.exists():
        return {}
    last: dict[str, dict] = {}
    with open(labels_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            fp = entry.get("filepath", "")
            if fp:
                last[fp] = entry
    return last


def _load_queue(md_detections: Path, flags_file: Path) -> list[dict]:
    """Build the ordered queue of multi-animal images to label."""
    md_by_path: dict[str, dict] = {}
    if md_detections.exists():
        with open(md_detections, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                md_by_path[entry["filepath"]] = entry

    queue_fps: set[str] = set()
    queue_items: list[dict] = []

    for fp, entry in md_by_path.items():
        n = entry.get("n_significant", 0)
        if n >= 2 or n == 0:
            queue_fps.add(fp)
            queue_items.append(entry)

    if flags_file.exists():
        flags: dict[str, str] = {}
        with open(flags_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                fp = entry.get("filepath", "")
                dec = entry.get("decision", "")
                if dec == "undo":
                    flags.pop(fp, None)
                elif dec in ("single", "multi"):
                    flags[fp] = dec
        for fp, dec in flags.items():
            if dec == "multi" and fp not in queue_fps and fp in md_by_path:
                queue_fps.add(fp)
                queue_items.append(md_by_path[fp])

    queue_items.sort(key=lambda x: (x.get("class", ""), x["filepath"]))
    return queue_items


# ── Startup ────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup() -> None:
    force = _state.get("force", False)
    md_detections = _state["md_detections"]
    flags_file = _state["flags_file"]
    labels_file = _state["labels_file"]

    print("\n── Synthetic Model Comparison — Bbox Labeling ───────────────", flush=True)
    print(f"  Cell: {_state['generator']}/{_state['prompt_regime']}", flush=True)
    print(f"  Loading queue from md_detections.jsonl ...", end=" ", flush=True)
    queue = _load_queue(md_detections, flags_file)
    print(f"{len(queue)} images in queue", flush=True)

    labels = _load_labels(labels_file)
    print(f"  {len(labels)} images already labeled", flush=True)

    if force:
        start_idx = 0
    else:
        start_idx = len(queue)
        for i, item in enumerate(queue):
            if item["filepath"] not in labels:
                start_idx = i
                break

    _state.update({"queue": queue, "idx": start_idx, "labels": labels})

    remaining = sum(1 for q in queue if q["filepath"] not in labels)
    print(f"  Ready — {remaining} remaining, starting at index {start_idx}\n", flush=True)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/")
async def root() -> HTMLResponse:
    return HTMLResponse(content=HTML_PAGE)


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


@app.get("/api/image")
async def api_image() -> JSONResponse:
    queue = _state["queue"]
    idx = _state["idx"]
    labels = _state["labels"]

    if idx >= len(queue):
        return JSONResponse({"done": True})

    item = queue[idx]
    fp = item["filepath"]

    if fp in labels:
        label = labels[fp]
        bboxes = label.get("bboxes", [])
        labeled = True
        skipped = label.get("skipped", False)
    else:
        bboxes = [
            {"bbox": det["bbox"], "source": "megadetector", "conf": det.get("conf")}
            for det in item.get("detections", [])
        ]
        labeled = False
        skipped = False

    total_labeled = sum(1 for q in queue if q["filepath"] in labels)

    return JSONResponse({
        "filepath": fp,
        "class": item.get("class", ""),
        "width": item.get("width", 0),
        "height": item.get("height", 0),
        "bboxes": bboxes,
        "idx": idx,
        "total": len(queue),
        "total_labeled": total_labeled,
        "labeled": labeled,
        "skipped": skipped,
        "done": False,
    })


class SaveBody(BaseModel):
    filepath: str
    bboxes: list[dict]


@app.post("/api/save")
async def api_save(body: SaveBody) -> JSONResponse:
    queue = _state["queue"]
    idx = _state["idx"]
    labels = _state["labels"]

    if idx >= len(queue):
        raise HTTPException(status_code=400, detail="queue exhausted")

    item = queue[idx]
    ts = datetime.now(timezone.utc).isoformat()
    entry = {
        "filepath": body.filepath,
        "class": item.get("class", ""),
        "skipped": False,
        "bboxes": body.bboxes,
        "labeled_at": ts,
    }

    with open(_state["labels_file"], "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    labels[body.filepath] = entry
    new_idx = min(idx + 1, len(queue))
    _state["idx"] = new_idx

    return JSONResponse({"ok": True, "done": new_idx >= len(queue)})


class SkipBody(BaseModel):
    filepath: str


@app.post("/api/skip")
async def api_skip(body: SkipBody) -> JSONResponse:
    queue = _state["queue"]
    idx = _state["idx"]
    labels = _state["labels"]

    if idx >= len(queue):
        raise HTTPException(status_code=400, detail="queue exhausted")

    item = queue[idx]
    ts = datetime.now(timezone.utc).isoformat()
    entry = {
        "filepath": body.filepath,
        "class": item.get("class", ""),
        "skipped": True,
        "bboxes": [],
        "labeled_at": ts,
    }

    with open(_state["labels_file"], "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    labels[body.filepath] = entry
    new_idx = min(idx + 1, len(queue))
    _state["idx"] = new_idx

    return JSONResponse({"ok": True, "done": new_idx >= len(queue)})


@app.post("/api/prev")
async def api_prev() -> JSONResponse:
    _state["idx"] = max(0, _state["idx"] - 1)
    return JSONResponse({"ok": True, "idx": _state["idx"]})


@app.get("/api/progress")
async def api_progress() -> JSONResponse:
    queue = _state["queue"]
    labels = _state["labels"]
    labeled = sum(1 for q in queue if q["filepath"] in labels)
    return JSONResponse({"labeled": labeled, "total": len(queue), "idx": _state["idx"]})


# ── Embedded single-page app ───────────────────────────────────────────────────
# Identical bbox-editor UI/UX to scripts/synthetic/5-bbox_labeling_server.py —
# only the cosmetic title/done-text differ (no dynamic path — this server can
# point at any cell).

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Bbox Labeling — Synthetic Model Comparison</title>
<style>
:root {
  --bg:      #111;
  --surface: #1c1c1c;
  --border:  #2e2e2e;
  --text:    #ddd;
  --dim:     #777;
  --green:   #00e07a;
  --cyan:    #00ccff;
  --red:     #ff4d4d;
  --yellow:  #f0c040;
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
#header {
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  padding: 7px 14px;
  display: flex;
  align-items: center;
  gap: 12px;
  flex-shrink: 0;
  min-height: 40px;
}
#logo { font-weight: 700; font-size: 13px; color: var(--green); white-space: nowrap; }
#progress-track { flex: 1; height: 5px; background: var(--border); border-radius: 3px; overflow: hidden; }
#progress-fill  { height: 100%; background: var(--green); width: 0; transition: width 0.3s; }
#progress-text  { font-size: 11px; color: var(--dim); white-space: nowrap; font-variant-numeric: tabular-nums; }
#help-btn {
  background: none; border: 1px solid var(--border); color: var(--dim);
  border-radius: 50%; width: 22px; height: 22px; cursor: pointer; font-size: 12px;
  display: flex; align-items: center; justify-content: center; flex-shrink: 0;
}
#help-btn:hover { color: var(--text); }
#main { flex: 1; display: flex; overflow: hidden; }
#canvas-wrap { flex: 1; position: relative; background: #000; }
#canvas { position: absolute; top: 0; left: 0; width: 100%; height: 100%; cursor: crosshair; display: block; }
#sidebar {
  width: 210px;
  flex-shrink: 0;
  background: var(--surface);
  border-left: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  padding: 14px 12px;
  gap: 10px;
  overflow-y: auto;
}
#class-name { font-size: 15px; font-weight: 700; letter-spacing: 0.05em; color: var(--green); word-break: break-all; }
#img-path { font-size: 9px; color: var(--dim); word-break: break-all; line-height: 1.4; }
.status-badge { display: inline-flex; align-items: center; font-size: 10px; font-weight: 700; padding: 2px 8px; border-radius: 3px; letter-spacing: 0.08em; }
.badge-unlabeled { background: #2a2000; color: var(--yellow); }
.badge-labeled   { background: #0a2a14; color: var(--green);  }
.badge-skipped   { background: #2a0a0a; color: var(--red);    }
.divider { border: none; border-top: 1px solid var(--border); }
#bbox-info { font-size: 11px; color: var(--dim); }
#bbox-info strong { color: var(--text); }
button {
  cursor: pointer; border: none; border-radius: 5px;
  font-size: 13px; font-weight: 600;
  transition: background 0.12s, opacity 0.1s;
}
button:disabled { opacity: 0.3; cursor: default; }
.btn { display: block; width: 100%; padding: 9px; text-align: center; }
#btn-save { background: #0f3d22; color: var(--green); }
#btn-save:hover:not(:disabled) { background: #145230; }
#btn-save.flash { background: var(--green); color: #000; }
#btn-skip { background: var(--bg); color: var(--dim); border: 1px solid var(--border); }
#btn-skip:hover:not(:disabled) { color: var(--text); }
#btn-prev { background: var(--bg); color: var(--dim); border: 1px solid var(--border); }
#btn-prev:hover:not(:disabled) { color: var(--text); }
#btn-delete { background: #2a0a0a; color: var(--red); }
#btn-delete:hover:not(:disabled) { background: #3d1010; }
.legend { width: 100%; font-size: 11px; border-collapse: collapse; }
.legend td { padding: 2px 0; vertical-align: middle; }
.legend td:last-child { color: var(--dim); }
.swatch { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 4px; vertical-align: middle; }
.keys { width: 100%; font-size: 11px; border-collapse: collapse; }
.keys td { padding: 2px 0; color: var(--dim); }
.keys td:last-child { text-align: right; }
.keys kbd { background: #252525; border: 1px solid var(--border); padding: 1px 5px; border-radius: 3px; font-family: monospace; font-size: 10px; color: var(--text); }
.overlay { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.82); z-index: 50; align-items: center; justify-content: center; }
.overlay.open { display: flex; }
#help-box {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 10px; padding: 26px 30px; width: 400px; max-width: 90vw;
}
#help-box h2 { color: var(--green); margin-bottom: 16px; font-size: 15px; }
.hrow { display: flex; justify-content: space-between; align-items: center; padding: 6px 0; border-bottom: 1px solid var(--border); font-size: 13px; }
.hrow:last-child { border-bottom: none; }
.hrow kbd { background: #252525; border: 1px solid var(--border); padding: 1px 7px; border-radius: 4px; font-family: monospace; font-size: 12px; }
.help-note { margin-top: 12px; font-size: 12px; color: var(--dim); line-height: 1.6; }
.help-note b { color: var(--text); }
.help-close { margin-top: 16px; width: 100%; padding: 8px; background: #252525; color: var(--dim); border-radius: 5px; cursor: pointer; border: none; font-size: 13px; }
.help-close:hover { color: var(--text); }
#done-overlay { flex-direction: column; gap: 16px; text-align: center; }
#done-overlay h1 { color: var(--green); font-size: 26px; }
#done-overlay p  { color: var(--dim); font-size: 14px; max-width: 420px; }
</style>
</head>
<body>

<div id="header">
  <span id="logo">Bbox Labeling &mdash; Synthetic Model Comparison</span>
  <div id="progress-track"><div id="progress-fill"></div></div>
  <span id="progress-text">0 / 0</span>
  <button id="help-btn" title="Help [?]">?</button>
</div>

<div id="main">
  <div id="canvas-wrap">
    <canvas id="canvas"></canvas>
  </div>
  <div id="sidebar">
    <div id="class-name">Loading&hellip;</div>
    <div id="img-path"></div>
    <div id="status-badge"></div>
    <hr class="divider">
    <div id="bbox-info">Boxes: <strong id="bbox-count">0</strong></div>
    <button id="btn-save"   class="btn">&#10003; Save &amp; Next [S]</button>
    <button id="btn-skip"   class="btn">&#10006; Skip [N]</button>
    <button id="btn-prev"   class="btn">&#8592; Prev</button>
    <button id="btn-delete" class="btn" disabled>&#128465; Delete Selected [Del]</button>
    <hr class="divider">
    <table class="legend">
      <tr><td><span class="swatch" style="background:#00e07a"></span></td><td>MegaDetector</td></tr>
      <tr><td><span class="swatch" style="background:#00ccff"></span></td><td>Manual</td></tr>
      <tr><td><span class="swatch" style="background:#ff4d4d"></span></td><td>Selected</td></tr>
    </table>
    <hr class="divider">
    <table class="keys">
      <tr><td>Save &amp; next</td><td><kbd>S</kbd></td></tr>
      <tr><td>Skip</td><td><kbd>N</kbd></td></tr>
      <tr><td>Delete box</td><td><kbd>Del</kbd></td></tr>
      <tr><td>Deselect</td><td><kbd>Esc</kbd></td></tr>
      <tr><td>Help</td><td><kbd>?</kbd></td></tr>
    </table>
  </div>
</div>

<div id="help-overlay" class="overlay">
  <div id="help-box">
    <h2>Controls</h2>
    <div class="hrow"><span>Save &amp; advance to next image</span><span><kbd>S</kbd></span></div>
    <div class="hrow"><span>Skip (no annotation)</span><span><kbd>N</kbd></span></div>
    <div class="hrow"><span>Delete selected box</span><span><kbd>Delete</kbd> / <kbd>Backspace</kbd></span></div>
    <div class="hrow"><span>Deselect / cancel draw</span><span><kbd>Escape</kbd></span></div>
    <div class="hrow"><span>This help</span><span><kbd>?</kbd></span></div>
    <div class="help-note">
      <b>Draw:</b> Click &amp; drag on empty canvas area<br>
      <b>Move:</b> Click inside a box, then drag<br>
      <b>Resize:</b> Drag the 8 white handles on the selected box<br>
      <b>Select:</b> Click any existing box
    </div>
    <button class="help-close" onclick="closeHelp()">Close</button>
  </div>
</div>

<div id="done-overlay" class="overlay">
  <h1>&#10003; All images labeled!</h1>
  <p>All multi-animal images in this cell's queue have been processed.</p>
  <p>Results saved to this cell's <code>manual_labels.jsonl</code></p>
  <p style="font-size:12px;color:#555">Re-run with <code>--force</code> to relabel from the start.</p>
</div>

<script>
'use strict';

var bboxes      = [];
var selectedIdx = -1;
var imgData     = null;
var img         = new Image();
var mouseState  = null;
var busy        = false;

var canvas = document.getElementById('canvas');
var ctx    = canvas.getContext('2d');

document.addEventListener('DOMContentLoaded', function() {
  document.getElementById('btn-save').addEventListener('click', saveAndNext);
  document.getElementById('btn-skip').addEventListener('click', skipImage);
  document.getElementById('btn-prev').addEventListener('click', prevImage);
  document.getElementById('btn-delete').addEventListener('click', deleteSelected);
  document.getElementById('help-btn').addEventListener('click', openHelp);

  canvas.addEventListener('mousedown',  onMouseDown);
  canvas.addEventListener('mousemove',  onMouseMove);
  canvas.addEventListener('mouseup',    onMouseUp);
  canvas.addEventListener('mouseleave', onMouseLeave);

  document.addEventListener('keydown', onKey);
  window.addEventListener('resize', function() { resizeCanvas(); render(); });

  resizeCanvas();
  loadImage();
});

function resizeCanvas() {
  var wrap = document.getElementById('canvas-wrap');
  canvas.width  = wrap.clientWidth;
  canvas.height = wrap.clientHeight;
}

function bboxToPx(b) {
  var cw = canvas.width, ch = canvas.height;
  return {
    x1: (b.xc - b.w / 2) * cw,
    y1: (b.yc - b.h / 2) * ch,
    x2: (b.xc + b.w / 2) * cw,
    y2: (b.yc + b.h / 2) * ch,
  };
}

function pxToYolo(ax, ay, bx, by) {
  var cw = canvas.width, ch = canvas.height;
  var lx = Math.max(0, Math.min(ax, bx));
  var rx = Math.min(cw, Math.max(ax, bx));
  var ty = Math.max(0, Math.min(ay, by));
  var vy = Math.min(ch, Math.max(ay, by));
  return {
    xc: (lx + rx) / 2 / cw,
    yc: (ty + vy) / 2 / ch,
    w:  (rx - lx) / cw,
    h:  (vy - ty) / ch,
  };
}

function getHandles(px) {
  var x1 = px.x1, y1 = px.y1, x2 = px.x2, y2 = px.y2;
  var xm = (x1 + x2) / 2, ym = (y1 + y2) / 2;
  return [
    {x: x1, y: y1}, {x: xm, y: y1}, {x: x2, y: y1},
    {x: x1, y: ym},                  {x: x2, y: ym},
    {x: x1, y: y2}, {x: xm, y: y2}, {x: x2, y: y2},
  ];
}

var HANDLE_CURSORS = [
  'nw-resize', 'n-resize', 'ne-resize',
  'w-resize',              'e-resize',
  'sw-resize', 's-resize', 'se-resize',
];

function hitHandle(px, mx, my) {
  var handles = getHandles(px);
  for (var i = 0; i < handles.length; i++) {
    var dx = handles[i].x - mx, dy = handles[i].y - my;
    if (dx * dx + dy * dy <= 144) return i;
  }
  return -1;
}

function hitBbox(px, mx, my) {
  return mx >= px.x1 && mx <= px.x2 && my >= px.y1 && my <= px.y2;
}

function getMousePos(e) {
  var rect = canvas.getBoundingClientRect();
  return {mx: e.clientX - rect.left, my: e.clientY - rect.top};
}

function applyHandleResize(origPx, handleIdx, dx, dy) {
  var x1 = origPx.x1, y1 = origPx.y1, x2 = origPx.x2, y2 = origPx.y2;
  if      (handleIdx === 0) { x1 += dx; y1 += dy; }
  else if (handleIdx === 1) { y1 += dy; }
  else if (handleIdx === 2) { x2 += dx; y1 += dy; }
  else if (handleIdx === 3) { x1 += dx; }
  else if (handleIdx === 4) { x2 += dx; }
  else if (handleIdx === 5) { x1 += dx; y2 += dy; }
  else if (handleIdx === 6) { y2 += dy; }
  else if (handleIdx === 7) { x2 += dx; y2 += dy; }
  return {x1: x1, y1: y1, x2: x2, y2: y2};
}

function applyMove(origPx, dx, dy) {
  var x1 = origPx.x1 + dx, y1 = origPx.y1 + dy;
  var x2 = origPx.x2 + dx, y2 = origPx.y2 + dy;
  var W = canvas.width, H = canvas.height;
  if (x1 < 0)  { x2 -= x1;       x1 = 0; }
  if (y1 < 0)  { y2 -= y1;       y1 = 0; }
  if (x2 > W)  { x1 -= x2 - W;  x2 = W; }
  if (y2 > H)  { y1 -= y2 - H;  y2 = H; }
  return {x1: x1, y1: y1, x2: x2, y2: y2};
}

function render() {
  var cw = canvas.width, ch = canvas.height;
  ctx.clearRect(0, 0, cw, ch);

  if (img.complete && img.naturalWidth > 0) {
    ctx.drawImage(img, 0, 0, cw, ch);
  } else {
    ctx.fillStyle = '#111';
    ctx.fillRect(0, 0, cw, ch);
  }

  for (var i = 0; i < bboxes.length; i++) {
    var b   = bboxes[i];
    var px  = bboxToPx(b);
    var bw  = px.x2 - px.x1, bh = px.y2 - px.y1;
    var sel = (i === selectedIdx);

    var color = sel ? '#ff4d4d' : (b.source === 'megadetector' ? '#00e07a' : '#00ccff');

    ctx.fillStyle   = color;
    ctx.globalAlpha = 0.07;
    ctx.fillRect(px.x1, px.y1, bw, bh);
    ctx.globalAlpha = 1;

    ctx.strokeStyle = color;
    ctx.lineWidth   = sel ? 2.5 : 2;
    ctx.strokeRect(px.x1, px.y1, bw, bh);

    var label = b.source === 'megadetector'
      ? ('MD ' + (b.conf != null ? (b.conf * 100).toFixed(0) + '%' : '?'))
      : 'manual';
    ctx.font = 'bold 12px monospace';
    var tw = ctx.measureText(label).width + 8;
    var lx = Math.min(px.x1, cw - tw);
    var labelTop = (px.y1 > 18) ? (px.y1 - 18) : px.y2;

    ctx.fillStyle   = color;
    ctx.globalAlpha = 0.9;
    ctx.fillRect(lx, labelTop, tw, 16);
    ctx.globalAlpha = 1;
    ctx.fillStyle = '#000';
    ctx.fillText(label, lx + 4, labelTop + 12);

    if (sel) {
      var handles = getHandles(px);
      for (var j = 0; j < handles.length; j++) {
        ctx.fillStyle   = '#fff';
        ctx.strokeStyle = '#333';
        ctx.lineWidth   = 1;
        ctx.beginPath();
        ctx.arc(handles[j].x, handles[j].y, 6, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }
    }
  }

  if (mouseState && mouseState.mode === 'draw') {
    var sx = mouseState.startX, sy = mouseState.startY;
    var cx2 = mouseState.curX,  cy2 = mouseState.curY;
    ctx.strokeStyle = '#fff';
    ctx.lineWidth   = 1.5;
    ctx.setLineDash([5, 5]);
    ctx.strokeRect(
      Math.min(sx, cx2), Math.min(sy, cy2),
      Math.abs(cx2 - sx), Math.abs(cy2 - sy)
    );
    ctx.setLineDash([]);
  }
}

function onMouseDown(e) {
  if (busy) return;
  var pos = getMousePos(e);
  var mx = pos.mx, my = pos.my;

  if (selectedIdx >= 0 && selectedIdx < bboxes.length) {
    var spx = bboxToPx(bboxes[selectedIdx]);
    var h   = hitHandle(spx, mx, my);
    if (h >= 0) {
      mouseState = {mode: 'resize', bboxIdx: selectedIdx, handleIdx: h,
                    startX: mx, startY: my, origPx: spx};
      return;
    }
  }

  for (var i = bboxes.length - 1; i >= 0; i--) {
    var bpx = bboxToPx(bboxes[i]);
    if (hitBbox(bpx, mx, my)) {
      selectedIdx = i;
      mouseState  = {mode: 'move', bboxIdx: i,
                     startX: mx, startY: my, origPx: bpx};
      updateDeleteBtn();
      render();
      return;
    }
  }

  selectedIdx = -1;
  mouseState  = {mode: 'draw', startX: mx, startY: my, curX: mx, curY: my};
  updateDeleteBtn();
  render();
}

function onMouseMove(e) {
  var pos = getMousePos(e);
  var mx = pos.mx, my = pos.my;

  if (!mouseState) {
    if (selectedIdx >= 0 && selectedIdx < bboxes.length) {
      var spx = bboxToPx(bboxes[selectedIdx]);
      var h   = hitHandle(spx, mx, my);
      if (h >= 0) { canvas.style.cursor = HANDLE_CURSORS[h]; return; }
    }
    for (var i = bboxes.length - 1; i >= 0; i--) {
      if (hitBbox(bboxToPx(bboxes[i]), mx, my)) { canvas.style.cursor = 'move'; return; }
    }
    canvas.style.cursor = 'crosshair';
    return;
  }

  if (mouseState.mode === 'draw') {
    mouseState.curX = mx;
    mouseState.curY = my;
  } else if (mouseState.mode === 'move') {
    var dx  = mx - mouseState.startX, dy = my - mouseState.startY;
    var npx = applyMove(mouseState.origPx, dx, dy);
    var y   = pxToYolo(npx.x1, npx.y1, npx.x2, npx.y2);
    bboxes[mouseState.bboxIdx] = merge(bboxes[mouseState.bboxIdx], y);
  } else if (mouseState.mode === 'resize') {
    var rdx = mx - mouseState.startX, rdy = my - mouseState.startY;
    var rpx = applyHandleResize(mouseState.origPx, mouseState.handleIdx, rdx, rdy);
    var ry  = pxToYolo(rpx.x1, rpx.y1, rpx.x2, rpx.y2);
    if (ry.w * canvas.width >= 10 && ry.h * canvas.height >= 10) {
      bboxes[mouseState.bboxIdx] = merge(bboxes[mouseState.bboxIdx], ry);
    }
  }

  render();
}

function onMouseUp(e) {
  if (!mouseState) return;
  var pos = getMousePos(e);
  var mx = pos.mx, my = pos.my;

  if (mouseState.mode === 'draw') {
    var dx = Math.abs(mx - mouseState.startX);
    var dy = Math.abs(my - mouseState.startY);
    if (dx > 10 && dy > 10) {
      var y = pxToYolo(mouseState.startX, mouseState.startY, mx, my);
      bboxes.push({xc: y.xc, yc: y.yc, w: y.w, h: y.h, source: 'manual', conf: null});
      selectedIdx = bboxes.length - 1;
    }
  }

  mouseState = null;
  canvas.style.cursor = 'crosshair';
  updateDeleteBtn();
  updateBboxCount();
  render();
}

function onMouseLeave() {
  if (mouseState && mouseState.mode === 'draw') {
    mouseState = null;
    render();
  }
}

function merge(base, patch) {
  return {xc: patch.xc, yc: patch.yc, w: patch.w, h: patch.h,
          source: base.source, conf: base.conf};
}

function updateSidebar(data) {
  document.getElementById('class-name').textContent =
    data.class ? data.class.toUpperCase() : '';

  var parts = data.filepath.split('/');
  document.getElementById('img-path').textContent =
    parts.slice(-3).join('/');

  var badgeEl = document.getElementById('status-badge');
  if (data.skipped) {
    badgeEl.innerHTML = '<span class="status-badge badge-skipped">SKIPPED</span>';
  } else if (data.labeled) {
    badgeEl.innerHTML = '<span class="status-badge badge-labeled">&#10003; LABELED</span>';
  } else {
    badgeEl.innerHTML = '<span class="status-badge badge-unlabeled">UNLABELED</span>';
  }

  var total   = data.total || 0;
  var labeled = data.total_labeled || 0;
  var pct     = total > 0 ? (labeled / total * 100) : 0;
  document.getElementById('progress-fill').style.width = pct.toFixed(2) + '%';
  document.getElementById('progress-text').textContent =
    labeled + ' / ' + total + '  (' + pct.toFixed(1) + '%)';

  document.getElementById('btn-prev').disabled = (data.idx === 0);
  updateBboxCount();
  updateDeleteBtn();
}

function updateBboxCount() {
  document.getElementById('bbox-count').textContent = bboxes.length;
}

function updateDeleteBtn() {
  document.getElementById('btn-delete').disabled = (selectedIdx < 0);
}

function apiBboxesFromData(data) {
  return data.bboxes.map(function(b) {
    return {xc: b.bbox[0], yc: b.bbox[1], w: b.bbox[2], h: b.bbox[3],
            source: b.source, conf: b.conf};
  });
}

function applyImageData(data) {
  imgData     = data;
  selectedIdx = -1;
  mouseState  = null;
  bboxes      = apiBboxesFromData(data);
  updateSidebar(data);

  var newImg   = new Image();
  newImg.onload  = function() { img = newImg; render(); };
  newImg.onerror = function() { img = newImg; render(); };
  newImg.src     = '/image?path=' + encodeURIComponent(data.filepath);
}

async function loadImage() {
  if (busy) return;
  busy = true;
  try {
    var resp = await fetch('/api/image');
    if (!resp.ok) throw new Error(resp.statusText);
    var data = await resp.json();
    if (data.done) { showDone(); return; }
    applyImageData(data);
  } catch(err) {
    console.error('loadImage:', err);
  } finally {
    busy = false;
  }
}

async function _fetchImage() {
  var resp = await fetch('/api/image');
  if (!resp.ok) throw new Error(resp.statusText);
  var data = await resp.json();
  if (data.done) { showDone(); return false; }
  applyImageData(data);
  return true;
}

async function saveAndNext() {
  if (busy || !imgData) return;
  busy = true;
  try {
    var payload = {
      filepath: imgData.filepath,
      bboxes: bboxes.map(function(b) {
        return {bbox: [b.xc, b.yc, b.w, b.h], source: b.source, conf: b.conf};
      }),
    };
    var btn = document.getElementById('btn-save');
    btn.classList.add('flash');
    setTimeout(function() { btn.classList.remove('flash'); }, 200);

    var resp = await fetch('/api/save', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
    });
    var result = await resp.json();
    if (result.done) { showDone(); return; }
    await _fetchImage();
  } catch(err) {
    console.error('saveAndNext:', err);
  } finally {
    busy = false;
  }
}

async function skipImage() {
  if (busy || !imgData) return;
  busy = true;
  try {
    var resp = await fetch('/api/skip', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({filepath: imgData.filepath}),
    });
    var result = await resp.json();
    if (result.done) { showDone(); return; }
    await _fetchImage();
  } catch(err) {
    console.error('skipImage:', err);
  } finally {
    busy = false;
  }
}

async function prevImage() {
  if (busy || !imgData) return;
  busy = true;
  try {
    await fetch('/api/prev', {method: 'POST'});
    await _fetchImage();
  } catch(err) {
    console.error('prevImage:', err);
  } finally {
    busy = false;
  }
}

function deleteSelected() {
  if (selectedIdx < 0 || selectedIdx >= bboxes.length) return;
  bboxes.splice(selectedIdx, 1);
  selectedIdx = -1;
  updateDeleteBtn();
  updateBboxCount();
  render();
}

function showDone() {
  document.getElementById('main').style.display = 'none';
  document.getElementById('done-overlay').classList.add('open');
}

function openHelp()  { document.getElementById('help-overlay').classList.add('open'); }
function closeHelp() { document.getElementById('help-overlay').classList.remove('open'); }

function onKey(e) {
  if (e.target.tagName === 'INPUT' || e.ctrlKey || e.metaKey || e.altKey) return;
  var help = document.getElementById('help-overlay');
  if (e.key === 'Escape') {
    if (help.classList.contains('open')) { closeHelp(); return; }
    selectedIdx = -1;
    updateDeleteBtn();
    render();
    return;
  }
  if (help.classList.contains('open')) { closeHelp(); return; }
  switch (e.key) {
    case 's': case 'S': e.preventDefault(); saveAndNext();    break;
    case 'n': case 'N': e.preventDefault(); skipImage();      break;
    case 'Delete': case 'Backspace': e.preventDefault(); deleteSelected(); break;
    case '?': openHelp(); break;
  }
}
</script>
</body>
</html>"""


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic Model Comparison — Multi-animal bbox labeling server")
    parser.add_argument("--generator", required=True, metavar="NAME")
    parser.add_argument("--prompt-regime", required=True, choices=["full", "compressed"])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8083)
    parser.add_argument("--force", action="store_true",
                        help="Re-label already-labeled images (start from index 0)")
    args = parser.parse_args()

    cell_dir = TRAIN_ROOT / args.generator / args.prompt_regime
    md_detections = cell_dir / "md_detections.jsonl"
    if not md_detections.exists():
        raise SystemExit(f"ERROR: {md_detections} not found — run 2-run_megadetector.py for this cell first")

    _state["generator"] = args.generator
    _state["prompt_regime"] = args.prompt_regime
    _state["md_detections"] = md_detections
    _state["flags_file"] = cell_dir / "single_detect_flags.jsonl"
    _state["labels_file"] = cell_dir / "manual_labels.jsonl"
    if args.force:
        _state["force"] = True

    print(f"Starting bbox labeling server for {args.generator}/{args.prompt_regime}")
    print(f"Open http://<tailscale-ip>:{args.port} in your browser")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
