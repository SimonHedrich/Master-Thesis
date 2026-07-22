#!/usr/bin/env python3
"""
Step 3 Manual Review Server — Wildlife Dataset Quality Check

Usage:
    uv run python scripts/dataset_quality/10-review_server.py [--port 8080] [--host 0.0.0.0]

Access via browser at http://<tailscale-ip>:8080

Keyboard shortcuts:  A/→ Approve   D/← Decline   Z/Backspace Undo   B Bbox   F Fullscreen   ? Help
"""
from __future__ import annotations

import argparse
import csv
import json
import random
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
ALL_SOURCES     = ["inaturalist", "gbif", "wikimedia", "openimages", "images_cv", "coco_humans"]
SOURCE_DISPLAY  = {
    "inaturalist": "iNaturalist",
    "gbif":        "GBIF",
    "wikimedia":   "Wikimedia",
    "openimages":  "OpenImages",
    "images_cv":   "ImagesCV",
    "coco_humans": "COCO",
}
PRIORITY_ORDER = {"P1 HIGH": 0, "P2 MED": 1, "P3 LOW": 2}
MAX_UNDO       = 20
COMPARE_N      = 8

# ── Global state ──────────────────────────────────────────────────────────────

_state: dict = {}
app = FastAPI(title="Wildlife Review")

# ── Data helpers ──────────────────────────────────────────────────────────────

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


def _load_decisions() -> tuple[set[str], dict[str, int]]:
    if not DECISIONS_FILE.exists():
        return set(), {}
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
    decided = set(last.keys())
    count_by_class: dict[str, int] = {}
    for fp in decided:
        parts = fp.split("/")
        if len(parts) >= 4:
            cls = parts[3].replace("_", " ")
            count_by_class[cls] = count_by_class.get(cls, 0) + 1
    return decided, count_by_class


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup() -> None:
    print("\n── Wildlife Review Tool ──────────────────────────────", flush=True)
    class_info              = _load_class_info()
    all_items               = _load_or_build_cache(class_info)
    decided, class_done_cnt = _load_decisions()

    undecided = [it for it in all_items if it["filepath"] not in decided]
    undecided.sort(key=lambda it: (
        PRIORITY_ORDER.get(class_info.get(it["class"], {}).get("review_priority", "P3 LOW"), 2),
        class_info.get(it["class"], {}).get("trusted_quality_pass", 999_999),
        it["filepath"],
    ))

    _state.update({
        "queue":           undecided,
        "idx":             0,
        "undo_stack":      [],
        "class_info":      class_info,
        "class_done_cnt":  class_done_cnt,
        "total_items":     len(decided) + len(undecided),
        "n_prior":         len(decided),
        "last_decision":   None,
    })
    print(
        f"Ready — {len(undecided)} remaining, {len(decided)} already decided, "
        f"{len(decided) + len(undecided)} total\n",
        flush=True,
    )


# ── API ───────────────────────────────────────────────────────────────────────

@app.get("/")
async def root() -> HTMLResponse:
    return HTMLResponse(content=HTML_PAGE)


@app.get("/api/current")
async def api_current() -> JSONResponse:
    queue = _state["queue"]
    idx   = _state["idx"]
    if idx >= len(queue):
        raise HTTPException(status_code=404, detail="done")
    item = queue[idx]
    cls  = item["class"]
    info = _state["class_info"].get(cls, {})
    return JSONResponse({
        "filepath":       item["filepath"],
        "class":          cls,
        "class_dir":      item["class_dir"],
        "source":         item["source"],
        "source_display": SOURCE_DISPLAY.get(item["source"], item["source"]),
        "filename":       Path(item["filepath"]).name,
        "bbox":           item["bbox"],
        "bbox_conf":      item["bbox_conf"],
        "detections":     item["detections"],
        "class_info":     info,
        "class_total":    info.get("trusted_quality_pass", 0),
        "class_done":     _state["class_done_cnt"].get(cls, 0),
        "total_done":     _state["n_prior"] + idx,
        "total_items":    _state["total_items"],
        "last_decision":  _state["last_decision"],
        "can_undo":       bool(_state["undo_stack"]),
    })


class DecideBody(BaseModel):
    decision: str


@app.post("/api/decide")
async def api_decide(body: DecideBody) -> JSONResponse:
    if body.decision not in ("approve", "decline"):
        raise HTTPException(status_code=400, detail="decision must be 'approve' or 'decline'")
    queue = _state["queue"]
    idx   = _state["idx"]
    if idx >= len(queue):
        return JSONResponse({"done": True})
    item = queue[idx]
    cls  = item["class"]
    ts   = datetime.now(timezone.utc).isoformat()
    with open(DECISIONS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"filepath": item["filepath"], "decision": body.decision, "ts": ts}) + "\n")
    _state["class_done_cnt"][cls] = _state["class_done_cnt"].get(cls, 0) + 1
    _state["undo_stack"].append({"filepath": item["filepath"], "class": cls, "decision": body.decision})
    if len(_state["undo_stack"]) > MAX_UNDO:
        _state["undo_stack"].pop(0)
    _state["idx"] += 1
    _state["last_decision"] = {
        "filename": Path(item["filepath"]).name,
        "decision": body.decision,
    }
    return JSONResponse({"done": _state["idx"] >= len(queue)})


@app.post("/api/undo")
async def api_undo() -> JSONResponse:
    if not _state["undo_stack"]:
        raise HTTPException(status_code=400, detail="nothing to undo")
    last = _state["undo_stack"].pop()
    _state["idx"] -= 1
    ts = datetime.now(timezone.utc).isoformat()
    with open(DECISIONS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"filepath": last["filepath"], "decision": "undo", "ts": ts}) + "\n")
    cls = last["class"]
    _state["class_done_cnt"][cls] = max(0, _state["class_done_cnt"].get(cls, 0) - 1)
    _state["last_decision"] = None
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


@app.get("/api/class_samples")
async def api_class_samples(cls: str) -> JSONResponse:
    cls_dir = cls.replace(" ", "_")
    samples: list[str] = []
    for source in ALL_SOURCES:
        class_path = DATA_DIR / source / "images" / cls_dir
        if class_path.exists():
            imgs = [f for f in class_path.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
            random.shuffle(imgs)
            for img in imgs[:4]:
                samples.append(f"data/{source}/images/{cls_dir}/{img.name}")
    random.shuffle(samples)
    return JSONResponse({"samples": samples[:COMPARE_N]})


# ── Embedded single-page app ──────────────────────────────────────────────────

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Wildlife Review</title>
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
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0;
}
#help-btn:hover { color: var(--text); border-color: var(--dim); }

/* ── Meta panel ── */
#meta {
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  padding: 8px 16px;
  display: flex;
  align-items: center;
  gap: 16px;
  flex-wrap: wrap;
  flex-shrink: 0;
  min-height: 44px;
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
#meta-right { margin-left: auto; display: flex; gap: 20px; font-size: 12px; color: var(--dim); white-space: nowrap; }
#meta-right strong { color: var(--text); }

/* ── Main area ── */
#main {
  display: flex;
  flex: 1;
  overflow: hidden;
}

/* ── Image viewer ── */
#viewer {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 12px;
  overflow: hidden;
  position: relative;
  gap: 6px;
}
#img-wrap {
  position: relative;
  display: inline-block;
  max-width: 100%;
  max-height: calc(100% - 28px);
  line-height: 0;
}
#main-img {
  display: block;
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
  border-radius: 4px;
  cursor: zoom-in;
}
#bbox-canvas {
  position: absolute;
  top: 0; left: 0;
  width: 100%; height: 100%;
  pointer-events: none;
  border-radius: 4px;
}
#filename-bar { font-size: 11px; color: var(--dim); font-family: monospace; text-align: center; }
#loader {
  display: none;
  position: absolute;
  top: 50%; left: 50%;
  transform: translate(-50%, -50%);
  color: var(--dim);
  font-size: 13px;
}

/* ── Sidebar (comparison) ── */
#sidebar {
  width: 152px;
  background: var(--surface);
  border-left: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  flex-shrink: 0;
}
#sidebar-header {
  padding: 8px 10px 4px;
  font-size: 10px;
  font-weight: 700;
  color: var(--dim);
  letter-spacing: 0.06em;
  flex-shrink: 0;
  border-bottom: 1px solid var(--border);
}
#compare-strip {
  display: flex;
  flex-direction: column;
  gap: 3px;
  padding: 6px;
  overflow-y: auto;
  flex: 1;
}
.thumb {
  width: 100%;
  aspect-ratio: 4/3;
  object-fit: cover;
  border-radius: 3px;
  cursor: zoom-in;
  opacity: 0.75;
  transition: opacity 0.15s;
  border: 1px solid var(--border);
  display: block;
}
.thumb:hover { opacity: 1; border-color: var(--accent); }

/* ── Controls ── */
#controls {
  background: var(--surface);
  border-top: 1px solid var(--border);
  padding: 10px 16px;
  display: flex;
  align-items: center;
  gap: 10px;
  flex-shrink: 0;
  flex-wrap: wrap;
}
button { cursor: pointer; border: none; border-radius: 5px; font-size: 13px; font-weight: 600; transition: background 0.12s, transform 0.08s, opacity 0.1s; }
button:disabled { opacity: 0.3; cursor: default; }
#btn-approve { background: #0f3d22; color: var(--accent); padding: 9px 22px; }
#btn-approve:hover:not(:disabled) { background: #145230; }
#btn-approve.flash { background: var(--accent); color: #000; }
#btn-decline { background: #3d0f0f; color: var(--danger); padding: 9px 22px; }
#btn-decline:hover:not(:disabled) { background: #521414; }
#btn-decline.flash { background: var(--danger); color: #fff; }
#btn-undo { background: var(--surface2); color: var(--dim); padding: 9px 16px; }
#btn-undo:hover:not(:disabled) { color: var(--text); }
#btn-bbox { background: var(--surface2); color: var(--dim); padding: 9px 14px; font-size: 12px; }
#btn-bbox:hover { color: var(--text); }
#last-info { margin-left: auto; font-size: 12px; color: var(--dim); }
.ok  { color: var(--accent); }
.bad { color: var(--danger); }

/* ── Overlays ── */
.overlay {
  display: none;
  position: fixed; inset: 0;
  background: rgba(0,0,0,0.82);
  z-index: 50;
  align-items: center;
  justify-content: center;
}
.overlay.open { display: flex; }

/* Help */
#help-box {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 28px 32px;
  width: 380px;
  max-width: 90vw;
}
#help-box h2 { color: var(--accent); margin-bottom: 18px; font-size: 16px; }
.hrow { display: flex; justify-content: space-between; align-items: center; padding: 7px 0; border-bottom: 1px solid var(--border); font-size: 13px; }
.hrow:last-child { border-bottom: none; }
.hrow kbd { background: var(--surface2); border: 1px solid var(--border); padding: 1px 7px; border-radius: 4px; font-family: monospace; font-size: 12px; color: var(--text); }
.help-close-btn { margin-top: 18px; width: 100%; padding: 8px; background: var(--surface2); color: var(--dim); border-radius: 5px; }
.help-close-btn:hover { color: var(--text); }

/* Lightbox */
#lightbox { cursor: zoom-out; }
#lb-img { max-width: 96vw; max-height: 96vh; object-fit: contain; border-radius: 4px; }

/* Done screen */
#done-screen {
  display: none;
  flex: 1;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  gap: 16px;
  padding: 40px;
}
#done-screen h1 { color: var(--accent); font-size: 28px; }
#done-screen p  { color: var(--dim); font-size: 15px; max-width: 500px; }
</style>
</head>
<body>

<!-- Header -->
<div id="header">
  <span id="logo">Wildlife Review</span>
  <div id="progress-track"><div id="progress-fill"></div></div>
  <span id="progress-text">0 / 0</span>
  <button id="help-btn" title="Keyboard shortcuts [?]">?</button>
</div>

<!-- Meta panel -->
<div id="meta">
  <span id="class-name">Loading...</span>
  <span id="priority-badge" class="badge"></span>
  <span id="review-note"></span>
  <div id="meta-right">
    <span><strong id="source-name"></strong></span>
    <span>conf <strong id="bbox-conf"></strong></span>
    <span>class <strong id="class-progress"></strong></span>
  </div>
</div>

<!-- Main layout -->
<div id="main">

  <!-- Viewer -->
  <div id="viewer">
    <div id="img-wrap">
      <img id="main-img" alt="Review image" />
      <canvas id="bbox-canvas"></canvas>
    </div>
    <div id="filename-bar"></div>
    <div id="loader">Loading...</div>
  </div>

  <!-- Comparison sidebar -->
  <div id="sidebar">
    <div id="sidebar-header">COMPARE &mdash; <span id="compare-class"></span></div>
    <div id="compare-strip"></div>
  </div>

</div>

<!-- Controls -->
<div id="controls">
  <button id="btn-approve">&#10003; Approve &nbsp;[A]</button>
  <button id="btn-decline">&#10007; Decline &nbsp;[D]</button>
  <button id="btn-undo" disabled>&#8617; Undo [Z]</button>
  <button id="btn-bbox">Hide Box [B]</button>
  <div id="last-info"></div>
</div>

<!-- Done screen (replaces main) -->
<div id="done-screen">
  <h1>&#10003; Review Complete</h1>
  <p>All images in the manual review queue have been decided.</p>
  <p style="margin-top:8px">Results saved to <code>reports/review_decisions.jsonl</code></p>
</div>

<!-- Help overlay -->
<div id="help-overlay" class="overlay">
  <div id="help-box">
    <h2>Keyboard Shortcuts</h2>
    <div class="hrow"><span>Approve image</span><span><kbd>A</kbd> or <kbd>&rarr;</kbd></span></div>
    <div class="hrow"><span>Decline image</span><span><kbd>D</kbd> or <kbd>&larr;</kbd></span></div>
    <div class="hrow"><span>Undo last decision</span><span><kbd>Z</kbd> or <kbd>Backspace</kbd></span></div>
    <div class="hrow"><span>Toggle bounding box</span><span><kbd>B</kbd></span></div>
    <div class="hrow"><span>Fullscreen image</span><span><kbd>F</kbd> or click image</span></div>
    <div class="hrow"><span>This help</span><span><kbd>?</kbd></span></div>
    <button class="help-close-btn" onclick="closeHelp()">Close</button>
  </div>
</div>

<!-- Lightbox overlay -->
<div id="lightbox" class="overlay" onclick="closeLightbox()">
  <img id="lb-img" alt="Fullscreen" />
</div>

<script>
'use strict';

// ── State ─────────────────────────────────────────────────────────────────────
let cur        = null;   // current /api/current response
let showBbox   = true;
let curClass   = null;
let busy       = false;

// ── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  fetchCurrent();
  document.addEventListener('keydown', onKey);
  window.addEventListener('resize', redrawBbox);

  document.getElementById('btn-approve').addEventListener('click', () => decide('approve'));
  document.getElementById('btn-decline').addEventListener('click', () => decide('decline'));
  document.getElementById('btn-undo').addEventListener('click', doUndo);
  document.getElementById('btn-bbox').addEventListener('click', toggleBbox);
  document.getElementById('help-btn').addEventListener('click', openHelp);
  document.getElementById('main-img').addEventListener('click', () => openLightbox(cur && cur.filepath));
});

// ── Fetch & render ────────────────────────────────────────────────────────────
async function fetchCurrent() {
  if (busy) return;
  busy = true;
  setLoader(true);
  try {
    const resp = await fetch('/api/current');
    if (resp.status === 404) { showDone(); return; }
    if (!resp.ok) throw new Error(resp.statusText);
    cur = await resp.json();
    render();
  } catch (e) {
    console.error('fetchCurrent error:', e);
  } finally {
    busy = false;
    setLoader(false);
  }
}

function render() {
  if (!cur) return;
  updateMeta();
  loadMainImage();
  if (cur.class !== curClass) {
    curClass = cur.class;
    loadComparisons();
  }
}

function updateMeta() {
  const d = cur;
  setText('class-name', d.class.toUpperCase());
  setText('review-note', d.class_info.review_notes || '');
  setText('source-name', d.source_display);
  setText('bbox-conf',   d.bbox_conf != null ? (d.bbox_conf * 100).toFixed(1) + '%' : '—');
  setText('class-progress', `${d.class_done} / ${d.class_total}`);
  setText('filename-bar', d.filename);

  const badge = document.getElementById('priority-badge');
  const pri   = d.class_info.review_priority || '';
  badge.textContent = pri;
  badge.className   = 'badge ' + (pri === 'P1 HIGH' ? 'p1' : pri === 'P2 MED' ? 'p2' : 'p3');

  const pct = d.total_items > 0 ? (d.total_done / d.total_items * 100) : 0;
  document.getElementById('progress-fill').style.width = pct.toFixed(2) + '%';
  setText('progress-text', `${d.total_done.toLocaleString()} / ${d.total_items.toLocaleString()}  (${pct.toFixed(1)}%)`);

  document.getElementById('btn-undo').disabled = !d.can_undo;

  const lastEl = document.getElementById('last-info');
  if (d.last_decision) {
    const ld = d.last_decision;
    const icon = ld.decision === 'approve' ? '&#10003;' : '&#10007;';
    const cls  = ld.decision === 'approve' ? 'ok' : 'bad';
    lastEl.innerHTML = `Last: <span class="${cls}">${icon} ${escHtml(ld.filename)}</span>`;
  } else {
    lastEl.textContent = '';
  }
}

function loadMainImage() {
  const img    = document.getElementById('main-img');
  const canvas = document.getElementById('bbox-canvas');
  canvas.style.display = 'none';
  img.onload = () => {
    if (showBbox) drawBbox();
  };
  img.onerror = () => {
    img.alt = 'Image not found: ' + (cur && cur.filename);
  };
  img.src = '/image?path=' + encodeURIComponent(cur.filepath) + '&_=' + Date.now();
}

function drawBbox() {
  if (!cur || !showBbox) return;
  const dets = cur.detections;
  if (!dets || dets.length === 0) return;

  const img    = document.getElementById('main-img');
  const canvas = document.getElementById('bbox-canvas');

  // Match canvas internal resolution to natural image size
  canvas.width  = img.naturalWidth;
  canvas.height = img.naturalHeight;
  // Match canvas CSS size to displayed image size
  const r = img.getBoundingClientRect();
  canvas.style.width  = r.width  + 'px';
  canvas.style.height = r.height + 'px';
  canvas.style.display = 'block';

  const ctx  = canvas.getContext('2d');
  const W    = canvas.width;
  const H    = canvas.height;
  const lw   = Math.max(2, W / 300);
  const fs   = Math.max(12, W / 50);

  ctx.clearRect(0, 0, W, H);

  for (const det of dets) {
    if (!det.bbox || det.bbox.length < 4) continue;
    // Stored as YOLO [x_center, y_center, width, height] normalized (see megadetector_to_yolo in filter script)
    const [xc, yc, wn, hn] = det.bbox;
    const w = wn * W, h = hn * H;
    const x = (xc - wn / 2) * W, y = (yc - hn / 2) * H;

    ctx.strokeStyle = '#00e07a';
    ctx.lineWidth   = lw;
    ctx.strokeRect(x, y, w, h);

    const label  = (det.conf * 100).toFixed(0) + '%';
    ctx.font     = `bold ${fs}px monospace`;
    const tw     = ctx.measureText(label).width;
    const lx     = Math.min(x, W - tw - 10);
    const ly     = y > fs + 8 ? y - 5 : y + h + fs + 5;

    ctx.fillStyle = 'rgba(0, 224, 122, 0.88)';
    ctx.fillRect(lx - 3, ly - fs - 1, tw + 10, fs + 6);
    ctx.fillStyle = '#000';
    ctx.fillText(label, lx + 2, ly);
  }
}

function redrawBbox() {
  const img = document.getElementById('main-img');
  if (img.complete && img.naturalWidth > 0 && showBbox) drawBbox();
}

async function loadComparisons() {
  setText('compare-class', curClass ? curClass.toUpperCase() : '');
  const strip = document.getElementById('compare-strip');
  strip.innerHTML = '<span style="color:var(--dim);font-size:11px;padding:6px">Loading...</span>';
  try {
    const resp = await fetch('/api/class_samples?cls=' + encodeURIComponent(curClass));
    const data = await resp.json();
    strip.innerHTML = '';
    for (const path of data.samples) {
      const img = document.createElement('img');
      img.src     = '/image?path=' + encodeURIComponent(path);
      img.className = 'thumb';
      img.loading = 'lazy';
      img.title   = path.split('/').pop();
      img.onclick = (e) => { e.stopPropagation(); openLightbox(path); };
      strip.appendChild(img);
    }
    if (!data.samples.length) {
      strip.innerHTML = '<span style="color:var(--dim);font-size:11px;padding:6px">No samples</span>';
    }
  } catch {
    strip.innerHTML = '<span style="color:var(--dim);font-size:11px;padding:6px">Error</span>';
  }
}

// ── Decisions ─────────────────────────────────────────────────────────────────
async function decide(decision) {
  if (busy) return;
  const btn = document.getElementById(decision === 'approve' ? 'btn-approve' : 'btn-decline');
  btn.classList.add('flash');
  setTimeout(() => btn.classList.remove('flash'), 180);

  busy = true;
  try {
    const resp = await fetch('/api/decide', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({decision}),
    });
    const data = await resp.json();
    if (data.done) { showDone(); return; }
    await fetchCurrentUnlocked();
  } finally {
    busy = false;
  }
}

async function doUndo() {
  if (busy || !cur || !cur.can_undo) return;
  busy = true;
  try {
    const resp = await fetch('/api/undo', {method: 'POST'});
    if (resp.ok) await fetchCurrentUnlocked();
  } finally {
    busy = false;
  }
}

// Fetch without the busy guard (used inside already-locked flows)
async function fetchCurrentUnlocked() {
  setLoader(true);
  try {
    const resp = await fetch('/api/current');
    if (resp.status === 404) { showDone(); return; }
    cur = await resp.json();
    render();
  } finally {
    setLoader(false);
  }
}

// ── UI helpers ────────────────────────────────────────────────────────────────
function toggleBbox() {
  showBbox = !showBbox;
  const canvas = document.getElementById('bbox-canvas');
  if (showBbox) {
    drawBbox();
    document.getElementById('btn-bbox').textContent = 'Hide Box [B]';
  } else {
    canvas.style.display = 'none';
    document.getElementById('btn-bbox').textContent = 'Show Box [B]';
  }
}

function openLightbox(path) {
  if (!path) return;
  document.getElementById('lb-img').src = '/image?path=' + encodeURIComponent(path);
  document.getElementById('lightbox').classList.add('open');
}
function closeLightbox() { document.getElementById('lightbox').classList.remove('open'); }

function openHelp()  { document.getElementById('help-overlay').classList.add('open'); }
function closeHelp() { document.getElementById('help-overlay').classList.remove('open'); }

function showDone() {
  document.getElementById('meta').style.display    = 'none';
  document.getElementById('main').style.display    = 'none';
  document.getElementById('controls').style.display = 'none';
  const ds = document.getElementById('done-screen');
  ds.style.display = 'flex';
}

function setLoader(show) {
  document.getElementById('loader').style.display = show ? 'block' : 'none';
}
function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}
function escHtml(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// ── Keyboard ──────────────────────────────────────────────────────────────────
function onKey(e) {
  if (e.target.tagName === 'INPUT' || e.ctrlKey || e.metaKey || e.altKey) return;

  // Escape closes any overlay
  if (e.key === 'Escape') {
    closeLightbox();
    closeHelp();
    return;
  }

  const lb   = document.getElementById('lightbox');
  const help = document.getElementById('help-overlay');
  if (lb.classList.contains('open'))   { closeLightbox(); return; }
  if (help.classList.contains('open')) { closeHelp(); return; }

  switch (e.key) {
    case 'a': case 'A': case 'ArrowRight': decide('approve'); break;
    case 'd': case 'D': case 'ArrowLeft':  decide('decline'); break;
    case 'z': case 'Z':
      e.preventDefault(); doUndo(); break;
    case 'Backspace':
      e.preventDefault(); doUndo(); break;
    case 'b': case 'B': toggleBbox(); break;
    case 'f': case 'F': openLightbox(cur && cur.filepath); break;
    case '?': openHelp(); break;
  }
}
</script>
</body>
</html>"""


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wildlife Image Review Server (Step 3)")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    args = parser.parse_args()
    print(f"Starting server — open http://<tailscale-ip>:{args.port} in your browser")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
