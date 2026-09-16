# /// script
# requires-python = ">=3.13"
# dependencies = []
# ///
"""
Copy matched source PDFs (per scripts/literature/mapping.json, produced by
1-build_zotero_mapping.py) into research/literature/sources/, renamed to
<citekey>.pdf, and (re)generate research/literature/sources/INDEX.md.

Idempotent: safe to re-run — skips PDFs already copied with a matching size,
and always regenerates INDEX.md (including the Extraction column, based on
whether <citekey>.md already exists on disk).

Run:
    uv run --script scripts/literature/2-copy_sources_and_build_index.py
    uv run --script scripts/literature/2-copy_sources_and_build_index.py --zotero-dir /path/to/Zotero
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from _lit_utils import render_index_md

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MAPPING_PATH = Path(__file__).resolve().parent / "mapping.json"
SOURCES_DIR = REPO_ROOT / "research" / "literature" / "sources"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zotero-dir",
        type=Path,
        default=Path.home() / "Zotero",
        help="Path to the Zotero data directory containing storage/ (default: ~/Zotero)",
    )
    args = parser.parse_args()

    mapping = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))
    storage_dir = args.zotero_dir / "storage"
    SOURCES_DIR.mkdir(parents=True, exist_ok=True)

    copied, skipped_existing, missing = 0, 0, 0
    rows = []
    for rec in mapping:
        citekey = rec["citekey"]
        has_pdf = False
        if rec["matched"] and rec["storage_key"] and rec["zotero_filename"]:
            src = storage_dir / rec["storage_key"] / rec["zotero_filename"]
            dst = SOURCES_DIR / f"{citekey}.pdf"
            if src.exists():
                if dst.exists() and dst.stat().st_size == src.stat().st_size:
                    skipped_existing += 1
                else:
                    shutil.copy2(src, dst)
                    copied += 1
                has_pdf = True
            else:
                missing += 1
                print(f"WARNING: source file missing on disk for {citekey}: {src}")

        note = rec.get("note") or ""

        rows.append(
            {
                "citekey": citekey,
                "title": rec["title"],
                "year": rec["year"],
                "zotero_filename": rec.get("zotero_filename") or "",
                "has_pdf": has_pdf,
                "has_extraction": (SOURCES_DIR / f"{citekey}.md").exists(),
                "note": note,
            }
        )

    (SOURCES_DIR / "INDEX.md").write_text(render_index_md(rows), encoding="utf-8")

    print(f"Copied:          {copied}")
    print(f"Already present: {skipped_existing}")
    print(f"Missing on disk: {missing}")
    print(f"No PDF expected: {sum(1 for r in rows if not r['has_pdf'])}")
    print(f"Wrote {SOURCES_DIR / 'INDEX.md'}")
    print("Unmatched Zotero PDFs (not copied): see scripts/literature/unmatched_zotero_pdfs.csv")


if __name__ == "__main__":
    main()
