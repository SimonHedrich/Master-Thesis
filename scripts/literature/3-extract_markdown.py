# /// script
# requires-python = ">=3.13"
# dependencies = ["docling>=2.15", "pdfplumber>=0.11", "tqdm>=4.66"]
# ///
"""
Extract each PDF in research/literature/sources/ to an LLM-readable Markdown
file next to it (<citekey>.md), using docling (layout-aware, handles
two-column academic PDFs and tables) with a pdfplumber raw-text fallback if
docling fails on a given file.

Idempotent/resumable: skips any <citekey>.pdf whose <citekey>.md already
exists, unless --force. Safe to interrupt (Ctrl-C) and rerun, including as a
long-running background job — updates sources/INDEX.md's Extraction column
after each file, not just at the end.

Run:
    uv run --script scripts/literature/3-extract_markdown.py
    uv run --script scripts/literature/3-extract_markdown.py --only zhangShuffleNetExtremelyEfficient2018
    uv run --script scripts/literature/3-extract_markdown.py --limit 5
    uv run --script scripts/literature/3-extract_markdown.py --force
"""

from __future__ import annotations

import argparse
import json
import re
import traceback
from pathlib import Path

from tqdm import tqdm

from _lit_utils import render_index_md

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SOURCES_DIR = REPO_ROOT / "research" / "literature" / "sources"
MAPPING_PATH = Path(__file__).resolve().parent / "mapping.json"
ERROR_LOG_PATH = Path(__file__).resolve().parent / "extraction_errors.log"


def extract_with_docling(converter, pdf_path: Path) -> str:
    result = converter.convert(str(pdf_path))
    return result.document.export_to_markdown()


def extract_with_pdfplumber_fallback(pdf_path: Path, error: str) -> str:
    import pdfplumber

    pages_text = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            pages_text.append(page.extract_text() or "")
    body = "\n\n".join(pages_text)
    header = f"<!-- extracted via pdfplumber fallback; docling failed: {error} -->\n\n"
    return header + body


def rebuild_index() -> None:
    mapping = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))
    rows = []
    for rec in mapping:
        citekey = rec["citekey"]
        has_pdf = (SOURCES_DIR / f"{citekey}.pdf").exists()
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", type=str, default=None, help="Process only this citekey")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N PDFs")
    parser.add_argument("--force", action="store_true", help="Re-extract even if .md already exists")
    args = parser.parse_args()

    pdf_paths = sorted(SOURCES_DIR.glob("*.pdf"))
    if args.only:
        pdf_paths = [p for p in pdf_paths if p.stem == args.only]
        if not pdf_paths:
            raise SystemExit(f"No PDF found for citekey {args.only!r} in {SOURCES_DIR}")

    todo = [p for p in pdf_paths if args.force or not p.with_suffix(".md").exists()]
    if args.limit is not None:
        todo = todo[: args.limit]

    print(f"PDFs total: {len(pdf_paths)}, to process: {len(todo)}")
    if not todo:
        return

    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    # This corpus is overwhelmingly born-digital text PDFs, not scans — OCR is
    # unneeded and its model loading/inference is the main memory cost per file.
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

    docling_ok, fallback_ok, failed = 0, 0, 0
    for pdf_path in tqdm(todo, desc="Extracting"):
        citekey = pdf_path.stem
        md_path = pdf_path.with_suffix(".md")
        try:
            md_text = extract_with_docling(converter, pdf_path)
            md_path.write_text(md_text, encoding="utf-8")
            docling_ok += 1
        except Exception as exc:  # noqa: BLE001 - want a fallback for any docling failure
            err_str = f"{type(exc).__name__}: {exc}"
            with open(ERROR_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(f"=== {citekey} ===\n{traceback.format_exc()}\n")
            try:
                md_text = extract_with_pdfplumber_fallback(pdf_path, err_str)
                md_path.write_text(md_text, encoding="utf-8")
                fallback_ok += 1
            except Exception as fallback_exc:  # noqa: BLE001
                failed += 1
                with open(ERROR_LOG_PATH, "a", encoding="utf-8") as f:
                    f.write(f"=== {citekey} (fallback also failed) ===\n{fallback_exc}\n")
                continue
        rebuild_index()

    print(f"docling:  {docling_ok}")
    print(f"fallback: {fallback_ok}")
    print(f"failed:   {failed}")
    if fallback_ok or failed:
        print(f"See {ERROR_LOG_PATH} for details")


if __name__ == "__main__":
    main()
