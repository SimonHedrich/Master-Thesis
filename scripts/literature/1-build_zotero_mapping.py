# /// script
# requires-python = ">=3.13"
# dependencies = ["bibtexparser>=1.4.1,<2"]
# ///
"""
Match research/literature/references.bib citekeys to PDF attachments in the
local Zotero library, via a read-only snapshot of zotero.sqlite (Zotero.app
holds a live lock on the real file, so this copies it to a scratch temp dir
first rather than querying it in place).

Writes:
    scripts/literature/mapping.json            — one record per bib entry
    scripts/literature/unmatched_zotero_pdfs.csv — Zotero PDFs with no bib match

Run:
    uv run --script scripts/literature/1-build_zotero_mapping.py
    uv run --script scripts/literature/1-build_zotero_mapping.py --zotero-dir /path/to/Zotero
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import shutil
import sqlite3
import tempfile
from pathlib import Path

import bibtexparser

from _lit_utils import bib_first_author_surname, fingerprint, normalize_title

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BIB_PATH = REPO_ROOT / "research" / "literature" / "references.bib"
MAPPING_PATH = Path(__file__).resolve().parent / "mapping.json"
UNMATCHED_CSV_PATH = Path(__file__).resolve().parent / "unmatched_zotero_pdfs.csv"

FUZZY_THRESHOLD = 0.92

ZOTERO_PDF_QUERY = """
SELECT
    p.itemID          AS parentItemID,
    tval.value        AS title,
    dval.value        AS date,
    attitem.key       AS storageKey,
    att.path          AS attPath
FROM items p
JOIN itemData td          ON td.itemID = p.itemID
JOIN fields tf             ON tf.fieldID = td.fieldID AND tf.fieldName = 'title'
JOIN itemDataValues tval   ON tval.valueID = td.valueID
LEFT JOIN itemData dd      ON dd.itemID = p.itemID
                          AND dd.fieldID = (SELECT fieldID FROM fields WHERE fieldName = 'date')
LEFT JOIN itemDataValues dval ON dval.valueID = dd.valueID
JOIN itemAttachments att  ON att.parentItemID = p.itemID
JOIN items attitem        ON attitem.itemID = att.itemID
WHERE att.contentType = 'application/pdf'
"""

ZOTERO_FIRST_AUTHOR_QUERY = """
SELECT c.lastName
FROM itemCreators ic
JOIN creators c ON c.creatorID = ic.creatorID
WHERE ic.itemID = ? AND ic.orderIndex = 0
LIMIT 1
"""


def load_bib_entries(bib_path: Path) -> list[dict]:
    with open(bib_path, encoding="utf-8") as f:
        db = bibtexparser.load(f)
    return db.entries


def load_zotero_pdf_rows(zotero_dir: Path) -> list[dict]:
    sqlite_src = zotero_dir / "zotero.sqlite"
    if not sqlite_src.exists():
        raise FileNotFoundError(f"zotero.sqlite not found at {sqlite_src}")

    with tempfile.TemporaryDirectory() as tmp:
        sqlite_copy = Path(tmp) / "zotero.sqlite"
        shutil.copy2(sqlite_src, sqlite_copy)
        for suffix in ("-wal", "-shm"):
            side = zotero_dir / f"zotero.sqlite{suffix}"
            if side.exists():
                shutil.copy2(side, Path(tmp) / f"zotero.sqlite{suffix}")

        conn = sqlite3.connect(f"file:{sqlite_copy}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            rows = [dict(r) for r in conn.execute(ZOTERO_PDF_QUERY)]
            for row in rows:
                cur = conn.execute(ZOTERO_FIRST_AUTHOR_QUERY, (row["parentItemID"],))
                first = cur.fetchone()
                row["first_author"] = first[0] if first else ""
                year = ""
                if row.get("date"):
                    for token in row["date"].replace("-", " ").split():
                        if token.isdigit() and len(token) == 4:
                            year = token
                            break
                row["year"] = year
                path = row["attPath"] or ""
                row["zotero_filename"] = path[len("storage:") :] if path.startswith("storage:") else path
        finally:
            conn.close()
    return rows


def match(bib_entries: list[dict], zotero_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    for row in zotero_rows:
        row["_norm_title"] = normalize_title(row["title"])
        row["_fp"] = fingerprint(row["_norm_title"], row["year"], row["first_author"])
        row["_used"] = False  # claimed by >=1 bib entry via an exact fingerprint match

    # Group bib entries by fingerprint so that true duplicate citekeys (same
    # paper exported twice from Zotero under two citekeys) share the same
    # underlying PDF instead of the second one losing a race for it.
    zotero_by_fp: dict[str, list[dict]] = {}
    for row in zotero_rows:
        zotero_by_fp.setdefault(row["_fp"], []).append(row)

    bib_with_fp = []
    for entry in bib_entries:
        author_surname = bib_first_author_surname(entry.get("author"))
        norm_title = normalize_title(entry.get("title", ""))
        fp = fingerprint(norm_title, entry.get("year", ""), author_surname)
        bib_with_fp.append((entry, fp))

    # Citekeys sharing a fingerprint with >=1 other citekey are true bib-file
    # duplicates (same paper exported twice from Zotero) — flag them so a
    # human can merge the underlying Zotero items later.
    citekeys_by_fp: dict[str, list[str]] = {}
    for entry, fp in bib_with_fp:
        citekeys_by_fp.setdefault(fp, []).append(entry.get("ID", ""))

    group_index: dict[str, int] = {}
    mapping = []
    for entry, fp in bib_with_fp:
        citekey = entry.get("ID", "")
        siblings = [k for k in citekeys_by_fp.get(fp, []) if k != citekey]
        record = {
            "citekey": citekey,
            "title": entry.get("title", ""),
            "year": entry.get("year", ""),
            "entrytype": entry.get("ENTRYTYPE", ""),
            "matched": False,
            "match_method": None,
            "confidence": None,
            "storage_key": None,
            "zotero_filename": None,
            "note": (
                "⚠ duplicate of " + ", ".join(f"`{k}`" for k in siblings) if siblings else None
            ),
        }

        candidates = zotero_by_fp.get(fp, [])
        if candidates:
            # Round-robin across same-fingerprint zotero rows so N bib
            # duplicates sharing 1 zotero PDF all get that PDF, while N bib
            # duplicates matching N distinct zotero rows get paired 1:1.
            idx = group_index.get(fp, 0)
            chosen = candidates[idx % len(candidates)]
            group_index[fp] = idx + 1
            chosen["_used"] = True
            record.update(
                matched=True,
                match_method="exact",
                confidence=1.0,
                storage_key=chosen["storageKey"],
                zotero_filename=chosen["zotero_filename"],
            )
            mapping.append(record)
            continue

        norm_title = normalize_title(entry.get("title", ""))
        best_ratio, best_row = 0.0, None
        for r in zotero_rows:
            if r["_used"]:
                continue
            ratio = difflib.SequenceMatcher(None, norm_title, r["_norm_title"]).ratio()
            if ratio > best_ratio:
                best_ratio, best_row = ratio, r
        dup_note = record["note"]
        if best_row is not None and best_ratio >= FUZZY_THRESHOLD:
            best_row["_used"] = True
            fuzzy_note = "fuzzy title match — review before trusting"
            record.update(
                matched=True,
                match_method="fuzzy",
                confidence=round(best_ratio, 3),
                storage_key=best_row["storageKey"],
                zotero_filename=best_row["zotero_filename"],
                note=f"{dup_note}; {fuzzy_note}" if dup_note else fuzzy_note,
            )
        else:
            no_pdf_note = "no source PDF found in Zotero library"
            record["note"] = f"{dup_note}; {no_pdf_note}" if dup_note else no_pdf_note
        mapping.append(record)

    unmatched_zotero = [r for r in zotero_rows if not r["_used"]]
    return mapping, unmatched_zotero


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zotero-dir",
        type=Path,
        default=Path.home() / "Zotero",
        help="Path to the Zotero data directory containing zotero.sqlite (default: ~/Zotero)",
    )
    args = parser.parse_args()

    bib_entries = load_bib_entries(BIB_PATH)
    zotero_rows = load_zotero_pdf_rows(args.zotero_dir)
    mapping, unmatched_zotero = match(bib_entries, zotero_rows)

    MAPPING_PATH.write_text(json.dumps(mapping, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    with open(UNMATCHED_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["storage_key", "zotero_filename", "zotero_title"])
        for r in unmatched_zotero:
            writer.writerow([r["storageKey"], r["zotero_filename"], r["title"]])

    exact = sum(1 for m in mapping if m["match_method"] == "exact")
    fuzzy = sum(1 for m in mapping if m["match_method"] == "fuzzy")
    unmatched_bib = sum(1 for m in mapping if not m["matched"])

    print(f"Bib entries:          {len(bib_entries)}")
    print(f"  matched (exact):    {exact}")
    print(f"  matched (fuzzy):    {fuzzy}")
    print(f"  unmatched:          {unmatched_bib}")
    print(f"Zotero PDFs found:    {len(zotero_rows)}")
    print(f"  unmatched (unused): {len(unmatched_zotero)}")
    print(f"\nWrote {MAPPING_PATH}")
    print(f"Wrote {UNMATCHED_CSV_PATH}")


if __name__ == "__main__":
    main()
