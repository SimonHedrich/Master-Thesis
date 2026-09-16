# Literature (Zotero-managed)

> Keep this file up to date whenever new files are added to this directory.

This directory holds the formal literature corpus for the thesis, managed via [Zotero](https://www.zotero.org/) rather than hand-curated like the rest of `research/`. It complements the papers and notes in the parent `research/` directory — see [`../README.md`](../README.md).

## Files

| File | Description |
|------|-------------|
| `references.bib` | BibTeX bibliography exported from the Zotero library (Better BibTeX citekeys, e.g. `zhangShuffleNetExtremelyEfficient2018`), 199 entries covering object detection, model compression/quantization/knowledge distillation, wildlife/camera-trap classification, and synthetic data generation. |
| `claude_literature.md` | Condensed LLM-generated literature survey (TL;DR + key findings + recommendations + caveats), organized by thesis-outline section, verifying DOIs/arXiv IDs and flagging literature gaps. |
| `gemini_literature.md` | Granular per-paper LLM-generated survey (author/venue/identifier/citekey/why-it-matters/confidence per entry), organized by the same thesis-outline sections, in much greater depth than `claude_literature.md`. |
| `sources/` | Source PDFs (copied from Zotero's local attachment storage, gitignored — see `make sync-ignored`) and their Markdown extractions (tracked in git), one pair per citekey. See [`sources/INDEX.md`](sources/INDEX.md) for the full citekey ↔ PDF ↔ extraction mapping. |

## Regenerating `sources/`

The `sources/` directory is built by `scripts/literature/` (a small pipeline of `uv run --script` utilities, not the repo's usual `-m scripts.<pkg>.<module>` form — see the note in `CLAUDE.md`'s "Running Code" section for why):

1. `uv run --script scripts/literature/1-build_zotero_mapping.py` — matches `references.bib` citekeys to PDF attachments in the local Zotero library (queries a snapshot of `~/Zotero/zotero.sqlite`), writing `scripts/literature/mapping.json` and `scripts/literature/unmatched_zotero_pdfs.csv`.
2. `uv run --script scripts/literature/2-copy_sources_and_build_index.py` — copies matched PDFs into `sources/<citekey>.pdf` and (re)generates `sources/INDEX.md`.
3. `uv run --script scripts/literature/3-extract_markdown.py` — extracts each PDF to `sources/<citekey>.md` (docling primary, pdfplumber fallback). Idempotent/resumable — safe to interrupt and rerun.

Not every bib entry has a source PDF: some are websites, software repos, or documentation pages (e.g. `ChatGPT`, `HuggingFaceAI`, `kaggleStanfordCarsDataset2018`) with no PDF to attach — these are flagged `no source PDF found in Zotero library` in `sources/INDEX.md` rather than treated as errors. Zotero library PDFs that don't correspond to any `references.bib` entry (this Zotero library also holds unrelated research, e.g. time-series forecasting papers) are reported in `scripts/literature/unmatched_zotero_pdfs.csv` but not copied.
