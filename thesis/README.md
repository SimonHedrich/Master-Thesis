# Thesis Writing

This directory is the home for everything related to producing the final Master's thesis **document** — as opposed to `docs/`, which holds research notes, experiment plans, and progress logs, and `research/`, which holds literature/reading material. Once manuscript writing begins, the LaTeX project (chapters, preamble, figures, bibliography) will live here.

## Current status

The LaTeX manuscript lives in `manuscript/` and is assembled by `manuscript/main.tex`. Chapter 2 (Literature Review) and the data-related sections of Chapter 3 (Methods and Implementation) are drafted; Chapters 1 and 5, the training/evaluation sections of Chapter 3 (`36-*`, `37-*`), and Results §4.2–§4.4 are still stubs marked with `% TODO`.

No LaTeX toolchain is installed in this repo's container (`pdflatex`/`latexmk`/`biber` are absent) and there is no `make` target for the thesis — the document is compiled externally (e.g. Overleaf). Structural checks (citation keys resolving against the two `.bib` files, `\label`/`\Cref` consistency, environment balance, figure paths) are therefore done by inspection rather than by compilation.

## Layout

```
manuscript/
  main.tex                        — top-level document
  preamble/                       — title page, declaration, abstracts, acronyms
  chapters/
    1-Introduction.tex            — stub
    2-Literature_Review.tex       — drafted
    3-Methods_and_Implementation/
      30-Overview.tex             — chapter lead-in, \input's the sections below
      31-Data_Sourcing_and_Taxonomy.tex
      32-Data_Quality_and_Curation.tex
      33-Synthetic_Data_Supplementation.tex
      34-Data_Augmentation.tex
      35-Synthetic_Generator_Comparison.tex
      36-Model_Training_and_Experiment_Tracking.tex   — stub
      37-Evaluation_Framework.tex                     — stub
    4-Results.tex                 — §4.1 drafted; §4.2–§4.4 stubs
    5-Discussion_and_Conclusion.tex — stub
  bibliography/
    references.bib                — project-specific entries (datasets, tools, model releases)
    references_zotero.bib         — copy of `research/literature/references.bib` (Better BibTeX keys)
  figures/plots/                  — charts copied from `reports/`
  appendices/
```

Both `.bib` files are registered with `\addbibresource`. When the Zotero library changes, refresh the copy with:

```
cp research/literature/references.bib thesis/manuscript/bibliography/references_zotero.bib
```

## Files

| File | Description |
|------|-------------|
| `manuscript/` | The LaTeX project (see layout above). |
| `bachelor-thesis-analysis.md` | Detailed analysis of the bachelor thesis's document structure, LaTeX setup, chapter-by-chapter content patterns, and academic writing style — with explicit notes on what to carry over as-is vs. what needs to change for the Master's thesis. The writing conventions in §7 are the style contract the manuscript follows. |

## Source material

- `resources/Thesis_Bachelor/` — the bachelor thesis Overleaf export (LaTeX source, figures, bibliography) that `bachelor-thesis-analysis.md` is based on.
