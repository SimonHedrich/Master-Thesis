# Thesis Writing

This directory is the home for everything related to producing the final Master's thesis **document** — as opposed to `docs/`, which holds research notes, experiment plans, and progress logs, and `research/`, which holds literature/reading material. Once manuscript writing begins, the LaTeX project (chapters, preamble, figures, bibliography) will live here.

## Current status

The LaTeX manuscript lives in `manuscript/` and is assembled by `manuscript/main.tex`. All five chapters are drafted. Chapter 5 was reworked on 2026-09-28 around the answers to the research questions and the author's limitations (see `docs/2026-09-28_chapter5-discussion-critique.md` for the analysis behind it).

No LaTeX toolchain is installed in this repo's container (`pdflatex`/`latexmk`/`biber` are absent) and there is no `make` target for the thesis — the document is compiled on Overleaf, which `manuscript/` is synced to two-way by `scripts/thesis/sync_overleaf.py` (`make overleaf` picks the direction; `make overleaf-push|overleaf-pull|overleaf-status` force one — on the `thesis/overleaf-sync` branch; see `scripts/thesis/README.md`). Structural checks (citation keys resolving against the two `.bib` files, `\label`/`\Cref` consistency, environment balance, figure paths) are therefore done by inspection rather than by compilation.

## Layout

```
docs/                             — how the document gets made (toolchain, publishing)
manuscript/
  main.tex                      — top-level document
  preamble/                       — title page, declaration, abstracts, acronyms
  chapters/
    1-Introduction.tex            — drafted
    2-Literature_Review.tex       — drafted
    3-Methods_and_Implementation/
      30-Overview.tex             — chapter lead-in, \input's the sections below
      31-Data_Sourcing_and_Taxonomy.tex
      32-Data_Quality_and_Curation.tex
      33-Synthetic_Data_Supplementation.tex
      34-Data_Augmentation.tex
      35-Synthetic_Generator_Comparison.tex
      36-Model_Training_and_Experiment_Tracking.tex
      37-Evaluation_Framework.tex
    4-Results.tex                 — drafted
    5-Discussion_and_Conclusion.tex — drafted, reworked 2026-09-28
  bibliography/
    references.bib                — the single bibliography (Better BibTeX keys), exported from Zotero
  figures/plots/                  — charts copied from `reports/`
  appendices/
```

`references.bib` is the manuscript's only bibliography and the only file registered with
`\addbibresource` (`main.tex`). It is a Zotero export: to refresh it, export the library from
Zotero (Better BibTeX) directly over `thesis/manuscript/bibliography/references.bib`. Every key
cited in `chapters/` resolves there — sources with no paper (dataset portals, model cards, docs
pages) are Zotero items too, so nothing is maintained by hand and an export never loses entries.

After each export, re-check that no citation has gone dangling:

```
comm -23 \
  <(grep -rhoE '\\(cite|parencite|textcite|autocite|footcite)\*?(\[[^]]*\])*\{[^}]*\}' \
      thesis/manuscript/{chapters,preamble,appendices} thesis/manuscript/main.tex \
    | sed -E 's/.*\{([^}]*)\}/\1/' | tr ',' '\n' | tr -d ' ' | sort -u) \
  <(grep -oE '^@[a-zA-Z]+\{[^,]+,' thesis/manuscript/bibliography/references.bib \
    | sed -E 's/^@[a-zA-Z]+\{(.*),$/\1/' | sort)
```

Empty output means every cited key resolves.

## Files

| File | Description |
|------|-------------|
| `manuscript/` | The LaTeX project (see layout above). |
| `docs/` | Documentation on producing the document itself — toolchain and publishing pipeline, as opposed to the research notes in the repo-root `docs/`. See `docs/README.md`. |
| `bachelor-thesis-analysis.md` | Detailed analysis of the bachelor thesis's document structure, LaTeX setup, chapter-by-chapter content patterns, and academic writing style — with explicit notes on what to carry over as-is vs. what needs to change for the Master's thesis. The conventions in §7 and the rough edges in §8 are half of the style contract (see below). |
| `writing/2026-09-17_manuscript-baseline.md` | Measurement of the drafted chapters taken before the writing skill existed: sentence and paragraph length per file, the longest sentences quoted, terminology drift counts, and the mechanical defects found. Exists so a later revision pass has a number to move — only the delta is meaningful. |
| `writing/anti-slop-skill.md` | Saved blog post (Sascha Becker, "A 1986 Aircraft Manual Fixed My Anti-Slop Skill") that prompted the writing skill. Reference reading, not a repo-authored contract. `anti-slop-skill.html` is the source page. |

## The style contract

Writing conventions for the manuscript live in two places, and they cover different things:

- **`bachelor-thesis-analysis.md` §7–§8** — *conventions and mechanics.* Numbers in math mode,
  `\textit{}` on first mention, `\enquote{}` never raw quotes, `\Cref` not `\ref`, the
  impersonal register, heavy cross-linking, and the rough edges to avoid repeating. Unchanged
  and still binding.
- **`.claude/skills/thesis-writing/`** — *construction, terminology, and claims.* How a sentence
  and a paragraph are built, one name per concept, and the traceability rules for numbers,
  citations and comparative claims. It restates §7 as a checklist in `references/mechanics.md`
  rather than replacing it, and writes down where its borrowed style guides conflict with §7
  (active voice vs. the impersonal register, hedge-cutting vs. scientific calibration, the em
  dash) with a verdict for each.

The skill loads automatically when writing or reviewing `.tex` prose under `manuscript/`.

## Source material

- `resources/Thesis_Bachelor/` — the bachelor thesis Overleaf export (LaTeX source, figures, bibliography) that `bachelor-thesis-analysis.md` is based on.
