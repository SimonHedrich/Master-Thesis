# Mechanics — the house conventions, as a checklist

The conventions in §7 of `thesis/bachelor-thesis-analysis.md` are correct and already followed.
They are restated here as checkable items so the skill has one place to point at, and so the
two places where the manuscript has actually drifted are visible.

No LaTeX toolchain exists in this container — `pdflatex`, `latexmk` and `biber` are all absent,
and the document is compiled externally. Every check below is therefore done by reading the
source, which `thesis/README.md` already establishes as this project's norm.

---

## 1. Formatting

| rule | check | state |
|---|---|---|
| Numbers in math mode | `$225$`, `$28\%$`, `$1{,}200$`, `$512 \times 512$`, `$1 \times 10^{-4}$` — never plain-text `10%` or `512x512` | 2 stray percentages (`2-Literature_Review.tex:5`, `:158`), 3 stray cardinals |
| Proper nouns stay in text | `Hexagon 685`, `SD 3.5`, `COCO 2017`, `Raspberry Pi 5` are names, not quantities | correct |
| Tool, dataset and model names | `\textit{}` on first mention, plain afterwards | correct, ×232 |
| Identifiers and slugs | `\texttt{}` — file names, model slugs, regime names, config keys | correct, ×90 |
| Quotation | `\enquote{}` (csquotes), never raw `"` | 4 raw pairs, all in Chapter 2 (`:5`, `:81`, `:86`, `:88`) |
| Cross-references | `\Cref{}` (cleveref), including multi-target `\Cref{sec:a,sec:b}` | correct, ×173; the 6 `\ref` are inside stub comments |
| First definition of a term | `\textbf{}` plus a definition, once | **drifted** — 4 uses in 30,004 words |

## 2. Acronyms — the live defect

`preamble/acronyms.tex` defines **25 acronyms** with `\newacronym`, and the chapters call
`\gls`, `\glspl`, `\acrshort` and `\acrlong` **zero times**. The acronym list renders empty
while the terms are spelled out by hand: "Knowledge Distillation" ×8, "mean Average Precision"
×5, "Quantization-Aware Training" ×4, "Vision-Language Model" ×1. `KD` never appears.

§8 of the style contract records that the bachelor thesis shipped two competing abbreviation
mechanisms. This is the same failure in different clothes: one mechanism declared, another one
practiced.

**The rule:** an acronym defined in `acronyms.tex` is introduced with `\gls{key}` on first use
and appears in short form afterwards. No acronym defined there is ever spelled out by hand. A
term that does not deserve an entry gets removed from `acronyms.tex` rather than left to render
an empty list entry.

Defined keys include: `gbif`, `md`, `sn`, `gt`, `vlm`, `llm`, `coco`, `iou`, `tide`, `api`,
`clip`, `fid`, `kid`, `vram`, `nf4`, `kd`, `qat`, `map`, `ap`, `ar`, `nms`, `dsp`, `gpu`,
plus the license acronyms.

## 3. Floats

- Every `figure` and `table` is referenced from the prose at least once with `\Cref`, and the
  referencing sentence says what the reader should see in it. See `claims.md` §7.
- Currently unreferenced: `fig:generator-headline-map`, `fig:prompt-length-ablation`,
  `fig:generator-cost-vs-map`, `fig:generator-per-class-heatmap` (all `4-Results.tex`) and
  `eq:quality-score` (`32-Data_Quality_and_Curation.tex:147`).
- Never refer to a float by position — "the table above" (`4-Results.tex:153`). Floats move.
- Multi-panel comparisons use `subcaption`; `\figurename` is renamed per figure group so the
  List of Figures reads as distinct categories. See `figures.md` for the layout pattern and for
  whether a comparison should be a figure, a table, or a sentence in the first place.

## 4. Citations

- biblatex, numeric, `sorting=none`. `\cite` is the default form; `\textcite` where the author
  is the grammatical subject.
- Keys resolve against `bibliography/references.bib` (project entries: datasets, tools, model
  releases, short hand-made keys) or `bibliography/references_zotero.bib` (Better BibTeX keys).
- Currently clean: 100 distinct keys, 158 calls, zero unresolved, zero dangling `\Cref`.
- Refresh the Zotero copy with
  `cp research/literature/references.bib thesis/manuscript/bibliography/references_zotero.bib`.

## 5. Structure

- Chapter file shape: `\newpage` → `\cleardoublepage` → `\chapter{}\label{chapter:...}` → an
  untitled lead-in paragraph previewing the chapter's sections → `\section`/`\subsection`, each
  with `\label{sec:snake_case}`, separated by the 50-character `%%%%` rules.
- Chapter 3 is split across `3-Methods_and_Implementation/`; `30-Overview.tex` holds the chapter
  header and `\input`s the numbered sections. `main.tex` includes only `30-Overview`.
- Every design choice is justified inline with at least one "chosen because…" sentence, usually
  with a citation. See the house move in `construction.md` Part 4.
- Cross-linking is heavy and deliberate, including forward references from the Introduction into
  later chapters, so the reader is never left wondering whether something will be explained.

## 6. The research-question echo

The strongest cross-file constraint in the thesis, from §5 of the style contract:

**The numbered research questions defined in Chapter 1 are reproduced verbatim in Chapter 5 and
answered one at a time.** Copy the text, do not paraphrase it. Both files are currently stubs
whose `% TODO` comments already point at each other; when Chapter 1 is written, Chapter 5's
skeleton gets the same text pasted into it in the same commit.

Chapter 1's skeleton, also from §5: lead-in → Motivation → Objective (numbered RQs) →
Environment, including an **AI-tool-usage disclosure** → Pre-Work → Structure. The disclosure
needs to be substantially more extensive than the bachelor thesis's, given how much of this
project ran through Claude Code.

## 7. Before submission

From §8 of the style contract, the rough edges that survived into the bachelor thesis:

- [ ] `grep -rn "TODO\|FIXME" thesis/manuscript/` returns nothing. All 32 current hits are
      intentional stub skeleton and must be gone or resolved.
- [ ] No superseded chapter file left in the compiled tree. One canonical file per chapter;
      drafts get deleted or moved outside `manuscript/`.
- [ ] One abbreviation mechanism only — `glossaries`, per §2 above.
- [ ] No unfilled figure slot. Writing prose first and back-filling figures is a fine workflow
      as long as none is missed; `% TODO: add images` comments are how they get missed.
- [ ] Every float referenced (§3).
- [ ] British spellings gone (`construction.md` §5.2): 63 `-ise/-isation`, 16 `behaviour`,
      10 `artefact`, 6 `judgement`, 3 `colour`, 2 `penalise`, 1 `favourable`, 1 `labelled`.
- [ ] Chapter 5 restates the Chapter 1 research questions verbatim (§6).
