# Formal Requirements Compliance Audit

*Audit of `thesis/manuscript/` against `thesis/docs/formal_requirements_en.md`
(the university's `äußerliche_Form.md`). Performed 2026-09-21 against the working
tree on branch `thesis/overleaf-sync`.*

**Method.** Static inspection of every file under `thesis/manuscript/` (main.tex,
preamble, five chapters, appendix, bibliography), plus scripted cross-checks:
citation-key resolution against `references.bib`, caption coverage per float
environment, proper-name italic consistency, dash and semicolon usage, and prose
word counts per chapter. **No compilation was performed** — no LaTeX toolchain
(`latexmk`, `pdflatex`, `biber`) exists on this machine and no built PDF is
present in the repository. Every requirement that can only be judged from the
rendered output is listed separately in §5 as *unverifiable here*. This matches
the open item already recorded in `2026-09-19_overleaf-sync.md`: the manuscript
has still never been compiled.

---

## 1. Verdict at a glance

| # | Requirement | Status |
|---|---|---|
| 1 | Prescribed cover page, outermost | ✅ Met (PDF spliced in) |
| 2 | Signed independence declaration directly behind it | ⚠️ Structure met, date and signature missing |
| 3 | German + English abstract, each on its own page | ❌ TODO |
| 4 | Well-structured TOC, every entry backed by ≈1 page | ⚠️ Met for Ch. 2–4, fails for Ch. 1 and Ch. 5 |
| 5 | No over-fine subdivision ("4.3.6.1") | ✅ Met (max depth x.y.z, TOC printed to x.y) |
| 6 | Numbered bracket citations, used as a word in the text | ✅ Met |
| 7 | Bibliography at the end | ✅ Met |
| 8 | No uncited take-over of text/tables/images | ⚠️ Mostly met, gaps listed in §3.3 |
| 9 | AI-tool disclosure per the separate information sheet | ❌ Section exists as a TODO, not written |
| 10 | Proper names italicised, first use points to source | ⚠️ 7 first-mention misses, **fixed 2026-09-21** except pending citations |
| 11 | Figures and tables numbered separately, per chapter | ➖ Met; the `\figurename` overrides are a documented divergence (§4.1) |
| 12 | Short summarising caption on every figure and table | ✅ Met (37/37 captioned, 22 with short forms since 2026-09-21) |
| 13 | List of figures / list of tables (discuss with supervisor) | ✅ Present, decision pending |
| 14 | Optional appendix for bulk material | ⚠️ Exists with one real section, rest is a TODO |
| 15 | Roman then Arabic page numbers matching the PDF | ⚠️ Set up correctly, unverifiable without a build |
| 16 | `--` for hyphens, `---` for em dashes | ➖ Deliberately superseded by the project's own rule |
| 17 | Manual hyphenation where LaTeX breaks badly | ➖ Unverifiable without a build |
| 18 | No typos, no overfull lines, no blurry images | ➖ Unverifiable without a build |
| 19 | Introduction with motivation / goals / environment / structure | ❌ All four sections are TODO stubs |
| 20 | State of the art with shortcomings and delimitation | ✅ Met, and unusually well framed |
| 21 | Design and implementation with justified decisions | ✅ Met, the manuscript's strongest part |
| 22 | Results contrasting before and after, with an outlook | ❌ One of four sections written, outlook missing |
| 23 | Written for a non-specialist external reader | ⚠️ Chapters 2–3 yes, unassessable for the unwritten parts |
| 24 | Brevity over page count | ⚠️ Chapter 2 is at risk of over-length |

Legend: ✅ met · ⚠️ partially met or at risk · ❌ not met · ➖ out of scope here.

**Headline.** The formal *scaffolding* is in good shape. Every structural element
the university prescribes exists in `main.tex` in the right order, the citation
mechanism is correct and fully resolved, and the captioning discipline is
complete. What is missing is *content*, and it is missing precisely where the
formal requirements are most specific: the two abstracts, the entire
Introduction including the AI-tool disclosure, the entire Discussion and
Conclusion including the required outlook, and three of the four Results
sections. Roughly 38,400 words of prose exist, essentially all of it in
Chapters 2 and 3.

---

## 2. Structural requirements

### 2.1 Cover page, declaration, abstracts

`main.tex` splices in the prescribed cover as page 1 via
`\includepdf[pages={1}]{preamble/Thesis_Master-SS 2026-Simon Hedrich.pdf}`,
then inputs the declaration, then the abstract pages, then the table of
contents. That is exactly the order the guideline demands.

Two notes on this block.

`preamble/title_page.tex` is a LaTeX-native title page carrying placeholder
fields (`Matrikelnummer`, `Abgabedatum`, `Betreuer: Name`). It is **not inputted
by `main.tex`** and therefore never compiled. It is dead code that contradicts
the comment above the `\includepdf` line, which claims no pre-rendered title
page exists. Either delete the file or correct the comment, otherwise a future
reader (or a future you) will wire the wrong page in.

`preamble/eidesstattliche_erklaerung.tex` has the full declaration text and a
signature rule, but the date line reads `Karlsruhe, % TODO: submission date`.
The declaration must be dated and signed; this is a known submission-time item,
not a defect in the draft.

`preamble/abstract.tex` correctly builds two separate pages, German first, each
in its own `minipage`. Both `abstract_ger.tex` and `abstract_eng.tex` contain
only a TODO comment. **The requirement is not met.** These are explicitly
deferred until the results are final, which is a reasonable sequencing choice,
but they are a hard requirement and need to be on the submission checklist.

### 2.2 Table of contents depth and substance

`\setcounter{tocdepth}{1}` prints chapters and sections only, so the deepest TOC
entry is of the form `4.3`. That matches the guideline's recommendation exactly
and avoids the `4.3.6.1` over-subdivision it warns about. Sub-subsections exist
in the source (max depth `3.5.4`) but do not surface in the TOC, and `\paragraph`
is used as an unnumbered fourth level, which is the right way to do it.

The substance test — "every entry followed by at least roughly one page of text"
— is where the current draft fails, and only because of unwritten chapters:

| Entry | Prose words | Verdict |
|---|---|---|
| 1 Introduction (4 sections) | 16 | ❌ empty |
| 2 Literature Review (4 sections) | 11,426 | ✅ |
| 3 Methods and Implementation (7 sections) | 23,185 | ✅ |
| 4.1 Practicability of Synthetic Training Data | ~2,400 | ✅ |
| 4.2 Detection Performance Across Regimes | 0 | ❌ empty |
| 4.3 Knowledge Distillation and Embedded Feasibility | ~1,300 | ⚠️ one of two subsections written |
| 4.4 Cross-Cutting Observations | 0 | ❌ empty |
| 5 Discussion and Conclusion (4 sections) | 8 | ❌ empty |

No written section is *too thin* for its heading. The problem is binary: a
section is either substantial or it is a stub. That is a much better position
to be in than a TOC full of half-page sections.

### 2.3 Page numbering

`\pagenumbering{Roman}` is set before the `\includepdf`, so the cover is page I
and the front matter runs in Roman numerals; `\cleardoublepage\pagenumbering{arabic}`
resets at Chapter 1. This satisfies the requirement in principle. Whether the
printed folios actually coincide with the PDF viewer's page counter depends on
how `\includepdf` and the `twoside` blank pages interact in the real build, and
**cannot be confirmed without compiling**. Check it once, on the first build.

---

## 3. Citation, sourcing and attribution

### 3.1 Citation style — met

`\usepackage[backend=biber,style=numeric,sorting=none,url=true]{biblatex}`
produces exactly the bracketed numerals the guideline prescribes, in citation
order. Spot-checks of the running text show citations used as words in the
flow, not introduced with "see" or "cf.":

> `Two detector architectures are trained. \textit{YOLOv5s} \cite{ultralytics...}, at ...`

A grep for the discouraged `see \cite` / `cf.` / `according to \cite`
constructions returns nothing in prose. This requirement is cleanly met.

### 3.2 Bibliography integrity

All **99** distinct citation keys used across the chapters resolve against
`bibliography/references.bib`. There are no undefined references. The `.bib`
holds **220** entries, so **121 are uncited** — harmless with `biblatex`
(uncited entries are not printed without `\nocite{*}`), but worth a cleanup pass
before submission so the file does not read as an undigested Zotero dump.

Citation density by chapter:

| Chapter | `\cite` calls |
|---|---|
| 1 Introduction | 0 (unwritten) |
| 2 Literature Review | 105 |
| 3 Methods and Implementation | 55 |
| 4 Results | 0 |
| 5 Discussion and Conclusion | 0 (unwritten) |

Chapter 4 having zero citations is defensible for a chapter reporting original
measurements, but see §3.3 — it is the chapter where uncited third-party tool
names cluster.

### 3.3 Proper names: italics and first-use source pointer

*Revised 2026-09-21. The first version of this section was wrong and is corrected here.*

The guideline asks for proper names to be highlighted, typically in italics, with
a pointer to literature or a website on first use. The house rule implementing
this is stated in `.claude/skills/thesis-writing/references/mechanics.md` §1:
**italic on first mention, plain afterwards**.

The first version of this audit counted every bare occurrence as a defect
(SpeciesNet 12, MegaDetector 10, QCS605 6, and so on). That was a
misreading — under the house rule those later plain occurrences are correct, and
almost all of them are. The real test is the **first** mention in document order
(Ch. 1, Ch. 2, Ch. 3 §§30-37, Ch. 4, Ch. 5, appendix). Checked that way, seven
names were genuinely wrong:

| Name | First mention | Defect | Status |
|---|---|---|---|
| QCS605 | `2-Literature_Review.tex:93` | bare, though all 14 later uses are italic | italicised ✅, citation pending |
| ONNX | `2-Literature_Review.tex:324` | bare, never cited | italicised ✅, citation pending |
| ONNX Runtime | `4-Results.tex:276` | bare, in a caption, never cited | italicised ✅, citation pending |
| TorchScript | `4-Results.tex:257` | bare, never cited | italicised ✅, citation pending |
| PyTorch | `4-Results.tex:262` | italic, never cited | citation pending |
| AX Visio | `2-Literature_Review.tex:11` | italic, never cited anywhere | citation pending |
| Raspberry Pi 400 | `2-Literature_Review.tex:334` | italic, never cited | citation pending |

`TFLite`, `NCNN` and `SNPE` were italicised in the same pass, since they sit in
one parenthetical list with `ONNX` at `2-Literature_Review.tex:324` and leaving
them bare beside an italicised sibling would read as an error.

Two apparent misses are **not** defects. `MegaDetector` and `SpeciesNet` first
appear inside the `\subsection{}` heading at `2-Literature_Review.tex:81`, where
italics would propagate into the table of contents. Their first prose mentions at
`:85` are already italicised and cited. `Hexagon` and `Ultralytics` are also
correct as they stand: `mechanics.md` §1 lists `Hexagon 685` as a proper noun
that stays in running text, and `Ultralytics` sits adjacent to its citation at
first mention.

**The citations remain open.** `references.bib` has no entry for ONNX, ONNX
Runtime, TorchScript, PyTorch, the AX Visio, the QCS605 or the Raspberry Pi 400 —
that is, for the thesis's target device, its proxy and its whole export
toolchain. The user is adding these via Zotero. Once the keys exist, attach each
one to the first mention above, in the house pattern
`\textit{YOLO26n} \cite{UltralyticsYOLO262025}`
(`35-Synthetic_Generator_Comparison.tex:280`).

### 3.4 Plagiarism and figure provenance

*Revised 2026-09-21 after the provenance was actually traced. The first version
of this section speculated; this one does not.*

The declaration commits to citing all borrowed text, tables and images, which
makes figure attribution a formal requirement rather than a nicety. Provenance
for all nine real photographs in the manuscript was reconstructed by image
signature matching against the source pools and cross-checked against
`data/wikimedia/metadata.csv`. The full mapping is recorded in
`2026-09-21_figures-guideline-and-new-manuscript-figures.md` §6b, which is now
the only record of it — the figure directories are untracked and `data/` is
gitignored, so it cannot be re-derived from git history.

**Six of the nine are resolved.** Credit lines naming author and licence were
added to the outer captions of `fig:zebra-species`, `fig:appendix-gallery-lion`,
`fig:appendix-gallery-saiga` and `fig:appendix-gallery-plains-zebra`. All six are
Wikimedia Commons photographs. Five are ShareAlike (CC BY-SA 2.0 or 4.0) and the
manuscript panels are lossy-resized derivatives, so the share-alike term attaches
to the modified images.

**Three remain open, and one of them is blocking.**

1. **The *AX Visio* pair is a rights-clearance problem, not a missing credit.**
   The first version of this audit assumed these were first-party captures. They
   are not.
   `docs/progress_notes/2026-03-11_dataset-stakeholder-meeting-and-model-architecture.md:23`
   records that the ten AX Visio frames were shared by the AX Visio Product Owner
   at Swarovski Optik. They are neither the author's own captures nor public
   material, and no licence or written permission is recorded anywhere in the
   repository. Reproducing them in a submitted and potentially published thesis
   needs explicit written permission. **`fig:ax-visio` should not ship until that
   permission exists.** This belongs on the blocking list, not the mechanical one.
2. **`kinkajou_real.jpg` cannot be attributed.** GBIF filenames originally
   embedded the occurrence ID, but `scripts/rename_dataset_images.py` renamed
   them in place and saved no mapping, and the original download directory is
   gone. The ID must not be guessed. Recommended fix is to swap that one panel
   for a Wikimedia kinkajou of known attribution, since
   `fig:kinkajou-generator-samples` does not depend on which real photograph it
   uses.
3. **`lion_real.jpg` has no nameable photographer.** The Wikimedia `artist` field
   reads "Meine Mutter (Erlaubnis liegt vor)". The caption now credits the
   uploader, L. Beck, without claiming they took the photograph.

Everything generated by the pipeline (synthetic samples, plots, TikZ diagrams)
is self-produced and needs no attribution beyond what the text already gives.

### 3.5 AI-tool disclosure — not met

`1-Introduction.tex` reserves a `\paragraph{Utilization of AI Tools in Thesis
Preparation}` with a TODO noting that the disclosure should be more extensive
than the bachelor thesis's, given heavy reliance on Claude Code across data
engineering and writing. The guideline defers the substance to the separate
*Umgang mit KI-Medien* information sheet. **That sheet is not present in this
repository and has not been consulted.** Obtain it before writing the paragraph
— it may prescribe a specific form, placement or wording that the current
placeholder does not anticipate.

---

## 4. Figures, tables and the appendix

### 4.1 Numbering and captions

`scrreprt` numbers figures and tables in separate sequences, divided by chapter
(`3.4`, `4.2`, …), which is exactly what the guideline asks for, and no
`\counterwithout` or similar override is present.

**Every one of the 37 float environments (14 `figure`, 23 `table`) has a
`\caption`.** The captions are genuinely summarising rather than bare labels —
for example, "Generation cost against downstream accuracy for the six locally-run
generators. The absence of any trend is the result…". This requirement is met
better than most theses manage.

Two issues:

**The `\figurename` overrides are an intentional divergence, not a defect.**
*Revised 2026-09-21.* Two places rename the float label — `4-Results.tex:131`
(`Samples`, reset at `:162`) and `appendices/appendix.tex:12` (`Appendix
Samples`, reset at `:158`) — so captions inside those blocks render as "Samples
4.6" and "Appendix Samples A.1", and the altered names carry into the List of
Figures.

The first version of this audit called for removing them. That was wrong. This is
a documented house convention: `.claude/skills/thesis-writing/references/figures.md:72-78`
prescribes renaming `\figurename` for a themed figure group "so the List of
Figures reads as a distinct section", `mechanics.md` §3 restates it, and the
precedent is the author's own bachelor thesis, which did exactly this for "Plots"
and "Images" and was accepted at this university. The numbering itself is
untouched — only the label word changes. Treat this like the dash-rule divergence
in §5.1: intentional, justified, and worth one sentence to the supervisor.

**Long captions would have flooded the two lists. Fixed 2026-09-21.** Only 2 of
37 floats used the optional short-caption form `\caption[short]{long}`, so the
rest would have pushed their full multi-sentence text into the List of Figures or
List of Tables. Short forms were added to the 20 floats whose caption body
exceeded 85 characters, bringing the total to 22. The 15 remaining bare captions
are already short enough to serve as their own list entries.

### 4.2 Appendix

The appendix is the right destination per the guideline ("extensive
explanations, data sheets, user manuals, source code"). It currently holds one
real section — the 21-image qualitative generator gallery — and a TODO reserving
space for the full 225-class band assignment table and the full look-alike group
listing. Both of those belong there and both are currently absent from the
thesis entirely, including from the main chapters that reference the underlying
groupings. Moving them in would also relieve any length pressure in Chapter 3.

The manuscript contains **no code listings** anywhere, which is fully in line
with the guideline's advice to describe algorithms in terms of their content
rather than "filling pages of paper with lengthy listings". `listings` is loaded
in the preamble but unused.

### 4.3 Lists of figures and tables

`\listoffigures` and `\listoftables` are both printed, after the appendix and
the abbreviations list, before the bibliography. The guideline asks you to
**discuss with your supervisor whether they add real value**. With 14 figures
and 23 tables spread across three content chapters, they plausibly do — but this
is an open decision, not a settled one. Raise it at the next supervision
meeting, and if the lists stay, do the short-caption pass from §4.1 first.

---

## 5. Requirements that cannot be checked without compiling

The guideline's closing paragraph — no spelling mistakes, no text running past
the right-hand margin, no blurry images or illegible labels — is a
rendered-output requirement, and so are the two LaTeX tips. None of it can be
assessed from source. Once the first build exists, check specifically:

1. **Overfull `\hbox` warnings.** The manuscript is heavy on long compound
   identifiers (`gemini-3.1-flash-image-preview`, `CocoYoloDataset`,
   `mAP\_detect`) inside `\texttt{}`, which LaTeX cannot hyphenate. These are
   the classic source of lines running into the margin. `\UrlBreaks` is already
   configured for URLs, but not for `\texttt` spans.
2. **Hyphenation.** `babel` is loaded with `english` only, so English
   hyphenation is active and the guideline's German-hyphenation warning does not
   apply — with one exception, the German abstract and the German declaration,
   which will be hyphenated by English rules. Consider `\selectlanguage{ngerman}`
   around those two blocks.
3. **Figure legibility.** Five of the plots are PNGs lifted from the reports
   pipeline (`embedded_latency_by_runtime.png`, `model_comparison_*.png`). Check
   their axis labels at final print size; regenerate as PDF or SVG if they blur.
   `svg` is already loaded in the preamble.
4. **Page-number correspondence** (§2.3).
5. **Glossary build.** `\makeglossaries` plus `\glsaddall` plus
   `\printglossary` requires a `makeglossaries` pass between LaTeX runs.
   Overleaf handles this automatically; a local `latexmk` may not.

### 5.1 The dash rule — a deliberate, documented divergence

The guideline gives a LaTeX tip: write `Client--System` for a hyphen and `---`
for a long dash. **This project deliberately does not follow it.** The house
style recorded in the `thesis-writing` skill forbids `--` and `;` in thesis
prose entirely, splitting such constructions into separate sentences instead.
The audit confirms the house rule is being followed: there are **zero**
semicolons in prose, and all 23 occurrences of ` -- ` are either TikZ edge
operators (`\draw[arr] (s0) -- (s1);`) or en-dash separators inside
`\paragraph{Stage 2 -- MegaDetector}` headings and table cells, none of them
running prose.

The guideline's tip is also simply wrong on its own terms for the hyphen case —
a single `-` is the correct hyphen in LaTeX and does *not* suppress hyphenation.
No change recommended. It is worth a one-line mention to the supervisor that the
divergence is intentional, in case they check for it.

---

## 6. Content-guideline requirements (the second half of the sheet)

### 6.1 Introduction — not met

The guideline is unusually prescriptive here: four sections, in order —
motivation (a problem the reader has "suffered" from), goals *and achievements*,
the environment including the company and department, and the structure with
two or three sentences per chapter. Chapter 1 has **exactly this skeleton and
no content**: `Motivation`, `Objective`, `Environment` (with the AI-disclosure
paragraph and a `Pre-Work and Preliminary Resources` subsection), and
`Structure`, all TODO.

Three things to get right when it is written:

- **State the achievements, not just the goals.** The guideline asks what new
  insight the reader gains. This thesis has at least three concrete ones ready
  to claim: that generation cost does not predict downstream utility, that
  prompt length matters less than model identity, and the single-seed-difference
  cautionary result.
- **Introduce inovex GmbH and the Data Management & Analytics department** for a
  reader who knows neither. One short paragraph.
- **Scope honesty belongs here too.** Chapters 2 and 4 already state plainly
  that knowledge distillation and the embedded check were reduced to a single
  basic comparison. The Objective section must state the research questions in
  the form they were actually answered, not the form originally planned, or the
  Discussion will be answering questions the Introduction never asked.

One structural defect: `\subsection{Pre-Work and Preliminary Resources}` is a
subsection placed at the end of `\section{Environment}` but *after* the
`\paragraph{Utilization of AI Tools}`. Once written, check that it nests where
intended — as it stands it will read as a trailing subsection of Environment,
which may or may not be the plan.

### 6.2 State of the art — met, and well done

Chapter 2 opens with a lead-in that does something the guideline does not even
ask for and that materially helps the reader: it states up front that coverage
is *weighted to mirror where the effort actually went* (~80% on dataset
construction and synthetic data), and says which topics are covered at
foundational depth only and why. It names the shortcomings of existing
approaches (the MegaDetector + SpeciesNet two-stage ensemble's cost against the
QCS605's DSP budget), delimits the work, and surfaces the fixed constraints the
guideline explicitly asks about — the YOLOv5 licence boundary, the hardware
budget, the 225-class count. The chapter-opening summary the guideline asks for
("later on, too, you should begin each chapter with a short summary") is present
here and in Chapter 3. It is missing from Chapters 1, 4 and 5, all of which are
stubs at that point.

### 6.3 Design and implementation — met

The guideline splits these into two chapters; this thesis merges them into
Chapter 3, which is a legitimate deviation ("you may of course also deviate").
The chapter does what both chapters are asked to do. Decisions are justified
rather than asserted, alternatives are contrasted with their trade-offs
(`Candidate Data Sources and Licensing Assessment`, `Generator Roster` with
excluded candidates *and their reasons*, the three prompt regimes), and
unforeseen difficulties are reported rather than hidden — the LILA BC rejection
gets its own subsection, and the SpeciesNet coverage gaps get another. That is
precisely the "where unforeseen difficulties arose, so that the reader does not
later run into the same problems" the guideline asks for.

The one item the guideline names that is **not** present: an explicit overview
of the languages and tools used. Tools appear scattered through the text
(Python, PyTorch, Ultralytics, MLflow, ONNX Runtime, uv, Docker) but never
collected. A short subsection or table in Chapter 3, or a paragraph in the
Introduction's Environment section, would close this and would double as the
place to give each tool its first-use italics and citation (§3.3).

### 6.4 Results — partially met

Section 4.1 is written and is strong: it reports the twelve-cell generator
comparison, states its qualifications up front rather than burying them, and
includes the "A Note on Seeds" paragraph, which is exactly the kind of honest,
transferable finding the guideline's "unforeseen difficulties" advice is
fishing for. Section 4.3's runtime benchmark subsection is written.

Missing: §4.2 (the four-subsection regime comparison, the thesis's other
headline result), §4.3.1 (the distillation vs. direct fine-tuning comparison —
**the thesis's stated core research question**), and §4.4 (cross-cutting
observations). The chapter also lacks its opening summary paragraph.

The guideline's "contrast the previous state and the now better state in detail"
maps here to contrasting the distilled student against the directly fine-tuned
one, and the synthetic-supplemented bands against the real-only baseline. Both
comparisons are planned in the TODOs and neither is written yet.

### 6.5 Outlook and conclusion — not met

Chapter 5's four sections (`Discussion`, `Limitations`, `Further Work`,
`Conclusion`) are all empty. The guideline asks specifically for a closing
outlook naming the steps you would have taken with more time, and encourages
initiative — "could the entire task perhaps have been accomplished quite
differently or more easily by another route?".

This thesis is unusually well placed to write that section, because the
reductions in scope are already documented in the source: the abandoned
quantization-aware training pass, the multi-strategy distillation survey that
became one comparison, the qualitative rubric that was designed and never
executed, the missing API quality-ceiling cell, the uneven seed counts. Each of
these is a ready-made "further work" item with a stated reason. The `Discussion`
TODO already commits to restating the numbered research questions from
`sec:objective` verbatim and answering them one by one, which is the right
structure — it just depends on §6.1 defining those questions first.

### 6.6 Audience and length

The guideline's target reader is a competent computer scientist who is *not* a
specialist in the subject area and does *not* know the company. Chapters 2 and 3
hold to this well: terminology is introduced before use, the acronym mechanism
is single-sourced through `glossaries`, and the reasoning is explained rather
than assumed. The unwritten Environment section is the one place where the
"does not know your company" instruction is currently unmet by construction.

On length, the guideline is blunt: "Would you feel like reading 150 pages if 70
would have done the job?" Current prose stands at roughly **38,400 words**
across two-and-a-bit chapters, which at this layout (12 pt, one-and-a-half
spacing, the configured margins) is on the order of 115–125 pages of body text
before floats — and the Introduction, three quarters of the Results and the
entire Discussion are still to come. **Chapter 2 alone, at 11,426 words, is
roughly 35 pages of literature review.** Its lead-in defends the weighting
convincingly, but a state-of-the-art chapter approaching a third of the body is
worth a compression pass, particularly §2.2 on synthetic data. Trim toward what
the thesis's own decisions actually rest on and push the rest into citations.

The guideline also warns against repeatedly noting that this is a final thesis
or an examination requirement, preferring "the present work". No violations were
found — the existing prose consistently says "this thesis", which is fine.

---

## 7. Prioritised action list

**Blocking for submission**

1. Write both abstracts (`preamble/abstract_ger.tex`, `preamble/abstract_eng.tex`).
2. Write Chapter 1 in full, including the numbered research questions in the
   form they were actually answered.
3. Obtain the *Umgang mit KI-Medien* information sheet and write the AI-tool
   disclosure to its requirements.
4. Write Chapter 5 in full, including the outlook the guideline requires.
5. Write §4.2, §4.3.1 and §4.4, and the Chapter 4 opening summary.
6. Date and sign the declaration, and fill the cover page's examiner and date
   fields.
7. **Obtain written permission from Swarovski Optik for the two *AX Visio*
   photographs**, or drop `fig:ax-visio`. They were supplied privately by the
   AX Visio Product Owner and carry no licence (§3.4).
8. **Compile the thesis** and work through §5's checklist. This has never been
   done and is the single largest unknown in this audit.

**Formal defects, mechanical — done 2026-09-21**

9. ~~Remove the `\renewcommand{\figurename}` overrides.~~ **Withdrawn.** A
   documented house convention with precedent, not a defect (§4.1).
10. ✅ Proper-name first mentions italicised (§3.3). **Citations still pending**
    — `references.bib` has no entry for ONNX, ONNX Runtime, TorchScript,
    PyTorch, the AX Visio, the QCS605 or the Raspberry Pi 400. The user is
    adding these via Zotero; attach the keys to the first mentions afterwards.
11. ✅ Credit and licence lines added for the six attributable photographs
    (§3.4). Two remain open: the AX Visio permission (item 7) and the
    unattributable kinkajou panel.
12. ✅ Short captions added to the 20 floats that needed them (§4.1).
13. ✅ `preamble/title_page.tex` deleted and the stale `main.tex` comment
    corrected (§2.1).

**Recommended**

14. Add a languages-and-tools overview to Chapter 3 (§6.3).
15. Move the 225-class band table and the look-alike group listing into the
    appendix (§4.2).
16. Compression pass on Chapter 2, especially §2.2 (§6.6).
17. Prune the 121 uncited entries from `references.bib` (§3.2).
18. Wrap the German abstract and declaration in `\selectlanguage{ngerman}` (§5).
19. Ask the supervisor whether the lists of figures and tables add value (§4.3),
    and mention the two intentional divergences: the `--` dash tip (§5.1) and
    the `\figurename` renaming (§4.1).
20. Replace the kinkajou real-photograph panel with a Wikimedia image of known
    attribution (§3.4).
