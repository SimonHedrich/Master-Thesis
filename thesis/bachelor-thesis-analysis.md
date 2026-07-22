# Bachelor Thesis Analysis — Structure & Style Reference

This document analyzes the author's bachelor thesis (Overleaf export at `resources/Thesis_Bachelor/`) in detail, as a reference for writing the Master's thesis in this repository. The bachelor thesis covers a different topic — augmenting training data with Stable-Diffusion-generated synthetic images to improve small-object (car) detection with Faster R-CNN — but it establishes the author's/institution's conventions for structure, LaTeX tooling, and academic writing style that the Master's thesis can largely reuse.

Source: `resources/Thesis_Bachelor/thesis.tex` and everything it includes.

## 1. Overview & Metadata

- **Institution**: Hochschule Karlsruhe, Fakultät für Informatik und Wirtschaftsinformatik
- **Degree**: Bachelor of Science, Studiengang Informatik
- **Company partner**: inovex GmbH, Data Management & Analytics department (same as the Master's thesis)
- **Document class**: `scrreport` (KOMA-Script), 12pt, A4, two-sided with binding correction
- **Bibliography engine**: biblatex + biber, numeric style, `sorting=none` (references appear in citation order, not alphabetical)
- **Length**: 5 chapters + appendix, ~2,000 lines of `.tex` chapter content, dozens of figures/plots/tables
- **Submission date**: January 8, 2025 (per the Eidesstattliche Erklärung)

## 2. File & Directory Organization

```
thesis.tex                         — top-level document; assembles everything via \include/\input
chapters/
  1-Introduction.tex
  2-Literature_and_Developments-new.tex   — the version actually \include'd
  2-Literature_and_Developments.tex       — orphaned earlier draft, NOT included (see §8)
  3-Methods_and_Implementation.tex
  4-Results.tex
  5-Conclusion.tex
preamble/
  title_page.tex                   — template; actual title page is a pre-rendered PDF, see §4
  eidesstattliche_erklaerung.tex   — signed declaration of originality
  abstract.tex                     — wraps abstract_ger.tex + abstract_eng.tex in minipages
  abstract_ger.tex / abstract_eng.tex
  acronyms.tex                     — \newacronym entries (glossaries package)
  Thesis_Bachelor-WS 2024-Simon Hedrich.pdf  — the rendered title page, \includepdf'd
  titelblatt.pdf
appendices/
  appendix.tex                     — supplementary figures/tables
  abbrevation.tex                  — abbreviations table (longtable)
bibliography/
  references.bib                   — ~2,550 lines, one shared .bib file
figures/
  images/                          — photos, architecture diagrams, prediction overlays
  images/appendix/, images/detections-letters/, images/predictions/
  plots/                           — data plots, each often saved as both .svg/.eps (vector, for LaTeX) and .png (preview)
```

**Naming convention**: chapter files are prefixed with their chapter number (`1-Introduction.tex`, `2-...`, etc.), which keeps `\include` order self-documenting in a file listing. Worth replicating for the Master's thesis repo.

**Figure convention**: source vector plots are kept in both `.eps`/`.svg` (compiled into the PDF) and `.png` (for quick preview outside LaTeX) — useful given this repo already generates plenty of matplotlib output.

## 3. LaTeX Preamble — Key Packages & Settings

| Package | Purpose |
|---|---|
| `geometry` | Custom margins (`outer=2cm, inner=2.5cm, top=3cm, bottom=3cm`) plus `BCOR=10mm` binding correction for double-sided printing |
| `babel[english]` | Document language (despite German abstract/declaration coexisting) |
| `setspace` (`\onehalfspacing`) | 1.5 line spacing throughout |
| `glossaries[acronym]` | Acronym management via `\newacronym`, printed as a glossary/acronym list |
| `algorithm2e` | Pseudocode (declared but apparently unused in the final text — `\listofalgorithms` is commented out) |
| `tikz` | Diagrams (declared, used sparingly) |
| `pdfpages` (`\includepdf`) | Embeds the externally-rendered title page PDF directly into the document |
| `graphicx`, `svg`, `float`, `subcaption`, `caption` | Figures, including multi-panel `subfigure` layouts |
| `booktabs`, `multirow`, `tabularx`, `adjustbox` | Tables, incl. results tables with merged cells |
| `listings`, `minted`, `xcolor` | Code listings with custom syntax-highlighting color scheme (though little/no code appears in the final chapters) |
| `biblatex` (`backend=biber, style=numeric, sorting=none, url=true`) | Bibliography |
| `hyperref` (`hidelinks, colorlinks=true, linkcolor=black, urlcolor=black, citecolor=blue`) | Clickable but visually unobtrusive cross-references; only citations are colored (blue) |
| `titletoc` | ToC formatting; custom `\contentsname` redefinition to vertically re-center the "Contents" heading |

**Custom commands worth reusing**:
- `\legendbox{color}` — draws a small colored square, used to build inline color-coded legends for prediction/bounding-box images (e.g., "cyan = all setups predicted this box").
- `\SetAlCapSty{}` / `\SetNlSty{footnotesize}{}{}` — algorithm caption/line-number style tweaks (only relevant if pseudocode is used).

**Structural options on `\documentclass`**: `toc = listofnumbered, toc = bibnumbered` (adds List of Figures/Tables and the bibliography to the ToC), `numbers = noendperiod` (chapter numbers like "2.3" not "2.3."), `open = right` (chapters always start on a right-hand page, consistent with double-sided printing).

## 4. Front Matter Pattern

- **Title page**: *not* compiled from `title_page.tex` in the final build — `thesis.tex` instead does `\includepdf[pages={1}]{preamble/Thesis_Bachelor-WS 2024-Simon Hedrich.pdf}`, i.e. a separately designed/exported PDF page is spliced in. `title_page.tex` remains as a LaTeX-native template/fallback with placeholder fields: university, faculty, degree program, thesis title, degree type ("Bachelorarbeit" — change to "Masterarbeit"), author + Matrikelnummer, submission date, and a three-row table for Betreuer/Erstgutachter/Zweitgutachter.
- **Eidesstattliche Erklärung** (declaration of originality): fixed German legal boilerplate + place/date + signature line. Needed verbatim (with updated date) for the Master's thesis.
- **Abstract**: dual-language, laid out as two `minipage`s in sequence (German "Zusammenfassung" first, then English "Abstract"), each `\input`-ing its own one-file-per-language content (`abstract_ger.tex`, `abstract_eng.tex`) so translations can be edited independently. Each abstract is 3 short paragraphs: problem framing → method summary → results/contribution summary.
- **Roman-numeral pagination** (`\pagenumbering{Roman}`) for everything before Chapter 1, switching to arabic at the first `\include{chapters/...}`.
- **AI-tool-usage disclosure**: a dedicated paragraph in the Introduction's "Environment" section (see §5) — not part of the front matter proper, but functions like a methodological disclosure and should be treated as a fixed, required component.

## 5. Chapter-by-Chapter Content Pattern

This is the reusable skeleton — the actual section names may need to change for the Master's topic, but the *shape* of each chapter is a strong template.

### Chapter 1 — Introduction
1. **Untitled lead-in paragraph** — one paragraph stating the thesis's core idea before any subsection headers.
2. **Motivation** — why the problem matters generally, then narrows to the specific gap, cites 3–5 papers per claim, ends by naming the concrete use case studied and why it was chosen as a good test case.
3. **Objective** (`\label{sec: objective}`) — one framing paragraph, then a **numbered list of explicit research questions**. Ends with a forward-reference: *"These questions are being answered in the [results/discussion] chapter."* This RQ list is the backbone the whole thesis is organized around — Results and Discussion both refer back to it explicitly.
4. **Environment** — describes the company partner (inovex), the department, available resources (cloud GPU access), and any constraints (or explicit lack thereof). Includes a **dedicated "Utilization of AI Tools in Thesis Preparation" paragraph**, explicitly naming each tool (ChatGPT, DeepL Translator/Writer) and drawing the line between "AI polished my wording" vs. "AI helped me analyze/develop code" vs. "the ideas and interpretation are mine," with a reference to the university's AI-use guidelines and their date. **This is directly relevant**: since Claude Code is being used throughout this Master's thesis project, an equivalent (updated, more extensive) disclosure paragraph will be needed.
5. **Pre-Work and Preliminary Resources** — a `\subsection` documenting work done *before* the official thesis start (early experiments, dataset scouting), broken into `\paragraph`s per topic.
6. **Structure** — final section, one paragraph per chapter, each with a `\hyperref` back-link to that chapter's label, explaining what the reader will find there.

### Chapter 2 — Literature Review and Current Developments
- Opens with a short paragraph mapping out the chapter's own sub-sections (meta-summary).
- **Object Detection** section: a historical narrative from foundational work (neocognitron, CNN/backprop) through named model families as `\subsection`s (R-CNN → Fast R-CNN → Faster R-CNN as `\subsubsection`s under one "R-CNN" arc, then a separate "Alternative Methods" subsection for SSD and YOLO), each subsection structured as: one-paragraph mechanism explanation → architecture figure with `\cite`d source → 1–2 paragraphs on strengths/limitations relative to the thesis's specific concern (small objects). Ends with a short "Challenges in Small Object Detection" subsection that ties the survey back to the thesis's angle.
- **Evaluation Metrics** section: standalone, covering general metrics (Precision/Recall/IoU/AP/AR) then the specific benchmark framework used (COCO), then an "Alternative Evaluation Metrics" subsection (NWD, DotD) with its own `\label` (`sec:alternative evaluation`) that gets cross-referenced later in Methods.
- **Synthetic Data Generation** section: covers the two competing technique families (GANs vs. diffusion models) at a conceptual level, then a `\paragraph` explicitly disambiguating two easily-confused terms (inpainting vs. outpainting) the thesis will use throughout — a good pattern for pre-empting terminology confusion.
- **Note on drafts**: two versions of this chapter exist in the source tree (`2-Literature_and_Developments.tex`, older/unused, and `-new.tex`, the one actually `\include`d). The unused draft is more citation-dense but written in a passive, name-dropping style ("(Author et al., YYYY)" instead of proper `\cite`), ends mid-sentence, and was clearly superseded. **Lesson**: keep exactly one canonical file per chapter in the include list, and delete or clearly archive superseded drafts before submission — don't leave stale `.tex` files sitting in `chapters/`.

### Chapter 3 — Methods and Implementation
1. Lead paragraph previews the chapter's own structure (Data Gathering → synthetic generation → model → training → evaluation).
2. **Data Gathering** → `\subsection{Dataset overview}` with one `\paragraph` per dataset, each justifying *why* that dataset was chosen and what its limitations were, followed by a `\paragraph{Analysis of the datasets}` that quantifies a key differentiator (relative object size) with a KDE plot comparison figure.
3. **Method section(s)** specific to the thesis's technique (here: Synthetic Dataset Generation, Stable Diffusion inpainting, SAM segmentation) — each described at the level of "what tool, why this tool over alternatives, what parameters, what problems came up and how they were fixed," with inline figures showing before/after or intermediate artifacts.
4. **Dataset Reference** (`\label{section:dataset_reference}`) — a small but important subsection that defines a **short-code naming scheme** for experiment variants (e.g. `bbox-gen`, `segm-noise`, prefixed with augmentation percentage like `50%-bbox-noise`), plus the term "setup" = dataset + trained model together. This naming scheme is then used consistently through Results — worth adopting whenever the Master's thesis has a combinatorial set of experiment configurations (e.g. model × quantization level × KD teacher).
5. **Model Selection** — states the chosen architecture, justifies it against alternatives with citations, describes the specific adaptation made (replacing the classification head, output-dimension math spelled out).
6. **Model Training** — overview paragraph, then named `\paragraph`s per hyperparameter/tool: optimizer (SGD, with all hyperparameter values justified), LR scheduler (StepLR, justified by simplicity given fixed epoch budget), and a **"Discrepancy of Training Results"** paragraph disclosing run-to-run variance and the mitigation (5 repeated runs, metrics averaged) — an important methodological transparency pattern.
7. **MLflow** subsection — documents the experiment-tracking setup (same tool this repo already uses) and exactly what was logged (hyperparameters, system metrics, training/validation loss).
8. **Evaluation Framework** — defines IoU, confidence threshold (with the specific value justified by a stated trade-off), NMS (same pattern), and a "Visual Representation of Predictions" paragraph explaining exactly how qualitative figures were constructed (merging 5 models' predictions per setup) — a transparency-about-visualization-methodology pattern.
9. Ends with a **COCO Evaluator** subsection spelling out every metric variant and, critically, a **"COCO Metric Selection"** paragraph (`\label{sec: metric selection}`) that explicitly states *which* 2–3 metrics were chosen as the headline numbers and why — directly analogous to this repo's own `docs/plans/2026-06-10_model-evaluation-strategy.md` metric-selection reasoning.

### Chapter 4 — Results
- One `\subsection` per experiment-family/"setup" (here: `no-aug` baseline, then `bbox-gen`, `bbox-noise`, `segm-gen`, `segm-noise`), each following an identical internal template:
  1. One-paragraph summary of the trend (peak augmentation %, direction of effect).
  2. A results **table**: rows = metrics, columns = augmentation levels, each non-baseline cell showing both the absolute score and the percentage improvement over baseline, with the *best* value in each row bolded (`\bm{}`).
  3. A **scatter/line plot** figure visualizing the same table across augmentation levels.
  4. A **qualitative prediction image** with a hand-built color legend (via `\legendbox`) showing which augmentation levels newly detected which objects, with a short walkthrough paragraph.
- After all setup subsections: a **"Comparison of the Augmentation Methods"** subsection with a combined bar/histogram figure and a synthesizing discussion contrasting the families.
- A final **"General Observations"** subsection collects cross-cutting findings that don't belong to any single setup: a metric-relationship note, a **training time** analysis (with its own plot), and a deeper hypothesis-driven discussion (here: latent-space clustering of visually similar classes) illustrated with one concrete example image.
- Consistently reuses **one held-out example image** across most subsections for qualitative continuity, explicitly justified ("this image offers multiple cars in different sizes and situations... represents the performance of the models").
- The same reused example image trick, applied per-metric bolding, and the paired (absolute value + % improvement) table format are strong, directly reusable patterns for any thesis comparing multiple training configurations against a baseline.

### Chapter 5 — Discussion and Conclusion
1. **Discussion** — re-states the numbered research questions from the Introduction (copy-pasted verbatim) and answers them **one by one**, each under its own `\paragraph{Question N — <short title>}`, synthesizing evidence from Results rather than introducing new data. Ends with a short summary paragraph.
2. **Limitations** — short, honest, specific (e.g., fixed/unoptimized training hyperparameters as a known confound; a hard technical constraint like fixed generative-model output resolution).
3. **Further Work** — several `\paragraph`s, each proposing one concrete follow-on direction motivated directly by a limitation or an interesting-but-out-of-scope Results observation (e.g., a follow-on for the latent-space clustering finding, a follow-on for evaluation-scenario diversity, a follow-on for extending the technique to new domains).
4. **Conclusion** — final section, ~5 paragraphs, restating problem → method → key quantitative result → limitations → closing statement of contribution. No new content; pure summary.

**Cross-referencing discipline**: the Discussion chapter's `\paragraph` questions are the *exact same numbered list* as the Introduction's Objective section — copy-paste identical wording. This 1:1 mapping is what makes the thesis feel tightly argued; the Master's thesis should establish its RQ list early and preserve exact wording everywhere it's echoed.

## 6. Appendix Conventions

- `appendices/appendix.tex`: supplementary material that would clutter the main chapters — here, a large gallery of example generated images (grouped by source dataset, 3-panel original/bbox/segm comparison per row) and per-image "which setup detected which lettered bounding box" comparison tables. Referenced from the main text via `\hyperref[appendix]{appendix}` rather than duplicated inline.
- `appendices/abbrevation.tex`: a flat two-column `longtable` (abbreviation → full term), separate from the `glossaries`-based acronym list in the preamble — i.e., two parallel abbreviation mechanisms exist in this thesis (a minor inconsistency; picking just one for the Master's thesis is cleaner).
- `\listoffigures` and `\listoftables` are auto-generated after the appendix; `\listofalgorithms` is present but commented out (no algorithms were ultimately used).

## 7. Writing Style & Conventions

- **Voice**: strictly impersonal/passive academic register — "This thesis explores...", "The research will investigate...", never "I". Even methodological choices are phrased as "X was chosen because Y" rather than "I chose X because Y".
- **Terminology discipline**: technical terms that could be ambiguous (inpainting vs. outpainting, "setup" as dataset+model pair) are explicitly defined once, in `\textbf{}`, the first time they're used, then used consistently afterward.
- **Numbers**: always in math mode for consistency — `$10\%$`, `$512 \times 512$`, `$1 \times 10^{-4}$` — never plain-text "10%" or "512x512".
- **Tool/dataset/model names**: consistently `\textit{}` on first mention (e.g., `\textit{Stable Diffusion}`, `\textit{inovex GmbH}`), plain thereafter.
- **Every design choice is justified inline**: dataset selection, model architecture, hyperparameter values, evaluation thresholds — each gets at least one sentence of "chosen because..." rationale, often with a citation backing the choice as standard practice.
- **Heavy internal cross-linking**: `\label`/`\ref` and `\hyperref[label]{descriptive text}` are used pervasively, including forward references from the Introduction into later chapters, so the reader is never left wondering "will this be explained later?"
- **Verbatim blocks for shorthand**: metric abbreviations (`AP-all`, `AP-small`, `AR-small`) are introduced in a `\begin{verbatim}` block mapping the shorthand to the full COCO metric name, then used as shorthand for the rest of the chapter — cleaner than repeating the full metric name every time.
- **Figures**: multi-panel comparisons via `subcaption`/`subfigure`, with a renamed `\figurename` per figure group (e.g. "Images", "Plots", "Figures") so the List of Figures reads as distinct categories rather than one undifferentiated "Figure" sequence.

## 8. Observed Rough Edges (avoid repeating)

1. **Orphaned draft file**: `chapters/2-Literature_and_Developments.tex` is a superseded draft that was never removed from the repo, even though only the `-new` version is `\include`d. It uses a weaker citation style (parenthetical author-year instead of `\cite`) and ends mid-sentence. For the Master's thesis, delete or clearly relocate (e.g. to a `drafts/` folder outside the compiled tree) any superseded chapter version.
2. **Unresolved `% TODO` comment** survives into the submitted text (`% TODO Note: metrics of 'large' objects correlate with AP over all sizes` in Chapter 3) — a reminder to grep for `TODO`/`FIXME` before final submission.
3. **Two parallel abbreviation mechanisms** (glossaries-based acronym list + a separate manual `longtable` in the appendix) — pick one for the Master's thesis.
4. A few other in-text `% TODO: add images` comments in Chapter 3 mark sections where images were planned but not yet inserted at some point — all were eventually resolved in the final chapter, but it's a sign of writing sections textually first and back-filling figures, which is a reasonable workflow to replicate deliberately (write the argument, mark figure slots, fill them in later) as long as none are missed at the end.

## 9. Adaptation Notes for the Master's Thesis

**Carry over largely as-is:**
- Document class, package set, and preamble structure (§3) — this is a proven, working LaTeX setup.
- Front-matter mechanics: PDF-spliced title page, Eidesstattliche Erklärung wording (update the date/degree), dual-language abstract via `minipage` + separate per-language files.
- The Introduction skeleton: Motivation → Objective (numbered RQs) → Environment (incl. an AI-tool-usage disclosure, which will need to be more extensive here given the heavier reliance on Claude Code throughout the project) → Structure.
- The RQ-driven backbone: define numbered research questions once in the Introduction, echo them verbatim in the Discussion, answer one-by-one.
- The "setup" / short-code naming convention for experiment variants, and the paired (absolute + % improvement) results-table format with bolded best values — directly applicable to comparing model architectures × quantization levels × KD (teacher/student) configurations.
- The "COCO Metric Selection" transparency pattern (state and justify the 2–3 headline metrics) — this repo already has an equivalent decision in `docs/plans/2026-06-10_model-evaluation-strategy.md`; that document's reasoning should be summarized into the Methods chapter the same way the bachelor thesis did.
- Reusing one or two held-out qualitative example images consistently across the Results chapter for continuity.

**Will need to change:**
- Title page fields: "Masterarbeit" / "Master of Science" instead of Bachelor, updated examiners.
- Chapter scope and count: the Master's thesis has a larger technical surface (dataset curation at scale, teacher fine-tuning, knowledge distillation, quantization-aware training, embedded-hardware benchmarking) that likely won't fit the bachelor thesis's single "Methods and Implementation" chapter — consider splitting into separate chapters per pipeline stage (e.g., Dataset, Distillation, Quantization & Deployment) while keeping the same per-chapter internal shape (lead-in → justified design choices → named sub-paragraphs per decision).
- The Literature Review will need broader coverage (object detection *and* KD *and* quantization *and* embedded inference), so the "historical narrative → alternatives → challenges" shape from §5 should probably be repeated as a mini-pattern per sub-topic rather than once for the whole chapter.
- The Evaluation section must reflect this repo's mixed real+synthetic evaluation strategy (real-only breakout as primary figure, mixed as headline, synthetic-vs-real delta as a watchdog per `CLAUDE.md`) rather than the bachelor thesis's single COCO-metric-on-one-dataset approach — this is a materially different evaluation design and deserves its own clearly-justified subsection, following the bachelor thesis's habit of explicitly stating *why* each metric/axis was chosen.
- Fix the two rough edges from §8 proactively: keep only one canonical chapter file per chapter (delete/archive drafts), grep for stray `TODO`s before submission, and pick a single abbreviation mechanism.
