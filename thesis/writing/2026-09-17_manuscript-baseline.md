# Manuscript Writing Baseline — 2026-09-17

**Purpose:** a measurement of the drafted manuscript taken *before* the
`thesis-writing` skill existed, so that a later revision pass has a number to move.

**How to read it:** the absolute numbers mean very little. A thesis is not graded on median
sentence length, and a text can score perfectly here and still say nothing. Only the *delta*
between this snapshot and a later one carries information. This caveat is borrowed from
`thesis/writing/anti-slop-skill.md`, which makes the same point about its own linter.

**Measured on:** the nine `.tex` files under `thesis/manuscript/chapters/` with drafted prose,
at commit `81fa911`. Chapters 1 and 5 and §§3.6–3.7 were comment skeletons and contributed no
prose.

## Method

Measurement was done with a throwaway script in a session scratchpad; nothing executable was
added to the repo. The stripping rules matter for reproducing the numbers:

- Comments removed from an unescaped `%` to end of line; `\%` preserved.
- Dropped wholesale: `tabular`, `tabularx`, `longtable`, `tikzpicture`, `equation`, `align`,
  `verbatim`, `lstlisting`, `minipage`. `figure` and `table` bodies dropped but `\caption{}`
  captured separately and excluded from prose statistics. `quote` blocks likewise excluded.
- `$...$` counts as one word; `\cite` as zero; `\Cref`/`\ref` as two (it renders as
  "Section 3.2"); `\texttt{}`/`\url{}` as one opaque token. Content of `\textit`, `\emph`,
  `\textbf`, `\enquote`, `\caption`, `\footnote` kept.
- Sentence splitting suppressed after `e.g`, `i.e`, `al`, `cf`, `vs`, `Fig`, `approx`, `etc`,
  single initials, and after the `.\ ` idiom the manuscript already uses.

Note on line numbers: the `.tex` sources are not hard-wrapped, so **one source line is one
paragraph**. A `file:line` reference below locates a paragraph, not a sentence.

## 1. Sentence length

| file | words | sents | median | mean | p75 | p90 | max | >30w | >40w | >55w |
|---|---|---|---|---|---|---|---|---|---|---|
| `2-Literature_Review.tex` | 10951 | 220 | 50 | 49.8 | 65 | 80 | 150 | 168 (76.4%) | 140 (63.6%) | 85 (38.6%) |
| `30-Overview.tex` | 307 | 8 | 43 | 38.4 | 56 | 66 | 66 | 6 (75.0%) | 4 (50.0%) | 2 (25.0%) |
| `31-Data_Sourcing_and_Taxonomy.tex` | 3005 | 88 | 34 | 34.1 | 45 | 59 | 80 | 47 (53.4%) | 29 (33.0%) | 12 (13.6%) |
| `32-Data_Quality_and_Curation.tex` | 4875 | 138 | 33 | 35.3 | 47 | 57 | 121 | 80 (58.0%) | 47 (34.1%) | 15 (10.9%) |
| `33-Synthetic_Data_Supplementation.tex` | 3696 | 104 | 32 | 35.5 | 44 | 61 | 119 | 54 (51.9%) | 33 (31.7%) | 14 (13.5%) |
| `34-Data_Augmentation.tex` | 1035 | 25 | 40 | 41.4 | 51 | 70 | 86 | 17 (68.0%) | 12 (48.0%) | 6 (24.0%) |
| `35-Synthetic_Generator_Comparison.tex` | 3991 | 136 | 28 | 29.3 | 40 | 51 | 73 | 58 (42.6%) | 33 (24.3%) | 9 (6.6%) |
| `4-Results.tex` | 2144 | 78 | 26 | 27.5 | 37 | 47 | 62 | 30 (38.5%) | 13 (16.7%) | 3 (3.8%) |
| **corpus** | **30004** | **797** | **34** | **37.6** | **51** | **65** | **150** | **460 (57.7%)** | **311 (39.0%)** | **146 (18.3%)** |

The single most useful fact in this table: **the newest writing already meets the targets.**
`35-*.tex` (median 28) and `4-Results.tex` (median 26) sit at or below the skill's 30-word
target, while `2-Literature_Review.tex` sits at 50 — its median sentence is longer than every
other file's p75. The rules in the skill are not a foreign register being imposed; they
describe what the author's most recent writing already does, and Chapter 2 is the outlier.

Clause length, measured as the longest run between `;`, `:` or a spaced `--`: mean 27.6,
median 26, p90 44, max 87. Only 2.5% of sentences contain a clause over 55 words, against
18.3% of sentences over 55 words overall — confirming that the long sentences are long by
*accumulation of clauses*, not by any one clause being unreadable. That is what makes them
fixable: the split points are already punctuated.

## 2. Paragraph size

231 paragraphs. Sentences per paragraph: mean 3.5, median 3, p90 6, max 9. Words per
paragraph: mean 130, median 123, p90 218, max 452. Paragraphs over 200 words: 34 (14.8%);
over 250 words: 10 (4.3%).

This is why the skill budgets paragraphs in **words rather than sentences**: at a median of 3
sentences per paragraph, every paragraph in the manuscript passes a six-sentence cap, including
the 452-word one.

## 3. The five longest sentences per file

Raw material for the before/after pairs in the skill's `examples.md`. Quoted from the stripped
text, so `[CITE]` is a `\cite{}`, `<REF>` a `\Cref{}`, `#` a math-mode token.

**`2-Literature_Review.tex`**

- **150w, `:90`** — "Across this body of work, three recurring failure modes motivate treating synthetic imagery as a supplement whose reliability must be actively verified rather than assumed: a persistent texture and lighting domain shift …"
- **134w, `:143`** — "TensorFlow Lite Micro [CITE] illustrates why the choice of export format and runtime matters for such a benchmark's validity in the first place: it argues for a portable, interpreter-based deployment framework …"
- **133w, `:105`** — "Taken together, these three axes – structured blind human rating, automatic distributional and embedding-based proxies, and downstream task utility – are exactly the three-way triangulation this thesis adopts …"
- **120w, `:88`** — "He et al. [CITE] conducted the first systematic study of this question for text-to-image diffusion output (using GLIDE), finding that synthetic images can measurably improve zero-shot classification …"
- **110w, `:88`** — "DA-Fusion [CITE] takes an intermediate position: rather than generating synthetic images from scratch, it edits real images semantically using a pretrained Stable Diffusion checkpoint …"

**`32-Data_Quality_and_Curation.tex`**

- **121w, `:36`** — "This stage requires no image I/O and is evaluated purely from source-provided metadata, and its pass criteria are deliberately per-source: GBIF records were to be admitted on a classifier-derived label …"
- **102w, `:9`** — "A blanket cloud vision-language model (VLM) pass over the entire corpus was considered and rejected: even paid API tiers cap out at approximately # requests per minute …"
- **100w, `:149`** — "# rewards a bounding box occupying between # and # of the frame (a score of #), decaying toward # below # or above # of frame area …"
- **89w, `:129`** — "Rather than manually reviewing all such images, the existing per-box SpeciesNet classification (<REF>) was reused to flag a secondary box only when its predicted taxon diverged …"
- **73w, `:116`** — "Treating a low pass rate under the match-level filter as evidence of a classifier coverage gap rather than a labeling problem required a fallback rule …"

**`33-Synthetic_Data_Supplementation.tex`**

- **119w, `:47`** — "Each generated image is specified along six largely independent axes to avoid the repetitive, canonical-pose imagery that a single fixed prompt template per species would otherwise produce …"
- **102w, `:44`** — "A generic instruction to depict an animal in its habitat fails here for three reasons that compound: behaviour is species-specific, so a cross-product of generic scenes produces a walrus climbing a tree …"
- **83w, `:78`** — "The fix added explicit diversity constraints to the scenario-generation step: at least one resting, lying, or sleeping behavior and at least one active foraging behavior per class …"
- **81w, `:31`** — "Band B (# classes, including Canada lynx, spectacled bear, caracal, giant panda, chimpanzee, and springbok) supplements a #-image real training set …"
- **73w, `:31`** — "Band A comprises the # most data-poor classes (for example walrus, old world porcupine family, raccoon dog, kinkajou, water deer, Eurasian badger, and nine-banded armadillo) …"

**`31-Data_Sourcing_and_Taxonomy.tex`** — longest 80w `:19`, then 78w `:110`, 71w `:11`, 71w `:46`, 71w `:112`.
**`34-Data_Augmentation.tex`** — longest 86w `:63`, then 73w `:70`, 70w `:68`, 59w `:61`, 56w `:9`.
**`35-Synthetic_Generator_Comparison.tex`** — longest 73w `:279`, then 66w `:73`, 65w `:50`, 65w `:184`, 62w `:252`.
**`4-Results.tex`** — longest 62w `:127`, then 57w `:129` (×2), 54w `:125`, 53w `:95`.
**`30-Overview.tex`** — longest 66w `:5`, then 56w `:7`, 49w `:7`, 43w `:7`, 38w `:5`.

## 4. What is already clean

Recorded so that no future pass wastes effort here, and so the skill is not credited with
fixing things that were never broken.

| Check | Result |
|---|---|
| Classic lexical slop tells (`delve`, `leverage`, `seamless`, `cutting-edge`, `pivotal`, `realm`, `testament`, `underscore`, `showcase`, `harness`, `utilize`, `myriad`, `plethora`, `not only X but also`, `some researchers`) | **3 hits / 30,004 words**, all legitimate usage |
| First-person pronouns | **0** |
| `perform an analysis of`-style nominalized verbs | **0** |
| Expletive openers (`There is/are/was/were`) | **0** |
| `\ref` in drafted prose (house form is `\Cref`) | **0** — all 6 `\ref` are inside stub comments |
| `TODO`/`FIXME` in non-comment text | **0** — all 32 belong to the intentional stub skeleton |
| Unresolved citation keys (100 distinct keys, 158 calls, against 224 bib entries) | **0** |
| Dangling `\Cref`/`\ref` targets | **0** |
| Stacked hedges (`may possibly`, `could arguably`) | **0** |
| Hedge density | **0.21 per 100 words** (≈62 tokens), which is calibration, not padding |

The hedging result deserves emphasis because it is the thing most likely to be damaged by a
careless "anti-slop" edit. The manuscript hedges *correctly* and sparingly. See the skill's
`conflicts.md` §2.

## 5. Defects found

### 5.1 Spelling register is mixed — the largest mechanical defect

The manuscript mixes British and American spelling throughout, including both spellings of the
same word inside one paragraph (`33-Synthetic_Data_Supplementation.tex:44`/`:47` contains both
`behaviour` and `behavior`).

| pair | British | American |
|---|---|---|
| `-ise/-isation` vs `-ize/-ization` | 63 | 94 |
| `behaviour*` / `behavior*` | 16 | 16 |
| `artefact` / `artifact` | 10 | 5 |
| `judgement` / `judgment` | 6 | 3 |
| `colour` / `color` | 3 | 4 |
| `labelled` / `labeled` | 1 | 5 |
| `neighbour` / `neighbor` | 0 | 4 |
| `modelling` / `modeling` | 0 | 2 |
| `analyse` / `analyze` | 0 | 1 |

**Decision recorded in the skill:** American English, matching `\usepackage[english]{babel}`
in `main.tex` (babel's `english` selects US patterns; `british` would have to be requested
explicitly). The pending fixes are 63 `-ise/-isation` forms, 16 `behaviour`, 10 `artefact`,
6 `judgement`, 3 `colour`, 2 `penalise`, 1 `favourable`, 1 `labelled`.

### 5.2 The acronym mechanism is declared and dead

`preamble/acronyms.tex` defines **25 acronyms** with `\newacronym`. The chapters contain
**zero** `\gls`, `\glspl`, `\acrshort` or `\acrlong` calls. The acronym list will therefore
render empty, while the terms are spelled out by hand instead — "Knowledge Distillation" ×8,
"mean Average Precision" ×5, "Quantization-Aware Training" ×4, "Vision-Language Model" ×1.
`KD` as an acronym appears 0 times.

This is `bachelor-thesis-analysis.md` §8 rough-edge 3 recurring in a new form: that thesis had
two competing abbreviation mechanisms, this one has a declared mechanism and a parallel manual
habit.

### 5.3 Four figures are placed but never referenced

A float that no sentence points at makes no claim and may float anywhere in the chapter.

```
fig:generator-headline-map        4-Results.tex:58
fig:prompt-length-ablation        4-Results.tex:90
fig:generator-cost-vs-map         4-Results.tex:106
fig:generator-per-class-heatmap   4-Results.tex:122
eq:quality-score                  32-Data_Quality_and_Curation.tex:147
```

All four tables in the same chapter are referenced correctly, so this is specific to the
figures. 39 further section labels are unreferenced, 21 of them in files that are still
skeletons; that is a label-everything convention rather than a defect.

### 5.4 `rather than` is an authorial tic

**202 occurrences, 0.67 per 100 words** — roughly one sentence in four. The construction is
correct English and often exactly right for this thesis, whose argument repeatedly takes the
form "X was chosen over Y". At this density it becomes a rhythm the reader can predict, and it
frequently marks a contrast that would land harder as two sentences.

### 5.5 Small mechanical hits

- **Percent outside math mode (2):** `2-Literature_Review.tex:5` (`80\%`), `:158` (`50\%`).
- **Cardinal numbers outside math mode (3 genuine):** `2-Literature_Review.tex:158` (`50`),
  `:162` (`225`), `32-Data_Quality_and_Curation.tex:38` (`2`). Proper nouns and version strings
  (`Hexagon 685`, `SD 3.5`, `COCO 2017`) are correctly left in text.
- **Raw `"` quotes instead of `\enquote{}` (4 pairs, all Chapter 2):** `:5`, `:81`, `:86`,
  `:88`. `\enquote` is used correctly 46 times elsewhere, including at `:118` in the same file.
- **Directional reference to a float (1):** `4-Results.tex:153`, "the table above". Eight
  further `described above`/`described below` phrases refer to prose sections, which is
  acceptable; the float reference is not.
- **`It is worth …` openers (3):** `2-Literature_Review.tex:30`,
  `31-Data_Sourcing_and_Taxonomy.tex:114`, `32-Data_Quality_and_Curation.tex:185`.

## 6. Terminology drift

Full counts behind the glossary in the skill's `terminology.md`. The four collisions where one
word carries two meanings are the ones that matter, because a reader cannot recover the
intended sense from context alone.

**Collision 1 — `distillation`.** Bare `distillation` ×34 and `distilled` ×7 refer to *knowledge*
distillation in Chapter 2 and §3.4, and to *diffusion step* distillation (SD 3.5 Large Turbo) in
§3.5 and Chapter 4. Two referents, one word, same document. `knowledge distillation` in full:
7. `KD`: 0.

**Collision 2 — `tier`.** 41 bare uses across three senses: taxonomic ranks (`Tier 1/2/3`,
`31-*.tex:19`), evaluation assessment levels, and generator API/quality tiers (18 uses in
`4-Results.tex`, 14 in `35-*.tex`).

**Collision 3 — `regime`.** 47 uses across prompt regime (12) and training/evaluation regime (35).

**Collision 4 — `label`.** 157 uses across the annotation record and the taxonomic name.

**Gap — the mixed test set has no name.** `mixed test set` appears **0 times**, although
`CLAUDE.md` names it as the headline evaluation set. `mixed` runs bare 10 times and `union` 4.
Meanwhile the real test set carries five surface forms: `real test set` (13), `real images`
(15), `real photographs` (12), `real imagery` (8), `real data` (7).

**Gap — `proxy hardware` appears 0 times.** `QCS605` ×12, `Raspberry Pi 5` ×6, `target hardware`
×5, `stand-in` ×1. Five of seven bare `proxy` uses mean something else entirely
(`recognizability proxy`, `automatic proxies`).

**Teacher/student:** `teacher` ×35 and `student` ×29 bare; `teacher model` and `student model`
exactly once each, both in the same sentence at `30-Overview.tex:5`.

**Experimental unit:** `cell` ×68 (only in `35-*.tex` and `4-Results.tex`), `Setup A/B/C` ×10
plus `setup` ×19 (only in `34-Data_Augmentation.tex`), `run` ×53, `condition` ×23,
`variant` ×11, `configuration` ×9. The two naming schemes do not currently leak into each
other, which is why the glossary fixes the boundary rather than renaming either.

**Class vocabulary:** `class` ×284 dominates, with `category` ×26 (partly forced by COCO's
`supercategory` schema), `species` ×143, `taxon`/`taxa` ×15.

**Hyphenation drift (four live):** `test set` 43 / `test-set` 6; `per-class` 18 / `per class`
14; `ground-truth` 15 / `ground truth` 11; `camera-trap` 13 / `camera trap` 5. Plus
`look-alike` 13 / `lookalike` 2. Most of these are not errors — the attributive form takes the
hyphen and the noun does not — but the manuscript applies the rule inconsistently.

## 7. What to re-measure after a revision pass

1. Median and p90 sentence length for `2-Literature_Review.tex`. Target: median ≤ 35,
   sentences over 55 words below 10% (from 38.6%).
2. Paragraphs over 200 words. Target: below 5% (from 14.8%).
3. `\gls` call count. Target: greater than 0 — every acronym in `acronyms.tex` introduced
   through the mechanism, or removed from it.
4. Unreferenced float labels. Target: 0.
5. British spellings. Target: 0.
6. `mixed test set` occurrences once Chapter 4 §§4.2–4.4 are written. Target: greater than 0.
7. Hedge density. Target: **unchanged at roughly 0.2 per 100 words.** This is the one number
   that should not move, and a revision pass that drives it toward zero has damaged the thesis.
