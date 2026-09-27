# Figure/table guideline and new manuscript figures

**Date:** 2026-09-21
**Status:** implemented, not yet committed or compiled
**Touches:** `.claude/skills/thesis-writing/`, `thesis/manuscript/chapters/`, `thesis/manuscript/appendices/`, `thesis/manuscript/figures/images/`

## Why this exists

Two related requests: give the `thesis-writing` skill a durable rule for choosing
between a plot, a table, or a sentence when reporting a numeric comparison (there
wasn't one), and add qualitative image figures the manuscript was missing — a
comparison of what the different synthetic-image generators actually produce, and
a check on whether the three zebra species used throughout the generator-comparison
study already had a dedicated subsection (they don't; see below).

`thesis/docs/old_bachelor_thesis/` (the author's own prior thesis, kept in-repo as a
worked example) supplied the layout pattern for multi-panel image figures, which is
new territory for `thesis/manuscript/` — its `figures/images/` directory was empty
before this change.

## 1. Skill guideline: `references/figures.md`

New file: `.claude/skills/thesis-writing/references/figures.md`. Two things it
covers that nothing else in the skill did:

**Medium choice.** Two numbers stay inline prose. More numbers where the
comparison itself isn't the point (a lookup, a ranking, a cost breakdown) go in a
table. A plot is justified only when prose or a table would lose the insight — a
trend, a cluster, a trade-off frontier, a distribution shape, a grid too dense to
scan as a table. The stated test: delete the plot and keep only the table it was
built from — if the reader still sees the point, the table alone was enough.

**Multi-panel image layout.** The `subfigure`/`subcaption` pattern lifted from
`thesis/docs/old_bachelor_thesis/chapters/3-Methods_and_Implementation.tex:20-44`
(three panels at `0.31\textwidth`) and `:68-86` (two panels at `0.49\textwidth`,
with `\renewcommand{\figurename}{...}` for themed figure groups) — reproduced as a
worked template, plus the rule that every panel keeps its own `\label` even though
only the outer caption is normally `\Cref`-referenced from prose.

Wired in: `SKILL.md`'s "Which reference to read" table gets a new row pointing to
`figures.md`; `mechanics.md` §3 ("Floats") now cross-references it instead of being
the only place the subcaption convention was mentioned.

## 2. Zebra species: no dedicated subsection

Checked as requested. The three look-alike zebra species (plains, Grévy's,
mountain) are **not** given their own subsection anywhere in the manuscript.
They're woven through the generator-comparison sub-study instead:

- `tab:generator-comparison-classes`
  (`3-Methods_and_Implementation/35-Synthetic_Generator_Comparison.tex:77-101`)
  lists all three as the study's "Look-alike" group, with band, real-test-image
  count, and one distinguishing visual marker per species.
- `tab:generator-ranking` (`4-Results.tex:30-52`) carries a "Zebra confusion"
  column, the within-group misclassification rate per generator/regime cell.
- `4-Results.tex:129` discusses what that confusion rate does and doesn't reveal
  about generator quality.

This is treated as intentional structure — a fine-grained-discrimination probe
folded into the generator study, not a standalone taxonomy topic — so no new
subsection was created. Instead, a figure was added at the existing anchor point
(§3 below) rather than restructuring the section.

## 3. New figures added

All four insertions use the `subcaption` pattern from `figures.md`. `subcaption`
and `float` are already loaded in `main.tex`, so no preamble changes were
needed. Every new figure is referenced from prose with a `\Cref` sentence, per
`mechanics.md` §3.

### 3.1 Three zebra species — real photographs

**Location:** `3-Methods_and_Implementation/35-Synthetic_Generator_Comparison.tex`,
immediately after `tab:generator-comparison-classes`.
**Figure:** `fig:zebra-species`, three panels (`0.31\textwidth` each) — plains
zebra, Grévy's zebra, mountain zebra — each captioned with the same distinguishing
marker already stated in the table, so the figure visually anchors what the table
claims in words.

### 3.2 Kinkajou generator comparison (in-chapter)

**Location:** `4-Results.tex`, §4.1.4 "Per-Class Structure", directly after the
paragraph that already describes FLUX.2-klein's kinkajou misrender in prose (a
ringed tail and spotted coat, "closer to a genet or civet than to the real
animal").
**Figure:** `fig:kinkajou-generator-samples`, four panels (`0.23\textwidth`
each, `\figurename` renamed to "Samples" for this one figure) — a real photo,
FLUX.2-klein's incorrect render, and correct renders from SD 3.5 Large and
HiDream-I1. This fills the gap the same section already names: downstream
accuracy alone can't detect this error (FLUX.2-klein and SD 3.5 Large score
statistically identically on it), which is exactly the argument
§4.1.5 ("The Qualitative Rubric: Designed, Not Executed") makes for the rubric
that was never run. §4.1.5's text was updated to point at the figure directly.

### 3.3 Appendix gallery — three classes × six generators

**Location:** `appendices/appendix.tex`, new §"Qualitative Generator Samples"
(`sec:appendix_generator_gallery`), replacing part of what had been a pure `%
TODO` stub reserving space for "additional qualitative example images once
training results exist."
**Three figures**, one per class, `\figurename` renamed to "Appendix Samples" for
the whole section: `fig:appendix-gallery-lion` (Band D, Anchor — the "sanity
ceiling" class), `fig:appendix-gallery-saiga` (Band A, Rare — only 50 real test
images), `fig:appendix-gallery-plains-zebra` (Band D, Look-alike — ties back to
§3.1's figure, with a caption that explicitly invites the comparison). Each
figure has 7 panels (`0.23\textwidth`, 4-across then 3-across) — a real photo
plus all six locally-run generators of `tab:generator-local-roster` at the
`maxlen` regime — and every one of the 21 panels has its own `\label`
(`fig:appendix-<class>-<model>`), even though none is individually referenced
yet, per `figures.md`'s panel-labeling rule.

### 3.4 AX Visio device photos

**Location:** `3-Methods_and_Implementation/33-Synthetic_Data_Supplementation.tex`,
in "Photography Style Correction", right after the paragraph that already cites
"the 10 genuine AX Visio captures available" as evidence that the binocular's
optics produce no background blur.
**Figure:** `fig:ax-visio`, two panels (`0.49\textwidth`) — a close-subject and a
distant-subject capture — visually substantiating a claim that was previously
backed only by prose citing a small (10-image) evidence base.

## 4. Image sourcing

No new images were generated. Everything was selected from data already on disk
and lightly resized. Two Haiku subagents did the mechanical work in parallel (view
2-3 candidates with the `Read` tool, pick the clearest one, convert, copy) so it
didn't consume the main session's context; I wrote all of the surrounding LaTeX
and prose myself.

**Conversion**, applied to every output file:
```
convert "<source>" -resize "1200x1200>" -quality 87 "<dest>.jpg"
```

**Sources:**
- Real photographs: `data/synthetic_model_comparison/test/images/<class>/` —
  Wikimedia-Commons-sourced files already cleared for training use during the
  dataset-licensing review (`31-Data_Sourcing_and_Taxonomy.tex`).
- Generator outputs: `data/synthetic_model_comparison/train/<model>/maxlen/images/<class>/`
  — the lowest-numbered image (`..._001.png`) for each class/model pair, from the
  generator-comparison sub-study's own archived output.
- AX Visio photos: `resources/AX_Visio_images/*.jpeg` (originals 3-7 MB each).

**Destinations** (all under `thesis/manuscript/figures/images/`, new
directories): `zebra_species/` (3 files), `generator_comparison/` (4 files, the
in-chapter kinkajou set), `generator_comparison/appendix/` (21 files, the
gallery), `ax_visio/` (2 files). 30 files, 7.1 MB total.

## 5. Verification done

No LaTeX toolchain exists in this container (`mechanics.md`'s standing note, also
true per the Overleaf-sync doc), so nothing was compiled. Checked instead by
inspection:

- Brace and `\begin{}`/`\end{}` balance in every edited file (all clean).
- No duplicate `\label{}` anywhere in `chapters/` or `appendices/` (all four new
  figures plus all 21 appendix panel labels are unique).
- Every `\Cref`/label target the new prose points at resolves to exactly one
  `\label{}` (`tab:generator-local-roster`, `tab:generator-comparison-classes`,
  `sec:results_per_class_structure`, `sec:results_rubric_not_executed`,
  `sec:generator_comparison_evaluation`, `fig:zebra-species`,
  `fig:kinkajou-generator-samples`).
- New prose contains no em dash, `--`, or `;` (the project's house rule).
- All 30 destination image files exist and were confirmed as valid images by the
  subagents that produced them (dimensions capped at 1200px on the long edge).

## 6. Open items

- **Images not eyeballed by me.** I can't render either the source PNGs/JPEGs or
  the compiled PDF in this container, so the specific frames the subagents picked
  (listed file-by-file in their reports) haven't been visually reviewed against
  the claims their captions make. Worth a look before this ships.
- **No photographer/source credit lines.** The real photographs are
  Wikimedia-Commons files cleared for training use, but reproducing them as
  thesis figures will likely need a credit line per their CC license — not yet
  added.
- **Nothing committed.** The working tree already carried a large amount of
  unrelated in-progress state (training runs, model exports, a bibliography change) and the current
  branch is `thesis/overleaf-sync`, which per project convention is never pushed
  directly — changes get merged into `main` (via a worktree if the tree is
  dirty, which it is) and `make overleaf-push` is left to the user. That merge
  hasn't been done.
- **Still-open TODOs in `appendix.tex`** (the full 225-class/band table, the full
  look-alike group listing) were left untouched — out of scope for this change.

## 6b. Photograph provenance (reconstructed 2026-09-21)

The per-file mapping from manuscript figure to source image was never written down when the
images were copied in (§4 names only the directories). It was reconstructed afterwards by
32x32 normalized-RGB signature matching against the source pools, restricted to candidates of
identical aspect ratio. Every one of the nine files matched at distance 0.0006 to 0.0023
against a next-best of 0.18 to 0.39, a roughly 100x separation, and every matched source is
larger than 1200 px on the long edge, consistent with the one-way `-resize 1200x1200>` used.
One file (`plains_zebra_real.jpg`) retains EXIF whose `DateTimeOriginal` matches its matched
Wikimedia record exactly, independently confirming the method. The other eight are stripped.

Attribution comes from `data/wikimedia/metadata.csv`, which carries `artist`, `license_short`,
`license_url` and `description_url`. **This table is the only record of the mapping.** The
figure directories are untracked and `data/` is gitignored, so it cannot be re-derived from
git history.

| Manuscript file | Source | Author | Licence |
|---|---|---|---|
| `zebra_species/plains_zebra.jpg` | Wikimedia `20220226_Zebra_at_Zhengzhou_Zoo.jpg` | Windmemories | CC BY-SA 4.0 |
| `zebra_species/mountain_zebra.jpg` | Wikimedia `Cape_Mountain_Zebra_(Equus_zebra_zebra)_(31766817534).jpg` | Bernard DUPONT | CC BY-SA 2.0 |
| `zebra_species/grevys_zebra.jpg` | Wikimedia `Grevy's_Zebra_(Equus_grevyi)_(48984753968).jpg` | Bernard DUPONT | CC BY-SA 2.0 |
| `generator_comparison/appendix/saiga_real.jpg` | Wikimedia `Saiga_antelope_at_the_Stepnoi_Sanctuary.jpg` | Andrey Giljov | CC BY-SA 4.0 |
| `generator_comparison/appendix/plains_zebra_real.jpg` | Wikimedia `2012-02-Cayo_Saetia_Zebras_01_anagoria.JPG` | Anagoria | CC BY 3.0 |
| `generator_comparison/appendix/lion_real.jpg` | Wikimedia `2012-09-15_Tierpark_Berlin_39.jpg` | uploader L. Beck, artist field reads "Meine Mutter (Erlaubnis liegt vor)" | CC BY-SA 4.0 |
| `generator_comparison/kinkajou_real.jpg` | GBIF `gbif_kinkajou_00017.jpg` | **UNRECOVERABLE** | **UNKNOWN** |
| `ax_visio/ax_visio_near.jpg` | `resources/AX_Visio_images/AXV_20241121-143038.012.jpeg` | Swarovski Optik (supplied by the AX Visio PO) | **NONE RECORDED** |
| `ax_visio/ax_visio_far.jpg` | `resources/AX_Visio_images/AXV_20241115-174745.987.jpeg` | Swarovski Optik (supplied by the AX Visio PO) | **NONE RECORDED** |

Credit lines for the six Wikimedia photographs were added to the outer captions of
`fig:zebra-species`, `fig:appendix-gallery-lion`, `fig:appendix-gallery-saiga` and
`fig:appendix-gallery-plains-zebra`. Five of the six are ShareAlike, and the manuscript panels
are lossy-resized derivatives, so the share-alike term attaches to the modified images.

Two blockers remain, both escalated rather than fixed:

- **`kinkajou_real.jpg` cannot be attributed.** GBIF filenames originally embedded the
  occurrence ID, but `scripts/rename_dataset_images.py` renamed them in place and saved no
  mapping, and the original download directory no longer exists. The ID must not be guessed.
  Recommended fix is to swap the panel for a Wikimedia kinkajou of known attribution, since
  `fig:kinkajou-generator-samples` does not depend on which real photograph is used.
- **The AX Visio pair is a rights-clearance problem, not a missing credit.**
  `docs/progress_notes/2026-03-11_dataset-stakeholder-meeting-and-model-architecture.md:23`
  records that the ten frames were shared by the AX Visio Product Owner at Swarovski Optik.
  They are neither first-party captures nor public material, and no licence or written
  permission exists anywhere in the repo. `fig:ax-visio` should not ship until permission is
  obtained in writing.

## 7. File manifest

Modified:
- `.claude/skills/thesis-writing/SKILL.md` (+1 line)
- `.claude/skills/thesis-writing/references/mechanics.md` (+2/-1 lines)
- `thesis/manuscript/chapters/3-Methods_and_Implementation/33-Synthetic_Data_Supplementation.tex` (+21 lines)
- `thesis/manuscript/chapters/3-Methods_and_Implementation/35-Synthetic_Generator_Comparison.tex` (+28 lines)
- `thesis/manuscript/chapters/4-Results.tex` (+37/-2 lines)
- `thesis/manuscript/appendices/appendix.tex` (+155/-3 lines)

New:
- `.claude/skills/thesis-writing/references/figures.md`
- `thesis/manuscript/figures/images/zebra_species/` (3 files)
- `thesis/manuscript/figures/images/generator_comparison/` (4 files)
- `thesis/manuscript/figures/images/generator_comparison/appendix/` (21 files)
- `thesis/manuscript/figures/images/ax_visio/` (2 files)
- This file.
