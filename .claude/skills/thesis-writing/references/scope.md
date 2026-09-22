# Scope — what belongs in the document at all

`construction.md` builds a good sentence. This file decides whether the sentence
should exist. It is the rule set that was missing when the manuscript grew to
57,291 words, roughly three and a half times the bachelor thesis, against a
department instruction that reads:

> "What matters is not producing as many pages as possible. Would you feel like
> reading 150 pages if 70 would have done the job?"
> — `thesis/docs/formal_requirements_en.md:41`

Read this file before drafting a section, before adding a paragraph to one, and
before deciding that something you did deserves to be written down.

---

## 1. The relevance test

A passage earns its place only if it does one of three things:

1. lets the reader **follow a method** that produced a number the thesis reports,
2. lets the reader **interpret a result**, or
3. **bounds a claim** the thesis makes.

Everything else goes, however true it is and however much work it represents.
Effort is not an argument for inclusion. The reader is one examiner reading
eighty pages under time pressure, and every paragraph that fails the test spends
their attention on something that will not appear again.

Apply the test to whole subsections, not only to sentences. A subsection that
survives only because deleting it feels wasteful is the single largest source of
length in this manuscript.

---

## 2. Proper names: explain or cut

**A proper name appears only if it is glossed in one clause at first use *and*
used, benchmarked, or explicitly rejected on a stated criterion later.** If both
conditions do not hold, delete the name and keep the claim.

> **Weak.** Applied to *StyleGAN*, *ProGAN*, *BEGAN*, and *WGAN-GP* on
> CelebA-$64$, even the best model, *StyleGAN* with the truncation trick,
> deceived only $50.7\%$ of raters.
>
> **Better.** Under blind human evaluation the strongest generator of its
> generation deceived only $50.7\%$ of raters, barely above the chance baseline
> \cite{zhouHYPEBenchmarkHuman2019}.

Four names, a dataset and a technique disappear; the claim that the chapter
actually uses survives intact. The same cut applies to `CLIP ViT-L`,
`OpenCLIP ViT-bigG`, `CIFAR-10`, `DA-Fusion`, `SDEdit`, `M2m`, `RSG`, `FTL`,
`LEAP`, `MiSLAS`, `Remix`, `FASA`, `MetaSAug`, and the YOLOv6–v12 block names
(`SimOTA`, `GELAN`, `C2f`, `C3k2`, `C2PSA`) — none of which the work uses.

Three supporting rules:

- **Budget roughly one to two new proper names per page.** The bachelor thesis
  introduces about twenty across a fourteen-page literature chapter and explains
  every one.
- **Gloss in one clause, then use the term bare forever:** "a feature map, a 2D
  grid of numerical values representing the learned features at each spatial
  location". Not a sentence, not a paragraph.
- **Never use an acronym before its definition.** The manuscript used FID
  sixty-nine lines before defining it and CAS thirty-seven lines before. A name
  that must be forward-referenced is a name in the wrong section.

A number attached to a cut name goes with it. `$2.6$ B vs $860$–$865$ M
parameters`, `FID 3.17`, `$\tau_c = 51.2 / 44.9$`, `$\rho = -0.029$, $p = 0.96$`
are evidence for a claim the thesis does not make.

---

## 3. Claim first, citation attached

**State the finding; hang the citation off the end. The source is not the
grammatical subject.** In the bachelor thesis only 7% of cited sentences name a
source in running text. Chapter 2 of this manuscript had 45 paragraphs that
open with one.

> **Weak.** *DA-Fusion* \cite{trabuccoEffectiveDataAugmentation2025} takes an
> intermediate position. Rather than generating synthetic images from scratch,
> it edits real images semantically…
>
> **Better.** Editing real images semantically, rather than generating them from
> scratch, outperforms both geometric augmentation and unconstrained diffusion
> augmentation in fine-grained few-shot settings
> \cite{trabuccoEffectiveDataAugmentation2025}.

Four cases earn a named source, and only these four:

| Case | Example |
|---|---|
| Borrowing someone's taxonomy | "categorized into three areas, as shown by Wahyudi et al. \cite{…}" |
| Historical priority is the claim | "the neocognitron, introduced by Kunihiko Fukushima \cite{…}" |
| Identifying which implementation was used | "Faster R-CNN, as implemented by Ren et al. \cite{…}" |
| The study itself is the object of discussion | a survey whose *scope* is being characterized |

Everything else is claim-first. `2-Literature_Review.tex:397` already does this
correctly with the same source that `:183` mishandles — use it as the model.

**In Results and Discussion, a citation must be a genuine external comparison.**
The bachelor thesis carries zero citations in either chapter. That is the right
instinct: those chapters report this work's own numbers, and a citation there is
either a real published figure being compared against or a literature claim that
belongs back in Chapter 2.

---

## 4. One family, one paragraph

Literature the work does **not** use gets a single claim-first paragraph stating
what the field does and why this work went another way. Not a paragraph per
paper, and never a named subsection per method.

The long-tail section ran to 1,016 words across re-sampling, re-weighting,
re-margining and decoupling, implemented none of them, and existed only to set
up one contrast. One paragraph closing on that contrast does the same work and
demonstrates the same reading.

This is how breadth is shown without cost. A reader can tell from one dense,
correctly cited paragraph that the author knows the field; they cannot tell it
any better from six.

---

## 5. Report an abandoned attempt only when the failure produced the final design

Then give it **four sentences at most: symptom, diagnosis, fix.**

> **Weak.** `31-Data_Sourcing_and_Taxonomy.tex` — "The LILA BC Camera-Trap
> Detour": 304 words and a filtering table describing three sub-datasets scanned
> against the taxonomy, for a source that contributed no images.
>
> **Better.** Three large camera-trap archives were scanned against the taxonomy
> and rejected: their species coverage overlapped the classes already
> well-supplied and missed the scarce ones, and their imagery is
> fixed-viewpoint. No images from them entered the corpus.

Deleted outright, not relocated: supervision-meeting narratives ("In a
supervision meeting this was identified as methodologically insufficient"), tool
bake-offs, bug post-mortems, billing anecdotes, earlier drafts of a design that
was later changed, and any sentence beginning "An early attempt…" whose failure
did not change the final method.

The distinction is whether the reader needs the failure to understand what they
are looking at. A photography style that was replaced explains nothing about the
images that were kept. A segmentation failure that forced the addition of anchor
points explains the anchor points.

---

## 6. One home per statement

**Every caveat, policy, definition and limitation is stated in full exactly
once. Everywhere else is a `\Cref`.** Before writing a caveat, grep for it.

The manuscript stated the real-only-breakout policy five times, "the qualitative
rubric was never executed" six times, and Chapter 4's scope boundaries
duplicated Chapter 5's Limitations near-verbatim for about 600 words.

Where a statement is genuinely needed in two places, the second is one sentence
plus a reference, never a re-argument. Ownership goes to the section the reader
reaches first *and* needs it in: evaluation policy to the evaluation framework,
result caveats to the result, scope limits to Limitations.

---

## 7. Route reference material to the appendix

Chapter prose gets one sentence and a pointer; the material itself goes to the
appendix. This is explicitly endorsed by the department
(`formal_requirements_en.md:65`) and is what the bachelor thesis does.

Route to the appendix: threshold lists and filter specifications, controlled
vocabularies and code lists, export paths and toolchain detail, experiment
tracking and infrastructure setup, per-group tables over roughly twelve rows
(keep a top-and-bottom excerpt inline), and validity campaigns that confirmed a
measurement rather than producing one.

Keep in the chapter: any number the reader must have to read a result, and any
threshold that a later correction or caveat depends on.

---

## 8. No apologia, no self-commentary

The thesis does not discuss itself. Delete:

- explanations of its own section proportions ("weighted to match where the
  experimental work actually went", "this is the chapter's largest section"),
- apologies for scope, which appeared five separate times in Chapter 2 alone and
  twice more in Chapter 4 ("This section is much smaller than the question it
  addresses"),
- defenses against objections nobody raised ("This is stated explicitly because
  the alternative would be presenting them without comment"),
- self-assessment of a decision's importance ("the most consequential single
  decision in the curation pipeline"), and
- meta-labels on the author's own reasoning ("deliberately asymmetric in three
  ways, and each is a considered choice").

Scope limits are real and must be stated — once, flatly, in Chapter 1 and in
`sec:limitations`. A limitation stated once is credible; the same limitation
stated five times reads as anxiety.

---

## 9. The standing word budget

The manuscript was cut from 57,291 words to 36,400 in September 2026. The table
below records where it landed, and those numbers are **ceilings to hold, not
targets to shrink toward**: each file was cut to the point where the next cut
would have removed evidence rather than noise.

| File | Before | Now |
|---|---|---|
| `1-Introduction.tex` | 2,303 | 2,000 |
| `2-Literature_Review.tex` | 11,455 | 6,800 |
| `3-…/30-Overview.tex` | 312 | 170 |
| `3-…/31-Data_Sourcing_and_Taxonomy.tex` | 3,529 | 1,400 |
| `3-…/32-Data_Quality_and_Curation.tex` | 5,422 | 2,100 |
| `3-…/33-Synthetic_Data_Supplementation.tex` | 4,134 | 1,700 |
| `3-…/34-Data_Augmentation.tex` | 1,268 | 800 |
| `3-…/35-Synthetic_Generator_Comparison.tex` | 5,510 | 2,550 |
| `3-…/36-Model_Training_and_Experiment_Tracking.tex` | 2,121 | 1,500 |
| `3-…/37-Evaluation_Framework.tex` | 2,218 | 1,550 |
| `4-Results.tex` | 11,316 | 7,400 |
| `5-Discussion_and_Conclusion.tex` | 4,524 | 3,000 |
| `appendices/appendix.tex` | 2,494 | 5,500 |
| **Total** | **57,291** | **36,470** |

Two things about that total are worth knowing before using it.

**The appendix more than doubled, and that is the design.** Rule 7 routes
reference material out of the chapters, so the appendix absorbing 3,000 words is
the mechanism working, not an overrun. The figure that measures how much thesis
there is to read is the body without it: **54,797 down to 30,970, a 43% cut.**

**Raw `wc -w` counts LaTeX markup, so a float-heavy file reads as longer than it
is.** `4-Results.tex` is 7,400 words by that measure and 5,500 words of prose.
Judge a Results chapter on the prose column, which
`scripts/thesis/check_manuscript.py` prints beside the budget.

A planning estimate of 31,400 preceded this pass and was about 15% optimistic,
because it was derived from the pre-cut files' float-to-prose ratio and did not
anticipate how much would be routed to the appendix. The measured allocation
replaces it. Do not treat the older figure as a target that was missed.

## 10. Structural habits that keep sections short

Adopted from the bachelor thesis, where they replace transitional prose
entirely:

- **Open a chapter or top-level section with a three-to-five sentence roadmap of
  its own contents. Open a subsection with the claim.** Do not roadmap twice.
- **Close a section with one sentence stating the consequence that motivates the
  next.** "These limitations highlight the need for X" is the house formula.
  This is what makes a compressed chapter still read as continuous, and it is
  the first thing to break when cutting hard.
- **Numbers in tables, trends in sentences.** Never restate a table cell in
  prose; state the direction, the peak, and the surprise.
- **Give a parameter a value and a trade-off in two sentences.** "X was set to
  $V$. Too high risks A, too low risks B."
- **Define a naming scheme once in its own short subsection**, then use it
  everywhere without re-expansion.

One rule from the bachelor thesis is *not* adopted: its 76-word median paragraph
is tighter than `construction.md` §2.2's 120-word target, which stays as written.
