# Manuscript prose review — 2026-09-25 (Part II added 2026-09-26)

**Part I** (2026-09-25): a full read of `thesis/manuscript/` (both abstracts, all five
chapters, the appendix) judged **only on the writing**, against the `thesis-writing`
skill's rules (construction, claims, terminology, scope, conflicts). Nothing here
questions the experiments. Findings are ordered by how much they would cost with the
examiner, not by where they occur.

**Part II** (2026-09-26, §§9–13): analysis of the supervisor's feedback — what the
"AI slop" flags have in common, where the same patterns live in the parts he did not
read, and the fix strategy. Part II supersedes Part I's §8 where the two conflict.

File:line references in Part I are to the tree of 2026-09-25; commit `cb07dab`
("take the advisor's wording edits from Overleaf") has since moved a few lines in
`1-Introduction.tex` and `5-Discussion_and_Conclusion.tex`.

**Overall verdict:** the manuscript is in strong shape. The word budget is met
(36,191 against 36,470), citations and cross-references are mechanically clean
(0 unresolved, 0 dangling), expletive openers and "It is worth" are gone, prose
carries no `--` dashes and no semicolons, hedges are calibrated rather than
decorative, and the claim-typing discipline (measured / cited / reasoned / **not
established, said out loud**) is applied consistently — the practicability verdict
(§4.1.6), the watchdog verdict (§4.2.4), and the Limitations chapter are the
strongest writing in the document. What remains is a set of localized defects:
a handful of terminology collisions the skill explicitly bans, two wrong
cross-reference targets, a few statements with more than one home, two numeric
inconsistencies, one overclaim in the Conclusion, and sentence-level polish.

---

## 1. Terminology collisions (the skill's one-name-one-meaning rules)

These matter most because the manuscript is otherwise disciplined about its
vocabulary, so the surviving collisions stand out.

### 1.1 "non-distilled" applied to image generators — the worst collision in the document

`4-Results.tex:41` — "the two **non-distilled** Stable Diffusion 3.5 variants lead …
the **non-distilled** HiDream-I1 ranks eleventh."

In a thesis whose central question is knowledge distillation, "distilled" must never
mean step distillation of a diffusion sampler (`terminology.md` §1.1: the diffusion
sense is always written out as *step distillation*, and a generator that underwent it
is a *step-distilled generator*). "Non-distilled" here means "without step
distillation" and will read, at least momentarily, as a claim about KD. Rewrite, e.g.
"the two Stable Diffusion 3.5 variants without step distillation lead … HiDream-I1,
which also runs its full sampling schedule, ranks eleventh." The same paragraph's
"SD 3.5 Large's four-step Turbo variant" is fine — "four-step" carries the meaning.

### 1.2 "taxonomic tiers"

`31-Data_Sourcing_and_Taxonomy.tex:17` — "The final list holds $225$ classes across
three **taxonomic tiers**." `terminology.md` §1.2 reserves *tier* for the generator
API/quality sense (which Chapters 3.5 and 4 use heavily and correctly) and prescribes
*taxonomic rank* for this sense. This is the exact line the terminology file flagged,
still unfixed. "…across three taxonomic ranks" (or "levels of taxonomic resolution").

### 1.3 "regime" for training bands, and a third loose sense

`terminology.md` §1.3 keeps *regime* for prompt regimes only; training-data
composition is a *band*. Live violations:

- `4-Results.tex:153` — section title "Detection Performance Across **Training
  Regimes** and Test Domains" (and its label `sec:results_regime_comparison`).
  This is the headline results section; "Across Training Bands and Test Domains"
  is both compliant and more precise.
- `4-Results.tex:227` — subsection title "Does **Training Regime** Change Accuracy?"
- `4-Results.tex:225` — "Whether the **training regime** moves it is the next question."
- `33-Synthetic_Data_Supplementation.tex:27` — "It makes three **regimes** comparable
  at matched sample size … confounded with the **training regime**."

A third, generic sense also appears twice: "the **regime** in which downstream
accuracy has no resolution" (`4-Results.tex:138`, `5-Discussion_and_Conclusion.tex:66`).
Harmless English on its own, but with prompt regimes live in the same sections it
is a third meaning for a word the skill allows one. "the region/setting in which…"
avoids it.

### 1.4 Proxy hardware has five names

`terminology.md` §2.5 approves "the proxy hardware". The manuscript uses:
"target-adjacent proxy device" (`1-Introduction.tex:29`, `5:16`), "proxy device"
(×3), "embedded proxy" (§4.3.2 title), "the proxy hardware" (×1, Conclusion), and
bare "the proxy" inside §4.3.2. None is wrong individually; five surface forms for
one referent is the drift the one-name rule exists to stop. Pick one ("the proxy
device" or "the proxy hardware") for running text and keep "Raspberry Pi 400" for
first mention per chapter. (Note the terminology file itself still says Raspberry
Pi 5 and should be updated to match the manuscript's Pi 400.)

### 1.5 Species-name spelling is inconsistent

- **Grévy's zebra** appears with the accent (`35-…:78`, `2-Literature_Review.tex:25`,
  appendix:434) and without it (`2-Literature_Review.tex:176`, `:192`). Pick one
  form for prose; the accented form is the standard English spelling.
- `4-Results.tex:350` — "**Asian elephants are called african elephants**": lowercase
  dataset class names leak into prose mid-sentence, colliding with the capitalized
  "Asian" three words earlier. In prose, either capitalize both normally or mark both
  as class names (`\texttt{}`); the appendix tables' lowercase forms are fine because
  they are declared verbatim reproductions of the category list.

---

## 2. Wrong cross-reference targets

The look-alike merge criterion ("confusing two classes in a single still frame is a
forgivable error vs. a real failure") is stated in `sec:lookalike_groups`
(`37-Evaluation_Framework.tex:38`). Two references point at `sec:gt_annotation_integrity`
instead, which discusses annotation integrity and only *feeds* the grouping:

- `appendices/appendix.tex:122` — "The governing criterion is the one stated in
  \Cref{sec:gt_annotation_integrity}" → should be `sec:lookalike_groups`.
- `35-Synthetic_Generator_Comparison.tex:263` — "a frozen look-alike group
  (\Cref{sec:gt_annotation_integrity})" → same fix.

Both compile and both send the reader to the wrong section, which is worse than a
dangling reference because nothing flags it.

---

## 3. Statements with more than one home (`scope.md` §6)

- **The YOLOv5 licensing constraint (commit `5cdad89`)** is stated in full three
  times: `1-Introduction.tex:55` (Pre-Work paragraph), `2-Literature_Review.tex:44`,
  and `36-Model_Training_and_Experiment_Tracking.tex:9`. One full statement (the
  literature review's, which argues it) plus one-sentence-and-a-\Cref elsewhere
  would do. The Introduction's Pre-Work paragraph is the most cuttable, since the
  Motivation section already gestures at licensing.
- **The bachelor-thesis noise-control finding** ("generated context conferred no
  advantage over per-pixel random noise") is stated four times: twice in the
  Introduction alone (`1:16` in Motivation and `1:61` in Pre-Work), then
  `2-Literature_Review.tex:98` (fullest, correctly the home) and
  `35-Synthetic_Generator_Comparison.tex:4`. The two Introduction statements should
  collapse to one; the §3.5 restatement is defensible as the sub-study's motivation
  but could shrink to a clause plus \Cref.

---

## 4. Numeric consistency

- **The teacher's parameter count.** Prose says "roughly $196\,\mathrm{M}$"
  (`1:12`, `36:9`, `36:46`), but the benchmark table (`tab:embedded-latency`)
  lists MegaDetector v5a at $140.06\,\mathrm{M}$ and the SpeciesNet classifier at
  $52.71\,\mathrm{M}$ — summing to $192.8\,\mathrm{M}$. Either reconcile the prose
  to ~$193\,\mathrm{M}$ or note once what the extra ~3M covers.
- **The teacher-student capacity ratio.** `36-…:68` says "roughly $70$ to $170$
  times in parameter count", with no derivation, while Results (`4:399`), the
  Discussion (`5:22`) and the Conclusion (`5:95`) all say "roughly $72$ to $1$"
  (which matches $196/2.71$). The "to $170$" end of the range is never explained
  anywhere and contradicts the single figure used three times later. Settle on one
  number, or state what the range's upper end counts.

---

## 5. One claim stated above its evidence

`5-Discussion_and_Conclusion.tex:97` — "**Nothing about this detector's accuracy
stands between the work reported here and a shippable model.** A compression pass
does…" (and the Summary's softer twin at `5:40`, "None describes an accuracy
problem for the chosen architecture").

The latency verdict rests on a stated budget ($\leq 30$ ms). The accuracy verdict
rests on no stated acceptance threshold anywhere in the document — no target mAP,
no comparison to what the product currently requires. As written, the sentence
asserts that $0.529$ real-set mAP (with a $19.3\%$ within-group confusion rate) is
shippable accuracy, which the thesis never establishes and the fine-tuned ensemble's
$0.599$ arguably undercuts. This is exactly the `claims.md` §1 case: a "reasoned"
claim phrased as if measured. A calibrated version keeps the rhetorical shape:
"No accuracy threshold was set for this work, and nothing in these measurements
identifies accuracy as the blocking axis. What measurably blocks deployment is
speed…" — or simply scope it: "the one *measured* obstacle to deployment is a
compression problem."

---

## 6. Sentence- and paragraph-level polish

### 6.1 The Discussion opener has no antecedent

`5-Discussion_and_Conclusion.tex:10` — §5.1 opens: "**They** are restated verbatim
and answered one at a time." "They" reaches back across the section heading to the
chapter roadmap's "four research questions". A section must be enterable at its
heading (`construction.md` §1.1). "The four research questions of \Cref{sec:objective}
are restated verbatim and answered one at a time." Separately, consider whether the
verbatim 120-word restatement earns its place at all, given each answer paragraph
already names its question in its `\paragraph` title.

### 6.2 The English abstract's hardest sentence

`preamble/abstract_eng.tex:5` — "Neither property a practitioner can judge before
training, the apparent realism of the generated images or their price, predicts how
useful they prove." Three problems: the apposition interrupts subject and verb
(`construction.md` §1.3), "or" after "Neither…" wants "nor", and the closing "they"
can bind to *properties* or *images*. E.g.: "Two properties can be judged before any
training run, the apparent realism of the generated images and their price. Neither
predicts how useful the images prove." Two more in the same paragraph:

- "evaluated on real and generated **photographs** alike" — a generated image is not
  a photograph; the German twin has the same issue ("generierten Fotografien").
  "on real photographs and generated images alike."
- "its speed, **which no compression step was applied to reduce**" — a strained
  relative clause; "its speed, which no compression step was yet applied to reduce"
  still strains. Simpler: "…but its speed. No compression step was applied in this
  work." Both abstracts' final paragraphs also exceed the 180-word warn threshold
  (193 EN / 212 DE) and each carries five findings; a paragraph break before the
  test-image-overstatement sentence would give the second half its own point.

### 6.3 Dangling "same" in the thesis's very first paragraph

`1-Introduction.tex:5` — "supervision from a much larger fine-tuned teacher model
and direct fine-tuning of **the same lightweight student model**." At first reading
nothing has been called the student yet, so "the same" has no referent, and
"lightweight model" brushes against the terminology file's not-approved list. E.g.:
"…two training strategies are compared: distilling a much larger fine-tuned teacher
model into a lightweight student, and fine-tuning that same student directly on the
wildlife dataset."

### 6.4 The four remaining >55-word sentences

All four are accumulation sentences with marked split points (`construction.md` §2.1):

- `5:32` (69 words) — the $\Delta_{\text{domain}}$ evidence sentence in Question 3.
  Split after "(\Cref{tab:headline-cross-model})".
- `5:63` (61 words) — the six-limits list-sentence in "The generator comparison
  supports ordinal claims only". Six limits in one sentence with an internal "so"
  clause; an itemized list or three sentences would let each limit land.
- `2:152` (61 words) — the FGD mechanism sentence. Split at "applies…" and "and adds…".
- `2:194` (57 words) — the Hoiem four-category sentence; the parenthetical
  "which they treat as including duplicate detections" is the natural extraction.

### 6.5 Remaining body paragraphs over 180 words

`4-Results.tex:350` ("Two things stand out…", 182 — it also absorbs the lion/tiger
cross-check and the segue, three points in one paragraph), `4-Results.tex:469`
("The two synthetic-data results agree", 182), `5:32` (Question 3's second half, 191).
Each has a clean split point where its second point starts.

### 6.6 Two British spellings

`35-…:257` "randomised", `35-…:263` "localised" — against "randomized"/"localized"
everywhere else (`4:133` "correctly localized"). These are the last two `-ise` forms
in the manuscript.

### 6.7 Smaller items

- `1-Introduction.tex:66` (Structure) — chapter names are inconsistently cased:
  "\textit{literature review}" and "\textit{results}" lowercase against
  "\textit{Methods and Implementation}" and "\textit{Discussion and Conclusion}"
  capitalized. Pick one convention.
- `4-Results.tex:351` — "confirming that lion and tiger do not confuse" uses
  "confuse" intransitively; "are not confused with each other".
- "mixed test set" and "real test set" are bolded twice, in `1:23` and again at
  their definitions in `37:23`. The first-mention rule wants one bold; the
  Introduction occurrence could drop `\textbf{}` since the definition lives in §3.7.
- `2-Literature_Review.tex:44` — "which is standard practice for this literature"
  is a trailing self-justification (`scope.md` §8 adjacent); the sentence is
  complete and stronger without it.
- `2-Literature_Review.tex:31` — the norouzzadeh sentence carries two accuracy
  numbers ($>93.8\%$, $96.6\%$) for a claim the work never uses beyond "near-human
  on its own distribution" (`scope.md` §2: numbers attached to unused claims go
  with them). One number, or none, suffices.

---

## 7. Housekeeping (invisible to the reader, worth clearing)

- `4-Results.tex:361-363` — a stale `%` comment claims the Methods setup table
  "still names YOLOv5s as the student"; `34-Data_Augmentation.tex:14` now names
  YOLO26n. The comment misinforms the next editor. Delete it.
- 3 unused figure files (`ax_visio_far`, `ax_visio_near`,
  `embedded_latency_by_runtime`) and 55 orphaned labels (mostly subfigure labels
  and `chapter:introduction`) per `check_manuscript`. Orphaned subfigure labels
  are harmless; the unused AX Visio device photos contradict the 2026-09-21
  figures doc, which says they back the no-bokeh claim in §3.3 — either wire them
  in or remove them.
- `rather than` is down to ~0.27 per 100 words (99 uses) from the measured 0.67 —
  no longer a tic, no action needed. Recorded here so the next pass doesn't
  re-litigate it.

---

## 8. What deserves to be kept exactly as it is

> **Revised 2026-09-26.** The supervisor's feedback (§9) showed that several of the
> sentences praised below belong to exactly the register he reads as AI-generated.
> The revised rule is in §10.3: a punch sentence survives only if it carries a fact
> or a finding; the ones below that merely dramatize ("Each arrived with the use
> case.") are now cut candidates. The calibration hedges and the
> qualification-before-claim discipline stand unchanged.

Named so a later edit pass doesn't sand it off:

- The **not-established discipline**: "Three conclusions are supported by the
  evidence assembled here, and one is not" (§4.1.6), the watchdog verdict's "The
  trigger fired… That is a deviation from the stated protocol and is recorded as
  one" (§4.2.4), and the kinkajou case as the concrete demonstration of a
  measurement's blind spot (§4.1.4). This is the manuscript's credibility engine.
- **Stress-position closers**: "It was never executed." (`35:257`), "None of these
  constraints was constructed for the experiment. Each arrived with the use case."
  (`1:18`), "The mixed default survived as a ranking instrument and failed as a
  description of real-world accuracy" (`4:312`).
- **Qualification-before-claim** is applied consistently in Results and Discussion
  ("On the mixed test set…", "at the one design point tested", "in two of the three
  models"), and epistemic hedges sit next to their claims. Hedge density should not
  be reduced further.
- The **section handoff sentences** ("Diverse images are of no use to a detector
  until they carry boxes, which the generator does not supply." `33:91`) keep the
  compressed chapters continuous exactly as `scope.md` §10 intends.

---

---

# Part II — Supervisor feedback of 2026-09-26: diagnosis and strategy

> **Status, end of 2026-09-26: applied.** The skill amendment landed as
> `construction.md` Part 6 plus `conflicts.md` §7, and `terminology.md` §2.5 now
> names the Raspberry Pi 400 and "the proxy device". Every item in §11 and every
> instance inventoried in §10 was edited in the manuscript, along with Part I's
> §§1–6 (terminology collisions, wrong cross-references, duplicated statements,
> the 193M/70:1 numeric reconciliation, the Conclusion overclaim, and the
> sentence-level polish). `check_manuscript` after the pass: 0 dangling
> references, 0 unresolved citations, 1 sentence over 55 words (a data
> enumeration), budgets held at 36,065 against 36,470. Still open: the three
> unused figure files (§7, needs a figure decision) and the abstracts' page
> fit, which needs a compiled PDF to confirm.

The supervisor read the abstract, parts of the Introduction, parts of the Literature
Review, and the Discussion chapter, and returned two kinds of feedback: specific
content requests (shorten the abstract, define "nano-scale", add a citation, merge
two challenges) and fourteen passages marked **"AI slop"**. He read only a subset of
the manuscript, so the job splits in two: fix what he named, and find every instance
of the same patterns in the chapters he did not read — above all the Results chapter,
which turns out to have the highest density of them.

Status first: five of his smallest wording items were already applied via Overleaf
and committed as `cb07dab` (the two bare "No." verdict openers, "Composition matters,
and it matters non-linearly" → "matters non-linearly", "What the comparison rules out
is" → "The comparison rules out", "Quantization and the accelerated path" →
"Quantization"). Everything below is still open.

## 9. What the "AI slop" flags have in common

The fourteen flagged passages are not random. Classified, they reduce to five
patterns, and knowing the patterns is what makes the rest of the manuscript
searchable.

### 9.1 Meta-discourse: the document talking about itself instead of the subject

Nine of the fourteen flags are sentences whose grammatical subject is the thesis's
own discourse — the answer, the question, the consequence, the item, the gap — rather
than a model, an image, or a class:

- "\Cref{chapter:results} reported the measurements. \Cref{sec:discussion} answers…"
- "They are restated verbatim and answered one at a time."
- "The consequence is specific rather than uniform."
- "This is the one item that deviates from a stated protocol…"
- "Four gaps bound it together."
- "each with what it costs."
- the four Further Work titles.

This is the deepest of the five patterns, and it explains why the Discussion drew the
most flags: that chapter is *structurally* meta (it discusses answers to questions),
so the prose defaults to abstractions acting on abstractions. An LLM writes this
register fluently because it narrates its own reasoning; a human academic writes
"the distilled run was slower to converge", not "the consequence is specific rather
than uniform". **Fix: give the sentence a domain subject.** "The consequence is
specific rather than uniform" becomes "A direction that holds across 17 views of one
run pair is unlikely to be seed noise; a difference of 0.003 carries no information."
(That content already follows the flagged sentence — the flagged sentence is a
fanfare announcing information the next sentence delivers, which is why it can
simply be deleted.)

### 9.2 The rhetorical antithesis: mirrored adjectives, "X, not Y"

- "a **hard** architectural ceiling…, not a **soft** practical guideline"
- "turns \Cref{…}'s **reasoned** argument … into a **measured** one"
- "The consequence is **specific** rather than **uniform**"

The shape is: two abstract adjectives swapped across a pivot. It is Gopen & Swan's
stress position weaponized into a flourish — the sentence performs a symmetry instead
of stating a fact. One of these per chapter reads as style; the manuscript's density
of them reads as a generator's cadence. **Fix: keep the claim, drop the mirror.**
"…capped at 77 tokens by its transformer text branch, an architectural limit"
says everything "hard ceiling, not soft guideline" says.

### 9.3 The commerce metaphor

- "That offer comes at a measured **discount**."
- "Accuracy … has largely been **bought** with scale" (also: citation needed)
- "**costs nothing** measurable"
- "each with what it **costs**"

Buy/sell/cost/price vocabulary applied to accuracy and evidence. It collides with the
places where cost is *literal* (GPU-hours, API billing, the word *budget* as a term
of art), and at this density it is a recognizable LLM register. **Fix: state the
quantity.** "costs nothing measurable" → "changes mAP by at most $0.003$";
"comes at a measured discount" → delete, and open the evidence directly:
"Training a classifier from scratch on generated images required roughly five
times as many of them…"

### 9.4 Colloquial verdicts and dramatics

"No." as a paragraph opener (fixed), "costs nothing", plus — unflagged but the same
voice — "home-field advantage", "tells a gentler story", "has a blunter answer",
"easy to state", "the trigger fired". Punchy spoken-register idiom inside an
impersonal academic document. **Fix: translate to the register or delete.**

### 9.5 Verbless and "and"-paired headings

- "Confidence intervals, computed retroactively."
- "The student's zero-shot floor and the multi-animal subset"
- "The distillation grid and other pairings"
- "The blind rating rubric, and carrying out the watchdog revision."
- "…Ensemble **and the** End-to-End Alternative"

Two failure modes at once: the noun-phrase-plus-participle fragment, and two items
bolted together with "and" (the supervisor's explicit rule: avoid "and" in titles).
The and-titles are usually a symptom that one item is actually two. **Fix: one topic
per heading, a plain noun phrase.**

### 9.6 The calibration point: the same shape can be praised

The supervisor marked "how much diagnostic detail a generator is given matters more
than which generator is chosen" as **"wichtiges finding!"** — and that sentence has
exactly the antithetical punch shape of the passages he flagged. So the shape itself
is not banned. The difference is what the rhetoric is spent on: that sentence carries
the thesis's single most quotable finding; "the consequence is specific rather than
uniform" carries a housekeeping observation. **The rule that falls out: rhetorical
emphasis is a budget, and it may only be spent on findings.** The operational test
for every punchy sentence: *delete it — did the reader lose a fact or a finding?*
Lost a finding → keep. Lost nothing (the neighbors carry the content) → delete for
real. Lost a fact → rewrite the fact plainly.

### 9.7 Why the manuscript sounds like this (and what it means for the skill)

This register did not appear by accident. The `thesis-writing` skill prescribes
stress-position endings, 20–30-word sentences, non-human actors, and open-with-the-
claim paragraphs — Gopen & Swan applied consistently. An LLM applies "consistently"
with machine uniformity: *every* paragraph opens with a counted abstraction ("Two
things stand out", "Three limits apply"), *every* section ends on a punch, every
contrast becomes an antithesis. The individual sentences pass every rule in the
skill; the **uniform cadence** is what a human reader — especially one primed to look
for it — identifies as generated text. Part I §8 of this review praised several
sentences the supervisor's feedback now condemns; §8 has been annotated accordingly.
The skill's own arbiter ("the reader settles every conflict") has spoken through the
closest available proxy for the examiner, so the skill needs an amendment (§12.4).

## 10. The sweep: the same patterns in what he did not read

Measured over the current tree with the greps in §12.5. The supervisor saw the
Discussion; the Results chapter is worse on every metric and is where the next batch
of flags would land.

### 10.1 Counted-abstraction openers (pattern 9.1) — 18 instances

Ch. 4 carries eight: "Two experiments carry this chapter" / "Three findings carry the
rest of the work" (4:5, twice in one paragraph), "Two qualifications govern every
number here" (4:159), "Two smaller observations round the grid out" (4:266), "Two
things limit the damage without repairing it" (4:312), "Two things stand out in the
pair-level counts" (4:350), "Four limits bound what the comparison establishes"
(4:399), "Three limits apply to everything above" (4:457), "Two observations cut
across the sections above and belong to none of them individually" (4:462). Ch. 5
keeps "Three properties bound how far it travels" (5:22) and "Four gaps bound it
together" (5:54, flagged). Ch. 3: "Three constraints carry consequences elsewhere"
(33:66), "Three deviations follow…" (36:50), "Three reasons support that default" /
"Four qualifications accompany every number" (37:25, 37:46). Ch. 2: "Three failure
modes recur across this body of work" (2:100).

Not all eighteen must go — announcing a genuine enumeration is normal academic
signposting. The fix targets (a) the ones where abstractions *act* ("carry", "bound
it together", "round the grid out", "limit the damage", "govern") and (b) the
doubling-up (4:5 uses the pattern twice in five sentences). A neutral form survives:
"Three deviations follow from the teacher being a classifier" is informative;
"Two things limit the damage without repairing it" is theater.

### 10.2 Commerce metaphors outside literal cost — the manuscript's signature tic

Beyond the flagged four: "what the long prompt **buys** is the diagnostic detail"
(4:64), "naming a group at all **costs** between $0.199$ and $0.438$" (4:218),
"fine-tuning the teacher's classifier **costs** Band A accuracy … rather than adding
it" and "**trades away** generic recognition capability" (4:266–267), "The ensemble
**buys** its $0.663$ … at roughly $83$ times the latency" (4:453), "a larger tier
**buys** detail the pipeline discards" (33:76), "The billed figures also **carry a
lesson**" (35:223), "Merging the look-alikes **recovers** at most…" (4:218, mild),
"replacing half … **costs nothing** measurable" (5:30 and its twin in 4:260), "The
teacher's cost is **the more decisive number**" (4:437 — literal cost, dramatic
framing). Keep the literal ones (GPU-hours, dollars, the *budget* term of art);
translate the metaphorical ones into their quantities.

### 10.3 Dramatics and colloquialisms not yet flagged

- "removes that **home-field advantage**" — in a table caption (`tab:band-grid-real`,
  4:233). A sports idiom in a float caption is the single most exposed instance in
  the document.
- "The mixed grid **tells a gentler story**" (4:264)
- "The second half of the question **has a blunter answer**" (5:32)
- "The practical status is **easy to state**" (5:97)
- "Band A is **where that conclusion stops**" (4:262)
- "Read together, the four answers **point one way**" (5:40)
- "**The trigger fired.**" (4:310) — borderline: it refers to a defined watchdog
  trigger, so it is arguably literal; if kept, it should be the only sentence of its
  kind on the page.
- "The question's own wording **carries the reason** this answer is not a verdict"
  (5:37), "\Cref{sec:further_work} **inherits** the rest / the revision" (4:356,
  4:312) — document parts as heirs and couriers.
- "is **the point of** the structure" (33:27), "None of these constraints was
  constructed for the experiment. **Each arrived with the use case.**" (1:18) —
  the second sentence adds rhetoric to a fact the first already states.

### 10.4 Headings with the §9.5 shapes

- "The Detector-Plus-Classifier Ensemble **and the** End-to-End Alternative" (2:51,
  flagged) → "The Detector-Plus-Classifier Ensemble" (the alternative is the
  subsection's conclusion, not its topic).
- "The Qualitative Rubric: **Designed, Not Executed**" (4:136) → "Execution Status of
  the Qualitative Rubric" or fold into §4.1.5's opening sentence.
- "Sourcing Execution **and the** LILA BC Rejection" (31:67) → "Sourcing Execution"
  (the rejection is part of the execution).
- "Training cost, **reported but not comparable**" (4:464, paragraph title) → "Training
  cost".
- Further Work items 2, 3, 4, 5 (5:78–81, all flagged): "Retroactive confidence
  intervals"; item 3 bundles two follow-ons and should either split or be titled by
  what unites them ("Re-scoring the archived predictions"); "A wider distillation
  design space"; item 5 likewise bundles two — "The blind rating rubric" with the
  watchdog revision moved into item text or its own item.
- Discussion answer titles "Question 2: Which Generator, **and Whether** Realism or
  Price Predicts It" and "Question 3: …, **and Whether** Mixed Accuracy Holds Up"
  (5:24, 5:29) — the questions genuinely have two halves, but the titles can carry
  just the subject: "Question 2: Generator Choice", "Question 3: Training-Data
  Composition".
- Limitations paragraph title "Error types are not separated, **and** training cost
  is not comparable" (5:68) — two unrelated items under one title; split.

## 11. The rest of the feedback (non-slop items), with proposed fixes

1. **Abstract to half a page.** Current: ~600 words in three paragraphs; the final
   paragraph alone is 193 words and over the house warn threshold. Target ~250 words:
   - P1 (2–3 sentences): task, constraint, and what the work does. The supervisor
     explicitly says the abstract need not re-motivate at the Introduction's depth —
     the whole fine-markings/long-tail exposition can compress to one clause.
   - P2 (2–3 sentences): dataset assembled and curated automatically; generators
     compared before use; distillation vs. direct training compared under matched
     conditions; speed measured on comparable hardware.
   - P3 (4–5 sentences): one sentence per research-question answer, keeping the
     prompt-detail finding (the one he starred) and the synthetic-test-overstatement
     disclosure. The German twin must shrink in lockstep.
2. **Widen the Introduction's framing** ("am Beispiel species detection"): the first
   sentence should present the general question — can a detector small enough for
   embedded hardware be trained when training data is scarce — with wildlife species
   detection as the test case, then instantiate. E.g. "This work investigates how an
   object detector small enough for embedded hardware can be trained when real
   training images are scarce, using wildlife species detection on a consumer
   binocular as the test case." This also matches the abstract's closing
   generalization, which already claims transfer "wherever a recognition model must
   be trained for a fine-grained domain".
3. **"Accuracy … bought with scale" needs a citation** (and is also a §9.3 metaphor).
   Replace with a cited factual form, e.g. "Detection accuracy has risen chiefly with
   model capacity \cite{zouObjectDetection202023}" — verify the survey supports the
   claim before attaching it; the capacity evidence already cited in §2.1.2
   (\cite{yuPPPicoDetBetterRealTime2021,geYOLOXExceedingYOLO2021}) is the in-house
   fallback.
4. **The complicated sentence** (1:14): "The class universe is fine-grained, with
   several of the $225$ classes separable only by markings a detector has to learn to
   attend to." Split: "The class universe is fine-grained. Many of the $225$ classes
   differ only in fine markings, and the detector has to learn exactly those markings
   \cite{weiFineGrainedImageAnalysis2022}."
5. **"nano-scale" is used before it means anything** (first at 1:23, sized only in
   2:46). Gloss at first use: "a nano-scale detector — the smallest members of a
   detector family, at roughly one to three million parameters —". Then bare
   "nano-scale" everywhere after, per the one-gloss rule.
6. **Name the two lineages in the topic sentence** (2:12): "Modern object detection
   descends from two lineages that emerged within a few years of each other:
   two-stage detectors, which first propose candidate regions and then classify
   them, and single-stage detectors, which predict boxes in one pass." The two
   existing paragraphs then elaborate.
7. **Merge challenges one and two** (2:23–25). The supervisor is right that they are
   one problem seen twice: fine-grained categorization *is* small inter-class
   distance, and the zebra/lynx/gazelle clusters are its instances in this taxonomy.
   Merge into one paragraph (property → instances → consequence for a
   general-purpose detector) and renumber four challenges to three.
8. **"provably biased" — verified, keep.** Bińkowski et al. do carry the proof: their
   footnote 10 argues the estimator cannot be unbiased (it is non-negative even when
   the true FID is zero), Appendix D.3 shows **no unbiased estimator of FID exists**,
   and Appendix D.1 constructs the analytic wrong-ordering example the sentence
   describes. If "provably" still reads as too strong, the drop-in replacement is
   "…is biased, and no unbiased estimator of it exists \cite{binkowskiDemystifyingMMDGANs2021}"
   — which is *more* precise, not less.
9. **"They are restated verbatim…" (5:10)** — supervisor confirms Part I §6.1: write
   "The research questions are restated verbatim and answered one at a time." Also
   rewrite the chapter roadmap above it (5:5), which he flagged as slop for its
   symmetric chapter-as-agent parade: "This chapter answers the four research
   questions of \Cref{sec:objective} from the evidence of \Cref{chapter:results},
   states the limitations that bound each answer, and derives the further work they
   motivate."

## 12. Strategy

### 12.1 Order of work

1. **Apply the remaining named items** (§11 and the flagged passages still open:
   "measured discount", "hard ceiling / soft guideline", "costs nothing" ×2, "each
   with what it costs", "the consequence is specific rather than uniform", "the one
   item", "Four gaps bound it together", "turns … reasoned … into a measured one",
   the four Further Work titles, the ensemble heading). These are reader-verified;
   there is no judgment call left.
2. **Sweep the chapters he did not read with the §9 patterns**, worst-density first:
   Results (§10's lists), then the Ch. 3 files, then the remaining Ch. 2 sections.
   The goal is that his *next* read of any section produces no new flags of a kind
   he has already given. The Results chapter's watchdog-verdict and look-alike
   sections, and the `home-field advantage` caption, are the priority instances.
3. **Shrink the abstract** (both languages) per §11.1.
4. **Amend the skill** (§12.4) so regenerated or newly drafted prose does not
   reintroduce the register.

### 12.2 The two tests to apply at every candidate sentence

- **The deletion test** (from §9.6): delete the sentence. Reader lost a finding →
  keep it, punch and all. Reader lost a fact → restate the fact plainly. Reader lost
  nothing → the deletion stands.
- **The subject test** (from §9.1): if the grammatical subject is a discourse noun
  (answer, question, consequence, item, gap, limit, qualification, observation,
  story, lesson, verdict), rewrite with the domain as subject — the model, the run,
  the class, the number — or attach the discourse noun to its content in one
  sentence instead of two.

### 12.3 What must not be over-corrected

The feedback is about register, not about the manuscript's structure or honesty.
Keep unchanged: the qualification-before-claim ordering, the epistemic hedges, the
scope-condition discipline, the one-home-per-caveat structure, the roadmap/handoff
sentences (in plain form), and the starred finding's phrasing. And keep *some* punch:
the fix for uniform cadence is variance, not a new uniformity of flatness. One
emphatic sentence per section, spent on a finding, is the calibration §9.6 supports.

### 12.4 Proposed skill amendment (for `construction.md`, new §; and a note in
`conflicts.md`)

> **Rhetoric is budgeted.** The stress-position and short-sentence rules exist to
> make facts land, not to decorate them. A sentence that performs — an antithesis of
> abstract adjectives, a metaphor (commercial vocabulary above all), an aphoristic
> fragment, a counted abstraction acting ("Four gaps bound it together") — must pass
> the deletion test: if removing it loses no fact or finding, remove it. Spend at
> most one emphatic sentence per section, and only on a finding. Sentences about the
> document's own discourse (answers, consequences, items, limits) take a domain
> subject instead wherever one exists. Headings name one topic, carry no "and", no
> verbless participle fragment, and no antithesis. Reader evidence: the supervisor
> feedback of 2026-09-26, which flagged fourteen such passages as recognizably
> machine-written while starring one — a punch sentence carrying the central
> finding — as important.

Part I §8's blanket protection of "stress-position closers" is revised to match.

### 12.5 Detection greps (re-runnable; candidates for `check_manuscript.py` warnings)

```bash
# counted abstractions acting
grep -rnE "(Two|Three|Four|Five|Six) [a-z-]+ (stand out|carry|carries|bound|govern|round the|limit the|cut across)" chapters
# commerce metaphors (then filter literal GPU/$ contexts by eye)
grep -rnE "bought with|buys |costs nothing|discount|trades away|carry a lesson|price of" chapters preamble
# discourse nouns as sentence subjects
grep -rnE "(The|This|That|Each|Every) (answer|question|consequence|item|gap|verdict|story|lesson|observation)s? [a-z]+" chapters
# antithesis flourish
grep -rnE ", not (a|an|the) [a-z]+ [a-z]+\." chapters
# and-titles / fragment titles
grep -rnE "\\\\(sub)?section\{[^}]*( and |, [A-Z][a-z]+ed\}|: [A-Z][a-z]+, Not)" chapters
# colloquial register
grep -rn "home-field|gentler story|blunter|easy to state|point one way|off the shelf" chapters
```

The counts these produced on 2026-09-26: 18 counted-abstraction openers, ~12
metaphorical commerce usages, 8 "X, not Y" antitheses, 8 heading hits, 7 colloquial
hits. A future `check_manuscript` warning class ("rhetoric-pattern hit") would keep
the register from regrowing; the pattern list above is the seed.

## 13. One meta-lesson for how this manuscript gets written

The manuscript's prose was largely LLM-drafted under a skill that encodes punchy,
stress-position-driven construction, and the supervisor identified the result's
*cadence* — not its facts, structure, or honesty — as machine-written. The durable
fix is not this one editing pass but the §12.4 rule, because any future LLM drafting
under the unamended skill will regenerate the same register. When the flagged
passages are rewritten, the word budget of `scope.md` §9 still applies: most fixes
above are length-neutral or negative (deletions), and the abstract shrinks, so no
budget conflict is expected.
