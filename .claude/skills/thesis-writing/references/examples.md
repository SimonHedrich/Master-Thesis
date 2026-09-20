# Examples — before and after, from this manuscript

Most examples below use **real text from this thesis** rather than invented bad writing. That is
deliberate: a catalogue of generic slop teaches a model to avoid generic slop, and this
manuscript does not have any. What it has is long sentences, a dead acronym mechanism, and four
words doing two jobs each.

**Every block is labelled, and the labels are load-bearing:**

| label | what it is |
|---|---|
| **BEFORE** | verbatim text from the manuscript, to be changed |
| **AFTER** | the proposed rewrite |
| **THIS IS THE STANDARD** | verbatim manuscript text that is already right — copy it |
| **CONSTRUCTED** | invented for illustration, never quoted from the thesis |

> **Warning to any reader, human or agent.** A **BEFORE** or **CONSTRUCTED** block is an example
> of what to change. It is not guidance and must never be imitated. The source blog post makes
> this point about its own reference files: a linter cannot tell quotation from confession, and
> neither can a language model skimming for patterns.

---

## 1. The 150-word sentence

`2-Literature_Review.tex:90`. The longest sentence in the manuscript, and a fair one to fix
because it is not badly written — it is three good ideas sharing one set of punctuation.

> **BEFORE**
>
> Across this body of work, three recurring failure modes motivate treating synthetic imagery as
> a supplement whose reliability must be actively verified rather than assumed: a persistent
> \textit{texture and lighting domain shift} between rendered or generated pixels and real
> sensor imagery, even after substantial fine-tuning effort (Azizi et al.); a \textit{label- and
> context-alignment} gap in which a generator's default visual associations for a class or scene
> diverge from the target task's actual distribution unless explicitly corrected (He et al.'s
> from-scratch results; Azizi et al.'s off-the-shelf failure case); and, at the level of what
> training on generative output can compound over time rather than within a single generation,
> \textit{model collapse} \cite{shumailovAIModelsCollapse2024} -- the theoretically and
> empirically demonstrated degeneration of a generative model's learned distribution when it is
> recursively retrained, indiscriminately, on data sampled from itself or its predecessors,
> first losing distributional tails and eventually collapsing toward a low-variance, degenerate
> output distribution.

**AFTER**

> Three recurring failure modes run through this body of work. Each is a reason to treat
> synthetic imagery as a supplement whose reliability is verified rather than assumed.
>
> The first is a persistent \textbf{texture and lighting domain shift} between generated pixels
> and real sensor imagery, which survives substantial fine-tuning effort (Azizi et al.). The
> second is a \textbf{label- and context-alignment} gap: a generator's default visual
> associations for a class diverge from the target task's distribution unless they are corrected
> explicitly (He et al.'s from-scratch results; Azizi et al.'s off-the-shelf failure case). The
> third operates across generations rather than within one. \textbf{Model collapse}
> \cite{shumailovAIModelsCollapse2024} is the degeneration of a generative model's learned
> distribution under recursive retraining on data sampled from itself. The distributional tails
> go first, and the distribution then collapses toward low-variance, degenerate output.

One sentence becomes seven, none over 30 words. What changed and why:

- Each failure mode now gets its own **stress position** (`construction.md` §1.2). In the
  original, three findings compete for the end of one sentence and none of them lands.
- The list is **numbered in prose** ("the first", "the second", "the third"), so the reader
  never has to hold an open colon across 120 words.
- "theoretically and empirically demonstrated" is dropped. The citation makes that claim, and
  the adjectives were doing the citation's job less precisely.
- "indiscriminately" is dropped; "recursive retraining on data sampled from itself" already
  says it.
- `\textit` becomes `\textbf`, because these are **first definitions of terms**, which §7 of the
  style contract marks with bold. `\textit` is for tool and dataset names.
- The closing sentence of the original paragraph ("None of these failure modes is a purely
  theoretical concern…") now has room to be the paragraph's landing, per `construction.md` §2.2.

---

## 2. The expletive opener

`2-Literature_Review.tex:30`, and two more like it at `31-*.tex:114` and `32-*.tex:185`.

> **BEFORE**
>
> It is worth stating plainly that none of the family's releases from YOLOv5 through YOLO26 …

**AFTER**

> None of the family's releases from YOLOv5 through YOLO26 …

Deleting the opener strengthens the sentence. "It is worth stating plainly" was the author
asking the reader's permission to state something plainly — and the sentence that follows had
already earned it. Same fix for "It is worth noting that".

---

## 3. Passive with an actor available

The manuscript is good at this, so both blocks are constructed. The pattern is worth showing
before §3.6 and §3.7 get drafted, where it is most likely to appear.

> **CONSTRUCTED, DO NOT IMITATE**
>
> The images were filtered and the remaining boxes were scored before the splits were assembled.

**AFTER**

> MegaDetector removes images without an animal. A composite quality score then ranks the boxes
> that remain, and split assembly draws from that ranking.

Three passives become three named non-human actors, and not one first-person pronoun appears.
This is the verdict in `conflicts.md` §1: §7's impersonal register is kept, and the
named-actor rule from STE and Microsoft survives by attaching to components instead of people.

The passive that *should* stay: "A single, uniform filtering policy across all sources was
rejected in favor of a source-aware, cost-staged design." The actor there is the author, and
naming the author is what §7 forbids.

---

## 4. Hedging: what to keep, what to cut, and a rule pulling the other way

The manuscript's real sentence, verbatim from `4-Results.tex` §4.1. Unlike every other section
here, this one starts with the **exemplar**, because the thesis already gets hedging right and
the failure mode to guard against is a careless edit that breaks it.

> **THIS IS THE STANDARD** (real text, keep this)
>
> The annotations underlying every cell are automatic rather than human-reviewed
> (\Cref{sec:generator_comparison_training}), so absolute accuracy is provisional even though
> the relative ranking is not obviously affected, every cell having been annotated identically.

"provisional" and "not obviously affected" are **epistemic hedges**: each calibrates a claim
against its evidence, and removing either would make the sentence assert more than the
experiment supports. They are mandatory, and they sit next to the claims they qualify.

Now a **constructed** degradation — this one is invented, not quoted, to show what a careless
anti-slop pass does in each direction:

> **CONSTRUCTED, DO NOT IMITATE — padded**
>
> It should be noted that the annotations underlying every cell are arguably somewhat automatic
> rather than human-reviewed, so absolute accuracy is provisional even though the relative
> ranking is not obviously affected.

> **CONSTRUCTED, DO NOT IMITATE — over-corrected**
>
> The annotations underlying every cell are automatic. Absolute accuracy is therefore unreliable
> and the ranking cannot be trusted.

The first adds "It should be noted that", "arguably" and "somewhat" — padding hedges, which
change tone and nothing else, and which the real sentence does not contain. The second deletes
the calibration along with the padding and ends up stating something the author never measured:
the ranking *is* trustworthy, and the sentence says why. Measured hedge density across the
manuscript is 0.21 per 100 words with zero stacked hedges. That number should not move.

**And one rule does pull against this sentence.** At 44 words it exceeds the 40-word warning in
`construction.md` §2.1, and the trailing absolute clause sits after the stress position. The
length rule applies; the hedges are not what to cut:

> **AFTER** (length fixed, calibration untouched)
>
> The annotations underlying every cell are automatic rather than human-reviewed
> (\Cref{sec:generator_comparison_training}). Absolute accuracy is therefore provisional,
> though the relative ranking is not obviously affected: every cell was annotated identically.

Two sentences, 20 and 24 words, every hedge intact, and the reason for the qualification now
closes the sentence instead of trailing off it.

---

## 5. The `rather than` tic

202 uses, roughly one sentence in four.

> **BEFORE** (`32-Data_Quality_and_Curation.tex:116`)
>
> Treating a low pass rate under the match-level filter as evidence of a classifier coverage gap
> rather than a labeling problem required a fallback rule.

**AFTER**

> A low pass rate under the match-level filter can mean two things: a gap in the classifier's
> coverage, or a labeling problem. Resolving that ambiguity required a fallback rule.

The contrast was the point of the sentence and it was compressed into two words in the middle.
Given its own clause, it also fixes the subject–verb distance — the original puts 18 words
between "Treating" and "required".

`rather than` is correct English and often exactly right here. The check is whether the contrast
deserves its own sentence, not whether the phrase is allowed.

---

## 6. Directional reference, and a British spelling

`4-Results.tex:153`. Both defects in one passage — and the passage is otherwise an exemplar,
which is why it is quoted at length.

> **BEFORE**
>
> One episode during this study is worth reporting because it bears on how the table above
> should be read. […] The final figure of $0.105 \pm 0.036$ has a spread roughly six times wider
> than any other API cell's, which after three seeds looks like a genuine property of that cell
> rather than an artefact of too few …

**AFTER**

> One episode during this study bears on how \Cref{tab:generator-ranking} should be read. […]
> The final figure of $0.105 \pm 0.036$ has a spread roughly six times wider than any other API
> cell's, which after three seeds looks like a genuine property of that cell rather than an
> artifact of too few …

Three changes: "the table above" names its target instead of its position, because floats move
(`construction.md` §3.3); "is worth reporting because it" goes, per §2 above; `artefact` becomes
`artifact`, per the American-spelling decision.

**What this passage gets right, and should be copied.** It reports that a conclusion was written
up at one seed, reversed at two, and reversed again at three — an episode that made the author
look wrong and the result look fragile, reported anyway because it changes how the table should
be read. That is `claims.md` §1 in practice.

---

## 7. A comparative claim with no axis

Constructed, for §4.3 and Chapter 5 — unwritten, and where this failure would appear.

> **CONSTRUCTED, DO NOT IMITATE**
>
> The distilled student outperforms the directly fine-tuned baseline.

**AFTER**

> On the mixed test set the distilled student model reaches a mAP of X, against Y for the
> directly fine-tuned baseline (\Cref{tab:kd-comparison}). On the real-only breakout the gap
> narrows to Z.

The first version cannot be checked, cannot be compared to a public benchmark, and does not say
which classes it holds for. The second satisfies `claims.md` §4 and the evaluation invariant in
§5 — the mixed-set headline and the real-only breakout appear together, every time.

---

## 8. A terminology collision in one sentence

Constructed, and the hardest kind to see, because each word is correct on its own.

> **CONSTRUCTED, DO NOT IMITATE**
>
> The distilled model was trained on cells from the top tier and evaluated across regimes.

Three collisions, and the reader cannot recover any of them from context: *distilled* (knowledge
distillation, or a step-distilled diffusion generator?), *tier* (API tier, taxonomic rank, or
assessment level?), *regimes* (prompt regimes, or training bands?).

**AFTER**

> The step-distilled generator supplied the cells in the top API tier, and each cell was
> evaluated under all three prompt regimes.

Every collision resolved by `terminology.md` §1. Note that the fix required knowing which sense
was meant — which is the argument for fixing the vocabulary before the chapters that use it get
written, not after.
