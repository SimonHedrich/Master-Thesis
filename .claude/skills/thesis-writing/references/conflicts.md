# Conflicts — where the sources disagree, and who wins

Three writing systems feed this skill, and they contradict each other. The contradictions are
written down here as contradictions, each with a verdict and the reason behind it, because a
rulebook that quietly drops the inconvenient half leaves the next reader unable to tell a
principled exception from an oversight.

Every verdict below resolves the same way: by asking who this document is written for. That is
the only question that can settle a style conflict, and it is the reason the reader is named
first in `SKILL.md`.

**The reader, restated.** One examiner who knows machine learning but not this project,
reading roughly eighty pages under time pressure, who wants the decision, the reason, the
evidence, and the limit. Not a mechanic who must not misread a procedure. Not a consumer
opening a dialog box who needs reassurance.

---

## 1. Named actor vs. impersonal register

**ASD-STE100 and Microsoft both require an active voice with a named actor.** STE because a
mechanic must know who does what; Microsoft because "you can access your files" is weaker than
"access your files from any device".

**§7 of the style contract requires the opposite:** strictly impersonal academic register,
never "I", choices phrased as "X was chosen because Y". The manuscript honors this completely —
zero first-person pronouns in 30,004 words.

**Verdict: the impersonal register wins, and the rule survives in a stronger form.**

The human actor in this thesis is constant and uninteresting. Naming it in every sentence would
add one bit of information to the document in total. But almost every passive in a methods
section has a *non-human* actor available, and that actor is genuinely informative:

> **Weak.** The images were filtered and the remaining boxes were scored.
>
> **Better.** MegaDetector removes images without an animal. A composite quality score then
> ranks the boxes that remain.

**The test:** can you name a non-human agent — a script, a model, a filter, a stage, a dataset,
a metric? Then name it. Passive is permitted only where the actor is the author, or where the
actor genuinely does not matter.

**What the verdict costs:** slightly more mechanical prose in the few places where the author
really is the actor and the passive was hiding a decision rather than a component. Watch for
"it was decided that" and "a choice was made to" — those are the passives worth rewriting into
"this thesis" constructions, not into named components.

---

## 2. Hedging

**The anti-slop instinct deletes hedges.** In a README, "this may improve performance in some
cases" is cowardice dressed as precision.

**Scientific writing requires them.** A claim stated more strongly than its evidence supports
is not confident prose; it is a false statement.

**Verdict: two kinds of hedge, and only one gets cut.**

**Epistemic hedges** calibrate a claim against its evidence. They are mandatory, and they sit
next to the claim they qualify, not at the end of the paragraph. The manuscript already does
this well:

> "absolute accuracy is provisional even though the relative ranking is not obviously affected"
>
> "[this thesis's] own extrapolation from that general capacity-accuracy relationship, not a
> literature-established figure, and it is treated as such throughout"
>
> "the comparison establishes a ranking for data-rich classes and establishes that the current
> instrument cannot rank generators for data-poor ones"

**Padding hedges** soften tone and carry no information: *arguably*, *somewhat*, *to some
extent*, *rather* (as an intensifier), *it is important to note that*, *it is worth stating
that*. Cut them.

**The test:** would a reader draw a different conclusion if the hedge were removed? Keep it.
Does only the tone change? Cut it.

**This conflict has a warning attached.** Measured hedge density in the manuscript is **0.21
per 100 words**, with **zero** stacked hedges. That is already right. This rule exists to
*protect* that calibration under editing pressure, not to reduce it. A revision pass that
drives hedge density toward zero has damaged the thesis, and §7 of
`thesis/writing/2026-09-17_manuscript-baseline.md` records the number so that damage is
visible. Of every rule in this skill, this is the one most likely to be over-applied.

---

## 3. Dashes and semicolons

**The source blog post calls the em dash the single loudest generated-text marker there is,**
and bans it outright. An earlier version of this rule argued the ban does not transfer to a
compiled PDF, since typeset `--` renders as an en dash rather than the ASCII em dash that marks
generated Markdown, and permitted it capped at one pair per paragraph.

**Verdict, revised: avoid the parenthetical `--` dash and the semicolon in prose. Split the
sentence instead.** The author overrode the earlier verdict directly: both marks make a sentence
harder to read than two sentences would, regardless of typeset rendering, and the fix is always
available — the dash or semicolon already marks the exact point where the sentence splits.

> **Weak.** MegaDetector removes images without an animal -- a composite quality score then
> ranks the boxes that remain.
>
> **Better.** MegaDetector removes images without an animal. A composite quality score then
> ranks the boxes that remain.

> **Weak.** The filter has four stages; each one runs independently of the others.
>
> **Better.** The filter has four stages. Each one runs independently of the others.

**Exception: numeric and version ranges are not parenthetical dashes and are untouched** —
`12--15`, `$200$--$300$`, `YOLOv2--YOLOv7`. Also untouched: semicolons inside code (TikZ/pgfplots
statement terminators, `\node[...] {...};`) and inside `.bib` entries.

This is the same defect §1.2 of `construction.md` already names — a sentence carrying two ideas,
one demoted into an aside instead of given its own sentence — just resolved by punctuation
before it reaches the structural fix.

---

## 4. Tense

**STE bans the present perfect** ("we received the reports", never "we have received").

**Academic convention requires it** for describing the state of a field.

**Verdict: STE loses.** The three-tense convention applies:

| tense | use | example |
|---|---|---|
| present perfect | the state of the field, work by others with continuing relevance | "prior work has shown that synthetic pretraining transfers unevenly" |
| simple past | a specific study, and this thesis's own completed actions | "He et al. found…", "the filter removed 12,000 images" |
| present | established facts, and what this document does | "a single-stage detector evaluates the image once", "this chapter presents…" |

---

## 5. Contractions, warmth, and second person

**Microsoft requires contractions** (`it's`, `you'll`, `we're`), asks for a voice that is "warm
and relaxed", "less head, more heart", and writes in the second person.

**Verdict: all three rejected, no exceptions.** This is the point where the Microsoft guide's
reader and this thesis's reader diverge completely. Microsoft writes consumer product UI for a
global audience, where a dialog box must reassure. An examiner reading eighty pages wants the
fact. Neither guide is wrong; they have different readers.

What survives from Microsoft is in `construction.md` Part 3 — the verb rules, the
name-the-target rule, and the bias-free vocabulary. Those are register-independent. A marketing
adjective is slop in anyone's house.

---

## 6. "This thesis" as a standing agent

Not a conflict between sources, but a consequence of the §7 verdict in §1 above, and it needs a
limit. The impersonal register produces "this thesis" as an agent noun — 59 uses so far, doing
the work that "I" would do elsewhere.

**Verdict: keep it, but budget it.** Prefer naming the component that actually acted
(`the quality filter`, `the Band A rule`, `\Cref{tab:generator-ranking}`). Reserve "this
thesis" for genuine document-level claims: what it argues, adopts, establishes, or declines to
conclude. When three paragraphs in a row open with "this thesis", two of them had a better
subject available.
