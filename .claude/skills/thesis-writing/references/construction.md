# Construction — how to build a sentence and a paragraph

These are the positive rules. They say how to build the sentence, not which words to avoid.

That distinction is the whole reason this file exists. The thesis already had a style
contract (`thesis/bachelor-thesis-analysis.md` §7) and it is entirely conventions — numbers in
math mode, `\textit` on first mention, never "I". Every rule in it is correct and none of them
touches how a sentence is put together. Measurement confirmed the gap: across 30,004 drafted
words the classic slop vocabulary returns **3 hits**, while **39% of sentences run over 40
words**. There was nothing left to ban and a great deal left to build.

Each rule names its source, because a rule that explains itself survives contact with an
exception and a rule that does not becomes superstition.

---

## Part 1 — Reader expectations (Gopen & Swan)

From George Gopen and Judith Swan, "The Science of Scientific Writing", *American Scientist*
78 (1990). Written for scientists, about scientific prose, three decades before any of this.
Its premise: readers have fixed structural expectations about where information appears in a
sentence, and prose is hard to read when the writer puts information somewhere else. The fix
is never "simplify" — it is "move it to the position the reader is already looking at".

### 1.1 Topic position — start with what the reader already has

The first few words of a sentence tell the reader what the sentence is *about* and link it
backwards. Put old, established material there; save the new material for later.

> **Weak.** A per-source reliability profile determines which of the four filter stages run on
> a given image.
>
> **Better.** Each filter stage runs only where the source's reliability profile calls for it.

The second version opens on "each filter stage", which the previous sentence just introduced.
The reader arrives already oriented.

### 1.2 Stress position — end on the point

Readers give natural emphasis to whatever closes a sentence. Put the new, important material
there, and nothing after it.

This is the mechanism behind Chapter 2's density, and it is worth being precise about why. A
50-word sentence is not hard to read because 50 is a large number. It is hard because it
contains four or five clauses, each of which ends in a stress position, so the reader is asked
to emphasize five things and ends up emphasizing none. Splitting the sentence does not just
shorten it; it gives each point a stress position of its own.

> **Weak.** The incumbent generator ranks third of twelve, which is a reasonable outcome given
> that it was selected on visual realism rather than downstream utility, although it is beaten
> by roughly 28% by a cheaper model under an identical prompt regime.
>
> **Better.** The incumbent generator ranks third of twelve. A cheaper model beats it by
> roughly 28% under an identical prompt regime. The original selection criterion was visual
> realism, not downstream utility.

### 1.3 Subject–verb proximity — nothing long in between

Readers hold a grammatical subject in working memory until its verb arrives. A long
interruption makes them do unpaid work.

> **Weak.** The quality score, which rewards a bounding box occupying between 5% and 60% of the
> frame and penalizes boxes clipped by the image boundary, guides validation-set sampling.
>
> **Better.** The quality score guides validation-set sampling. It rewards a bounding box
> occupying between 5% and 60% of the frame and penalizes boxes clipped by the image boundary.

### 1.4 Action in the verb — not in a noun

If something happens in the sentence, a verb should be doing it.

> **Weak.** The removal of images without an animal is performed by MegaDetector.
>
> **Better.** MegaDetector removes images without an animal.

**Exception, and it is a large one.** Nominalizations that are terms of art — *detection*,
*distillation*, *classification*, *augmentation*, *quantization*, *inference*, *evaluation* —
are the correct names of the things they name. Do not "fix" them. The rule targets a verb that
has been turned into a noun and then propped up by a colorless one (`perform`, `conduct`,
`carry out`, `undertake`, `provide`, `achieve`). The manuscript currently has **zero** of these,
which is the standard to hold.

### 1.5 One unit of discourse, one point

A sentence makes one claim; a paragraph makes one point. If you cannot say what a paragraph is
for in a single clause, it is two paragraphs.

---

## Part 2 — Length and shape (ASD-STE100, relaxed)

From ASD-STE100 Simplified Technical English, the controlled language the European aerospace
industry wrote in 1986 for maintenance manuals. Its caps are relaxed here: a thesis carries
qualifications that a hydraulic-line procedure does not, and a 25-word ceiling would read as
choppy to an examiner.

### 2.1 Sentence length

| | target | warn | rewrite |
|---|---|---|---|
| words per sentence | ≤ 30 | > 40 | > 55 |

These are calibrated against this manuscript, not invented. `35-Synthetic_Generator_Comparison`
(median 28) and `4-Results` (median 26) already meet the target; `2-Literature_Review`
(median 50, 38.6% of sentences over 55 words) does not. **The targets describe the author's
own most recent writing.** They are a floor already reached, not a new register.

A sentence over 55 words is almost always a paragraph that lost its punctuation. Clause-length
measurement shows why it is fixable: only 2.5% of sentences contain a single clause over 55
words, against 18.3% of sentences over 55 words overall. The long sentences are long by
*accumulation*, and the split points are already marked with colons, semicolons and dashes —
turn each one into a full stop rather than keeping the semicolon or dash as the joiner; see
`conflicts.md` §3.

### 2.2 Paragraph length

| | target | warn |
|---|---|---|
| words per paragraph | ≤ 120 | > 180 |

**Budget paragraphs in words, not sentences.** STE caps a paragraph at six sentences; this
manuscript averages 3.5 sentences per paragraph and has a 452-word one. Every paragraph here
passes the sentence cap, including the ones that need splitting.

Shape: the first sentence states the point, the last sentence lands the consequence. A
paragraph that opens with throat-clearing ("It is worth noting that…", "In this section, we
will…") has wasted its strongest position.

### 2.3 Qualification before claim

STE requires the condition before the command, so nobody acts on the first half of a sentence
before reaching the second. Transposed to a thesis: **the reader must never be able to believe
a claim and then have it withdrawn.**

> **Weak.** The incumbent generator is beaten by 28%, although only on data-rich classes and
> under a single evaluation axis.
>
> **Better.** On data-rich classes, and under the single evaluation axis that was executed, a
> cheaper generator beats the incumbent by 28%.

This rule matters more here than anywhere else in the file, because the scope conditions in
this thesis are load-bearing: a result that holds only on Band D classes, or only on the real
test set, is a different result. See `claims.md`.

### 2.4 One term, one meaning

STE's dictionary gives each approved word exactly one meaning: "check" is permitted as a noun
and forbidden as a verb, because a mechanic under time pressure should never have to choose
between "inspect it", "test it" and "stop it". The same discipline, applied to this project's
vocabulary, lives in `terminology.md`. Four words in this manuscript currently carry two
meanings each.

---

## Part 3 — Verbs and references (Microsoft Writing Style Guide)

The public style guide behind one of the largest documentation estates in software. Most of it
is written for consumer product UI and does not survive this register — see `conflicts.md` §3
and §5 — but three rules transfer intact.

### 3.1 "If you mean the same thing, use the same word"

This is STE's one-name rule, reached from a completely different argument by a completely
different industry. A rule found twice from two directions is a real one. See `terminology.md`.

### 3.2 Kill expletive openers

"There are three regimes that…" and "It is important to note that…" burn the verb slot on
nothing and push the real subject backwards.

> **Weak.** There are three failure modes that motivate treating synthetic imagery as a
> supplement.
>
> **Better.** Three failure modes motivate treating synthetic imagery as a supplement.

> **Weak.** It is worth stating plainly that the real images carry no hand-drawn ground truth.
>
> **Better.** The real images carry no hand-drawn ground truth.

The manuscript has **zero** "There is/are" openers and **three** instances of "It is
worth …" (`2-Literature_Review.tex:30`, `31-*.tex:114`, `32-*.tex:185`). Note what the fix
does in the second example: deleting the opener does not weaken the statement, it strengthens
it. "Worth stating plainly" was the author asking permission to state something plainly.

### 3.3 Name the target, never its position

Microsoft forbids "the button above" because a screen reader moves through a page in a line
and "above" means nothing to it. The same ban follows here from a different mechanism: LaTeX
floats move across page breaks, so "the table above" may end up below, overleaf, or two pages
back. The fix is identical — name the target.

> **Weak.** …it bears on how the table above should be read.
>
> **Better.** …it bears on how `\Cref{tab:generator-ranking}` should be read.

One live instance: `4-Results.tex:153`. Phrases like "the filter described above" that point at
a *section* rather than a float are acceptable, though `\Cref` is better.

### 3.4 Bias-free vocabulary

The substitutions that reach machine-learning writing: *primary/subordinate* not
*master/slave*, *allowlist/denylist* not *whitelist/blacklist*, *stops responding* not *hangs*,
*placeholder* not *dummy*. Person-first phrasing where people are described at all. None of
this reads as generated text, and all of it is a defect an agent writing at volume introduces
faster than a person would.

---

## Part 4 — The house move

The strongest pattern already in this manuscript, named so it can be reused deliberately:

> **alternative considered → rejected → because → what was done instead → evidence**

§7 of the style contract requires every design choice to be justified inline. This is the shape
that justification takes here. The worked example is the VLM paragraph at
`32-Data_Quality_and_Curation.tex:9`: a blanket cloud VLM pass is named, rejected, and four
reasons are given — rate limits, cost, terms of service, and redundancy with MegaDetector — in
ascending order of importance, closing on the one that actually decided it.

Two notes on using it. Order the reasons so the decisive one lands last, in the stress position
(§1.2). And at 102 words that paragraph is one sentence carrying four reasons; the same content
as four sentences would be stronger, which is exactly the Chapter 2 pattern in miniature.

---

## Part 5 — Two habits specific to this manuscript

### 5.1 `rather than` is a tic — 202 uses, 0.67 per 100 words

Roughly one sentence in four. The construction is correct, and it fits a thesis whose argument
repeatedly takes the form "X was chosen over Y". At this density the reader starts predicting
the rhythm.

When you reach for it, check whether the contrast deserves its own sentence:

> **Tic.** Treating a low pass rate as evidence of a classifier coverage gap rather than a
> labeling problem required a fallback rule.
>
> **Better.** A low pass rate can mean two things: a gap in the classifier's coverage, or a
> labeling problem. Resolving that ambiguity required a fallback rule.

### 5.2 Prefer American spelling

The manuscript currently mixes registers, including both `behaviour` and `behavior` inside one
paragraph (`33-*.tex:44` and `:47`). American English is the decision, matching
`\usepackage[english]{babel}` in `main.tex` — babel's `english` selects US patterns, and
`british` would have to be requested explicitly. It is also the majority usage already (94
American `-ize` forms against 63 British `-ise`).

So: *behavior, artifact, judgment, color, labeled, analyze, modeling, neighbor, penalize,
favorable, -ize, -ization*.
