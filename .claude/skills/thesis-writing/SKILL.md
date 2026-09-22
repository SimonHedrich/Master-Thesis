---
name: thesis-writing
description: Write, edit, or review prose for the Master's thesis manuscript in thesis/manuscript/ — any .tex chapter, section, abstract, figure caption, or research question. Use when drafting a new section, filling a % TODO stub, rewriting existing thesis prose, or checking a draft for sentence construction, terminology consistency, claim-evidence traceability, or LaTeX house style. Also use before committing thesis prose.
---

# Thesis writing

Rules for building sentences in `thesis/manuscript/`, not a list of words to avoid.

That distinction is the whole point. This thesis already had a style contract
(`thesis/bachelor-thesis-analysis.md` §7) and it is entirely conventions — numbers in math
mode, `\textit` on first mention, never "I". All of it correct; none of it about how a sentence
is built. Measurement across the 30,004 drafted words found **3 hits** from the classic slop
vocabulary and **39% of sentences over 40 words**. There was nothing left to ban.

## The reader

**One examiner who knows machine learning but not this project, reading roughly eighty pages
under time pressure, who wants the decision, the reason, the evidence, and the limit.**

Read that again before applying any rule, because it is what settles every conflict in
`references/conflicts.md`. Not a mechanic who must not misread a procedure. Not a consumer
opening a dialog box who needs reassurance. Decoration costs credibility with this reader; so
does a claim stated more strongly than its evidence supports.

## The five rules

1. **Start where the reader already is; end on the point.** Old information opens the sentence,
   new information closes it. *Test: does the last thing in the sentence deserve the emphasis
   the reader will give it?*

2. **Put the action in a verb, and give the verb an actor.** Name the script, the model, the
   filter, the stage — never the author. *Test: can you name a non-human agent? Then name it.*

3. **One sentence, one claim. Target 20-30 words, rewrite above 55.** Paragraphs are budgeted in
   words, not sentences: target 120, warn above 180. *Test: can you say what the paragraph is
   for in one clause?*

4. **Qualify before you claim.** Scope conditions — band, test domain, granularity, which
   evaluation axes were executed — come first, so the reader never believes the unconditional
   version. *Test: could a reader stop mid-sentence and be wrong?*

5. **One name per concept, one meaning per name.** Four words in this manuscript currently
   carry two meanings each. *Test: is this word already doing another job here?*

These targets are not imported from anywhere. `35-Synthetic_Generator_Comparison.tex` (median
28 words) and `4-Results.tex` (median 26) already meet them; `2-Literature_Review.tex`
(median 50) does not. **The rules describe the author's own most recent writing.**

## Which reference to read

| Doing this | Read |
|---|---|
| Deciding whether a passage, a name, or a whole section belongs at all | `references/scope.md` |
| Drafting or rewriting any prose | `references/construction.md` |
| Reporting a number, a comparison, or a conclusion | `references/claims.md` |
| Using a project term, or introducing a new one | `references/terminology.md` |
| Tempted to delete a hedge, add a dash or semicolon, use the passive, or pick a tense | `references/conflicts.md` |
| Checking LaTeX house style, acronyms, floats, or a submission checklist | `references/mechanics.md` |
| Choosing plot vs. table vs. inline prose, or laying out a multi-panel image figure | `references/figures.md` |
| Wanting to see a rule applied to real text from this thesis | `references/examples.md` |

Read the one you need, not all seven. **`scope.md` is the one to read first when
editing existing prose**, because the cheapest fix for a bad paragraph is usually
deleting it.

Two things to know without opening anything: the **mixed-set headline number is always reported
with its real-only breakout** (`claims.md` §5, from `CLAUDE.md`), and **hedges that calibrate a
claim against its evidence are mandatory** — only hedges that soften tone get cut
(`conflicts.md` §2). That second one is the rule most likely to be over-applied.

## Where the rules come from

Three systems that predate this thesis and disagree with each other. Gopen and Swan's *The
Science of Scientific Writing* (1990) supplies the reader-expectation rules; ASD-STE100, the
1986 aerospace controlled language, supplies length, shape, and one-name-one-meaning; the
Microsoft Writing Style Guide supplies the verb and reference rules. Where they conflict —
active voice against §7's impersonal register, hedge-cutting against scientific calibration,
the em dash, the present perfect — the disagreement is written down as a disagreement with a
verdict and a reason in `references/conflicts.md`, rather than averaged away.

`thesis/writing/anti-slop-skill.md` is the blog post that prompted this skill, and
`thesis/writing/2026-09-17_manuscript-baseline.md` holds every measurement quoted above.

## The length problem

The five rules above fix sentences, and sentences were never the reason the manuscript reached
57,291 words — roughly three and a half times the bachelor thesis, against a department
instruction asking whether seventy pages would have done the job. The prose was locally correct
and globally far too much of it: literature the work never uses reviewed one paper per
paragraph, abandoned approaches narrated as stories, and fifteen caveats each stated in two to
six places.

**`references/scope.md` is the rule set for that**, with the standing per-file word budget in
its §9 (36,470 words, of which the appendix is 5,500). `uv run python -m scripts.thesis.check_manuscript` measures
every file against that budget and checks that nothing was severed in the cutting.

## The limit

These rules fix form and scope. A section can satisfy every one of them and still say nothing,
because no rule knows whether a claim is true. `references/claims.md` goes as far as a rule can
— it makes every number traceable to an artifact and every citation resolvable — and stops
there. Whether the experiment was the right one stays with the author. It is only easier to see
once the noise around it is gone.
