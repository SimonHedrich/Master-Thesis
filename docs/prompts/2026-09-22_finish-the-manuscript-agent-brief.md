# Agent Brief — Finish the Manuscript

Handover for an agent asked to "write the chapters and subsections that are
still empty" in `thesis/manuscript/`. Written 2026-09-22, immediately after the
shortening pass that cut the manuscript from 57,291 to 36,402 words.

---

## Read this first: almost nothing is empty

The premise behind this task is out of date. A full survey on 2026-09-22 found
**no empty chapter and no empty subsection.** Every `\section` and
`\subsection` in all five chapters and the appendix carries drafted prose.
`grep -rn "TODO\|FIXME" thesis/manuscript/` returns **four** hits, all in the
preamble.

Do not "fill in" a section that looks short. Section lead-ins of 40 to 150
words are the house pattern: a `\section` opens with a short roadmap, then its
`\subsection`s carry the content. Those are finished, not stubs. Chapter 1 is
complete, including the numbered research questions and the AI-tool-usage
disclosure, and Chapter 5 already quotes those questions verbatim.

**If you add prose to a section that already has prose, you are undoing the
shortening pass.** The manuscript was just cut by 36% because it was overloaded.
Adding words back is the failure mode to avoid, not the task.

## What is actually outstanding

| # | Item | File | Size | Blocked on |
|---|---|---|---|---|
| 1 | English abstract | `preamble/abstract_eng.tex` | ~350 w | nothing |
| 2 | German abstract | `preamble/abstract_ger.tex` | ~370 w | item 1 |
| 3 | Submission date | `preamble/eidesstattliche_erklaerung.tex:13` | 1 date | **the author** |

Item 3 is not yours. It needs a real submission date. Leave the `% TODO` and
say so.

Two optional items, only if the author asks. Neither is a defect:

- `§Structure` in Chapter 1 is 185 words, under the department's rough
  one-page-per-TOC-entry guidance (`thesis/docs/formal_requirements_en.md:11`).
  So is `§Cross-Cutting Observations` in Chapter 4 at 281 words. Both are
  complete; they are simply short. Expanding them costs words the manuscript
  just spent effort removing, so check before doing it.
- 55 labels are defined but never referenced. 32 are subfigure labels, which is
  normal. The other 23 are harmless. This is not a compile error.

---

## The task: the two abstracts

### Shape

Three paragraphs, mirroring the bachelor thesis
(`thesis/docs/old_bachelor_thesis/preamble/abstract_eng.tex`, 363 words):
**problem framing → method summary → results and contribution.** Read that file
before writing. It is the model for length, register and level of detail.

The German abstract is a translation of the English one, not a separate
composition. Write English first. `preamble/abstract.tex` already `\input`s
both into the two minipages and needs no change.

### What must be in it

Draw every number from the manuscript. Do not compute, round or restate a figure
that is not already in a chapter.

- The research question: does distilling a large teacher into a lightweight
  student beat directly fine-tuning that student on the wildlife domain.
- The scope: 225 non-bird mammal classes, embedded deployment, Qualcomm QCS605
  as the target and a Raspberry Pi 400 as the measured proxy.
- Synthetic supplementation and the band structure, in one clause.
- The headline accuracy, **and here the invariant below is not optional.**
- The distillation result and the runtime result, both flagged as the
  single-design-point evidence they are.

### The invariant you must not break

From `CLAUDE.md`: the primary evaluation is the mixed real+synthetic test set,
and **a mixed headline number is always reported together with its real-only
breakout.** An abstract that quotes the mixed mAP alone is wrong, however
tempting the brevity. `\Cref{tab:headline-cross-model}` in
`chapters/4-Results.tex` has both columns side by side.

Also carry the honest finding rather than burying it: the watchdog on the mixed
default **fired**, and the designed revision was not carried out. Chapter 4's
`sec:results_domain_shift` and Chapter 5 both state this. An abstract that omits
it oversells the work.

### Length check

`\small` inside a `minipage` at `0.95\textwidth`. The bachelor thesis fits ~370
words on one page at the same settings. Do not exceed ~400.

---

## How to write here

### Load the skill, and read `scope.md` first

Invoke the `thesis-writing` skill. Then, in this order:

1. `references/scope.md` — **the most important file.** Ten rules for what
   belongs in the document at all. It is new as of 2026-09-21 and it is what the
   shortening pass was executed against. §1 is the relevance test, §3 is
   claim-first citation, §9 is the standing word budget.
2. `references/construction.md` — sentence and paragraph shape.
3. `references/claims.md` — §4 on comparative claims, §5 on the evaluation
   invariant above.
4. `references/conflicts.md` — §3 on dashes and semicolons.

### Two hard house rules

- **No `--` and no `;` in prose.** Split into separate sentences instead.
  Numeric ranges such as `200--250` are the only permitted `--`.
- **American spelling.** `behavior`, `quantization`, `tokenizer`, `-ize`.

### Claim first, citation attached

The source is never the grammatical subject. Not "Azizi et al. showed that…"
but the finding, with `\cite{}` at the end. Target is under 10% of cited
sentences naming a source in running text. In an abstract, cite nothing at all.

### The bachelor thesis: use the right file

`thesis/docs/old_bachelor_thesis/` is the reference for register and economy.
One trap: `chapters/2-Literature_and_Developments.tex` is an **orphaned,
superseded draft** that `thesis.tex` does not include and that ends mid-sentence.
The live file is `2-Literature_and_Developments-new.tex`. Check `thesis.tex`
before quoting anything from that directory.

What makes that thesis work, measured: 15,536 prose words; only 7% of cited
sentences name a source in running text; median paragraph 75 words; every proper
name explained at first use, about 1.5 new ones per page; every section closing
on the consequence that motivates the next.

---

## Verify, do not assume

```
uv run python -m scripts.thesis.check_manuscript        # from the repo root
uv run python -m scripts.thesis.check_manuscript -v     # list every warning
```

No LaTeX toolchain is installed in this container, so this script plus an
Overleaf compile is the only verification path. Run it **before and after** your
edits and compare, rather than reading the absolute numbers cold.

State as of this handover, which your run must not regress:

- **0** dangling cross-references
- **0** unresolved citation keys (92 distinct keys, 170 calls, 230 bib entries)
- **0** semicolons and **0** non-numeric dashes in prose
- 36,402 words against a 36,470 budget
- 6 sentences over 55 words, 4 paragraphs over 180 (one is a false positive:
  the splitter counts an `enumerate` block as a paragraph)

Errors fail the run. Warnings need judgment and never fail it.

### Never invent a citation key

Use only keys already in `bibliography/references.bib`. An invented key that
survives to submission is a fabricated citation. If a claim loses its source,
cut the claim. The abstract should carry no citations anyway.

---

## Lessons from the shortening pass

These cost real time to learn. They apply to any large edit here, and especially
if you delegate.

### On delegating to subagents

- **Give each agent a disjoint set of files.** Five agents ran concurrently over
  13 files with zero write conflicts because no two shared a file.
- **Arbitrate shared statements explicitly, in the brief.** Fifteen caveats had
  two to six homes each. Without a table naming exactly one owner per statement,
  parallel agents will each keep their copy, or each delete it. Both happened in
  rehearsal; neither happened with the table.
- **Pass content between agents through a scratchpad file, never by assuming
  order.** Chapter 2 handed two blocks to Chapters 3 and 4 through
  `handoff_from_ch2.tex`. Tell the receiving agent the file may not exist yet,
  to do its other work first and then poll, and what to do if it never arrives.
  Both receivers reported the handoff landed late; both handled it because the
  brief said it would.
- **Word targets are in raw `wc -w`, and LaTeX float markup counts.** Section
  targets that look right will sum above the file target once tables are
  included. Say so, or agents will cut prose to pay for table markup. Chapter 4
  is 7,376 words by `wc -w` and 5,513 words of prose.
- **Expect agents to land ~10% over target and plan for it.** All five did,
  consistently, and in every case the overshoot was defensible.
- **Do not let a subagent commit.** The working tree churns while others are
  running, and `CLAUDE.md` forbids `git add -A` here for good reason.
- **Read their reports skeptically but do not dismiss them.** One agent's
  claim that the instructed cuts could not coexist with the word target was
  correct, and it had done the arithmetic. Another routed two core evidence
  tables to the appendix, which was wrong and was reverted. Check the judgment
  calls; they are where the risk is.
- **Use Opus for prose judgment and a cheaper model for mechanical sweeps.**
  Every rewriting package here needed judgment. Grep-able checks do not.

### On this repository

- **Branch workflow.** Work happens on `thesis/overleaf-sync`. Never push that
  branch. Commit there, merge into `main`, push `main`. Use a git worktree,
  because the tree is usually dirty. The author runs `make overleaf-push`
  themselves after `make overleaf-status`.
- **Check `make overleaf-status` before any Overleaf operation.** As of this
  handover there was **1 commit to pull** from Overleaf. Pull before pushing or
  you will conflict.
- **The working tree routinely carries unrelated modified files** — in-flight
  training runs, scratch logs, another session's edits. Check `git status` and
  stage only your own files. At this handover, one pending change is
  deliberately uncommitted: the title-page switch in `main.tex`, which needs its
  untracked cover PDF staged alongside it.
- **`main.tex` is shared and easy to break.** It gained `amsmath`, `amssymb` and
  the tikz `calc` library only after chapters started using `\text{}` inside
  math. If you add a construct, check the preamble supports it, because nothing
  local will tell you.
- **Trust `wc -w` over impressions.** Every length claim in the pass was
  measured. Several confident guesses were wrong by 15%.
