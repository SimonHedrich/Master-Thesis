# Claims — evidence, traceability, and saying what is not supported

The source blog post ends by conceding that its rules fix the *form* of bad writing and never
the substance. A text can satisfy every construction rule and still say nothing. No rule knows
whether a claim is true, or worth making.

A thesis can go further than a README can. Not because judgment can be automated, but because
in this repository every number in the manuscript has an artifact behind it, every claim about
prior work has a bib entry behind it, and both of those are checkable by reading. This file is
the only part of the skill that touches substance, and it works entirely by **traceability**.

---

## 1. Every claim has a type, and the fourth type is the important one

Every claim-bearing sentence in the thesis is exactly one of:

| type | what backs it | how it appears |
|---|---|---|
| **measured** | a table or figure in this thesis, or a run artifact in this repo | a `\Cref` to the float, or a `%` comment naming the artifact path |
| **cited** | someone else's published result | a `\cite` key that resolves |
| **reasoned** | an argument from premises the reader already accepts | the premises stated in the same paragraph |
| **not established** | nothing | **said out loud** |

The fourth line is where a thesis earns credibility, and this manuscript already has the
exemplar. `4-Results.tex` §4.1.6 opens:

> "Three conclusions are supported by the evidence assembled here, and one is not."

and closes:

> "What is not supported is any conclusion about the rare, long-tail classes — which is to say,
> about the case that motivated synthetic data in this thesis at all. […] The honest summary is
> that the comparison establishes a ranking for data-rich classes and establishes that the
> current instrument cannot rank generators for data-poor ones."

That is the standard for Chapter 5 and for §§4.2–4.4. A limitation stated by the author is a
result. The same limitation found by an examiner is a hole.

**When drafting:** before writing a claim, name its type. If the answer is "reasoned" but the
sentence is phrased as if measured, fix the sentence, not the evidence.

---

## 2. Every number traces

A figure in the prose is backed by one of:

- a table or figure in the manuscript, reached with `\Cref`;
- a named artifact in this repo — `reports/…`, a run directory under
  `scripts/training/*/model_exports/…`, a document under `docs/plans/…`.

Where the source is a repo artifact rather than a thesis float, record the path in a `%`
comment on the same line:

```latex
The ensemble reaches a mixed-set mAP of $0.663$.  % reports/speciesnet_bandA_retrain/eval.json
```

The comment is for the author and for whoever checks the thesis later. It costs nothing in the
compiled document, and it separates a number that can be re-derived in five minutes from one
that cannot be re-derived at all.

**Numbers that change are the dangerous ones.** Any figure taken from a run that might be
re-run — every mAP in Chapters 4 and 5 — needs the artifact path, not just the value.

**This is not only a numbers rule.** The same comment-not-prose placement applies to any
repo-internal reference at all — a script name, a data file, a planning or scoping doc under
`docs/`. The rendered sentence states what was done or found; the path that backs it, if it
needs recording, goes in a `%` comment on the same line, never inline as `\texttt{}` in the
visible text. A reader has no access to this repository, so a path inside the sentence is not
traceability for them — it is a dead end.

> **Weak.** The candidate labels were filtered down to 483 using a dedicated filtering script
> (`\texttt{resources/refineLabels.py}`).
>
> **Better.** The candidate labels were filtered down to 483 using a dedicated filtering
> script.  % resources/refineLabels.py

The same holds when the source is a repo-internal *methodology* or *design* document rather
than a script or data file — "documented in `\texttt{docs/2026-03-10\_...md}`" is the same
defect with a different file extension. State the methodology or finding itself in the
sentence, and point the reader at the thesis section that develops it in full with `\Cref`; if
the internal doc still needs recording for traceability, that goes in a `%` comment, never
both stated as a claim and left inline as a visible path.

**Git history is repo-internal too, and it cannot be dropped into a `%` comment as a fallback.**
A commit trailer, a commit message, or a branch name is not evidence a reader can check, and
unlike a script path there is no artifact to point at from a comment either — the fix is to cut
the claim, not relocate it. This applies whenever a disclosure or methods sentence is tempted to
lean on "this is recorded in the repository" as its own evidence.

> **Weak.** Every code change produced with the tool is committed to this thesis's repository
> with a co-authorship trailer naming the model, so the model and date behind each AI-assisted
> contribution are recorded in the repository's history.
>
> **Fix.** Cut it. The reader has no access to this repository, so its git history cannot serve
> as anyone's evidence. State only what the sentence can support on its own — which tool did
> what, and that the author reviewed the output — and stop there.

---

## 3. Every borrowed claim carries a key that resolves

This section and `scope.md` §3 are two halves of one rule. Here: a citation must attach to a
stated claim. There: the claim, not the source, is the sentence's grammatical subject.

**A citation attaches to a stated claim; it never stands alone as a pointer.** "For a
comprehensive account of X, see \cite{key}" tells the reader nothing except that something
else exists — it spends a sentence sending the reader out of the document instead of putting
the relevant content of `key` into it. State what that source actually found or established,
in the same register as every other citation in the chapter, and put `\cite{key}` at the end of
that sentence as its evidence.

> **Weak.** For an independent comparative review of the YOLOv2–YOLOv7 generations
> specifically, see \cite{nazirYouOnlyLook2023}.
>
> **Better.** An independent comparative review of the YOLOv2–YOLOv7 generations finds
> \dots \cite{nazirYouOnlyLook2023}.

The substance for the "Better" version has to come from the source, not be invented to fill
the template — read the cited work (or its extraction under `research/literature/sources/`)
for an actual finding worth stating. If nothing in it adds information the surrounding
paragraph doesn't already have, cut the sentence rather than keep the pointer.

Against `thesis/manuscript/bibliography/references.bib` (project entries: datasets, tools,
model releases) or `references_zotero.bib` (Better BibTeX keys from the Zotero library).

The manuscript is currently clean here: 100 distinct keys across 158 citation calls, **zero
unresolved**. Keep it that way. When the Zotero library changes:

```
cp research/literature/references.bib thesis/manuscript/bibliography/references_zotero.bib
```

Do not invent a key and fix it later. An invented key that survives to submission is a
fabricated citation.

**The bachelor thesis is a prior publication, not a preceding chapter.** This Master's thesis is
an independent work. Treat `\cite{hedrichGeneratingSyntheticDatasets2026}` exactly like any
other cited paper by the same author: state what it found, cite it, move on. Never write "as in
the author's bachelor thesis", "more than the bachelor thesis did", or "continues the bachelor
thesis" as though the two documents shared a section, a decision, or a required disclosure
paragraph. Both happening to have, say, an "Environment" section or an AI-tools disclosure is
shared house style between two otherwise-unrelated documents, not a joint one, and needs no
cross-reference at all.

> **Weak.** As in the author's bachelor thesis, ChatGPT and DeepL were used to polish wording.
>
> **Better.** ChatGPT and DeepL Translator were used to polish wording.

When the bachelor thesis is genuinely the source of a *finding* — a result, a method, a number —
cite it like the manuscript already does correctly at `2-Literature_Review.tex:193` and
`35-Synthetic_Generator_Comparison.tex:4` ("Earlier work by the present author found …
\cite{hedrichGeneratingSyntheticDatasets2026}"). That is a legitimate borrowed claim under the
rule above. Reaching for it to justify anything else — practice, tooling, boilerplate — is not.

---

## 4. No comparative claim without the measurement

"Better", "faster", "outperforms", "improves", "more robust" each require a number in the same
paragraph or a `\Cref` to the table holding it. Also required: **the axis**. Better at what,
measured how, on which test domain, for which classes.

> **Weak.** The distilled student outperforms the directly fine-tuned baseline.
>
> **Better.** On the mixed test set the distilled student reaches a mAP of X against Y for the
> directly fine-tuned baseline (`\Cref{tab:kd-comparison}`); on the real-only breakout the gap
> narrows to Z.

The second version is longer and it is the only one that can be checked.

---

## 5. The evaluation invariant

From `CLAUDE.md`, and it is not negotiable because the whole evaluation design rests on it:

- The **mixed test set** — the union of the real test images and the balanced 225×50 synthetic
  test set — carries the headline number.
- The **real-only breakout** is reported **alongside it, every time**, as the primary-evaluation
  figure and as the only number comparable to public benchmarks.
- The **real-vs-synthetic delta** is stated wherever it bears on the claim. It is a watchdog: a
  clear discrepancy between mixed and real results means the evaluation axes get revised, and
  that revision belongs in the thesis if it happens.
- **No model is ever judged on synthetic images alone.**

Rationale, for the Methods chapter: the consistent 50 synthetic images per class stabilize
evaluation for Band A classes, which have few or low-quality real photographs. For Band D
classes, with up to 500 real test images, the same 50 are a negligible and consistent
addition. `docs/plans/2026-06-10_model-evaluation-strategy.md` has the full reasoning, and §7 of
the style contract asks for exactly this kind of stated-and-justified metric choice.

**A terminology consequence.** The phrase "mixed test set" appears **zero times** in the
manuscript as drafted, while the real test set carries five different surface forms. A reporting
invariant that has no name cannot be applied consistently. See `terminology.md`.

---

## 6. Scope conditions are load-bearing

A result in this thesis is almost never unconditional. It holds on a training band, on a test
domain, at a granularity, under a single evaluation axis. Those conditions are part of the
claim, and `construction.md` §2.3 puts them **before** it so the reader never believes the
unconditional version first.

The axes that qualify a result here:

- **training band** — A (synthetic only), B (mixed), C (real-200), D (real-large);
- **test domain** — mixed, real, synthetic;
- **granularity** — detection only, look-alikes merged, full 225-way;
- **which axes were actually executed** — §4.1 reports a three-axis evaluation design of which
  one axis was carried out, and says so.

---

## 7. Floats make claims too

A figure that no sentence points at makes no claim, and LaTeX may place it anywhere. Every
float is referenced from the prose at least once, with `\Cref`, and the referencing sentence
says what the reader should see in it.

Four figures in `4-Results.tex` are currently placed but never referenced —
`fig:generator-headline-map`, `fig:prompt-length-ablation`, `fig:generator-cost-vs-map`,
`fig:generator-per-class-heatmap` — along with `eq:quality-score` in
`32-Data_Quality_and_Curation.tex`. All four tables in the same chapter are referenced
correctly, so this is a figure-specific habit worth breaking.

---

## 8. What this file cannot do

It cannot tell you whether a claim is worth making, whether the experiment was the right one,
or whether a section should exist. Traceability catches a number with no source and a citation
with no entry. It does not catch a true, well-sourced, perfectly punctuated paragraph that
nobody needed.

That judgment stays with the author. It is only easier to see once the noise around it is gone.
