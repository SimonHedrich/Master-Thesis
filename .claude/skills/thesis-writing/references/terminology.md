# Terminology — one name per concept

ASD-STE100 gives each approved word exactly one meaning and one part of speech. "Check" is
permitted as a noun and forbidden as a verb, because "check the valve" could mean inspect it,
test it, or stop it, and a mechanic under time pressure should never have to choose. The
Microsoft guide reaches the same rule from a completely different direction: "if you mean the
same thing, use the same word". A rule found twice by two industries that never spoke to each
other is a real one.

A 900-word approved dictionary is not possible for a machine-learning thesis. What is possible,
and what this file is, is a table of **the concepts in this project that already carry more than
one name**, and the four words that already carry more than one meaning.

Every entry below is grounded in counts measured from the drafted manuscript, recorded in
`thesis/writing/2026-09-17_manuscript-baseline.md` §6. Nothing here was invented.

**Two standing rules:**

1. **First mention takes `\textbf{}` and a definition** (§7 of the style contract), then the
   same word forever. The manuscript has drifted here — 4 `\textbf` in 30,004 words.
2. **A word that carries two meanings in this thesis loses one.** Pick the sense that is used
   more often or is harder to rename, and give the other sense a different word.

**Abbreviations are not in this file.** They live in `preamble/acronyms.tex`, which is already
declared as the single mechanism for this thesis. Adding them here would recreate exactly the
rough edge §8 of the style contract warns about. See `mechanics.md` §2.

---

## 1. The four collisions — one word, two meanings

### 1.1 `distillation`

| sense | approved term | never |
|---|---|---|
| teacher → student transfer | **knowledge distillation**, then bare *distillation* and `\gls{kd}` | — |
| diffusion sampler shortening (SD 3.5 Large Turbo) | **step distillation**, always both words | bare *distillation*, *distilled model* |

The manuscript uses bare `distillation` ×34 and `distilled` ×7 across both senses, in the same
document. Bare *distillation* means knowledge distillation. The diffusion sense is always
written out as *step distillation*, and a generator that underwent it is a *step-distilled
generator*, never "a distilled model". This matters most in §3.5 and §4.1, where both senses
are live in the same paragraph.

### 1.2 `tier`

| sense | approved term | never |
|---|---|---|
| generator API and quality levels (`gpt-image-2` low/medium, API vs local) | **tier** | — |
| taxonomic levels (species / genus / family) | **taxonomic rank** | *tier* |
| the evaluation assessment levels of the eval-strategy document | **assessment level** | *tier* |

`tier` ×41 currently spans all three. It keeps the generator sense, which is the vendors' own
word and the one used most (18 uses in `4-Results.tex`, 14 in `35-*.tex`), and loses the other
two. `31-*.tex:19` uses `Tier 1/2/3` for taxonomic ranks and is the one place to change during
the Chapter 3 pass.

### 1.3 `regime`

| sense | approved term | never |
|---|---|---|
| the prompt-construction variants (`full`, `compressed`, `maxlen`) | **prompt regime** | bare *regime* where ambiguous |
| how a class's training data was composed | **band** (see §2.1) | *regime* |
| which held-out set a metric was computed on | **test domain** (see §2.2) | *regime* |

`regime` ×47 spans prompt regime (12) and training or evaluation regime (35). It keeps the
prompt sense. Note that `CLAUDE.md` itself uses "regimes" loosely for training data conditions;
in the thesis those are **bands**.

### 1.4 `label`

| sense | approved term | never |
|---|---|---|
| the annotation record attached to an image or box | **annotation** | *label*, where the class is meant |
| the class assigned to a box | **class** | *label* |
| the biological name | **species name** or **taxon** | *label* |

`label` ×157 is the loosest word in the manuscript. It is not worth a global rewrite of
drafted text, but new chapters use *annotation* for the record and *class* for the assigned
category.

---

## 2. Concepts with more than one name

### 2.1 Training bands

| concept | approved | not approved |
|---|---|---|
| the four data-composition conditions | **Band A / B / C / D**, capitalized; *band* lowercase for the generic noun | *tier*, *group*, *stratum*, *bracket*, *regime* |

`band` ×74 and `Band A/B/C/D` ×37 already dominate; this entry protects them against the
`tier` and `group` drift. Definitions: **Band A** synthetic only, **Band B** mixed real plus
synthetic, **Band C** real-200, **Band D** real-large.

### 2.2 Test domains — the gap that matters most

| concept | approved | not approved |
|---|---|---|
| the union of real and synthetic test images; carries the headline number | **the mixed test set** | bare *mixed*, *the union*, *the combined set*, *the full test set* |
| the real held-out photographs; the primary-evaluation figure | **the real test set** | *real images*, *real photographs*, *real imagery*, *real data*, when the test set is meant |
| the balanced 225×50 synthetic held-out set | **the synthetic test set** | — |

**`mixed test set` appears zero times in the manuscript.** `CLAUDE.md` names it as the headline
evaluation set, and `claims.md` §5 makes reporting it an invariant — but a reporting rule with
no noun phrase cannot be applied consistently. Introduce the term in §3.7 (Evaluation
Framework) with `\textbf{}` and a definition, and use it in every results sentence after that.

The real test set currently carries five surface forms (`real test set` ×13, `real images` ×15,
`real photographs` ×12, `real imagery` ×8, `real data` ×7). Those phrases are all fine when
they mean real imagery in general; they are not fine when they mean the held-out set.

### 2.3 Teacher and student

| concept | approved | not approved |
|---|---|---|
| the large model that supervises | **teacher model** on first use per chapter, then *teacher* | *large model*, *teacher network* |
| the deployable model being trained | **student model** on first use per chapter, then *student* | *lightweight model*, *compact model*, *small model*, *student network* |

Currently `teacher` ×35 and `student` ×29 run bare, with the full forms appearing exactly once
each, in one sentence at `30-Overview.tex:5`. This entry is preventive: §3.6, §4.3 and
Chapter 5 are unwritten, and they are where the pair is used most.

### 2.4 The experimental unit

Two naming schemes exist and they do not currently leak into each other. The boundary is fixed
rather than merged, because renaming drafted text buys nothing:

| scope | approved | meaning |
|---|---|---|
| generator comparison (§3.5, §4.1) | **cell** | one generator × prompt-regime combination (×68) |
| augmentation study (§3.4) | **Setup A / B / C** | one augmentation configuration (×28) |
| training generally | **run** | one training execution of one configuration |

Not approved as the unit noun anywhere: *condition*, *variant*, *arm*, *configuration*, *grid
point*. `configuration` ×9 currently straddles both schemes and is the one to stop using.

### 2.5 Hardware

| concept | approved | not approved |
|---|---|---|
| Qualcomm QCS605, the deployment target | **the target hardware**, **QCS605** | *target device*, *target platform* |
| Raspberry Pi 400, the benchmark stand-in | **the proxy device**, **Raspberry Pi 400** | *stand-in*, bare *proxy*, *embedded proxy*, *proxy hardware* |
| the product context | **AX Visio** (`\textit` on first mention) | — |

Updated 2026-09-26: the manuscript benchmarks on a **Raspberry Pi 400**, not the
Raspberry Pi 5 that `CLAUDE.md` names as the planned proxy, and the settled term is
**the proxy device** (chosen in the 2026-09-26 fix pass, which collapsed five surface
forms — *target-adjacent proxy device*, *proxy device*, *embedded proxy*, *proxy
hardware*, bare *proxy*). The phrase "target-adjacent proxy device" survives only
inside research question 4, whose wording is restated verbatim in Chapter 5. Several
bare uses of *proxy* in the manuscript mean something else entirely (*recognizability
proxy*, *automatic proxies*), which is exactly why the bare form is not approved for
the hardware.

### 2.6 Class vocabulary

| concept | approved | not approved |
|---|---|---|
| one of the 225 entries in the label universe | **class** | *category*, *label* |
| the biological entity | **species** | — |
| a taxonomic grouping above species | **taxon** / **taxa** | *clade*, unless phylogeny is meant |
| COCO's schema fields | **category**, quoted as COCO's own term | — |

`class` ×284 already dominates. `category` ×26 survives only where COCO's `supercategory` and
`category_id` fields are being described, and should be visibly marked as COCO's vocabulary
(`\texttt{}`) when it is.

---

## 3. Hyphenation

Four live inconsistencies. Most are not errors so much as an inconsistently applied rule, so
the rule is what is recorded here: **the attributive form takes the hyphen, the noun does not.**

| noun | attributive | current split |
|---|---|---|
| the test set | a test-set image | 43 / 6 |
| per class | per-class accuracy | 14 / 18 |
| the ground truth | a ground-truth box | 11 / 15 |
| a camera trap | camera-trap imagery | 5 / 13 |

`per class` is the exception to the pattern: it is almost always attributive in this thesis, so
**per-class** is the default and bare *per class* is correct only after a verb ("images are
counted per class").

Settled spellings, no variants: **look-alike** (not *lookalike*), **trade-off**, **sub-study**,
**fine-tune** / **fine-tuning**, **long-tailed**, **real-world** (attributive), **real-only**,
**held-out**.

---

## 4. Adding a term

When a new concept appears in a chapter:

1. Check this file and `preamble/acronyms.tex` for an existing name.
2. If the concept is new, define it once with `\textbf{}` at first use, and add a row here.
3. If it collides with a word already in use, resolve the collision here first. One of the two
   senses gets a different word, and it is written down before either is used again.
