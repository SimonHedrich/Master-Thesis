# Chapter 5 (Discussion and Conclusion): critique and rework outline

Date: 2026-09-28. Subject: `thesis/manuscript/chapters/5-Discussion_and_Conclusion.tex` at
commit `e61d396` (3,058 words against a 3,000-word budget, 101 lines). Read against Chapters
1, 2 and 4, the `thesis-writing` skill, the supervisor's register feedback
(`2026-09-25_manuscript-prose-review.md`, Part II), and the department's guidance on the
closing chapter (`formal_requirements_en.md`, "Results" and the outlook paragraph).

Nothing here questions an experiment. Every number in the chapter was checked against the
Chapter 4 tables and is correct (see §2.6). The problems are what the chapter chooses to say,
and how often it says it. Line numbers are `5:NN` in the file as of today.

---

## 0. Verdict

1. **The chapter is a compressed second copy of Chapter 4, and the Conclusion is a third
   copy.** Nearly every sentence of §5.1 and every paragraph of §5.2 has a twin in Chapter 4,
   usually with the same numbers. Inside this one chapter, 0.591 appears four times, 0.599
   and 0.529 three times each, "upper bound" four times, "single run / one design point"
   five times, and one sentence ("…the long-tail classes that motivated generating images in
   the first place") appears verbatim twice (5:27, 5:95) after already appearing in §4.1.6.
2. **It does not discuss.** There is no mechanism argument (why did distillation lose, why
   does the prompt matter, why does Band A fail), no comparison to prior work (zero citations,
   while Chapter 2 sets up at least eight direct hooks and in one place says outright that a
   point "is important for interpreting the distillation comparison"), no statement of what
   generalizes beyond this dataset, and no implication for the product.
3. **The strongest findings that have no research question fell through.** Classification,
   not localization, limits every model (Δcoarse is 16 to 22 times Δfine, Asian elephants
   collapse onto African ones 1,545 to 9). The two synthetic-data studies agree independently.
   YOLO26n beats the production architecture on size, accuracy and speed at once. The first
   appears only inside a Limitations paragraph, the second not at all, the third only in the
   Conclusion's last paragraph.
4. **Limitations and Further Work are inventories, weighted equally.** "Training cost is not
   comparable" gets the same paragraph status as "no confidence intervals anywhere", although
   no answer depends on it. Meanwhile a limitation that the Methods chapter promises for this
   section (no architecture comparison, `36-…:9`) is missing, and the licensing status of the
   winning detector, which Chapter 2 rules out by name, is never mentioned. Further Work is
   mostly the thesis's own housekeeping (rerun one evaluation, run the bootstrap, rewrite
   tables), where the department asks for initiative and alternative routes.
5. **The register the supervisor flagged is still the chapter's default** (see §2.5), and the
   thesis ends on a not-done caveat: "…has not been measured at all."

---

## 1. What the chapter does now

| Section | Lines | ≈Words | What it does | Verdict |
|---|---|---|---|---|
| Lead-in | 5 | 50 | Roadmap of the chapter | Fine, trim |
| §5.1 RQ restatement | 10–17 | 170 | Copies the four questions from §1.2 verbatim | Optional (bachelor pattern); the sentence at 5:10 is signposting |
| §5.1 Q1 | 19–22 | 260 | Restates §4.3.1 including all three asymmetries and all four limits | Re-report |
| §5.1 Q2 | 24–27 | 330 | Restates §4.1.1, §4.1.2, §4.1.3, §4.1.4, §4.1.6 | Re-report |
| §5.1 Q3 | 29–32 | 310 | Restates §4.2.3 and §4.2.4 | Re-report |
| §5.1 Q4 | 34–37 | 330 | Restates §4.3.2; two sentences are identical to Chapter 4 | Re-report |
| §5.1 Summary | 39–40 | 90 | Meta-commentary on which questions "yielded usable results" | Delete |
| §5.2 Limitations | 43–72 | 720 | Eight paragraphs, each a restated Chapter 4 caveat | Rebuild around what bounds the answers |
| §5.3 Further Work | 75–88 | 520 | Six items, three of them compute- or rewriting-only | Rebuild as research directions |
| §5.4 Conclusion | 91–101 | 560 | Restates §5.1 with the same numbers, ends on a caveat | Rewrite to half its length |

---

## 2. Diagnosis

### 2.1 Re-reporting instead of discussing

The Discussion should be readable by someone who has read Chapter 4 without making them read
it again, and by someone who skipped Chapter 4 without giving them a worse Chapter 4. At the
moment it does the second thing.

| Chapter 5 passage | Chapter 4 twin |
|---|---|
| 5:20 Q1 verdict (17 cells, 0.599/0.560, 0.529/0.479, Band A widest) | §4.3.1, paragraph after `tab:kd-vs-direct`, near verbatim |
| 5:22 "Three properties limit…" (batch 16 vs 32, early stop, one (T, α), 70:1, no zero-shot) | §4.3.1 "Three asymmetries" and "Four limits"; §4.2.1 last paragraph |
| 5:25 Q2 (0.134 ± 0.014, third of twelve, 28 %, 2.9×, 0.008 vs 0.018) | §4.1.1, §4.1.3, §4.1.6 |
| 5:27 prompt effect (−45 %, −48 %, kinkajou, "instrument cannot rank") | §4.1.2, §4.1.4; last sentence identical to §4.1.6 |
| 5:30 Q3 (0.588/0.591, 0.638/0.639, 0.341/0.386, 0.300) | §4.2.3, first two paragraphs |
| 5:32 mixed vs real (0.042–0.089, 208/224, 76, 0.21/0.72, 1.44 boxes) | §4.2.1 paragraph 4, §4.2.4 |
| 5:35 Q4 (423, 360–381, 949, 400 MB, 35 s, 5.4 s, 83×, 13×, 71×, 22×, 0.064/0.070) | §4.3.2 paragraphs 4–6; "the teacher ensemble cannot ship on this hardware" identical |
| 5:37 (Hexagon has no analogue, order-of-magnitude gains) | §4.3.2 last paragraph, identical |
| §5.2, all eight paragraphs | §4.2 intro (CIs, teacher upper bound), §4.2.4 (revision), §4.3.1 (four limits), §4.3.2 (three limits), §4.1.1 and §4.1.6 (ordinal), §4.1.5 (rubric), §4.2.2 (error types), §4.4 (training cost) |
| §5.4 paragraphs 2–5 | §5.1 itself, same numbers |

The skill's one-home rule (`scope.md` §6) is met at sentence level and broken at chapter
level. The fix is not to trim sentences but to change what each section is for: Chapter 4
owns the numbers and their immediate caveats, Chapter 5 owns the verdicts, the mechanisms,
the comparison to prior work, and the consequences. A number should reappear in Chapter 5
only when the verdict cannot be stated without it.

### 2.2 No interpretation, no literature

Chapter 2 was written to be used here and is not. Each row is a finding of this thesis next
to a claim Chapter 2 already makes and cites, which the Discussion could confirm, contradict
or qualify in one or two sentences. Citations in a Discussion are allowed by the skill exactly
for this: a genuine external comparison (`scope.md` §3).

| Finding (Chapter 4) | Chapter 2 hook, already cited there | What the Discussion could say |
|---|---|---|
| Distillation loses in all 17 cells, widest on Band A | Negative transfer at large capacity gaps, teacher assistant as remedy (`mirzadeh…2020`, §2.3.1 and `sec:kd_loss`) | The result is the documented failure mode at 70:1, not an anomaly |
| Same | Naive transplant of the classification recipe onto a detector should not be expected to match a detection-specific scheme (`yangFocalGlobal…2022`; §2.3.1 says this "is important for interpreting the distillation comparison") | The sentence Chapter 2 promised is missing from Chapter 5 |
| Same, plus Δfine ≤ 0.027 for every model | Hinton's premise: soft targets carry the similarity structure among look-alikes | This taxonomy's errors are not among look-alikes (Δfine is tiny), so the information soft targets add is aimed at the wrong error. A new interpretive point the thesis can make from its own data |
| Prompt detail outranks generator identity | Bare class-name prompts produce a very poor classifier; alignment, not fidelity, is the operative variable (`azizi…2023`) | Direct confirmation on a detection task |
| Apparent realism did not predict usefulness | Domain randomization (`tobin…2017`, `tremblay…2018`); human realism ratings diverge from other axes (`zhouHYPE…2019`); the author's own noise-control result (`hedrich…2026`) | The incumbent was chosen on the one criterion the literature already said was unreliable |
| Band A (synthetic only) halves accuracy; Band B (half synthetic) loses nothing | ~5× as many synthetic images needed; raw supervised training cannot absorb the gap, a pretrained representation partly can (`heSyntheticData…2023`); fine-tuned generator "still short of real-data training" (`azizi…2023`) | The band result reproduces the literature's shape and adds the ratio at which it breaks |
| Rare classes stay below 0.05 AP for every generator; a wrong-animal generator goes unpenalized | CAS revealed per-class zeros hidden by global scores (`ravuriCAS…2019`) | The downstream axis is CAS by another name and shows the same blind spot; the rubric was the designed complement |
| Synthetic test images are easier by ≥ 0.2 mAP even for a model never trained on them | §2.2.2 warns that a synthetic-only headline "could appear artificially strong" | The warning is now a measurement, with a 0.2 floor and a band-dependent remainder |
| Classification, not localization, is the bottleneck; cross-group errors dominate | Fine-grained recognition (`wei…2022`); the ensemble's confidence-gated taxonomic rollup, which a single detector loses by default (`gadot…2024`, §2.1.3); WordTree as the detector-side precedent (`redmon…2017`) | The single detector gave up exactly the mechanism that would have turned its dominant error into a coarse-correct answer. This is also the strongest outlook item |
| 423 ms full precision on the proxy CPU | Up to 10× from 8-bit quantization on Hexagon-class DSPs (`krishnamoorthi…2018`) | A reasoned, clearly labelled estimate of where the accelerated path would land (tens of milliseconds), instead of "still open" |
| Test set is iNaturalist photographs, deployment is binocular stills | Location and viewpoint shift degrade wildlife models sharply (`beery…2018`); the thesis rejected camera-trap data for this reason | The same argument applies to the thesis's own test domain and is currently not stated as a limitation |

Mechanisms the chapter could argue from its own evidence, all currently absent:

- **Why Band A fails.** Three measured facts point the same way: the synthetic test set holds
  one centred animal per image (1.00 vs 1.44 boxes), the off-the-shelf ensemble already finds
  synthetic images 0.2 mAP easier, and fine-tuning the teacher on synthetic-only Band A
  supervision lowers its real accuracy from 0.324 to 0.182 (§4.2.3). Together they argue that
  the generated images lack compositional diversity rather than realism, and that
  synthetic-only supervision teaches something actively wrong. Chapter 4 says the two are
  "not separable"; the Discussion is the place to make the reasoned case and label it as such.
- **Why distillation lost.** Candidates are scattered across `sec:kd_loss` and §4.3.1 and
  never weighed: the 70:1 gap, one teacher vector broadcast to every foreground anchor, the
  probability-cache temperature approximation, classification-branch-only supervision, a
  teacher that sees 480 px crops while the student sees whole frames, plus the batch-size
  confound. A discussion says which of these it considers likely and why.
- **Why the prompt matters that much.** §4.1.2 attributes it to text-encoder capacity and
  diagnostic detail; the within-group confusion moving in step (0.253 to 0.457) is the
  evidence that the detail is species-discriminating detail. One paragraph, with Azizi.

### 2.3 Findings that have no research question

Because §5.1 is organized strictly by the four questions, everything Chapter 4 found outside
them is absent from the Discussion:

1. **Classification is the bottleneck** (§4.2.2, §4.2.5): AR100 of 0.845 against mAP 0.599,
   Δcoarse 16 to 22 times Δfine, 19.3 % within-group confusion concentrated in five groups,
   the one-directional elephant collapse. The Introduction (1:5) forward-references this very
   section as the reason classification is hard, so Chapter 1 promises it and Chapter 5 drops
   it. It is also the most product-relevant finding in the thesis.
2. **YOLO26n against YOLOv5s** (§4.2.1, §4.3.2): 2.8× smaller, +0.103 mixed and +0.122 real,
   2.2× faster. YOLOv5s is the production architecture, so this is the "previous state
   against the new state" comparison the department explicitly asks the closing chapter to
   make. It sits only in the Conclusion's last paragraph.
3. **The two synthetic-data studies agree** (§4.4): the useful range of synthetic data is the
   middle of the availability spectrum, not the tail that motivated it. This is the thesis's
   best synthesis and it is not in Chapter 5.
4. **The generation pipeline produces compositionally uniform images** (one animal, centred).
   Used only as a caveat; it is also a design finding about the production dataset.
5. **Robustness checks on the headline** (drop classes with < 30 test images: 0.594; weight
   by test count: 0.597). Useful in the Conclusion where the headline is "advertised".
6. **Pre-processing cost of device stills** (256–275 ms to decode and letterbox a native
   4192×3120 still, §4.3.2). At 320 px this would exceed the model's own time. A practical
   observation for the product that is buried in Chapter 4.

### 2.4 Limitations and Further Work as inventories

**Limitations, current eight paragraphs, with a recommendation each:**

| Paragraph (line) | Bounds which answer | Recommendation |
|---|---|---|
| No confidence intervals (47) | All | Keep, one paragraph. The seed episode is told in §4.1.6, cite it, do not retell |
| Revision of evaluation axes not carried out (50) | Q3 | Keep once. Currently stated in §4.2.4 in full and again at 5:32, 5:51, 5:85, 5:97. See decision D3 |
| Q1 rests on one cell (53) | Q1 | Split. Batch size and early stop are *confounds* (defects), the (T, α) point, one pairing and response-only are *scope*. The "missing data directory" bug post-mortem goes (`scope.md` §5) |
| No quantization-aware training (56) | Q4 | Merge into one external-validity paragraph with the proxy device, the absent DSP and the absent device stills |
| Teacher figures are upper bounds (59) | Q1, Q4 | Keep as one sentence with `\Cref`; §4.2 already argues it in full |
| Generator comparison ordinal only (62) | Q2 | Keep one sentence. The six sub-limits are Chapter 4 material |
| Rubric never run (65) | Q2 | Merge into the Q2 limitation; it is the reason for the rare-class blind spot, not a separate limit |
| Error types not separated (68) | none directly | Cut here. It becomes the caveat inside the classification-bottleneck discussion |
| Training cost not comparable (71) | none | Cut. No answer depends on it |

**Limitations that are missing:**

- **No architecture comparison.** `36-Model_Training…:9` says nano-scale detectors outside
  the YOLO family were set aside on time grounds, "a limitation carried in
  \Cref{sec:limitations}". It is not there. Dangling promise.
- **Bands confound species with composition.** Stated inline at 5:30 and in §4.2.3, but it is
  the structural limitation of Q3 and deserves the paragraph.
- **Automatic, unreviewed labels for every model.** Ground-truth boxes come from MegaDetector
  and species labels from SpeciesNet-based curation (§3.2.4). The chapter names only the
  teacher's circularity, but label noise bounds the students' absolute figures too.
- **Test domain is not the deployment domain.** iNaturalist photographs against binocular
  stills; no image from the device was evaluated although example stills exist. Beery's
  location-shift finding, which the thesis used to reject camera-trap data, applies to its
  own test set.
- **17 Band A classes have fewer than 30 real test images** (§3.3.1), so Band A figures are
  test-limited before they are model-limited.
- **The synthetic test set comes from the same pipeline as the synthetic training images**
  (verify against `sec:synthetic_test_set`). If so, the mixed condition partly scores
  generator-style fit, which sharpens the Q3 finding.
- **Licensing of the winning detector.** See §2.7.
- **The production pipeline's accuracy was never measured.** The ensemble evaluated is
  MegaDetector + SpeciesNet, not the deployed YOLOv5s + SpeciesNet pairing. Chapter 1
  motivates the work by improving the production model's accuracy, and the thesis has a
  latency estimate for that pairing but no accuracy figure. Either state this as a
  limitation or reframe the accuracy reference as the teacher ensemble only.

**Further Work, current six items:**

| Item | Nature | Recommendation |
|---|---|---|
| 1 Quantization | Research direction | Keep, and make the reasoned estimate (§2.2, Krishnamoorthi row) |
| 2 Retroactive CIs | Compute only, suite supports it | Do it before submission (D2) or collapse to one sentence |
| 3 Re-score archived predictions (zero-shot student, multi-animal subset) | One eval run plus a re-score | Same as item 2 |
| 4 Wider distillation space | Research direction | Keep, but reorient: the compatible-teacher route (a large single-stage YOLO as teacher, enabling FGD-style feature distillation) is the one Chapter 2 argues for |
| 5 Blind rating rubric | Needs raters | Keep, short |
| 6 Evaluation-axes revision | Rewriting only | Not further work. Resolve by D3 and state the outcome once in Limitations |

The department's outlook paragraph asks two things this section does not deliver: further
steps the author lacked time for (partly delivered), and "could the entire task have been
accomplished differently or more easily by another route? Show initiative." Candidate
directions of that second kind are in §5.4 of this report.

### 2.5 Register

The supervisor's five patterns, as they stand in the file today:

- **Discourse-noun subjects** (the document acting on itself): 5:5, 5:10, 5:22 "Three
  properties limit how far that answer generalizes", 5:27 "A third criterion the question
  did not anticipate outranks both", 5:35 "The comparison with the two ensembles is the
  substantive answer", 5:37 "This answer is not a verdict on the hardware. The question
  asks…", 5:40 the whole Summary, 5:45, 5:51 "This limitation, alone in this section,
  deviates from a stated protocol", 5:54 "The answer is bounded by four gaps", 5:63 "Six
  limits apply", 5:77 "Six follow-ons emerge", 5:88 "The first item decides whether…",
  5:93 "The substantive contribution sits in two sub-studies".
- **Adjective-swap antitheses:** 5:37 "a scope decision rather than an attempt that failed",
  5:51 "a ranking instrument, not an estimate", 5:54 "an absolute level rather than a gain",
  5:57 "a feasibility statement… Calling it a deployment verdict would require", 5:69 "an
  upper bound and not a measurement", 5:81 "compute rather than redesign", 5:84 "raters
  rather than compute", 5:97 "directional rather than exact". Eight in one chapter.
- **Commerce metaphors:** 5:77 "cost-to-value ratio", 5:80 "largest remaining lever" and
  "second lever", 5:35 "gives up 0.064 mAP", 5:101 "at a cost of 0.064 mAP".
- **Colloquial verdicts:** 5:20 "No, at the one design point tested." (the supervisor removed
  the Q4 twin in `cb07dab` and left this one, so it may be acceptable to him; still the only
  such opener). "in the first place" twice (5:27, 5:95).
- **Counted-abstraction openers:** 5:22, 5:54, 5:63, 5:77.
- **Ending.** The last sentence of the thesis is "…the accelerated path such a pass would
  target has not been measured at all." The abstract rule in memory (no "not done" caveats,
  frame the detectors as much faster and smaller at little accuracy loss) applies with more
  force to the final sentence of the document.

The deeper cause is structural: a section that discusses answers to questions naturally
produces sentences about answers and questions. Giving each paragraph a domain subject (the
detector, the generator, the band, the elephant pair) removes most of it without word-level
policing.

### 2.6 Fact check

All figures in the chapter reconcile with the Chapter 4 tables and the Methods text: the 17
cells, 0.599/0.560 and 0.529/0.479, 70:1 (193/2.71), 0.134 ± 0.014, 28 %, 2.9×, 0.008 vs
0.018, −45/−48 %, 0.588/0.591, 0.638/0.639, 0.341/0.386, 0.300, 0.042–0.089, 208/224, 76,
0.21/0.72, 1.44, 423 ms, 360–381 ms, 949 ms, 83× and 13× (35,000/423, 5,400/423), 71× and
22× (193/2.71, 60.34/2.71), 0.064/0.070, 0.176/0.154, 50 classes under 150 images. Four
calibration issues:

- 5:99 "Three disclosed asymmetries prevent a clean single-factor comparison." §4.3.1 says
  only the first (batch size) breaks the design; the other two are convergence notes.
- 5:32 "About 0.2 mAP of that difference is not due to domain shift since a synthetic test
  image contains one animal in the center". Chapter 4 bounds the 0.2 with the off-the-shelf
  ensemble and offers the one-animal composition as the plausible reason. The "since" turns
  a hypothesis into the cause.
- 5:63 lists "evaluation is real-only" as a limitation of the generator study. Under the
  thesis's own invariant the real-only figure is the preferred one, so this needs a reason
  or should go.
- 5:93 restates the effort distribution across the four questions, which §1.2 already states.

### 2.7 Two content gaps that are not writing problems

**Licensing.** Chapter 1 lists "a product requirement that the resulting weights remain
commercially licensable" and says any replacement is drawn from the same family as YOLOv5
for licensing reasons. Chapter 2 (2:42) says every Ultralytics release after commit `5cdad89`
"is ruled out, independent of any architectural or accuracy comparison". YOLO26n is an
Ultralytics release, and the manuscript never says under what terms it could ship. The
Conclusion then says the only thing between this detector and a shippable model is a
compression pass. Whether Swarovski Optik holds a license that covers YOLO26, or whether the
thesis's recommendation is conditional on one, is a fact only the author knows. It has to be
stated somewhere in Chapter 5, and the Conclusion's "shippable" sentence depends on it.

**The resolution runs.** The working tree holds untracked `yolo26n-res320-*` run directories
and a `resolution_sweep/` folder under the primary detector's export directory, dated
2026-09-20. Chapter 4 reports nothing at 320 px and Further Work item 1 lists a
resolution-native training arm as not done. If these runs produced results, they belong in
§4.3.2 and change both Q4's answer and the outlook. If they were abandoned, nothing changes.

---

## 3. Decisions to make before rewriting

**D1. Structure.** Three options, detailed in §4.

- A. Keep the bachelor pattern (one paragraph per question, then Limitations, Further Work,
  Conclusion) and strip all re-reporting. Cheapest, but the interpretation has nowhere to
  live except inside the answers, which then grow again.
- B. Thematic Discussion (synthetic data, classification bottleneck, distillation,
  deployability) with the four answers as a short list at the end. Best for synthesis, but
  departs from the pattern Chapter 1 promises ("restates and answers each question in turn").
- C. Short answers per question, then an Interpretation section with three or four thematic
  subsections, then Limitations, Further Work, Conclusion. **Recommended.** It keeps the
  Chapter 1 promise and gives mechanisms, literature and orphan findings a home.

**D2. Run the cheap missing measurements before submission?** Bootstrap confidence intervals
(compute only, the suite accepts a resample count), the student's zero-shot baseline (one
evaluation run), the multi-animal re-score (archived predictions). Each removes a Further
Work item and turns an "unquantified" into a number. The bootstrap alone would let the
chapter say whether the 17-cell direction and the Band B against Band C tie are significant.
Recommendation: at least the bootstrap on the four comparisons the chapter leans on.

**D3. The evaluation-axes revision.** (a) Re-tabulate Chapter 4 with the real test set as
default. (b) Keep the tables, but lead every Chapter 5 verdict and the abstract with the
real-only figure, say once in Limitations that this is the revision the protocol called for
and that the mixed figure is kept as the ranking instrument. (c) Leave as is and keep
apologizing in five places. Recommendation: (b). It is honest, cheap, and stops the repeated
"deviation from protocol" language.

**D4. How to frame Q4.** The AX Visio takes a still after a button press and shows a name.
"Real-time" in that interaction is a response delay, not a frame rate. The chapter currently
says real-time "is not reached" without any reference point. Options: (a) keep the
comparison-only framing (13× faster than the production pairing, 83× than the teacher);
(b) add the response-time framing and let the reader judge 0.4 s against 5.4 s;
(c) add the reasoned quantization estimate from the Krishnamoorthi row. (b) and (c) can be
combined. Whether a response-time expectation exists for the product is the author's
knowledge; the memory rule says no fixed budget existed.

**D5. Licensing of YOLO26n.** Must be addressed (§2.7). Where: inside the deployability
discussion or as a Limitation, and the Conclusion's "shippable" sentence has to agree.

**D6. Which orphan findings to promote.** Recommendation: the classification bottleneck as
its own interpretation subsection; YOLO26n against YOLOv5s into the deployability
discussion and the Conclusion's "previous state against now"; the two-studies-agree
synthesis as the spine of the synthetic-data discussion.

**D7. Which literature comparisons.** From the §2.2 table, one or two sentences each, never
a paragraph per paper. Recommended set: Mirzadeh and Yang (Q1), Azizi and the author's
noise finding (Q2), He and Ravuri (Q2/Q3), Gadot's rollup and WordTree (classification
bottleneck), Krishnamoorthi (Q4), Beery (external validity).

**D8. Further Work as research directions.** Candidate directions in §5.4. Recommendation:
three or four directions, one paragraph each, ordered by what they would change about the
product, plus a single sentence for whatever housekeeping survives D2.

**D9. The resolution runs** (§2.7): results or abandoned?

**D10. The final sentence.** Options in §5.5.

---

## 4. Recommended outline (option C), with word budget

Target ≈ 2,900 words, under the 3,000 ceiling.

**Lead-in (≈ 40 words).** One sentence per section. No "this chapter answers".

**5.1 Answers to the research questions (≈ 450 words, ~110 each).** Optionally keep the
verbatim question list (bachelor pattern) but drop the sentence at 5:10. Each answer: one
verdict sentence with a domain subject, the one or two numbers that carry it (real-only
first under D3b, with the mixed figure beside it), one sentence of scope. No asymmetries, no
sub-limits, no mechanisms here.

- Q1: Direct fine-tuning wins on every measured axis, widest on Band A; holds for one
  response-based configuration at 70:1.
- Q2: gpt-image-2, cheapest tier, full prompt; neither realism nor price predicted it; prompt
  detail did. Ranking holds for data-rich classes only.
- Q3: Half synthetic costs nothing measurable at a 200-image budget, all-synthetic halves
  accuracy; mixed-set figures overstate real accuracy in every band.
- Q4: 0.4 s per still on the proxy CPU at full precision, 13× faster and 22× smaller than
  the production pairing at 0.07 mAP less than the teacher ensemble on real photographs.

**5.2 Interpretation (≈ 1,100 words, four subsections).**

- *5.2.1 Where synthetic imagery helps and where it does not.* The two-studies-agree spine.
  Mechanism argument for Band A (composition, not realism; the teacher's 0.324 → 0.182). The
  mixed-set inflation as the measured version of Chapter 2's warning. Azizi, He, Ravuri, the
  author's earlier noise result. Practical recipe for the product: full prompts, cheapest
  tier, spend synthetic budget on classes with some real images, not on the tail.
- *5.2.2 Classification, not localization.* AR100 against mAP, Δcoarse against Δfine, the
  elephant collapse and imbalance as the candidate cause, the Δcoarse-is-an-upper-bound
  caveat. Wei; the rollup the ensemble has and the single detector lost (Gadot); WordTree.
  This subsection feeds the first outlook item.
- *5.2.3 Why distillation did not help here.* Weigh the candidates (capacity gap, recipe
  transplant, broadcast approximation, crop-versus-frame teacher, batch confound). The
  soft-target premise against the measured Δfine. What the negative rules out and what it
  does not. Mirzadeh, Yang.
- *5.2.4 Deployability.* YOLO26n against YOLOv5s and both ensembles. Response-time framing
  (D4). Reasoned quantization estimate, labelled as reasoned (Krishnamoorthi). Decode cost of
  native stills. Licensing (D5).

**5.3 Limitations (≈ 550 words, five paragraphs).** Grouped by what they bound, each with
its consequence in one clause:

1. Statistical: single runs, no confidence intervals, the seed reversal by `\Cref`.
2. Design confounds: bands hold different species; the distillation pair differs in batch
   size and convergence.
3. Evidence base: automatic labels for every model, the teacher's circular localization, the
   mixed-set inflation and the once-stated protocol outcome (D3).
4. External validity: proxy CPU, no DSP, no device stills, no architecture comparison, 17
   thin Band A classes, licensing if not handled in 5.2.4.
5. Generator study resolution: ordinal only, rare classes unresolved, rubric not run.

**5.4 Further Work (≈ 450 words).** Three or four research directions (candidates below),
then one sentence for the deferred measurements if D2 leaves any.

**5.5 Conclusion (≈ 350 words).** Problem in one sentence. What was built in two (dataset
and curation, twelve-cell generator study, four trained models, proxy benchmark). Three
findings in three sentences with at most one number each. Previous state against now in one
sentence (production pairing against YOLO26n). Final sentence on the contribution (D10).

**Option A sketch.** Same as C without 5.2; each answer grows to ~200 words carrying its own
mechanism and one citation; Limitations and Further Work as above. Fits the budget but
leaves the orphan findings homeless.

**Option B sketch.** 5.1 becomes the four thematic subsections; 5.2 "Answers to the research
questions" is a 150-word list of verdicts with `\Cref`s back into 5.1. Then Limitations,
Further Work, Conclusion. Change the Chapter 1 sentence that promises per-question answers.

---

## 5. Content menu: what could be mentioned, by section

Tags: **keep** (in the chapter now, stays), **promote** (in Chapter 4, absent here),
**new** (not in the manuscript yet), **cut** (in the chapter now, goes).

### 5.1 Answers

- keep: the four verdicts and their headline numbers.
- cut: asymmetries, sub-limits, "instrument cannot rank" sentence (moves to 5.2.1 once),
  the Hexagon paragraph (moves to 5.2.4), the Summary.

### 5.2.1 Synthetic data

- promote: the two studies agree (§4.4); the useful range is the middle band.
- promote: teacher fine-tuned on Band A drops 0.324 → 0.182.
- promote: one centred animal per synthetic image; 1.00 vs 1.44 boxes.
- new: mechanism argument (composition and diversity over realism), labelled reasoned.
- new: Azizi (bare prompts, alignment over fidelity), He (5× and pretrained absorption),
  Ravuri (per-class zeros), Tobin/Tremblay (realism unnecessary), the author's noise result.
- new: product recipe (full prompt, cheapest tier, target Band-B-like classes).
- new: what the incumbent choice cost, stated neutrally (third of twelve, chosen on the
  criterion the literature calls unreliable).

### 5.2.2 Classification bottleneck

- promote: AR100 0.845/0.798 against mAP 0.599/0.529; Δcoarse 16–22× Δfine; 19.3 % in-group
  confusion, five groups carry 60 %; elephant 1,545 vs 9 with the larger test count on the
  losing side.
- promote: Δcoarse is an upper bound (error types not separated).
- new: the rollup argument (Gadot) and WordTree; a hierarchical label space or loss as the
  route to coarse-correct fallbacks.
- new: imbalance remedies the thesis surveyed and did not use (Kang's classifier
  re-adjustment) as the natural fix for one-directional collapse.
- new: Wei on fine-grained cues.

### 5.2.3 Distillation

- keep: the 17-cell direction, Band A widest, one configuration at 70:1.
- promote: the broadcast approximation and probability-cache temperature (from `sec:kd_loss`),
  the crop-versus-frame teacher, the multi-animal subset never scored.
- new: which cause is most likely and why (author's judgment, stated as such).
- new: the soft-target premise against the measured Δfine.
- new: Mirzadeh (teacher assistant), Yang (detection-specific distillation needs a
  compatible teacher).

### 5.2.4 Deployability

- promote: YOLO26n against YOLOv5s on all three axes; sensitivity checks 0.594 / 0.597.
- keep: 13× and 83×, 22× and 71×, 0.07 mAP on real photographs against the teacher.
- new: response-time framing for a button-press still (D4).
- new: reasoned quantization estimate (D4c).
- promote: decode cost of native stills, 256–275 ms.
- new: licensing (D5).
- new: the per-crop latency scaling of the ensemble confirmed (1.7 crops per image).

### 5.3 Limitations

See the table in §2.4 for keep/merge/cut, and the missing-limitations list.

### 5.4 Further Work candidates

Ordered by how much they would change the product, not by cost:

1. **Hierarchical label space in the single detector** (rollup to genus or family below a
   confidence threshold, WordTree-style or a hierarchical loss). Targets the dominant error
   and restores the fallback the ensemble had. The thesis has the frozen look-alike table
   and the per-pair confusion counts to design it from.
2. **Distillation with a compatible teacher.** A large single-stage YOLO fine-tuned on the
   same data as teacher, enabling feature and attention distillation (FGD) and closing the
   capacity gap; per-detection teacher vectors for multi-animal images. Equal batch size.
3. **Synthetic data spent differently.** Target the middle band; generator fine-tuning on the
   real corpus instead of prompting; semantic editing of real photographs for classes with
   few real images; compositional diversity (multi-animal, off-centre, occlusion) by design.
4. **Quantized measurement on the Hexagon** (PTQ, then QAT via SNPE and AIMET), plus the
   resolution arm (or its results, D9), plus evaluation on device stills.
5. **Class-imbalance remedies for one-directional collapses** (classifier re-adjustment,
   re-weighting), tested on the elephant and gazelle pairs.
6. **The class-count ceiling.** The 200–250 class figure is the thesis's own extrapolation
   (Chapter 2); a class-count sweep at fixed capacity would test it.
7. Housekeeping, one sentence if D2 leaves any: bootstrap, zero-shot baseline, multi-animal
   re-score, the rubric.

### 5.5 Conclusion

- keep: problem, what was built, the three findings, the size and speed contrast.
- cut: every number that appears in 5.1 except the three the findings need; "The
  substantive contribution sits in…"; the effort-distribution sentence; the two verbatim
  "in the first place" sentences.
- new: previous state against now in one sentence (production pairing against YOLO26n).
- D10, final sentence options:
  (a) the contribution: a nano-scale detector that locates and names 225 species at a
  fraction of the ensemble's size and time, with a training recipe whose useful range for
  synthetic data is now measured;
  (b) the product consequence: the ensemble cannot ship on this hardware and the single
  detector can, pending compression;
  (c) the open question, stated as a question rather than as a thing not done: how much of
  the remaining gap the accelerated path closes.
  Recommendation: (a), with (b) as the sentence before it.

---

## 6. Cut list

Passages in the current file that should go entirely, with the reason:

| Line | Passage | Reason |
|---|---|---|
| 5:10 | "The research questions are restated verbatim and answered one at a time." | Signposting |
| 5:22, 5:54 | The asymmetries and the four gaps, in full | Chapter 4 owns them; one `\Cref` |
| 5:37, second half | Hexagon has no analogue, order-of-magnitude gains | Identical to §4.3.2; reappears in 5.2.4 as an estimate |
| 5:39–40 | Summary paragraph | Meta-commentary, no fact |
| 5:45 | "Each of the answers above is limited by the constraints below, each of which is stated with its consequence." | Signposting |
| 5:54 | "failed due to a missing data directory and was not retried" | Bug post-mortem |
| 5:63 | The six sub-limits | Chapter 4 material |
| 5:68–72 | Error types; training cost | Moves / no answer depends on it |
| 5:85 | Further Work item 6 | Not research; resolved by D3 |
| 5:88 | "The first item decides whether the system is deployable at all. The rest decide how well it would perform once it is." | Rhetoric, restates item 1 |
| 5:93 | Effort distribution sentence | §1.2 already says it |
| 5:95–99 | Conclusion paragraphs 2–4 as written | Restate 5.1 with the same numbers |
| 5:27, 5:95 | "…in the first place" | Same sentence twice, and a third time in §4.1.6 |

---

## 7. Rules to hold while rewriting

- Every claim typed: measured (with `\Cref`), cited, reasoned (premises in the same
  paragraph), or not established and said so (`claims.md` §1). The mechanism arguments in
  5.2 are reasoned claims and must read as such.
- One emphatic sentence per section, spent on a finding, with a domain subject. Run the
  deletion test on every short sentence (supervisor memory).
- No "--", no ";" in prose. No "and" in headings, no verbless headings.
- A citation is a tag on a stated claim, never a subject. In this chapter it is always an
  external comparison.
- The real-only figure beside every mixed figure; under D3b, real first.
- No number from Chapter 4 reappears unless the verdict cannot be stated without it.
- No "not done" caveat in the Conclusion; state limits once, in Limitations.
- Keep the file at or under 3,000 words (`uv run python -m scripts.thesis.check_manuscript`).
