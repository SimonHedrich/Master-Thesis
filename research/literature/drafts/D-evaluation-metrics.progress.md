# Progress: Section D — Evaluation Metrics for Fine-Grained Wildlife Detection

## Planned subsection breakdown

1. **General Object Detection Metrics** — `\label{sec:lit_detection_metrics}`
   IoU, precision/recall, interpolated AP, the COCO AP@[.5:.95]/AP50/AP75/AR vector.
2. **Hierarchical and Coarse-to-Fine Evaluation for Confusable Classes** — `\label{sec:lit_hierarchical_evaluation}`
   Taxonomic-rollup vs empirical-confusion-driven grouping; a-priori semantic grouping precedent.
3. **Error-Type Decomposition** — `\label{sec:lit_error_decomposition}`
   Localization vs classification/confusion vs background error taxonomies (TIDE-style diagnostics).
4. **Evaluating Across a Real/Synthetic Test-Domain Split** — `\label{sec:lit_real_synthetic_evaluation}`
   Precedent search for mixed real+synthetic headline test sets (confirmed gap) + adjacent literature.

## Citekeys planned per subsection (bib-verified via grep before use)

### Subsection 1 — General Object Detection Metrics
- [x] read, supports: `everinghamPascalVisualObject2010` — IoU (intersection/union, 50% threshold), precision/recall definitions, interpolated 11-point AP definition (PASCAL VOC 2007 protocol). Confirmed present in references.bib.
- [x] read, supports: `zouObjectDetection202023` — historical evolution of detection metrics (FPPW -> FPPI -> AP -> mAP@0.5 -> COCO's AP averaged over IoU 0.5:0.95). Confirmed present.
- [x] read, supports: `linMicrosoftCOCOCommon2015` — introduces the COCO benchmark itself; NOTE: the archived arXiv v1 text explicitly says "a full discussion of evaluation metrics will be added once the evaluation server is complete" — so the specific AP50/AP75/AR@1/10/100 vector claim is NOT attributed to this source directly, only the benchmark/dataset itself. Confirmed present.

### Subsection 2 — Hierarchical and Coarse-to-Fine Evaluation for Confusable Classes
- [x] read, supports: `hoiemDiagnosingErrorObject2012` — defines a priori semantic similarity sets ({all animals}, {all vehicles}, etc.) and shows "confusion with similar objects" is one of the most impactful error types — direct precedent for a-priori (non-circular) look-alike/coarse grouping, used as the closest available anchor. NOTE: no dedicated taxonomic-rollup-vs-confusion-driven-grouping paper exists in this corpus (Koul et al. 2021 and Bertinetto et al. 2020, both flagged as relevant in claude_literature.md, have NO entry in references.bib — verified absent by grep, see Unresolved below). Gap stated plainly in the prose.

### Subsection 3 — Error-Type Decomposition
- [x] read, supports: `hoiemDiagnosingErrorObject2012` — four-way false-positive taxonomy (localization error / confusion with similar objects / confusion with other labeled objects / confusion with background) — the direct intellectual precedent for later automated multi-category error-decomposition toolboxes. NOTE: TIDE (Bolya et al., ECCV 2020) has NO entry in references.bib despite being marked "verified" in claude_literature.md and gemini_literature.md, and despite `\gls{tide}` already being predefined in `thesis/manuscript/preamble/acronyms.tex` and referenced in the Results chapter TODO. Grepped exhaustively (author names, DOI fragments, "TIDE", "toolbox") — absent. Cannot cite it under the citekey rule. Handled by grounding the subsection entirely in Hoiem et al. 2012 and deferring the *name* of the specific modern toolbox used operationally to the Methods/Results chapters. Flagged as unresolved below.

### Subsection 4 — Evaluating Across a Real/Synthetic Test-Domain Split
- [x] read, supports: `UnbiasedLookDataset` (Torralba & Efros, CVPR 2011 "Unbiased Look at Dataset Bias" — confirmed same paper via DOI/URL cross-check against gemini_literature.md's torralba_unbiased_2011 identifier, DOI 10.1109/CVPR.2011.5995347) — bib entry is metadata-thin (title+URL only, no abstract/authors, no source PDF in sources/) — cited minimally, for the general dataset-bias/cross-dataset-generalization framing only.
- [x] read, supports: `tremblayTrainingDeepNetworks2018` — concrete sim-to-real precedent: trains on domain-randomized synthetic data, evaluates *exclusively* on the real-world KITTI test set (never mixes synthetic images into the test set) — standard practice this thesis departs from.
- [x] read, supports: `heSyntheticDataGenerative2023` — evaluates synthetic-data-trained/pretrained models strictly against standard, established real test sets (CIFAR-100, PASCAL VOC); identifies domain gap as the mechanism behind synthetic-vs-real performance differences.
- [x] read, supports: `zhouDomainGeneralizationSurvey2023` — general domain-shift/OOD-generalization framing (train/test distribution mismatch), used only as background vocabulary for why a real-vs-synthetic delta is a meaningful watchdog signal, not as support for the mixed-set methodology itself.
- Explicit statement: no established citable precedent found for (a) reporting a headline metric over a real∪synthetic test-set union, or (b) using a class-balanced synthetic set as a diagnostic instrument. Presented as this thesis's own explicit, justified methodological choice (per claude_literature.md's confirmed gap finding), anchored to `docs/plans/2026-06-10_model-evaluation-strategy.md`.

## Status
All planned subsections drafted and appended to `D-evaluation-metrics.tex`.

## Unresolved / flagged issues
1. **TIDE (Bolya et al., ECCV 2020) has no entry in `research/literature/references.bib`.** Both survey docs (`claude_literature.md` line 38, `gemini_literature.md` §4.2-01) list it as "verified", and `thesis/manuscript/preamble/acronyms.tex` already predefines `\gls{tide}` = "Toolbox for Identifying Detection Errors", and `4-Results.tex`'s TODO already forward-references "a TIDE error-type chart" — implying the thesis intends to use/cite it. I could not add a bib entry (out of scope / forbidden) or fabricate the key. Subsection 3 is grounded entirely in Hoiem et al. (2012) instead, and does not name "TIDE" in the prose. **Recommend**: add a `bolya`-keyed entry to `references.bib` (DOI 10.1007/978-3-030-58580-8_33 per claude_literature.md, or 10.1007/978-3-030-58595-2_33 per gemini_literature.md — the two survey docs disagree on the DOI's last digits, so verify before adding) so that Methods/Results can cite it properly.
2. Also absent from the bib despite being named in the task brief / survey docs as candidates: Everingham et al. 2015 PASCAL retrospective, Hosang et al. 2016, Oksuz et al. 2021 (LRP metric — note `oksuzImbalanceProblemsObject2020` IS present but is a *different* Oksuz paper, "Imbalance Problems in Object Detection: A Review", not the LRP metric paper — not used here, weak fit), Bertinetto et al. "Making Better Mistakes" CVPR 2020, Koul et al. 2021 "Taxonomic Loss Functions", Koch et al. 2021, DeGrave et al. 2021, Hataya et al. ICCV 2023. None fabricated; all simply omitted or their claims dropped/softened.

## Citekeys used
- everinghamPascalVisualObject2010
- zouObjectDetection202023
- linMicrosoftCOCOCommon2015
- hoiemDiagnosingErrorObject2012
- UnbiasedLookDataset
- tremblayTrainingDeepNetworks2018
- heSyntheticDataGenerative2023
- zhouDomainGeneralizationSurvey2023
