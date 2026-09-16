# Literature Review drafting — progress

Plan: ~/.claude/plans/recap-built-a-pipeline-delegated-wreath.md

Read this file first if resuming after an interruption. Resume at the first unchecked step below,
then re-read the plan file's matching numbered section for that step's exact instructions.

## Steps
- [x] 0. Backup original TODO skeleton -> research/literature/drafts/2-Literature_Review.tex.orig
- [x] 1. Section A draft (Object Detection + Model Selection) -> research/literature/drafts/A-object-detection.tex
- [x] 2. Section B draft (Synthetic Training Data) -> research/literature/drafts/B1-generative-and-simtoreal.tex + B2-evaluation-and-longtail.tex (split into two files, see notes below; no single B-synthetic-data.tex was ever written)
- [x] 3. Section C draft (KD + Embedded Inference) -> research/literature/drafts/C-kd-embedded.tex
- [x] 4. Section D draft (Evaluation Metrics) -> research/literature/drafts/D-evaluation-metrics.tex
- [x] 5. Chapter lead-in paragraph written
- [x] 6. Assembled into thesis/manuscript/chapters/2-Literature_Review.tex
- [x] 7. Citekey verification pass — all 79 unique citekeys used resolve in research/literature/references.bib or thesis/manuscript/bibliography/references.bib. Zero missing/hallucinated keys.
- [x] 8. Coherence read-through — full chapter (179 lines) read end-to-end. All internal \Cref{} targets resolve to labels defined in this chapter; all external \Cref{} targets (Methods/Results chapters) resolve to labels that already exist there. No local LaTeX toolchain available to compile (expected — Overleaf is the actual compile target); skipped per plan (nice-to-have, not blocking).
- [x] 9. Handed to user for review — see final report in conversation.

## OUTSTANDING ITEMS FOR THE USER (not blockers, but need action before submission)
1. **Fill the three placeholder bib stubs** in `thesis/manuscript/bibliography/references.bib`
   (`gadot_speciesnet_2024`, `beery_megadetector_2019`, `jocher_yolov5_2020`) — currently just
   `{title=..., note={TODO: authors, year, venue}}`. For SpeciesNet, `gadotCropNotCrop2024` in the
   Zotero corpus (IET Computer Vision, DOI 10.1049/cvi2.12318) is confirmed to be the real
   peer-reviewed paper — consider using its data to fill the stub or unifying the two keys.
2. **Add a real TIDE (Bolya et al., ECCV 2020) bib entry** — confirmed absent from both bib files
   despite being referenced by `thesis/manuscript/preamble/acronyms.tex`'s `\gls{tide}` and the
   Results chapter's own TODO. Section D's error-decomposition subsection was written to avoid
   naming "TIDE" without a citation, but Methods/Results will need this citation when written.
3. Both flags above are Overleaf/bibliography-side fixes, consistent with "no .bib editing" scope
   for this task.

## Per-section citekey logs (filled in as each section drafts)

### Section A — Object Detection for Wildlife Monitoring + Model Selection
DONE. Draft in `drafts/A-object-detection.tex`, full checklist in `drafts/A-object-detection.progress.md`.
Added a new `\subsection{Challenges in Wildlife Object Detection}\label{sec:lit_wildlife_challenges}`
(TODO brief asked for a "Challenges" close but the skeleton had no subsection header for it).

Citekeys used (31, all verified verbatim against their source .bib):
girshickRichFeatureHierarchies2014a, uijlingsSelectiveSearchObject2013, girshickFastRCNN2015a,
renFasterRCNNRealTime2017, linFeaturePyramidNetworks2017, redmonYouOnlyLook2016a,
liuSSDSingleShot2016a, linFocalLossDense2017, zouObjectDetection202023,
weiFineGrainedImageAnalysis2022, cuiClassBalancedLossBased2019, zhangDeepLongTailedLearning2023,
beeryRecognitionTerraIncognita2018, williIdentifyingAnimalSpecies2019,
tabakMachineLearningClassify2019, norouzzadehAutomaticallyIdentifyingCounting2018,
tuiaPerspectivesMachineLearning2022, redmonYOLO9000BetterFaster2017,
redmonYOLOv3IncrementalImprovement2018, jocher_yolov5_2020 (legacy), nazirYouOnlyLook2023,
geYOLOXExceedingYOLO2021, liYOLOv6SingleStageObject2022, wangYOLOv9LearningWhat2025,
wangYOLOv10RealTimeEndtoEnd2024, khanamYOLOv11OverviewKey2024, tianYOLOv12AttentionCentricRealTime2025,
yuPPPicoDetBetterRealTime2021, beery_megadetector_2019 (legacy), gadot_speciesnet_2024 (legacy),
gadotCropNotCrop2024, rangilyuRangiLyuNanodet2026.

**FLAG for user, needs action before Overleaf sync:** the three "legacy" keys in
`thesis/manuscript/bibliography/references.bib` are currently EMPTY PLACEHOLDER STUBS
(`@misc{gadot_speciesnet_2024, title = {SpeciesNet}, note = {TODO: authors, year, venue}}` — same
pattern for the other two). They must be filled in with real bibliographic data before the chapter
compiles/renders citations correctly. For SpeciesNet specifically: `gadotCropNotCrop2024` in
`research/literature/references.bib` (Gadot et al., *IET Computer Vision* 18(8):1193-1208, DOI
10.1049/cvi2.12318, "To crop or not to crop") is confirmed by `claude_literature.md` to be the
actual peer-reviewed SpeciesNet paper — consider using its real bibliographic data to fill the
`gadot_speciesnet_2024` stub (or unify the two keys) rather than leaving it a placeholder.

Softened/dropped: YOLOv4 named but not separately cited (no corpus entry, legacy key off-limits).
YOLO26 discussed only qualitatively, explicit note that no independent analysis of it exists yet.
EfficientDet-Lite named with no citation (no bib entry anywhere). The ~200-250 class ceiling is
written as an explicit literature gap (no direct paper establishes it), with PicoDet/YOLOX-Nano
capacity-accuracy figures cited only as indirect evidence.

### Section B — Synthetic Training Data for Visual Recognition
IN PROGRESS. Note: three whole-section attempts hit a recurring `[bio]` API safety false-positive
(benign CV/generative-AI content, reported via SendFeedback) right at the prose-drafting step, so
this section was split into B1 (subsections 1-2) and B2 (subsections 3-4), each its own agent.

**B1 — sec:lit_generative_models + sec:lit_sim_to_real: DONE** despite the launching agent's run
itself being reported "failed" (hit the same `[bio]` error on its very last step, after saving
output) — verified complete on disk: `drafts/B1-generative-and-simtoreal.tex` (both subsections,
full prose) + `drafts/B1-generative-and-simtoreal.progress.md` (19/20 citekeys marked read; the
20th, `openaiDALLECreatingImages2021`, is used only for a minimal non-archival lineage mention, and
its key existence was independently grep-verified). Citekeys used: goodfellowGenerativeAdversarialNetworks2014,
radfordUnsupervisedRepresentationLearning2016, karrasStyleBasedGeneratorArchitecture2019,
hoDenoisingDiffusionProbabilistic2020, songDenoisingDiffusionImplicit2022,
hoClassifierFreeDiffusionGuidance2022, rombachHighResolutionImageSynthesis2022,
podellSDXLImprovingLatent2023, esserScalingRectifiedFlow2024, labsBlackForestLabs,
radfordLearningTransferableVisual2021, raffelExploringLimitsTransfer2023, luoLatentConsistencyModels2023,
openaiDALLECreatingImages2021, tobinDomainRandomizationTransferring2017, tremblayTrainingDeepNetworks2018,
heSyntheticDataGenerative2023, trabuccoEffectiveDataAugmentation2025, shumailovAIModelsCollapse2024,
aziziSyntheticDataDiffusion2023.

**B2 — sec:lit_synthetic_evaluation + sec:lit_synthetic_long_tail: DONE**, no interruption this
time. `drafts/B2-evaluation-and-longtail.tex` + `.progress.md` complete. Citekeys used:
salimansImprovedTechniquesTraining2016, heuselGANsTrainedTwo2018, binkowskiDemystifyingMMDGANs2021,
zhouHYPEBenchmarkHuman2019, hesselCLIPScoreReferencefreeEvaluation2021,
ravuriClassificationAccuracyScore2019, zhangDeepLongTailedLearning2023,
chawlaSMOTESyntheticMinority2002, cuiClassBalancedLossBased2019, caoLearningImbalancedDatasets2019,
kangDecouplingRepresentationClassifier2020. Dropped `weiFineGrainedImageAnalysis2022` for this
subsection (re-checked full text: no actual long-tail/imbalance discussion in body, only in its
own bibliography) despite the pre-discovery list suggesting it — correct call, don't re-add it here.

**Section B overall: all four subsections now drafted across B1 + B2.** Step 2 (assembly) will
concatenate: intro paragraph (write fresh) + B1's two subsections + B2's two subsections, in that
order, under one `\section{Synthetic Training Data for Visual Recognition}\label{sec:lit_synthetic_data}`.

**Original whole-section files** `drafts/B-synthetic-data.progress.md` (citekey discovery only, no
prose) and (never created) `drafts/B-synthetic-data.tex` — the `.progress.md` is being kept as a
reference for B2's citekey shortlist; final assembly (step 6) will pull prose from
`B1-generative-and-simtoreal.tex` + `B2-*.tex` instead of a single `B-synthetic-data.tex`.

### Section C — Knowledge Distillation and Embedded Inference Feasibility
DONE, no interruptions. `drafts/C-kd-embedded.tex` + `.progress.md` complete, appropriately concise
("foundational depth only" framing honored throughout).

Citekeys used:
KD Fundamentals: buciluaModelCompression2006, hintonDistillingKnowledgeNeural2015,
gouKnowledgeDistillationSurvey2021 (author corrected: Gou not Guo), romeroFitNetsHintsThin2015,
yangFocalGlobalKnowledge2022 (FGD, lead author Zhendong Yang, correctly attributed),
mirzadehImprovedKnowledgeDistillation2020 (capacity-gap/teacher-assistant, mentioned only in a
one-paragraph "not pursued" acknowledgment per the TODO's constraint).
Embedded Inference Feasibility: howardMobileNetsEfficientConvolutional2017, yuPPPicoDetBetterRealTime2021,
liPruningFiltersEfficient2017, reddiMLPerfInferenceBenchmark2020 (no source PDF, cited from bib
abstract only), banburyMLPerfTinyBenchmark2021, davidTensorFlowLiteMicro2021,
krishnamoorthiQuantizingDeepConvolutional2018, jacobQuantizationTrainingNeural2018,
QualcommAimet2026 (vendor doc, explicitly flagged in-text as such).

Notable: both quantization papers (jacobQuantizationTrainingNeural2018,
krishnamoorthiQuantizingDeepConvolutional2018) explicitly target Qualcomm Hexagon/QDSP hardware in
their own text — strong direct grounding for the Hexagon 685 discussion, found without needing the
unavailable SNPE/AIMET vendor docs.

Softened: "thread/affinity sensitivity" and "cold-start effects" pitfalls softened from the TODO's
specific framing since no source directly evidences ARM big.LITTLE affinity variance or cold-start
per se — phrased more cautiously, attributed to general practice rather than a specific paper.

### Section D — Evaluation Metrics for Fine-Grained Wildlife Detection
DONE, no interruptions. `drafts/D-evaluation-metrics.tex` + `.progress.md` complete. This section
had no subsections yet in the skeleton; the agent designed and added four:
1. General Object Detection Metrics -> `sec:lit_detection_metrics`
2. Hierarchical and Coarse-to-Fine Evaluation for Confusable Classes -> `sec:lit_hierarchical_evaluation`
3. Error-Type Decomposition -> `sec:lit_error_decomposition`
4. Evaluating Across a Real/Synthetic Test-Domain Split -> `sec:lit_real_synthetic_evaluation`

Citekeys used: everinghamPascalVisualObject2010, zouObjectDetection202023, linMicrosoftCOCOCommon2015,
hoiemDiagnosingErrorObject2012 (used for both subsections 2 and 3), tremblayTrainingDeepNetworks2018,
heSyntheticDataGenerative2023, zhouDomainGeneralizationSurvey2023, UnbiasedLookDataset (Torralba &
Efros — bib entry has no author/abstract field and no source PDF; cited minimally).

**FLAG for user — IMPORTANT, needs action:** TIDE (Bolya et al., ECCV 2020) has NO entry in
`research/literature/references.bib`, despite being marked "verified" in both survey docs, AND
despite `thesis/manuscript/preamble/acronyms.tex` already defining `\gls{tide}` and the Results
chapter's own TODO already forward-referencing "a TIDE error-type chart." The agent grepped
exhaustively (author names, DOI fragments, "TIDE") in both `research/literature/references.bib`
and `thesis/manuscript/bibliography/references.bib` and confirmed it's genuinely absent from both —
did NOT fabricate a key. Subsection 3 (`sec:lit_error_decomposition`) is grounded entirely in
Hoiem et al. (2012) instead and does not name "TIDE" anywhere in the prose, to avoid an uncited
name-drop. **Action needed before Methods/Results chapters cite TIDE by name: add a real
Bolya/TIDE bib entry** (the two survey docs disagree on its DOI's last digits — verify when adding,
likely DOI 10.1007/978-3-030-58580-8_33 or arXiv 2008.08115 per `claude_literature.md`).

Also confirmed absent from the bib (all simply omitted from prose, not fabricated): PASCAL VOC 2015
retrospective, Hosang et al. 2016, Oksuz LRP 2021, Bertinetto et al. 2020 "Making Better Mistakes",
Koul/Koch/DeGrave et al. 2021, Hataya et al. 2023.

Softened/dropped: subsection 2 states plainly no paper in the corpus performs taxonomic rollup at
species/genus/family resolution or compares it against empirical-confusion grouping — grounded only
in Hoiem et al.'s general a-priori-grouping principle. Subsection 4 (the confirmed literature gap,
mixed real+synthetic test sets) explicitly states no precedent exists and uses adjacent literature
(domain randomization, synthetic-data-readiness, domain-generalization survey, dataset-bias) only as
contrast/context for what standard practice looks like, never as endorsement.

## Fixed citekey rules (apply to all sections)
- Use exactly `jocher_yolov5_2020`, `beery_megadetector_2019`, `gadot_speciesnet_2024` for those
  three topics — they exist only in `thesis/manuscript/bibliography/references.bib`, not in the
  Zotero corpus (confirmed via grep, no overlap).
- Every other citation: copy the citekey verbatim from `research/literature/references.bib`
  (Better BibTeX camelCase style, e.g. `zhangShuffleNetExtremelyEfficient2018`). Never guess or
  reformat a key.
- Do not touch any `.bib` file — Overleaf owns bibliography sync.
