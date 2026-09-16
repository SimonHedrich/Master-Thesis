# Progress: Section C — Knowledge Distillation and Embedded Inference Feasibility

## Subsection 1: Knowledge Distillation Fundamentals (sec:lit_knowledge_distillation)

- [x] hintonDistillingKnowledgeNeural2015 — read, supports: soft-target/temperature-scaling formulation, dark knowledge, KD loss (soft CE + hard CE)
- [x] buciluaModelCompression2006 — read, supports: precursor "model compression" concept (Bucila et al. 2006), ensemble mimicking predating Hinton's formalization
- [x] gouKnowledgeDistillationSurvey2021 — read (first ~300 lines), supports: response-based vs. feature-based vs. relation-based knowledge taxonomy; correct author Jianping Gou (not Guo)
- [x] romeroFitNetsHintsThin2015 — read, supports: feature-based distillation example (FitNets hint/guided-layer matching)
- [x] yangFocalGlobalKnowledge2022 — read, supports: object-detection-specific adaptation (FGD), why plain logit-matching/equal-weighting fails without spatial/foreground-background awareness; correct author Zhendong Yang
- [x] mirzadehImprovedKnowledgeDistillation2020 — read, supports: one-paragraph acknowledgment of capacity-gap / teacher-assistant work not pursued

## Subsection 2: Embedded Inference Feasibility (sec:lit_embedded_inference)

### (a) lightweight architecture design principles
- [x] howardMobileNetsEfficientConvolutional2017 — read (abstract/bib), supports: depthwise separable convolutions as core efficiency primitive
- [x] liPruningFiltersEfficient2017 — read (abstract/bib), supports: channel/filter pruning concept
- [x] yuPPPicoDetBetterRealTime2021 — read (abstract/bib), supports: anchor-free heads in nano-scale detectors, ties to model-selection recap

### (b) on-device benchmarking methodology
- [x] banburyMLPerfTinyBenchmark2021 — read (abstract/bib), supports: standardized latency/accuracy/energy measurement protocol for embedded ML
- [x] davidTensorFlowLiteMicro2021 — read (abstract/bib), supports: export/interpreter-based deployment framework, embedded fragmentation, cross-platform interoperability
- [x] reddiMLPerfInferenceBenchmark2020 — no source file (not in Zotero PDF library); cited only for what bib abstract establishes (general standardized ML-inference benchmarking across architecture-neutral hardware), said briefly per grounding rules

### (c) quantization + Hexagon/SNPE toolchain (explicitly scoped down)
- [x] jacobQuantizationTrainingNeural2018 — read (abstract/bib), supports: integer-only inference quantization scheme + joint QAT training procedure (PTQ vs QAT)
- [x] krishnamoorthiQuantizingDeepConvolutional2018 — read (abstract/bib), supports: PTQ whitepaper, explicitly benchmarks "Qualcomm QDSPs with HVX" speedups — direct grounding for Hexagon DSP mention
- [x] QualcommAimet2026 — no source file (vendor GitHub repo, no DOI); cited as vendor documentation only, flagged explicitly in prose

### (d) hardware-proxy recap
- no citation needed; recap of docs/2026-03-09_hardware-proxy-selection.md (internal project document, not a literature source)

## Softened/dropped claims
- Subsection 2(b): "thread/affinity sensitivity" pitfall named in the TODO is not directly evidenced by any source in the corpus as an on-device-benchmarking pitfall per se; softened to lean on davidTensorFlowLiteMicro2021's documented multi-thread/multi-core interpreter design discussion (Sec. 4.6 "Multithreading") as partial grounding, phrased as "underscoring that thread and core-affinity choices ... are a further variable" rather than asserting it as an established literature finding.
- Subsection 2(b): "cold-start effects" softened to "one-time effects (initial model loading, cache and frequency-scaling warm-up)" and described as "standard practical concern" rather than attributed to a specific source, since neither reddiMLPerfInferenceBenchmark2020 nor banburyMLPerfTinyBenchmark2021 discusses cold-start explicitly (banbury only documents excluding pre-/post-processing from the measurement window, which is what is directly cited).
- reddiMLPerfInferenceBenchmark2020 has no source file in research/literature/sources/ (INDEX.md: "no source PDF found in Zotero library"); cited only for the general architecture-neutral benchmarking methodology established in its bib abstract, per the grounding rules for non-archival/unavailable sources.
- QualcommAimet2026 has no source file (vendor GitHub repo, no DOI); cited only to name the toolchain, explicitly flagged in-text as "vendor documentation rather than peer-reviewed literature."

## Citekeys used

### Subsection 1 — Knowledge Distillation Fundamentals (sec:lit_knowledge_distillation)
- buciluaModelCompression2006
- hintonDistillingKnowledgeNeural2015
- gouKnowledgeDistillationSurvey2021
- romeroFitNetsHintsThin2015
- yangFocalGlobalKnowledge2022
- mirzadehImprovedKnowledgeDistillation2020

### Subsection 2 — Embedded Inference Feasibility (sec:lit_embedded_inference)
- howardMobileNetsEfficientConvolutional2017
- yuPPPicoDetBetterRealTime2021
- liPruningFiltersEfficient2017
- reddiMLPerfInferenceBenchmark2020 (no source file — cited from bib abstract only)
- banburyMLPerfTinyBenchmark2021
- davidTensorFlowLiteMicro2021
- krishnamoorthiQuantizingDeepConvolutional2018
- jacobQuantizationTrainingNeural2018
- QualcommAimet2026 (no source file — vendor documentation, flagged in-text)

All 15 keys verified verbatim against research/literature/references.bib via grep before use. None of the three excluded "legacy" keys (jocher_yolov5_2020, beery_megadetector_2019, gadot_speciesnet_2024) were used.
