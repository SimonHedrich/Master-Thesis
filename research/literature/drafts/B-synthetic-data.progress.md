# Progress: Section B — Synthetic Training Data for Visual Recognition

## sec:lit_generative_models (Generative Models for Image Synthesis)
- [ ] goodfellowGenerativeAdversarialNetworks2014 — GAN, adversarial minimax framework
- [ ] radfordUnsupervisedRepresentationLearning2016 — DCGAN, stable conv architecture constraints
- [ ] karrasStyleBasedGeneratorArchitecture2019 — StyleGAN, style-based generator
- [ ] hoDenoisingDiffusionProbabilistic2020 — DDPM
- [ ] songDenoisingDiffusionImplicit2022 — DDIM, faster sampling
- [ ] hoClassifierFreeDiffusionGuidance2022 — classifier-free guidance
- [ ] rombachHighResolutionImageSynthesis2022 — Latent Diffusion / Stable Diffusion
- [ ] podellSDXLImprovingLatent2023 — SDXL
- [ ] esserScalingRectifiedFlow2024 — SD3, rectified flow transformer (MMDiT)
- [ ] labsBlackForestLabs — FLUX.1, non-archival (model card only)
- [ ] radfordLearningTransferableVisual2021 — CLIP, text-conditioning backbone for SD/SDXL
- [ ] raffelExploringLimitsTransfer2023 — T5, text encoder used by FLUX/SD3
- [ ] luoLatentConsistencyModels2023 — LCM, few-step distilled sampling
- [ ] openaiDALLECreatingImages2021 — DALL-E / GPT-image lineage, non-archival (model card only)

## sec:lit_sim_to_real (Synthetic Data for Object Detection and the Sim-to-Real Gap)
- [ ] tobinDomainRandomizationTransferring2017 — domain randomization
- [ ] tremblayTrainingDeepNetworks2018 — domain randomization for detection, real+synthetic mix
- [ ] heSyntheticDataGenerative2023 — "Is synthetic data from generative models ready for image recognition?"
- [ ] trabuccoEffectiveDataAugmentation2025 — DA-Fusion, semantics-preserving diffusion augmentation
- [ ] shumailovAIModelsCollapse2024 — model collapse / self-consuming generative loops
- [ ] aziziSyntheticDataDiffusion2023 — synthetic diffusion data improves ImageNet classification (also failure modes)

## sec:lit_synthetic_evaluation (Evaluating Synthetic Image Quality)
- [ ] zhouHYPEBenchmarkHuman2019 — HYPE, structured/blind human perceptual evaluation protocol
- [ ] heuselGANsTrainedTwo2018 — FID
- [ ] binkowskiDemystifyingMMDGANs2021 — KID
- [ ] hesselCLIPScoreReferencefreeEvaluation2021 — CLIPScore
- [ ] ravuriClassificationAccuracyScore2019 — Classification Accuracy Score / TSTR-style downstream utility
- [ ] salimansImprovedTechniquesTraining2016 — Inception Score (brief mention only, as FID/CLIPScore predecessor)
- NOTE: Cohen 1960 / Fleiss 1971 / Landis & Koch 1977 / Krippendorff 2004 (kappa statistics) are NOT present in references.bib despite claude_literature.md claiming they were verified — grepped by author/title/DOI, no match. DROPPED, will not cite; softened to describe blind multi-rater protocol via HYPE only.
- NOTE: FID critiques (Kynkäänniemi et al. CVPR 2023 "Rethinking FID", Parmar et al. CVPR 2022 aliased resizing, Stein et al. NeurIPS 2023, Chong & Forsyth CVPR 2020) also NOT in references.bib — grepped, no match. DROPPED.
- NOTE: Esteban et al. 2017 (original TSTR) NOT in references.bib. DROPPED — using ravuriClassificationAccuracyScore2019 (CAS) as the downstream-utility anchor instead, consistent with claude_literature.md's own fallback framing.

## sec:lit_synthetic_long_tail (Synthetic Data for Long-Tail and Rare-Class Problems)
- [ ] chawlaSMOTESyntheticMinority2002 — SMOTE, feature-space resampling baseline (contrast case)
- [ ] cuiClassBalancedLossBased2019 — class-balanced loss (re-weighting alternative)
- [ ] kangDecouplingRepresentationClassifier2020 — decoupling representation/classifier (re-sampling alternative)
- [ ] caoLearningImbalancedDatasets2019 — LDAM loss (re-weighting alternative)
- [ ] zhangDeepLongTailedLearning2023 — long-tail survey
- [ ] weiFineGrainedImageAnalysis2022 — fine-grained review (already used in Section A for a different subsection; reuse if relevant here)
- NOTE: Samuel et al. 2021 "Generating Tail Samples", Beery et al. 2021 "Synthetic Animals", He et al. 2022 "Synthetic Data Augmentation for Long-Tailed Visual Recognition" (the ICLR 2022 He/Sun/Yu et al paper, DISTINCT from heSyntheticDataGenerative2023 despite similar author list) — all checked, NOT in references.bib. DROPPED.

## Internal docs checked
- docs/synthetic-model-comparison/05_prompt-strategy-and-length-limits.md — FOUND, read in full.
- docs/synthetic-model-comparison/06_evaluation-methodology.md — FOUND, read in full.

## Citekeys used
(to be filled at the end)
