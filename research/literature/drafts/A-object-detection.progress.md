# Progress: Section A — Object Detection for Wildlife Monitoring

## sec:lit_object_detection (intro, two-stage vs single-stage)
- [x] read, supports: girshickRichFeatureHierarchies2014a — R-CNN, two-stage origin (region proposals + CNN classification)
- [x] read, supports: uijlingsSelectiveSearchObject2013 — Selective Search region-proposal algorithm used by R-CNN
- [x] read, supports: girshickFastRCNN2015a — Fast R-CNN, shared conv features across proposals
- [x] read, supports: renFasterRCNNRealTime2017 — Faster R-CNN, learned Region Proposal Network, end-to-end trainable
- [x] read, supports: linFeaturePyramidNetworks2017 — FPN, multi-scale top-down feature pyramid
- [x] read, supports: redmonYouOnlyLook2016a — YOLOv1, single-stage regression framing
- [x] read, supports: liuSSDSingleShot2016a — SSD, multi-scale default boxes, single pass
- [x] read, supports: linFocalLossDense2017 — Focal Loss/RetinaNet, closes 1-stage/2-stage accuracy gap via class-imbalance fix
- [x] read, supports: zouObjectDetection202023 — 20-years survey, secondary reference for full historical detail

## sec:lit_wildlife_challenges (new subsection, added per TODO's "close with a Challenges subsection"; label sec:lit_wildlife_challenges)
- [x] read, supports: weiFineGrainedImageAnalysis2022 — fine-grained visual categorization framing, look-alike species
- [x] read, supports: cuiClassBalancedLossBased2019 — long-tail / effective number of samples
- [x] read, supports: zhangDeepLongTailedLearning2023 — long-tail survey
- [x] read, supports: beeryRecognitionTerraIncognita2018 — camera-trap location domain-shift (good at trained locations, poor at new ones), LILA BC rejection grounding
- [x] read, supports: williIdentifyingAnimalSpecies2019 — cross-project transferability costs accuracy
- [x] read, supports: tabakMachineLearningClassify2019 — cross-regional generalization (98% US -> 82% Canada out-of-sample)
- [x] read, supports: norouzzadehAutomaticallyIdentifyingCounting2018 — near-human accuracy in-distribution (93.8%/99.3%/96.6% figures), contrast point for domain-shift claim
- [x] read, supports: tuiaPerspectivesMachineLearning2022 — wildlife ML survey, domain shift as primary obstacle framing

## sec:lit_yolo_family
- [x] read, supports: redmonYouOnlyLook2016a — YOLOv1 (reuse, single-stage origin)
- [x] read, supports: redmonYOLO9000BetterFaster2017 — YOLOv2/9000, k-means anchors, WordTree hierarchical label space
- [x] read, supports: redmonYOLOv3IncrementalImprovement2018 — YOLOv3, Darknet-53, 3-scale heads
- jocher_yolov5_2020 — YOLOv5 (legacy key; fields are TODO-placeholder in thesis bib, cited per rule, no further detail asserted beyond what CLAUDE.md/Methods already establish)
- [x] read, supports: geYOLOXExceedingYOLO2021 — YOLOX, first anchor-free YOLO, decoupled head, SimOTA; also used for capacity-accuracy gap evidence (YOLO-Nano 0.91M params / 25.3% AP)
- [x] read, supports: liYOLOv6SingleStageObject2022 — YOLOv6, re-parameterizable backbone, industrial focus
- [x] read (abstract only, no source PDF in Zotero — flagged), supports: wangYOLOv9LearningWhat2025 — YOLOv9, PGI + GELAN, ECCV 2024 peer-reviewed version used (not preprint)
- [x] read, supports: wangYOLOv10RealTimeEndtoEnd2024 — YOLOv10, NMS-free consistent dual assignment
- [x] read, supports: khanamYOLOv11OverviewKey2024 — YOLOv11, C3k2 replacing C2f, C2PSA attention (also describes YOLOv8's C2f/anchor-free head as prior art)
- [x] read, supports: tianYOLOv12AttentionCentricRealTime2025 — YOLOv12, attention-centric, beats YOLOv10-N/YOLOv11-N
- [x] read, supports: nazirYouOnlyLook2023 — secondary YOLO family review (YOLOv2-v7), substitute for missing Terven survey
- [x] read, supports: yuPPPicoDetBetterRealTime2021 — PicoDet capacity-accuracy numbers, used as indirect (not direct) evidence for the class-count-ceiling gap

## sec:lit_md_speciesnet
- beery_megadetector_2019 (legacy key, MegaDetector — species-agnostic localizer)
- gadot_speciesnet_2024 (legacy key, SpeciesNet — name/product only, fields are TODO-placeholder)
- [x] read, supports: gadotCropNotCrop2024 — Gadot et al., IET Computer Vision 2024 (DOI 10.1049/cvi2.12318), same team/system as SpeciesNet: EfficientNetV2-M backbone, ~36M-image training corpus / 2176 labels, confidence-gated taxonomic label-rollup algorithm (aggregate confidence upward if top pred < 0.75), geofencing allow-list rollup, ~25% macro-F1 gain from crop-then-classify vs whole-image

## sec:lit_yolo_vs_ensemble
- gadotCropNotCrop2024 (reuse — accuracy gain, rollup graceful degradation)
- beery_megadetector_2019 / gadot_speciesnet_2024 (reuse)
- redmonYOLO9000BetterFaster2017 (reuse — WordTree as hierarchical-fallback precedent for detectors)

## Model Selection: Practical Constraints (factual record, not a survey)
- jocher_yolov5_2020, beery_megadetector_2019, gadot_speciesnet_2024 (reuse, context)
- yuPPPicoDetBetterRealTime2021 (reuse, PicoDet named as considered-not-benchmarked candidate)
- rangilyuRangiLyuNanodet2026 (NanoDet named as considered-not-benchmarked candidate; bib note only, no source file — GitHub repo, non-archival)
- EfficientDet-Lite: no bib entry in either corpus — named without \cite
- Sources used: `git show HEAD:thesis/manuscript/chapters/3-Methods_and_Implementation/35-Model_Selection.tex` (was TODO-only, nothing substantive lost besides the label, which moved here); CLAUDE.md "Important Constraints" (commit 5cdad89 licensing cutoff); docs/2026-03-10_object-detection-models-for-embedded-systems.md (NanoDet/PicoDet/EfficientDet-Lite considered-early context)

## Confirmed unavailable in research/literature/references.bib (checked, do not use)
- terven_comprehensive_2023 (Terven et al. YOLO survey) — NOT present; substituted with nazirYouOnlyLook2023
- sapkota / bertinetto / femtodet / leyolo — NOT present
- bochkovskiy_yolov4_2020 — exists ONLY as a legacy TODO placeholder in thesis bib; citekey rules restrict legacy-bib reuse to MegaDetector/SpeciesNet/YOLOv5 only, and no Zotero-corpus equivalent exists — YOLOv4 is mentioned by name only, folded into the YOLOv5/CSPNet sentence, without a dedicated citation
- EfficientDet (Tan et al., CVPR 2020) — NOT present despite being named in docs/2026-03-10 doc — mentioned by name without \cite in Model Selection subsection
- YOLO26 — no bib entry, no independent architectural analysis found anywhere; treated explicitly as an acknowledged absence of literature in sec:lit_yolo_family, no specific technical claim made about it

## Citekeys used (final, grouped by subsection)

### sec:lit_object_detection (intro)
girshickRichFeatureHierarchies2014a, uijlingsSelectiveSearchObject2013, girshickFastRCNN2015a, renFasterRCNNRealTime2017, linFeaturePyramidNetworks2017, redmonYouOnlyLook2016a, liuSSDSingleShot2016a, linFocalLossDense2017, zouObjectDetection202023

### sec:lit_wildlife_challenges
weiFineGrainedImageAnalysis2022, cuiClassBalancedLossBased2019, zhangDeepLongTailedLearning2023, beeryRecognitionTerraIncognita2018, williIdentifyingAnimalSpecies2019, tabakMachineLearningClassify2019, norouzzadehAutomaticallyIdentifyingCounting2018, tuiaPerspectivesMachineLearning2022

### sec:lit_yolo_family
redmonYouOnlyLook2016a, redmonYOLO9000BetterFaster2017, redmonYOLOv3IncrementalImprovement2018, jocher_yolov5_2020, nazirYouOnlyLook2023, geYOLOXExceedingYOLO2021, liYOLOv6SingleStageObject2022, wangYOLOv9LearningWhat2025, wangYOLOv10RealTimeEndtoEnd2024, khanamYOLOv11OverviewKey2024, tianYOLOv12AttentionCentricRealTime2025, yuPPPicoDetBetterRealTime2021, geYOLOXExceedingYOLO2021 (reused for gap evidence)

### sec:lit_md_speciesnet
beery_megadetector_2019, gadot_speciesnet_2024, gadotCropNotCrop2024

### sec:lit_yolo_vs_ensemble
gadotCropNotCrop2024, beery_megadetector_2019, gadot_speciesnet_2024, redmonYOLO9000BetterFaster2017

### sec:model_selection
jocher_yolov5_2020, beery_megadetector_2019, gadot_speciesnet_2024, yuPPPicoDetBetterRealTime2021, rangilyuRangiLyuNanodet2026
