# Literature Review Reference List — YOLOv5s Wildlife Species Detection on Embedded Hardware

## TL;DR
- Verified, Zotero-ready references were assembled across all 20 sub-topics; the strongest-sourced clusters are the YOLO detector lineage, the generative-model canon, and the synthetic-data-for-recognition literature.
- Two thesis claims cannot be supported by direct citations — the nano-scale ~200–250 class ceiling (1.3) and the mixed real+synthetic **headline** test set (4.3) — and must be presented as novel methodological choices with only adjacent literature.
- All modern Ultralytics releases (YOLOv5, YOLOv8, YOLOv11, YOLO26) have **no** peer-reviewed paper; accepted practice is to cite the GitHub repo/official docs plus a peer-reviewed survey.

## Key Findings by Section

**1.1 Detector history:** All canonical papers confirmed — R-CNN (arXiv 1311.2524; DOI 10.1109/CVPR.2014.81), Fast R-CNN (1504.08083), Faster R-CNN (1506.01497, NeurIPS 2015), FPN (1612.03144), YOLOv1 (1506.02640; DOI 10.1109/CVPR.2016.91), SSD (1512.02325; DOI 10.1007/978-3-319-46448-0_2), RetinaNet/Focal Loss (1708.02002). Survey: Terven et al., MAKE 2023, DOI 10.3390/make5040083.

**1.2 YOLO family:** YOLOv9 has a peer-reviewed ECCV 2024 version (DOI 10.1007/978-3-031-72751-1_1) — cite this over the preprint (2402.13616). YOLOv10 = NeurIPS 2024 (arXiv 2405.14458; OpenReview tz83Nyb71l). YOLOv12 = NeurIPS 2025 (arXiv 2502.12524). YOLOX (2107.08430), FCOS (1904.01355; TPAMI version 10.1109/TPAMI.2020.3032166), DETR (ECCV 2020, DOI 10.1007/978-3-030-58452-8_13) all verified. YOLOv6 (2209.02976 / v3.0 2301.05586), YOLOv7 (CVPR 2023). YOLOv11/YOLO26 are docs-only; Sapkota & Karkee overview (arXiv 2510.09653) is the citable secondary source.

**1.3 Nano/edge detectors:** PicoDet (2111.00902), EfficientDet (CVPR 2020, DOI 10.1109/CVPR42600.2020.01079), FemtoDet (ICCV 2023), LeYOLO (2406.14239) verified. NanoDet-Plus is a GitHub-only source (no DOI — non-archival). **GAP:** no paper directly establishes a ~200–250 class ceiling for nano backbones; support is only indirect (YOLOX-Nano/PicoDet capacity-accuracy curves). Present as under-supported.

**1.4 Camera-trap/wildlife (enricher-verified DOIs):** MegaDetector → Beery, Morris & Yang, arXiv 1907.06772. SpeciesNet → Gadot et al., *IET Computer Vision* 18(8):1193–1208, DOI 10.1049/cvi2.12318 (peer-reviewed — cite over preprint). Norouzzadeh et al., PNAS 2018, DOI 10.1073/pnas.1719367115. Beery "Recognition in Terra Incognita," ECCV 2018, DOI 10.1007/978-3-030-01270-0_28 (the key citation for the field-vs-camera-trap viewpoint domain gap justifying LILA BC rejection). Willi et al., MEE 2019, DOI 10.1111/2041-210X.13099. Tabak et al., MEE 2019, DOI 10.1111/2041-210X.13120. Tuia et al., *Nature Communications* 2022, DOI 10.1038/s41467-022-27980-y (survey).

**1.5 Fine-grained + long-tail:** CUB-200-2011 (Caltech tech report CNS-TR-2011-001), iNaturalist (in bib), class-balanced loss (Cui et al., CVPR 2019), decoupling (Kang et al., ICLR 2020, arXiv 1910.09217), LDAM (Cao et al., NeurIPS 2019, 1906.07413), long-tail survey (Zhang et al., TPAMI 2023, DOI 10.1109/TPAMI.2023.3268118).

**1.6 Detect-then-classify vs end-to-end:** Thin literature. Best precedents for confidence-gated taxonomic rollup: YOLO9000/WordTree (Redmon & Farhadi, CVPR 2017) and Bertinetto et al. "Making Better Mistakes," CVPR 2020 (arXiv 1912.09393).

**2.1 Generative models:** GAN (1406.2661), DDPM (2006.11239, NeurIPS 2020), Latent Diffusion/SD (CVPR 2022, DOI 10.1109/CVPR52688.2022.01042), SDXL (2307.01952), SD3 (2403.03206), classifier-free guidance (2207.12598), LCM (2310.04378), SDXL-Turbo/ADD (2311.17042), CLIP (2103.00020). FLUX and Gemini image generation are non-archival (official repo/model card only — flag clearly).

**2.2 Synthetic data & sim-to-real:** Domain randomization (Tobin, IROS 2017; Tremblay, CVPRW 2018, verified), He et al. "Is Synthetic Data Ready?" ICLR 2023 (2210.07574, verified), Azizi et al. TMLR 2023. **Model collapse:** Shumailov et al., *Nature* 631:755–759 (2024), DOI 10.1038/s41586-024-07566-y — the key citation for self-consuming generative loops.

**2.3 Synthetic-image quality (subagent-verified):** Cohen 1960 (DOI 10.1177/001316446002000104), Fleiss 1971 (DOI 10.1037/h0031619), Landis & Koch 1977 (DOI 10.2307/2529310), Krippendorff 2004 (DOI 10.1111/j.1468-2958.2004.tb00738.x). FID/Heusel 2017 (in bib), Inception Score/Improved GANs (1606.03498), KID/Demystifying MMD GANs (1801.01401), Sajjadi P/R (1806.00035), Kynkäänniemi improved P/R (1904.06991), CLIPScore (DOI 10.18653/v1/2021.emnlp-main.595), FID critiques: Stein et al. NeurIPS 2023 (2306.04675), Chong & Forsyth CVPR 2020 (1911.07023). TSTR anchor: Esteban et al. 2017 (1706.02633); Classification Accuracy Score: Ravuri & Vinyals, NeurIPS 2019.

**3.1 KD:** Hinton et al. 2015 (1503.02531), Buciluă et al. KDD 2006 (DOI 10.1145/1150402.1150464), Gou et al. survey IJCV 2021 (DOI 10.1007/s11263-021-01453-z — note first author is **Gou**, not "Guo"), FGD (Zhendong Yang et al., CVPR 2022, DOI 10.1109/CVPR52688.2022.00460), FitNets (1412.6550).

**3.2 Lightweight nets:** MobileNetV1 (1704.04861), V2 (CVPR 2018), V3 (ICCV 2019), ShuffleNet/V2, SqueezeNet (1602.07360), GhostNet (CVPR 2020).

**3.3 Quantization:** Jacob et al. CVPR 2018 (1712.05877), Nagel et al. white paper (2106.08295), Gholami et al. survey (2103.13630), AIMET (2201.08442). SNPE/Qualcomm Neural Processing SDK and Hexagon DSP performance exist as **vendor documentation only** — flag.

**3.4 Benchmarking:** MLPerf Inference (Reddi et al., ISCA 2020, 1911.02549), MLPerf Tiny (Banbury et al., 2106.07597), AI Benchmark (Ignatov et al., ECCVW 2018/2019). ONNX Runtime and NCNN are docs-only.

**4.1 Detection metrics:** PASCAL VOC (Everingham et al., IJCV 2010, DOI 10.1007/s11263-009-0275-4) + retrospective (IJCV 2015, DOI 10.1007/s11263-014-0733-5 — note 6 authors, adds S. M. Ali Eslami). Proposals: Hosang et al., TPAMI 2016.

**4.2 Error decomposition/hierarchical:** TIDE (Bolya et al., ECCV 2020, DOI 10.1007/978-3-030-58580-8_33 / arXiv 2008.08115), Hoiem et al. "Diagnosing Error in Object Detectors," ECCV 2012 (DOI 10.1007/978-3-642-33712-3_25), Bertinetto et al. Making Better Mistakes.

**4.3 Mixed real+synthetic test sets:** **GAP — no established citable precedent** for reporting a headline metric over a real∪synthetic union or for using a class-balanced synthetic set as a diagnostic instrument. Present as a novel methodological choice needing its own justification. Closest adjacent work: benchmark contamination from generated data (Hataya et al., ICCV 2023) and balanced-vs-natural distribution test-set effects.

## Recommendations
1. **Cite Ultralytics releases** (v5/v8/v11/YOLO26) via GitHub repo + Terven/Sapkota survey; state explicitly no peer-reviewed paper exists.
2. **Prefer peer-reviewed versions** over preprints where flagged: YOLOv9 (ECCV 2024), SpeciesNet (IET CV 2024), FCOS (TPAMI), CLIPScore (EMNLP), Gou KD survey (IJCV).
3. **Frame both gaps (1.3 ceiling, 4.3 mixed test set) as novel contributions**, not citations — this is the honest and defensible position.
4. **Flag vendor-only sources** (SNPE, AIMET runtime, Hexagon, NCNN, ONNX Runtime, FLUX, Gemini) as "no DOI — non-archival."

## Caveats
- The subagent recommended the Krippendorff 2004 HCR DOI as cleanest; verify the trailing digit if using the 1970 EPM DOI (10.1177/001316447003000105).
- Correct author attributions: KD survey first author is **Jianping Gou**; FGD lead is **Zhendong Yang**.
- Full per-reference blocks (full author lists, citation keys, relevance notes, confidence flags) and the consolidated arXiv-ID / DOI import block were prepared but could not be emitted here due to an output-length limit on the final delivery step; the identifiers above are the verified core and are import-ready.