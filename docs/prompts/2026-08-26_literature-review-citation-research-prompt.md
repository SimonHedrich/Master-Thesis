# Deep Research Prompt — Literature Review Citations (Chapter 2)

Prompt below is meant to be pasted into a deep-research-capable LLM (with web
access). Target output: a Zotero-importable reference list for
`thesis/manuscript/chapters/2-Literature_Review.tex`.

---

## PROMPT (copy from here)

You are a research librarian and computer-vision domain expert assisting with the
literature review chapter of a Master's thesis in applied machine learning. Your
task is **not** to write prose. Your task is to **find, verify, and report real
academic references** that I can import into Zotero and cite in LaTeX/Overleaf.

### Thesis context (so you can judge relevance)

The thesis is titled, in substance: *optimizing deep learning object detection
models for real-time wildlife species detection on embedded hardware*.

- **Domain:** fine-grained wildlife species detection, restricted to non-bird
  mammals, 225 target classes.
- **Model actually trained:** YOLOv5s (fixed to the last commercially-licensed
  YOLOv5 revision, commit `5cdad89`; AGPL-relicensing after that point made
  newer revisions unusable).
- **Target hardware:** Qualcomm QCS605 SoC (Hexagon 685 DSP, Adreno 615 GPU), as
  found in the Swarovski Optik AX Visio smart binocular. Development proxy
  hardware: Raspberry Pi 5 (8 GB).
- **Where the work actually went:** roughly 80 % of the available time went into
  **dataset construction and synthetic training-data generation**. The dataset
  was assembled from iNaturalist / GBIF-sourced field observations, filtered
  with a staged pipeline that uses **MegaDetector + SpeciesNet** as a
  pseudo-labeling and quality-filtering ensemble, and then supplemented for
  data-poor classes with **text-to-image generated synthetic images** (compared
  across Stable Diffusion 3.5, FLUX, SDXL-lightning derivatives, and proprietary
  APIs such as Gemini image generation).
- **What was deliberately scoped down:** knowledge distillation was reduced to a
  single basic response-based KD comparison; embedded work was reduced to a
  single unoptimized-model runtime/latency check on the AX Visio hardware. No
  quantization-aware training was performed.
- **Evaluation design:** the headline metric is computed over a **mixed test set**
  (real test images ∪ a class-balanced 225 × 50 synthetic test set), with a
  **real-only breakout always reported alongside**; the real-vs-synthetic delta is
  monitored as a domain-shift watchdog. Look-alike species groups (zebra, lynx,
  gazelle clusters) are evaluated with coarse-to-fine / hierarchical grouping.

### What I need from you

For **each numbered topic** in the section list below, return **the best
available academic references**, ranked by how citable and canonical they are.
Aim for the counts I give per topic; if a topic genuinely has thin literature,
say so explicitly rather than padding it with weak or off-topic hits.

### Hard requirements on every reference (this is the critical part)

For each reference, give me **all** of the following fields, in the exact
structure shown in "Output format" below:

1. **Full author list** (not "et al.") — first author's surname is enough for the
   citation key, but I need the full list for the BibTeX record.
2. **Year** of the version you are pointing at.
3. **Exact title.**
4. **Venue**: conference name + year (e.g. "CVPR 2023"), journal name + volume,
   or "arXiv preprint" / "technical report" if unpublished. If a paper was an
   arXiv preprint that was **later published at a venue**, tell me both and mark
   the peer-reviewed version as the one to cite.
5. **A Zotero-importable identifier — at least one of these, in this priority
   order:**
   - **DOI** (preferred; give it as a bare DOI, e.g. `10.1109/CVPR.2016.91`)
   - **arXiv ID + the `https://arxiv.org/abs/XXXX.XXXXX` landing page URL**
     (the `/abs/` page, *never* the `/pdf/` URL — Zotero's translator needs
     `/abs/`)
   - **A stable publisher landing page URL** (ACM DL, IEEE Xplore, SpringerLink,
     OpenReview, CVF Open Access, PMLR, journal page)
   - Only if none of the above exist: an official GitHub/model-card/docs URL,
     and flag it clearly as "no DOI — non-archival source".
6. **A one- to two-sentence relevance note**: what specifically this paper
   supports in *my* chapter, phrased so I can see where it belongs.
7. **Citation-key suggestion** in the style already used in my `.bib`:
   `firstauthorsurname_shorttitle_year`, all lowercase (e.g.
   `hinton_distilling_2015`, `heusel_fid_2017`).

**Verification rules — take these seriously:**

- **Do not invent references.** Every entry must be a paper you have actually
  located. If you are not certain a paper exists with that exact title and
  author list, drop it.
- **Do not invent DOIs or arXiv IDs.** A wrong identifier is worse than a
  missing one, because it silently imports the wrong record into Zotero. If you
  cannot confirm the identifier, write `identifier: UNVERIFIED — search title on
  <site>` instead of guessing.
- Prefer **peer-reviewed venue versions** over arXiv preprints where both exist.
- Prefer the **canonical/original** paper for a concept over a later paper that
  merely uses it, but do include one or two **recent surveys (2023–2026)** per
  major topic, since surveys are the most efficient thing to cite for the
  historical-narrative passages.
- Note **retracted, withdrawn, or heavily contested** work if you encounter it.

### Already in my bibliography — do not re-find these, but DO tell me if a better/peer-reviewed version exists

`gadot_speciesnet_2024`, `beery_megadetector_2019`, `gbif_2024`,
`vanhorn_inaturalist_2018`, `lila_bc_2024`, `swanson_snapshot_2015`,
`wcs_camera_traps_2022`, `kuznetsova_open_2020`, `lin_microsoft_2014`,
`stevens_bioclip_2024`, `tan_efficientnetv2_2021`, `xiao_florence2_2023`,
`qwen_2024`, `google_gemini_2024`, `voxel51_fiftyone_2024`,
`jocher_yolov5_2020`, `bochkovskiy_yolov4_2020`, `zhang_mixup_2017`,
`ghiasi_copypaste_2021`.

---

## Topics to cover

The numbering below mirrors my chapter structure. Please keep it, so I can map
your output straight onto my sections.

### 1. Object detection for wildlife monitoring

**1.1 — Detector architecture history (~8–12 refs).** Canonical papers for the
two-stage lineage (R-CNN, Fast R-CNN, Faster R-CNN, Feature Pyramid Networks)
and the single-stage lineage (YOLOv1, SSD, RetinaNet/focal loss). I need these at
*conceptual depth only*, so also give me 1–2 modern general object-detection
survey papers I can cite instead of enumerating the whole family tree.

**1.2 — The YOLO family in depth (~10–15 refs).** The strongest available
citation for each of: YOLOv1–v4, YOLOv5, YOLOv6, YOLOv7, YOLOv8, YOLOv9,
YOLOv10 (NMS-free / consistent dual assignments), YOLOv11, YOLOv12, and any
YOLO release documented up to 2026 (including YOLO26 if a citable technical
report or paper exists — if it is documentation-only, say so and give the
official docs URL). Note explicitly that several Ultralytics releases have **no
peer-reviewed paper** and tell me what the accepted citation practice is for
those. Additionally: 2–3 **YOLO-family review/survey papers** (2023–2026), and
the key papers behind the architectural transitions I need to describe:
anchor-based → anchor-free (FCOS, and the YOLOX anchor-free decoupled head), and
label-assignment/NMS-free end-to-end detection (DETR as the reference point for
set prediction).

**1.3 — Nano-scale / edge detectors (~6–8 refs).** NanoDet (or NanoDet-Plus),
PicoDet, EfficientDet + EfficientDet-Lite, YOLO-nano-class variants, and any
paper that empirically studies the **class-count vs. accuracy tradeoff** for
small-capacity detectors — I claim a practical ~200–250-class ceiling for
nano-scale backbones and I need the best available literature support for that
claim, or an honest statement that this specific claim is under-supported in the
literature and can only be cited indirectly.

**1.4 — Camera-trap and wildlife-specific detection/classification (~10–14
refs).** MegaDetector (peer-reviewed version if one exists), SpeciesNet /
Google's camera-trap classifier, Wildlife Insights, plus the broader literature
on: automated species identification from camera-trap imagery (Norouzzadeh et
al., Willi et al., Tabak et al., Beery et al. "Recognition in Terra Incognita"),
**generalization to new camera-trap locations**, and wildlife CV surveys
(2022–2026). Critically, I also need references for the **field-observation vs.
camera-trap domain gap** — my thesis rejected LILA BC as a training source
because its camera-trap imagery does not match the handheld/binocular
field-observation viewpoint of the AX Visio. Find whatever literature exists on
viewpoint/deployment domain shift in wildlife imagery to support that decision.

**1.5 — Fine-grained visual categorization and long-tail ecological data (~8–10
refs).** Fine-grained recognition foundations (CUB-200, Stanford Dogs,
iNaturalist challenge papers), the difficulty of visually near-identical
species, and long-tail recognition (class-balanced loss, decoupled
representation/classifier learning, LDAM, long-tail survey 2023+).

**1.6 — Two-stage detect-then-classify vs. end-to-end detection (~4–6 refs).**
Literature that directly compares a detect-then-classify cascade against a
single end-to-end multi-class detector, and anything on the **latency
characteristics** of per-instance classification cascades (cost scaling with
instance count) versus fixed-cost single-pass detection. Also: hierarchical /
taxonomic classification with confidence-gated rollup to a coarser taxonomic
rank — this is what SpeciesNet does and I need the methodological precedent for
it.

### 2. Synthetic training data (LARGEST section — weight your effort here)

**2.1 — Generative image models (~12–16 refs).** GANs (Goodfellow, DCGAN,
StyleGAN/StyleGAN2, BigGAN) and diffusion (DDPM, DDIM, score-based SDE,
classifier-free guidance, Latent Diffusion / Stable Diffusion, SDXL, Stable
Diffusion 3 / rectified-flow, Imagen, DALL·E 2/3, FLUX if citable, consistency /
distilled few-step models such as LCM / SDXL-Lightning / SDXL-Turbo). For
proprietary APIs with no paper (Gemini image generation, GPT-image), give the
official technical report or model card URL and flag it as non-archival.
Additionally: literature on **text encoders and prompt-conditioning capacity** —
CLIP, T5 as a diffusion text encoder, and anything studying the effect of
**prompt length / token limits** on generation fidelity, since prompt-length
limits per model family became a controlled experimental factor in my
comparison.

**2.2 — Synthetic data for detection & the sim-to-real gap (~14–18 refs).** This
is the core of my contribution's literature grounding. Cover:
- Rendered/simulated training data for detection (domain randomization, Virtual
  KITTI, SYNTHIA, Synscapes, "Training Deep Networks with Synthetic Data",
  Falling Things / synthetic-data-for-detection lineage).
- **Generative** (diffusion-based) synthetic data for classification and
  detection: "Is Synthetic Data From Generative Models Ready for Image
  Recognition?", DA-Fusion, "Fake it till you make it", diffusion-based dataset
  augmentation/expansion, X-Paste, DiffusionDet-adjacent data work, synthetic
  data for detection specifically (not just classification).
- **Documented failure modes**: texture/lighting domain shift, anatomical and
  geometric artifacts in generated animals/humans, background–context
  implausibility, and **model collapse / self-consuming generative loops** when
  training on generated data.
- Papers reporting **how much** synthetic data helps and where it stops helping
  (mixing-ratio / real-to-synthetic ratio studies).
- Domain adaptation and domain generalization surveys, as the framing for the
  sim-to-real gap.

**2.3 — Evaluating synthetic image quality (~14–18 refs).** Three axes, all
three of which I need grounded:
- *(a) Human evaluation:* protocols for blind/side-by-side human rating of
  generated images, two-alternative forced choice / HYPE, best-practice
  critiques of human eval in generative modeling, and the statistics of
  inter-rater agreement — **Cohen's kappa (Cohen 1960)**, **Fleiss' kappa (Fleiss
  1971)**, Krippendorff's alpha, and Landis & Koch's kappa-interpretation
  benchmarks. I need the *original* statistics citations here, not just ML papers
  that use them.
- *(b) Automatic proxies:* Inception Score, **FID (Heusel et al. 2017)**, **KID
  (Bińkowski et al. 2018)**, precision/recall for generative models, CLIPScore,
  known **criticisms and failure modes of FID** (small-sample bias, backbone
  sensitivity, the "rethinking FID" line of work), and any paper validating
  **classifier confidence on generated images** as a recognizability proxy.
- *(c) Downstream-task utility:* the "train on synthetic, test on real" (TSTR)
  evaluation protocol and papers arguing downstream utility is the only
  trustworthy quality signal for synthetic training data.

**2.4 — Synthetic data for long-tail / rare classes (~8–10 refs).** Work that
uses generative augmentation *specifically* for the tail of the distribution,
and how it compares against the classical remedies (re-sampling, class-balanced
loss, re-weighting, SMOTE as the historical anchor). Also anything on
**few-shot / rare-species** recognition in ecology where synthetic or generated
imagery was used.

### 3. Knowledge distillation and embedded inference (concise — foundational only)

**3.1 — KD fundamentals (~8–10 refs).** Hinton/Vinyals/Dean soft targets and
temperature scaling, Buciluă et al. 2006 model compression as the precursor,
FitNets (feature-based), attention transfer, response- vs. feature- vs.
relation-based taxonomy (Gou et al. survey), and **KD for object detection**
specifically — why plain logit matching is insufficient without spatial
awareness (Chen et al. 2017 detection KD, fine-grained feature imitation,
FGD/focal-and-global distillation, and a recent detection-KD survey). Also one
citation each, for a *single-sentence acknowledgment* of what I did **not**
pursue: capacity gap / teacher-assistant, multi-teacher, and cross-domain
distillation.

**3.2 — Lightweight architecture design (~6–8 refs).** Depthwise separable
convolutions (MobileNetV1/V2/V3), ShuffleNet, SqueezeNet, GhostNet, structured
channel pruning, and a recent efficient-architecture survey.

**3.3 — Quantization (~6–8 refs), explicitly scoped as context only.**
Post-training quantization vs. quantization-aware training (Jacob et al. 2018
integer-arithmetic-only inference, white paper on quantization for efficient
inference, LSQ, a 2023+ quantization survey), plus the citable Qualcomm-side
material: **AIMET**, the SNPE / Qualcomm Neural Processing SDK, and anything
academic on **Hexagon DSP** inference performance. Note where only vendor
documentation exists.

**3.4 — On-device benchmarking methodology (~6–8 refs).** MLPerf Inference,
MLPerf Tiny / MLPerf Mobile, AI Benchmark (Ignatov et al.) for mobile-SoC
inference measurement, and any paper documenting **measurement pitfalls** on
mobile/ARM SoCs: cold-start and warm-up effects, thermal throttling, DVFS,
thread count and core-affinity sensitivity, and run-to-run variance. Also:
citable references for the export/runtime formats — ONNX / ONNX Runtime, TFLite
(and the "TensorFlow Lite / LiteRT" on-device inference paper if one exists),
NCNN. And anything on **Raspberry Pi 5 / ARM SBC as a proxy platform** for
embedded DL benchmarking, to support my proxy-hardware argument.

### 4. Evaluation metrics for fine-grained detection

**4.1 — Standard detection metrics (~6–8 refs).** The canonical source for
IoU/AP/mAP as used in detection: PASCAL VOC (Everingham et al., including the
"retrospective" paper), MS COCO (already in my bib — but tell me if there is a
better metric-definition citation), papers analyzing **what AP actually measures
and its pathologies**, Average Recall / "What makes for effective detection
proposals?", and Localization-Recall-Precision (LRP) or similar alternative
metrics.

**4.2 — Error decomposition & hierarchical evaluation (~6–8 refs).** TIDE
("A General Toolbox for Identifying Object Detection Errors"), Hoiem et al.
"Diagnosing Error in Object Detectors", and the literature on **hierarchical /
taxonomy-aware evaluation** — hierarchical precision/recall, cost-sensitive
misclassification over a taxonomy, superclass or coarse-to-fine evaluation for
confusable classes, and metrics that credit a correct coarse prediction when the
fine-grained class is wrong.

**4.3 — Evaluating on mixed real + synthetic test sets (~4–8 refs, plus an
honest gap statement).** This is a **deliberate probe for a literature gap.**
Search for precedent for: (a) reporting a headline metric over a *union* of real
and synthetic test images; (b) using a **class-balanced synthetic test set as a
diagnostic instrument** rather than as a benchmark; (c) work on benchmark
contamination or bias introduced by evaluating on generated data; (d) balanced
vs. natural-distribution test sets and how that choice changes the reported
metric. If the specific practice of a mixed real+synthetic headline test set is
**not** established in the literature, **say so explicitly and clearly** — I need
to know whether to present this as a novel methodological choice that requires
its own justification, or as an existing practice I can cite. Give me the
closest adjacent literature either way.

---

## Output format

Produce **Markdown**, grouped under the exact section headings above, with each
reference as a block in this shape:

```
#### [1.2-04] Wang et al. (2024) — YOLOv10: Real-Time End-to-End Object Detection
- **Authors:** Ao Wang, Hui Chen, Lihao Liu, Kai Chen, Zijia Lin, Jungong Han, Guiguang Ding
- **Venue:** NeurIPS 2024 (arXiv preprint 2024)
- **Identifier:** arXiv:2405.14458 — https://arxiv.org/abs/2405.14458
- **DOI:** <bare DOI or "none">
- **Citation key:** `wang_yolov10_2024`
- **Why it matters here:** NMS-free end-to-end detection via consistent dual
  assignments; the endpoint of the anchor-free/NMS-free trajectory I trace from
  YOLOv5 forward in Sec. 2.1.1.
- **Confidence:** verified / unverified-identifier
```

Then, at the very end, add two things:

1. **A single ready-to-import block** containing every arXiv ID and DOI you
   found, one per line, nothing else — so I can paste it straight into Zotero's
   "Add Item by Identifier" field in one go. Separate arXiv IDs and DOIs into
   two lists.
2. **A "Gaps and cautions" section**: topics where the literature is thin or
   where my thesis's claims cannot be directly supported by a citation
   (I expect at least the nano-scale class-count ceiling in 1.3 and the mixed
   real+synthetic test set in 4.3), plus any references you had to mark
   unverified.

Do not write chapter prose, do not summarize the papers at length, and do not
suggest changes to my chapter structure. References and identifiers only.

## END OF PROMPT
