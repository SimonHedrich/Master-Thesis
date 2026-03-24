# Supplementary Dataset Research — LLM Prompt

This document contains a prompt for deep LLM research on open wildlife image datasets that can supplement Danielle's GBIF/SpeciesNet dataset for training the AX Visio student model.

---

## Context for the LLM

> **Task:** You are a machine learning researcher specializing in wildlife computer vision. You need to identify and evaluate open image datasets that can supplement an existing training dataset for a mammal species detection model. The model will be deployed commercially on the Swarovski AX Visio smart binocular ($4,800 premium consumer product). **Commercial usability of the data is a hard requirement.**
>
> ### Background
>
> We are training a lightweight object detection model (1–3M parameters, YOLO-nano class) to classify non-bird mammals in real time on embedded hardware (Qualcomm QCS605). The model has two target label sets: a focused 225-class list and an extended 480-class list (see details below).
>
> ### Existing Dataset
>
> Our primary dataset comes from GBIF (Global Biodiversity Information Facility) images processed through Google's SpeciesNet v4.0.2a pipeline:
>
> - **66,881 usable images** on disk (91,291 in the JSON, but 24,410 are missing from disk)
> - **35,594 images** (53%) have confident species-level SpeciesNet predictions
> - **19,420 images** (29%) have rolled-up predictions (genus/family/order level — uncertain species)
> - **11,867 images** (18%) are detector-only (animal detected but not classified) or blank
> - Labels are **SpeciesNet-generated, not human-verified** — approximately 23% of predictions disagree with the filename-inferred species (based on a 10k-image sample)
> - Heavily imbalanced: 120 species have >100 images, but 146 species have <10 images
> - **405 unique species** at confident prediction level
>
> ### Coverage Gaps
>
> Against our 225-class target label set:
> - **194 of 225 classes** have at least 1 image
> - **31 classes have zero images**, including:
>   - African antelopes: bongo, nyala, roan antelope, kob, dik-dik, klipspringer
>   - Asian mammals: Asiatic wild ass, water deer, yak, water buffalo, Japanese macaque, sloth bear
>   - Marine mammals: sea otter, walrus, elephant seals, eared seals, pinnipeds
>   - Primates: aye-aye, ring-tailed lemur, patas monkey, drill
>   - Other: meerkat, saiga, brown hyena, Eurasian lynx, domestic pig, sloths
> - **55 classes have fewer than 10 images**
> - **94 classes have fewer than 50 images** (likely insufficient for robust training)
>
> Against our 480-class extended label set:
> - **308 of 478 classes** have at least 1 image
> - **170 classes have zero images**
> - **243 classes have fewer than 10 images**
>
> ### Commercial Use Constraint
>
> The model will be deployed in a commercial product (Swarovski AX Visio). **Only datasets with licenses that permit commercial use of derived models are acceptable.** This means:
> - **Acceptable licenses:** CC0, CC-BY, CC-BY-SA, Apache 2.0, MIT, public domain, CDLA-Permissive, custom licenses that explicitly allow commercial model training
> - **Unacceptable licenses:** CC-BY-NC (non-commercial), CC-BY-NC-SA, CC-BY-NC-ND, restrictive academic-only licenses
> - **Gray area (needs legal review):** CC-BY-SA (share-alike may impose obligations on the trained model), CDLA-Sharing, licenses that are silent on model training
> - Note: Using images for *model training* may be treated differently from *redistributing* the images. Some datasets allow training commercial models even if image redistribution is restricted. This distinction should be noted.
>
> ### Technical Requirements for Supplementary Data
>
> Ideal supplementary datasets should have:
> 1. **Bounding box annotations** (not just image-level labels) — needed for detection model training
> 2. **Species-level labels** aligned with or mappable to SpeciesNet taxonomy
> 3. **Daylight, color images** — not infrared/grayscale camera trap images (the AX Visio produces color photos)
> 4. **Reasonable image quality** — at least 200×200px, focused on the animal
> 5. **Geographic diversity** — multiple locations per species to avoid background overfitting
>
> However, datasets with only image-level labels (no bounding boxes) are still useful — we can run MegaDetector to generate bounding boxes. Similarly, camera trap datasets (which are often infrared) can supplement training if mixed with daylight data. These should be noted as "partially suitable" rather than excluded.
>
> ### Research Questions
>
> Please investigate and produce a structured report covering:
>
> **1. Major Open Wildlife Image Datasets**
>
> For each dataset, provide:
> - Name and URL
> - Organization/creator
> - License (exact license name, and explicit assessment: commercially usable YES / NO / NEEDS REVIEW)
> - Total image count
> - Number of mammal species covered
> - Annotation type (bounding boxes, image-level labels, segmentation masks, etc.)
> - Image type (camera trap IR, camera trap color, citizen science photo, professional photo)
> - Geographic coverage
> - Overlap with our target species (rough estimate: how many of our 225/480 classes are covered)
> - Key limitations
>
> Specifically investigate these known datasets and any others you find:
>
> a) **iNaturalist Competition datasets** (iNat 2017, 2018, 2019, 2021)
>    - Multiple years with different class counts
>    - Citizen science photos (high quality, daylight, color)
>    - Which year/version is most useful for mammal detection?
>
> b) **LILA BC (Labeled Information Library of Alexandria: Biology and Conservation)**
>    - Aggregator of many camera trap datasets
>    - Which sub-datasets are most relevant and commercially usable?
>
> c) **Snapshot Serengeti / Snapshot Safari**
>    - Camera trap datasets from African ecosystems
>    - License status for commercial use?
>
> d) **iWildCam** (2019, 2020, 2021, 2022)
>    - FGVC competition datasets
>    - Camera trap images from global sites
>
> e) **Caltech Camera Traps (CCT-20)**
>    - North American camera trap data
>
> f) **Open Images V7** (Google)
>    - Large-scale dataset with bounding boxes
>    - How many mammal classes are included?
>
> g) **COCO (Common Objects in Context)**
>    - Has some animal classes with bounding boxes
>    - Which mammal classes?
>
> h) **Wildlife-10 / Wildlife-71 / similar benchmarks**
>    - Any standardized wildlife classification benchmarks?
>
> i) **TreeOfLife-200M / BioCLIP datasets**
>    - Very large biological image datasets
>    - Usable for pre-training or fine-tuning?
>
> j) **Pl@ntNet / iNaturalist GBIF exports**
>    - Are there direct GBIF image exports with permissive licenses?
>
> k) **Any other datasets** you identify that cover our gap species (African antelopes, Asian mammals, marine mammals, primates, domestic animals)
>
> **2. Gap Analysis**
>
> For our 31 zero-image classes and 55 under-10-image classes, identify which supplementary datasets would provide coverage:
> - Create a matrix: gap species × dataset → estimated image count
> - Identify species that remain uncovered even after all known open datasets
> - For persistently uncovered species: suggest alternative strategies (synthetic generation, web scraping with permissive license filtering, zoo photo campaigns)
>
> **3. Dataset Combination Strategy**
>
> Recommend a practical strategy for combining datasets:
> - Label harmonization challenges (different taxonomies, naming conventions)
> - How to handle mixed image types (camera trap IR + daylight photos)
> - How to handle label noise (SpeciesNet predictions vs. human-verified vs. citizen science)
> - Recommended train/val/test split strategy across multiple data sources
> - Total estimated image count after combining all recommended datasets
>
> **4. License Risk Assessment**
>
> Provide a summary table of all recommended datasets with:
> - Dataset name
> - License
> - Commercial model training: SAFE / NEEDS REVIEW / NOT ALLOWED
> - Image redistribution: SAFE / NEEDS REVIEW / NOT ALLOWED
> - Specific license concerns or obligations
>
> **Output format:** Structure your response as a detailed research report with clear sections, tables, and citations. Be specific about license names and version numbers. When uncertain about commercial usability, say so explicitly rather than guessing.
