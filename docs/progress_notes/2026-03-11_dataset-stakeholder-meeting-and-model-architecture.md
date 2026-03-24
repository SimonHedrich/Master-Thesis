# Progress Notes – 11.03.2026

---

## Meeting Notes – Danielle (AX Visio Product Owner)

**Background:** Danielle is the Product Owner for [AX Visio](https://www.swarovskioptik.com/de/de/beobachten/fernglaese/ax-visio), Swarovski Optik's smart binocular with a built-in AI model for wildlife species recognition. She is independently preparing training data for the binocular's species detection model, which overlaps with the goals of this thesis.

### Dataset Preparation

Danielle is still in the process of cleaning up approximately 80,000 images she has gathered for training. As an additional data source, she is considering the [iNaturalist Competition](https://www.kaggle.com/competitions/inaturalist-2021) — a public machine learning benchmark with large-scale, annotated wildlife images.

To decide which species to include, she scraped image counts from [GBIF](https://www.gbif.org) (Global Biodiversity Information Facility — a public open-access database of biodiversity observations worldwide) and stored the results in `resources/GBIF_image_counts.csv`. Species below a certain image count threshold will be excluded: too few training images leads to poor model performance, and such rare species are also less likely to be encountered by binocular users in practice.

### YOLOv5 License Constraint

The YOLOv5 codebase and model weights are only licensed for commercial use up to this specific commit:

> https://github.com/ultralytics/yolov5/commit/5cdad8922c83b0ed49a0173cd1a8b0739acbb336

Beyond this commit, Ultralytics changed its licensing terms. For commercial deployment in the AX Visio product, only code and weights at or before this commit may be used without additional licensing agreements.

Danielle also shared ten example images taken from the AX Visio binocular.

### Geo-Location Filtering

We discussed whether GPS location data should be incorporated to narrow down which species are plausible at any given time and place. Two approaches were considered:

1. **Model-level integration** – Feed GPS coordinates directly as an additional input to the detection model, so the model's predictions are inherently location-aware.
2. **Post-hoc output filtering** – Run the model normally and apply a rule-based filter afterward that removes or suppresses species classes that are not native to the detected GPS location.

**Decision:** The post-hoc output filter is sufficient for now. Modifying the model architecture to accept location data would add significant complexity with unclear benefit at this stage.

---

## Thinking Update – Class Design and Data Strategy

### Flexible Class Output Design

Rather than defining a narrow, hand-curated list of target species upfront, the plan is to use a broad, established taxonomy as the model's output space. Specifically, the class set from [SpeciesNet](https://github.com/google/cameratrapml) (Google's wildlife species identification model) would serve as the full class universe — excluding classes outside the scope of the use case (e.g. birds, marine life).

This approach decouples the model's capability from any specific application's requirements. A downstream filter can then be applied to the model's output predictions to remove irrelevant classes or to aggregate classes into coarser categories that better match Danielle's goals for the binocular.

### Synthetic Data Generation for Rare Species

A known challenge in wildlife datasets is that some species have very few available training images. Given recent advances in image generation (diffusion models), there is potential to synthesize training images for underrepresented species. The proposed experimental setup:

- **Train:** Mix of real images and synthetically generated images for rare species
- **Evaluate:** Only real photographs — to measure whether synthetic data actually helps generalization

Exploring this is of personal interest and will be a focus for the first month of experimentation.

---

## Model Architecture Research

*(Based on `research/A Review of Real-Time Deep Learning–Based Object Detection Models.md` and `docs/object-detection-models-for-embedded-systems.md`)*

For knowledge distillation to work, two tiers of models are needed: a large, accurate **teacher** and a small, deployable **student**. Research identified the following candidates:

### Full (Heavy) Architectures — Teacher Models

Models like **YOLOv12**, **RT-DETR**, **SpeciesNet**, and **DINOv3** fall into this category. They are large, computationally expensive, and cannot run on the target embedded hardware (Qualcomm QCS605). However, they serve as strong teacher models — they can be fine-tuned on the target species classes and used to distill knowledge into a lightweight student model.

### Lightweight Architectures — Student Models

Models like **YOLO26**, **NanoDet**, **PicoDet**, and **EfficientDet-Lite** are designed for edge and embedded deployment. These models *can* run on the target hardware. The workflow for these:

1. Adapt the output layer to the target wildlife species classes (replacing standard COCO classes)
2. Perform fine-tuning using knowledge distillation from the teacher model
3. Apply quantization-aware training to further optimize for embedded inference

---

## Potential Research Question

The architecture research led to an interesting comparison that could form a core research question for the thesis:

> **Does distilling a large, accurate teacher model into a lightweight student model yield better results than directly training the lightweight model on the target domain — especially given the domain shift from standard benchmarks (e.g. COCO classes) to wildlife species?**

This question is particularly relevant because: (a) domain shift from COCO-style classes to animal species is significant, and (b) it is unclear whether the teacher's general knowledge survives this shift well enough to benefit the student beyond supervised fine-tuning alone.
