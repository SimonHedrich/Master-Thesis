# Towards a Visipedia: Combining Computer Vision and Communities of Experts

**Author:** Grant Van Horn
**Institution:** California Institute of Technology (Caltech)
**Year:** 2019 (defended September 7, 2018)
**Advisor:** Pietro Perona

---

## Overview

This PhD thesis explores how to combine computer vision with communities of experts and citizen scientists to work towards **Visipedia** — a community-generated visual encyclopedia analogous to Wikipedia. The thesis addresses two core challenges: (1) how to build large-scale, high-quality fine-grained visual datasets, and (2) how to train efficient computer vision models when training data follows a long-tailed distribution.

The work spans five published papers and covers crowdsourcing methodology, dataset construction (NABirds, iNaturalist), citizen science platforms (Merlin, iNaturalist apps), and model efficiency for large-scale classification.

---

## Original Source Files

| File | Description |
|------|-------------|
| [Towards a Visipedia.pdf](Towards%20a%20Visipedia.pdf) | Original PhD thesis PDF (authoritative source) |
| [Towards a Visipedia.md](Towards%20a%20Visipedia.md) | Full text extracted from PDF as a single markdown file (~6,973 lines) |

---

## Files in This Directory

| File | Chapter | Lines | Key Topics |
|------|---------|-------|------------|
| [00_front_matter.md](00_front_matter.md) | Front Matter | 189 | Acknowledgments, abstract, table of contents |
| [01_introduction.md](01_introduction.md) | Ch. 1 — Introduction | 224 | Visipedia vision, citizen science, thesis roadmap |
| [02_devil_in_tails.md](02_devil_in_tails.md) | Ch. 2 — The Devil Is In The Tails | 951 | Long-tail classification, eBird analysis, transfer learning |
| [03_lean_crowdsourcing.md](03_lean_crowdsourcing.md) | Ch. 3 — Lean Crowdsourcing | 1628 | Human-machine annotation, Mechanical Turk, probabilistic models |
| [04_lean_multiclass.md](04_lean_multiclass.md) | Ch. 4 — Lean Multiclass Crowdsourcing | 1158 | Multiclass annotation, taxonomy-aware crowdsourcing, cost reduction |
| [05_bird_recognition_app.md](05_bird_recognition_app.md) | Ch. 5 — Bird Recognition App | 859 | NABirds dataset, Merlin app, citizen scientist quality |
| [06_inaturalist_dataset.md](06_inaturalist_dataset.md) | Ch. 6 — iNaturalist Dataset | 1010 | iNat2017 benchmark, 5000+ species, class imbalance, detection |
| [07_memory_computation.md](07_memory_computation.md) | Ch. 7 — Memory & Computation | 833 | Model compression, taxonomic parameter sharing, mobile deployment |
| [08_conclusions.md](08_conclusions.md) | Ch. 8 — Conclusions | 121 | Future directions, expert/user interfaces, open problems |

---

## Chapter Summaries

### [Chapter 1: Introduction](01_introduction.md)
Motivates the Visipedia vision: a community-generated visual encyclopedia where experts contribute visual knowledge and users can ask visual questions about photographs. Introduces the four types of stakeholders (users, experts, annotators, engineers) and identifies the two key interfaces Visipedia must provide. Outlines how subsequent chapters address the challenges of data collection, annotation efficiency, and model scalability.

### [Chapter 2: The Devil Is In The Tails](02_devil_in_tails.md)
*arXiv:1709.01450 — Van Horn & Perona, 2017*

Analyzes the impact of long-tailed class distributions on fine-grained visual classification using the eBird dataset and deep CNNs. Key findings: (a) peak accuracy on well-represented classes is excellent; (b) adding more classes has minimal impact given sufficient data; (c) performance degrades steeply when training examples are scarce; (d) transfer learning is largely absent in current methods. Concludes that the community must address long-tailed distributions as a first-class problem.

### [Chapter 3: Lean Crowdsourcing](03_lean_crowdsourcing.md)
*CVPR 2017 — Branson, Van Horn & Perona*

Introduces a human-machine hybrid annotation system that dramatically reduces redundant labeling effort. Uses a sequential risk-estimation framework over a probabilistic model combining worker skill, image difficulty, and an incrementally trained CV model. Achieves 4–11× reduction in annotation time for binary filtering and 2–4× for bounding box annotation, while reducing errors. Covers models for binary labels, part keypoints, and bounding box sets.

### [Chapter 4: Lean Multiclass Crowdsourcing](04_lean_multiclass.md)
*CVPR 2018 — Van Horn et al.*

Extends the Lean Crowdsourcing framework to the multiclass setting with hundreds to thousands of categories. The method handles taxonomic label structures, worker history/dependence, and integrates CV model predictions to minimize annotations needed per image. Reduces required annotations by up to 5.4× and residual error by up to 90% vs. majority voting. Validated on real-world iNaturalist and Merlin annotation tasks.

### [Chapter 5: Building a Bird Recognition App and Large Scale Dataset](05_bird_recognition_app.md)
*CVPR 2015 — Van Horn et al.*

Describes the methodology for building NABirds — a dataset of 48,562 images across 555 North American bird species, collected with citizen scientists from eBird. Demonstrates that citizen scientists significantly outperform Mechanical Turk workers at zero cost. Measures label error rates (~4%) in popular datasets like CUB-200-2011 and ImageNet, and shows that while models are robust to training noise, high-quality expert-curated test sets are essential. The trained model powers the Merlin bird identification app.

### [Chapter 6: The iNaturalist Species Classification and Detection Dataset](06_inaturalist_dataset.md)
*CVPR 2018 — Van Horn et al.*

Introduces the iNat2017 benchmark: ~859,000 images across 5,089 species of plants and animals, collected via the iNaturalist citizen science platform. The dataset features real-world class imbalance, diverse camera types, and multi-expert verification. Baseline experiments show state-of-the-art models achieve only 67% top-1 accuracy, highlighting difficulty — especially for low-data species. Also includes an object detection track. Became a standard benchmark for long-tail and fine-grained recognition research.

### [Chapter 7: Reducing Memory & Computation Demands](07_memory_computation.md)
*Van Horn & Perona, 2019*

Addresses the bottleneck of large fully-connected classification layers when deploying fine-grained models (>1K categories) on mobile devices. Proposes **Taxonomic Parameter Sharing** — using the species taxonomy to share parameters across related classes. Joint training with a rank-factorized layer achieves a 25× memory reduction in the classification head without loss in top-1 accuracy on iNaturalist 8K-class tasks (reducing the final layer from 64MB to 2.6MB).

### [Chapter 8: Conclusions and Future Directions](08_conclusions.md)
Reflects on the gap between current deployed apps (Merlin, iNaturalist) and the full Visipedia vision. Identifies three open problems: (1) expert interface design for structured visual knowledge contribution; (2) interactive question-answering interface for navigating visual regions; (3) improving human-machine collaboration in ambiguous classification scenarios. Proposes hierarchical/taxonomic approaches for both.

---

## Published Papers

| # | Citation | Thesis Chapter |
|---|----------|---------------|
| 1 | Van Horn & Perona (2015). "Building a bird recognition app..." CVPR 2015. | Ch. 5 |
| 2 | Branson, Van Horn & Perona (2017). "Lean Crowdsourcing..." CVPR 2017. | Ch. 3 |
| 3 | Van Horn & Perona (2017). "The Devil is in the Tails..." arXiv:1709.01450. | Ch. 2 |
| 4 | Van Horn et al. (2018). "The iNaturalist Species Classification and Detection Dataset." CVPR 2018. | Ch. 6 |
| 5 | Van Horn et al. (2018). "Lean Multiclass Crowdsourcing." CVPR 2018. | Ch. 4 |

---

## Notes on Text Extraction Quality

This markdown was extracted from the original PDF. The text is ~95% clean. Known limitations:
- **Figures and diagrams are absent** — only their captions remain
- **Mathematical notation** may have broken subscripts/superscripts in some equations
- **Tables** may have minor alignment issues from the PDF extraction
