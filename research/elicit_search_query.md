# Elicit Search Queries

[Elicit](https://elicit.com) is an AI-powered academic research tool that searches and summarizes papers from Semantic Scholar. Unlike keyword search, it works best with natural language questions or descriptive topic statements — the more specific the framing, the more targeted the results.

These queries are collected here so searches can be reproduced and refined over time. Each entry includes the query text, the research angle it targets, and what kind of papers it is intended to surface.

---

## Query 1 – General Thesis Topic

**Query:**
> Optimization of deep learning object detection models for real-time inference on embedded hardware (Qualcomm QCS605) using knowledge distillation, quantization, and transfer learning from large-scale wildlife datasets like SpeciesNet.

**Angle:** Broad overview query covering the full thesis scope — embedded inference, model compression techniques (quantization, knowledge distillation), and wildlife-specific transfer learning. Useful for finding survey papers and foundational work.

**Target papers:** Surveys on real-time object detection for embedded systems, knowledge distillation for object detection, quantization-aware training, and any prior work applying these techniques in ecological / wildlife contexts.

---

## Query 2 – Knowledge Distillation with Domain Shift

**Query:**
> Does knowledge distillation from a large general-domain teacher model improve a lightweight student model's accuracy on fine-grained wildlife species classification compared to directly fine-tuning the student on the target domain?

**Angle:** Targets the core research question that emerged from the 2026-03-11 progress notes: it is unclear whether a teacher model trained on broad classes (e.g. COCO) can meaningfully transfer knowledge to a student that must recognize fine-grained wildlife species — a significant domain shift. This query is intended to find papers that study the interaction between knowledge distillation and domain shift, particularly in fine-grained or long-tail visual classification settings.

**Target papers:** Studies on knowledge distillation under distribution shift, teacher–student performance comparisons across domain boundaries, fine-grained recognition with distillation, and any work on compressing models for ecological or species-level classification.
