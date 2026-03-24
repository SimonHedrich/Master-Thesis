# Species Label Selection — Extended List (480 Classes)

This document describes an alternative, larger label set targeting **480 output classes** for the student model. It preserves the vast majority of the PO's expert-curated 483-label list ([`resources/2026-03-17_SN_labels_refined_only.txt`](../resources/2026-03-17_SN_labels_refined_only.txt)), removing only clearly problematic entries and adding genus/family fallback labels for hierarchical inference.

For the full research background, selection methodology, and detailed justification of the three-tier hierarchy, see [`docs/species-label-selection.md`](species-label-selection.md) (225-class list) and [`docs/species-label-research.md`](species-label-research.md) (research report).

---

## 1. Rationale for a Larger Label Set

The 225-class list (see [`docs/species-label-selection.md`](species-label-selection.md)) is optimized for the strict 1–3M parameter constraint, where research suggests ~225 classes as the accuracy-optimal ceiling.<sup>[9][11]</sup> However, there are reasons to also produce a larger list:

- **Larger student models:** If the student model is closer to 5–10M parameters (e.g., YOLOv5s at 7.2M — the current AX Visio detector), 400–500 classes become feasible.
- **Two-stage pipeline fallback:** If the thesis ultimately retains a two-stage pipeline (detector + classifier), the classifier (EfficientNetV2-M at 54M params) can easily handle 500 classes.
- **Ablation baseline:** Comparing a 225-class model against a 478-class model on the same architecture reveals the empirical accuracy cost of class count in the wildlife domain.
- **PO alignment:** This list stays much closer to the PO's expert curation, minimizing the risk of dropping species that stakeholders care about.

---

## 2. Approach

**Starting point:** 483 entries from the PO's refined SpeciesNet label list.

**Step 1 — Deduplicate:** Remove 2 exact duplicates (same species appearing twice with different UUIDs or as repeated entries).

**Step 2 — Remove clearly inappropriate entries (-18):** Only entries that fail basic inclusion criteria are removed:
- Species too small to observe through binoculars (shrews, treeshrews)
- Strictly nocturnal species that are also range-limited or tiny
- Species restricted to a single small island or region with virtually no training data

**Step 3 — Add genus/family fallback labels (+17):** For groups with 3+ species in the list, add genus-level entries from the SpeciesNet taxonomy to enable hierarchical inference (see Section 3.5 in the 225-class document for the fallback strategy).

**Result:** 483 (PO) − 20 (removed) + 17 (fallbacks) = **480 classes**

*Note: The PO list contains 2 entries where the same SpeciesNet UUID is used for both a species-level and a genus-level entry (Arctonyx and Lama). Both are retained as they serve different purposes in the hierarchy.*

---

## 3. Removed Entries

### 3.1 Duplicates (2 entries)

| Scientific Name | Common Name | Issue |
|:----------------|:------------|:------|
| *Bos grunniens* | Yak | Appears at both line 463 (PO added) and line 480 (duplicate add). Remove the second occurrence. |
| *Ictonyx striatus* | Striped polecat / zorilla | Appears at line 89 (original filter, "zorilla") and line 460 (added back, "striped polecat"). Same species, different UUIDs. Keep the added-back entry (line 460) which has the more recognizable common name. |

### 3.2 Too Small for Binocular Observation (5 entries)

These species weigh under 100g and are virtually impossible to identify through binoculars at any practical distance. They were likely included in SpeciesNet for camera trap purposes (where the camera is centimeters away), not for optical observation.

| Scientific Name | Common Name | Reason |
|:----------------|:------------|:------|
| *Blarina brevicauda* | Northern short-tailed shrew | 15–30g, cryptic, underground/leaf-litter dwelling |
| *Suncus murinus* | House shrew | 30–60g, commensal rodent-like, rarely seen |
| *Sylvisorex vulcanorum* | Volcano shrew | <10g, extremely range-limited (Albertine Rift), nocturnal |
| *Tupaia glis* | Common treeshrew | Small (150g), SE Asian only, easily mistaken for squirrel |
| *Tupaia belangeri* | Northern treeshrew | Same issues as common treeshrew |

### 3.3 Strictly Nocturnal and Range-Limited (8 entries)

These species are active only at night, are restricted to specific regions, and have minimal open-dataset coverage. A binocular user will essentially never encounter them in daylight.

| Scientific Name | Common Name | Reason |
|:----------------|:------------|:------|
| *Microcebus murinus* | Grey mouse lemur | Nocturnal, 60g, Madagascar only |
| *Arctogalidia trivirgata* | Small-toothed palm civet | Nocturnal, SE Asian canopy specialist |
| *Poiana richardsonii* | Central African oyan | Nocturnal, dense forest only, rarely photographed |
| *Nandinia binotata* | African palm civet | Strictly nocturnal, arboreal |
| *Petaurista elegans* | Spotted giant flying squirrel | Nocturnal, SE Asian |
| *Petaurista philippensis* | Indian giant flying squirrel | Nocturnal, South Asian |
| *Aeromys tephromelas* | Black flying squirrel | Nocturnal, Borneo/Malay Peninsula |
| *Eupleres goudotii* | Eastern falanouc | Nocturnal, Madagascar, extremely elusive |

### 3.4 Small Nocturnal Australian Dasyurids (3 entries)

The PO included several small Australian marsupial predators from SpeciesNet's camera trap data. These are tiny, nocturnal, and rarely encountered even by Australian wildlife enthusiasts. The spotted-tailed quoll (*Dasyurus maculatus*) is retained as it is larger, occasionally diurnal, and more commonly observed.

| Scientific Name | Common Name | Reason |
|:----------------|:------------|:------|
| *Phascogale tapoatafa* | Brush-tailed phascogale | Nocturnal, <200g, arboreal |
| *Dasykaluta rosamondae* | Kaluta | Nocturnal, tiny (<40g), restricted to Pilbara |
| *Dasycercus blythi* | Brush-tailed mulgara | Nocturnal, arid inland Australia |

### 3.5 Very Range-Limited with Insufficient Data (2 entries)

| Scientific Name | Common Name | Reason |
|:----------------|:------------|:------|
| *Anomalurus derbianus* | Lord Derby's scaly-tailed squirrel | West/Central African forest, nocturnal glider, very few images |
| *Sundasciurus lowii* | Low's squirrel | Borneo/Malay Peninsula, small, easily confused with other tropical squirrels |

**Total removed: 20 entries**

---

## 4. Added Genus/Family Fallback Labels (17 entries)

These entries are sourced from the full SpeciesNet taxonomy ([`resources/speciesnet_taxonomy_release.txt`](../resources/speciesnet_taxonomy_release.txt)) and enable the hierarchical fallback inference strategy described in the 225-class document (Section 3.5). Each covers a group with multiple species-level entries already in the list.

| Taxonomic Level | Scientific Name | Common Name | Species Covered |
|:----------------|:----------------|:------------|:----------------|
| Genus | *Gorilla* | Gorillas | 2 (G. beringei, G. gorilla) |
| Genus | *Macaca* | Macaques | 11 species + 2 PO-added |
| Genus | *Cercopithecus* | Guenons | 10 species + 2 PO-added |
| Genus | *Colobus* | Colobus monkeys | 5 species (incl. Piliocolobus) |
| Genus | *Ateles* | Spider monkeys | 3 species |
| Genus | *Cebus* | Capuchins | 9 species (Cebus + Sapajus) |
| Genus | *Saguinus* | Tamarins | 6 species (incl. Leontocebus) |
| Genus | *Eulemur* | True lemurs | 4 species (+ Hapalemur, Prolemur) |
| Genus | *Saimiri* | Squirrel monkeys | 2 species |
| Genus | *Callithrix* | Marmosets | 3 species (Callithrix + Mico) |
| Genus | *Cephalophus* | Forest duikers | 12+ species |
| Genus | *Lycalopex* | South American foxes | 3 species |
| Genus | *Martes* | Martens | 4 species |
| Genus | *Mustela* | Weasels | 4 species |
| Genus | *Spilogale* | Spotted skunks | 2 species |
| Family | *Macropodidae* | Kangaroos/wallabies | 16 species |
| Family | *Manidae* | Pangolins | 5 species |

**Total added: 17 entries**

---

## 5. Summary Statistics

| Metric | 480-class list | 225-class list | PO list |
|--------|:--------------:|:--------------:|:-------:|
| **Total output classes** | **480** | **225** | **483** |
| Species-level | ~413 | 164 | ~413 |
| Genus-level | ~47 | 41 | ~30 |
| Family/clade-level | ~20 | 20 | ~15 |
| | | | |
| PO entries kept | 463 | ~208 | 483 |
| PO entries removed | 20 | ~275 | — |
| New fallback entries | 17 | 17 | — |

### Comparison of the Two Lists

| Aspect | 225-class list | 480-class list |
|--------|:---------------|:---------------|
| **Target model size** | 1–3M params (YOLO-nano, NanoDet, PicoDet) | 3–10M params or two-stage pipeline |
| **Consolidation strategy** | Aggressive: genus-level for most primate, mustelid, civet, duiker groups | Minimal: only deduplicate, remove nocturnal/tiny/range-limited |
| **Individual species retained** | ~164 | ~413 |
| **Primates** | 9 species + 15 genera = 24 | ~63 species + 15 genera = ~78 |
| **Mustelids** | 11 species + 3 genera = 14 | ~32 species + 3 genera = ~35 |
| **Duikers** | 1 species + 1 genus = 2 | ~12 species + 1 genus = ~13 |
| **Civets/Viverrids** | 2 species + 2 genera = 4 | ~12 species + 1 genus = ~13 |
| **Marsupials** | 5 species + 1 family = 9 | ~20 species + 1 family = ~21 |
| **Alignment with PO** | Significant departures (detailed justification) | Very close to PO's expert curation |

### When to Use Which List

- **225-class list:** For the primary nano-model experiments (1–3M params). This is the thesis's main contribution.
- **480-class list:** For (a) larger student model experiments, (b) the two-stage pipeline baseline comparison, (c) ablation study on class count vs. accuracy.

---

## 6. Output Files

- **This document:** Reasoning for the 478-class extended label set
- **[`resources/2026-03-20_student_model_labels_extended.txt`](../resources/2026-03-20_student_model_labels_extended.txt):** 480 labels in SpeciesNet taxonomy format
- **Reference:** [`docs/species-label-selection.md`](species-label-selection.md) (225-class list with full methodology)
