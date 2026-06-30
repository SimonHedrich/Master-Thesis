# Species Label Selection for the Student Model

This document captures the analysis, research findings, and final decisions on the output class set for the wildlife detection student model deployed on the Swarovski AX Visio binocular.

---

## 1. Context and Goals

The thesis trains a lightweight one-shot object detection model (1–3M parameters) for the **Swarovski AX Visio** smart binocular, targeting real-time inference on the Qualcomm QCS605 (see [`docs/thesis-overview.md`](thesis-overview.md)).

**Product vision** (from Danielle, AX Visio Product Owner — [`docs/progress_notes/2026-03-11`](progress_notes/2026-03-11_dataset-stakeholder-meeting-and-model-architecture.md)):
The model should recognize all commonly observed mammals — not only charismatic safari wildlife, but also everyday animals a user would encounter in urban and suburban settings. A user who buys an expensive binocular should not be disappointed when it fails to identify a dog in the park.

**Target audience:** North America is the primary market, but the model should have reasonable global coverage.

**Scope exclusions:** Birds (separate model), marine mammals (except pinnipeds visible from shore), bats (nocturnal, rarely seen through binoculars).

---

## 2. Starting Point — Danielle's Refined SpeciesNet Labels

Danielle curated an initial label set by filtering the full SpeciesNet taxonomy (3,537 labels, ~1,415 mammal entries) down to **483 labels** using [`resources/refineLabels.py`](../resources/refineLabels.py). The resulting labels are in [`resources/sci_names_2026-03-17_SN_labels_refined_only.txt`](../resources/sci_names_2026-03-17_SN_labels_refined_only.txt).

**What was removed:** Taxonomic placeholders (order/family-level entries with more specific alternatives), nocturnal/cryptic species (bush babies, possums, dunnarts), small rodents (most mice, voles, mole rats), bats, tenrecs, very range-limited species, and visually similar subspecies consolidated to genus level.

**What was added:** Genus-level groupings for visually similar species (chipmunks, baboons, hares, mongooses, etc.), iconic species added individually (meerkat, stoat, ocelot, sea otter, walrus), common North American species, pinnipeds, and rodent family fallbacks.

**Danielle's note on squirrels:** *"Probably some of the squirrels could be added back in tbh (especially ones that occur in North America because that's our target audience...)"*

---

## 3. Research Findings

An extensive literature review was conducted on the optimal taxonomic output design for lightweight detection models (see [`docs/species-label-research.md`](species-label-research.md) for the full report). Key findings are summarized below.

### 3.1 Class Count Ceiling

Models in the 1–3M parameter range experience diminishing returns and rapid accuracy degradation as the class count exceeds approximately 200–300 categories.<sup>[9]</sup> YOLO11n achieves 39.5% mAP@0.5:0.95 on COCO's 80 classes,<sup>[6]</sup> but increasing from 100 to 500 classes in lightweight YOLO variants results in a multi-point mAP drop as the model becomes more susceptible to noise.<sup>[15][16]</sup>

Training lightweight models on datasets with heavy class imbalances exacerbates this issue. Open-source datasets like iNaturalist exhibit extreme long-tailed distributions where common species have millions of images while rare species have fewer than 1,000.<sup>[18]</sup> A 1–3M parameter model trained on such a distribution without aggressive class consolidation will suffer from catastrophic forgetting of minority classes or severe overfitting on majority classes.<sup>[21]</sup>

**Recommended ceiling: ~225 classes.** This provides sufficient model capacity per class while covering all major charismatic, common, and expected mammals across North America, Europe, and African safari circuits.<sup>[11]</sup>

### 3.2 Label Granularity Strategy

The research recommends a three-tier hierarchical approach:

- **Tier 1 (Species):** Reserved for mammals with high visual distinctiveness, charismatic/common status, and abundant training data (tens of thousands of images). Examples: American Black Bear, Moose, Lion, Eastern Gray Squirrel.
- **Tier 2 (Genus):** For groups of species that are genetically distinct but phenotypically nearly identical, especially at binocular distances. A user observing a chipmunk is generally satisfied with "Chipmunk" rather than distinguishing Alpine from Lodgepole Chipmunk.<sup>[34]</sup> Examples: *Tamias* (chipmunks), *Sylvilagus* (cottontails), *Papio* (baboons).
- **Tier 3 (Family):** Catch-all categories for highly diverse but morphologically similar groups, or inference fallback when confidence is low. Examples: Muridae (mice/rats), Herpestidae (mongooses), Sciuridae (squirrel fallback).

### 3.3 North American Coverage Gaps

The research found that the original 483-label list aggressively pruned North American squirrel species — a critical error given that squirrels are among the most frequently documented mammals on the continent, second only to white-tailed deer in many regions.<sup>[18]</sup> The major North American tree squirrels possess distinct visual phenotypes that a 1–3M parameter model can separate:

- **Eastern Gray Squirrel** (*Sciurus carolinensis*): Most common urban/suburban squirrel in eastern NA and Europe. Melanistic morphs must be represented in training data.<sup>[40]</sup>
- **Fox Squirrel** (*Sciurus niger*): Largest NA tree squirrel, distinct orange belly and larger head.<sup>[40]</sup>
- **American Red Squirrel** (*Tamiasciurus hudsonicus*): Smaller, reddish-brown, distinct white eye-ring.<sup>[45]</sup>
- **Western Gray Squirrel** (*Sciurus griseus*): Pure silver-gray, no brown hues.<sup>[46]</sup>

### 3.4 Domestic Animals and Zoo Use

Failing to identify common domestic animals, or misclassifying a domestic dog as a coyote, instantly undermines user trust in the AI.<sup>[5]</sup> Domestic dog, cat, horse, cattle, sheep, goat, and llama/alpaca must be included. Zoo-iconic species (tiger, giant panda, polar bear) should also be retained for urban usage scenarios.

### 3.5 Hierarchical Fallback at Inference

Rather than a mutually exclusive softmax, the model can be trained with hierarchical labels so it activates species, genus, and family nodes simultaneously.<sup>[25]</sup> At inference:

1. If species confidence > 80%: display species name (e.g., *"Eastern Gray Squirrel"*)
2. If genus confidence > 85% but species < threshold: display genus (e.g., *"Tree Squirrel (Sciurus)"*)
3. If only family is confident > 90%: display family (e.g., *"Squirrel Family (Sciuridae)"*)

This ensures graceful degradation rather than forcing low-confidence guesses.<sup>[64]</sup>

---

## 4. Selection Methodology

### Principles

1. **Respect the PO's expert curation.** The 483-label list was created by domain experts. Species are only removed or consolidated with clear justification.
2. **Visual distinctiveness drives granularity.** Species kept at Tier 1 must be distinguishable by a convolutional network from daylight binocular images. If expert-level examination (cranial morphology, dental formula, DNA) is needed to distinguish species, they are consolidated to genus.
3. **User encounter frequency matters.** Common species in target markets (NA, Europe, Africa) are prioritized. Range-limited endemics with <1,000 training images are deprioritized.
4. **Additions are cheaper than removals.** Adding a few missing common species costs less model capacity than keeping many obscure ones. The research recommended adding back NA squirrels and domestic animals.
5. **Target: 225 classes.** This sits in the optimal 200–250 range for 1–3M parameter models.

### How the target was reached

| Action | Count |
|--------|------:|
| PO species kept as-is | ~208 |
| New genus/family consolidation entries | 17 |
| **Total final labels** | **225** |
| PO species removed (consolidated or dropped) | ~275 |

---

## 5. Consolidation Decisions by Taxonomic Group

### 5.1 Carnivora (75 labels, down from ~155 in PO list)

**Bears (7 kept):** All seven PO bear species are retained — each is visually highly distinctive (size, coloring, facial markings) and represents a different continent/habitat.

**Red Panda (1 kept):** Unique appearance, no confusion with any other species.

**Canids (14, down from 24):** The PO list had 24 canid entries including many small fox species. Retained: all major wolves, jackals, and foxes that are either (a) commonly encountered in target markets or (b) visually highly distinctive. Consolidated: all three *Lycalopex* species (Argentine gray fox, culpeo, hoary fox) to genus — visually near-identical South American foxes.

*Removed canids (10):* Canis rufus (red wolf — <30 individuals in the wild, functionally indistinguishable from coyote hybrids), Canis adustus (side-striped jackal — very similar to black-backed/golden jackal, less commonly encountered), Vulpes velox and Vulpes macrotis (swift fox and kit fox — nearly identical small foxes), Vulpes corsac and Vulpes chama (corsac fox and cape fox — range-limited, similar to red fox at distance), Urocyon littoralis (island fox — Channel Islands endemic only), Speothos venaticus (bush dog — extremely elusive, rarely photographed), Cerdocyon thous (crab-eating fox — similar body plan to Lycalopex), Atelocynus microtis (short-eared dog — extremely rare, almost never observed).

**Felids (18, down from 25):** All five Panthera species, cheetah, puma, three lynx species, both Felis species, serval, caracal, clouded leopard, leopard cat, Leopardus genus + ocelot retained. These are all visually distinctive big/medium cats with strong user recognition.

*Removed felids (7):* Neofelis diardi (Sunda clouded leopard — nearly identical to mainland clouded leopard), Felis chaus (jungle cat — easily confused with wildcat/domestic cat), Catopuma temminckii (Asiatic golden cat — range-limited), Prionailurus rubiginosus (rusty-spotted cat — tiny, range-limited), Pardofelis marmorata (marbled cat — range-limited), Lynx pardinus (Iberian lynx — near-identical to Eurasian lynx at binocular distances), Herpailurus yagouaroundi (jaguarundi — range-limited, similar to other small cats).

**Mustelids (14, down from ~34):** This was the most aggressively consolidated group. PO had 34 mustelid entries including 4 individual marten species, 4 individual weasel species (specifically added back), and 5 otter species.

- *Martens:* Pine marten, beech marten, yellow-throated marten, American marten → **Martes genus**. These four species have similar elongated body plans and are often indistinguishable at distance.
- *Weasels:* Stoat, least weasel, long-tailed weasel, western polecat → **Mustela genus**. The PO added these back individually as "more iconic weasels," but at 1–3M parameters the model cannot reliably distinguish them. Genus level is a reasonable compromise between the PO's species-level and the original family-level consolidation.
- *Otters:* Kept 4 highly distinctive otters (N. American river otter, Eurasian otter, sea otter, giant otter). Removed 5 similar-looking medium-sized freshwater otters (neotropical otter, Asian small-clawed otter, cape clawless otter, spotted-necked otter, smooth-coated otter).
- *Other removals:* Asian badger (similar to Eurasian), grisons (rare), African striped weasel (rare), Chinese ferret-badger (range-limited).

**Hyaenids (4 kept):** All four — spotted hyena, striped hyena, brown hyena, and aardwolf — are visually distinct and commonly encountered in Africa.

**Procyonids (5, down from 9):** Kept raccoon, ringtail, white-nosed coati, South American coati, kinkajou. Removed: crab-eating raccoon (similar to raccoon), cacomistle (similar to ringtail), western mountain coati (very similar to coati, range-limited), olingo (elusive, nocturnal).

**Herpestids (2 kept):** Already consolidated by PO to family + meerkat.

**Mephitids (2, down from 7):** Kept striped skunk (most common/iconic). Consolidated spotted skunks to Spilogale genus. Removed: hooded skunk (similar to striped), all three hog-nosed skunks (Conepatus — rare, similar to each other).

**Pinnipeds (4 kept):** PO's existing consolidation — pinnipedia clade + elephant seal genus + eared seals family + walrus.

**Viverrids + Eupleridae (4, down from ~16):** This group was heavily pruned. Nearly all civets are nocturnal, range-limited, and visually similar (dark elongated body with spots/bands). Kept: African civet (distinctive, common), binturong (very distinctive), genetta genus (PO consolidation), fossa (iconic Madagascar predator). Removed 12 species including common palm civet, masked palm civet, small-toothed palm civet, all Viverra species, banded civet, central African oyan, brown palm civet, African palm civet, spotted fanaloka, eastern falanouc.

### 5.2 Cetartiodactyla (65 labels, down from ~140 in PO list)

**Cervidae (14, down from ~22):** Most cervids are retained — deer species are generally distinctive (antler shape, body size, habitat) and frequently encountered by binocular users. Removed: Dama mesopotamica (Persian fallow deer — nearly identical to common fallow, critically endangered), Rusa timorensis (Javan deer — similar to sambar, range-limited), Blastocerus dichotomus (marsh deer — limited range), Hippocamelus antisensis (taruca — rare Andean), Ozotoceros bezoarticus (pampas deer — limited range), Elaphodus cephalophus (tufted deer — range-limited), Pudu mephistophiles and Pudu puda (pudus — interesting but extremely rare/range-limited; PO added southern pudu back but given model constraints, removed).

**Bovidae — African antelopes (25, down from ~40):** Safari animals are critical for the premium binocular market. Kept all commonly seen species on East/Southern African safaris. Key consolidations:

- *Duikers:* 12+ forest duiker species (Cephalophus spp.) → **Cephalophus genus**. Forest duikers are small, elusive, and visually near-identical. Only common duiker (Sylvicapra grimmia) kept at species level — it's an open-habitat species, much more commonly seen.
- *Removed antelopes:* Tragelaphus imberbis (lesser kudu — similar to greater kudu), Tragelaphus spekii (sitatunga — swamp-dwelling, elusive), Damaliscus lunatus (topi — very similar to hartebeest), various gazelles (dorcas, mountain, goitered, chinkara, Mongolian, red-fronted — consolidated out; Thomson's, Grant's, springbok, and gerenuk are sufficient), Oryx beisa and Oryx leucoryx (consolidated with gemsbok), various small antelopes (Sharpe's grysbok, cape grysbok, suni, oribi, grey rhebok, Guenther's dik-dik — too similar or range-limited).
- *Added back:* Hippotragus equinus (roan antelope) and Kobus kob (kob) — both commonly seen on safari and PO had included them.

**Bovidae — non-African (10, down from ~18):** Kept all commonly encountered mountain/domestic ungulates. Removed: Capra aegagrus (wild goat — too similar to domestic goat for model), Capra nubiana (similar to Alpine ibex), Ovis ammon (argali — range-limited), Pseudois nayaur (blue sheep — range-limited), Tetracerus quadricornis (four-horned antelope — range-limited), Budorcas taxicolor (takin — range-limited), Naemorhedus griseus (goral — range-limited), both Capricornis species (serows — range-limited).

**Bovidae — large/domestic (6):** All kept. Note: Bos grunniens (yak) had a duplicate entry in PO list — deduplicated.

**Suidae (4, down from 9):** Kept wild boar, domestic pig, warthog, and red river hog (very distinctive orange coloring). Removed: bushpig (elusive), forest hog (elusive), desert warthog (similar to common warthog), bearded pig (range-limited), Palawan bearded pig (range-limited).

**Other families:** Collared peccary, dromedary camel, Lama genus, pronghorn, hippopotamus, giraffe — all kept from PO.

*Removed from Cetartiodactyla:* Water chevrotain and lesser chevrotain (tiny, elusive), alpine musk deer (elusive), guanaco (merged into Lama genus).

### 5.3 Perissodactyla (10 labels, down from 11)

All equids kept (6 species — horse, donkey, 3 zebra species, Asiatic wild ass). All three tapirs kept — each is from a different continent and visually distinct (Malay tapir's black-and-white pattern is unmistakable). Rhinocerotidae family kept as PO consolidation.

### 5.4 Primates (24 labels, down from ~65)

This was the second most aggressively consolidated group. The PO list contained ~65 individual primate species. Many primate genera contain species that are phenotypically near-identical and geographically separated — a nano model cannot and should not try to distinguish them.

**Species kept (9):**
- *Homo sapiens* — essential for the model to recognize humans
- *Pan troglodytes* (chimpanzee) — iconic great ape, very distinctive
- *Pongo pygmaeus* (orangutan) — iconic, unmistakable orange fur
- *Chlorocebus pygerythrus* (vervet monkey) — most common African monkey, frequently seen on safari
- *Lemur catta* (ring-tailed lemur) — most iconic lemur, distinctive banded tail
- *Macaca fuscata* (Japanese macaque) — iconic "snow monkey," PO specifically added it
- *Mandrillus leucophaeus* (drill) — distinctive facial markings
- *Erythrocebus patas* (patas monkey) — distinctive long-legged build, common in Africa
- *Daubentonia madagascariensis* (aye-aye) — extremely distinctive appearance, iconic

**Genus consolidations (15):**

| Genus | PO Species Consolidated | Reason |
|-------|------------------------|--------|
| *Gorilla* | G. beringei, G. gorilla | Both iconic but nano model unlikely to distinguish eastern/western |
| *Papio* (baboons) | Already PO consolidated | — |
| *Alouatta* (howler monkeys) | Already PO consolidated | — |
| *Presbytis* (leaf monkeys) | Already PO consolidated + absorbs Trachypithecus (5 spp.) and Semnopithecus (2 spp.) — all Asian forest monkeys with similar leaf-monkey body plan |
| *Cercocebus* (mangabeys) | Already PO consolidated | — |
| *Callicebus* (titis) | Already PO consolidated | — |
| *Macaca* (macaques) | 11 PO species → genus fallback (Japanese macaque kept separately) | Macaques are extremely similar across species; only Japanese macaque is distinctive enough |
| *Cercopithecus* (guenons) | 10 PO species → genus | Guenons have subtle face markings only experts can distinguish |
| *Colobus* | Colobus guereza, C. angolensis, C. polykomos, Piliocolobus gordonorum, P. badius → genus | Black-and-white vs. red colobus are distinguishable, but within each group species are identical |
| *Ateles* (spider monkeys) | A. hybridus, A. belzebuth, A. chamek → genus | Nearly identical long-limbed primates |
| *Cebus* (capuchins) | Cebus (5 spp.) + Sapajus (4 spp.) → genus | Capuchins share the same body plan; subtle face mask differences unreliable |
| *Saguinus* (tamarins) | Saguinus (5 spp.) + Leontocebus (1 sp.) → genus | Small, fast, similar body proportions |
| *Eulemur* (true lemurs) | E. rufifrons, E. fulvus, E. rubriventer, E. albifrons + Hapalemur (2 spp.) + Prolemur simus + Propithecus + Microcebus → genus | Non-ring-tailed lemurs are generally similar; model can fall back to genus |
| *Saimiri* (squirrel monkeys) | S. boliviensis, S. sciureus → genus | Nearly identical |
| *Callithrix* (marmosets) | Callithrix jacchus + Mico (2 spp.) → genus | Similar tiny primates with tufted ears |

**Removed primates (~40):** All individual species within the consolidated genera above, plus: Pygathrix nemaeus (red-shanked douc — stunning but extremely range-limited, zoo only), Rhinopithecus roxellana (golden snub-nosed monkey — range-limited China), all individual Trachypithecus/Semnopithecus langur species (absorbed into Presbytis genus for practical purposes), Lophocebus albigena (absorbed into Cercocebus), Allochrocebus lhoesti (absorbed into Cercopithecus), Pithecia pithecia (white-faced saki — distinctive but rarely encountered), and several lemur species merged into Eulemur genus.

### 5.5 Rodentia (24 labels, down from ~45)

**North American squirrels reinstated:** Following the research recommendation and Danielle's note, four major NA tree squirrel species were confirmed at Tier 1 (eastern gray, fox, western gray, American red squirrel). Additionally kept: California ground squirrel (very common Pacific coast), golden-mantled ground squirrel (common trail species, often confused with chipmunks).

**Key consolidations:** All chipmunks to Tamias genus (PO already did this), flying squirrels to Glaucomys genus, agoutis to Dasyprocta genus (PO already did this), beavers to Castor genus (PO already did this).

**Removed rodents (~20):** Douglas's squirrel (very similar to red squirrel), all tropical/Asian squirrels (~10 species — sundasciurus, tamiops, callosciurus, funambulus, guerlinguetus, rheithrosciurus, aeromys, petaurista spp.), rock cavy, guinea pig, lesser capybara (similar to capybara), acouchi (similar to agouti), lord derby's scaly-tailed squirrel (range-limited), springhare (nocturnal), bushy-tailed woodrat, white-tailed antelope squirrel, Brazilian porcupine, spotted paca.

### 5.6 Marsupials + Monotremes (9, down from ~25)

Kept the five most commonly seen/iconic macropods (red kangaroo, eastern grey kangaroo, swamp wallaby, red-necked wallaby, quokka) plus wombat, koala, and echidna. Added Macropodidae family as a fallback for other wallaby/kangaroo species.

**Removed (~16):** All other wallaby species (common wallaroo, western grey kangaroo, agile wallaby, parma wallaby, whiptail wallaby, tammar wallaby, black-striped wallaby, rock wallaby, pademelons, spectacled hare-wallaby), sugar glider (nocturnal), greater glider (nocturnal), ringtail possum (nocturnal), quolls (2 spp.), phascogale, kaluta, mulgara, bandicoot order.

### 5.7 Other Orders

**Proboscidea (2):** Both elephant species kept — very distinctive.

**Pilosa (3, down from 4):** Giant anteater (very distinctive), brown-throated sloth, Hoffmann's two-toed sloth. The two sloth species are distinguishable (2 vs. 3 claws, different face shape).

**Pholidota (1, down from 5):** All 5 pangolin species consolidated to **Manidae family**. Pangolins all share the same unique scaled body plan and a nano model cannot distinguish species.

**Cingulata (2, down from 6):** Nine-banded armadillo (most common, widespread) and giant armadillo (very distinctive largest species). Removed: tolypeutes (three-banded), euphractus (yellow), cabassous × 2 (naked-tailed) — all similar armored body plan.

**Lagomorpha (6 kept):** PO's existing consolidation — European rabbit, Lepus genus + European hare, Sylvilagus genus + Eastern cottontail, Ochotona genus.

**Other (4):** Aardvark (iconic), rock hyrax (commonly seen on safari), hedgehog family, opossum family.

**Removed entirely:** All sengis/elephant shrews (5 spp. — tiny, fast, rarely seen), both treeshrews (range-limited), both shrews (too small for binocular ID).

---

## 6. Final Label Table

**225 output classes** organized by taxonomic order. Tier indicates: **S** = species, **G** = genus, **F** = family/clade.

### Carnivora (75)

| # | Scientific Name | Common Name | Tier | Notes |
|--:|:----------------|:------------|:----:|:------|
| 1 | *Ailuropoda melanoleuca* | Giant panda | S | |
| 2 | *Helarctos malayanus* | Sun bear | S | |
| 3 | *Melursus ursinus* | Sloth bear | S | |
| 4 | *Tremarctos ornatus* | Spectacled bear | S | |
| 5 | *Ursus americanus* | American black bear | S | |
| 6 | *Ursus arctos* | Brown / grizzly bear | S | |
| 7 | *Ursus thibetanus* | Asiatic black bear | S | |
| 8 | *Ailurus fulgens* | Red panda | S | |
| 9 | *Canis lupus* | Grey wolf | S | |
| 10 | *Canis latrans* | Coyote | S | |
| 11 | *Canis familiaris* | Domestic dog | S | |
| 12 | *Canis lupus dingo* | Dingo | S | |
| 13 | *Canis mesomelas* | Black-backed jackal | S | |
| 14 | *Canis aureus* | Golden jackal | S | |
| 15 | *Vulpes vulpes* | Red fox | S | |
| 16 | *Urocyon cinereoargenteus* | Grey fox | S | |
| 17 | *Lycaon pictus* | African wild dog | S | |
| 18 | *Chrysocyon brachyurus* | Maned wolf | S | |
| 19 | *Cuon alpinus* | Dhole | S | |
| 20 | *Nyctereutes procyonoides* | Raccoon dog | S | |
| 21 | *Otocyon megalotis* | Bat-eared fox | S | |
| 22 | *Lycalopex* | South American foxes | G | Consolidates 3 PO species |
| 23 | *Panthera leo* | Lion | S | |
| 24 | *Panthera pardus* | Leopard | S | |
| 25 | *Panthera tigris* | Tiger | S | |
| 26 | *Panthera onca* | Jaguar | S | |
| 27 | *Panthera uncia* | Snow leopard | S | |
| 28 | *Acinonyx jubatus* | Cheetah | S | |
| 29 | *Puma concolor* | Puma / cougar | S | |
| 30 | *Lynx rufus* | Bobcat | S | |
| 31 | *Lynx canadensis* | Canada lynx | S | |
| 32 | *Lynx lynx* | Eurasian lynx | S | |
| 33 | *Felis catus* | Domestic cat | S | |
| 34 | *Felis silvestris* | Wildcat | S | |
| 35 | *Leptailurus serval* | Serval | S | |
| 36 | *Caracal caracal* | Caracal | S | |
| 37 | *Neofelis nebulosa* | Clouded leopard | S | |
| 38 | *Prionailurus bengalensis* | Leopard cat | S | |
| 39 | *Leopardus* | Leopardus cats | G | PO consolidation |
| 40 | *Leopardus pardalis* | Ocelot | S | PO added back from genus |
| 41 | *Gulo gulo* | Wolverine | S | |
| 42 | *Mellivora capensis* | Honey badger | S | |
| 43 | *Taxidea taxus* | American badger | S | |
| 44 | *Meles meles* | Eurasian badger | S | |
| 45 | *Lontra canadensis* | N. American river otter | S | |
| 46 | *Lutra lutra* | Eurasian otter | S | |
| 47 | *Enhydra lutris* | Sea otter | S | |
| 48 | *Pteronura brasiliensis* | Giant otter | S | |
| 49 | *Neovison vison* | American mink | S | |
| 50 | *Pekania pennanti* | Fisher | S | |
| 51 | *Eira barbara* | Tayra | S | |
| 52 | *Martes* | Martens | G | Consolidates 4 PO species |
| 53 | *Mustela* | Weasels | G | Consolidates 4 PO species |
| 54 | *Arctonyx* | Hog badgers | G | PO consolidation |
| 55 | *Crocuta crocuta* | Spotted hyena | S | |
| 56 | *Hyaena hyaena* | Striped hyena | S | |
| 57 | *Parahyaena brunnea* | Brown hyena | S | |
| 58 | *Proteles cristata* | Aardwolf | S | |
| 59 | *Procyon lotor* | Northern raccoon | S | |
| 60 | *Bassariscus astutus* | Ringtail | S | |
| 61 | *Nasua narica* | White-nosed coati | S | |
| 62 | *Nasua nasua* | South American coati | S | |
| 63 | *Potos flavus* | Kinkajou | S | |
| 64 | *Herpestidae* | Mongooses | F | PO consolidation |
| 65 | *Suricata suricatta* | Meerkat | S | PO added back from family |
| 66 | *Mephitis mephitis* | Striped skunk | S | |
| 67 | *Spilogale* | Spotted skunks | G | Consolidates 2 PO species |
| 68 | *Pinnipedia* | Seals and sea lions | F | PO consolidation (clade) |
| 69 | *Mirounga* | Elephant seals | G | PO consolidation |
| 70 | *Otariidae* | Eared seals | F | PO consolidation |
| 71 | *Odobenus rosmarus* | Walrus | S | |
| 72 | *Civettictis civetta* | African civet | S | |
| 73 | *Arctictis binturong* | Binturong | S | |
| 74 | *Genetta* | Genets | G | PO consolidation |
| 75 | *Cryptoprocta ferox* | Fossa | S | |

### Cetartiodactyla (65)

| # | Scientific Name | Common Name | Tier | Notes |
|--:|:----------------|:------------|:----:|:------|
| 76 | *Odocoileus virginianus* | White-tailed deer | S | |
| 77 | *Odocoileus hemionus* | Mule deer | S | |
| 78 | *Cervus canadensis* | Elk / wapiti | S | |
| 79 | *Cervus elaphus* | Red deer | S | |
| 80 | *Cervus nippon* | Sika deer | S | |
| 81 | *Alces alces* | Moose | S | |
| 82 | *Rangifer tarandus* | Reindeer / caribou | S | |
| 83 | *Capreolus capreolus* | European roe deer | S | |
| 84 | *Dama dama* | Common fallow deer | S | |
| 85 | *Axis axis* | Chital / spotted deer | S | |
| 86 | *Hydropotes inermis* | Water deer | S | |
| 87 | *Rusa unicolor* | Sambar | S | |
| 88 | *Muntiacus* | Muntjacs | G | PO consolidation |
| 89 | *Mazama americana* | Red brocket | S | PO added back |
| 90 | *Aepyceros melampus* | Impala | S | |
| 91 | *Connochaetes taurinus* | Blue wildebeest | S | |
| 92 | *Connochaetes gnou* | Black wildebeest | S | |
| 93 | *Eudorcas thomsonii* | Thomson's gazelle | S | |
| 94 | *Nanger granti* | Grant's gazelle | S | |
| 95 | *Antidorcas marsupialis* | Springbok | S | |
| 96 | *Oryx gazella* | Gemsbok | S | |
| 97 | *Litocranius walleri* | Gerenuk | S | |
| 98 | *Alcelaphus buselaphus* | Hartebeest | S | |
| 99 | *Kobus ellipsiprymnus* | Waterbuck | S | |
| 100 | *Kobus kob* | Kob | S | PO added |
| 101 | *Tragelaphus oryx* | Common eland | S | |
| 102 | *Tragelaphus strepsiceros* | Greater kudu | S | |
| 103 | *Tragelaphus angasii* | Nyala | S | |
| 104 | *Tragelaphus scriptus* | Bushbuck | S | |
| 105 | *Tragelaphus eurycerus* | Bongo | S | |
| 106 | *Hippotragus niger* | Sable antelope | S | |
| 107 | *Hippotragus equinus* | Roan antelope | S | |
| 108 | *Raphicerus campestris* | Steenbok | S | |
| 109 | *Madoqua kirkii* | Kirk's dik-dik | S | |
| 110 | *Sylvicapra grimmia* | Common duiker | S | |
| 111 | *Oreotragus oreotragus* | Klipspringer | S | |
| 112 | *Damaliscus pygargus* | Blesbok / bontebok | S | Deduplicated from PO |
| 113 | *Redunca* | Reedbucks | G | PO consolidation |
| 114 | *Cephalophus* | Forest duikers | G | Consolidates 12+ PO species |
| 115 | *Ovis canadensis* | Bighorn sheep | S | |
| 116 | *Ovis aries* | Domestic sheep | S | |
| 117 | *Ovis orientalis* | Mouflon | S | |
| 118 | *Oreamnos americanus* | Mountain goat | S | |
| 119 | *Capra ibex* | Alpine ibex | S | |
| 120 | *Capra aegagrus hircus* | Domestic goat | S | |
| 121 | *Rupicapra rupicapra* | Northern chamois | S | |
| 122 | *Antilope cervicapra* | Blackbuck | S | PO added |
| 123 | *Boselaphus tragocamelus* | Nilgai | S | |
| 124 | *Saiga tatarica* | Saiga | S | PO added |
| 125 | *Syncerus caffer* | African buffalo | S | |
| 126 | *Bison bison* | American bison | S | |
| 127 | *Bison bonasus* | European bison | S | |
| 128 | *Bos taurus* | Domestic cattle | S | |
| 129 | *Bos grunniens* | Yak | S | PO added |
| 130 | *Bubalus bubalis* | Domestic water buffalo | S | |
| 131 | *Hippopotamus amphibius* | Hippopotamus | S | |
| 132 | *Giraffa camelopardalis* | Giraffe | S | |
| 133 | *Sus scrofa* | Wild boar | S | |
| 134 | *Sus scrofa scrofa* | Domestic pig | S | |
| 135 | *Phacochoerus africanus* | Common warthog | S | |
| 136 | *Potamochoerus porcus* | Red river hog | S | |
| 137 | *Pecari tajacu* | Collared peccary | S | |
| 138 | *Camelus dromedarius* | Dromedary camel | S | |
| 139 | *Lama* | Llamas and alpacas | G | PO consolidation |
| 140 | *Antilocapra americana* | Pronghorn | S | |

### Perissodactyla (10)

| # | Scientific Name | Common Name | Tier | Notes |
|--:|:----------------|:------------|:----:|:------|
| 141 | *Equus caballus* | Domestic horse | S | |
| 142 | *Equus asinus* | Domestic donkey | S | |
| 143 | *Equus quagga* | Plains zebra | S | |
| 144 | *Equus grevyi* | Grevy's zebra | S | |
| 145 | *Equus zebra* | Mountain zebra | S | |
| 146 | *Equus hemionus* | Asiatic wild ass | S | |
| 147 | *Tapirus terrestris* | Lowland tapir | S | |
| 148 | *Tapirus indicus* | Malay tapir | S | |
| 149 | *Tapirus bairdii* | Baird's tapir | S | |
| 150 | *Rhinocerotidae* | Rhinoceroses | F | PO consolidation |

### Primates (24)

| # | Scientific Name | Common Name | Tier | Notes |
|--:|:----------------|:------------|:----:|:------|
| 151 | *Homo sapiens* | Human | S | |
| 152 | *Pan troglodytes* | Chimpanzee | S | |
| 153 | *Pongo pygmaeus* | Bornean orangutan | S | |
| 154 | *Chlorocebus pygerythrus* | Vervet monkey | S | |
| 155 | *Lemur catta* | Ring-tailed lemur | S | PO added back |
| 156 | *Macaca fuscata* | Japanese macaque | S | PO added |
| 157 | *Mandrillus leucophaeus* | Drill | S | |
| 158 | *Erythrocebus patas* | Patas monkey | S | |
| 159 | *Daubentonia madagascariensis* | Aye-aye | S | |
| 160 | *Gorilla* | Gorillas | G | Consolidates 2 PO species |
| 161 | *Papio* | Baboons | G | PO consolidation |
| 162 | *Alouatta* | Howler monkeys | G | PO consolidation |
| 163 | *Presbytis* | Leaf monkeys / langurs | G | PO consolidation + absorbs Trachypithecus, Semnopithecus |
| 164 | *Cercocebus* | Mangabeys | G | PO consolidation |
| 165 | *Callicebus* | Titi monkeys | G | PO consolidation |
| 166 | *Macaca* | Macaques | G | Consolidates 11 PO species (fallback) |
| 167 | *Cercopithecus* | Guenons | G | Consolidates 10 PO species |
| 168 | *Colobus* | Colobus monkeys | G | Consolidates 5 PO species |
| 169 | *Ateles* | Spider monkeys | G | Consolidates 3 PO species |
| 170 | *Cebus* | Capuchins | G | Consolidates 9 PO species (Cebus + Sapajus) |
| 171 | *Saguinus* | Tamarins | G | Consolidates 6 PO species |
| 172 | *Eulemur* | True lemurs | G | Consolidates 8+ PO lemur species |
| 173 | *Saimiri* | Squirrel monkeys | G | Consolidates 2 PO species |
| 174 | *Callithrix* | Marmosets | G | Consolidates 3 PO species |

### Rodentia (24)

| # | Scientific Name | Common Name | Tier | Notes |
|--:|:----------------|:------------|:----:|:------|
| 175 | *Sciurus carolinensis* | Eastern gray squirrel | S | |
| 176 | *Sciurus niger* | Eastern fox squirrel | S | |
| 177 | *Sciurus griseus* | Western gray squirrel | S | |
| 178 | *Sciurus vulgaris* | Eurasian red squirrel | S | |
| 179 | *Tamiasciurus hudsonicus* | American red squirrel | S | |
| 180 | *Callospermophilus lateralis* | Golden-mantled ground squirrel | S | |
| 181 | *Otospermophilus beecheyi* | California ground squirrel | S | PO added back |
| 182 | *Marmota monax* | Groundhog / woodchuck | S | |
| 183 | *Marmota marmota* | Alpine marmot | S | |
| 184 | *Marmota flaviventris* | Yellow-bellied marmot | S | |
| 185 | *Cynomys ludovicianus* | Black-tailed prairie dog | S | |
| 186 | *Erethizon dorsatum* | N. American porcupine | S | |
| 187 | *Hydrochoerus hydrochaeris* | Capybara | S | |
| 188 | *Ondatra zibethicus* | Muskrat | S | |
| 189 | *Myocastor coypus* | Nutria / coypu | S | |
| 190 | *Tamias* | Chipmunks | G | PO consolidation |
| 191 | *Glaucomys* | Flying squirrels | G | Consolidates 2 PO species |
| 192 | *Dasyprocta* | Agoutis | G | PO consolidation |
| 193 | *Castor* | Beavers | G | PO consolidation |
| 194 | *Rattus* | Rats | G | PO consolidation |
| 195 | *Muridae* | Old World mice / rats | F | PO consolidation |
| 196 | *Sciuridae* | Squirrel family (fallback) | F | PO consolidation |
| 197 | *Cricetidae* | New World mice / voles | F | PO consolidation |
| 198 | *Hystricidae* | Old World porcupines | F | PO consolidation |

### Marsupials and Monotremes (9)

| # | Scientific Name | Common Name | Tier | Notes |
|--:|:----------------|:------------|:----:|:------|
| 199 | *Osphranter rufus* | Red kangaroo | S | |
| 200 | *Macropus giganteus* | Eastern grey kangaroo | S | |
| 201 | *Wallabia bicolor* | Swamp wallaby | S | |
| 202 | *Macropus rufogriseus* | Red-necked wallaby | S | |
| 203 | *Setonix brachyurus* | Quokka | S | |
| 204 | *Vombatus ursinus* | Common wombat | S | |
| 205 | *Phascolarctos cinereus* | Koala | S | |
| 206 | *Macropodidae* | Kangaroos / wallabies (fallback) | F | New fallback |
| 207 | *Tachyglossus aculeatus* | Short-beaked echidna | S | |

### Other Orders (16)

| # | Scientific Name | Common Name | Tier | Order | Notes |
|--:|:----------------|:------------|:----:|:------|:------|
| 208 | *Loxodonta africana* | African elephant | S | Proboscidea | |
| 209 | *Elephas maximus* | Asian elephant | S | Proboscidea | |
| 210 | *Myrmecophaga tridactyla* | Giant anteater | S | Pilosa | |
| 211 | *Bradypus variegatus* | Brown-throated sloth | S | Pilosa | |
| 212 | *Choloepus hoffmanni* | Hoffmann's two-toed sloth | S | Pilosa | |
| 213 | *Manidae* | Pangolins | F | Pholidota | Consolidates 5 PO species |
| 214 | *Dasypus novemcinctus* | Nine-banded armadillo | S | Cingulata | |
| 215 | *Priodontes maximus* | Giant armadillo | S | Cingulata | |
| 216 | *Oryctolagus cuniculus* | European rabbit | S | Lagomorpha | |
| 217 | *Lepus* | Hares / jackrabbits | G | Lagomorpha | PO consolidation |
| 218 | *Lepus europaeus* | European hare | S | Lagomorpha | PO added back |
| 219 | *Sylvilagus* | Cottontail rabbits | G | Lagomorpha | PO consolidation |
| 220 | *Sylvilagus floridanus* | Eastern cottontail | S | Lagomorpha | PO added back |
| 221 | *Ochotona* | Pikas | G | Lagomorpha | PO consolidation |
| 222 | *Orycteropus afer* | Aardvark | S | Tubulidentata | |
| 223 | *Procavia capensis* | Rock hyrax | S | Hyracoidea | |
| 224 | *Erinaceidae* | Hedgehogs | F | Eulipotyphla | PO consolidation |
| 225 | *Didelphidae* | Opossums | F | Didelphimorphia | PO consolidation |

---

## 7. Summary Statistics

| Metric | Value |
|--------|------:|
| **Total output classes** | **225** |
| Species-level (Tier 1) | 164 |
| Genus-level (Tier 2) | 41 |
| Family/clade-level (Tier 3) | 20 |
| | |
| PO entries kept as-is | ~208 |
| New genus/family consolidations | 17 |
| PO entries removed/consolidated | ~275 |
| | |
| **By order** | |
| Carnivora | 75 (33%) |
| Cetartiodactyla | 65 (29%) |
| Perissodactyla | 10 (4%) |
| Primates | 24 (11%) |
| Rodentia | 24 (11%) |
| Marsupials + Monotremes | 9 (4%) |
| Other orders | 18 (8%) |

### Comparison with PO list

| Group | PO count | Final count | Reduction |
|-------|:--------:|:-----------:|:---------:|
| Primates | ~65 | 24 | −63% |
| Mustelids | ~34 | 14 | −59% |
| Viverrids/Eupleridae | ~16 | 4 | −75% |
| Duikers (Cephalophus) | ~13 | 2 | −85% |
| Marsupials | ~25 | 9 | −64% |
| Canids | ~24 | 14 | −42% |
| Felids | ~25 | 18 | −28% |
| Bovidae (African) | ~40 | 25 | −38% |
| Cervidae | ~22 | 14 | −36% |
| Squirrels | ~8 | 8 | 0% |

---

## 8. Output Files

- **This document:** Analysis and reasoning for the label selection
- **[`resources/2026-03-19_student_model_labels.txt`](../resources/2026-03-19_student_model_labels.txt):** Final 225 labels in SpeciesNet taxonomy format (`UUID;class;order;family;genus;species;common_name`)
- **Source data:** [`resources/2026-03-17_SN_labels_refined_only.txt`](../resources/2026-03-17_SN_labels_refined_only.txt) (PO's 483 labels), [`resources/speciesnet_taxonomy_release.txt`](../resources/speciesnet_taxonomy_release.txt) (full SpeciesNet taxonomy)

---

## 9. Next Steps

1. Review with Danielle — discuss removed species and any strong objections
2. Cross-reference against actual training data availability (iNaturalist, SpeciesNet training image counts)
3. Verify that all 225 labels have sufficient training data (target: >1,000 images per class)
4. For classes with insufficient data: consider synthetic augmentation or further consolidation
5. Finalize the label mapping and begin model training pipeline

---

## Works Cited

Works cited numbers reference the research report [`docs/species-label-research.md`](species-label-research.md).

1. Qualcomm QCS603/605 SoCs for IoT product brief
5. AddaxAI — Simplifying camera trap image analysis with AI
6. YOLO11 vs YOLOv8: A Comprehensive Technical Comparison (Ultralytics)
9. Fast, Accurate Detection of 100,000 Object Classes on a Single Machine (Google Research)
11. Optimal Size of Agricultural Dataset for YOLOv8 Training (Semantic Scholar)
15. Benchmarking YOLOv8 to YOLOv11 for Real-Time Traffic Sign Recognition (MDPI)
16. A Comparative Benchmark of Real-time Detectors for Blueberry Detection (arXiv)
18. SNAPSHOT USA: First-ever nationwide mammal survey (NC Museum of Natural Sciences)
21. Fine-Tuning Without Forgetting: Adaptation of YOLOv8 Preserves COCO Performance (arXiv)
25. The effect of grouping classes into hierarchical structures for object detection (TU Delft)
34. Squirrels — Montana Field Guide
40. North America's Most Common Squirrels and How to Tell Them Apart (A-Z Animals)
45. Different Types Of Squirrels Found in North America (Active Wild)
46. Tree Squirrels (Internet Center for Wildlife Damage Management)
64. Hierarchical Multi-Label Object Detection Framework for Remote Sensing Images (MDPI)
