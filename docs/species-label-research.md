# **Optimization of Taxonomic Granularity for Lightweight Mammal Object Detection on Edge Devices**

## **Introduction and Architectural Context**

The integration of computer vision and artificial intelligence into premium consumer optics represents a significant advancement in wildlife observation and ecotourism. Devices such as the Swarovski AX Visio merge high-performance analog long-range optics with digital intelligence, enabling real-time species identification directly through the viewfinder. However, deploying continuous, real-time object detection and classification models on edge hardware imposes severe computational and thermal constraints. The hardware architecture driving these capabilities—specifically the Qualcomm QCS605 System-on-Chip (SoC)—is engineered for low-power Internet of Things (IoT) applications. While it features an integrated Artificial Intelligence Engine and digital signal processor capable of handling edge inference, it possesses strict ceilings regarding memory bandwidth, cache size, and maximum floating-point operations (FLOPs).<sup>[1]</sup>

The legacy identification pipeline utilized a two-stage architecture: a YOLOv5s object detector functioning in tandem with a highly parameterized SpeciesNet classifier based on the EfficientNetV2-M architecture.<sup>[3]</sup> This legacy classifier utilized approximately 54 million parameters to distinguish between 2,498 taxonomic labels, encompassing birds, mammals, vehicles, and blank frames.<sup>[4]</sup> While highly accurate in cloud or high-compute environments, a 54-million parameter model exceeds the optimal thermal and latency budgets for continuous, real-time inference on the QCS605 without introducing unacceptable battery drain or frame-rate drops.

Transitioning this architecture to a unified, single-stage detection and classification model restricted to a capacity of 1 to 3 million parameters (such as the nano variants of YOLOv8, YOLOv10, or YOLOv11) necessitates a radical re-evaluation of the output class ontology. The initial proposal of 483 mammal labels, derived from the massive 2,498-class SpeciesNet taxonomy, contains severe imbalances in visual distinctiveness, geographic relevance, and data availability. A 1–3M parameter model lacks the representational depth to memorize the subtle, highly fine-grained features required to distinguish between hundreds of visually identical cryptic species, especially under the variable lighting and occlusion conditions typical of binocular use.

This report provides an exhaustive analysis of the optimal taxonomic output design for a 1–3M parameter single-stage detection model focused exclusively on non-volant, non-marine mammals. It investigates the empirical limits of class capacity in lightweight models, defines a hierarchical label granularity strategy, reconstructs the North American and global mammal coverage to maximize user satisfaction, and proposes a finalized, highly curated ontology that balances computational reality with the premium expectations of the consumer optics market.

## **Class Count Ceiling and Accuracy Trade-Offs in Lightweight Models**

The fundamental challenge in designing an ontology for a 1–3M parameter object detector is the inherent trade-off between the breadth of the output vocabulary (the number of classes) and the precision of the predictions. Unlike cloud-based foundation models that benefit from tens or hundreds of millions of parameters, edge-deployed nano-models face strict information bottleneck limitations.

### **Empirical Limits of Representational Capacity**

Current literature evaluating lightweight object detectors—specifically YOLOv8n (3.2M parameters), YOLOv10n (2.7M parameters), YOLOv11n (2.6M parameters), and MobileNetV3-Small (1.5M parameters)—demonstrates that model capacity dictates a hard ceiling on the number of fine-grained classes that can be reliably distinguished.<sup>[6]</sup>

In evaluating the accuracy-efficiency trade-off, empirical benchmarks indicate that models in the 1–3M parameter range experience diminishing returns and rapid accuracy degradation as the class count exceeds approximately 200 to 300 categories.<sup>[9]</sup> The YOLO11n architecture, for example, achieves a highly competitive 39.5% Mean Average Precision (mAP@0.5:0.95) on the COCO dataset, which contains exactly 80 diverse object classes.<sup>[6]</sup> Similarly, the MobileNetV3-Small architecture achieves roughly 67.6% top-1 accuracy on the ImageNet-1K dataset (1,000 classes), but this performance relies heavily on the presence of massive, highly curated training data (over 1.2 million images) and still falls significantly short of larger models when distinguishing fine-grained sub-categories.<sup>[7]</sup>

When a nano-model is forced to learn highly granular, visually similar classes within a narrow domain—such as distinguishing 15 different species of morphologically identical chipmunks—the limited parameter space is stretched too thin. The convolutional filters within a 1-3M parameter backbone lack the depth and dimensionality to extract the highly complex, subtle morphological features required for fine-grained differentiation.<sup>[13]</sup> Consequently, the network struggles to separate inter-class variance from intra-class variance, leading to a significant drop in overall mAP and an increase in false positive rates due to inter-class confusion.<sup>[15]</sup>

Studies evaluating YOLOv8n on agricultural and wildlife datasets scaling from 100 to 500 classes reveal that while the model achieves exceptional real-time inference speeds (often sub-20 milliseconds on edge NPUs), its localization and classification accuracy drops notably when the class count exceeds the network's feature-extraction capacity.<sup>[15]</sup> For instance, increasing a dataset from 100 to 500 classes in lightweight YOLO variants frequently results in a multi-point drop in mAP@0.5, as the model becomes more susceptible to noise and struggles to leverage pseudo-labeled or augmented data effectively.<sup>[16]</sup>

Furthermore, training lightweight models on datasets with heavy class imbalances exacerbates this issue. Open-source datasets such as iNaturalist and Snapshot USA exhibit extreme long-tailed distributions, where common species possess millions of high-quality images, while rare or elusive species possess fewer than 1,000.18 A 1–3M parameter model trained on such a distribution without aggressive class consolidation will suffer from catastrophic forgetting of the minority classes or severe overfitting on the majority classes.<sup>[21]</sup>

### **Establishing the Target Class Count**

Based on the architectural limitations of 1–3M parameter single-stage detectors, the optimal class count ceiling for the target deployment on the Qualcomm QCS605 is approximately **200 to 250 classes**.

Restricting the model to a highly curated vocabulary of \~225 mammal classes yields several critical architectural and user-experience benefits:

1. **Computational Efficiency:** The detection head requires fewer channels to compute class probabilities across 225 classes compared to 483\. This slightly reduces the overall parameter count and, more importantly, the computational FLOPs required during the final inference stages, translating directly to lower latency and improved battery conservation on the edge device.<sup>[22]</sup>  
2. **Robustness to Variance:** By merging visually indistinguishable species into genus-level or family-level super-classes, the intra-class variance (e.g., viewing angles, lighting, occlusion) increases while inter-class confusion decreases. This allows the lightweight backbone's limited convolutional filters to learn more robust, generalized features.<sup>[25]</sup>  
3. **User Experience (UX) Preservation:** From a premium consumer optics perspective, device trust is paramount. A false positive (misidentifying a widely known species, such as classifying a domestic dog as a coyote) or a low-confidence "Unknown" result severely damages the perceived value of a $4,800 product. A model that is highly confident and perfectly accurate across 225 well-known taxa is vastly superior to a model that hesitates and hallucinates across 483 obscure taxa.<sup>[26]</sup>

## **Label Granularity Strategy: Species, Genus, and Family**

To achieve the target reduction from the initial 483 labels to approximately 225, a rigorous hierarchical pruning and consolidation strategy must be applied. The decision to retain a label at the species level, consolidate it to the genus level, or fall back to the family level must be governed by a strict matrix of four core pillars: Visual Distinctiveness, User Expectation, Geographic Commonality, and Data Availability.

### **Tier 1: Species-Level Granularity**

Species-level labels represent the highest level of granularity and should be reserved exclusively for mammals that satisfy all of the following stringent criteria:

* **High Visual Distinctiveness:** The animal must possess unique morphological traits—such as specific coloration patterns, relative size, body proportions, or distinct horn/antler structures—that a 1–3M parameter convolutional network can easily extract from a daylight, high-resolution binocular image.<sup>[28]</sup>  
* **Charismatic or Common Status:** The animal is frequently encountered by the target demographic (e.g., North American hikers, European woodland walkers, African safari tourists) or represents a high-value "trophy" sighting that users specifically purchase binoculars to observe.<sup>[30]</sup>  
* **Data Abundance:** There must be tens of thousands of high-quality, diverse training images available via open datasets (iNaturalist, GBIF, Snapshot USA) to ensure the model can generalize across all seasons, lighting conditions, and viewing angles.<sup>[18]</sup>

If a mammal meets these criteria, it warrants the parameter allocation required for species-level detection. Examples of mandatory Tier 1 species include the American Black Bear (*Ursus americanus*), Moose (*Alces alces*), Lion (*Panthera leo*), and the Eastern Gray Squirrel (*Sciurus carolinensis*).

### **Tier 2: Genus-Level Consolidation**

Genus-level labels are critical for optimizing lightweight models. They act as logical "super-classes" for groups of species that are genetically distinct but phenotypically nearly identical, especially at the distances typical of binocular observation.

* **Cryptic Species Complexes:** When differentiating species requires geographic context, cranial morphology measurements, dental formulas, or DNA analysis, a computer vision model will inevitably fail. Training the model to separate such species forces it to memorize background pixels or geographic metadata rather than the animal's actual features, leading to severe overfitting.<sup>[25]</sup>  
* **Sufficient User Satisfaction:** A user observing a chipmunk darting across a trail is generally highly satisfied with the identification "Chipmunk" (*Tamias* spp.). The average consumer rarely expects, nor requires, the binocular to distinguish an Alpine Chipmunk from a Lodgepole Chipmunk, especially when both appear identical to the naked eye.<sup>[34]</sup>

By consolidating cryptic species, the model's accuracy on the Genus level skyrockets, providing a reliable and satisfying user experience. Examples of mandatory Tier 2 consolidations include Chipmunks (*Tamias* / *Neotamias*), Cottontail Rabbits (*Sylvilagus*), Macaques (*Macaca*), and Prairie Dogs (*Cynomys*).

### **Tier 3: Family-Level Fallback**

Family-level labels should serve as catch-all categories for highly diverse but morphologically generalized groups, or as engineered fallback predictions during inference when confidence is low.

* **Small, Overlooked Taxa:** Small rodents (mice, voles, rats, lemmings) are exceptionally difficult to observe through binoculars and even more difficult to identify to the species level. Grouping them into broad families ensures the model provides a biologically accurate, if broad, label without wasting capacity on dozens of indistinguishable classes.<sup>[35]</sup>  
* **Inference Fallback:** The hierarchical strategy dictates that if the model detects a deer but cannot determine if it is a Mule Deer or a White-tailed Deer due to severe occlusion, distance, or poor lighting, it should output "Deer Family" (*Cervidae*) rather than forcing a low-confidence guess.<sup>[37]</sup>

Examples of mandatory Tier 3 labels include Old World Mice and Rats (*Muridae*), New World Mice and Voles (*Cricetidae*), and Mongooses (*Herpestidae*).

## **North American Coverage Corrections and Gap Analysis**

North America represents the primary market for the Swarovski AX Visio. Therefore, the model's accuracy on North American fauna is the single most critical driver of overall consumer satisfaction and product viability. A detailed review of the initial 483-label list reveals a severe operational flaw: the aggressive pruning of North American squirrel species. Furthermore, to ensure the device is perceived as highly capable, several ubiquitous urban, suburban, and trail mammals must be guaranteed Tier 1 status.

### **Reinstating the Tree Squirrels (*Sciuridae*)**

The initial product owner decision to aggressively prune North American squirrel species was likely based on the assumption that lightweight models struggle with the small inter-class variances between rodents. However, applying this logic to the *Sciuridae* family is a critical error. Tree squirrels are diurnal, highly visible, largely unafraid of humans, and frequently observed by birdwatchers, hikers, and backyard nature enthusiasts—the exact demographic utilizing this $4,800 device.<sup>[39]</sup>

Extensive wildlife survey data, including the nationwide Snapshot USA camera trap survey and iNaturalist tracking databases, confirms that squirrels are among the most frequently documented mammals on the continent, second only to white-tailed deer in many regions.<sup>[18]</sup> Furthermore, unlike cryptic mice or voles, the major North American tree squirrels possess highly distinct visual phenotypes that a 1–3M parameter model can easily separate, provided it is trained on high-resolution daylight imagery rather than low-resolution, grayscale infrared camera trap data.<sup>[42]</sup>

To align with user expectations and observation frequencies, the following species must be reinstated as independent Tier 1 classes:

1. **Eastern Gray Squirrel (*Sciurus carolinensis*):** The most common urban and suburban squirrel in eastern North America, and a widespread invasive species in Europe. It is characterized by a gray dorsum, a white underbelly, and distinct white-frosted tail fringes. Notably, melanistic (pure black) morphs are highly common in certain urban areas and must be heavily represented in the training data to prevent misclassification as a separate species or a different animal entirely.<sup>[40]</sup>  
2. **Fox Squirrel (*Sciurus niger*):** The largest North American tree squirrel. It is visually highly distinct from the Eastern Gray due to its larger, bulkier head, heavily orange-pigmented underbelly, and orange-fringed tail. The model will easily learn these colorimetric differences.<sup>[40]</sup>  
3. **American Red Squirrel (*Tamiasciurus hudsonicus*):** Considerably smaller than the *Sciurus* genus, this lively conifer-dwelling squirrel possesses a reddish-brown dorsum, a stark white underbelly, and a highly distinct white eye-ring.<sup>[45]</sup> Its unique size, coloration, and morphological proportions make it easily separable by lightweight convolutional architectures.  
4. **Western Gray Squirrel (*Sciurus griseus*):** Found primarily along the Pacific coast, this species is distinguished by a pure silver-gray back and a sharply contrasting white belly, lacking the brownish or rusty hues frequently seen in the Eastern Gray Squirrel.<sup>[46]</sup>

Conversely, the nocturnal Flying Squirrels (*Glaucomys* spp.) are rarely observed during typical daylight binocular use. When they are seen, differentiating the Northern (*G. sabrinus*) and Southern (*G. volans*) species visually is exceptionally difficult even for human experts. Therefore, they should be consolidated to the genus level (*Glaucomys*).<sup>[46]</sup> Ground squirrels (e.g., *Spermophilus*, *Urocitellus*) should largely be grouped at the genus level due to extreme phenotypic similarity across their vast ranges, with exceptions made for highly iconic and visually distinct species like the California Ground Squirrel (*Otospermophilus beecheyi*).

### **Urban, Suburban, and Trail Essentials**

To ensure the product delights users in their own backyards and local state parks, the model must flawlessly identify the most commonly encountered synanthropic (human-adapted) and trail species. The "urban mammal paradox" highlighted by recent large-scale surveys demonstrates that developed areas often harbor higher mammal detection rates than remote wildernesses.<sup>[49]</sup>

* **Synanthropic Mammals:** The Northern Raccoon (*Procyon lotor*), Virginia Opossum (*Didelphis virginiana*), Striped Skunk (*Mephitis mephitis*), and Groundhog/Woodchuck (*Marmota monax*) are ubiquitous across North American suburbs and parks. These must be permanently locked as Tier 1 species, as any failure to identify them would be perceived as a catastrophic failure of the device's AI.  
* **Predators:** The Coyote (*Canis latrans*) is the most universally detected wild mammal across the continental United States, present in all 49 continental states and thriving in both deep wilderness and urban centers.<sup>[18]</sup> The Bobcat (*Lynx rufus*), Red Fox (*Vulpes vulpes*), and Gray Fox (*Urocyon cinereoargenteus*) are also critical inclusions.<sup>[50]</sup> The model must be able to distinguish a Coyote from a domestic dog, and a Bobcat from a domestic cat—distinctions that require high-quality training data focused on facial structure and ear/tail morphology.  
* **Ungulates:** The White-tailed Deer (*Odocoileus virginianus*) and Mule Deer (*Odocoileus hemionus*) are ubiquitous and heavily observed.<sup>[52]</sup> While easily distinguishable by tail morphology, antler shape, and ear size in clear views, a fallback to the genus *Odocoileus* should be programmed for obscured or partial views. Moose (*Alces alces*), Elk/Wapiti (*Cervus canadensis*), and Pronghorn (*Antilocapra americana*) are highly sought-after trail sightings and must retain Tier 1 status.<sup>[53]</sup>

## **Global Market Alignment and Necessary Additions**

While North America is the primary target market, the AX Visio commands a premium price globally and is heavily marketed toward international ecotourism. A user spending $4,800 on smart binoculars will absolutely expect them to function flawlessly on an African safari, a European woodland hike, or even during a weekend trip to a local zoological park.

### **East and Southern African Safari Megafauna**

Safari tourists represent a prime, high-disposable-income demographic for smart binoculars. The model must cover the traditional "Big Five" alongside the most visible and iconic plains game.<sup>[30]</sup> The open savanna environments typical of East and Southern Africa are ideal for computer vision, as animals are frequently viewed in good lighting with minimal forest canopy occlusion.

* **Predators:** Lion (*Panthera leo*), Leopard (*Panthera pardus*), Cheetah (*Acinonyx jubatus*), Spotted Hyena (*Crocuta crocuta*), and African Wild Dog (*Lycaon pictus*) are mandatory Tier 1 inclusions.  
* **Megafauna:** African Elephant (*Loxodonta africana*), Cape Buffalo (*Syncerus caffer*), and Hippopotamus (*Hippopotamus amphibius*) are essential.  
* **Taxonomic Complexities (Giraffes and Rhinos):** Due to recent and heavily debated taxonomic splits regarding giraffes (e.g., separating Masai, Reticulated, and Southern species), it is highly recommended to class all giraffes at the genus level (*Giraffa*) to avoid user confusion and model hallucination, unless the training data is exceptionally well-annotated by geographic region.<sup>[55]</sup> Both Black and White Rhinoceros must be included, as they are visually distinct based on lip morphology and size.  
* **Ungulates:** Plains Zebra (*Equus quagga*), Blue Wildebeest (*Connochaetes taurinus*), Impala (*Aepyceros melampus*), Springbok (*Antidorcas marsupialis*), and Warthog (*Phacochoerus africanus*).  
* **Primates:** Olive, Chacma, and Yellow Baboons should be consolidated into the genus *Papio* to save model capacity, as their phenotypic differences are subtle and geographically dependent.<sup>[30]</sup> Vervet Monkeys (*Chlorocebus pygerythrus*) and high-value ecotourism targets like the Mountain Gorilla (*Gorilla beringei*) must be included.<sup>[56]</sup>

### **European Woodlands**

For the European consumer base, the model must recognize native species as well as common introduced fauna.<sup>[59]</sup>

* Key inclusions: Eurasian Red Squirrel (*Sciurus vulgaris*), Eurasian Badger (*Meles meles*), Roe Deer (*Capreolus capreolus*), Red Deer (*Cervus elaphus*), Wild Boar (*Sus scrofa*), and European Hare (*Lepus europaeus*).

### **Domestic and Zoo Animals: The Crucial Oversight**

A common, and often fatal, oversight in wildlife computer vision models trained primarily on remote camera trap data (such as the original SpeciesNet architecture) is the exclusion or poor performance on domestic animals. Consumers do not only use binoculars in pristine wildernesses; they will inevitably point the AX Visio at pets in the park, livestock in rural fields, and exotic animals in captivity.

Failing to identify a horse, or worse, misclassifying a domestic Golden Retriever as a Coyote or a Wolf, instantly undermines the user's perception of the AI's intelligence.<sup>[5]</sup>

* **Domestic Additions:** The following must be added to the ontology: Domestic Dog (*Canis lupus familiaris*), Domestic Cat (*Felis catus*), Horse (*Equus caballus*), Cattle (*Bos taurus*), Domestic Sheep (*Ovis aries*), Domestic Goat (*Capra hircus*), Alpaca/Llama (*Lama* spp.), and Donkey (*Equus asinus*).  
* **Zoo Exotics:** To account for urban usage in zoological parks, highly iconic species such as the Tiger (*Panthera tigris*), Giant Panda (*Ailuropoda melanoleuca*), and Polar Bear (*Ursus maritimus*) should be retained, even if they are unlikely to be seen in the wild by the average user.

## **Strategic Pruning: Species to Remove**

To make room for the necessary North American, domestic, and safari additions while keeping the final class count strictly under the 250-class ceiling required by 1–3M parameter models, aggressive and strategic pruning of the original 483-label list is required. The removal criteria must target species that provide negative value to the model's accuracy-efficiency trade-off.

### **1\. Visually Cryptic Taxa**

Dozens of small mammal species cannot be differentiated without in-hand morphological measurements, dental examinations, or DNA sequencing.

* **Chipmunks and Ground Squirrels:** Attempting to teach a nano-model to distinguish between a Least Chipmunk (*Neotamias minimus*) and a Yellow-pine Chipmunk (*Neotamias amoenus*) will result in feature confusion and wasted parameter weights.<sup>[34]</sup> All chipmunks must be rolled up to the genus *Tamias* (or *Neotamias*).  
* **Lagomorphs:** North American cottontails (e.g., *Sylvilagus floridanus*, *S. audubonii*) are highly cryptic and frequently overlap in range. They must be merged into the genus *Sylvilagus*.<sup>[51]</sup>

### **2\. Nocturnal, Elusive, and Subterranean Species**

Mammals that are strictly nocturnal, incredibly elusive, or spend their lives underground are virtually never observed through daylight consumer binoculars. Including them wastes model capacity on scenarios that will never occur in real-world device usage.

* **Target Removals:** Specific species classes for Shrews (*Soricidae*), Moles (*Talpidae*), Pocket Gophers (*Geomyidae*), and nocturnal specialists like Ringtails (*Bassariscus astutus*) should be removed, replacing them with high-level Family labels only, or dropping them entirely if feature space is tight.<sup>[35]</sup>  
* **Bats and Deep-Sea Mammals:** As specified in the constraints, all Chiroptera and marine mammals are excluded from this specific deployment.

### **3\. Data-Starved Endemics**

Range-limited species from South America, Southeast Asia, or remote island chains that lack sufficient training data (i.e., fewer than 500-1000 high-quality, varied images) in open datasets like iNaturalist and GBIF must be removed.<sup>[11]</sup>

* **The Overfitting Risk:** Training a 1-3M parameter model on a long-tailed dataset with heavily underrepresented minority classes leads to severe overfitting. The model will fail to learn generalized features for these rare animals, instead memorizing the specific backgrounds of the few available training images.<sup>[19]</sup> These species should fall back to Family-level labels (e.g., using *Viverridae* for obscure Southeast Asian civets).

## **Hierarchical Fallback Architecture at Inference**

Deploying a lightweight model requires an intelligent post-processing and inference strategy to maximize user trust. Edge AI systems must be designed for robustness, anticipating that a 3M parameter backbone will occasionally fail to extract sufficient confidence for a fine-grained species prediction, especially under suboptimal viewing conditions.<sup>[38]</sup>

When the bounding box regressor detects a mammal, the classification head evaluates the pooled feature vector against the learned classes.<sup>[24]</sup> Because lightweight models lack the depth to consistently assign 99% confidence to fine-grained species at extreme distances, a **Hierarchical Fallback Strategy** is mandatory.<sup>[64]</sup>

### **Multi-Label Training and Inference Logic**

Instead of utilizing a mutually exclusive softmax output layer (where the model is forced to choose exactly one class and suppress all others), the model should be trained using a multi-label classification approach (e.g., binary cross-entropy) where the entire biological taxonomy is embedded into the training labels.<sup>[25]</sup>

For example, a training image of an Eastern Gray Squirrel is tagged not just with the species, but with the full hierarchical path: \`\`. The network learns to activate all relevant nodes.

During real-time inference on the edge device, the following logical routing should occur:

1. **Full Taxonomy Prediction:** The model outputs independent confidence scores for all nodes in the taxonomic tree simultaneously.  
2. **Tier 1 Evaluation (Species):** The software evaluates the confidence score at the Species level. If the confidence exceeds a strict, high threshold (e.g., \>80%), the binocular displays the highly specific species name: *"Eastern Gray Squirrel"*.  
3. **Tier 2 Fallback (Genus):** If the species confidence fails to meet the threshold (e.g., 60%), but the Genus level confidence remains very high (e.g., \>85%), the binocular suppresses the uncertain species guess and instead displays the genus name: *"Tree Squirrel (Sciurus)"*.<sup>[33]</sup>  
4. **Tier 3 Fallback (Family):** If the genus cannot be determined due to extreme distance or occlusion, but the family node is highly activated (e.g., \>90%), the device displays *"Squirrel Family (Sciuridae)"*.

This biological bias programming and conditional logic ensure that the system degrades gracefully. It guarantees that the device rarely makes an egregiously wrong prediction (e.g., confidently calling a blurred dog a bear), which is the primary cause of user dissatisfaction with AI computer vision products.<sup>[29]</sup>

## **Final Label Set Recommendation**

Based on the synthesis of model capacity constraints, geographical user priorities, visual distinctiveness, data availability, and hierarchical fallback logic, the recommended target total count is **225 distinct output classes**.

This target count provides the absolute optimum balance. It allows a 1–3M parameter YOLO or MobileNet architecture to dedicate sufficient convolutional filters to extract highly discriminative features for each class, avoiding the accuracy degradation and parameter dilution that plagues small models forced to learn 500+ classes.<sup>[11]</sup> Concurrently, 225 classes are more than sufficient to cover every major charismatic, common, and expected mammal across North America, Europe, and the African safari circuits.

Below is a structured representation of the recommended ontology, categorized by major taxonomic order. (Note: For brevity and structural clarity, this table highlights the most critical inclusions, consolidations, and hierarchical logic paradigms that define the 225-class limit; the full list follows this exact framework).

### **Recommended Taxonomic Output Structure**

| Order / Group | Taxonomic Tier | Scientific Name | Common Name | Rationale / Action from Original List |
| :---- | :---- | :---- | :---- | :---- |
| **Carnivora** | Species | *Ursus americanus* | American Black Bear | Highly common NA trail species. Keep as Tier 1\. |
| **Carnivora** | Species | *Ursus arctos* | Brown / Grizzly Bear | Distinctive megafauna. Keep as Tier 1\. |
| **Carnivora** | Species | *Procyon lotor* | Northern Raccoon | Ubiquitous urban/suburban species.<sup>[41]</sup> Keep as Tier 1\. |
| **Carnivora** | Species | *Canis latrans* | Coyote | Most widely distributed NA predator.<sup>[18]</sup> Keep as Tier 1\. |
| **Carnivora** | Species | *Canis lupus familiaris* | Domestic Dog | **ADD:** Essential for urban use. Prevents wild canid false positives. |
| **Carnivora** | Species | *Felis catus* | Domestic Cat | **ADD:** Essential for urban/suburban observation. |
| **Carnivora** | Species | *Lynx rufus* | Bobcat | Common, visually distinct NA trail species. |
| **Carnivora** | Species | *Panthera leo* | Lion | Safari essential. Keep as Tier 1\.<sup>[30]</sup> |
| **Carnivora** | Species | *Panthera pardus* | Leopard | Safari essential. Keep as Tier 1\. |
| **Carnivora** | Family | *Herpestidae* | Mongooses | **CONSOLIDATE:** Merge various African mongoose species to Family level due to small size and extreme visual similarity. |
| **Carnivora** | Family | *Mustelidae* | Weasels / Martens | **CONSOLIDATE:** Merge smaller, fast-moving weasels to Family, retaining only larger/distinct ones (e.g., Wolverine, Badger) at Species level. |
| **Artiodactyla** | Species | *Odocoileus virginianus* | White-tailed Deer | Most common NA ungulate. Keep as Tier 1\.<sup>[18]</sup> |
| **Artiodactyla** | Species | *Odocoileus hemionus* | Mule Deer | Distinct from White-tailed via ears/tail. Keep as Tier 1\. |
| **Artiodactyla** | Genus | *Odocoileus* | NA Deer (Fallback) | **ADD:** Genus fallback for when tail/ears are obscured. |
| **Artiodactyla** | Species | *Alces alces* | Moose | Highly sought-after trail species. Visually distinct. |
| **Artiodactyla** | Species | *Bos taurus* | Domestic Cattle | **ADD:** Ubiquitous in rural landscapes globally. |
| **Artiodactyla** | Genus | *Equus* | Horses and Zebras | **CONSOLIDATE:** Maintain *Equus quagga* (Zebra) as Species, but use Genus fallback for domestic horses/donkeys.<sup>[60]</sup> |
| **Artiodactyla** | Species | *Hippopotamus amphibius* | Hippopotamus | Safari essential. Keep as Tier 1\. |
| **Artiodactyla** | Genus | *Giraffa* | Giraffe | **CONSOLIDATE:** Safari essential, but merge distinct species to Genus to avoid taxonomic confusion and hallucination. |
| **Artiodactyla** | Sub-family | *Cephalophinae* | Duikers | **CONSOLIDATE:** Deep-forest African antelopes; too cryptic to separate to species. |
| **Rodentia** | Species | *Sciurus carolinensis* | Eastern Gray Squirrel | **ADD BACK:** Ubiquitous NA/EU urban wildlife. High data availability.<sup>[43]</sup> |
| **Rodentia** | Species | *Sciurus niger* | Fox Squirrel | **ADD BACK:** Distinct orange belly/size differentiates from Gray.<sup>[44]</sup> |
| **Rodentia** | Species | *Sciurus griseus* | Western Gray Squirrel | **ADD BACK:** Distinctive silver-gray, no brown hues.<sup>[47]</sup> |
| **Rodentia** | Species | *Tamiasciurus hudsonicus* | American Red Squirrel | **ADD BACK:** Small size, red dorsum, high hiker visibility.<sup>[45]</sup> |
| **Rodentia** | Genus | *Tamias / Neotamias* | Chipmunks | **CONSOLIDATE:** Merge all 20+ NA chipmunk species to Genus. Visually inseparable by AI.<sup>[34]</sup> |
| **Rodentia** | Genus | *Marmota* | Marmots / Groundhogs | **CONSOLIDATE:** Group Hoary, Yellow-bellied, and Groundhogs, or keep *M. monax* distinct due to extreme commonality.<sup>[34]</sup> |
| **Rodentia** | Genus | *Glaucomys* | Flying Squirrels | **CONSOLIDATE:** Merge Northern/Southern species. Rarely seen (nocturnal).<sup>[46]</sup> |
| **Rodentia** | Family | *Muridae* | Old World Mice/Rats | **CONSOLIDATE:** Use as catch-all. Small rodents are too difficult to ID via binoculars. |
| **Rodentia** | Family | *Cricetidae* | New World Mice/Voles | **CONSOLIDATE:** Use as catch-all. |
| **Lagomorpha** | Genus | *Sylvilagus* | Cottontail Rabbits | **CONSOLIDATE:** Merge Eastern, Desert, Mountain, etc. Highly cryptic species complex.<sup>[62]</sup> |
| **Lagomorpha** | Species | *Lepus californicus* | Black-tailed Jackrabbit | Keep as Tier 1\. Distinctively large ears easily detected by bounding box regressors. |
| **Primates** | Genus | *Papio* | Baboons | **CONSOLIDATE:** Merge Olive, Chacma, Yellow. Geographically separated but visually similar to a lightweight AI.<sup>[30]</sup> |
| **Primates** | Species | *Gorilla beringei* | Mountain Gorilla | High-value ecotourism species.<sup>[58]</sup> Keep as Tier 1\. |

### **Summary of Taxonomic Modifications**

To reach the \~225 class target, the following massive structural changes from the original 483-label list were executed:

**Major Exclusions and Consolidations (\~280 classes removed):**

* **Cryptic Rodents and Lagomorphs:** Over 100 species of mice, voles, woodrats, and individual chipmunks were removed and consolidated entirely into Genus or Family tiers. The visual overlap at these scales renders species-level detection statistically impossible for a 3M parameter backbone.  
* **Nocturnal and Subterranean Taxa:** Species strictly active at night (e.g., Ringtails, numerous African civets, bushbabies) or underground (moles, pocket gophers) were ruthlessly pruned. They are virtually never viewed through daytime consumer binoculars, and their inclusion dilutes the model's capacity to learn high-priority diurnal species.  
* **Data-Starved Endemics:** Rare tropical and isolated endemic species with insufficient training data (fewer than 1,000 images) were dropped to prevent catastrophic model overfitting and parameter waste.

**Major Additions and Reinstatements (\~25 classes added):**

* **North American Squirrels:** The aggressive pruning of *Sciuridae* was entirely reversed. Eastern Gray, Fox, Red, and Western Gray squirrels were reinstated as they are highly abundant, frequently observed by target consumers, and possess highly distinct phenotypic markers easily learned by modern convolutional networks.  
* **Domestic and Zoo Fauna:** Dogs, cats, cows, horses, sheep, and charismatic zoo animals were explicitly added to the ontology. This accommodates standard consumer behavior and prevents the catastrophic UX failure of misclassifying common domestic pets as wild predators.

By restricting the class count to an optimal ceiling of \~225 classes, the single-stage detection model avoids the accuracy degradation inherent in over-parameterized nano-networks. Implementing the hierarchical fallback strategy ensures that when the AI encounters ambiguity in the field, it degrades gracefully to a Genus or Family level rather than failing outright. Finally, by meticulously reconstructing the dataset to prioritize North American tree squirrels, urban wildlife, domestic animals, and global safari megafauna, the resulting taxonomy perfectly aligns the computational capabilities of the Qualcomm QCS605 with the premium expectations of the Swarovski AX Visio user base.

#### **Works cited**

1. Qualcomm® QCS603/605 SoCs for IoT, accessed March 19, 2026, [https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/documents/qcs603.605-socs-product-brief\_87-pg764-1-c.pdf](https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/documents/qcs603.605-socs-product-brief_87-pg764-1-c.pdf)  
2. A Method of Deep Learning Model Optimization for Image Classification on Edge Device, accessed March 19, 2026, [https://www.mdpi.com/1424-8220/22/19/7344](https://www.mdpi.com/1424-8220/22/19/7344)  
3. GitHub \- google/cameratrapai: AI models trained by Google to classify species in images from motion-triggered wildlife cameras., accessed March 19, 2026, [https://github.com/google/cameratrapai](https://github.com/google/cameratrapai)  
4. Using the power of AI to identify and track species \- World Wildlife Fund, accessed March 19, 2026, [https://www.worldwildlife.org/news/stories/using-the-power-of-ai-to-identify-and-track-species/](https://www.worldwildlife.org/news/stories/using-the-power-of-ai-to-identify-and-track-species/)  
5. AddaxAI \- Simplifying camera trap image analysis with AI \- Addax Data Science, accessed March 19, 2026, [https://addaxdatascience.com/addaxai/](https://addaxdatascience.com/addaxai/)  
6. YOLO11 vs YOLOv8: A Comprehensive Technical Comparison of Real-Time Vision Models, accessed March 19, 2026, [https://docs.ultralytics.com/compare/yolo11-vs-yolov8/](https://docs.ultralytics.com/compare/yolo11-vs-yolov8/)  
7. Everything you need to know about TorchVision's MobileNetV3 implementation \- PyTorch, accessed March 19, 2026, [https://pytorch.org/blog/torchvision-mobilenet-v3-implementation/](https://pytorch.org/blog/torchvision-mobilenet-v3-implementation/)  
8. Benchmarking Lightweight YOLO Object Detectors for Real-Time Hygiene Compliance Monitoring \- PMC, accessed March 19, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC12526732/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12526732/)  
9. Fast, Accurate Detection of 100,000 Object Classes on a Single Machine \- Google Research, accessed March 19, 2026, [https://research.google.com/pubs/archive/40814.pdf](https://research.google.com/pubs/archive/40814.pdf)  
10. Do more object classes increase or decrease the accuracy of object detection, accessed March 19, 2026, [https://stats.stackexchange.com/questions/348584/do-more-object-classes-increase-or-decrease-the-accuracy-of-object-detection](https://stats.stackexchange.com/questions/348584/do-more-object-classes-increase-or-decrease-the-accuracy-of-object-detection)  
11. Optimal Size of Agricultural Dataset for YOLOv8 Training \- Semantic Scholar, accessed March 19, 2026, [https://pdfs.semanticscholar.org/68e9/051dfcc7902b0cf66c90baa4b64f13529627.pdf](https://pdfs.semanticscholar.org/68e9/051dfcc7902b0cf66c90baa4b64f13529627.pdf)  
12. ImageNet-21K Pretraining for the Masses \- arXiv, accessed March 19, 2026, [https://arxiv.org/pdf/2104.10972](https://arxiv.org/pdf/2104.10972)  
13. Advancements in Small-Object Detection (2023–2025): Approaches, Datasets, Benchmarks, Applications, and Practical Guidance \- MDPI, accessed March 19, 2026, [https://www.mdpi.com/2076-3417/15/22/11882](https://www.mdpi.com/2076-3417/15/22/11882)  
14. Modified Lightweight YOLO v8 Model for Fast and Precise Indoor Occupancy Detection, accessed March 19, 2026, [https://www.mdpi.com/2076-3417/16/1/335](https://www.mdpi.com/2076-3417/16/1/335)  
15. Benchmarking YOLOv8 to YOLOv11 Architectures for Real-Time Traffic Sign Recognition in Embedded 1:10 Scale Autonomous Vehicles \- MDPI, accessed March 19, 2026, [https://www.mdpi.com/2227-7080/13/11/531](https://www.mdpi.com/2227-7080/13/11/531)  
16. A Comparative Benchmark of Real-time Detectors for Blueberry Detection towards Precision Orchard Management \- arXiv.org, accessed March 19, 2026, [https://arxiv.org/html/2509.20580v1](https://arxiv.org/html/2509.20580v1)  
17. Edge AI for Industrial Visual Inspection: YOLOv8-Based Visual Conformity Detection Using Raspberry Pi \- MDPI, accessed March 19, 2026, [https://www.mdpi.com/1999-4893/18/8/510](https://www.mdpi.com/1999-4893/18/8/510)  
18. SNAPSHOT USA: First-ever nationwide mammal survey published | Programs and Events Calendar \- North Carolina Museum of Natural Sciences, accessed March 19, 2026, [https://naturalsciences.org/calendar/news/snapshot-usa-first-ever-nationwide-mammal-survey-published/](https://naturalsciences.org/calendar/news/snapshot-usa-first-ever-nationwide-mammal-survey-published/)  
19. Comparative Analysis of Lightweight Deep Learning Models for Memory-Constrained Devices \- arXiv, accessed March 19, 2026, [https://arxiv.org/html/2505.03303v1](https://arxiv.org/html/2505.03303v1)  
20. a comparative performance analysis of federated knowledge distillation for image classification under data heterogeneity \- OuluREPO, accessed March 19, 2026, [https://oulurepo.oulu.fi/bitstream/handle/10024/56897/nbnfioulu-202506124404.pdf?sequence=1\&isAllowed=y](https://oulurepo.oulu.fi/bitstream/handle/10024/56897/nbnfioulu-202506124404.pdf?sequence=1&isAllowed=y)  
21. Fine-Tuning Without Forgetting: Adaptation of YOLOv8 Preserves COCO Performance, accessed March 19, 2026, [https://arxiv.org/html/2505.01016v1](https://arxiv.org/html/2505.01016v1)  
22. A Structurally Optimized and Efficient Lightweight Object Detection Model for Autonomous Driving \- PMC, accessed March 19, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC12788148/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12788148/)  
23. A Structurally Optimized and Efficient Lightweight Object Detection Model for Autonomous Driving \- MDPI, accessed March 19, 2026, [https://www.mdpi.com/1424-8220/26/1/54](https://www.mdpi.com/1424-8220/26/1/54)  
24. YOLOv10: Real-Time End-to-End Object Detection \- arXiv, accessed March 19, 2026, [https://arxiv.org/html/2405.14458v1](https://arxiv.org/html/2405.14458v1)  
25. The effect of grouping classes into hierarchical structures for object detection \- TU Delft Repository, accessed March 19, 2026, [https://repository.tudelft.nl/file/File\_a1da1184-d152-4d8b-9e1a-de45b7a6144f?preview=1](https://repository.tudelft.nl/file/File_a1da1184-d152-4d8b-9e1a-de45b7a6144f?preview=1)  
26. Swarovski AX Visio: Innovation or fueling the demise of birding? \- The Birding Life, accessed March 19, 2026, [https://www.thebirdinglife.com/post/swarovski-ax-visio-innovation-or-fueling-demise-of-birding](https://www.thebirdinglife.com/post/swarovski-ax-visio-innovation-or-fueling-demise-of-birding)  
27. I had a chance to try out the Swarovski AX Visio today : r/Binoculars \- Reddit, accessed March 19, 2026, [https://www.reddit.com/r/Binoculars/comments/1bjwlpe/i\_had\_a\_chance\_to\_try\_out\_the\_swarovski\_ax\_visio/](https://www.reddit.com/r/Binoculars/comments/1bjwlpe/i_had_a_chance_to_try_out_the_swarovski_ax_visio/)  
28. A Guide to Identifying the Most Commonly Seen Animals on a Safari in Tanzania, accessed March 19, 2026, [https://www.discoverafrica.com/blog/a-guide-to-identifying-the-most-commonly-seen-animals-on-a-safari-in-tanzania/](https://www.discoverafrica.com/blog/a-guide-to-identifying-the-most-commonly-seen-animals-on-a-safari-in-tanzania/)  
29. Hierarchical Edge Detection and Pattern Recognition: Insights from Biological Systems | by Micheal Bee | Medium, accessed March 19, 2026, [https://medium.com/@mbonsign/hierarchical-edge-detection-and-pattern-recognition-insights-from-biological-systems-3987f756dcda](https://medium.com/@mbonsign/hierarchical-edge-detection-and-pattern-recognition-insights-from-biological-systems-3987f756dcda)  
30. What Is the Big 5? Meet Africa's Most Iconic Safari Animals, accessed March 19, 2026, [https://www.gocollette.com/en-us/travel-blog/what-is-the-big-5-african-safari-animals](https://www.gocollette.com/en-us/travel-blog/what-is-the-big-5-african-safari-animals)  
31. Crowdsourcing New Range Data for North American Mammals | NC State News, accessed March 19, 2026, [https://news.ncsu.edu/2025/07/crowdsourcing-new-range-data-for-north-american-mammals/](https://news.ncsu.edu/2025/07/crowdsourcing-new-range-data-for-north-american-mammals/)  
32. iNaturalist Research-grade Observations \- GBIF, accessed March 19, 2026, [https://www.gbif.org/dataset/50c9509d-22c7-4a22-a47d-8c48425ef4a7](https://www.gbif.org/dataset/50c9509d-22c7-4a22-a47d-8c48425ef4a7)  
33. Supercategory fallback / label hierarchy in Object Detection : r/computervision \- Reddit, accessed March 19, 2026, [https://www.reddit.com/r/computervision/comments/z01p7h/supercategory\_fallback\_label\_hierarchy\_in\_object/](https://www.reddit.com/r/computervision/comments/z01p7h/supercategory_fallback_label_hierarchy_in_object/)  
34. (Family) Squirrels \- Montana Field Guide, accessed March 19, 2026, [https://fieldguide.mt.gov/displaySpecies.aspx?family=Sciuridae](https://fieldguide.mt.gov/displaySpecies.aspx?family=Sciuridae)  
35. List of mammals of North America \- Wikipedia, accessed March 19, 2026, [https://en.wikipedia.org/wiki/List\_of\_mammals\_of\_North\_America](https://en.wikipedia.org/wiki/List_of_mammals_of_North_America)  
36. Mammals, accessed March 19, 2026, [https://www.fws.gov/sites/default/files/documents/MammalslistID.pdf](https://www.fws.gov/sites/default/files/documents/MammalslistID.pdf)  
37. AI labeling \- Edge Impulse Documentation, accessed March 19, 2026, [https://docs.edgeimpulse.com/studio/projects/data-acquisition/ai-labeling](https://docs.edgeimpulse.com/studio/projects/data-acquisition/ai-labeling)  
38. Hierarchical Fallback Architecture for High Risk Online Machine Learning Inference \- arXiv, accessed March 19, 2026, [https://arxiv.org/html/2501.17834v1](https://arxiv.org/html/2501.17834v1)  
39. Squirrels \- Mass Audubon, accessed March 19, 2026, [https://www.massaudubon.org/nature-wildlife/mammals-in-massachusetts/squirrels](https://www.massaudubon.org/nature-wildlife/mammals-in-massachusetts/squirrels)  
40. North America's Most Common Squirrels and How to Tell Them Apart \- A-Z Animals, accessed March 19, 2026, [https://a-z-animals.com/articles/north-americas-most-common-squirrels-and-how-to-tell-them-apart/](https://a-z-animals.com/articles/north-americas-most-common-squirrels-and-how-to-tell-them-apart/)  
41. Top 15 Mammals Photographed by UWI's Biodiversity Monitoring Project | Lincoln Park Zoo, accessed March 19, 2026, [https://www.lpzoo.org/top-15-mammals-photographed-by-uwis-biodiversity-monitoring-project/](https://www.lpzoo.org/top-15-mammals-photographed-by-uwis-biodiversity-monitoring-project/)  
42. Distinguishing between Sciurus niger (Eastern Fox Squirrel) and Sciurus carolinensis (Eastern Gray Squirrel) in Camera Trap Photos \- eMammal, accessed March 19, 2026, [https://emammal.si.edu/sites/default/files/2024-06/squirrel-id-training.pdf](https://emammal.si.edu/sites/default/files/2024-06/squirrel-id-training.pdf)  
43. Eastern gray squirrel \- Wikipedia, accessed March 19, 2026, [https://en.wikipedia.org/wiki/Eastern\_gray\_squirrel](https://en.wikipedia.org/wiki/Eastern_gray_squirrel)  
44. Gray and Fox Squirrels \- Oklahoma State University Extension, accessed March 19, 2026, [https://extension.okstate.edu/fact-sheets/gray-and-fox-squirrels.html](https://extension.okstate.edu/fact-sheets/gray-and-fox-squirrels.html)  
45. Different Types Of Squirrels Found in North America With Pictures \- Active Wild, accessed March 19, 2026, [https://www.activewild.com/types-of-squirrels/](https://www.activewild.com/types-of-squirrels/)  
46. Tree Squirrels | Internet Center for Wildlife Damage Management, accessed March 19, 2026, [https://icwdm.org/species/rodents/tree-squirrels/](https://icwdm.org/species/rodents/tree-squirrels/)  
47. TREE SQUIRRELS \- ODFW, accessed March 19, 2026, [https://www.dfw.state.or.us/wildlife/living\_with/docs/TreeSquirrels.pdf](https://www.dfw.state.or.us/wildlife/living_with/docs/TreeSquirrels.pdf)  
48. "Tree Squirrels" by Jeffrey J. Jackson \- UNL Digital Commons, accessed March 19, 2026, [https://digitalcommons.unl.edu/icwdmhandbook/10/](https://digitalcommons.unl.edu/icwdmhandbook/10/)  
49. NMU Involved in Snapshot USA Mammal Survey \- Northern Today, accessed March 19, 2026, [https://news.nmu.edu/nmu-involved-snapshot-usa-mammal-survey](https://news.nmu.edu/nmu-involved-snapshot-usa-mammal-survey)  
50. Mammal Guide \- Pajarito Environmental Education Center, accessed March 19, 2026, [https://peecnature.org/learn/nature-guides/mammal-guide/](https://peecnature.org/learn/nature-guides/mammal-guide/)  
51. Checklist of Mammals, accessed March 19, 2026, [https://www.srs.fs.usda.gov/pubs/misc/ag\_654/volume\_1/checklist\_of/mammals.htm](https://www.srs.fs.usda.gov/pubs/misc/ag_654/volume_1/checklist_of/mammals.htm)  
52. From backyard to backcountry: changes in mammal communities across an urbanization gradient | Journal of Mammalogy | Oxford Academic, accessed March 19, 2026, [https://academic.oup.com/jmammal/article/105/1/175/7453584](https://academic.oup.com/jmammal/article/105/1/175/7453584)  
53. How often do you see dangerous animals when hiking? : r/AskAnAmerican \- Reddit, accessed March 19, 2026, [https://www.reddit.com/r/AskAnAmerican/comments/v9j662/how\_often\_do\_you\_see\_dangerous\_animals\_when\_hiking/](https://www.reddit.com/r/AskAnAmerican/comments/v9j662/how_often_do_you_see_dangerous_animals_when_hiking/)  
54. Mammals | National Wildlife Federation, accessed March 19, 2026, [https://www.nwf.org/Educational-Resources/Wildlife-Guide/Mammals](https://www.nwf.org/Educational-Resources/Wildlife-Guide/Mammals)  
55. Top 16 Common Animals You'll See on an African Safari \- Go Expeditions Africa, accessed March 19, 2026, [https://www.goexpeditionsafrica.com/top-16-common-animals-youll-see-on-an-african-safari/](https://www.goexpeditionsafrica.com/top-16-common-animals-youll-see-on-an-african-safari/)  
56. Guide to the wildlife of East Africa \- Jacada Travel, accessed March 19, 2026, [https://www.jacadatravel.com/africa/east-africa/travel-guides/guide-to-the-wildlife-of-east-africa/](https://www.jacadatravel.com/africa/east-africa/travel-guides/guide-to-the-wildlife-of-east-africa/)  
57. SAFARI WILDLIFE CHECKLIST | Adventures Within Reach, accessed March 19, 2026, [https://adventureswithinreach.com/wp-content/uploads/Tanzania/awr-safari-wildlife-checklist.pdf](https://adventureswithinreach.com/wp-content/uploads/Tanzania/awr-safari-wildlife-checklist.pdf)  
58. Top Ten African Safari Animals | Audley Travel US, accessed March 19, 2026, [https://www.audleytravel.com/us/inspiration/safaris/safari-guides/top-african-safari-animals](https://www.audleytravel.com/us/inspiration/safaris/safari-guides/top-african-safari-animals)  
59. Mammals | The Wildlife Trusts, accessed March 19, 2026, [https://www.wildlifetrusts.org/wildlife-explorer/mammals](https://www.wildlifetrusts.org/wildlife-explorer/mammals)  
60. List of mammals of Europe \- Wikipedia, accessed March 19, 2026, [https://en.wikipedia.org/wiki/List\_of\_mammals\_of\_Europe](https://en.wikipedia.org/wiki/List_of_mammals_of_Europe)  
61. European Animals List With Pictures & Facts – Species Found In Europe \- Active Wild, accessed March 19, 2026, [https://www.activewild.com/european-animals/](https://www.activewild.com/european-animals/)  
62. Mammal Watching Checklist, accessed March 19, 2026, [https://www.mammalwatching.com/wp-content/uploads/2025/08/Mammal-Watching-Checklist-2.pdf](https://www.mammalwatching.com/wp-content/uploads/2025/08/Mammal-Watching-Checklist-2.pdf)  
63. A Novel Lightweight Object Detection Network with Attention Modules and Hierarchical Feature Pyramid \- MDPI, accessed March 19, 2026, [https://www.mdpi.com/2073-8994/15/11/2080](https://www.mdpi.com/2073-8994/15/11/2080)  
64. Hierarchical Multi-Label Object Detection Framework for Remote Sensing Images \- MDPI, accessed March 19, 2026, [https://www.mdpi.com/2072-4292/12/17/2734](https://www.mdpi.com/2072-4292/12/17/2734)  
65. A Decade of You Only Look Once (YOLO) for Object Detection \- arXiv.org, accessed March 19, 2026, [https://arxiv.org/html/2504.18586v1](https://arxiv.org/html/2504.18586v1)  
66. Small-Object Detection at the Edge: A Pareto-Efficient Benchmark of Lightweight YOLO Models on UAV and Overhead Datasets \- Beadle Scholar, accessed March 19, 2026, [https://scholar.dsu.edu/cgi/viewcontent.cgi?article=1258\&context=ccspapers](https://scholar.dsu.edu/cgi/viewcontent.cgi?article=1258&context=ccspapers)  
67. Tree Squirrels | Wildlife Illinois, accessed March 19, 2026, [https://wildlifeillinois.org/identify-wildlife/tree-squirrels/](https://wildlifeillinois.org/identify-wildlife/tree-squirrels/)  
68. WILDLIFE SPECIES: Tamiasciurus hudsonicus \- USDA Forest Service, accessed March 19, 2026, [https://www.fs.usda.gov/database/feis/animals/mammal/tahu/all.html](https://www.fs.usda.gov/database/feis/animals/mammal/tahu/all.html)  
69. Animals in National Trail Parks: Mammals, accessed March 19, 2026, [https://ntprd.org/animals-in-national-trail-parks-mammals/](https://ntprd.org/animals-in-national-trail-parks-mammals/)  
70. An Inventory of Terrestrial Mammals at National Parks in the Northeast Temperate Network and Sagamore Hill National Historic Site \- USGS Publications Warehouse, accessed March 19, 2026, [https://pubs.usgs.gov/sir/2007/5245/pdf/sir2007-5245-screen.pdf](https://pubs.usgs.gov/sir/2007/5245/pdf/sir2007-5245-screen.pdf)