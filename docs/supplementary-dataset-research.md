# **Commercial Viability and Integration Strategy for Open Wildlife Image Datasets in Embedded Mammal Detection Models**

## **1\. Architectural Context and Deployment Constraints**

The deployment of real-time computer vision models on embedded consumer hardware represents a complex intersection of machine learning efficiency, ecological data aggregation, and intellectual property law. The target hardware for this deployment—the Qualcomm QCS605 embedded within the Swarovski AX Visio smart binocular—imposes strict computational constraints. The objective is to train a lightweight object detection architecture, specifically within the 1 to 3 million parameter YOLO-nano class, capable of classifying non-bird mammals in real-time directly on the edge device. The taxonomy targets a focused 225-class list and an extended 480-class list, requiring a highly robust and diverse training corpus to achieve acceptable mean Average Precision (mAP) across all classes.

The primary training corpus currently consists of 66,881 usable images sourced from the Global Biodiversity Information Facility (GBIF) and processed through the SpeciesNet v4.0.2a pipeline. However, this dataset exhibits severe class imbalance and label noise. Notably, 31 classes in the focused 225-class list contain zero images, and 55 classes have fewer than 10 images. Such sparsity precludes a deep learning model from learning generalized feature representations for these species. In the extended 480-class list, the data starvation is even more pronounced, with 170 classes completely absent from the training data. Furthermore, because the AX Visio is a premium commercial product, the intellectual property lineage of the training data must be pristine. Datasets governed by non-commercial (NC), no-derivatives (ND), or share-alike (SA) copyleft licenses introduce unacceptable legal risks to the proprietary model weights.

This report provides an exhaustive evaluation of major open wildlife image datasets, assesses their commercial viability for proprietary model training, maps them against the existing taxonomic coverage gaps, and outlines a comprehensive data engineering strategy to harmonize, clean, and combine these disparate data sources into a unified, legally compliant training corpus.

## **2\. Evaluation of Major Open Wildlife Image Datasets**

The global open-source ecosystem offers a vast repository of ecological data, but these datasets vary drastically in annotation quality, modality (infrared versus daylight), and licensing frameworks. A rigorous assessment of each primary repository is necessary to isolate data that satisfies both the technical requirement of bounding box annotations and the legal requirement of commercial usability.

### **2.1. iNaturalist Competition Datasets (2017, 2018, 2019, 2021\)**

The iNaturalist platform represents one of the largest aggregations of citizen science biodiversity data globally. Over several years, subsets of this data have been packaged into benchmark datasets for the Fine-Grained Visual Categorization (FGVC) workshops at computer vision conferences.1

**Dataset Characteristics and Content** The iNaturalist competition datasets are massive in scale and exceptionally high in image quality, consisting predominantly of high-resolution, daylight, color photographs taken by citizen scientists and professional photographers.2 The 2018 iteration, for instance, contains 437,513 training images spanning 8,142 species, of which 234 categories are mammals comprising 20,104 training images.3 The 2021 iteration expanded this significantly, featuring 246 distinct mammal species with 68,917 training images dedicated solely to the Mammalia super-category.4 The annotations primarily consist of image-level labels mapped to a rigorous taxonomic hierarchy, though subsets of bounding box annotations have been retroactively added to older competition subsets.4 The geographical coverage is global, and the overlap with the target 225-class and 480-class label sets is exceptionally high, making it technically ideal for supplementing daylight-focused inference engines like a smart binocular.

**Commercial Viability Assessment** Despite their technical perfection, the pre-packaged iNaturalist *competition* datasets are strictly barred from commercial use.5 The terms of use governing the downloads from the Visipedia GitHub and Kaggle repositories explicitly stipulate that users may use the data only for "non-commercial research and educational purposes" and that users "will NOT distribute the dataset images".4 Consequently, downloading the competition archives directly is legally non-viable and introduces an unacceptable vector for intellectual property litigation. The commercial usability of the pre-packaged competition datasets is evaluated as **NOT ALLOWED**.

**Alternative Data Extraction Strategy** While the competition bundles are restricted, the underlying iNaturalist platform functions differently. Photographers on iNaturalist retain their intellectual property but can apply voluntary Creative Commons licenses to their uploads.6 By default, iNaturalist applies a Creative Commons Attribution-NonCommercial (CC-BY-NC) license, but users frequently opt for fully permissive CC0 (Public Domain) or CC-BY (Attribution) licenses.7 iNaturalist regularly exports "Research Grade" observations that hold CC0, CC-BY, or CC-BY-NC licenses to the Global Biodiversity Information Facility (GBIF) and the Amazon Web Services (AWS) Open Data Registry.8

Therefore, a commercial entity can entirely bypass the restricted competition datasets by writing a targeted script to query the GBIF API or the AWS S3 bucket (s3://inaturalist-open-data/). By explicitly filtering the metadata payload to isolate records where license=CC0 or license=CC-BY, machine learning engineers can reconstruct a massive, commercially safe mirror of the iNaturalist data.10 This process requires processing the images through an object detector to generate bounding boxes, but it cleanly avoids the non-commercial stipulations of the FGVC competition rules.

### **2.2. Labeled Information Library of Alexandria: Biology and Conservation (LILA BC)**

LILA BC is the premier aggregator of ecological image datasets, heavily focused on camera trap deployments. Hosted in partnership with Microsoft AI for Earth, Google Cloud, and Source Cooperative, LILA BC standardizes diverse wildlife datasets into the common COCO Camera Traps format.13 Crucially, the organizers of LILA BC have made a concerted effort to publish these datasets under the Community Data License Agreement (CDLA) \- Permissive variant.14 The CDLA-Permissive license functions similarly to an MIT or Apache 2.0 license for data, expressly permitting commercial use, modification, and the training of proprietary machine learning models.

#### **2.2.2. Snapshot Serengeti and the Snapshot Safari Network**

The Snapshot Serengeti project, which subsequently expanded into the cross-continental Snapshot Safari network, represents one of the highest-volume camera trap repositories in existence.16 The Serengeti dataset alone contains approximately 2.65 million sequences totaling 7.1 million images, while the 2024 Expansion dataset contributes an additional 4.02 million images from 15 separate African camera trapping projects.17

These datasets are particularly critical for the current deployment because they explicitly cover the African antelope gap species that are missing from the primary training corpus. The 2024 Expansion dataset provides labels for 151 categories, including massive volumes of impala and zebra, but also critical gap species such as the bongo, nyala, roan antelope, kob, dik-dik, and klipspringer.18 The images are a mixture of daylight color photography and nighttime infrared (IR) imagery, reflecting standard motion-triggered camera trap behavior.

While Snapshot Serengeti provides approximately 150,000 human-annotated bounding boxes across 78,000 images, the vast majority of the 11 million combined images feature only sequence-level or image-level labels.17 Furthermore, up to 76% of the images in certain splits are labeled as empty, reflecting misfires caused by wind or vegetation.17 Both the Serengeti and Safari Expansion datasets are licensed under the CDLA-Permissive variant, making their commercial usability evaluation **SAFE**.17 The Zooniverse platform also notes that the raw images are licensed under Creative Commons Attribution 4.0 International (CC-BY 4.0), which provides redundant legal protection for commercial use.20

#### **2.2.3. Wildlife Conservation Society (WCS) Camera Traps**

The WCS Camera Traps dataset is among the most taxonomically diverse camera trap collections publicly available, encompassing approximately 1.4 million images representing roughly 675 species from 12 countries.21 The geographical spread is highly advantageous, heavily sampling from South American and Asian ecosystems. This directly addresses the Asian and neotropical mammal coverage gaps, providing vital training data for the Asiatic wild ass, water deer, various macaques, sloths, jaguars, and tapirs.21

The WCS dataset is uniquely valuable because it includes approximately 375,000 ground-truth bounding box annotations across 300,000 images.21 These bounding boxes bypass the need for secondary localization modeling. However, the dataset exhibits severe class imbalance. The most frequent species, the White-lipped peccary (*Tayassu pecari*), accounts for nearly 95,000 images, while rare target species may possess fewer than 100 images.21 Like Snapshot Safari, the WCS dataset is released under the CDLA-Permissive license, rendering its commercial usability **SAFE**.21

#### **2.2.4. Caltech Camera Traps (CCT-20)**

The Caltech Camera Traps dataset consists of 243,100 images from 140 camera locations concentrated in the Southwestern United States.22 It provides labels for 21 animal categories, predominantly common North American fauna such as opossums, raccoons, and coyotes, accompanied by approximately 66,000 bounding box annotations.22 While the taxonomic overlap with the specific African, Asian, and marine gap species is exceptionally low, CCT-20 remains highly relevant for algorithmic development. It is widely used in literature to benchmark domain generalization algorithms—specifically evaluating how well a model trained on camera locations A through M performs on novel camera location N.23 The dataset is governed by the CDLA-Permissive license, making its commercial usability **SAFE**.22

### **2.3. The iWildCam Competitions (2019, 2020, 2021, 2022\)**

The iWildCam challenges are hosted annually on Kaggle in conjunction with the FGVC workshops at the Computer Vision and Pattern Recognition (CVPR) conference.25 The objective of these challenges usually involves species classification or counting individual animals across temporal sequences.25

**Dataset Characteristics and Licensing Hazards** The iWildCam datasets are not monolithic, original collections; rather, they are complex aggregations of existing repositories. The 2022 iteration, for example, contained approximately 260,000 camera trap images derived primarily from the WCS Camera Traps dataset, supplemented with multispectral Landsat-8 satellite imagery and iNaturalist data.25

The licensing environment surrounding iWildCam is treacherous for commercial entities. The competition rules mandate that winning submissions must release their source code and models under an Open Source Initiative (OSI) approved license (e.g., Apache 2.0) that permits commercial use.27 However, this permissive licensing applies solely to the *competitors' code and trained model weights*, not to the underlying images provided by the sponsors. The underlying image corpora inherit the licenses of their source material. Because the iWildCam bundles explicitly incorporate data from the iNaturalist 2017-2021 competitions 29—which strictly prohibit commercial use 5—the aggregate iWildCam bundle is legally contaminated. Downloading and training directly on the pre-packaged iWildCam dataset presents a severe copyright risk. The commercial usability of the aggregated iWildCam dataset is evaluated as **NOT ALLOWED**. The optimal engineering strategy is to ignore the iWildCam bundles entirely and independently download the CDLA-Permissive WCS Camera Traps dataset directly from LILA BC.

### **2.4. Open Images V7 (Google)**

Google's Open Images V7 is a massive, multi-modal computer vision dataset encompassing approximately 9 million images.31 Unlike ecological datasets that utilize precise Linnaean taxonomy, Open Images relies on a machine-generated and human-verified semantic hierarchy containing 600 boxable object classes.31

**Mammalian Coverage and Annotation Quality** The dataset contains a staggering 16 million manually verified bounding boxes, making it one of the most reliable sources for spatial localization training.31 The mammalian coverage is structurally different from GBIF or LILA BC. It includes a generic "Mammal" class featuring 95,335 images with bounding boxes, deliberately capped by the creators due to the overwhelming presence of over 1.3 million mammalian instances in the raw corpus.31 Beyond the generic node, Open Images provides specific bounding boxes for numerous target megafauna and gap species, including the Jaguar, Kangaroo, Leopard, Elephant, Rhinoceros, Sea Lion, and Cattle.32

The primary limitation of Open Images V7 is taxonomic precision. A bounding box labeled "Elephant" does not distinguish between *Loxodonta africana* (African elephant) and *Elephas maximus* (Asian elephant). Consequently, while Open Images is highly valuable for providing the YOLO-nano architecture with robust, daylight-color bounding box regressions for mammals, the labels must be manually audited or passed through a secondary taxonomic classifier before integration into the 480-class SpeciesNet framework.

**Commercial Usability** The annotations (bounding boxes, segmentation masks) in Open Images V7 are licensed under CC-BY 4.0, while the underlying images are listed as CC-BY 2.0.33 Google explicitly notes that while they attempted to identify images under permissive licenses, they make no warranties regarding the license status, advising users to verify licenses independently.33 For the purpose of training a commercial model, utilizing CC-BY data is widely accepted as legally compliant, provided the deploying entity generates a manifest attributing the original creators. The commercial usability for model training is evaluated as **SAFE**, though direct image redistribution requires rigorous attribution tracking.

### **2.5. MS COCO 2017 (Common Objects in Context)**

The MS COCO dataset is the foundational benchmark for object detection and instance segmentation, containing 330,000 images, of which 118,000 constitute the standard training split.34

**Coverage and Annotations** COCO utilizes a highly constrained taxonomy of exactly 80 object categories.34 Within this taxonomy, there are exactly 10 mammal classes: person, cat, dog, horse, sheep, cow, elephant, bear, zebra, and giraffe.35 The annotations are exceptionally high quality, offering both bounding box coordinates and polygon segmentation masks for every object.35 While COCO will not assist in distinguishing rare African antelopes, it provides pristine, daylight, complex-background training examples for domestic animals (pigs, cows, sheep) and generalized megafauna, which are explicitly listed in the target gap analysis. The severe class imbalance inherent to COCO—where "person" dominates the distribution while other classes trail behind—must be managed via class-aware sampling during training.35

**Commercial Usability** The annotations within COCO are owned by the COCO Consortium and licensed under CC-BY 4.0.37 The images themselves are governed by the Flickr Terms of Use, as they were scraped directly from the Flickr platform.37 In corporate machine learning environments, training on COCO is generally considered safe under fair use doctrines and the implied licenses of open-web indexing, though directly redistributing the raw images is discouraged. The commercial usability for model training is evaluated as **SAFE**.

### **2.6. Wildlife-10, Wildlife-71, and Re-Identification Benchmarks**

Standardized wildlife classification benchmarks occasionally emerge from the academic sector, such as the Wildlife-71 dataset (often associated with Amur Tiger Re-identification, ATRW, and UniReID architectures).

**Characteristics and Legal Hazards** Wildlife-71 is designed for Re-Identification (ReID)—the task of identifying a specific individual animal across multiple camera views, rather than general species-level object detection.38 The dataset spans 71 wildlife categories and provides bounding boxes. However, the provenance of the data is highly problematic. The dataset was constructed by aggregating existing tracking datasets (like GOT-10k) and actively crawling and scraping YouTube videos to extract target bounding boxes.39

Scraping YouTube videos without explicit programmatic consent or permissive licensing creates massive copyright infringement liabilities. Because the dataset organizers cannot confer a permissive commercial license upon scraped proprietary video content, utilizing Wildlife-71 in a commercial deployment is an extreme legal risk. The commercial usability of Wildlife-71 is evaluated as **NOT ALLOWED**.

### **2.7. TreeOfLife-200M and BioCLIP**

Curated by the Imageomics Institute at Ohio State University, TreeOfLife-200M represents the largest and most diverse machine-learning-ready dataset for biological computer vision.41

**Scale and Modality** TreeOfLife-200M aggregates images and metadata from the Global Biodiversity Information Facility (GBIF), the Encyclopedia of Life (EOL), BIOSCAN-5M, and FathomNet.41 It contains an unprecedented 214 million images covering 952,257 unique taxa.41 The dataset provides vast contextual diversity, encompassing museum specimens, camera trap images, and citizen science photos.41

**Licensing Complexity and Strategic Application** The sheer scale of TreeOfLife-200M introduces profound licensing heterogeneity. The dataset acts as a metadata index pointing to files with varying copyright statuses.41 While BIOSCAN-5M images are largely CC-BY 3.0, FathomNet images (which cover highly valuable marine life) are restricted under CC-BY-NC-ND 4.0, forbidding commercial use and derivative works.43 Therefore, executing a bulk download of the TreeOfLife-200M tarballs is legally unsafe for the Swarovski deployment. The commercial usability of the raw dataset bundle is evaluated as **NEEDS REVIEW / REQUIRES FILTERING**.

However, the researchers trained a vision foundation model, BioCLIP 2, on this data. The BioCLIP 2 model weights are officially released under the permissive MIT License.44 This creates a powerful strategic loophole: while the underlying FathomNet images are restricted from commercial use, the MIT-licensed BioCLIP 2 model can be legally deployed as a "teacher" network. BioCLIP 2 can be used to run zero-shot inference over millions of commercially safe, unannotated wildlife images (e.g., CC0 images from Unsplash or public domain archives), effectively pseudo-labeling a massive new dataset for the target YOLO-nano model without ever touching the restricted CC-BY-NC-ND images.

### **2.8. Marine and Specialized Datasets**

Marine mammals (sea otters, walruses, elephant seals, and other pinnipeds) constitute a significant coverage gap. Marine datasets heavily favor telemetry, acoustic tracking, and spatial polygons over annotated optical imagery.45 Datasets like NOAA's SEAMAP or the SWFSC juvenile loggerhead tracking focus on coordinate data rather than bounding-box computer vision.45

Optical marine datasets do exist, such as the LILA BC Beluga ID dataset (5,902 bounding-boxed images of *Delphinapterus leucas*).47 However, to cover the specific gaps of pinnipeds and sea otters, the optimal strategy avoids searching for pre-packaged marine computer vision benchmarks. Instead, data engineers must query the GBIF API explicitly for the taxonomic families *Odobenidae* (walruses), *Phocidae* (earless seals), *Otariidae* (eared seals), and *Mustelidae* (specifically *Enhydra lutris* for sea otters) 48, strictly filtering the query for CC0 and CC-BY licenses.50

## ---

**3\. Taxonomic Gap Analysis and Mitigation**

The existing SpeciesNet-processed dataset exhibits critical omissions against the commercial target list. 31 classes possess zero images, and 55 classes have fewer than 10 images, precluding the YOLO-nano architecture from establishing decision boundaries in the latent space.

### **3.1. Gap Coverage Matrix**

The following matrix estimates the volume of commercially viable imagery available to satisfy the specific zero-image and low-image classes required for the AX Visio deployment.

| Taxonomic Category | Target Gap Species | Recommended Primary Data Source | Estimated Viable Image Yield | Annotation Status |
| :---- | :---- | :---- | :---- | :---- |
| **African Antelopes** | Bongo, Nyala, Roan antelope, Kob, Dik-dik, Klipspringer | **Snapshot Safari / Serengeti** (LILA BC) | 10,000+ images per species. | Image-level labels. Requires MegaDetector processing to generate bounding boxes. |
| **Asian Mammals** | Asiatic wild ass, Water deer, Yak, Water buffalo, Japanese macaque, Sloth bear | **WCS Camera Traps** (LILA BC) & **GBIF** (CC-BY/CC0 filtered) | 500 \- 2,000 images per species. | WCS provides ground-truth bounding boxes. GBIF requires MegaDetector. |
| **Marine Mammals** | Sea otter, Walrus, Elephant seals, Eared seals, Pinnipeds | **GBIF** (CC-BY/CC0 filtered) & **Open Images V7** | 1,000 \- 5,000 images per species. | Open Images ("Sea lion" class) provides boxes. GBIF requires MegaDetector. |
| **Primates** | Aye-aye, Ring-tailed lemur, Patas monkey, Drill | **GBIF** (CC-BY/CC0 filtered) | 50 \- 300 images per species. | Image-level labels. High risk of poor background diversity due to zoo photography. Requires MegaDetector. |
| **Other / Domestic** | Meerkat, Saiga, Brown hyena, Eurasian lynx, Domestic pig, Sloths | **WCS Camera Traps**, **COCO 2017**, **Open Images V7** | 5,000+ images per species. | High-quality bounding boxes natively available in WCS, COCO, and Open Images. |

### **3.2. Strategies for Persistently Uncovered Species**

While African antelopes and domestic animals can be easily satisfied via LILA BC and COCO, niche species like the Aye-aye (*Daubentonia madagascariensis*) or the Saiga antelope (*Saiga tatarica*) may fail to yield the 50 to 100 diverse, high-quality daylight images necessary to prevent model overfitting. Camera trap data for arboreal primates is notably scarce, and citizen science photos of deep-forest nocturnal primates are often heavily blurred or flash-washed. To mitigate persistent data starvation without violating commercial licensing constraints, the following alternative strategies are recommended:

**1\. Synthetic Data Generation via Latent Diffusion Models**

Modern text-to-image architectures (e.g., Stable Diffusion XL, Adobe Firefly) can synthesize photorealistic training data. If only 10 high-quality CC0 images of an Aye-aye exist, a Low-Rank Adaptation (LoRA) module can be trained on these 10 images. The LoRA can then guide the diffusion model to generate thousands of synthetic Aye-ayes placed against highly diverse, procedurally generated backgrounds (e.g., daylight canopies, riverbanks, savanna scrub). This technique forces the downstream YOLO-nano model to learn the morphological invariants of the mammal rather than overfitting to the specific foliage present in the 10 real photographs. Using an enterprise-licensed generative model ensures the resulting synthetic dataset is unencumbered by copyright claims.

**2\. Teacher-Student Pseudo-Labeling via BioCLIP 2** Because the BioCLIP 2 model weights are released under an MIT license 44, the model can be legally deployed in a commercial pipeline. Data engineers can download massive repositories of commercially safe, unlabeled nature photography (for instance, filtering the Unsplash API for animal imagery). BioCLIP 2 can be run over these unlabeled millions in a zero-shot capacity, querying specifically for the 31 gap species. When BioCLIP 2 outputs a high-confidence prediction (e.g., \>0.95 softmax probability for *Patas monkey*), the image is routed to MegaDetector to draw a bounding box, creating a fully automated, legally pristine data mining pipeline.

**3\. Targeted Web Scraping via Wikimedia Commons**

Unlike broad Google Images scraping, the Wikimedia Commons API allows programmatic extraction of images explicitly filtered by license type (CC0, CC-BY, Public Domain). Images tagged with CC-BY-SA (ShareAlike) must be programmatically dropped to prevent copyleft contamination. While volume will be low, the taxonomic precision of Wikimedia files ensures high-quality additions to rare classes like the Drill or Water deer.

## ---

**4\. Dataset Combination and Engineering Strategy**

Fusing highly disparate datasets—spanning the citizen science daylight photos of GBIF and the infrared night-vision motion-captures of Snapshot Safari—presents severe optimization challenges for a lightweight object detection model. The following data engineering pipeline establishes the practical strategy for harmonization and combination.

### **4.1. Taxonomic Harmonization**

The integration of Open Images V7, COCO, WCS, and GBIF introduces severe taxonomic collision. Open Images utilizes colloquial semantic nodes (e.g., "pig"), while WCS utilizes strict Linnaean classification (e.g., *Sus scrofa*). The target model must output labels mapped to the existing SpeciesNet v4.0.2a JSON backbone.

To resolve this, LILA BC provides a centralized taxonomic mapping CSV file that translates every localized, idiosyncratic category from its constituent camera trap datasets into the standardized iNaturalist taxonomy.51 All supplementary data must be programmatically forced through this mapping file to align with the SpeciesNet hierarchy. For datasets lacking strict taxonomic depth (e.g., COCO's "bear" class), the label should be mapped to the highest confident taxonomic node (e.g., the *Ursidae* family). The YOLO-nano loss function must be modified to accept hierarchical roll-up predictions; if the model predicts *Ursidae* on a COCO image, it should not be penalized for failing to predict *Sloth bear*, thereby preventing the injection of false gradients during backpropagation.

### **4.2. Handling Mixed Modalities (Infrared vs. Daylight RGB)**

The Swarovski AX Visio operates as an optical, daylight-capable binocular.52 Training a computer vision model heavily on LILA BC's infrared (IR) night-time images introduces a severe domain shift. If the model learns that a dik-dik is characterized by glowing white retinas and a hyper-luminous body against a pitch-black background, the convolutional filters will fail to detect the animal's natural camouflage in broad daylight.

This modality gap requires a highly structured training curriculum:

1. **Filtering for Inference Equivalence:** The validation and test splits must perfectly mirror the target deployment environment. Therefore, all grayscale/IR images must be excluded from the validation and test sets.  
2. **Curriculum Pre-Training:** The IR images from Snapshot Safari and WCS should be utilized during the initial pre-training phase. While the color statistics are inverted, the IR images provide excellent data for learning robust morphological shapes, edge detection, and pose variations, which are modality-invariant.  
3. **Aggressive Augmentation:** To prevent the model from relying on color signatures during the final fine-tuning phase, aggressive data augmentation must be applied to the daylight RGB images (from GBIF and Open Images). Applying severe color jitter, random channel dropping, and spontaneous grayscale conversion forces the network to rely on structural shape rather than color, mathematically bridging the gap between the IR training data and the RGB inference environment.

### **4.3. Bounding Box Generation via MegaDetector**

The target YOLO-nano architecture is an object detector, making spatial bounding boxes a strict requirement for the loss function. While WCS, COCO, and Open Images provide native bounding boxes, datasets like Snapshot Serengeti and GBIF exports primarily provide image-level labels.17

This necessitates the integration of the Microsoft AI for Earth MegaDetector (currently v5a/v5b).53 MegaDetector is an open-source, commercially viable object detection model trained exclusively to draw highly accurate bounding boxes around "animals," "people," and "vehicles," operating entirely agnostically of species.53

The pipeline involves passing all un-boxed Snapshot Safari and GBIF images through MegaDetector. If MegaDetector returns a high-confidence bounding box for an "animal," and the original metadata confirms the presence of exactly one species (e.g., "1 Impala"), the pipeline assigns the metadata label to the bounding box coordinate. To prevent box-label mismatch, any image containing multiple species in the metadata must be discarded from this automated pipeline.

### **4.4. Managing Label Noise**

The existing SpeciesNet-generated dataset exhibits a 23% disagreement rate with filename-inferred species, indicating profound label noise. Training a lightweight model on noisy data degrades bounding box regression and classification accuracy.

This requires the implementation of a Confident Learning framework (utilizing open-source libraries such as cleanlab). Confident learning algorithms identify out-of-distribution label errors by analyzing the predicted softmax probabilities of a heavy teacher model against the provided labels. Images flagged with high epistemic uncertainty should be either dynamically discarded from the batch or re-weighted. Furthermore, the loss function should implement sample weighting: pristine, human-verified datasets (COCO, Open Images, Caltech Camera Traps) should command a higher learning rate multiplier than the computationally inferred SpeciesNet labels or unverified citizen science uploads.

### **4.5. Spatial Train/Validation/Test Split Strategy**

Camera trap datasets are notoriously susceptible to background overfitting. If a camera is bolted to a tree for six months, the neural network may memorize the static arrangement of the rocks and foliage behind the animal rather than learning the features of the animal itself. If a random 80/20 train/test split is applied across the images, the test set will contain near-identical backgrounds to the training set, resulting in artificially inflated accuracy metrics that collapse upon real-world deployment.

To combat this, the data splits must be executed strictly at the **Location/Camera ID level**.24 All images generated by "Camera Location A" must be routed entirely to the training set, while all images from "Camera Location B" must be routed entirely to the validation set. This "trans-location" evaluation ensures the model generalizes to novel environments. Finally, the hold-out test set should consist exclusively of citizen science (GBIF) and Open Images photography, as the variable, hand-held nature of these images most closely simulates the user experience of looking through the Swarovski AX Visio binoculars.

### **4.6. Total Estimated Image Count**

By combining the existing 66,881 usable images with the WCS Camera Traps (filtered for target species), Snapshot Safari (MegaDetector boxed), Open Images V7 (Mammal subsets), and a strictly filtered GBIF export (CC0/CC-BY), the unified dataset size will achieve an estimated **250,000 to 450,000 highly curated, bounding-boxed, commercially safe images**. This volume sits comfortably within the optimal training regime for a 1–3 million parameter YOLO-nano architecture, providing sufficient data depth without exceeding the training compute budget.

## ---

**5\. Intellectual Property and License Risk Assessment**

The hard requirement for commercial usability dictates the exclusion of several prominent datasets. In software and data licensing, the distinction between *model training* and *image redistribution* is critical. While fair use arguments exist for training AI on copyrighted data, corporate compliance for a premium product demands strict adherence to permissive licenses to prevent copyleft "viral" infection of the proprietary model weights.

### **5.1. The Copyleft / ShareAlike (SA) Gray Area**

Licenses such as CC-BY-SA (Creative Commons Attribution-ShareAlike) and CDLA-Sharing impose a "copyleft" obligation, requiring that derivative works be distributed under the identical license. In the context of deep learning, it remains a heavily contested legal question whether the mathematical matrix weights of a trained neural network constitute a "derivative work" of the underlying training images. If a court rules affirmatively, training the AX Visio's model on CC-BY-SA images could legally force Swarovski to open-source the proprietary YOLO-nano model weights. To completely insulate the commercial deployment from open-source infection, **all ShareAlike (SA), Non-Commercial (NC), and No-Derivatives (ND) licenses must be strictly excluded from the training pipeline.**

### **5.2. Recommended Dataset Licensing Summary**

The following table summarizes the legal viability of the investigated datasets concerning the AX Visio deployment.

| Dataset Name | Primary License(s) | Commercial Model Training | Image Redistribution | Specific Legal Concerns & Obligations |
| :---- | :---- | :---- | :---- | :---- |
| **GBIF / iNaturalist (Filtered Export)** | CC0, CC-BY | **SAFE** | **SAFE** | **Must explicitly filter out CC-BY-NC and CC-BY-SA.** For CC-BY images used, a manifest of attributions (photographer names/URIs) must be generated programmatically.50 |
| **iNaturalist Competition Data** | Custom Academic | **NOT ALLOWED** | **NOT ALLOWED** | Explicitly prohibits commercial use and redistribution.5 Bypassed by using the GBIF filtered API export instead. |
| **Snapshot Safari / Serengeti** | CDLA-Permissive / CC-BY 4.0 | **SAFE** | **SAFE** | The CDLA-Permissive was built specifically by the Linux Foundation for data sharing and explicitly permits commercial model training without copyleft obligations.17 |
| **WCS Camera Traps** | CDLA-Permissive | **SAFE** | **SAFE** | Governed by the same legal protections as Snapshot Safari.21 |
| **Caltech Camera Traps (CCT-20)** | CDLA-Permissive | **SAFE** | **SAFE** | Fully safe for commercial model training.22 |
| **iWildCam Competition Bundle** | Mixed / Restricted | **NOT ALLOWED** | **NOT ALLOWED** | Avoid downloading the Kaggle bundle. It is contaminated with restricted iNat data.26 Source constituent datasets directly from LILA BC instead. |
| **Open Images V7** | CC-BY 2.0 (Images) / CC-BY 4.0 (Boxes) | **SAFE** | **NEEDS REVIEW** | Requires attribution. Google states they "tried to identify" CC images but makes no legal warranty; commercial users should ideally cross-reference source URLs before attempting redistribution.33 |
| **COCO 2017** | CC-BY 4.0 (Boxes) / Flickr (Images) | **SAFE** | **NEEDS REVIEW** | Bounding boxes are CC-BY. Images belong to Flickr users. Training is generally accepted as safe in industry, but direct image redistribution inside the product software is legally unsafe.37 |
| **TreeOfLife-200M** | Mixed (CC0 to CC-BY-NC-SA) | **NEEDS REVIEW** | **NOT ALLOWED** | Do not train on the raw dataset. Must explicitly filter the metadata for CC0/CC-BY before fetching images, as components like FathomNet are strictly CC-BY-NC-ND.43 |
| **BioCLIP 2 Model Weights** | MIT License | **SAFE** | **SAFE** | The *model weights* are MIT licensed.44 Fully safe to use as a zero-shot teacher model for pseudo-labeling unannotated commercial imagery. |
| **Wildlife-71 (ATRW)** | Unknown (Web Scraped) | **NOT ALLOWED** | **NOT ALLOWED** | Contains unverified YouTube scrapes.39 Presents an extreme copyright infringement risk. Must be excluded entirely. |

### **5.3. Compliance Execution**

To ensure absolute legal compliance, the data engineering pipeline must implement an automated, programmatic gatekeeper. An ingestion script must parse the metadata JSON or CSV of every proposed image, check the license string, and aggressively drop any record containing "NC" (Non-Commercial), "ND" (No-Derivatives), "SA" (ShareAlike), or "All Rights Reserved".

Following the completion of the training cycle, the data pipeline must auto-generate a master attribution document containing the URIs, author names, and license types of all CC-BY images utilized in the final training corpus. Embedding this text manifest within the AX Visio's digital documentation, user manual, or companion mobile application satisfies the attribution clauses inherent to CC-BY, CDLA-Permissive, and Apache 2.0 licenses, ensuring the product launches with a legally pristine machine learning architecture.

#### **Works cited**

1. visipedia/inat\_comp: iNaturalist competition details \- GitHub, accessed March 23, 2026, [https://github.com/visipedia/inat\_comp](https://github.com/visipedia/inat_comp)  
2. A Guide to iNaturalist | Atlas of Living Australia, accessed March 23, 2026, [https://ala.org.au/app/uploads/2024/04/A\_Guide\_to\_iNaturalist\_Apr2024-1.pdf](https://ala.org.au/app/uploads/2024/04/A_Guide_to_iNaturalist_Apr2024-1.pdf)  
3. README.md \- iNaturalist 2018 Competition \- GitHub, accessed March 23, 2026, [https://github.com/visipedia/inat\_comp/blob/master/2018/README.md](https://github.com/visipedia/inat_comp/blob/master/2018/README.md)  
4. README.md \- iNaturalist 2021 Competition \- GitHub, accessed March 23, 2026, [https://github.com/visipedia/inat\_comp/blob/master/2021/README.md](https://github.com/visipedia/inat_comp/blob/master/2021/README.md)  
5. iNat Challenge 2021 \- FGVC8 \- Kaggle, accessed March 23, 2026, [https://www.kaggle.com/competitions/inaturalist-2021/rules](https://www.kaggle.com/competitions/inaturalist-2021/rules)  
6. How do licenses work on iNaturalist? Should I change my licenses?, accessed March 23, 2026, [https://help.inaturalist.org/en/support/solutions/articles/151000173511-how-do-licenses-work-on-inaturalist-should-i-change-my-licenses-](https://help.inaturalist.org/en/support/solutions/articles/151000173511-how-do-licenses-work-on-inaturalist-should-i-change-my-licenses-)  
7. What are licenses? How can I update the licenses on my content? \- iNaturalist Help, accessed March 23, 2026, [https://help.inaturalist.org/en/support/solutions/articles/151000175695-what-are-licenses-how-can-i-update-the-licenses-on-my-content-](https://help.inaturalist.org/en/support/solutions/articles/151000175695-what-are-licenses-how-can-i-update-the-licenses-on-my-content-)  
8. Documentation for iNaturalist Open Data \- GitHub, accessed March 23, 2026, [https://github.com/inaturalist/inaturalist-open-data](https://github.com/inaturalist/inaturalist-open-data)  
9. Which iNaturalist observations are exported for GBIF, and how often does this export happen?, accessed March 23, 2026, [https://help.inaturalist.org/en/support/solutions/articles/151000170346-which-inaturalist-observations-are-exported-for-gbif-and-how-often-does-this-export-happen-](https://help.inaturalist.org/en/support/solutions/articles/151000170346-which-inaturalist-observations-are-exported-for-gbif-and-how-often-does-this-export-happen-)  
10. PhenoVision: A framework for automating and delivering research-ready plant phenology data from field images \- bioRxiv.org, accessed March 23, 2026, [https://www.biorxiv.org/content/10.1101/2024.10.10.617505.full.pdf](https://www.biorxiv.org/content/10.1101/2024.10.10.617505.full.pdf)  
11. PhenoVision: A framework for automating and delivering research-ready plant phenology data from field images \- bioRxiv.org, accessed March 23, 2026, [https://www.biorxiv.org/content/10.1101/2024.10.10.617505v1.full-text](https://www.biorxiv.org/content/10.1101/2024.10.10.617505v1.full-text)  
12. iNaturalist Research-grade Observations \- GBIF, accessed March 23, 2026, [https://www.gbif.org/dataset/50c9509d-22c7-4a22-a47d-8c48425ef4a7](https://www.gbif.org/dataset/50c9509d-22c7-4a22-a47d-8c48425ef4a7)  
13. LILA BC: Home, accessed March 23, 2026, [https://lila.science/](https://lila.science/)  
14. society-ethics/lila\_camera\_traps · Datasets at Hugging Face, accessed March 23, 2026, [https://huggingface.co/datasets/society-ethics/lila\_camera\_traps](https://huggingface.co/datasets/society-ethics/lila_camera_traps)  
15. North American Camera Trap Images \- LILA BC, accessed March 23, 2026, [https://lila.science/datasets/nacti/](https://lila.science/datasets/nacti/)  
16. Snapshot Safari: A large-scale collaborative to monitor Africa's remarkable biodiversity, accessed March 23, 2026, [https://www.researchgate.net/publication/348882187\_Snapshot\_Safari\_A\_large-scale\_collaborative\_to\_monitor\_Africa's\_remarkable\_biodiversity](https://www.researchgate.net/publication/348882187_Snapshot_Safari_A_large-scale_collaborative_to_monitor_Africa's_remarkable_biodiversity)  
17. Snapshot Serengeti \- LILA BC, accessed March 23, 2026, [https://lila.science/datasets/snapshot-serengeti/](https://lila.science/datasets/snapshot-serengeti/)  
18. Snapshot Safari 2024 Expansion \- LILA BC, accessed March 23, 2026, [https://lila.science/datasets/snapshot-safari-2024-expansion/](https://lila.science/datasets/snapshot-safari-2024-expansion/)  
19. A Scoping Review of Viral Diseases in African Ungulates \- MDPI, accessed March 23, 2026, [https://www.mdpi.com/2306-7381/8/2/17](https://www.mdpi.com/2306-7381/8/2/17)  
20. Snapshot Safari \- Zooniverse, accessed March 23, 2026, [https://www.zooniverse.org/organizations/meredithspalmer/snapshot-safari](https://www.zooniverse.org/organizations/meredithspalmer/snapshot-safari)  
21. WCS Camera Traps \- LILA BC, accessed March 23, 2026, [https://lila.science/datasets/wcscameratraps/](https://lila.science/datasets/wcscameratraps/)  
22. Caltech Camera Traps \- LILA BC, accessed March 23, 2026, [https://lila.science/datasets/caltech-camera-traps/](https://lila.science/datasets/caltech-camera-traps/)  
23. Caltech Camera Traps \- COVE \- Computer Vision Exchange, accessed March 23, 2026, [https://cove.thecvf.com/datasets/330](https://cove.thecvf.com/datasets/330)  
24. Towards Zero-Shot Camera Trap Image Categorization \- arXiv.org, accessed March 23, 2026, [https://arxiv.org/html/2410.12769v1](https://arxiv.org/html/2410.12769v1)  
25. iWildCam 2022 \- FGVC9 \- Kaggle, accessed March 23, 2026, [https://www.kaggle.com/competitions/iwildcam2022-fgvc9](https://www.kaggle.com/competitions/iwildcam2022-fgvc9)  
26. iWildCam 2022 \- LILA BC, accessed March 23, 2026, [https://lila.science/datasets/iwildcam-2022/](https://lila.science/datasets/iwildcam-2022/)  
27. iWildcam 2021 \- FGVC8 \- Kaggle, accessed March 23, 2026, [https://www.kaggle.com/competitions/iwildcam2021-fgvc8/rules](https://www.kaggle.com/competitions/iwildcam2021-fgvc8/rules)  
28. iWildCam 2020 \- FGVC7 \- Kaggle, accessed March 23, 2026, [https://www.kaggle.com/competitions/iwildcam-2020-fgvc7/rules](https://www.kaggle.com/competitions/iwildcam-2020-fgvc7/rules)  
29. visipedia/iwildcam\_comp: iWildCam competition details \- GitHub, accessed March 23, 2026, [https://github.com/visipedia/iwildcam\_comp](https://github.com/visipedia/iwildcam_comp)  
30. iwildcam\_comp/2019/readme.md at master \- GitHub, accessed March 23, 2026, [https://github.com/visipedia/iwildcam\_comp/blob/master/2019/readme.md](https://github.com/visipedia/iwildcam_comp/blob/master/2019/readme.md)  
31. Open Images V7 \- Description \- Googleapis.com, accessed March 23, 2026, [https://storage.googleapis.com/openimages/web/factsfigures\_v7.html](https://storage.googleapis.com/openimages/web/factsfigures_v7.html)  
32. List of classes from the OpenImages dataset that are segmentable. \- GitHubのGist, accessed March 23, 2026, [https://gist.github.com/hgaiser/960811a7191acbbf772103ff7bbc002a](https://gist.github.com/hgaiser/960811a7191acbbf772103ff7bbc002a)  
33. READMEV3.md \- openimages/dataset \- GitHub, accessed March 23, 2026, [https://github.com/openimages/dataset/blob/master/READMEV3.md](https://github.com/openimages/dataset/blob/master/READMEV3.md)  
34. COCO Dataset \- Ultralytics YOLO Docs, accessed March 23, 2026, [https://docs.ultralytics.com/datasets/detect/coco/](https://docs.ultralytics.com/datasets/detect/coco/)  
35. COCO Dataset: All You Need to Know to Get Started, accessed March 23, 2026, [https://www.v7labs.com/blog/coco-dataset-guide](https://www.v7labs.com/blog/coco-dataset-guide)  
36. An Introduction to the COCO Dataset \- Roboflow Blog, accessed March 23, 2026, [https://blog.roboflow.com/coco-dataset/](https://blog.roboflow.com/coco-dataset/)  
37. COCO Dataset (2017) \- EdgeFirst Studio Documentation, accessed March 23, 2026, [https://doc.edgefirst.ai/test/datasets/coco/](https://doc.edgefirst.ai/test/datasets/coco/)  
38. ATRW: A Benchmark for Amur Tiger Re-identification in the Wild \- ResearchGate, accessed March 23, 2026, [https://www.researchgate.net/publication/346179472\_ATRW\_A\_Benchmark\_for\_Amur\_Tiger\_Re-identification\_in\_the\_Wild](https://www.researchgate.net/publication/346179472_ATRW_A_Benchmark_for_Amur_Tiger_Re-identification_in_the_Wild)  
39. WildlifeReID-10k: Wildlife re-identification dataset with 10k individual animals \- CVF Open Access, accessed March 23, 2026, [https://openaccess.thecvf.com/content/CVPR2025W/FGVC/papers/Adam\_WildlifeReID-10k\_Wildlife\_re-identification\_dataset\_with\_10k\_individual\_animals\_CVPRW\_2025\_paper.pdf](https://openaccess.thecvf.com/content/CVPR2025W/FGVC/papers/Adam_WildlifeReID-10k_Wildlife_re-identification_dataset_with_10k_individual_animals_CVPRW_2025_paper.pdf)  
40. Deep Relative Distance Learning: Tell the Difference between Similar Vehicles, accessed March 23, 2026, [https://www.researchgate.net/publication/311611186\_Deep\_Relative\_Distance\_Learning\_Tell\_the\_Difference\_between\_Similar\_Vehicles](https://www.researchgate.net/publication/311611186_Deep_Relative_Distance_Learning_Tell_the_Difference_between_Similar_Vehicles)  
41. imageomics/TreeOfLife-200M · Datasets at Hugging Face, accessed March 23, 2026, [https://huggingface.co/datasets/imageomics/TreeOfLife-200M](https://huggingface.co/datasets/imageomics/TreeOfLife-200M)  
42. BioCLIP 2: Emergent Properties from Scaling Hierarchical Contrastive Learning \- arXiv.org, accessed March 23, 2026, [https://arxiv.org/html/2505.23883v1](https://arxiv.org/html/2505.23883v1)  
43. Add dataset card · imageomics/TreeOfLife-200M at 4c4d6ea \- Hugging Face, accessed March 23, 2026, [https://huggingface.co/datasets/imageomics/TreeOfLife-200M/commit/4c4d6ea0285387c5cb6b78fd2d03a79b21355b32](https://huggingface.co/datasets/imageomics/TreeOfLife-200M/commit/4c4d6ea0285387c5cb6b78fd2d03a79b21355b32)  
44. README.md · imageomics/bioclip-2 at main \- Hugging Face, accessed March 23, 2026, [https://huggingface.co/imageomics/bioclip-2/blob/main/README.md](https://huggingface.co/imageomics/bioclip-2/blob/main/README.md)  
45. Datasets List \- OBIS-SEAMAP, accessed March 23, 2026, [https://seamap.env.duke.edu/dataset/list](https://seamap.env.duke.edu/dataset/list)  
46. wildlife \- Dataset \- Catalog \- Data.gov, accessed March 23, 2026, [https://catalog.data.gov/dataset/?tags=wildlife](https://catalog.data.gov/dataset/?tags=wildlife)  
47. Beluga ID 2022 \- LILA BC, accessed March 23, 2026, [https://lila.science/datasets/beluga-id-2022/](https://lila.science/datasets/beluga-id-2022/)  
48. New Technologies for Monitoring Marine Mammal Health \- PMC, accessed March 23, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC7149946/](https://pmc.ncbi.nlm.nih.gov/articles/PMC7149946/)  
49. Survey of Federally-Funded Marine Mammal Research and Conservation, accessed March 23, 2026, [https://www.mmc.gov/wp-content/uploads/FINAL-MMC-FY15-FFR-BodyAppendicesA-DCovers-2017-07-06.pdf](https://www.mmc.gov/wp-content/uploads/FINAL-MMC-FY15-FFR-BodyAppendicesA-DCovers-2017-07-06.pdf)  
50. Data quality requirements: Occurrence datasets \- GBIF, accessed March 23, 2026, [https://www.gbif.org/data-quality-requirements-occurrences](https://www.gbif.org/data-quality-requirements-occurrences)  
51. Taxonomy Mapping for Camera Trap Datasets \- LILA BC, accessed March 23, 2026, [https://lila.science/taxonomy-mapping-for-camera-trap-data-sets/](https://lila.science/taxonomy-mapping-for-camera-trap-data-sets/)  
52. ShadowWolf – Automatic Labelling, Evaluation and Model Training Optimised for Camera Trap Wildlife Images \- arXiv, accessed March 23, 2026, [https://arxiv.org/html/2512.06521v1](https://arxiv.org/html/2512.06521v1)  
53. MegaDetector is an AI model that helps conservation folks spend less time doing boring things with camera trap images. \- GitHub, accessed March 23, 2026, [https://github.com/agentmorris/MegaDetector](https://github.com/agentmorris/MegaDetector)  
54. Improving and Making Use of iNaturalist Data \- Institute for Natural Resources, accessed March 23, 2026, [https://inr.oregonstate.edu/sites/inr.oregonstate.edu/files/2021-10/inaturalist-qc-export-pnw-citsci-summit-oct-2021.pdf](https://inr.oregonstate.edu/sites/inr.oregonstate.edu/files/2021-10/inaturalist-qc-export-pnw-citsci-summit-oct-2021.pdf)