This document reproduces the complete content of the research paper "To crop or not to crop: Comparing whole-image and cropped classification on a large dataset of camera trap images," published in *IET Computer Vision* in 2024\.

---

**DOI**: 10.1049/cvi2.12318  **Received**: 5 March 2024 | **Revised**: 16 September 2024 | **Accepted**: 11 October 2024

## **To crop or not to crop: Comparing whole-image and cropped classification on a large dataset of camera trap images**

**Authors**: Tomer Gadot, Ştefan Istrate, Hyungwon Kim, Dan Morris, Sara Beery, Tanya Birch, Jorge Ahumada

### **Abstract**

Camera traps facilitate non-invasive wildlife monitoring, but their widespread adoption has created a data processing bottleneck: a camera trap survey can create millions of images, and the labour required to review those images strains the resources of conservation organisations. AI is a promising approach for accelerating image review, but AI tools for camera trap data are imperfect; in particular, classifying small animals remains difficult, and accuracy falls off outside the ecosystems in which a model was trained. It has been proposed that incorporating an object detector into an image analysis pipeline may help address these challenges, but the benefit of object detection has not been systematically evaluated in the literature. In this work, the authors assess the hypothesis that classifying animals cropped from camera trap images using a species-agnostic detector yields better accuracy than classifying whole images. We find that incorporating an object detection stage into an image classification pipeline yields a macro-average F1 improvement of around 25% on a large, long-tailed dataset; this improvement is reproducible on a large public dataset and a smaller public benchmark dataset. The authors describe a classification architecture that performs well for both whole and detector-cropped images, and demonstrate that this architecture yields state-of-the-art benchmark accuracy.

**KEYWORDS**: computer vision, object detection

### ---

**1\. Introduction and Related Work**

Camera traps are static, ground-level cameras used widely across ecology and conservation to monitor species, with more than a million camera traps estimated to be deployed worldwide. Images are collected from camera traps usually based on a heat/motion trigger or timelapse trigger. These images are then processed into scientific observations, such as records of species occurrence. This processing is typically done by human experts, but as cameras became cheaper over the last 30 years, the average number of deployed cameras per project increased, leading to a processing bottleneck, as humans could not keep up with large-scale data streams.

The need for automated computer vision systems to assist in processing camera trap data was established as early as 2018, and many papers have been published in subsequent years demonstrating the potential for AI to help automate species identification. AI has helped researchers address a variety of ecological topics, including the impact of fire on wildlife, beaver activity on river flow, species occupancy, abundance, and faunal biomass. These works have reported time reductions up to 8.4x.

However, challenges remain. Animals that occupy only a small region of interest (due to being small or far from the camera) are harder to classify. AI systems also struggle with geospatial generalizability due to "visual shift" (changing backgrounds/lighting) and "subpopulation shift" (changing relative frequency of species).

One proposed approach is to use a class-agnostic detection model to locate animals with a bounding box as a first stage; these boxes are cropped, and classification is performed on the crops. This allows higher-resolution information to pass to the model and reduces the impact of background context. In this work, the authors demonstrate that incorporating a detection stage improves accuracy relative to whole-image classification alone across several datasets.

### ---

**2\. Methods**

#### **2.1 Data**

The study utilizes three datasets:

1. **WI**: A non-public dataset of \~42M images that is geographically diverse and extremely long-tailed.

2. **LILA**: A public dataset of \~15M images that is less geographically diverse.

3. **iWildCam**: A smaller public benchmark dataset of \~180k images.

Images with multiple labels (approximately 0.4% of the WI dataset) were removed to maintain a conservative evaluation of detector benefits.

**Table 1: Summary of the WI dataset splits.**  | Split | \# of blank images | \# of non-blank images | \# of locations | \# of labels | | :--- | :--- | :--- | :--- | :--- | | Train | 16,477,843 | 19,670,994 | 27,277 | 2176 | | Validation | 1,609,250 | 1,953,246 | 2,978 | 1205 | | Test | 1,048,018 | 1,691,427 | 2,830 | 657 |

**Table 2: Summary of the LILA dataset splits.**  | Split | \# of blank images | \# of non-blank images | \# of locations | \# of labels | | :--- | :--- | :--- | :--- | :--- | | Train | 7,319,982 | 6,202,518 | 5,011 | 773 | | Validation | 635,567 | 151,950 | 267 | 317 | | Test | 346,833 | 148,577 | 298 | 309 |

**Table 3: Summary of the iWildCam dataset splits.**  | Split | \# of blank images | \# of non-blank images | \# of locations | \# of labels | | :--- | :--- | :--- | :--- | :--- | | Train | 47,831 | 64,150 | 243 | 175 | | ID validation | 2,949 | 3,387 | 146 | 69 | | OOD validation| 2,391 | 10,003 | 32 | 75 | | ID test | 2,006 | 6,148 | 164 | 86 | | OOD test | 14,106 | 28,685 | 48 | 101 |

#### **2.2 Data Preparation**

The authors used **MegaDetector (MD) version 5a**, based on the YOLOv5x6 architecture, which accepts images 1280 pixels wide.

To handle label noise (errors in human annotation), the authors created "filtered" variants of the datasets. Images were removed if MD results conflicted significantly with labels (e.g., non-blank labels where MD found nothing with confidence $\\ge 0.4$). All training used the "filtered" datasets.

**Table 4: Impact of data filtering on WI and LILA test sets.**  | Split | \# of blank (unfiltered) | \# of non-blank (unfiltered) | \# of blank (filtered) | \# of non-blank (filtered) | | :--- | :--- | :--- | :--- | :--- | | WI test | 1,067,885 | 2,015,007 | 1,048,018 | 1,691,427 | | LILA test | 350,673 | 178,840 | 346,833 | 148,577 |

#### **2.3 Models**

All classifiers are based on **EfficientNetV2-M**, initialized with ImageNet weights.

* **Whole-image classifier**: Images were cropped vertically (400 pixels or 30% height) to remove manufacturer overlays that could cause spurious correlations, then resized to 480x480.

* **Crop classifier**: Images were cropped to the highest-confidence MD detection ($\\ge 0.01$) and resized to 480x480.

**Ensembles**:

* **Whole-image \+ MD**: Combines whole-image softmax scores with MD detections using manual thresholds to maximize F1 for animal, blank, human, and vehicle classes.

* **Two-classifier ensemble**: Uses both whole-image and crop classifiers to get species predictions, selecting the non-blank prediction if either is confident.

### ---

**3\. Experiments and Metrics**

Four architectures were evaluated for each test set:

1. Whole-image classification alone

2. Whole-image classifier \+ MD

3. Crop classifier

4. Whole-image classifier \+ crop classifier

**Metrics** included Overall weighted F1, Species-level weighted F1 (excluding blank/human/vehicle), Overall macro-averaged F1 (to capture accuracy across the long tail), and Blank F1.

### ---

**4\. Results and Discussion**

#### **4.1 Primary Experiments**

Key takeaways from the results:

1. **Detection helps**: No metric achieved its highest score using a whole-image classifier alone.

2. **Cropping helps**: Crop classification outperformed whole-image ensembles with detectors.

3. **Ensembling benefits are marginal**: Combining whole-image and crop classifiers yielded results very similar to the crop classifier alone.

#### **4.2 Absolute Accuracy**

The whole-image \+ crop classifier achieved state-of-the-art results on the iWildCam leaderboard as of March 2024, with a **Test OOD macro F1 of 54.2**.

**Table 7: Leaderboard Comparison.**  | Model | Rank | Test ID macro F1 | Test OOD macro F1 | | :--- | :--- | :--- | :--- | | **Present work: whole-image \+ crop** | n/a | **61.9** | **54.2** | | \[16\] | 1 | 63.5 | 52.0 | | \[25\] | 2 | 59.9 | 46.0 | | \[57\] | 3 | 57.6 | 43.3 |

#### **4.4 Production Considerations**

The authors explored strategies to improve user experience:

* **"Unknown" predictions**: Predicting "unknown" for low-confidence images increased precision (e.g., blank precision increased from 0.81 to 0.84 on WI).

* **Label Rollup**: Aggregating scores to higher taxonomic levels (e.g., Felidae instead of Lion) when species confidence is low. This reduced the "unknown" rate from 12.1% to 5.7%.

* **Geofencing**: Suppressing species predictions that are not on a regional "allow-list." This increased species-level macro-average precision on WI from 0.44 to 0.46 and reduced jarring errors.

### ---

**5\. Discussion**

The results confirm that incorporating a detector and cropping images provides significant benefits. Detectors handle multi-species images naturally and minimize risk from background or overlay correlations. However, whole-image classifiers remain useful for efficiency, as detectors are larger and slower (MD runs at roughly half the speed of the classifier).

### ---

**6\. Conclusion and Future Work**

While beneficial, the exact scenarios where detection helps (e.g., small animals) require further study.  The community is encouraged to:

* Contribute bounding-box annotations to shared repositories.

* Develop standardized metrics for multi-taxonomic-level labels.

* Conduct human factors experiments to assess real-world efficiency gains.

### ---

**Acknowledgements**

The authors acknowledge Partners of Wildlife Insights (Roland Kays, William McShea, Walter Jetz, Jonathan Palmer, Maggie Kinnaird, Alison Swanson, Nicole Flores, Anthony Dancer, Martin Wikelski, Chrissy Durkin) and sponsors (Lyda Hill Foundation, Gordon and Betty Moore Foundation, and Google).

Would you like me to detail the specific data filtering thresholds or the taxonomic rollup logic used in the production experiments?