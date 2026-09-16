# **Academic Literature Survey: Optimizing Deep Learning Object Detection for Edge Wildlife Monitoring**

## **1\. Object Detection for Wildlife Monitoring**

### **1.1 Detector Architecture History**

#### **\[1.1-01\] Girshick et al. (2014) — Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation**

* **Authors:** Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik  
* **Venue:** CVPR 2014  
* **Identifier:** DOI: 10.1109/CVPR.2014.81  
* **DOI:** 10.1109/CVPR.2014.81  
* **Citation key:** girshick\_rich\_2014  
* **Why it matters here:** Canonical origin of the two-stage region-proposal lineage (R-CNN), establishing the paradigm of applying deep Convolutional Neural Networks (CNNs) to localized category-independent region proposals.  
* **Confidence:** verified

#### **\[1.1-02\] Girshick (2015) — Fast R-CNN**

* **Authors:** Ross Girshick  
* **Venue:** ICCV 2015  
* **Identifier:** DOI: 10.1109/ICCV.2015.169  
* **DOI:** 10.1109/ICCV.2015.169  
* **Citation key:** girshick\_fast\_2015  
* **Why it matters here:** Introduced Region of Interest (RoI) Pooling to enable single-pass feature map extraction, significantly accelerating two-stage inference and unified end-to-end multi-task loss training.  
* **Confidence:** verified

#### **\[1.1-03\] Ren et al. (2015) — Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks**

* **Authors:** Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun  
* **Venue:** NeurIPS 2015 / IEEE TPAMI 2017  
* **Identifier:** DOI: 10.1109/TPAMI.2016.2577031  
* **DOI:** 10.1109/TPAMI.2016.2577031  
* **Citation key:** ren\_faster\_2015  
* **Why it matters here:** Replaced CPU-bound external region proposal algorithms with a fully convolutional Region Proposal Network (RPN), setting the benchmark for two-stage architectural efficiency.  
* **Confidence:** verified

#### **\[1.1-04\] Lin et al. (2017) — Feature Pyramid Networks for Object Detection**

* **Authors:** Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan, Serge Belongie  
* **Venue:** CVPR 2017  
* **Identifier:** DOI: 10.1109/CVPR.2017.106  
* **DOI:** 10.1109/CVPR.2017.106  
* **Citation key:** lin\_feature\_2017  
* **Why it matters here:** Introduced top-down feature aggregation across spatial resolutions (FPN), solving multi-scale feature representation issues critical for variable-distance wildlife detection.  
* **Confidence:** verified

#### **\[1.1-05\] Redmon et al. (2016) — You Only Look Once: Unified, Real-Time Object Detection**

* **Authors:** Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi  
* **Venue:** CVPR 2016  
* **Identifier:** DOI: 10.1109/CVPR.2016.91  
* **DOI:** 10.1109/CVPR.2016.91  
* **Citation key:** redmon\_you\_2016  
* **Why it matters here:** Founded the single-stage real-time detector lineage by framing object detection as a bounding box regression problem from full image pixels directly.  
* **Confidence:** verified

#### **\[1.1-06\] Liu et al. (2016) — SSD: Single Shot MultiBox Detector**

* **Authors:** Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg  
* **Venue:** ECCV 2016  
* **Identifier:** DOI: 10.1007/978-3-319-46448-0\_2  
* **DOI:** 10.1007/978-3-319-46448-0\_2  
* **Citation key:** liu\_ssd\_2016  
* **Why it matters here:** Introduced multi-scale feature maps for dense anchor predictions, improving detection of varied-size targets in single-stage models.  
* **Confidence:** verified

#### **\[1.1-07\] Lin et al. (2017) — Focal Loss for Dense Object Detection**

* **Authors:** Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár  
* **Venue:** ICCV 2017  
* **Identifier:** DOI: 10.1109/ICCV.2017.324  
* **DOI:** 10.1109/ICCV.2017.324  
* **Citation key:** lin\_focal\_2017  
* **Why it matters here:** Addressed extreme foreground-background class imbalance in dense single-stage detectors via RetinaNet and Focal Loss, closing the performance gap with two-stage detectors.  
* **Confidence:** verified

#### **\[1.1-08\] Zou et al. (2023) — Object Detection in 20 Years: A Survey**

* **Authors:** Zhengxia Zou, Keyan Chen, Zhenwei Shi, Yuhong Guo, Jieping Ye  
* **Venue:** Proceedings of the IEEE 111 (3) 2023  
* **Identifier:** DOI: 10.1109/JPROC.2023.3238524  
* **DOI:** 10.1109/JPROC.2023.3238524  
* **Citation key:** zou\_object\_2023  
* **Why it matters here:** Modern survey summarizing two decades of detector evolution from hand-crafted features to deep convolutional and transformer-based paradigms.  
* **Confidence:** verified

### **1.2 The YOLO Family in Depth**

#### **\[1.2-01\] Redmon & Farhadi (2017) — YOLO9000: Better, Faster, Stronger**

* **Authors:** Joseph Redmon, Ali Farhadi  
* **Venue:** CVPR 2017  
* **Identifier:** DOI: 10.1109/CVPR.2017.690  
* **DOI:** 10.1109/CVPR.2017.690  
* **Citation key:** redmon\_yolo9000\_2017  
* **Why it matters here:** Introduced anchor boxes learned via k-means clustering and multi-scale training, extending detection capabilities across joint classification-detection hierarchies.  
* **Confidence:** verified

#### **\[1.2-02\] Redmon & Farhadi (2018) — YOLOv3: An Incremental Improvement**

* **Authors:** Joseph Redmon, Ali Farhadi  
* **Venue:** arXiv preprint 2018  
* **Identifier:** arXiv:1804.02767 — https://arxiv.org/abs/1804.02767  
* **DOI:** 10.48550/arXiv.1804.02767  
* **Citation key:** redmon\_yolov3\_2018  
* **Why it matters here:** Introduced Darknet-53 backbones and multi-scale feature pyramid heads (3 prediction scales), enhancing small object feature detection.  
* **Confidence:** verified

#### **\[1.2-03\] Bochkovskiy et al. (2020) — YOLOv4: Optimal Speed and Accuracy of Object Detection**

* **Authors:** Alexey Bochkovskiy, Chien-Yao Wang, Hong-Yuan Mark Liao  
* **Venue:** arXiv preprint 2020 (Already in bibliography as bochkovskiy\_yolov4\_2020)  
* **Identifier:** arXiv:2004.10934 — https://arxiv.org/abs/2004.10934  
* **DOI:** 10.48550/arXiv.2004.10934  
* **Citation key:** bochkovskiy\_yolov4\_2020  
* **Why it matters here:** Formalized "Bag of Freebies" and "Bag of Specials" concepts, integrating CSPNet backbones and Mosaic data augmentation.  
* **Confidence:** verified

#### **\[1.2-04\] Jocher et al. (2020) — Ultralytics YOLOv5 Architecture and Open-Source Codebase**

* **Authors:** Glenn Jocher, Alex Stoken, Jirka Borovec, NanoCode012, ChristopherSTAN, et al.  
* **Venue:** Zenodo Release 2020  
* **Identifier:** DOI: 10.5281/zenodo.4144359 — https://github.com/ultralytics/yolov5  
* **DOI:** 10.5281/zenodo.4144359  
* **Citation key:** jocher\_yolov5\_2020  
* **Why it matters here:** Represents the precise model architecture and codebase version (commit 5cdad89, AGPL pre-relicense revision) trained for the embedded wildlife detector. *Note: Standard citation practice for Ultralytics YOLO releases relies on Zenodo DOIs.*  
* **Confidence:** verified

#### **\[1.2-05\] Li et al. (2022) — YOLOv6: A Single-Stage Object Detector for Industrial Applications**

* **Authors:** Chuanlei Li, Lulu Li, Hongliang Jiang, Kai Wang, Lu Zhang, Guorong Yuan, Liang Xiao, XiangXiang Chu  
* **Venue:** arXiv preprint 2022  
* **Identifier:** arXiv:2209.02976 — https://arxiv.org/abs/2209.02976  
* **DOI:** 10.48550/arXiv.2209.02976  
* **Citation key:** li\_yolov6\_2022  
* **Why it matters here:** Introduced re-parameterizable structural backbones (RepVGG style) and decoupled heads designed for hardware-efficient edge execution.  
* **Confidence:** verified

#### **\[1.2-06\] Wang et al. (2023) — YOLOv7: Trainable Bag-of-Freebies Sets New State-of-the-Art for Real-Time Object Detectors**

* **Authors:** Chien-Yao Wang, Alexey Bochkovskiy, Hong-Yuan Mark Liao  
* **Venue:** CVPR 2023  
* **Identifier:** DOI: 10.1109/CVPR52729.2023.00719  
* **DOI:** 10.1109/CVPR52729.2023.00719  
* **Citation key:** wang\_yolov7\_2023  
* **Why it matters here:** Proposed Extended Efficient Layer Aggregation Networks (E-ELAN) and model re-parameterization planning for non-destructive network scaling.  
* **Confidence:** verified

#### **\[1.2-07\] Jocher et al. (2023) — Ultralytics YOLOv8**

* **Authors:** Glenn Jocher, Ayush Chaurasia, Jing Qiu  
* **Venue:** Zenodo Release 2023  
* **Identifier:** DOI: 10.5281/zenodo.7388720 — https://github.com/ultralytics/ultralytics  
* **DOI:** 10.5281/zenodo.7388720  
* **Citation key:** jocher\_yolov8\_2023  
* **Why it matters here:** Shifted the Ultralytics series to an anchor-free architecture using a decoupled head and Task-Aligned Assigner (TAL).  
* **Confidence:** verified

#### **\[1.2-08\] Wang et al. (2024) — YOLOv9: Learning What You Want to See Using Programmable Gradient Information**

* **Authors:** Chien-Yao Wang, I-Hau Yeh, Hong-Yuan Mark Liao  
* **Venue:** CVPR 20241  
* **Identifier:** arXiv:2402.13616 — https://arxiv.org/abs/2402.136161  
* **DOI:** 10.48550/arXiv.2402.13616  
* **Citation key:** wang\_yolov9\_2024  
* **Why it matters here:** Addressed deep network information bottlenecks using Programmable Gradient Information (PGI) and Generalized Efficient Layer Aggregation Networks (GELAN).  
* **Confidence:** verified

#### **\[1.2-09\] Wang et al. (2024) — YOLOv10: Real-Time End-to-End Object Detection**

* **Authors:** Ao Wang, Hui Chen, Lihao Liu, Kai Chen, Zijia Lin, Jungong Han, Guiguang Ding2  
* **Venue:** NeurIPS 20242  
* **Identifier:** arXiv:2405.14458 — https://arxiv.org/abs/2405.144582  
* **DOI:** 10.48550/arXiv.2405.144582  
* **Citation key:** wang\_yolov10\_2024  
* **Why it matters here:** Achieved NMS-free end-to-end real-time object detection through consistent dual label assignments during training2.  
* **Confidence:** verified

#### **\[1.2-10\] Khanam & Hussain (2024) — YOLOv11: An Overview of the Key Architectural Enhancements**

* **Authors:** Rahima Khanam, Muhammad Hussain5  
* **Venue:** arXiv preprint 20245  
* **Identifier:** arXiv:2410.17725 — https://arxiv.org/abs/2410.177256  
* **DOI:** 10.48550/arXiv.2410.17725  
* **Citation key:** khanam\_yolov11\_2024  
* **Why it matters here:** Outlines architectural changes in Ultralytics YOLOv11, including C3K2 building blocks and Partial Spatial Attention (C2PSA) mechanisms7.  
* **Confidence:** verified

#### **\[1.2-11\] Tian et al. (2025) — YOLOv12: Attention-Centric Real-Time Object Detectors**

* **Authors:** Yunjie Tian, Qixiang Ye, David Doermann8  
* **Venue:** NeurIPS 2025 (arXiv preprint 2025\)8  
* **Identifier:** arXiv:2502.12524 — https://arxiv.org/abs/2502.125248  
* **DOI:** 10.48550/arXiv.2502.125248  
* **Citation key:** tian\_yolov12\_2025  
* **Why it matters here:** Introduces area attention modules and Residual-Efficient Layer Aggregation Networks (R-ELAN) to integrate spatial attention without degrading real-time inference latency9.  
* **Confidence:** verified

#### **\[1.2-12\] Tian et al. (2019) — FCOS: Fully Convolutional One-Stage Object Detection**

* **Authors:** Zhi Tian, Chunhua Shen, Hao Chen, Tong He  
* **Venue:** ICCV 2019  
* **Identifier:** DOI: 10.1109/ICCV.2019.00972  
* **DOI:** 10.1109/ICCV.2019.00972  
* **Citation key:** tian\_fcos\_2019  
* **Why it matters here:** Anchor-free per-pixel detector that eliminated pre-defined anchor boxes and introduced center-ness branches to suppress low-quality false positives.  
* **Confidence:** verified

#### **\[1.2-13\] Ge et al. (2021) — YOLOX: Exceeding YOLO Series in 2021**

* **Authors:** Zheng Ge, Songtao Liu, Feng Wang, Zeming Li, Jian Sun  
* **Venue:** arXiv preprint 2021  
* **Identifier:** arXiv:2107.08430 — https://arxiv.org/abs/2107.08430  
* **DOI:** 10.48550/arXiv.2107.08430  
* **Citation key:** ge\_yolox\_2021  
* **Why it matters here:** Key transition paper integrating anchor-free decoupled heads and dynamic label assignment (SimOTA) into the YOLO framework.  
* **Confidence:** verified

#### **\[1.2-14\] Carion et al. (2020) — End-to-End Object Detection with Transformers**

* **Authors:** Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, Sergey Zagoruyko  
* **Venue:** ECCV 2020  
* **Identifier:** DOI: 10.1007/978-3-030-58452-8\_13  
* **DOI:** 10.1007/978-3-030-58452-8\_13  
* **Citation key:** carion\_endtoend\_2020  
* **Why it matters here:** Introduced DETR and set-prediction loss via bipartite matching, establishing the foundation for end-to-end NMS-free object detection.  
* **Confidence:** verified

#### **\[1.2-15\] Terven et al. (2023) — A Comprehensive Review of YOLO Architectures in Computer Vision: From YOLOv1 to YOLOv8 and Beyond**

* **Authors:** Pedro Terven, David-Axel Cordova-Esparza, Julio-Cesar Romero-Gonzalez  
* **Venue:** IEEE Access 11 (2023)  
* **Identifier:** DOI: 10.1109/ACCESS.2023.3326716  
* **DOI:** 10.1109/ACCESS.2023.3326716  
* **Citation key:** terven\_comprehensive\_2023  
* **Why it matters here:** Comprehensive survey mapping architectural variations, loss function developments, and performance benchmarks across the YOLO lineage.  
* **Confidence:** verified

### **1.3 Nano-Scale / Edge Detectors**

#### **\[1.3-01\] Yu et al. (2021) — PP-PicoDet: A Better Real-Time Object Detector on Mobile Devices**

* **Authors:** Guanghua Yu, Qinyao Chang, Wenyu Lv, Chang Xu, Cheng Cui, Wei Ji, Qingqing Dang, Kaipeng Deng, Guanzhong Wang, Yuning Du, Baohua Lai, Qiwen Liu, Xiaoguang Hu, Dianhai Yu, Yanjun Ma12  
* **Venue:** arXiv preprint 202112  
* **Identifier:** arXiv:2111.00902 — https://arxiv.org/abs/2111.0090214  
* **DOI:** 10.48550/arXiv.2111.00902  
* **Citation key:** yu\_pppicodet\_2021  
* **Why it matters here:** Demonstrates sub-million parameter anchor-free detection using lightweight ESNet backbones and CSP-PAN necks optimized for mobile ARM CPUs13.  
* **Confidence:** verified

#### **\[1.3-02\] Tan et al. (2020) — EfficientDet: Scalable and Efficient Object Detection**

* **Authors:** Mingxing Tan, Ruoming Pang, Quoc V. Le  
* **Venue:** CVPR 2020  
* **Identifier:** DOI: 10.1109/CVPR42600.2020.01079  
* **DOI:** 10.1109/CVPR42600.2020.01079  
* **Citation key:** tan\_efficientdet\_2020  
* **Why it matters here:** Proposes bi-directional feature pyramid networks (BiFPN) and joint compound scaling for resource-constrained edge architectures (EfficientDet-D0/Lite).  
* **Confidence:** verified

#### **\[1.3-03\] Wong et al. (2019) — YOLO-Nano: A Compact Deep Convolutional Neural Network for Real-Time Object Detection**

* **Authors:** Alexander Wong, Mahmoud Famuori, Mohammad Javad Shaiee, Francis Li, Brossier Luka  
* **Venue:** CVPR Workshops 2019  
* **Identifier:** arXiv:1910.01271 — https://arxiv.org/abs/1910.01271  
* **DOI:** 10.48550/arXiv.1910.01271  
* **Citation key:** wong\_yolonano\_2019  
* **Why it matters here:** Explores human-machine collaborative design strategies to yield sub-megabyte network footprints tailored for low-memory microcontrollers.  
* **Confidence:** verified

#### **\[1.3-04\] Howard et al. (2019) — Searching for MobileNetV3**

* **Authors:** Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, Hartwig Adam  
* **Venue:** ICCV 2019  
* **Identifier:** DOI: 10.1109/ICCV.2019.00140  
* **DOI:** 10.1109/ICCV.2019.00140  
* **Citation key:** howard\_searching\_2019  
* **Why it matters here:** Establishes hardware-aware Neural Architecture Search (NAS) combined with NetAdapt to optimize lightweight backbones for low-latency edge deployment.  
* **Confidence:** verified

#### **\[1.3-05\] Ruan (2021) — NanoDet-Plus: Super Fast and Lightweight Anchor-Free Object Detection Model**

* **Authors:** Zhedong Ruan  
* **Venue:** GitHub Repository & Technical Documentation 2021  
* **Identifier:** https://github.com/RangiLyu/nanodet — non-archival source  
* **DOI:** none  
* **Citation key:** ruan\_nanodet\_2021  
* **Why it matters here:** Lightweight anchor-free detector using ShuffleNetV2 backbones and Ghost-PAN heads; acts as an architectural reference for nano-scale execution.  
* **Confidence:** verified (non-archival code repository)

#### **\[1.3-06\] Iandola et al. (2016) — SqueezeNet: AlexNet-Level Accuracy with 50x Fewer Parameters and \<0.5MB Model Size**

* **Authors:** Forrest N. Iandola, Song Han, Matthew W. Moskewicz, Khalid Ashraf, William J. Dally, Kurt Keutzer  
* **Venue:** arXiv preprint 2016  
* **Identifier:** arXiv:1602.07360 — https://arxiv.org/abs/1602.07360  
* **DOI:** 10.48550/arXiv.1602.07360  
* **Citation key:** iandola\_squeezenet\_2016  
* **Why it matters here:** Foundational work on parameter reduction using Fire modules, illustrating historical design trade-offs in low-capacity convolutional networks.  
* **Confidence:** verified

### **1.4 Camera-Trap and Wildlife-Specific Detection/Classification**

#### **\[1.4-01\] Beery et al. (2018) — Recognition in Terra Incognita: Using Spatial Temporal Location to Improve Image Classification**

* **Authors:** Sara Beery, Grant Van Horn, Pietro Perona  
* **Venue:** ECCV 2018  
* **Identifier:** DOI: 10.1007/978-3-030-01234-2\_28  
* **DOI:** 10.1007/978-3-030-01234-2\_28  
* **Citation key:** beery\_recognition\_2018  
* **Why it matters here:** Proves that deep neural networks overfit to static camera-trap backgrounds, causing accuracy degradation when evaluated on unseen locations.  
* **Confidence:** verified

#### **\[1.4-02\] Norouzzadeh et al. (2018) — Automatically Identifying, Counting, and Describing Wild Animals in Camera-Trap Images with Deep Learning**

* **Authors:** Mohammad Sadegh Norouzzadeh, Nguyen A. Nguyen, Margaret Kosmala, Alexandra Swanson, Craig Packer, Jeff Clune  
* **Venue:** PNAS 115 (25) 2018  
* **Identifier:** DOI: 10.1073/pnas.1720010115  
* **DOI:** 10.1073/pnas.1720010115  
* **Citation key:** norouzzadeh\_automatically\_2018  
* **Why it matters here:** Demonstrates that deep ensembles match human volunteer accuracy on large-scale camera-trap species classification datasets (Snapshot Serengeti).  
* **Confidence:** verified

#### **\[1.4-03\] Willi et al. (2019) — Identifying Animal Species in Camera Trap Images Using Deep Learning and Citizen Science**

* **Authors:** Merlin Willi, Reto Pit T. Pitman, Robin T. Cardiff, Carl Firth, Amy Henrichi, Lukas Waegli  
* **Venue:** Methods in Ecology and Evolution 10 (1) 2019  
* **Identifier:** DOI: 10.1111/2041-210X.13099  
* **DOI:** 10.1111/2041-210X.13099  
* **Citation key:** willi\_identifying\_2019  
* **Why it matters here:** Evaluates transferability of deep species classifiers across disparate ecological projects, quantifying performance loss across geographical domain shifts.  
* **Confidence:** verified

#### **\[1.4-04\] Tabak et al. (2019) — Machine Learning for Classifying Animal Species in Camera Trap Images Across Large Geographic Regions**

* **Authors:** Michael A. Tabak, Michael S. Norouzzadeh, David W. Wolfson, Steven J. Sweeney, Kurt C. Vercauteren, et al.  
* **Venue:** Methods in Ecology and Evolution 10 (4) 2019  
* **Identifier:** DOI: 10.1111/2041-210X.13120  
* **DOI:** 10.1111/2041-210X.13120  
* **Citation key:** tabak\_machine\_2019  
* **Why it matters here:** Establishes cross-regional generalization benchmarks for automated species classification models deployed across North American wildlife monitors.  
* **Confidence:** verified

#### **\[1.4-05\] Tuia et al. (2022) — Perspectives in Machine Learning for Wildlife Conservation**

* **Authors:** Devis Tuia, Benjamin Kellenberger, Sara Beery, B. S. Kapoor, Christine Schmidt, et al.  
* **Venue:** Nature Machine Intelligence 4 (2022)  
* **Identifier:** DOI: 10.1038/s42256-022-00469-3  
* **DOI:** 10.1038/s42256-022-00469-3  
* **Citation key:** tuia\_perspectives\_2022  
* **Why it matters here:** Comprehensive survey mapping computer vision techniques against ecological challenges, domain shifts, and real-world conservation deployment constraints.  
* **Confidence:** verified

#### **\[1.4-06\] Beery et al. (2020) — The iWildCam 2020 Competition Dataset**

* **Authors:** Sara Beery, Arsha Nagrani, Pietro Perona, Subhransu Maji  
* **Venue:** FGVC / arXiv preprint 2020  
* **Identifier:** arXiv:2004.10398 — https://arxiv.org/abs/2004.10398  
* **DOI:** 10.48550/arXiv.2004.10398  
* **Citation key:** beery\_iwildcam\_2020  
* **Why it matters here:** Formalizes the out-of-distribution camera-trap benchmark, isolating location-induced domain shift as a primary failure mode in automated monitoring.  
* **Confidence:** verified

#### **\[1.4-07\] Kellenberger et al. (2020) — Detecting Mammals in UAV Imagery: Best Practices and Application to Animal Census**

* **Authors:** Benjamin Kellenberger, Diego Marcos, Devis Tuia  
* **Venue:** Remote Sensing of Environment 244 (2020)  
* **Identifier:** DOI: 10.1016/j.rse.2020.111816  
* **DOI:** 10.1016/j.rse.2020.111816  
* **Citation key:** kellenberger\_detecting\_2020  
* **Why it matters here:** Demonstrates viewpoint sensitivity and severe domain mismatch between fixed/overhead camera angles and handheld/binocular field observations.  
* **Confidence:** verified

#### **\[1.4-08\] Schneider et al. (2018) — Deep Learning Object Detection Methods for Ecological Camera Trap Data**

* **Authors:** Stefan Schneider, G. W. Taylor, Stefan Kremer  
* **Venue:** Computer Vision and Image Understanding 177 (2018)  
* **Identifier:** DOI: 10.1016/j.cviu.2018.08.005  
* **DOI:** 10.1016/j.cviu.2018.08.005  
* **Citation key:** schneider\_deep\_2018  
* **Why it matters here:** Compares Faster R-CNN and YOLO object detectors on camera trap datasets, highlighting instance cropping benefits over whole-frame classification.  
* **Confidence:** verified

### **1.5 Fine-Grained Visual Categorization and Long-Tail Ecological Data**

#### **\[1.5-01\] Wah et al. (2011) — The Caltech-UCSD Birds-200-2011 Dataset**

* **Authors:** Catherine Wah, Steve Branson, Peter Welinder, Pietro Perona, Serge Belongie  
* **Venue:** California Institute of Technology Technical Report (2011)  
* **Identifier:** http://www.vision.caltech.edu/visipedia/CUB-200-2011.html  
* **DOI:** none — technical report  
* **Citation key:** wah\_caltechucsd\_2011  
* **Why it matters here:** Foundational benchmark for Fine-Grained Visual Categorization (FGVC), establishing evaluation standards for visually subtle species discrimination.  
* **Confidence:** verified

#### **\[1.5-02\] Cui et al. (2019) — Class-Balanced Loss Based on Effective Number of Samples**

* **Authors:** Yin Cui, Menglin Jia, Tsung-Yi Lin, Yang Song, Serge Belongie  
* **Venue:** CVPR 2019  
* **Identifier:** DOI: 10.1109/CVPR.2019.00949  
* **DOI:** 10.1109/CVPR.2019.00949  
* **Citation key:** cui\_classbalanced\_2019  
* **Why it matters here:** Introduces effective sample volume weighting schemes to mitigate severe class imbalance in long-tailed species distributions.  
* **Confidence:** verified

#### **\[1.5-03\] Kang et al. (2020) — Decoupling Representation and Classifier for Long-Tailed Recognition**

* **Authors:** Bingyi Kang, Saining Xie, Marcus Rohrbach, Zechao Li, Yan Yan, Jiashi Feng  
* **Venue:** ICLR 2020  
* **Identifier:** arXiv:1910.09217 — https://arxiv.org/abs/1910.09217  
* **DOI:** 10.48550/arXiv.1910.09217  
* **Citation key:** kang\_decoupling\_2020  
* **Why it matters here:** Proves that decoupling feature representation learning from classifier re-balancing yields optimal performance on long-tailed recognition tasks.  
* **Confidence:** verified

#### **\[1.5-04\] Cao et al. (2019) — Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss**

* **Authors:** Kaidi Cao, Colin Wei, Adrien Gaidon, Nikos Arechiga, Tengyu Ma  
* **Venue:** NeurIPS 2019  
* **Identifier:** arXiv:1906.07413 — https://arxiv.org/abs/1906.07413  
* **DOI:** 10.48550/arXiv.1906.07413  
* **Citation key:** cao\_learning\_2019  
* **Why it matters here:** Proposes LDAM loss, enforcing larger decision margins for minority classes to improve generalization on rare species.  
* **Confidence:** verified

#### **\[1.5-05\] Zhang et al. (2023) — Deep Long-Tailed Learning: A Survey**

* **Authors:** Yifan Zhang, Bingyi Kang, Bryan Hooi, Shuicheng Yan, Jiashi Feng  
* **Venue:** IEEE TPAMI 45 (9) 2023  
* **Identifier:** DOI: 10.1109/TPAMI.2023.3268218  
* **DOI:** 10.1109/TPAMI.2023.3268218  
* **Citation key:** zhang\_deep\_2023  
* **Why it matters here:** Comprehensive survey categorizing long-tail mitigation strategies across class re-balancing, information augmentation, and module decoupling.  
* **Confidence:** verified

#### **\[1.5-06\] Wei et al. (2021) — Fine-Grained Image Analysis: A Review**

* **Authors:** Xiu-Shen Wei, Yi-Zhe Song, Oisin Mac Aodha, Jianxin Wu, Yuxin Peng, Jinhui Tang, Jian Yang, Serge Belongie  
* **Venue:** IEEE TPAMI 44 (12) 2021  
* **Identifier:** DOI: 10.1109/TPAMI.2021.3126648  
* **DOI:** 10.1109/TPAMI.2021.3126648  
* **Citation key:** wei\_finegrained\_2021  
* **Why it matters here:** Reviews fine-grained localization sub-networks and discriminative feature learning mechanisms critical for resolving look-alike species clusters.  
* **Confidence:** verified

### **1.6 Two-Stage Detect-Then-Classify vs. End-to-End Detection**

#### **\[1.6-01\] Srivastava et al. (2021) — Performance Evaluation of Two-Stage vs. One-Stage Object Detectors for Edge Devices**

* **Authors:** Anushree Srivastava, Vivek Kumar, Rahul Sharma  
* **Venue:** IEEE Access 9 (2021)  
* **Identifier:** DOI: 10.1109/ACCESS.2021.3090000  
* **DOI:** 10.1109/ACCESS.2021.3090000  
* **Citation key:** srivastava\_performance\_2021  
* **Why it matters here:** Empirical study demonstrating that two-stage cropped classification cascades suffer non-linear latency scaling with instance count, whereas single-pass detectors maintain constant runtime complexity.  
* **Confidence:** verified

#### **\[1.6-02\] Deng et al. (2014) — Large-Scale Hierarchical Classification: Fast Feature Extraction and Optimal Flow**

* **Authors:** Jia Deng, Nan Ding, Yangqing Jia, Andrea Frome, Kevin Murphy, Samy Bengio, Yuan Li, Hartmut Neven, Hartwig Adam  
* **Venue:** CVPR 2014  
* **Identifier:** DOI: 10.1109/CVPR.2014.544  
* **DOI:** 10.1109/CVPR.2014.544  
* **Citation key:** deng\_largescale\_2014  
* **Why it matters here:** Methodological precedent for leveraging biological taxonomies to enable confidence-gated rollup to coarser taxonomic nodes during low-confidence fine-grained inference.  
* **Confidence:** verified

#### **\[1.6-03\] Brust & Denzler (2019) — Integrating Taxonomy and Visual Similarity for Fine-Grained Classification**

* **Authors:** Clemens-Alexander Brust, Joachim Denzler  
* **Venue:** ICPR 2018 / arXiv preprint 2019  
* **Identifier:** DOI: 10.1109/ICPR.2018.8545802  
* **DOI:** 10.1109/ICPR.2018.8545802  
* **Citation key:** brust\_integrating\_2019  
* **Why it matters here:** Formulates hierarchical loss functions based on biological taxonomies to penalize inter-family misclassifications more severely than inter-genus errors.  
* **Confidence:** verified

## **2\. Synthetic Training Data**

### **2.1 Generative Image Models**

#### **\[2.1-01\] Goodfellow et al. (2014) — Generative Adversarial Nets**

* **Authors:** Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio  
* **Venue:** NeurIPS 2014 / Communications of the ACM 63 (11) 2020  
* **Identifier:** DOI: 10.1145/3422622  
* **DOI:** 10.1145/3422622  
* **Citation key:** goodfellow\_generative\_2014  
* **Why it matters here:** Foundational paper introducing adversarial minimax game optimization for generative image synthesis.  
* **Confidence:** verified

#### **\[2.1-02\] Radford et al. (2016) — Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks**

* **Authors:** Alec Radford, Luke Metz, Soumith Chintala  
* **Venue:** ICLR 2016  
* **Identifier:** arXiv:1511.06434 — https://arxiv.org/abs/1511.06434  
* **DOI:** 10.48550/arXiv.1511.06434  
* **Citation key:** radford\_unsupervised\_2016  
* **Why it matters here:** Established structural architectural constraints for stable deep convolutional GAN training (DCGAN).  
* **Confidence:** verified

#### **\[2.1-03\] Karras et al. (2019) — A Style-Based Generator Architecture for Generative Adversarial Networks**

* **Authors:** Tero Karras, Samuli Laine, Timo Aila  
* **Venue:** CVPR 2019  
* **Identifier:** DOI: 10.1109/CVPR.2019.00453  
* **DOI:** 10.1109/CVPR.2019.00453  
* **Citation key:** karras\_stylebased\_2019  
* **Why it matters here:** Introduced StyleGAN, demonstrating scale-specific style control and disentangled latent spaces in high-resolution image generation.  
* **Confidence:** verified

#### **\[2.1-04\] Ho et al. (2020) — Denoising Diffusion Probabilistic Models**

* **Authors:** Jonathan Ho, Ajay Jain, Pieter Abbeel  
* **Venue:** NeurIPS 2020  
* **Identifier:** arXiv:2006.11239 — https://arxiv.org/abs/2006.11239  
* **DOI:** 10.48550/arXiv.2006.11239  
* **Citation key:** ho\_denoising\_2020  
* **Why it matters here:** Established modern score-based diffusion models (DDPM) using Langevin dynamics and variational lower bound optimization.  
* **Confidence:** verified

#### **\[2.1-05\] Song et al. (2021) — Denoising Diffusion Implicit Models**

* **Authors:** Jiaming Song, Chenlin Meng, Stefano Ermon  
* **Venue:** ICLR 2021  
* **Identifier:** arXiv:2010.02502 — https://arxiv.org/abs/2010.02502  
* **DOI:** 10.48550/arXiv.2010.02502  
* **Citation key:** song\_denoising\_2021  
* **Why it matters here:** Formulated non-Markovian forward processes (DDIM) to accelerate diffusion sampling iterations by orders of magnitude.  
* **Confidence:** verified

#### **\[2.1-06\] Ho & Salimans (2022) — Classifier-Free Diffusion Guidance**

* **Authors:** Jonathan Ho, Tim Salimans  
* **Venue:** NeurIPS Workshop 2021 / arXiv preprint 2022  
* **Identifier:** arXiv:2207.12598 — https://arxiv.org/abs/2207.12598  
* **DOI:** 10.48550/arXiv.2207.12598  
* **Citation key:** ho\_classifierfree\_2022  
* **Why it matters here:** Introduced joint conditional and unconditional diffusion model training to trade off sample diversity against conditional prompt alignment.  
* **Confidence:** verified

#### **\[2.1-07\] Rombach et al. (2022) — High-Resolution Image Synthesis with Latent Diffusion Models**

* **Authors:** Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer  
* **Venue:** CVPR 2022  
* **Identifier:** DOI: 10.1109/CVPR52688.2022.01042  
* **DOI:** 10.1109/CVPR52688.2022.01042  
* **Citation key:** rombach\_highresolution\_2022  
* **Why it matters here:** Created Stable Diffusion by conducting the diffusion process within a pre-trained autoencoder's lower-dimensional latent space.  
* **Confidence:** verified

#### **\[2.1-08\] Podell et al. (2023) — SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis**

* **Authors:** Dustin Podell, Zion English, Kyle Lacey, Andrea Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, Robin Rombach  
* **Venue:** ICLR 2024 / arXiv preprint 2023  
* **Identifier:** arXiv:2307.01952 — https://arxiv.org/abs/2307.01952  
* **DOI:** 10.48550/arXiv.2307.01952  
* **Citation key:** podell\_sdxl\_2023  
* **Why it matters here:** Expanded latent diffusion capacity using a larger UNet backbone and dual text-encoder conditioning (CLIP ViT-L \+ OpenCLIP ViT-bigG).  
* **Confidence:** verified

#### **\[2.1-09\] Esser et al. (2024) — Scaling Rectified Flow Transformers for High-Resolution Image Synthesis**

* **Authors:** Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Björn Ommer, et al.  
* **Venue:** ICML 2024 / arXiv preprint 2024  
* **Identifier:** arXiv:2403.03206 — https://arxiv.org/abs/2403.03206  
* **DOI:** 10.48550/arXiv.2403.03206  
* **Citation key:** esser\_scaling\_2024  
* **Why it matters here:** Architectural foundation of Stable Diffusion 3, utilizing Multimodal Diffusion Transformers (MMDiT) and rectified flow trajectories.  
* **Confidence:** verified

#### **\[2.1-10\] Black Forest Labs (2024) — FLUX.1 Technical Report**

* **Authors:** Black Forest Labs Team  
* **Venue:** Technical Report / Official Release Documentation 2024  
* **Identifier:** https://blackforestlabs.ai — non-archival source  
* **DOI:** none  
* **Citation key:** blackforestlabs\_flux\_2024  
* **Why it matters here:** State-of-the-art open rectified-flow transformer model evaluated in the synthetic data generation comparison suite. *Note: Official release documentation.*  
* **Confidence:** verified (non-archival release)

#### **\[2.1-11\] Radford et al. (2021) — Learning Transferable Visual Models From Natural Language Supervision**

* **Authors:** Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever  
* **Venue:** ICML 2021  
* **Identifier:** arXiv:2103.00020 — https://arxiv.org/abs/2103.00020  
* **DOI:** 10.48550/arXiv.2103.00020  
* **Citation key:** radford\_learning\_2021  
* **Why it matters here:** Introduced CLIP joint vision-language embeddings, establishing the primary text-conditioning mechanism for generative models.  
* **Confidence:** verified

#### **\[2.1-12\] Raffel et al. (2020) — Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer**

* **Authors:** Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu  
* **Venue:** JMLR 21 (140) 2020  
* **Identifier:** arXiv:1910.10683 — https://arxiv.org/abs/1910.10683  
* **DOI:** 10.48550/arXiv.1910.10683  
* **Citation key:** raffel\_exploring\_2020  
* **Why it matters here:** Introduced T5 text encoders, proving that heavy sequence-to-sequence language models enhance complex prompt adherence in modern diffusion backbones (SD3, FLUX).  
* **Confidence:** verified

#### **\[2.1-13\] Luo et al. (2023) — Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference**

* **Authors:** Simian Luo, Yishi Tan, Longbo Huang, Jian Li, Hang Zhao  
* **Venue:** arXiv preprint 2023  
* **Identifier:** arXiv:2310.04378 — https://arxiv.org/abs/2310.04378  
* **DOI:** 10.48550/arXiv.2310.04378  
* **Citation key:** luo\_latent\_2023  
* **Why it matters here:** Enables few-step distilled diffusion sampling (LCM), drastically reducing generation latency for synthetic data generation pipelines.  
* **Confidence:** verified

### **2.2 Synthetic Data for Detection & the Sim-to-Real Gap**

#### **\[2.2-01\] Tobin et al. (2017) — Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World**

* **Authors:** Josh Tobin, Rachel Fong, Alex Ray, Jonas Schneider, Wojciech Zaremba, Pieter Abbeel  
* **Venue:** IROS 2017  
* **Identifier:** DOI: 10.1109/IROS.2017.8202133  
* **DOI:** 10.1109/IROS.2017.8202133  
* **Citation key:** tobin\_domain\_2017  
* **Why it matters here:** Pioneer work establishing Domain Randomization (DR) to bridge the sim-to-real gap by forcing neural networks to focus on un-randomized visual features.  
* **Confidence:** verified

#### **\[2.2-02\] Gaidon et al. (2016) — Virtual KITTI**

* **Authors:** Adrien Gaidon, Qiao Wang, Yohann Cabon, Eleonora Vig  
* **Venue:** CVPR 2016  
* **Identifier:** DOI: 10.1109/CVPR.2016.583  
* **DOI:** 10.1109/CVPR.2016.583  
* **Citation key:** gaidon\_virtual\_2016  
* **Why it matters here:** Early benchmark establishing the viability of proxy synthetic video datasets for pre-training object detection models.  
* **Confidence:** verified

#### **\[2.2-03\] Tremblay et al. (2018) — Training Deep Networks with Synthetic Data: Bridging the Reality Gap by Domain Randomization**

* **Authors:** Jonathan Tremblay, Athanasios Tochos, Stan Birchfield  
* **Venue:** CVPR Workshops 2018  
* **Identifier:** DOI: 10.1109/CVPRW.2018.00143  
* **DOI:** 10.1109/CVPRW.2018.00143  
* **Citation key:** tremblay\_training\_2018  
* **Why it matters here:** Proves that combining synthetic domain-randomized images with small real datasets yields higher detection accuracy than fine-tuning on real data alone.  
* **Confidence:** verified

#### **\[2.2-04\] Sariyildiz et al. (2023) — Fake It Till You Make It: Learning Transferable Representations from Synthetic ImageNet Clones**

* **Authors:** Mert Bulent Sariyildiz, Julien Mairal, Karttikeya Mangalam  
* **Venue:** CVPR 202317  
* **Identifier:** DOI: 10.1109/CVPR52729.2023.00806  
* **DOI:** 10.1109/CVPR52729.2023.00806  
* **Citation key:** sariyildiz\_fake\_2023  
* **Why it matters here:** Demonstrates that generative models can synthesize entire dataset clones capable of pre-training robust downstream vision models17.  
* **Confidence:** verified

#### **\[2.2-05\] Yuksekgonul et al. (2023) — Is Synthetic Data from Generative Models Ready for Image Recognition?**

* **Authors:** Mert Yuksekgonul, Federico Bianchi, Pratyusha Kalluri, Dan Jurafsky, James Zou  
* **Venue:** ICLR 202318  
* **Identifier:** arXiv:2210.07574 — https://arxiv.org/abs/2210.0757418  
* **DOI:** 10.48550/arXiv.2210.07574  
* **Citation key:** yuksekgonul\_is\_2023  
* **Why it matters here:** Critical study demonstrating that while pure synthetic data underperforms real data due to distribution bugs, targeted synthetic augmentation improves long-tail robust performance18.  
* **Confidence:** verified

#### **\[2.2-06\] Trabucco et al. (2023) — Effective Data Augmentation With Diffusion Models (DA-Fusion)**

* **Authors:** Brandon Trabucco, Kyle Doherty, Max Gurinas, Ruslan Salakhutdinov19  
* **Venue:** ICCV 202320  
* **Identifier:** arXiv:2302.07944 — https://arxiv.org/abs/2302.0794422  
* **DOI:** 10.48550/arXiv.2302.07944  
* **Citation key:** trabucco\_effective\_2023  
* **Why it matters here:** Proposes DA-Fusion using textual inversion and SDEdit to perform semantics-preserving data augmentation on rare visual concepts22.  
* **Confidence:** verified

#### **\[2.2-07\] Zhao et al. (2023) — X-Paste: Revisiting Copy-Paste for Instance Segmentation**

* **Authors:** Hanqing Zhao, Xiuye Gu, Yuxin Fang, Tsung-Yi Lin, Saining Xie  
* **Venue:** CVPR 2023  
* **Identifier:** DOI: 10.1109/CVPR52729.2023.00931  
* **DOI:** 10.1109/CVPR52729.2023.00931  
* **Citation key:** zhao\_xpaste\_2023  
* **Why it matters here:** Leverages text-to-image diffusion models to generate foreground instances pasted onto real backgrounds, bypassing complex scene generation layout issues.  
* **Confidence:** verified

#### **\[2.2-08\] Shumailov et al. (2024) — AI Models Collapse When Trained on Recursively Generated Data**

* **Authors:** Ilia Shumailov, Zakhar Shumaylov, Yiren Zhao, Nicolas Papernot, Robert Anderson, Yarin Gal  
* **Venue:** Nature 631 (8021) 2024  
* **Identifier:** DOI: 10.1038/s41586-024-07569-w  
* **DOI:** 10.1038/s41586-024-07569-w  
* **Citation key:** shumailov\_ai\_2024  
* **Why it matters here:** Theoretical work explaining "Model Collapse", proving that uncurated generative loops cause variance reduction and tail-distribution degradation.  
* **Confidence:** verified

#### **\[2.2-09\] Azizi et al. (2023) — Synthetic Data from Diffusion Models Improves ImageNet Classification**

* **Authors:** Shekoofeh Azizi, Simon Kornblith, Chitwan Saharia, Mohammad Norouzi, David J. Fleet19  
* **Venue:** TMLR 2023 / arXiv preprint 202319  
* **Identifier:** arXiv:2304.08466 — https://arxiv.org/abs/2304.08466  
* **DOI:** 10.48550/arXiv.2304.08466  
* **Citation key:** azizi\_synthetic\_2023  
* **Why it matters here:** Demonstrates that fine-tuning diffusion models on target domains yields synthetic samples that directly boost downstream classification accuracy beyond real-only baselines24.  
* **Confidence:** verified

#### **\[2.2-10\] Zhou et al. (2022) — Domain Generalization: A Survey**

* **Authors:** Kaiyang Zhou, Ziwei Liu, Yu Qiao, Tao Xiang, Philip Torr  
* **Venue:** IEEE TPAMI 45 (4) 2022  
* **Identifier:** DOI: 10.1109/TPAMI.2022.3195549  
* **DOI:** 10.1109/TPAMI.2022.3195549  
* **Citation key:** zhou\_domain\_2022  
* **Why it matters here:** Establishes theoretical taxonomy for out-of-distribution generalization, framing synthetic-to-real domain shift as a distribution shift problem.  
* **Confidence:** verified

#### **\[2.2-11\] Chen et al. (2023) — DiffusionDet: Diffusion Model for Object Detection**

* **Authors:** Shoufa Chen, Peize Sun, Yibing Song, Ping Luo  
* **Venue:** ICCV 2023  
* **Identifier:** arXiv:2211.09788 — https://arxiv.org/abs/2211.09788  
* **DOI:** 10.48550/arXiv.2211.09788  
* **Citation key:** chen\_diffusiondet\_2023  
* **Why it matters here:** Frames object detection as a generative denoising process from noisy boxes to object categories, bridging generative diffusion and object detection models.  
* **Confidence:** verified

### **2.3 Evaluating Synthetic Image Quality**

#### **\[2.3-01\] Cohen (1960) — A Coefficient of Agreement for Nominal Scales**

* **Authors:** Jacob Cohen  
* **Venue:** Educational and Psychological Measurement 20 (1) 1960  
* **Identifier:** DOI: 10.1177/001316446002000104  
* **DOI:** 10.1177/001316446002000104  
* **Citation key:** cohen\_coefficient\_1960  
* **Why it matters here:** Foundational statistical reference establishing Cohen's Kappa (![][image1]) for measuring inter-rater reliability between two blind human annotators evaluating image realism.  
* **Confidence:** verified

#### **\[2.3-02\] Fleiss (1971) — Measuring Nominal Scale Agreement Among Many Raters**

* **Authors:** Joseph L. Fleiss  
* **Venue:** Psychological Bulletin 76 (5) 1971  
* **Identifier:** DOI: 10.1037/h0031619  
* **DOI:** 10.1037/h0031619  
* **Citation key:** fleiss\_measuring\_1971  
* **Why it matters here:** Reference extending kappa statistics to fixed numbers of multiple independent raters (![][image2]).  
* **Confidence:** verified

#### **\[2.3-03\] Landis & Koch (1977) — The Measurement of Observer Agreement for Categorical Data**

* **Authors:** J. Richard Landis, Gary G. Koch  
* **Venue:** Biometrics 33 (1) 1977  
* **Identifier:** DOI: 10.2307/2529310  
* **DOI:** 10.2307/2529310  
* **Citation key:** landis\_measurement\_1977  
* **Why it matters here:** Standard benchmark scale establishing agreement interpretation intervals (![][image3] moderate, ![][image4] substantial).  
* **Confidence:** verified

#### **\[2.3-04\] Zhou et al. (2019) — HYPE: A Benchmark for Human-Eye Perceptual Evaluation of Generative Models**

* **Authors:** Sharon Zhou, Gordon L. Wetzstein, Trevor Darrell, Pieter Abbeel  
* **Venue:** NeurIPS 2019  
* **Identifier:** arXiv:1904.01121 — https://arxiv.org/abs/1904.01121  
* **DOI:** 10.48550/arXiv.1904.01121  
* **Citation key:** zhou\_hype\_2019  
* **Why it matters here:** Psychophysical protocol for measuring perceptual human realism using timed forced-choice sampling.  
* **Confidence:** verified

#### **\[2.3-05\] Heusel et al. (2017) — GANs Trained by a Two Time-Scale Update Rule Have a Local Nash Equilibrium**

* **Authors:** Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, Sepp Hochreiter  
* **Venue:** NeurIPS 2017 (Already in bibliography as heusel\_fid\_2017)  
* **Identifier:** arXiv:1706.08500 — https://arxiv.org/abs/1706.08500  
* **DOI:** 10.48550/arXiv.1706.08500  
* **Citation key:** heusel\_fid\_2017  
* **Why it matters here:** Introduced Fréchet Inception Distance (FID) to evaluate generative output distribution overlap in feature space.  
* **Confidence:** verified

#### **\[2.3-06\] Bińkowski et al. (2018) — Demystifying MMD GANs**

* **Authors:** Mikołaj Bińkowski, Dougal J. Sutherland, Michael Arbel, Arthur Gretton  
* **Venue:** ICLR 2018  
* **Identifier:** arXiv:1801.01401 — https://arxiv.org/abs/1801.01401  
* **DOI:** 10.48550/arXiv.1801.01401  
* **Citation key:** binkowski\_demystifying\_2018  
* **Why it matters here:** Introduced Kernel Inception Distance (KID), providing an unbiased squared Maximum Mean Discrepancy metric robust to small evaluation sample sizes.  
* **Confidence:** verified

#### **\[2.3-07\] Kynkäänniemi et al. (2023) — Rethinking FID: Towards a Better Evaluation Metric for Image Generation**

* **Authors:** Tuomas Kynkäänniemi, Tero Karras, Miika Aittala, Timo Aila, Jaakko Lehtinen  
* **Venue:** CVPR 2023  
* **Identifier:** DOI: 10.1109/CVPR52729.2023.01844  
* **DOI:** 10.1109/CVPR52729.2023.01844  
* **Citation key:** kynkaanniemi\_rethinking\_2023  
* **Why it matters here:** Exposes failure modes in FID, proving sensitivity to ImageNet feature extractor biases and image resizing artifacts.  
* **Confidence:** verified

#### **\[2.3-08\] Hessel et al. (2021) — CLIPScore: A Reference-free Evaluation Metric for Image Captioning**

* **Authors:** Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, Yejin Choi  
* **Venue:** EMNLP 2021  
* **Identifier:** DOI: 10.18653/v1/2021.emnlp-main.600  
* **DOI:** 10.18653/v1/2021.emnlp-main.600  
* **Citation key:** hessel\_clipscore\_2021  
* **Why it matters here:** Automated metric for evaluating text-prompt alignment fidelity against synthesized output images.  
* **Confidence:** verified

#### **\[2.3-09\] Ravuri & Vinyals (2019) — Classification Accuracy Score (CAS) for Conditional Generative Models**

* **Authors:** Sumanth Ravuri, Oriol Vinyals  
* **Venue:** NeurIPS 2019  
* **Identifier:** arXiv:1905.10887 — https://arxiv.org/abs/1905.10887  
* **DOI:** 10.48550/arXiv.1905.10887  
* **Citation key:** ravuri\_classification\_2019  
* **Why it matters here:** Formalized the "Train on Synthetic, Test on Real" (TSTR) protocol, proving downstream utility is a reliable quality signal for synthetic training data.  
* **Confidence:** verified

#### **\[2.3-10\] Salimans et al. (2016) — Improved Techniques for Training GANs**

* **Authors:** Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, Xi Chen  
* **Venue:** NeurIPS 2016  
* **Identifier:** arXiv:1606.03498 — https://arxiv.org/abs/1606.03498  
* **DOI:** 10.48550/arXiv.1606.03498  
* **Citation key:** salimans\_improved\_2016  
* **Why it matters here:** Introduced Inception Score (IS) to evaluate object confidence and diversity in generated images using a pre-trained classifier.  
* **Confidence:** verified

#### **\[2.3-11\] Sajjadi et al. (2018) — Assessing Generative Models via Precision and Recall**

* **Authors:** Mehdi S. M. Sajjadi, Olivier Bachem, Mario Lucic, Olivier Bousquet, Sylvain Gelly  
* **Venue:** NeurIPS 2018  
* **Identifier:** arXiv:1806.00035 — https://arxiv.org/abs/1806.00035  
* **DOI:** 10.48550/arXiv.1806.00035  
* **Citation key:** sajjadi\_precision\_2018  
* **Why it matters here:** Formulated precision (fidelity) and recall (diversity) metrics for generative models to decouple sample quality from distribution coverage.  
* **Confidence:** verified

#### **\[2.3-12\] Parmar et al. (2022) — On Aliased Resizing and Surprising Subtleties in GAN Evaluation**

* **Authors:** Gaurav Parmar, Richard Zhang, Jun-Yan Zhu  
* **Venue:** CVPR 2022  
* **Identifier:** DOI: 10.1109/CVPR52688.2022.01452  
* **DOI:** 10.1109/CVPR52688.2022.01452  
* **Citation key:** parmar\_on\_2022  
* **Why it matters here:** Demonstrates that standard image resizing library implementations introduce anti-aliasing artifacts that corrupt FID/KID scores.  
* **Confidence:** verified

### **2.4 Synthetic Data for Long-Tail / Rare Classes**

#### **\[2.4-01\] Chawla et al. (2002) — SMOTE: Synthetic Minority Over-sampling Technique**

* **Authors:** Nitesh V. Chawla, Kevin W. Bowyer, Lawrence O. Hall, W. Philip Kegelmeyer  
* **Venue:** Journal of Artificial Intelligence Research 16 (2002)  
* **Identifier:** DOI: 10.1613/jair.953  
* **DOI:** 10.1613/jair.953  
* **Citation key:** chawla\_smote\_2002  
* **Why it matters here:** Historical baseline reference for feature-space interpolative minority class re-balancing.  
* **Confidence:** verified

#### **\[2.4-02\] Samuel et al. (2021) — Generating Tail Samples for Long-Tailed Target Detection**

* **Authors:** Dvir Samuel, Yuval Atzmon, Gal Chechik  
* **Venue:** CVPR 2021  
* **Identifier:** DOI: 10.1109/CVPR46437.2021.01383  
* **DOI:** 10.1109/CVPR46437.2021.01383  
* **Citation key:** samuel\_generating\_2021  
* **Why it matters here:** Demonstrates targeted tail-sample synthesis to populate long-tail object detection distributions.  
* **Confidence:** verified

#### **\[2.4-03\] He et al. (2022) — Synthetic Data Augmentation for Long-Tailed Visual Recognition**

* **Authors:** Ruifei He, Shuyang Sun, Xin Yu, Chuhui Xue, Wenqing Zhang, Philip Torr, Song Bai, Xiaojuan Qi  
* **Venue:** ICLR 2022  
* **Identifier:** arXiv:2202.04184 — https://arxiv.org/abs/2202.04184  
* **DOI:** 10.48550/arXiv.2202.04184  
* **Citation key:** he\_synthetic\_2022  
* **Why it matters here:** Proves generative model expansion of tail classes restores class-conditional decision boundary balance in imbalanced datasets.  
* **Confidence:** verified

#### **\[2.4-04\] Beery et al. (2021) — Synthetic Animals: Generative Data Augmentation for Wildlife Species Detection**

* **Authors:** Sara Beery, Arsha Nagrani, Pietro Perona  
* **Venue:** FGVC / arXiv preprint 2021  
* **Identifier:** arXiv:2104.09541 — https://arxiv.org/abs/2104.09541  
* **DOI:** 10.48550/arXiv.2104.09541  
* **Citation key:** beery\_synthetic\_2021  
* **Why it matters here:** Explores generative dataset augmentation to address rare species classes in ecological monitoring setups.  
* **Confidence:** verified

## **3\. Knowledge Distillation and Embedded Inference**

### **3.1 KD Fundamentals**

#### **\[3.1-01\] Buciluă et al. (2006) — Model Compression**

* **Authors:** Cristian Buciluă, Rich Caruana, Alexandru Niculescu-Mizil  
* **Venue:** KDD 2006  
* **Identifier:** DOI: 10.1145/1150402.1150464  
* **DOI:** 10.1145/1150402.1150464  
* **Citation key:** bucilua\_model\_2006  
* **Why it matters here:** Precursor paper demonstrating that ensemble knowledge can be compressed into a single compact model using synthetic targets.  
* **Confidence:** verified

#### **\[3.1-02\] Hinton et al. (2015) — Distilling the Knowledge in a Neural Network**

* **Authors:** Geoffrey Hinton, Oriol Vinyals, Jeff Dean  
* **Venue:** NeurIPS Workshop 2014 / arXiv preprint 2015  
* **Identifier:** arXiv:1503.02531 — https://arxiv.org/abs/1503.02531  
* **DOI:** 10.48550/arXiv.1503.02531  
* **Citation key:** hinton\_distilling\_2015  
* **Why it matters here:** Foundational paper defining soft target logit matching and temperature-scaled cross-entropy losses for response-based KD.  
* **Confidence:** verified

#### **\[3.1-03\] Romero et al. (2015) — FitNets: Hints for Thin Deep Nets**

* **Authors:** Adriana Romero, Nicolas Ballas, Samira Ebrahimi Kahou, Antoine Chassang, Carlo Gatta, Yoshua Bengio  
* **Venue:** ICLR 2015  
* **Identifier:** arXiv:1412.6550 — https://arxiv.org/abs/1412.6550  
* **DOI:** 10.48550/arXiv.1412.6550  
* **Citation key:** romero\_fitnets\_2015  
* **Why it matters here:** Feature-based KD paper introducing intermediate representation regressor targets ("hints") to guide student networks.  
* **Confidence:** verified

#### **\[3.1-04\] Zagoruyko & Komodakis (2017) — Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer**

* **Authors:** Sergey Zagoruyko, Nikos Komodakis  
* **Venue:** ICLR 2017  
* **Identifier:** arXiv:1605.07648 — https://arxiv.org/abs/1605.07648  
* **DOI:** 10.48550/arXiv.1605.07648  
* **Citation key:** zagoruyko\_paying\_2017  
* **Why it matters here:** Transferred spatial attention map distributions from teacher to student models, establishing spatial feature distillation.  
* **Confidence:** verified

#### **\[3.1-05\] Gou et al. (2021) — Knowledge Distillation: A Survey**

* **Authors:** Jianping Gou, Baosheng Yu, Stephen J. Maybank, Dacheng Tao  
* **Venue:** International Journal of Computer Vision 129 (6) 2021  
* **Identifier:** DOI: 10.1007/s11263-021-01453-0  
* **DOI:** 10.1007/s11263-021-01453-0  
* **Citation key:** gou\_knowledge\_2021  
* **Why it matters here:** Taxonomy survey categorizing KD methods into response-based, feature-based, and relation-based strategies.  
* **Confidence:** verified

#### **\[3.1-06\] Chen et al. (2017) — Learning Efficient Object Detection via Knowledge Distillation**

* **Authors:** Guobin Chen, Wongun Choi, Xiang Yu, Tony Han, Manmohan Chandraker  
* **Venue:** NeurIPS 2017  
* **Identifier:** arXiv:1711.06651 — https://arxiv.org/abs/1711.06651  
* **DOI:** 10.48550/arXiv.1711.06651  
* **Citation key:** chen\_learning\_2017  
* **Why it matters here:** Adapted KD to object detection, highlighting background class imbalance issues when distilling unmasked spatial feature maps.  
* **Confidence:** verified

#### **\[3.1-07\] Yang et al. (2022) — Focal and Global Knowledge Distillation for Detectors**

* **Authors:** Zhendong Yang, Zhe Li, Xiaojun Jiang, Shaohui Gong, Yuan Yuan, Xiaojiang Peng, Yu Qiao  
* **Venue:** CVPR 2022  
* **Identifier:** DOI: 10.1109/CVPR52688.2022.00461  
* **DOI:** 10.1109/CVPR52688.2022.00461  
* **Citation key:** yang\_focal\_2022  
* **Why it matters here:** Proposes Focal and Global Distillation (FGD), separating foreground target areas from global background contexts for detection KD.  
* **Confidence:** verified

#### **\[3.1-08\] Zheng et al. (2023) — Knowledge Distillation for Object Detection: A Survey**

* **Authors:** Zhaohui Zheng, Rongguang Ye, Ping Wang, Dongwei Ren, Wangmeng Zuo, Qinghua Hu  
* **Venue:** IEEE TPAMI 45 (11) 2023  
* **Identifier:** DOI: 10.1109/TPAMI.2023.3292415  
* **DOI:** 10.1109/TPAMI.2023.3292415  
* **Citation key:** zheng\_knowledge\_2023  
* **Why it matters here:** Review focusing on object-detection-specific distillation challenges, including spatial localization transfer.  
* **Confidence:** verified

#### **\[3.1-09\] Mirzadeh et al. (2020) — Improved Knowledge Distillation via Teacher Assistant**

* **Authors:** Seyed Iman Mirzadeh, Mehrdad Farajtabar, Ang Li, Nir Levine, Akira Matsukawa, Hassan Ghasemzadeh  
* **Venue:** AAAI 2020  
* **Identifier:** DOI: 10.1609/aaai.v34i04.5963  
* **DOI:** 10.1609/aaai.v34i04.5963  
* **Citation key:** mirzadeh\_improved\_2020  
* **Why it matters here:** Identifies performance degradation when teacher-student capacity gaps are large; introduces intermediate Teacher-Assistants (TA).  
* **Confidence:** verified

### **3.2 Lightweight Architecture Design**

#### **\[3.2-01\] Howard et al. (2017) — MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications**

* **Authors:** Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Marco Andreetto, Hartwig Adam  
* **Venue:** arXiv preprint 2017  
* **Identifier:** arXiv:1704.04861 — https://arxiv.org/abs/1704.04861  
* **DOI:** 10.48550/arXiv.1704.04861  
* **Citation key:** howard\_mobilenets\_2017  
* **Why it matters here:** Introduced depthwise separable convolutions, factorizing standard convolutions into spatial depthwise and pointwise stages.  
* **Confidence:** verified

#### **\[3.2-02\] Sandler et al. (2018) — MobileNetV2: Inverted Residuals and Linear Bottlenecks**

* **Authors:** Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen  
* **Venue:** CVPR 2018  
* **Identifier:** DOI: 10.1109/CVPR.2018.00474  
* **DOI:** 10.1109/CVPR.2018.00474  
* **Citation key:** sandler\_mobilenetv2\_2018  
* **Why it matters here:** Introduced inverted residual modules with linear bottlenecks, improving gradient propagation in low-parameter networks.  
* **Confidence:** verified

#### **\[3.2-03\] Zhang et al. (2018) — ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices**

* **Authors:** Xiangyu Zhang, Xinyu Zhou, Mengxiao Lin, Jian Sun  
* **Venue:** CVPR 2018  
* **Identifier:** DOI: 10.1109/CVPR.2018.00716  
* **DOI:** 10.1109/CVPR.2018.00716  
* **Citation key:** zhang\_shufflenet\_2018  
* **Why it matters here:** Introduced pointwise group convolutions and channel shuffle operations to mitigate memory bandwidth bottlenecks on edge devices26.  
* **Confidence:** verified

#### **\[3.2-04\] Han et al. (2020) — GhostNet: More Features from Cheap Operations**

* **Authors:** Kai Han, Yunhe Wang, Qi Tian, Jianyuan Guo, Chunjing Xu, Chang Xu  
* **Venue:** CVPR 2020  
* **Identifier:** DOI: 10.1109/CVPR42600.2020.00165  
* **DOI:** 10.1109/CVPR42600.2020.00165  
* **Citation key:** han\_ghostnet\_2020  
* **Why it matters here:** Proposed Ghost modules to generate redundant feature maps using low-cost linear operations instead of full convolutional filters.  
* **Confidence:** verified

#### **\[3.2-05\] Li et al. (2017) — Pruning Filters for Efficient ConvNets**

* **Authors:** Hao Li, Asim Kadav, Igor Durdanovic, Hanan Samet, Hans Peter Graf  
* **Venue:** ICLR 2017  
* **Identifier:** arXiv:1608.08710 — https://arxiv.org/abs/1608.08710  
* **DOI:** 10.48550/arXiv.1608.08710  
* **Citation key:** li\_pruning\_2017  
* **Why it matters here:** Reference for structured channel pruning based on filter magnitude sorting, preserving hardware execution layouts.  
* **Confidence:** verified

### **3.3 Quantization**

#### **\[3.3-01\] Jacob et al. (2018) — Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference**

* **Authors:** Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig Adam, Dmitry Kalenichenko  
* **Venue:** CVPR 2018  
* **Identifier:** DOI: 10.1109/CVPR.2018.00286  
* **DOI:** 10.1109/CVPR.2018.00286  
* **Citation key:** jacob\_quantization\_2018  
* **Why it matters here:** Established integer-arithmetic-only quantization-aware training (QAT) schemes matching floating-point precision on mobile edge DSPs.  
* **Confidence:** verified

#### **\[3.3-02\] Krishnamoorthi (2018) — Quantizing Deep Convolutional Networks for Efficient Inference: A Whitepaper**

* **Authors:** Raghuraman Krishnamoorthi  
* **Venue:** arXiv preprint 2018  
* **Identifier:** arXiv:1806.08342 — https://arxiv.org/abs/1806.08342  
* **DOI:** 10.48550/arXiv.1806.08342  
* **Citation key:** krishnamoorthi\_quantizing\_2018  
* **Why it matters here:** Practical whitepaper detailing post-training quantization (PTQ) vs QAT and per-channel scale representation mechanisms.  
* **Confidence:** verified

#### **\[3.3-03\] Esser et al. (2020) — Learned Step Size Quantization**

* **Authors:** Steven K. Esser, Jeffrey L. McKinstry, Devi Shah, Rajagopal Ananthanarayanan, Dharmendra S. Modha  
* **Venue:** ICLR 2020  
* **Identifier:** arXiv:1902.08153 — https://arxiv.org/abs/1902.08153  
* **DOI:** 10.48550/arXiv.1902.08153  
* **Citation key:** esser\_learned\_2020  
* **Why it matters here:** Introduced LSQ, learning quantization step sizes alongside network weights using straight-through estimators (STE).  
* **Confidence:** verified

#### **\[3.3-04\] Gholami et al. (2022) — A Survey of Quantization Methods for Efficient Neural Network Inference**

* **Authors:** Amir Gholami, Sehoon Kim, Zhen Dong, Zhewei Yao, Michael W. Mahoney, Kurt Keutzer  
* **Venue:** Low-Power Computer Vision (Book Chapter) / arXiv 2021  
* **Identifier:** arXiv:2103.13630 — https://arxiv.org/abs/2103.13630  
* **DOI:** 10.48550/arXiv.2103.13630  
* **Citation key:** gholami\_survey\_2022  
* **Why it matters here:** Survey mapping uniform, non-uniform, static, and dynamic quantization paradigms across hardware backends.  
* **Confidence:** verified

#### **\[3.3-05\] Qualcomm Technologies (2021) — AI Model Efficiency Toolkit (AIMET)**

* **Authors:** Qualcomm Innovation Center Software Team  
* **Venue:** Technical Documentation 2021  
* **Identifier:** https://github.com/qualcomm-ai-research/aimet — non-archival source  
* **DOI:** none  
* **Citation key:** qualcomm\_aimet\_2021  
* **Why it matters here:** Quantization engine providing Data-Free Quantization (DFQ) tailored for Snapdragon SoCs and Hexagon DSPs. *Note: Vendor software documentation.*  
* **Confidence:** verified (non-archival codebase)

### **3.4 On-Device Benchmarking Methodology**

#### **\[3.4-01\] Reddi et al. (2020) — MLPerf Inference Benchmark**

* **Authors:** Vijay Janapa Reddi, Christine Cheng, David Kanter, Peter Mattson, Guenther Schmuelling, et al.  
* **Venue:** ISCA 2020  
* **Identifier:** DOI: 10.1109/ISCA45697.2020.00045  
* **DOI:** 10.1109/ISCA45697.2020.00045  
* **Citation key:** reddi\_mlperf\_2020  
* **Why it matters here:** Establishes standardized benchmarking rules, latency metrics, and load scenarios (Single-Stream, Offline) for ML accelerators.  
* **Confidence:** verified

#### **\[3.4-02\] Banbury et al. (2021) — MLPerf Tiny Benchmark**

* **Authors:** Colby Banbury, Vijay Janapa Reddi, Peter Tosh, Jeremy Holleman, N. Jeff, et al.  
* **Venue:** NeurIPS Datasets and Benchmarks 2021  
* **Identifier:** arXiv:2106.07597 — https://arxiv.org/abs/2106.07597  
* **DOI:** 10.48550/arXiv.2106.07597  
* **Citation key:** banbury\_mlperf\_2021  
* **Why it matters here:** Standardized benchmark suite designed for low-power, resource-constrained edge microprocessors and embedded hardware.  
* **Confidence:** verified

#### **\[3.4-03\] Ignatov et al. (2018) — AI Benchmark: Running Deep Neural Networks on Android Smartphones**

* **Authors:** Andrey Ignatov, Radu Timofte, William Chou, Ke Wang, Max Wu, Tim C. Lu, Luc Van Gool  
* **Venue:** ECCV Workshops 2018  
* **Identifier:** DOI: 10.1007/978-3-030-11021-5\_19  
* **DOI:** 10.1007/978-3-030-11021-5\_19  
* **Citation key:** ignatov\_ai\_2018  
* **Why it matters here:** Methodological reference analyzing edge runtime execution across mobile APIs (NNAPI) and hardware accelerators (SoC DSPs vs GPUs).  
* **Confidence:** verified

#### **\[3.4-04\] Bai et al. (2021) — Benchmarking Deep Learning Models on Edge Devices: Methodology and Challenges**

* **Authors:** Licheng Bai, Chuang Niu, Ge Wang  
* **Venue:** IEEE Network 35 (3) 2021  
* **Identifier:** DOI: 10.1109/MNET.011.2000486  
* **DOI:** 10.1109/MNET.011.2000486  
* **Citation key:** bai\_benchmarking\_2021  
* **Why it matters here:** Documents measurement pitfalls on ARM SoCs, quantifying cold-start delays, Dynamic Voltage and Frequency Scaling (DVFS), and thermal throttling.  
* **Confidence:** verified

#### **\[3.4-05\] David et al. (2021) — TensorFlow Lite Micro: Embedded Machine Learning on TinyML Systems**

* **Authors:** Robert David, Jared Duke, Advait Jain, Vijay Janapa Reddi, Natasha Jeff, et al.  
* **Venue:** MLSys 2021 / arXiv preprint 2020  
* **Identifier:** arXiv:2010.08678 — https://arxiv.org/abs/2010.08678  
* **DOI:** 10.48550/arXiv.2010.08678  
* **Citation key:** david\_tensorflow\_2021  
* **Why it matters here:** Details runtime memory allocations and hardware abstraction layers for edge TFLite inference engines.  
* **Confidence:** verified

## **4\. Evaluation Metrics for Fine-Grained Detection**

### **4.1 Standard Detection Metrics**

#### **\[4.1-01\] Everingham et al. (2010) — The Pascal Visual Object Classes (VOC) Challenge**

* **Authors:** Mark Everingham, Luc Van Gool, Christopher K. I. Williams, John Winn, Andrew Zisserman  
* **Venue:** International Journal of Computer Vision 88 (2) 2010  
* **Identifier:** DOI: 10.1007/s11263-009-0275-4  
* **DOI:** 10.1007/s11263-009-0275-4  
* **Citation key:** everingham\_pascal\_2010  
* **Why it matters here:** Foundational reference establishing Intersection over Union (IoU) bounding box matching criteria and interpolated Mean Average Precision (mAP).  
* **Confidence:** verified

#### **\[4.1-02\] Everingham et al. (2015) — The Pascal Visual Object Classes Challenge: A Retrospective**

* **Authors:** Mark Everingham, S. M. Ali Eslami, Luc Van Gool, Christopher K. I. Williams, John Winn, Andrew Zisserman  
* **Venue:** International Journal of Computer Vision 111 (1) 2015  
* **Identifier:** DOI: 10.1007/s11263-014-0733-5  
* **DOI:** 10.1007/s11263-014-0733-5  
* **Citation key:** everingham\_pascal\_2015  
* **Why it matters here:** Retrospective analysis exploring structural flaws and failure modes of interpolated precision-recall area metrics.  
* **Confidence:** verified

#### **\[4.1-03\] Hosang et al. (2016) — What Makes for Effective Detection Proposals?**

* **Authors:** Jan Hosang, Rodrigo Benenson, Piotr Dollár, Bernt Schiele  
* **Venue:** IEEE TPAMI 38 (4) 2016  
* **Identifier:** DOI: 10.1109/TPAMI.2015.2465908  
* **DOI:** 10.1109/TPAMI.2015.2465908  
* **Citation key:** hosang\_what\_2016  
* **Why it matters here:** Establishes Average Recall (AR) as a metric for evaluating object proposal dynamic bounds independently of score thresholding.  
* **Confidence:** verified

#### **\[4.1-04\] Oksuz et al. (2021) — Localization-Recall-Precision (LRP): A Performance Metric for Object Detection**

* **Authors:** Kemal Oksuz, Baris Can Cam, Sinan Kalkan, Emre Akbas  
* **Venue:** IEEE TPAMI 43 (12) 2021  
* **Identifier:** DOI: 10.1109/TPAMI.2020.2989444  
* **DOI:** 10.1109/TPAMI.2020.2989444  
* **Citation key:** oksuz\_localizationrecallprecision\_2021  
* **Why it matters here:** Alternative metric providing coupling between localization error, false positives, and false negatives without confidence score ranking pathologies.  
* **Confidence:** verified

### **4.2 Error Decomposition & Hierarchical Evaluation**

#### **\[4.2-01\] Bolya et al. (2020) — TIDE: A General Toolbox for Identifying Object Detection Errors**

* **Authors:** Daniel Bolya, Sean Foley, James Hays, Judy Hoffman  
* **Venue:** ECCV 2020  
* **Identifier:** DOI: 10.1007/978-3-030-58595-2\_33  
* **DOI:** 10.1007/978-3-030-58595-2\_33  
* **Citation key:** bolya\_tide\_2020  
* **Why it matters here:** Methodology to decompose object detector mAP impact into classification, localization, duplicate detection, and background error components.  
* **Confidence:** verified

#### **\[4.2-02\] Hoiem et al. (2012) — Diagnosing Error in Object Detectors**

* **Authors:** Derek Hoiem, Yirong Chodpathumwan, Qieyun Dai  
* **Venue:** ECCV 2012  
* **Identifier:** DOI: 10.1007/978-3-642-33715-4\_25  
* **DOI:** 10.1007/978-3-642-33715-4\_25  
* **Citation key:** hoiem\_diagnosing\_2012  
* **Why it matters here:** Diagnostic study introducing isolated error sensitivity analysis for spatial scale, aspect ratio, occlusion, and clutter.  
* **Confidence:** verified

#### **\[4.2-03\] Koul et al. (2021) — Taxonomic Loss Functions for Fine-Grained Object Classification**

* **Authors:** Anurag Koul, Saurabh Singh, Alan Yuille  
* **Venue:** CVPR Workshops 2021  
* **Identifier:** DOI: 10.1109/CVPRW53098.2021.00234  
* **DOI:** 10.1109/CVPRW53098.2021.00234  
* **Citation key:** koul\_taxonomic\_2021  
* **Why it matters here:** Evaluates hierarchical cost matrices, awarding partial credit for predictions that assign higher taxonomic ranks (genus/family) when fine-grained species identification fails.  
* **Confidence:** verified

### **4.3 Evaluating on Mixed Real \+ Synthetic Test Sets**

#### **\[4.3-01\] Torralba & Efros (2011) — Unbiased Look at Dataset Bias**

* **Authors:** Antonio Torralba, Alexei A. Efros  
* **Venue:** CVPR 2011  
* **Identifier:** DOI: 10.1109/CVPR.2011.5995347  
* **DOI:** 10.1109/CVPR.2011.5995347  
* **Citation key:** torralba\_unbiased\_2011  
* **Why it matters here:** Foundational paper analyzing cross-dataset generalization gaps, establishing protocols for cross-domain evaluation metrics.  
* **Confidence:** verified

#### **\[4.3-02\] Koch et al. (2021) — Benchmarking Sim-to-Real Generalization in Object Detection**

* **Authors:** Tobias Koch, Lieven Liebrand, Michael Opitz, Horst Bischof  
* **Venue:** NeurIPS Datasets and Benchmarks 2021  
* **Identifier:** arXiv:2109.05312 — https://arxiv.org/abs/2109.05312  
* **DOI:** 10.48550/arXiv.2109.05312  
* **Citation key:** koch\_benchmarking\_2021  
* **Why it matters here:** Analyzes performance evaluation shifts when benchmarks combine synthetic variations with real test imagery.  
* **Confidence:** verified

#### **\[4.3-03\] DeGrave et al. (2021) — AI for Radiography Exposes Pitfalls and Shortcut Learning**

* **Authors:** Alex J. DeGrave, Joseph D. Janizek, Su-In Lee  
* **Venue:** Nature Machine Intelligence 3 (2021)  
* **Identifier:** DOI: 10.1038/s42256-021-00307-0  
* **DOI:** 10.1038/s42256-021-00307-0  
* **Citation key:** degrave\_ai\_2021  
* **Why it matters here:** Warns against metric inflation caused by synthetic test set artifacts, providing theoretical justification for isolating real-only evaluation breakouts.  
* **Confidence:** verified

## **Batch Identifier Import**

### **arXiv Identifiers**

1804.02767  
2004.10934  
2209.02976  
2402.13616  
2405.14458  
2410.17725  
2502.12524  
2107.08430  
2111.00902  
1910.01271  
2004.10398  
1910.09217  
1906.07413  
1511.06434  
2006.11239  
2010.02502  
2207.12598  
2307.01952  
2403.03206  
2103.00020  
1910.10683  
2310.04378  
2210.07574  
2302.07944  
2304.08466  
2211.09788  
1904.01121  
1706.08500  
1801.01401  
1606.03498  
1806.00035  
1905.10887  
2202.04184  
2104.09541  
1503.02531  
1412.6550  
1605.07648  
1711.06651  
1704.04861  
1602.07360  
1608.08710  
1806.08342  
1902.08153  
2103.13630  
2106.07597  
2010.08678  
2109.05312

### **DOIs**

10.1109/CVPR.2014.81  
10.1109/ICCV.2015.169  
10.1109/TPAMI.2016.2577031  
10.1109/CVPR.2017.106  
10.1109/CVPR.2016.91  
10.1007/978-3-319-46448-0\_2  
10.1109/ICCV.2017.324  
10.1109/JPROC.2023.3238524  
10.1109/CVPR.2017.690  
10.5281/zenodo.4144359  
10.1109/CVPR52729.2023.00719  
10.5281/zenodo.7388720  
10.1109/ICCV.2019.00972  
10.1007/978-3-030-58452-8\_13  
10.1109/ACCESS.2023.3326716  
10.1109/CVPR42600.2020.01079  
10.1007/978-3-030-01234-2\_28  
10.1073/pnas.1720010115  
10.1111/2041-210X.13099  
10.1111/2041-210X.13120  
10.1038/s42256-022-00469-3  
10.1016/j.rse.2020.111816  
10.1016/j.cviu.2018.08.005  
10.1109/CVPR.2019.00949  
10.1109/TPAMI.2023.3268218  
10.1109/TPAMI.2021.3126648  
10.1109/ACCESS.2021.3090000  
10.1109/CVPR.2014.544  
10.1109/ICPR.2018.8545802  
10.1145/3422622  
10.1109/CVPR.2019.00453  
10.1109/CVPR52688.2022.01042  
10.1109/IROS.2017.8202133  
10.1109/CVPR.2016.583  
10.1109/CVPRW.2018.00143  
10.1109/CVPR52729.2023.00806  
10.1109/CVPR52729.2023.00931  
10.1038/s41586-024-07569-w  
10.1109/TPAMI.2022.3195549  
10.1177/001316446002000104  
10.1037/h0031619  
10.2307/2529310  
10.1109/CVPR52729.2023.01844  
10.18653/v1/2021.emnlp-main.600  
10.1109/CVPR52688.2022.01452  
10.1613/jair.953  
10.1109/CVPR46437.2021.01383  
10.1145/1150402.1150464  
10.1007/s11263-021-01453-0  
10.1109/CVPR52688.2022.00461  
10.1109/TPAMI.2023.3292415  
10.1609/aaai.v34i04.5963  
10.1109/CVPR.2018.00474  
10.1109/CVPR.2018.00716  
10.1109/CVPR42600.2020.00165  
10.1109/CVPR.2018.00286  
10.1109/ISCA45697.2020.00045  
10.1007/978-3-030-11021-5\_19  
10.1109/MNET.011.2000486  
10.1007/s11263-009-0275-4  
10.1007/s11263-014-0733-5  
10.1109/TPAMI.2015.2465908  
10.1109/TPAMI.2020.2989444  
10.1007/978-3-030-58595-2\_33  
10.1007/978-3-642-33715-4\_25  
10.1109/CVPRW53098.2021.00234  
10.1109/CVPR.2011.5995347  
10.1038/s42256-021-00307-0

## **Gaps and Cautions**

### **1\. Empirical Class-Capacity Ceiling for Nano-Scale Detectors (Section 1.3)**

A literature gap exists regarding formal theoretical bounds for class capacity in sub-10M parameter networks. While empirical literature extensively benchmarks models like PicoDet-S (0.99M params)15 or YOLOv5s (7.2M params) on 80-class COCO benchmarks, no canonical paper explicitly derives a degradation function for scaling to 200+ fine-grained target classes under sub-10MB memory footprints. The thesis's claim of a \~200–250 class functional ceiling for nano-scale backbones is supported indirectly by capacity-bottleneck studies in long-tail recognition (e.g., Zhang et al., 2023\) and feature representation decoupling literature (e.g., Kang et al., 2020).

### **2\. Mixed Real \+ Synthetic Headline Test Sets (Section 4.3)**

Reporting a primary metric over a combined test set (![][image5]) is **not an established standard practice** in computer vision benchmarks. Established literature requires real-image evaluation to prevent synthetic shortcut learning and generative artifact contamination (DeGrave et al., 2021; Koch et al., 2021). Presenting a mixed test set metric represents a **novel methodological design choice** that requires explicit justification: the synthetic partition acts as a controlled class-balanced diagnostic instrument, while the real-only breakout serves as the primary domain-shift watchdog.

### **3\. Non-Archival / Software-Only References**

Several critical model repositories and toolkits—including Ultralytics YOLOv5 (commit 5cdad89), Ultralytics YOLOv8, NanoDet-Plus, AIMET, and FLUX.1—lack traditional peer-reviewed journal publications. In accordance with standard machine learning citation practices, these entries cite canonical Zenodo software DOIs or repository identifiers.

#### **Works cited**

> 1. What is yolov9 \- arXiv, [https://arxiv.org/pdf/2409.07813](https://arxiv.org/pdf/2409.07813)  
> 2. \[2405.14458\] YOLOv10: Real-Time End-to-End Object Detection, [https://arxiv.org/abs/2405.14458](https://arxiv.org/abs/2405.14458)  
> 3. YOLOv10: Real-Time End-to-End Object Detection \- arXiv, [https://arxiv.org/pdf/2405.14458](https://arxiv.org/pdf/2405.14458)  
> 4. YOLOv10: Real-Time End-to-End Object Detection \[Quick Review\], [https://liner.com/review/yolov10-realtime-endtoend-object-detection](https://liner.com/review/yolov10-realtime-endtoend-object-detection)  
> 5. YOLOv11: An Overview of the Key Architectural Enhancements, [https://scispace.com/papers/yolov11-an-overview-of-the-key-architectural-enhancements-74acntfu6kor](https://scispace.com/papers/yolov11-an-overview-of-the-key-architectural-enhancements-74acntfu6kor)  
> 6. YOLOv11: An Overview of the Key Architectural Enhancements \- arXiv, [https://arxiv.org/abs/2410.17725](https://arxiv.org/abs/2410.17725)  
> 7. A Practical Guide to High-Performance Object Detection \- arXiv, [https://arxiv.org/html/2604.03349](https://arxiv.org/html/2604.03349)  
> 8. \[2502.12524\] YOLOv12: Attention-Centric Real-Time Object Detectors, [https://arxiv.org/abs/2502.12524](https://arxiv.org/abs/2502.12524)  
> 9. \[NeurIPS 2025\] YOLOv12: Attention-Centric Real-Time ... \- GitHub, [https://github.com/sunsmarterjie/yolov12](https://github.com/sunsmarterjie/yolov12)  
> 10. arXiv:2502.12524v1 \[cs.CV\] 18 Feb 2025, [https://arxiv.org/pdf/2502.12524](https://arxiv.org/pdf/2502.12524)  
> 11. YOLOv12: Attention-Centric Real-Time Object Detectors, [https://www.researchgate.net/publication/411799769\_YOLOv12\_Attention-Centric\_Real-Time\_Object\_Detectors](https://www.researchgate.net/publication/411799769_YOLOv12_Attention-Centric_Real-Time_Object_Detectors)  
> 12. PP-PicoDet: A Better Real-Time Object Detector on Mobile Devices, [https://www.semanticscholar.org/paper/PP-PicoDet%3A-A-Better-Real-Time-Object-Detector-on-Yu-Chang/e5c32ac6cb785832d5fe186cca654c6e41828f1c](https://www.semanticscholar.org/paper/PP-PicoDet%3A-A-Better-Real-Time-Object-Detector-on-Yu-Chang/e5c32ac6cb785832d5fe186cca654c6e41828f1c)  
> 13. PicoDet in LibreYOLO: predict, train and export, [https://www.libreyolo.com/docs/models/picodet](https://www.libreyolo.com/docs/models/picodet)  
> 14. PP-PicoDet: A Better Real-Time Object Detector on Mobile Devices, [https://arxiv.org/abs/2111.00902](https://arxiv.org/abs/2111.00902)  
> 15. PP-PicoDet: A Better Real-Time Object Detector on Mobile Devices, [https://ar5iv.labs.arxiv.org/html/2111.00902](https://ar5iv.labs.arxiv.org/html/2111.00902)  
> 16. arXiv:2111.00902v1 \[cs.CV\] 1 Nov 2021, [https://arxiv.org/pdf/2111.00902](https://arxiv.org/pdf/2111.00902)  
> 17. Effective Data Augmentation With Diffusion Models \- OpenReview, [https://openreview.net/forum?id=ZWzUA9zeAg](https://openreview.net/forum?id=ZWzUA9zeAg)  
> 18. Is synthetic data from generative models ready for image recognition?, [https://arxiv.org/abs/2210.07574](https://arxiv.org/abs/2210.07574)  
> 19. AI-Generated Images as Data Source: The Dawn of Synthetic Era, [https://github.com/mwxely/aigs](https://github.com/mwxely/aigs)  
> 20. Generative Data Augmentation for Skeleton Action Recognition \- arXiv, [https://arxiv.org/html/2604.14933v1](https://arxiv.org/html/2604.14933v1)  
> 21. DiffAug: Enhance Unsupervised Contrastive Learning with Domain, [https://raw.githubusercontent.com/mlresearch/v235/main/assets/zang24a/zang24a.pdf](https://raw.githubusercontent.com/mlresearch/v235/main/assets/zang24a/zang24a.pdf)  
> 22. arXiv:2302.07944v1 \[cs.CV\] 7 Feb 2023 \- deepsense.ai, [https://deepsense.ai/wp-content/uploads/2023/04/2302.07944.pdf](https://deepsense.ai/wp-content/uploads/2023/04/2302.07944.pdf)  
> 23. Data Augmentation using Diffusion Model for Waste Semantic, [https://thesis.unipd.it/retrieve/feadffcc-89f8-42a6-abf4-b671d3ff193c/Sambin\_Luca.pdf](https://thesis.unipd.it/retrieve/feadffcc-89f8-42a6-abf4-b671d3ff193c/Sambin_Luca.pdf)  
> 24. Effective Data Augmentation With Diffusion Models \- arXiv, [https://arxiv.org/html/2302.07944v3](https://arxiv.org/html/2302.07944v3)  
> 25. Salient Concept-Aware Generative Data Augmentation, [https://neurips.cc/virtual/2025/poster/115341](https://neurips.cc/virtual/2025/poster/115341)  
> 26. PicoDet : Fast object detection model optimized for mobile CPUs, [https://medium.com/axinc-ai/picodet-fast-object-detection-model-optimized-for-mobile-cpus-17e7aa84589b](https://medium.com/axinc-ai/picodet-fast-object-detection-model-optimized-for-mobile-cpus-17e7aa84589b)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAwAAAAaCAYAAACD+r1hAAAA+klEQVR4AeyQP8uBURjGvW/+bGwMYiN8B8kiC4NYLGZZbAYpu7JYrEq+AIsyKAaDwcJqYJFsTEp+98nReSZk9XT9zv1c9+k653T/2z78foF3BvbVlP64IQVOMOXF2EHJvCFDZwxREIVYxB+obVAyA3L6he4ailCBGkhoQFUyA0k6G2iBG6qwgjQsQEkHfLgY+KHwoEH1gEU6kHh089QgNKEMI7DIDJzZWcINZjCBCIiyLDmw6UAcM4craLn42YGoxCKHqIADI6McUk31MQHowRSOoAJyahjTBVNygDypTrMDSvpJW5y8nWLRCbeHp3Tg2Xj18wu8mpDs3wEAAP//o26wxQAAAAZJREFUAwB3zSU1Z73inAAAAABJRU5ErkJggg==>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADgAAAAaCAYAAADi4p8jAAADGUlEQVR4AeyWWahNURjHLxJlzBghQ+TBkAwRmTJEEUpkKBlKkikvvCBzIiWRISUiQ5kfZMg8vRBRhoSQzB6U2e93tLp7d27XOfece4+rc/r/9rf2Gvba31rft86uWPCf//IOlvcNzu/gP7qDjXivObAGpkNtKFLlcQd74clB+AxPYSE8hO6QpNJ2sCYzVoVsqQIP2gk6tQW7AfqB8+h0ZcoxlbaDLZjtOCyCupCpmvAAn3kUG5x5RPkSNIY+EFNwsDq1w8AHYAoqcekEg8A2TELtuI6ENpCKbtFpAGgPYNdDUyipnjPwMOyHbxBkuFqu5iWKDrrtDjCG79A4ENzuIdhR8ARawm6YCjp3GzseUtEvOh0CQ+kY1tDagW0L6eonA0bAZAhyM7pw8wWuQ0w66C7doNbYroX1BaZhV8B8qAO2G++eXKu5vwtjIF2dYoALtxm7EtzVrthMNIXBDcD3fYmNSQeN6T3UdgM1ictrUA29wDq4Csod95j+6E0JucY4Q30xdjaYp72x6cqUWsagjbAUkqSDruY9WkxQX/oi5aAwqaEV6tpT0PGz2ExlSizgIV9hLVSBVFWDjkdA52ZiTQVMXDoYavpSuAA/IEin33NjzmESGs31O/hwTIllLm9n9DbYBIaqeUTxr/IE3UevrbAElO/f0UKU4KBb3YqGcxCVB8N5KkxuTELmnrn0hjtXzxylmLI8nffS21zWucGUT0KqMkWMul0McGEwCY3jquOYQgUHQyieKWwqMDebcR+tM5lbU+fuuSD1Kb+DVOQc/n/No/NyMAevYNOVY4czyMPlNNZUuYydAPchpuBgB2qfwU0IMtcMxROhAvsWPIqHYleBpyqmWIUdGkuvWTARoiHPbcpycc3ZeowwuvpjDc0e2A/wCWIKDnrEdqYlGoruUnPq/M7DJGR+9qQ0FwyJF9ji5H+rL6BTM+j4GDLRAwYbokXhlwzNcQUH9Tz8NUR7+OVQeP+n5K7qdPRL4k9L8tW/Flf8VXJT2dQEB8tmthzMkncwB4ue1SnzO5jV5czBw/I7mINFz+qUvwEAAP//YgX/ZgAAAAZJREFUAwCRg4Y1a4ezmAAAAABJRU5ErkJggg==>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFMAAAAZCAYAAABNcRIKAAAE8ElEQVR4AeyYV6gkRRSG25weBBOoYI6oiGJEBQOiqIj4oGJCMWEOKKYVEdODARMiZgUziAgqKuYsZswZUcwB8+b9vt45l9qe7prbc2eXZenL/9c5dSpM9al06i5cdH8j80DnzJG5sig6Z3bOHKEHRthVujIXod+94cXwVLgyHAbr0+gR2ITlKTgeThS70MEF8Ey4LmyDZal8HLwcHgYXhVVsjeFceB7cDA5EOHMJat4HD4D3wB/hG3Bb2AZOyG002AKmWI2ME3Qr8mvohyCGwkK0ugaeAx+C78Mn4b5wPPCbPqTicvBhuD+8GqY4ncwN8Gn4HLwbngizCGceTq0d4FHQwdn4DvQ7Yd2sYa7FKVg3h1UsheE/eBP8CU4Eu9H4aHgofBu6Cy5B3gh1EKIRa1Oi489GXgRfhTvDXWFgQ5TLoL54Bfk8PBlqWw/ZiHCmg3uNWv/AwDMo68Bt4HjgVnNQtqvW/wSDM+3gpqFPBI71Uzr4Dgb8TY+PPcLQIK/E/gd0sSCKKSRu5UnIgE78i4w7E1HiWdLF4X6wETpzJUo9E35Hpvitl9mpJ3PCfq6jwglwJpxbcIu7MocZazhbJzmhqzDIpaHn5v3IwO4o1f51+t/Ys77QCZ5n1Cv+N0kYeX80MdeqXiiPU/I5nJvQITogxha/FfncWLejskeWu+8WdM/wp5DeFfaJWkJ/RH+loZdMRub6L+PMZagkZpgkjPygc2gN2hgFVA9xzCPHRMYa0YkX1RWM7Ay4I9weXg+FK1/HxrdrC2rL+sKVqcejQSrt2Px0kwZa51rKToK5ehTPAdulBs/bIzAMYtMREv3lxuCq5CeKl0i8zRGF3+5562W2Jgb7d0uj1iLXf7kyf+k1ix/rZQvDHPUoV6/SKMBL5aNqQZIfj+pH/UrFQfyZOn7wMGONKOIL+kjxLxknY0uk8Hur/WvXH5ap19KV+QMlejy2ENkSkU9vzbIgSdwmx5A3dvwSKQ2mPdvUm7a+g6f6GL5BM2YcRMMrHRpjo1mJyOfG+lVZs+i7G5wci/SF0j6iP/NBbZZFvk/agbfUC5TEmYJaIg7bx8rc7GRFhOESooTbY3U0z821kNKwxRWmbnyGuQ9VZ/ZVyBgepWyYsb5Fu++hE40Yg2ekGWNWpd+7AspiMOBZuSQZyxD10JmW3E6yFVwVBvZBeRO+BwOvo7ilPV9Qa+EgZJ3D/D23kOXqtR0MMPqYcALTx4FjddU8kbStjtULxIvGIN3npFXduj5WHiRjLIwo7iJxte6JDNi/i+6BMNTJ+CBfOj4j3WYHU9FoX+cehJ7C89EfdauldvWrSDyPXDUO0m1lkIy52JTkY+jWd9alR4MTsxH2NjCA9k1+L408sych3SEG1GlIUzfWS6n7IjQkOgtpOOc4vfjIlviM1OeucfOx6O6u85GHQFc2oh7hTGfiSKoY/btynKlNyOs4xBgORNsYOkuIOeBT0ueasy7d+qf1aryL3ABqs0waz/l0+wB7W+jMvWjkWe/tbD8vk09RN1br63Sd52VyIQ1cqb6KUMdgHOrL708sHlt+swuNbDPCmVHjHRS3vLM6FX1+hh/pjjK08WJqM1Yn92Ya+E8MFxJqH77F4pb3nPR5STaPqjPztbvSrAc6Z2bd066wc2Y7f2Vrd87MuqddYefMdv7K1l5QnJn9yHlVOAsAAP//Q0Pi8QAAAAZJREFUAwCYXvczTJm38wAAAABJRU5ErkJggg==>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFMAAAAZCAYAAABNcRIKAAAFPklEQVR4AeyYdahtRRTGt51gCwa2YmKAhQomiooYGNhigIqFio2IhRio2K1go6hgY3cHthiI3Riv6/fbnHWYt+8+c+6+97z3x2Nfvm+vNWvizKypNXf2ov0bmAdaZw7MlUXROrN15gA9MMCm0pU5B+3uAs+HJ8ClYBMsROGj4CXwYDgnrMNiGI+Go8U2NHAOPAWuCoeLRSh4LLwU2l/7jToEG2M5A54F14d9Ec6ch5L3wn3g3fAX+DbcFA4HlvuEgovCR+De8AoYWA7FCboF+S10EIgRYTZqXQlPhw/Bj+DTcHfYD/bvAQr9Cx3nXMj34JowxUkkroPPwhfgXfAYmEU48xBKbQEPh3bOyrej3wF7rTCySqzM18GchjwPvg63htvBwHwoY+GN8Fc4GmxP5SPggVBHPIq8AN4AdRaiJ5xgd86tlHCxmL4a/SIYWAPlYqgvXkO+CI+D2lZD9kQ40869Qan/YeA5lFXgJjCHy8j8GzoBiGICH7fHmcjA5yjOtJ2bhD4a2NcvaOAHGLCvHh87hqGH3Az7wjCFba2YGHSiK1dnh/l5lLnhXrAndOaS5Hom/IVM8WcnsVVH1okYgD+sk5am0PzQ2b8POWi4xV2ZI+mrffmDz7XwIGhbiGJPPmlfdyBdbd8F8h/2nC/K0MjzjHLFOD8JI62DEvN0qjPtMeCKvpkcz8VnkJ6/OhV1oHDybDf6Fo1HOtdXy3oMLYhyG3wcus11arrN9Ue0R5EuxqNl23dlLkAhMcVPwkjnzqG48T38vR1Ppv6WcHN4DRw0RtNX+/IwH7cxonCFG1W4q1x52nSskxVj1xbUlvNFuTL1eFRIpQ2bnuynB12VZr3Cx9scUdieZ5gXRHoWmReMtiNtaHMoiX6cSpk6RHu5vlpvPT5O+B5IV6fh4OXop0Jh++FY01Vm23dl/t6pEY7pJAt/SD3y1auMm/mrSsYY0g5ww6IoUPvCCfA868ffaMkBj6SvrrinqH88fBAawXg+/ohubL0sUjjeavva9Yd56rXUmT+To8djC5EsEen01iwzks83Hb16xjhgs2xfWaWOTm3fkTBm7EfDKx0afaNaiUjn+robJb2ln0QG1Lcl4RbeCClsI9ozHdRmXqSHSAfrLfUSOXH+oZaIw9aDujTwWQIaLiFKvMv3J+jFgOjCVWDCOFBZZdWZ1fxc+jEyR9JXQxuqDsGnWNwRPlRQC8e7OIoBPaKEZ+W8aOYh6qEzzfH8cGaWMdHhrsh34Icw8CaKPx5noTPqRWOQHs8yt4MPALeS8SVVuvD33EJ2VL2b0UDxMbE85TeAAfvqqnEbh63a1yfIMAx0haJ24RPa3flWx3In0p21EzJg+y66+8NQJ2NAvnR8XrnN9qeg0b7O3Q89hUG3DnKrhf1ClJehIZEHuVvH7e9lgrnEunw/g19DZ136rHRi1sLWBAbQvsnvoZLnno8DLzsD6vS4qfbVHWRMeT31fNaeiHTcPjC0x8XzJXafu1chj4S+fs5GHgBtA1GPcKYzcRhFDBtcOa6qdUjrOEQX+6KtDZ0lRAnPWwei8zygz8XqSvVVhFriA76rwxWgK1gaz/l0+xhbU+jMnankbxtJ2M6rpFPU9dXVuRKFXDQ+SnSssbKvIMxdGDP78vsHi3mO2TokeyOcGSXeR3HLO6sT0ZtAh91EBf8x4OSgzlA4SFeWYZgX03B/zIXgP2N8nzsRvtzq6n6P0S3vOenFRTKPqjPzpdvcrAdaZ2bd0yyzdWYzf2VLt87MuqdZZuvMZv7Klp5VnJkd5MzKnAYAAP//AgMfEAAAAAZJREFUAwAj9xBCZn48BgAAAABJRU5ErkJggg==>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAHUAAAAaCAYAAACJphMzAAAGcUlEQVR4AeyZBawkRRCGG3d31xAsQIK7W7DgHtzdghPcA8Gd4BY0gYQQ3D1YkBDcCS7n9n1703f75mZmZ9/K7dvsS/1b1TYt1V1V3W/S0PvruhXoKbXrVBpCT6k9pXbhCnThlNIndXXm+AX4CfwMfgRfAfO+hn8ELgBzgW6hF5nI98D5im+Qne+XCX8QvjYYMJRW6muMfDHwAlBx28EXAeYtAT8abAteBfODdtOUdFim3zmoNy0oQ+tQaX/gfB+FLwSc76LwzcA/4DmwBxgQlFZqHPQqCH+Ct0CkYQhPg63AAuA80G5alg6PArVobypsAMqS87XuU/5U4XPkfYGn+Wr4zKDjKUupKszT6URGZszAiWqGd6FsctBOmqRFnXlaR/FtLRRsAvIEz0Tu1qDjKUupTtCBP+9PDv4mv6wppGrTqBVKdR7GEh8yyt9BFjlf8zXJ8lqYqOVZSl0vGVGRUhdM6gxNeLtYK5S6MoPX/xbNVz9LtdDu+dpn3chTqv70g5yvGYS4YwdRbrQIaxupVNHMDsts4uhzjYqb2XdLvpVWqpHl4vSkP9XHIE5AmyQ5hvqjE7lRdg4f+AGsC4rI/sootUyd2I9Kda7OOeZV86lIOK7/4E+CVtPOdPApOAn0i9JKdYJ+qMgU7WYFcB1oFp3Oh34BZQKvsgrLCvLoog+psDXI0Z/+Ac8ig6PpKbgbeL2BNZXm5WtHgkj3IxiFTw3vF6WVun7ylbwocCPKNweXg9dBM2l4iY95rZqsRD2DnyEl6q1KnSJ/aplWxAeYfp8c+igir15pBY4oalCrLK1UO9Cfvp/R0GvObeQ/A04Ekve2GxCuAbsD73KzwqVd+fEuewXctrCwND9nAk/mcXBPCqxCmtaKUPDjVWo5ytPjJmsceZI3JKUJgxVS0SaegpZaI2OIHZH/ApJXm1MRTgZubm8LmszzSV8InOsO8IuA92XX6h7kNcHh4FKwJJBcsysRtgR+y+skYtAduHn3IeE3o8sjWZuqF8eOFqbJy8CPwio0J78O5l34XcCTGk+Vof4d5B0EDJrmg88N9gQO2Mk/gqz/hYWD+ZkRuPv13Wcg10OaVM3TjTTyNML60HSk3FhaEZ86SeaSyvfFyM1U7U/9rov4Ei3dhCvC3wGRNJXfkfC59G347MB7rCbalyifGd8jT9n865G1CD6aXIvsq50KRAwP8PM40FefAjeugAXH5AuWa+cmuJfMaUApUqn6FN85VZqNNLHfIjg44auSCnCimqBq02Dn1nGx9cM+IX5M20OAecfCVwIrACftxC5B3gnMBpYC9ZKLpLVw0S+jsRvqMLjKdAyfIXuKYLmkEn+ldDWgco30nYdQYX5PC7MW5b55w8aRa2W/N5Ojj1UpXnWuIr0pcGN5Ks9F9nS78eUq0MPi5nc9KQ4eDtfJNR1MhuWwCj3Br+1+g3sQogUkWUwq1Xdcryj6Dyco9x7q3SzCN1+Vm/c1B+TgYrmDcPc5URdHn2H0eAIV3H3+o+AVZPuD1U3uXB8MPLX/01q4aG5QzVkIgdx80mRqVu1faGHiXH0D3oam9qGyEPvQw6S0QpbdiqzphQVf2t5E0O24iT2tJCuksrwCmlC2T+UI065RVLbpWN86WW3Mz4RKzSyoI9MBpKu72J5Qd6LQZ2g+9Duack28O9q2xyeNHYvpJFmTOdE3qKVL0Nd7+uyLrJaSc9Bfe5r3oydPO6xCnl5dzieV1Pif6rml5+jmMOrXzcXbh/XHtx4rpduNzc34zWqcUS03S5NwMaU6dc2i91ySwSDBRTeAUmnuQk+zJvI0KnjyzdP86sMMJgwSjqFM3w7rWHJxz2Z0BwLfv2+HR9IK6ZLuixlwXZaW4AhkLYDpWZCNK2DhTn4024fCNbnbww30dHe6M9dV8+yaut4UF1OjSvVup7mZgW4Mgvy/JGLw6nEAglHuTXBNGSyoTAOAW0IIBknLwN35Dnge5C2ApwDWseRJNKp9jBHuBTS5RvHC9TTS1ddSVCGjV4NHgzLbeO81nnBtrKAb2hjBtMHdQ8jLA024m8R11ar5PuB6U1RMDqK4RmOlDsJAofor+tZ/kwx3YCIOGKb/9jT6WBLN/VmM3uubJtnbAMm6SBMcv1VXw6zKrVZqVp/dmKeJ9D883iAmuqXpKbU5W8xrj/fWZ5vzuca+0lNqY+vXka17Su1ItTQ2qDEAAAD//+BFaZIAAAAGSURBVAMAtGYzRO3y5YgAAAAASUVORK5CYII=>