## Retina U-Net: Embarrassingly Simple Exploitation of Segmentation Supervision for Medical Object Detection

Paul F. Jaeger 1 , Simon A. A. Kohl 1 , Sebastian Bickelhaupt 2 , Fabian Isensee 1 , Tristan Anselm Kuder 3 , Heinz-Peter Schlemmer 2 , and Klaus H. Maier-Hein 1

1 Division of Medical Image Computing, 2 Department of Radiology, 3 Medical Physics in Radiology, German Cancer Research Center, Heidelberg, Germany

## Abstract

The task of localizing and categorizing objects in medical images often remains formulated as a semantic segmentation problem. This approach, however, only indirectly solves the coarse localization task by predicting pixel-level scores, requiring ad-hoc heuristics when mapping back to object-level scores. State-of-the-art object detectors on the other hand, allow for individual object scoring in an end-toend fashion, while ironically trading in the ability to exploit the full pixel-wise supervision signal. This can be particularly disadvantageous in the setting of medical image analysis, where data sets are notoriously small. In this paper, we propose Retina U-Net, a simple architecture, which naturally fuses the Retina Net one-stage detector with the U-Net architecture widely used for semantic segmentation in medical images. The proposed architecture recaptures discarded supervision signals by complementing object detection with an auxiliary task in the form of semantic segmentation without introducing the additional complexity of previously proposed two-stage detectors. We evaluate the importance of full segmentation supervision on two medical data sets, provide an in-depth analysis on a series of toy experiments and show how the corresponding performance gain grows in the limit of small data sets. Retina U-Net yields strong detection performance only reached by its more complex two-staged counterparts. Our framework including all methods implemented for operation on 2D and 3D images is available at github.com/pfjaeger/medicaldetectiontoolkit.

## 1. Introduction

Semantic segmentation algorithms, such as the U-Net architecture [22], constitute the de facto standard approach for detection of anatomical structures in clincial context [6, 14, 12, 26, 24]. This is owed to the circumstance that annotation of medical images is commonly performed by delineating the structures of interest resulting in pixelwise annotations, which are optimally exploited in the segmentation scenario learning from per pixel supervision signals. Furthermore, MRI and CT imaging capture 3D spaces, inducing inherent spatial separation of objects, therefore not requiring the discrimination of (overlapping) instances. While in tasks like radiation therapy planning or tumor growth monitoring pixel-wise predictions are clinically required, in most settings coarse localization or mere knowledge of object presence are of relevance. This relevance is optimally reflected in study design, when evaluating models on an object-level. In order to bridge the discrepancy between pixel-wise predictions from semantic segmentation and object-level evaluation, however, ad-hoc heuristics or additional models have to be introduced. By extracting predictions from coarser representation levels to enable end-to-end object scoring one naturally converges towards the architecture which most current object detectors are based on: The feature pyramid network (FPN) [15]. Two methodologies are applied for generating object-level predictions from coarse representations: two-stage detectors, where first objects are discriminated from background irrespective of class, accompanied by bounding box regression to generate region proposals of variable sizes [10, 17, 8]. Subsequently, proposals are resampled to a fixed-sized grid to ensure scale-invariance for categorization. On the other hand, one-stage detectors have been proposed, where categoryspecific classification is performed immediately on the coarse representations [21, 18, 16]. Predicting objects on coarse levels, however, comes at the prize of coarsening pixel-wise annotations to bounding boxes (or cubes). This conversion entails information loss which contradicts the need for data-efficient training on comparably small data sets in the medical domain. We show that fully exploiting the available semantic segmentation signal results in significant performance gains for object detection tasks on medical images. While previous work in the non-medical domain has pursued this concept [23, 19, 20, 3], the extra supervision signal is either used sub-optimally or combined with approaches of arguably significant model complexity. The applicability of computer aided detection systems in clinical environments however, among other requirements, hinges on their interpretability and robustness, which intuitively decreases with model complexity. Furthermore, we argue that the explicit scale variance enforced by the resampling operation in two-stage detectors is not helpful in the medical domain, since unlike in natural images, scale does not depict artifacts caused by varying distances between object and camera, but encodes semantic information.

Figure 1. Example predictions of a malignant lesion in a CT scan of the lung (left) and a benign lesion on a Diffusion MRI of the breast (right). Object detection in medical images is challenging due to small objects, large images, and limited training data. We tackle this problem by exploiting semantic segmentation supervision in a simple architecture.

<!-- image -->

Towards the goal of both model simplicity and optimally leveraging available supervision signals, we propose Retina U-Net, a simple approach to recapturing full semantic segmentation supervision based on Retina Net, a plain one-stage detector. Inspired by the U-Net, a very successful model for semantic segmentation of medical images [22], we complement the top-down part of the Feature Pyramid Network by additional high resolution levels to learn the auxiliary task of semantic segmentation. From a segmentation perspective, we retrofit the U-Net with two sub-networks operating on the coarse feature levels of the decoder part to allow for end-to-end object scoring.

We demonstrate the effectiveness of our model on the task of detecting and categorizing lesions on two data sets: Lung-CT (publicly available data [2]) and BreastDiffusion-MRI (in-house data set described below). We support our analysis by a series of toy experiments that help shed light on the reasons behind the observed performance gains. The proposed model, Retina U-Net, shows results superior to a U-Net like segmentation model with detection heuristics as well as the prevalent object detectors without full semantic supervision: Mask R-CNN [10] and Retina Net [16]. In order to provide a fair and insightful comparison, we enhance the two-stage baselines such that they can be trained with a full segmentation signal and show that Retina U-Net is able to hold up against such more complex models. This paper makes the following contributions:

- A simple but effective method for leveraging semantic segmentation training signals in object detection focused on application in medical images.
- An in-depth analysis of the prevalent object detectors (operating in 2D as well as 3D) by means of comparative studies on medical data sets.
- Weighted box clustering: An algorithm to consolidate object detections across different predictions of the same image in 2D and 3D.
- A comprehensive framework including e.g. modular implementations of all explored models and an efficient implementation of weighted box clustering.

## 2. Related Work

Since object detection in natural images is increasingly formulated as an instance segmentation problem, several two-stage object detectors learn to predict proposal-based segmentations [10, 4, 17]. However, we argue that this setup does not fully exploit semantic segmentation supervision:

- The mask loss is only evaluated on cropped proposal regions, i.e. context gradients of surrounding regions are not backpropagated.
- The proposal region as well as the ground truth mask are typically resampled to a fixed-sized grid (known as RoIAlign [10]).
- Only positive matched proposals are utilized for the mask loss, which induces a dependency on the region proposition performance.
- Gradients of the mask loss do not flow through the entire model, but merely from the corresponding pyramid level upwards.

Auxiliary tasks for exploiting semantic segmentation supervision have been applied in two stage detectors with bottom-up feature extractors (i.e. encoders) [23, 19]. In the one-stage domain, the work of Uhrig et al. is the one most similar to ours: Semantic segmentation is performed on top of a single-shot detection (SSD) architecture for instance segmentation, where segmentation outputs are assigned to box proposals in a post-processing step [25]. Zhang et al. propose a similar architecture, but learn segmentation in a weakly-supervised manner, using pseudo-masks created from bounding box annotations [28].

Figure 2. The Retina U-Net architecture in 2D.

<!-- image -->

As opposed to bottom-up backbones for feature extraction, we follow the argumentation of feature pyramid networks [15], where a top-down (i.e. decoder) pathway is installed to allow for semantically rich representations at different scales. This concept is adapted from stateof-the-art segmentation architectures [22, 5] and used in both current oneand two-stage detectors. Building on FPN-based two-stage detectors the winning entries of the COCO object detection challenge 2017 [20] and the COCO instance segmentation challenge 2018 [3] report to have used additional semantic segmentation supervision in increasingly complex backbones like FishNet [3], without disclosing implementation details at the time of submission (e.g. the resolution at which segmentation is learned remains unclear). In contrast, we propose a FPN-based one-stage detector, which allows to naturally fuse object detection and segmentation, resulting in the simple Retina U-Net architecture.

## 3. Methods

## 3.1. Retina U-Net

Retina Net. Retina Net is a simple one-stage detector based on a FPN for feature extraction [16], where two sub-networks operate on the pyramid levels P 3 - P 6 for classification and bounding box regression, respectively. Here P j denotes the feature-maps of the j th decoder level, where j increases as the resolution decreases. To factor in the existence of small object sizes in medical images, we shift sub-network operations by one pyramid level towards P 2 - P 5 . This comes at a computational price, since a vast number of dense positions are produced in the higher resolution P 2 level. We further exchange the sigmoid non-linearity in the classification sub-network for a softmax operation, to account for mutual exclusiveness of classes due to non-overlapping objects in 3D images. For the 3D implementation, the number of feature maps in the network-heads was reduced to 64, to reduce GPU memory consumption.

Adding Semantic Segmentation Supervision. In Retina U-Net training signals from full semantic supervision are added to the top-down path by means of additional pyramid levels P 1 and P 0 , including the respective skip connections with the bottom-up path. The resulting Feature Pyramid resembles the symmetric U-Net architecture (see Figure 2), which in the following we refer to as U-FPN for clarity. The detection sub-networks do not operate on P 1 and P 0 , which keeps the number of parameters at inference time unchanged. The segmentation loss is calculated from P 0 logits. In addition to a pixel-wise cross entropy loss L CE , a soft Dice loss is applied, which has been shown to stabilize training on highly class imbalanced segmentation tasks e.g. in the medical domain [12]:

<!-- formula-not-decoded -->

where u is the softmax output of the network and v is a one hot encoding of the ground truth segmentation map. Both u and v have shape I × K with i ∈ I being the number of pixels in the training batch and k ∈ K being the classes. Since dice scores are determined on a per class basis, images with zero foreground pixels in one class lead to unstable scores, where false positive predictions are not penalized. To alleviate this problem, we compute the dice scores over a pseudovolume consisting of all images in one batch.

## 3.2. Weighted Box Clustering

Medical images typically are comparatively large, due to e.g. very high resolutions as in mammograms, or since they are acquired as 3D volumes like in MRI. Image resolutions are expected to keep rising in the future driven by advances in imaging technologies, such as the recent introduction of 7T MRI scanners. For this reason models are trained on patch crops, resulting in a trade-off between patch size and batch size limited by available GPU memory. If a single image exceeds GPU memory, inference is performed patch-wise as well, where tiling strategies are designed to avoid potential artifacts that arise due to effects at the patch boundaries (e.g. by allowing for sufficient overlap between patches). The tiling strategies as well as test time augmentations and model ensembling can amount to a large number of predictions per patch and image (particularly in medical object detection, where validation metrics for model selection are often based on limited validation data, ensembling over multiple selected epochs is able to reduce noise effects in this process). The resulting predictions for different views of the same image need to be consolidated, which in semantic segmentation is done via simple per-pixel averaging. For consolidation of predictions in object detection, we propose weighted box clustering (WBS): Similar to the commonly used non-maximum suppression algorithm, WBS clusters predictions to be consolidated according to an IoU threshold, but instead of selecting the highest scoring box in the cluster, weighted averages o c per coordinate and a weighted confidence score o s per resulting box are computed. Further, the prior knowledge about the expected number of predictions at a position (number of views from ensembling, test time augmentations and patch overlaps at the position) is used to down-weight o s for views that did not contribute at least one box to the cluster ( n missing ):

<!-- formula-not-decoded -->

where i is the cluster members' index, s and c the corresponding confidence scores and coordinates. w = f · a · p is the weighting factor, consisting of:

- overlap factor f : weights according to the overlap between a box and the highest scoring box (softmax confidence) in the cluster.
- area a : assigns higher weights to larger boxes based on empirical observations indicating an increase in image evidence from larger areas.
- patch center factor p : down-weights boxes based on the distance to the input patch center, where most image context is captured. Scores are assigned according to the density of a normal distribution that is centered at the patch center.

An example for application of WBC is shown in Figure 4.

## 4. Experiments

## 4.1. Backbone &amp; Baselines

In this study, we compare Retina U-Net against a set of one- and two-stage object detectors. We evaluate the performance of all models processing the data in both 2D (slice-based) and 3D (volumetric patches). For the sake of unrestricted comparability, all methods are implemented in one framework, using a FPN [15] based on a ResNet50 backbone [11] as identical architecture for feature extraction. The anchor sizes are divided by a factor of 4 to account for smaller objects in the medical domain resulting in anchors of size { 4 2 , 8 2 , 16 2 , 32 2 } for the corresponding pyramid levels { P 2 , P 3 , P 4 , P 5 } . In the 3D implementation, the z-scale of anchor-cubes is set to { 1 , 2 , 4 , 8 } factoring in the typically lower resolutions along the z-axis.

Retina Net . The implementation of Retina Net is identical to the one used in Retina U-Net and described in 3.1 (see Figure 3c).

Mask R-CNN . Minor adjustments have to be made for the 3D implementation: The number of feature maps in the region proposal network is lowered to 64 to reduce GPU memory consumption. The poolsize of 3D-RoIAlign is set to (7, 7, 3) for the classification head and (14, 14, 5) for the mask head. The matching IoU for positive proposals is lowered to 0.3 (see Figure 3a).

Faster R-CNN+ . In order to single out the Mask R-CNN's performance gain obtained by segmentation supervision from the mask head, we run ablations on the toy data sets while disabling the mask-loss, thereby effectively reducing the model to the Faster R-CNN architecture [9] except for the RoIAlign operation (which we denote by a +) (see Figure 3b).

U-Faster R-CNN+ . As the currently leading COCO challenge architecture's exact implementation details are presently undisclosed [3], we explore the performance of additional semantic segmentation in two-stage detectors by deploying Faster R-CNN+ on top of U-FPN. (see Figure 3d).

DetU-Net . Essentially formulating the problem as a semantic segmentation task, as is common on medical imaging, we implement a U-Net-like baseline using UFPN. Therefore softmax predictions are extracted from P 0 via 1x1 convolution and utilized to identify connected components for all foregorund classes. Subsequently, bounding boxes (or cubes) are drawn around connected components and the highest softmax probability per component and class is assigned as object score. To reduce noise, only the 5 (15 in 3D) largest components per image are considered (see Figure 3e).

Figure 3. The upper panel shows all baselines utilizing a regular FPN feature extractor while the lower panel depicts baselines that employ a symmetric FPN feature extractor akin to a U-Net (U-FPN). Subfigures a) - e) show the detection sub networks (heads) that are characteristic of each model and operate on FPN features. All models employ their respective head topology to different decoder scales which are denoted in red. Boxes in green indicate logits that are trained on an auxiliary semantic segmentation task.

<!-- image -->

## 4.2. Training &amp; Inference Setup

For comparability, experiments for all methods are run with identical training and inference schemes. Since experiments are performed on 3D images, several possible ways of processing the volumetric data arise. In this study, we compare slice-wise 2D processing, slice-wise 2D processing feeding the ± 3 neighbouring slices as additional input channels (2Dc) [27], and 3D volume processing (i.e. using volumetric convolutions). Oversampling of foreground regions is applied when training on patch crops. To account for the class-imbalance of object level classification losses, we stochastically mine the hardest negative object candidates according to softmax probability. Models are trained in a 5-fold cross validation (splits: train 60% / val 20% / test 20%) with batch size 20 (8) in 2D (3D) using the Adam [13] optimizer at a learning rate of 10 - 4 . Extensive data augmentation in 2D and 3D is applied to account for overfitting. To compensate for unstable statistics on small data sets, we report results on the aggregated inner loop test sets and ensemble by performing test-time mirroring as well as by testing on multiple models selected as the 5 highest scoring epochs according to validation metrics. Consolidation of box predictions from ensemble-members and overlapping tiles is done via clustering and weighted averaging of scores and coordinates, as detailed in Section 3.2. Since evaluation is performed entirely in 3D, a adaption of non-maximum suppression (NMS) is applied to consolidate box predictions from 2D networks to 3D cube predictions: Boxes of all slices are projected into one plane while retaining the slice-origin information. When applying NMS, only boxes with direct or indirect connection to the slice of the highest scoring box are considered as matches. The minimal and maximal slice numbers of all matches are assigned as z-coordinates to the resulting prediction cube.

Table 1. Results for lung lesion detection on CT.

| Dim.   | Model        |   mAP 10 [%] |   AP pat m [%] |
|--------|--------------|--------------|----------------|
| 3D     | Retina U-Net |         49.8 |           70.4 |
| 3D     | Retina Net   |         45.9 |           68.8 |
| 3D     | DetU-Net     |         36.6 |           62.8 |
| 3D     | U-FRCNN+     |         50.5 |           70.7 |
| 3D     | Mask R-CNN   |         48.3 |           71.8 |
| 2Dc    | Retina U-Net |         50.2 |           73.9 |
| 2Dc    | Retina Net   |         48.2 |           71.5 |
| 2Dc    | DetU-Net     |         41.1 |           66.1 |
| 2Dc    | U-FRCNN+     |         49.1 |           71.6 |
| 2Dc    | Mask R-CNN   |         45.4 |           69.1 |
| 2D     | Retina U-Net |         40.8 |           68.0 |
| 2D     | Retina Net   |         39.5 |           67.7 |
| 2D     | DetU-Net     |         29.0 |           59.8 |
| 2D     | U-FRCNN+     |         38.6 |           66.4 |
| 2D     | Mask R-CNN   |         35.3 |           63.6 |

Table 2. Results for breast lesion detection on Diffusion-MRI.

| Dim.   | Model        |   mAP 10 [%] |   AP pat m [%] |
|--------|--------------|--------------|----------------|
| 3D     | Retina U-Net |         35.8 |           88.0 |
| 3D     | Retina Net   |         31.9 |           86.4 |
| 3D     | DetU-Net     |         26.9 |           85.1 |
| 3D     | U-FRCNN+     |         35.1 |           86.5 |
| 3D     | Mask R-CNN   |         34.0 |           84.8 |
| 2Dc    | Retina U-Net |         33.4 |           86.9 |
| 2Dc    | Retina Net   |         33.2 |           84.4 |
| 2Dc    | DetU-Net     |         25.8 |           81.6 |
| 2Dc    | U-FRCNN+     |         33.2 |           84.7 |
| 2Dc    | Mask R-CNN   |         32.3 |           86.4 |
| 2D     | Retina U-Net |         31.8 |           85.2 |
| 2D     | Retina Net   |         32.3 |           86.1 |
| 2D     | DetU-Net     |         22.3 |           80.1 |
| 2D     | U-FRCNN+     |         33.1 |           83.7 |
| 2D     | Mask R-CNN   |         33.6 |           86.8 |

## 4.3. Evaluation

Experiments are evaluated using mean average precision (mAP). We determine mAP at a relatively low matching intersection over union (IoU) threshold of IoU = 0 . 1 . This choice respects the clinical need for coarse localization and also exploits the non-overlapping nature of objects in 3D. Note, that evaluation and matching is performed in 3D for all models and processing setups. Patient-level scoring is often used in the clinical context for staging and treatment and is therefore sometimes used for model selection. In our setting the patient-level metric is to be taken with a grain of salt: By disregarding whether box predictions match the ground truth, it is blind to the issue of 'being right for the wrong reasons' and can further over-estimate performance due to class-imbalance on the patient-level. For the purpose of comparability, we however also report patient-scores in Table 2 which are determined as the maximum of predicted scores per class and patient and compute the AP thereof.

Figure 4. Example application of the weighted box clustering (WBC) algorithm. (a) lung CT scan containing a benign lesion marked in green. (b) box predictions of Retina U-Net trained in 2D with context slices: 4 mirror settings from 5 tested epochs lead yield 100 box predictions in the two foreground classes. (c) Remaining predictions after application of WBC: The true lesion is predicted correctly as benign (class 1) with 93% confidence, surrounded by one false positive prediction for class 2 (malignant). (d) Remaining predictions after applying Non-maximum suppression instead of WBC: The benign lesion is predicted correctly, but multiple false positives remain in the image with confidence scores as high as 83% .

<!-- image -->

## 4.4. Lung nodule detection and categorization

We consider the task of detecting lesions and assigning them to one of two categories, benign or malignant . This is both a difficult and very frequent problem setting in radiology and therefore constitutes a highly-relevant domain of application with its own characteristics that our approach factors in and addresses. Furthermore, fine-grained categorization is expected to gain relevance in the context of growing data sets and image resolutions.

## 4.4.1 Utilized data set

The lung nodule detection and categorization task is performed on the publicly available LIDC-IDRI data set [2, 1, 7], consisting of 1035 lung CT scans with manual lesion segmentations and malignancy likelihood scores (1-5) from four experts. We aggregate labels across raters per lesion by applying a pixel-wise majority voting on the segmentations and taking the mean of malignancy scores. Scores are then re-labelled into benign (1-2, n=1319) and malignant (3-5, n=494). CT scans are resampled to a resolution of 0 . 7 × 0 . 7 × 1 . 25 mm , which roughly corresponds to the mean resolution of the data set. For training, patches of size 288 × 288 (for 3D training: 128 × 128 × 64 ) are sampled.

## 4.4.2 Results

Results for the lung lesion detection task are shown in Table 1. Retina U-Net performs best on the 2D setups ( 0 . 41 mAP and 50 . 2 mAP with context slices) and only slightly worse ( 0 . 50 mAP ) than its two-stage counterpart ( 0 . 51 mAP ), the U-FRCNN+ in 3D. Comparing the two to the remaining baselines shows a clear performance margin, hinting upon the importance of full segmentation supervision. The DetUNet performs worse with a notable margin ( 0 . 41 mAP in 2D with context slices and 0 . 37 mAP in 3D), seemingly suffering from high confidence false positive predictions caused by the ad-hoc score aggregation. Generally, 3D context shows to be important for this task, yet operating entirely in 3D seems to yield no benefits with respect to feeding the ± 3 neighbouring slices to a 2D network.

## 4.5. Breast lesion detection and categorization

## 4.5.1 Utilized data set

The breast lesion detection and categorization task is performed on an in-house Diffusion MRI data set of 331 Patients with suspicious findings in previous mammography. Lesion annotations are provided by experts. Categorization labels are given by subsequent pathology results (benign: n=141, malignant: n=190). Images are resampled to a resolution of 1 . 25 × 1 . 25 × 3 mm . For training, images are cropped to a size of 160 × 160 (for 3D training: 160 × 160 × 56 ).

## 4.5.2 Results

Results for the breast lesion detection task are shown in Table 2. Retina U-Net performs best in 2D with context slices ( 0 . 33 mAP ) and in 3D ( 0 . 36 mAP ). Mask R-CNN yields best results on plain 2D ( 0 . 34 mAP ). The overall best results are achieved by Retina U-Net and U-FRCNN+, again indicating that leveraging the full segmentation information is crucial in medical object detection. As opposed to the lung CT data set, 3D context information seems to be less important here, which is expected, when operating on highly anisotropic data, i.e. data with low z-resolution. Notably, the patient-level scores are only poorly correlated to the actual model performance due to high ambiguities in score aggregation and should not be used for model selection in clinical context.

## 4.6. Toy Experiments

We create a series of toy experiments to get a handle on the sub-tasks commonly involved in object-categorization on medical images, such as distinguishing scales, shapes and intensities. More specifically, the aim is to investigate the importance of full segmentation supervision in the context of limited training data. Therefore we consider three tasks for each of which we gradually decrease the amount of training data:

1. Distinguishing object shapes: Two classes of objects are to be detected and distinguished, where class 1 consists of filled circles with 20 pixels diameter and class 2 consists of filled circles with 20 pixels diameter and a centered whole of 4 pixels diameter, resembling the shape of donuts (see Figure 6a). For this task, the corresponding segmentation mask's shape explicitly contains the discriminative feature, since the centered whole is cut out from the foreground mask. Hence, full semantic supervision is expected to yield significant performance gains on this task, particularly in the small data regime.
2. Learning discriminative object patterns: This task is identical to the previous one, except the central hole is not cut out from the segmentation masks of the donuts (class 2). This requires the model to pick up the discriminative pattern (the hole) without explicitly pointing it out by means of the mask's shape (see Figure 6a). This setup could be considered more realistic in the context of medical images.
3. Distinguishing object scales: Two classes of objects are to be detected and distinguished, where class 1 consists of circles with 19 pixels diameter and class 2 consists of circles with 20 pixels diameter (see Figure 6b). Here, class information is entirely encoded in object

Figure 5. Results of the of toy experiment series. The three tasks are displayed as (a) distinguishing objects of different shapes, (b) learning discriminative image patterns unrelated to an object's shape, and (c) distinguishing objects of different scales. Explored models are divided into two groups: One-stage methods have blue/green color, while two-stage methods are drawn in red. n denotes the number of utilized training images.

<!-- image -->

scales and hence in target box coordinates. No significant gain from semantic supervision is expected.

## 4.6.1 Utilized data set

Both data sets consist of artificially generated 2D images of size 320 × 320 , where 1000 images were created for training, 500 for validation, and another 1000 as held-out test set. Images are zero-initialized and foreground objects imprinted by increasing intensity values by 0.2. Subsequently, uniform noise is added to all pixels.

## 4.6.2 Results

Results are shown in Figure 5. In the first task, where explicit class information is contained in segmentation annotations, models which explicitly leverage those, i.e. Retina U-Net and U-FRCNN, perform best. The resulting margin increases with decreasing amount of available training data. The second task, where class information is effectively removed from segmentation annotations, similar margins of Retina U-Net and U-FRCNN+ towards the other models are observed. This indicates the importance of full segmentation supervision even in implicit setups and shows a particularly strong edge in the small data set regime, where models that discard this supervision essentially collapse. In the third task, where class information is entirely contained in the target boxes, no gain from segmentation supervision is observed, at least for small training data sets. Interestingly, two-shot detectors perform better at this task, which seems counter-intuitive given the scale-invariance enforced by the RoIAlign operation. We hypothesize, that this bottleneck regularizes the architecture in a sense that scale information is enforced to be encoded broadly over spatial dimensions of FPN features. Comparing Mask R-CNN to Faster R-CNN+, the sub-optimal mask-supervision seems to yield no gains in detection performance when working with limited training data.

Figure 6. Image samples of the toy experiments. (a) the first two tasks require to detect and distinguish filled circles (left) from donuts (right) over noisy background. (b) For the third task, filled circles of different scales have to be distinguished: 19 pixels diameter (left) versus 20 pixels diameter (right).

<!-- image -->

## 5. Conclusion

In this work, we propose Retina U-Net, a simple but effective one-stage detection model which leverages semantic segmentation inspired by the current state-of-the-art architecture in medical segmentation. We show the importance of exploiting these training signals on multiple data sets, input dimensions and meticulously compare against the prevalent object detection models, with a particular emphasis on the context of limited training data. Therefore, we consider the task of localizing and classifying lesions, which constitutes a difficult and very frequent problem setting in radiology and therefore a highly-relevant domain of application. On the publicly available LIDC-IDRI lung CT dataset as well as on our in-house breast lesion MRI dataset Retina U-Net yields detection performance superior to models without full segmentation supervision and only reached by its more complex two-stage counterpart. By means of a set of toy experiments we shed light on a important set of scenarios that can profit from the additional full supervision: Any such problem where there is discriminative power in features beyond mere scale can expect to pocket an edge in detection performance. Among other distinguishing characteristics, the domain of medical image analysis holds one prominent feature: scarcity of labelled data. Retina U-Net is designed to make the most of the given supervision signal which is a key advantage on small datasets as high-lighted by our experiments. Our architecture stands out with another feature: its embarrassingly simple model formulation that takes aim at the clinical requirement of interpretability and robustness.

## References

- [1] S. Armato III, G. McLennan, L. Bidaut, M. McNittGray, C. Meyer, A. Reeves, and L. Clarke. Data from lidc-idri. the cancer imaging archive. DOI http://doi. org/10.7937 K , 9, 2015. 7
- [2] S. G. Armato III, G. McLennan, L. Bidaut, M. F. McNitt-Gray, C. R. Meyer, A. P. Reeves, B. Zhao, D. R. Aberle, C. I. Henschke, E. A. Hoffman, et al. The lung image database consortium (lidc) and image database resource initiative (idri): a completed reference database of lung nodules on ct scans. Medical physics , 38(2):915-931, 2011. 2, 7
- [3] K. Chen, J. Pang, J. Wang, Y. Xiong, X. Li, S. Sun, W. Feng, Z. Liu, J. Shi, W. Ouyang, C. C. Loy, and D. Lin. Hybrid task cascade for instance segmentation. Technical report, 2018. 2, 3, 4
- [4] L.-C. Chen, A. Hermans, G. Papandreou, F. Schroff, P. Wang, and H. Adam. Masklab: Instance segmentation by refining object detection with semantic and direction features. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , June 2018. 2
- [5] L.-C. Chen, Y. Zhu, G. Papandreou, F. Schroff, and H. Adam. Encoder-decoder with atrous separable convolution for semantic image segmentation. ECCV , 2018. 3
- [6] G. Chlebus, A. Schenk, J. H. Moltz, B. van Ginneken, H. K. Hahn, and H. Meine. Automatic liver tumor segmentation in ct with fully convolutional neural networks and object-based postprocessing. Scientific reports , 8(1):15497, 2018. 1
- [7] K. Clark, B. Vendt, K. Smith, J. Freymann, J. Kirby, P. Koppel, S. Moore, S. Phillips, D. Maffitt, M. Pringle, et al. The cancer imaging archive (tcia): maintaining and operating a public information repository. Journal of digital imaging , 26(6):1045-1057, 2013. 7
- [8] J. Dai, Y. Li, K. He, and J. Sun. R-fcn: Object detection via region-based fully convolutional networks. In Advances in neural information processing systems , pages 379-387, 2016. 2
- [9] R. Girshick. Fast r-cnn. In Proceedings of the IEEE international conference on computer vision , pages 1440-1448, 2015. 4
- [10] K. He, G. Gkioxari, P. Doll´ ar, and R. Girshick. Mask rcnn. In Computer Vision (ICCV), 2017 IEEE International Conference on , pages 2980-2988. IEEE, 2017. 2
- [11] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 770-778, 2016. 4
- [12] F. Isensee, J. Petersen, A. Klein, D. Zimmerer, P. F. Jaeger, S. Kohl, J. Wasserthal, G. Koehler, T. Norajitra, S. Wirkert, et al. nnu-net: Self-adapting framework for u-net-based medical image segmentation. arXiv preprint arXiv:1809.10486 , 2018. 1, 3
- [13] D. P. Kingma and J. Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 , 2014. 5
- [14] S. Kohl, D. Bonekamp, H.-P. Schlemmer, K. Yaqubi, M. Hohenfellner, B. Hadaschik, J.-P. Radtke, and K. Maier-Hein. Adversarial networks for the detection of aggressive prostate cancer. arXiv preprint arXiv:1702.08014 , 2017. 1
- [15] T.-Y. Lin, P. Doll´ ar, R. B. Girshick, K. He, B. Hariharan, and S. J. Belongie. Feature pyramid networks for object detection. In CVPR , volume 1, page 4, 2017. 2, 3, 4
- [16] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Doll´ ar. Focal loss for dense object detection. IEEE transactions on pattern analysis and machine intelligence , 2018. 2, 3
- [17] S. Liu, L. Qi, H. Qin, J. Shi, and J. Jia. Path aggregation network for instance segmentation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition , pages 8759-8768, 2018. 2
- [18] W. Liu, D. Anguelov, D. Erhan, C. Szegedy, S. Reed, C.-Y. Fu, and A. C. Berg. Ssd: Single shot multibox detector. In European conference on computer vision , pages 21-37. Springer, 2016. 2
- [19] J. Mao, T. Xiao, Y. Jiang, and Z. Cao. What can help pedestrian detection? In 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , pages 6034-6043. IEEE, 2017. 2
- [20] C. Peng, T. Xiao, Z. Li, Y. Jiang, X. Zhang, K. Jia, G. Yu, and J. Sun. Megdet: A large mini-batch object detector. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2018. 2, 3
- [21] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi. You only look once: Unified, real-time object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 779-788, 2016. 2
- [22] O. Ronneberger, P. Fischer, and T. Brox. U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention , pages 234-241. Springer, 2015. 1, 2, 3
- [23] A. Shrivastava and A. Gupta. Contextual priming and feedback for faster r-cnn. In European Conference on Computer Vision , pages 330-348. Springer, 2016. 2
- [24] A. Shvets, V. Iglovikov, A. Rakhlin, and A. A. Kalinin. Angiodysplasia detection and localization using deep convolutional neural networks. arXiv preprint arXiv:1804.08024 , 2018. 1
- [25] J. Uhrig, E. Rehder, B. Fr¨ ohlich, U. Franke, and T. Brox. Box2pix: Single-shot instance segmentation by assigning pixels to object boxes. In IEEE Intelligent Vehicles Symposium (IV) , 2018. 3
- [26] N. Wang, C. Bian, Y. Wang, M. Xu, C. Qin, X. Yang, T. Wang, A. Li, D. Shen, and D. Ni. Densely deep supervised networks with threshold loss for cancer detection in automated breast ultrasound. In International Conference on Medical Image Computing and Computer-Assisted Intervention , pages 641-648. Springer, 2018. 1
- [27] K. Yan, M. Bagheri, and R. M. Summers. 3d context enhanced region-based convolutional neural network

for end-to-end lesion detection. In International Conference on Medical Image Computing and ComputerAssisted Intervention , pages 511-519. Springer, 2018. 5

- [28] Z. Zhang, S. Qiao, C. Xie, W. Shen, B. Wang, and A. L. Yuille. Single-shot object detection with enriched semantics. Technical report, Center for Brains, Minds and Machines (CBMM), 2018. 3