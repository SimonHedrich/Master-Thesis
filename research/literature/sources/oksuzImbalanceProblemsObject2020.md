## Imbalance Problems in Object Detection: A Review

Kemal Oksuz † , Baris Can Cam , Sinan Kalkan ‡ , and Emre Akbas ‡

<!-- image -->

Abstract -In this paper, we present a comprehensive review of the imbalance problems in object detection. To analyze the problems in a systematic manner, we introduce a problem-based taxonomy. Following this taxonomy, we discuss each problem in depth and present a unifying yet critical perspective on the solutions in the literature. In addition, we identify major open issues regarding the existing imbalance problems as well as imbalance problems that have not been discussed before. Moreover, in order to keep our review up to date, we provide an accompanying webpage which catalogs papers addressing imbalance problems, according to our problem-based taxonomy. Researchers can track newer studies on this webpage available at: https://github.com/kemaloksuz/ObjectDetectionImbalance.

F

## 1 INTRODUCTION

Object detection is the simultaneous estimation of categories and locations of object instances in a given image. It is a fundamental problem in computer vision with many important applications in e.g. surveillance [1], [2], autonomous driving [3], [4], medical decision making [5], [6], and many problems in robotics [7], [8], [9], [10], [11], [12].

Since the time when object detection (OD) was cast as a machine learning problem, the first generation OD methods relied on hand-crafted features and linear, max-margin classifiers. The most successful and representative method in this generation was the Deformable Parts Model (DPM) [13]. After the extremely influential work by Krizhevsky et al. in 2012 [14], deep learning (or deep neural networks) has started to dominate various problems in computer vision and OD was no exception. The current generation OD methods are all based on deep learning where both the hand-crafted features and linear classifiers of the first generation methods have been replaced by deep neural networks. This replacement has brought significant improvements in performance: On a widely used OD benchmark dataset (PASCAL VOC), while the DPM [13] achieved 0.34 mean average-precision (mAP), current deep learning based OD models achieve around 0.80 mAP [15].

In the last five years, although the major driving force of progress in OD has been the incorporation of deep neural networks [16], [17], [18], [19], [20], [21], [22], [23], imbalance problems in OD at several levels have also received significant attention [24], [25], [26], [27], [28], [29], [30]. An imbalance problem with respect to an input property occurs when the distribution regarding that property affects the performance. When not addressed, an imbalance problem has adverse effects on the final detection performance. For example, the most commonly known imbalance problem in OD is the foreground-to-background imbalance which

All authors are at the Dept. of Computer Engineering, Middle East Technical University (METU), Ankara, Turkey. E-mail:{kemal.oksuz@metu.edu.tr, can.cam@metu.edu.tr, skalkan@metu.edu.tr, emre@ceng.metu.edu.tr}

† Corresponding author.

‡ Equal contribution for senior authorship.

manifests itself in the extreme inequality between the number of positive examples versus the number of negatives. In a given image, while there are typically a few positive examples, one can extract millions of negative examples. If not addressed, this imbalance greatly impairs detection accuracy.

In this paper, we review the deep-learning-era object detection literature and identify eight different imbalance problems. We group these problems in a taxonomy with four main types: class imbalance, scale imbalance, spatial imbalance and objective imbalance (Table 1). Class imbalance occurs when there is significant inequality among the number of examples pertaining to different classes. While the classical example of this is the foreground-tobackground imbalance, there is also imbalance among the foreground (positive) classes. Scale imbalance occurs when the objects have various scales and different numbers of examples pertaining to different scales. Spatial imbalance refers to a set of factors related to spatial properties of the bounding boxes such as regression penalty, location and IoU. Finally, objective imbalance occurs when there are multiple loss functions to minimize, as is often the case in OD (e.g. classification and regression losses).

## 1.1 Scope and Aim

Imbalance problems in general have a large scope in machine learning, computer vision and pattern recognition. We limit the focus of this paper to imbalance problems in object detection. Since the current state-of-the-art is shaped by deep learning based approaches, the problems and approaches that we discuss in this paper are related to deep object detectors. Although we restrict our attention to object detection in still images, we provide brief discussions on similarities and differences of imbalance problems in other domains. We believe that these discussions would provide insights on future research directions for object detection researchers.

Presenting a comprehensive background for object detection is not among the goals of this paper; however, some background knowledge on object detection is required to make the most out of this paper. For a thorough background on the subject, we refer the readers to the recent, comprehensive object detection surveys [31], [32], [33]. We provide only a brief background on state-of-the-art object detection in Section 2.1.

TABLE 1: Imbalance problems reviewed in this paper. We state that an imbalance problem with respect to an input property occurs when the distribution regarding that property affects the performance. The first column shows the major imbalance categories. For each Imbalance problem given in the middle column, the last column shows the associated input property concerning the definition of the imbalance problem.

| Type      | Imbalance Problem                            | Related Input Property                                                                                                |
|-----------|----------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| Class     | Foreground-Background Class Imbalance (§4.1) | The numbers of input bounding boxes pertain- ing to different classes                                                 |
| Class     | Foreground-Foreground Class Imbalance (§4.2) | The numbers of input bounding boxes pertain- ing to different classes                                                 |
| Scale     | Object/box-level Scale Imbalance (§5.1)      | The scales of input and ground-truth bounding boxes                                                                   |
| Scale     | Feature-level Imbalance (§5.2)               | Contribution of the feature layer from different abstraction levels of the backbone network (i.e. high and low level) |
| Spatial   | Imbalance in Regression Loss (§6.1)          | Contribution of the individual examples to the regression loss                                                        |
| Spatial   | IoU Distribution Imbalance (§6.2)            | IoU distribution of positive input bounding boxes                                                                     |
| Spatial   | Object Location Imbalance (§6.3)             | Locations of the objects throughout the image                                                                         |
| Objective | Objective Imbalance (§7)                     | Contribution of different tasks (i.e. classifica- tion, regression) to the overall loss                               |

Our main aim in this paper is to present and discuss imbalance problems in object detection comprehensively. In order to do that,

- 1) We identify and define imbalance problems and propose a taxonomy for studying the problems and their solutions.
- 2) We present a critical literature review for the existing studies with a motivation to unify them in a systematic manner. The general outline of our review includes a definition of the problems, a summary of the main approaches, an in-depth coverage of the specific solutions, and comparative summaries of the solutions.
- 3) We present and discuss open issues at the problemlevel and in general.
- 4) We also reserved a section for imbalance problems found in domains other than object detection. This section is generated with meticulous examination of methods considering their adaptability to the object detection pipeline.
- 5) Finally, we provide an accompanying webpage 1 as a living repository of papers addressing imbalance problems, organized based on our problem-based taxonomy. This webpage will be continuously updated with new studies.

## 1.2 Comparison with Previous Reviews

Recent object detection surveys [31], [32], [33] aim to present advances in deep learning based generic object detection. To this end, these surveys propose a taxonomy for object

1. [https://github.com/kemaloksuz/ObjectDetectionImbalance](https://github.com/kemaloksuz/ObjectDetectionImbalance)

detection methods, and present a detailed analysis of some cornerstone methods that have had high impact. They also provide discussions on popular datasets and evaluation metrics. From the imbalance point of view, these surveys only consider the class imbalance problem with a limited provision. Additionally, Zou et al. [32] provide a review for methods that handle scale imbalance. Unlike these surveys, here we focus on a classification of imbalance problems related to object detection and present a comprehensive review of methods that handle these imbalance problems.

There are also surveys on category specific object detection (e.g. pedestrian detection, vehicle detection, face detection) [34], [35], [36], [37]. Although Zehang Sun et al. [34] and Dollar et al. [35] cover the methods proposed before the current deep learning era, they are beneficial from the imbalance point of view since they present a comprehensive analysis of feature extraction methods that handle scale imbalance. Zafeiriou et al. [36] and Yin et al. [38] propose comparative analyses of non-deep and deep methods. Litjens et al. [39] discuss applications of various deep neural network based methods i.e. classification, detection, segmentation to medical image analysis. They present challenges with their possible solutions which include a limited exploration of the class imbalance problem. These category specific object detector reviews focus on a single class and do not consider the imbalance problems in a comprehensive manner from the generic object detection perspective.

Another set of relevant work includes the studies specifically for imbalance problems in machine learning [40], [41], [42], [43]. These studies are limited to the foreground class imbalance problem in our context (i.e. there is no background class). Generally, they cover dataset-level methods such as undersampling and oversampling, and algorithmlevel methods including feature selection, kernel modifications and weighted approaches. We identify three main differences of our work compared to such studies. Firstly, the main scope of such work is the classification problem, which is still relevant for object detection; however, object detection also has a 'search' aspect, in addition to the recognition aspect, which brings in the background (i.e. negative) class into the picture. Secondly, except Johnson et al. [43], they consider machine learning approaches in general without any special focus on deep learning based methods. Finally, and more importantly, these works only consider foreground class imbalance problem, which is only one of eight different imbalance problems that we present and discuss here (Table 1).

## 1.3 A Guide to Reading This Review

The paper is organized as follows. Section 2 provides a brief background on object detection, and the list of frequentlyused terms and notation used throughout the paper. Section 3 presents our taxonomy of imbalance problems. Sections 47 then cover each imbalance problem in detail, with a critical review of the proposed solutions and include open issues for each imbalance problem. Each section dedicated to a specific imbalance problem is designed to be self-readable, containing definitions and a review of the proposed methods. In order to provide a more general perspective, in Section 8, we present the solutions addressing imbalance in other but closely related domains. Section 9 discusses open issues that are relevant to all imbalance problems. Finally, Section 10 concludes the paper.

Readers who are familiar with the current state-of-the-art object detection methods can directly jump to Section 3 and use Figure 1 to navigate both the imbalance problems and the sections dedicated to the different problems according to the taxonomy. For readers who lack a background in stateof-the-art object detection, we recommend starting with Section 2.1, and if this brief background is not sufficient, we refer the reader to the more in-depth reviews mentioned in Section 1.1.

## 2 BACKGROUND, DEFINITIONS AND NOTATION

In the following, we first provide a brief background on state-of-the-art object detection methods, and then present the definitions and notations used throughout the paper.

## 2.1 State of the Art in Object Detection

Today there are two major approaches to object detection: top-down and bottom-up. Although both the top-down and bottom-up approaches were popular prior to the deep learning era, today the majority of the object detection methods follow the top-down approach; the bottom-up methods have been proposed relatively recently. The main difference between the top-down and bottom-up approaches is that, in the top-down approach, holistic object hypotheses (i.e., anchors, regions-of-interests/proposals) are generated and evaluated early in the detection pipeline, whereas in the bottom-up approach, holistic objects emerge by grouping sub-object entities like keypoints or parts, later in the processing pipeline.

Methods following the top-down approach are categorized into two: two-stage and one-stage methods. Twostage methods [16], [17], [18], [21] aim to decrease the large number of negative examples resulting from the predefined, dense sliding windows, called anchors, to a manageable size by using a proposal mechanism [21], [44], [45] which determines the regions where the objects most likely appear, called Region of Interests (RoIs). These RoIs are further processed by a detection network which outputs the object detection results in the form of bounding boxes and associated object-category probabilities. Finally, the non-maxima suppression (NMS) method is applied on the object detection results to eliminate duplicate or highly-overlapping results. NMS is a universal post-processing step used by all state-of-the-art object detectors.

TABLE 2: Frequently used notations in the paper.

| Symbol            | Domain                  | Denotes                                                              |
|-------------------|-------------------------|----------------------------------------------------------------------|
| B                 | See.Def.                | A Bounding box                                                       |
| C                 | A set of integers       | Set of Class labels in a dataset                                     |
| &#124; C i &#124; | &#124; C i &#124; ∈ Z + | Number of examples for the i th class in a dataset                   |
| Ci                | i ∈ Z +                 | Backbone feature layer at depth i                                    |
| I ( P )           | I ( P ) ∈ { 0 , 1 }     | Indicator function. 1 if predicate P is true, else 0                 |
| Pi                | i ∈ Z +                 | Pyramidal feature layer corresponding to i th backbone feature layer |
| p i               | p i ∈ [0 , 1]           | Confidence score of i th class (i.e. output of classifier)           |
| p s               | p s ∈ [0 , 1]           | Confidence score of the ground truth class                           |
| p 0               | p 0 ∈ [0 , 1]           | Confidence score of background class                                 |
| u                 | u ∈ C                   | Class label of a ground truth                                        |
| ˆ x               | ˆ x ∈ R                 | Input of the regression loss                                         |

One-stage top-down methods, including SSD Variants [19], [46], YOLO variants [15], [20], [47] and RetinaNet [22], are designed to predict the detection results directly from anchors - without any proposal elimination stage - after extracting the features from the input image. We present a typical one-stage object detection pipeline in Figure 1(a). The pipeline starts with feeding the input image to the feature extraction network, which is usually a deep convolutional neural network. A dense set of object hypotheses (called anchors) are produced, which are then sampled and labeled by matching them to ground-truth boxes. Finally, labeled anchors (whose features are obtained from the output of the feature extraction network) are fed to the classification and regression networks for training. In a two-stage method, object proposals (or regions-of-interest) are first generated using anchors by a separate network (hence, the two stages).

On the other hand, bottom-up object detection methods [23], [48], [49] first predict important key-points (e.g. corners, centers, etc.) on objects and then group them to form whole object instances by using a grouping method such as associative embedding [50] and brute force search [49].

## 2.2 Frequently Used Terms and Notation

Table 2 presents the notation used throughout the paper, and below is a list of frequently used terms.

<!-- image -->

(b)

Fig. 1: (a) The common training pipeline of a generic detection network. The pipeline has 3 phases (i.e. feature extraction, detection and BB matching, labeling and sampling) represented by different background colors. (b) Illustration of an example imbalance problem from each category for object detection through the training pipeline. Background colors specify at which phase an imbalance problem occurs.

Feature Extraction Network/Backbone: This is the part of the object detection pipeline from the input image until the detection network.

Classification Network/Classifier: This is the part of the object detection pipeline from the features extracted by the backbone to the classification result, which is indicated by a confidence score.

Regression Network/Regressor: This is the part of the object detection pipeline from the features extracted by the backbone to the regression output, which is indicated by two bounding box coordinates each of which consisting of an x-axis and y-axis values.

Detection Network/Detector: It is the part of the object detection pipeline including both classifier and regressor.

Region Proposal Network (RPN): It is the part of the two stage object detection pipeline from the features extracted by the backbone to the generated proposals, which also have confidence scores and bounding box coordinates.

Bounding Box: A rectangle on the image limiting certain features. Formally, [ x 1 , y 1 , x 2 , y 2 ] determine a bounding box with top-left corner ( x 1 , y 1 ) and bottom-right corner ( x 2 , y 2 ) satisfying x 2 &gt; x 1 and y 2 &gt; y 1 .

Anchor: The set of pre defined bounding boxes on which the RPN in two stage object detectors and detection network in one stage detectors are applied.

Region of Interest (RoI)/Proposal: The set of bounding boxes generated by a proposal mechanism such as RPN on which the detection network is applied on two state object detectors.

Input Bounding Box: Sampled anchor or RoI the detection network or RPN is trained with.

Ground Truth: It is tuple ( B,u ) such that B is the bounding box and u is the class label where u ∈ C and C is the enumeration of the classes in the dataset.

Detection: It is a tuple ( ¯ B,p ) such that ¯ B is the bounding box and p is the vector over the confidence scores for each class 2 and bounding box.

Intersection Over Union: For a ground truth box B and a detection box ¯ B , we can formally define Intersection over Union(IoU) [51], [52], denoted by IoU( B, ¯ B ) , as

<!-- formula-not-decoded -->

such that A ( B ) is the area of a bounding box B .

Under-represented Class: The class which has less samples in a dataset or mini batch during training in the context of class imbalance.

Over-represented Class: The class which has more samples in a dataset or mini batch during training in the context of class imbalance.

Backbone Features: The set of features obtained during the application of the backbone network.

Pyramidal Features/Feature Pyramid: The set of features obtained by applying some transformations to the backbone features.

Regression Objective Input: Some methods make predictions in the log domain by applying some transformation which can also differ from method to method (compare transformation in Fast R-CNN [17] and in KL loss [53] for Smooth L1 Loss), while some methods directly predict the bounding box coordinates [23]. For the sake of clarity, we use ˆ x to denote the regression loss input for any method.

2. We use class and category interchangeably in this paper.

Fig. 2: Problem based categorization of the methods used for imbalance problems. Note that a work may appear at multiple locations if it addresses multiple imbalance problems - e.g. Libra R-CNN [29]

<!-- image -->

## 3 A TAXONOMY OF THE IMBALANCE PROBLEMS AND THEIR SOLUTIONS IN OBJECT DETECTION

In Section 1, we defined the problem of imbalance as the occurrence of a distributional bias regarding an input property in the object detection training pipeline. Several different types of such imbalance can be observed at various stages of the common object detection pipeline (Figure 1). To study these problems in a systematic manner, we propose a taxonomy based on the related input property.

We identify eight different imbalance problems, which we group into four main categories: class imbalance, scale imbalance, spatial imbalance and objective imbalance. Table 1 presents the complete taxonomy along with a brief definition for each problem. In Figure 2, we present the same taxonomy along with a list of proposed solutions for each problem. Finally, in Figure 1, we illustrate a generic object detection pipeline where each phase is annotated with their typically observed imbalance problems. In the following, we elaborate on the brief definitions provided earlier, and illustrate the typical phases where each imbalance problem occurs.

Class imbalance (Section 4; blue branch in Figure 2)

occurs when a certain class is over-represented. This problem can manifest itself as either 'foreground-background imbalance,' where the background instances significantly outnumber the positive ones; or 'foreground-foreground imbalance,' where typically a small fraction of classes dominate the dataset (as observed from the long-tail distribution in Figure 5). The class imbalance is usually handled at the 'sampling' stage in the object detection pipeline (Figure 1).

Scale imbalance (Section 5; green branch in Figure 2) is observed when object instances have various scales and different number of examples pertaining to different scales. This problem is a natural outcome of the fact that objects have different dimensions in nature. Scale also could cause imbalance at the feature-level (typically handled in the 'feature extraction' phase in Figure 1), where contribution from different abstraction layers (i.e. high and low levels) are not balanced. Scale imbalance problem suggests that a single scale of visual processing is not sufficient for detecting objects at different scales. However, as we will see in Section 5, proposed methods fall short in addressing scale imbalance, especially for small objects, even when small objects are surprisingly abundant in a dataset.

Spatial imbalance (Section 6; orange branch in Figure 2) refers to a set of factors related to spatial properties of the bounding boxes. Owing to these spatial properties, we identify three sub-types of spatial imbalance: (i) 'imbalance in regression loss' is about the contributions of individual examples to the regression loss and naturally the problem is related to loss function design (ii) 'IoU distribution imbalance' is related to the biases in the distribution of IoUs (among ground-truth boxes vs. anchors or detected boxes), and typically observed at the bounding-box matching and labeling phase in the object detection pipeline (Figure 1) (iii) 'object location imbalance' is about the location distribution of object instances in an image, which is related to both the design of the anchors and the sampled subset to train the detection network.

Finally, objective imbalance (Section 7; purple branch in Figure 2) occurs when there are multiple objectives (loss functions) to minimize (each for a specific task, e.g. classification and box-regression). Since different objectives might be incompatible in terms of their ranges as well as their optimum solutions, it is essential to develop a balanced strategy that can find a solution acceptable in terms of all objectives.

Figure 2 gives an overall picture of the attention that different types of imbalance problems have received from the research community. For example, while there are numerous methods devised for the foreground-background class imbalance problem, the objective imbalance and object location imbalance problems are examples of the problems that received relatively little attention. However, recently there have been a rapidly increasing interest (Figure 3) in these imbalance problems as well, which necessitates a structured view and perspective on the problems as well as the solutions, as proposed in this paper.

Note that some imbalance problems are directly owing to the data whereas some are the by-products of the specific methods used. For example, class imbalance, object location imbalance etc. are the natural outcomes of the distribution of classes in the real world. On the other hand, objective imbalance, feature-level imbalance and imbalance in regression loss are owing to the selected methods and possibly avoidable with a different set of methods; for example, it is possible to avoid IoU distribution imbalance altogether by following a bottom-up approach where, typically, IoU is not a labeling criterion (e.g. [23], [49]).

Fig. 3: Number of papers per imbalance problem category through years.

<!-- image -->

## 4 IMBALANCE 1: CLASS IMBALANCE

Class imbalance is observed when a class is overrepresented, having more examples than others in the dataset. This can occur in two different ways from the object detection perspective: foreground-background imbalance and foreground-foreground imbalance.

Figure 4 illustrates the presence of class imbalance. To generate the figure, we apply the default set of anchors from RetinaNet [22] on the MS-COCO dataset 3 [90] and calculated the frequencies for the cases where the IoU of an anchor with a ground truth bounding box exceeds 0 . 5 and where it is less than 0 . 4 (i.e. it is a background box) following the labeling rule of RetinaNet [22]. When an anchor overlaps with a foreground class (with IoU&gt;0.5), we kept a count for each class separately and normalized the resulting frequencies with the number of images in the dataset.

These two types of class imbalance have different characteristics and have been addressed using different types of solutions. Therefore, in the following, we will cover them separately. However, some solutions (e.g. generative modeling) could be employed for both problem types.

## 4.1 Foreground-Background Class Imbalance

Definition. In foreground-background class imbalance, the over-represented and under-represented classes are background and foreground classes respectively. This type of problem is inevitable because most bounding boxes are labeled as background (i.e. negative) class by the bounding box matching and labeling module as illustrated in Figure 4(a). Foreground-background imbalance problem occurs during training and it does not depend on the number of examples per class in the dataset since they do not include any annotation on background.

3. Throughout the text, unless otherwise specified, 'COCO', 'PASCAL', and 'Open Images' respectively correspond to the Pascal VOC 2012 trainval [51], MS-COCO 2017 train [90], Open Images v5 training (subset with bounding boxes) [91] and Objects365 train [92] dataset partitions. And, when unspecified, the backbone is ResNet-50 [93].

Fig. 4: Illustration of the class imbalance problems. The numbers of RetinaNet [22] anchors on MS-COCO [90] are plotted for foreground-background (a) , and foreground classes (b) . The values are normalized with the total number of images in the dataset. The figures depict severe imbalance towards some classes.

<!-- image -->

Solutions. We can group the solutions for the foregroundbackground class imbalance into four: (i) hard sampling methods, (ii) soft sampling methods, (iii) sampling-free methods and (iv) generative methods. Each set of methods are explained in detail in the subsections below.

In sampling methods, the contribution ( w i ) of a bounding box ( BB i ) to the loss function is adjusted as follows:

<!-- formula-not-decoded -->

where CE() is the cross-entropy loss. Hard and soft sampling approaches differ on the possible values of w i . For the hard sampling approaches, w i ∈ { 0 , 1 } , thus a BB is either selected or discarded. For soft sampling approaches, w i ∈ [0 , 1] , i.e. the contribution of a sample is adjusted with a weight and each BB is somehow included in training.

## 4.1.1 Hard Sampling Methods

Hard sampling is a commonly-used method for addressing imbalance in object detection. It restricts w i to be binary; i.e., 0 or 1. In other words, it addresses imbalance by selecting a subset of positive and negative examples (with desired quantities) from a given set of labeled BBs. This selection is performed using heuristic methods and the non-selected examples are ignored for the current iteration. Therefore, each sampled example contributes equally to the loss (i.e. w i = 1 ) and the non-selected examples ( w i = 0 ) have no contribution to the training for the current iteration. See Table 3 for a summary of the main approaches.

A straightforward hard-sampling method is random sampling . Despite its simplicity, it is employed in R-CNN family of detectors [16], [21] where, for training RPN, 128 positive examples are sampled uniformly at random (out of all positive examples) and 128 negative anchors are sampled in a similar fashion; and 16 positive examples and 48 negative RoIs are sampled uniformly from each image in the batch at random from within their respective sets, for training the detection network [17]. In any case, if the number of positive input bounding boxes is less than the desired values, the mini-batch is padded with randomly sampled negatives. On the other hand, it has been reported that other sampling strategies may perform better when a property of an input box such as its loss value or IoU is taken into account [24], [29], [30].

The first set of approaches to consider a property of the sampled examples, rather than random sampling, is the Hard-example mining methods 4 . These methods rely on the hypothesis that training a detector more with hard examples (i.e. examples with high losses) leads to better performance. The origins of this hypothesis go back to the bootstrapping idea in the early works on face detection [55], [94], [95], human detection [96] and object detection [13]. The idea is based on training an initial model using a subset of negative examples, then using the negative examples on which the classifier fails (i.e. hard examples), a new classifier is trained. Multiple classifiers are obtained by applying the same procedure iteratively. Currently deep-learning-based methods also adopt some versions of the hard example mining in order to provide more useful examples by using the loss values of the examples. The first deep object detector to use hard examples in the training was Single-Shot Detector [19], which chooses only the negative examples incurring the highest loss values. A more systematic approach considering the loss values of positive and negative samples is proposed in Online Hard Example Mining (OHEM) [24]. However, OHEM needs additional memory and causes the training speed to decrease. Considering the efficiency and memory problems of OHEM, IoU-based sampling [29] was proposed to associate the hardness of the examples with their IoUs and to use a sampling method again for only negative examples rather than computing the loss function for the entire set. In the IoU-based sampling, the IoU interval for the negative samples is divided into K bins and equal number of negative examples are sampled randomly within each bin to promote the samples with higher IoUs, which are expected to have higher loss values.

To improve mining performance, several studies proposed to limit the search space in order to make hard examples easy to mine. Two-stage object detectors [18], [21] are among these methods since they aim to find the most probable bounding boxes (i.e. RoIs) given anchors, and then choose top N RoIs with the highest objectness scores, to which an additional sampling method is applied. Fast RCNN [17] sets the lower bound of IoU of the negative RoIs to 0 . 1 rather than 0 for promoting hard negatives and then applies random sampling. Kong et al. [56] proposed a method that learns objectness priors in an end-to-end setting in order to have a guidance on where to search for the objects. All of the positive examples having an objectness prior larger than a threshold are used during training, while the negative examples are selected such that the desired balance (i.e. 1 : 3 ) is preserved between positive and negative classes. Zhang et al. [57] proposed determining the confidence scores of anchors with the anchor refinement module in a one-stage detection pipeline and again adopted a threshold to eliminate the easy negative anchors. The authors coined their approach as negative anchor filtering . Nie et al. [58] used a cascaded-detection scheme in the SSD pipeline which includes an Objectness Module before each prediction module. These objectness modules are binary classifiers to filter out the easy anchors.

4. In this paper, we adopt the boldface font whenever we introduce an approach involving a set of different methods, and the method names themselves are in italic.

## 4.1.2 Soft Sampling Methods

Soft sampling adjusts the contribution ( w i ) of each example by its relative importance to the training process. This way, unlike hard sampling, no sample is discarded and the whole dataset is utilized for updating the parameters. See Table 3 for a summary of the main approaches.

A straightforward approach is to use constant coefficients for both the foreground and background classes. YOLO [20], having less number of anchors compared to other one-stage methods such as SSD [19] and RetinaNet [22], is a straightforward example for soft sampling in which the loss values of the examples from the background class are halved (i.e. w i = 0 . 5 ).

Focal Loss [22] is the pioneer example that dynamically assigns more weight to the hard examples as follows:

<!-- formula-not-decoded -->

where p s is the estimated probability for the ground-truth class. Since lower p s implies a larger error, Equation (3) promotes harder examples. Note that when γ = 0 , focal loss degenerates to vanilla cross entropy loss and Lin et al. [22] showed that γ = 2 ensures a good trade-off between hard and easy examples for their architecture.

Similar to focal loss [22], Gradient Harmonizing Mechanism (GHM) [59] suppresses gradients originating from easy positives and negatives. The authors first observed that there are too many samples with small gradient norm, only limited number of samples with medium gradient norm and significantly large amount of samples with large gradient norm. Unlike focal loss, GHM is a counting-based approach which counts the number of examples with similar gradient norm and penalizes the loss of a sample if there are many samples with similar gradients as follows:

<!-- formula-not-decoded -->

where G ( BB i ) is the count of samples whose gradient norm is close to the gradient norm of BB i ; and m is the number of input bounding boxes in the batch. In this sense, the GHM method implicitly assumes that easy examples are those with too many similar gradients. Different from other methods, GHM is shown to be useful not only for classification task but also for the regression task. In addition, since the purpose is to balance the gradients within each task, this method is also relevant to the 'imbalance in regression loss' discussed in Section 6.1.

Different from the latter soft sampling methods, PrIme Sample Attention (PISA) [30] assigns weights to positive and negative examples based on different criteria. While the positive ones with higher IoUs are favored, the negatives with larger foreground classification scores are promoted. More specifically, PISA first ranks the examples for each class based on their desired property (IoU or classification score) and calculates a normalized rank, ¯ r i , for each example i as follows:

<!-- formula-not-decoded -->

where r i ( 0 ≤ r i ≤ N j - 1 ) is the rank of the i th example and N max is the maximum number of examples over all classes in the batch. Based on the normalized rank, the weight of each example is defined as:

<!-- formula-not-decoded -->

where β adjusts the contribution of the normalized rank and hence, the minimum sample weight; and γ is the modulating factor again. These parameters are validated for positives and negatives independently. Note that the balancing strategy in Equations (5) and (6) increases the contribution of the samples with high IoUs for positives and high scores for negatives to the loss.

## 4.1.3 Sampling-Free Methods

Recently, alternative methods emerged in order to avoid the aforementioned hand-crafted sampling heuristics and therefore, decrease the number of hyperparameters during training. For this purpose, Chen et al. [60] added an objectness branch to the detection network in order to predict Residual Objectness scores. While this new branch tackles the foreground-background imbalance, the classification branch handles only the positive classes. During inference, classification scores are obtained by multiplying classification and objectness branch outputs. The authors showed that such a cascaded pipeline improves the performance. This architecture is trained using vanilla cross entropy loss.

Another recent approach [54] suggests that if the hyperparameters are set appropriately, not only the detector can be trained without any sampling mechanism but also the performance can be improved. Accordingly, the authors propose methods for setting the initialization bias, loss weighting (see Section 7) and class-adaptive threshold, and in such a way they trained the network using vanilla cross entropy loss achieving better performance.

Finally, an alternative method is to directly model the final performance measure and weigh examples based on this model. This approach is adopted by AP Loss [61] which formulates the classification part of the loss as a ranking task (see also DR Loss [62] which also uses a ranking method to define a classification loss based on Hinge Loss) and uses average precision (AP) as the loss function for this task. In this method, the confidence scores are first transformed as x ij = - ( p i - p j ) such that p i is the confidence score of the i th bounding box. Then, based on the transformed values, the primary terms of the AP loss are computed as (presented here in a simplified form):

TABLE 3: A toy example depicting the selection methods of common hard and soft sampling methods. One positive and two negative examples are to be chosen from six bounding boxes (drawn at top-right). The properties are the basis for the sampling methods. p s is the predicted ground truth probability (i.e. positive class probability for positive BBs, and background probability for negative BBs). If we set a property or hyperparameter for this example, it is shown in the table. For soft sampling methods, the numbers are the weights of each box (i.e. w i ). We assumed one foreground class for PISA.

| Legend&Method                                                           | Considered Property   | Intuition                                                                                              | Addit. Params.   | Positive Examples                                               | Positive Examples                                               | Negative Examples                                                                                                  | Negative Examples                                                                                                  | Negative Examples                                                                                                  | Negative Examples                                                                                                  |
|-------------------------------------------------------------------------|-----------------------|--------------------------------------------------------------------------------------------------------|------------------|-----------------------------------------------------------------|-----------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| Boxes Negative BB Positive BB Ground Truth 3 Selected BB 7 Discarded BB | N/A                   | N/A                                                                                                    | N/A              | p s =0.8 Loss = 0.22 IoU = 0.95 p s =0.2 Loss = 1.61 IoU = 0.55 | p s =0.8 Loss = 0.22 IoU = 0.95 p s =0.2 Loss = 1.61 IoU = 0.55 | p s =0.1 Loss = 2.30 IoU = 0.35                                                                                    | p s =0.3 Loss = 1.20 IoU = 0.15                                                                                    | p s =0.6 Loss = 0.51 IoU = 0.06                                                                                    | p s =1.0 Loss = 0.00 IoU = 0.00                                                                                    |
| Random Sampling                                                         | -                     | Samples uniformly among boxes                                                                          | -                | Random Sample × 1                                               | Random Sample × 1                                               | Random Sample × 2                                                                                                  | Random Sample × 2                                                                                                  | Random Sample × 2                                                                                                  | Random Sample × 2                                                                                                  |
| SSD [19]                                                                | Loss                  | Chooses neg. with max loss                                                                             | -                | Random Sample × 1                                               | Random Sample × 1                                               | 3                                                                                                                  | 3                                                                                                                  | 7                                                                                                                  | 7                                                                                                                  |
| OHEM [24]                                                               | Loss                  | Chooses pos.&neg. with max loss                                                                        | -                | 7                                                               | 3                                                               | 3                                                                                                                  | 3                                                                                                                  | 7                                                                                                                  | 7                                                                                                                  |
| Sampling IoU-Based Sampling [29]                                        | IoU                   | Samples equal number of negatives from IoU bins                                                        | K=2              | Does not specify a crite- rion for positives                    | Does not specify a crite- rion for positives                    | Random Sample × 1                                                                                                  | Random Sample × 1                                                                                                  | Random Sample × 1                                                                                                  | Random Sample × 1                                                                                                  |
| Hard IoU Lower Bound [17]                                               | IoU=0.05              | Discards neg. having IoU less than lower bound                                                         | -                | Does not specify a crite- rion for positives                    | Does not specify a crite- rion for positives                    | Random Sample × 2                                                                                                  | Random Sample × 2                                                                                                  | Random Sample × 2                                                                                                  | 7                                                                                                                  |
| Objectness Prior [56]                                                   | -                     | Learns to predict objectness priors for pos.                                                           | -                | Chooses all with prior larger than threshold.                   | Chooses all with prior larger than threshold.                   | The negatives are sampled randomly by preserving 1:3 ratio with positives. So, fixed batch size is not considered. | The negatives are sampled randomly by preserving 1:3 ratio with positives. So, fixed batch size is not considered. | The negatives are sampled randomly by preserving 1:3 ratio with positives. So, fixed batch size is not considered. | The negatives are sampled randomly by preserving 1:3 ratio with positives. So, fixed batch size is not considered. |
| Negative Anchor Filter- ing [57]                                        | p s = 0 . 9           | Discards negs. less than threshold                                                                     | -                | Does not specify a crite- rion for positives                    | Does not specify a crite- rion for positives                    | Uses OHEM for the remaining negatives. In this case, the first two boxes are selected                              | Uses OHEM for the remaining negatives. In this case, the first two boxes are selected                              | Uses OHEM for the remaining negatives. In this case, the first two boxes are selected                              | 7                                                                                                                  |
| Focal Loss [22]                                                         | p s                   | Promotes examples with larger loss values                                                              | γ = 1            | 0.2                                                             | 0.8                                                             | 0.9                                                                                                                | 0.7                                                                                                                | 0.4                                                                                                                | 0.0                                                                                                                |
| Sampling Gradient Harmonizing Mechanism [59]                            | p s                   | Promotes hard examples by suppressing effects of outliers                                              | ϵ = 0 . 4        | 0.66                                                            | 0.66                                                            | 0.60                                                                                                               | 0.66                                                                                                               | 1.2                                                                                                                | 0.60                                                                                                               |
| Soft Prime Sample Attention [30]                                        | IoU, p                | Promotes examples with larger IoUs for positives and larger fg scores for negatives (i.e. 1- p s here) | γ = 1 β = 0      | 1.00                                                            | 0.75                                                            | 1.00                                                                                                               | 0.75                                                                                                               | 0.50                                                                                                               | 0.25                                                                                                               |

<!-- formula-not-decoded -->

where P , N are the set of positive and negative labeled examples respectively. Note that U ij is zero if p i &lt; p j and a positive value less than one otherwise. With this quantity, AP loss is defined as follows:

<!-- formula-not-decoded -->

where y ij is the ranking label, set to 1 only if the i th box is a foreground bounding box and j th box is a background bounding box.

To make this differentiable, the authors propose a novel error-driven update rule, which performs better than the previous works aiming to include average precision as a training objective [97], [98]. However, this optimization algorithm is beyond the scope of our work.

## 4.1.4 Generative Methods

Unlike sampling-based and sampling-free methods, generative methods address imbalance by directly producing and injecting artificial samples into the training dataset. Table 4 presents a summary of the main approaches.

One approach is to use generative adversarial networks (GANs). A merit of GANs is that they adapt themselves to generate harder examples during training since the loss values of these networks are directly based on the classification accuracy of the generated examples in the final detection. An example is the Adversarial-Fast-RCNN model [63], which generates hard examples with occlusion and various deformations. In this method, the generative manipulation is directly performed at the feature-level, by taking the fixed size feature maps after RoI standardization layers (i.e. RoI pooling [17]). For this purpose, they proposed two networks: (i) adversarial spatial dropout network for occluded feature map generation, and (ii) adversarial spatial transformer network for deformed (transformed) feature map generation. These two networks are placed sequentially in the network design in order to provide harder examples and they are integrated into the conventional object training pipeline in an end-to-end manner.

Alternatively, artificial images can be produced to augment the dataset [64], [65], [99], [100] by generating composite images in which multiple crops and/or images are blended. A straightforward approach is to randomly place cropped objects onto images as done by Dwibedi et al. [99]. However, the produced images may look unrealistic. This problem is alleviated by determining where to paste and the size of the pasted region according to the visual context

[100]. In a similar vein, the objects can be swapped between images: Progressive and Selective Instance-Switching (PSIS) [65] swaps single objects belonging to the same class between a pair of images considering also the scales and shapes of the candidate instances. Producing images by swapping objects of low-performing classes improves detection quality. For this reason, they use the performance ranking of the classes while determining the objects to swap and the number of images to generate.

A more prominent approach is to use GANs to generate images rather than copying existing objects: an example is Task Aware Data Synthesis [64] which uses three competing networks to generate hard examples: a synthesizer, a discriminator and a target network where the synthesizer is expected to fool both the discriminator and the target network by yielding high quality synthetic hard images. Given an image and a foreground object mask, the synthesizer aims to place the foreground object mask onto the image to produce realistic hard examples. The discriminator is adopted in order to enforce the synthesizer towards realistic composite images. The target network is an object detector, initially pretrained to have a baseline performance.

Instead of generating images, the Positive RoI (pRoI) Generator [66] generates a set of positive RoIs with given IoU, BB relative spatial and foreground class distributions. The approach relies on a bounding box generator that is able to generate bounding boxes (i.e. positive example) with the desired IoU with a given bounding box (i.e. ground truth). Noting that the IoU of an input BB is related to its hardness [29], the pRoI generator is a basis for simulating, thereby analyzing, hard sampling methods (Section 4.1.1).

## 4.2 Foreground-Foreground Class Imbalance

Definition. In foreground-foreground class imbalance, the over-represented and the under-represented classes are both foreground classes. This imbalance among the foreground classes has not attracted as much interest as foregroundbackground imbalance. We discuss this problem in two categories according to their origin: (i) dataset-level, and (ii) batch-level.

## 4.2.1 Foreground-Foreground Imbalance Owing to the Dataset

Definition. Objects exist at different frequencies in nature, and therefore, there is naturally an imbalance among the object classes in datasets - see Figure 5(a) which shows that the datasets suffer from significant gap in class examples. For this reason, overfitting in favor of the over-represented classes may be inevitable for naive approaches on such datasets.

Solutions. Owing to the fact that some of the generative methods are able to generate new images or bounding boxes (see Section 4.1.4) that cannot be obtained by the conventional training pipeline, these methods can also be adopted to alleviate the foreground-foreground class imbalance problem.

Another method addressing this imbalance problem from the object detection perspective is proposed by Ouyang et al. [25]. Their method of finetuning long-tail distribution for object detection (here, 'long tail' corresponds to under-represented classes) provides an analysis on the effects of this level to the training process and uses clustering based on visual similarity. In their analysis, two factors affecting the training are identified: (i) the accuracy of the prediction, and (ii) the number of examples. Based on this observation, they handcrafted a similarity measure among classes based on the inner product of the features of the last layer of the pretrained backbone network (i.e. GoogLe Net [101]), and grouped the classes hierarchically in order to compensate for dataset-level foreground class imbalance. For each node in the designed hierarchy tree, a classifier is learned based on the confidence scores of the classifier. The leaves of the tree are basically an SVM classifier that determines the final detection for the given input BB.

## 4.2.2 Foreground-Foreground Imbalance Owing to the Batch

Definition. The distribution of classes in a batch may be uneven, introducing a bias in learning. To illustrate the batch-level foreground imbalance, Figure 4(b) provides the mean number of anchors per class on the MS COCO dataset [90]. A random sampling approach is expected to allocate an unbalanced number of positive examples in favor of the one with more anchors, which may lead the model to be biased in favor of the over-represented class during training. Also see Figure 5(b) and (c), which display that the numbers of objects and classes in an image vary significantly.

Solutions. A solution for this problem, Online Foreground Balanced (OFB) sampling by Oksuz et a. [66], shows that the foreground-foreground class imbalance problem can be alleviated at the batch level by assigning probabilities to each bounding box to be sampled, so that the distribution of different classes within a batch is uniform. In other words, the approach aims to promote the classes with lower number of positive examples during sampling. While the method is efficient, the performance improvement is not significant. Moreover, the authors have not provided an analysis on whether or not batch-level imbalance causes a bias in learning, therefore, we discuss this as an open problem in Section 4.4.

## 4.3 Comparative Summary

In early deep object detectors, hard sampling methods were commonly used to ensure balanced training. OHEM [24] was one of the most effective methods with a relative improvement of 14 . 7% over the Fast R-CNN baseline (with VGG-16 backbone) on MS COCO 2015. On the other hand, recent methods aim to use all examples either by soft sampling or without sampling. Among the seven papers investigated under these categories, six of them have been published during the last year. The pioneering Focal Loss method [22] achieves 9 . 3% relative improvement compared to the baseline alpha-balanced cross entropy on Retina-Net (i.e. from 31 . 1 to 34 . 0 mAP 5 ). AP Loss [61], with a relative improvement of 3 . 9% over Focal Loss (i.e. from 33 . 9 to 35 . 0 mAP), is also promising.

Sampling methods for positive examples are not many. One approach, prime sample attention (PISA) [30], reported a relative improvement of 4 . 4% (from 36 . 4 to 38 . 0 mAP when applied to positives only) and showed that sampling also matters for positives. We notice several crucial points regarding prime sampling. First of all, contrary to the popular belief that hard examples are more preferable over easy examples, PISA shows that, if balanced properly, the positive examples with higher IoUs, which normally incur smaller loss values, are more useful for training compared to OHEM [24] applied to positives. Moreover, the results suggest that the major improvement of the method is on localization since there is no performance improvement in AP @0 . 50 , and there is significant improvement for APs with higher IoUs (i.e. up to 2 . 6% in AP @0 . 75 ). As a result, the improvement can be due to the changing nature of the IoU distribution rather than presenting more descriptive samples to the classifier since the classifier performs worse but the regressor improves (see the discussion on IoU distribution imbalance in Section 6.2).

5. Unless otherwise stated, MS COCO 2017 minival results are reported using the COCO-style mAP.

TABLE 4: Comparison of major generative methods addressing class imbalance.

| Generative Method              | Generates                                                                                                                        |
|--------------------------------|----------------------------------------------------------------------------------------------------------------------------------|
| Adversarial-Fast-RCNN [63]     | Occluded and spatially transformed features during RoI pooling in order to make the examples harder                              |
| Task Aware Data Synthesis [64] | Images with hard examples in a GAN setting in which given foreground mask is to be placed onto the given image by the generator. |
| PSIS [65]                      | Images by switching the instances between existing images considering the performance of the class during training               |
| pRoI Generator [66]            | Positive RoIs (i.e. BBs) following desired IoU, spatial and foreground class distributions                                       |

Fig. 5: Some statistics of common datasets (training sets). For readability, the y axes are in logarithmic scale. (a) The total number of examples from each class. (b) The number of images vs. the number of examples. (c) The number of images vs. the number of classes. (This figure is inspired from Kuznetsova et al. [91] but calculated and drawn from scratch).

<!-- image -->

## 4.4 Open Issues

As we have highlighted before, class imbalance problem can be analyzed in two main categories: foreground-background class imbalance and foreground-foreground class imbalance. In the following, we identify issues to be addressed with a more focus on foreground-foreground imbalance since it has received less attention.

## 4.4.1 Sampling More Useful Examples

Open Issue. Many criteria to identify useful examples (e.g. hard example, example with higher IoUs etc.) for both positive and negative classes have been proposed. However, recent studies point out interesting phenomena that need further investigation: (i) For the foreground-background class imbalance, soft sampling methods have become more prominent and optimal weighting of examples needs further investigation. (ii) Hard example mining [24] for the positive examples is challenged by the idea that favors examples with higher IoUs, i.e. the easier ones [30]. (iii) The usefulness of the examples are not considered in terms of foregroundforeground class imbalance.

Explanation 6 . In terms of the usefulness of the examples, there are two criteria to be identified: (i) The usefulness of the background examples, and (ii) the usefulness of the foreground examples.

The existing approaches mostly concentrated around the first criterion using different properties (i.e. IoU, loss value, ground truth confidence score) to sample a useful example for the background. However, the debate is still open after Li et al. [59] showed that there are large number of outliers during sampling using these properties, which will result in higher loss values and lower confidence scores. Moreover, the methods preferring a weighting over all the examples

6. Issues that may not be immediately clear include a detailing explanation section

[22], [59] rather than discarding a large portion of samples have proven to yield more performance improvement. For this reason, currently, soft sampling approaches that assign weights to the examples are more popular, but which negative examples are more useful needs more investigation.

idea

the

For the foreground examples, Cao et al. [30] apply a specific sampling methodology to the positives based on the IoU, which has proven useful. Note that, this conflicts with the hard example mining approach since while OHEM [24] offers to pick the difficult samples, prime samples concentrate on the positive samples with higher IoUs with the ground truth, namely the easier ones. For this reason, currently it seems that the usefulness of examples is different for the positives and negatives. To sum up, further investigation is required for identifying the best set of examples, or figuring out how to weigh the positive examples during training.

Moreover, sampling methods have proven to be useful for foreground-background class imbalance, however, their effectiveness for the foreground-foreground class imbalance problem needs to be investigated.

## 4.4.2 Foreground-Foreground Class Imbalance Problem

Open Issue. Foreground-foreground class imbalance has not been addressed as thoroughly as foreground-background class imbalance. For instance, it is still not known why a class performs worse than others; a recent work [65] discredits the correlation between the number of examples in a class and its detection performance. Moreover, despite its differences, the rich literature for addressing imbalance in image classification has not been utilized in object detection.

Explanation. One important finding of a recent study [65] is that a class with the fewest examples in the dataset can yield one of the best detection performances and thus the total number of instances in the dataset is not the only issue to balance the performance of foreground classes. Such discrepancies presents the necessity of an in-depth analysis to identify the root cause and investigate for better sampling mechanisms to employ while balancing a dataset.

Moreover, we identify a similarity and a difference between the class imbalance from image classification perspective and the foreground-foreground class imbalance problem. The similarity is that neither has a background class. On the other hand, the input BBs are labeled and sampled in an online fashion in object detection, which makes the data that the detector is trained with not static.

Class imbalance problem is addressed in image classification from a larger scope by using not only classical oversampling and under-sampling methods but also (i) transfer learning to transfer the information from over-represented to under-represented classes, (ii) data redundancy methods to be useful for under-sampling the over-represented classes, (iii) weak supervision in order to be used in favor of under-represented class and (iv) a specific loss function for balancing foreground classes. Such approaches can be adopted for addressing foreground-foreground imbalance in object detection as well.

Fig. 6: Illustration of batch-level class imbalance. (a) An example that is consistent with the overall dataset (person class has more instances than parking meter). (b) An example that has a different distribution from the dataset. Images are from the MS COCO dataset.

<!-- image -->

## 4.4.3 Foreground-Foreground Imbalance Owing to the Batch

Open Issue. The distribution of the foreground classes in a batch might be very different than that of the overall dataset, especially when the batch size is small. An important question is whether this may have an influence on the overall performance.

Explanation. A specific example is provided in Figure 6 from the MS COCO dataset [90]: An over-represented class ('person' class in this example) across the dataset may be under-represented in a batch or vice versa. Similar to the foreground-background class imbalance problem, overrepresenting a class in a batch will increase the probability of the corresponding class to dominate more.

Even when a batch has uniform foreground-foreground class distribution, an imbalance may occur during sampling due to the fact that sampling algorithms either select a subset or apply a weighting to the large number of boxes. If the sampling mechanism tends to choose the samples from specific classes in the image, balancing the dataset (or the batch) may not be sufficient to address the foregroundforeground class imbalance entirely.

## 4.4.4 Ranking-Based Loss Functions

Open Issue. AP Loss [61] sorts the confidence scores pertaining to all classes together to make a ranking between the detection boxes. However, this conflicts with the observation that the optimal confidence scores vary from class to class [102].

Explanation. Chen et al. [61] use all the confidence scores all together without paying attention to the fact that the optimal threshold per class may vary [102]. Therefore, a method that sorts the confidence scores in a class-specific manner might achieve a better performance. Addressing this issue is an open problem.

## 5 IMBALANCE 2: SCALE IMBALANCE

We discuss scale the imbalance problem in two parts: the first part, Object/Box-Level Scale Imbalance, analyses the problems arising from the imbalanced distribution of the

<!-- image -->

(a)

(b)

(c)

Fig. 7: Imbalance in scales of the objects in common datasets: the distributions of BB width

(a)

Values are with respect to the normalized image (i.e. relative to the image). For readability, the

y

Fig. 8: An illustration and comparison of the solutions for scale imbalance. 'Predict' refers to the prediction performed by a detection network. The layered boxes correspond to convolutional layers. (a) No scale balancing method is employed. (b) Prediction is performed from backbone features at different levels (i.e. scales) (e.g. SSD [19]). (c) The intermediate features from different scales are combined before making prediction at multiple scales (e.g. Feature Pyramid Networks [26]). (d) The input image is scaled first and then processed. Each I corresponds to an image pyramidal feature (e.g. Image Pyramids [27]). (e) Image and feature pyramids are combined. Rather than applying backbones, light networks are used in order to extract features from smaller images.

<!-- image -->

scales of the objects and input bounding boxes. The second part, Feature Imbalance, analyses the problems arising at the feature extraction level and concerns the methods using pyramidal features.

## 5.1 Object/Box-Level Scale Imbalance

Definition. Scale imbalance occurs when certain sizes of the objects or input bounding boxes are over-represented in the dataset. It has been shown that this affects the scales of the estimated RoIs and the overall detection performance [27]. Figure 7 presents the relative width, height and the area of the objects in the MS-COCO dataset [90]; we observe a skewness in the distributions in favor of smaller objects.

Many of the deep object detectors rely on a backbone convolutional neural network (e.g. [93], [101], [103], [104],

, height (b) , and area (c) . axes are in log-scale.

[105]), pretrained on an image classification task, in order to extract visual features from the input image. Li et al. [106] discuss the cons of employing such networks designed for the classification task and propose a backbone specially designed for the object detection task where they limit the spatial downsampling rate for the high level features. Overall, these networks, also called the backbones, play an important role for the performance of the object detectors, but they alone are insufficient for handling the scale diversity of the input bounding boxes.

Solutions. First examples of deep object detectors made predictions from the final layer of the backbone network (see [16], [17] and Figure 8(a)), and therefore, neglected the scalediversity of BBs. The solutions to addressing scale imbalance can be grouped into four (Figure 8): methods predicting from the hierarchy of the backbone features (Figure 8(b)), methods based on feature pyramids (Figure 8(c)), methods based on image pyramids (Figure 8(d)) and finally methods combining image and feature pyramids (Figure 8(e)).

## 5.1.1 Methods Predicting from the Feature Hierarchy of Backbone Features

These methods make independent predictions from the features at different levels of the backbone network (Figure 8(b)). This approach naturally considers object detection at multiple scales since the different levels encode information at different scales; e.g., if the input contains a small object, then earlier levels already contain strong indicators about the small object [69].

An illustratory example for one-stage detectors is the Single Shot Detector ( SSD ) [19], which makes predictions from features at different layers.

Two-stage detectors can exploit features at different scales while either estimating the regions (in the first stage) or extracting features from these regions (for the second stage). For example, the Multi Scale CNN (MSCNN) [70] uses different layers of the backbone network while estimating the regions in the first stage whereas Yang et al. [69] choose an appropriate layer to pool based on the scale of the estimated RoI; called Scale Dependent Pooling (SDP), the method, e.g., pools features from an earlier layer if the height of the RoI is small. Alternatively, the Scale Aware Fast R-CNN [71] learns an ensemble of two classifiers, one for the small scale and one for the large scale objects, and combines their predictions.

## 5.1.2 Methods Based on Feature Pyramids

Methods based on feature hierarchies use features from different levels independently without integrating low-level and high-level features. However, the abstractness (semantic content) of information varies among different layers, and thus it is not reliable to make predictions directly from a single layer (especially the lower layers) of the backbone network.

To address this issue, the Feature Pyramid Networks (FPN) [26] combine the features at different scales before making predictions. FPN exploits an additional top-down pathway along which the features from the higher level are supported by the features from a lower level using lateral connections in order to have a balanced mixed of these features (see Figure 8(c)). The top-down pathway involves upsampling to ensure the sizes to be compatible and lateral connections are basically 1 × 1 convolutions. Similar to feature hierarchies, a RoI pooling step takes the scale of the RoI into account to choose which level to pool from. These improvements allow the predictor network to be applied at all levels which improves the performance especially for small and medium sized objects.

Although FPN was originally proposed for object detection, it quickly became popular and has been used for different (but related) tasks such as shadow detection [107], instance segmentation [108], [109] and panoptic segmentation [110].

Despite its benefits, FPN is known to suffer from a major shortcoming due to the straightforward combination of the features gathered from the backbone network - the feature imbalance problem . We discuss this problem in Section 5.2.

## 5.1.3 Methods Based on Image Pyramids

The idea of using multi-scale image pyramids, presented in Figure 8(d), for the image processing tasks goes back to the early work by Adelson et al. [111] and was popular before deep learning. In deep learning, these methods are not utilized as much due to their relatively high computational and memory costs. However, recently, Singh and Davis [27] presented a detailed analysis on the effect of the scale imbalance problem with important conclusions and proposed a method to alleviate the memory constraint for image pyramids. They investigated the conventional approach of training object detectors in smaller scales but testing them in larger scales due to memory constraints, and showed that this inconsistency between test and training time scales has an impact on the performance. In their controlled experiments, upsampling the image by two performed better than reducing the stride by two. Upon their analysis, the authors proposed a novel training method coined as SNIP based on image pyramids rather than feature pyramids. They argue that, while training scale-specific detectors by providing the input to the appropriate detector will lose a significant portion of the data, and that using multiscale training on a single detector will increase the scale imbalance by preserving the variation in the data. Therefore, SNIP trains multiple proposal and detector networks with images at different sizes, however, for each network only the appropriate input bounding box scales are marked as valid, by which it ensures multi-scale training without any loss in the data. Another challenge, the limitation of the GPU memory, is overcome by an image cropping approach. The image cropping approach is made more efficient in a followup method, called SNIPER [28].

## 5.1.4 Methods Combining Image and Feature Pyramids

Image pyramid based methods are generally less efficient than feature pyramid based methods in terms of computational time and memory. However, image pyramid based methods are expected to perform better, since feature pyramid based methods are efficient approximations of such methods. Therefore, to benefit from advantages of both approaches, they can be combined in a single model.

Even though there are different architectures, the generic approach is illustrated in Figure 8(e). For example, Efficient Featurized Image Pyramids [72] uses five images at different scales, four of which are provided to the light-weight featurized image pyramid network module (i.e. instead of additional backbone networks) and the original input is fed into the backbone network. This light-weight network consists of 4 consecutive convolutional layers designed specifically for each input. The features gathered from this module are integrated to the backbone network features at appropriate levels according to their sizes, such that the image features are combined with the features extracted from the backbone network using feature attention modules. Furthermore, after attention modules, the gathered features are integrated with the higher levels by means of forward fusion modules before the final predictions are obtained.

A similar method that is also built on SSD and uses downsampled images is Enriched Feature Guided Refinement Network [58] in which a single downsampled image is provided as an input to the Multi-Scale Contextual Features (MSCF) Module. The MSCF consists of two consecutive convolutional layers followed by three parallel dilated convolutions, which is also similar to the idea in Trident Networks [74]. The outputs of the dilated convolutions are again combined using 1 × 1 convolutions. The authors set the downsampled image size to meet the first prediction layer in the regular SSD architecture (e.g. for 320 × 320 input, the size of the downsampled image is 40 × 40 ). While Pang et al. [72] only provide experiments with the SSD backbone, Nie et al. [58] show that their modules can also be used in the ResNet backbone.

An alternative approach is to generate super-resolution feature maps. Noh et al. [73] proposed super-resolution for small object detection for two-stage object detectors which lack strong representations of small RoIs after RoI standardization layers. Their architecture adds four additional modules to the baseline detector: (i) Given the original image, target extractor outputs the targets for the discriminator by using dilated convolutions. This network also shares parameters with the backbone network. (ii) Given the features obtained from the backbone network using the original image and a 0 . 5 x downsampled image, the generator network generates super resolution feature maps for small RoIs. (iii) Given the outputs of (i) and (ii), a discriminator is trained in the conventional GAN setting. (iv) Finally, if the RoI is a small object according to a threshold, then the prediction is carried out by the small predictor, or else by the large predictor.

OKSUZ

et al.

: IMBALANCE PROBLEMS IN OBJECT DETECTION: A REVIEW

Fig. 9: Feature-Level imbalance is illustrated on the FPN architecture.

<!-- image -->

Another approach, Scale Aware Trident Networks [74], combines the advantages of the methods based on feature pyramids and image pyramids without using multiple downsampled images but employing only dilated convolutions. The authors use dilated convolutions [112] with dilation rates 1 , 2 and 3 in parallel branches in order to generate scale-specific feature maps, making the approach more accurate compared to feature pyramid based methods. In order to ensure that each branch is specialized for a specific scale, an input bounding box is provided to the appropriate branch according to its size. Their analysis on the effect of receptive field size on objects of different scales shows that larger dilation rates are more appropriate for objects with larger scales. In addition, since using multiple branches is expected to degrade the efficiency due to the increasing number of operations, they proposed a method for approximating these branches with a single parametersharing branch, with minimal (insignificant) performance loss.

## 5.2 Feature-level Imbalance

Definition. The integration of the features from the backbone network is expected to be balanced in terms of low- and high-level features so that consistent predictions can follow. To be more specific, if we analyze the conventional FPN architecture in Figure 9, we notice that, while there are several layers from the C 2 layer of the bottom-up pass with low-level features to the P 5 layer of the feature pyramid, the C 2 layer is directly integrated to the P 2 layer, which implies the effect of the high-level and low-level features in the P 2 and P 5 layers to be different.

Solutions. There are several methods to address imbalance in the FPN architectures, which range from designing improved top-down pathway connections [56], [57] to completely novel architectures. Here, we consider the methods to alleviate the feature-level imbalance problem using novel architectures, which we group into two according to what they use as a basis, pyramidal or backbone features.

## 5.2.1 Methods Using Pyramidal Features as a Basis

These methods aim to improve the pyramidal features gathered by FPN using additional operations or steps - see an overview of these methods in Figure 10(a,b).

Path Aggregation Network (PANet) [75] is the first to show that the features gathered by an FPN can be further enhanced and an RoI can be mapped to each layer of the pyramid rather than associating it with a single one. The authors suggest that low-level features, such as edges, corners, are useful for localizing objects, however, the FPN architecture does not sufficiently make use of these features. Motivated from this observation, PANet, depicted in Figure 10(a), improves the FPN architecture with two new contributions:

- 1) Bottom-up path augmentation extends the feature pyramid in order to allow the low-level features to arrive at the layers where the predictions occur in shorter steps (see red arrows in Figure 10(a) within FPN and to the final pyramidal features to see the shortcut). For this reason, in a way a shortcut is created for the features in the initial layers. This is important since these features have rich information about localization thanks to edges or instance parts.
- 2) While in the FPN, each RoI is associated with a single level of feature based on its size, PANet associates each RoI to every level, applies RoI Pooling, fuses using element-wise max or sum operation and the resulting fixed-sized feature grid is propagated to the detector network. This process is called Adaptive Feature Pooling .

Despite these contributions, PANet still uses a sequential pathway to extract the features.

Different from the sequential enhancement pathway of PANet, Libra FPN [29] aims to learn the residual features by using all of the features from all FPN layers at once (see Figure 10(b)). Residual feature layer computation is handled in two steps:

- 1) Integrate: All feature maps from different layers are reduced to one single feature map by rescaling and averaging. For this reason, this step does not have any learnable parameter.
- 2) Refine: The integrated feature map is refined by means of convolution layers or non-local neural networks [113].

Finally the refined features are added to each layer of the pyramidal features. The authors argue that in addition to FPN, their method is complementary to other methods based on pyramidal features as well, such as PANet [75].

## 5.2.2 Methods Using Backbone Features as a Basis

These methods build their architecture on the backbone features and ignore the top-down pathway of FPN by employing different feature integration mechanisms, as displayed in Figure 10(c-h).

Scale-Transferrable Detection Network (STDN) [76] generates the pyramidal features from the last layer of the backbone features which are extracted using DenseNet [114] blocks (Figure 10(c)). In a DenseNet block, all the lower level features are propagated to every next layer within a block. In Figure 10(c), the number of DenseNet (dense) blocks is four and the i th block is denoted by D i . Motivated by the idea that direct propagation of lower-level layers to the subsequent layers also carries lower-level information, STDN builds pyramidal features consisting of six layers by using the last block of DenseNet. In order to map these layers to lower sizes, the approach uses mean pooling with different receptive field sizes. For the fourth feature map, an identity mapping is used. For the last two layers which the feature maps of DenseNet are to be mapped to higher dimensions, the authors propose a scale transfer layer approach. This layer does not have any learnable parameter and given r , the desired enlargement for a feature map, the width and height of the feature map are enlarged by r by decreasing the total number of feature maps (a.k.a. channels). STDN incorporates highand low-level features with the help of DenseNet blocks and is not easily adaptable to other backbone networks. In addition, no method is adopted to balance the low- and high-level features within the last block of DenseNet.

Fig. 10: High-level diagrams of the methods designed for feature-level imbalance. (a) Path Aggregation Network . FPN is augmented by an additional bottom-up pathway to facilitate a shortcut of the low-level features to the final pyramidal feature maps. Red arrows represent the shortcuts. (b) Libra FPN . FPN pyramidal features are integrated and refined to learn a residual feature map. We illustrate the process originating from P 2 feature map. Remaining feature maps ( P 3 - P 5 ) follow the same operations. (c) Scale Transferrable Detection Network . Pyramidal features are learned via pooling, identity mapping and scale transfer layers depending on the layer size. (d) Parallel FPN . Feature maps with difference scales are followed by spatial pyramid pooling. These feature maps are fed into the multi-scale context aggregation (MSCA) module. We show the input and outputs of MSCA module for P 3 by red arrows. (e) Deep Feature Pyramid Reconfiguration . A set of residual features are learned via global attention and local reconfiguration modules. We illustrate the process originating from P 2 feature map. Remaining feature maps ( P 3 - P 5 ) follow the same operations. (f) Zoom Out-And-In Network . Azoomin phase based on deconvolution (shown with red arrows) is adopted before stacking the layers of zoom-out and zoom-in phases. The weighting between them is determined by map attention decision module. (g) Multi-Level FPN . Backbone features from two different levels are fed into Thinned U-Shape Module (TUM) recursively to generate a sequence of pyramidal features, which are finally combined into one by acale-wise feature aggregation module. (h) NAS-FPN. The layers between backbone features and pyramidal features are learned via Neural Architecture Search.

<!-- image -->

Similar to STDN, Parallel FPN [77] also employs only the last layer of the backbone network and generates multiscale features by exploiting the spatial pyramid pooling (SPP) [115] - Figure 10(d). Differently, it increases the width of the network by pooling the last D feature maps of the backbone network multiple times with different sizes, such that feature maps with different scales are obtained. Figure 10(e) shows the case when it is pooled for three times and D = 2 . The number of feature maps is decreased to 1 by employing 1 × 1 convolutions. These feature maps are then fed into the multi-scale context aggregation (MSCA) module , which integrates context information from other scales for a corresponding layer. For this reason, MSCA, operating on a scale-based manner, has the following inputs: Spatial pyramid pooled D feature maps and reduced feature maps from other scales. We illustrate the inputs and outputs to the MSCA module for the middle scale feature map by red arrows in Figure 10(e). MSCA first ensures the sizes of each feature map to be equal and applies 3 × 3 convolutions.

While previous methods based on backbone features only use the last layer of the backbone network, Deep Feature Pyramid Reconfiguration [78] combines features from different levels of backbone features into a single tensor ( X in Figure 10(e)) and then learns a set of residual features from this tensor. A sequence of two modules are applied to the tensor X in order to learn a residual feature map to be added to each layer of the backbone network. These modules are,

- 1) Global Attention Module aims to learn the interdependencies among different feature maps for tensor X . The authors adopt Squeeze and Excitation Blocks [116] in which the information is squeezed to lower dimensional features for each feature map initially (i.e. squeeze step), and then a weight is learned for each feature map based on learnable functions including non-linearity (i.e. excitation step).
- 2) Local Configuration Module aims to improve the features after global attention module by employing convolutional layers. The output of this module presents the residual features to be added for a feature layer from the backbone network.

Similarly, Zoom Out-and-In Network [79] also combines low- and high-level features of the backbone network. Additionally, it includes deconvolution-based zoom-in phase in which intermediate step pyramidal features, denoted by Bi s in Figure 10(f), are learned. Note that, unlike FPN [26], there is no lateral connection to the backbone network during the zoom-in phase, which is basically a sequence of deconvolutional layers (see red arrows for the zoom-in phase).

Integration of the high- and low-level features are achieved by stacking the same-size feature maps from zoom-out and zoom-in phases after zoom-in phase (i.e. B 3 and C 3 ). On the other hand, these concatenated feature maps are to be balanced especially for the B 3 - C 3 and B 4 - C 4 blocks since B 3 and C 3 (or B 4 and C 4 ) are very far from each other in the feature hierarchy, which makes them have different representations of the data. In order to achieve this, the proposed map attention decision module learns a weight distribution on the layers. Note that the idea is similar to the squeeze and excitation modules [116] employed by Kong et al. [78], however, it is shown by the authors that their design performs better for their architecture. One drawback of the method is that it is built upon Inception v2 (a.k.a. Inception BN) [117] and corresponding inception modules are exploited throughout the method, which may make the method difficult to adopt for other backbone networks.

Different from Kong et al. [78] and Li et al. [79], MultiLevel FPN [80] stacks one highest and one lower level feature layers and recursively outputs a set of pyramidal features, which are all finally combined into a single feature pyramid in a scale-wise manner (see Figure 10(g)). Feature fusion module (FFM) v1 equals the dimensions of the input feature maps by a sequence of 1 × 1 convolution and upsampling operations. Then, the resulting two-layer features are propagated to thinned U-shape modules (TUM) . Excluding the initial propagation, each time these two-layer features are integrated to the output of the previous TUM by 1 × 1 convolutions in FFMv2 . Note that the depth of the network is increasing after each application of the TUM and the features are becoming more high level. As a result of this, a similar problem with the FPN feature imbalance arise again. As in the work proposed by [78], the authors employed squeeze and excitation networks [116] to combine different pyramidal shape features.

Rather than using hand-crafted architectures, Neural Architecture Search FPN (NAS-FPN) [81] aims to search for the best architecture to generate pyramidal features given the backbone features by using neural architecture search methods [118] - Figure 10(h). This idea was also previously applied to the image classification task and showed to perform well [119], [120], [121]. Auto-FPN is also another example for using NAS while learning the connections from backbone features to pyramidal features and beyond. While NAS-FPN achieves higher performance, Auto-FPN is more efficient and has less memory footprint. The idea is also applied to backbone design for object detection by Chen et al. [122], however it is not within the scope of our paper. Considering their performance in other tasks such as EfficientNet [121] and different definitions of search spaces maylead to better performance in NAS methods, more work is expected for FPN design using NAS.

## 5.3 Comparative Summary

SSD [19] provides an analysis on the importance of making predictions from different number of layers with varying scales. Increasing the number of prediction layers leads to a significant performance gain. While predicting from one layer provides 62 . 4 mAP@0.5 on Pascal VOC 2007 test set [51] with SSD-300, it is 70 . 7 and 74 . 6 when the number of prediction layers is 3 and 5 respectively (while keeping the number of predictions almost equal). Therefore, once addressed in the detection pipeline, scale imbalance methods can significantly boost performance.

Feature pyramid based methods increased the performance significantly compared to SSD. Once included in Faster R-CNN with ResNet-101, the pioneering FPN study [26] has a relative improvement of 3 . 7% on MS COCO testdev (from 34 . 9 to 36 . 2 ). Another virtue of the feature pyramids is the efficiency due to having lighter detection networks: e.g., the model with FPN is reported to be more than two times faster than baseline Faster R-CNN with ResNet-50 backbone ( 150 ms vs 320 ms per image on a single NVIDIA M40 GPU) [26]. Currently, the most promising results are obtained via NAS by learning how to combine the backbone features from different levels. Using the ResNet-50 backbone with 1024 × 1024 image size, a 10 . 2% relative improvement is obtained on MS COCO testdev (from 40 . 1 to 44 . 2 ) by NAS-FPN [81] but it needs also to be noted that number of parameters in the learned structure is approximately two times higher and inference time is 26% more ( 92 . 1 ms vs 73 . 0 ms per image on a single P100 GPU). Inference time is also a major drawback of the image pyramid based methods. While SNIP reports inference at approximately 1 image/sec, the faster version, SNIPER, improves it to 5 images/sec on a V100 GPU ( 200 ms/image).

In order to adjust the inference time-performance tradeoff in scale imbalance, a common method is to use multiscale images with one backbone and multiple lighter networks. All methods in Section 5.1.4 are published last year. To illustrate the performance gains: Pang et al. [72] achieved 19 . 5% on MS COCO test-dev relative improvement with 12 ms per image inference time compared to SSD300 with the same size image and backbone network having 10 ms per image inference time (mAPs are 25 . 1 vs 30 . 0 ).

## 5.4 Open Issues

Here we discuss open problems concerning scale imbalance.

## 5.4.1 Characteristics of Different Layers of Feature Hierarchies

In feature-pyramid based methods (Section 5.2), a prominent and common pattern is to include a top-down pathway in order to integrate higher-layer features with lowerlayer ones. Although this approach has yielded promising improvements in performance, an established perspective about what critical aspects of features (or information) are handled differently in those methods is missing. Here, we highlight three such aspects:

(i) Abstractness . Higher layers in a feature hierarchy carry high-level, semantically more meaningful information about the objects or object parts whereas the lower layers represent low-level information in the scene, such as edges, contours, corners etc. In other words, higher-layer features are more abstract.

(ii) Coarseness . To reduce dimensions, feature networks gradually reduce the size of the layers towards the top of the hierarchy. Although this is reasonable given the constraints, it has an immediate outcome on the number of neurons that a fixed bounding box at the image level encapsulates at the different feature layers. Namely, the BB will include less neurons when projected to the highest layer. In other words, higher layers are more coarse.

(iii) Cardinality . In FPN and many of its variants, prediction is performed for an object from the layer that matches the object's scale. Since the scales of objects are not balanced, this approach has a direct affect on the number of predictions and backpropagation performed through a layer.

We argue that analyzing and addressing these aspects in a more established manner is critical for developing more profound solutions. Although we see that some methods handle imbalance in these aspects (e.g. Libra FPN [29] addresses all three aspects, Path Aggregation Network [75] handles abstractness and cardinality, whereas FPN solves only abstractness to a certain extent), these aspects should be quantified and used for comparing different methods.

## 5.4.2 Image Pyramids in Deep Object Detectors

Open Issue. It is hard to exploit image pyramids using neural networks due to memory limitations. Therefore, finding solutions alleviating this constraint (e.g., as in SNIP [27]) is still an open problem.

Explanation. The image pyramids (Figure 8(d)) were commonly adopted by the pre-deep learning era object detectors. However, memory limitations motivated the methods based on pyramidal features which, with less memory, are still able to generate a set of features with different scales allowing predictions to occur at multiple scales. On the other hand, feature-pyramids are actually approximations of the features extracted from image pyramids, and there is still room for improvement given that using image pyramids is not common among deep object detectors.

## 6 IMBALANCE 3: SPATIAL IMBALANCE

Definition. Size, shape, location - relative to both the image or another box - and IoU are spatial attributes of bounding boxes. Any imbalance in such attributes is likely to affect the training and generalization performance. For example, a slight shift in position may lead to drastic changes in the regression (localization) loss, causing an imbalance in the loss values, if a suitable loss function is not adopted. In this section, we discuss these problems specific to the spatial attributes and regression loss.

## 6.1 Imbalance in Regression Loss

Definition. This imbalance problem is concerned with the uneven contributions of different individual examples to the regression loss. Figure 11 illustrates the problem using L 1 and L 2 losses, where the hard example (i.e. the one with low IoU, the yellow box) is dominating the L 2 loss, whereas L 1 loss assigns relatively more balanced errors to all examples. Solutions. The regression losses for object detection have evolved under two main streams: The first one is the Lp - norm-based (e.g. L 1 , L 2 ) loss functions and the second one is the IoU-based loss functions. Table 5 presents a comparison of the widely used regression losses.

Replacing L 2 regression loss [17], [123], Smooth L 1 Loss [17] is the first loss function designed specifically for deep object detectors, and it has been widely adopted (e.g. [19], DIoU Loss, CIoU Loss [86]

TABLE 5: A list of widely used loss functions for the BB regression task.

| Loss Function              | Explanation                                                                                                                                                                       |
|----------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| L 2 Loss                   | Employed in earlier deep object detectors [16]. Stable for small errors but penalizes outliers heavily.                                                                           |
| L 1 Loss                   | Not stable for small errors.                                                                                                                                                      |
| Smooth L 1 Loss [17]       | Baseline regression loss function. More robust to outliers compared to L 1 Loss.                                                                                                  |
| Balanced L 1 Loss [29]     | Increases the contribution of the inliers compared to smooth L 1 loss.                                                                                                            |
| Kullback-Leibler Loss [53] | Predicts a confidence about the input bounding box based on KL diver- gence.                                                                                                      |
| IoU Loss [83]              | Uses an indirect calculation of IoU as the loss function.                                                                                                                         |
| Bounded IoU Loss [84]      | Fixes all parameters of an input box in the IoU definition except the one whose gradient is estimated during backpropagation.                                                     |
| GIoU Loss [85]             | Extends the definition of IoU based on the smallest enclosing rectangle of the inputs to the IoU, then uses directly IoU and the extended IoU, called GIoU, as the loss function. |

<!-- image -->

Extends the definition of IoU by adding additional penalty terms concerning aspect ratio difference and center distances between two boxes.

|   IoU |    L1 |    L2 |
|-------|-------|-------|
|  0.94 |  1.00 |  0.50 |
|  0.65 |  6.00 |  6.00 |
|  0.51 | 10.00 | 14.00 |

increases the gradient contribution of the inliers to the total loss value. To achieve this, the authors first derive the definition of the loss function originating from the desirable balanced gradients across inliers and outliers:

IoU L1 L2 L1Smooth 0.88 2.00 2.00 1 0.65 6.00 4.82 2.82 0.51 10.00 6.28 4.28 Fig. 11: An illustration of imbalance in regression loss. Blue denotes the ground truth BB. There are three prediction boxes, marked with green, red and yellow colors. In the table on the right, L 1 and L 2 columns show the sum of L 1 and L 2 errors between the box corners of the associated prediction box and the ground-truth (blue) box. Note that the contribution of the yellow box to the L 2 loss is more dominating than its effect on total L 1 error. Also, the contribution of the green box is less for the L 2 error.

[21], [22]) since it reduces the effect of the outliers (compared to L 2 loss) and it is more stable for small errors (compared to L 1 loss). Smooth L 1 loss, a special case of Huber Loss [124], is defined as:

<!-- formula-not-decoded -->

where ˆ x is the difference between the estimated and the target BB coordinates.

Motivated by the fact that the gradients of the outliers still have a negative effect on learning the inliers with smaller gradients in Smooth L 1 loss, Balanced L 1 Loss [29]

<!-- formula-not-decoded -->

where α controls how much the inliers are promoted (small α increases the contribution of inliers); θ is the upper bound of the error to help balancing between the tasks. Integrating Equation (10), L 1 balanced is derived as follows:

<!-- formula-not-decoded -->

where b ensures L 1 balanced (ˆ x = 1) is a continuous function, Z is a constant and the association between the hyperparamaters is:

<!-- formula-not-decoded -->

Having put more emphasis on inliers, Balanced L 1 Loss improves the performance especially for larger IoUs (namely, AP @0 . 75 improves by %1 . 1 ).

Another approach, Kullback-Leibler Loss (KL Loss) [53], is driven by the fact that the ground truth boxes can be ambiguous in some cases due to e.g. occlusion, shape of the object or inaccurate labeling. For this reason, the authors aim to predict a probability distribution for each BB coordinate rather than direct BB prediction. The idea is similar to the networks with an additional localization confidence associated prediction branch [89], [125], [126], [127], [128], besides classification and regression, in order to use the predicted confidence during inference. Differently, KL Loss, even without its proposed NMS, has an improvement in localization compared to the baseline. The method assumes that each box coordinate is independent and follows a Gaussian distribution with mean ˆ x and standard deviation σ . Therefore, in addition to conventional boxes, a branch is added to the network to predict the standard deviation, that is σ , and the loss is backpropagated using the KL divergence between the prediction and the ground truth such that the ground truth boxes are modelled by the dirac delta distribution centered at the box coordinates. With these assumptions, KL Loss is proportional to:

<!-- formula-not-decoded -->

They also employ gradient clipping similar to smooth L 1 in order to decrease the effect of the outliers. During NMS, a voting scheme is also proposed to combine bounding boxes with different probability distributions based on the certainty of each box; however, this method is out of the scope of our paper. Note that the choice of probability distribution for bounding boxes matters since the loss definition is affected by this choice. For example, Equation (13) degenerates to Euclidean distance when σ = 1 .

In addition to the Lp -norm-based loss functions, there are also IoU based loss functions which exploit the differentiable nature of IoU. An earlier example is the IoU Loss [83], where an object detector is successfully trained by directly formulating a loss based on the IoU as:

<!-- formula-not-decoded -->

Another approach to exploit the metric-nature of 1 - IoU is the Bounded IoU loss [84]. This loss warps a modified version of 1 - IoU to the smooth L 1 function. The modification involves bounding the IoU by fixing all the parameters except the one to be computed, which implies the computation of the maximum attainable IoU for one parameter:

<!-- formula-not-decoded -->

where the bounding boxes are represented by center coordinates, width and height as [ c x , c y , w, h ] . Here, we define the bounded IoU only for c x and w since c y and h have similar definitions. We follow our notation in Section 2 to denote ground truth and detection (i.e. c x for ground truth and ¯ c x for detection). With this notation, IoU B , the bounded IoU, is defined as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In such a setting, IoU B ≥ IoU . Also, an IoU based loss function is warped into the smooth L 1 function in order to make the ranges of the classification and localization task consistent and to decrease the effect of outliers.

Motivated by the idea that the best loss function is the performance metric itself, in Generalized Intersection over Union (GIoU) [85] showed that IoU can be directly optimized and that IoU and the proposed GIoU can be used as a loss function. GIoU is proposed as both a performance measure and a loss function while amending the major drawback of the IoU (i.e. the plateau when IoU=0) by incorporating an additional smallest enclosing box E . In such a way, even when two boxes do not overlap, a GIoU value can be assigned to them and this allows the function to have non-zero gradient throughout the entire input domain rather being limited to IoU&gt;0. Unlike IoU, the GIoU( B, ¯ B ) ∈ [ - 1 , 1] . Having computed E , GIoU is defined as:

<!-- formula-not-decoded -->

where GIoU is a lower bound for IoU, and it converges to IoU when A ( B ∪ ¯ B ) = A ( E ) . GIoU preserves the advantages of IoU, and makes it differentiable when IoU=0. On the other hand, since positive labeled BBs have IoU larger than 0.5 by definition, this portion of the function is never visited in practice, yet still, GIoU Loss performs better than using IoU directly as a loss function.

A different idea proposed by Zheng et al. [86] is to add penalty terms to the conventional IoU error (i.e. 1 - IoU ( B, ¯ B ) ) in order to ensure faster and more accurate convergence. To achieve that, in Distance IoU (DIoU) Loss, a penalty term related with the distances of the centers of B and ¯ B is added as:

<!-- formula-not-decoded -->

where d () is the Euclidean distance, B C is the center point of box B and E D is the diagonal length of E (i.e. smallest enclosing box). To further enhance their method, L DIoU is extended with an additional penalty term for inconsistency in aspect ratios of two boxes. The resulting loss function, coined as Complete IoU (CIoU), is defined as:

<!-- formula-not-decoded -->

such that

and

<!-- formula-not-decoded -->

In this formulation, α , the trade-off parameter, ensures the importance of the cases with smaller IoUs to be higher. In the paper, one interesting approach is to validate faster and more accurate convergence by designing simulation scenarios since it is not straightforward to analyze IoUbased loss functions (see Section 6.5).

## 6.2 IoU Distribution Imbalance

Definition. IoU distribution imbalance is observed when input bounding boxes have a skewed IoU distribution. The problem is illustrated in Figure 12(a), where the IoU distribution of the anchors in RetinaNet [22] are observed to be

<!-- formula-not-decoded -->

Fig. 12: The IoU distribution of the positive anchors for a converged RetinaNet [22] with ResNet-50 [93] on MS COCO [90] before regression (a) , and (b) the density of how the IoU values are affected by regression ( IoU B : before regression, IoU A : after regression). For clarity, the density is plotted in log-scale. (c) A concise summary of how regression changes IoUs of anchors as a function of their starting IoUs ( IoU B ). Notice that while it is better not to regress the boxes in larger IoUs, regressor causes false positives more in lower IoUs.

<!-- image -->

skewed towards lower IoUs. It has been previously shown that this imbalance is also observed while training two-stage detectors [87]. Differently, we present in Figure 12(b) how each anchor is affected by regression. The rate of degraded anchors after regression (i.e. positive anchors under the blue line) decreases towards the threshold that the regressor is trained upon, which quantitatively confirms the claims of Cai et al. (see Figure 12(c)). On the other hand, the rate of the anchors that become a false positive (i.e. positive anchors under the red line), is increasing towards the 0 . 5 - 0 . 6 bin, for which around 5% of the positive anchors are lost by the regressor (see Figure 12(c)). Furthermore, comparing the average IoU error of the anchors before and after regression on a converged model, we notice that it is better off without applying regression and use the unregressed anchors for the IoU B intervals 0 . 8 - 0 . 9 and 0 . 9 - 1 . 0 . These results suggest that there is still room to improve by observing the effect of the regressor on the IoUs of the input bounding boxes.

Solutions. The Cascade R-CNN method [87] has been the first to address the IoU imbalance. Motivated by the arguments that (i) a single detector can be optimal for a single IoU threshold, and (ii) skewed IoU distributions make the regressor overfit for a single threshold, they show that the IoU distribution of the positive samples has an effect on the regression branch. In order to alleviate the problem, the authors trained three detectors, in a cascaded pipeline, with IoU thresholds 0 . 5 , 0 . 6 and 0 . 7 for positive samples. Each detector in the cascade uses the boxes from the previous stage rather than sampling them anew. In this way, they show that the skewness of the distribution can be shifted from the left-skewed to approximately uniform and even to the right-skewed, thereby allowing the model to have enough samples for the optimal IoU threshold that it is trained with. The authors show that such a cascaded scheme works better compared to the previous work, such as MultiRegion CNN [129] and AttractioNet [130], that iteratively apply the same network to the bounding boxes. Another cascade-based structure implicitly addressing the IoU imbalance is Hierarchical Shot Detector (HSD) [88]. Rather than using a classifier and regressor at different cascade-levels, the method runs its classifier after the boxes are regressed, resulting in a more balanced distribution.

In another set of studies, randomly generated bounding boxes are utilized to provide a set of positive input bounding boxes with balanced IoU distribution to the second stage of Faster R-CNN. IoU-uniform R-CNN [89] adds controllable jitters and in such a way provides approximately uniform positive input bounding boxes to the regressor only (i.e. the classification branch still uses the RPN RoIs). Differently, pRoI Generator [66] trains both branches with the generated RoIs but the performance improvement is not that significant probably because the training set covers a much larger space than the test set. However, one significant contribution of Oksuz et al. [66] is, rather than adding controllable jitters, they systematically generate bounding boxes using the proposed bounding box generator. Using this pRoI generator, they conducted a set of experiments for different IoU distributions and reported the following: (i) The IoU distribution of the input bounding boxes has an effect not only on the regression but also on the classification performance. (ii) Similar to the finding of Pang et al. [29], the IoU of the examples is related to their hardness. However, contrary to Cao et al. [30], who argued that OHEM [24] has an adverse effect when applied only to positive examples, Oksuz et al. [66] showed that the effect of OHEM depends on the IoU distribution of the positive input BBs. When a right-skewed IoU distribution is used with OHEM, a significant performance improvement is observed. (iii) The best performance is achieved when the IoU distribution is uniform.

## 6.3 Object Location Imbalance

Definition. The distribution of the objects throughout the image matters because current deep object detectors employ densely sampled anchors as sliding window classifiers. For most of the methods, the anchors are evenly distributed within the image, so that each part in the image is considered with the same importance level. On the other hand, the objects in an image do not follow a uniform distribution (see Figure 13), i.e. there is an imbalance about object locations. Solutions. Motivated by the fact that the objects are not distributed uniformly over the image, Wang et al. [67] aim to learn the location, scale and aspect ratio attributes of the anchors concurrently in order to decrease the number of anchors and improve recall at the same time. Specifically, given the backbone feature maps, a prediction branch is designed for each of these tasks to generate anchors: (i) anchor location prediction branch predicts a probability for each location to determine whether the location contains an object, and a hard thresholding approach is adopted based on the output probabilities to determine the anchors, (ii) anchor shape prediction branch generates the shape of the anchor for each location. Since the anchors vary depending on the image, different from the conventional methods (i.e. one-stage generators and RPN) using a fully convolutional classifier over the feature map, the authors proposed anchor-guided feature adaptation based on deformable convolutions [131] in order to have a balanced representation depending on the anchor size. Rather than learning the anchors, free anchor method [68] loosens the hard constraint of the matching strategy (i.e. a sample being positive if IoU &gt; 0 . 5 ) and in such a way, each anchor is considered a matching candidate for each ground truth. In order to do that, the authors pick a bag of candidate anchors for each ground truth using the sorted IoUs of the anchors with the ground truth. Among this bag of candidates, the proposed loss function aims to enforce the most suitable anchor by considering both the regression and the classification task to be matched with the ground truth.

<!-- image -->

## 6.4 Comparative Summary

Addressing spatial imbalance problems has resulted in significant improvements for object detectors. In general, while the methods for imbalance in regression loss and IoU distribution imbalance yield improvement especially in the regressor branch, removing the bias in the anchors in the location imbalance improves classification, too.

Despite the widespread use of the Smooth L 1 loss, four new loss functions have been proposed last year. Chen et al. [132] compared Balanced L 1 , IoU, GIoU, Bounded IoU on Faster R-CNN+FPN against Smooth L 1 , and reported relative improvement of 0 . 8% , 1 . 1% , 1 . 4 and - 0 . 3% respectively. With the same configuration, DIoU and CIoU losses are reported to have a relative improvement of 0 . 2% and 1 . 7% against the GIoU baseline (without DIoU-NMS). Compared to the loss functions designed for the foregroundbackground imbalance problem (i.e. Focal Loss - see Section 4.1.2), the relative improvement of the regression losses over Smooth L 1 are smaller. We also analyzed for which cases the baseline loss function fails in Figure 12(b), which can be used as a tool to compare the pros/cons of different regression loss functions.

Cascaded structures are proven to be very useful for object detection by regulating the IoU distribution. Cascade R-CNN, being a two-stage detector, has a relative improvement of 18 . 2% on MS-COCO testdev compared to its baseline Faster R-CNN+FPN with ResNet-101 backbone ( 36 . 2 vs. 42 . 8 ). A one-stage architecture, HSD, also has a relative improvement of 34 . 7% compared to the SSD512 with VGG16 backbone ( 28 . 8 vs 38 . 8 ). It is difficult to compare Cascade R-CNN and HSD since they are different in nature (onestage and two-stage pipelines), and their performances were reported for different input resolutions. However, we observed that HSD with a 768 × 768 input image runs approximately 1 . 5 × faster than Cascade R-CNN for a 1333 × 800 image and achieves slightly lower performance on MSCOCO testdev ( 42 . 3 vs. 42 . 8 ). One observation between these two is that, even though HSD performs worse than Cascade R-CNN at AP @0 . 5 , it yields better performance for AP @0 . 75 than Cascade R-CNN, which implies that the regressor of HSD is better trained.

As for the methods that we discussed for location imbalance, guided anchoring [67] increases average recall by 9 . 1% with 90% fewer anchors via learning the parameters of the anchors. Compared to guided anchoring, free anchor [68] reports a lower relative improvement for average recall with 2 . 4% against RetinaNet with ResNet50, however it has a 8 . 4% relative improvement (from 35 . 7 to 38 . 7 ).

## 6.5 Open Issues

This section discusses the open issues related to the spatial properties of the input bounding boxes and objects.

## 6.5.1 A Regression Loss with Many Aspects

Open Issue. Recent studies have proposed alternative regression loss definitions with different perspectives and aspects. Owing to their benefits, a single regression loss function that can combine these different aspects can be beneficial.

Explanation. Recent regression loss functions have different motivations: (i) Balanced L 1 Loss [29] increases the contribution of the inliers. (ii) KL Loss [53] is motivated from the ambiguity of the positive samples. (iii) IoU-based loss functions have the motive to use a performance metric as a loss function. These seemingly mutually exclusive motives can be integrated to a single regression loss function.

## 6.5.2 Analyzing the Loss Functions

In order to analyze how outliers and inliers affect the regression loss, it is useful to analyze the loss function and its gradient with respect to the inputs. To illustrate such an analysis, in Focal Loss [22], the authors plot the loss function with respect to the confidence score of the ground truth class with a comparison to the cross entropy loss, the baseline. Similarly, in Balanced L1 Loss [29], both the loss function itself and the gradients are depicted with a comparison to Smooth L1 Loss. Such an analysis might be more difficult for the recently proposed more complex loss functions. As an example, AP Loss [61] is computed considering the ranking of the individual examples, which is based on the confidence scores of all BBs. So, the loss depends on the entire set rather than individual examples, which makes it difficult to plot the loss (and its gradient) for a single input as conventionally done. Another example is GIoU Loss [85], which uses the ground truth box and the smallest enclosing box in addition to the detection box. Each box is represented by four parameters (see Section 6.1), which creates a total of twelve parameters. For this reason, it is necessary to develop appropriate analysis methods to observe how these loss functions penalize the examples.

Fig. 14: The (relative) spatial distributions of 1 K RoIs with IoU between 0 . 5 to 0 . 6 from the RPN (of Faster R-CNN with ResNet101 backbone) collected from Pascal [51] during the last epoch of the training. A bias is observed around the top-left corners of ground truths such that RoIs are densely concentrated at the top-left corner of the normalized ground truth box.

<!-- image -->

## 6.5.3 Designing Better Anchors

Designing an optimal anchor set with high recall has received limited attention. The Meta Anchor [133] method attempts to find an optimal set of aspect ratios and scales for anchors. More recently, Wang et al. [67] have improved recall more than 9% on MS COCO dataset [90] while using 90% less anchors than RPN [21]. Addressing the imbalanced nature of the locations and scales of the objects seems to be an open issue.

## 6.5.4 Relative Spatial Distribution Imbalance

Open Issue. As we discussed in Section 6, the distribution of the IoU between the estimated BBs and the ground truth is imbalanced and this has an influence on the performance. A closer inspection [66] reveals that the locations of estimated BBs relative to the matching ground truths also have an imbalance. Whether this imbalance affects the performance of the object detectors remains to be investigated.

Explanation. During the conventional training of the object detectors, input bounding boxes are labeled as positive

23

Fig. 15: An illustration of the imbalance in overlapping BBs. The grid represents the pixels of the image/feature map, and blue bounding box is the ground-truth. (a) Four negative input bounding boxes. (b) Two positive input bounding boxes. (c) Per-pixel number-of-overlaps of the input bounding boxes. Throughout the image, the number of sampling frequencies for the pixels vary due to the variation in the overlapping number of bounding boxes.

<!-- image -->

when their IoU with a ground truth is larger than 0 . 5 . This is adopted in order to provide more diverse examples to the classifier and the regressor, and to allow good quality predictions at test time from noisy input bounding boxes. The work by Oksuz et al. [66] is currently the only study that points to an imbalance in the distribution of the relative location of BBs. Exploiting the scale-invariance and shiftinvariance properties of the IoU, they plotted the top-left point of the RoIs from the RPN (of Faster R-CNN) with respect to a single reference box representing the ground truth (see Figure 14). They reported that the resulting spatial distribution of the top-left points of the RPN RoIs are skewed towards the top-left point of the reference ground truth box. We see that the samples are scarce away from the top-left corner of the reference box.

## 6.5.5 Imbalance in Overlapping BBs

Open Issue. Due to the dynamic nature of bounding box sampling methods (Section 4.1), some regions in the input image may be over-sampled (i.e. regions coinciding with many overlapping boxes) and some regions may be undersampled (or not even sampled at all). The effect of this imbalance caused by BB sampling methods has not been explored.

Explanation. Imbalance in overlapping BBs is illustrated in Figure 15(a-c) on an example grid representing the image and six input BBs (four negative and two positive). The number of overlapping BBs for each pixel is shown in Figure 15(c); in this example, this number ranges from 0 to 5 .

This imbalance may affect the performance for two reasons: (i) The number of highly sampled regions will play more role in the final loss functions, which can lead the method to overfit for specific features. (ii) The fact that some regions are over-sampled and some are undersampled might have adverse effects on learning, as the size of sample (i.e. batch size) is known to be related to the optimal learning rate [134].

## 6.5.6 Analysis of the Orientation Imbalance

Open Issue. The effects of imbalance in the orientation distribution of objects need to be investigated.

Explanation. The distribution of the orientation of object instances might have an effect on the final performance. If there is a typical orientation for the object, then the detector will likely overfit to this orientation and will make errors for the other orientations. To the best of our knowledge, this problem has not yet been explored.

Fig. 16: (a) Randomly sampled 32 positive RoIs using the pRoI Generator [66]. (b) Average classification and regression losses of these RoIs at the initialization of the object detector for MS COCO dataset [90] with 80 classes. We use cross entropy for the classification task assuming that initially each class has the same confidence score, and smooth L1 loss for regression task. Note that right after initialization, the classification loss has more effect on the total loss.

<!-- image -->

## 7 IMBALANCE 4: OBJECTIVE IMBALANCE

Definition. Objective imbalance pertains to the objective (loss) function that is minimized during training. By definition, object detection requires a multi-task loss in order to solve classification and regression tasks simultaneously. However, different tasks can lead to imbalance because of the following differences: (i) The norms of the gradients can be different for the tasks, and one task can dominate the training (see Figure 16). (ii) The ranges of the loss functions from different tasks can be different, which hampers the consistent and balanced optimization of the tasks. (iii) The difficulties of the tasks can be different, which affects the pace at which the tasks are learned, and hence hinders the training process [135].

Figure 16 illustrates a case where the loss of the classification dominates the overall gradient.

Solutions. The most common solution is Task Weighting which balances the loss terms by an additional hyperparameter as the weighting factor. The hyper-parameter is selected using a validation set. Naturally, increasing the number of tasks, as in the case of two-stage detectors, will increase the number of weighting factors and the dimensions of the search space (note that there are four tasks in two-stage detectors and two tasks in one-stage detectors).

An issue arising from the multi-task nature is the possible range inconsistencies among different loss functions. For example, in AP Loss , smooth L1, which is in the logarithmic range (since the input to the loss is conventionally provided after applying a logarithmic transformation) with [0 , ∞ ) , is used for regression while L AP ∈ [0 , 1] . Another example is the GIoU Loss [85], which is in the [ - 1 , 1] range and used together with cross entropy loss. The authors set the weighting factor of GIoU Loss to 10 and regularization is exploited to balance this range difference and ensure balanced training.

Since it is more challenging to balance terms with different ranges, it is a better strategy to first make the ranges comparable.

A more prominent approach to combine classification and regression tasks is Classification-Aware Regression Loss (CARL) [30], which assumes that classification and regression tasks are correlated. To combine the loss terms, the regression loss is scaled by a coefficient determined by the (classification) confidence score of the bounding box:

<!-- formula-not-decoded -->

where c ′ i is a factor based on p i , i.e., an estimation from the classification task. In this way, regression loss provides gradient signals to the classification branch as well, and therefore, this formulation improves localization of highquality (prime) examples.

An important contribution of CARL is to employ the correlation between the classification and regression tasks. However, as we discuss in Section 7.2, this correlation should be investigated and exploited more extensively.

Recently, Chen et al. [54] showed that cumulative loss originating from cross entropy needs to be dynamically weighted since, when cross entropy loss is used, the contribution rate of individual loss component at each epoch can be different. To prevent this imbalance, the authors proposed Guided Loss which simply weights the classification component by considering the total magnitude of the losses as:

<!-- formula-not-decoded -->

The motivation of the method is that regression loss consists of only foreground examples and is normalized only by number of foreground classes, therefore it can be used as a normalizer for the classification loss.

## 7.1 Comparative Summary

Currently, except for linear task weighting, there is no method suitable for all architectures alleviating objective imbalance, and unless the weights of the tasks are set accordingly, training diverges [54]. While many studies use equal weights for the regression and classification tasks [19], [22], it is also shown that appropriate linear weighting can lead to small improvements on the performance [132]. However, an in-depth analysis on objective imbalance is missing in the literature.

CARL [30] is a method to promote examples with higher IoUs, and in such a way, 1 . 6% relative improvement is achieved compared to prime sampling without CARL (i.e. from 37 . 9 to 38 . 5 ). Another method, Guided Loss [54], weights the classification loss dynamically to ensure balanced training. This method achieves a similar performance with the baseline RetinaNet, without using Focal Loss. However, their method does not discard the linear weighting, and they search for the optimal weight for different architectures. Moreover, the effect of Guided Loss is not clear for the two-stage object detectors, which conventionally employ cross entropy loss.

Fig. 17: Paces of the tasks in RPN (with Resnet-101 [93] backbone) [21] on Pascal VOC 2007 training set [51].

<!-- image -->

## 7.2 Open Issues

Open Issue. Currently, the most common approach is to linearly combine the loss functions of different tasks to obtain the overall loss function (except for classificationaware regression loss in [30]). However, as shown in Figure 18(a-c), as the input bounding box is slid over the image, both classification and regression losses are affected, implying their dependence. This suggests that current linear weighting strategy may not be able to address the imbalance of the tasks that is related to (i) the loss values and their gradients, and (ii) the paces of the tasks.

Explanation. The loss function of one task (i.e. classification) can affect the other task (i.e. regression). To illustrate, AP Loss [61] did not modify the regression branch; however, COCO style AP @0 . 75 increased around 3% . This example shows that the loss functions for different branches (tasks) are not independent (see also Figure 18). This interdependence of tasks has been explored in classification-aware regression loss by Cao et al. [30] (as discussed in Section 7) to a certain extent. Further research is needed for a more detailed analysis of this interdependence and fully exploiting it for object detection.

Some studies in multi-task learning [135] pointed out that learning pace affects performance. With this in mind, we plotted the regression and classification losses of the RPN [21] during training on the Pascal VOC dataset [51] in Figure 17. We observe that the classification loss decreases faster than the regression loss. Analyzing and balancing the learning paces of different tasks involved in the object detection problem might be another fruitful research direction.

## 8 IMBALANCE PROBLEMS IN OTHER DOMAINS

In this section, we cover imbalance problems in other related domains in order to motivate the adaptation of methods from related domains to the object detection problem. We identify two problems that are closely related to object detection: image classification and metric learning, discussed in individual subsections. In addition, the methods pertaining to the other domains are discussed in the last subsection.

## 8.1 Image Classification

Image classification is the problem of assigning a category label for an image. This problem is closely related to the object detection problem since it is one of two tasks in object detection. For image classification problem, class imbalance has been extensively studied from very different perspectives (compared to other imbalance problems), and in this section, we will focus only on class imbalance.

A common approach is Resampling the dataset [136], [137], [138], [139], [140], including oversampling and undersampling to balance the dataset. While oversampling adds more samples from the under-represented classes, undersampling balances over the classes by ignoring some of the data from the over-represented classes. When employed naively, oversampling may suffer from overfitting because duplication of samples from the under-represented classes can introduce bias. Therefore, despite it means ignoring a portion of the training data, undersampling was preferable for non-deep-learning approaches [137]. On the other hand, Buda et al. [139] showed that deep neural networks do not suffer from overfitting under oversampling and in fact, better performance can be achieved compared to undersampling.

Over the years, methods more complicated than just duplicating examples have been developed. For example, Chawla et al. [136] proposed Smote as a new way of oversampling by producing new samples as an interpolation of neighboring samples. Adasyn [138], an extension of Smote to generate harder examples, aims to synthesize samples from the underrepresented classes. Li et al. [140] sample a mini-batch as uniform as possible from all classes also by restricting the same example and class to appear in the same order, which implies promoting the minority classes.

Another approach addressing class imbalance in image classification is transfer learning [141], [142]. For example, Wang et al. [141] design a model (i.e. meta-learner) to learn how a model evolves when the size of the training set increases. The meta-learner model is trained gradually by increasing the provided number of examples from the classes with a large number of examples. The resulting meta-model is able to transform another model trained with less examples to a model trained with more examples, which makes it useful to be exploited by an underrepresented class. Another study [142] adopted a different strategy during transfer learning: Firstly, the network is trained with the entire imbalanced dataset, and then the resulting network is fine-tuned by using a balanced subset of the dataset.

Weighting the loss function [143], [144] is yet another way of balancing the classes (we called it soft sampling in the paper). Among these approaches, Huang et al. [143] use the inverse class frequency (i.e. 1 / | C i | ) to give more weight to under-represented classes. This is extended by class-balanced loss [144] to adopt inverse class frequency by employing a hyperparameter β as w i = (1 - β ) / (1 - β | C i | ) . When β = 0 , the weight degenerates to 1 ; and β → 1 makes the weight approximate 1 / | C i | , the inverse class frequency.

Similar to object detection, giving more importance to 'useful' examples is also common [145], [146], [147], [148]. As done in object detection, hard examples have been utilized, e.g. by Dong et al. [147] who identify hard (positive and negative) samples in the batch level. The proposed method uses the hardness in two levels: (i) Class-level hard samples are identified based on the predicted confidence for the ground-truth class. (ii) Instance-level hard examples are the ones with larger L 2 distance to a representative example in the feature space. An interesting approach that has not been utilized in object detection yet is to focus training on examples on which the classifier is uncertain (based on its prediction history) [148]. Another approach is to generate hard examples by randomly cropping parts of an image as in Hide and Seek [146], or adopting a curricular learning approach by first training the classifier on easy examples and then on hard examples [145].

Another related set of methods addresses image classification problem from the data redundancy perspective [149], [150], [151], [152], [153]. Birodkar et al. [153] showed that around 10% of the data in ImageNet [52] and CIFAR-10 datasets is redundant during training. This can be exploited not only for balancing the data but also to obtain faster convergence by ignoring useless data. The methods differ from each other on how they mine for the redundant examples. Regardless of whether imbalance problem is targeted or not, a subset of these methods uses the active learning paradigm, where an oracle is used to find out the best set of training examples. The core set approach [151] uses the relative distances of the examples in the feature space to determine redundant samples whereas Vodrahalli et al. [152] determine redundancies by looking at the magnitude of the gradients.

Another mechanism to enrich a dataset is to use weak supervision for incorporating unlabelled examples. An example study is by Mahajan et al. [154], who augment the dataset by systematically including Instagram images with the hashtags as the labels, which are rather noisy. In another example, Liu et al. [155] selectively add unlabeled data to the training samples after labeling these examples using the classifier.

Generative models (e.g. GANs) can also be used for extending the dataset to address imbalance. Many studies [156], [157], [158] have successfuly used GANs to generate examples for under-represented classes for various image classification problems.

Special cases of image classification (e.g. face recognition) are also affected by imbalance [159], [160], [161]. The general approaches are similar and therefore, due to the space constraint, we omit imbalance in specialized classification problems.

Comparative Summary. Our analysis reveals that object detection community can benefit from the imbalance studies in image classification in many different aspects. The discussed methods for image classification are (by definition) the present solutions for the foreground-foreground class imbalance problem (see Section 4.2); however, they can possibly be extended to the foreground-background class imbalance problem. Foreground-background class imbalance is generally handled by under-sampling for the object detection problem, and other advanced resampling or transfer learning methods are not adopted yet for object detection from a class imbalance perspective. While there are loss functions (discussed in Section 4.1.2) that exploit a weighting scheme [22], [59], Cui et al. [144] showed that class-balanced loss is complementary to the focal loss [22] in that focal loss aims to focus on hard examples while class-balanced loss implies balancing over all classes. However, since the number of background examples is not defined, the current definition does not fit into the object detection context. Similarly, the adoption of weakly supervised methods to balance underrepresented classes or data redundancy by also decreasing the samples from an over-represented class can be used to alleviate the class imbalance problem. Finally, there are only a few generative approaches in object detection, much less than those proposed for addressing imbalance in image classification.

## 8.2 Metric Learning

Metric learning methods aim to find an embedding of the inputs, where the distance between the similar examples is smaller than the distance between dissimilar examples. In order to model such similarities, the methods generally employ pairs [162], [163] or triplets [164], [165], [166] during training. In the pair case, the loss function uses the information about whether both of the samples are from the same or different classes. In contrast, training using triplets require an anchor example, a positive example from the same class with the anchor and a negative example from a different class from the anchor. The triplet-wise training scheme introduces imbalance over positive and negative examples in favor of the latter one, and the methods also look for the hard examples as in object detection in both the pair-wise and triplet-wise training schemes. Accordingly, to present the imbalance and useful example mining requirement, note that there are approximately O ( n 2 ) pairs and O ( n 3 ) triplets assuming that the dataset size is n . This increase in the dataset size makes it impossible to mine for hard examples by processing the entire dataset to search for the most useful (i.e. hard) examples. For this reason, similar to object detection, a decision is to be made on which samples to be used during training.

The metric learning methods use sampling, generative methods or novel loss functions to alleviate the problem. Note that this is very similar to our categorization of foreground-background class imbalance methods in Section 4.1, which also drove us to examine metric learning in detail. Owing to its own training, and resulting loss function configuration, here we only examine sampling strategies and generative methods.

Sampling Methods aim to find a useful set of training examples from a large training dataset. The usefulness criterion is considered to be highly relevant to the hardness of an example. Unlike the methods in object detection, some of the methods avoid using the hardest possible set of examples during traning. One example proposed by Schroff et al. [164] use a rule based on Euclidean distance to define 'semihard' triplets since selecting the hardest triplets can end up in local minima during the early stages of the training. This way, they decrease the effect of confusing triplets and avoid repeating the same hardest examples. Another approach that avoids considering the hardest possible set of examples is Hard-Aware Deeply Cascaded Embedding [167], which proposes training a cascaded model such that the higher layers are trained with harder examples while the first layers are trained with the entire dataset. Similarly, Smart Mining [168] also mines semi-hard examples exploiting the distance between nearest neighbor of anchor and the corresponding anchor and one novelty is that they increase the hardness of the negative examples in the latter epochs adaptively.

Note that neither semi-hardness nor adaptive setting of the hardness level is considered by object detectors.

of the

As a different approach, Cui et al. [169] consider to exploit humans to label false positives during training, which are identified as hard examples and be added to the minibatch for the next iteration. Song et al. [170] mine hard negatives not only for the anchor but also for all positives. Note that no object detection method considers the relation between all positives and the negative example while assigning a hardness value to a negative example. One promising idea shown by Huang et al. [171] is that larger intra-class distances can confuse the hard example mining process while only inter-class distance is considered during mining. For this reason, a position-dependent deep

metric unit is proposed to take into account the intra-class variations.

Similar to the generative methods for object detection (Section 4.1.4), Generative Methods have been used for generating examples or features for metric learning as well. Deep Adversarial Metric Learning [172] simultaneously learns and embedding and a generator. Here, the generator outputs a synthetic hard negative example given the original triplet. Similar to Tripathi et al. [64], Zhao et al. [173] also use a GAN [174] in order to generate not only hard negatives but also hard positives. The idea to consider interclass similarity have proven well as in the work by Huang et al. [171]. Finally, a different approach from the previous generative models, Hardness Aware Metric Learning [175], aims to learn an autoencoder in the feature space. The idea is as follows: The authors first manipulate the features after the backbone such that the hardness of the example can be controlled by linearly interpolating the embedding towards the anchor by employing a coefficient relying on the loss values at the last epoch. Since it is not certain that the interpolated embedding preserves the original label, a label preserving mapping back to the feature space is employed using the autoencoder. Also, similar to Harwood et al. [168], the hardness of the examples in the latter epochs is increased.

Comparative Summary. Looking at the studies presented above, we observe that the metric learning methods are able to learn an embedding of the data that preserves the desired similarity between data samples. Object detection literature have used different measures and metrics that have been designed by humans. However, as shown by the metric learning community, a metric that is directly learned from the data itself can yield better results and have interesting properties. Moreover, the self-paced learning, where the hardness levels of the examples is increased adaptively, is definitely an important concept for addressing imbalance in object detection. Another idea that can be adopted by the object detectors is to label the examples by humans in an online manner (similar to the work by Yao [176]) during training and to use the semi-hardness concept.

## 8.3 Multi-Task Learning

Multi-task learning involves learning multiple tasks (with potentially conflicting objectives) simultaneously. A common approach is to weigh the objectives of the tasks to balance them.

27

Fig. 18: An example suggesting the necessity of considering different imbalance problems together in a unified manner. Blue, green and red colors indicate ground-truth, positive example (prediction) and negative examples, respectively. The larger example (i.e. the one with higher IoU with the blue box) in (a) is shifted right. In (b), its IoU is significantly decreased and it eventually becomes a negative example in (c). This example shows the interplay between class imbalance, scale imbalance, spatial imbalance and objective imbalance by a change in BB positions.

<!-- image -->

Many methods have been proposed for assigning the weights in a more systematic manner. For example, Li et al. [177] extended the self-paced learning paradigm to multitask learning based on a novel regularizer. The hyperparameters in the proposed regularizer control the hardness of not only the instances but also the tasks, and accordingly, the hardness level is increased during training. In another work motivated by the self-paced learning approach, Guo et al. [135] use more diverse set of tasks, including object detection. Their method weighs the losses dynamically based on the exponential moving average of a predefined key performance indicator (e.g. accuracy, average precision) for each task. Similar to image classification, one can also use the uncertainty of the estimations [178] or their loss values [179] to assign weights to the tasks.

In addition to the importance or hardness of tasks, Zhao Chen and Rabinovich [180] identified the importance of the pace at which the tasks are learned. They suggested that the tasks are required to be trained in a similar pace. For this end, they proposed balancing the training pace of different tasks by adjusting their weights dynamically based on a normalization algorithm motivated by batch normalization [117].

Comparative Summary. Being a multi-task problem, object detection can benefit significantly from the multi-task learning approaches. However, this aspect of object detectors has not received attention from the community.

## 9 OPEN ISSUES FOR ALL IMBALANCE PROBLEMS

In this section, we identify and discuss the issues relevant to all imbalance problems. For open issues pertaining to a specific imbalance problem, please see its corresponding section in the text.

## 9.1 A Unified Approach to Addressing Imbalance

Open Issue. One of the main challenges is to come up with a unified approach that addresses all imbalance problems by considering the inter-dependence between them.

Explanation. We illustrate this inter-dependency using a toy example in Figure 18. In this figure, we shift an input bounding box with a high IoU (see Figure 18(a)) to worse qualities in terms of IoU in two steps (see Figure 18(b-c)) and observe how this shift affects the different imbalance problems. For the base case in Figure 18(a), there are two positive bounding boxes (relevant for class imbalance) with different scales (relevant for scale imbalance), loss values (relevant for objective imbalance) and IoUs (relevant for BB imbalance). Shifting the box to the right, we observe the following:

- In Figure 18(b), we still have two positives, both of which now have less IoU with the ground truth (compared to (a)).
- This leads to the following: (i) There are more hard examples (considering hard example mining [24], [29]), and less prime samples [30]. For this reason, the methods for class imbalance are affected. (ii) The scales of the RoIs and ground truth do not change. Considering this, the scale imbalance seems not affected. (iii) Objective imbalance is affected in two ways: Firstly, the shifted BB will incur more loss (for regression, and possibly for classification) and thus, become more dominant in its own task. Secondly, since the cumulative individual loss values change, the contribution to the total loss of the individual loss values will also change, which implies its effect on objective imbalance. (iv) Finally, both BB IoU distribution and spatial distribution of the positive examples will be affected by this shift.
- In Figure 18(c), by applying a small shift to the same BB, its label changes.

This leads to the following: (i) There are less positive examples and more negative examples. The ground truth class loses an example. Note that this example evolves from being a hard positive to a hard negative in terms hard example mining [24], [29], and the criterion that the prime sample attention [30] considers for the example changed from IoU to classification score. For this reason, in this case, the methods involving class imbalance and foreground-foreground class imbalance are affected. (ii) A positive RoI is removed from the set of RoIs with similar scales. Therefore, there will be less positive examples with the same scale, which affects scale imbalance. (iii) Objective imbalance is affected in two ways: Firstly, the now-negative BB is an additional hard example in terms of classification possibly with a larger loss value. Secondly, the shifted example is totally free from the regression branch, and moved to the set of hard examples in terms of classification. Hence, the contribution of the regression loss to the total loss is expected to decrease, and the contribution of classification would increase. (iv) Finally, the IoU distribution of the positive examples will be affected by this shift since a positive example is lost.

Therefore, the aforementioned imbalance problems have an intertwined nature, which needs to be investigated and identified in detail to effectively address all imbalance problems.

## 9.2 Measuring and Identifying Imbalance

Open Issue. Another critical issue that has not been addressed yet is how to quantify or measure imbalance, and how to identify imbalance when there is one. We identify three questions that need to be studied:

- 1) What is a balanced distribution for a property that is critical for a task? This is likely to be uniform distribution for many properties like, e.g. class distribution. However, different modalities may imply a different concept of balance. For example, OHEM prefers a skewed distribution around 0 . 5 for the IoU distribution; left-skewed for the positives and rightskewed for the negatives.
- 2) What is the desired distribution for the properties that are critical for a task? Note that the desired distribution may be different from the balanced distribution since skewing the distribution in one way may be beneficial for faster convergence and better generalization. For example, online hard negative mining [24] favors a right-skewed IoU distribution towards 0.5 [29], whereas prime sample attention prefers the positive examples with larger IoUs [30] and the class imbalance methods aim to ensure a uniform distribution from the classes.
- 3) How can we quantify how imbalanced a distribution is? A straightforward approach is to consider optimal transport measures such as the Wasserstein distance; however, such methods would neglect the effect of a unit change (imbalance) in the distribution on the overall performance, thereby jeopardizing a direct and effective consideration (and comparison) of the imbalance problems using the imbalance measurements.

## 9.3 Labeling a Bounding Box as Positive or Negative

Open Issue. Currently, object detectors use IoU-based thresholding (possibly with different values) for labeling an example as positive or negative and there is no consensus on this (i.e. Fast R-CNN [17], Retina Net [22] and RPN [21] label input BBs with IoUs 0 . 5 , 0 . 4 and 0 . 3 as negatives respectively.). However, a consensus on this is critical since labeling is very relevant to determining whether an example is a hard example.

Explanation. Labeling bounding boxes is highly relevant to imbalance problems since this is the step where the set of all bounding boxes are split as positives and negatives in an online manner. Of special concern are the bounding boxes around the decision boundary, which are typically considered as hard examples, and a noisy labeling over them would result in large gradients in opposite directions. In other words, in order to define the hard negatives reliably, the number of outliers should be as small as possible. For this reason, consistent labeling of the input bounding boxes as positive or negative is a prerequisite of the imbalance problems in object detection.

Currently the methods use a hard IoU threshold (generally 0 . 5 ) to split the examples; however, Li et al. [59] showed that this scheme results in a large number of noisy examples. In Figure 19, we illustrate two input bounding boxes that can be misleadingly labeled as positive; and once they are labeled as positive, it is likely that they will be sampled as hard positives:

Fig. 19: An illustration on ambiguities resulting from labeling examples. The blue boxes denote the ground truth. The green boxes are the estimated positive boxes with IoU &gt; 0 . 5 . The input image is from the MS COCO [90].

<!-- image -->

- The estimated (positive) box for the bicycle (green) has two problems: It has occlusion (for the bicycle), and a big portion of it includes another object (a person). For this reason, during training, this is not only a hard example for the bicycle class but also a misleading example for the person class in that this specific example will try to suppress the probability of this box to be classified as person.
- The estimated (positive) box for the person class (green) consists of black pixels in most of it. In other words, the box does hardly include any visually descriptive part for a person. For this reason, this is a very hard example which is likely to fail in capturing the ground truth class well.

## 9.4 Imbalance in Bottom-Up Object Detectors

Open Issue. Bottom-up detectors [23], [48], [49] adopt a completely different approach to object detection than the one-stage and the two-stage detectors (see Section 2.1). Bottom-up detectors might share many of the imbalance problems seen in the top-down detectors, and they may have their own imbalance issues as well. Further research needs to be conducted for (i) analyzing the known methods addressing imbalance problems in the context of bottomup object detectors, and (ii) imbalance problems that are specific to bottom-up detectors.

Explanation. Addressing imbalance issues in bottom-up object detectors has received limited attention. CornerNet [23] and ExtremeNet [49] use focal loss [22] to address foreground-background class imbalance, and the hourglass network [104] to compensate for the scale imbalance. On the other hand, use of hard sampling methods and the effects of other imbalance problems have not been investigated. For the top-down detectors, we can recap some of the findings: from the class imbalance perspective, Shrivastava et al. [24] show that the examples with larger losses are important; from the scale imbalance perspective, different architectures [29], [75], [81] and training methods [27], [28] involving feature and image pyramids are proven to be useful and finally from the objective imbalance perspective, Pang et al. [29] showed that smooth L 1 loss underestimates the effect of the inliers. Research is needed to come up with such findings for bottom-up object detectors.

## 10 CONCLUSION

In this paper, we provided a thorough review of the imbalance problems in object detection. In order to provide a more complete and coherent perspective, we introduced a taxonomy of the problems as well as the solutions for addressing them. Following the taxonomy on problems, we discussed each problem separately in detail and presented the solutions with a unifying yet critical perspective.

In addition to the detailed discussions on the studied problems and the solutions, we pinpointed and presented many open issues and imbalance problems that are critical for object detection. In addition to the many open aspects that need further attention for the studied imbalance problems, we identified new imbalance issues that have not been addressed or discussed before.

With this review and our taxonomy functioning as a map, we, as the community, can identify where we are and the research directions to be followed to develop better solutions to the imbalance problems in object detection.

## ACKNOWLEDGMENTS

This work was partially supported by the Scientific and Technological Research Council of Turkey (TÜB ˙ ITAK) through the project titled 'Object Detection in Videos with Deep Neural Networks' (grant number 117E054). Kemal Öksüz is supported by the TÜB ˙ ITAK 2211-A National Scholarship Programme for Ph.D. students.

## REFERENCES

- [1] Q. Fan, L. Brown, and J. Smith, 'A closer look at faster r-cnn for vehicle detection,' in IEEE Intelligent Vehicles Symposium , 2016.
- [2] Z. Fu, Y. Chen, H. Yong, R. Jiang, L. Zhang, and X. Hua, 'Foreground gating and background refining network for surveillance object detection,' IEEE Transactions on Image Processing-Accepted , 2019.
- [3] A. Geiger, P. Lenz, and R. Urtasun, 'Are we ready for autonomous driving? the kitti vision benchmark suite,' in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2012.
- [4] X. Dai, 'Hybridnet: A fast vehicle detection system for autonomous driving,' Signal Processing: Image Communication , vol. 70, pp. 79 - 88, 2019.
- [5] P. F. Jaeger, S. A. A. Kohl, S. Bickelhaupt, F. Isensee, T. A. Kuder, H. Schlemmer, and K. H. Maier-Hein, 'Retina u-net: Embarrassingly simple exploitation of segmentation supervision for medical object detection,' arXiv , vol. 1811.08661, 2018.
- [6] S.-g. Lee, J. S. Bae, H. Kim, J. H. Kim, and S. Yoon, 'Liver lesion detection from weakly-labeled multi-phase ct volumes with a grouped single shot multibox detector,' in Medical Image Computing and Computer Assisted Intervention (MICCAI) , A. F. Frangi, J. A. Schnabel, C. Davatzikos, C. Alberola-López, and G. Fichtinger, Eds., 2018.
- [7] M. Rad and V. Lepetit, 'Bb8: A scalable, accurate, robust to partial occlusion method for predicting the 3d poses of challenging objects without using depth,' in The IEEE International Conference on Computer Vision (ICCV) , 2017.
- [8] W. Kehl, F. Manhardt, F. Tombari, S. Ilic, and N. Navab, 'Ssd6d: Making rgb-based 3d detection and 6d pose estimation great again,' in The IEEE International Conference on Computer Vision (ICCV) , 2017.

- [9] B. Tekin, S. N. Sinha, and P. Fua, 'Real-Time Seamless Single Shot 6D Object Pose Prediction,' in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2018.
- [10] T. Hodan, F. Michel, E. Brachmann, W. Kehl, A. GlentBuch, D. Kraft, B. Drost, J. Vidal, S. Ihrke, X. Zabulis, C. Sahin, F. Manhardt, F. Tombari, T.-K. Kim, J. Matas, and C. Rother, 'Bop: Benchmark for 6d object pose estimation,' in The European Conference on Computer Vision (ECCV) , 2018.
- [11] G. Du, K. Wang, and S. Lian, 'Vision-based robotic grasping from object localization, pose estimation, grasp detection to motion planning: A review,' arXiv , vol. 1905.06658, 2019.
- [12] I. Bozcan and S. Kalkan, 'Cosmo: Contextualized scene modeling with boltzmann machines,' Robotics and Autonomous Systems , vol. 113, pp. 132-148, 2019.
- [13] P. F. Felzenszwalb, R. B. Girshick, D. McAllester, and D. Ramanan, 'Object detection with discriminatively trained partbased models,' IEEE Transactions on Pattern Analysis and Machine Intelligence , vol. 32, no. 9, pp. 1627-1645, 2010.
- [14] A. Krizhevsky, I. Sutskever, and G. E. Hinton, 'Imagenet classification with deep convolutional neural networks,' in Advances in Neural Information Processing Systems (NIPS) , 2012.
- [15] J. Redmon and A. Farhadi, 'YOLO9000: Better, faster, stronger,' in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2017.
- [16] R. Girshick, J. Donahue, T. Darrell, and J. Malik, 'Rich feature hierarchies for accurate object detection and semantic segmentation,' in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2014.
- [17] R. Girshick, 'Fast R-CNN,' in The IEEE International Conference on Computer Vision (ICCV) , 2015.
- [18] J. Dai, Y. Li, K. He, and J. Sun, 'R-FCN: Object detection via region-based fully convolutional networks,' in Advances in Neural Information Processing Systems (NIPS) , 2016.
- [19] W. Liu, D. Anguelov, D. Erhan, C. Szegedy, S. E. Reed, C. Fu, and A. C. Berg, 'SSD: single shot multibox detector,' in The European Conference on Computer Vision (ECCV) , 2016.
- [20] J. Redmon, S. K. Divvala, R. B. Girshick, and A. Farhadi, 'You only look once: Unified, real-time object detection,' in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2016.
- [21] S. Ren, K. He, R. Girshick, and J. Sun, 'Faster R-CNN: Towards real-time object detection with region proposal networks,' IEEE Transactions on Pattern Analysis and Machine Intelligence , vol. 39, no. 6, pp. 1137-1149, 2017.
- [22] T. Lin, P. Goyal, R. B. Girshick, K. He, and P. Dollár, 'Focal loss for dense object detection,' in The IEEE International Conference on Computer Vision (ICCV) , 2017.
- [23] H. Law and J. Deng, 'Cornernet: Detecting objects as paired keypoints,' in The European Conference on Computer Vision (ECCV) , 2018.
- [24] A. Shrivastava, A. Gupta, and R. Girshick, 'Training regionbased object detectors with online hard example mining,' in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2016.
- [25] W. Ouyang, X. Wang, C. Zhang, and X. Yang, 'Factors in finetuning deep model for object detection with long-tail distribution,' in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2016.
- [26] T. Lin, P. Dollár, R. B. Girshick, K. He, B. Hariharan, and S. J. Belongie, 'Feature pyramid networks for object detection,' in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2017.
- [27] B. Singh and L. S. Davis, 'An analysis of scale invariance in object detection - snip,' in The Conference on Computer Vision and Pattern Recognition (CVPR) , 2018.
- [28] B. Singh, M. Najibi, and L. S. Davis, 'Sniper: Efficient multi-scale training,' in Advances in Neural Information Processing Systems (NIPS) , 2018.
- [29] J. Pang, K. Chen, J. Shi, H. Feng, W. Ouyang, and D. Lin, 'Libra R-CNN: Towards balanced learning for object detection,' in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2019.
- [30] Y. Cao, K. Chen, C. C. Loy, and D. Lin, 'Prime Sample Attention in Object Detection,' arXiv , vol. 1904.04821, 2019.
- [31] L. Liu, W. Ouyang, X. Wang, P. W. Fieguth, J. Chen, X. Liu, and M. Pietikäinen, 'Deep learning for generic object detection: A survey,' arXiv , vol. 1809.02165, 2018.
- [32] Z. Zou, Z. Shi, Y. Guo, and J. Ye, 'Object detection in 20 years: A survey,' arXiv , vol. 1905.05055, 2018.
- [33] S. Agarwal, J. O. D. Terrail, and F. Jurie, 'Recent advances in object detection in the age of deep convolutional neural networks,' arXiv , vol. 1809.03193, 2018.
- [34] Zehang Sun, G. Bebis, and R. Miller, 'On-road vehicle detection: a review,' IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI) , vol. 28, no. 5, pp. 694-711, 2006.
- [35] P. Dollar, C. Wojek, B. Schiele, and P. Perona, 'Pedestrian detection: An evaluation of the state of the art,' IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI) , vol. 34, no. 4, pp. 743-761, 2012.
- [36] S. Zafeiriou, C. Zhang, and Z. Zhang, 'A survey on face detection in the wild,' Computer Vision and Image Understanding , vol. 138, pp. 1-24, 2015.
- [37] Q. Ye and D. Doermann, 'Text detection and recognition in imagery: A survey,' IEEE Transactions on Pattern Analysis and Machine Intelligence , vol. 37, no. 7, pp. 1480-1500, 2015.
- [38] X. Yin, Z. Zuo, S. Tian, and C. Liu, 'Text detection, tracking and recognition in video: A comprehensive survey,' IEEE Transactions on Image Processing , vol. 25, no. 6, pp. 2752-2773, 2016.
- [39] G. Litjens, T. Kooi, B. E. Bejnordi, A. A. A. Setio, F. Ciompi, M. Ghafoorian, J. A. van der Laak, B. van Ginneken, and C. I. Sánchez, 'A survey on deep learning in medical image analysis,' Medical Image Analysis , vol. 42, pp. 60 - 88, 2017.
- [40] B. Krawczyk, 'Learning from imbalanced data: open challenges and future directions,' Progress in Artificial Intelligence , vol. 5, no. 4, pp. 221-232, 2016.
- [41] J. L. Leevy, T. M. Khoshgoftaar, R. A. Bauder, and N. Seliya, 'A survey on addressing high-class imbalance in big data,' Journal of Big Data , vol. 5, no. 42, 2018.
- [42] A. Fernández, S. García, M. Galar, R. Prati, B. Krawczyk, and F. Herrera, Learning from Imbalanced Data Sets . Springer International Publishing, 2018.
- [43] J. M. Johnson, Khoshgoftaar, and T. M., 'Survey on deep learning with class imbalance,' Journal of Big Data , vol. 6, no. 21, 2019.
- [44] J. R. R. Uijlings, K. E. A. van de Sande, T. Gevers, and A. W. M. Smeulders, 'Selective search for object recognition,' International Journal of Computer Vision , vol. 104, no. 2, pp. 154-171, 2013.
- [45] C. L. Zitnick and P. Dollár, 'Edge boxes: Locating object proposals from edges,' in The European Conference on Computer Vision (ECCV) , 2014.
- [46] C.-Y. Fu, W. Liu, A. Ranga, A. Tyagi, and A. C. Berg, 'DSSD: Deconvolutional single shot detector,' arXiv , vol. 1701.06659, 2017.
- [47] J. Redmon and A. Farhadi, 'Yolov3: An incremental improvement,' arXiv , vol. 1804.02767, 2018.
- [48] K. Duan, S. Bai, L. Xie, H. Qi, Q. Huang, and Q. Tian, 'Centernet: Keypoint triplets for object detection,' in The IEEE International Conference on Computer Vision (ICCV) , 2019.
- [49] X. Zhou, J. Zhuo, and P. Krahenbuhl, 'Bottom-up object detection by grouping extreme and center points,' in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2019.
- [50] A. Newell, Z. Huang, and J. Deng, 'Associative embedding: Endto-end learning for joint detection and grouping,' in Advances in Neural Information Processing Systems (NIPS) , 2017.
- [51] M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn, and A. Zisserman, 'The pascal visual object classes (voc) challenge,' International Journal of Computer Vision (IJCV) , vol. 88, no. 2, pp. 303-338, 2010.
- [52] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, 'ImageNet: A Large-Scale Hierarchical Image Database,' in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2009.
- [53] Y. He, C. Zhu, J. Wang, M. Savvides, and X. Zhang, 'Bounding box regression with uncertainty for accurate object detection,' in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2019.
- [54] J. Chen, D. Liu, T. Xu, S. Zhang, S. Wu, B. Luo, X. Peng, and E. Chen, 'Is sampling heuristics necessary in training deep object detectors?' arXiv , vol. 1909.04868, 2019.
- [55] H. A. Rowley, S. Baluja, and T. Kanade, 'Human face detection in visual scenes,' in The Advances in Neural Information Processing Systems (NIPS) , 1995.
- [56] T. Kong, F. Sun, A. Yao, H. Liu, M. Lu, and Y. Chen, 'Ron: Reverse connection with objectness prior networks for object detection,'

- in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2017.
- [57] S. Zhang, L. Wen, X. Bian, Z. Lei, and S. Z. Li, 'Single-shot refinement neural network for object detection,' in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2018.
- [58] J. Nie, R. M. Anwer, H. Cholakkal, F. S. Khan, Y. Pang, and L. Shao, 'Enriched feature guided refinement network for object detection,' in The IEEE International Conference on Computer Vision (ICCV) , 2019.
- [59] B. Li, Y. Liu, and X. Wang, 'Gradient harmonized single-stage detector,' in AAAI Conference on Artificial Intelligence , 2019.
- [60] J. Chen, D. Liu, B. Luo, X. Peng, T. Xu, and E. Chen, 'Residual objectness for imbalance reduction,' arXiv , vol. 1908.09075, 2019.
- [61] K. Chen, J. Li, W. Lin, J. See, J. Wang, L. Duan, Z. Chen, C. He, and J. Zou, 'Towards accurate one-stage object detection with ap-loss,' in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2019.
- [62] Q. Qian, L. Chen, H. Li, and R. Jin, 'DR Loss: Improving Object Detection by Distributional Ranking,' arXiv , vol. 1907.10156, 2019.
- [63] X. Wang, A. Shrivastava, and A. Gupta, 'A-fast-rcnn: Hard positive generation via adversary for object detection,' in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2017.
- [64] S. Tripathi, S. Chandra, A. Agrawal, A. Tyagi, J. M. Rehg, and V. Chari, 'Learning to generate synthetic data via compositing,' in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2019.
- [65] H. Wang, Q. Wang, F. Yang, W. Zhang, and W. Zuo, 'Data augmentation for object detection via progressive and selective instance-switching,' arXiv , vol. 1906.00358, 2019.
- [66] K. Oksuz, B. C. Cam, S. Kalkan, and E. Akbas, 'Generating positive bounding boxes for balanced training of object detectors,' in IEEE Winter Applications on Computer Vision (WACV) , 2020.
- [67] J. Wang, K. Chen, S. Yang, C. C. Loy, and D. Lin, 'Region proposal by guided anchoring,' in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2019.
- [68] X. Zhang, F. Wan, C. Liu, R. Ji, and Q. Ye, 'Freeanchor: Learning to match anchors for visual object detection,' in Advances in Neural Information Processing Systems (NIPS) , 2019.
- [69] F. Yang, W. Choi, and Y. Lin, 'Exploit all the layers: Fast and accurate cnn object detector with scale dependent pooling and cascaded rejection classifiers,' in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2016.
- [70] Z. Cai, Q. Fan, R. Feris, and N. Vasconcelos, 'A unified multiscale deep convolutional neural network for fast object detection,' in The European Conference on Computer Vision (ECCV) , 2016.
- [71] J. Li, X. Liang, S. Shen, T. Xu, J. Feng, and S. Yan, 'Scale-aware fast r-cnn for pedestrian detection,' IEEE Transactions on Multimedia , vol. 20, no. 4, pp. 985-996, 2018.
- [72] Y. Pang, T. Wang, R. M. Anwer, F. S. Khan, and L. Shao, 'Efficient featurized image pyramid network for single shot detector,' in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2019.
- [73] J. Noh, W. Bae, W. Lee, J. Seo, and G. Kim, 'Better to follow, follow to be better: Towards precise supervision of feature superresolution for small object detection,' in The IEEE International Conference on Computer Vision (ICCV) , 2019.
- [74] Y. Li, Y. Chen, N. Wang, and Z. Zhang, 'Scale-aware trident networks for object detection,' in The IEEE International Conference on Computer Vision (ICCV) , 2019.
- [75] S. Liu, L. Qi, H. Qin, J. Shi, and J. Jia, 'Path aggregation network for instance segmentation,' in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2018.
- [76] P. Zhou, B. Ni, C. Geng, J. Hu, and Y. Xu, 'Scale-transferrable object detection,' in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2018.
- [77] S.-W. Kim, H.-K. Kook, J.-Y. Sun, M.-C. Kang, and S.-J. Ko, 'Parallel feature pyramid network for object detection,' in The European Conference on Computer Vision (ECCV) , 2018.
- [78] T. Kong, F. Sun, W. Huang, and H. Liu, 'Deep feature pyramid reconfiguration for object detection,' in The European Conference on Computer Vision (ECCV) , 2018.
- [79] H. Li, Y. Liu, W. Ouyang, and X. Wang, 'Zoom out-and-in network with map attention decision for region proposal and
- object detection,' International Journal of Computer Vision , vol. 127, no. 3, pp. 225-238, 2019.
- [80] Q. Zhao, T. Sheng, Y. Wang, Z. Tang, Y. Chen, L. Cai, and H. Ling, 'M2det: A single-shot object detector based on multi-level feature pyramid network,' in AAAI Conference on Artificial Intelligence , 2019.
- [81] G. Ghiasi, T. Lin, R. Pang, and Q. V. Le, 'NAS-FPN: learning scalable feature pyramid architecture for object detection,' in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2019.
- [82] H. Xu, L. Yao, W. Zhang, X. Liang, and Z. Li, 'Auto-fpn: Automatic network architecture adaptation for object detection beyond classification,' in The IEEE International Conference on Computer Vision (ICCV) , 2019.
- [83] J. Yu, Y. Jiang, Z. Wang, Z. Cao, and T. Huang, 'Unitbox: An advanced object detection network,' in The ACM International Conference on Multimedia , 2016.
- [84] L. Tychsen-Smith and L. Petersson, 'Improving object localization with fitness nms and bounded iou loss,' in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2018.
- [85] H. Rezatofighi, N. Tsoi, J. Gwak, A. Sadeghian, I. Reid, and S. Savarese, 'Generalized intersection over union: A metric and a loss for bounding box regression,' in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2019.
- [86] Z. Zheng, P. Wang, W. Liu, J. Li, Y. Rongguang, and R. Dongwei, 'Distance-iou loss: Faster and better learning for bounding box regression,' in AAAI Conference on Artificial Intelligence , 2020.
- [87] Z. Cai and N. Vasconcelos, 'Cascade R-CNN: Delving into high quality object detection,' in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2018.
- [88] J. Cao, Y. Pang, J. Han, and X. Li, 'Hierarchical shot detector,' in The IEEE International Conference on Computer Vision (ICCV) , 2019.
- [89] Z. Li, Z. Xie, L. Liu, B. Tao, and W. Tao, 'Iou-uniform r-cnn: Breaking through the limitations of rpn,' arXiv , vol. 1912.05190, 2019.
- [90] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollár, and C. L. Zitnick, 'Microsoft COCO: Common Objects in Context,' in The European Conference on Computer Vision (ECCV) , 2014.
- [91] A. Kuznetsova, H. Rom, N. Alldrin, J. R. R. Uijlings, I. Krasin, J. Pont-Tuset, S. Kamali, S. Popov, M. Malloci, T. Duerig, and V. Ferrari, 'The open images dataset V4: unified image classification, object detection, and visual relationship detection at scale,' arXiv , vol. 1811.00982, 2018.
- [92] S. Shao, Z. Li, T. Zhang, C. Peng, G. Yu, X. Zhang, J. Li, and J. Sun, 'Objects365: A large-scale, high-quality dataset for object detection,' in The IEEE International Conference on Computer Vision (ICCV) , 2019.
- [93] K. He, X. Zhang, S. Ren, and J. Sun, 'Deep residual learning for image recognition,' in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2016.
- [94] K.-K. Sung and T. Poggio, 'Example-based learning for viewbased human face detection,' IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI) , vol. 20, no. 1, pp. 39-51, 1998.
- [95] P. Viola and M. Jones, 'Rapid object detection using a boosted cascade of simple features,' in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2001.
- [96] N. Dalal and B. Triggs, 'Histograms of oriented gradients for human detection,' in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2005.
- [97] Y. Song, A. Schwing, Richard, and R. Urtasun, 'Training deep neural networks via direct loss minimization,' in The International Conference on Machine Learning (ICML) , 2016.
- [98] P. Henderson and V. Ferrari, 'End-to-end training of object class detectors for mean average precision,' in The Asian Conference on Computer Vision (ACCV) , 2017.
- [99] D. Dwibedi, I. Misra, and M. Hebert, 'Cut, paste and learn: Surprisingly easy synthesis for instance detection,' in The IEEE International Conference on Computer Vision (ICCV) , 2017.
- [100] N. Dvornik, J. Mairal, and C. Schmid, 'Modeling visual context is key to augmenting object detection datasets,' in The European Conference on Computer Vision (ECCV) , 2018.
- [101] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich, 'Going deeper with

- convolutions,' in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2015.
- [102] K. Oksuz, B. C. Cam, E. Akbas, and S. Kalkan, 'Localization recall precision (LRP): A new performance metric for object detection,' in The European Conference on Computer Vision (ECCV) , 2018.
- [103] K. Simonyan and A. Zisserman, 'Very deep convolutional networks for large-scale image recognition,' in The International Conference on Learning Representations (ICLR) , 2015.
- [104] A. Newell, K. Yang, and J. Deng, 'Stacked hourglass networks for human pose estimation,' in The European Conference on Computer Vision (ECCV) , B. Leibe, J. Matas, N. Sebe, and M. Welling, Eds., 2016.
- [105] A. G. Howard, M. Zhu, B. Chen, D. Kalenichenko, W. Wang, T. Weyand, M. Andreetto, and H. Adam, 'Mobilenets: Efficient convolutional neural networks for mobile vision applications,' in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2017.
- [106] Z. Li, C. Peng, G. Yu, X. Zhang, Y. Deng, and J. Sun, 'Detnet: Design backbone for object detection,' in The European Conference on Computer Vision (ECCV) , 2018.
- [107] L. Zhu, Z. Deng, X. Hu, C.-W. Fu, X. Xu, J. Qin, and P.-A. Heng, 'Bidirectional feature pyramid network with recurrent attention residual modules for shadow detection,' in The European Conference on Computer Vision (ECCV) , 2018.
- [108] Y. Sun, P. S. K. P, J. Shimamura, and A. Sagata, 'Concatenated feature pyramid network for instance segmentation,' arXiv , vol. 1904.00768, 2019.
- [109] S. S. Seferbekov, V. I. Iglovikov, A. V. Buslaev, and A. A. Shvets, 'Feature pyramid network for multi-class land segmentation,' in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops , 2018.
- [110] A. Kirillov, R. B. Girshick, K. He, and P. Dollár, 'Panoptic feature pyramid networks,' in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2019.
- [111] E. H. Adelson, C. H. Anderson, J. R. Bergen, P. J. Burt, and J. M. Ogden, 'Pyramid methods in image processing,' RCA Engineer , vol. 29, no. 6, pp. 33-41, 1984.
- [112] F. Yu and V. Koltun, 'Multi-scale context aggregation by dilated convolutions,' in The International Conference on Learning Representations (ICLR) , 2016.
- [113] X. Wang, R. Girshick, A. Gupta, and K. He, 'Non-local neural networks,' in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2018.
- [114] G. Huang, Z. Liu, and K. Q. Weinberger, 'Densely connected convolutional networks,' in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2017.
- [115] K. He, X. Zhang, S. Ren, and J. Sun, 'Spatial pyramid pooling in deep convolutional networks for visual recognition,' in The European Conference on Computer Vision (ECCV) , D. Fleet, T. Pajdla, B. Schiele, and T. Tuytelaars, Eds., 2014.
- [116] J. Hu, L. Shen, and G. Sun, 'Squeeze-and-excitation networks,' in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2018.
- [117] S. Ioffe and C. Szegedy, 'Batch normalization: Accelerating deep network training by reducing internal covariate shift,' in The International Conference on Machine Learning (ICML) , 2015.
- [118] B. Zoph and Q. V. Le, 'Neural architecture search with reinforcement learning,' in The International Conference on Learning Representations (ICLR) , 2017.
- [119] B. Zoph, V. Vasudevan, J. Shlens, and Q. V. Le, 'Learning transferable architectures for scalable image recognition,' in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2018.
- [120] E. Real, A. Aggarwal, Y. Huang, and Q. V. Le, 'Regularized evolution for image classifier architecture search,' in AAAI Conference on Artificial Intelligence , 2019.
- [121] M. Tan and Q. V. Le, 'Efficientnet: Rethinking model scaling for convolutional neural networks,' in The International Conference on Machine Learning (ICML) , 2019.
- [122] Y. Chen, T. Yang, X. Zhang, G. Meng, C. Pan, and J. Sun, 'Detnas: Backbone search for object detection,' in Advances in Neural Information Processing Systems (NIPS) , 2019.
- [123] P. Sermanet, D. Eigen, X. Zhang, M. Mathieu, R. Fergus, and Y. Lecun, 'Overfeat: Integrated recognition, localization and detection using convolutional networks,' in International Conference on Learning Representations (ICLR) , 2014.
- [124] P. J. Huber, 'Robust estimation of a location parameter,' Annals of Statistics , vol. 53, no. 1, pp. 73-101, 1964.
- [125] B. Jiang, R. Luo, J. Mao, T. Xiao, and Y. Jiang, 'Acquisition of localization confidence for accurate object detection,' in The European Conference on Computer Vision (ECCV) , 2018.
- [126] E. Goldman, R. Herzig, A. Eisenschtat, J. Goldberger, and T. Hassner, 'Precise detection in densely packed scenes,' in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2019.
- [127] J. Choi, D. Chun, H. Kim, and H.-J. Lee, 'Gaussian yolov3: An accurate and fast object detector using localization uncertainty for autonomous driving,' in The IEEE International Conference on Computer Vision (ICCV) , 2019.
- [128] Z. Tan, X. Nie, Q. Qian, N. Li, and H. Li, 'Learning to rank proposals for object detection,' in The IEEE International Conference on Computer Vision (ICCV) , 2019.
- [129] S. Gidaris and N. Komodakis, 'Object detection via a multiregion and semantic segmentation-aware cnn model,' in The IEEE International Conference on Computer Vision (ICCV) , 2015.
- [130] --, 'Attend refine repeat: Active box proposal generation via in-out localization,' in The British Machine Vision Conference (BMVC) , 2016.
- [131] J. Dai, H. Qi, Y. Xiong, Y. Li, G. Zhang, H. Hu, and Y. Wei, 'Deformable convolutional networks,' in The IEEE International Conference on Computer Vision (ICCV) , 2017.
- [132] K. Chen, J. Wang, J. Pang, Y. Cao, Y. Xiong, X. Li, S. Sun, W. Feng, Z. Liu, J. Xu, Z. Zhang, D. Cheng, C. Zhu, T. Cheng, Q. Zhao, B. Li, X. Lu, R. Zhu, Y. Wu, J. Dai, J. Wang, J. Shi, W. Ouyang, C. C. Loy, and D. Lin, 'MMDetection: Open mmlab detection toolbox and benchmark,' arXiv preprint arXiv:1906.07155 , 2019.
- [133] T. Yang, X. Zhang, Z. Li, W. Zhang, and J. Sun, 'Metaanchor: Learning to detect objects with customized anchors,' in Advances in Neural Information Processing Systems (NIPS) , 2018.
- [134] C. Peng, T. Xiao, Z. Li, Y. Jiang, X. Zhang, K. Jia, G. Yu, and J. Sun, 'Megdet: A large mini-batch object detector,' in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2018.
- [135] M. Guo, A. Haque, D.-A. Huang, S. Yeung, and L. Fei-Fei, 'Dynamic task prioritization for dynamic task learning,' in The European Conference on Computer Vision (ECCV) , 2018.
- [136] N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, 'Smote: Synthetic minority over-sampling technique,' Journal of Artificial Intelligence Research , vol. 16, pp. 321-357, 2002.
- [137] C. Drummond and R. C. Holte, 'C4.5, class imbalance, and cost sensitivity: Why under-sampling beats over-sampling,' in The International Conference on Machine Learning (ICML) , 2003.
- [138] H. He, Y. Bai, E. A. Garcia, and S. Li, 'Adasyn: Adaptive synthetic sampling approach for imbalanced learning,' in International Joint Conference on Neural Networks , 2008.
- [139] M. Buda, A. Maki, and M. A. Mazurowski, 'A systematic study of the class imbalance problem in convolutional neural networks,' Neural Networks , vol. 106, pp. 249-259, 2018.
- [140] S. Li, L. Zhouche, and H. Qingming, 'Relay Backpropagation for Effective Learning of Deep Convolutional Neural Networks,' in The European Conference on Computer Vision (ECCV) , 2016.
- [141] Y.-X. Wang, D. Ramanan, and M. Hebert, 'Learning to model the tail,' in Advances in Neural Information Processing Systems (NIPS) , 2017.
- [142] Y. Cui, Y. Song, C. Sun, A. G. Howard, and S. J. Belongie, 'Large scale fine-grained categorization and domain-specific transfer learning,' in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2018.
- [143] C. Huang, Y. Li, C. C. Loy, and X. Tang, 'Learning deep representation for imbalanced classification,' in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2016.
- [144] Y. Cui, M. Jia, T. Lin, Y. Song, and S. J. Belongie, 'Class-balanced loss based on effective number of samples,' in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2019.
- [145] M. P. Kumar, B. Packer, and D. Koller, 'Self-paced learning for latent variable models,' in Advances in Neural Information Processing Systems (NIPS) , 2010.
- [146] K. Kumar Singh and Y. Jae Lee, 'Hide-and-seek: Forcing a network to be meticulous for weakly-supervised object and action localization,' in The IEEE International Conference on Computer Vision (ICCV) , 2017.

- [147] Q. Dong, S. Gong, and X. Zhu, 'Class rectification hard mining for imbalanced deep learning,' in The International Conference on Computer Vision (ICCV) , 2017.
- [148] H.-S. Chang, E. Learned-Miller, and A. McCallum, 'Active bias: Training more accurate neural networks by emphasizing high variance samples,' in Advances in Neural Information Processing Systems (NIPS) , 2017.
- [149] K. Wei, R. Iyer, and J. Bilmes, 'Submodularity in data subset selection and active learning,' in The International Conference on Machine Learning (ICML) , 2015, pp. 1954-1963.
- [150] V. Kaushal, A. Sahoo, K. Doctor, N. R. Uppalapati, S. Shetty, P. Singh, R. K. Iyer, and G. Ramakrishnan, 'Learning from less data: Diversified subset selection and active learning in image classification tasks,' arXiv , vol. 1805.11191, 2018.
- [151] O. Sener and S. Savarese, 'Active learning for convolutional neural networks: A core-set approach,' in The International Conference on Learning Representations (ICLR) , 2018.
- [152] K. Vodrahalli, K. Li, and J. Malik, 'Are all training examples created equal? an empirical study,' arXiv , vol. 1811.12569, 2018.
- [153] V. Birodkar, H. Mobahi, and S. Bengio, 'Semantic redundancies in image-classification datasets: The 10% you don't need,' arXiv , vol. 1901.11409, 2019.
- [154] D. Mahajan, R. Girshick, V. Ramanathan, K. He, M. Paluri, Y. Li, A. Bharambe, and L. van der Maaten, 'Exploring the Limits of Weakly Supervised Pretraining,' in The European Conference on Computer Vision (ECCV) , 2018.
- [155] Y. Liu, T. Gao, and H. Yang, 'Selectnet: Learning to sample from the wild for imbalanced data training,' arXiv , vol. 1905.09872, 2019.
- [156] S. S. Mullick, S. Datta, and S. Das, 'Generative Adversarial Minority Oversampling,' arXiv , vol. 1903.09730, 2019.
- [157] X. Zhu, Y. Liu, Z. Qin, and J. Li, 'Data augmentation in emotion classification using generative adversarial networks,' arXiv preprint arXiv:1711.00648 , 2017.
- [158] C. Bowles, L. Chen, R. Guerrero, P. Bentley, R. Gunn, A. Hammers, D. A. Dickie, M. V. Hernández, J. Wardlaw, and D. Rueckert, 'Gan augmentation: augmenting training data using generative adversarial networks,' arXiv preprint arXiv:1810.10863 , 2018.
- [159] X. Zhang, Z. Fang, Y. Wen, Z. Li, and Y. Qiao, 'Range loss for deep face recognition with long-tailed training data,' in The IEEE International Conference on Computer Vision (ICCV) , 2016.
- [160] X. Yin, X. Yu, K. Sohn, X. Liu, and M. K. Chandraker, 'Feature transfer learning for deep face recognition with long-tail data,' arXiv , vol. 1803.09014, 2018.
- [161] C. Huang, Y. Li, C. L. Chen, and X. Tang, 'Deep imbalanced learning for face recognition and attribute prediction,' IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI) , 2019.
- [162] J. Hu, J. Lu, and Y. Tan, 'Discriminative deep metric learning for face verification in the wild,' in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2014.
- [163] S. Bell and K. Bala, 'Learning visual similarity for product design with convolutional neural networks,' ACM Trans. on Graphics (SIGGRAPH) , vol. 34, no. 4, 2015.
- [164] F. Schroff, D. Kalenichenko, and J. Philbin, 'Facenet: A unified embedding for face recognition and clustering,' in Conference on Computer Vision and Pattern Recognition (CVPR) , 2015.
- [165] E. Hoffer and N. Ailon, 'Deep metric learning using triplet network,' in The International Conference on Learning Representations (ICLR) , 2015.
- [166] W. Ge, 'Deep metric learning with hierarchical triplet loss,' in The European Conference on Computer Vision (ECCV) , 2018.
- [167] Y. Yuan, K. Yang, and C. Zhang, 'Hard-aware deeply cascaded embedding,' in The IEEE International Conference on Computer Vision (ICCV) , 2017.
- [168] B. Harwood, G. K. B. VijayKumarB., G. Carneiro, I. D. Reid, and T. Drummond, 'Smart mining for deep metric learning,' in The IEEE International Conference on Computer Vision (ICCV) , 2017.
- [169] Y. Cui, F. Zhou, Y. Lin, and S. Belongie, 'Fine-grained categorization and dataset bootstrapping using deep metric learning with humans in the loop,' in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2016.
- [170] H. O. Song, Y. Xiang, S. Jegelka, and S. Savarese, 'Deep metric learning via lifted structured feature embedding,' in The IEEE Computer Vision and Pattern Recognition (CVPR) , 2016.
- [171] C. Huang, C. C. Loy, and X. Tang, 'Local similarity-aware deep feature embedding,' in Advances in Neural Information Processing Systems (NIPS) , 2016.
- [172] Y. Duan, W. Zheng, X. Lin, J. Lu, and J. Zhou, 'Deep adversarial metric learning,' in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2018.
- [173] Y. Zhao, Z. Jin, G.-j. Qi, H. Lu, and X.-s. Hua, 'An adversarial approach to hard triplet generation,' in The European Conference on Computer Vision (ECCV) , 2018.
- [174] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. WardeFarley, S. Ozair, A. Courville, and Y. Bengio, 'Generative adversarial nets,' in Advances in Neural Information Processing Systems (NIPS) , 2014.
- [175] W. Zheng, Z. Chen, J. Lu, and J. Zhou, 'Hardness-aware deep metric learning,' in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2019.
- [176] A. Yao, 'Interactive object detection,' in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2012.
- [177] C. Li, J. Yan, F. Wei, W. Dong, Q. Liu, and H. Zha, 'Self-paced multi-task learning,' in AAAI Conference on Artificial Intelligence , 2017.
- [178] A. Kendall, Y. Gal, and R. Cipolla, 'Multi-task learning using uncertainty to weigh losses for scene geometry and semantics,' in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2018.
- [179] S. Liu, E. Johns, and A. J. Davison, 'End-to-end multi-task learning with attention,' in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2019.
- [180] C.-Y. L. Zhao Chen, Vijay Badrinarayanan and A. Rabinovich, 'Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks,' in International Conference on Machine Learning (ICML) , 2018.

<!-- image -->

Kemal Oksuz received B.Sc. in System Engineering from Land Forces Academy of Turkey in 2008 and his M.Sc. in Computer Engineering in 2016 from Bogazici University with high honor degree. He is currently pursuing his Ph.D. in Middle East Technical University, Ankara, Turkey. His research interests include computer vision with a focus on object detection.

<!-- image -->

<!-- image -->

Baris Can Cam received his B.Sc. degree in Electrical and Electronics Engineering from Eskisehir Osmangazi University, Turkey in 2016. He is currently pursuing his M.Sc. in Middle East Technical University, Ankara, Turkey. His research interests include computer vision with a focus on object detection.

Sinan Kalkan received his M.Sc. degree in Computer Engineering from Middle East Technical University (METU), Turkey in 2003, and his Ph.D. degree in Informatics from the University of Göttingen, Germany in 2008. After working as a postdoctoral researcher at the University of Gottingen and at METU, he became an assistant professor at METU in 2010. Since 2016, he has been working as an associate professor on problems within Computer Vision, and Developmental Robotics.

<!-- image -->

Emre Akbas is an assistant professor at the Department of Computer Engineering, Middle East Technical University (METU), Ankara, Turkey. He received his Ph.D. degree from the Department of Electrical and Computer Engineering, University of Illinois at Urbana-Champaign in 2011. His M.Sc. and B.Sc. degrees in computer science are both from METU. Prior to joining METU, he was a postdoctoral research associate at the Department of Pyschological and Brain Sciences, University of California Santa Barbara. His re-

search interests are in computer vision and deep learning with a focus on object detection and human pose estimation.