[DOI: 10.1049/cvi2.12318](https://doi.org/10.1049/cvi2.12318)

## ORIGINAL RESEARCH

<!-- image -->

<!-- image -->

WILEY

## To crop or not to crop: Comparing whole - image and cropped classification on a large dataset of camera trap images

Tomer Gadot 1 Sara Beery 1,2

S ¸ tefan Istrate 1 Tanya Birch 1 |

| Hyungwon Kim 1 Jorge Ahumada 3

|

|

1 Google, Mountain View, California, USA

2 Department of Electrical Engineering and Computer Science, Massachusetts Institute of Technology, Cambridge, Massachusetts, USA

3 WildMon, Dale, Texas, USA

## Correspondence

Tomer Gadot, Google, 1600 Amphitheatre Parkway, Mountain View, California, USA. Email: tomerg@google.com

## Funding information

Whitehead Institute for Biomedical Research, Massachusetts Institute of Technology; Wildlife Conservation Society; Gordon and Betty Moore Foundation; Google

<!-- image -->

<!-- image -->

<!-- image -->

## Abstract

Camera traps facilitate non - invasive wildlife monitoring, but their widespread adoption has created a data processing bottleneck: a camera trap survey can create millions of images, and the labour required to review those images strains the resources of conservation organisations. AI is a promising approach for accelerating image review, but AI tools for camera trap data are imperfect; in particular, classifying small animals remains difficult, and accuracy falls off outside the ecosystems in which a model was trained. It has been proposed that incorporating an object detector into an image analysis pipeline may help address these challenges, but the benefit of object detection has not been systematically evaluated in the literature. In this work, the authors assess the hypothesis that classifying animals cropped from camera trap images using a species - agnostic detector yields better accuracy than classifying whole images. We find that incorporating an object detection stage into an image classification pipeline yields a macro - average F1 improvement of around 25% on a large, long - tailed dataset; this improvement is reproducible on a large public dataset and a smaller public benchmark dataset. The authors describe a classification architecture that performs well for both whole and detector - cropped images, and demonstrate that this architecture yields state - of - the - art benchmark accuracy.

## KEYWORDS

computer vision, object detection

## 1 | INTRODUCTION AND RELATED WORK

Camera traps are static, ground - level cameras used widely across ecology and conservation to monitor species, with more than a million camera traps estimated to be deployed worldwide. Images are collected from camera traps usually based on a heat/motion trigger or timelapse trigger. These images are then processed into scientific observations, such as records of species occurrence. This processing is typically done by human experts, but as cameras became cheaper over the last 30 years, the average number of deployed cameras per project increased, leading to a processing bottleneck, as humans could not keep up with large - scale data streams.

The need for automated computer vision systems to assist in processing camera trap data was established as early as 2018 [53], and many papers have been published in subsequent years demonstrating the potential for AI to help automate species identification on benchmark datasets, often with associated training datasets, AI models, and/or software tools [51]. Starting in 2022, we began to see work published that moved beyond demonstrating the hypothetical potential of AI to save human time to the direct deployment and analysis of AI 'in production', as a key component of data processing pipelines for camera trap data. AI has helped researchers address a variety of ecological topics, including the impact of fire on wildlife [6, 8, 42], the impact of beaver activity on river flow [1], species occupancy [3, 29], species abundance [14, 26, 50], faunal biomass

This is an open access article under the terms of the Creative Commons Attribution License, which permits use, distribution and reproduction in any medium, provided the original work is properly cited.

© 2024 The Author(s). IET Computer Vision published by John Wiley &amp; Sons Ltd on behalf of The Institution of Engineering and Technology.

|

Dan Morris 1 D

|

-

[37], domestic hybridisation [7], the impact of grazing on landscapes [38], the impact of human activity on animal space use [4, 22, 43], multispecies interactions [15], and the effectiveness of managed harvest [5]. These works have clearly demonstrated the practical value of AI for camera trap data processing, with reported time reductions up to 8.4x [23].

However, though the value of AI has been shown, there are still challenges and limitations that often restrict its use. First, animals which occupy only a small region of interest relative to the image frame, either because the animal itself is small or because it is far from the camera, have proven more challenging to accurately detect and classify [34, 54]. Second, it has proven difficult to build AI systems that are geospatially generalisable, or even in some cases generalisable to new sensors in the same part of the world [9, 44]. This is likely due to a combination of visual shift -changing backgrounds, orientations, and sometimes even colour morphs of the animals of interest at different camera locations or for different camera types-and subpopulation shift -changing relative frequency of observation for different species over time and space, due to variance in species range and density as well as environmental and human factors such as climate change or land use change [33].

Several different methods and best practices have been proposed to address these limitations. One approach is to rely on a class - agnostic detection model which locates animals with a bounding box (regardless of species) as a first stage of an AI - based processing pipeline; these proposed boxes are cropped from the original images, and species classification is performed on crops. This serves the dual purpose of enabling higher - resolution information to pass through the machine learning model as these crops can often be run through a machine learning model at or near full resolution, as well as reducing some of the impact of the background context which in some cases may induce a machine learning model to learn spurious correlations. This approach has been shown to improve accuracy on novel locations compared to classifying whole images [9, 18, 28, 39]. Additional work has extended this approach to study model calibration [20], active learning [13, 41], neural architecture search [30], and interpretability [21].

Not all methods using a first - stage detector focus on species identification; this type of generic detection as a first filtering pass has been shown to improve performance for diverse downstream tasks including animal re - identification [19, 46], distance estimation [27], and counting [52]. Additional work has investigated the use of a first - stage detector paired with manual segmentation or a semantic segmentation model plus copy - paste augmentation to generate more diverse training data for crop classification [10, 24].

However, while crop - based classification is popular, there is concurrent work focusing on whole - image classification without a first - stage detector, often incorporating ensembles of multiple classification models [40, 48, 56, 58]. Other work has explored end - to - end multiclass species detection models, as opposed to an animal detector followed by a separate species classification model [11, 36, 45, 47].

In this work, we demonstrate that incorporating a detection stage into camera trap image classification improves accuracy relative to a whole - image classifier alone. We demonstrate these results in a context of a very large, long - tailed private dataset, and we demonstrate that our findings are reproducible on a large public dataset, and a smaller public benchmark dataset. We also present hyperparameters for our classification architecture, and demonstrate that our classification approach is competitive with the state of the art on a public benchmark.

## 2 | METHODS

## 2.1 | Data

In Section 4, we will present results on three different datasets:

1. A non - public dataset of ~42M images that is geographically diverse and extremely - long - tailed ('WI')
2. A public dataset of ~15M images that is less geographically diverse ('LILA')
3. A smaller public benchmark dataset of ~180k images ('iWildCam')

In this section, we describe these three datasets in detail.

A very small minority (approximately 0.4% of the Wildlife Insights (WI) dataset, for example) of images originally contained multiple labels, which in some cases reflect multi - species images, but in other cases represent disagreement among labellers or other metadata errors. For the 'filtered' datasets described below, images with multiple labels were removed from the dataset. We consider this to be a conservative approach to evaluating our hypotheses about the benefit of detectors: incorporating multi - species images trivially increases the benefit of detectors, so if we demonstrate the benefit of detectors on single - label images, we are likely only underestimating that benefit. We will revisit this theme in our conclusion.

We note that all three datasets contain more training labels than test labels; removing non - test classes from training would likely have increased our reported accuracy values, but that would be less reflective of real - world use, where you cannot predict which classes from the long tail of a training dataset will be observed at test time.

## 2.1.1 | WI dataset

WI(wildlifeinsights.org) is a cloud - based platform for managing, reviewing, and analysing camera trap images [2]. The WI platform hosts approximately 118M images with human - verified labels, of which approximately 42M contain animals. Images on WI generally become publicly available after an embargo period. A WI 'project' is a collection of cameras and their associated images, typically representing a single ecosystem and managed by a single organisation. Labels on WI may include multiple taxonomic levels, for example, 'white - tailed deer' (a species) and 'cervidae' (a family) are both valid labels.

In this paper, we report results on a dataset of ~51M images derived from both public and non - public WI projects; we will refer to this dataset as 'WI'. Section 2.2 describes the process by which we filtered the entire dataset to arrive at the 51M image 'WI' dataset. All projects that contributed images to this work are cited in the 'WI data references' section.

Training and evaluating camera trap AI systems on the same cameras can artificially inflate the results a user might expect in new locations [9], so it is generally recommended that locations be assigned to either training or test data splits, but never both. For the 'WI' dataset, we take an even more conservative approach to data splitting: a test set of 31 projects -representing a total of 2897 locations-was constructed manually to ensure both geographic and taxonomic diversity. From the remaining projects, 90% of locations were assigned randomly to a training split, and 10% to a validation split.

Table 1 summarises the WI dataset. Figure 1 summarises the geographic breadth of the three data splits. Supplementary Table ST1 lists all of the species present in the WI dataset, with the corresponding counts in the train/validation/test splits.

## 2.1.2 | LILA dataset

The Labelled Information Library of Alexandria: Biology and Conservation (https://lila.science) (LILA) is a public repository of labelled datasets related to wildlife conservation. Of 42 datasets on LILA, 21 represent labelled camera trap images, containing a total of ~19.3M images. Datasets on LILA do not contain precise location information, but locations within each cameratrap dataset are assigned unique identifiers to facilitate the creation of training and test sets. As was the case with the WI dataset, labels may represent multiple taxonomic levels.

TABLE 1 Summary of the train, validation, and test splits for the WI dataset. 'Blank' images are images whose labels indicate that no animals, humans, or vehicles are present.

| Split      |   # of blank images |   # of non - blank images |   # of locations |   # of labels |
|------------|---------------------|---------------------------|------------------|---------------|
| Train      |          16,477,843 |                19,670,994 |           27,277 |          2176 |
| Validation |           1,609,250 |                 1,953,246 |             2978 |          1205 |
| Test       |           1,048,018 |                 1,691,427 |             2830 |           657 |

-

In this paper, we report results on a dataset of ~14.8M images from LILA. Section 2.2 describes the process by which we filtered the entire dataset to arrive at the 14.8M image dataset that we will refer to as 'LILA'. Locations from LILA are randomly split into train (90%), validation (5%), and test (5%) data splits. To facilitate reproducibility of our results, we include the list of images in each of these splits as supplementary material.

Table 2 summarises the LILA dataset.

## 2.1.3 | iWildCam dataset

iWildcam - 2020 [33] is a public benchmark dataset that includes images from 323 cameras; we will refer to this dataset as 'iWildCam'. In order to facilitate independent evaluation of a model's accuracy 'in domain' (i.e. on the same locations on which the model was trained) versus 'out of domain' (i.e. on new locations), the dataset is broken into five splits: 'Train' (most of the images from most of the locations), 'ID validation'/'ID test' (both are held - out days from the training locations), and 'OOD validation'/'OOD test' (locations not included in training).

Table 3 summarises the iWildCam dataset.

## 2.2 | Data preparation

## 2.2.1 | MegaDetector overview

The data preparation procedure we describe in this section (as well as the experiments we describe in the next section) relies on MegaDetector (MD) [12], an object detection model that detects animals, humans, and vehicles in camera trap images.

TABLE 2 Summary of the train, validation, and test splits for the LILA dataset.

| Split      |   # of blank images |   # of non - blank images |   # of locations |   # of labels |
|------------|---------------------|---------------------------|------------------|---------------|
| Train      |           7,319,982 |                 6,202,518 |             5011 |           773 |
| Validation |             635,567 |                   151,950 |              267 |           317 |
| Test       |             346,833 |                   148,577 |              298 |           309 |

FIGURE 1 Geographic distribution of locations used in the WI (a) training, (b) validation, and (c) test sets.

<!-- image -->

(c)

-

TABLE 3 Summary of the train, validation, and test splits for the iWildCam dataset.

| Split          |   # of blank images |   # of non - blank images |   # of locations |   # of labels |
|----------------|---------------------|---------------------------|------------------|---------------|
| Train          |              47,831 |                    64,150 |              243 |           175 |
| ID validation  |                2949 |                      3387 |              146 |            69 |
| OOD validation |               2,391 |                    10,003 |               32 |            75 |
| ID test        |                2006 |                      6148 |              164 |            86 |
| OOD test       |              14,106 |                    28,685 |               48 |           101 |

Specifically, we use MD version 5a; for brevity, in this paper, we will use 'MD' specifically to refer to MD version 5a. MD is based on the YOLOv5x6 object detection architecture [31], which accepts images that are 1280 pixels wide. Importantly, this is considerably larger than the input size of even the largest available image classification architectures; as per above, the opportunity to look at more pixels in the first stage of AI processing is a major reason we hypothesise that a two - stage approach will provide better results than whole - image classification alone.

Though our results are influenced by MD's accuracy, a full evaluation of MD is outside the scope of this paper; for evaluations of MD's performance, see refs. [23, 51, 55].

## 2.2.2 | Data preparation overview

Though AI studies frequently use human - annotated camera trap images as ground truth, camera trap data frequently contains numerous errors [32, 59]. This may include the mislabelling of one species as another, or the omission of small or distant animals due to fatigue or lack of annotator expertise. However, data errors are not necessarily human errors: camera trap data may be transformed several times before it's available to AI researchers, which may result in errors of taxonomic misinterpretation, replacement of null labels based on invalid assumptions (often resulting in repetition of a species label), or the misinterpretation of episode labels as image labels. Camera trap images are frequently collected in episodes (aka bursts, sequences, events) of 3-10 images, and it is common for an animal to be present in only some images within an episode; if that label is applied to all images in the episode, blank images may be interpreted as non - blank when training and evaluating AI systems.

Consequently, the use of 'raw' camera trap data presents challenges for training and evaluating AI systems: incorrect labels may confuse the model during training, and incorrect labels create noise in evaluation metrics. It is possible to use AI to remove images with unreliable labels (in particular, MD can help eliminate labels that confuse blank with non - blank images), but results that depend on datasets that have been filtered with AI incur the biases of that AI.

In this paper, we attempt to get the 'best of both worlds' by reporting metrics on two variants of each dataset: we will refer to the 'filtered' variant of each dataset when we are describing a version that has been filtered by MD to minimise label noise (details in the next subsection). The class labels that survive this filtering are generally quite reliable, and the size of our dataset allows us adequate support even after filtering out possibly - unreliable labels, but we cannot meaningfully report on blank/non - blank accuracy using those labels, since the labels are now limited by MD's accuracy. Consequently, we also report on the 'unfiltered' variant of each dataset, which has not been subjected to this filtering, and thus comes with considerable label noise, but allows us to report on blank/non - blank accuracy. All of our training uses the 'filtered' datasets.

## 2.2.3 | Data preparation procedure

Except where otherwise noted, the following procedure is applied to all three datasets described above.

1. Remove images with labels that are ambiguous, or outside the scope of this evaluation (e.g. images labelled as 'unknown', 'fire', 'snow', blank images from projects that used animal carcasses as bait, or blank images from projects that used inflatable decoys as bait).
2. For the filtered variants of each dataset only , remove images where MD results conflict with the labels; in particular, remove:
- a. Images with a non - blank label, in which MD finds no objects with confidence ≥ 0.4
- b. Images with a blank label, in which MD finds an object with confidence ≥ 0.7
- c. Images with an animal label, in which MD finds a human or a vehicle with confidence ≥ 0.7
- d. Images with a human label, in which MD finds an animal or a vehicle with confidence ≥ 0.6
- e. Images with a vehicle label, in which MD finds an animal or a human with confidence ≥ 0.6

The thresholds used here to create the 'filtered' datasets were chosen based on (a) qualitative evaluations from samples of the training data, and (b) MD's documentation: these thresholds are well above the typical threshold recommended for wildlife monitoring applications (0.2), and by using only very high - confidence predictions to alter the original labels, we are biasing our process towards trusting the original labels, and only relying on MD's output if there is an extremely clear discrepancy.

3. For the WI dataset only , for each (project, label) pair, sample at most N of the remaining images to keep in each split, with N = 1,000,000 for train, and N = 100,000 for validation and test. These are high thresholds that are used only to prevent severe overrepresentation of projects with enormous numbers of blanks or very common species (e.g. cattle).

Because iWildCam is a standard benchmark dataset, we don't create filtered variants of the iWildCam ID test or OOD test splits.

TABLE 4 The impact of data filtering on the WI and LILA test sets.

-

| Split                  |   # of blank images |   # of non - blank images |   # of locations |   # of labels |
|------------------------|---------------------|---------------------------|------------------|---------------|
| WI test (unfiltered)   |           1,067,885 |                 2,015,007 |             2831 |           667 |
| WI test (filtered)     |           1,048,018 |                 1,691,427 |             2830 |           657 |
| LILA test (unfiltered) |             350,673 |                   178,840 |              298 |           316 |
| LILA test (filtered)   |             346,833 |                   148,577 |              298 |           309 |

Note : Rows 2 and 4 are identical to the 'test' rows in Tables 1 and 2, respectively.

Abbreviation: LILA, Labelled Information Library of Alexandria.

Other than the iWildCam test splits, the tables in Section 2.1 present the filtered variants of each dataset. These are the only variants used in training, but we evaluate and report on both the filtered and unfiltered variants of the WI and LILA datasets. Table 4 presents the impact of this filtering process on both datasets.

## 2.3 | Models

In this section, we describe two approaches to classifier training (on whole images and crops), then two strategies for ensembling multiple models together. The architecture is the same for both classifiers, so we will first describe the classifier architecture, then the process by which images are fed to the classifier for the whole - image and crop scenarios, then the ensembling strategies. All the models and ensembling strategies we describe produce a single label per image.

## 2.3.1 | Classifier architecture

The architecture and hyperparameters are the same for the whole - image and crop classifiers; due to variations in dataset size, some hyperparameters vary across datasets. All classifiers are based on EfficientNetV2 - M [49], initialised with weights from the Keras library that have been pretrained on ImageNet. We selected EfficientNetV2 - M based on the published accuracy/ cost tradeoffs: as of [49], EfficientNetV2 - Mis at near state - of - the - art accuracy on ImageNet, and any models outperforming it-even marginally-have at least twice as many parameters.

Dropout is applied before the dense classification layer (dropout rate 0.3 for WI and LILA, 0.5 for iWildCam). Training uses a cross - entropy loss with a batch size of 480, and the Adam optimiser with Keras's default optimiser parameters. Images are shuffled randomly at each epoch. For WI and LILA, five warmup epochs are run with all layers frozen other than the dense classification layer and a learning rate of 1e - 3; after the warmup period, no layers are frozen. The learning rate is initialised to 1e - 4, after which it decays according to the Keras ReduceLROnPlateau function, which reduces the learning rate by half when no improvements are seen in the validation loss for five epochs. Training concludes when no improvements are seen in the validation loss for 20 epochs. Experimental results are based on the checkpoint with the lowest validation loss.

## 2.3.2 | Whole - image classifier

For the whole - image classifier, during both training and inference, the following modifications are applied to each image:

- 400 pixels (or 30% of the image height, whichever is smaller) are cropped vertically from the image, equally split between the top and bottom of the image.
- To offer some intuition as to why this is important: camera trap images typically contain manufacturer - specific overlays at the top or bottom of the image (with a manufacturer logo, along with the date and time). Camera trap data are typically ingested as 'projects', where 'project' is a collection of images from a finite number of cameras, maintained by a single organisation, in a specific geographic area. Because species are inherently tied to geography, and organisations typically purchase cameras in bulk and use a consistent brand/model throughout the life of a project, an inevitable correlation exists between [camera model] and [species] that is not predictive of future data. This is consistent with our anecdotal observations that whole - image classifiers learn correlations between camera brand and species priors that are eliminated when we crop the top and bottom of the image.
- The remaining pixels are resized to 480 � 480 (the input size of the EfficientNetV2 - M architecture).
- In training only, we apply random augmentations, using the Keras RandAugment function [17] (augmentations\_per\_- image = 4, magnitude = 0.4, magnitude\_stddev = 0.2, rate = 0.9), followed by a random horizontal flip.

## 2.3.3 | Crop classifier

- During training, images with a maximum MD detection confidence &lt; 0.01 are discarded.
- All other images are cropped to the highest - confidence MD detection. Note that this will include many detections with confidence ≥ 0.01 that are associated with a 'blank' label. In contrast to some prior work that relies entirely on MD to eliminate blanks, we also allow our classifier to learn a 'blank' class.
- The crop is resized to 480 � 480 (the input size of the EfficientNetV2 - M architecture).

-

- In training only, the same random augmentations are applied for the crop classifier that were used for the whole - image classifier.

The vertical cropping described for the whole - image classifier is not applied when training or evaluating the crop classifier.

## 2.3.4 | Ensembles

We will present experimental results for the whole - image classifier and the crop classifier, but using a classifier and a detector are not mutually exclusive; in this section, we describe two strategies for using a classifier in addition to a detector.

## Whole-image classifier, ensembled with coarse detector

Not all AI errors are equally costly for ecologists. In particular, most AI - accelerated workflows still involve human review of either all non - blank images or all images other than the most common species, so AI errors that prevent review of important images-for example, predicting a non - blank image as blankare much more costly than other errors. Consequently, even systems that use a whole - image classifier may benefit from getting a 'vote' on coarse - level categorisation from a coarse - grained detector. For example, if the detector is very confident that an animal is present in an image, the system may prefer to call that image non - blank, even if the whole - image classifier is confident that the image is blank.

Here we describe a series of empirically - optimised heuristics for using both a whole - image classifier and a coarse detector (MD). In Section 4, we will refer to this ensemble strategy as 'whole - image classifier þ MD'.

In this ensembling strategy, each image is processed by the whole - image classifier and by MD. Softmax scores from the whole - image classifier are combined with the highest - confidence MD detection. Thresholds were manually selected to maximise the F1 of four coarse - grained categories ( animal , blank , human , vehicle ) on the WI validation set; the same thresholds were used for experiments on all three datasets.

In this ensembling strategy, the final predicted class is determined according to the following procedure (these steps are run in order):

1. Accept high - confidence MD predictions ( ≥ 0.7) for the human and vehicle classes, regardless of the classifier output.
2. If MD is somewhat confident ( ≥ 0.2 confidence) that a human or vehicle is present, and the highest - confidence classifier prediction agrees ( ≥ 0.5 for humans, ≥ 0.7 for vehicles), predict the corresponding class ( human or vehicle ).
3. Use the classifier's top prediction if it is not blank.
4. If MD has a high confidence score for an object of any category ( ≥ 0.8), while the classifier's top class is blank …
- a. If the classifier's blank prediction confidence is ≤ 0.5, use the classifier's second - most - confident class.
- b. Otherwise, predict the coarse - level animal class.
5. If no other rule applies, predict blank.

TABLE 5 Training datasets and their corresponding test datasets.

| Training dataset          | Test dataset        | Motivation                                                                        |
|---------------------------|---------------------|-----------------------------------------------------------------------------------|
| WI (train) þ LILA (train) | WI (test)           | Train the best model possible for the Wildlife Insights platform                  |
| LILA (train)              | LILA (test)         | Train the best possible model that can be externally reproduced                   |
| iWildCam (train)          | iWildCam (ID test)  | Evaluate performance on a standardised, reproducible, within - location benchmark |
| iWildCam (train)          | iWildCam (OOD test) | Evaluate performance on a standardised, reproducible, generalisability benchmark  |

## 2.3.5 | Two - classifier ensemble

In the previous ensemble strategy, MD was used as a 'backup classifier'; no cropping was performed. An alternative ensembling strategy uses both the whole - image classifier and the crop classifier to get two species - level predictions, then ensembles those results together to determine the final image - level prediction. In Section 4, we will refer to this ensemble strategy as 'whole - image classifier þ crop classifier'.

In this ensembling strategy, the final predicted class is determined according to the following procedure (these steps are run in order):

1. If both classifiers' top prediction is a non - blank class, select the prediction with the higher confidence.
2. If one classifier predicted a non - blank class with confidence ≥ 0.5, while the other predicted blank , select the non - blank class.
3. Otherwise, predict blank.

## 3 | EXPERIMENTS AND METRICS

## 3.1 | Experiments

Whole - image and crop classifiers were trained according to Section 2.3 on three training datasets: WI þ LILA, LILA, and iWildCam - train, and evaluated on the corresponding test set; as per the iWildCam benchmark procedure, iWildCam - train is evaluated on both the iWildCam - ID (in domain) and iWildCam - OOD (out of domain) test sets. The relationships between training and test sets are summarised in Table 5, along with a concise summary of the reason each train/test pair was included.

For each test dataset, we evaluated four AI architectures:

1. Whole - image classification alone
2. Whole - image classifier þ MD
3. Crop classifier
4. Whole - image classifier þ crop classifier

For each dataset and each AI architecture, we provide test results on both filtered and unfiltered test datasets (Section 2.2), with the exception that because iWildCam is a standardised benchmark test set, we do not report filtered results on iWildCam ID or iWildCam OOD.

## 3.2 | Metrics

For each combination of (architecture, test set), we report four metrics:

1. Overall weighted F1 : the average F1 score for all classes in the test set, including the blank class, weighted by the frequency of each class in the test set being evaluated.
2. Species-level weighted F1 : the average F1 score across all animal classes in the test set (i.e. excluding the blank , human , and vehicle classes), weighted by the frequency of each class, considering only species - level predictions when computing precision, and species - level labels when computing recall. The overall weighted F1 is disproportionately influenced by the blank class (because it is so common); we include species-level weighted F1 to verify that trends in our results are not indicative only of performance on the blank class. The restriction to species - level predictions/labels is conservative (i.e. underestimates true performance), but simplifies evaluation.
3. Overall macro-averaged F1 : the average per - class F1 score. Although the species - level weighted F1 is not influenced by the blank class, and reasonably reflects the accuracy a user would experience on non - blank images, it is disproportionately impacted by common animals, and does not capture the full range of accuracy on our long - tailed distribution. The overall macro-averaged F1 metric reflects performance across the full range of species.
4. Blank F1 : as per above, many AI - accelerated workflows will still require that humans at least quickly review all non - blank images, so the assignment of a blank prediction on a non - blank image is particularly costly. The blank F1 metric allows us to assess accuracy on the single most important class, and to assess whether advantages in species - level accuracy for one model over another may come at the expense of more missed animals.

A non - blank prediction is considered correct only if the prediction precisely matches the ground truth label. Our ensembling logic allows prediction of a general animal class; this is important for real workflows, but we treat this as an incorrect prediction, just as if an incorrect species had been

-

predicted. Similarly, we have labels at multiple taxonomic levels; a prediction at a taxonomic level other than the one in the ground truth is considered wrong for purposes of evaluation, that is, if the ground truth label is 'bird', and a model predicts 'song sparrow', we treat that as an incorrect prediction, and vice - versa. These approaches are conservative; that is, they will in many cases penalise what is a correct prediction. However, variable - taxonomic - level labelling is a common property of camera trap data, and any approach to evaluation will either require some heuristics (such as the conservative heuristics we chose here) or will limit evaluation to unrealistic scenarios (such as discarding all labels other than species - level labels). In Section 6, we discuss the need for the community to converge on a standardised way of handling this issue.

## 4 | RESULTS AND DISCUSSION

## 4.1 | Primary experiments

Table 6 presents the results for our primary experiments. In this section, we briefly highlight the key takeaways from Table 6; the implications of these results will be discussed in more Section 5.

1. Detection helps ; there is not a single metric for any of our experiments where the highest score is achieved using a whole - image classifier alone.
2. Cropping helps, above and beyond the benefit of detection ; there is not a single metric for any of our experiments where the highest score is achieved using a whole - image classifier ensemble with a detector, without crop classification.
3. Combining a whole - image classifier with a crop classifier may not provide any benefit above and beyond the crop classifier ; the results for the crop classifier and (whole - image classifier þ crop classifier) cases are quite similar in all experiments. Each of those two options slightly outperforms the other on some experiments.

## 4.2 | Absolute accuracy

The primary goal of our experiments was to evaluate the benefits of using a detector as part of a classification pipeline, rather than to assess the absolute accuracy of any individual classifier or hyperparameter set. However, the availability of iWildCam as a public benchmark and leaderboard facilitates an absolute comparison, so we also compare our results to the top - performing models on the iWildCam leaderboard (https:// wilds.stanford.edu/leaderboard/#iwildcam) as of the time of this writing (March 2024) in Table 7. A discussion of the absolute utility of any individual model or hyperparameter set (e.g. the time it would save an ecologist per image) is outside the scope of this paper, but the fact that our results correspond to

-

TABLE 6 Results for all of the experimental scenarios described in Section 3.

|              |                                            | Unfiltered                | Unfiltered   | Unfiltered                        | Unfiltered             | Filtered                  | Filtered   | Filtered                          | Filtered               |
|--------------|--------------------------------------------|---------------------------|--------------|-----------------------------------|------------------------|---------------------------|------------|-----------------------------------|------------------------|
| Test dataset | Method                                     | Overall weighted - avg F1 | Blank F1     | Species - level weighted - avg F1 | Overall macro - avg F1 | Overall weighted - avg F1 | Blank F1   | Species - level weighted - avg F1 | Overall macro - avg F1 |
| LILA         | Whole - image classifier                   | 0.899                     | 0.954        | 0.825                             | 0.455                  | 0.951                     | 0.986      | 0.885                             | 0.509                  |
| LILA         | Whole - image classifier þ MD              | 0.902                     | 0.955        | 0.825                             | 0.458                  | 0.952                     | 0.988      | 0.885                             | 0.510                  |
| LILA         | Crop classifier                            | 0.914                     | 0.957        | 0.862                             | 0.576                  | 0.968                     | 0.992      | 0.927                             | 0.644                  |
| LILA         | Whole - image classifier þ crop classifier | 0.918                     | 0.959        | 0.871                             | 0.566                  | 0.969                     | 0.992      | 0.930                             | 0.633                  |
| WI           | Whole - image classifier                   | 0.795                     | 0.872        | 0.784                             | 0.204                  | 0.868                     | 0.963      | 0.845                             | 0.225                  |
| WI           | Whole - image classifier þ MD              | 0.796                     | 0.876        | 0.784                             | 0.204                  | 0.871                     | 0.971      | 0.845                             | 0.225                  |
| WI           | Crop classifier                            | 0.823                     | 0.881        | 0.832                             | 0.254                  | 0.900                     | 0.980      | 0.893                             | 0.280                  |
| WI           | Whole - image classifier þ crop classifier | 0.827                     | 0.883        | 0.832                             | 0.250                  | 0.901                     | 0.978      | 0.893                             | 0.275                  |
| iWildCam OOD | Whole - image classifier                   | 0.813                     | 0.889        | 0.790                             | 0.490                  |                           |            |                                   |                        |
| iWildCam OOD | Whole - image classifier þ MD              | 0.815                     | 0.893        | 0.790                             | 0.489                  |                           |            |                                   |                        |
| iWildCam OOD | Crop classifier                            | 0.817                     | 0.912        | 0.789                             | 0.519                  |                           |            |                                   |                        |
| iWildCam OOD | Whole - image classifier þ crop classifier | 0.826                     | 0.898        | 0.807                             | 0.542                  |                           |            |                                   |                        |
| iWildCam ID  | Whole - image classifier                   | 0.796                     | 0.892        | 0.781                             | 0.597                  |                           |            |                                   |                        |
| iWildCam ID  | Whole - image classifier þ MD              | 0.797                     | 0.897        | 0.781                             | 0.598                  |                           |            |                                   |                        |
| iWildCam ID  | Crop classifier                            | 0.810                     | 0.905        | 0.793                             | 0.619                  |                           |            |                                   |                        |
| iWildCam ID  | Whole - image classifier þ crop classifier | 0.822                     | 0.906        | 0.808                             | 0.619                  |                           |            |                                   |                        |

Note : Within each metric for each experiment, the highest score is in bold. In some cases, apparent 'ties' were broken by decimal places not shown here; we limit bolding to one cell per metric for readability. Columns 3-6 show results for unfiltered (i.e., noisy) test data (Section 2.2); columns 7-10 show results for filtered (i.e., reliable, but possibly biased by our filtering process) test data. Because iWildCam is a standard benchmark, no results are presented for filtered iWildCam data.

Abbreviation: LILA, Labelled Information Library of Alexandria.

the top of a standardised leaderboard provides evidence that the primary questions on which we are focused are being asked on state - of - the - art models.

We note that these results should be taken with some caution, as some of the data used in the iWildCam benchmark was also used to train MD, and our results depend on MD. Anecdotally, the intersection between iWildCam and MD's training data is small, and we don't expect that the effect of that intersection is significant, but for this reason, we are not making claims that our hyperparameters are 'better' than others used on the iWildCam leaderboard, only that we are confident that the models on which we are basing our experiments are in the same accuracy range as other state - of - the - art models.

TABLE 7 Comparison of the present results to the three best - performing models on the iWildCam leaderboard.

| Model                                                    | Leaderboard rank   |   Test ID macro F1 |   Test ID accuracy |   Test OOD macro F1 |   Test OOD accuracy |
|----------------------------------------------------------|--------------------|--------------------|--------------------|---------------------|---------------------|
| Present work: whole - image classifier þ crop classifier | n/a                |               61.9 |               82.8 |                54.2 |                81.9 |
| [16]                                                     | 1                  |               63.5 |               82.9 |                52.0 |                83.1 |
| [25]                                                     | 2                  |               59.9 |               76.2 |                46.0 |                76.2 |
| [57]                                                     | 3                  |               57.6 |               79.1 |                43.3 |                79.3 |

Note : The Test OOD Macro F1 column (highlighted) is used for leaderboard ranking.

## 4.3 | Data filtering

For brevity, we do not conduct a complete set of experiments to assess the benefit of the training data preparation procedures described in Section 2.2. However, we find that filtering camera trap training data with AI provides a substantial benefit, and we evaluate that benefit on the standardised iWildCam benchmark, in the simplest case of the whole - image classifier. For Test OOD Macro F1 (the primary metric used for the iWildCam leaderboard), in Table 6, we report a value of 0.490 for the whole - image classifier. This is based on filtered training data, but unfiltered test data. When we instead train the same model on unfiltered training data, Test OOD Macro F1 falls dramatically, from 0.490 to 0.378.

As per above, we note that this result should be taken with some caution, since some of the data used in the iWildCam benchmark was also used to train MD, and our data filtering is based largely on MD.

## 4.4 | Production considerations

All of the analyses we have presented in this section-in fact, all of the analyses we are aware of in the literature-have assumed that the 'best' model is the most accurate model. However, in a production workflow for camera trap image analysis, a user's efficiency and subjective satisfaction with the system depend on a number of factors beyond accuracy. A comprehensive evaluation of efficiency or user experience is outside the scope of this work, but in this section, we will introduce some of these concerns, and propose candidate mechanisms for addressing those concerns. All of these solutions involve tradeoffs between quantitative accuracy and expected user experience; in Section 6, we encourage the community to develop metrics and experimental protocols that allow these tradeoffs to be addressed more systematically.

## 4.4.1 | Trading 'unknown' predictions for precision

In the systems we have described so far, each image is assigned a label during inference; choosing not to assign any label at all for some images-that is, predicting unknown -complicates evaluation. However, anecdotally, we have observed that up to a

-

point, users respond more negatively to incorrect predictions than to unknown predictions. This is particularly true for the blank label: allowing users not to look at blank images at all is perhaps the most important advantage of AI systems for camera trap data, and even a small number of incorrect blank predictions may encourage users to review blank images, eliminating the benefits of AI. Consequently, production systems may choose not to output any predictions when an AI system's confidence is low; this will generally increase precision and reduce recall for all classes.

For example, without any unknown predictions, the crop classifier described above yields a blank precision of 0.81 and a species - level weighted average precision of 0.92 on the WI unfiltered test set. If we predict unknown for all images where the top prediction is blank but the confidence is &lt; 0.8, and all other images where the top (non - blank) prediction has a confidence &lt; 0.65, 12.1% of images are predicted as unknown , but blank precision and species - level weighted average precision increase to 0.84 and 0.95, respectively.

## 4.4.2 | Label rollup

Similarly, although an ideal system will always make predictions at the species level, an incorrect species - level prediction may yield a worse subjective experience than a correct prediction at a higher taxonomic level. For example, if an image contains a coyote, predicting the 'canidae' family may be preferable to predicting 'wolf'. We propose a 'label rollup' strategy that favours confident predictions at a higher taxonomic level over either unknown predictions or very - low - confidence predictions at the species level. This will necessarily reduce accuracy, but may improve user experience.

Specifically, when employing label rollup, if a model's confidence in its top prediction is &lt; 0.75, we aggregate scores to higher taxonomic levels by summing confidence values at higher taxonomic levels. For example, if the model predictions are (puma, mule deer, lion, leopard) with confidence values (0.6, 0.2, 0.1, 0.1), respectively, no prediction is above the confidence threshold of 0.75, so the model will not output a species - level prediction. In this case, if we look at the genus level, we are no more confident in any particular genus: puma, lion, and leopard are in different genera. But if we look at the family level, we see a total confidence of 0.8 for the 'felidae' family (cats), which is now above our threshold.

-

We find that when we employ this approach, in conjunction with the use of the unknown prediction described above, the unknown rate for the crop classifier on the WI unfiltered test set falls from 12.1% to 5.7%, and the aggregated recall for animals-that is, the likelihood that we predict some animal class when the true label is an animal class-increases from 75.0% to 85.0%.

## 4.4.3 | Geofencing

Finally, we have observed anecdotally that users are particularly frustrated by predictions of species that are not observed in the ecosystem where their data originates. For example, predicting an Australian species in North America has a disproportionately negative impact on user experience, even if that prediction is part of a strategy that maximises overall accuracy. We propose a 'geofencing' strategy that uses a regional 'allow - list', specifying all taxa that are expected in a region (e.g. a country). When geographic information is available for an image, and a model's top prediction is not on the allow - list for the location from which this image originates, we invoke the 'label rollup' strategy described above until we find a taxonomic level that is allowable. For example, if the model's top prediction is lion with a confidence of 0.8, but the image originates in North America (where lion is not on the allow - list), we would examine higher taxonomic levels, finding that felidae is on the allow - list for North America, and the model would predict felidae .

We find that when we employ this approach, the species - level macro - average precision for the WI unfiltered test set increases from 0.44 to 0.46. This is a small increase in accuracy, but it disproportionately benefits user experience, since the incorrect predictions that are suppressed by this approach are particularly jarring.

We note that this geofencing approach is distinct from the use of geographic priors in training [35]; this is a complementary strategy that may also be beneficial for large, geographically diverse camera trap dataset. We also note that both strategies involve complex tradeoffs with respect to allowing the prediction of invasive species.

## 5 | DISCUSSION

The primary conclusions from our experiments are that (a) incorporating an object detector provides a benefit over whole - image classification, and that (b) cropping images to detected animals prior to classification provides a benefit above and beyond the coarse - categorisation benefit that a detector provides. While it may seem intuitive that an object detector provides some benefit, given that a typical detector is a larger model that sees more pixels than a typical whole - image classifier, this is an important result: most of the literature in the field has focused on whole - image classification alone, and this result may help steer the field towards embracing object detection.

This is independent of the benefits that detectors provide by construct, which are outside of the scope of this paper, but useful to consider in designing new systems: detectors provide a natural way to handle multi - species images, for example, which is extremely challenging for whole - image classifiers. Multi - label classification is possible, but multiple - species images represent a very small fraction of camera trap data overall, so it would be difficult to train a practical multi - label whole - image classifier. Detectors also provide a natural mechanism for counting individual animals [52]; although it is possible to train whole - image models for counting [40], this requires a dedicated model beyond the model used for species classification, and is still subject to the impact of limited input size for whole - image models. Finally, in addition to minimising the risk of learning spurious correlations with natural backgrounds, cropped classification minimises the risk of learning spurious correlations with camera overlays, which as per above we have anecdotally observed to be problematic on very - long - tailed datasets, where a species may only be observed by a single camera model.

However, these results should not be taken to suggest that a whole - image classifier is not an important component of a camera trap AI pipeline in many scenarios. Object detectors typically have large input sizes because they are large models, which incurs larger inference time than a typical classifier: for example, the classification architecture that we use in this work is quite large (51M parameters), and it is still much smaller than the YOLOv5x6architecture that MD employs (140.7M parameters), and runs around twice as fast. Consequently, there are scenarios when whole - image classification alone may be preferred for efficiency. Furthermore, although the ensemble of a whole - image classifier with a crop classifier did not perform better than the crop classifier alone in our experiments, these results are averaged over a very large dataset, and there are numerous scenarios where MD is known to perform poorly on particular image types or particular taxa that are under - represented in its training data [12], and a whole - image classifier that is trained either on a larger dataset or for a particular domain may compensate for weaknesses in the detector.

## 6 | CONCLUSION, FUTURE WORK, AND COMMUNITY CALLS TO ACTION

Although we have demonstrated that incorporating an object detector into a camera trap image processing pipeline is beneficial, a comprehensive assessment of when it helps is left to future work. While it seems intuitive that the benefits are focused on small or distant animals, supporting that claim is beyond the scope of this work. Along the same lines, while intuition suggests that multi - species images trivially benefit from a detector stage, we did not explore multi - species images in this study. Similarly, some hyperparameter exploration related to object detection is deferred to future work; in particular, in our experiments, we cropped animals according to MD output with no padding, but it is anecdotally easier for humans to identify cropped animals when the detection box is expanded by a few pixels prior to cropping. Future work should assess whether this is also true for AI classifiers, and what the optimal cropping strategy is, including whether resizing to the classifier's input size (with aspect ratio distortion) outperforms padding (with aspect ratio preservation). Along the same lines, we assess the combination of a detector with an image classifier; future work should compare that to a single model that incorporates species classification into the object detector's classification head. We also encourage follow - up work that explores alternative architectures and hyperparameters: EfficientNetV2 - M was selected here to strike an application - specific balance between accuracy and inference cost, but architectures continue to evolve, and this is already quite a large architecture that may be more than what's required for regional classifiers with fewer species.

The detector employed here-MD version 5a-should also be viewed as a baseline; it was trained more than 3 years ago as of the time of this writing, using a commodity architecture (YOLOv5x6), on data with a limited geographic distribution. MD's utility is derived largely from its use of a large dataset of bounding boxes for training (~2.3M boxes), but this is not a large number in the context of global camera trapping, and could quickly be eclipsed by a collaborative annotation effort. If complementary work confirms that a detection stage is as beneficial as our results indicate, we encourage the community to contribute bounding - boxannotations to a shared repository that will enable the continued development of coarse - but - generalisable detectors.

Furthermore, the uniqueness of MD as a tool for asking these questions-coupled with the fact that the only available public leaderboard with a standardised benchmark overlaps with MD's training data-poses a challenge for evaluation. For example, we highlight above that all of our comparisons to the public iWildCam leaderboard should be treated cautiously, since there is some possibility that our results are inflated by the overlap between the iWildCam benchmark and MD's training data. Therefore, if complementary work confirms that a detection stage is as beneficial as our results indicate, we encourage the community to develop (a) analogous benchmarks for object detection accuracy and (b) an iWildCam - like classification benchmark on previously - unseen data.

Finally, as we discussed in Section 2.1, we used very conservative heuristics to accommodate the fact that our data includes labels at multiple taxonomic levels (for example, 'bird', 'owl', and 'great horned owl' are all valid labels). Our evaluation likely penalised numerous correct predictions, for example, cases where our model correctly predicted 'great horned owl' but the ground truth label was either 'owl' or 'bird'. Excluding labels other than species - level labels would simplify evaluation, but would create an unrealistic evaluation distribution. We encourage the community to develop a standardised metric for evaluating datasets that include labels at multiple taxonomic levels, and to create an iWildCam - like public benchmark that incorporates this metric. Similarly, we discussed user experience tradeoffs between risking an incorrect prediction and reporting 'unknown'; we encourage the community to develop metrics that integrate these user experience concerns into model

-

evaluation, and to conduct human factors experiments that assess both efficiency gains and subject satisfaction in the context of AI systems for camera traps.

## AUTHOR CONTRIBUTIONS

Tomer Gadot : Conceptualisation; Data curation; Investigation; Methodology; Software; Writing - original draft; Writing - review &amp; editing. S ¸ tefan Istrate : Conceptualisation; Data curation; Investigation; Methodology; Software; Writing - original draft; Writing - review &amp; editing. Hyungwon Kim : Conceptualisation; Investigation; Methodology; Software. Dan Morris : Conceptualisation; Methodology; Writing - original draft; Writing - review &amp; editing. Sara Beery : Conceptualisation; Methodology; Writing - original draft. Tanya Birch : Conceptualisation; Methodology; Project administration; Writing - original draft. Jorge Ahumada : Conceptualisation; Data curation; Methodology; Project administration.

## ACKNOWLEDGEMENTS

We would like to acknowledge the Partners of Wildlife Insights for collaboration on the WI platform, including Roland Kays, NC Museum of Natural Sciences; William McShea, Smithsonian Conservation Biology Institute; Walter Jetz, Yale Univ; Jonathan Palmer, WCS, Maggie Kinnaird, WWF; Alison Swanson and Nicole Flores, Conservation International; Anthony Dancer, ZSL; Martin Wikelski, Max Planck Institute of Animal Behavior; Chrissy Durkin, WildMon; as well as the users of the Wildlife Insights platform who licence their data to support improving computer vision models. We thank Siyu Yang for her work developing the MegaDetector model. We also acknowledge sponsors of the Wildlife Insights platform: Lyda Hill Foundation, Gordon and Betty Moore Foundation, and founding technology partner Google. We would like to thank the team who worked on the Wildlife Insights AI system: Burcu Karagol Ayan, Keyvan Azami, Michael Brooks, Sarabeth Craig, Sarah Dudley, Shivaji Dutta, Eric Fegraus, Ryan Ffrench, Logan Goldberg, Matt Hancher, Jacqueline Hea, Jonathan Huang, Zach Hynes, Camilla Ibrahim, Simon Ilyushchenko, Charbel Kaed, Sayali Kulkarni, Juan Liang, Chen Luo, Maggie Lynn, Pravar Mahajan, Rebecca Moore, Anish Nangia, Alison Reichenthal, Dusty Reid, Scott Riddle, Brian Rutkowski, Sophie Su, Karin Tuxen - Bettman, Karen Wang, and Leon Zhou.

## CONFLICT OF INTEREST STATEMENT

The authors declare no conflicts of interest.

## DATA AVAILABILITY STATEMENT

Methods are benchmarked against the public iWildCam dataset and a novel set of splits based on the public LILA repository; our splits are provided as supplementary material.

## ORCID

[S ¸tefan Istrate https://orcid.org/0000-0003-0695-4174](https://orcid.org/0000-0003-0695-4174)

Dan Morris https://orcid.org/0000-0003-0264-3504

[Tanya Birch https://orcid.org/0000-0002-6504-9409](https://orcid.org/0000-0002-6504-9409)

17519640, 2024, 8, Downloaded from https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/cvi2.12318, Wiley Online Library on [09/09/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License

-

## REFERENCES

1. Aguirre, I., Hood, G.A., Westbrook, C.J.: Short - term dynamics of beaver dam flow states. Sci. Total Environ. 170825, (2024 Feb 9). https://doi. org/10.1016/j.scitotenv.2024.170825
2. Ahumada, J.A., et al.: Wildlife insights: a platform to maximize the potential of camera trap and other passive sensor wildlife data for the planet. Environ. Conserv. 47(1), 1-6 (2020 Mar). https://doi.org/10. 1017/s0376892919000298
3. Anderson, A.K., Waller, J.S., Thornton, D.H.: Canada lynx occupancy and density in glacier national park. J. Wildl. Manag. (2023) May;87(4): e22383, https://doi.org/10.1002/jwmg.22383
4. Anderson, A.K., Waller, J.S., Thornton, D.H.: Partial COVID - 19 closure of a national park reveals negative influence of low - impact recreation on wildlife spatiotemporal ecology. Sci. Rep. 13(1), 687 (2023 Jan 13). https://doi.org/10.1038/s41598 - 023 - 27670 - 9
5. Ausband, D.E., et al.: Examining dynamic occupancy of gray wolves in Idaho after a decade of managed harvest. J. Wildl. Manag. 87(6), e22453 (2023). https://doi.org/10.1002/jwmg.22453
6. Ayars, J., et al.: Camera traps link population - level activity patterns with wildfire smoke events for mammals in Eastern Washington State. Fire Ecol. 19(1), 1-5 (2023 Dec). https://doi.org/10.1186/s42408 - 023 - 00207 - 1
7. Barash, A., et al.: Possible origins and implications of atypical morphologies and domestication - like traits in wild golden jackals (Canis aureus). Sci. Rep. 13(1), 7388 (2023). https://doi.org/10.1038/s41598 - 023 - 34533 - w
8. Bassing, S.B., et al.: Are we telling the same story? Comparing inferences made from camera trap and telemetry data for wildlife monitoring. Ecol. Appl. (2023) Jan;33(1):e2745, https://doi.org/10. 1002/eap.2745
9. Beery, S., Van Horn, G., Perona, P.: Recognition in terra incognita. In: Proceedings of the European Conference on Computer Vision (ECCV), pp. 456-473 (2018)
10. Beery, S., et al.: Synthetic examples improve generalization for rare classes. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 863-873 (2020)
11. Beery, S., et al.: Context r - cnn: long term temporal context for per - camera object detection. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13075-13085 (2020)
12. Beery, S., Morris, D., Yang, S.: Efficient Pipeline for Camera Trap Image Review [Computer Software]. (2024). http://github.com/agentmorris/ MegaDetector. Accessed 3/2024
13. Bothmann, L., et al. Automated Wildlife Image Classification: An Active Learning Tool for Ecological Applications. arXiv preprint arXiv: 2303.15823. (2023 Mar 28)
14. Boyce, P.: Feral Horse Ecology in the Rocky Mountain Foothills of Alberta. Canada (Doctoral dissertation, University of Saskatchewan) (2022)
15. Cabon, V., et al.: Endangered animals and plants are positively or neutrally related to wild boar (Sus scrofa) soil disturbance in urban grasslands. Sci. Rep. 12(1):1 - 0 (2022 Oct 5). https://doi.org/10.1038/ s41598 - 022 - 20964 - 4
16. Choi, C., et al. AutoFT: Robust Fine - Tuning by Optimizing Hyperparameters on OOD Data. arXiv preprint arXiv:2401.10220. (2024 Jan 18)
17. Cubuk, E.D., et al.: Randaugment: practical automated data augmentation with a reduced search space. In: 2020 IEEE. InCVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pp. 3008-3017 (2019)
18. Cunha, F., dos Santos, E.M., Colonna, J.G.: Bag of tricks for long - tail visual recognition of animal species in camera - trap images. Ecol. Inf. 76:102060 (2023 Sep 1). https://doi.org/10.1016/j.ecoinf.2023.102060
19. De Lorm, T., et al.: Optimising the automated recognition of individual animals to support population monitoring. Ecol. Evol., 13(7) (2023 Jun 28). https://doi.org/10.1002/ece3.10260
20. Dussert, G., et al.: Beyond Accuracy: Score Calibration in Deep Learning Models for Camera Trap Image Sequences. bioRxiv. (2023: 2023 - 11)
21. Evans, B.C., et al.: Reasoning about neural network activations: an application in spatial animal behaviour from camera trap classifications. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases, pp. 26-37. Springer (2020 Sep 14)
22. Fennell, M.J., et al.: Assessing the impacts of recreation on the spatial and temporal activity of mammals in an isolated alpine protected area. Ecol. Evol. (2023 Nov);13(11):e10733, https://doi.org/10.1002/ece3.10733
23. Fennell, M.J.: Multispecies Mammal Monitoring in Cathedral Provincial Park. Doctoral dissertation, University of British, Columbia (2022)
24. Gao, I., et al.: Out - of - Distribution robustness via targeted augmentations. In: NeurIPS 2022 Workshop on Distribution Shifts: Connecting Methods and Applications (2022)
25. Goyal, S., et al.: Finetune like you pretrain: improved finetuning of zero - shot vision models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 19338-19347 (2023)
26. Goward, S.I.: Spy through a camera's eye: Divii in the Gwich'in settlement area. Arctic 75(4), 510-515 (2022 Dec 15). https://doi.org/10. 14430/arctic76639
27. Henrich, M., et al.: A semi - automated camera trap distance sampling approach for population density estimation. Rem. Sens. Ecol. Conserv. 10(2), 156-171 (2023). https://doi.org/10.1002/rse2.362
28. He, W., et al.: Long - tailed metrics and object detection in camera trap datasets. Appl. Sci. 13(10), 6029 (2023 May 14). https://doi.org/10. 3390/app13106029
29. Hewitt, M.O., et al.: Modeling habitat use and potential distribution of kit fox in the Trans - Pecos, Texas. J. Wildl. Manag. 86(8), e22303 (2022 Nov). https://doi.org/10.1002/jwmg.22303
30. Jia, L., Tian, Y., Zhang, J.: Neural architecture search based on packed samples for identifying animals in camera trap images. Neural Comput. Appl. (14), 1-23 (2023 Jan 29). https://doi.org/10.1007/s00521 - 023 - 08247 - z
31. Jocher, G., et al.: ultralytics/yolov5: v3. 0. Zenodo. (2020 Aug)
32. Johansson, Ö., et al.: Identification errors in camera - trap studies result in systematic population overestimation. Sci. Rep. 10(1), 6393 (2020 Apr 14). https://doi.org/10.1038/s41598 - 020 - 63367 - z
33. Koh, P.W., et al.: Wilds: a benchmark of in - the - wild distribution shifts. In: International Conference on Machine Learning, pp. 5637-5664. PMLR (2021, July)
34. Leorna, S., Brinkman, T.: Human vs. machine: detecting wildlife in camera trap images. Ecol. Inf. 72, 101876 (2022 Oct 27). https://doi.org/10. 1016/j.ecoinf.2022.101876
35. Mac Aodha, O., Cole, E., Perona, P.: Presence - only geographical priors for fine - grained image classification. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 9596-9606 (2019)
36. Maile, R.E., Duggan, M.T., Mousseau, T.A.: The successes and pitfalls: deep - learning effectiveness in a Chernobyl field camera trap application. Ecol. Evol. 13(9), e10454 (2023 Sep). https://doi.org/10.1002/ece3. 10454
37. Munguía - Carrara, M., et al.: Comparison of biomass of exotic and native mammals between temperate and tropical forests of Mexico. In Mexican Fauna in the Anthropocene, pp. 515-525. Cham: Springer International Publishing (2023 Jan 31)s
38. Muthoka, J.M., et al.: Assessing drivers of intra - seasonal grassland dynamics in a Kenyan savannah using digital repeat photography. Ecol. Indicat. 142, 109223 (2022 Sep 1). https://doi.org/10.1016/j.ecolind. 2022.109223
39. Norman, D.L., et al.: Can CNN - based species classification generalise across variation in habitat within a camera trap survey? Methods Ecol. Evol. 14(1), 242-251 (2023 Jan). https://doi.org/10.1111/2041 - 210x. 14031
40. Norouzzadeh, M.S., et al.: Automatically identifying, counting, and describing wild animals in camera - trap images with deep learning. Proc. Natl. Acad. Sci. USA 115(25), E5716-E5725 (2018 Jun 19). https://doi. org/10.1073/pnas.1719367115
41. Norouzzadeh, M.S., et al.: A deep active learning system for species identification and counting in camera trap images. Methods Ecol. Evol. 12(1), 150-161 (2021 Jan). https://doi.org/10.1111/2041 - 210x.13504

42. Nouzille, C.M.: Mammal Recolonization and Recovery Following the Woolsey Fire. MSc thesis. University of California, Los Angeles (2022)
43. Procko, M., et al.: Human presence and infrastructure impact wildlife nocturnality differently across an assemblage of mammalian species. PLoS One 18(5), e0286131 (2023). https://doi.org/10.1371/journal. pone.0286131
44. Schneider, S., et al.: Three critical factors affecting automated image species recognition performance for camera traps. Ecol. Evol. 10(7), 3503-3517 (2020 Apr). https://doi.org/10.1002/ece3.6147
45. Simões, F., et al.: DeepWILD: Wildlife Identification, Localisation and estimation on camera trap videos using Deep learning. Ecol. Inf. 75, 102095 (2023 Jul 1). https://doi.org/10.1016/j.ecoinf.2023.102095
46. Suessle, V., et al.: Automatic Individual Identification of Patterned Solitary Species Based on Unlabeled Video Data. arXiv preprint arXiv: 2304.09657. (2023 Apr 19)
47. Tabak, M.A., et al.: CameraTrapDetectoR: Automatically Detect, Classify, and Count Animals in Camera Trap Images Using Artificial Intelligence. bioRxiv. (2022 Jan 1)
48. Tabak, M.A., et al.: Improving the accessibility and transferability of machine learning algorithms for identification of animals in camera trap images: MLWIC2. Ecol. Evol., 10, (19), 10374-10383 (September 2020) https://doi.org/10.1002/ece3.6692
49. Tan, M., Le, Q.: Efficientnetv2: smaller models and faster training. In International Conference on Machine Learning, pp. 10096-10106. PMLR (2021 Jul 1)
50. Thompson, S., et al.: Camera - Based Estimation of Statewide Wolf Abundance in Idaho - 2019-2021. Interim Report, 2/28/22 (2022)
51. Vélez, J., et al.: An evaluation of platforms for processing camera - trap data using artificial intelligence. Methods Ecol. Evol. 14(2), 459-477 (2023 Feb). https://doi.org/10.1111/2041 - 210x.14044
52. Wang, Y., et al.: Deep learning methods for animal counting in camera trap images. In: 2022 IEEE 34th International Conference on Tools with Artificial Intelligence (ICTAI), pp. 939-943. IEEE (2022 Oct 31)
53. Weinstein, B.G.: A computer vision for animal ecology. J. Anim. Ecol. 87(3), 533-545 (2018 May). https://doi.org/10.1111/1365 - 2656.12780
54. Westworth, S.O., et al.: Understanding external influences on target detectionandclassificationusingcameratrapimagesandmachinelearning. Sensors 22(14), 5386 (2022 Jul 19). https://doi.org/10.3390/s22145386
55. WildEye: MegaDetector Version 5 evaluation. https://wildeyeco nservation.org/megadetector - version - 5/ (2022). Accessed March 2024
56. Willi, M., et al.: Identifying animal species in camera trap images using deep learning and citizen science. Methods Ecol. Evol. 10(1), 80-91 (2019 Jan). https://doi.org/10.1111/2041 - 210x.13099
57. Wortsman, M., et al.: Model soups: averaging weights of multiple fine - tuned models improves accuracy without increasing inference time. In: International Conference on Machine Learning, pp. 23965-23998. PMLR (2022, June)
58. Yang, D.Q., et al.: An automatic method for removing empty camera trap images using ensemble learning. Ecol. Evol. 11(12), 7591-7601 (2021 May 2). https://doi.org/10.1002/ece3.7591
59. Zett, T., Stratford, K.J., Weise, F.J.: Inter - observer variance and agreement of wildlife information extracted from camera trap images. Biodivers. Conserv. 31(12), 3019-3037 (2022 Oct). https://doi.org/10. 1007/s10531 - 022 - 02472 - z

## WILDLIFE INSIGHTS DATA REFERENCES

60. WWF - Hong Kong: Kadoorie Farm and Botanic Garden. Incubator Project - Mammals (2022). Last updated Feb 2022 http://n2t.net/ark:/ 63614/w12003827
61. One Tam: Tam (2021). Last updated Feb 2022 http://n2t.net/ark:/ 63614/w12001631
62. Iannarilli, F.: Snapshot Europe 2022 (2022). Last updated Sep 2022 http://n2t.net/ark:/63614/w12004446
63. Will, D.: Mona Island Restoration Project (2021). Last updated Sep 2021 http://n2t.net/ark:/63614/w12001652
64. Rich, N.L., et al.: CEMAP Medium - Large Mammals (2021). Last updated Jun 2023 http://n2t.net/ark:/63614/w12001650

-

65. Ayala, G.: Ocelot Survey Hondo River Jan - Mar 2005 (2021). Last updated Sep 2021 http://n2t.net/ark:/63614/w12000258
66. Nelson, L.D.: Swift Fox Pilot (2021). Last updated Jan 2022 http://n2t. net/ark:/63614/w12002979
67. Bonjorne, L.: Floresta Nacional de Ipanema (2022). Last updated Dec 2022 http://n2t.net/ark:/63614/w12004127
68. Fletcher, C., Campos - Arceiz, A.: Pasoh Forest Reserve (2021). Last updated Apr 2022 http://n2t.net/ark:/63614/w12002650
69. ICMBio/CENAP: Rio São Benedito (2021). Last updated Jul 2022 http://n2t.net/ark:/63614/w12002563
70. Mena, J.L.: Fauna del Santuario Nacional Tabaconas Namballe (2021). Last updated Sep 2021 http://n2t.net/ark:/63614/w12003071
71. Kays, R.: Kasanka Mammal Surveys (2021). Last updated Sep 2022 http://n2t.net/ark:/63614/w12003452
72. Nelson, D., Shamon, H., Jachowski, D.: Swift Fox Scat 2022 (2022). Last updated Dec 2022 http://n2t.net/ark:/63614/w12004526
73. Green, A.: Wasatch Wildlife Watch - Community Science Project (2021). Last updated Feb 2022 http://n2t.net/ark:/63614/w12003225
74. Naidoo, R.: WWF US: South Chilcotins Wildlife Survey (2021). Last updated Sep 2022 http://n2t.net/ark:/63614/w12000286
75. Farris, Z.: Makira NP Carnivore Survey 2009: Lokaitra Forest (2021). Last updated Sep 2021 http://n2t.net/ark:/63614/w12000276
76. Rosa, F.L.: Mecanismos de conservación para asegurar la sostenibilidad de las poblaciones del Oso andino y Tapir de montaña en los Andes del norte y centro del Perú (2022). Last updated Sep 2022 http://n2t.net/ ark:/63614/w12004209
77. Marthy, W.: Bukit Barisan (2021). Last updated Sep 2021 http://n2t. net/ark:/63614/w12003026
78. Programa Monitora ICMBio: Parque Nacional Do Mapinguari (2021). Last updated Feb 2022 http://n2t.net/ark:/63614/w12003207
79. None: WI\_Forest\_Chequamegon - Nicolet\_20 (2021). Last updated Jan 2022 http://n2t.net/ark:/63614/w12002685
80. Cook, V.: Nutria Eradication - Private Lands (2021). Last updated Jan 2023 http://n2t.net/ark:/63614/w12003570
81. Farris, Z.: Makira NP Carnivore Survey 2008: Anjanaharibe Forest (2021). Last updated Sep 2021 http://n2t.net/ark:/63614/w12000273
82. Vasquez, R.: Yanachaga Chimillén National Park (2021). Last updated Apr 2022 http://n2t.net/ark:/63614/w12003019
83. Tucker, J.: R5 Sierra Nevada Carnivore Monitoring Program (2022). Last updated Sep 2022 http://n2t.net/ark:/63614/w12004223
84. Kayijamahe, C., Uzabaho, E.: Virunga Massif (2021). Last updated Mar 2023 http://n2t.net/ark:/63614/w12002655
85. Furnas, B.: Carson River 2018 (2021). Last updated Mar 2023 http:// n2t.net/ark:/63614/w12002828
86. Wallace, R.: Rio Heath Survey (2021). Last updated Sep 2021 http://n2t. net/ark:/63614/w12000262
87. Hatfield, B.: Sierra Nevada Alpine Mesocarnivores (2021). Last updated Sep 2022 http://n2t.net/ark:/63614/w12002818
88. USDA Forest Service: Pacific Southwest region, carnivore monitoring program. In: 2013 Marten Hair Snares (2021). Last updated Dec 2022 http://n2t.net/ark:/63614/w12000517
89. None: RI\_Forest\_University\_of\_Rhode\_Island\_20 (2021). Last updated Jan 2022 http://n2t.net/ark:/63614/w12002684
90. Akins, J.: Cascade Red Fox and Wolverine Study Project (2022). Last updated Jan 2022 http://n2t.net/ark:/63614/w12004591
91. Kays, R.: Snapshot USA 2021 (2021). Last updated Jan 2023 http://n2t. net/ark:/63614/w12003286
92. Will, D.: Ulithi Atoll Restoration Project (2021). Last updated Sep 2021 http://n2t.net/ark:/63614/w12001655
93. Maitz, N.: Brush - Tailed Rock Wallabies in SEQ (2023). Last updated Feb 2023 http://n2t.net/ark:/63614/w12004931
94. Spencer, E.: Quokka Recovery in Northcliffe Fire Zone (2021). Last updated Jun 2022 http://n2t.net/ark:/63614/w12001410
95. One Tam: Wood (2021). Last updated Feb 2022 http://n2t.net/ark:/ 63614/w12001634
96. Farris, Z.: Makira NP Carnivore Survey 2010: Lohan'sanjinja Forest (2021). Last updated Sep 2021 http://n2t.net/ark:/63614/w12000277

-

97. Stutzman, J.: Riparian Networks in a Northern Great Plains Ecosystem (2021). Last updated Feb 2023 http://n2t.net/ark:/63614/w12003608
98. Alvarez, P.: Cocha Cashu - Manu National Park (2021). Last updated Apr 2022 http://n2t.net/ark:/63614/w12003037
99. Will, D.: Palau Islands Restoration Project (2021). Last updated Sep 2021 http://n2t.net/ark:/63614/w12001654
100. WWF - Ecuador: Monitoreo de oso (2021). Last updated Sep 2021 http://n2t.net/ark:/63614/w12002681
101. Lima, M., Santos, F.: TEAM - Floresta Nacional de Caxiuanã (2021). Last updated Jan 2023 http://n2t.net/ark:/63614/w12002641
102. Farris, Z.: Makira NP Carnivore Survey 2011: Anjanaharibe Forest (2021). Last updated Sep 2021 http://n2t.net/ark:/63614/w12000278
103. Jimenez, I.: Seguimiento Lince WWF España (2021). Last updated Jun 2023 http://n2t.net/ark:/63614/w12001260
104. Nguyen Duc, T., et al.: Trung Khanh: Musk Deer &amp; Bears 2021 - 2022 (2022). Last updated Dec 2022 http://n2t.net/ark:/63614/ w12003996
105. Wallace, R.: Tuichi TCO San Jose (2021). Last updated Sep 2021 http:// n2t.net/ark:/63614/w12000266
106. Carvalho, E.A.R.Jr., Nienow, S.S., Bonavigo, P.H.: Floresta Nacional Do Jamari (2021). Last updated Sep 2022 http://n2t.net/ark:/63614/ w12002562
107. Programa Monitora ICMBio: Parque Nacional Do Juruena (2021). Last updated Feb 2022 http://n2t.net/ark:/63614/w12002576
108. ENTRMPA: ENTMRPA Wildlife Monitoring (2021). Last updated Feb 2022 http://n2t.net/ark:/63614/w12001356
109. Cabrera, J.A.: Corredor Guaviare Meta 2022 (2023). Last updated Feb 2023 http://n2t.net/ark:/63614/w12004972
110. Carvalho, E.A.R.Jr., et al.: Estação Ecológica da Terra do Meio (2021). Last updated Feb 2022 http://n2t.net/ark:/63614/w12002554

111.

Downes, S., Oates, B., Meyer, W .: Teanaway Wildlife Cameras (2023).

[Last updated Jan 2024 http://n2t.net/ark:/63614/w12004869](http://n2t.net/ark:/63614/w12004869)

112. Sink, C.: Clear Lake Predators (2021). Last updated Sep 2022 http:// n2t.net/ark:/63614/w12002779
113. Casady, D.: Marin County Deer Abundance 2015 - 16 (MCDA 2015 - 16) (2021). Last updated Mar 2023 http://n2t.net/ark:/63614/w12003054
114. Eustrate, U., Akampurira, E.: Volcanoes - Bwindi Ecosystems (2021). Last updated Jan 2022 http://n2t.net/ark:/63614/w12003219
115. Levin, M.: Solar Array Cameras (2022). Last updated Dec 2022 http:// n2t.net/ark:/63614/w12004578
116. Le Khac, Q., et al.: Phong Nha - Ke Bang NP (2022). Last updated Jan 2023 http://n2t.net/ark:/63614/w12004117
117. Wallace, R.: Rio Hondo 2012 Survey (2021). Last updated Sep 2021 http://n2t.net/ark:/63614/w12000264
118. Giumelli, P.: Royal National Park (2023). Last updated Mar 2023 http:// n2t.net/ark:/63614/w12004988
119. Will, D.: Juan Fernandez Islands Restoration Project (2021). Last updated Sep 2021 http://n2t.net/ark:/63614/w12001656
120. Stutzman, J.: GPS Riparian Mammals (2021). Last updated Dec 2022 http://n2t.net/ark:/63614/w12003074
121. Farris, Z.: Makira NP Carnivore Survey 2009: Vinanibe Forest (2021). Last updated Sep 2021 http://n2t.net/ark:/63614/w12003109
122. Bonjorne, L.: Veadeiros (2022). Last updated May 2023 http://n2t.net/ ark:/63614/w12004795
123. Wallace, R.: Rio Tuichi Survey (2021). Last updated Sep 2021 http:// n2t.net/ark:/63614/w12000265
124. Harms, J.: Velebit Mountains (2022). Last updated Jan 2022 http://n2t. net/ark:/63614/w12003886
125. Fleytas, M.: Uso de trampas - cámara para el monitoreo del jaguareté (Panthera onca) y los conflictos que amenazan su conservación (Using camera - traps to monitors jaguars) (2021). Last updated Sep 2021 http://n2t.net/ark:/63614/w12000283
126. One Tam: East (2021). Last updated Jan 2023 http://n2t.net/ark:/ 63614/w12001648
127. Furnas, B.: East Tehama Deer Abundance (ETDA 2015 - 2019) (2021). Last updated Mar 2023 http://n2t.net/ark:/63614/w12003058
128. Sernanp, W.W.F.: Monitoreo Oso Y Tapir\_TabaconasNamballe\_ WWFSERNANP (2022). Last updated Sep 2022 http://n2t.net/ark:/ 63614/w12003941
129. Espinosa, S., Salvador, J.: Yasuni (2021). Last updated Jan 2023 http:// n2t.net/ark:/63614/w12002654
130. Organization for Tropical Studies: Volcán Barva (2021). Last updated Apr 2022 http://n2t.net/ark:/63614/w12003020
131. Berlinck, C.N.: Levantamento e Monitoramento de Fauna na Parte Alta do PN da Serra da Canastra (2022). Last updated Jul 2023 http://n2t. net/ark:/63614/w12004102
132. Magioli, M.: REVIS Rio Dos Frades (2022). Last updated Sep 2022 http://n2t.net/ark:/63614/w12004349
133. Cabrera, J.: Corredor del Jaguar en los Municipios de el Retorno y San José (2022). Last updated Sep 2022 http://n2t.net/ark:/63614/w12003902
134. Linley, G.: Do Wombat Burrows Act as Refugia to Other Wildlife Species in Post -fi re Landscapes? (2021). Last updated Sep 2022 http:// n2t.net/ark:/63614/w12003801
135. Shamon, H.: IT - LTSER 2023 (2023). Last updated Aug 2023 http://n2t. net/ark:/63614/w12006485
136. Bresnan, C.: Bison Grazing 2023 (2023). Last updated Jun 2023 http:// n2t.net/ark:/63614/w12006261
137. Cabrera, J.A.: Corredor Guaviare - Meta 2021 (2023). Last updated Jan 2023 http://n2t.net/ark:/63614/w12004930
138. Jansen, P.: Zofin ForestGEO Project (2022). Last updated Feb 2023 http://n2t.net/ark:/63614/w12004581
139. Harris, S., et al.: EagleCam\_CU (2021). Last updated Jul 2023 http:// n2t.net/ark:/63614/w12003589
140. Masozera, M.: Long Term Monitoring of Wildlife Communities in Nyungwe National Park (2021). Last updated Dec 2022 http://n2t.net/ ark:/63614/w12000284
141. Harris, S., et al.: EagleCam (2021). Last updated Jul 2023 http://n2t. net/ark:/63614/w12001570
142. Ayala, G.: Lower Undumo River and Lower Tequeje River Jaguar Survey (2021). Last updated Sep 2021 http://n2t.net/ark:/63614/w12000260
143. Spironello, W .: Manaus (2021). Last updated Apr 2022 http://n2t.net/ ark:/63614/w12002646
144. WWF: WWF MY Tiger Temengor 2007 (2021). Last updated Jun 2022 http://n2t.net/ark:/63614/w1stg\_2003214
145. McMurry, S.: Mast and Mammals 2 (2023). Last updated Jul 2023 http://n2t.net/ark:/63614/w12006449
146. Anderson, S.: Siskiyou Deer Project R1 Wildlife (2021). Last updated Jul 2023 http://n2t.net/ark:/63614/w12003821
147. Chen, R.: Hamaarag Long Term Monitoring T3 2019 - 2020 (2021). Last updated Sep 2022 http://n2t.net/ark:/63614/w12000664
148. Farris, Z.: Makira NP Carnivore Survey 2012: Mangabe Forest (2021). Last updated Sep 2021 http://n2t.net/ark:/63614/w12000282
149. Jansen, P.: Wabikon Lake Forest ForestGEO Project (2022). Last updated Feb 2023 http://n2t.net/ark:/63614/w12004587
150. Farris, Z.: Makira NP Carnivore Survey 2011: Mangabe Forest (2021). Last updated Sep 2021 http://n2t.net/ark:/63614/w12000279
151. WWF - Ecuador: Monitoreo de Nutria (2021). Last updated Sep 2021 http://n2t.net/ark:/63614/w12001614
152. Rosa, F.L.: Achieving Jaguar Conservation and Local Communities Well - Being in the Southwest Amazon Jaguar Priority Landscape - PER (2022). Last updated Sep 2022 http://n2t.net/ark:/63614/w12003931
153. Shamon, H.: American Prairie Fence Pronghorn 2023 (2023). Last updated May 2023 http://n2t.net/ark:/63614/w12006229
154. Spencer, E.: BMCC Post -fi re Faunal Monitoring 2021 (2021). Last updated Sep 2022 http://n2t.net/ark:/63614/w12002996
155. Carvalho, E.A.R.Jr., Mendonça, E.N., Martins, A.: Reserva Biológica Do Gurupi (2021). Last updated Feb 2023 http://n2t.net/ark:/63614/ w12002545
156. One Tam: Beach (2021). Last updated Feb 2022 http://n2t.net/ark:/ 63614/w12001639
157. Ayala, G.: Jaguar Survey Tuichi Vallley 2001, Second Survey (2021). Last updated Sep 2021 http://n2t.net/ark:/63614/w12000257

158. Kamau, M.: Q Fever (2021). Last updated Sep 2022 http://n2t.net/ ark:/63614/w12003397
159. Ayala, G.: Heath River Jaguar Survey 2005 (2021). Last updated Sep 2021 http://n2t.net/ark:/63614/w12000259
160. Valencia, D.: Parque Nacional La Campana (2021). Last updated Sep 2022 http://n2t.net/ark:/63614/w12001235
161. Spencer, E.: Brush - Tailed Rock Wallaby Survey (2023). Last updated Mar 2023 http://n2t.net/ark:/63614/w12006048
162. Cook, V.: Nutria - CDFW Lands (2022). Last updated Feb 2022 http:// n2t.net/ark:/63614/w12004221
163. Farris, Z.: Makira NP Carnivore Survey 2009: Soavera Forest (2021). Last updated Sep 2021 http://n2t.net/ark:/63614/w12000274
164. Ahumada, J.: Cafe - Fauna, Alto Mayo (2021). Last updated Jan 2023 http://n2t.net/ark:/63614/w12002510
165. Ngoc, D.V.: Biodiversity Moniotring (2022). Last updated Aug 2022 http://n2t.net/ark:/63614/w12004370
166. Berlinck, C.N.: PN Pantanal Matogrossense (2022). Last updated Jan 2023 http://n2t.net/ark:/63614/w12004097
167. Casady, D.: North Coast Deer Abundance (NCDA 2019 - 2020) (2021). Last updated Mar 2023 http://n2t.net/ark:/63614/w12002901
168. Kays, R., et al.: Calloway Forest Preserve (2022). Last updated Jul 2022 http://n2t.net/ark:/63614/w12004251
169. ICMBio/CENAP: Parque Natural Municipal da Grota Funda (2021). Last updated Sep 2021 http://n2t.net/ark:/63614/w12002544
170. Will, D.: Cabritos Island Restoration Project (2021). Last updated Sep 2021 http://n2t.net/ark:/63614/w12001651
171. Will, D.: Floreana Island Restoration Project (2021). Last updated Sep 2021 http://n2t.net/ark:/63614/w12001653
172. Brncic, T.: Nouabalé Ndoki (2021). Last updated Apr 2022 http://n2t. net/ark:/63614/w12003030
173. Bresnan, C.: Bison Grazing 2022 (2022). Last updated Dec 2022 http:// n2t.net/ark:/63614/w12004234
174. Archer, L., et al.: Safeguarding the Philippine Pangolin (2021). Last updated May 2023 http://n2t.net/ark:/63614/w12002705
175. Smith, R.B., et al.: Investigating Responses by Wildlife to the Presence of Livestock Guarding Dogs (LGDs) (2021). Last updated Sep 2022 http:// n2t.net/ark:/63614/w12003795
176. Nelson, D., Shamon, H., Jachowski, D.: Swift Fox Scat 2021 (2021). Last updated Dec 2022 http://n2t.net/ark:/63614/w12003597

177.

Veracel Celulose, S.A.: Veracel Pequenos Remanescentes (2022). Last

[updated Jan 2024 http://n2t.net/ark:/63614/w12004350](http://n2t.net/ark:/63614/w12004350)

178. Bitariho, R.: Bwindi Impenetrable Forest (2021). Last updated Apr 2022 http://n2t.net/ark:/63614/w12003040
179. Furnas, B.: Mendocino Fisher Project (2021). Last updated Dec 2022 http://n2t.net/ark:/63614/w12003401
180. McNab, R.: Jaguar survey in Parque Nacional Laguna del Tigre (2021). Last updated Sep 2021 http://n2t.net/ark:/63614/w12000270
181. Brasil, W.: Parque Nacional Juruena (2021). Last updated Sep 2021 http://n2t.net/ark:/63614/w12001206
182. Farris, Z.: Makira NP Carnivore Survey 2011: Farankarina Forest (2021). Last updated Sep 2021 http://n2t.net/ark:/63614/w12000280
183. Carvalho, E.A.R.Jr., Nienow, S.S., Bonavigo, P.H.: Floresta Nacional de Jacundá (2022). Last updated Sep 2022 http://n2t.net/ark:/63614/ w12004252
184. Carvalho, E.A.R.Jr., Laranjeiras, T.O., Reis, M.L.: Parque Nacional Do Monte Roraima (2021). Last updated Sep 2022 http://n2t.net/ark:/ 63614/w12000521
185. Programa Monitora ICMBio: Parque Nacional Montanhas Do Tumucumaque (2021). Last updated Mar 2023 http://n2t.net/ark:/63614/ w12003586
186. Johnson, A.P., et al.: Lands R6 North Wildlife Area and Ecological Reserve Cameras (2022). Last updated Dec 2023 http://n2t.net/ark:/ 63614/w12004413
187. Jansen, P.A., den Ouden, J.: Speulderbos ForestGEO Project (2022). Last updated Feb 2023 http://n2t.net/ark:/63614/w12004580

-

188. Casady, D.: Surprise Valley Deer Abundance 2017 (SVDA 2017) (2021). Last updated Mar 2023 http://n2t.net/ark:/63614/w12002923
189. Schiavini, A.: Feral dog ecology in Tierra del Fuego Argentina (2021). Last updated Sep 2022 http://n2t.net/ark:/63614/w12000531
190. Shamon, H., McShea, J.W.: Smithsonian Great Plains Science 2020 (2021). Last updated Dec 2022 http://n2t.net/ark:/63614/ w12002982
191. Swanepoel, B.: Nam Kading (2021). Last updated Apr 2022 http://n2t. net/ark:/63614/w12003034
192. Sayol, F., et al.: European Wildcat Project (2013 - 2016) (2021). Last updated Sep 2022 http://n2t.net/ark:/63614/w12002768
193. Casady, D.: Lower Sacramento River (LSR 2016 - 2018) (2021). Last updated Mar 2023 http://n2t.net/ark:/63614/w12002889
194. Cowan, M.: Pilbara Mine Camp Fauna Monitoring (2022). Last updated Mar 2023 http://n2t.net/ark:/63614/w12004109
195. Kimuyu, D.: MpalaForestGeo (2021). Last updated Sep 2022 http:// n2t.net/ark:/63614/w12002761
196. Iannarilli, F.: Snapshot Europe 2023 (2023). Last updated Dec 2023 http://n2t.net/ark:/63614/w12006593
197. Casady, D.: Central Coast Deer Abundance 2017 - 18 (CCDA 2017 - 18) (2021). Last updated Mar 2023 http://n2t.net/ark:/63614/w12002877
198. Farris, Z.: Makira NP Carnivore Survey 2012: Anjanaharibe Forest (2021). Last updated Sep 2021 http://n2t.net/ark:/63614/w12000281
199. Wallace, R.: Alto Madidi 2011 (2021). Last updated Sep 2021 http://n2t. net/ark:/63614/w12000263
200. Spencer, E.: Southern Ark Predator Transects (2022). Last updated Feb 2023 http://n2t.net/ark:/63614/w12004553
201. Souter, N.: Chheu Teal Community Forest Revegetation (2021). Last updated Sep 2022 http://n2t.net/ark:/63614/w12001239
202. Brewster, J.R., Giumelli, J.P ., Ashman, R.K.: Plateau Survey (2022). Last updated Sep 2022 http://n2t.net/ark:/63614/w12004207
203. Endo, W., Campos - Souza, B., Carvalho, E.A.R., Jr: Estação Ecológica de Maracá (2021). Last updated Sep 2022 http://n2t.net/ark:/63614/ w12002584
204. Aiello, M.C., et al.: Enhancing Function of an Increasingly Fragmented Metapopulation of Desert Bighorn Sheep (2021). Last updated Dec 2023 http://n2t.net/ark:/63614/w12003007
205. Andrianarisoa, M.: Ranomafana (2021). Last updated Jan 2023 http:// n2t.net/ark:/63614/w12003032
206. Jansen, P.: Smithsonian Environmental Research Center ForestGEO Project (2022). Last updated Feb 2023 http://n2t.net/ark:/63614/ w12004305
207. Rovero, F.: Udzungwa (2021). Last updated Apr 2022 http://n2t.net/ ark:/63614/w12003041
208. Martin, E.M., Green, S.D., Matthews, M.S.: Klamath Carnivore Community Project (2021). Last updated Feb 2022 http://n2t.net/ark:/ 63614/w12001294
209. Wallace, R.: Acero Marka Survey 2012 (2021). Last updated Sep 2021 http://n2t.net/ark:/63614/w12000268
210. Valenzuela, L.: Fototrampeo SFPM Orito Ingi Ande, Putumayo (2021). Last updated Sep 2022 http://n2t.net/ark:/63614/w12003815
211. ICMBio: Estação Ecológica de Taiamã (2021). Last updated Sep 2022 http://n2t.net/ark:/63614/w12003800
212. Gajapersad, K.: CI Suriname: Central Suriname Nature Reserve (2021). Last updated Apr 2022 http://n2t.net/ark:/63614/w12003038
213. Kays, R.: Snapshot Europe 2021 (2021). Last updated Sep 2022 http:// n2t.net/ark:/63614/w12003388
214. Wallace, R.: Puina, Pusupunku, Pasto Grande Jaguar Survey 2012 (2021). Last updated Sep 2021 http://n2t.net/ark:/63614/w12000267
215. Uzabaho, E., Turikunkiko, E.: Gishwati - Mukura National Park (2021). Last updated Sep 2022 http://n2t.net/ark:/63614/w12003590
216. Pacheco, J.: WWF EC: Monitoreo Tri Nacional de jaguar (2021). Last updated Dec 2022 http://n2t.net/ark:/63614/w12000288
217. Ayala, G.: San Pedro Savannah Diversity Survey (2021). Last updated Sep 2021 http://n2t.net/ark:/63614/w12000261

-

218. Giumelli, P.: Bowen Island (2022). Last updated Mar 2023 http://n2t. net/ark:/63614/w12004130
219. Rovero, F.: Inventory of Medium - To - Large, Terrestrial Forest Mammals and Birds in the Tanzanian Eastern Arc Mountains (2021). Last updated Sep 2021 http://n2t.net/ark:/63614/w12000285
220. Butti, M.: Floresta Nacional de Silvânia (2021). Last updated Aug 2022 http://n2t.net/ark:/63614/w12002556
221. MMPL Palawan: Mount Mantalingahan Protected Landscape (MMPL) - Wildlife (2021). Last updated Feb 2022 http://n2t.net/ark:/63614/ w12001564
222. Ragai, R.: Community Monitoring of Large Mammals Distribution Around an Important Protected Area in Sarawak. Malaysian Borneo (2023). Last updated Feb 2023 http://n2t.net/ark:/63614/w12004822
223. Boekee, K., Kenfack, D., Jansen, P.A.: Korup National Park (2021). Last updated Feb 2023 http://n2t.net/ark:/63614/w12002642

## SUPPORTING INFORMATION

Additional supporting information can be found online in the Supporting Information section at the end of this article.

How to cite this article: Gadot, T., et al.: To crop or not to crop: comparing whole - image and cropped classification on a large dataset of camera trap images. IET Comput. Vis. 18(8), 1193-1208 (2024). https://doi. org/10.1049/cvi2.12318