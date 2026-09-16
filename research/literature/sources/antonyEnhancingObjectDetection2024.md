## Enhancing Object Detection Performance for Small Objects through Synthetic Data Generation and Proportional Class-Balancing Technique: A Comparative Study in Industrial Scenarios

Jibinraj Antony :; , Vinit Hegiste ˚ , Ali Nazeri : , Hooman Tavakoli : , Snehal Walunj :

, Christiane Plociennik : , Martin Ruskowski :˚

: German Research Center for Artificial Intelligence (DFKI), Kaiserslautern, Germany

˚ Rheinland-Pf¨ alzische Technische Universit¨ at (RPTU) Kaiserslautern-Landau, Kaiserslautern, Germany

; Corresponding Author. Tel.: +49-176 6290 8490 ; E-mail : jibinraj.antony@dfki.de

** The first five Authors contributed to the Research equally.

Abstract -Object Detection (OD) has proven to be a significant computer vision method in extracting localized class information and has multiple applications in the industry. Although many of the state-of-the-art OD models perform well on medium and large sized objects, they seem to under perform on small objects. In most of the industrial use cases, it is difficult to collect and annotate data for small objects, as it is time-consuming and prone to human errors. Additionally, those datasets are likely to be unbalanced and often results in inefficient model convergence. To tackle this challenge, this study presents a novel approach that injects additional data points to improve the performance of the OD models. Using synthetic data generation, the difficulties in data collection and annotations for small object data points can be minimized and a balanced distribution of dataset can be created. This paper discusses the effects of a simple proportional classbalancing technique, to enable better anchor matching of the OD models. A comparison has been made on the performances of the state-of-the-art OD models: YOLOv5, YOLOv7 and SSD, for combinations of real and synthetic datasets within an industrial use case.

Index Terms -Small Object Detection, Class Balancing, Synthetic Data Generation, YOLOv7, YOLOv5, SSD.

## I. INTRODUCTION

The birth of Convolution Neural Networks (CNN) set down an important milestone in the performance improvement of image classification challenges, where networks like AlexNet [1] and their successors exceed human level performances. As the researches in CNN advanced, so did the academic interest in the localization of the specific objects, where networks like OverFeat [2] made their first success with a deep CNN architecture. Further improvements in the field of Object Detection (OD) were possible with networks like Regions with CNN features (R-CNN) [3], Fast R-CNN [4], Faster R-CNN [5], YOLO [6] and its variants. These networks and their current adaptations were the catalyst in solving some of the most challenging OD problems, enabling it to be integrated into the industry with a high degree of confidence.

Detecting small objects is particularly challenging as they have very small features-space to offer for training. As per COCO evaluation matrix [7], the small objects are the objects in a image having an area less than 32x32 pixels. Although the state-of-the-art Machine Learning (ML) models are excellent in general OD applications involving prominently visible or dominant objects, they perform poorly in scenarios involving small objects. For instance, the latest COCO Dataset Object Detection challenge shows that even the top performing models tends to perform badly on small objects [8] (see Figure 1). More specifically, taking a YOLOv4 with CSPDarknet-53 backbone on MS COCO Dataset as example [9], the mean average precision (mAP) for the smaller object is only 20%, where mAP for the dominant objects are more than twice as much (mAP of 45% for the medium and 56% for the large objects), highlights the big gap in the performance of models in smaller OD problems.

<!-- image -->

Fig. 1. Detection Leader board of COCO Dataset Object Detection Challenge from 2020. The Average Precision of Small ( AP S ) and Large ( AP L ) objects in the dataset for the top performing models are highlighted in the figure.

|                        |    AP |   Ap50 |   Ap75 |   APS |   APM |   APL |   AR1 |   AR10 |
|------------------------|-------|--------|--------|-------|-------|-------|-------|--------|
| Noah CV Lab (Huawei)   | 0.588 |  0.766 |  0.649 | 0.407 | 0.616 | 0.720 | 0.418 |  0.700 |
| mmdet                  | 0.578 |  0.770 |  0.637 | 0.399 | 0.605 | 0.706 | 0.414 |  0.690 |
| DeepAR(ETRIxKAIST_AIM) | 0.553 |  0.746 |  0.609 | 0.378 | 0.583 | 0.668 | 0.403 |  0.674 |
| DetectoRS              | 0.550 |  0.736 |  0.604 | 0.377 | 0.578 | 0.669 | 0.401 |  0.678 |
| KiwiDet2               | 0.547 |  0.728 |  0.597 | 0.362 | 0.576 | 0.685 | 0.401 |  0.680 |
| 360 AI Research        | 0.546 |  0.737 |  0.600 | 0.369 | 0.583 | 0.674 | 0.398 |  0.665 |
| ZFTurbo                | 0.544 |  0.728 |  0.600 | 0.367 | 0.579 | 0.671 | 0.400 |  0.657 |
| Hyundai Mobis AD Lab   | 0.538 |  0.721 |  0.591 | 0.343 | 0.571 | 0.675 | 0.398 |  0.646 |
| ByteDance_VC           | 0.534 |  0.723 |  0.589 | 0.351 | 0.572 | 0.663 | 0.396 |  0.653 |

However, when it comes to the integration of ML application into the industry, the data driven nature of these approaches make them highly dependent on the data used, both quality and quantity, which is why data availability is one of the most crucial factors in the project's success. But in industrial environments, the collection of enough useful data is unfortunately a painful process, accounting for additional costs, efforts, and time. Data has to be collected error-free and has to be categorized correctly in order to ensure its effectiveness in the ML application. Since the industrial machines of today are less prone to errors, the availability of the useful data necessary for the ML implementation becomes even more limited, turning the industrial realization a challenging and complex task. In some cases, the data obtained from the industrial environment is unbalanced, which consequently affects the model's performance on all classes. The low performance for the detection of small objects is often attributed to their unbalanced distribution over the entire dataset. A balanced dataset could eventually improve the learning of the model and hence perform better in all associated tasks.

As the manual labelling of small objects on image data is a time and effort intensive task, the latest advancements of synthetic data generation methods tend to address these difficulties in data preparation [10]. For many industrial objects, CAD data is available and can be utilized as a blue-print for the generation of synthetic data. Although CAD data lacks photo-realism, a simple rendering can be imparted and the corresponding images of the target class could be generated from their CAD models without high computational costs. Upon adding the newly generated images, the class distribution of the dataset could be balanced. This approach of leveraging synthetic data generation together with real datasets for class focused data balancing and its effectiveness in improving the performance of use cases with small object detection has been addressed in this paper.

The structure of the paper is as follows: In section II, we discuss the related works in OD applications in industry, as well as the various data manipulation techniques commonly used for the ML applications. This is followed by detailed description of the methodology and the experimental setup performed in section III. In section IV we present the results of our approach and compares it with the other popular implementations. Finally, we conclude with the summary of the work and discussion on the future scope in section V.

## II. BACKGROUND &amp; RELATED WORKS

Object Detection is a crucial problem in computer vision that involves recognizing and localizing objects of interest within an image or a video stream [11]. In recent years, deep learning (DL) techniques using CNNs, have achieved remarkable success on improving the accuracy of OD models, even in challenging scenarios where the object features are partially occluded, poorly illuminated, or exhibit low contrast. Single Shot Detector (SSD) [12] and You Only Look Once (YOLO) [13] are two CNN-based models that have gained popularity in the field of OD. These models have various applications in transportation, military, and industrial use cases. For example, an OD-model based on YOLOv2 was utilized in [14] for the surface inspection on conveyor belts, while [15] compares the effect of SSD, F-RCNN, YOLOv3 and YOLOv5 OD models in detecting surface defects in metals using a generic dataset.

Beyond general OD applications, the small OD finds its application in areal object inspection, industrial scenarios, etc. Small OD also forms the basis of many other computer vision applications such as object tracking [16], instance segmentation [17], action recognition [18] and others. The field of application of small OD is therefore diverse, where it is used for spider detection and removal application [19], in autonomous driving application [20], remote sensing [21], or even for improving the synchronization of Digital Twin of a manufacturing system [22].

Due to the data-driven nature of small OD models, they are highly dependent on the available data. In most industrial situations, collecting new datasets to train a model is a challenging, expensive and a time-consuming task [23]. One way to address this is data augmentation, which is a powerful technique to alleviate over-fitting. Many researchers have proposed data augmentation methods [24], [25] to artificially increase the size of training data using the data in hand without collecting new data [26]. One or more morphs are applied to the data while preserving the labels during transformation [27].

Synthetic data [28], and can be generated based on custom requirements and other controlled conditions. More importantly, such a dataset should follow the underlying distribution of the real dataset [29]. Synthetic data generation can be achieved generally via two techniques, first by using 3D rendering and simulation tools such as Game Engines and second by using DL technique such as Variational Autoencoder [30], Denoising Diffusion Probabilistic Models [31], GANs [32] (such as StyleGANs3) [33], etc. The most simple form of synthetic data generation is using the cut-and-paste method , where one can have various types of background and the dedicated objects of different types needed to be detected. Then a combination of these objects with the background is undertaken to create an image, i.e. to cut the object and paste it on to the background image. This method is good to train OD for some simple tasks, but would not yield better results for complex OD tasks, since this approach does not help in model generalization. While DL techniques are faster than normal cut-paste in creating a large dataset, a similar issue would arise as these models do not guarantee the desired outcome with specific angles, backgrounds, and lighting conditions. These issues can be easily tackled with synthetic data created using 3D rendering and simulation tools [34].

Usage of synthetic data for OD makes many applications easier to implement. The authors, in [35], utilize synthetic data generated using DeepGTAV framework to work on Unmanned Aerial Vehicles scenarios. While synthetic data was leveraged in [36] to create OD models for robotic object grasping application. CAD models offer an important asset for synthesizing image data. Leveraging the geometric precision and comprehensive annotations inherent in CAD models, it is possible to create diverse and labeled synthetic datasets that enhance the performance and generalization of OD models [34]. Object-related CAD models are available in a highlydetailed geometric form since they are necessary attributes in manufacturing. Game engines allow the use of these CAD models of compatible formats within a simulation. Taking advantage of this feature of the renderer space and a game camera, the dataset capturing process can be simulated [37]. There are also packages like Unity Perception package that enable the generation of synthetic datasets for multi-OD and segmentation applications.

In addition to the benefits associated with data generation, synthetic data generated using CAD models exhibits a notable challenge in terms of domain dissimilarity when compared to real-image-based test data. On the other hand, the synthetic data generated from CAD models come with the problem of being inherently different to the domain of the real images. Although there are various domain adaptation techniques in the literature such as synthetic-to-real domain adaptation in [38], which aim to modify synthetic data, there are also works, which present how photo-realistic rendering alone cannot reduce the domain gap and therefore attempt to resolve the issue by domain randomization [39], [40]. Furthermore, investigating the combined effect of synthetic and real data as a hybrid dataset, without altering its nature, presents an intriguing avenue for exploration. In scenarios where there is an amount of limited real data, this approach could help leverage it, with synthetic data alongside.

## III. METHODOLOGY

The synthetic data generation approaches described in section II, could address the data limitations hindering the successful development of small OD-models, by generating class specific balanced datasets. A photo-realistic synthetic datasets may not be necessary for some scenarios, where the generated synthetic datasets are mixed along with the available real dataset.

To validate this approach, an industrial use case of a manual assembly scenario has been taken into consideration. In the assembly process, an actor/worker performs a circuit breadboard assembly of a previously designed configuration, using the specific electronics components, while wearing an Augmented Reality (AR) or Mixed Reality (MR) glasses. The camera sensor in the AR glasses can capture the scene consisting of various objects available within the worker's field-of-view and feed it into an OD-model to perform object recognition. Based on the obtained results, a worker assistance-systems in AR will assist the worker by providing instructions and visual augmentations. The worker assistance and AR methods will not be discussed further, as they are outside the scope of this paper. The focus will be primarily only on the performance of the OD model.

After identifying the distribution of each target class in the original dataset, synthetic data generation will be used to balance the classes. With this approach, we are able to create a proportionally balanced dataset, with more instances of small objects in the combined dataset. Later, the state-of-the-art OD models were trained in combinations of experiments and the results were evaluated.

## A. Data Distribution

The data driven nature of DL/ML models makes the data distribution a significant factor in the project's success. The focus of this paper is the detection of small objects in the manual assembly scenario, where a correct identification is crucial in the worker assistance context. As the first step of

<!-- image -->

X

Fig. 2. Data Distribution of Initial real dataset (left) to the combined dataset DS-3 (right). The target classes have been proportionally balanced, giving more significance for the less occurring classes.

this work, the original distribution of the data was identified. There were Five objects in consideration in the target dataset, out of which the Three objects (LED, Resister &amp; Button) fit the criteria of a small object i.e. below 32x32 pixels. The other Two objects (Buzzer, Arduino) are between 32x32 pixels and 64x64 pixels and are thus considered as medium-sized objects.

After identifying the distribution of the target objects, a proportionally balanced dataset was created using synthetic data generation. The Figure 2 shows the initial and final distribution of the dataset (original dataset on left and the combined dataset DS-3 on the right. Details on DS-3 are shown in table I).

## B. Dataset Generation

In order to create a balanced dataset, synthetic data generation techniques was utilized. The table I lists the datasets and their combination of real and synthetic data instances used for this work. A total of Five datasets were generated combining the real and synthetic data and different OD models were trained on those datasets with various combinations of hyperparameters. The datasets are later referenced using the acronyms given in this table I.

The datasets are generated using a scene simulation with a script for image capturing of the game-engine camera. Initially the CAD models are imported in the Unity scene in compatible format. On imparting the 3D models with materials and textures with minimal rendering. The dataset is simple and consists of objects in assembled and disassembled states. The object backgrounds and scales in the images are randomly set within a predefined scale range. We also varied illumination and object viewpoints. We used white lights and yellow lights with random illumination in the virtual scene while capturing

|   Exp. No | Dataset Name   | Real Data   | Synthetic Data   |   Total Size of Training Data |
|-----------|----------------|-------------|------------------|-------------------------------|
|         1 | DS-1           | 300         | 0                |                           300 |
|         2 | DS-2           | 300         | 100 (33% more)   |                           400 |
|         3 | DS-3           | 300         | 150 (50% more)   |                           450 |
|         4 | DS-4           | 300         | 300 (100% more)  |                           600 |
|         5 | DS-5           | -           | 300              |                           300 |

## TABLE I

THE LIST OF DATASETS AND THEIR DIVISION OF REAL AND SYNTHETIC DATA INSTANCES USED. THE DATASETS ARE LATER REFERENCED USING THE ACRONYMS GIVEN IN THIS TABLE.

Fig. 3. A sample synthetic image generated using the 3D rendering Game Engine, used in the dataset.

<!-- image -->

images for the dataset. We do not add variable viewpoints or object occlusions in the dataset. In our use case, the objects are perceived from a limited number of viewpoints. Hence, it's a simple dataset that could be generated using the available CAD models of industrial parts. A sample of synthetic image from the dataset is shown in 3.

Figure 4 shows a sample original image from the dataset and figure 3 shows a synthetically generated image. As it is observable, the generated data is rather simple and mainly focuses in balancing the target object instances, rather than creating a photo-realistic scene.

## C. Objected Detection Models

After generating the datasets and its combinations, they were used to train and compare state-of-the-art OD-models.

Fig. 4. A sample of real image data from the Assembly Scenario, taken from the live-stream video of AR device.

<!-- image -->

Based on popularity and ease of implementation, three models were selected and utilized for the experiments. Two of the models were based on YOLO family developed on PyTorch framework and the other model was the SSD based on TensorFlow. These models are considered to be effective on small OD use cases, faster in inferences and having remarkable performances.

1) YOLOv5: YOLO (You Only Look Once) is a family of OD algorithms, which performs detection in an image through a single forward pass, unlike other algorithms like Fast-RCNN or Faster-RCNN using two stages. YOLO was the first OD algorithm to combine the procedure of predicting bounding box with class label in an end to end differentiable network [41]. YOLOv5 is open source and original version of the code is written in python with PyTorch framework. YOLO models use a Non-Maxima Suppression (NMS) as a post-processing step to obtain the final bounding box for each detected object. This technique filters out redundant bounding boxes based on their overlap values and ensures that only the most relevant and accurate predictions remain in the final results [6].YOLOv5 offers a range of architectures, each tailored to specific application requirements and datasets. The YOLOv5s (small), YOLOv5m (medium), YOLOv5l (large), and YOLOv5x (Extra Large) architectures exhibit increasing complexity in their design. As the architecture complexity grows, so does the accuracy of the object detection results. However, this improvement in accuracy comes at the expense of reduced model speed during object detection. Therefore, the choice of architecture depends on striking the right balance between accuracy and real-time object detection speed, considering the specific needs of the application and datasets at hand.

2) YOLOv7: Wang et al. in [42], introduced YOLOv7 as an extension of the YOLO series of real-time OD models. It introduces various models tailored for different GPU environments, including edge GPU, normal GPU, and cloud GPU. Examples of these models include YOLOv7-tiny, YOLOv7, and YOLOv7-W6. Additionally, YOLOv7 incorporates model scaling techniques to cater to diverse service requirements, resulting in the development of models such as YOLOv7-X, YOLOv7-E6, YOLOv7-D6, and YOLOv7-E6E. They apply stack scaling to the neck component and employ the suggested compound scaling technique to increase the depth and width of the entire model, resulting in YOLOv7-X. On the other hand, for YOLOv7-W6, they utilize the newly introduced compound scaling method to derive YOLOv7-E6 and YOLOv7-D6.

YOLOv7 in OD offers notable improvements in speed and accuracy compared to previous models. The speed ranges from 5 to 160 frames per second (FPS), enabling real-time applications. In terms of accuracy, YOLOv7 achieves the highest average precision (AP) of 56.8% among all real-time object detectors operating at 30 FPS or higher [42].

3) SSD: SSD (Single Shot MultiBox Detector) is a prominent OD model that considers bounding box prediction as a regression problem. It starts by selecting the anchor box with the highest intersection over union (IoU) with the ground truth bounding box and gradually refines the prediction by minimizing the loss between the predicted and ground truth boxes. This iterative regression process allows SSD to achieve precise localization of objects in the image [12]. This model provides a straightforward procedure for eliminating duplicate predictions, while SSD's regression-based approach focuses on refining the bounding box estimate by progressively minimizing the error.

## D. Hyper-Parameters

After identifying and selecting the OD models as described above, the experiments have been performed. To ensure a comprehensive comparison among the three OD algorithms, it is essential to maintain consistent hyper-parameters during the training and testing phases on the identical dataset. In order to achieve this, the experiments employed the following hyperparameters:

- , Training Batch size : 8
- , Input Image size : 1080 x 1080 pixels (height x width)

To optimize the training process, all the experiments utilized the Adam optimizer, which has been widely recognized for its efficiency in deep learning tasks. The Adam optimizer effectively combines the benefits of adaptive gradient algorithms and momentum-based optimization methods, facilitating faster convergence and improved performance during training.

To evaluate the performance of the algorithms, experiments were conducted for different numbers of Epochs: specifically, 30, 50 and 100 Epochs. By observing the algorithms' performance over multiple epochs, insights into their convergence rates and overall accuracy as the training progresses are observed.

The OD models are then trained on the dataset combinations and evaluated on a general test dataset consisted of real images. The model performances are evaluated on the COCO test evaluation matrix [43] and the results are reviewed and observations are noted.

## IV. RESULTS

After training the Three OD models on the five sets of datasets with three hyper-parameter combinations, the models are tested on a common test dataset, that consists of real images from the Assembly scenario. The test was carried out using the COCO evaluation matrix [43], examining the performance on small, medium and large sized objects and the results are noted. Table II, Table III and Table IV represents the evaluation results from YOLOv5, YOLOv7 and SSD models respectively. The silent feature of this table are the Average Precision and Average Recall of Small objects (APs &amp; ARs) which calculates the actual Precision and Recall of the model based on the true bounding box and predicted bounding box overlap for small objects (less than 32x32 pixels). This was necessary because only by observing the mAP, a true analysis cannot be made as the models are performing well on most of the cases.

Upon comparing the models, the results from Table II show that YOLOv5 performs the best out of all the three models, especially for small objects. Upon comparing the datasets, the models trained on a dataset containing real data components (DS-1, DS-2, DS-3, DS-5) seems to perform better on the dataset with only synthetic data (DS-4). This is due to the synthetic data generation features, as mentioned in section III-B, where the samples created were non-photo-realistic and therefore upon testing in real dataset, the model trained only on synthetic dataset seem to perform inadequately.

It was also observed that as the number of synthetic data points increases (from DS-1 to DS-3), the models tend to learn more features through the targeted class balancing technique and reaches a maximum performance in the DS-3 dataset, where the number of synthetic data in the dataset is exactly the half of real data instances. Upon further increasing the synthetic dataset (DS-4 &amp; DS-5) the models tend to learn more features from the prominent synthetic data and therefore performed poorly upon testing in the real world. And comparing the hyper-parameters, the models trained for 100 epochs seem to perform better compared to the other instances.

Comparing the recent advancements in the YOLO model families, the YOLOv7 model performs very well detecting small objects in the scenario of this work. The results of this model for different experiments which are presented in Table III, clearly show that by increasing the number of synthetic data samples, can improve the accuracy of the model to detect small objects as well as medium and large objects, but there is a limitation for that and increasing these samples too much can affect the results negatively. The results show that the highest performance occurs for DS-3, when the number of synthetic data is equal to half of real data samples.

The results of SSD model which are presented in Table IV show that while this model can not achieve the best results compared to the other YOLO models, using and combining an amount of synthetic data with the real data samples to train the model can affect the performance and improve the accuracy of the model. The table shows that the results of training the SSD model with only synthetic data are very weak because of the difference between the synthetic dataset and the real object images, especially considering the background. On the other hand, when the amount of synthetic data samples is half of the real dataset, the performance of the SSD model is the highest (same as the YOLOv7 model), which means combining onethird of synthetic data with two-third of real dataset can affect and improve the performance of the SSD model to achieve the best result.

In a nutshell, adding synthetic data with class balancing technique can improve the performance especially for small objects, even by synthetically generating additional data points build using simple open source CAD models. The results show an increase of up to 11.4% in precision on small objects (APs) to the base model trained only on real data set (DS-1) and best combination ratio of real and synthetic dataset (DS-3) for 100 epochs. Similar percentage increase in precision and recall can be observed for models trained on 30 and 50 epochs as well.

TABLE II

|   No |   Epoch | Dataset   |   mAP |   AP50 |   APs |   APm |   APl |   ARs |   ARm |   ARl |
|------|---------|-----------|-------|--------|-------|-------|-------|-------|-------|-------|
|    1 |      30 | DS-1      | 0.922 |  0.912 | 0.373 | 0.516 |   0.3 | 0.446 | 0.574 | 0.339 |
|    2 |      50 | DS-1      | 0.946 |  0.939 |  0.47 | 0.532 |  0.35 | 0.515 | 0.588 | 0.375 |
|    3 |     100 | DS-1      | 0.976 |  0.967 | 0.493 | 0.557 | 0.373 | 0.564 |  0.61 |   0.4 |
|    4 |      30 | DS-2      | 0.936 |  0.929 | 0.401 | 0.511 | 0.329 | 0.476 | 0.583 | 0.364 |
|    5 |      50 | DS-2      | 0.953 |  0.949 | 0.475 | 0.576 | 0.362 | 0.537 | 0.631 | 0.394 |
|    6 |     100 | DS-2      | 0.955 |  0.952 | 0.543 | 0.605 | 0.366 | 0.594 | 0.654 | 0.397 |
|    7 |      30 | DS-3      | 0.955 |  0.953 | 0.475 | 0.576 |  0.33 |  0.53 |  0.63 | 0.367 |
|    8 |      50 | DS-3      | 0.959 |  0.953 | 0.488 | 0.566 | 0.364 | 0.354 | 0.624 | 0.398 |
|    9 |     100 | DS-3      | 0.986 |  0.982 | 0.594 | 0.611 | 0.369 | 0.659 | 0.667 | 0.395 |
|   10 |      30 | DS-4      | 0.966 |  0.959 | 0.449 | 0.594 | 0.346 | 0.484 | 0.661 | 0.375 |
|   11 |      50 | DS-4      | 0.977 |  0.968 | 0.577 | 0.613 | 0.382 | 0.627 | 0.672 | 0.402 |
|   12 |     100 | DS-4      | 0.983 |  0.979 | 0.607 | 0.628 | 0.384 | 0.658 | 0.688 |  0.41 |
|   13 |      30 | DS-5      | 0.369 |  0.357 | 0.143 | 0.093 | 0.137 | 0.143 | 0.169 | 0.183 |
|   14 |      50 | DS-5      | 0.336 |  0.323 | 0.057 | 0.104 | 0.170 | 0.057 | 0.139 | 0.207 |
|   15 |     100 | DS-5      | 0.412 |  0.399 | 0.083 | 0.121 | 0.149 | 0.128 | 0.182 | 0.193 |

EXPERIMENT RESULTS FOR YOLOV5 MODEL, THE EXPERIMENT INSTANCE CORRESPONDING TO DS-3 AT 100 EPOCHS SEEM TO GIVE THE OPTIMUM RESULTS, EVEN COMPARING THE DS-4 WITH A LOT MORE SYNTHETIC DATA.

TABLE III

|   No |   Epoch | Dataset   |   mAP |   AP50 |   APs |   APm |   APl |   ARs |   ARm |   ARl |
|------|---------|-----------|-------|--------|-------|-------|-------|-------|-------|-------|
|    1 |      30 | DS-1      | 0.885 |  0.873 | 0.259 | 0.416 | 0.273 | 0.306 | 0.487 | 0.309 |
|    2 |      50 | DS-1      | 0.905 |  0.897 |  0.36 |  0.45 | 0.285 | 0.447 | 0.505 | 0.328 |
|    3 |     100 | DS-1      | 0.951 |  0.946 |  0.48 | 0.566 | 0.361 | 0.538 | 0.621 | 0.393 |
|    4 |      30 | DS-2      | 0.838 |  0.831 | 0.331 | 0.404 | 0.313 | 0.403 | 0.461 |  0.35 |
|    5 |      50 | DS-2      | 0.914 |  0.907 | 0.463 | 0.479 | 0.347 |  0.56 | 0.538 | 0.372 |
|    6 |     100 | DS-2      | 0.949 |  0.939 | 0.499 | 0.588 | 0.379 | 0.552 | 0.635 | 0.412 |
|    7 |      30 | DS-3      | 0.947 |  0.943 | 0.424 | 0.514 |  0.32 | 0.475 | 0.588 | 0.352 |
|    8 |      50 | DS-3      | 0.957 |   0.95 |  0.42 | 0.575 |  0.34 | 0.472 |  0.64 | 0.375 |
|    9 |     100 | DS-3      | 0.976 |  0.967 | 0.512 | 0.624 |  0.38 | 0.553 | 0.689 | 0.404 |
|   10 |      30 | DS-4      | 0.906 |  0.902 | 0.305 | 0.538 | 0.319 | 0.349 | 0.618 | 0.354 |
|   11 |      50 | DS-4      | 0.928 |  0.922 | 0.372 |  0.51 | 0.322 | 0.434 | 0.616 | 0.359 |
|   12 |     100 | DS-4      | 0.946 |  0.939 |  0.47 | 0.578 | 0.578 | 0.519 | 0.659 | 0.398 |
|   13 |      30 | DS-5      | 0.243 |  0.235 | 0.083 | 0.059 | 0.072 | 0.103 | 0.108 | 0.147 |
|   14 |      50 | DS-5      | 0.304 |    0.3 | 0.105 | 0.075 | 0.085 | 0.114 | 0.117 | 0.145 |
|   15 |     100 | DS-5      | 0.374 |  0.364 | 0.122 | 0.102 | 0.129 | 0.144 | 0.173 | 0.246 |

EXPERIMENT RESULTS FOR YOLOV7 MODEL. THE DS-3 DATASET CLEARLY GIVES BEST RESULTS COMPARED TO THE OTHER DATASET COMBINATIONS.

TABLE IV

|   No |   Epoch | Dataset   |   mAP |   AP50 |   APs |   APm |   APl |   ARs |   ARm |   ARl |
|------|---------|-----------|-------|--------|-------|-------|-------|-------|-------|-------|
|    1 |      30 | DS-1      | 0.492 |  0.480 | 0.015 | 0.174 | 0.254 | 0.051 | 0.251 | 0.290 |
|    2 |      50 | DS-1      | 0.630 |  0.617 | 0.079 | 0.240 | 0.300 | 0.170 | 0.292 | 0.330 |
|    3 |     100 | DS-1      | 0.826 |  0.765 | 0.204 | 0.329 | 0.314 | 0.265 | 0.411 | 0.347 |
|    4 |      30 | DS-2      | 0.414 |  0.409 | 0.036 | 0.142 | 0.274 | 0.062 | 0.214 | 0.304 |
|    5 |      50 | DS-2      | 0.556 |  0.540 | 0.079 | 0.177 | 0.263 | 0.113 | 0.232 | 0.305 |
|    6 |     100 | DS-2      | 0.813 |  0.752 | 0.147 | 0.301 | 0.315 | 0.217 | 0.368 | 0.352 |
|    7 |      30 | DS-3      | 0.333 |  0.307 | 0.007 | 0.087 | 0.246 | 0.027 | 0.134 | 0.289 |
|    8 |      50 | DS-3      | 0.574 |  0.555 | 0.033 | 0.175 | 0.242 | 0.091 | 0.244 | 0.280 |
|    9 |     100 | DS-3      | 0.850 |  0.742 | 0.190 | 0.302 | 0.329 | 0.265 | 0.379 | 0.355 |
|   10 |      30 | DS-4      | 0.253 |  0.232 | 0.002 | 0.041 | 0.230 | 0.021 | 0.079 | 0.263 |
|   11 |      50 | DS-4      | 0.446 |  0.430 | 0.028 | 0.144 | 0.210 | 0.072 | 0.214 | 0.242 |
|   12 |     100 | DS-4      | 0.775 |  0.720 | 0.155 | 0.264 | 0.305 | 0.227 | 0.322 | 0.329 |
|   13 |      30 | DS-5      | 0.081 |  0.076 | 0.011 | 0.034 | 0.063 | 0.062 | 0.071 | 0.092 |
|   14 |      50 | DS-5      | 0.121 |  0.099 | 0.067 | 0.059 | 0.091 | 0.079 | 0.108 | 0.136 |
|   15 |     100 | DS-5      | 0.198 |  0.142 | 0.088 | 0.101 | 0.123 | 0.092 | 0.132 | 0.169 |

EXPERIMENT RESULTS FOR SSD MODEL. SSD MODEL SEEM TO PERFORM NOT REALLY GOOD ON THE DATA, BUT COMPARING THE OVERALL PERFORMANCE, THE MODEL PERFORMED COMPARATIVELY BETTER ON DS-3.

## V. CONCLUSION

Certain scenarios, where real-world dataset are be limited and biased towards prominent objects, present challenges for detecting less prominent and significant objects, such as smallsized objects. However, by increasing the dataset through targeted class balancing techniques using synthetic data generation, there appears to be an improvement in the performance of data-driven object detection models. The results from our use case show that even with low quality of synthetic dataset, an increase of 10% in the mAP for small objects can be achieved using this technique. But Adding more synthetic images seem to degrade the model's performance as the generated data were not photo-realistic compared to the real world data. In comparison with three state-of-the-art OD Models, YOLOv5 performed better in this particular use case. It is worth to note that, even though YOLOv7 claims to perform best for small objects [42], it did not perform as well as its predecessor YOLOv5 under similar training conditions for small objects.

The task of annotating the small object dataset is tedious, and using synthetic data for automatic annotations is an optimum approach to improve the model performance. Together with class balanced data generation techniques, through synthetic data generation without implementing photo-realistic rending, a better performing models can be easily generated without much effort. For certain industrial use cases, these techniques can be really helpful in achieving the business goals.

Although this study gives a clear comparison of state-ofthe-art OD models, it has potential for many future works. By extending the quality of the generated images to more photo-realistic and bring closer to the real world could potentially improve the performance a lot. But the data diversity challenges in the real world and the corresponding model performance and comparing it to the costs of data generation can still debatable.

## ACKNOWLEDGMENT

This research work is funded by the German BMBF - Bundesministerium f¨ ur Bildung und Forschung (0IW19002, project InCoRAP). We would like to thank Al Harith Farhad for proofreading the paper.

## REFERENCES

- 1 Krizhevsky, A., Sutskever, I., and Hinton, G. E., 'Imagenet classification with deep convolutional neural networks,' in Advances in Neural Information Processing Systems , Pereira, F., Burges, C. J. C., Bottou, L., and Weinberger, K. Q., Eds., vol. 25. Curran Associates, Inc., 2012. [Online]. Available: https://proceedings.neurips.cc/paper/2012/file/ c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
- 2 Sermanet, P., Eigen, D., Zhang, X., Mathieu, M., Fergus, R., and LeCun, Y., 'Overfeat: Integrated recognition, localization and detection using convolutional networks,' arXiv preprint arXiv:1312.6229 , 2013.
- 3 Girshick, R. B., Donahue, J., Darrell, T., and Malik, J., 'Rich feature hierarchies for accurate object detection and semantic segmentation,' CoRR , vol. abs/1311.2524, 2013. [Online]. Available: http://arxiv.org/abs/ 1311.2524
- 4 Girshick, R., 'Fast r-cnn,' in 2015 IEEE International Conference on Computer Vision (ICCV) , 2015, pp. 1440-1448.
- 5 Ren, S., He, K., Girshick, R., and Sun, J., 'Faster r-cnn: Towards realtime object detection with region proposal networks,' IEEE Transactions on Pattern Analysis and Machine Intelligence , vol. 39, no. 6, pp. 11371149, 2017.
- 6 Redmon, J., Divvala, S., Girshick, R., and Farhadi, A., 'You only look once: Unified, real-time object detection,' in 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2016, pp. 779-788.
- 7 Chen, X., Fang, H., Lin, T., Vedantam, R., Gupta, S., Doll´ ar, P., and Zitnick, C. L., 'Microsoft COCO captions: Data collection and evaluation server,' CoRR , vol. abs/1504.00325, 2015. [Online]. Available: http://arxiv.org/abs/1504.00325
- 8 Papers with code - coco test-dev benchmark (object detection). [Online]. Available: https://paperswithcode.com/sota/ object-detection-on-coco?metric=APS
- 9 Bochkovskiy, A., Wang, C., and Liao, H. M., 'Yolov4: Optimal speed and accuracy of object detection,' CoRR , vol. abs/2004.10934, 2020. [Online]. Available: https://arxiv.org/abs/2004.10934
- 10 Borkman, S., Crespi, A., Dhakad, S., Ganguly, S., Hogins, J., Jhang, Y.- C., Kamalzadeh, M., Li, B., Leal, S., Parisi, P. et al. , 'Unity perception: Generate synthetic data for computer vision,' arXiv preprint arXiv:2107.04259 , 2021.
- 11 Ren, S., He, K., Girshick, R., and Sun, J., 'Faster r-cnn: Towards realtime object detection with region proposal networks,' Advances in neural information processing systems , vol. 28, 2015.
- 12 Liu, W., Anguelov, D., Erhan, D., Szegedy, C., Reed, S., Fu, C.-Y., and Berg, A. C., 'Ssd: Single shot multibox detector,' in Computer VisionECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part I 14 . Springer, 2016, pp. 2137.
- 13 Redmon, J., Divvala, S., Girshick, R., and Farhadi, A., 'You only look once: Unified, real-time object detection,' in Proceedings of the IEEE conference on computer vision and pattern recognition , 2016, pp. 779788.
- 14 D'Angelo, T., Mendes, M., Keller, B., Ferreira, R., Delabrida, S., Rabelo, R., Azpurua, H., and Bianchi, A., 'Deep learning-based object detection for digital inspection in the mining industry,' in 2019 18th IEEE International Conference On Machine Learning And Applications (ICMLA) , 2019, pp. 633-640.
- 15 Usamentiaga, R., Lema, D. G., Pedrayes, O. D., and Garcia, D. F., 'Automated surface defect detection in metals: A comparative review of object detection and semantic segmentation using deep learning,' IEEE Transactions on Industry Applications , vol. 58, no. 3, pp. 4203-4213, 2022.
- 16 Kang, K., Li, H., Yan, J., Zeng, X., Yang, B., Xiao, T., Zhang, C., Wang, Z., Wang, R., Wang, X., and Ouyang, W., 'T-cnn: Tubelets with convolutional neural networks for object detection from videos,' IEEE Transactions on Circuits and Systems for Video Technology , vol. 28, no. 10, pp. 2896-2907, 2018.
- 17 Dai, J., He, K., and Sun, J., 'Instance-aware semantic segmentation via multi-task network cascades,' in 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2016, pp. 3150-3158.
- 18 Herath, S., Harandi, M., and Porikli, F., 'Going deeper into action recognition: A survey,' Image and Vision Computing , vol. 60, pp. 4-21, 2017, regularization Techniques for High-Dimensional Data Analysis. [Online]. Available: https://www.sciencedirect.com/science/ article/pii/S0262885617300343
- 19 Menikdiwela, M., Nguyen, C., Li, H., and Shaw, M., 'Cnn-based small object detection and visualization with feature activation mapping,' in 2017 International Conference on Image and Vision Computing New Zealand (IVCNZ) , 2017, pp. 1-5.
- 20 Benjumea, A., Teeti, I., Cuzzolin, F., and Bradley, A., 'Yolo-z: Improving small object detection in yolov5 for autonomous vehicles,' 2023.
- 21 Zhang, W., Wang, S., Thachan, S., Chen, J., and Qian, Y., 'Deconv r-cnn for small object detection on remote sensing images,' in IGARSS 2018 - 2018 IEEE International Geoscience and Remote Sensing Symposium , 2018, pp. 2483-2486.
- 22 Zhou, X., Xu, X., Liang, W., Zeng, Z., Shimizu, S., Yang, L. T., and Jin, Q., 'Intelligent small object detection for digital twin in smart manufacturing with industrial cyber-physical systems,' IEEE Transactions on Industrial Informatics , vol. 18, no. 2, pp. 1377-1386, 2022.
- 23 Zheng, X., Zheng, S., Kong, Y., and Chen, J., 'Recent advances in surface defect inspection of industrial products using deep learning techniques,' The International Journal of Advanced Manufacturing Technology , vol. 113, 03 2021.
- 24 L´ opez de la Rosa, F., G´ omez-Sirvent, J. L., S´ anchez-Reolid, R., Morales, R., and Fern´ andez-Caballero, A., 'Geometric transformationbased data augmentation on defect classification of segmented images of semiconductor materials using a resnet50 convolutional neural network,' Expert Systems with Applications , vol. 206, p. 117731, 2022. [Online]. Available: https://www.sciencedirect.com/science/article/ pii/S0957417422010120
- 25 Wang, J. and Lee, S., 'Data augmentation methods applying grayscale images for convolutional neural networks in machine vision,' Applied Sciences , vol. 11, no. 15, 2021. [Online]. Available: https://www.mdpi. com/2076-3417/11/15/6721
- 26 Martins, D. H., de Lima, A. A., Pinto, M. F., Hemerly, D. d. O., Prego, T. d. M., Tarrataca, L., Monteiro, U. A., Guti´ errez, R. H., Haddad, D. B. et al. , 'Hybrid data augmentation method for combined failure recognition in rotating machines,' Journal of Intelligent Manufacturing , pp. 1-19, 2022.
- 27 Salamon, J. and Bello, J. P., 'Deep convolutional neural networks and data augmentation for environmental sound classification,' IEEE Signal Processing Letters , vol. 24, no. 3, pp. 279-283, 2017.
- 28 Emam, K., Mosquera, L., and Hoptroff, R., Chapter 1: Introducing Synthetic Data Generation . O'Reilly Media, Inc., 2020.
- 29 Figueira, A. and Vaz, B., 'Survey on synthetic data generation, evaluation methods and gans,' Mathematics , vol. 10, no. 15, p. 2733, 2022. [Online]. Available: http://dx.doi.org/10.3390/math10152733
- 30 Kingma, D. P. and Welling, M., 'An introduction to variational autoencoders,' Foundations and Trends® in Machine Learning , vol. 12, no. 4, pp. 307-392, 2019. [Online]. Available: https://doi.org/10.1561% 2F2200000056
- 31 Ho, J., Jain, A., and Abbeel, P., 'Denoising diffusion probabilistic models,' in Proceedings of the 34th International Conference on Neural Information Processing Systems , ser. NIPS'20. Red Hook, NY, USA: Curran Associates Inc., 2020.
- 32 Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., and Bengio, Y., 'Generative adversarial nets,' in Advances in Neural Information Processing Systems , Ghahramani, Z., Welling, M., Cortes, C., Lawrence, N., and Weinberger, K., Eds., vol. 27. Curran Associates, Inc., 2014. [Online]. Available: https://proceedings. neurips.cc/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf
- 33 Karras, T., Aittala, M., Laine, S., H¨ ark¨ onen, E., Hellsten, J., Lehtinen, J., and Aila, T., 'Alias-free generative adversarial networks,' 2021. [Online]. Available: https://arxiv.org/abs/2106.12423
- 34 Rajpura, P. S., Bojinov, H., and Hegde, R. S., 'Object detection using deep cnns trained on synthetic images,' arXiv preprint arXiv:1706.06782 , 2017.
- 35 Kiefer, B., Ott, D., and Zell, A., 'Leveraging synthetic data in object detection on unmanned aerial vehicles,' in 2022 26th International Conference on Pattern Recognition (ICPR) , 2022, pp. 3564-3571.
- 36 Josifovski, J., Kerzel, M., Pregizer, C., Posniak, L., and Wermter, S., 'Object detection and pose estimation based on convolutional neural networks trained with synthetic data,' in 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) , 2018, pp. 62696276.
- 37 Planche, B., Wu, Z., Ma, K., Sun, S., Kluckner, S., Lehmann, O., Chen, T., Hutter, A., Zakharov, S., Kosch, H. et al. , 'Depthsynth: Real-time realistic synthetic data generation from cad models for 2.5 d recognition,' in 2017 International conference on 3d vision (3DV) . IEEE, 2017, pp. 1-10.
- 38 Nikolenko, S. I., Synthetic-to-Real Domain Adaptation and Refinement . Cham: Springer International Publishing, 2021, pp. 235-268.
- 39 Tremblay, J., Prakash, A., Acuna, D., Brophy, M., Jampani, V., Anil, C., To, T., Cameracci, E., Boochoon, S., and Birchfield, S., 'Training deep networks with synthetic data: Bridging the reality gap by domain randomization,' in Proceedings of the IEEE conference on computer vision and pattern recognition workshops , 2018, pp. 969-977.
- 40 Man, K. and Chahl, J., 'A review of synthetic image data and its use in computer vision,' Journal of Imaging , vol. 8, no. 11, 2022. [Online]. Available: https://www.mdpi.com/2313-433X/8/11/310
- 41 Redmon, J., Divvala, S., Girshick, R., and Farhadi, A., 'You only look once: Unified, real-time object detection,' 2016.
- 42 Wang, C.-Y., Bochkovskiy, A., and Liao, H.-Y. M., 'Yolov7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors,' in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , 2023, pp. 7464-7475.
- 43 Lin, T.-Y., Maire, M., Belongie, S., Bourdev, L., Girshick, R., Hays, J., Perona, P., Ramanan, D., Zitnick, C. L., and Doll´ ar, P., 'Microsoft coco: Common objects in context,' 2015.