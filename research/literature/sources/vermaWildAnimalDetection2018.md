## Wild Animal Detection Using Deep Convolutional Neural Network

Gyanendra K. Verma and Pragya Gupta

Abstract Wildlife monitoring and analysis are an active research field since last many decades. In this paper, we focus on wildlife monitoring and analysis through animal detection from natural scenes acquired by camera-trap networks. The image sequences obtained from camera-trap consist of highly cluttered images that hinder the detection of animal resulting in low-detection rates and high false discovery rates. To handle this problem, we have used a camera-trap database that has candidate animal proposals using multilevel graph cut in the spatiotemporal domain. These proposals are used to create a verification phase that identifies whether a given patch is animal or background. We have designed animal detection model using self-learned Deep Convolutional Neural Network (DCNN) features. This efficient feature set is then used for classification using state-of-the-art machine learning algorithms, namely support vector machine, k-nearest neighbor, and ensemble tree. Our intensive results show that our detection model using DCNN features provides accuracy of 91.4% on standard camera-trap dataset.

Keywords Animal detection · Camera-trap images · DCNN features · Deep learning · SVM · KNN · Ensemble tree

## 1 Introduction

Nowadays, huge data on wildlife activity and behavior can be obtained over larger space and time domain. Camera-trap methods and many other digital technologies can be used in wildlife monitoring and analysis due to relatively low cost and easyto-use. With the growth in data on wildlife, the study related to wildlife has become

G. K. Verma ( B ) · P. Gupta

Department of Computer Engineering, National Institute of Technology Kurukshetra,

Haryana, India

e-mail: gyanendra@nitkkr.ac.in

P. Gupta

e-mail: pragyagupta10@gmail.com

©Springer Nature Singapore Pte Ltd. 2018

B. B. Chaudhuri et al. (eds.), Proceedings of 2nd International Conference

on Computer Vision &amp; Image Processing

, Advances in Intelligent Systems more convenient such as studying the effects of climate change on wildlife, alterations in habitat, impact of human intervention on animals and biodiversity over different seasons, areas and species [1]. For monitoring wildlife, sensor cameras are placed on trees in a region creating a stationary camera-trap network. The camera traps are activated; each time motion is sensed, and a short video of animals activities are recorded with details about the surroundings (illumination levels, humidity, temperature, and location). Camera-trap networks are vital for acquisition of wildlife data without any disturbance. Moreover, camera-trap networks are economically feasible, easy to deploy at larger space and have low maintenance cost; as a result, they are widely used for wildlife monitoring. We can easily obtain data about the visual aspects of animal from the camera-trap images, that help to know the behavior and biometric features of species along with the relevant features related to wildlife habitat and surroundings [2]. Recently, a huge set of camera-trap images have been acquired which challenges the capability of manual annotation and image processing. There is a dire need to design multiple tools for automated processing of these huge camera-trap images such as animal identification, segmentation, extraction, and tracking. In this work, we propose a method to detect wildlife animal using CNN based on camera-trap images [3].

<!-- image -->

Object segmentation and detection from the background based on the motion of object are a necessary step for automated analysis from image sequences [4]. Several studies [5, 6] are based on background and foreground modeling for object detection, however, challenges are involved with complex dynamic scene modeling. The image sequences captured by camera traps consist of natural and dynamic scenes that are challenging to analyze using existing techniques. The natural scenes are usually highly cluttered with swinging trees, waving water, shifting shadows, changing weather, rains, etc. Also, the natural camouflage of animals poses another difficulty for analyzing natural scenes. The prime challenge for wildlife detection is to design models that can handle complex backgrounds and efficiently detect animals from dynamic scenes. Conventional approaches based on motion are inefficient with dynamic scenes.

Lately, techniques based on deep neural networks are employed for object detection such as Region-based Convolutional Neural Networks (RCNN) [7], Fast-RCNN [8], and Faster-RCNN [9]. Generally, the object detection can be divided into two steps: first; the detection of image regions using region proposal methods that may contain desired object and second; classification step that detects whether the regions contain the desired object or not. Object detection in terms of animal detection deals with problem of accuracy and speed due to the highly dynamic and highly cluttered image sequences obtained from camera traps. The available region proposal approaches [10, 11] create a huge amount of candidate regions. We observe that DCNNis computationally comprehensive and also requires performing region classification multiple times for all candidate regions. Hence, it is important to study the distinct characteristics of camera-trap image sequences in spatiotemporal domain to model an efficient region proposal approach that creates a small number of candidate regions. Hence, we used the camera-trap image sequences that are analyzed using Iterative Embedded Graph Cut (IEGC) technique to create a small group of candidate animal regions [3]. Also for better candidate animal region classification, DCNN features are used with different classifiers such as Support Vector Machine (SVM) and its variants and Ensemble classifiers.

In this paper, we have considered both the animal motion and spatial context to develop candidate animal regions using IEGC. Moreover, we have observed that DCNNimagefeaturesworkwellandimprovetheperformance of classification using different classifiers. We have designed a robust and reliable animal detection model based on camera-trap image sequences containing highly dynamic and cluttered images.

## 2 Related Work

In this section, we review the related study on the topic of wildlife detection such as foreground-background segmentation, object detection, image classification, and verification.

Existing study on background subtraction was done with the assumption of static background. Many models for background segmentation use a base frame that consists of only background explicitly. The challenge of dynamic background is handled by considering the local segmentation sensitivity using feedback loops at pixel level of images in a video [12].

We observe that reliable and efficient object segmentation and detection require the removal of assumption about stationary backgrounds. For segmentation in dynamic images, there are multiple points to consider, i.e., evaluation of scenes at pixel level, spatial context, graph cut at area level and foreground-background identification at sequential level, etc. X. Ren et al. [13] used camera-trap images that are acquired from highly dynamic images using a reliable and efficient object cut method.

Recent work in object detection and recognition display the exemplary work of DCNNs [14, 15]. To avoid the intensive analysis of whole image, region of interest is analyzed resulting fast processing of DCNN-based object detection. Some studies have been done on object detection using object proposal approach [16-19]. C. Szegedy et al. [16] used Deep Neural Network (DNN) to predict objects using bounding box object regions through regression models. Sermanet et al. [18] modeled a fully connected neural network for prediction of box coordinates and detection of objects. D. Erhan et al. [17] utilized DNN for region proposals and multi-class prediction.

The study of image verification is relevant as image verification is the problem where candidate object proposal is verified, i.e., whether animal is present or not. The process of image verification has two key phases: feature representation and distance or similarity measures. Features can be handcrafted such as HOG, HAARlike descriptors. In this work, we have used features extracted using DCNN for camera-trap data.

## 3 Algorithm Overview

Reliable and robust wildlife detection from highly dynamic and cluttered image sequences of camera-trap network is a challenging task. Hence to gain high performance, images need to be analyzed at pixel or small region level. However, due to low contrast and cluttered images, it becomes difficult to identify whether a particular region or pixel based on local information represents animal or background. Hence, we need to analyze global image features also. For example, the region of an animal body may be counted as background region. In such case, local information processing will not be sufficient, leading to requirement of global processing (to extract global image features) to detect animal. For example, recognize whether an animal is present or not, one should also identify body parts like head, legs etc. rather than body only.

## 3.1 Animal-Background Verification Model

In this work, we have designed a model that verifies the animal and background patches from the camera-trap images. The challenges associated with the model are the huge variations in background such as dynamic texture of background, change of position of irrelevant objects (like leaf, branch), illumination differences due to weather, season and shadows. Therefore, features must be invariant to all above changes. Also, our model has to work with candidate animal patches that are of variable sizes and ratios since they are obtained through ensemble graph cuts.

To handle above challenges, we present the following scheme for animalbackground verification model. Our scheme has three steps: (1) preprocessing, (2) fine-tuned DCNN features, and (3) classification through learning algorithms.

Existing literature has shown that DCNN is very efficient descriptors for object recognition, classification and retrieval, etc. [20]. There are multiple convolutional layers and at least one fully connected layer in a DCNN. For translation invariant features, DCNN has pooling layer. In our work, we pursue the architecture portrayed by Krizhevsky et al. [20] using the VGG-F pretrained model [21]. The pretrained model has been learned on huge auxiliary ILSVRC 2012 dataset. The pretrained model has an image size for input of 224 × 224; hence, we resize the images to 224 × 224, without considering its actual size and ratio. The image resize incurs image distortions which can be neglected due to the fact that all the images go through the same distortions, and the effect of resizing is negligible. The DCNN provides a feature vector of 1000 dimensions. We use DCNN features as they are self-learned features that enhance the performance of the system, and these features contain information that describes components of an image like edge, shape. In Table1, the architecture of the VGG-F model is described in detail [21].

Table 1 Architecture of CNN used (No: number of filters, Size: size of filters, st: stride, pad: spatial padding, LRN: local response normalization is used, Factor: max-pooling downsampling factor, dim: dimensionalty) [21]

| conv1                                             | conv2                                            | conv3                       | conv4                       | conv5                                       | full6                                   | full7                                   | full8                        |
|---------------------------------------------------|--------------------------------------------------|-----------------------------|-----------------------------|---------------------------------------------|-----------------------------------------|-----------------------------------------|------------------------------|
| No:64 Size:11x11 st.4, pad 0 LRN, Factor: x2 pool | No:256 Size:5x5 st.1, pad.2 LRN, Factor: x2 pool | No:256 Size:3x3 st.1 pad, 1 | No:256 Size:3x3 st.1, pad 1 | No:256 Size:3x3 st.1, pad 1 Factor: x2 pool | Dim:4096 Drop-out regularization method | Dim:4096 Drop-out regularization method | Dim:1000 Soft-max classifier |

As can be seen from Table1, there are five convolutional layers, namely conv1-5 while three fully connected layers, namely full6-8. The parameters of each layer are given in the Table1 as convolutional layer; number of filters with their size; stride value; spatial padding and down-sampling factor of max-pooling. Stride tells the allocation of spatial dimensions over the input while padding tells the size of the padding along the borders of the input in a convolutional layer. Also, the pooling layer along with a convolutional layer helps in reducing the size of the representations. Moreover, pooling aids in overcoming the problem of overfitting. Similarly for fully connected layers, the dimensionality of each layer along with the method used for regularisation is given, and in last layer, soft-max classifier is used that evaluates the deviation of output to the target [21].

## 3.2 Animal-Background Verification Model Training

The self-learned DCNN features are a 1000-dimension vector as presented in our experiment. The 1000-dimension vector is made available for each of the candidate animal proposed image region for verification. As training data, we have taken camera-trap image sequences that have bounding boxes around proposed animal region in an image (approx. 1100 images) along with some images of background (with no animal present in the scene). The patches obtained from bounding boxes constitute positive set while some random scenes with no animal are considered as negative sample that constitutes negative set. The next step is classification of the features obtained from both positive and negative sets. Classification uses machine learning algorithms such as SVMs (linear, quadratic, cubic, medium Gaussian), KNNs (cosine, weighted), and ensemble algorithms (boosted tree, bagged tree).

## 4 Experimental Results

## 4.1 Datasets

Wehave used the standard camera-trap dataset [3] for experimentation and assessing the system performance. Camera-trap images allow assessing the system for wildlife detection even in highly dynamic and highly cluttered natural images. The dataset contains 20 species of animals with around 100 image sequence for each species. The available images are present in both daytime format and nighttime format, resulting in wildlife monitoring system for both daytime and nighttime. Camera-trap networks provide complex images with highly cluttered natural videos and also high-resolution images. The images obtained vary in resolution from 1920 × 1080 to 2048 × 1536. The number of images in each sequence varies from 10 to 300 and more. The number of images in an image sequence depends on the period of action by animal. A total of 1110 patches are extracted from the dataset using the given bounding boxes and their locations.

## 4.2 Experimental Setup and Results

We use MATLAB 2016a with hardware configuration as Intel Core i7-4790 CPU @3.60 GHz (8 GB RAM) for wildlife detection system. We have used a pretrained ImageNet ILSVRC classification model with tenfold cross-validation approach. In tenfold cross-validation, whole feature dataset is divided into ten uniform folds. The primary reason for using tenfold cross-validation is to ensure that results remain unbiased to given partitioned data. Out of ten folds, nine are considered as training data and remaining one is used as test data. Hence, 90% data is used for training and 10% data is used for test purpose. The process is then repeated ten times so that each sample is used as test data. The final outcome is the average of all the ten results. Weuse 1110 images as positive samples from the camera-trap images and randomly choose background images from the Web. Figure1 displays a few examples of animal patches from camera-trap image sequences. The camera-trap database used here has bounding boxes around animal regions in the natural scenes.

We have used multiple state-of-the-art machine learning algorithms such as Support Vector Machine, K-Nearest Neighbor and ensemble classifiers and its variants. SVMs are supervised learning model; they can be used for both linear and nonlinear classification. We have used linear SVM with linear kernel function giving linearly separable planes. Also, we have used quadratic and cubic SVM that uses quadratic and cubic kernel functions and automated kernel scale. The medium Gaussian SVM uses Gaussian kernel function with kernel scale as 32. The training time and prediction speed vary for each SVM.

KNN is an instance-based algorithm widely used in various areas such as medical imaging, pattern recognition, information retrieval. We have employed cosine and weighted KNN each using ten number of neighbors. Cosine KNN uses cosine distance metric with equal distance weights, while weighted KNN uses Euclidean distance metric with squared inverse distance weights.

Fig. 1 Samples of positive images (patches from camera-trap dataset)

<!-- image -->

Table 2 System performance using Camera-Trap database (TPR: True Positive Rate, PPV: Positive Predictive Value, FDR: False Discovery Rate, AUC: Area Under ROC Curve)

| Classifier            |   Accuracy(%) |   TPR(%) |   PPV(%) |   FDR(%) |   F1-measure |   AUC |
|-----------------------|---------------|----------|----------|----------|--------------|-------|
| Linear SVM            |          90.2 |       98 |       91 |        9 |        0.945 |  0.92 |
| Quadratic SVM         |          91.1 |       97 |       93 |        7 |        0.949 |  0.90 |
| Cubic SVM             |          91.2 |       96 |       94 |        6 |        0.949 |  0.90 |
| Medium Gaussian SVM   |          90.2 |      100 |       90 |       10 |        0.946 |  0.88 |
| Cosine KNN            |          89.5 |       99 |       90 |       10 |        0.942 |  0.89 |
| Weighted KNN          |          91.4 |       98 |       92 |        8 |        0.951 |  0.91 |
| Ensemble boosted tree |          91.2 |       99 |       92 |        8 |        0.951 |  0.91 |
| Ensemble bagged tree  |          90.8 |       99 |       91 |        9 |        0.948 |  0.92 |

Ensemble classifiers are also used; that uses multiple weak learners for classification. Ensemble boosted tree is employed with adaboost ensemble method, decision tree learner, 20 number of splits, 30 number of learners, and 0.1 learning rate. Ensemble bagged tree with bag ensemble method, decision tree learner type, 30 number of learners, 20 number of splits, and 0.1 learning rate.

Table2 shows the system performance in terms of different performance evaluation metrics such as recall, precision, accuracy, F1-measure on the camera-trap database. Figure2 shows the Receiver Operating Characteristic (ROC) curves for results obtained from different classifiers.

Fig. 2 ROC curves with respect to multiple machine learning algorithms ( a Linear SVM, b quadratic SVM, c cubic SVM, d medium Gaussian SVM, e cosine KNN, f weighted KNN, g Ensemble boosted tree, h ensemble bagged tree)

<!-- image -->

We observe that DCNN features along with learning algorithm for classification enhance the system performance for animal-background verification. The verification step provides high true positive rates along with low false discovery rates.

## 4.3 Result Analysis and Discussion

We have used different performance evaluation metrics as (1) True Positive Rate (TPR) or recall, (2) Positive Predictive Values (PPV) or precision, (3) False Discovery Rate (FDR), (4) F1-measure, and (5) Area Under Curve (AUC) to asses the performance of system.

True positive rate (TPR) is the proportion of number of true positives to the total numberofpositivesamples(imagesofanimal)whilepositivepredictive values (PPV) is the proportion of number of true positives to the number of positive calls as shown in Eqs.1 and 3, respectively.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

False discovery rate (FDR) is the proportion of number of false positives to the number of positive calls as shown in Eq.3.

<!-- formula-not-decoded -->

F1-measure is the trade-off between recall and precision as shown in Eq.4. The area under curve (AUC) is a measure of the overall performance of the classifier. The higher the value of area under curve represents better classifier performance.

<!-- formula-not-decoded -->

We have achieved average accuracy 90.7% with SVMs, KNNs, and ensemble classifiers. The experiments are performed using variants of SVM such as linear SVM, quadratic SVM, cubic SVM, and medium Gaussian SVM. The highest accuracy achieved using nonlinear SVMs, i.e., quadratic and cubic is 91.1% &amp; 91.2% respectively. However, weighted KNN and ensemble boosted tree also perform well similar to SVM with 91.4% and 91.2% accuracy, respectively. We observed that proposed system obtained significant results with highly cluttered images. The overall accuracy of the system is obtained in the range of 89-91.4% and highest accuracy is 91.4% achieved with weighted KNN classifier. The result shows that DCNN features provide good result with different machine learning algorithms.

AscanbeseenfromTable2, medium Gaussian SVM outperforms for true positive rate, inline with other classifier, therefore, we can conclude that self-learned DCNN features are prominent for animal detection. The verification phase shows reduction in false discovery rate while achieving high positive predictive values as low false discovery rate is up to 6%, while, the highest rate is 10%. Also, the highest positive predictive value is 94% with cubic SVM while others have positive predictive values in the range of 90-93% that can be consider as a good performance.

Highest F1-measure is 0.951 obtained with weighted KNN and ensemble boosted tree classifiers. Also, the F1-measure is observed in the range of 0.94-0.95 approximately for all classifiers used. Hence, we can deduce that self-learned DCNN features with machine learning algorithm provide an efficient and robust system for wildlife monitoring and analysis in highly cluttered scenes.

Table 3 Performance comparison with existing animal detection systems

| Study Group           |   Year | Attributes                                                                                                      | Performance   | Performance   | Performance   |
|-----------------------|--------|-----------------------------------------------------------------------------------------------------------------|---------------|---------------|---------------|
|                       |        |                                                                                                                 | Recall        | Precision     | F-score       |
| Zhi Zhang et al. [3]  |   2016 | Combination of deep learning and HOG features encoded with fishers vector for object verification               | 0.8597        | 0.8209        | 0.8398        |
| Zhi Zhang et al. [22] |   2015 | Ensemble graph cuts for object classifier as foreground-background segmentation followed by object verification | 0.9137        | 0.8293        | 0.8695        |
| Our Work              |   2017 | DCNN features with machine learning algorithms for classification                                               | 0.9825        | 0.91625       | 0.9476        |

The results and performances of our system show that it provides an efficient and robust mechanism for wildlife detection and analysis. The animal detection has shown accuracy up to 91% with F1-measure up to 0.95. We observe that our system is robust to pose as we have taken images of animals from different views for animal-background verification. Moreover, the system works well in both daytime andnighttime as our database contains both categories of images, i.e., daytime images and nighttime images. Since the database used is camera-trap images, the system is also invariant to dynamic nature of natural scenes and invariant to cluttered images of animals.

We have obtained accuracy of 91.4% with weighted KNN and DCNN features which outperform the work in [3, 22]. Table3 shows comparison of state-of-the-art existing works with our approach. In [3], the researchers have performed animal detection using deep learning and HOG-based features for patch verification. They claimed F-score 0.839, which is lower than our study i.e. 0.951. Similarly, in [22], the researchers have applied graph cut for object classification and object verification for animal detection with F1-measure 0.8695, which is also lower than this work.

## 5 Conclusion

In this paper, we have proposed a reliable and robust method for animal detection in highly cluttered images using DCNN. The cluttered images are obtained using camera-trap networks. The images in camera-trap image sequences also provide the candidate animal region proposals done by multilevel graph cut. We have introduce a verification step in which the proposed region is classified into animal or background classes, Thus, determining whether the proposed region is truly animal or not. We applied DCNN features to machine learning algorithm to achieve better performance. The experimental results shows that proposed system is efficient and robust wild animal detection system for both daytime and nighttime.

## References

1. S. Tilak et al., Monitoring wild animal communities with arrays of motion sensitive camera, in Int. J. Res. Rev. Wireless Sensor Netw., vol. 1, pp. 1929, 2011.
2. R. Kays et al., eMammalCitizen science camera trapping as a solution for broad-scale, longterm monitoring of wildlife populations, in Proc. North Am. Conservation Biol., 2014, pp. 8086.
3. Zhi Zhang, Zhihai He, Guitao Cao, and Wenming Cao, Animal Detection From Highly Cluttered Natural Scenes Using Spatiotemporal Object Region Proposals and Patch Verification, in IEEE Transactions on Multimedia, vol. 18, no. 10, October 2016.
4. Mario I. Chacon-Murguia, and Sergio Gonzalez-Duarte, An Adaptive Neural-Fuzzy Approach for Object Detection in Dynamic Backgrounds for Surveillance Systems, in IEEE Transaction on industrial electronics, vol. 59, no. 8, August 2012.
5. Wei Liu, Hongfei Yu, Huai Yuan, Hong Zhao, Xiaowei Xu, Effective background modelling and subtraction approach for moving object detection, in IET Computer Vision, April 2014.
6. V. Mahadevan and N. Vasconcelos, Background subtraction in highly dynamic scenes, in Proc. IEEE Conf. Comput. Vis. Pattern Recog., Jun. 2008, pp. 16.
7. R. Girshick, J. Donahue, T. Darrell, and J. Malik, Rich feature hierar-chies for accurate object detection and semantic segmentation, in Proc. IEEE Conf. Comput. Vis. Pattern Recog., Jun. 2014, pp. 580587.
8. R. Girshick,Fast r-CNN, in Proc. Int. Conf. Comput. Vis., pp. 14401448, 2015.
9. K. H. Shaoqing Ren and J. S. Ross Girshick, Faster R-CNN: Towards real-time object detection with region proposal networks, in Adv. Neural Inf. Process. Syst., 2015, pp. 9199.
10. J. R. Uijlings, K. E. van de Sande, T. Gevers, and A. W. Smeulders, Selective search for object recognition, in Int. J. Comput. Vis., vol. 104, no. 2, pp. 154171, 2013.
11. M. M. Cheng, Z. Zhang, W. Y. Lin, and P. Torr, BING: Binarized normed gradients for objectness estimation at 300fps, in Proc. IEEE Conf. Comput. Vis. Pattern Recog., Jun. 2014, pp. 32863293.
12. P.L. St-Charles, G.-A. Bilodeau, and R. Bergevin, Flexible background subtraction with selfbalanced local sensitivity, in Proc. IEEE Conf. Comput. Vis. Pattern Recog. Workshops, Jun. 2014, pp. 414419.
13. X. Ren, T. X. Han, and Z. He, Ensemble video object cut in highly dynamic scenes, in Proc. IEEE Conf. Comput. Vis. Pattern Recog., Jun. 2013, pp. 19471954.
14. M. Oquab, L. Bottou, I. Laptev, and J. Sivic, Learning and transferring mid-level image representations using convolutional neural networks, in Proc. IEEE Conf. Comput. Vis. Pattern Recog., Jun. 2014, pp. 17171724.
15. A. S. Razavian, H. Azizpour, J. Sullivan, and S. Carlsson, CNN features off-the-shelf: An astounding baseline for recognition, in Proc. IEEE Conf. Comput. Vis. Pattern Recog. Workshops, Jun. 2014, pp. 512519.
16. C. Szegedy, A. Toshev, and D. Erhan,Deep neural networks for object detection, in Proc. Adv. Neural Inf. Process. Syst., 2013, pp. 25532561.
17. D. Erhan, C. Szegedy, A. Toshev, and D. Anguelov, Scalable object detection using deep neural networks, in Proc. IEEE Conf. Comput. Vis. Pattern Recog., Jun. 2014, pp. 21552162.
18. P. Sermanet et al., Overfeat: Integrated recognition, localization and detection using convolutional networks, in Proc. Int. Conf. Learn. Represent., Dec. 2013. [Online]. Available: http:// adsabs.harvard.edu/abs/2014arXiv1412.1441S.

19. C. Szegedy, S. Reed, D. Erhan, and D. Anguelov, Scalable, high-quality object detection, Dec. 2014.
20. A. Krizhevsky, I. Sutskever, and G. E. Hinton, Imagenet classification with deep convolutional neural networks, in Proc. Adv. Neural Inf. Process. Syst., 2012, pp. 10971105.
21. Ken Chatfield, Karen Simonyan, Andrea Vedaldi, Andrew Zisserman, Return of the Devil in the Details: Delving Deep into Convolutional Nets, in proceedings of BMVC 2014.
22. Zhang, Z., Han, T.X., He, Z.: Coupled ensemble graph cuts and object verification for animal segmentation from highly cluttered videos. In: 2015 IEEE International Conference on Image Processing (ICIP), pp. 2830-2834. IEEE (2015).