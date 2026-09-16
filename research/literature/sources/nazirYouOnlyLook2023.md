## You Only Look Once - Object Detection Models: A Review

## Aabidah Nazir

Dept. of Computer Sciences University of Kashmir Srinagar, India aabidh.nazir@gmail.com

Abstract - Object detection is the task of detecting instances of  particular  classes  in  an  image.  The  You  Only  Look  Once (YOLO)  object  detection  algorithms  have  become  popular  in recent  years  due  to  their  high  accuracy  and  fast  inference speed. In this review, an overview of YOLO variants, including YOLOv2, YOLOv3, YOLOv4, YOLOv5, YOLOv6 and YOLOv7, is performed and compared on the basis of evaluation metrics. We begin by discussing the basic principles and  architecture  of  YOLO,  which  involves  a  single  network architecture that predicts bounding boxes and class probabilities directly from full images. In addition, the changes made  in  each  version  of  YOLO,  such  as  incorporating  skip connections,  feature  pyramid  networks,  and  anchor  boxes  to improve accuracy and speed is discussed. A critical comparative analysis of YOLO variants is performed, highlighting the trade-offs between accuracy and speed. Finally, we highlight some of the future research directions for YOLO variants, such as improving their robustness to different environmental  conditions  like  motion  blur,  lighting  condition and  integrating  them  with  other  computer  vision  tasks  like image  segmentation,  image  classification  and  object  tracking. This work will help a researcher to select a version that is best suited for a given application.

Keywords - Object Detection, YOLO, DarkNet, E-ELAN, You Only Look Once

## I. INTRODUCTION

A  human  eye's  ability  to  differentiate,  recognise,  and classify the objects it sees is trivial. However, as real-world objects are extremely versatile and can take on a diversification  of  shapes,  sizes,  textures,  and  colours,  it  is challenging  for  machines  to  understand  them.  However, recent  improvements  in  computer  vision  have  changed  the process of object detection. The usage of object recognition and tracking technologies is prevalent in autonomous vehicles,  medical  diagnosis,  tracking  of  sports  balls,  and video surveillance systems, among other applications [1]. In a digital picture or video frame, object detection algorithms locate things and draw a bounding box throughout them with a  tag  indicating  to  which  class  the  object  present  in  the image belongs. However, some items might not be picked up by sensors, which could be crucial for autonomous vehicles as they need to operate with complete precision. For instance,  a  death  involving  a  self-driving  car  has  been reported.  Unable  to  sense  its  surroundings,  an  Uber  selfdriving  car  struck  a  pedestrian.  As  a  result,  perception deserves much more attention because it can make or break someone. Modern state-of-art models such as Region based Convolutional  Neural  Networks  (R-CNN),  YOLO,  SingleShot  Multi-box  Detection  (SSD),  and  other  cutting-edge

## Mohd. Arif Wani

Dept. of Computer Sciences University of Kashmir Srinagar, India awani@uok.edu.in

models  were  discovered  as  a  result  of  advances  in  neural networks  and  deep  learning  [2].  These  detectors'  primary responsibilities include producing bounding boxes, calculating class probabilities, and based on class probabilities, confidence score is calculated. This document provides  an  outline  of  the  operation  and  differences  of  the various object detection algorithms of YOLO  and  its variations. The mean Average Precision (mAP), test duration, and memory requirements are used to assess these models. The model that most closely matches the requirements  of  your  application  can  now  be  chosen  and used  in  accordance  with.  The  YOLO  algorithm  and  its modification are briefly evaluated in this study. Through the evaluation,  it  is  clear  that  various  utterances  and  accurate findings demonstrate the similarities and dissimilarities between the YOLO  version and convolutional neural networks  (CNNS).  The  development  of  the  YOLO  is  in progress  with  its  new  variants,  and  this  is  the  pertinent perception.  A  novel  method  of  object  detection  is  called YOLO.  The  classifiers  are  utilised  for  object  detection  in earlier  works.  As  a  replacement  for  object  detection,  frame object detection is taken into account as a regression problem  to  contiguous  splitted  bounding  package  container and related class probabilities. By analysing the entire set of images, a single neural network simultaneously identifies the class and the boundary bin. It being a single network allows for  optimization  of  the  complete  direction  pipeline  from beginning  to end. A  unified architecture operates very quickly  and  an  image  is  processed  by  2D  YOLO  model  in real time at a rate of 45 frames each frame for the efficacy of detection  without  delay  [3].  Scaled-down  YOLO,  which operates  at  an  astounding  125  frames  per  second,  achieves twice the coverage of the other modern real-time detectors. YOLO does  not  predict  false  positives  based  on  historical data, but it produces more localization errors than contemporary detection algorithms. YOLO gradually generalises  from  natural  images  to  further  domain  names like emulsion and becomes a widely preferred representation of stuff. It outperforms competing detection approaches like Deformable Parts Model (DPM) and R-CNN [4].

The  purpose  of  the  research  work  on  YOLO  and  its variants  is  to  improve  the  performance  and  efficiency  of object  detection  algorithms.  YOLO  is  a  popular  real-time object detection algorithm that uses a single neural network to  predict  the  bounding  boxes  and  class  probabilities  of objects  in  an  image.  YOLO  can  process  images  extremely quickly  and  is  able  to  detect  multiple  objects  in  a  single image.  Over  the  years,  researchers  have  proposed  various improvements  and  modifications  to  the  original  YOLO

algorithm, such as YOLOv2, YOLOv3, and YOLOv4. These variants  aim  to  address  some  of  the  limitations  of  the original  YOLO  algorithm,  such  as  improving  accuracy, reducing computational complexity, and increasing the detection speed. Overall, the research work on YOLO and its variants aims to improve the state-of-the-art performance of object  detection  algorithms,  making  them  more  efficient, accurate, and scalable for real-world applications.

The  sections  of  this  review  paper  is  organised  as:  In section  II,  the  Object  Detection  Model  YOLO  and  its variants are explained in detail with their architectural modifications.  In section III,  a  comparative  analysis  of YOLO and its  versions  are  discussed  with  their  evaluation metrics. In section IV, the review is concluded by convincing the performance of YOLO and its variants.

## II. YOLO AND ITS VARIANTS

YOLO is a family of object detection models developed by  Joseph  Redmon  and  his  team  at  the  University  of Washington  and  later  at  the  company  he  founded,  called YOLOv1, also known as Darknet.

YOLOv1 was released in 2016 and it was the first realtime  object  detection  model  to  achieve  high  accuracy.  It achieved this by dividing the input image into a grid of cells and  predicting  the  object  class  and  bounding  box  for  each cell.

YOLOv2  was  released in 2017 and made several improvements over YOLOv1. It introduced a new architecture called Darknet-19, which had fewer layers than the original Darknet-53 architecture used  in  YOLOv1, making  it  faster  and  more  efficient.  YOLOv2  also  used anchor  boxes  to  improve  the  accuracy  of  bounding  box predictions  and  introduced  batch  normalization  to  improve training.

YOLOv3, released in 2018, was a significant improvement over YOLOv2. It introduced a new detection architecture called Darknet-53, which was deeper and more powerful than the Darknet-19 architecture used in YOLOv2. YOLOv3  also  used  feature  pyramid  networks  (FPN)  to detect objects at different scales, improving accuracy further.

YOLOv4,  released  in  2020, was  another significant improvement  over  YOLOv3.  It  introduced  several  new techniques, including the use of the CSP (cross-stage partial) architecture  to  improve  efficiency  and  accuracy,  the  use  of SPP  (spatial pyramid  pooling)  to improve  detection  of objects at different scales, and the introduction of the Mish activation function to improve training stability.

YOLOv5, released in 2020, was developed by Ultralytics,  an  AI  software  company.  It  was  not  an  official release by the original YOLO creators. It introduced a new architecture  called  YOLOv5,  which  was  designed  to  be smaller, faster, and more accurate than YOLOv4. YOLOv5 also  introduced  the  use  of  anchor-free  detection,  which improved accuracy.

YOLOv6 and YOLOv7 are not  official  releases  by  the original  YOLO  creators.  YOLOv6  was  also  developed  by Ultralytics  and  introduced  several  new  features,  including the  use  of  self-attention  mechanisms  to  improve  detection accuracy.  YOLOv7  was  developed  by  the  community  and introduced  the  use  of  Transformer-based  architectures  for object detection. YOLO  makes  localization errors but anticipates fewer false positives in the background [5].

## A. YOLO (V1)

This object identification network was created by Redmon  et  al. [6] to simplify the enormous  run-time complexity  of R-CNN  and  its  suggested modifications. Unlike R-CNN [7] and its modified versions, YOLO divides a complete image into S×S grids and finds bounding boxes as  'm'  number  inside  each  grid,  allowing  it  to  localise  and classify objects without the need for region recommendations.  Each  bounding  box  forecasts  an  offset value and a class probability. Bounding boxes are suppressed that predict class probabilities below a specific threshold. Fig 1 provides a visual breakdown of the procedure involved in object detection using YOLO version

The major limitations of this variant are:

- A YOLO detector can only detect one thing per grid, hence  the  greatest  number  of  objects  it  can  detect always depends on the grid's dimensions. For instance, if  the  grid size is S×S, then  mostly objects can be found is S×S.
- The maximum number of objects YOLO can detect is 1  but  when  there  is  more  than  one  object  within  a grid, it makes an incorrect detection.

Fig. 1. Architecture of YOLO v1

<!-- image -->

## B. YOLO (V2)

An enhanced variant of YOLO, also known as YOLO9000, has been proposed by Redmon et al. [8]. It not only  outperforms  modern  models  like  Fast  R-CNN  and Faster  R-CNN  in  terms  of  performance  and  efficiency  but also completes detection in an acceptable length of time. To address  the  drawbacks  of  YOLO  version  1,  the  creators  of this version of the detector made a number of adjustments to the architecture.

Some  remarkable  architectural  modifications  which  are done in YOLO version 1:

- The  modifications  of  the  detector  are  enhanced  and the  possibility  of  overfitting  is  removed  with  the addition of this batch normalisation after every convolutional layer, all without the need for dropout layers.

- As  opposed  to  the  YOLO  version,  this  enhanced model eliminates using a fully connected layer from the architecture and a fully connected layer above the convolutional layer to predict the offset value, we use anchor  boxes  to  predict  the  objectness  score.  Even though YOLO version 2's mAP is lower than YOLO version  1's,  the  inclusion  of  anchor  boxes  raises  the latter's Recall value.
- Direct Location Prediction: This attribute mostly relates  to  the  method's  stability  with  the  addition  of anchor boxes, as anchor boxes to some extent increase  the  model's  instability.  The  authors  used logistic activation to confine the bounding box coordinates inside [0 1] in order to boost the model's stability.
- Fine grained features: A pass-through layer is added to  the  network  and  instead  of  using  features  with varying resolutions to execute the network, both lowand high-resolution features were viewed horizontally rather than geographically lined up and concatenated.
- Unlike the YOLO detector, that employs images with a dimension of 224×224 for training and 448×448 for testing. The  effectiveness in  performance of the YOLO  detector  version  1  is  reduced  by  the  abrupt increase  in  resolution  of  image  during  the  testing phase.  The  network  is  fine-tuned  and  trained  on images having size 448×448 for approx 10 epochs in YOLO v2 in order to overcome this issue. As a result, the  issue  with  the  rapid  rise  in  image  dimension  in YOLO  causing  a  fall  in  mean  Average  Precision (mAP) is resolved.
- YOLO version 1 trains the  network  using  bounding boxes  that  have  been  manually  annotated,  but  the authors in [6] have trained the network where bounding  boxes  are  produced  using  the  k-means technique in conjunction with their suggested evaluation metric i.e. distance to make learning more straightforward.

Fig. 2. Architecture of YOLO v2

<!-- image -->

Distance (box, centroid) = 1-IOU (box, centroid)

The standard value in [6] of "k" in their work to be 5, as this attains a reasonable commutation among the complexity and performance of the network.

## C. YOLO (V3)

An  updated  version  of  the  YOLO  detector  created  by Redmon et al. [9] is called YOLO version 3. Because YOLO version  3  only  permits  the  prediction  of  one  class  per  item and  cannot  effectively  handle  multiclass  prediction,  it  does not  employ  the  softmax  classifier  to  predict  the  classes  of observed objects. To  get around this  problem,  YOLO version 3 uses distinct logistic classifiers for each class and with  DarkNet-19  uses  a  hybrid  feature  extraction  strategy and  the  residual  network,  in  contrast  to  YOLO  v2  which utilizes  Darknet-19  as  a  feature  extractor.  The  YOLO  v3 proposed design features a number of alternative connections, which  improves  its  efficiency in terms of performance while identifying small items but diminishes it when detecting large and medium objects. Although YOLOv3  was  speedier  than  YOLOv2,  it  didn't  offer  any revolutionary improvements over its predecessor. In terms of speed,  accuracy,  and  class  specificity,  YOLOv3  and  earlier versions differ significantly. In terms of mAP and intersection over union (IOU) values, YOLOv3 is quick and accurate.  The  AP  for  small  objects  increased  by  13.3  in YOLOv3 is a significant improvement over YOLOv2. Even yet, RetinaNet still outperforms all objects (small, medium, and large) in terms of average precision (AP).

There  are  notable  gaps  among  YOLOv3  and  other variants  with  reference  to  precision/accuracy,  speed  and class specificity.

- Precision/Accuracy for tiny/small objects: The AP for tiny or small objects increased by 13.3 in YOLOv3, a significant  improvement  over  YOLOv2.  Even  yet, RetinaNet  still  outperforms  all  objects  like  small, medium-sized, and large in terms of average precision (AP).
- Speed: While YOLOv3 currently uses Darknet-53 as its backbone for feature extractor. Darknet-53 is more formidable than Darknet-19 and also more efficacious than  other  backbones  as  it  utilizes  53  convolutional layers  as  hostile  to  the  precursory  19  layers  as  in ResNet-152  or  ResNet-101.  In  connection  with  the values of IOU and mAP, YOLOv3 is expeditious and specific.

Fig. 3. Architecture of YOLO v3

<!-- image -->

- Specificity of Classes: Binary cross-entropy and Independent  logistic  classifiers  are  utilized  by  the new  YOLOv3  to  predict  classes  during  training. These changes enable the use of complicated datasets for  YOLOv3  model  training,  including  Microsoft's Open  Images  Dataset (OID). For photos in the collection,  OID  has  a  large  number  of  overlapping labels, including "man" and "person."

The YOLO v3 enables classes to be more detailed using multilable technique and to have several parameters for each bounding box and when a softmax is utilised; each bounding box can only belong to one class.

## D. YOLO (V4)

The YOLO version 4 [10] and its design was inspired by a  number  of  Bag-of-Specials  and  Bag-of-Freebies  item recognition  techniques.  Its  accuracy  is  increased  but  its inference time and training costs are increased with the Bagof-Freebies approach and, to a lesser extent, with the Bag-ofSpecials method.

There  were  also  improvements  to  the  YOLO  version  4 model in the form of genetic algorithms for selecting the best hyper-parameter  values,  SAT  (Self-Adversarial  Training) and  Mosaic  techniques  for  data  augmentation,  as  well  as adjustments  to  existing  methods  such  as  Cross  mini-Batch Normalization and Spatial Attention Module.

The CSPResNext53 and CSPDarknet50 are both built on top of DenseNet. The dense network consists of convolutional  neural  networks  connected  together,  solving the vanishing gradient problem (back-propagating loss signals  through  a  dense  network  is  difficult),  improving feature propagation, promoting reuse of features, and reducing  network  parameters  by  reducing  the  number  of layers.  CSPResNext50  and  CSPDarknet53  have  modified DenseNet so that the feature map of the plinth layer is split into two in which one copy is send through the dense block and other one to the next step.

EfficientNet performs better in terms of image classification  than  other  networks.  However,  the  authors  of YOLO  v4  postulate  that  the  networks  used  as  backbone might  shows  better  performance  in  the  models  of  object detection  and  choose  to  analyse  them  all.  The  YOLOv4 network  uses  CSPDarknet53  for  the  backbone  network  for experimental results (i.e., A LOT of experimental results).

Fig. 4. Architecture of YOLO v4

<!-- image -->

## E. YOLO (v5)

This is the first version of YOLO [11] to be created in the PyTorch  framework,  as  opposed  to  earlier  versions,  which were developed using the Darknet research framework. Due to PyTorch's ease of configuration over Darknet, version 5 of YOLO  is  now  significantly  more  production  ready  than earlier iterations.

The run-time of this version of YOLO  is  another noteworthy advancement. In comparison to its earlier suggested versions, YOLO version 5 is considerably faster. When developed using the same PyTorch library as YOLO version 5, YOLO v4's inference time is 50 frames per second while YOLO v5's is 140 frames per second. The YOLO v5 is smaller in size than YOLO v4 and thus is faster.

Like other single-stage object detectors,  YOLO  v5 contains  three  influential  components  i.e.  Neck,  Backbone and Head. The intent of Backbone is to extricate remarkable information  from  the  supplied  input  image.  To  extricate salient features from an image given as input to YOLO v5, the Cross Stage Partial Networks (CSP) are employed as its backbone.  CSPNet  has  exhibit  a  considerable  depletion  in processing time.

The key purpose of Neck is to fabricate feature pyramids. They allow models to successfully scale objects in general. Recognizing  the  same  thing  in  different  scales  and  sizes  is useful. On unobserved data, feature pyramid models perform well.  In  YOLO  v5  model  to  acquire  the  feature  pyramid, PANet is used as a neck.

The Head is used for the final detecting process. It gives final  output  vectors  and  the  vector  comprises  of  bounding using anchor boxes on the features boxes, objectness scores, and class probabilities. The heads of the YOLO V3 and V4 versions are interchangeable with the head of the YOLO v5.

## F. YOLO (v6)

YOLOv6  [12]  accomplishes  the  best  commutation  in terms of metrics like speed and accuracy. Modern quantization  techniques,  such  as  QAT  (quantization-aware training) and PTQ (post-training quantization), are investigated and integrated into YOLOv6  to increase inference speed without significantly degrading performance networks. The main modifications of YOLOv6 are as:

- Different size line networks are redesigned for industrial  applications  in  a  variety  of  contexts.  To achieve  the  greatest  speed  and  accuracy  swap,  the designs  at  different  sizes  differ,  with  tiny  models having a simple single-path  backbone  and  more models  being  constructed  on  effective  multi-branch blocks.
- A self-distillation approach is a feature of YOLO v6, and  it  is  used  for  both  classification  and  regression tasks.  To  help  the  learner  learn  knowledge  more effectively  across  all  training  phases,  the  teacher's expertise and labels are dynamically adjusted.
- Substantiate the latest methods of object detection for loss function, label assignment, and data augmentation  generally  before  deciding  which  ones to use to improve performance.

- RepOptimizer [13] and channel-wise distillation [14] are used  to reform  the quantization strategy for detection, resulting in an ever-faster and more precise object  detector  with  43.3%  MS  COCO  AP  and  a throughput with batch size of 32 is 869 FPS.

Fig. 5. Architecture of YOLO v6

<!-- image -->

## G. YOLO (v7)

The  computer  vision  and  machine  learning  are  buzzing about  the  YOLO  v7  [15]  model.  The  most  recent  and  fast algorithm of YOLO  outperforms other earlier object detection algorithms and YOLO iterations in terms of speed and precision. It can be taught significantly on tiny datasets without any pre-learned weights than other neural networks and requires technology that is several times less expensive. Therefore, YOLOv7 is anticipated to overtake YOLO v4 as the current state-of-the-art for real-time demands and become the calibre  for  object  detection  in  industries  in  the future.  The  authors  of  YOLOv7  build  on  prior  research  on the subject while taking into account the memory requirements  for  maintaining  layers  in  memory  and  the distance over  which a gradient can propagate back through the  layers;  the  smaller  the  gradient,  the  more  productively their network will be able to master. They settle on E-ELAN, an  extended  variant  of  the  ELAN  computational  block,  as their final layer aggregate. The depth, breadth, and resolution of  the  network  that  was  used  to  train  the  network  are frequently taken into account by object detection models. In YOLOv7,  the  authors  concatenate  layers  while  scaling  the network's depth and width simultaneously. Studies on ablations  demonstrate  that  this  method  maintains  the  ideal model design while scaling for various sizes.

Backbone Re-parameterization  techniques  build  a  model  that  is more resilient to the broad patterns they are trying to capture by  averaging  a  group  of  model  weights.  Module  level  reparameterization,  in  which  individual  network  components have their own re-parameterization techniques, has been the recent focus of research. The authors of  YOLOv7 experiment  with  various  levels  of  supervision  for  this  head before  deciding  on  a  coarse-to-fine  definition  in  which supervision  is  handed  back  from  the  lead  head  at  varying granularities.  Main YOLO network improvement initiatives from V1 to V7:

Fig. 6. Architecture of YOLO v7

<!-- image -->

In YOLO, the division into grids is key point of detection of objects and confidence loss. In YOLO  V2, full convolutional networks, two-stage training, and anchor using K-means  are  employed.  Multi-scale  detection  with  FPN  is employed in YOLO V3. SPP, MISH activation function, and data improvement are utilised in YOLO V4. The YOLO V5 model with its variable model size management utilizes the Hardswish  activation  function,  and  data  augmentation,  as well as Mosaic/Mixup and the GIOU (Generalized Intersection over Union) loss function. YOLO  v6 is quantized with QAT and PTQ for better speed and accuracy. The latest version of YOLO i.e. YOLO v7 is settled on EELAN.

The comparison of architectures  of  YOLO  and  variants are given Table I:

TABLE I. COMPARISON OF YOLO ARCHITECTURES

| Object Detector   | Architecture                                                                                            | Limitations                                                   |
|-------------------|---------------------------------------------------------------------------------------------------------|---------------------------------------------------------------|
| YOLO v1           | Inspired by GoogleNet and uses DarkNet framework                                                        | Localization error                                            |
| YOLO v2           | Inspired by VGG & uses DarkNet-19 framework                                                             | Small sized objects can't be identified                       |
| YOLO v3           | Inspired by Feature Pyramid Network and uses DarkNet 53 framework                                       | It lacks accuracy with medium and large sized objects         |
| YOLO v4           | Inspired by path aggregation Network and used CSP Darknet-53                                            | Difficult to deploy in embedded devices because of large size |
| YOLO v5           | Uses Focus structure with CSP DarkNet-53                                                                | The accuracy of detection and inference speed is not optimal  |
| YOLO v6           | EfficientRep Backbone and Rep-PAN Neck                                                                  | Speed and accuracy is less                                    |
| YOLO v7           | Extended Efficient Layer Aggregation Network (E- ELAN) and Model scaling for concatenation based models | -                                                             |

## III. RESULTS AND DISCUSSION

YOLO is  the  major  development  in  the  field  of  object identification  as  it  is  the  first  object  detector  model  that identifies  objects  in  only  single  stage  and  also  addresses detection  of  objects  as  a  regression  problem.  The  object detection  models  architecture  [16]  just  needed  to  take  a single  look  at  the  image  to  identify  the  elements'  locations and class labels. When benchmarked on a Titan X GPU, the basic  YOLO  model  predicts  images  at  a  rate  of  45  FPS (Frames per Second), as it is designed to train in a way that is similar to image classification from beginning to end. The fact that YOLO achieved mAP (mean average precision) of 63.4,  more  than  twice  as much  as  the  other  real-time detectors, makes it even more exceptional.

On  detection  datasets  like  MS  COCO,  YOLOv2  was trained.  By  simultaneously  training  it  on  the  ImageNet  and MS COCO datasets, the YOLO9000 was created to predict over 9000 different item categories. The enhanced YOLOv2 model outperformed cutting-edge approaches in both accuracy and speed by utilising a variety of unique techniques.  The  multi-scale  training  approach  [17]  uses  the network  as  an  input  to  predict  at  several  scales,  allowing time and accuracy to be traded off. In the VOC 2007 dataset [18], YOLOv2  achieved  76.8 mAP  at 416×416 input resolution  and  67  frames  per  second.  On  that  dataset  with dimensions  of  544×544  as  input,  YOLOv2  obtained  78.6 mAP and 40 FPS.

The developers of YOLOv3: An YOLOv2 adopted many of  the  other  strategies  from  YOLOv1  and  this  network architecture was significantly modified by Incremental Improvement.  Darknet-53  [19],  new  network  architecture, was unveiled. The Darknet-53 is a lot larger, more precise, and  swifter  network  than  the  one  it  replaced.  It  has  been honed at different image resolutions, including 320x320 and 416x416. On the Titan X GPU, YOLOv3 runs at 45 FPS and reaches  28.2  mAP  at  320320  resolutions,  making  it  more precise and rapid.

YOLOv4 is the result of numerous studies and experiments  that  combine  a  variety  of  tiny,  one-of-a-kind methods  to  increase the convolutional neural network's accuracy and speed. In addition to CSP (Cross-Stage Partial Connections), WRC (Weighted-Residual-Connections), SAT (Self-adversarial training), CmBN (Cross-mini-Batch Normalization),  DropBlock  regular  sampling,  and  Mosaic data  augmentation,  there  are  also  other  methods  which  can be  applied.  It  was  found  that  Optimal  Accuracy  and  Speed [20] of Object Detection in YOLOv4 runs twice as rapidly as EfficientDet and is having equal performance. Version 4 of the YOLO model [21] integrates features to improve model training as well as object detection accuracy.

On the datasets like PASCAL VOC and MS COCO [22], respectively,  the  experimental  findings  of  YOLO  v5.  The approach has higher large object recognition accuracy compared  to  the  four  YOLO  models.  The  fact  that  our technique on the mAP  [0.5:0.95] is 5.4%  better than YOLOv5  on  the  MS  COCO  datasets  deserves greater attention [23].

On the COCO dataset, YOLOv6-N [24] achieves 35.9% AP  at  a  throughput  of  1234  frames  per  second  using  an NVIDIA Tesla T4 GPU.

By striking at 495 FPS and 43.5% AP, YOLOv6-S beats other significant detectors on that  very scale. The  modified version [25] of YOLOv6-S adds new metrics as 43.3% AP and 869 FPS, which is even better. Additionally, YOLOv6M/L performs  better  in  terms  of  accuracy  (i.e.,  49.5%  and 52.3%)  than  other  modern  object  detectors  with  equivalent inference speeds.

The comparative analysis of YOLO models on PASCAL VOC dataset is given in Table II.

With performance spanning to 160 FPS from 5 FPS and the greatest accuracy of all modern real-time object detectors above  30  FPS on  GPU  V100  with  56.8%  You  have%  AP, YOLOv7  exceeds  all  known  object  detectors  [26].  The performance of the object detector YOLOv7-E6 with 56 FPS V100  and  55.9D44AP  is  509%  faster  and  2  percentage points  more  accurate  than  that  of  the  transformer-based detector  SWINL  Cascade-Mask  R-CNN  having  53.9%  AP and  9.2  FPS  A100.  YOLOv7  surpasses  object  detectors trained on the MS COCO dataset with reference to accuracy and speed.

TABLE II. COMPARISON OF YOLO MODELS USING PASCAL VOC DATASET

| Object Detector Model                  |   Mean Average Precision |   No of Frames Per Second |
|----------------------------------------|--------------------------|---------------------------|
| YOLO version1                          |                     63.4 |                        45 |
| YOLO version2 with input image 288*288 |                       69 |                        91 |
| YOLO version2 with input image 352*352 |                     73.7 |                        81 |
| YOLO version2 with input image 416*416 |                     76.8 |                        67 |
| YOLO version2 with input image 480*480 |                     77.8 |                        59 |
| YOLO version2 with input image 544*544 |                     78.6 |                        40 |

The  comparative  analysis  of  YOLO  models  on  MS COCO dataset is given in Table III.

TABLE III. COMPARISON OF YOLO MODELS USING MS COCO DATASET

| Object Detector Model            |   Mean Average Precision |   No of Frames Per Second |
|----------------------------------|--------------------------|---------------------------|
| YOLO v2 with input image 608*608 |                     48.1 |                        40 |
| YOLO v3                          |                     57.9 |                        20 |
| YOLO v4                          |                     65.7 |                        33 |
| YOLO v5                          |                     50.7 |                        48 |
| YOLO v6                          |                     43.1 |                       520 |
| YOLO v7                          |                     56.8 |                       160 |

There  are  several  environmental  conditions  that  can improve the robustness of the YOLO (You Only Look Once) object detection algorithm (Table IV).

TABLE IV. COMPARISON OF MODELS WITH DIFFERENT ENVIRONMENTAL CONDITIONS

|         | Accuracy using MS COCO Dataset   | Accuracy using MS COCO Dataset                             | Accuracy using KITTI Dataset   | Accuracy using KITTI Dataset                               | Accuracy using SUN-RGBD Dataset   | Accuracy using SUN-RGBD Dataset                            |
|---------|----------------------------------|------------------------------------------------------------|--------------------------------|------------------------------------------------------------|-----------------------------------|------------------------------------------------------------|
| Model   | Motion Blur (%)                  | Other environmental conditions (occlusion, truncation) (%) | Motion Blur (%)                | Other environmental conditions (occlusion, truncation) (%) | Motion Blur (%)                   | Other environmental conditions (occlusion, truncation) (%) |
| YOLO v1 | 13.7                             | 33.1                                                       | 20.3                           | 63.4                                                       | 19.8                              | 38.8                                                       |
| YOLO v2 | 20.1                             | 44                                                         | 25.2                           | 70.9                                                       | 24.7                              | 49.5                                                       |
| YOLO v3 | 21.8                             | 45.5                                                       | 25.3                           | 71.4                                                       | 26.4                              | 51.9                                                       |
| YOLO v4 | 33.8                             | 55.3                                                       | 29.8                           | 73.2                                                       | -                                 | -                                                          |
| YOLO v5 | 31.3                             | 53.4                                                       | 29.8                           | 73                                                         | -                                 | -                                                          |

Here are some of them:

- Good lighting conditions: YOLO performs best when there  is  ample  lighting  and  the  objects  are  clearly visible.
- Poor  lighting  conditions  can  cause  the  algorithm  to miss objects or detect false positives.
- Minimal  occlusion:  The  algorithm  performs  better when  objects  are  not  heavily  occluded  or  partially obstructed  by  other  objects.  Heavy  occlusion  can make it difficult for YOLO  to detect objects accurately.
- Clear and uncluttered background: Objects are easier to  detect  when  they  are  set  against  a  clear  and uncluttered background. A busy or cluttered background  can make  it difficult for  YOLO  to accurately identify objects.
- Adequate  training  data:  Adequate  training  data  is crucial  for  YOLO  to  accurately  detect  objects.  The more varied the training data, the better the algorithm can perform.
- No motion blur: Motion blur can make it difficult for YOLO to accurately detect objects. A still image or a video  with  minimal  motion  blur  is  ideal  for  the algorithm.
- Limited  camera  distortion:  YOLO  works  best  when there is minimal camera distortion. Camera distortion can  cause  objects  to  appear  distorted,  making  it difficult  for  the  algorithm  to  accurately  detect  and classify them.

Overall,  YOLO  performs  best  when  the  environmental conditions are optimized for object detection.

## IV. CONCLUSION AND FUTURE SCOPE

In  this  paper,  we  have  performed  a  critical  analysis  of object  detection  using  various  YOLO  variants.  We  review each variant from YOLO 1 to YOLO V7 examining in detail how  they  perform  single  stage  object  detection.  We  found out  that  YOLO  models  achieve  high  accuracy  and  fast inference speed, making them suitable for various real-world applications. We also found out YOLO V1 and V2 are able to  accurately  detect  large  objects,  however  the  accuracy reduces for small objects. The other variants like YOLO V3V7  actively  focused  on  this  limitation  and  achieve  good accuracy on small object detection, but identifying multiple small  objects  in  a  group  is  still  a  challenge.  The  current state-of-the-art variants of YOLO include YOLOv4, YOLOv5, and YOLOv6, have introduced several improvements  in  terms  of  accuracy,  speed,  and  efficiency. The  future  scope  of  YOLO  variants  is  vast,  with  several research directions being explored to enhance its capabilities.  One  such  direction  is  to  integrate  YOLO  with other computer vision tasks, such as semantic segmentation and  instance  segmentation,  to  build  more  comprehensive models. Additionally, researchers are also working on improving the robustness of these models to different environmental conditions and occlusion. Overall, YOLO and its variants are powerful tools for real-time object detection, and their future looks promising with ongoing research and advancements.

## ACKNOWLEDGEMENT

The  authors  are  thankful  to  the  Artificial  Intelligence Research  Center  at  the  Department  of  Computer  Science, University of Kashmir for acquiring High-performance NIVIDA  Server  (DGX  A100)  under  RUSA  2.0  grant  and providing access to it for the smooth conduction of research.

## REFERENCES

- [1] J. V. Raju, P. Rakesh, and N. Neelima, 'Driver drowsiness monitoring system,'  Intelligent  Manufacturing  and  Energy  Sustainabilit y,  pp. 675 - 683, 2020.
- [2] T.  Iqball  and  M.  A.  Wani, ' Weighted  ensemble  model  for  image classification, ' Int. j. inf. tecnol., vol. 15, pp. 557 - 564, 2023.
- [3] M. A. Sofi and M. A. Wani, ' Protein secondary structure prediction using  data-partitioning  combined  with  stacked  convolutional  neural networks and bidirectional gated recurrent units, ' International Journal of Information Technology, vol. 14(5), pp. 2285-2295, 2022.
- [4] M. Maity, S. Banerjee, and S. Sinha Chaudhuri, 'Faster R -CNN and YOLO  based  Vehicle  detection:  A  Survey,' In  Proc  of  the  5th International Conference on Computing Methodologies and Communication, ICCMC 2021, Apr. 2021, pp. 1442 - 1447.
- [5] S.  Geethapriya,  N.  Duraimurugan,  and  S.  P.  Chokkalingam,  'Real time object detection with yolo, ' International Journal of Engineering and Advanced Technology, vol. 8, pp. 578-581, 2019.
- [6] J.  Redmon, S. Divvala, R. Girshick, and A. Farhadi, 'You only look once:  Unified,  realtime  object  detection,' In  Proc.  of  the  IEEE Computer  Society Conference on Computer  Vision and Pattern Recognition, Dec. 2016, vol. 2016-December, pp. 779 - 788.
- [7] K.  He,  X.  Zhang,  S.  Ren,  and  J.  Sun,  'Spatial  Pyramid  P ooling  in Deep Convolutional Networks for Visual Recognition,' IEEE Trans. Pattern Anal. Mach. Intell., vol. 37, no. 9, pp. 1904 - 1916, Sep. 2015.
- [8] J.  Redmon and A. Farhadi, 'YOLO9000: Better, faster, stronger,' In Proc.  of  the  30 th IEEE  Conference  on  Computer  Vision  and  Pattern Recognition,  CVPR 2017,  Nov.  2017,  vol.  2017-January,  pp.  6517 - 6525.
- [9] J. Redmon and A. Farhadi, 'YOLOv3: An Incremental Improvement,' unpublished, Apr. 2018, [Online]. Available: http://arxiv.org/abs/1804. 02767
- [10] A. Bochkovskiy, C.-Y. Wang, and H.- Y. M. Liao, 'YOLOv4: Optimal Speed  and  Accuracy  of  Object  Detection,'    Apr.  2020,  [Online]. Available: http://arxiv.org/abs/2004.10934
- [11] G . Jocher, 'ultralytics/yolov5,' GitHub, Aug. 21, 2020. https://github. com/ultralytics/yolov5.
- [12] C. Li et al. ,  'YOLOv6: A Single -Stage Object Detection Framework for Industrial Appli cations,' Sep. 2022, [Online]. Available: http://arxiv.org/abs/2209.02976
- [13] X.  Ding,  H.  Chen,  X.  Zhang,  K.  Huang,  J.  Han, and  G.  Ding,  'Re - parameterizing  Your  Optimiz ers rather than Architectures,' May 2022, [Online]. Available: http://arxiv.org/abs/2205.15242
- [14] C.  Shu,  Y.  Liu,  J.  Gao,  Z.  Yan,  and  C.  Shen,  'Channel -wise Knowledge Distillation for Dense Prediction,' In Proc. of the IEEE/CVF  International  Conference  on  Computer  Vision  (ICCV), 2021.
- [15] C.-Y. Wang,  A.  Bochkovskiy, and H.-Y.  M.  Lia o,  'YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors,' ,  Jul.  2022,  [Online].  Available:  http://arxiv.org/abs/2207. 02696
- [16] I.  Bello  et  al. ,  'Revisiting  ResNets:  Improved  Training  and  Scaling Strategie s,'  2021.  [Online].  Available: https://github.com/tensorflow/ tpu/tree/master/
- [17] K.  Chen,  W.  Lin,  J.  Li,  J.  See,  J.  Wang,  and  J.  Zou,  'AP -Loss  for Accurate  OneStage  Object  Detection,' IEEE  Trans.  Pattern  Anal. Mach.  Intell.,  vol.  43,  no.  11,  pp.  3782 - 3798,  Nov.  2021,  doi: 10.1109/TPAMI.2020.2991457.
- [18] D.  Jia,  W.  Dong,  R.  Socher,  L.-J.  Li,  K.  Li,  and  F.-F.  and  Li, 'ImageNet:  A  Large - Scale  Hierarchical  Image  Database,'  in IEEE conference on computer vision and pattern recognition, 2009.
- [19] 'Darknet:  O pe n Source Neural Networks in C,' pjreddie.com. http://pjreddie.com/darknet/

- [20] J.  Huang et al. ,  'Speed/accuracy trade -offs for modern convolutional object  detectors  Jonathan,'  in IEEE  conference  on  computer  vision and pattern recognition, 2017, vol. 84, no. 3 - 4, pp. 7310 - 7311.
- [21] J.  Guo  et  al. ,  'Hit -Detector: Hierarchical Trinity Architecture Search for  Object  Detection,' In  Proc.  of  the  IEEE/CVF  Conference  on Computer Vision and Pattern Recognition (CVPR), 2020.
- [22] T.  Y.  Lin  et  al. ,  'Microsoft  COCO:  Common  objects  in  context,' Computer Vision - ECCV 2014, 2014.
- [23] G. Jocher, 'Releases ultralytics/yolov5,' GitHub, 2022. https://github. com/ultralytics/yolov5/releases.
- [24] D.  Feng  et  al. ,  'Deep  Multi -Modal  Object  Detection  and  Semantic Segmentation for Autonomous  Driving: Datasets, Methods, and Challenges,' IEEE  Trans.  Intell.  Transp.  Syst.,  vol.  22,  no.  3,  pp. 1341 - 1360, Mar. 2021.
- [25] M. Hu et al. , 'Online Convolutional Reparameterization,' In Proc. of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 558 - 567.
- [26] C.  Y.  Wang,  A.  Bochkovskiy,  and  H.  Y.  M.  L iao,  'Scaled -yolov4: Scaling  cross  stage  partial  network, '  In  Proc.  of  the IEEE/CVF Conference  on  Computer  Vision  and  Pattern  Recognition  (CVPR), 2021.