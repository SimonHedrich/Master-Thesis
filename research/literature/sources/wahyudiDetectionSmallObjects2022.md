## Toward Detection of Small Objects Using Deep Learning Methods: A Review

Dwi Wahyudi Department of Electrical and Information Technology Universitas Gadjah Mada Yogyakarta, Indonesia dwiwahyudi@mail.ugm.acid

## Indah Soesanti

Department of Electrical and Information Technology Universitas Gadjah Mada Yogyakarta, Indonesia indahsoesanti@ugm.ac.id Hanung Adi Nugroho Department of Electrical and Information Technology Universitas Gadjah Mada Yogyakarta, Indonesia adinugroho@ugm.ac.id

AbstractThe field of computer vision, particularly object detection, has undergone significant changes. Most cutting-edge object  detectors  can  accurately  detect  medium  and  large objects.  Small  object  detection  remains  challenging  for  the majority  of  object  detectors  due  to  low  resolution,  lack  of feature  information,  small  objects  appearing  in  unexpected areas  or  overlapping  with  other  objects,  and  small  object dataset  limitations.  Several  solutions  have  been  developed  to address this issue. This paper provides a brief description and analysis  of  contemporary  general  object  detectors,  such  as Faster  R-CNN,  SSD,  and  YOLO.  In  addition,  we  investigate several  techniques  to  improve  object  detection  performance, particularly for small object detection, from three perspectives: network improvement (multiscale feature, contextual information), input data optimization (super-resolution, image tiling), and dataset enhancement (data augmentation, creating own dataset). Implementing these techniques has been shown to improve the accuracy of contemporary object detectors, particularly for small objects.

Keywords-computer  vision,  object  detection,  small  object detection

## I. INTRODUCTION

Object  detection  is  a  significant  topic  in  the  field  of computer  vision  that  seeks  to  locate  objects  in  images  by providing bounding boxes and classifying these objects into specific classes [1]. Classification accuracy  and  object location are crucial indicators of a detection model's effectiveness [2]. Object detection methods are applicable to a  variety  of  fields,  including  autonomous  vehicles,  video surveillance, environmental conservation, and disaster mitigation, among others. On top of handcrafted features and a  shallow  training  architecture,  traditional  object  detection methods such as SIFT (Scale Invariant Feature Transform), Haar, HOG (Histogram of Gradients), and SURF (Speeded Up Robust Features) were developed. The performance of the methods is insufficient for complex tasks that combine multiple low-level image characteristics with the high-level context of the object detector and classifier [1][3]. Therefore, the emphasis in this field is on deep learning. Deep learning generates hierarchical features from raw data to high-level semantic  information  that  is  automatically  learned  from training data and exhibits enhanced expression capabilities in complex  contexts.  Moreover,  with  larger  data  sets,  deep learning can better represent features [1][4].

Object detection methods are typically divided into two approaches [5]; region proposal-based network or two-stage approach (e.g., R-CNN [6], Fast R-CNN [7] and Faster RCNN  [8])  and  one-stage  approaches  (e.g.,  SSD  [9]  and YOLO [10]) Modern object detectors  with  both  one-stage and  two-stage  approaches  have  achieved  high  levels  of precision  for  medium  and  large  objects.  However,  the majority of them still have difficulty detecting small objects. The obtained accuracy is still inadequate. In order to develop a dependable model, it is essential to enhance the precision of  small  object  detection.  Small  object  detection  is  widely used in aerial image detection, small things on the road (e.g., stones), object detection based on SAR imagery, small traffic signs, etc.

There  are  numerous  perspectives  on  the  definition  of small objects. First, small objects are defined as real-world small objects [11]. Second, based on the evaluation of the metrics  in  MSCOCO [12],  objects  are  considered  small  if their size in an image scene is smaller or equal to 32 × 32 . Moreover, small objects are those whose size is less than 1% of  the  overall  image  size [13].  Small objects,  according to SPIE (Society of Photo-Optical Instrumentation Engineers) [14],  are  objects  with  a  size  of  fewer  than  80  pixels  in  a 256 × 256 image.

Multiple  factors  make  it  difficult  or  impossible  for modern  object  detectors  to  detect  small  objects.  First,  the object's image resolution and pixel size are lower [15], so the features  do  not  contain  sufficient  information  about  the object's characteristics [2]. Second, small objects can occasionally appear in unexpected areas of an image, such as in  a  corner  or  overlapping  other  objects.  In  addition,  the complexity of the background makes it difficult to distinguish and precisely define small objects in the background  [1][15][16].  Thirdly,  the  dataset  for  detecting small objects is limited. Using a suitable dataset to train a model  can  improve  its  performance.  The  vast  majority  of existing public datasets contain data on medium and large objects, so the trained model cannot effectively detect small objects.

This article reviewed and analyzed a selection of recent articles on object detection, particularly small object detection. We examined the R-CNN, SSD, and YOLO object detection algorithms. In addition, we identify and summarize several methods to improve the performance of small object detection from three perspectives, namely network improvement  (multiscale  feature,  contextual  information), input data optimization (super-resolution, image tiling), and dataset enhancement  (data augmentation, creating  own dataset). In addition, we investigated the training datasets for supervised learning algorithms.

## II. OBJECT DETECTION METHODS

Object  detection  is  used  to  comprehend  what  is  in  an image,  to  describe  the  image's  content,  and  to  find  the location of objects. As previously mentioned, state-of-the-art object detection algorithms have produced excellent results. Object  detection  algorithms  are  typically  divided  into  two categories: two-stage and one-stage detectors. In this article, we investigate three well-known object detectors.

## A. Faster R-CNN

Faster R-CNN [8] is a two-stage object detector based on Fast R-CNN [7], one of the variants of R-CNN [6]. Regional Proposal Network (RPN) is utilized by R-CNN to generate superior regional proposals. Each convolution requires RPN to extract the features in each proposal region. RPN is a fully convolutional network that predicts both the bounding box and the value at each position simultaneously. As input, RPN applies  a 3 × 3 filter  to  a  feature  map  from  the  fifth convolution layer (conv5). RPN has been shown to improve both accuracy and speed. With a detection speed of 7 fps, faster  R-CNN  can  achieve  up  to  73.2%  accuracy  on  the PASCAL  VOC  dataset.  Faster  R-CNN  can  only  achieve 7.7% mAP for small objects on the MS COCO dataset. In addition, the two-stage network architecture can reduce the inference time of Faster R-CNN.

## B. Single Shot Multibox Detector (SSD)

Single Shot Multibox Detector (SSD) [9] is a real-time object detection method that employs one-stage deep neural network  approach.  The  SSD  architecture  is  based  on  the VGG16  architecture,  but  fully  connected  layers  are  not implemented.  The  two  primary  components  of  SSD  are feature map extraction and object detection using a convolution  layer.  Eliminating  the  network  proposal  is essential for SSD computing speed, although this reduces the mAP  value.  SSDs  results  have  fewer  errors  in  detecting locations  compared  to  R-CNN,  indicating  that  SSD  is superior in location detection because it uses regression on the  object's  shape  and  classifies  it.  SSD  is  excellent  at detecting large objects, but it is still inadequate at detecting small objects. SSD achieves a mAP of only 20.7% for small objects in the VOC2007 dataset [17]. Consequently, there is much room for improvement.

## C. You Only Look Once (YOLO)

Similar  to  SSD,  YOLO  was  created  using  a  one-stage approach.  The  YOLOv1  [10]  algorithm  divides  the  input image into a S x S grid. Each grid predicts the bounding box as  well  as  its  confidence  or  precision.  YOLOv1  provides excellent  detection  speed,  but  it  still  has  trouble  detecting small objects and  detecting object locations. In 2017, YOLOv2 or YOLO9000  [18] was proposed to address the limitations of YOLOv1. It can identify up to 9000 distinct object categories. YOLOv2 employs Darknet19 as a feature extractor to recognize smaller objects than its predecessor. Joseph Redmon and Ali Farhadi introduced YOLOv3 [19] in 2018, the successor to YOLOv2 [18]. To improve performance on smaller objects, they added object scores to bounding box predictions, added connections to the backbone layer, and made predictions at three different levels of detail. YOLOv3  employs  Darknet-53 as a feature extractor, which has 53 convolution layers. YOLOv3 results have comparable accuracy to SSDs but are three times faster. In 2020, Bochkovskiy et al. proposed YOLOv4 [20] as an enhancement of YOLOv3. YOLOv4 employs CSPDarknet53 as a backbone, SPP module and PANet as the neck, and YOLOv3 as the head. YOLOv4 is faster and more precise  than  other  object  detection  techniques.  YOLOv4 makes  training  neural  networks  on  a  single  GPU  simpler. With 512 × 512 input  images,  YOLOv4  achieves  24.3% mAP and 31 fps when detecting small objects. It is vastly superior to other detection methods.

Most object detectors have achieved good accuracy for medium and large objects, but their precision for small ones remains unsatisfactory. Table I provides a summary of the general object detection results.

## III. SMALL OBJECT DETECTION METHODS

As explained previously, detecting small objects remains an obstacle and challenge for original state-of-the-art object detectors. Several methods have been developed to enhance the  performance  of  object  detectors  in  detecting  small objects.

## A. Network Improvement Approach

## a. Multiscale Feature

In convolutional neural architecture, a number of convolution  combinations  are  followed  by  pooling  layers, which  generate  multiple  feature  map  layers  (multiscale feature maps) in the form of pyramid features. Therefore, the small object features extracted in the initial layer disappear after  several  downsampling  and  never  actually  reach  the detection and classification step. It causes small objects to fail to be detected. The multiscale feature method combines features  at  different  layers,  with  the  low  and  high  levels responsible for location and semantic information, respectively. The performance of a network can be enhanced by combining features from multiple layers, especially when detecting small objects. Lin et al. [21] introduced the Feature Pyramid Network (FPN) based on Faster R-CNN consisting of  pyramid-shaped  feature  maps  by  combining  lower  and higher layer features via a top-down path. This operation is performed on each and every feature layer. After multiple downsampling operations, however, a great deal of information is lost and difficult to recover. Deng et al. [22] proposed  the  Extended  Feature  Pyramid  Network  (EFPN) method by adding a detection layer to the top-down path and combining it with the previous layer. The process is then sent to  feature  texture  transfer  (FTT)  to  improve  the  prediction process, particularly for small objects.

Using features at all levels generates a large amount of unnecessary representation, which eventually degrades into noise, reduces performance and speed, and increases computational load. Cao et al. [23]  use the third, fourth, and fifth layers, whereas G. Cao et al. [24] only use the fourth and  fifth  layers  to  perform  feature  fusion.  Yin  et  al.  [25] proposed feature fusion and dilation SSD (FD-SSD), which is  comprised  of  a  multilayer  feature  fusion  module  to improve the semantic information of shallow features and a multi-branch residual dilated convolution module to obtain multiscale feature context information.

According to the preceding analysis, multiscale feature learning  can  improve  the  performance  of  object  detection, particularly for small objects. However, this method requires considerable  time  and  memory.    Moreover,  feature  fusion alone is insufficient to improve the performance of small NM = Not mentioned in the research papers clearly objects detection. Therefore, a number of researchers have added other techniques, such as optimization of loss function, bilinear interpolation, and non-maximum suppression (NMS) to achieve optimal results.

TABLE I SUMMARY OF GENERAL OBJECT DETECTION RESULTS

| Method       | Dataset   | Backbone         | Input Size   | FPS   | AP   |   AP 50 | AP 75   | AP S   | AP M   | AP L   |   Year | Ref   |
|--------------|-----------|------------------|--------------|-------|------|---------|---------|--------|--------|--------|--------|-------|
| Fast R-CNN   | VOC2007   | VGG-16           | NM           | 0.5   | -    |    70.0 | -       | -      | -      | -      |   2015 | [7]   |
|              | MSCOCO    | VGG-17           | NM           | -     | 19.7 |    35.9 | -       | -      | -      | -      |   2015 | [7]   |
| Faster R-CNN | VOC2007   | VGG-16           | 600x         | 7     | -    |    73.2 | -       | -      | -      | -      |   2017 | [8]   |
|              | VOC2007   | ResNet-101* 1    | 600x         | 2.4   | -    |    76.4 | -       | -      | -      | -      |   2017 | [8]   |
|              | MSCOCO    | VGG-16           | NM           | -     | 24.2 |    45.3 | 23.5    | 7.7    | 26.4   | 37.1   |   2016 | [9]   |
| SSD          | VOC2007   | VGG-16* 2        | 300          | 59    | -    |    74.3 | -       | -      | -      | -      |   2016 | [9]   |
|              | VOC2007   | VGG-16           | 512          | 22    | -    |    76.8 | -       | -      | -      | -      |   2016 | [9]   |
|              | VOC2007   | VGG-16* 3        | 300          | 23.91 | -    |    77.5 | -       | 20.7   | 62.0   | 83.3   |   2021 | [17]  |
|              | MSCOCO    | VGG-16           | 300          | 43    | 25.1 |    43.1 | 25.8    | 6.6    | 25.9   | 41.4   |   2016 | [9]   |
|              | MSCOCO    | VGG-16           | 512          | 33    | 28.8 |    48.5 | 30.3    | 10.9   | 31.8   | 43.5   |   2016 | [9]   |
| YOLOv1       | VOC2007   | VGG-16           | 448          | 21    | -    |    66.4 | -       | -      | -      | -      |   2016 | [10]  |
| YOLOv2       | VOC2007   | Darknet-19       | 288          | 91    | -    |    78.6 | -       | -      | -      | -      |   2017 | [18]  |
|              |           |                  | 352          | 81    | -    |    73.7 | -       | -      | -      | -      |   2017 | [18]  |
|              |           |                  | 416          | 67    | -    |    76.8 | -       | -      | -      | -      |   2017 | [18]  |
|              |           |                  | 480          | 59    | -    |    77.8 | -       | -      | -      | -      |   2017 | [18]  |
|              |           |                  | 544          | 40    | -    |    78.6 | -       | -      | -      | -      |   2017 | [18]  |
|              | MSCOCO    | Darknet-19       | NM           | -     | 21.6 |    44.0 | 19.2    | 9.0    | 22.4   | 35.5   |   2017 | [18]  |
| YOLOv3       | MSCOCO    | Darknet-53       | 320          | 45    | 28.2 |    51.5 | 29.7    | 11.9   | 30.6   | 43.4   |   2018 | [19]  |
|              | MSCOCO    | Darknet-53       | 416          | 35    | 31.0 |    55.3 | 32.3    | 15.2   | 33.2   | 42.8   |   2018 | [19]  |
|              | MSCOCO    | Darknet-53       | 608          | 51    | 33.0 |    57.9 | 34.4    | 18.3   | 35.4   | 41.9   |   2018 | [19]  |
| YOLOv4       | MSCOCO    | CSPDarknet-53* 4 | 416          | 96    | 41.2 |    62.8 | 44.3    | 20.4   | 44.4   | 56.0   |   2020 | [20]  |
|              | MSCOCO    | CSPDarknet-53* 4 | 512          | 83    | 43.0 |    64.9 | 46.5    | 24.3   | 46.1   | 55.2   |   2020 | [20]  |
|              | MSCOCO    | CSPDarknet-53* 4 | 608          | 62    | 43.5 |    65.7 | 47.3    | 26.7   | 46.7   | 53.3   |   2020 | [20]  |

*1 Use K40 GPU

*2 Use Titan X

*3 Use Titan Xp

*4 Use Volta GPU that can be Titan Volta or Tesla V100 GPU

## b. Contextual Information

On occasion, an object appears in a specific location and is related to other objects in an image. Small object detection relies not only on  internal characteristics but also  on information from external sources [26]. Context is additional information  required  to  improve  the  accuracy  of  object detection based on the object's relationship to others or the background. For instance, ships sail on the sea. Oliva et al. [27]  explained  that  the  area  surrounding  an  object  might provide contextual information to aid in object recognition. Yu et al. [28] also explain that adding a unique context can significantly improve detection accuracy. Contextual information  derives  data  from  the  region  surrounding  the region of interest and enhances classification by analyzing the relationship between objects and surrounding information. Small objects only occupy a small portion of the image, so the local RoI or information obtained is minimal. Lim et al. [17] propose feature fusion and attention based on SSD (FA-SSD), which focuses on small objects in the early layer by combining shallow and high-level context features with the attention module of SSD. Pyramid context learning (PCL) was proposed by Ding et al. [29]; it uses all feature contexts at all levels. However, concatenating shallow and deep features for all layers will increase computational load and  memory  consumption.  Chen  et  al.  [30]  only  use  data from four different scale feature maps (64, 32, 16, and 8) on SSD's  model  for  classification  and  regression  prediction. Leng et al. [26] proposed Internal-External Network (IENet), which evaluates the internal characteristic (appearance feature) and surrounding environment (context information) of objects for robust detection. This method includes three customized models: the Bidirectional Feature Fusion Module (Bi-FFM), the Context Reasoning Module (CRM), and the Context Feature Augmentation Module (CFAM) (CFAM).

## B. Input Data Optimization Approach

## a. Super Resolution

Super-resolution (SR) aims to reconstruct an image with low resolution (LR) and minimal information detail into a high-resolution image (HR) with clearer detailed information [31].  There  are  two  major  approaches  to  super-resolution: Single  Image  Super  Resolution  (SISR)  and  Multi-Image Super Resolution (MISR) [32]. For certain applications, the SISR approach is quick, low computational, and capable of generating sharp HR images. Super  Resolution Convolutional Neural Network  (SRCNN)  [33]  and  Fast Super Resolution Convolution Neural Network (FSRCNN) [34] are two popular SISR-based methods. Some scholars proposed Super  Resolution  with  Generative  Adversarial  (SRGAN) [37][38], which comprises generator and discriminator networks to generate HR images from LR images. Sometimes  low  resolution  image  includes  a  great  deal  of noise, so a non-local denoising algorithm is needed [36][31]. Super  Resolution  has  shown  promise  in  resolving  the problem of  detecting  small  objects.  However,  this  method requires a great deal of memory and a substantial amount of processing power.

TABLE II METHODS OF SMALL OBJECT DETECTION

| Approach                | Method                 |                                                                                                                                   | Object Detector                   | Dataset                         |
|-------------------------|------------------------|-----------------------------------------------------------------------------------------------------------------------------------|-----------------------------------|---------------------------------|
| Network Improvement     | Multiscale Feature     | Extended Feature Pyramid Network (EFPN) [22]                                                                                      | Faster R-CNN                      | TT100k (Tsinghua- Tencent 100K) |
|                         |                        | Multiscale features fusion [35]                                                                                                   | SSD                               | MS COCO                         |
|                         |                        | Multiscale features fusion [23]                                                                                                   | Faster R-CNN                      | TT100k (Tsinghua- Tencent 100K) |
|                         |                        | Multiscale features fusion [2]                                                                                                    | Faster R-CNN                      | PASCAL VOC, MS COCO, SUN        |
|                         |                        | Multiscale features fusion [24]                                                                                                   | SSD                               | VOC2007                         |
|                         |                        | Multiscale features fusion and dilated convolution [25]                                                                           | SSD                               | VOC2007, VOC2012, MS COCO       |
|                         | Contextual Information | Bidirectional Feature Fusion Module (Bi-FFM), Context Reasoning Module (CRM), and Context Feature Augmentation Module (CFAM) [26] | Internal-External Network (IENet) | MS COCO, WIDER FACE             |
|                         |                        | Feature fusion and attention [17]                                                                                                 | SSD                               | VOC2007                         |
|                         |                        | Pyramid context learning (PCL) [29]                                                                                               | SSD, Faster R-CNN, RetinaNet      | VOC2007, MS COCO                |
|                         |                        | Contextual Information Fusion [30]                                                                                                | SSD                               | NWPU VHR-10                     |
| Input Data Optimization | Super Resolution       | SR + multi-parallel modules [31]                                                                                                  | CNN                               | DIV2K                           |
|                         |                        | Fast Super Resolution Convolutional Nural Network (FSRCNN) [36]                                                                   | CNN                               | Personal traffic images dataset |
|                         |                        | Super-Resolution Convolutional Neural Network (SRCNN) [33]                                                                        | Caffe                             | ILSVRC                          |
|                         |                        | Generative Adversarial Network [37]                                                                                               | Faster R-CNN                      | Personal dataset                |
|                         |                        | SRGAN [38]                                                                                                                        | Improved Faster R- CNN            | Personal dataset                |
|                         | Image tiling           | Tiling [39]                                                                                                                       | PeleeNet                          | VisDrone2018                    |
|                         |                        | Multi-block [40]                                                                                                                  | SSD                               | ILSVRC                          |
| Dataset                 | Data Augmentation      | Copy-paste technique [41]                                                                                                         | Mask R-CNN                        | MS COCO                         |
|                         |                        | Color, geometric, and bounding box operation [42]                                                                                 | RetinaNet                         | MS COCO                         |
|                         | Create own dataset     | Image-bot [43]                                                                                                                    | YOLOv5                            | MS COCO                         |

## b. Image Tiling

Typically,  modern  object  detectors  perform  well  with large images. Memory and computational limitations prevent these models from detecting small objects in high-resolution images satisfactorily. In addition, the time required to draw conclusions  from  high-resolution  images  is  proportionally longer.

In addition, inference time for high-resolution images is correspondingly longer. A strategy based on image tiling is utilized  to  solve  this  issue.  The  method  of  tiling  can  be utilized for training, testing, or both. The core concept of this method  is  to  divide  a  high-resolution  image  into  identical tiles or blocks [40][39]. This strategy comprises three steps: Initially,  divide  the  original  input  into  overlapping  blocks.

Second, focus on the development of CNNs in which object detection is performed on each tile/block. In the final stage, detections  of  final  objects  are  made  by  combining  object suggestions from each tile in the original image resolution. Non-maximum suppression (NMS) algorithm is performed to  reduce false detection. Our analysis demonstrates that a tiling approach can improve the detection and classification of small objects. The precision improves as the size of the grid tiles grows. Unfortunately, as accuracy increases, inference speed decreases.

## C. Dataset Approach

## a. Data augmentation

The limitation of the small object data set becomes an obstacle  in  detecting  small  objects.  A  data  augmentation strategy  was  implemented  to  overcome  this  issue.  Data Augmentation  refers  to  the  transformations  applied  to  an image, such as flipping, cropping, rotating, and scaling. Data augmentation  aims  to  increase  the  variability  of  the  input image  to  improve  the  performance  of  the  object  detection model on images obtained from different environments [20]. Liu et al. [9] added small object data by resizing big ones and flipping  them  horizontally  to  increase  data  variations.  The overlap  between  predicted  anchors  and  small  ground-truth objects is significantly less than the expected threshold for IoU.  It  is  caused  by  the  infrequency  of  small  objects  in images containing small objects. Kisantal et al. [41] proposed a solution with two approaches, oversampling and augmentation, to address this issue. Oversampling is used to overcome  the  problem  of  comparatively  fewer  images containing  small  objects  during  training.  Second,  small objects are duplicated by copy-pasting technique. On the MS COCO dataset, their experiment demonstrated an increase of 9.7%  in  instance  segmentation  and  7.1%  in  small  object detection. Zoph  et al. [42] identified  22  implemented operations on TensorFlow, including Color operations, Geometric operations, and Bounding box operations. While augmentation  is  excellent  for  improving  final  detection performance,  it  can  increase  training  phase  computational complexity and load.

## b. Create Own Dataset

The majority of small and medium-sized manufacturing companies produce  their  own  specialized  product,  so  they cannot utilize existing datasets for supervised learning. But creating their own training dataset is prohibitively expensive. Block  et  al.  [43]  proposed  Image-bot,  a  combination  of hardware  and  software  to  easily  generate  a  new  synthetic dataset  from  real-world  objects.  Diverse  perspectives  of objects  are  captured  in  front  of  a  green  screen.  The  green areas  are  eliminated  and  then  randomly  combined  with various backgrounds. Variables such as illumination, camera location,  and  object  position  should  be  included  in  robust training datasets. Poisson Image Editing combines image and background since it is better than the previous algorithmic mixing techniques. The proposed method is performed on 23 objects  and  can  generate  2000  new  images  per  object  in under 45 minutes.

Table II provides a summary of the various techniques that have been shown to improve the accuracy of detecting small objects, as described in the previous section.

## IV. DATASET

Large-object dataset-trained detection methods may not be applicable when dealing with small objects. The dataset serves as a benchmark for comparing various algorithms and facilitates the achievement of the solution's intended objectives.  Using  distinct  datasets  will  result  in  distinct training and testing outcomes.

Table III presents several free public datasets that can be used  for  training,  validating,  and  testing  object  detection networks based on the research domain.

TABLE III PUBLIC DATASETS

| Purpose                      | Datasets                                                                                                         |
|------------------------------|------------------------------------------------------------------------------------------------------------------|
| General                      | MSCOCO [2], [25], [26], [30], [41]-[43] PASCAL VOC 2007 [2], [17], [24], [25], [30] PASCAL VOC 2012 [25] SUN [2] |
| HR Images                    | DIV2K [32] ILSVRC - ImageNet [34], [40]                                                                          |
| Traffic sign Pedestrian      | Tsinghua-Tencent 100K [22], [23] Caltech [38] WIDER PERSON [38] USC [38]                                         |
| Face                         | CityPerson [38] WIDER FACE [26]                                                                                  |
| Unnamed Aerial Vehicle (UAV) | VisDrone2018 [40] NWPU VHR-10 [31]                                                                               |

## V. CONCLUSION

Significant progress has been made in computer vision, particularly in object detection based on deep learning. Twostage  detectors  (e.g.,  R-CNN,  Fast  R-CNN,  and  Faster  RCNN) and one-stage detectors (e.g., R-CNN, Fast R-CNN, and Faster R-CNN) are the two most common types of object detectors (e.g., SSD and YOLO). Advanced object detectors such as Faster R-CNN, SSD, and YOLO are able to detect objects of medium and large sizes with high precision and speed.  However,  it  is  still  challenging  to  recognize  small objects  due  to  their  low  resolution  and  lack  of  feature information, as well as their unexpected location and dataset limitations.  Several  methods  are  developed  to  overcome these issues from three different perspectives, including the Network Improvement Approach (multiscale feature, contextual information), the Input Data Optimization Approach (super-resolution,  image  tiling),  and  the  Dataset Approach  (Data  Augmentation,  Create  Own  Dataset)  to provide  adequate  datasets.  Our  analysis  demonstrates  that implementing  these  techniques  can  enhance  the  object detector's performance, particularly in terms of recognizing small objects. Several techniques, including optimization of loss  function,  bilinear  interpolation,  and  optimization  of NMS  algorithm,  were  added  to improve  performance. Nonetheless, this increases computational load and decreases inference time.

## REFERENCES

- [1] X. Wu, D. Sahoo, and S. C. H. Hoi, 'Recent advances in deep learning for object detection,' Neurocomputing , vol. 396, pp. 39-64, 2020, doi: 10.1016/j.neucom.2020.01.085.
- [2] G. X. Hu, Z. Yang, L. Hu, L. Huang, and J. M. Han, 'Small Object Detection with Multiscale Features,' Int. J. Digit. Multimed. Broadcast. , vol. 2018, 2018, doi: 10.1155/2018/4546896.
- [3] G.  Chen et  al. ,  'A  Survey  of  the  Four  Pillars  for  Small  Object Detection: Multiscale Representation, Contextual Information, SuperResolution, and Region Proposal,' IEEE Trans. Syst. Man, Cybern. Syst. , pp. 1-18, 2020, doi: 10.1109/tsmc.2020.3005231.
- [4] Z. Q. Zhao, P. Zheng, S. T. Xu, and X. Wu, 'Object Detection with Deep  Learning:  A  Review,' IEEE  Trans.  Neural  Networks  Learn. Syst. , vol. 30, no. 11, pp. 3212-3232, 2019, doi: 10.1109/TNNLS.2018.2876865.
- [5] V. Sharma and R. N. Mir, 'A comprehensive and systematic look up into  deep  learning  based  object  detection  techniques:  A  review,' Comput. Sci. Rev. , vol. 38, p. 100301, 2020, doi: 10.1016/j.cosrev.2020.100301.
- [6] R.  Girshick,  J.  Donahue,  T.  Darrell,  and  J.  Malik,  'Rich  feature hierarchies for accurate object detection and semantic segmentation,' Proc. IEEE Comput. Soc. Conf. Comput. Vis. Pattern Recognit. , pp. 580-587, 2014, doi: 10.1109/CVPR.2014.81.
- [7] R. Girshick, 'Fast R-CNN,' Proc. IEEE Int. Conf. Comput. Vis. , vol. 2015 Inter, pp. 1440-1448, 2015, doi: 10.1109/ICCV.2015.169.
- [8] S. Ren, K. He, R. Girshick, and J. Sun, 'Faster R-CNN: Towards RealTime Object Detection with Region Proposal Networks,' IEEE Trans. Pattern Anal. Mach. Intell. , vol. 39, no. 6, pp. 1137-1149, 2017, doi: 10.1109/TPAMI.2016.2577031.
- [9] W.  Liu et  al. ,  'SSD:  Single  shot  multibox  detector,' Lect.  Notes Comput. Sci. (including Subser. Lect. Notes Artif. Intell. Lect. Notes Bioinformatics) , vol. 9905 LNCS, pp. 21-37, 2016, doi: 10.1007/9783-319-46448-0\_2.
- [10]  J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, 'You only look once: Unified, real-time object detection,' Proc. IEEE Comput. Soc. Conf. Comput. Vis. Pattern Recognit. , vol. 2016-Decem, pp. 779-788, 2016, doi: 10.1109/CVPR.2016.91.
- [11] K.  Tong,  Y.  Wu,  and  F.  Zhou,  'Recent  advances  in  small  object detection based on deep learning: A review,' Image Vis. Comput. , vol.

- 97, p. 103910, 2020, doi: 10.1016/j.imavis.2020.103910.
- [12]  T. Y. Lin et al. , 'Microsoft COCO: Common objects in context,' Lect. Notes Comput. Sci. (including Subser. Lect. Notes Artif. Intell. Lect. Notes Bioinformatics) ,  vol.  8693 LNCS, no. PART 5, pp. 740-755, 2014, doi: 10.1007/978-3-319-10602-1\_48.
- [13]  L. Yan, M. Yamaguchi, N. Noro, Y. Takara, and F. Ando, 'A novel two-stage deep learning-based small-object detection using hyperspectral images,' Opt. Rev. , vol. 26, no. 6, pp. 597-606, 2019, doi: 10.1007/s10043-019-00528-0.
- [14]  H. Luo, P. Wang, H. Chen, and V. P. Kowelo, 'Small Object Detection Network  Based  on  Feature  Information  Enhancement,' Comput. Intell.  Neurosci. ,  vol.  2022,  no.  6394823,  pp.  1-12,  2022,  doi: 10.1155/2022/6394823.
- [15]  C. Sun, Y. Ai, S. Wang, and W. Zhang, 'Mask-guided SSD for smallobject detection,' Appl. Intell. , vol. 51, no. 6, pp. 3311-3322, 2021, doi: 10.1007/s10489-020-01949-0.
- [16]  N. D. Nguyen, T. Do, T. D. Ngo, and D. D. Le, 'An Evaluation of Deep  Learning  Methods  for  Small  Object  Detection,' J.  Electr. Comput. Eng. , vol. 2020, 2020, doi: 10.1155/2020/3189691.
- [17] J. S. Lim, M. Astrid, H. J. Yoon, and S. I. Lee, 'Small Object Detection using Context and Attention,' 3rd Int. Conf. Artif. Intell. Inf. Commun. ICAIIC 2021 , pp. 181-186, 2021, doi: 10.1109/ICAIIC51459.2021.9415217.
- [18]  J.  Redmon  and  A.  Farhadi,  'YOLO9000:  Better,  faster,  stronger,' Proc.  -  30th  IEEE  Conf.  Comput.  Vis.  Pattern  Recognition,  CVPR 2017 , vol. 2017-Janua, pp. 6517-6525, 2017, doi: 10.1109/CVPR.2017.690.
- [19]  J. Redmon and A. Farhadi, 'YOLOv3: An Incremental Improvement,' Comput. Vis. Pattern Recognit. , 2018, [Online]. Available: http://arxiv.org/abs/1804.02767.
- [20]  A. Bochkovskiy, C.-Y. Wang, and H.-Y. M. Liao, 'YOLOv4: Optimal Speed and Accuracy of Object Detection,' 2020, [Online]. Available: http://arxiv.org/abs/2004.10934.
- [21] X.  Li,  T.  Lai,  S.  Wang,  Q.  Chen,  C.  Yang,  and  R.  Chen,  'Feature pyramid networks for object detection,' Proc. - 2019 IEEE Intl Conf Parallel Distrib. Process. with Appl. Big Data Cloud Comput. Sustain. Comput. Commun. Soc. Comput. Networking, ISPA/BDCloud/SustainCom/SocialCom 2019 ,  pp.  1500-1504,  2019, doi: 10.1109/ISPA-BDCloud-SustainComSocialCom48970.2019.00217.
- [22]  C. Deng, M. Wang, L. Liu, Y. Liu, and Y. Jiang, 'Extended Feature Pyramid Network for Small Object Detection,' IEEE Trans. Multimed. , vol. 24, pp. 1968-1979, 2022, doi: 10.1109/TMM.2021.3074273.
- [23]  C.  Cao et al. ,  'An  Improved  Faster  R-CNN  for  Small  Object Detection,' IEEE  Access ,  vol.  7,  no.  August,  pp.  106838-106846, 2019, doi: 10.1109/ACCESS.2019.2932731.
- [24]  X. Xie, G. Cao, W. Yang, Q. Liao, G. Shi, and J. Wu, 'Feature-fused SSD: fast detection for small objects,' Proc. Vol. 10615, Ninth Int. Conf. Graph. Image Process. (ICGIP 2017) , vol. 10615, pp. 381-388, 2018, doi: 10.1117/12.2304811.
- [25]  Q. Yin, W. Yang, M. Ran, and S. Wang, 'FD-SSD: An improved SSD object detection  algorithm  based  on  feature  fusion and  dilated convolution,' Signal Process. Image Commun. , vol. 98, no. March, p. 116402, 2021, doi: 10.1016/j.image.2021.116402.
- [26] J.  Leng,  Y.  Ren,  W.  Jiang,  X.  Sun,  and  Y.  Wang,  'Realize  your surroundings: Exploiting context information for small object detection,' Neurocomputing ,  vol.  433,  pp.  287-299,  2021,  doi: 10.1016/j.neucom.2020.12.093.
- [27]  A. Oliva and A. Torralba, 'The role of context in object recognition,' Trends Cogn.  Sci. , vol. 11, no. 12, pp. 520-527,  2007, doi: 10.1016/j.tics.2007.09.009.
- [28] F.  Yu  and  V.  Koltun,  'Multi-scale  context  aggregation  by  dilated convolutions,' 4th  Int.  Conf.  Learn.  Represent.  {ICLR}  2016,  San Juan,  Puerto  Rico,  May  2-4,  2016,  Conf.  Track  Proc. ,  2016,  doi: https://doi.org/10.48550/arXiv.1511.07122.
- [29]  P. Ding, J. Zhang, H. Zhou, X. Zou, and M. Wang, 'Pyramid context learning for object detection,' J.  Supercomput. ,  vol.  76,  no.  12,  pp. 9374-9387, 2020, doi: 10.1007/s11227-020-03168-3.
- [30]  J. Chen, X. Chen, L. Luo, and G. Wang, 'Conual Information Fusion
- for Small Object Detection,' Chinese Control Conf. CCC , vol. 2021July, pp. 7971-7975, 2021, doi: 10.23919/CCC52363.2021.9550159.
- [31]  T. B. Lee and Y. Seok Heo, 'Single Image Super Resolution Using Convolutional  Neural  Networks  for  Noisy  Images,' Int.  Conf.  ICT Converg. , vol. 2020-Octob, pp. 195-199, 2020, doi: 10.1109/ICTC49870.2020.9289414.
- [32]  W.  Symolon  and  C.  Dagli,  'Single-Image  Super  Resolution  Using Convolutional Neural Network,' Procedia Comput. Sci. , vol. 185, no. June, pp. 213-222, 2021, doi: 10.1016/j.procs.2021.05.022.
- [33]  C. Dong, C. C. Loy, K. He, and X. Tang, 'Image Super-Resolution Using  Deep  Convolutional  Networks,' IEEE  Trans.  Pattern  Anal. Mach. Intell. , vol. 38, no. 2, pp. 295-307, 2016, doi: 10.1109/TPAMI.2015.2439281.
- [34]  C. Dong, C. C. Loy, and X. Tang, 'Accelerating the super-resolution convolutional neural network,' Lect.  Notes  Comput.  Sci.  (including Subser. Lect. Notes Artif. Intell. Lect. Notes Bioinformatics) , vol. 9906 LNCS, pp. 391-407, 2016, doi: 10.1007/978-3-319-46475-6\_25.
- [35]  Z. Xue, W. Chen, and J. Li, 'Enhancement and Fusion of Multi-Scale Feature  Maps  for  Small  Object  Detection,' Chinese  Control  Conf. CCC , vol. 2020-July, pp. 7212-7217, 2020, doi: 10.23919/CCC50068.2020.9189352.
- [36]  I. García, R. M. Luque, and E. López, 'Improved detection of small objects  in  road  network  sequences,'  pp.  1-20,  2021,  [Online]. Available: http://arxiv.org/abs/2105.08416.
- [37]  C. Xing, X. Liang, and Z. Bao, 'A Small Object Detection Solution by Using  Super-  Resolution  Recovery,' Proc.  IEEE  7th  Int.  Conf. Comput. Sci. Netw. Technol. ICCSNT 2019 , pp. 313-316, 2019, doi: 10.1109/ICCSNT47585.2019.8962422.
- [38] Y.  Jin,  Y.  Zhang,  Y.  Cen,  Y.  Li,  V.  Mladenovic,  and  V.  Voronin, 'Pedestrian  detection  with  super-resolution  reconstruction  for  lowquality  image,' Pattern  Recognit. ,  vol.  115,  p.  107846,  2021,  doi: 10.1016/j.patcog.2021.107846.
- [39]  F. O. Unel, B. O. Ozkalayci, and C. Cigla, 'The power of tiling for small  object  detection,' IEEE  Comput.  Soc.  Conf.  Comput.  Vis. Pattern  Recognit.  Work. ,  vol.  2019-June,  pp.  582-591,  2019,  doi: 10.1109/CVPRW.2019.00084.
- [40]  Y.  LI,  H.  DONG,  H.  LI,  X.  ZHANG,  B.  ZHANG,  and  Z.  XIAO, 'Multi-block SSD based on small object detection for UAV railway scene surveillance,' Chinese J. Aeronaut. ,  vol.  33,  no.  6,  pp.  17471755, 2020, doi: 10.1016/j.cja.2020.02.024.
- [41]  M.  Kisantal,  Z.  Wojna,  J.  Murawski,  J.  Naruniec,  and  K.  Cho, 'Augmentation  for  small  object  detection,' 9th  Int.  Conf.  Adv. Comput. Inf. Technol. (ACITY 2019), December 21~22, 2019, Sydney, Aust. , vol. 9, no. 17, pp. 119-133, 2019, doi: 10.5121/csit.2019.91713.
- [42]  B. Zoph, E. D. Cubuk, G. Ghiasi, T. Y. Lin, J. Shlens, and Q. V. Le, 'Learning Data Augmentation Strategies for Object Detection,' Lect. Notes Comput. Sci. (including Subser. Lect. Notes Artif. Intell. Lect. Notes  Bioinformatics) ,  vol.  12372  LNCS,  pp.  566-583,  2020,  doi: 10.1007/978-3-030-58583-9\_34.
- [43]  L. Block, A. Raiser, L. Schön, F. Braun, and O. Riedel, 'Image-Bot: Generating Synthetic Object Detection Datasets for Small and Medium-Sized Manufacturing Companies,' Procedia CIRP , vol. 107, pp. 434-439, 2022, doi: 10.1016/j.procir.2022.05.004.