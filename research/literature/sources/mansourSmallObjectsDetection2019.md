## Small Objects Detection in Satellite Images Using Deep Learning

Ahmad Mansour Mechatronics Engineering Military Technical College Cairo, Egypt Ahmad.mansour\_44@yahoo.com Wessam M Hussein Mechatronics Engineering Military Technical College Cairo, Egypt Wessam\_hussein@mtc.edu.eg Ehab Said Mechatronics Engineering Military Technical College Cairo, Egypt ehab\_said@mtc.edu.eg

AbstractUsing  the  deep  Convolution  Neural  Networks (CNNs)  for  Object  detection  in  satellite  images  accomplish promising  results,  especially  for  large  objects.  While  Small objects detection in the same spatial resolution images does not attain the same results. For instance, vehicle detection in highresolution satellite images, the targeted object maybe existed in an  area  that  does  not  exceed  15  square  pixels,  which  will  not make  a  sufficient  effect  in  the  deeper  layers.  In  addition;  the interfering  with  the  surrounding  background,  noise  effect,  the neighboring object's shadows, and various vehicle colors. In the proposed paper, an analysis study is performed to evaluate the effect  of  changing  the  object  size  on  the  detection  results.  A separate resampling algorithm is applied to the input test images to  change  its  size  -  bear  in  mind  the  built-in  detection  model resampling  layer-,  which  results  in  changing  the  object  size, and  accordingly  extends  the  object  impact  in  deep  layers. Through  Transfer  Learning,  the  Faster  R-CNN  pre-trained object  detection  model  with  Inception-V2  is  applied  to  submeter  satellite  images  and  passenger  vehicles  as  the  target objects. The Experimental results show the change in detection accuracy with the change of the object size.

Keywords-Object  detection,  Convolution  Neural  Networks, deep learning, Satellite image.

## I. INTRODUCTION

Satellite  images are  a rich  source  of  information,  used  in many fields, urban planning applications, environmental applications, traffic control, and military applications[1]. These images generated from capturing the light sensors the electromagnetic  waves  reflected  from  the  objects,  and  store them  in  pixels-shaped  matrices  which  represent  the  raster images [2,3].

Raster  image  resolution  depends  on  the  sensor's  physical characteristics,  which  can  be  expressed  in  four  terms.  The temporal resolution, spectral resolution, radiometric resolution and the spatial resolution which describes the image real pixel size [4]. The last one called also the Ground Sample Distance (GSD) in aerial and satellite images, which properly describes the  minimum  separation  distance  between  two  objects  to differentiate  among  them.  Frequently  the  pixel  size  gives  a wrong  indication  for  the  spatial  resolution,  as  a  result  of performing some image processing operations to enhance the image visualization [5].

Most of earth observation satellites operate on low altitude orbits, between 500 and 700-kilometers height [6], and have various  spatial  resolutions,  starts  from  over  one  kilometer  to 30 centimeters, depending on its application, with radiometric resolutions between hundreds to 4 bands [7,8], and different images swathes.

Rapid  development  in  the  computer  vision  and  digital cameras, contemporaneous with computers capabilities progress emerges for the arena the concept of deep learning, as  the  development  of  machine  learning,  which  has  been already available since 1985 but not efficiently used. Convolution Neural Networks (CNNs)[9,10] - which are the base  of  deep  learning - were  inspired  by  the  human  brain interactions  with  neurons  connection[11],  where  one  neuron activates  another  neuron,  exactly  the  same  way,  the  CNNs were built.

Object  detection  task  could  be  addressed  into  two  subtasks,  the  first  one  is  identifying  the  objects  in  the  image, which  called  image  classification  according  to  the  training samples,  the  second  task  is  localizing  these  objects  in  the image.  Object  detection  based  on  CNNs  proposed  in  Many algorithms and architectures[12-15].

Two main series used in object detection assignment. The first  series depends on a combination of region proposal and classification  as  a  two-stage  object  detection  framework, Represented  by  R-CNN,  including  Fast  and  Faster  R-CNN [16], SPP-NET  [17],  and  others. The  other series uses convolution  neural  networks  to  make  object  detection  in  a single-stage framework, which is represented by YOLO (You Only  Look  Once)  [18],  and  SSD  (Single  Shot  Multi-Box Detector  [19].  The  last  series  usually  used  with  the  data streams like videos, because it presents a high-speed performance  but  with  detection  accuracy  less  than  the  first series.

The  neural  networks  in  deep  learning  models  used  in object detection, designed to manage fixed image sizes, to be able to perform the classification and localization tasks for the targets.  That's  why  the  models  take  any  input  image  and resize it to the predefined size. Here, the problem of the small objects  appears.  For  instance,  using  a  relatively  large  image size  in  the  model,  to  fit  the  specified  input  size  the  model resizes this image generating a new image in different spatial resolution,  which  results  in  changing  the  number  of  pixels occupied by the objects. These objects become smaller. And as  a  result  of  applying  the  Convolution  layers  and  MAXPOOL layers in the model workflow, the objects' effect in the later layers will vanish [20-22], accordingly the classification layer will not be able to effectively detect these objects.

In this paper, the proposed technique depends on increasing  the  object  area  in  pixels,  by  partitioning  the  test image  into  smaller  images,  and  depending  on  the  built-in resample layer in  Faster  R-CNN model  - the most popular technique  for  object  detection  in  the  first  series  -  to  adapt the  image  to  the  desired  model  image  size,  with  suitable overlap  between  partitioned  images.  Then  import  the  test images into the model. Finally,  measuring the impact on the detection  accuracy  and  confidence  level,  to  decide  the  most suitable object size to be efficiently detected.

II. THEORETICAL OVERVIEW.

## A. Deep Learning.

The conventional method of solving mathematical tasks, to obtain  an  output  from  known  input,  is  applying  a  known algorithm to the input to get an output, both the input and the algorithm are known and the output is the unknown part.

Machine learning does not work that way, here the situation is the input and the output are known and we want to train an algorithm,  to  map  the  output  to  the  input.  This  seems  to  be easy  in  linear  relations,  but  in  non-linear  relations,  it  is  very complicated [23].

<!-- image -->

The Requires hyperparameters: Number of filters K ,  Filter size F, Stride S, Output volume size: W2×H2×D2









<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Fig. 1. Flow of data through a deep neural network.

<!-- image -->

The  training  process  is  really  about  tuning  the  internal variables of  the  network, to  the  best  possible  values,  so  that the  algorithm  can  map  the  input  to  the  output  [24].    This  is achieved  through  an  optimization  process  called  Gradient Descent,  which  uses  Numeric  Analysis  to  find  the  best possible  values  to  the  internal  variables  of  the  model.  As shown in figure 1, starting with a forward pass, after feeding the  neural  network  with  the  input.  Then  the  model  applies convolution operation to the input with its internal variables to predict the answer.

Once the output value is predicted, the loss is estimated as the difference between that predicted value and the true value. This loss measures the model performance. Then adjusting the internal  variables  (W,  B)  for  all  layers,  to  diminish  the  loss, and  making  the  predictions  convenient  as  possible  to  the correct value [25].

## B. Convolution over RGB image.

To  recognize features  in  the  RGB  image,  the  model convolves  the  image  by  a  matrix  which  represents  the  filter. So, the filter  should  be  3-D  analogous  to  the  R,  G,  B channels.- as shown in figure 2- the number of channels in the applied filters and the image depth must match. The number of the  applied  filters  decides  the  number  of  convolution  output depth [26,27].

Input volume size:

W1×H1×D1

Fig. 2. Convolution over 3-D image.

## C. Pooling Layer

Pooling layers are important bricks of CNNs. Its function is to  progressively  shrink  the  size  of  its  input  to  minimize  the number of used parameters and reduce the computations in the network. Pooling layers operate on each feature map independently.  Two  kinds  of  pooling  layers  Are  used,  Max and  Mean-Pooling.  The  most  common  approach  used  in pooling is the Max-Pooling, which takes the maximum pixel value  of  the  matrix  under  the  convolved  filter[28].  Using kernel 2×2 filter reduces the image size to half - as shown in figure 3-, and using a 3×3 filter reduces the size to one-third [29].

Input volume size W 1× H 1× D 1 Requires two hyperparameters:

The spatial extent F&amp; The stride S.

The Output volume size: W2×H2×D2

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->









Fig. 3. Illustration of Max-Pooling Algorithm.

<!-- image -->

## D. Faster R-CNN

The Faster  R-CNN split  into  three  successively  networks. The first  network  is  the  base  network,  used  for  feature  map production to generate proper features from the input images, that  uses  a  pre-trained  CNN  for  the  classification  task  [30]. This mode is commonly used in training a classifier in case of small  datasets  utilizing  the  weights  of  another  pre-trained network on a bigger dataset.

Fig. 4. Illustration of Faster R-CNN model.

<!-- image -->

The  base  network  used  in  this  approach  is  Inception-V2. The second network is what called Region Proposal Network (RPN)  is  used  to  generate  region  proposals  with  different extents and different aspect ratios for the input image. The last network is the Fast R-CNN detector, which takes the Regions Of Interest (ROIs), that previously generated  from  the  RPN. Then  the  (ROI)  layer  generates  a  feature  vector  for  each Region. The generated features are fed into Fully  Connected (FC)  layers  then  applying  Regression  and  SoftMax layers  to determine the position of the bounding boxes of the classified objects  [31].  Faster  R-CNN  Inception-V2  model  accepts different images size and then apply the resampling algorithm to resize the images to (1,024x600) pixels.

## III. IMPLEMENTATION

The used model, have been obtained from the open-source framework Tensor-flow.

Faster R-CNN models with Inception-V2 as base network Object Detection APIs.

## A. Faster R-CNN Inception-V2.

the Learning Rate (LR) used set to start with LR=0.0001 till step number 40,000. Then changing LR the next 40,000 steps to  be LR=0.00001, and end with LR= 0.000001, For the last 2000 steps.

The total number of steps is 100K step. With applying data augmentations  horizontal  flip  and  random  rotation  for  the training dataset. Setting the batch size to one.

## B. The Proposed Workflow.

Fig. 5. Test Image proposed workflow

<!-- image -->

High-resolution satellite images could exceed 10,000 x 10,000 pixels, to apply object detection model to such an image it must be split into smaller images.

To ensure that the Faster R-CNN Inception-V2 model will not change the image pixel size, the test image will be cut and resampled using a separated resample algorithm to (1,024x600)  pixels  image  to  represent  the  unit  image  in  the experiment,  then  Splitting  the  requested  to  valuation  input image  to  smaller  images  with  different  sizes  with  enough overlap to ensure that no object has been lost in image cutting - this step is which decides the object size - , then applying the selected object detection model on each single image and the original one.

Image  Resampling  is  a  common  method  used  in  image processing, to enhance the image visualization, without changing the image spatial resolution, by breaking down the image  pixels  into  different  pixels  size  and  orientation,  the newly generated pixels calculated in different methods, Nearest Neighbor, Bilinear Interpolation, Cubic Convolution, and Bicubic Spline [32,33].

The  models  perform  its  built-in  resizing  function,  which subsequently changes the object's area of pixels in the divided image - see figure 6 -.

Finally, recording and evaluating the detection Results and confidence  levels,  and  making  an  experimental  comparison between results.

Fig. 6. Difference in pixel size after Image Resample

<!-- image -->

## C. Training and testing Datasets.

Object detection  models are applicable  to any  kind  of 3band  images,  digital  cameras  images,  satellite  images,  aerial images, and UAV images.

Selecting  satellites  images  to  the  experiment,  because  of the  ability  to  geolocate  these  images,  to  represent  a  good measuring  for  the  object  size  in  pixel  and  reference  GSD values.

The  Datasets  used  in  this  paper  are  collected  from  free sources satellite image for different locations on earth, Google Earth  and,  WORLD-VIEW-4  0.3-meter  GSD,  WORLDVIEW-3    0.3-meter  GSD  and,  Super-View  0.5-meter  GSD, Geo-Eye-1 0.5-meter GSD and Pleiades 0.7-meter GSD. Also, UAVs and aerial images are used.

Fig. 7. Same object in different spatial resolution (GSD)

<!-- image -->

Training  the  chosen  model  on  247  images  with  5,487 instances of labeled vehicles. Selecting 548 images to be the testing dataset and 4,939 as training. The training and testing datasets  collected  from  different  lighting  and  environment conditions and different spatial resolution satellites to increase the validity and reality of the attempts, all the used images are sub-meter GSD.

## IV. RESULTS.

Experimental  evaluation  of  the  results  is  performed  by Mean Average Precision (MAP) before and after changing the sizes of the objects.

MAP  computes  the  value  of  the  average  precision  and Recall.

The Precision measures prediction accuracy(correct positive predictions among all positive cases in reality), The Recall  measures  how  good  the  positives  results  to  the  total results.





<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Where: TP= True Positive, TN= True Negative, FP= False Positive and FN= False Negative.

Fig. 8. Vehicles detection in the Original Image (1024*600).

<!-- image -->

Fig. 9. Vehicles detection in the (682*400) Image.

<!-- image -->

Fig. 10. Vehicles detection in the (512*300) Image.

<!-- image -->

In  the  performed  comparison  between  results,  the  best result obtained with the first proposed object size, the precision increased by 7.2% from the basic precision, to reach 87.6 %.

The time consumed increased to be 550 milliseconds from 300 milliseconds in the original image.

The recall value decreases with 1.5%, due to the detection of some wrong objects.

96.1

Fig. 11. Different objects sizes detection result.

<!-- image -->

Increasing the object length by 50% gives the best results in detection Precision, according to the vehicles samples used in  the  test,  further  size  increase  resulted  in  an  exponential decrease in Recall. The mean suggested size to a small object to be effectively detected about (10 x 20) pixel.

Up-Sampling the test images pixel size makes the detection  easier  but  increases  the  processing  time.  Excessive Up-Sample gives a lot of wrong detections.

The training quality is the most important parameter in the experiment.

## V. CONCLUSION

In this paper, a comparison is made based on the sizes of the  objects,  for  small  object  detection  in  the  high-resolution satellite image, using state of the art algorithms Faster R-CNN Inception-V2 object detection model, which considered one of the most popular object detection models, choosing vehicles in these images to express the target objects. The presented study based on increasing the object area in pixels to keep the object effect in the detection models deep layers which is a quite new technique,  that  generally  increases  the  detection  precision, with  a  slight  increase  in  processing  time,  to  select  the  most effective  object  size.  Training  quality  should  be  taken  into consideration,  that  is,  it  is  the  most  important  parameter  in object detection. The testing and training datasets are chosen from open-sources digital images on the internet. Applying the proposed solution to any kind of image will be performed in the same procedures. The Mean Average Precision (MAP) is used in the performance evaluation..

## REFERENCES.

- [1] (Dial, Gene, et al. "IKONOS satellite, imagery, and products." Remote sensing of Environment 88.1-2 (2003): 23-36.
- [2] Asra, Ghassem. Theory and applications of optical remote sensing. Ed. Ghassem Asrar. New York: Wiley, 1989.
- [3] Landgrebe,  David  A.  Signal  theory  methods  in  multispectral  remote sensing. Vol. 29. John Wiley &amp; Sons, 2005.
- [4] ERDAS IMAGINE. "ERDAS Imagine 2013 User Guide." (2013).
- [5] Hood, Joy, Lyman Ladner, and Richard Champion. "Image processing techniques  for  digital  orthophotoquad  production." Photogram.  Eng. Remote Sens 55.9 (1989): 1323-1329.
- [6] Capderou, Michel. Satellites: Orbits and missions. Springer Science &amp; Business Media, 2006.
- [7] Govender,  M.,  et  al.  "A  comparison  of  satellite  hyperspectral  and multispectral  remote  sensing  imagery  for  improved  classification  and mapping of vegetation." Water SA 34.2 (2008): 147-154.
- [8] Aplin, P., Peter M. Atkinson, and P. J. Curran. "Fine spatial resolution satellite  sensors  for  the  next  decade." International  journal  of  remote sensing 18.18 (1997): 3873-3881.
- [9] LeCun, Yann, Yoshua Bengio, and Geoffrey Hinton. "Deep learning." nature 521.7553 (2015): 436.
- [10] Schmidhuber, Jürgen. "Deep learning in neural networks: An overview." Neural networks 61 (2015): 85-117.
- [11] Cox, David Daniel, and Thomas Dean. "Neural networks and neuroscience-inspired computer vision." Current Biology 24.18 (2014): R921-R929.
- [12] Zheng,  Hong,  Li  Pan,  and  Li  Li.  "A  morphological  neural  network approach for vehicle detection from high-resolution satellite imagery." International  Conference  on  Neural  Information  Processing.  Springer, Berlin, Heidelberg, 2006.
- [13] Chen, Xueyun, et al. "Vehicle detection in satellite images by parallel deep convolutional neural networks." 2013 2nd IAPR Asian Conference on Pattern Recognition. IEEE, 2013.
- [14] Hinz,  Stefan.  "Detection  and  counting  of  cars  in  aerial  images." Proceedings 2003 International Conference on Image Processing (Cat. No. 03CH37429). Vol. 3. IEEE, 2003.
- [15] Erhan,  Dumitru,  et  al.  "Scalable  object  detection  using  deep  neural networks." Proceedings of the IEEE conference on computer vision and pattern recognition. 2014.
- [16] Ren, Shaoqing, et al. "Faster r-cnn: Towards real-time object detection with region proposal networks."  Advances  in neural information processing systems. 2015.
- [17] Purkait,  Pulak,  Cheng  Zhao,  and  Christopher  Zach.  "Spp-net:  Deep absolute pose regression with synthetic views." arXiv preprint arXiv:1712.03452 (2017).
- [18] Redmon, Joseph, et al. "You only look once: Unified, real-time object detection." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
- [19] Liu, Wei,  et al. "Ssd:  Single  shot  multibox  detector."  European conference on computer vision. Springer, Cham, 2016.
- [20] Zeiler,  Matthew  D.,  and  Rob  Fergus.  "Visualizing  and  understanding convolutional  networks."  European  conference  on  computer  vision. springer, Cham, 2014.

- [21] Long, Jonathan, Evan Shelhamer, and Trevor Darrell. "Fully convolutional networks for semantic segmentation." Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.
- [22] Hinton,  Geoffrey  E.,  and  Ruslan  R.  Salakhutdinov.  "Reducing  the dimensionality of data with neural networks." science 313.5786 (2006): 504-507.
- [23] Bengio, Yoshua. "Learning deep architectures for AI." Foundations and trends® in Machine Learning 2.1 (2009): 1-127
- [24] Snoek,  Jasper,  Hugo  Larochelle,  and  Ryan  P.  Adams.  "Practical bayesian  optimization  of  machine  learning  algorithms."  Advances  in neural information processing systems. 2012.
- [25] Kim, Hoyong, Yunseok Ko, and K-H. Jung. "Artificial neural-network based feeder reconfiguration for loss reduction in distribution systems." IEEE Transactions on Power Delivery 8.3 (1993): 1356-1366.
- [26] Simonyan,  Karen,  and  Andrew  Zisserman.  "Very  deep  convolutional networks for large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014).
- [27] Sangwine, Stephen J., and Todd A. Ell. "Colour image filters based on hypercomplex convolution." IEE Proceedings-Vision, Image and Signal Processing 147.2 (2000): 89-93.
- [28] Graham, Benjamin. "Fractional max-pooling." arXiv preprint arXiv:1412.6071 (2014).
- [29] Zeiler, Matthew D., and Rob Fergus. "Stochastic pooling for regularization  of  deep  convolutional  neural  networks."  arXiv  preprint arXiv:1301.3557 (2013).
- [30] Alamsyah,  Derry,  and  Muhammad  Fachrurrozi.  "Faster  R-CNN  with Inception  V2  for  Fingertip  Detection  in  Homogenous  Background Image." Journal of Physics: Conference Series. Vol. 1196. No. 1. IOP Publishing, 2019.
- [31] Fleury,  Daniel,  and  Angelica  Fleury.  "Implementation  of  RegionalCNN and SSD machine learning object detection architectures for the real time analysis of blood borne pathogens in dark field microscopy." (2018).
- [32] Thévenaz, Philippe, Thierry Blu, and Michael Unser. "Image interpolation and resampling." Handbook of medical imaging, processing and analysis 1.1 (2000): 393-420.
- [33] Arif,  Fahim,  and  Muhammad  Akbar.  "Resampling  air  borne  sensed data using bilinear interpolation algorithm." IEEE International Conference on Mechatronics, 2005. ICM'05.. IEEE, 2005.