## A Surveillance System Using CNN for Face Recognition with Object, Human and Face Detection

Yeong-Hyeon Byeon, Sung-Bum Pan,

Sang-Man Moh and Keun-Chang Kwak

1

Abstract Recently, surveillance system plays an important role in solving several crimes  by  replacing  human  to  watch  monitors.  Not  only  many  functions  are complicatedly  integrated  to  a  system  but  also  a  system  is  evolved  to  capture statistical  data  to  extract  useful  information.  But  integrating  many  functions should be considered to make it have reduced processing time because a system has  limited  processing  ability.  If  a  system  considers  moving  objects,  it  could reduce processing time because surveillance system normally has static background that is useless information. People and face detection are performed in detected objects. Detected faces are recognized using CNN(Convolutional Neural Network). The processing time of the proposed system is reduced and true rate of face recognition is 72.7% under varying distance from 2m to 5m.

Keywords Surveillance · CNN · Face recognition · Object detection

## 1 Introduction

Solution  of  security  using  camera  recently  is  progressed,  security  enhanced  by international terrors.  It  is  needed  for public  and  private  parts  more  and  more  by playing an important role in solving several heavy crimes that occurred near. It is difficult  for    watchman  to  keep  eyes  at  monitor  all  day  long.  An  intelligent surveillance  system  that  automatically  cautions  against  dangers  is  rising.  But  a system  capturing images  under  high  resolution needs to process a lot of

Y.-H. Byeon · S.-B. Pan · K.-C. Kwak(  ) Department of Control and Instrumentation Engineering, Chosun University,

375 Seosuk-dong, Dong-gu, Gwangju 501-759, Korea e-mail: qasdfghjt@daum.net, kwak@chosun.ac.kr

S.-M. Moh

Department of Computer Engineering, Chosun University, 375 Seosuk-dong, Dong-gu, Gwangju 501-759, Korea

© Springer Science+Business Media Singapore 2016

K.J. Kim and N. Joukov (eds.), Information Science and Applications (ICISA) 2016 ,

information or a system capturing images under low resolution makes information insufficient to be processed precisely.

An  intelligent  surveillance  system  aims  to  prevent  accidents,  reducing  the number  of  watchman  and  raising  efficiency.  It  leads  to  reduced  expense  for maintenance. In a case of showing strange patterns, it can be perceived as planned crime because it is reported culprits have certain moves[1,2].

Normally,  processing  an  entire  image  needs  many  computing  powers  and  a camera captures many images for a second that are hard to be processed in a short time. Under using fixed camera, however, background isn't dynamic enough for reducing process time. Static background in video is less important than dynamic one for surveillance system. So, focusing moving object is effective for time.

Furthermore, identity of individual using computer needed, interest of biometrics recognitions increase steadily. Related researches are fingerprint recognition[3], speaker recognition[4], document  recognition[5],facial  expression recognition[6] and face recognition[7].

In  this  paper,  we  consider  a  surveillance  system  using  CNN(Convolutional Neural  Network)  for  face  recognition  with  object,  human  and  face  detection. Region of object in entire image is picked by object detection and we discriminate whether the area is human or not using human or face detection. If it is human, we can analyze his movement. Section 2 describes object, face and human detection. Section 3 shows surveillance system. Section 4 is experimental result. Section 5 makes conclusions.

## 2 Related Works

## 2.1 Object Detection

In  the  design  of  adaptive  Gaussian  mixture  model,  each  pixel  in  the  scene  is modelled by a mixture of K Gaussian distributions. The probability that a certain pixel has a value of ݔ ே at time N can be written as

<!-- formula-not-decoded -->

where ݓ ௞ is  the  weight  parameter  of  the ݇ ௧௛ Gaussian  component. η(x; ߠ ௞ ) is the Normal distribution of ݇ ௧௛ component represented by

<!-- formula-not-decoded -->

where ߤ ௞ is the mean and Σ௞ ߪ = ௞ ଶ ܫ is the covariance of the ݇ ௧௛ component.

The K distributions are ordered based on the fitness value ݓ ௞ ߪ/ ௞ and the first B distributions  are  used  as  a  model  of  the  background  of  the  scene  where  B  is estimated as

<!-- formula-not-decoded -->

The  threshold  T  is  the  minimum  fraction  of  the  background  model.  In  other words,  it  is  the  minimum  prior  probability  that  the  background  is  in  the  scene. Background subtraction is performed by marking a foreground pixel any pixel that is more than 2.5 standard deviations away from any of the B distributions. The first Gaussian component that matches the test value will be updated by the following update equations,

̂

̂

̂

<!-- formula-not-decoded -->

̂

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ߱ ௞ is the ݇ ௧௛ Gaussian component. 1/α defines the time constant which determines change. If none of the K distributions match that pixel value, the least probable component is replaced by a distribution with the current value as its mean, an initially high variance, and a low weight parameter. According to their papers [8,9,10], only two parameters, α and T, needed to be set for the system.

The  EM  algorithms  by  expected  sufficient  statistics  are  shown  in  the  left column while the by L-recent window version in the right[11].

̂

<!-- formula-not-decoded -->

̂

̂

<!-- formula-not-decoded -->

̂

<!-- formula-not-decoded -->

̂

̂

̂

̂

<!-- formula-not-decoded -->

̂

̂

̂

<!-- formula-not-decoded -->

̂

̂

̂

<!-- formula-not-decoded -->

̂

̂

̂

## 2.2 People Detection

The  method  is  based  on  evaluating  well-normalized  local  histograms  of  image gradient orientations in a dense grid. The basic idea is that local object appearance and shape can often be characterized rather well by the distribution of local intensity gradients or edge directions, even  without precise knowledge of the corresponding gradient  or  edge  positions.  In  practice  this  is  implemented  by  dividing  the  image window  into  small  spatial  regions('cells'),  for  each  cell  accumulating  a  local 1-D histogram of gradient directions or edge orientations over the pixels of the cell. The  combined  histogram  entries  form  the  representation.  For  better  invariance  to

̂

̂

̂

̂

illumination,  shadowing,  etc.,  it  is  also  useful  to  contrast-normalize  the  local responses before using them. This can be done by accumulating a measure of local histogram  'energy'  over  somewhat  larger  spatial  regions('blocks')  and  using  the results to normalize all of the cells in the block. The people detector detects people in an  input  image  using  the  Histogram  of  Oriented  Gradient(HOG)  features  and  a trained Support Vector Machine(SVM) classifier[12].

## 2.3 Face Detection

This  face  detector  finds  optimal  boundary  using  weak  classifier  extracting  haarbased  feature  that  classifies  images  between  positive  ones  and  negative  ones.  If such  weak  classifiers  are  gathered  by  cascade  they  become  strong  one.  Table  1 shows algorithm of adaboost to train classifier.

## Table 1 Algorithm of adaboost to train classifier

-  Given example images ݔ( ଵ ݕ , ଵ ), … , ( ݔ ௡ ݕ , ௡ ) where ݕ ௜ = 0,1 for negative and positive examples respectively.
-  Initialize  weights ݓ ଵ,௜ = ଵ ଶ௠ , ଵ ଶ௟ for ݕ ௜ = 0,1 respectively,  where m and l are the number of negatives and positives respectively.
-  For t=1,…,T:
1. Normalize the weights.

<!-- formula-not-decoded -->

2. For each feature, j, train a classifier ℎ௝ which is restricted to using a  single  feature.  The  error  is  evaluated  with  respect  to ݓ ௧ ߳ , ௝ = ∑ ݓ ௜ หℎ ௝ ݔ( ௜ ) - ݕ ௜ ൯| ௜ .
3. Choose the classifier, ℎ௧ , with the lowest error ߳ ௧ .
4. Update the weights:

<!-- formula-not-decoded -->

where ݁ ௜ = 0 if example ݔ ௜ is classified correctly, ݁ ௜ = 1 otherwise, and ߚ ௧ = ఢ ೟ ଵିఢ೟ .

-  The final strong classifier is[13]:

<!-- formula-not-decoded -->

where ߙ ௧ ݃݋݈ = ଵ ఉ೟

## 3 Surveillance System

In the case of a fixed camera, surveillance system captures images directing same area. The images does not change suddenly as long as the camera does not move. It means that there are wide parts overlapped in an image. Processing entire image takes  bags  of  time  for  unnecessary  background.  Actually,  important  part  in surveillance  system  is  changes  from  previous  information.  Object  detection  is performed to extract changes. And people and face detection are followed. This strategy is more favorable because object detection takes less time than people and face detection.

In  a  case  using  a  fixed  camera,  object  detection  is  performed  by  subtracting current image from standard one. Although the camera is fixed, it moves a little by wind  or  unplanned  impact.  Subtracting  current  image  from  standard  one  is  not good for  the  case.  On  the  other  hand,  adaptive  object  detection  using  Gaussian mixture  model  don't  use  standard  image.  It  uses  near  images  from  current  one. The algorithm considers stationary pixels as background. If the camera moves and stops,  pixels  are  changed  and  fixed  to  new  values.  The  algorithm  will  adapts changes because stationary pixels become background soon. But its result is not clearer  than  subtracting's  one.  So,  after  binary  image  is  gotten  from  subtracting current image from standard one, it is dealt under adaptive object detection using Gaussian mixture model again. Doing so makes object clearer and the algorithm have stability against camera's move. Then, changed parts are gained. People and face detection are performed on the only changed parts as input. Figure 1 shows the flow chart about surveillance system.

Fig. 1 Flow chart about surveillance system

<!-- image -->

Also, the detected face is recognized by CNN. The structure of CNN consists of 5 layers. First is input layer that images enter the network. Second and forth layers are  convolution  that  extracts  features  from  images.  Third  and  fifth  layers  are subsampling that reduces size of images. Input images are normalized before CNN recognizes it. If input images are passed through network, output features become much tinier than input images. Then, the output features are classified by training a  MLP(Multilayer Perceptron). MLP is a classifier that imitates brain system of human. Figure 2 shows the structure of CNN. The size of input image is 28x28. The size of window for convolution at second layer is 3x3 so the size of second layer is  26x26. The size of window for subsampling at third layer is 2x2 so the size of third layer is 13x13. The size of window for convolution at fourth layer is 6x6 so the size of fourth layer is 8x8. The size of window for subsampling at fifth layer is 2x2 so the size of fifth layer is 4x4. The number of feature map is 30 at fifth layer. So the size of final feature is 1x480(30x4x4)[7,14].

Fig. 2 Structure of CNN

<!-- image -->

## 4 Experimental Results

In  this  experiment,  we  use  windows  7,  64bit  architecture,  i5-4440  cpu  @ 3.10GHz, 8GB RAM and NVIDIA GeForce GT 630. The program is developed using matlab(2013a)'s image processing toolbox. Device for capturing images is Logitech's  QuickCam  Sphere  AF.  Resolutions  for  capturing  are  360x240  and 640x480 as SD. We captured images for studying performance in several working environment including moving people and cars in indoor and outdoor environments.  The  first  process  is  object  detection.  Using  subtracting  current image  from  standard  one  for  object  detection  depends  on  the  standard  image directly.  And  using  adaptive  object  detection  for  object  detection  is  sensitive against noise. So, we integrate subtracting current image from standard one with adaptive object detection using Gaussian mixture model. Fig. 3 shows comparison of object detection. There are three images every row and they are same. (a) and (d) are original image. (b) is a fail to detect object under subtracting image from standard one. But (c) succeeds to detect object under integrated object detection. (e)  considers  background  as  object.  But  (f)  does  not  consider  background  as object. The experiment shows that integrated object detection has better performance.

Fig. 3 Comparison of object detection

<!-- image -->

Fig. 4 Entire system working in indoor and outdoor environments

<!-- image -->

After object is detected, people and face detection are performed only in area of object. Then, if  face or people is detected,  system  considers  area  as  staying people. Face  detection  works  within  1~5m  away  from  a  camera  using  640x480  resolution.

People detection works within 1~10m away from a camera using 640x480 resolution. And object should be taken roughly for people detection. Fig. 4 shows entire system working indoor and outdoor environments. Integrated systems are faster than simple ones. Frame rates depending on detection flow are listed in Table 2.

Table 2 Frame rates depending on detection flow

| Detection Flow   | Detection Flow   |   Frame Rate |
|------------------|------------------|--------------|
| Face             | Face             |      07.0738 |
| People           | People           |      08.3174 |
| Object →         | Face             |      15.3352 |
| Object →         | People           |      14.3998 |
| Object →         | Face & People    |      10.5228 |

Detected  face  is  recognized  by  CNN.  Figure  5  shows  detected  faces.  Face images  are  3  channel  captured  varying  light  and  saturation.  Face  images  turn grayscale and histogram equalization. There are faces of 10 people. Each person has 500 images from 1m to 5m. Totally face images are 5000 images. Only 1000 images are used for training and 4000 images are used for checking. True rate for recognition is 72.7% under training 1m faces and checking 2-5m faces. Table 3 shows recognition rate of varying distance.

Fig. 5 Detected faces

<!-- image -->

Table 3 Recognition rate of varying distance

|                   | Recognition Rate   | Recognition Rate   | Recognition Rate   | Recognition Rate   | Recognition Rate   |
|-------------------|--------------------|--------------------|--------------------|--------------------|--------------------|
| Algorithm(Param)  | mean               | 2m                 | 3m                 | 4m                 | 5m                 |
| CNN(epoch=50)[12] | 72.7               | 79.7               | 67.8               | 66.3               | 75.8               |

## 5 Conclusions

We developed a surveillance system using CNN for face recognition with object, human  and  face  detection.  And  it  shows  different  performance  depending  on detection flows. Integrated system was faster than simple one. We use a database that  is  captured  indoor  and  outdoor  environments  including  moving  people  and cars.  Furthermore,  we  want  this  system  to  become  faster  by  using  tracking algorithm. But there are plenty of problems for applying. Among several frames, there  are  successful  and  unsuccessful  ones.  It  is  difficult  to  judge  why  object disappear. We will study this more.

Acknowledgments This  research  was  financially  supported  by  the  Ministry  of  Trade, Industry and Energy(MOTIE) and Korea Institute for Advancement of Technology(KIAT) through the Promoting Regional specialized Industry. This research was also supported by Basic Science Research Program through the National Research Foundation of Korea(NRF) funded by the Ministry of Science, ICT and Future Planning (NRF2015R1D1A1A01060701).

## References

1. Jeon, B.J.: Industrial Trend on Visual Observation of Domestic and Foreign Intelligent CCTV. J. TTA 142 , 50-55 (2012)
2. Kang,  J.H.:  Detection  of  Intrusion,  Sudden  running,  Loitering  for  Intelligent  Video Surveillance System. Hanbat National University (2014)
3. Yuan, W., Lixiu, Y., Fuqiang, Z.: A real time fingerprint recognition system based on novel  fingerprint  matching  strategy.  In:  Proc.  IEEE  Electronic  Measurement  and Instruments, pp. 81-85 (2007)
4. Abdel-Hamid, O., Mohamed, A., Jiang, H., Penn, G.: Applying convolutional neural networks concepts to hybrid NN-HMM model for speech recognition. In: Proc. IEEE Acoustics, Speech and Signal Processing, pp. 4277-4280 (2012)
5. LeCun, Y., Bottou, L., Bengio, Y., Haffner, P.: Gradient-Based Learning Applied to Document Recognition. Proceeding of IEEE 86 (11), 2278-2324 (1998)
6. Matsugu, M., Mori, K., Mitari, Y., Kaneda, Y.: Subject independent facial expression recognition  with  robust  face  detection  using  a  convolutional  neural  network.  Neural Networks 16 (5-6), 555-559 (2003)

7. Lawrence, S., Giles, C.L., Tsoi, A.C., Back, A.D.: Face Recognition: A Convolutional Neural-Network Approach. IEEE Trans. on Neural Networks 8 (1), 98-113 (1997)
8. Grimson,  W.E.L.,  Stauffer,  C.,  Romano,  R.,  Lee,  L.:  Using  adaptive  tracking  to classify and monitor activities in a site. In: Proc. IEEE Computer Society Conference on Computer Vision and Pattern Recognition (1998)
9. Stauffer,  C.,  Grimson,  W.E.L.:  Adaptive  background  mixture  models  for  real-time tracking.  In:  Proc.  IEEE  Computer  Society  Conference  on  Computer  Vision  and Pattern Recognition, vol. 2 (1999)
10. Stauffer, C., Grimson, W.E.L.: Learning patterns of activity using real-time tracking. Proc. IEEE Transactions on Pattern Analysis &amp; Machine Intelligence 22 (8), 747-757 (2000)
11. Kaewtrakulpong, P., Bowden, R.: An improved adaptive background mixture  model for  realtime  tracking  with  shadow  detection.  In:  Proc.  2nd  European  Workshop  on Advanced Video Based Surveillance Systems (2001)
12. Dalal, N., Triggs, B.: Histograms of oriented gradients for human  detection. In:  Proceedings  of  IEEE  Conference  on  Computer  Vision  and  Pattern  Recognition, pp. 886-893 (2005)
13. Viola,  P.A.,  Jones,  M.J.:  Rapid  object  detection  using  a  boosted  cascade  of  simple features. In: IEEE CVPR (2001)
14. Byeon, Y.H., Kwak, K.C.: Performance comparison of distance-based face recognition using CNN. In: KISM Fall Conference, vol. 4(2), pp. 338-339 (2015)