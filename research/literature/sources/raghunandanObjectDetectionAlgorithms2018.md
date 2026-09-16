## Object Detection Algorithms for Video Surveillance Applications

Apoorva Raghunandan, Mohana, Pakala Raghav and H. V. Ravish Aradhya

AbstractObject  Detection  algorithms  find  application  in various  fields  such  as  defence,  security,  and  healthcare.  In  this paper various Object Detection Algorithms such as face detection, skin detection, colour detection, shape detection, target detection are simulated and implemented using MATLAB 2017b to detect various types of objects for video surveillance applications with improved accuracy. Further, various challenges and applications of Object Detection methods are elaborated.

Index  TermsColour Detection, Face Detection, Object Detection  Algorithms,  Skin  Detection,  Target  Detection,  Video Surveillance

## I. INTRODUCTION

BJECT  detection  mainly  deals  with  identification  of real-world objects such as people, animals, and objects  of  suspense  or  threatening  objects.  Object  detection algorithms use a wide range of image processing applications for  extracting  the  object's  desired  portion.  It  is  commonly used in applications such as image retrieval, security, Medical field, and defense. O

Fig. 1. Basic block diagram of object detection process

<!-- image -->

Fig.1  shows  the  Basic  block  diagram  of  object  detection process.  Frames  are  extracted  from  image  or  video.  Objects are detected based on user's desired choice such as face, skin, colour,  target  of  interest.  Further  various  features  of  Object are extracted for video surveillance applications

Apoorva Raghunandan, Mohana, Pakala Raghav, H. V. Ravish Aradhya is with  the  Department  of  Electronics  &amp;  Communication  Engineering  and Department of Telecommunication Engineering, R. V. College of Engineering, Bangalore-560059, India (e-mail: apoorva.rsi@gmail.com, mohana.rvce@gmail.com, pakalaraghav@gmail.com, ravisharadhya@rvce.edu.in)

978-1-5386-3521-6/18/$31.00 ©2018 IEEE

In  section  II  Literature  Survey  elaborates  the  current research work  in this  domain. Section III deals with fundamental concepts of object detection. Section IV describes various object detection algorithms with mathematical equations. Section V discusses various simulation  results.  Further,  Section  VI  elaborates  various challenges and applications of Object detection techniques for video surveillance applications.

## II. LITERATURE SURVEY

K.K. Hati [1] an efficient Background Subtraction Method for accurate Object Detection is proposed. Local Illumination based Background Subtraction (LIBS) method is used. Background modeling is done by defining an intensity range for  each  pixel,  shadows  are  eliminated,  which  is  an  added advantage of this method. I. Haritaoglu [2] in monochromatic imagery for tracking people and their body parts, W4 (Who? When? Where? What?) is used. For modelling the background each pixel is represented by three values, maximum, minimum intensity values and maximum intensity difference  between  consecutive frames  is observed.  The locations of these parts are verified and refined using dynamic template matching D. H. Santosh [3], three algorithms Gaussian  Mixture  Model  (GMM),  Extended  Kalman  Filter and  Mean  Shift  Algorithm  are  compared  in  the  context  of multiple  object  tracking.  The  performance  of  GMM  was observed  to  be  good  in  the  presence  of  occlusions.  During Nonlinear  transformation,  random  variables  behaved  in  an abnormal manner, due to this Extended Kalman filter failed. Identification of Multiple objects becomes challenging when there  are  occlusions.  For  single  object  tracking  Mean  shift algorithm  is  best  suited,  which  is  very  sensitive  to  window size.  Jacinto  Nascimento  [4]  In  this  paper  the  evaluation  of object  detection  was  performed  taking  five  algorithms  into consideration  such  as  Lehigh  Omni  directional  Tracking System (LOTS), Basic Background Subtraction (BBS), Multiple Gaussian Model (MGM), W4 and Single Gaussian Model (SGM).Best results were achieved by LOTS and SGM algorithm in terms of the number of correct detections. False alarms, splits and merges were much less compared to other algorithms. H. Fradi[5] The detection rate is improved without compromising on precision. This approach has been tested  on  a  dataset  of  complex  background  scenes.  The advantage of this method over other existing methods is that it improves  the  accuracy  of  foreground  segmentation.  This  is evident from  the  results  obtained  by  this  method.  P.M. Jodoin[6] Behavior subtraction finds application in characterizing  of  dynamic  events  especially  behavior  of  the object.  Each  event  is  composed  of  various  moving  objects which have been defined as spatio-temporal signatures. Modelling  of  these  events  has  been  done  using  stationary random processes.

<!-- image -->

## III. FUNDAMENTALS OF OBJECT DETECTION

Object detection is a technique of detecting a foreground object in a frame. The desired object could be person, animal or any other object or target of interest.

Foreground Object: A  foreground  object  is  distinct  from the  stationary  background.  It  could  be  with  respect  to  its appearance or local motion. It tends to change from frame to frame.

Background  Object :  Stationary  objects  in  a  frame  which are part of the background are called background objects.

Fig. 2. Block Diagram of Background Subtraction process

<!-- image -->

Fig. 2 shows the Block Diagram of Background Subtraction process. Background Subtraction Process is performed  on  the  current  frame  and  foreground  object  is detected. Next step involves the Background Update processor which current frame is compared with the previous frame to detect the object.

Fig. 3. The RGB and YCbCr Model

<!-- image -->

Fig.  3  shows  the  relationship  between RGB colour  space and YCbCr colour space . RGB colour space is represented by inner  cube  and YCbCr colour  space  is  represented  by  outer cube. Conversion from RGB to YCbCr model can be represented by mathematical expressions

<!-- formula-not-decoded -->

Cr = R - Y                                                                               (2)

<!-- formula-not-decoded -->

## IV. Object Detection Algorithms

This  section  describes  various  algorithms  that  are  used for object detection. An object can be characterized by the detection of face, skin and colour.

## A. Face Detection

Face  detection  is  a  technique  used  to  identify  human faces. The Viola Jones algorithm is used to detect all facial features - eyes, mouth and nose.

## B. Skin detection

In  skin  detection  technique,  skin  pixels  are  identified. The skin pixels and non-skin pixels are represented by '1' and '0' respectively. Four widely used categories of colour spaces for skin detection are RGB colour space, orthogonal colour space, perpetual colour space and perpetually uniform  colour  space.  Skin  Detection  is  performed  using the  YCbCr  Model.  The  translation  of  RGB  into  YCbCr colour space mainly involves separation of luminance from chrominance.  Therefore,  the  model  does  not  change  with variation  of  illumination.  Here  Y'  represents  Luminance and Cb and Cr indicate the Chrominance parameters.

Fig. 4. Flow chart for Skin Detection

<!-- image -->

Fig. 4 shows the flow chart of Skin Detection. First step is to read the image and convert RGB to YCbCr model. Adjust values  of  Cb  and  Cr  to  detect  skin  pixels  accurately.  In  the output image, skin pixels are represented by '1' and non-skin pixels are represented by'0'.

## C. Target Detection

In target detection, object of interest is detected. One of the most widely used methods for target detection is Background Subtraction[6-11]. Kommireddy Akhila

Background  Subtraction  from  a  stationary  Camera :  The image  is  captured  from  a  stationary  camera.  The  posterior probability is computed using Bayes rule

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Where )ݐݔ/ܨ(݌ and )ݐݔ / ߚ(݌ indicate probabilities of observing  'xt'  in  the  foreground  and  background  models respectively. p(F) and ߚ(݌ ) are  probabilities  of  the  pixels belonging to the foreground and background respectively. If equal priorities are assumed, then expression can be denoted by the likelihood ratio given by [12].

<!-- formula-not-decoded -->

## D. Colour Detection

Fig..  5.  Flowchart for Colour detection

<!-- image -->

Colour is an essential parameter for object recognition. Illumination  which  varies across the scene is important to be  considered  when  selecting  colour  models.  Effect  of robustness  with  geometry  of  the  object,  occlusion  and cluttering also play an important role.

Fig. 5 shows the flow chart of colour detection. First, the image is read. A thresholding process is used for conversion  of  Gray  scale  image  to  binary  which  mainly involves  comparison  of  each  pixel  value  with  the  pre-set threshold  value.  When  the  Threshold  value  is  observed  to be lower in comparison with pixel value then a value '1' is assigned  representing  white,  else  a  value  '0'  is  assigned representing black.

## V. SIMULATION RESULTS

This  section  describes  the  simulation  results  of  Face detection,  Skin  detection,  Colour  detection  and  Target detection.  The  algorithms  are  simulated  using  MATLAB 2017b.

## A. Face detection

It is the technique of detecting human faces. Different parts of the face have been detected using Viola Jones Algorithms.

<!-- image -->

.

Fig.. 6.(a)  Input Image (b) Face Detection

Fig. 6(a) is the input image of human face. Fig. 6(b) shows the face detected image of 6(a).

Fig.. 7. (a)  Eyes Detection (b) Nose detection (c) Detection of all features

<!-- image -->

Fig.7  shows  the  different  parts  of  the  face  with  respect  to 6(a). 7(a) indicates eye detection 7(b) indicates nose detection and 7(c) shows the detection of all features.

## B. Skin Detection

In Skin detection, the non-skin pixels and skin pixels are detected.

Skin Detection for a Single face - Four cases have been considered for skin detection analysis.

## Case 1: Cb - 77 to 127 and Cr -133 to173

Fig.. 8.  Skin detection for Case 1 - (a) Input Image (b) and (c) are output images of skin detection

<!-- image -->

Fig.  8(a)  is  the  input  image  8(b)  is  the  skin  detection output  of  8(a)  and  8(c)  is  the  output  binary  image  of 8(b).An  improvisation  has  been  performed  and  different outputs for different cases have been obtained[11].

Case 2: Cb - 79 to 136 and Cr - 142 to 165

Fig.. 9.  Skin detection for Case 2 - (a) Input Image (b) Skin detection represented by blue colour (c) Binary image

<!-- image -->

Fig.  9(a)  is  the  input  image,  9(b)  is  the  skin  detection output of 9(a) and 9(c) is the output binary image of 9(b). In this case shows only the skin is being detected.

Case 3 : Cb - 73 to 130 and Cr - 148 to 167

<!-- image -->

Fig.. 10.  Skin detection for Case 3 - (a) Input Image (b) Skin detection represented by blue colour (c) Output image 2

Fig. 10(a) is the input image, 10(b) is the skin detection output  of  10(a)  and  10(c)  is  the  output  binary  image  of 10(b).

Case 4: Cb - 77 to 137 and Cr - 142 to 167

Fig.. 11 . Skin detection for Case 4(a) Input Image (b) Skin detection represented by blue colour (c) Output image 2

<!-- image -->

Fig. 11(a) is the input image, 11(b) is the skin detection output  of  11(a)  and  11(c)  is  the  output  binary  image  of 11(b).For this value of Y, Cb and Cr, the results obtained are most accurate.

Multiple  People  Skin  Detection -  The  image  shows multiple people seated in different positions. Slight variations in skin complexion and different colour clothing are observed.

<!-- image -->

<!-- image -->

Fig.. 12. (a)  Input Image (b) Skin Detection Image (c) Binary Image

<!-- image -->

Fig. 12(a) is the Input Image 12(b) is the Skin Detection output  of  12(a)  and  12(c)  is  the  output  binary  image  of 12(b).The value of Cb is between 77 to Cr is between 142 to  167.  As  observed  from  the  output  images  12(b)  and 12(c), even clothes have been mis-detected as skin [12].

## C. Colour and Shape Detection

Colour  Detection -  Colour  detection  is  performed  for accurate detection of an object. It helps in correct classification  of  an  object.  Simulations  of  the  code  detect both the shape and different shades of colour in the image.

<!-- image -->

Fig. 13. (a) Input Image for colour detection (b) Colour detection

<!-- image -->

Fig.  13(a)  shows  the  input  image.  13(b)  shows  the colour detection output image of 13(a). In the output image, each region having a different shade is highlighted in green colour.

Shape Detection - Shapes of different parts of the object in the image have been detected as shown in Fig 14.

Fig.. 14. (a) Output Image of Shape detection, (b) Output Image Shape detection magnified view

<!-- image -->

Fig.  14(a)  shows  the  shape  detection  output  image  of 13(a) each shape is detected and displayed in red text. The shapes of the tennis racket and the face are detected.

Fig.  14(b)  is  a  magnified  view  of  14(a).  However, uneven  shapes  have  been  displayed  as  unknown.  This aspect of the algorithm could be improved to display with more clarity and accuracy.

## D. Target Detection

In  target  detection  the  desired  object  in  a  frame  is detected.  Background  Subtraction  was  used  for  Target Detection  and  simulation  was  performed  using  MATLAB 2017b.  The  outputs  were  obtained  for  different  values  of Euclidian threshold. The foreground, the cleaned-up foreground, shadows and the object of interest was detected

.

Fig. 15. Case 1: Original output for RGB Euclidian Threshold 'T'= 80

<!-- image -->

Fig.  15  shows  the  simulations  for  T  =  80  from  the original code. It detects the foreground, a cleaned-up foreground with shadows removed and the Object detected is displayed separately.

Case 2: RGB Euclidian Threshold T = 60

Fig.  16  shows  the  simulations  for  T  =  60.  In  the  first case the object was not completely detected. That, problem has  been  eliminated  in  this  case.  However,  shadows  have not been detected accurately.

Fig..16. Case 2: Output images for RGB Euclidian Threshold 'T'= 60

<!-- image -->

Case 3: RGB Euclidian Threshold T = 40

<!-- image -->

Fig. 17. Case 3: Output image for RGB Euclidian Threshold 'T'= 40

Fig. 17 shows the simulations for T = 40. For this value of  T,  shadows  are  detected,  but  not  very  accurately.  The image is shown with only a part of the shadow detected and a light yellow shade is observed.

## Case 4: RGB Euclidian Threshold T = 20

Fig. 18. Case 4: Output image for RGB Euclidian Threshold 'T'= 20

<!-- image -->

Fig. 18 shows the simulations for T = 20. For this value, the foreground and shadows are detected with high accuracy.

## VI. OBJECT DETECTION ALGORITHMS -CHALLENGES AND APPLICATIONS

This section elaborates challenges and applications of various object detection algorithms.

## A. Challenges

The challenges of each method - LIBS, W4, Behaviour Subtraction, Kalman Filter, Mean Shift Algorithm, Colour detection and Skin detection has been highlighted. In LIBS, the  model fails to provide the most accurate results in the presence of dynamic objects in the background. If there are small changes in the background like the waving of leaves or any subtle changes that may occur in the background. In W4, only people in upright position can be detected using the cardboard model. If people are in different poses, or are crawling and climbing, it becomes challenging. In Behavior Subtraction,  detection  of  spatial  anomalies  like  U-turns  is challenging  in  this  method.  If  it  is  necessary  in  detecting outliers,  only  the  ones  that  are  spatially  localized  and temporal can be detected. Behavior camouflage takes place especially when there is a foreground object during background activity. Kalman Filter, Mean Shift Algorithm and GMM face the challenge of detecting multiple objects when  there  is  slight  occlusion.  If  multiple  objects  are present in the image, existing skin detection algorithms will fail  to  detect  skin  region.  In  colour  Detection,  existing algorithms can only detect primary colours with accuracy. If  other  different  colours are  present in an image, existing methods mis-detect the colours.

Apart from these, some of the general issues are, if there is any change in illumination in the background, it could be mis-considered for a foreground object. Some methods also face  challenges  in  detecting  shadows.  Similarity  between the appearance of foreground object and background could create  confusion  of  camouflage.  Modelling  of  non-static backgrounds is another challenge. In High-traffic areas the background  is  frequently  obstructed  by  many  different foreground objects. Therefore, it will be difficult to classify a  fixed  foreground  and  background  due  to  continuous change

## B. Applications

Object  Detection  finds  scope  in  various  areas  such  as defense  and  border  security,  medical  image  processing, video  surveillance,  astronomy  and  other  security  related applications.

## VII. CONCLUSION

The  various  object  detection  algorithms  such  as  skin detection, colour detection, face detection and target detection  are  simulated  using  MATLAB  2017b  with  an accuracy of approximately 95%. Parameters such as detection accuracy, RGB Euclidian Threshold 'T' in Target Detection,  Y,  Cb  and  Cr  in  Skin  Detection  have  been simulated and implemented to improve the efficiency of the algorithms  for  video  surveillance  applications.  Further  a single  algorithm  maybe  designed  by  considering  various detection parameters such as Colour, Face, Skin and Target of interest to meet video surveillance applications.

## REFERENCES

- [1] K. K. Hati, P. K. Sa and B. Majhi, "Intensity Range Based Background Subtraction for Effective Object Detection," in IEEE Signal Processing Letters , vol. 20, no. 8, pp. 759-762, Aug. 2013
- [2] I. Haritaoglu, D. Harwood and L. S. Davis, "W4: Who? When? Where? What? A real time system for detecting and tracking people," Proceedings  Third  IEEE  International  Conference  on  Automatic  Face and Gesture Recognition , Nara, 1998, pp. 222-227
- [3] D.  H.  Santosh  and  P.  G.  K.  Mohan,  "Multiple  objects  tracking  using Extended Kalman  Filter, GMM  and  Mean  Shift  Algorithm -  A comparative study," 2014 IEEE International Conference on Advanced Communications, Control and Computing Technologies , Ramanathapuram, 2014, pp. 1484-1488
- [4] J. C. Nascimento and J. S. Marques, "Performance evaluation of object detection  algorithms  for  video  surveillance,"  in IEEE Transactions on Multimedia , vol. 8, no. 4, pp. 761-774, Aug. 2006
- [5] H.  Fradi  and  J.  L.  Dugelay,  "Robust  foreground  segmentation  using improved Gaussian Mixture Model and optical flow," 2012 International Conference on Informatics, Electronics &amp; Vision (ICIEV) , Dhaka, 2012, pp. 248-253.
- [6] P.  M.  Jodoin,  V.  Saligrama and  J.  Konrad,  "Behavior  Subtraction,"  in IEEE Transactions on Image Processing , vol. 21, no. 9, pp. 4244-4255, Sept. 2012.
- [7] Mohana and H. V. R. Aradhya, "Elegant and efficient algorithms for real time object detection, counting and classification for video surveillance applications  from  single  fixed  camera," 2016 International  Conference on Circuits, Controls, Communications and Computing (I4C) , Bangalore, 2016, pp. 1-7.
- [8] H.  V.  Ravish  Aradhya,  Mohana  and  Kiran  Anil  Chikodi,  "Real  time objects detection and positioning in multiple regions using single fixed camera  view  for  video  surveillance  applications," 2015  International Conference  on  Electrical,  Electronics,  Signals,  Communication  and Optimization (EESCO) , Visakhapatnam, 2015, pp. 1-6.
- [9] F.  Z.  Chelali,  N.  Cherabit  and  A.  Djeradi,  "Face  recognition  system using skin detection in RGB and YCbCr colour space," 2015 2nd World Symposium  on  Web  Applications  and  Networking  (WSWAN) ,  Sousse, 2015, pp. 1-7.
- [10] H. Ye, L. Zheng and P. Liu, "Colour detection and segmentation of the scene  based  on  Gaussian  mixture  model  clustering," 2017  7th  IEEE International  Conference  on  Electronics  Information  and  Emergency Communication (ICEIEC) , Macau, 2017, pp. 503-506.
- [11] (MathWorks - Makers of MATLAB and Simulink, n.d.)
- [12] Ahmed Elgammal, 'Background Subtraction- Theory and Practice'