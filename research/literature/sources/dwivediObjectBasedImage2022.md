## AN OBJECT BASED IMAGE ANALYSIS OF MULTISPECTRAL SATELLITE AND DRONE IMAGES FOR PRECISION AGRICULTURE MONITORING

Arun Kant Dwivedi 1 , Arun Kumar Singh 2 , Dharmendra Singh 2

1 Department of Computer Science and Engineering 2 Department of Electronics and Communication Engineering Indian Institute of Technology Roorkee, India

Email-IDs: adwivedi@cs.iitr.ac.in, asingh@ec.iitr.ac.in, dharm@ec.iitr.ac.in

## ABSTRACT

Accurate  information  on  spatial  distribution  of  crop  and vegetation indices for crop health monitoring is important for precision agriculture monitoring. However, freely available multispectral  satellite  images  and  unmanned  aerial  vehicle based multispectral images provides great opportunities for crop area estimation and extraction of vegetation indices. An object-based image analysis is better than pixel-based analysis because  it is used  statistical,  geometrical,  and topographic  feature  of  the  objects.  Therefore,  this  paper presents  an  object-based  image  analysis  of  multispectral satellite  and  drone  images  for  crop  area  estimation  and extraction  of  vegetation  indices  for  precision  agriculture monitoring.  Object  segmentation,  feature  extraction,  and classification of multispectral satellite and drone images was done. The experimental results show that the high-resolution drone  imagery  provides  better  crop  area  estimation  and vegetation indices compared  to  freely available coarse resolution  satellite  imagery  due  to  mixed  pixels  especially boundary of the crop classes.

Index  Terms - Precision  agriculture  monitoring,  area estimation, object-based image analysis, mixed pixels

## 1. INTRODUCTION

Precision agriculture monitoring  is  important  for better agricultural productivity and food management. The goal of precision agriculture monitoring is mapping, measuring, and monitoring. In recent years, an information technology-based precision  agriculture  monitoring  system  uses  geographical information system (GIS), global positioning system (GPS), satellite imagery, and drone imagery for mapping, measuring, and  monitoring  agriculture  fields  [1].  Satellite  images  are successfully  used  in  land  cover  classification  into  various classes like bare soil, urban, water, short vegetation and tall vegetation [2], [3]. Artificial intelligence made a significant improvement  in  satellite  image  classification  with  use  of various machine learning techniques [4]. Various vegetation indices,  texture,  phenology  and  spectral  signatures  of  crop classes  from  satellite  images  are  also  used  for  crop  type classification [5]. However, satellite images are also used in intra  class  classification  for  crop  monitoring  such  as  the segregation of sparse and dense crop area in agriculture fields [6]. From these classifications, we can observe that some of dense class pixels in the boundary of barren land or boundary of sparse classes are classified as sparse pixels. These things are happening in coarse resolution satellite images due to the mixed pixels in the boundary of land cover classes [7].

In  recent  years,  Drones,  or  unmanned  aerial  vehicles (UAV)  are  gaining  popularity  and  available  at  affordable price. Nowadays drones are used in various applications such as asset inspection, mapping, surveying, disaster management, agriculture, mining, and defense services [8]. A drone  equipped  with  multispectral  camera  have  taken  the advantage of red edge and near infrared imaging over crop, which provide crop health vegetation indices. Drone images provide us fast,  easy,  and  precise  information  about  crops. The drone images provide us an alternative to space-borne satellite images with clear high spatial and temporal resolution  for  a  region  of  interest.  The  drone  images  are extensively  used  to  measure  canopy  leaf  area  index  and various  biochemical  parameter  for  crop  health  monitoring [9].  The  use  of  drone  is  rapidly  increasing  in  precision agriculture monitoring due to ease of handling, high automation, and decision making.

Several  recent  studies  demonstrate  that  the  object  based image classification gives better results than the pixel based classification [10]. Nowadays the object-based image analysis is used for classification of remote sensing imagery because in object-based image analysis a group of pixels with homogeneous  characteristics  are  used  for  classification.  In object-based  image  classification  various  spectral,  spatial, and  textural  properties  are  used  rather  than  only  spectral features in pixel-based classification.

Recognizing  these paradigms,  the  main  objective  of  this paper is an object-based image  analysis of crop with vegetation  indices  using  multispectral  satellite  and  drone images. This study shows the utility of UAV or drone with multispectral camera for precision agriculture monitoring.

## 2. STUDY AREA AND DATASET

2.1. Study area Study is carried out in agriculture fields near Roorkee region in  Haridwar  district,  Uttarakhand,  India.  This  region  is predominantly  composed  of  smallholder  agriculture  land, with  farms  size  less  than  two  hectare.  There  are  monsoon (kharif) and winter (rabi) seasons are most growing agricultural seasons in this region. Sugarcane and wheat are most prominent crops cultivated in winter (rabi) season in the district  of  Haridwar.  The  study  area  shown  in  Figure.  1 includes  sugarcane,  wheat,  bare  land,  and  ploughed  land cover  classes  with  center  latitude  29.93°  N  and  longitude 77.97° E.

## 2.2. Sentinel-2 data

Sentinel-2 image  of study  region  is  downloaded  from sentinel scientific data hub for the study. The acquired image has  13  spectral  bands.  The  spatial  resolution  of  acquired image is in the range of 10m to 60m. Sentinel -2 images have temporal resolution of 5 days. The Sentinel-2 data is acquired on November 21, 2018, and the study area is marked in the image as shown in Figure 1. The acquired Sentinel-2 data is preprocessed to obtain atmospherically corrected data using Sen2cor  tool.  To  simplifying  the  processing,  atmospheric corrected Sentinel-2 image bands further resampled to unique spatial  resolution  (10m)  using  nearest-neighbor  assignment method in SNAP tool. The subsetting is done in resampled sentinel-2 image where drone was flown. This final preprocessed dataset is used for the further implementation of algorithms.

## 2.3. Multispectral drone data

The images of the agriculture field are collected using the DJI Matrix -100 quadcopter UAV  that have RedEdge-M multispectral  camera  sensor  by  MicaSense.  This  sensor captures  the  high-  resolution  images  in  five  narrow  bands having (blue, green, red, red edge and NIR (near infra-red)) filter.  The  sensor  has  its  separate  GPS  module  that  help  it capture the geo-tagged image with precision and the geotagged position is saved in the capture.kmz file that help in  creating  mosaic  image.  The  images  used  for  the  study purpose  is  captured  on  November  22,  2018.  The  drone  is flown  at  height  of  120m  above  the  ground  to  acquire  the images with image size of 4000 × 3000. The flight path is set such that drone acquired images with 65% forward overlap and 40% side overlap to produce the good result for feature point matching while the images are mosaicked to obtain the desired study area. The acquired images of drone are carried to lab and processes using Pix4d tool. The mosaicked image of  study  area  obtained  from  the  Pix4d  is  0.08m  spatial resolution. This mosaic image of area is georeferenced and orthorectified  which  means  it  have  location  information of every  pixel  and  projection  of  each  pixel  is  corrected.  This obtained image is used for further processing.

## 3. METHODOLOGY

The fundamental aim of this study is to analyze the use of

<!-- image -->

(a)

(b)

Figure 1. (a) Mosaicked drone image of the study area, (b) Sentinel-2 image of the study area.

multispectral satellite images and multispectral drone images in  precision  agriculture  monitoring.  For  vegetation  health monitoring, the study shows that the reflectance of blue (B) and red (R) band is low, while it is peak in green (G) band. However, the reflectance of near infrared (NIR) band is much higher  than  the  visible  bands.  Mostly  vegetation  health indices utilized these R, G, B, and NIR band for crop health monitoring. Therefore, in this study preprocessed multispectral satellite and drone images are subsetted to study area  and  R,  G,  B,  and  NIR  band  are  extracted  for  further processing. The proposed methodology includes three major steps: multiscale segmentation for object creation; spectral, spatial,  and  texture  feature  extraction  for  object  analysis; finally,  object-based  classification  of  agriculture  field  for vegetation indices extraction and area estimation. The flowchart  of  proposed  methodology  of  object-based  image analysis is shown in Figure 2.

## 3.1. Image segmentation and object creation

Image segmentation is a bottom up process, where initially all pixels are a segment. These single pixels are merged with homogeneous pixels at different scale to create meaningful objects. The scale parameters are used to determine the shape, compactness,  and  size  of  the  objects.  Since,  we  are  using multiresolution  images  (satellite  and  drone)  for  crop  field segmentation, we need to set scale parameter according to image  resolution  and  crop  parcel  size  for  parcel  boundary extraction. The watershed algorithm is used to segment the crop  parcels  as  catchment  basins  and  parcel  boundaries  as crest  lines  [11].  The  advantage  of  watershed  segmentation algorithm is that the boundaries of the segmented objects are always connected and closed, compared to edge or boundary detection  algorithms.  The  merging  combines  neighboring segments  with  similar  statistical  features  to  overcome  the over-segmentation problem.

## 3.2. Statistical feature extraction

Objects basically a group of homogeneous pixels that gives some meaningful information to the real world. Segmentation process creates objects. However,  to  recognizing  these objects  we  need  to  analyze  various  spectral,  spatial,  and textural features. Spectral feature describes the sensitivity of objects with multispectral bands, spatial feature describes the size  of  objects,  and  texture  feature  describe  the  shape  of objects. The list of statistical features used in this study are shown in table 1.

Table 1. Statistical features used in this study.

| Spectral                           | Spatial                      | Texture                      |
|------------------------------------|------------------------------|------------------------------|
| Mean, min, max, standard deviation | Area, convexity, compactness | Texture range, mean, entropy |

## 3.3. Object classification

The  supervised  machine  learning  model  support  vector machine (SVM) are used in classification of these segmented objects.  The  SVM  model  was  trained  with  these  spectral, spatial, and texture features. After fine tuning of parameter of SVM model, this model classifies the segmented multispectral  satellite  and  drone  images  into  various  crop classes. These classified objects are further used for precise crop monitoring  and  estimation  of crop area for crop production  estimation.  The  NDVI,  SAVI,  and  GNDVI vegetation indices are used for crop health monitoring.

Figure 2. Flowchart of proposed method.

<!-- image -->

## 4. RESULTS AND DISCUSSION

The experimental results of object-based image classification are shown in Figure 3. The segmentation of multiresolution drone and Sentinel-2 images are done, and boundary of crop classes  are  generated.  The  spectral,  spatial,  and  texture features of these objects are extracted, and these features are used for developing a machine learning model for classification  of  these  objects.  The  classification  results  of drone  and  satellite  images  are  shown  in  Figure  3(c)  and Figure 3(f). After object-based classification of crop classes, these objects are further evaluated with multiple vegetation indices and area estimation for precision agriculture monitoring.

Figure  3. Object  based  classification  results.  (a)  and  (d) segmented,  (b)  and  (e)  mean  spectral  feature, (c)  and  (f) classified drone and Sentinel-2 images, respectively.

<!-- image -->

The NDVI, SAVI, and GNDVI indices are used for crop health  analysis  of  crop  classes  using  satellite  and  drone images. These vegetation indices are used to estimate crop health parameter like leaf area, canopy biomass, and chlorophyll content in the crop. The statistical measurement of  crop  health  using  satellite  images  and  drone  images  are shown  in  Table  2  and  Table  3.  Mean  values  of  these vegetation  indices  indicate  that  the  drone  images  provide more enhanced information of crop health. However, standard deviation of vegetation indices indicates the measurement  of  spatial  variability.  In  both  the  cases  high resolution drone imagery perform better than coarse resolution satellite imagery.

Table 2. Statistical measurement of vegetation indices using satellite images.

|           | NDVI   | NDVI   | SAVI   | SAVI   | GNDVI   | GNDVI   |
|-----------|--------|--------|--------|--------|---------|---------|
| Areas     | Mean   | SD     | Mean   | SD     | Mean    | SD      |
| Sugarcane | 0.56   | 0.06   | 0.30   | 0.05   | 0.50    | 0.06    |
| Wheat     | 0.80   | 0.02   | 0.58   | 0.02   | 0.72    | 0.03    |
| Bare land | 0.14   | 0.02   | 0.08   | 0.01   | 0.25    | 0.02    |
| Ploughed  | 0.31   | 0.03   | 0.20   | 0.02   | 0.45    | 0.04    |

Table 3. Statistical measurement of vegetation indices using drone images.

|           | NDVI   | NDVI   | SAVI   | SAVI   | GNDVI   | GNDVI   |
|-----------|--------|--------|--------|--------|---------|---------|
| Areas     | Mean   | SD     | Mean   | SD     | Mean    | SD      |
| Sugarcane | 0.58   | 0.13   | 0.38   | 0.09   | 0.55    | 0.11    |
| Wheat     | 0.83   | 0.06   | 0.56   | 0.05   | 0.78    | 0.03    |
| Bare land | 0.17   | 0.03   | 0.02   | 0.01   | 0.18    | 0.04    |
| Ploughed  | 0.32   | 0.09   | 0.13   | 0.10   | 0.47    | 0.11    |

However, area estimation of various crop classes is also done using both satellite and drone images. Table 4 indicate the possible errors in area estimation of various crop classes using coarse resolution satellite images due to mixed pixels in the boundary of objects.

Table 4. Area estimation of crop classes.

| Class         |   Area computed by satellite image (in km 2 ) |   Area computed by drone image (in km 2 ) |
|---------------|-----------------------------------------------|-------------------------------------------|
| Sugarcane     |                                        0.1293 |                                    0.1220 |
| Wheat         |                                        0.0666 |                                    0.0693 |
| Bare land     |                                        0.0761 |                                    0.0732 |
| Ploughed land |                                        0.2159 |                                    0.2235 |

## 5. CONCLUSIONS AND FUTURE WORK

This paper reports an object-based analysis of multispectral satellite and drone images for precision agriculture monitoring. An object-based multispectral image analysis is suitable  for  crop  health  monitoring  and  area  estimation  of crop classes because it utilizes various spectral, spatial, and textural features for segmentation and classification of crop field.  Moreover,  multispectral  (R,  B,  G,  and  NIR)  satellite and  drone  images  are  used  to  compute  various  vegetation indices for crop health monitoring. However, the experimental results show that the high-resolution multispectral drone imagery provides better crop area estimation and vegetation indices compared to freely available  coarse  resolution  satellite  imagery  due  to  mixed pixels especially boundary of the crop classes. Furthermore, our  results  suggest  the  research  gap  for  low  resolution satellite  images  to  upsampled  or  calibration  of  area  and vegetation indices using multispectral drone data. Finally, in future we call an extension for decomposition of mixed pixels in coarse resolution satellite images to improve estimation of vegetation indices and area of crop classes.

.

## 6. ACKNOWLEDGMENTS

The authors would like to thank Drone Research Centre, IIT Roorkee for supporting this work and ESA (European Space Agency) for providing Sentinel-2 data.

## 7. REFERENCES

- [1] D.  J.  Mulla,  'Twenty  five  years  of  remote  sensing  in precision agriculture: Key advances and remaining knowledge gaps', Biosystems Engineering ,  vol. 114, no. 4, pp. 358 - 371, Apr. 2013.
- [2] P.  Mishra  and  D.  Singh,  'A  Statistical -Measure-Based Adaptive Land Cover Classification Algorithm by Efficient  Utilization  of  Polarimetric  SAR  Observables', IEEE Transactions on Geoscience and Remote Sensing , vol. 52, no. 5, pp. 2889 - 2900, May 2014.
- [3] A.  Garg  and  D.  Singh,  'Development  of  an  Efficient Contextual Algorithm for Discrimination of Tall Vegetation and Urban for PALSAR Data', IEEE Transactions on Geoscience and Remote Sensing , vol. 56, no. 6, pp. 3413 - 3420, Jun. 2018.
- [4] A. E. Maxwell, T. A. Warner, and F. Fang, 'Implementation  of  machine -learning  classification  in remote sensing: an applied review', International Journal of Remote Sensing ,  vol.  39,  no.  9,  pp.  2784 - 2817, May 2018.
- [5] N.  R.  Rao,  'Development  of  a  crop-specific  spectral library  and  discrimination  of  various  agricultural  crop varieties using hyperspectral imagery', International Journal of Remote Sensing , vol. 29, no. 1, pp. 131 - 144, Jan. 2008.
- [6] D. Murugan, A. Garg, and D. Singh, 'Development of an Adaptive Approach for Precision Agriculture Monitoring with Drone and Satellite Data', IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing , vol. 10, no. 12, pp. 5322 - 5328, Dec. 2017.
- [7] A.  K.  Dwivedi,  S.  Roy,  and  D.  Singh,  'An  Adaptive Neuro-Fuzzy  Approach  for  Decomposition  of  Mixed Pixels to Improve Crop Area Estimation Using Satellite Images',  in IGARSS  2020  -  2020  IEEE  International Geoscience and Remote Sensing Symposium , Sep. 2020, pp. 4191 - 4194.
- [8] A.  K.  Singh,  A.  K.  Dwivedi,  N.  Nahar,  and  D.  Singh, 'Railway Track Sleeper Detection in Low Altitude UAV Imagery Using Deep Convolutional Neural Network', in 2021 IEEE International Geoscience and Remote Sensing Symposium IGARSS , Jul. 2021, pp. 355 - 358.
- [9] A.  J.  Mathews  and  J.  L.  R.  Jensen,  'Visualizing  and Quantifying Vineyard Canopy LAI Using an Unmanned Aerial Vehicle (UAV) Collected High Density Structure from Motion Point Cloud', Remote Sensing , vol. 5, no. 5, Art. no. 5, May 2013.
- [10] S. W. Myint, C. S. Galletti, S. Kaplan, and W. K. Kim, 'Object vs. pixel: a systematic evaluation in  urban environments', Geocarto International , vol. 28, no. 7, pp. 657 - 678, Nov. 2013.
- [11] L. Vincent and P. Soille, 'Watershed s in digital spaces: an efficient  algorithm  based  on  immersion  simulations', IEEE  Transactions  on  Pattern  Analysis  and  Machine Intelligence , vol. 13, no. 6, pp. 583 - 598, Jun. 1991.