## Image In-Painting Techniques - A Survey and Analysis

S.  Ravi Dept.  Computer Science Pondicherry University Pondicherry, India sravicite@gmail.com P  .Pasupathi Centre for IT &amp; Engg. MS University Tirunelveli, India. pp.cite.msu@gmail.com

Abstract-Digital in-painting is relatively a  young research area,  yet  a  large  variety  of  techniques  were  proposed  by  the researchers to  correct  the  occlusion.  Image  in-painting  aims  to restore  images  with partly information loss  and  tries  to  make in-painting results as these missing parts in such a way that the reconstructed  image  looks  natural. Many  different  types  of image  in-painting  algorithms  exist  in  the  literature.  However no recent study has been undertaken for a comparative evaluation  of  these  algorithms  to  provide  a  comprehensive visualization.  This  paper compares different types  of image in­ painting algorithms. The algorithms are analyzed in both theoretical and experimental ways, which have made  the suitability  of  these  image  in-painting  algorithms  over  different kinds of applications in diversified areas.

Keywords-In-Painting; Occlusion Removal; Image Matting, Object Removal,  Crack Removal,  Image Editing

## I. I NTRODUCTION

The  process  of  filling  the  missing  regions  of  an  image from  the  surrounding  parts  is  known  as  Digital  Image  In­ painting. The digital Image In-painting has various applications such as restoration of damaged old printing and old photographs, error recovery of images and videos, multimedia  editing  (computer  assisted), transmission  loss and replacing large regions in an  image or video for privacy protection. The  main  aim  of  the  image  in-painting  is  to modify the damaged regions of an image or video so that the in-painted  region  is  undetectable  to  a  neutral  observer  [25]. Indeed,  natural  images  are  complex,  an  image  information may  contain  three  parts  viz., shape  (or  structure),  texture and color information. Textures have been traditionally classified  as  either  regular  (consisting  of  repeated  texels)  or stochastic  (without  explicit  texels).  However,  almost  in  all real-world  textures  lie  somewhere  in  between  these  two extremes and should be captured with a single model  [22,28, 33, 34].

## II. I MAGE AND  V IDEO  I N-PAINTING  T ECHNIQUES

The  challenging  task  in  image  in-painting  technique  is the evaluation  of  the quality of an image  because  it  is fundamentally  different from  the general motion  of the image  quality.  An  image  or  video  can  still  be  classified  as unsatisfactory  in-painting  implementation  if  it  is  free  from visual  artifacts,  for  example,  the  geometric  attribute  of  the Muthukumar. S Dept.CSE NIT, Puducherry Karaikal,  India sm.cite.msu@gmail.com N.  Krishnan Centre for IT &amp; Engg. MS University Tirunelveli,  India. krishnann@ieee.org synthetic  image  to  present  a  visually  plausible  image  could be  considered  is  be  more  satisfactory  than  the  approximate correct  texture  replication [36]. Similarly  for  the  natural images, combinations  of  the  texture  inside  natural  edge boundaries  might  indicate  a  better  in-painting algorithm. The  researchers  used  different  approaches  to  digital  image in-painting and can be classified into following categories:

- Texture Synthesis based in-painting
- Semi-automatic and Fast Digital In-painting
- Partial Differential Equation (PDE) based in-painting
- Exemplar and Search based in-painting
- Hybrid in-painting

## A.  Texture Synthesis and block applying algorithm

Chan and Shen  [35] developed a new in-painting called Curvature  Driven  Delusion  (CDD),  and  in  a  later  paper remarkably  showed  how  the  Euler  elastic  encapsulate  both CCD in-painting and transportation in-painting. Many additional types of in-paintings methodologies were proposed  subsequently,  including  textural  in-painting  [15]­ [18]  which  relies  on  texture  matching  and  replication,  or global image statistics [19], or templates matching functional  [20].  The  matching  process  of  the  texture  inside the hole region can be speeded up through Principal Component  Analysis  (PCA)  and  Vector  Quantization  (VQ) based  techniques  [22]. The  variants  of  this  approach  are discussed below:

- i) Pixel Synthesis

S.  E.  Chen  and  L.Williams  [39]  used  "Pixel  Synthesis by  non parametric Sampling".  It is based on pixel data.  Jing Xu  and  Jian  Wu,  et  al.  [11]  implemented  8-neighborhood Fast Sweeping Method to remove the text and hidden errors on video.

- ii) Texture Synthesis

Efros and Leung  [6] have given an effective and simple algorithm for highly connected problem of texture synthesis. It  is  based on Markov Random Field and texture synthesis is done  by  pixel  by  pixel. The  pixel  is  compared  with  the neighboring  pixel  randomly.  This  algorithm  is  very  slow because the filling-in is being done pixel by pixel.  Criminisi et.al [1], [12] introduced another algorithm for texture synthesis and priority is given for filling the edges  [2],  [5].

- iii) Multi-Resolution Approach

Heeger and Bergen [28] presented Pyramid based Textures analysis based on texture at various thickness scales. They used image pyramid and matched the histogram  of  random  noise image  source  texture. lS.D Bonet [3] proposed algorithm called,  "Multi Region Texture Sampling  Procedure"  for  analysis  and  synthesis  of  texture image  which  preserves  the  structural  properties.  Igehy  and pereiva [13] described "Image Replacement through texture Synthesis". They tried smooth transition between the existing  image and the synthesized  structure. Wei and Levoy's [20] described  "Fast  Texture  synthesis  using  tree structured  vector  quantization". The  algorithm  created  by them was efficient, general and produces high quality fmeable textures.

## iv) Multi-scale Pyramids Decompositions

Mohammad Faizal,  et  al [23] introduced  this  algorithm for improving  the blurring image  done  by single scale algorithm  is  affected  by  the  effect  of  blurring.  S.Roth  and MJ.Black [37,  38] presented  a  technique  based  on  prior models. The diffusion technique is used for denoising approaches  and  was changed  and  applied to repair the damaged images.  Elad et at. [29] developed an approach by separating the image into cartoon texture layers and sparsely  represented  two  layers  by  two  incoherent over complete transforms.

## v) Block Replicating method

EFros and Freeman [6] set pixel by pixel texture synthesis method. It is used to find a minimum  error boundary  cut  between  the  existing  texture  pixel  and  new block  to  be  replaced.  Chen  and  Williams [39] extended  the idea to 3D  by calculating a linear warp held  between corresponding 3D points of two scenes,  and interpolated for views  in  between. Their  research  tries  to  deal  with  both holes and visibility ordering.

## B.  PDE based in-painting

Partial Differential Equation (PDE) needs a lot of iterations  before  the  convergence  can  be  reached. These methods [8] are  computationally expensive.  Bertalmio et al. [4] proposed  an  algorithm  based  on  both  geometric  and photometric information.  It gives border of occluded area by interactive  process  that  propagates  linear  structures  (edges) of  the  surrounding  area  also  called  Isophotes,  into  the  hole region,  using a  diffusion process.  Olivera et al [17] presents two  different  in-painting  techniques  based  on  second  order and third order PDE's. Tschumperle and Derche [18] present general vector  value  image  regularization  approach. They used high order PDE's.  Chan and Shen [35] developed a  local  variation  model based  on  Rudin-Osher-Faterin's  De­ noising  sounded  variation  image  model. Their model  is based on the users to remove  occlusion and  minimize posterior energy using second order PDE's. Mansnou proposed  an algorithm  based  on  around  connecting 'T­ Functions'. It hits the closed area through the use of geodesic  curves  with  an  aim  to  minimize  the  connected curves [9].

C. Exemplar and search based In-painting This  exemplar-based  removal  technique  performs well  for  a  wide  range  of  images  with  good  texture  and structure  replication.  But  it  has  some  difficulty  in  handling curved structures.  Criminist et.al [7] developed an algorithm that  combined  the  use  of  texture  synthesis  and  Isophote driven using priority based mechanism. This algorithm removes  big  objects  from  digital  photographs.  Wu [14] is developed a cross isophotes examplar based in-painting algorithm  based  on  the  analysis  of  anisotropic  diffusion. Wong [30] projected  a  non-local  means  approach  set  of candidate patches for examplar in-painting algorithm. Bertalmio  et  al [8] presented  new  method  which  combines the  advantages  of  partial  equations  and  texture  synthesis. The image is decomposed into structure and texture component. Fadili et al [42] developed an expectation minimization  (EM)  based  Bayesian  model  for  in-painting using  sparse  representation.  Guleryuz  et  al [41] algorithm uses adaptive sparse representation of image. A fast exemplar-based image in-painting approach is anticipated to perk up the Computation efficiency [47].

Muthukumar.S, Krishnan,et al. [16] recommended hybrid algorithms for removing objects and in-painting damaged regions.  The approach can be used in both one and two  dimensional  images  and  also  to  improve  the  efficiency of filling by Poisson method successive elimination algorithm is  employed.  K.  A.  Patwardhan, G.  Sapiro,  and M. Bertalmio [44] described  a  complete  framework  for generating new views from arbitrary viewpoints. First, cylindrical image samples are generated,  and then a form of stereo  matching  is  perfonned  on  the  cylinders  for  dense correspondence.  A.  Wong  and  J.  Orchard [43] proposed  an exemplar  based image  completion  algorithm.  It  consist  of four aspects  viz., the  size  of  image  patch  can  be  decided based  on  the  gradient  domain  of  image, the  filling  priority is  decided  by  the  geometrical  structuring  the  known  region instead  of  a  single  best  match  patches  feature  of  image, especially the curvature and the direction of the isophotes,  it introduces a better patch-matching scheme, which incorporates the curvature and color of image, the determined  source  template  is  copied  into  the  destination template  and  the  information  of  the  destination  template  is updated. Fast  exemplar-bas  is projected to perk up the computation efficiency [47].

## D.  Pixel Based In-painting Algorithm

This  method  used  to  estimate  the  non-reference  pixel which approximately fixes the point.

## i) Patch Priority Using Structure Sparsity

Hui  Yi  Huang  et  al [26] proposed  a  patch  based  in­ painting. It distinguishes the damaged areas and non damaged  areas  based  on  structures  and  textures  of  frame work and also redefme confidence and illumination variation. R. Gribonval, C. Fvotte, and  E. Vincent [45] proposed a technique is to solve the blind image  in-painting problem with the sparsity prior to the  damaged pixel.  M.  J. Fadili, J. L. Starck  and  F.  Murtagh [46] introduced In­ painting  and  zooming  using  sparse  representations  to  save the computation time.

## ii) Structure Sparsity

Structures  are  thinly  distributed  in  the  image  domain. Its neighboring  patches with larger similarities are also distributed  in  the  same  structure  or  texture  as  the  path  of interest [21], [22]. The patches with more sparsely distributed nonzero similarities are prone to be located sparsely.  Masnou and Morel  [27]  implemented  a  method  of non-occlusion  rather  than  in  traditional  one.  Zhilin  Feng  ,et al[31]  proposed  Mumford  and  Shah model  in-painting model,  which  takes  extra  care  of  the  edges  on  the  functions to  be  minimized. Esedoglu  and  Shen [32]  extension to curvature base,  proposed using the Euler's elastic.

## iii) Structure In-painting

Bertalmio  et  al  [4]  proposed  a  new  method  whose  idea is to spread the isophotes that are coming at the boundaries / edges  of  the  in-painting  region.  This  method  preserves  the arrival  angle  and  smoothens  the  inner  region.  A  curvature diffusion  based  in-painting  method  with  a  third  order  PDE is  utilized  to  implement  pixel  prediction.  Chan  and  Shen [35]  derived  an  in-painting  model  by  considering  the  image as  an  element  of  the  space  of  Bounded  Variation  (BV) images,  endowed  with  the  Total  Variation  (TV)  norm.  The solution of the in-painting problem comes from the minimization of an appropriate function, taken into consideration, curvature and connectivity principle according to the broken edges reconstruction.

## E.  Hybrid Image in- Painting Model

Jiying Wu et al  [24]  developed a hybrid  model  that  uses total  variation  equation,  divided  into  a  structure  part  and texture  part.  Textures  contain  more  dynamic  information and  the  PDF  preserves  the  linear  structure.  Structure  part  is processed  by  a  bi-directional  diffused  PDF  and  the  texture part by an exemplars-based model.  K.A.  Narayanakutti et.  al [34] developed the Hybrid Image In-painting for  Occlusion Removal  algorithm  FE  - BEMD.  This  algorithm  is  fully data-driven, unsupervised decomposition method that decomposes images into intrinsic mode functions and residue by both structure and texture synthesis.  Casells et.al [10]  proposed  a  variational  framework  exemplar  algorithm in which the image is split into structure and texture components.  Aujol  et.al  [40]  proposed  a  method  for  texture decomposition  variation, which  discriminates  texture  and noise.

## III. RESULTS ANALYSIS AND DISCUSSION

The  area  to  be  in-painted  is  selected  based  on  color, shape,  orientation,  region  selected  by  the  user  or  a  binary image specifying the missing area. Interpolation based method  works  well  if  the  in-painting  area  is  uniform.  The performance  is  high,  even  when  large  number  of  smaller areas  is  to  be  in-painted. It  results  in  blurring  along  the border and does not preserve the objects shape. The diffusion based method and TV in-painting,  work well if the unknown  area  is  smaller.  Texture  synthesis  methods  work well for  larger  unknown  area  but  generally  resulting  in undesirable boundaries.

<!-- image -->

The  more  accurate  propagation  of  the  structure,  the more  perfect  will  be  the  restored  image. It  could  also  be observed that as the thickness of the in-painting area increases,  TV in-painting  requires  more  iteration  to  in-paint the area. These methods perform well for smaller in­ painting  area;  images  with  definite  shapes  and  images  with lesser  isophotes  in  the  boundary  fail  inside  the  textured region.  Anisotropic  diffusion  based  in-painting  presents  as the  number of iterations increase beyond a certain limit the algorithm  starts  oscillating  and  does  not  produce  a  better result.

The exemplar based in-painting algorithm is capable of propagating  both  linear  structure  and  2D  texture  into  the target  region  with  a  single,  simple  algorithm  and  it  has  the limitations  similar  patches  produce  reasonable  results  and the algorithm will not handle curved structures. PDE algorithm  works  surprisingly  well,  yet  it  has  a  problem  of reconstructing  the  curved structure in  the  occlusion. The hybrid  in-painting  algorithm  works  well  only  if  the  missing region  consists  of  simple  structure  and  texture. A  failure case is due to inappropriate selection of patch size.

TABLE II COMP  ARISION OF INP AINTING TECHNIQUES

## TABLE III VIDEO (FRAME BASED) INPAINTING COMP  ARISION

| Method                                                         | Merit(s)                                                                  | Demerit(s)                                                                   |
|----------------------------------------------------------------|---------------------------------------------------------------------------|------------------------------------------------------------------------------|
| Fast Marching Method (Fmm)                                     | Good as it breaks up the known from the unknown image area                | Iff, in-painting regions are thicker and the region's boundary most agilely. |
| Modified Convolution Based Method                              | Produces the result without blurring                                      | blurring occurs, if the region to be painted thicker than 10 pixels          |
| Edge Based Greedy In-Painting Method                           | Speeded up by the design of hashing works                                 | Guides to visual inconsistencies                                             |
| Content - Aware Image In-Painting Method                       | Delivers better results for coherent pixels                               | Achieves a local optimum but not the global optimum output.                  |
| Geometry-Oriented Methods                                      | Interpolate the in- painting domain by continuing the geometric structure | Fails in the presence of structure                                           |
| Manifold Learning based Position Sequence Estimation synthesis | Used for both periodic and non periodic motion                            | Segmentation generates artifacts                                             |
| Partial Differential Equation (POE)                            | Preserve all structure information                                        | Outcome may display the blurring artifacts                                   |
| Robust image synthesis by adaptive tensor voting               | Works well for big holes in images                                        | Requires expensive segmentation stage                                        |
| Total Variation in- painting (TV)                              | Fine for removing Salt and pepper noise                                   | Models in-painting in miniature regions                                      |
| The Curvature Driven diffusion (COD) Model                     | Applicable to bigger area                                                 | Connects (some) broken edges also                                            |
| Texture Synthesis Approach                                     | Outcome not may display blurring artifacts                                | Not fit for curved structure and large and thick scratched regions.          |
| Image Replacement Through Texture Synthesis                    | Save any sharp transitions between textures                               | Inappropriate in case of wrong patch size.                                   |
| Pixel Based Synthesis                                          | Imitates both stochastic and deterministic textures.                      | Whole source sample area required to be checked                              |
| Tree Structured Vector Quantization                            | F aster output                                                            | Requires a large amounts of memory                                           |
| Pyramid Based Texture Analysis Synthesis                       | Supplies best outcomes for stochastic terms                               | Matching of histograms fails to capture more structured textures             |
| Multi Resolution Sampling                                      | Creates improved Better performance                                       | Fails when the source texture is more complex                                |
| Edge Based Algorithm                                           | To create more robust in-painting method                                  | It is powerless to fruitfully reproduce textures                             |

| Method                                                                                                 | Merit(s)                                                                                                                                              | Demerit(s)                                                                                                                  |
|--------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| Navier - Stokes, Fluid Dynamics and Image And Video In- Painting.                                      | fine outcome maintained throughout                                                                                                                    | Only fit for filling miniature non textured holes.                                                                          |
| Video refurbish: Inference of foreground and background under severe occlusion.                        | Works for large class of camera motion also                                                                                                           | Generates artifacts due to rapid change of light and shadow. Not pertinent for composite structures                         |
| Space-time completion of video                                                                         | Can entire frames or part of frames that in no way be present in the dataset. Simultaneously it handle. Handle both spatial and temporal information. | Multi-scale nature of outcome may produce blurring and also more expensive to compute. Considers only low resolution videos |
| Video epitomes                                                                                         | Synthesize data that do not have structure information                                                                                                | Includes over- smoothing artifacts and low resolution outcomes                                                              |
| Efficient object-based video in-painting.                                                              | Deals both static and moving cameras.                                                                                                                 | The numbers of positions are inadequate when the outcomes are insufficient                                                  |
| Video Repairing Under Variable Illumination                                                            | Preserves illumination variable and spatio- temporal reliability                                                                                      | It cannot mend shadows of a spoiled model                                                                                   |
| Efficient object-based video in-painting.                                                              | Address videos from both static and moving cameras.                                                                                                   | Outputs are unsatisfactory when the number of postures is insufficient                                                      |
| A Rank Minimization Approach                                                                           | Used for non-periodic motion.                                                                                                                         | Cannot handle scaling or deformations                                                                                       |
| Video in-painting under constrained camera motion                                                      | Works well in rich and It cluttered backgrounds and it well in rich work                                                                              | Do not maintain temporal continuity and hence produces flickering artifacts                                                 |
| Exemplar-based video in-painting (without ghost shadow arti facts by maintaining temporal continuity ) | It gives very few ghost shadows different other in-painting techniques. It can deal with various camera movements                                     | Block matching and select the continuous blocks block matching issues are issues which are to be handled                    |
| Generalized Patch Match correspondence                                                                 | It is very important in field in CAD which is used to reduce the spike noise.                                                                         | It is difficult to measure a visual sensitivity                                                                             |

## IV  C ONCLUSION

In this paper, different types of in-painting techniques  are  studied  and  analyzed  for  removing  objects and in-painting in damaged photographs. For each of the

algorithms,  researchers  have  experimented the  images  for  a

different environmental conditions used for filling an

occlusion  making  use  of  images and/or video, wherever

appropriate.  Digital  in-painting  techniques  can  be  used  for

all kinds of image/frame repairs,  such as removing text from

an image, erasing lines/object from

a

scenic view, or

repairing cracks and scratches.

The success of the in­

painting algorithm lies in how well the information

(photometry),  color,  shape  and  the  structure  (geometry)  are

propagated into the unknown area.

TABLE IV COMPARISION OF PSNR AND COMPUTATION TlME

<!-- image -->

It  can  be seen that  each  methods have good restoration capability, but  the  problem  is  with  the  reconstruction  of original  images.  The  main  problem  is  in  edges.  The  sharp edges  lead  to  higher  errors  than  in  the  image  with  slow changes.  Each  of the  algorithms  presented  here have  a number  of advantages and limitations. If there are not enough  samples  in the  image, it will be impossible  to synthesize the desired image. In addition to these shortcomings,  there  are  certain  cases  where  the  in-painting algorithms would fail to successfully reconstruct the image.

The  researchers  are  currently  working  on  this  area and would like to propose an efficient method for  obtaining better  results  and  to  design  a  generalized  framework  for  all kinds of occlusions/distortions.  The authors future work will concentrate  on  extending  analysis  to  in-painting  of  video  / images  where  the  challenge  is  to  preserve  visual  coherency of in-painting results over time with better quality.

## REFERENCES

- [I] ACriminisi,P. Perez. and K. Toyama, "Object Removal by Exemplar-based In-painting," in Conf.  Computer Vision and Pattern Recong, CVPR'03, vol.  2, pp. 721-728, June 2003.
- [2] ARares, M. J. T. Reinders,  and  J.  Biemond,  " Edge-based Image Restoration",  IEEE Transactions on Image Processing, vol.  14, pp. 1454-1468, Oct 2005.
- [3] J.S.D.  Bonet,  "Multi  resolution  Sampling Procedure  for  Analysis and  Synthesis  of  Texture  Images",  Computer  Graphics,  vol.  31, Annual Conference Series, pp.361-368, 1997.
- [4] M.Bertalmio,G.Sapiro, V.Caselles and C.Ballester, "Image in­ painting", in Siggraph 2000, Computer Graphics Proceedings, PP.417-424, ACM  Press  / ACM SIGGRAPH  /  Addison Wesley Longman, 2000.
- [5] Aurelie Bugeau, Marcelo BertalmIO, "Combining Texture Synthesis and Diffusion for Image In-painting",  Proceedings  of the  4  th  Inter.  Conf.  on Computer  Vision  Theory  and Applications, 2009.
- [6] AA  Efors and T.K.  Leung, 'Texture synthesis by non-parametric sampling," in ICCV(2), pp.  1033-1038,1999.
- [7] ACriminisi,  P.Perez  and K.Toyama,  "Region  Filling  and  Object Removal  by  Exemplar  based  Image  In-Painting", IEEE  Trans.  on Image Processing, vol. 13, pp. 1200-1212, Sep 2004.
- [8] M.Bertalmio, "Processing of flat and non-flat image information on arbitrary  manifolds using  partial  Differential  Equations",  Computer Eng. Program, 200  I.
- [9] G.  T.  N.  Komodakis  (2007),  "Image  completion  using  efficient belief  propagation  via  priority  scheduling  and  dynamic  pruning," IEEE Trans. Image Process., vol. 16,  pp. 2649- 2661.
- [10] P.  Arias,  V.  Caselles,  G. Facciolo,  and  G.  Sapiro,  Variational framework for exemplar-based image in-painting, Int. Journ. Computer.Vision, 93:3 (2011), 1-29.
- [II] Jing Xu,Daming Feng,Jian Wu,Zhiming Cui, "An Image In-painting Technique Based on 8-Neighborhood Fast Sweeping Method", WRI International Conference on Communications and Mobile Computing - Volume 03, Pages 626-630, ISBN: 978-0-7695-3501-2, IEEE Computer Society Washington, DC, USA, 2009.
- [12] ARares, M. J. T. Reinders , and  J.  Biemond,  " Edge-based Image Restoration,"  IEEE Transactions on Image Processing, vol.  14, pp. 1454-1468, Oct 2005.
- [13] Homan  Igehy,  and Lucas Pereira,"Image Replacement  through Texture Synthesis" ICIP 3, page 186-189. (1997).
- [14] J.  Y.  Wu,  Q.  Q.  Ruan,  "Object  Removal  By  Cross  Isophotes Exemplar-based  In-painting",  Proc.  Conf.  on  Pattern  Recongnition (ICPR2006), Hong Kong, Aug. 2006, vol .3, pp:810-813.
- [IS] Jiying  Wu,  Qiuqi  Ruan,  Gaoyun,  "An.  Examplar based  Image Completion Model Employing POE Corrections", Informatic, 21  (2), 259- 276, 2010.

- [16] Muthukumar. S, Dr. N. Krishnan, Pasupathi P and Deepa.S, "Analysis of Image In-painting Techniques with Exemplar Poisson, Successive Elimination and 8 Pixel Method" , International Journal of Computer Applications, vol. 9, issue II, pp. IS-18., Nov 2010.
- [17] M. Oliveria,  B. Bowen,  R.  McKenna,  and  Y.-S.  Chang,  "Fast Digital Image In-Painting."in VIIP, pp. 261-266, 2001.
- [18] D.Tschumperle and R. Deriche, "Vector-Valued Image Regularization With  PDEs: A  Common Framework for  Different Applications," IEEE  Trans  on  Pattern  Analysis  and  Machine Intelligence, vol. 27 , pp.  S06-SI7, Apr 200S.
- [19] Zhaolin Lu, He Huang, Leida Li, and 0 Cheng, "A Novel Hybrid Image In-painting Model" Presented at the IEEE International conf on Genetic and Evolutionary Computing, sep 2010.
- [20] Li-Yi  Wei  and  Marc  Levoy,  "Fast  Texture  Synthesis  using  Tree­ structured  Vector  Quantization",  In  Proceedings  of  SIGGRAPH 2000.
- [21] S.Masnou,  "Disocclusion: A  Variational  Approach  using  Level Lines,"  IEEE Transactions on Image Processing, vol. II, pp.68-76, feb 2002.
- [22] M.  Bertalmio,  L.  Vese,  G.  Sapiro,  et.al.,  "Simultaneous  Structure and Texture Image in-painting", Proc. Conf.  Compo  Vision Pattern Rec., Madison, WI,  2003.
- [23] Mohammad  Faizal,  Ahmad  Fauzi,  Paul  H.  Lewis,  "A  Multiscale Approach to Texture-based Image Retrieval", Pattern Analysis and Applications  Springer -Verlag London Limited, September 2007, pp II:I4I-IS7.
- [24] Jiying Wu, Qiuqi Ruan,"A Novel Hybrid Image in-painting" IEEE Transactions on Image Processing, May, 2008.
- [25] Zongben Xu and Jian Sun ,"Image in-painting by Patch Propagation Using Patch Sparsity" IEEE Transactions on Image Processing, vol 19, no.5, May 2010.
- [26] Hui-Yu Huang and Chun-Nan Hsiao," A Patch Based Image In-Painting  Based  on  Structure  Consistence"  IEEE Trans  on Image Processing, 978-10/20  I O.
- [27] S.Masnou, 1M Morel,"Level Lines based Disocclusion", Proceeding of the 5th IEEE International conference on Image processing, Chicago,  1998,vol 3,pp.259-263.
- [28] D.J.Heeger and J.R.Bergen, "Pyramid based Texture Analysis Synthesis", SIGGRAPH,PP,229238,199S.
- [29] M.Elad, J,L,Starck, P.Querre, and .L.Donoho, "Simultaneous cartoon and Texture Image in-painting using Morphological Component  analysis",  Compu!..  Harmon  Anal.,  Vo1.l9,  PP:  340358,200S.
- [30] A. wong and  1.0rchard," A Nonlocal Means Approach to Exemplar based In-painting," in proce. IEEE ICIP, 2008, pp: 2600-2603.
- [31] Zhilin  Feng,  Jianwei  Yin,Duanyang  Zhao  and  Xiaoming  Liu  "A Variational Approach  to  Medical  Image  In-painting  Based  on Mumford Shah Model",  IEEE  Tranaction on Service Systems and Service management, 2007 Transaction, page I-S.
- [32] S.Esedoglu, J.shen, Digital In-painting based  on  the  Mumford Shaheuler Image Model , Eur.J.Appl. Math. 13 (2002) 3S3-370.
- [33] M.M. Oliveria,  B. Bowen,  R.  McKenna,  and  Y.-S.  Chang,  "Fast digital image inpainting", in VIlP , pp. 261-266,200  I.
- [34] K.A.Narayanakutty,  Hema  P.Menon  and  Devasruthi.D,  "FEBEMD and Exemplar based Hybrid Image In  painting for Occlusion Removal", International Journal of Computer Applications(09758887) volume 28 No 8, August 2011.
20. [3S] I.F.  Chan ,  and  J.Shen, "Non  Texture  in-painting  by  Curvature­ Driven Diffusions (COD)", Journal of  Vis. Comm. Image Rep., vol. 4,no. 12, pp:436-449, 2001.
- [36] X.Li and Y.Zheng "Patch based Video Processing : a Variational  Bayesian  Approach",  IEEE  Transaction  on  circuits  and Systems for video Technology, Vo1.l9, no 10, pp.2476, b2491,2007.
- [37] S.Roth  and  M.J.  Black  "Fields  of  Experts  :  A  Framework  for Learning  Image  Priors,"  in  Proc  .IEEE  Computer  Society  Conf. Computer vision and pattern recognition, 200S, pp.860-867.
- [38] S.Roth and MJ. Black "Steerable  Random Fields", IEEE Computer Society Conf  Computer vision and pattern Recognition ,2007,pp.I8.
- [39] S. E. Chen and L. Williams. View interpolation for image synthesis. Computer Graphics (SIGGRAPH 1993), 27 (Annual Conference Series): 279-288, 1993.
- [40] 1.-F. Aujol and A. Chambolle, "Dual Norms And Image Decomposition Models," Int. Journal on Computer Vis., vol. 63, no. II, pp. 8S-I04, Jun. 200S.
- [41] O. G. Guleryuz, "Nonlinear Approximation Based Image Recovery using Adaptive Sparse Reconstructures  and Iterated Denoising-Part II:  Adaptive  Algorithms," IEEE Trans.  Image Process., vol. IS, pp. 555-571, 2006.
- [42] M. J. Fadili, J. L. Starck, and F. Murtagh, "In  painting and Zooming using Sparse  Representations," The Com  put. Journal , vol. S2, no. I, pp. 64-79, 2009.
- [43] A. Wong and J. Orchard, "A Nonlocal-Means Approach to Examplar-Based  In-Painting,"  presented  at  the  IEEE  Int.  Conf Image Processing, 2008.
- [44] K.  A.  Patwardhan,  G.  Sapiro,  and  M.  Bertalmio,  "Video InPainting  of  Occluding  And  Occluded  Objects,"  ProC.  IEEE  Int. Conf. Image Process., pp. 69-72, 200S.
30. [4S] R. Gribonval, C. Fvotte, and E. Vincent, "Performance Measurement  in  Blind  Audio  Source  Separation,"  IEEE  Trans. Speech, Audio, Lang. Process., vol. 14, no. 4, pp. 1462-1469, 2006.
- [46] M.1. Fadili,  1. L. Starck, and F. Murtagh, "Inpainting and Zooming using Sparse Representations," The Comput.  1. , vol. 52, no. I, pp. 64--79, 2009.
- [47] Hui-qin  Wang,  Qing  Cheni,  et  aI,"  Fast  Exemplar-based  Image Inpainting Approach" Proceedings of the 2012 International Conference on Machine Learning and Cybernetics, Xian, IS-I7 July, 2012.