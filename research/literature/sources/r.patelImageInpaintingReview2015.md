## Image Inpainting - A Review of the Underlying Different Algorithms and Comparative Study of the Inpainting Techniques

Kaushikkumar R. Patel PG Student, Digital Communication, at NIIST, Bhopal, Madhya Pradesh, India Lalit Jain Asst.Prof. in Electronics &amp; Communication Engineering, at NIIST, Bhopal, Madhya Pradesh, India Ankurkumar G.Patel Asst.Prof. in Computer Sci.&amp; Engineering at MGITER, Navsari, Gujarat, India

## ABSTRACT

Inpainting also known as retouching is the process by which we try to fill in the damaged or missing portions of an image in such a way that it is unable for the person seeing the image to find the fault within the image.  Digital Image In painting, a relatively  young  research  area  is  an  art  of  filling  in  the missing  or  corrupted  regions  in  an  image  using  information from  the  neighboring  pixels  in  a  visually  plausible  manner, while  restoring  its unity. It is helpfully used  for object removal  in  digital  photographs,  image  reconstruction,  text removal, video restoration, special effects in movie discussions and so on. There are numbers of method used for image inpainting. All methods have their own advantage and disadvantages.  This  paper  presents  a  comparative  study  and review of different image in painting techniques. The algorithms are analyzed theoretically as well as experimentally.

## Keywords

Inpainting, Texture, Structure,  Image,  Occlusion,  Object Removal, Algorithm, Exemplar

## 1. INTRODUCTION

There are lots of advantages multimedia  instruments in today's world peoples are clicking lots of images or picture of theirs  and  also  trying  to  preserve  their  past  pictures.  And  as the  time  passes  on,  those  pictures  got  damaged  (cracks, starches, image data loss, unwanted etc.)

Image  In-painting  has  received  considerable  attention  in  the past few years and has become a very active field of research in image processing. The concept of image inpainting commonly  known  as disocclusion or image completion originates from the medieval art practices, the motive being to recreate  the  lost  or  damaged  structures  in  order  to  bring  the medieval  art  up  to  date  so  as  to  make  the  damaged  artwork more discernible, while conserving its unicity[4],[23].

<!-- image -->

Fig.1: Manual Inpainting performed by a professional artist.

<!-- image -->

It has been widely used in restoring scratched old photos and old  films,  text  removal  from  images,  digital  zooming  and special  effects  in  movies,  now  it  is  also  applied  in  add  or remove objects in photos [13]. Figure 1 shows the manual inpainting done by creative people for centuries to retouch the oil  paintings  in  a  room.  The  inside-painting  problem  first appeared in the digital domain under the name 'error concealment'  in  telecommunications  where  the  need  was  to recreate lost or damaged image structures / blocks that have been lost during data transmission and coding (e.g. streaming video) in order to make it more legible. The principal goal of Digital  Image  In-painting  is  to  redo  the  lost  or  deteriorated components of an image or videos utilizing spatial data from its neighboring countries in order to take  the  damaged paintings  or  old  pictures  close  to  the  original  or  back  to  the original state so that it seems natural to the human eye [16]. Many works on Inpainting have been proposed these  recent years.  The  image  is  to  decompose  the  original  image  into  a structure  and  a  texture  image.  That  reconstruction  of  each picture is performed individually. The missing structure component  is  fixed  using  a  structure  Inpainting  algorithm, while  the  texture  component  is  repaired  by  exemplar  based synthesis technique [18].

Paper Outline : The next section of this paper presents various digital  image  inpainting  techniques  and  their  experimental results which include PDE based inpainting, Texture synthesis based inpainting, Exemplar based inpainting, Hybrid inpainting and Fast Digital in-painting and Convolution based Method.  Results  from  various  techniques  are  analyzed  and compared  in  Section  III.  Diverse  applications  of  in-painting techniques  talk  over  in  Section  IV.  At  long  last,  concluding remarks based on experimental results obtained from both real scene photographs and synthetic images are shown in part IV.

## 2. INPAINTING ALGORITHMS

The  most challenging task in  image  in  painting  technique  is the  evaluation  of  the  quality  of  the  image  so  that  the  inside painted image seems reasonable to the human beholder. The survey contains several inpainting algorithms that were developed.

.

Fig.2: Inpainting method

<!-- image -->

Nowadays, there are different techniques of image inpainting are  available.  Several  attacks  have  been  employed  by  the researchers for  digital  image  in  painting  which  may  be separated into the following broad classes:



-  Partial Differential Equation (PDE) based In painting. 



-  Texture Synthesis based In painting. 
-  Hybrid In painting. 
-  Exemplar based in painting. 
-  Fast Digital In painting and Convolution Based Method.



Thus  we  can  easily  read  the  methods  and  class  them  into various categories as follows.

## 2.1 Partial Differential Equation (PDE) based In painting

The first digital inpainting technique was the diffusion based in-painting. In this technique the corrupted or the missing part is filled by diffusing information from the surrounding countries into the missing region at the pixel point in a way that the changes seem undetectable to the beholder(observer).

Isotropic Diffusion stems from the one-dimensional heat flow equation:

<!-- formula-not-decoded -->

Where ID is the deteriorated version of the original image I and ∆I is  the  image  Laplacian.  The  time  variable  t'  denotes the  progression  of  the  function  I  in  a  sequential  continuous manner.  Isotropic  diffusion,  which  minimizes  the  variations acts as a low pass filter that suppresses high frequencies in an image. And it is because of this reason that isotropic method suffers  from  blur  close  to  the  contours.  Thus  more  general methods  using  nonlinear  Partial  Differential  Equations  that were earlier used for fluid dynamics have been introduced to preserve sharpness and edges thereby increasing the efficiency  of  this  method.  Figure  4  shows  the  in-painted output of the damaged image using PDE technique. Bertalmio et.al [20] have proposed a digital image inpainting algorithm based  on  Partial  Differential  Equation  (PDE)  in  which  the counselling of the lines of equal luminescence i.e. isophotes (figure  5)  is  maintained  by  measuring  the  direction  of  the largest    spatial  change  obtained    by  working  out  a  gradient vector    and  rotating  this    vector  by    90    radians.  Bertalmio et.al took the ideas from computational fluid dynamics (CFD) to continue the isophote lines into the region to be in painted [21]. It processes the image intensity as a stream function and the  Laplacian  of  the  image  as  vorticity  of  the  fluid  which continues  into  the  area  to  be  in  painted  by  vector  fields defined by stream function.

<!-- image -->

<!-- image -->

A                                   B

Fig.3: In-painted results of PDE based image-painting(a) The Original Image, (b) The mask, (c) The Result

<!-- image -->

This  is  done  by  formulating  a  partial  differential  equation which propagates the information (Laplacian of image) in the direction of minimal change using 'isophote lines' (the lines of  equal  gray  value)  [8],  [9].  This  method  is  thus  based directly on the Navier Strokes Equations for CFD. The PDE technique for in painting produces good results if the region to be filled in are small but if the missed regions are large, the efficiency of this algorithm decreases as it takes a long time to fill in large missing areas and the results produced will also be unsatisfactory.

Fig.4: Direction of the melodic phrases of equal gray value (isophot   lines).

<!-- image -->

The  first  PDE  base  approach  given  by  Bertalmio  [13].  It employs the concept of the isophotes (linear edges of surrounding country) and dissemination process. The primary trouble  with  this  method  is  that  due  to  blurring  effect  of diffusion process replication of large texture is not executing good. Pixels on  edges are  also  not  covered  right.  TV  (Total Vibrational)  model  is  proposed  by  Chang  and  Shen  which uses anisotropic diffusion and Euler-Lagrange equation. From TV  model,  another  algorithm  presented  based  on  the  CDD (Curvature Driven Diffusion model) which include curvature information of the isophotes.

## 2.2 Texture synthesis based image inpainting

PDE  based  techniques  are  easily  suited  for  filling  in  small gaps,  text  overlays,  etc.  But  PDE  technique  usually  fails  if applied  to  areas  containing  regular  patterns  or  to  a  textured field. This failure is because of the following reason:

1. Mostly high intensity gradients are present in grains which may be interpreted wrongly as edges and falsely carried into the region to be painted.
2.  In case of PDE based in painting, the information used is simply the boundary condition that is present within a narrow circle around the area to be painted. Therefore, it is unacceptable to recognize structures, textured areas or regular patterns from such an insignificant quantity of data.

In  this  method,  holes  are  filled  by  sampling  and  copying neighboring  pixels  [19],  [21].  The  chief  dispute  between different texture based  algorithms  is  how  they  maintain continuity  between  hole's  pixel  and  original  image  pixels. This method is simply running for selected number of images, not with all. Yamauchi  et.al presented algorithm  which generates  texture  under  different  brightness  conditions  and workplace for multi resolution [3].

(a) Input Image             (b) Texture Synthesis

<!-- image -->

Fig.5: Example of texture synthesis (a) Input Corrupted Image, (b) In-painted output using texture synthesis

The fast synthesizing algorithm presented in [19], uses image quilting  (stitching  small  batches  of  existing  images). All texture based methods are different in terms of their capability to generate texture with different color intensity, gradient and statistical features. Texture synthesis based inpainting method not handle real well for natural images. These methods don't handle  edges  and  boundaries  well.  In  some  fonts  the  user requires to enter which texture to replace with which texture. Hence  these  methods  are  used  for  small  area  of  Inpainting [10].  These  algorithms  have  difficulty  in  handling  natural images.  Texture  synthesis  techniques  could  be  applied  in repairing digitized photographs and if the damaged area needs to  be  filled-in  with  some  regular  pattern,  texture  synthesis does a good job.

## 2.3 Exemplar based Inpainting

The  exemplar  based  approach  is  an  important  inpainting algorithm and they have proven to be very effective. Basically it  consists  of  two  basic  steps:    first  step  priority  assignment and  the  second  step  is  the  selection  of  best  matching  patch. The  exemplar  based  approach  regarding  the  best  matching pieces  from  the  source  region,  whose  similarity  is  measured by  certain  metrics,  and  pastes  into  the  aim  field.  Exemplar based Inpainting iteratively synthesizes the target neighborhood, by the most similar patch in the source region [8]. According to the filling order, the method fills structures in the missing regions using known information of neighboring regions. Generally, an exemplar-based Inpainting algorithm includes the following main steps:

- 1) Initializing the Target Region, in which the initial missing areas are extracted and interpreted with appropriate data structures.
- 2) Computing Filling Priorities, in this a predefined priority function  is  applied  to  compute  the  filling  order  for  all unfilled pixels p ∈ δΩ in the root of each filling iteration.
- 3) Searching Example and Compositing, in which the most similar  model  is  explored  from  the  source  region  Φ  to compose the given patch, Ψ (of size N × N pixels) that centered on the given pixel p.

<!-- image -->

<!-- image -->

(a)                                               (b)

Fig. 6: Removing Large Objects using Exemplar based Texture Synthesis (a) Original Image, (b) Developed Mask, (c) In-painted Result

<!-- image -->

4) Updating Image Information, in which the boundary δΩ of the target region Ω and the required information for computing, filling priorities are updated. Numbers of algorithms  are  developed  for  the  exemplar  based  image Inpainting.

Bertalmio  [13]  produced  a  hybrid  algorithm  to  combine  the diffusion-based system and texture synthesis [14]. This algorithm works well in recovering not only the geometrical structures,  but  besides  the  small  texture  regions.  Criminisi [12]  developed  an  effective  and  uncomplicated  access  to encourage filling in from the boundary of the missing region where  the  strength  of  isophote  nearby  was  strong,  and  then used  the  sum  of  squared  difference  (SSD)  to  choose  a  best matching  patch  among  the  candidate  source  patches.  This algorithm of criminisi the region filling order is specified by the  priority  based  mechanism.  Cheng  [10]  generalized  the priority function for the family of algorithms given in [12] to provide a more robust operation.

Wong  [7]  developed  a  weighted  similarity  function.  That function uses several source patches to reconstruct the target patch instead of utilizing a single source patch J. WU [3] used the structure generation and  Bezier curves to  make  the missing  edge  information.  Using  the  structural  information and reconnecting contours by curve filling process, the damaged regions will be planted. Inwards [13] first decomposing  the  original  image  into  the  entirety  of  two pictures, one capturing the basic image structure and the other capturing  the  texture.  The  first  image  (structure  image)  is inpainted  following  the  work  of  Bertalmio  et  al.  [21],  while the  other  one  is  filled-in  with  a  texture  synthesis  algorithm following the work of Efros et al. [22].  The algorithm works well enough for well designed structures in the image, but in the  event  of  natural  images  the  structures  do  not  have  well defined boundaries so the effects might not be correct.

In [12] Inpainting techniques fill holes in images by propagating linear structures (called isophotes in the inventing literature) into the target  region  via  diffusion.  They  are motivated by the partial differential equations of physical heat flow, and work convincingly as restoration algorithms. Their drawback is that  the  diffusion  process  introduces  some  blur, which  becomes  noticeable  when  filling  larger  areas.  The technique  presented  here  combines  the  hard  spots  of  both approaches into a single, efficient algorithm. In this story the author proposed an exemplar-based image completion algorithm. Their method computes the priorities of patches to be  synthesized  through  a  best-first  greedy  strategy  which depends upon the precedence attributed to each spell on the filling-front,  where  the  patch  filling  order  is  defined  by  the angle between the isophotes direction and the normal way of the  local  filling  front.  This  algorithm  works  well  in  large missing areas and textured areas. The technique is capable of propagating  both  linear  structure  and  flat  texture  into  the target region with a single, simple algorithm.

In [10] instead of creating a new inpainting scheme, they try to generalize the priority function for the family of algorithms given  in  [12]  to  provide  more  robust  performance.  In  this paper  presents,  the  new  priority  function  is  able  to  resist undesired  noises and  robust to  the  aforementioned  over amplified phenomenon. The computational complexity of the proposed  algorithm  is  dominated  by  two  tasks:  exemplar search and component weight selection. Hence, the limitation of  this  report  will  concentrate  on  the  investigation  of  an efficient searching scheme and on the automatic discovery of component weights for different sorts of pictures.

In  [5]  they  develop  the  proposed  algorithm  is  basically  an extension to the algorithm proposed by Criminisi et al. [12]. Using this algorithm, we can inpaint large missing regions in an  image  as  well  as  reconstructed  small  defects.  This  attack used a priority count based on specifying a means  of differentiating between patches that bear the same  minimum mean squared error with the selected tour. This technique can be applied to fill minor scratches in the icons/photos as well as to remove larger objects from them. It is also computationally efficient and runs well with larger images.

In [3] a new hybrid image inpainting method based on Bezier curves which combines the exemplar-based inpainting technique and the edge-based image restoration algorithm. For restoring image structures In that first use the segmentation by Otsu's  thresholding  to  obtain  the  information  of  edges  and Bezier  curves  used  to  reconstruct  the  image  skeletons  in missing  areas.  Using  this  approach  we  restore  approximates incoming edges with only lines and circle arcs which preserves successfully the curvature structures in the damaged picture.  These  methods  solve  the  disadvantage  of  Exemplar based approach for handling ambiguities in which the missing region covers the convergence of two countries and achieves better  results  than  the  conventional  edge  based  restoration method.

In [2] a robust exemplar-based image in-painting algorithm is proposed.  The  algorithm  uses  a  novel  priority  function  to regulate the inside-painting order. Structure tensor is adopted in the priority role. Structure tensor contains the transformation direction of the image and the size of transformation  in  these  ways.  So,  a  patch  along  a  strong geometric  structure  has  a  higher  in-painting  priority  and would be discharged prior to other pieces to keep the analog structure.

In [1] here they deliver a novel of exemplar-based inpainting as a global energy optimization problem, written in terms of the offset map. The proposed energy function combines a data attachment term that ensures the continuity of reconstruction at the limit of the inpainting domain with a smoothness term that ensures a visually coherent reconstruction inside the butt area.  This  novel  formulation  is  adjusted  to  receive  a  global minimum  using  the  graph  cut  algorithm.  To  reduce  the computational  complexity,  they  propose  an  efficient  multiscale  graph  cut  algorithm.  Propose  a  multi-scale  graph  cut algorithm to efficiently resolve the energy minimization problem  in  which  a  feature  vector  representation  is  brought out  to  compare  patches  at  low  resolution,  to  redress  the information loss. This office can significantly eliminate ambiguities and improve the accuracy of the offset map. The reasons  for using  a multi-scale graph  cut algorithm  are twofold: the first one is obvious to reduce the computational cost, the graph labeling at the original image resolution being computationally  intensive  because  of  the  large  number  of labels and pixels in the objective country. The second is that the  smoothness  term,  that  uses  at  each  scale  the  4  nearest neighbors, can capture more spatial information at the lowest resolution  layer.  This  aids  toward  off  getting  stuck  into  bad local  minima.  For  the  same  cause,  besides  the  multi-scale strategy is used in the Patch Match method.

## 2.4 Hybrid Inpainting

Partial  differential  equation  based  methods  perform  well  for low  and  sparingly  distributed  gaps.  They  are  suitable  for sparse smooth images and for disseminating strong structures, but  PDE  based  methods  are  unable  to  synthesize  textures [22]. On the other hand, exemplar based in painting performs well in synthesizing textures with regular/homogenous patterns. Unlike PDE they do not maintain the linear structure of  images  with  small  holes  spread  over  most  of  the  image field.

Natural  images  are  composed  of  composite  structures  and textures. The structures comprise of primary sketches (corners,  edges)  of  an  image  and  textures  are  regions  with regular/ homogenous feature statistics. A technique is therefore  needed  to  handle  complex  images  containing  both structures and textures.

Fig.7: Decomposition of an image into Structural (geometric) and textural components

<!-- image -->

Hybrid  image  in  painting  technique  mostly  used  for  filling large missing  regions  preserve  both  the  texture  and  the structure of an image so that the changes in the in painted area are  undetectable  to  the  human  eye.  The  hybrid  approaches combine both texture synthesis and PDE based Inpainting for completing the objective country. The main idea behind these approaches is that it cracked down the icon into two separate parts, Structure region and texture regions [17].

The  decomposed  regions  are  occupied  by  edge  propagating algorithms and texture synthesis techniques. These algorithms are computationally intensive unless the fill area is low. This technique uses a two-step approach: the first stage is structure completion done by texture synthesis. The  second step consists  of  synthesizing  a  texture  and  color  information  in each segment, again using tensor voting [14].

## 2.5 Fast Semi-automatic Inpainting and Convolution based Method

Depending on the size of the gap that needs to be filled in, all the inside painting techniques discussed earlier require minutes to hours for an in painting algorithm to finish to fully in  paint  the  missing  areas,  thereby  making  it  inefficient  for interactive user applications.. Therefore, a novel class of fast in  painting  techniques  is  developed  to  accelerate  up  the conventional in painting techniques. A Telea introduced a fast marching  algorithm  almost  similar  to  PDE  technique,  but considerably simpler and faster to implement than other PDE methods  [15].  This  algorithm  computes  smoothness  of  an image  from  already  known  neighborhood  of  an  image  as  a weighted sum to in paint the gap [19].

Fig.8: Block Diagram of Convolution based Image Inpainting Method

<!-- image -->

The  restriction  of  this  method  is  it  produces  blur  when  the area to be in painted is thicker than ten pixels. Semi-automatic image  inpainting  requires  user  assistance  in  the  form  of guidelines  to  assist  in  structure  completion.  The  method  by Jian et.al [6] proposed inpainting with Structure propagation. This technique completes within a two-step operation. In the first  step  a  user  manually  selects  missing  information  in  the hole  by  sketching  object  boundaries  from  the  source  to  the butt  area  and  then  a  patch based  texture  synthesis  is  used  to generate the grain. For multiple objects, the optimization is a great deal more difficult and then proposes approximated the answer by using belief propagation. All the methods discussed above will take minutes two hours to fill out depending on the size of the Inpainting area and hence making it unacceptable for  interactive  user  applications.  To  hurry  up  the  image Inpainting algorithms, new  categories of fast Inpainting techniques are being developed. A new method which treats the  missing  regions  as  level  sets  and  uses  Fast  Marching Method  (FMM)  to  propagate  image  information  has  been proposed  by  Thalia  in  [11].  These  fast  techniques  are  not suitable  in  filling  large  hole  regions  as  they  lack  explicit methods  to  implant  edge  regions.  This  technique  results  in blur effect in the photo.

It is observed that the PDE based image Inpainting algorithms cannot  fill  the  large  target  (missing)  region  and  it  cannot rejuvenate  the  texture  shape.  Based  on  theoretical  analysis exemplar  based  Inpainting  will  produce  sound  results  for  In painting  the  large  missing  region  also  these  algorithms  can imprint  both  structure  and  textured  image  as  well.  But  they function  well  only  if  the  missing  region  consists  of  simple structure and texture. The Exemplar based technique combines the strong points of both approaches into a single, efficient algorithm. As with inpainting, we pay extra attention to linear structures.  But, linear structures abutting the target region only influence the fill order of  what is at  the core an exemplar  based  texture  synthesis  algorithm.  The  outcome  is an algorithm that delivers the efficiency and  qualitative performance  of  exemplar  based  texture  synthesis,  but  which also  observes  the  image  constraints  imposed  by  surrounding linear  structure.  Here  we  can  briefly  survey  on  Exemplar based Image Inpainting.

## 3. RESULTS, ANALYSIS AND DISCUSSION

Digital  image  in-painting  has  become  a  hotspot  and  has received  a  lot  of  attention  since  a  decade.  Various  advances have been presented with varying applicability in disocclusion,  texture  synthesis,  object  removal,  text  removed, and  image  reconstruction.  Table  1  shows  the  merits  and  the demerits of different in-painting approaches. Established along the varying applicability of different in-painting techniques,  all  the  approaches  are  as  significant  in  diverse fields of application.

Table 1 Comparison of image Inpainting methods

| METHODS                             | MERITS                                              | DEMERITS                                                                 |
|-------------------------------------|-----------------------------------------------------|--------------------------------------------------------------------------|
| Partial Differential Equation (PDE) | Fine Outcome & preserves All Structural information | Results display Blurring artifacts When applied to Large missing Regions |
| Total variation In-painting (TV)    | Works well for Removing salt And pepper Noise       | Models Inpainting Only in small/ Miniature regions                       |

| The curvature Driven Diffusion Model (CDD)   | Used for bigger Areas                                                            | Connects only Few broken edges                                                              |
|----------------------------------------------|----------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| Texture Synthesis Based In-painting          | Results do not Display blurs/ Artifacts                                          | Not applicable for Thick scratched Regions & curved structure                               |
| Exemplar Based Texture Synthesis             | Gives Impressive Results and Preserves all Structural & Textural Information     | Gives Unsatisfactory Results of the Corrupted region Is spread along Most of the image Area |
| Hybrid In- Painting                          | Restores Smoothness and Preserves both The linear Structure and Texture of Image | Produces blocky Effect If the damaged area is large & the Patch size is Inappropriate       |
| Convolution Based In-painting Method         | Produces fine Results without Blurring                                           | For corrupted Region thicker Than ten pixels, blurring occurs                               |

## 4. CONCLUSION AND FUTURE WORK

In this comparative study, we review the existing techniques of image Inpainting. We also have provided a concept and a description of the different Inpainting.

From this analysis, we also highlighted a number of shortcomings  and  limitations  of  these  different  techniques. Here, we highlight some of the possible research areas which can merit future investigations to extend the art of in-painting.

(i) Extension  to  3-D :  - For  the  last  few  years,  attempts have  been  made  to  extend  the  in-painting  techniques  from two dimensions (2-D) to three dimensions (3-D). This modified methodology  would  help  in  the  restoration of historical artifacts and damaged monuments. This is because the  objects  that  are  symmetrical  in  three  dimensions  (3-D) may  not  appear  symmetric  in  their  2-D  projection  /  image plane.  However,  extensive  research  is  still  needed  in  this direction for the results to be visually pleasing.

## (ii) In-painting   for   high   resolution   images :-

Reconstruction/In-painting  of  high  resolution  images  is  still an issue of concern. Dealing with high resolution images, the main issue for the inside-painting techniques is a significant increase in the computation time (up to several hours). Thus to quantify the quality of in-painting techniques, new algorithms  can  be  developed  that  would  be  able  to  implant high-resolution images.

(iii) Video In-painting :  -Video  In-painting  still  remains  a challenging and a crucial task as the arbitrary camera motion, tracking moving objects in a video, and variations in illumination conditions remain difficult problems to be solved and hence complicate the inside-painting performance. Another  issue  in  video  in-painting  is  to  preserve  the  visual coherency of the in-painted results over time.

- (iv) Reducing In-painting time : - More efficient algorithms  are  required  to  be  developed  for  reducing  the computational cost and In-painting time.
- (v) Quality Evaluation :  -Accessing the quality of the inpainted  images  is  another  open  problem  as  no  such  quality metrics exists that does not depend upon the reference image. So one has to rely on the human visual comparisons for the quality assessment.

## 5. REFERENCES

- [1] Yanking  Liu  and  Vicent Caselles,'  Exemplar-Based Image  Inpainting  Using  Multiscale  Graph  Cuts', IEEE Transactions On Image Processing ,  vol. 22, no. 5, May 2013.
- [2] Liu  Kui,  Tan  Jieqing,  Su  Benyue  '  Exemplar-based Image Inpainting using Structure Tesnor', International Conference on Advanced Computer Science and Electronics Information (ICACSEI 2013).
- [3] Jiunn-Lin Wu and Yi-Ying Chou Department of Computer  Science and Engineering National  Chung Hsing University, Taichung, 402 Taiwan'  An Effective Content-Aware  Image  Inpainting  Method', Journal  of information science and engineering 28, 755-770 (2012).
- [4] Alexandra  Ioana  Oncu  Feier.  (2012).  Digital  Inpainting for  Artwork  Restoration:  Algorithms  and  Evaluation, Master Thesis Report.
- [5] Anupam,  Pulkit  Goyal,  Sapan  Diwakar,  Information Technology,  IIIT  Allahabad,  India  'Fast  and  Enhanced Algorithm for Exemplar Based Image Inpainting', Fourth  Pacific-Rim  Symposium  on  Image  and  Video Technology 2010.
- [6] Z. Xu and S. Jian, 'Image inpainting by patch propagation using patch sparsity', IEEE Transactions on Image Processing , vol. 19, 2010, pp. 1153-1165.
- [7] A. Wong and J. Orchard, 'A nonlocal-means approach to exemplar-based  inpainting',  presented  at  the IEEE  Int. Conf. Image Processing , 2008, pp. 2600-2603.
- [8] Zhongyu  Xu, Xiaoli Lian and Lili Feng. (2008). 'Image Inpainting Algorithm Based on Partial Differential Equation', IEEE Computer Society, International Colloquium  on  Computing,  Communication,  Control, and Management ISSN:978-0-7695-3290-5.
- [9] Michael E Taschler. (2006). 'A Comparative Analysis of Image Inpainting Techniques', The University of York, pp. 01-120.
- [10] W.  Cheng,  C.  Hsieh,  S.  Lin,  C.  Wang,  and  J.  Wu, 'Robust algorithm for exemplar-based image inpainting,'  in Processing  of  International  Conference on Computer Graphics, Imaging and Visualization , 2005, pp. 64-69.
- [11] Telea,'An  Image  Inpainting  Technique  Based  On  The Fast  Marching  Method', Journal  Of  Graphics  Tools , vol.9, no. 1, ACM Press, 2004.
- [12] A  Criminisi,  P.  Pérez,  and  K.  Toyama,  'Region  filling and object removal by exemplar-based image inpainting,' IEEE Trans. Image Process. , vol. 13, no. 9, pp. 1200-1212, Sep. 2004.

- [13] Bertalmio M, Vese L, Sapiro G, Osher S. 'Simultaneous structure and texture image inpainting,' IEEE Transactions on Image Processing , 2003, 12, 882-889.
- [14] Drori D. Cohen-Or, Yeshurun H. 'Fragment-based image  completion', ACM  Transactions  on  Graphics , 2003, 22, 303-312.
- [15] J. Jia and C. K. Tang, 'Image is repairing: Robust image synthesis by adaptive ND tensor voting,' in Proceedings of  IEEE  Computer  Society  Conference  on  Computer Vision Pattern Recognition , 2003, pp. 643-650.
- [16] Shen,  J.  (June  2003).  Inpainting  and  the  Fundamental Problem of Image Processing, SIAM News, 36 (5).
- [17] Rane S,  Sapiro  G,  Bertalmio  M.  'Structure  and  texture filling of  missing image blocks in wireless transmission and compression applications'. IEEE. Trans.  Image Processing , 2002.
- [18] Chan  T,  Shen  J.  'Non  texture  inpainting  by  curvaturedriven  diffusions', Journal  of  Visual  Communication and Image Representation , 2001, 4, 436-449.
- [19] Oliviera,  B.  Bowen,  R.  Mckenna,  and  Y.  -S.  Chang. 'Fast  Digital  Image  Inpainting',  in Proc.  Of  Intl.  Conf. On Visualization, Imaging And Image Processing (VIIP) , pp. 261-266, 2001.
- [20] Bertalmio, M., Bertozzi,A.L. And Sapiro, G. (Dec 2001). 'Navier Stokes, Fluid  Dynamics, and Image and Video Inpainting', Proceedings of Conf. Comp. Vision, Pattern Rec ., Hawai, pp. 355-362.
- [21] Bertalmio M, Sapiro G, Caselles V, Ballester C. 'Image inpainting', In Proceedings of ACM  Conf. Comp. Graphics  (SIGGRAPH) ,  417-424,  New  Orleans,  USA, July 2000.
- [22] Alexei A. Efros and Thomas K. Leung, 'Texture Synthesis by Non-Parametric Sampling', IEEE International Conference on Computer Vision , pp. 10331038, 1999.
- [23] Emile-Male,  G.  The    Restorer's    Handbook    of  Easel Painting, Van Nostrand Reinhold, New York. (1976).