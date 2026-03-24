# Chapter 6: The iNaturalist Species Classification And Detection Dataset
Van Horn, Grant et al. (2018). “The iNaturalist Species Classification and Detection
Dataset”. In: _Proceedings of the IEEE Conference on Computer Vision and Pattern
Recognition_. Salt Lake City, UT.doi:10.1109/CVPR.2018.00914.

### 6.1 Abstract

Existing image classification datasets used in computer vision tend to have a uni-
form distribution of images across object categories. In contrast, the natural world
is heavily imbalanced, as some species are more abundant and easier to photograph
than others. To encourage further progress in challenging real world conditions,
we present the iNaturalist species classification and detection dataset, consisting of
859 , 000 images from over 5 , 000 different species of plants and animals. It fea-
tures visually similar species captured in a wide variety of situations, from all over
the world. Images were collected with different camera types, have varying image
quality, feature a large class imbalance, and have been verified by multiple citizen
scientists. We discuss the collection of the dataset and present extensive baseline
experiments using state-of-the-art computer vision classification and detection mod-
els. Results show that current non-ensemble based methods achieve only67%top
one classification accuracy, illustrating the difficulty of the dataset. Specifically, we
observe poor results for classes with small numbers of training examples, suggesting
more attention is needed in low-shot learning.

### 6.2 Introduction

Performance on existing image classification benchmarks such as (Russakovsky
et al., 2015) is close to being saturated by the current generation of classification
algorithms (He et al., 2016; Szegedy, Vanhoucke, et al., 2016; Szegedy, Ioffe, et al.,
2016; Xie et al., 2017). However, the number of training images is crucial. If one
reduces the number of training images per category, performance typically suffers. It
may be tempting to try and acquire more training data for the classes with few images,
but this is often impractical, or even impossible, in many application domains. We
argue that class imbalance is a property of the real world, and computer vision


Figure 6.1: Two visually similar species from the iNat2017 dataset. Through close
inspection, we can see that the ladybug on the left has _two_ spots while the one on
the right has _seven_.

models should be able to deal with it. Motivated by this problem, we introduce
the iNaturalist Classification and Detection Dataset (iNat2017). Just like the real
world, it exhibits a large class imbalance, as some species are much more likely to
be observed.

It is estimated that the natural world contains several million species with around
1.2 million of these having already been formally described (Mora et al., 2011).
For some species, it may only be possible to determine the species via genetics
or by dissection. For the rest, visual identification in the wild, while possible,
can be extremely challenging. This can be due to the sheer number of visually
similar categories that an individual would be required to remember along with the
challenging inter-class similarity; see Fig. 6.1. As a result, there is a critical need for
robust and accurate automated tools to scale up biodiversity monitoring on a global
scale (Cardinale et al., 2012).

The iNat2017 dataset is comprised of images and labels from the citizen science
website iNaturalist 1. The site allows naturalists to map and share photographic
observations of biodiversity across the globe. Each observation consists of a date,
location, images, and labels containing the name of the species present in the images.
As of November 2017, iNaturalist has collected over 6.6 million observations from
127,000 species. From this, there are close to 12,000 species that have been observed
by at least twenty people and have had their species ID confirmed by multiple
annotators.

The goal of iNat2017 is to push the state-of-the-art in image classification and

(^1) [http://www.inaturalist.org](http://www.inaturalist.org)


detection for ‘in the wild’ data featuring large numbers of imbalanced, fine-grained,
categories. iNat2017 contains over 5,000 species, with a combined training and
validation set of 675,000 images, 183,000 test images, and over 560,000 manually
created bounding boxes. It is free from one of the main selection biases that are
encountered in many existing computer vision datasets - as opposed to being scraped
from the web, all images have been collected and then verified by multiple citizen
scientists. It features many visually similar species captured in a wide variety of
situations from all over the world. We outline how the dataset was collected and
report extensive baseline performance for state-of-the-art classification and detection
algorithms. Our results indicate that iNat2017 is challenging for current models due
to its imbalanced nature and will serve as a good experimental platform for future
advances in our field.

### 6.3 Related Datasets

In this section, we review existing image classification datasets commonly used in
computer vision. Our focus is on large scale, fine-grained, object categories as
opposed to datasets that feature common everyday objects, e.g. (Fei-Fei, Fergus,
and Perona, 2007; Everingham et al., 2010; T.-Y. Lin et al., 2014). Fine-grained
classification problems typically exhibit two distinguishing differences from their
coarse-grained counter parts. First, there tends to be only a small number of domain
experts that are capable of making the classifications. Second, as we move down
the spectrum of granularity, the number of instances in each class becomes smaller.
This motivates the need for automated systems that are capable of discriminating
between large numbers of potentially visually similar categories with small numbers
of training examples for some categories. In the extreme, face identification can be
viewed as an instance of fine-grained classification, and many existing benchmark
datasets with long tail distributions exist e.g. (G. B. Huang et al., 2007; Omkar
M Parkhi, Vedaldi, Zisserman, et al., 2015; Guo et al., 2016; Cao et al., 2017).
However, due to the underlying geometric similarity between faces, current state-
of-the-art approaches for face identification tend to perform a large amount of face
specific pre-processing (Taigman et al., 2014; Schroff, Kalenichenko, and Philbin,
2015; Omkar M Parkhi, Vedaldi, Zisserman, et al., 2015).

The vision community has released many fine-grained datasets covering several
domains such as birds (Welinder et al., 2010; Wah et al., 2011; Berg et al., 2014;
Van Horn et al., 2015; Krause, Sapp, et al., 2016), dogs (Khosla et al., 2011; O. M.
Parkhi et al., 2012; J. Liu et al., 2012), airplanes (Maji et al., 2013; Vedaldi et al.,


2014), flowers (Nilsback and Zisserman, 2006), leaves (Kumar et al., 2012), food
(Hou, Y. Feng, and Wang, 2017), trees (Wegner et al., 2016), and cars (Krause,
Stark, et al., 2013; Y.-L. Lin et al., 2014; Yang et al., 2015; Gebru et al., 2017).
ImageNet (Russakovsky et al., 2015) is not typically advertised as a fine-grained
dataset, yet it contains several groups of fine-grained classes, including about 60 bird
species and about 120 dog breeds. In Table 6.1, we summarize the statistics of some
of the most common datasets. With the exception of a small number e.g.(Krause,
Sapp, et al., 2016; Gebru et al., 2017), many of these datasets were typically
constructed to have an approximately uniform distribution of images across the
different categories. In addition, many of these datasets were created by searching the
internet with automated web crawlers and as a result, can contain a large proportion
of incorrect images e.g.(Krause, Sapp, et al., 2016). Even manually vetted datasets
such as ImageNet (Russakovsky et al., 2015) have been reported to contain up to4%
error for some fine-grained categories (Van Horn et al., 2015). While current deep
models are robust to label noise at training time, it is still very important to have
clean validation and test sets to be able to quantify performance (Van Horn et al.,
2015; Rolnick et al., 2017).

Unlike web scraped datasets (Krause, Sapp, et al., 2016; Krasin et al., 2016; Wilber et
al., 2017; Hou, Y. Feng, and Wang, 2017), the annotations in iNat2017 represent the
consensus of informed enthusiasts. Images of natural species tend to be challenging,
as individuals from the same species can differ in appearance due to sex and age and
may also appear in different environments. Depending on the particular species,
they can also be very challenging to photograph in the wild. In contrast, mass-
produced, man-made object categories are typically identical up to nuisance factors,
i.e. they only differ in terms of pose, lighting, or color, but not necessarily in their
underlying object shape or appearance (Yu and Grauman, 2014; Gebru et al., 2017;
Zhang et al., 2017).

### 6.4 Dataset Overview

In this section, we describe the details of the dataset, including how we collected
the image data (Section 6.4), how we constructed the train, validation and test splits
(Section 6.4), how we vetted the test split (Section 6.4), and how we collected
bounding boxes (Section 6.4). Future researchers may find our experience useful
when constructing their own datasets.


Dataset Name # Train # Classes Imbalance
Flowers 102 (Nilsback and Zisserman, 2006) 1,020 102 1.00
Aircraft (Maji et al., 2013) 3,334 100 1.03
Oxford Pets (O. M. Parkhi et al., 2012) 3,680 37 1.08
DogSnap (J. Liu et al., 2012) 4,776 133 2.85
CUB 200-2011 (Wah et al., 2011) 5,994 200 1.03
Stanford Cars (Krause, Stark, et al., 2013) 8,144 196 2.83
Stanford Dogs (Khosla et al., 2011) 12,000 120 1.00
Urban Trees (Wegner et al., 2016) 14,572 18 7.51
NABirds (Van Horn et al., 2015) 23,929 555 15.00
LeafSnap∗(Kumar et al., 2012) 30,866 185 8.00
CompCars∗(Yang et al., 2015) 136,727 1,716 10.15
VegFru∗(Hou, Y. Feng, and Wang, 2017) 160,731 292 8.00
Census Cars (Gebru et al., 2017) 512,765 2,675 10.00
ILSVRC2012 (Russakovsky et al., 2015) 1,281,167 1,000 1.78
iNat2017 579,184 5,089 435.44
Table 6.1: Summary of popular general and fine-grained computer vision classifi-
cation datasets. ‘Imbalance’ represents the number of images in the largest class
divided by the number of images in the smallest. While susceptible to outliers,
it gives an indication of the imbalance found in many common datasets. ∗Total
number of train, validation, and test images.

**Dataset Collection**
iNat2017 was collected in collaboration with iNaturalist, a citizen science effort that
allows naturalists to map and share observations of biodiversity across the globe
through a custom made web portal and mobile apps. Observations, submitted by
_observers_ , consist of images, descriptions, location and time data, and community
identifications. If the community reaches a consensus on the taxa in the observation,
then a “research-grade” label is applied to the observation. iNaturalist makes an
archive of research-grade observation data available to the environmental science
community via the Global Biodiversity Information Facility (GBIF) (Ueda, 2017).
Only research-grade labels at genus, species, or lower are included in this archive.
These archives contain the necessary information to reconstruct which photographs
belong to each observation, which observations belong to each observer, as well as
the taxonomic hierarchy relating the taxa. These archives are refreshed on a rolling
basis, and the iNat2017 dataset was created by processing the archive from October
3rd, 2016.


Super-Class Class Train Val BBoxes
Plantae 2,101 158,407 38,206 -
Insecta 1,021 100,479 18,076 125,679
Aves 964 214,295 21,226 311,669
Reptilia 289 35,201 5,680 42,351
Mammalia 186 29,333 3,490 35,222
Fungi 121 5,826 1,780 -
Amphibia 115 15,318 2,385 18,281
Mollusca 93 7,536 1,841 10,821
Animalia 77 5,228 1,362 8,536
Arachnida 56 4,873 1,086 5,826
Actinopterygii 53 1,982 637 3,382
Chromista 9 398 144 -
Protozoa 4 308 73 -
Total 5,089 579,184 95,986 561,767
Table 6.2: Number of images, classes, and bounding boxes in iNat2017 broken
down by super-class. ‘Animalia’ is a catch-all category that contains species that
do not fit in the other super-classes. Bounding boxes were collected for nine of the
super-classes. In addition, the public and private test sets contain 90,427 and 92,280
images, respectively.

**Dataset Construction**
The complete GBIF archive had 54k classes (genus level taxa and below), with
1.1M observations and a total of 1.6M images. However, over 19k of those classes
contained only one observation. In order to construct train, validation, and test
splits that contained samples from all classes, we chose to employ a taxa selection
criterion: we required that a taxa have at least 20 observations, submitted from
at least 20 unique observers (i.e. one observation from each of the 20 unique
observers). This criterion limited the candidate set to 5,089 taxa coming from 13
super-classes, see Table 6.2.

The next step was to partition the images from these taxa into the train, validation,
and test splits. For each of the selected taxa, we sorted the _observers_ by their
number of observations (fewest first), selected the first 40% of observers to be in
the test split, with the remaining 60% to be in the “train-val” split. By partitioning
the observers in this way and subsequently placing all of their photographs into
one split or the other, we ensure that the behavior of a particular user (e.g. camera
equipment, location, background, _etc._ ) is contained within a single split, and not
available as a useful source of information for classification on the other split for
a specific taxa. Note that a particular observer may be put in the test split for one


(^01000) Sorted Species 2000 3000 4000 5000
101
102
103
Number of Training Images
Figure 6.2: Distribution of training images per species. iNat2017 contains a large
imbalance between classes, where the top1%most populated classes contain over
16%of training images.
taxa, but the “train-val” split for another taxa. By first sorting the observers by
their number of observations, we ensure that the test split contains a high number
of unique observers and therefore a high degree of variability. To be concrete, at
this point, for a taxa that has exactly 20 unique observers (the minimum allowed), 8
observers would be placed in the the test split, and the remaining 12 observers would
be placed in the “train-val” split. Rather than release all test images, we randomly
sampled∼183,000 to be included in the final dataset. The remaining test images
were held in reserve in case we encountered unforeseen problems with the dataset.
To construct the separate train and validation splits for each taxa from the “train-val”
split, we again partition on the observers. For each taxa, we sort the observers by
increasing observation counts and repeatedly add observers to the validation split
until either of the following conditions occurs: (1) the total number of _photographs_
in the validation set exceeds 30, or (2) 33% of the available _photographs_ in the
“train-val” set for the taxa have been added to the validation set. The remaining
observers and all of their photographs are added to the train split. To be concrete,
and continuing the example from above, exactly 4 images would be placed in the
validation split, and the remaining 8 images would be placed in the train split for a
taxa with 20 unique observers. This results in a validation split that has at least 4
and at most∼30 images for each class (the last observer added to the validation split
for a taxa may push the number of photographs above 30), and a train split that has


Figure 6.3: Sample bounding box annotations. Annotators were asked to annotate
up to 10 instances of a super-class, as opposed to the fine-grained class, in each
image.

at least 8 images for each class. See Fig. 6.2 for the distribution of train images per
class.

At this point we have the final image splits, with a total of 579,184 training images,
95,986 validation images, and 182,707 test images. All images were resized to
have a max dimension of 800px. Sample images from the dataset can be viewed in
Fig. 6.11. The iNat2017 dataset is available from our project website 2.

**Test Set Verification**

Each observation on iNaturalist is made up of one or more images that provide
evidence that the taxon _was present_. Therefore, a small percentage of images may
not contain the taxon of interest but instead can include footprints, feces, and habitat
shots. Unfortunately, iNaturalist does not distinguish between these types of images
in the GBIF export, so we crowdsourced the verification of three super-classes
(Mammalia, Aves, and Reptilia) that might exhibit these “non-instance” images.
We found that less than 1.1% of the test set images for Aves and Reptilia had non-
instance images. The fraction was higher for Mammalia due to the prevalence of
footprint and feces images, and we filtered these images out of the test set. The
training and validation images were not filtered.

(^2) https://github.com/visipedia/inat_comp/tree/master/2017


0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
Relative bbox size
0.00
0.02
0.04
0.06
0.08
Frequency
Mammalia
Aves
Mollusca
Insecta
Arachnida
Reptilia
Amphibia
Animalia
Actinopterygii
Figure 6.4: The distribution of relative bounding box sizes (calculated by√
wbbox×hbbox/

√

wimg×himg) in the training set, per super-class. Most objects
are relatively small or medium sized.

**Bounding Box Annotation**
Bounding boxes were collected on 9 out of the 13 super-classes (see Table 6.2),
totaling 2,854 classes. Due to the inherent difficultly of asking non-expert crowd
annotators to both recognize and box specific fine-grained classes, we instructed
annotators to instead box all instances of the associated super-class for a taxon
(e.g. “Box all Birds” rather than “Box all Red-winged Black Birds”). We col-
lected super-class boxes only on taxa that are part of that super-class. For some
super-classes (e.g. Mollusca), there are images containing taxa which are unfamiliar
to many of the annotators (e.g. Fig. 6.3(a)). For those cases, we instructed the
annotators to box the prominent objects in the images.

The task instructions specified to draw boxes tightly around all parts of the animal
(including legs, horns, antennas, _etc._ ). If the animal is occluded, the annotators were
instructed to draw the box around the visible parts (e.g. Fig. 6.3(b)). In cases where
the animal is blurry or small (e.g. Fig. 6.3(c) and (d)), the following rule-of-thumb
was used: “if you are confident that it is an animal from the requested super-class,
regardless of size, blurriness or occlusion, put a box around it.” For images with
multiple instances of the super-class, all of them are boxed, up to a limit of 10
(Fig. 6.3(f)), and bounding boxes may overlap (Fig. 6.3(e)). We observe that12%
of images have more than 1 instance and 1 .3%have more than 5. If the instances are
physically connected (e.g. the mussels in Fig. 6.3(g)), then only one box is placed
around them.


Bounding boxes were not collected on the Plantae, Fungi, Protozoa, or Chromista
super-classes because these super-classes exhibit properties that make it difficult
to box the individual instances (e.g. close up of trees, bushes, kelp, _etc._ ). An
alternate form of pixel annotations, potentially from a more specialized group of
crowd workers, may be more appropriate for these classes.

Under the above guidelines, 561,767 bounding boxes were obtained from 449,313
images in the training and validation sets. Following the size conventions of
COCO T.-Y. Lin et al., 2014, the iNat2017 dataset is composed of 5 .7%small
instances (area< 322 ), 23 .6%medium instances ( 322 ≤area≤ 962 ), and 70 .7%
large instances (area> 962 ), with area computed as50%of the annotated bounding
box area (since segmentation masks were not collected). Figure 6.4 shows the dis-
tribution of relative bounding box sizes, indicating that a majority of instances are
relatively small and medium sized.

### 6.5 Experiments

In this section, we compare the performance of state-of-the-art classification and
detection models on iNat2017.

**Classification Results**
To characterize the classification difficulty of iNat2017, we ran experiments with sev-
eral state-of-the-art deep network architectures, including ResNets (He et al., 2016),
Inception V3 (Szegedy, Vanhoucke, et al., 2016), Inception ResNet V2 (Szegedy,
Ioffe, et al., 2016), and MobileNet (Howard et al., 2017). During training, random
cropping with aspect ratio augmentation (Szegedy, W. Liu, et al., 2015) was used.
Training batches of size 32 were created by uniformly sampling from all available
training images as opposed to sampling uniformly from the classes. We fine-tuned
all networks from ImageNet pre-trained weights with a learning rate of 0.0045,
decayed exponentially by 0.94 every 4 epochs, and RMSProp optimization with
momentum and decay both set to 0.9. Training and testing were performed with an
image size of 299 × 299 , with a single centered crop at test time.

Table 6.3 summarizes the top-1 and top-5 accuracy of the models. From the
Inception family, we see that the higher capacity Inception ResNet V2 outperforms
the Inception V3 network. The addition of the Squeeze-and-Excitation (SE) blocks
(Hu, Shen, and G. Sun, 2017) further improves performance for both models by a
small amount. ResNets performed worse on iNat2017 compared to the Inception
architectures, likely due to over-fitting on categories with a small number of training


bf Validation Public Test Private Test
Top1 Top5 Top1 Top5 Top1 Top5
IncResNetV2 SE 67.3 87.5 68.5 88.2 67.7 87.9
IncResNetV2 67.1 87.5 68.3 88.0 67.8 87.8
IncV3 SE 65.0 85.9 66.3 86.7 65.2 86.3
IncV3 64.2 85.2 65.5 86.1 64.8 85.7
ResNet152 drp 62.6 84.5 64.2 85.5 63.1 85.1
ResNet101 drp 60.9 83.1 62.4 84.1 61.4 83.6
ResNet152 59.0 80.5 60.6 81.7 59.7 81.3
ResNet101 58.4 80.0 59.9 81.2 59.1 80.9
MobileNet V1 52.9 75.4 54.4 76.8 53.7 76.3
Table 6.3: Classification results for various CNNs trained on only the training
set, using a single center crop at test time. Unlike some current datasets where
performance is near saturation, iNat2017 still poses a challenge for state-of-the-art
classifiers.

images. We found that adding a 0.5 probability dropout layer (drp) could improve
the performance of ResNets. MobileNet, designed to efficiently run on embedded
devices, had the lowest performance.

Overall, the Inception ResNetV2 SE was the best performing model. As a com-
parison, this model achieves a single crop top-1 and top-5 accuracy of 80.2% and
95.21% respectively on the ILSVRC 2012 (Russakovsky et al., 2015) validation
set (Szegedy, Ioffe, et al., 2016), as opposed to 67.74% and 87.89% on iNat2017,
highlighting the comparative difficulty of the iNat2017 dataset. A more detailed
super-class level breakdown is available in Table 6.4 for the Inception ResNetV2
SE model. We can see that the Reptilia super-class (with 289 classes) was the most
difficult with an average top-1 accuracy of 45.87%, while the Protozoa super-class
(with 4 classes) had the highest accuracy at 89.19%. Viewed as a collection of
fine-grained datasets (one for each super-class), we can see that the iNat2017 dataset
exhibits highly variable classification difficulty.

In Figure 6.5, we plot the top one public test set accuracy against the number of
training images for each class from the Inception ResNet V2 SE model. We see
that as the number of training images per class increases, so does the test accuracy.
However, we still observe a large variance in accuracy for classes with a similar
amount of training data, revealing opportunities for algorithmic improvements in
both the low data and high data regimes.


Super-Class Avg Train Public Test
Top1 Top5
Plantae 75.4 69.5 87.1
Insecta 98.4 77.1 93.4
Aves 222.3 67.3 88.0
Reptilia 121.8 45.9 80.9
Mammalia 157.7 61.4 85.1
Fungi 48.1 74.0 92.3
Amphibia 67.9 51.2 81.0
Mollusca 81.0 72.4 90.9
Animalia 67.9 73.8 91.1
Arachnida 87.0 71.5 88.8
Actinopterygii 37.4 70.8 86.3
Chromista 44.2 73.8 92.4
Protozoa 77.0 89.2 96.0
Table 6.4: Super-class level accuracy (computed by averaging across all species
within each super-class) for the best performing model Inception ResNetV2 SE (Hu,
Shen, and G. Sun, 2017). “Avg Train” indicates the average number of training
images per class for each super-class. We observe a large difference in performance
across the different super-classes.

**Additional Classification Results**
We performed an experiment to understand if there was any relationship between
real world animal size and prediction accuracy. Using existing records for bird
(Lislevand, Figuerola, and Székely, 2007) and mammal (Jones et al., 2009) body
sizes, we assigned a mass to each of the classes in iNat2017 that overlapped with
these datasets. For a given species, mass will vary due to the life stage or gender
of the particular individual. Here, we simply take the average value. This resulted
in data for 795 species, from the small Allen’s hummingbird ( _Selasphorus sasin_ )
to the large Humpback whale ( _Megaptera novaeangliae_ ). In Figure 6.6, we can see
that median accuracy decreases as the mass of the species increases. These results
are preliminary, but reinforce the observation that it can be challenging for humans
to take good photographs of larger mammals. More analysis of these failure cases
may allow us to produce better, species-specific, instructions for the photographers
on iNaturalist.

The IUCN Red List of Vulnerable Species monitors and evaluates the extinction risk
of thousands of species and subspecies (Baillie, Hilton-Taylor, and Stuart, 2004). In
Figure 6.7, we plot the Red List status of 1,568 species from the iNat2017 dataset.
We see that the vast majority of the species are in the ‘Least Concern’ category and
that test accuracy decreases as the threatened status increases. This can perhaps be


4 465 2602 778 590 438 145 53 14
Binned Number of Training Images
5-
10
10-
20
20-
50
50-
100
100-
200
200-
500
500-
1K
1K-
2K
2K-
4K
Test Accuracy
0
20
40
60
80
100
Figure 6.5: Top one public test set accuracy per class for IncResNet V2 SE (Hu,
Shen, and G. Sun, 2017). Each box plot represents classes grouped by the number
of training images. The number of classes for each bin is written on top of each box
plot. Performance improves with the number of training images, but the challenge
is how to maintain high accuracy with fewer images.

Binned Mass (KG)
Test Accuracy
0
20
40
60
80
100
339
259 167
30
0.0-
0.1
0.1-
1.0
1.0-
100
100-
40K
Figure 6.6: Top one public test set accuracy per class for (Szegedy, Ioffe, et al.,
2016) for a subset of 795 classes of birds and mammals binned according to mass.
The number of classes appears to the bottom right of each box.

explained by the reduced number of images for these species in the dataset.

Finally, in Figure 6.8 we examine the relationship between the number of images
and the validation accuracy. The median number of training images per class for
our entire training set is 41. For this experiment, we capped the maximum number
of training images per class to 10, 20, 50, or all, and trained a separate Inception V3


Test Accuracy
0
20
40
60
80
100
IUCN Red List Status
1433
65
36
31
3
Least
Concern
Near
Threatened Vulnerable Endangered
Critically
Endangered
Figure 6.7: Top one public test set accuracy for (Szegedy, Ioffe, et al., 2016) for
a subset of 1,568 species binned according to their IUCN Red List of Threatened
Species status (Baillie, Hilton-Taylor, and Stuart, 2004). The number of classes
appears to the bottom right of each box.

for each case. This corresponds to starting with 50,000 for the case of 10 images
per class and then doubling the total amount of training data each time. For each
species, we randomly selected the images up until the maximum amount. As noted
in the main paper, more attention is needed to improve performance in the low data
regime.

105 106
Number of train images
30
40
50
60
70
80
90
Validation accuracy
10 ims per class
20 ims per class
50 ims per class
all train
top5
top1
Figure 6.8: As the maximum number of training images per class increases, so does
the accuracy. However, we observe diminishing returns as the number of images
increases. Results are plotted on the validation set for the Inception V3 network
(Szegedy, Vanhoucke, et al., 2016).


**iNat2017 Competition Results**
From April to mid July 2017, we ran a public challenge on the machine learning
competition platform Kaggle 3 using iNat2017. Similar to the classification tasks in
(Russakovsky et al., 2015), we used the top five accuracy metric to rank competitors.
We used this metric as some species can only be disambiguated with additional data
provided by the observer, such as location or date. Additionally, in a small number
of cases, multiple species may appear in the same image (e.g. a bee on a flower).
Overall, there were 32 submissions and we display the final results for the top five
teams in Table 6.5.

The top performing entry from _GMV_ consisted of an ensemble of Inception V4 and
Inception ResNet V2 networks Szegedy, Ioffe, et al., 2016. Each model was first
initialized on the ImageNet-1K dataset and then finetuned with the iNat2017 training
set along with 90% of the validation set, utilizing data augmentation at training time.
The remaining 10% of the validation set was used for evaluation. To compensate for
the imbalanced training data, the models were further fine-tuned on the 90% subset
of the validation data that has a more balanced distribution. To address small object
size in the dataset, inference was performed on 560 × 560 resolution images using
twelve crops per image at test time.

The additional training data amounts to 15% of the original training set, which,
along with the ensembling, multiple test crops, and higher resolution, account for
the improved 81.58% top 1 public accuracy compared to our best performing single
model which achieved 68.53%.

Rank Team name Public Test Private Test
Top1 Top5 Top1 Top5
1 GMV 81.58 95.19 81.28 95.13
2 Terry 77.18 93.60 76.76 93.50
3 Not hotdog 77.04 93.13 76.56 93.01
4 UncleCat 77.64 93.06 77.44 92.97
5 DLUT_VLG 76.75 93.04 76.19 92.96
Table 6.5: Final public challenge leaderboard results. ‘Rank’ indicates the final
position of the team out of 32 competitors. These results are typically ensemble
models, trained with higher input resolution, with the validation set as additional
training data.

(^3) [http://www.kaggle.com/c/inaturalist-challenge-at-fgvc-2017](http://www.kaggle.com/c/inaturalist-challenge-at-fgvc-2017)


**Detection Results**
To characterize the detection difficulty of iNat2017, we adopt Faster-RCNN (Ren
et al., 2017) for its state-of-the-art performance as an object detection setup (which
jointly predicts object bounding boxes along with class labels). We use a Ten-
sorFlow (Abadi et al., 2016) implementation of Faster-RCNN with default hyper-
parameters (J. Huang et al., 2017). Each model is trained with 0.9 momentum and
asynchronously optimized on 9 GPUs to expedite experiments. We use an Incep-
tion V3 network, initialized from ImageNet, as the backbone for our Faster-RCNN
models. Finally, each input image is resized to have 600 pixels as the short edge
while maintaining the aspect ratio.

As discussed in Section 6.4, we collected bounding boxes on 9 of the 13 super-
classes, translating to a total of 2,854 classes with bounding boxes. In the following
experiments, we only consider performance on this subset of classes. Additionally,
we report performance on the validation set in place of the test set, and we only
evaluate on images that contained a single instance. Images that contained only
evidence of the species’ presence and images that contained multiple instances were
excluded. We evaluate the models using the detection metrics from COCO (T.-Y.
Lin et al., 2014).

We first study the performance of fine-grained localization and classification by
training the Faster-RCNN model on the 2,854 class subset. Figure 6.10 shows some
sample detection results. Table 6.6 provides the break down in performance for
each super-class, where super-class performance is computed by taking an average
across all classes within the super-class. The precision-recall curves (again at the
super-class level) for 0.5 IoU are displayed in Figure 6.9. Across all super-classes
we achieve a comprehensive average precision (AP) of 43.5. Again the Reptilia
super-class proved to be the most difficult, with an AP of 21.3 and an AUC of 0.315.
At the other end of the spectrum, we achieved an AP of 49.4 for Insecta and an
AUC of 0.677. Similar to the classification results, when viewed as a a collection
of datasets (one for each super-class), we see that iNat2017 exhibits highly variable
detection difficulty, posing a challenge to researchers to build improved detectors
that work across a broad group of fine-grained classes.

Next we explored the effect of label granularity on detection performance. We
trained two more Faster-RCNN models, one trained to detect super classes rather
fine-grained classes (so 9 classes in total), and another model trained with all
labels pooled together, resulting in a generic object / not object detector. Table 6.7


shows the resulting AP scores for the three models when evaluated at different
granularities. When evaluated on the coarser granularity, detectors trained on finer-
grained categories have lower detection performance when compared with detectors
trained at coarser labels. The performance of the 2,854-class detector is particularly
poor on super-class recognition and object localization. This suggests that the Faster-
RCNN algorithm has plenty of room for improvements on end-to-end fine-grained
detection tasks.

0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
Recall
0.0
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1.0
Precision
Insecta (0.677)
Aves (0.670)
Arachnida (0.664)
Animalia (0.557)
Actinopterygii (0.521)
Mollusca (0.500)
Mammalia (0.486)
Amphibia (0.402)
Reptilia (0.315)
Figure 6.9: Precision-Recall curve with 0.5 IoU for each super-class, where the Area-
Under-Curve (AUC) corresponds to AP^50 in Table 6.6. Super-class performance is
calculated by averaging across all fine-grained classes. We can see that building a
detector that works well for all super-classes in iNat2017 will be a challenge.

**Additional Detection Results**
In Table 6.8, we investigate detector performance for the 2,854-class model across
different bounding box sizes using the size conventions of the COCO dataset (T.-Y.
Lin et al., 2014). As expected, performance is directly correlated with size, where


AP AP^50 AP^75 AR^1 AR^10
Insecta 49.4 67.7 59.3 64.5 64.9
Aves 49.5 67.0 59.1 63.3 63.6
Reptilia 21.3 31.5 24.9 44.0 44.8
Mammalia 33.3 48.6 39.1 49.8 50.6
Amphibia 28.7 40.2 35.0 52.0 52.3
Mollusca 34.8 50.0 41.6 52.0 53.0
Animalia 35.6 55.7 40.8 48.3 50.5
Arachnida 43.9 66.4 49.6 57.3 58.6
Actinopterygii 35.0 52.1 41.6 49.1 49.6
Overall 43.5 60.2 51.8 59.3 59.8
Table 6.6: Super-class-level Average Precision (AP) and Average Recall (AR)
for object detection, where AP, AP^50 and AP^75 denotes AP@[IoU=.50:.05:.95],
AP@[IoU=.50] and AP@[IoU=.75] respectively; AR^1 and AR^10 denotes AR given
1 detection and 10 detections per image.

Training Evaluation
2854-class 9-super-class 1-generic
2854-class 43.5 55.6 63.7
9-super-class - 65.8 76.7
1-generic - - 78.5
Table 6.7: Detection performance (AP@[IoU=.50:.05:.95]) with different training
and evaluation class granularity. Using finer-grained class labels during training
has a negative impact on coarser-grained super-class detection. This presents an
opportunity for new detection algorithms that maintain precision at the fine-grained
level.

smaller objects are more difficult to detect. However, examining Table 6.9, we can
see that total number of these small instances is low for most super-classes.

### 6.6 Conclusions and Future Work

We present the iNat2017 dataset, in contrast to many existing computer vision
datasets it is: (1) unbiased, in that it was collected by non-computer vision re-
searchers for a well defined purpose; (2) more representative of real-world chal-
lenges than previous datasets; (3) represents a long-tail classification problem; and
(4) is useful in conservation and field biology. The introduction of iNat2017 en-
ables us to study two important questions in a real world setting: (1) do long-tailed
datasets present intrinsic challenges; and (2) do our computer vision systems ex-
hibit transfer learning from the well-represented categories to the least represented
ones. While our baseline classification and detection results are encouraging, from


Chaetodon lunula(1.00)
Chaetodon lunula(0.98)
Anaxyrus fowleri(0.95) Pseudacris regilla(0.58)
Setophaga petechia(0.91)
Orcinus orca(0.99) Rabdotus dealbatus(0.92) Sylvilagus audubonii(0.97) Equus quagga(0.98)Equus quagga(1.00)
Megaptera novaeangliae(0.74) Zalophus californianus(0.88)
Hippodamia convergens(0.83) Phalacrocorax auritus(0.54)
Figure 6.10: Sample detection results for the 2,854-class model that was evaluated
across all validation images. Green boxes represent correct species level detections,
while reds are mistakes. The bottom row depicts some failure cases. We see that
small objects pose a challenge for classification, even when localized well.

our experiments we see that state-of-the-art computer vision models have room to
improve when applied to large imbalanced datasets. Small, efficient models de-
signed for mobile applications and embedded devices have even more room for
improvement (Howard et al., 2017).

Unlike traditional, researcher-collected datasets, the iNat2017 dataset has the oppor-
tunity to grow with the iNaturalist community. Currently, every 1.7 hours another
species passes the 20 unique observer threshold, making it available for inclusion
in the dataset (already up to 12k as of November 2017, up from 5k when we started
work on the dataset). Thus, the current challenges of the dataset (long tail with
sparse data) will only become more relevant.

In the future we plan to investigate additional annotations such as sex and life stage
attributes, habitat tags, and pixel level labels for the four super-classes that were
challenging to annotate. We also plan to explore the “open-world problem” where
the test set contains classes that were never seen during training. This direction
would encourage new error measures that incorporate taxonomic rank (Mittal et al.,
2012; Yan et al., 2015). Finally, we expect this dataset to be useful in studying
how to teach fine-grained visual categories to humans (Singla et al., 2014; Johns,


Actino
Amphib
Animal
Arachn
Aves
Chromi
Fungi
Insect
Mammal
Mollus
Planta
Protoz
Reptil
Figure 6.11: Example images from the training set. Each row displays randomly
selected images from each of the 13 different super-classes. For ease of visualization
we show the center crop of each image.


APS APM APL ARS ARM ARL
Insecta 13.4 34.7 51.8 13.5 38.9 67.7
Aves 11.5 41.7 55.1 13.3 49.2 69.9
Reptilia 0.0 12.4 22.0 0.0 16.3 46.5
Mammalia 6.7 27.8 37.1 9.0 36.1 55.8
Amphibia 0.0 23.2 29.9 0.0 28.7 54.9
Mollusca 17.5 30.8 35.8 17.5 33.6 55.9
Animalia 24.0 22.7 37.1 26.7 28.2 52.0
Arachnida 16.2 32.9 46.5 16.2 38.5 61.6
Actinopterygii 5.0 16.3 36.1 5.0 17.9 51.1
Overall 11.0 34.7 46.7 12.5 40.7 63.7
Table 6.8: Super-class level Average Precision (AP) and Average Recall (AR) with
respect to object sizes. S, M and, L denote small (area< 322 ), medium ( 322 ≤area
≤ 962 ) and, large (area> 962 ) objects. The AP for each super-class is calculated by
averaging the results for all species belonging to it. Best and worst performance for
each metric are marked by green and red, respectively.

Small Medium Large
Insecta 445 2432 16429
Aves 2375 8898 16239
Reptilia 32 400 5426
Mammalia 280 1068 2751
Amphibia 20 253 2172
Mollusca 74 466 1709
Animalia 72 414 1404
Arachnida 12 152 909
Actinopterygii 32 144 634
Table 6.9: The number of super-class instances at each bounding box size in the
validation set. While AP and AR is low for some super-classes at a particular size
(see Table 6.8), the actual number of instances at that size may also be low.

Mac Aodha, and Brostow, 2015), and plan to experiment with models of human
learning.

**Acknowledgments.** This work was supported by a Google Focused Research Award.
We would like to thank: Scott Loarie and Ken-ichi Ueda from iNaturalist; Steve
Branson, David Rolnick, Weijun Wang, and Nathan Frey for their help with the
dataset; Wendy Kan and Maggie Demkin from Kaggle; the iNat2017 competitors,
and the FGVC2017 workshop organizers. We also thank NVIDIA and Amazon Web
Services for their donations.


**References**

Abadi, Martin et al. (2016). “Tensorflow: Large-scale machine learning on hetero-
geneous distributed systems”. In: _arXiv preprint arXiv:1603.04467_.

Baillie, Jonathan, Craig Hilton-Taylor, and Simon N Stuart (2004). _2004 IUCN red
list of threatened species: a global species assessment_. IUCN.

Berg, Thomas et al. (2014). “Birdsnap: Large-scale fine-grained visual categoriza-
tion of birds”. In: _Computer Vision and Pattern Recognition (CVPR), 2014 IEEE
Conference on_. IEEE, pp. 2019–2026.

Cao, Qiong et al. (2017). “VGGFace2: A dataset for recognising faces across pose
and age”. In: _arXiv preprint arXiv:1710.08092_.

Cardinale, Bradley J et al. (2012). “Biodiversity loss and its impact on humanity”.
In: _Nature_.

Everingham, Mark et al. (2010). “The pascal visual object classes (voc) challenge”.
In: _IJCV_.

Fei-Fei, Li, Rob Fergus, and Pietro Perona (2007). “Learning generative visual
models from few training examples: An incremental Bayesian approach tested on
101 object categories”. In: _CVIU_.

Gebru, Timnit et al. (2017). “Fine-grained car detection for visual census estima-
tion”. In: _AAAI_.

Guo, Yandong et al. (2016). “Ms-celeb-1m: A dataset and benchmark for large-scale
face recognition”. In: _ECCV_.

He, Kaiming et al. (2016). “Deep residual learning for image recognition”. In: _Pro-
ceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ ,
pp. 770–778.

Hou, Saihui, Yushan Feng, and Zilei Wang (2017). “VegFru: A Domain-Specific
Dataset for Fine-grained Visual Categorization”. In: _ICCV_.

Howard, Andrew G et al. (2017). “Mobilenets: Efficient convolutional neural net-
works for mobile vision applications”. In: _arXiv preprint arXiv:1704.04861_.

Hu, Jie, Li Shen, and Gang Sun (2017). “Squeeze-and-Excitation Networks”. In:
_arXiv preprint arXiv:1709.01507_.

Huang, Gary B. et al. (2007). _Labeled Faces in the Wild: A Database for Study-
ing Face Recognition in Unconstrained Environments_. Tech. rep. University of
Massachusetts, Amherst.

Huang, Jonathan et al. (2017). “Speed/accuracy trade-offs for modern convolutional
object detectors”. In: _CVPR_.

Johns, Edward, Oisin Mac Aodha, and Gabriel J Brostow (2015). “Becoming the
expert-interactive multi-class machine teaching”. In: _CVPR_.


Jones, Kate E et al. (2009). “PanTHERIA: a species-level database of life history,
ecology, and geography of extant and recently extinct mammals”. In: _Ecology_.

Khosla, Aditya et al. (2011). “Novel Dataset for Fine-Grained Image Categoriza-
tion”. In: _First Workshop on Fine-Grained Visual Categorization, IEEE Confer-
ence on Computer Vision and Pattern Recognition_. Colorado Springs, CO.

Krasin, I et al. (2016). “OpenImages: A public dataset for large-scale multi-label
and multiclass image classification”. In: _Dataset available from https://github.
com/openimages_.

Krause, Jonathan, Benjamin Sapp, et al. (2016). “The unreasonable effectiveness of
noisy data for fine-grained recognition”. In: _European Conference on Computer
Vision_. Springer, pp. 301–320.

Krause, Jonathan, Michael Stark, et al. (2013). “3d object representations for fine-
grained categorization”. In: _Proceedings of the IEEE International Conference
on Computer Vision Workshops_ , pp. 554–561.

Kumar, Neeraj et al. (2012). “Leafsnap: A computer vision system for automatic plant
species identification”. In: _Computer Vision–ECCV 2012_. Springer, pp. 502–516.

Lin, Tsung-Yi et al. (2014). “Microsoft COCO: Common objects in context”. In:
_ECCV_.

Lin, Yen-Liang et al. (2014). “Jointly optimizing 3d model fitting and fine-grained
classification”. In: _Computer Vision–ECCV 2014_. Springer, pp. 466–480.

Lislevand, Terje, Jordi Figuerola, and Tamás Székely (2007). “Avian body sizes in
relation to fecundity, mating system, display behavior, and resource sharing”. In:
_Ecology_.

Liu, Jiongxin et al. (2012). “Dog breed classification using part localization”. In:
_Computer Vision–ECCV 2012_. Springer, pp. 172–185.

Maji, Subhransu et al. (2013). “Fine-grained visual classification of aircraft”. In:
_arXiv preprint arXiv:1306.5151_.

Mittal, Arpit et al. (2012). “Taxonomic multi-class prediction and person layout
using efficient structured ranking”. In: _ECCV_.

Mora, Camilo et al. (2011). “How many species are there on Earth and in the ocean?”
In: _PLoS biology_ 9.8, e1001127.

Nilsback, Maria-Elena and Andrew Zisserman (2006). “A visual vocabulary for
flower classification”. In: _Computer Vision and Pattern Recognition, 2006 IEEE
Computer Society Conference on_. Vol. 2. IEEE, pp. 1447–1454.

Parkhi, O. M. et al. (2012). “Cats and Dogs”. In: _CVPR_.

Parkhi, Omkar M, Andrea Vedaldi, Andrew Zisserman, et al. (2015). “Deep Face
Recognition.” In: _BMVC_.


Ren, Shaoqing et al. (2017). “Faster r-cnn: Towards real-time object detection with
region proposal networks”. In: _PAMI_.

Rolnick, David et al. (2017). “Deep Learning is Robust to Massive Label Noise”.
In: _arXiv preprint arXiv:1705.10694_.

Russakovsky, Olga et al. (2015). “Imagenet large scale visual recognition challenge”.
In: _International Journal of Computer Vision_ 115.3, pp. 211–252.

Schroff, Florian, Dmitry Kalenichenko, and James Philbin (2015). “Facenet: A
unified embedding for face recognition and clustering”. In: _CVPR_.

Singla, Adish et al. (2014). “Near-Optimally Teaching the Crowd to Classify.” In:
_ICML_.

Szegedy, Christian, Sergey Ioffe, et al. (2016). “Inception-v4, inception-resnet and
the impact of residual connections on learning”. In: _arXiv preprint arXiv:1602.07261_.

Szegedy, Christian, Wei Liu, et al. (2015). “Going deeper with convolutions”. In:
_CVPR_.

Szegedy, Christian, Vincent Vanhoucke, et al. (2016). “Rethinking the inception
architecture for computer vision”. In: _Proceedings of the IEEE Conference on
Computer Vision and Pattern Recognition_ , pp. 2818–2826.

Taigman, Yaniv et al. (2014). “Deepface: Closing the gap to human-level perfor-
mance in face verification”. In: _Computer Vision and Pattern Recognition (CVPR),
2014 IEEE Conference on_. IEEE, pp. 1701–1708.

Ueda, K (2017). “iNaturalist Research-grade Observations via GBIF.org.” In:url:
https://doi.org/10.15468/ab3s5x.

Van Horn, Grant et al. (2015). “Building a bird recognition app and large scale
dataset with citizen scientists: The fine print in fine-grained dataset collection”.
In: _Proceedings of the IEEE Conference on Computer Vision and Pattern Recog-
nition_ , pp. 595–604.doi:10.1109/CVPR.2015.7298658.

Vedaldi, Andrea et al. (2014). “Understanding objects in detail with fine-grained
attributes”. In: _Proceedings of the IEEE Conference on Computer Vision and
Pattern Recognition_ , pp. 3622–3629.

Wah, Catherine et al. (2011). “The caltech-ucsd birds-200-2011 dataset”. In:

Wegner, Jan D et al. (2016). “Cataloging public objects using aerial and street-level
images-urban trees”. In: _Proceedings of the IEEE Conference on Computer Vision
and Pattern Recognition_ , pp. 6014–6023.

Welinder, Peter et al. (2010). “Caltech-UCSD birds 200”. In:

Wilber, Michael J et al. (2017). “BAM! The Behance Artistic Media Dataset for
Recognition Beyond Photography”. In: _ICCV_.

Xie, Saining et al. (2017). “Aggregated residual transformations for deep neural
networks”. In: _CVPR_.


Yan, Zhicheng et al. (2015). “HD-CNN: hierarchical deep convolutional neural
networks for large scale visual recognition”. In: _ICCV_.

Yang, Linjie et al. (2015). “A large-scale car dataset for fine-grained categorization
and verification”. In: _CVPR_.

Yu, Aron and Kristen Grauman (2014). “Fine-grained visual comparisons with local
learning”. In: _CVPR_.

Zhang, Xiao et al. (2017). “The iMaterialist Challenge 2017 Dataset”. In: _FGVC
Workshop at CVPR_.


