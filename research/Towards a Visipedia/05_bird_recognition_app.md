# Chapter 5: Building A Bird Recognition App And Large Scale Dataset With Citizen Scientists: The Fine Print In Fine-Grained Dataset Collection
Van Horn, Grant et al. (2015). “Building a bird recognition app and large scale
dataset with citizen scientists: The fine print in fine-grained dataset collection”.
In: _Proceedings of the IEEE Conference on Computer Vision and Pattern Recog-
nition_ , pp. 595–604.doi:10.1109/CVPR.2015.7298658.

### 5.1 Abstract

We introduce tools and methodologies to collect high quality, large scale, fine-
grained, computer vision datasets using citizen scientists – crowd annotators who
are passionate and knowledgeable about specific domains such as birds or airplanes.
We worked with citizen scientists and domain experts to collect NABirds, a new
high quality dataset containing 48,562 images of North American birds with 555
categories, part annotations and bounding boxes. We find that citizen scientists
are significantly more accurate than Mechanical Turkers at zero cost. We worked
with bird experts to measure the quality of popular datasets like CUB-200-2011 and
ImageNet and found class label error rates of at least4%. Nevertheless, we found
that learning algorithms are surprisingly robust to annotation errors, and this level
of training data corruption can lead to an acceptably small increase in test error
if the training set has sufficient size. At the same time, we found that an expert-
curated high quality test set like NABirds is necessary to accurately measure the
performance of fine-grained computer vision systems. We used NABirds to train a
publicly available bird recognition service deployed on the web site of the Cornell
Lab of Ornithology. 1

### 5.2 Introduction

Computer vision systems – catalyzed by the availability of new, larger scale datasets
like ImageNet (Jia Deng, Dong, et al., 2009) – have recently obtained remarkable
performance at object recognition (Krizhevsky, Sutskever, and Hinton, 2012; Taig-
man et al., 2014) and detection (Girshick et al., 2013). Computer vision has entered

(^1) merlin.allaboutbirds.org


an era of big data, where the ability to collect larger datasets – larger in terms of
the number of classes, the number of images per class, and the level of annotation
per image – appears to be paramount for continuing performance improvement and
expanding the set of solvable applications.

Unfortunately, expanding datasets in this fashion introduces new challenges beyond
just increasing the amount of human labor required. As we increase the number
of classes of interest, classes become more fine-grained and difficult to distinguish
for the average person (and the average annotator), more ambiguous, and less likely
to obey an assumption of mutual exclusion. The annotation process becomes more
challenging, requiring an increasing amount of skill and knowledge. Dataset _quality_
appears to be at direct odds with dataset _size_.

In this chapter, we introduce tools and methodologies for constructing large, high
quality computer vision datasets, based on tapping into an alternate pool of crowd
annotators – citizen scientists. Citizen scientists are nonprofessional scientists or
enthusiasts in a particular domain such as birds, insects, plants, airplanes, shoes, or
architecture. Citizen scientists contribute annotations with the understanding that
their expertise and passion in a domain of interest can help build tools that will be
of service to a community of peers. Unlike workers on Mechanical Turk, citizen
scientists are unpaid. Despite this, they produce higher quality annotations due to
their greater expertise and the absence of spammers. Additionally, citizen scientists
can help define and organically grow the set of classes and its taxonomic structure
to match the interests of real users in a domain of interest. Whereas datasets like
ImageNet (J. Deng et al., 2009) and CUB-200-2011 (Wah et al., 2011) have been
valuable in fostering the development of computer vision algorithms, the particular
set of categories chosen is somewhat arbitrary and of limited use to real applications.

Figure 5.1: **Merlin Photo ID** : a publicly available tool for bird species classification
built with the help of citizen scientists. The user uploaded a picture of a bird, and
server-side computer vision algorithms identified it as an immature Cooper’s Hawk.


The drawback of using citizen scientists instead of Mechanical Turkers is that the
throughput of collecting annotations may be lower, and computer vision researchers
must take the time to figure out how to partner with different communities for each
domain.

We collected a large dataset of 48 , 562 images over 555 categories of birds with
part annotations and bounding boxes for each image, using a combination of citizen
scientists, experts, and Mechanical Turkers. We used this dataset to build a publicly
available application for bird species classification. In this chapter, we provide
details and analysis of our experiences with the hope that they will be useful and
informative for other researchers in computer vision working on collecting larger
fine-grained image datasets. We address questions like: what is the relative skill
level of different types of annotators (MTurkers, citizen scientists, and experts)
for different types of annotations (fine-grained categories and parts)? What are
the resulting implications in terms of annotation quality, annotation cost, human
annotator time, and the time it takes a requester to finish a dataset? Which types of
annotations are suitable for different pools of annotators? What types of annotation
GUIs are best for each respective pool of annotators? How important is annotation
quality for the accuracy of learned computer vision algorithms? How significant are
the quality issues in existing datasets like CUB-200-2011 and ImageNet, and what
impact has that had on computer vision performance?

We summarize our contributions below:

1. Methodologies to collect high quality, fine-grained computer vision datasets
    using a new type of crowd annotators: citizen scientists.
2. NABirds: a large, high quality dataset of 555 categories curated by experts.
3. Merlin Photo ID: a publicly available tool for bird species classification.
4. Detailed analysis of annotation quality, time, cost, and throughput of MTurk-
    ers, citizen scientists, and experts for fine-grained category and part annota-
    tions.
5. Analysis of the annotation quality of the popular datasets CUB-200 and Ima-
    geNet.
6. Empirical analysis of the effect that annotation quality has when training
    state-of-the-art computer vision algorithms for categorization.


A high-level summary of our findings is: (a) citizen scientists have 2-4 times
lower error rates than MTurkers at fine-grained bird annotation, while annotating
images faster and at zero cost. Over 500 citizen scientists annotated images in
our dataset – if we can expand beyond the domain of birds, the pool of possible
citizen scientist annotators is massive. (b) A curation-based interface for visualizing
and manipulating the full dataset can further improve the speed and accuracy of
citizen scientists and experts. (c) Even when averaging answers from 10 MTurkers
together, MTurkers have a more than30%error-rate at 37-way bird classification.
(d) The general high quality of Flickr search results (84%accurate when searching
for a particular species) greatly mitigates the errors of MTurkers when collecting
fine-grained datasets. (e) MTurkers are as accurate and fast as citizen scientists
at collecting part location annotations. (f) MTurkers have faster throughput in
collecting annotations than citizen scientists; however, using citizen scientists it is
still realistic to annotate a dataset of around 100k images in a domain like birds in
around 1 week. (g) At least4%of images in CUB-200-2011 and ImageNet have
incorrect class labels, and numerous other issues including inconsistencies in the
taxonomic structure, biases in terms of which images were selected, and the presence
of duplicate images. (h) Despite these problems, these datasets are still effective for
computer vision research; when training CNN-based computer vision algorithms
with corrupted labels, the resulting increase in test error is surprisingly low and
significantly less than the level of corruption. (i) A consequence of findings (a),
(d), and (h) is that training computer vision algorithms on unfiltered Flickr search
results (with no annotation) can often outperform algorithms trained when filtering
by MTurker majority vote.

### 5.3 Related Work

**Crowdsourcing with Mechanical Turk**
Amazon’s Mechanical Turk (AMT) has been an invaluable tool that has allowed
researchers to collect datasets of significantly larger size and scope than previously
possible (Sorokin and Forsyth, 2008; J. Deng et al., 2009; Lin et al., 2014). AMT
makes it easy to outsource simple annotation tasks to a large pool of workers. Al-
though these workers will usually be non-experts, for many tasks it has been shown
that repeated labeling of examples by multiple non-expert workers can produce high
quality labels (Sheng, Provost, and Ipeirotis, 2008; Welinder and Pietro Perona,
2010; Ipeirotis, Provost, et al., 2013). Annotation of fine-grained categories is a
possible counter-example, where the average annotator may have little to no prior


knowledge to make a reasonable guess at fine-grained labels. For example, the aver-
age worker has little to no prior knowledge as to what type of bird a "Semipalmated
Plover" is, and her ability to provide a useful guess is largely dependent on the efforts
of the dataset collector to provide useful instructions or illustrative examples. Since
our objective is to collect datasets of thousands of classes, generating high quality
instructions for each category is difficult or infeasible.

**Crowdsourcing with expertise estimation**
A possible solution is to try to automatically identify the subset of workers who
have adequate expertise for fine-grained classification (Welinder, Branson, et al.,
2010; Whitehill et al., 2009; Raykar et al., 2009; Long, Hua, and Kapoor, 2013).
Although such models are promising, it seems likely that the subset of Mechanical
Turkers with expertise in a particular fine-grained domain is small enough to make
such methods impractical or challenging.

**Games with a purpose**
Games with a purpose target alternate crowds of workers that are incentivized by
construction of annotation tasks that also provide some entertainment value. Notable
examples include the ESP Game (Von Ahn, 2006), reCAPTCHA (Von Ahn et al.,
2008), and BubbleBank (Jia Deng, Krause, and Fei-Fei, 2013). A partial inspiration
to our work was Quizz (Ipeirotis and Gabrilovich, 2014), a system to tap into new,
larger pools of unpaid annotators using Google AdWords to help find and recruit
workers with the applicable expertise. 2 A limitation of games with a purpose is that
they require some artistry to design tools that can engage users.

**Citizen science**
The success of Wikipedia is another major inspiration to our work, where citizen
scientists have collaborated to generate a large, high quality web-based encyclopedia.
Studies have shown that citizen scientists are incentivized by altruism, sense of
community, and reciprocity (Kuznetsov, 2006; Nov, 2007; Yang and Lai, 2010), and
such incentives can lead to higher quality work than monetary incentives (Gneezy
and Rustichini, 2000).

(^2) The viability of this approach remains to be seen, as our attempt to test it was foiled by a
misunderstanding with the AdWords team.


**Datasets**
Progress in object recognition has been accelerated by dataset construction. These
advances are fueled both by the release and availability of each dataset but also
by subsequent competitions on them. Key datasets/competitions in object recog-
nition include Caltech-101 (Fei-Fei, Fergus, and Pietro Perona, 2006), Caltech-
256 (Griffin, Holub, and P Perona, 2007), Pascal VOC (Everingham et al., 2010),
and ImageNet/ILSVRC (J. Deng et al., 2009; Russakovsky et al., 2014).

Fine-grained object recognition is no exception to this trend. Various domains have
already had datasets introduced including Birds (the CUB-200 (Wah et al., 2011) and
recently announced Birdsnap (T. Berg et al., 2014) datasets), Flowers (Nilsback and
Zisserman, 2008; Angelova and Zhu, 2013), Dogs and Cats (Khosla et al., 2011;
Parkhi et al., 2011; Liu et al., 2012), Stoneflies (Martinez-Munoz et al., 2009),
Butterflies (Lazebnik, Schmid, and Ponce, 2004), and Fish (Boom et al., 2014)
along with man-made domains such as Airplanes (Maji et al., 2013), Cars (Krause
et al., 2013), and Shoes (T. L. Berg, A. C. Berg, and Shih, 2010).

### 5.4 Crowdsourcing with Citizen Scientists

The communities of enthusiasts for a taxon are an untapped work force and partner
for vision researchers. The individuals comprising these communities tend to be
very knowledgeable about the taxon. Even those that are novices make up for
their lack of knowledge with passion and dedication. These characteristics make
these communities a fundamentally different work force than the typical paid crowd
workers. When building a large, fine-grained dataset, they can be utilized to curate
images with a level of accuracy that would be extremely costly with paid crowd
workers, see Section 5.6. There is a mutual benefit as the taxon communities gain
from having a direct influence on the construction of the dataset. They know their
taxon, and their community, better than vision researchers, and so they can ensure
that the resulting datasets are directed towards solving real world problems.

A connection must be established with these communities before they can be utilized.
We worked with ornithologists at the Cornell Lab of Ornithology to build NABirds.
The Lab of Ornithology provided a perfect conduit to tap into the large citizen
scientist community surrounding birds. Our partners at the Lab of Ornithology
described that the birding community, and perhaps many other taxon communities,
can be segmented into several different groups, each with their own particular
benefits. We built custom tools to take advantage of each of the segments.


**Experts**
Experts are the professionals of the community, and our partners at the Lab of
Ornithology served this role. Figure 5.4 is an example of an expert management
tool (Vibe 3 ) and was designed to let expert users quickly and efficiently curate
images and manipulate the taxonomy of a large dataset. Beyond simple image
storage, tagging, and sharing, the benefit of this tool is that it lets the experts define
the dataset taxonomy as they see fit, and allows for the dynamic changing of the
taxonomy as the need arises. For NABirds, an interesting result of this flexibility
is that bird species were further subdivided into “visual categories." A “visual
category" marks a sex or age or plumage attribute of the species that results in
a visually distinctive difference from other members within the same species, see
Figure 5.2. This type of knowledge of visual variances at the species level would
have been difficult to capture without the help of someone knowledgeable about the
domain.

**Citizen Scientist Experts**
After the experts, these individuals of the community are the top tier, most skilled
members. They have the confidence and experience to identify easily confused
classes of the taxonomy. For the birding community, these individuals were iden-
tified by their participation in eBird, a resource that allows birders to record and
analyze their bird sightings. 4 Figure 5.3a shows a tool that allows these members
to take bird quizzes. The tool presents the user with a series of images and requests
the species labels. The user can supply the label using the autocomplete box, or, if
they are not sure, they can browse through a gallery of possible answers. At the end
of the quiz, their answers can be compared with other expert answers.

(^3) vibe.visipedia.org
(^4) ebird.org
Figure 5.2: Two species of hawks from the NABirds dataset are separated into 6
categories based on their visual attributes.


(a) Quiz Annotation GUI (b) Part Annotation GUI
Figure 5.3: **(a)** This interface was used to collect category labels on images. Users
could either use the autocomplete box or scroll through a gallery of possible birds.
**(b)** This interface was used to collect part annotations on the images. Users were
asked to mark the visibility and location of 11 parts. See Section 5.4 and 5.4

Figure 5.4: Expert interface for rapid and efficient curation of images, and easy
modification of the taxonomy. The taxonomy is displayed on the left and is similar
to a file system structure. See Section 5.4.

**Citizen Scientist Turkers**
This is a large, passionate segment of the community motivated to help their cause.
This segment is not necessarily as skilled in difficult identification tasks, but they
are capable of assisting in other ways. Figure 5.3b shows a part annotation task that
we deployed to this segment. The task was to simply click on all parts of the bird.
The size of this population should not be underestimated. Depending on how these
communities are reached, this population could be larger than the audience reached
in typical crowdsourcing platforms.

### 5.5 NABirds

We used a combination of experts, citizen scientists, and MTurkers to build NABirds,
a new bird dataset of 555 categories with a total of 48 , 562 images. Members from
the birding community provided the images, the experts of the community curated
the images, and a combination of CTurkers and MTurkers annotated 11 bird parts on
every image along with bounding boxes. This dataset is free to use for the research
community.


The taxonomy for this dataset contains 1011 nodes, and the categories cover the most
common North American birds. These leaf categories were specifically chosen to al-
low for the creation of bird identification tools to help novice birders. Improvements
on classification or detection accuracy by vision researchers will have a straight-
forward and meaningful impact on the birding community and their identification
tools.

We used techniques from (Branson et al., 2014) to baseline performance on this
dataset. Using Caffe and the fc6 layer features extracted from the entire image, we
achieved an accuracy of 35 .7%. Using the best performing technique from (Branson
et al., 2014) with ground truth part locations, we achieved an accuracy of75%.

**5.6 Annotator Comparison**
In this section, we compare annotations performed by Amazon Mechanical Turk
workers (MTurkers) with citizen scientists reached through the Lab of Ornithol-
ogy’s Facebook page. The goal of these experiments was to quantify the followings
aspects of annotation tasks: (1) **Annotation Error** : The fraction of incorrect an-
notations; (2) **Annotation Time** : The average amount of human time required per
annotation; (3) **Annotation Cost** : The average cost in dollars required per annota-
tion; (4) **Annotation Throughput** : The average number of annotations obtainable
per second, this scales with the total size of the pool of annotators.

In order to compare the skill levels of different annotator groups directly, we chose
a common user interface that we considered to be appropriate for both citizen
scientists and MTurkers. For category labeling tasks, we used the quiz tool that was
discussed in Section 5.4. Each question presented the user with an image of a bird
and requested the species label. To make the task feasible for MTurkers, we allowed
users to browse through galleries of each possible species and limited the space of
possible answers to< 40 categories. Each quiz was focused on a particular group
of birds, either sparrows or shorebirds. Random chance was 1 / 37 for the sparrows
and 1 / 32 for the shorebirds. At the end of the quiz, users were given a score (the
number of correct answers) and could view their results. Figure 5.3a shows our
interface. We targeted the citizen scientist experts by posting the quizzes on the
eBird Facebook page.

Figure 5.5 shows the distribution of scores achieved by the two different worker
groups on the two different bird groups. Not surprisingly, citizen scientists had
better performance on the classification task than MTurkers; however we were


uncertain as to whether or not averaging a large number of MTukers could yield
comparable performance. Figure 5.6a plots the time taken to achieve a certain error
rate by combining multiple annotators for the same image using majority voting.
From this figure, we can see that citizen scientists not only have a lower median time
per image (about 8 seconds vs. 19 seconds), but that one citizen scientist expert
label is more accurate than the average of 10 MTurker labels. We note that we are
using a simple-as-possible (but commonly used) crowdsourcing method, and the
performance of MTurkers could likely be improved by more sophisticated techniques
such as CUBAM (Welinder, Branson, et al., 2010). However, the magnitude of
difference in the two groups and overall large error rate of MTurkers led us to
believe that the problem could not be solved simply using better crowdsourcing
models.

Figure 5.6c measures the raw throughput of the workers, highlighting the size of
the MTurk worker pool. With citizen scientists, we noticed a spike of participation
when the annotation task was first posted on Facebook, and then a quick tapering off
of participation. Finally, Figure 5.6b measures the cost associated with the different
levels of error; citizen scientists were unpaid.

We performed a similar analysis with part annotations. For this task, we used the
tool shown in Figure 5.3b. Workers from the two different groups were given an
image and asked to specify the visibility and position of 11 different bird parts. We
targeted the citizen scientist Turkers with this task by posting the tool on the Lab of
Ornithology’s Facebook page. The interface for the tool was kept the same between
the workers. Figures 5.7a, 5.7b, and 5.7c detail the results of this test. From
Figure 5.7a, we can see there is not a difference between the obtainable quality from
the two worker groups, and that MTurkers tended to be faster at the task. Figure 5.7c
again reveals that the raw throughput of Mturkers is larger than that of the citizen
scientists. The primary benefit of using citizen scientists for this particular case is
made clear by their zero cost in Figure 5.7b.

**Summary**
From these results, we can see that there are clear distinctions between the two
different worker pools. Citizen scientists are clearly more capable at labeling fine-
grained categories than MTurkers. However, the raw throughput of MTurk means
that you can finish annotating your dataset sooner than when using citizen scientists.
If the annotation task does not require much domain knowledge (such as part


(a) Sparrow Quiz Scores (b) Shorebird Quiz Scores
Figure 5.5: Histogram of quiz scores. Each quiz has 10 images, a perfect score is

10. **(a)** Score distributions for the sparrow quizzes. Random chance per image is
2.7%. **(b)** Score distributions for the shorebird quizzes. Random chance per image
is 3.1%. See Section 5.6.

0.0 (^412) Annotation Time (hours) 20 28 36 44 52
0.10.2
0.30.4
0.50.6
0.70.8
0.91.0
Error
1x
5x 10x
1x
1x3x5x
MTurkersCitizen Scientists
Citizen Scientists + Vibe
(a) Annotation Time
0.0 $0 $40Annotation Cost$80$120$160$200
0.10.2
0.30.4
0.50.6
0.70.8
0.91.0
Error
MTurkersCitizen Scientists
Citizen Scientists + Vibe
(b) Annotation Cost
(^1000002448) Time (hours) 72 96 120 144
20003000
40005000
60007000
80009000
10000
Annotations Completed
MTurkersCitizen Scientists
(c) Throughput
Figure 5.6: **Category Labeling Tasks:** workers used the quiz interface (see Fig-
ure 5.3a) to label the species of birds in images. **(a)** Citizen scientists are more
accurate and faster for each image than MTurkers. If the citizen scientists use an
expert interface (Vibe), then they are even faster and more accurate. **(b)** Citizen
scientists are not compensated monetarily, they donate their time to the task. **(c)**
The total throughput of MTurk is still greater, meaning you can finish annotating
your dataset sooner, however this comes at a monetary cost. See Section 5.6.
annotation), then MTurkers can perform on par with citizen scientists. Gathering
fine-grained category labels with MTurk should be done with care, as we have shown
that naive averaging of labels does not converge to the correct label. Finally, the cost
savings of using citizen scientists can be significant when the number of annotation
tasks grows.

### 5.7 Measuring the Quality of Existing Datasets

CUB-200-2011 (Wah et al., 2011) and ImageNet (J. Deng et al., 2009) are two pop-
ular datasets with fine-grained categories. Both of these datasets were collected by
downloading images from web searches and curating them with Amazon Mechani-
cal Turk. Given the results in the previous section, we were interested in analyzing


0.8 (^412) Annotation Time (hours) 20 28 36
0.91.0
1.21.1
1.31.4
1.51.6
1.71.8
1.9
Error (Ave # Incorrect Parts)
1x
5x 10x
1x
5x
MTurkersCitizen Scientists
(a) Annotation Time
0.8 $20$60$100Annotation Cost ($)$140$180$220$260
0.91.0
1.11.2
1.31.4
1.51.6
1.71.8
1.9
Error (Ave # Incorrect Parts)
1x
5x
1x
5x
MTurkersCitizen Scientists
(b) Annotation Cost
0K0.0 1.0Time (hours)2.0 3.0
20K10K
30K40K
50K60K
70K80K
100K90K
Annotations Completed
MTurkersCitizen Scientists
(c) Throughput
Figure 5.7: **Parts annotation tasks:** workers used the interface in Figure 5.3b
to label the visibility and location of 11 parts. **(a)** For this task, as opposed to
the category labeling task, citizen scientists and MTurkers perform comparable on
individual images. **(b)** Citizen scientists donate their time and are not compensated
monetarily. **(c)** The raw throughput of MTurk is greater than that of the citizen
scientists, meaning you can finish your total annotation tasks sooner, but this comes
at a cost. See Section 5.6.
the errors present in these datasets. With the help of experts from the Cornell Lab of
Ornithology, we examined these datasets, specifically the bird categories, for false
positives.
**CUB-200-2011:**
The CUB-200-2011 dataset has 200 classes, each with roughly 60 images. Experts
went through the entire dataset and identified a total of 494 errors, about 4 .4%
of the entire dataset. There was a total of 252 images that did not belong in the
dataset because their category was not represented, and a total of 242 images that
needed to be moved to existing categories. Beyond this 4 .4%percent error, an
additional potential concern comes from dataset bias issues. CUB was collected
by performing a Flickr image search for each species, then using MTurkers to filter
results. A consequence is that the most difficult images tended to be excluded from
the dataset altogether. By having experts annotate the raw Flickr search results,
we found that on average 11 .6%of correct images of each species were incorrectly
filtered out of the dataset. See Section 5.8 for additional analysis.
**ImageNet:**
There are 59 bird categories in ImageNet, each with about 1300 images in the training
set. Table 5.1 shows the false positive counts for a subset of these categories. In
addition to these numbers, it was our general impression that error rate of ImageNet
is probably at least as high as CUB-200 within fine-grained categories; for example,
the synset “ruffed grouse, partridge, Bonasa umbellus" had overlapping definition


and image content with the synset “partridge" beyond what was quantified in our
analysis.

Category Training Images False Positives
magpie 1300 11
kite 1294 260
dowitcher 1298 70
albatross, mollymark 1300 92
quail 1300 19
ptarmigan 1300 5
ruffed grouse, par-
tridge, Bonasa um-
bellus
1300 69
prairie chicken,
prairie grouse,
prairie fowl
1300 52
partridge 1300 55
Table 5.1: False positives from ImageNet LSVRC dataset.
### 5.8 Effect of Annotation Quality & Quantity

(^100) log(Number of Categories) 101 102 103
0.10
0.30
0.500.70
0.90
log(Classification Error)
5% corruption15% corruption
50% corruptionPure
(a) Image level features,
train+test corruption
(^100) log(Number of Categories) 101 102 103
0.10
0.30
0.500.70
0.90
log(Classification Error)
5% corruption15% corruption
50% corruptionPure
(b) Image level features, train
corruption only
(^10010) log(Dataset Size)^1102103
0.100.20
0.300.40
0.50
log(Classification Error)
0.05 corruption0.15 corruption
0.50 corruptionPure
(c) Localized features, train
corruption only
Figure 5.8: Analysis of error degradation with corrupted training labels: **(a)** Both
the training and testing sets are corrupted. There is a significant difference when
compared to the clean data. **(b)** Only the training set is corrupted. The induced
classification error is much less than the corruption level. **(c)** Only the training set
is corrupted but more part localized features are utilized. The induced classification
error is still much less than the corruption level. See Section 5.8
In this section, we analyze the effect of data quality and quantity on learned vision
systems. Does the4%+error in CUB and ImageNet actually matter? We begin
with simulated label corruption experiments to quantify reduction in classification
accuracy for different levels of error in Section 5.8, then perform studies on real
corrupted data using an expert-vetted version of CUB in Section 5.8.


**Label Corruption Experiments**
In this experiment, we attempted to measure the effect of dataset quality by cor-
rupting the image labels of the NABirds dataset. We speculated that if an image of
true classXis incorrectly labeled as classY, the effect might be larger if classYis
included as a category in the dataset (i.e., CUB and ImageNet include only a small
subset of real bird species). We thus simulated class subsets by randomly picking
N ≤ 555 categories to comprise our sample dataset. Then, we randomly sampled
Mimages from theNselected categories and corrupted these images by swapping
their labels with another image randomly selected from all 555 categories of the
original NABirds dataset. We used this corrupted dataset ofNcategories to build
a classifier. Note that as the number of categoriesNwithin the dataset increases,
the probability that a corrupted label is actually in the dataset increases. Figure 5.8
plots the results of this experiment for different configurations. We summarize our
conclusions below.

**5-10% Training error was tolerable**
Figures 5.8b and 5.8c analyze the situation where only the training set is corrupted,
and the ground truth testing set remains pure. We see that the increase in classifica-
tion error due to5%and even15%corruption is remarkably low–much smaller than
5%and15%. This result held regardless of the number of classes or computer vision
algorithm. This suggests that the level of annotation error in CUB and ImageNet
(≈5%) might not be a big deal.

**Obtaining a clean test set was important**
On the other hand, one cannot accurately measure the performance of computer
vision algorithms without a high quality test set, as demonstrated in Figure 5.8a,
which measures performance when the test set is also corrupted. There is clearly a
significant drop in performance with even5%corruption. This highlights a potential
problem with CUB and ImageNet, where train and test sets are equally corrupted.

**Effect of computer vision algorithm**
Figure 5.8b uses computer vision algorithms based on raw image-level CNN-fc6
features (obtaining an accuracy of35%on 555 categories) while Figure 5.8c uses
a more sophisticated method (Branson et al., 2014) based on pose normalization
and features from multiple CNN layers (obtaining an accuracy of74%on 555
categories). Label corruption caused similar additive increases in test error for both


methods; however, this was a much higher percentage of the total test error for the
higher performing method.

**Error Analysis on Real CUB-200-2011 Labels**
The results from the previous section were obtained on simulated label corruptions.
We performed additional analysis on real annotation errors on CUB-200-2011.
CUB-200-2011 was collected by performing Flickr image search queries for each
species and filtering the results using votes from multiple MTurkers. We had experts
provide ground truth labels for all Flickr search results on 40 randomly selected
categories. In Figure 5.9, we compare different possible strategies for constructing
a training set based on thresholding the number of MTurk votes. Each method
resulted in a different training set size and level of precision and recall. For each
training set, we measured the accuracy of a computer vision classifier on a common,
expert-vetted test set. The classifier was based on CNN-fc6 features from bounding
box regions. Results are summarized below:

Datasetvote 0 Images 6475 ACC0.78
vote 1vote 2 64676080 0.780.77
vote 3vote 4 50023410 0.770.75
expertvote 5^12775257 0.680.78
Precision
Recall
Figure 5.9: Different datasets can be built up when modifying the MTurker agree-
ment requirement. Increasing the agreement requirement results in a dataset with
low numbers of false positives and lower amounts of training data due to a high
number of false negatives. A classifier trained on all the images performs as well or
better than the datasets that attempt to clean up the data. See Section 5.8.

**The level of training error in CUB was tolerable**
The results were consistent with those predicted by the simulated label corruption
experiments, where a 5-15% error rate in the training errors yielded only a very small
(roughly 1%) increase in test error. This provides comfort that CUB-200-2011 and
ImageNet are still useful despite label errors. We emphasize though that an error
free test set is still necessary–this is still an advantage of NABirds over CUB and
ImageNet.


**Keeping all Flickr images without any MTurk curation does surprisingly well**
This “free dataset" was as good as the expert dataset and slightly better than the
MTurk curated datasets. The raw Flickr image search results had a reasonably
high precision of 81%. Keeping all images resulted in more training images than
the MTurk and expert filtered datasets. If we look at the voter agreement and the
corresponding dataset training sizes, we see that having high MTurk agreement
results in much smaller training set sizes and a correspondingly low recall.

**Quantity can be more important than quality**
This underlines the point that having a large training set is extremely important,
and having strict requirements on annotation quality can come at the expense of
reducing training set size. We randomly reduced the size of the training set within
the 40 class dataset and measured performance of each resulting computer vision
classifier. The results are shown in Table 5.2; we see that classification accuracy is
more sensitive to training set size than it was to label corruption (see Figures 5.8b
and 5.9).

Scale
Size
1 1/2 1/4 1/8 1/16 1/32 1/64
ACC .77 .73 .676 .612 .517 .43 .353
Table 5.2: Classification accuracy with reduced training set size. See Section 5.8.
**Similar results when scaling to more classes**
One caveat is that the above results were obtained on a 40 class subset, which was
the limit of what was reasonable to ask of experts to annotate all Flickr image
search results. It is possible that annotation quality becomes more important as the
number of classes in the dataset grows. To test this, we had experts go through
all 200 classes in CUB-200-2011, annotating all images that were included in the
dataset (see Section 5.7). We obtained a similar result as on the 40-class subset,
where the expert filtered dataset performed at about the same level as the original
CUB-200-2011 trainset that contains 4-5% error. These results are consistent with
simulated label corruption experiments in Figure 5.8b.

### 5.9 Conclusion

We introduced tools for crowdsourcing computer vision annotations using citizen
scientists. In collecting a new expert-curated dataset of 48,562 images over 555
categories, we found that citizen scientists provide significantly higher quality la-


bels than Mechanical Turk workers, and found that Turkers have alarmingly poor
performance annotating fine-grained classes. This has resulted in error rates of
over4%in fine-grained categories in popular datasets like CUB-200-2011 and Im-
ageNet. Despite this, we found that learning algorithms based on CNN features and
part localization were surprisingly robust to mislabeled training examples as long
as the error rate is not too high, and we would like to emphasize that ImageNet and
CUB-200-2011 are still very useful and relevant datasets for research in computer
vision.

Our results so far have focused on experiences in a single domain (birds) and have
resulted in a new publicly available tool for bird species identification. We are
currently working on expanding to other types of categories such as shoes and
Lepidoptera. Given that over 500 citizen scientists helped provide high quality
annotations in just a single domain, working with citizen scientists has potential to
generate datasets of unprecedented size and quality while encouraging the landscape
of computer vision research to shape around the interests of end users.

### 5.10 Acknowledgments

We would like to thank Nathan Goldberg, Ben Barkley, Brendan Fogarty, Graham
Montgomery, and Nathaniel Hernandez for assisting with the user experiments.
We appreciate the feedback and general guidance from Miyoko Chu, Steve Kelling,
Chris Wood, and Alex Chang. This work was supported in part by a Google Focused
Research Award, the Jacobs Technion-Cornell Joint Research Fund, and Office of
Naval Research MURI N000141010933.

**References**

Angelova, Anelia and Shenghuo Zhu (2013). “Efficient Object Detection and Seg-
mentation for Fine-Grained Recognition”. In: _The IEEE Conference on Computer
Vision and Pattern Recognition (CVPR)_.

Berg, Tamara L, Alexander C Berg, and Jonathan Shih (2010). “Automatic attribute
discovery and characterization from noisy web data”. In: _European Conference
on Computer Vision_. Springer, pp. 663–676.

Berg, Thomas et al. (2014). “Birdsnap: Large-Scale Fine-Grained Visual Catego-
rization of Birds”. In: _2014 IEEE Conference on Computer Vision and Pattern
Recognition_. IEEE, pp. 2019–2026.isbn: 978-1-4799-5118-5.doi:10.1109/
CVPR.2014.259.url:http://ieeexplore.ieee.org/lpdocs/epic03/
wrapper.htm?arnumber=6909656.


Boom, Bastiaan J. et al. (2014). “A research tool for long-term and continuous
analysis of fish assemblage in coral-reefs using underwater camera footage”.
In: _Ecological Informatics_ 23, pp. 83–97.issn: 15749541.doi:10.1016/j.
ecoinf.2013.10.006.url:http://www.sciencedirect.com/science/
article/pii/S1574954113001003.

Branson, Steve et al. (2014). “Bird Species Categorization Using Pose Normalized
Deep Convolutional Nets”. In: _arXiv preprint arXiv:1406.2952_.

Deng, Jia, Wei Dong, et al. (2009). “Imagenet: A large-scale hierarchical image
database”. In: _Computer Vision and Pattern Recognition, 2009. CVPR 2009.
IEEE Conference on_. IEEE, pp. 248–255.

Deng, Jia, Jonathan Krause, and Li Fei-Fei (2013). “Fine-grained crowdsourcing for
fine-grained recognition”. In: _Proceedings of the IEEE Conference on Computer
Vision and Pattern Recognition_ , pp. 580–587.

Deng, J. et al. (2009). “ImageNet: A Large-Scale Hierarchical Image Database”. In:
_CVPR09_.

Everingham, M. et al. (2010). “The Pascal Visual Object Classes (VOC) Challenge”.
In: _International Journal of Computer Vision_ 88.2, pp. 303–338.

Fei-Fei, Li, Robert Fergus, and Pietro Perona (2006). “One-shot learning of object
categories”. In: _Pattern Analysis and Machine Intelligence, IEEE Transactions
on_ 28.4, pp. 594–611.

Girshick, Ross et al. (2013). “Rich feature hierarchies for accurate object detection
and semantic segmentation”. In: _arXiv preprint arXiv:1311.2524_.

Gneezy, Uri and Aldo Rustichini (2000). “Pay enough or don’t pay at all”. In:
_Quarterly journal of economics_ , pp. 791–810.

Griffin, G, A Holub, and P Perona (2007). _Caltech-256 Object Category Dataset_.
Tech. rep. CNS-TR-2007-001. California Institute of Technology.url:http:
//authors.library.caltech.edu/7694.

Ipeirotis, Panagiotis G. and Evgeniy Gabrilovich (2014). “Quizz: targeted crowd-
sourcing with a billion (potential) users”. In: pp. 143–154. doi: 10. 1145 /
2566486. 2567988. url: http : / / dl. acm. org / citation. cfm? id =
2566486.2567988.

Ipeirotis, Panagiotis G., Foster Provost, et al. (2013). “Repeated labeling using mul-
tiple noisy labelers”. In: _Data Mining and Knowledge Discovery_ 28.2, pp. 402–
441.issn: 1384-5810. doi: 10.1007/s10618- 013- 0306- 1.url: [http:](http:)
//link.springer.com/10.1007/s10618-013-0306-1.

Khosla, Aditya et al. (2011). “Novel Dataset for Fine-Grained Image Categoriza-
tion”. In: _First Workshop on Fine-Grained Visual Categorization, IEEE Confer-
ence on Computer Vision and Pattern Recognition_. Colorado Springs, CO.


Krause, Jonathan et al. (2013). “Collecting a Large-Scale Dataset of Fine-Grained
Cars”. In: _Second Workshop on Fine-Grained Visual Categorization (FGVC2)_.

Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E Hinton (2012). “ImageNet Clas-
sification with Deep Convolutional Neural Networks.” In: _NIPS_.

Kuznetsov, Stacey (2006). “Motivations of contributors to Wikipedia”. In: _ACM
SIGCAS computers and society_ 36.2, p. 1.

Lazebnik, S., C. Schmid, and Jean Ponce (2004). “Semi-Local Affine Parts for Object
Recognition”. In: _Proc. BMVC_. doi:10.5244/C.18.98, pp. 98.1–98.10.isbn: 1-
901725-25-1.

Lin, Tsung-Yi et al. (2014). “Microsoft COCO: Common objects in context”. In:
_ECCV_.

Liu, Jiongxin et al. (2012). “Dog Breed Classification Using Part Localization.” In:
_ECCV_.

Long, Chengjiang, Gang Hua, and Ashish Kapoor (2013). “Active Visual Recogni-
tion with Expertise Estimation in Crowdsourcing”. In: _2013 IEEE International
Conference on Computer Vision_. IEEE, pp. 3000–3007.isbn: 978-1-4799-2840-
8.doi:10.1109/ICCV.2013.373.url:http://ieeexplore.ieee.org/
lpdocs/epic03/wrapper.htm?arnumber=6751484.

Maji, S. et al. (2013). _Fine-Grained Visual Classification of Aircraft_. Tech. rep.
arXiv:1306.5151 [cs-cv].

Martinez-Munoz, G. et al. (2009). “Dictionary-free categorization of very similar
objects via stacked evidence trees”. In: _2009 IEEE Conference on Computer
Vision and Pattern Recognition_. IEEE, pp. 549–556.isbn: 978-1-4244-3992-8.
doi:10.1109/CVPR.2009.5206574.url:http://ieeexplore.ieee.org/
lpdocs/epic03/wrapper.htm?arnumber=5206574.

Nilsback, M-E. and A. Zisserman (2008). “Automated Flower Classification over a
Large Number of Classes”. In: _Proceedings of the Indian Conference on Computer
Vision, Graphics and Image Processing_.

Nov, Oded (2007). “What motivates wikipedians?” In: _Communications of the ACM_
50.11, pp. 60–64.

Parkhi, Omkar M et al. (2011). “The truth about cats and dogs”. In: _ICCV_.

Raykar, Vikas C et al. (2009). “Supervised learning from multiple experts: whom to
trust when everyone lies a bit”. In: _Proceedings of the 26th Annual international
conference on machine learning_. ACM, pp. 889–896.

Russakovsky, Olga et al. (2014). _ImageNet Large Scale Visual Recognition Chal-
lenge_. eprint:arXiv:1409.0575.


Sheng, Victor S., Foster Provost, and Panagiotis G. Ipeirotis (2008). “Get another
label? improving data quality and data mining using multiple, noisy labelers”. In:
_Proceeding of the 14th ACM SIGKDD international conference on Knowledge
discovery and data mining - KDD 08_. New York, New York, USA: ACM Press,
p. 614.isbn: 9781605581934.doi:10.1145/1401890.1401965.url:http:
//dl.acm.org/citation.cfm?id=1401890.1401965.

Sorokin, Alexander and David Forsyth (2008). “Utility data annotation with Amazon
Mechanical Turk”. In: _2008 IEEE Computer Society Conference on Computer
Vision and Pattern Recognition Workshops_. IEEE, pp. 1–8.isbn: 978-1-4244-
2339-2.doi:10.1109/CVPRW.2008.4562953.url:http://ieeexplore.
ieee.org/lpdocs/epic03/wrapper.htm?arnumber=4562953.

Taigman, Yaniv et al. (2014). “Deepface: Closing the gap to human-level perfor-
mance in face verification”. In: _Computer Vision and Pattern Recognition (CVPR),
2014 IEEE Conference on_. IEEE, pp. 1701–1708.

Von Ahn, Luis (2006). “Games with a purpose”. In: _Computer_ 39.6, pp. 92–94.

Von Ahn, Luis et al. (2008). “recaptcha: Human-based character recognition via
web security measures”. In: _Science_ 321.5895, pp. 1465–1468.

Wah, C. et al. (2011). _The Caltech-UCSD Birds-200-2011 Dataset_. Tech. rep. CNS-
TR-2011-001. California Institute of Technology.

Welinder, Peter, Steve Branson, et al. (2010). “The Multidimensional Wisdom of
Crowds”. In: _Advances in Neural Information Processing Systems 23_. Ed. by J D
Lafferty et al. Curran Associates, Inc., pp. 2424–2432.url:http://papers.
nips.cc/paper/4074-the-multidimensional-wisdom-of-crowds.pdf.

Welinder, Peter and Pietro Perona (2010). “Online crowdsourcing: Rating annotators
and obtaining cost-effective labels”. In: _2010 IEEE Computer Society Conference
on Computer Vision and Pattern Recognition - Workshops_. IEEE, pp. 25–32.
isbn: 978-1-4244-7029-7.doi:10.1109/CVPRW.2010.5543189.url:http:
//ieeexplore.ieee.org/lpdocs/epic03/wrapper.htm?arnumber=
5543189.

Whitehill, Jacob et al. (2009). “Whose vote should count more: Optimal integration
of labels from labelers of unknown expertise”. In: _Advances in neural information
processing systems_ , pp. 2035–2043.

Yang, Heng-Li and Cheng-Yu Lai (2010). “Motivations of Wikipedia content con-
tributors”. In: _Computers in Human Behavior_ 26.6, pp. 1377–1383.


