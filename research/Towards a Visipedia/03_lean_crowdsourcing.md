# Chapter 3: Lean Crowdsourcing: Combining Humans And Machines In An Online System
Branson, Steve, Grant Van Horn, and Pietro Perona (2017). “Lean Crowdsourcing:
Combining Humans and Machines in an Online System”. In: _Proceedings of the
IEEE Conference on Computer Vision and Pattern Recognition_ , pp. 7474–7483.
doi:10.1109/CVPR.2017.647.

### 3.1 Abstract

We introduce a method to greatly reduce the amount of redundant annotations
required when crowdsourcing annotations, such as bounding boxes, parts, and class
labels. For example, if two Mechanical Turkers happen to click on the same pixel
location when annotating a part in a given image–an event that is very unlikely to
occur by random chance– it is a strong indication that the location is correct. A
similar type of confidence can be obtained if a single Turker happened to agree
with a computer vision estimate. We thus incrementally collect a variable number
of worker annotations per image based on online estimates of confidence. This is
done using a sequential estimation of risk over a probabilistic model that combines
worker skill, image difficulty, and an incrementally trained computer vision model.
We develop specialized models and algorithms for binary annotation, part keypoint
annotation, and sets of bounding box annotations. We show that our method can
reduce annotation time by a factor of 4-11 for binary filtering of websearch results,
2-4 for annotation of boxes of pedestrians in images, while in many cases also
reducing annotation error. We will make an end-to-end version of our system
publicly available.

### 3.2 Introduction

Availability of large labeled datasets like ImageNet (Deng, Dong, et al., 2009; Lin et
al., 2014) is one of the main catalysts for recent dramatic performance improvement
in computer vision (Krizhevsky, Sutskever, and Hinton, 2012; He et al., 2015;
Szegedy, W. Liu, et al., 2015; Szegedy, Reed, et al., 2014). While sophisticated
crowdsourcing algorithms have been developed for classification (Whitehill et al.,
2009; Welinder, Branson, et al., 2010; Welinder and Perona, 2010), there is a


relative lack of methods and publicly available tools that use smarter crowdsourcing
algorithms for other types of annotation.

Computer
Vision
Online Model ofWorker Skill
and Crowdsourcing
Amazon
Mechanical
Turk
Dataset (^) AnnotationsHuman
**2
3
5
4
6
1**
Figure 3.1: A schematic of our proposed method. **(1)** The system is initialized with
a dataset of images. Each global step of the method will add annotations to this
dataset. **(2)** The computer vision system incrementally retrains using current worker
labels. **(3)** The crowdsourcing model updates its predictions of worker skills and
image labels and decides which images are finished based on a risk-based quality
assurance threshold. Unfinished images are sent to Amazon Mechanical Turk. **(4-5)**
Workers on AMT annotate the images. **(6)** The crowdsourcing model continues to
update its predictions of worker skills and image labels, and the cycle is repeated
until all images are marked as complete.
We have developed a simple-to-use, publicly available tool that incorporates and
extends many recent advances in crowdsourcing methods to different types of an-
notation, like part annotation and multi-object bounding box annotation, and also
interfaces directly with Mechanical Turk. One key inspiration is the notion of on-
line crowdsourcing (Welinder and Perona, 2010), where instead of obtaining the
same number of annotations for all images, the parameters of the crowdsourcing
model are estimated incrementally until a desired confidence level on image labels
is achieved. We find that this type of approach is very effective for annotation
modalities such as parts and bounding boxes, if one first develops an appropriate
probabilistic model of annotation. Second, we develop and test models of worker
skill and image difficulty, which we develop for parts, bounding boxes, and binary
classification. Further, online crowdsourcing can naturally be extended by machine-
in-the-loop methods, where an incrementally-trained, computer vision predictor is
another source of information in the online crowdsourcing early stoppage criterion.
Our main contributions are: (1) an online algorithm and stopping criterion for
binary, part, and object crowdsourcing, (2) a worker skill and image difficulty
crowdsourcing model for binary, part, and object annotations, (3) incorporation of
online learning of computer vision algorithms to speedup crowdsourcing, and (4) a


publicly available tool that interfaces with Mechanical Turk and incorporates these
algorithms. We show that contributions 1–3 lead to significant improvements in
annotation time and/or annotation quality for each type of annotation. For binary
classification, annotation error with 1. 37 workers per image is lower using our
method than when using majority vote and 15 workers per image. For bounding
boxes, our method produces lower error with 1. 97 workers per image, compared
to majority vote using 7 workers per image. For parts, a variation of our system
without computer vision was used to annotate accurately a dataset of 11 semantic
parts on 55,000 images, averaging 2. 3 workers per part.

We note that while incorporating computer vision in the loop speeds up annotation
time, computer vision researchers wishing to collect datasets for benchmarking
algorithms may choose to toggle off this option to avoid potential issues with bias.
At the same time, we believe that it is a very valuable feature in applied settings.
For example, a biologist may need to annotate the location of all cells in a dataset of
images, not caring if the annotations come from humans or machines, but needing
to ensure a certain level of annotation quality. Our method offers an end-to-end tool
for collecting training data, training a prediction algorithm, combining human and
machine predictions and vetting their quality, while attempting to minimize human
time. This may be a useful tool for several applications.

### 3.3 Related Work

Kovashka et al. (A. Kovashka et al., 2016) provide a thorough overview of crowd-
sourcing in computer vision. Sorokin and Forsyth (Sorokin and Forsyth, 2008)
proposed three methods for collecting quality annotations on crowdsourcing plat-
forms. The first is to build a gold standard set (Larlus et al., 2014) to verify work
and filter out underperforming workers. The second is to use a grading scheme
to evaluate the performance of the workers. This scheme can be accomplished by
workers grading each other through a variety of interfaces (Su, Deng, and Fei-Fei,
2012; Russakovsky, L.-J. Li, and Fei-Fei, 2015; Lin et al., 2014), workers grading
themselves through monetary incentives (Nihar Bhadresh Shah and Denny Zhou,
2015), and heuristic grading (Russell et al., 2008; Vittayakorn and J. Hays, 2011).
The third method resorts to redundantly annotating images and aggregate the results.
This has become the standard method for crowdsourcing.

There is a large body of work that explores aggregating worker answers to maximize
the accuracy of the estimated labels. Approaches that propose methods to combine


multiple annotations with an assurance on quality are the most similar to our method.
Existing work has predominantly used the Dawid-Skene model (Dawid and Skene,
1979). The Dawid-Skene model iteratively infers the reliability of each worker and
updates the belief on the true labels. In this setup, individual tasks are assumed to
be equally difficult, and researchers often reduce the task to binary classification.
Inference algorithms for this model include (Dawid and Skene, 1979; Smyth et
al., 1995; Jin and Ghahramani, 2002; Sheng, Provost, and Ipeirotis, 2008; Ghosh,
Kale, and McAfee, 2011; Karger, Oh, and D. Shah, 2011; Q. Liu, Peng, and Ihler,
2012; Denny Zhou et al., 2012; H. Li and Yu, 2014; Y. Zhang et al., 2014; Dalvi
et al., 2013; Karger, Oh, and D. Shah, 2013; Ok et al., 2016). Some work has
focused on theoretical bounds for this setup (Karger, Oh, and D. Shah, 2011; Ghosh,
Kale, and McAfee, 2011; Gao and Dengyong Zhou, 2013; Y. Zhang et al., 2014;
Dalvi et al., 2013). (Sheng, Provost, and Ipeirotis, 2008; Tian and Zhu, 2015)
reconcile multiple annotators through majority voting and worker quality estimates.
(Welinder and Perona, 2010; Welinder, Branson, et al., 2010; Long, Hua, and
Kapoor, 2013; Wang, Ipeirotis, and Provost, 2013) jointly model labels and the
competence of the annotators. (Hua et al., 2013; Long, Hua, and Kapoor, 2013;
Long and Hua, 2015) explore the active learning regime of selecting the next data
to annotate, as well as which annotator to query. Our approach differs from these
previous methods by merging the online notion of (Welinder and Perona, 2010)
with the worker modeling of (Welinder, Branson, et al., 2010), and we incorporate a
computer vision component as well as provide the framework for performing binary
classification, bounding box and part annotations.

In online marketplaces, it is typically unrealistic to assume that all workers are
equally adept at a task. The previously listed work specifically tries to model
this. However, it is also unrealistic to assume that task difficulty is constant for all
tasks. Methods have been developed to model both worker skills and task difficulty
(Carpenter, 2008; Raykar et al., 2010; Whitehill et al., 2009; Welinder, Branson,
et al., 2010; Snow et al., 2008; Sheng, Provost, and Ipeirotis, 2008; Denny Zhou
et al., 2012; Dengyong Zhou et al., 2015), and there is a growing body of work
that moves past the Dawid-Skene model (Carpenter, 2008; Whitehill et al., 2009;
Welinder, Branson, et al., 2010; Long, Hua, and Kapoor, 2013; Wang, Ipeirotis,
and Provost, 2013; Dengyong Zhou et al., 2015; Nihar B Shah, Balakrishnan, and
Wainwright, 2016).

Our work is related to human-in-the-loop active learning. Prior work in this area


has contributed methods for tasks such as fine-grained image classification (Branson
et al., 2010; Wah, Branson, Perona, et al., 2011; Deng, Krause, and Fei-Fei, 2013;
Wah, Van Horn, et al., 2014), image segmentation (Rubinstein, C. Liu, and Freeman,
2012; Dutt Jain and Grauman, 2013; Gurari et al., 2015; Jain and Grauman, 2016),
attribute-based classification (A. Kovashka, Vijayanarasimhan, and Grauman, 2011;
Parkash and Parikh, 2012; Biswas and Parikh, 2013), image clustering (Lad and
Parikh, 2014), image annotation (Vijayanarasimhan and Grauman, 2009a; Vijaya-
narasimhan and Grauman, 2009b; Siddiquie and Gupta, 2010; Yao et al., 2012;
Russakovsky, L.-J. Li, and Fei-Fei, 2015), human interaction (Khodabandeh et al.,
2015) and object annotation (Vondrick, D. Patterson, and Ramanan, 2013) and seg-
mentation (Shankar Nagaraja, Schmidt, and Brox, 2015) in videos. For simplicity,
we do not incorporate an active learning component when selecting the next batch
of images to annotate or question to ask, but this can be included in our framework.
Additional methods to reduce annotation effort include better interfaces, better task
organization (Chilton et al., 2013; Deng, Russakovsky, et al., 2014; Wilber, Kwak,
and S. J. Belongie, 2014), and gamifcation (Von Ahn and Dabbish, 2004; Von Ahn
and Dabbish, 2005; Kazemzadeh et al., 2014; Deng, Krause, and Fei-Fei, 2013).
Our work is different from the previous work because we combine a worker skill
model, a task difficulty model, a computer vision component, and presenting frame-
works for binary classification, multi-instance bounding box annotation, and part
keypoint annotation. Additional methods have focused on filtering out bad workers
(Long, Hua, and Kapoor, 2013; Hua et al., 2013; Long and Hua, 2015) or com-
bining known weak and strong annotators (Chicheng Zhang and Chaudhuri, 2015;
Gurari et al., 2015; G. Patterson, G. V. H. S. Belongie, and P. P. J. Hays, 2015), or
optimizing the payment to the workers (Wang, Ipeirotis, and Provost, 2013).
### 3.4 Method

LetX = {xi}iN= 1 be a set of images we want to label with unknown true labels
Y={yi}iN= 1 using a pool of imperfect crowd workers. We first describe the problem
generally, where depending on the desired application, each yi may represent a
class label, bounding box, part location, or some other type of semantic annotation.
For each imagei, our goal is to recover a labely ̄ithat is equivalent toyiwith high
probability by combining multiple redundant annotationsZi={zi j}|Wj= 1 i|, where each
zi jis an imperfect worker label (i.e., their perception ofyi), andWiis that set of
workers that annotated imagei.


Importantly, the number of annotations|Wi| can vary significantly for different
imagesi. This occurs because our confidence in an estimated labely ̄iwill depend not
only on the number of redundant annotations|Wi|, but also on the level of agreement
between those annotationsZi, the skill level of the particular workers that annotated
i, and the agreement with a computer vision algorithm (that is incrementally trained).
For example, if two different annotators both happened to provide nearly identical
bounding box annotations for a given image–the probability of which is very small
by random chance– we could be fairly confident of their correctness. On the
other hand, we couldn’t, as a general rule, collect only two bounding boxes per
image, because workers will occasionally make mistakes. Our objective is then to
implement an online policy for choosing whether or not to augment each image with
additional annotations and to incrementally train a computer vision algorithm with
the annotations that have been collected so far.
Online Crowdsourcing
We first describe a simplified model that does not include a worker skill model or
computer vision in the loop. We will augment this simplified model in subsequent
sections. At any given time step, letZ={Zi}iN= 1 be the set of worker annotations for
all images. We define the probability over observed images, true labels, and worker
labels asp(Y,Z)=Œip(yi)
(Œ

j∈Wip(zi j|yi)
)

, wherep(yi)is a prior probability over
possible labels, andp(zi j|yi)is a model of noisy worker annotations. Here we have
assumed that each worker label is independent. The maximum likelihood solution
Y ̄=arg maxp(Y|Z)=arg maxp(Y,Z)can be found for each image separately:

y ̄i=arg maxy
i
©≠

́

p(yi)
÷

j∈Wi
p(zi j|yi)™Æ
̈
(3.1)

The riskR(y ̄i)=
∫

yi`(yi,y ̄i)p(yi|Zi)associated with the predicted label is
R(y ̄i)=
∫

yi`(yi,y ̄i)p(yi)
Œ

j∈Wip(zi j|yi)
∫
yip(yi)
Œ

j∈Wip(zi j|yi)
(3.2)

where`(yi,y ̄i) is the loss associated with the predicted labely ̄i when the true
label isyi. A logical criterion is to accepty ̄ionce the risk drops below a certain
thresholdR(y ̄i) ≤τ(i.e.,τis the minimum tolerable error per image). The basic
online crowdsourcing algorithm, shown in Algorithm 1, processes images in batches
(because sending images to services like Mechanical Turk is easier in batches).

Currently, we give priority to annotating unfinished images with the fewest number
of worker annotation|Wi|; however, one could incorporate more sophisticated active
learning criteria in future work. Each time a new batch is received, combined image
labelsy ̄iare re-estimated, and the risk criterion is used to determine whether or not
an image is finished or may require more worker annotations.

**Algorithm 1** Online Crowdsourcing

1: input : unlabeled imagesX={xi}iN= 1
2:Initialize unfinished/finished sets:U←{i}iN= 1 ,F←∅
3:InitializeW ̄,I ̄using prior probabilities
4: repeat
5: Select a batchB⊆Uof unfinished examples
6: Fori∈Bobtain new crowd labelzi j:Zi←Zi∪zi j
7: repeat .Max likelihood estimation
8: Estimate dataset-wide priorsp(di),p(wj)
9: Predict true labels:
∀i,y ̄i←arg maxyip(yi|xi,θ ̄)p(Zi|yi,d ̄i,W ̄)
10: Predict image difficulties:
∀i,d ̄i←arg maxdip(di)p(Zi|y ̄i,di,W ̄)
11: Predict worker parameters:
∀j,w ̄j←arg maxwjp(wj)Œ
i∈Ij
p(zi j|y ̄i,d ̄i,wj)
12: until Until convergence
13: UsingK-fold cross-validation, train computer vision on dataset{(xi,y ̄i)}i,|Wi|> 0 ,
and calibrate probabilitiesp(yi|xi,θ ̄k)
14: Predict true labels:
∀i,y ̄i←arg maxyip(yi|xi,θ ̄)p(Zi|yi,d ̄i,W ̄)
15: for i∈B do .Check for finished labels
16: Ri←
∫
yi`(yi,y ̄i)p(y ̄i|xi,θ ̄)
Œ
∫ j∈Wip(zi j|yi,di,wj)
yip(y ̄i|xi,θ ̄)
Œj∈W
ip(zi j|yi,di,wj)
17: ifRi≤τ:F←F∪i,U←U\i
18: end for
19: until U=∅
20: return Y←{y ̄i}iN= 1
**Adding Computer Vision**
A smarter algorithm can be obtained by using the actual pixel contentsxiof each
image as an additional source of information. We consider two possible approaches:
(1) a naive algorithm that treats computer vision the same way as a human worker
by appending the computer vision predictionzi,cvto the set of worker labelsWi,


and (2) a smarter algorithm that exploits the fact that computer vision can provide
additional information than a single label output (e.g., confidence estimates that a
bounding box occurs at each pixel location in an image).

For the smarter approach, the joint probability over observed images, true labels,
and worker labels is:

p(Y,Z, θ|X)=p(θ)
÷

©≠

́

p(yi|xi, θ)
÷

j∈Wi
p(zi j|yi)™Æ
̈
(3.3)

wherep(yi|xi, θ)is the estimate of a computer vision algorithm with parametersθ.
Ifθis fixed, the predicted labely ̄ifor each image and its associated riskR(y ̄i)can
be simply found by using the computer vision predictionp(yi|xi, θ)instead of the
priorp(yi)in Equations 3.1, 3.2.

**Training Computer Vision:**
The main challenge is then training the computer vision system (estimating computer
vision parametersθ), given that we incrementally obtain new worker labels over time.
While many possible approaches could be used, in practice we retrain the computer
vision algorithm each time we obtain a new batch of labels from Mechanical Turk.
For each step, we treat the currently predicted labelsy ̄ifor each image with at least
one worker label|Wi| ≥ 1 as training labels to an off-the-shelf computer vision
algorithm. While the predicted labelsy ̄iare clearly very noisy when the number of
workers per image is still small, we rely on a post-training probability calibration
step to cope with resulting noisy computer vision predictions. We use a modified
version ofK-fold cross validation: for each splitk, we use(K− 1 )/Kexamples
for training and the remaining(k− 1 )/Kexamples for probability calibration. We
filter out images with |Wi| < 1 from both training and probability calibration;
however, all 1 /Kimages are used for outputting probability estimatesp(yi|xi, θk),
including images with|Wi|= 0. This procedure ensures that estimatesp(yi|xi, θk)
are produced using a model that wasn’t trained on labels from imagei.

**Worker Skill and Image Difficulty Model**
More sophisticated methods can model the fact that some workers are more skillful
or careful than others, and some images are more difficult or ambiguous than others.
LetW={wj}Mj= 1 be parameters encoding the skill level of our pool ofMcrowd
workers, and letD={di}in= 1 be parameters encoding the level of inherent difficulty
of annotating each imagei(to this point, we are just definingWandDabstractly).


Then the joint probability is

p(Y,Z,W,D, θ|X)=
p(θ)
÷

(p(di)p(yi|xi, θ))
÷

j
p(wj)
÷

i,j∈Wi
p(zi j|yi,di,wj) (3.4)
where p(di) is a prior on the image difficulty, p(wj)is a prior on a worker’s
skill level, andp(zi j|yi,di,wj)models noisy worker responses as a function of the
ground truth label, image difficulty, and worker skill parameters. LetY ̄,W ̄,D ̄,θ ̄=
arg maxY,W,D,θp(Y,W,D, θ|X,Z)be the maximum likelihood solution to Eq. 3.4: In
practice, we estimate parameters using alternating maximization algorithms, where
we optimize with respect to the parameters of one image or worker at a time (often
with fast analytical solutions):

y ̄i = arg maxy
i
p(yi|xi,θ ̄)
÷

j∈Wi
p(zi j|yi,di,wj) (3.5)
d ̄i = arg max
di p(di)
÷

j∈Wi
p(zi j|yi,di,wj) (3.6)
w ̄j = arg maxwj p(wj)
÷

i∈Ij
p(zi j|y ̄i,d ̄i,wj) (3.7)
θ ̄ = arg max
θ p(θ)
÷

p(y ̄i|xi, θ) (3.8)
whereIjis the set of images labeled by workerj. Exact computation of the risk
Ri=R(y ̄i)is difficult because labels for different images are correlated throughW
andθ. An estimate is to assume our approximationsW ̄,I ̄, andθ ̄are good enough
R(y ̄i)≈

∫

yi`(yi,y ̄i)p(yi|X,Z,θ, ̄W ̄,D ̄)
R(y ̄i) ≈
∫

yi`(yi,y ̄i)p(yi|xi,θ ̄)
Œ

j∈Wip(zi j|yi,d ̄i,w ̄j)
∫
yip(yi|xi,θ ̄)
Œ

j∈Wip(zi j|yi,d ̄i,w ̄j)
such that Eq. 3.9 can be solved separately for each imagei.

**Considerations in designing priors:**
Incorporating priors is important to make the system more robust. Due to the online
nature of the algorithm, in early batches the number of images|Ij|annotated by each
workerjis likely small, making worker skillwjdifficult to estimate. Additionally,
in practice many images will satisfy the minimum risk criterion with two or less
labels|Wi| ≤ 2 , making image difficultydidifficult to estimate. In practice we use
a tiered prior system. A dataset-wide worker skill priorp(wj)and image difficulty


prior p(di)(treating all workers and images the same) is estimated and used to
regularize per worker and per image parameters when the number of annotations
is small. As a heuristic to avoid over-estimating skills, we restrict ourselves to
considering images with at least 2 worker labels|Wi|> 1 when learning worker
skills, image difficulties, and their priors, since agreement between worker labels
is the only viable signal for estimating worker skill. We also employ a hand-coded
prior that regularizes the learned dataset-wide priors. We will describe what this
means specifically in subsequent sections when we describe each type of annotation.

### 3.5 Models For Common Types of Annotations

Algorithm 1 provides pseudo-code to implement the online crowdsourcing algorithm
for any type of annotation. Supporting a new type of annotation involves defining
how to represent true labelsyiand worker annotationszi j, and implementing solvers
for inferring the (1) true labelsy ̄i(Eq. 3.5), (2) image difficultiesd ̄i(Eq. 3.6), (3)
worker skillsw ̄j(Eq. 3.7), (4) computer vision parametersθ ̄(Eq. 3.8), and (5) risk
Ri associated with the predicted true label (Eq. 3.9). Although this is somewhat
involved, we note that each component 2-5 individually (though useful in practice)
could optionally be omitted. In the sections below, we detail models and algorithms
for 3 common types of annotations: class labels, part annotations, and multi-object
bounding box labels. These cover several building blocks for different annotation
types including boolean variables, continuous variables, and unordered sets.

### 3.6 Binary Annotation

Here, each labelyi∈ 0 , 1 , denotes the absence/presence of a class of interest. This
is the simplest type of annotation and has also been covered most extensively in
crowdsourcing literature, therefore we try to keep this section as simple as possible.
At the same time, many important datasets such as ImageNet (Deng, Dong, et al.,
2009) and Caltech-256 (Griffin, Holub, and Perona, 2007) are obtained by binary
filtering of image search results, and we are unaware of an existing crowdsourcing
tool that incorporates online crowdsourcing with a worker skill model (not to mention
computer vision), so we believe our binary annotation tool is worthwhile.

**Binary worker skill model:**
We model worker skillwj=[p^1 j,p^0 j]using two parameters representing the worker’s
skill at identifying true positives and true negatives, respectively. Here, we assume
zi jgivenyiis Bernoulli, such thatp(zi j|yi= 1 )=p^1 j andp(zi j|yi= 0 )=p^0 j. As


described in Section 3.4, we use a tiered set of priors to make the system robust
in corner case settings where there are few workers or images. Ignoring worker
identity and assuming a worker labelzgiven y is Bernoulli such thatp(z|y =
1 ) = p^1 and p(z|y = 0 ) = p^0 , we add Beta priorsBeta

(

nβp^0 ,nβ( 1 −p^0 )
)

and
Beta

(

nβp^1 ,nβ( 1 −p^1 )
)

on p^0 j and p^1 j, respectively, wherenβ is the strength of
the prior. An intuition of this is that worker j’s own labels zi j softly start to
dominate estimation ofwjonce she has labeled more thannβimages, otherwise
the dataset-wide priors dominate. We also place Beta priorsBeta

(

nβp,nβ( 1 −p)
)

onp^0 andp^1 to handle cases such as the first couple batches of Algorithm 1. In
our implementation, we usep=. 8 as a general fairly conservative prior on binary
variables andnβ= 5. This model results in simple estimation of worker skill priors
p(wj)in line 8 of Algorithm 1 by counting the number of labels agreeing with
combined predictions:

pk=
nβp+Õi j 1 [zi j=y ̄i=k,|Wi|> 1 ]
nβ+Õi j 1 [y ̄i=k,|Wi|> 1 ] , k=^0 ,^1 (3.9)
where 1 []is the indicator function. Analogously, we estimate worker skillswj in
line 11 of Algorithm 1 by counting worker j’s labels that agree with combined
predictions:

pkj =
nβpk+Õi∈Ij 1 [zi j=y ̄i=k,|Wi|> 1 ]
nβ+Õi∈Ij 1 [y ̄i=k,|Wi|> 1 ] , k=^0 ,^1 (3.10)
For simplicity, we decided to omit a notion of image difficulty in our binary model
after experimentally finding that our simple model was competitive with more so-
phisticated models like CUBAM (Welinder, Branson, et al., 2010) on most datasets.

**Binary computer vision model:**
We use a simple computer vision model based on training a linear SVM on features
from a general purpose pre-trained CNN feature extractor (our implementation uses
VGG), followed by probability calibration using Platt scaling (Platt et al., 1999)
with the validation splits described in Sec. 3.4. This results in probability estimates
p(yi|xi, θ)=σ(γ θ·φ(xi))for each imagei, whereφ(xi)is a CNN feature vector,θis
a learned SVM weight vector,γis probability calibration scalar from Platt scaling,
andσ()is the sigmoid function. This simple procedure is easily fast enough to run
in time less than the time to annotate a batch on Mechanical Turk and is reasonably
general purpose.


### 3.7 Part Keypoint Annotation

Part keypoint annotations are popular in computer vision and included in datasets
such as MSCOCO (Lin et al., 2014), MPII human pose (Andriluka et al., 2014),
and CUB-200-2011 (Wah, Branson, Welinder, et al., 2011). Here, each part is
typically represented as anx,ypixel locationland binary visibility variablev, such
thatyi =(li,vi). While we can modelvusing the exact same model as for binary
classification (Section 3.6),l is a continuous variable that necessitates different
models. For simplicity, even though most datasets contain several semantic parts of
an object, we model and collect each part independently. This simplifies notation
and collection; in our experience, Mechanical Turkers tend to be faster/better at
annotating a single part in many images than multiple parts in the same image.

We first note that modeling keypoint visibility as binary classification results in two
worker skill parameters p^0 j andp^1 j, which correspond to the probability that the
worker thinks a part is visible whenvi = 0 andvi= 1 , respectively–these encode
different annotators’ tendencies and biases in annotating a part’s visibility. The
reader can refer to Eqs. 3.10 and 3.9 for computation ofp^0 j,p^1 jand their priors.

**Keypoint worker skill image difficulty model:**
Letlibe the true location of a keypoint in imagei, whileli jis the location clicked by
workerj. We assumeli jis Gaussian distributed aroundliwith varianceσi j^2. This
variance is governed by the worker’s skill or image difficultyσi j^2 =ei jσj^2 +( 1 −ei j)σi^2 ,
whereσj^2 represents worker noise (e.g., some workers are more precise than others)
andσi^2 represents per image noise (e.g., the precise location of a bird’s belly in a given
image maybe inherently ambiguous), andei jis a binary variable that determines if
the variance will be governed by worker skill or image difficulty. However, workerj
sometimes makes a gross mistake and clicks somewhere very far from the Gaussian
center (e.g., worker j could be a spammer or could have accidentally clicked an
invalid location). mi jindicates whether or not jmade a mistake–with probability
pmj– in which caseli jis uniformly distributed in the image. Thus

p(li j|yi,di,wj)=
’

mi j∈ 0 , 1
p(mi j|pmj)p(li j|li,mi j, σi j) (3.11)
wherep(mi j|pmj)=mi jpmj+( 1 −mi j)( 1 −pmj),p(li j|li,mi j, σi j)=|exi ji|+( 1 −ei j)g(‖li j−
li‖^2 ;σi j^2 ),|xi|is the number of pixel locations ini, andg(x^2 ;σ^2 )is the probability
density function for the normal distribution. In summary, we have 4 worker skill
parameterswj=[σj,pmj,p^0 j,p^1 j]describing errors in clicking precise locations, the


Worker 1 Worker 2 Predicted^ Ground Truth^
Worker 1 Worker 2 Predicted Ground Truth
Figure 3.2: Example part annotation sequence showing the common situation where
the responses from 2 workers correlate well and are enough for the system to mark
the images as finished.

probability of making a mistake, probabilities of correctly identifying visibility,
and one image difficulty parameterdi = σi describing ambiguity of the exact
keypoint location in the image. As described in Section 3.6, we place a dataset-
wide Beta priorBeta

(

nβpm,nβ( 1 −pm)
)

on pmj, wherepm is a worker agnostic
probability of making a mistake and an additional Beta priorBeta

(

nβp,nβ( 1 −p)
)

onpm. Similarly, we place Scaled inverse chi-squared priors onσ^2 j andσi^2 , such
thatσ^2 j ∼scale−inv−χ^2 (nβ, σ^2 )andσi^2 ∼scale−inv−χ^2 (nβ, σ^2 )whereσ^2 is a
dataset-wide variance in click location.

**Inferring worker and image parameters:**
These priors would lead to simple analytical solutions toward inferring the maximum
likelihood image difficulties (Eq. 3.6) and worker skills (Eq. 3.7), ifmi j,ei j, andθ
were known. In practice, we handle latent variablesmi jandei jusing expectation
maximization, with the maximization step over all worker and image parameters,
such that worker skill parameters are estimated as

σi^2 =
nβσ^2 +Õj∈Wi( 1 −Eei j)( 1 −Emi j)‖li j−li‖^2
nβ+ 2 +Õj∈Wi( 1 −Eei j)( 1 −Emi j) (3.12)
σ^2 j =
nβσ^2 +
Õ

i∈IjEei j(^1 −Emi j)‖li j−li‖^2
n+ 2 +
Õ

i∈IjEei j(^1 −Emi j)
(3.13)

pmj =
nβpm+Õi∈IjEmi j
nβ+|Ij| (3.14)
These expressions all have intuitive meaning of being like standard empirical es-
timates of variance or binomial parameters, except that each example might be
soft-weighted byEmi jorEei j, andnβsynthetic examples have been added from the
global prior distribution. Expectations are then

Eei j=
gj
gi+gj, Emi j=
1 /|xi|
1 /|xi|+( 1 −Eei j)gi+Eei jgj
gi=g(‖li j−li‖^2 ;σi^2 ), gj=g(‖li j−li‖^2 ;σ^2 j) (3.15)

We alternate between maximization and expectation steps, where we initialize with
Emi j= 0 (i.e., assuming an annotator didn’t make a mistake) andEei j =. 5 (i.e.,
assuming worker noise and image difficulty have equal contribution).

**Inferring true labels:**
Inferringy ̄i(Eq. 3.5) must be done in a more brute-force way due to the presence
of the computer vision termp(yi|xi, θ). LetXibe a vector of length|xi|that stores
a probabilistic part detection map; that is, it stores the value ofp(yi|xi, θ)for each
possible value ofyi. LetZi jbe a corresponding vector of length|xi|that stores the
value ofp(zi j|yi,di,wj)at each pixel location (computed using Eq. 3.11 1 ). Then the
vectorYi=Xi

Œ

j∈WiZi jdensely stores the likelihood of all possible values ofyi,
where products are assumed to be computed using component-wise multiplication.
The maximum likelihood labely ̄iis simply the argmax ofYi.

**Computing risk:**
LetLi be a vector of length |xi| that stores the loss`(yi,y ̄i)for each possible
value ofyi. We assume a part prediction is incorrect if its distance from ground
truth is bigger than some radius (in practice, we compute the standard deviation
of Mechanical Turker click responses on a per part basis and set the radius equal
to 2 standard deviations). The risk associated with predicted labely ̄iaccording to
Eq. 3.9 is thenRi=LTiYi/‖Yi‖ 1.

**Computer Vision:**
We can use any detection system that can produce dense detection scores for pixel
locations in the image, such as a fully convolutional CNN. The part detection scores
can be converted to probabilities using cross-validation, such that part detection
scoresm(xi,li;θ)are converted to probabilities

p(yi|xi, θ)=Õexp{γm(xi,yi;θ)}
liexp{γm(xi,yi;θ)}
(3.16)

withγlearned to maximize the likelihood on the validation set.

### 3.8 Multi-Object Bounding Box Annotations

Similar types of models that were used for part keypoints can be applied to other
types of continuous annotations like bounding boxes. However, a significant new
challenge is introduced if multiple objects are present in the image, such that each

(^1) In practice, we replaceei jandmi jwithEei j andEmi j in Eq. 3.11, which corresponds to
marginalizing over latent variablesei jandmi j, instead of using maximum likelihood estimates.


worker may label a different number of bounding boxes and may label objects in
a different order. Checking for finished labels means ensuring not only that the
boundaries of each box are accurate, but also that there are no false negatives or
false positives.

Computer Vision Worker 1 Prediction Ground Truth
Computer Vision Worker 1 Worker 2 Prediction Ground Truth
Figure 3.3: Bounding box annotation sequences. The top sequence highlights a good
case where only the computer vision system and one human are needed to finish the
image. The bottom sequence highlights the average case where two workers and the
computer vision system are needed to finish the image.

**Bounding box worker skill and image difficulty model**
An image annotationyi={bri}r|B=i 1 |is composed of a set of objects in the image where
boxbriis composed of x,y,x2,y2 coordinates. Workerj’s corresponding annotation
zi j={bki j}|kB=i j 1 |is composed of a potentially different number|Bi j|of box locations

with different ordering. However, if we can predict latent assignments{aki j}k|B=i j 1 |,

wherebki jis workerj’s perception of true boxba

ki j
i , we can model annotation of
a matched bounding box exactly as for keypoints, where 2D vectorslhave been
replaced by 4D vectorsb.

Thus, as for keypoints the difficulty of imageiis represented by a set of bounding
box difficulties:di={σir}r|B=i 1 |, which measure to what extent the boundaries of each
object in the image are inherently ambiguous. A worker’s skillwj ={pfpj,pfnj, σj}
encodes the probabilitypfpj that an annotated boxbki jis a false positive (i.e.,ai jk =∅),
the probabilitypfnj that a ground truth boxbriis a false negative (i.e.,∀k,aki j,r), and
the worker’s varianceσ^2 j in annotating the exact boundary of a box is modeled as in
Section 3.7. The number of true positivesntp, false positivesnfp, and false negatives
benfncan be written asntp=Õ|kB=i j 1 | 1 [ai jk ,∅],nfn=|Bi|−ntp,nfp=|Bi j|−ntp.
This leads to annotation probabilities

p(zi j|yi,di,wj)=
÷

k= 1 ...Bi j,aki j,∅
g
(

(^)
b
aki j
i −b
ki j
(^)
(^)
2
;σi jk^2

)

(pfnj)nfn( 1 −pfnj)ntp(pfpj)nfp( 1 −pfpj)ntp (3.17)

As in the previous sections, we place dataset-wide priors on all worker and image
parameters.

**Computer vision**
We train a computer vision detector based on MSC-MultiBox (Szegedy, Reed, et
al., 2014), which computes a shortlist of possible object detections and associated
detection scores: {(bki,cv,mik,cv)}|kB=i, 1 cv|. We choose to treat computer vision like a
worker, with learned parameters[pfpcv,pfncv, σcv]. The main difference is that we
replace the false positive parameterpfpcvwith a per bounding box prediction of the
probability of correctness as a function of its detection scoremki,cv. The shortlist of
detections is first matched to boxes in the predicted labely ̄i={bri}r|B=i 1 |. Letrik,cvbe 1
or− 1 if detected boxbki,cvwas matched or unmatched to a box iny ̄i. Detection scores
are converted to probabilities using Platt scaling and the validation sets described in
Section 3.4.

10 -2 (^0) Avg Number of Human Workers per Image 2 4 6 8 10 12 14
10 -1
100
Error
Method Comparisonprob-worker-cv-online-.02prob-worker-cv-online-.01
prob-worker-cv-online-.005prob-worker-cv-naive-onlineprob-worker-online
prob-onlineprob-worker-cvprob-worker
probmajority-vote
(a) Binary Method Com-
parison
(^002) Annotations Per Image 4 6 8 10 12 14
1000
2000
3000
4000
5000
6000
7000
Image Count
Number of Annotationsprob-worker-cv-online-.02prob-worker-cv-online-.005
prob-worker-online
(b) Binary # Human An-
notations
**Ground Truth: ScorpionPrediction: Scorpion0 Workers**^
**Ground Truth: ScorpionPrediction: Scorpion1 Workers**^ **Ground Truth: Not ScorpionPrediction: Scorpion14 Workers**^
(c) Binary Qualitative Examples
Figure 3.4: **Crowdsourcing Binary Classification Annotations: (a)** Comparison
of methods. Our full model prob-worker-cvonline-0.02 obtains results as good as
typical baselines with 15 workers (majority-vote and prob) using only 1. 37 workers
per image on average. **(b)** Histogram of the number of human annotations required
for each image. **(c)** The image on the left represents an average annotation situation
where only the computer vision label and one worker label are needed to confidently
label the image. The image on the right (which is not a scorpion) represents a
difficult case in which many workers disagreed on the label.
**Inferring true labels and assignments**
We devise an approximate algorithm to solve for the maximize likelihood labely ̄i
(Eq. 3.5) concurrently with solving for the best assignment variablesai jk between
worker and ground truth bounding boxes:
y ̄i,ai=arg maxy
i,ai
log

’

j∈Wi
logp(zi j|yi,di,wj) (3.18)

wherep(zi j|yi,di,wj)is defined in Eq. 3.17. We formulate the problem as a fa-
cility location problem Erlenkotter, 1978, a type of clustering problem where the
objective is to choose a set of "facilities" to open up given that each "city" must
be connected to a single facility. One can assign custom costs for opening each
facility and connecting a given city to a given facility. Simple greedy algorithms are
known to have good approximation guarantees for some facility location problems.
In our formulation, facilities will be boxes selected to add to the predicted combined
label y ̄i, and city-facility costs will be costs associated with assigning a worker
box to an opened box. Due to space limitations we omit derivation details; how-
ever, we set facility open costsCopen(bi jk)=Õj∈Wi−logpfnj and city-facility costs
Cmatch(bki j,bki j′′)=−log( 1 −pfnj)+logpfnj −log( 1 −pfpj)−logg(‖bki j−bi jk′′‖^2 ;σ^2 j)for
matching worker box bi jk to facilitybki j′′, while not allowing connections where
j = j′ unless k = k′,j = j′. We add a dummy facility with open cost 0,
such that cities matched to it correspond to worker boxes that are false positives:
Cmatch(bki j,dummy)=−logpfpj.

**Computing risk**
We assume that the loss`(y ̄i,yi)for annotating bounding boxes is defined as the
number of false positive bounding boxes plus the number of false negatives, where
boxes match if their area of intersection over union is at least 50%. Previously in this
section, we described a procedure for inferring assignments{ai jk}|kB=i j 1 |between boxes

in each worker labelzi j ={bki j}k|B=i j 1 |and predicted combined labelsy ̄i ={b ̄ri}rB ̄=i 1.
To simplify calculation of risk, we assume our inferred correspondences are valid.
In this case, the probabilityp(fp(b ̄ri))that a boxb ̄ri ∈ y ̄iis a false positive can be
computed by evaluating the likelihood of worker labels with and withoutb ̄ri:

p(fp(b ̄ri)) =
Œ

j∈Wi jp(zi j|y ̄i\b ̄ri,di,wj)
Œ
j∈Wi jp(zi j|y ̄i\b ̄ri,di,wj)+
Œ

j∈Wi jp(zi j|y ̄i,di,wj)
(3.19)

=^1

1 +Œj∈Wi jp(pz(i jzi j|y ̄|iy\ ̄ib, ̄dri,wj)
i,di,wj)
(3.20)

=^1

1 +Œj∈Wi jpfnj Œk,aki j=r
( 1 −pfnj)( 1 −pfpj)g
(
bki j;b ̄ri,σi jk^2
)
pfnjpfpj
(3.21)

where the second line was found by substituting inp(zi j|y ̄i,di,wj)as defined in Eq.
18 of the main paper. Computing the expected number of false negatives is more
complicated, because it involves considering each possible bounding box location


b ̄ri′in the image (not just iny ̄i), and evaluating the relative likelihood of all worker
labels ifb ̄ri′were addedy ̄i, sop(fn(b ̄ri′))=:
Œ
j∈Wi jp(zi j|y ̄i∪b ̄r

′
Œ i,di,wj)
j∈Wi jp(zi j|y ̄i∪b ̄r
′
i,di,wj)+
Œ

j∈Wi jp(zi j|y ̄i,di,wj)
(3.22)

= 1 −

Œ

j∈Wi jp(zi j|y ̄i,di,wj)
Œ
j∈Wi jp(zi j|y ̄i,di,wj)+
Œ

j∈Wi jp(zi j|y ̄i∪b ̄r
′
i,di,wj)
(3.23)

= 1 −^1

1 +Œj∈Wi jpfnj Œk,ai j′k=r′
( 1 −pfnj)( 1 −pfpj)g
(
bki j;b ̄ri′,σi jk^2
)
pfnjpfpj
(3.24)

whereai j′k=r′represents worker boxes that would be assigned tobri′if it existed in
the true label. To infer these assignments, we extend the facility location algorithm.
Note that in this algorithm, some worker boxesbki jmay be assigned to the dummy
facilityai jk=∅, which represents worker boxes that are inferred to be false positives.
Such worker boxes could also be interpreted as providing probabilistic evidence that
a box may occur in a nearby location. We setup a second facility location problem
where we enumerate all possible values ofbri′(in practice, we incrementally grow
a big set of all possible bounding box locationsBibig, adding a new one if its
intersection over union is at least 50% from all previous boxes in the set). Each
bri′ in this set is a possible facility with open costCopen(bri′) =−log

Œ

j∈Wipfnj,
and each unassigned worker boxbki jwithaki j=∅is a city that can be connected

to itCmatch(bki j,bri′)=−log

(

( 1 −pfnj)( 1 −pfpj)g
(
bki j;bri′,σi jk^2
)
pfnj
)

. This procedure allows us to

infer assignmentsai j′k =r′such that evaluating Eq. 3.24 to estimatep(fn(b ̄ri′))is
possible. The facility location algorithm can be understood as a way of predicting
correspondences/assignments between worker boxes (since each worker may label
objects in a different order); the criterion to infer these correspondences is to
minimize the negative log-likelihood of all worker labels.

The last consideration is that predicted boxes where the boundaries are too inaccurate
will incur both a false positive and false negative according to our loss function.
Again assuming that assignmentsai jkbetween worker boxes and predicted boxes are
correct, the probabilityp(bnd_off(b ̄ri))that a predicted boxb ̄ri has boundaries that


are too far off is:

p(bnd_off(b ̄ri)) =
π
b,IOU(b,b ̄ri)>. 5
÷

j k,j∈Wi,aki j=r
g(bki j;b, σi j^2 )db (3.25)
≈

π
b,‖b−b ̄
ri‖ 2
‖b ̄ri‖^2 >.^5
÷

j k,j∈Wi,aki j=r
g(bi jk;b, σi j^2 )db (3.26)
= 1 −erf
©≠

≠

́

. 5

√√√

√ ’

j k,j∈Wi,ai jk=r
‖b ̄ri‖^2
σi j^2
™Æ

Æ

̈

(3.27)

whereg(x;μ, σ^2 )is the density function of the Normal distribution anderf(x)is
the error function. In the 2nd line, we use an approximation that the region where
the intersection over union of two boxes is greater than a threshold is similar to the
region where their Euclidean distance is greater than a threshold. This enables the
integral to have a simple analytical solution.

The total risk Ri is then the expected number of false positives (computed by
summing over eachbriand computingp(fp(b ̄ri))according to Eq. 3.21), the expected
number of false negatives (computed by summingp(fn(b ̄ri))over all possible boxes
bri′in the image according to Eq. 3.24), and the expected number of true positives that
were too inaccurate to meet the area of intersection over union criterion (computed
by summing over eachbri and computingp(bnd_off(b ̄ri)using Eq. 3.27):

Ri=
’|B ̄i|
r= 1
p(fp(b ̄ri))+
|’Bbigi |
r′= 1
p(fn(b ̄ri′))+
’|B ̄i|
r= 1
( 1 −p(fp(b ̄ri)))p(bnd_off(b ̄ri)) (3.28)
### 3.9 Experiments

We used a live version of our method to collect parts for the NABirds dataset (see
Chapter 5). Additionally, we performed ablation studies on datasets for binary,
part, and bounding box annotation based on simulating results from real-life MTurk
worker annotations.

**Evaluation Protocol**
For each image, we collected an over-abundance of MTurk annotations per image,
which were used to simulate results by adding MTurk annotations in random order.
The online crowdsourcing algorithm chose whether or not to terminate receiving
additional annotations. For lesion studies, we crippled portions of Algorithm 1


as follows: (1) we removed online crowdsourcing by simply running lines 7-14
over the whole dataset withk workers per image and sweeping over choices of
k, (2) we removed the worker skill, image difficulty model by using dataset-wide
priors, and (3) we removed computer vision by using label priorsp(yi)instead of
computer vision estimatesp(yi|xi, θ). As a baseline, the _majority-vote_ method in
plots 3.4a,3.5a,3.5c shows what we consider to be the most standard and commonly
used method/baseline for crowdsourcing. For binary annotation, this selects the
label with the most worker votes. For parts, it selects the median worker part
location (i.e., the one that matches the most other worker annotations with minimal
loss). The same basic method is used for bounding boxes, adding a box if the
majority of workers drew a box that could be matched to it. Figs. 3.4a,3.5a,3.5c
show results for different lesioned methods. In each method name, the tag _worker_
means that a worker skill and image difficulty model was used, the tag _online_ means
that online crowdsourcing was used (with parameterτ =. 005 , unless a different
number appears in the method name), the tag _cvnaive_ means that a naive method
to incorporate computer vision was used (by treating computer vision like a human
worker, see Section 3.4), and the tag _cv_ means that computer vision probabilities
described in Section 3.6-3.7,3.8 were used.

10 -2Avg Number of Human Workers per Image 0 1 2 3 4 5 6
10 -1
100
101
Error
Method Comparisonprob-worker-cv-online-.02prob-worker-cv-online-.01
prob-worker-cv-online-.005prob-worker-cv-naive-onlineprob-worker-online
prob-onlineprob-worker-cvprob-worker
probmajority-vote
(a) BBox Method
Comp

(^001) Annotations Per Image 2 3 4 5 6 7
200
400
600
800
1000
Image Count
Number of Annotationsprob-worker-cv-online-.02prob-worker-cv-online-.005
prob-worker-online
(b) BBox # Human
Annotations
Avg # of Human Workers per Part per Image^0123456789
10 -1
0.05
0.06
0.07
0.08
0.09
Error
Method Comparisonprob-worker-online
prob-onlineprob-worker
probmajority-vote
(c) Parts
(^0012) Annotations Per Part 345678910
5000
10000
15000
20000
25000
Image Count
Number of Annotationsprob-worker-online
prob-online
(d) Parts
Figure 3.5: **Crowdsourcing Multi-Object Bounding Box and Part Annotations:
(a)** Our full model prob-worker-cvonline-0.02 obtains slightly lower error than
majority-vote while using only 1.97 workers per image. **(b)** Histogram of the number
of human annotators per image. **(c)** The worker skill model (prob-worker) led to
10%reduction in error over the majority-vote baseline, and the online model cut
annotation time roughly in half. **(d)** Histogram of the number of human annotators
per part.
**Binary Annotation**
We collected 3 datasets (scorpions, beakers, and cardigan sweaters) which we believe
to be representative of the way datasets like ImageNet (Deng, Dong, et al., 2009)
and CUB-200-2011 (Wah, Branson, Welinder, et al., 2011) were collected. For each


category, we collected 4000 Flickr images by searching for the category name. 15
MTurkers per image were asked to filter search results. We obtained ground truth
labels by carefully annotating images ourselves. Fig. 3.4a summarizes performance
for the scorpion category (which is typical, see supplementary material for results
on more categories), whereas Fig. 3.4c shows qualitative examples.

The full model prob-worker-cvonline-0.02 obtained results as good as typical base-
lines with 15 workers (majority-vote and prob) using only 1. 37 workers per image on
average. The method prob-online corresponds to the online crowdsourcing method
of Welinder et al. (Welinder and Perona, 2010), which used 5. 1 workers per im-
age and resulted in an error of 0. 045 ; our full method prob-worker-cvonline-0.005
obtained lower error 0. 041 with only 1. 93 workers per image. We see that incor-
porating a worker skill model reduced converged error by about33%(comparing
prob-worker to majority-vote or prob). Adding online crowdsourcing roughly halved
the number of annotations required to obtain comparable error (comparing prob-
worker-online vs. prob-worker). Adding computer vision reduced the number of
annotations per image by an additional factor of 2. 4 with comparable error (compar-
ing prob-worker-cvonline-0.005 to prob-worker-online). It also reduced annotations
by a factor of 1. 8 compared to the naive method of using computer vision (prob-
worker-cvnaive-online), showing that using computer vision confidence estimates
is useful. Interestingly, in Fig. 3.4b we see that adding computer vision allowed
many images to be predicted confidently using no worker labels. Lastly, comparing
prob-worker-cvonline-0.02 to prob-worker-cvonline-0.005, which resulted in errors
of 0. 051 and 0. 041 , respectively, and 1. 37 vs. 1. 93 workers per image, we see that
the error tolerance parameterτoffers an intuitive parameter to tradeoff annotation
time and quality.

**Bounding Box Annotation**
To evaluate bounding box annotation, we used a 1448 image subset of the Caltech
Roadside Pedestrian dataset (Hall and Perona, 2015). The dataset is challenging,
because it contains images of pedestrians in the wild; some images contain no
pedestrians, others contain many, pedestrians are often walking next to each other
causing overlapping bounding boxes, and some pedestrians are far away and less
than 10 pixels. We obtained ground truth annotations and 7 MTurk annotations per
image from the creators of the dataset. We incur error for all false positives and
negatives using a .5 IOU overlap criterion.


Prediction: Scorpion0 Workers^
Ground Truth: Scorpion
Prediction: Scorpion1 Workers^
Ground Truth: Scorpion
Prediction: Scorpion14 Workers^
Ground Truth: Not Scorpion
Figure 3.6: Binary classification sequences. The image on the left represents a best
case scenario where the computer vision is able to confidently label the image. The
image in the center represents an average annotation situation where the computer
vision label and one worker label is needed to confidently label the image. The
image on the right (which is not a scorpion) represents a difficult case in which
many workers disagreed on the label.

10 -2 (^0) Avg Number of Human Workers per Image 2 4 6 8 10 12 14
10 -1
100
Error
Method Comparisonprob-worker-cv-online-.02
prob-worker-cv-online-.01prob-worker-cv-online-.005
prob-worker-cv-naive-onlineprob-worker-online
prob-onlineprob-worker-cv
prob-workerprob
majority-vote
(a) Binary Method Comparison (Scorpi-
ons)
10 -2 (^0) Avg Number of Human Workers Per Image 2 4 6 8 10 12 14
10 -1
100
Error
Method Comparisonprob-worker-cv-online-.02
prob-worker-cv-online-.005prob-worker-cv-naive-online
prob-worker-onlineprob-online
prob-worker-cvprob-worker
probmajority-vote
(b) Binary Method Comparison
(Beakers)
Figure 3.7: **Crowdsourcing Binary Classification Annotations:** We ran binary
classification experiments on two different datasets: scorpions and beakers, which
were selected because they are two different types of objects that pose different
challenges: scorpions are natural objects, with some grossly different objects in
search results (the band scorpions, the video game character, the motorcycle) and
some very related objects (scorpion spiders are spiders that resemble scorpions).
Beakers are man-made objects, which are often not the subject of the photo (and
thus not centered and very small in the image), and many people mistakenly tag
images of flasks and granulated cylinders as beakers. **(a) Results on the Scorpion
Dataset:** Our full model prob-worker-cvonline-0.02 obtains results as good as
typical baselines with 15 workers (majority-vote and prob) using only 1. 37 workers
per image on average. **(b) Results on the Beakers Dataset:** Similar trends occur
for both beakers and scorpions


In Fig. 3.5a, we see that the full model prob-worker-cvonline-0.02 obtained slightly
lower error than majority-vote while using only 1.97 workers per image. This is en-
couraging, given that most publicly available crowdsourcing tools for bounding box
annotation use simple crowdsourcing methods. Incorporating a probabilistic model
(comparing prob to majority-vote) reduced the error by a factor of 2, demonstrat-
ing that it is useful to account for probabilities of false positive and false negative
boxes, and precision in drawing box boundaries. Online crowdsourcing reduced
the number of required workers per image by a factor of 1.7 without increasing
error (comparing prob-worker-online to prob-worker), while adding computer vi-
sion (method prob-worker-online-.005) reduced annotation by an additional29%.
Examining Fig. 3.5b, we see that computer vision allowed many images to be con-
fidently annotated with a single human worker. The naive computer vision method
prob-worker-cvnaive-online performed as well as our more complicated method.

Computer Vision Worker 1 Worker 2 Worker 3 Worker 4 Worker 5 Worker 6 Prediction Ground Truth
**Computer Vision Worker 1** (^) **Prediction Ground Truth Computer Vision Worker 1 Worker 2 Prediction Ground Truth**
Figure 3.8: Bounding box annotation sequences. The top left annotation sequence
highlights a good case where only the computer vision system and one human are
needed to finish the image. The top right annotation sequence highlights the average
case where two workers and the computer vision system are needed to finish the
image. The bottom row highlights a difficult annotation sequence where workers
disagree on the number of instances, forcing the image to remain unfinished longer
than usual.
**Part Annotation**
To evaluate part keypoint annotation, we used the 1000 image subset of the NABirds
dataset (Van Horn et al., 2015), for which a detailed analysis comparing experts to
MTurkers was performed in (Van Horn et al., 2015). This subset contained 10
MTurker labels per image of 11 semantic keypoint locations as well as expert
part labels. Although our algorithm processed each part independently, we report
error averaged over all 11 parts, using the loss defined in Section 3.7. We did not
implement a computer vision algorithm for parts; however, a variant of our algorithm
(prob-worker-online) was used by the creators of the dataset to collect its published
part annotations (11 parts on 55,000 images), using only 2.3 worker annotations per
part on average.


Simulated results on the 1000 image subset are shown in Fig. 3.5c. We see that the
worker skill model (prob-worker) led to10%reduction in error over the majority-
vote baseline, and online model cut annotation time roughly in half, with most parts
finishing with 2 worker clicks (Fig.3.4b)

Worker 1 Worker 2 Predicted Ground Truth
Worker 1 Worker 2 Predicted Ground Truth
Worker 1 Worker 2 Predicted Ground Truth
Worker 3 Worker 4 Worker 5 Worker 6
Figure 3.9: Part annotation sequences. The two sequences on the top row are the
common situation where the responses from two workers correlate well and are
enough for the system to mark the images as finished. The annotation sequence on
the bottom row highlights a difficult situation where workers toggle back and forth
on the visibility of the wings, forcing the image to remain unfinished for longer than
usual. This toggling behavior can be attributed to task ambiguity and/or insufficient
instructions.

**Worker Skills**
One bonus of our algorithm is that it predicts the skill of each worker according
to a small number of semantically interpretable features. These could be used for
blocking spammers, giving bonuses to good workers, or debugging ambiguities in
the annotation task.

**Discussion and Failure Cases**
All crowdsourcing methods resulted in some degree of error when crowd labels
converged to something different than expert labels. The most common reason was
ambiguous images. For example, most MTurkers incorrectly thought scorpion spi-
ders (a type of spider resembling scorpions) were actual scorpions. Visibility of a
part annotation can become ambiguous as an object rotates from frontal to rear view.
However, all variants of our method (with and without computer vision, with and
without online crowdsourcing) resulted in higher quality annotations than majority
vote (which is commonly used for many computer vision datasets). Improvement
in annotation quality came primarily from modeling worker skill. Online crowd-
sourcing can increase annotation errors; however, it does so with an interpretable
parameter for trading off annotation time and error. Computer vision also reduces
annotation time, with greater gains coming as dataset size increases. However, we


0.00.0 0.2 Prob Correct Given Present0.4 0.6 0.8 1.0
0.2
0.4
0.6
0.8
1.0
Prob Correct Given Not Present
Worker Skill
(a) Worker Skills for Binary Classi-
fication
0.00.2Location Sigma0.4 0.60.8 1.00.00.2Prob Mistake0.40.6
0.81.0
0.700.750.80Prob Vis Correct
0.850.90
0.951.00
Worker Skill
(b) Worker Skills for Part Annota-
tions
0.0Prob False Positive0.2 0.40.6 0.8
1.00.00.2Boundary Sigma
0.40.6
0.81.0
0.000.05Prob False Negative
0.05
0.10
0.15
0.20
Worker Skill
(c) Worker Skills for Bounding Box
Annotations
Figure 3.10: **Inferred Worker Skills:** These plots easily surface worker capabilities
and can inform researchers on the difficulty of their task and help debug ambiguous
tasks. **(a)** On the scorpion dataset (binary classification), most workers on this
task can tell fairly accurately when a scorpion is not present; however, there is a
large spread in worker miss rate, possibly due to less careful workers in picking
out smaller or more ambiguous objects. **(b)** On the bird dataset (part annotation),
workers tend to be highly accurate when annotating parts. Visibility issues are often
due to left/right part mistakes or when a part is partially (self) occluded. **(c)** On
the pedestrian dataset (bounding box annotation), workers tend to not hallucinate
people (low false positive), however there is a chance that they miss a (probably
small) instance.


note that in some cases, adding computer vision in the loop might be inappropriate
for research datasets due to bias toward certain algorithms. We allow it to be toggled
on or off in our source code.

### 3.10 Conclusion

In this work, we introduced crowdsourcing algorithms and online tools for collecting
binary, part, and bounding box annotations. We showed that each component of
the system–a worker skill / image difficulty model, an online stoppage criterion for
collecting a variable number of annotations per image, and integration of computer
vision in the loop– led to significant reductions in annotation time and/or annotation
error for each type of annotation. In future work, we plan to extend the approach
to other types of annotation, like segmentation and video, use inferred worker skill
parameters to block spammers, choose which worker should annotate an image,
and incorporate active learning criteria to choose which images to annotate next or
choose between different types of user interfaces.

**Acknowledgments**
This paper was inspired by work from and earlier collaborations with Peter Welinder
and Boris Babenko. Much thanks to Pall Gunnarsson for helping to develop an early
version of the method. Thank you to David Hall for supplying data for bounding
box experiments. This work was supported by a Google Focused Research Award
and Office of Naval Research MURI N000141010933.

**References**

Andriluka, Mykhaylo et al. (2014). “2D Human Pose Estimation: New Benchmark
and State of the Art Analysis”. In: _IEEE Conference on Computer Vision and
Pattern Recognition (CVPR)_.

Biswas, Arijit and Devi Parikh (2013). “Simultaneous active learning of classifiers
& attributes via relative feedback”. In: _Proceedings of the IEEE Conference on
Computer Vision and Pattern Recognition_ , pp. 644–651.

Branson, Steve et al. (2010). “Visual recognition with humans in the loop”. In:
_European Conference on Computer Vision_. Springer, pp. 438–451.

Carpenter, Bob (2008). “Multilevel bayesian models of categorical data annotation”.
In: _Unpublished manuscript_.

Chilton, Lydia B et al. (2013). “Cascade: Crowdsourcing taxonomy creation”. In:
_Proceedings of the SIGCHI Conference on Human Factors in Computing Systems_.
ACM, pp. 1999–2008.


Dalvi, Nilesh et al. (2013). “Aggregating crowdsourced binary ratings”. In: _Proceed-
ings of the 22nd international conference on World Wide Web_. ACM, pp. 285–
294.

Dawid, Alexander Philip and Allan M Skene (1979). “Maximum likelihood esti-
mation of observer error-rates using the EM algorithm”. In: _Applied statistics_ ,
pp. 20–28.

Deng, Jia, Wei Dong, et al. (2009). “Imagenet: A large-scale hierarchical image
database”. In: _Computer Vision and Pattern Recognition, 2009. CVPR 2009.
IEEE Conference on_. IEEE, pp. 248–255.

Deng, Jia, Jonathan Krause, and Li Fei-Fei (2013). “Fine-grained crowdsourcing for
fine-grained recognition”. In: _Proceedings of the IEEE Conference on Computer
Vision and Pattern Recognition_ , pp. 580–587.

Deng, Jia, Olga Russakovsky, et al. (2014). “Scalable multi-label annotation”. In:
_Proceedings of the SIGCHI Conference on Human Factors in Computing Systems_.
ACM, pp. 3099–3102.

Dutt Jain, Suyog and Kristen Grauman (2013). “Predicting sufficient annotation
strength for interactive foreground segmentation”. In: _Proceedings of the IEEE
International Conference on Computer Vision_ , pp. 1313–1320.

Erlenkotter, Donald (1978). “A dual-based procedure for uncapacitated facility lo-
cation”. In: _Operations Research_ 26.6, pp. 992–1009.

Gao, Chao and Dengyong Zhou (2013). “Minimax optimal convergence rates for esti-
mating ground truth from crowdsourced labels”. In: _arXiv preprint arXiv:1310.5764_.

Ghosh, Arpita, Satyen Kale, and Preston McAfee (2011). “Who moderates the
moderators?: crowdsourcing abuse detection in user-generated content”. In: _Pro-
ceedings of the 12th ACM conference on Electronic commerce_. ACM, pp. 167–
176.

Griffin, Gregory, Alex Holub, and Pietro Perona (2007). “Caltech-256 object cate-
gory dataset”. In:

Gurari, Danna et al. (2015). “How to collect segmentations for biomedical images?
A benchmark evaluating the performance of experts, crowdsourced non-experts,
and algorithms”. In: _2015 IEEE Winter Conference on Applications of Computer
Vision_. IEEE, pp. 1169–1176.

Hall, David and Pietro Perona (2015). “Fine-grained classification of pedestrians in
video: Benchmark and state of the art”. In: _Proceedings of the IEEE Conference
on Computer Vision and Pattern Recognition_ , pp. 5482–5491.

He, Kaiming et al. (2015). “Deep residual learning for image recognition”. In: _arXiv
preprint arXiv:1512.03385_.


Hua, Gang et al. (2013). “Collaborative active learning of a kernel machine ensem-
ble for recognition”. In: _Proceedings of the IEEE International Conference on
Computer Vision_ , pp. 1209–1216.

Jain, Suyog Dutt and Kristen Grauman (2016). “Active Image Segmentation Propa-
gation”. In: CVPR.

Jin, Rong and Zoubin Ghahramani (2002). “Learning with multiple labels”. In:
_Advances in neural information processing systems_ , pp. 897–904.

Karger, David R, Sewoong Oh, and Devavrat Shah (2011). “Iterative learning for
reliable crowdsourcing systems”. In: _Advances in neural information processing
systems_ , pp. 1953–1961.

- (2013). “Efficient crowdsourcing for multi-class labeling”. In: _ACM SIGMETRICS_
    _Performance Evaluation Review_ 41.1, pp. 81–92.

Kazemzadeh, Sahar et al. (2014). “ReferItGame: Referring to Objects in Photographs
of Natural Scenes.” In: _EMNLP_ , pp. 787–798.

Khodabandeh, Mehran et al. (2015). “Discovering human interactions in videos
with limited data labeling”. In: _Proceedings of the IEEE Conference on Computer
Vision and Pattern Recognition Workshops_ , pp. 9–18.

Kovashka, Adriana, Sudheendra Vijayanarasimhan, and Kristen Grauman (2011).
“Actively selecting annotations among objects and attributes”. In: _2011 Interna-
tional Conference on Computer Vision_. IEEE, pp. 1403–1410.

Kovashka, A. et al. (2016). “Crowdsourcing in Computer Vision”. In: _ArXiv e-
prints_. arXiv:1611.02145 [cs.CV].url:%7Bhttps://arxiv.org/abs/
1611.02145%7D.

Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E Hinton (2012). “ImageNet Clas-
sification with Deep Convolutional Neural Networks.” In: _NIPS_.

Lad, Shrenik and Devi Parikh (2014). “Interactively guiding semi-supervised clus-
tering via attribute-based explanations”. In: _European Conference on Computer
Vision_. Springer, pp. 333–349.

Larlus, Diane et al. (2014). “Generating Gold Questions for Difficult Visual Recog-
nition Tasks”. In:

Li, Hongwei and Bin Yu (2014). “Error rate bounds and iterative weighted majority
voting for crowdsourcing”. In: _arXiv preprint arXiv:1411.4086_.

Lin, Tsung-Yi et al. (2014). “Microsoft COCO: Common objects in context”. In:
_ECCV_.

Liu, Qiang, Jian Peng, and Alexander T Ihler (2012). “Variational inference for
crowdsourcing”. In: _Advances in Neural Information Processing Systems_ , pp. 692–
700.


Long, Chengjiang and Gang Hua (2015). “Multi-class multi-annotator active learn-
ing with robust Gaussian Process for visual recognition”. In: _Proceedings of the
IEEE International Conference on Computer Vision_ , pp. 2839–2847.

Long, Chengjiang, Gang Hua, and Ashish Kapoor (2013). “Active visual recogni-
tion with expertise estimation in crowdsourcing”. In: _Proceedings of the IEEE
International Conference on Computer Vision_ , pp. 3000–3007.

Ok, Jungseul et al. (2016). “Optimality of Belief Propagation for Crowdsourced
Classification”. In: _arXiv preprint arXiv:1602.03619_.

Parkash, Amar and Devi Parikh (2012). “Attributes for classifier feedback”. In:
_European Conference on Computer Vision_. Springer, pp. 354–368.

Patterson, Genevieve, Grant Van Horn2 Serge Belongie, and Pietro Perona2 James
Hays (2015). “Tropel: Crowdsourcing Detectors with Minimal Training”. In:
_Third AAAI Conference on Human Computation and Crowdsourcing_.

Platt, John et al. (1999). “Probabilistic outputs for support vector machines and
comparisons to regularized likelihood methods”. In: _Advances in large margin
classifiers_ 10.3, pp. 61–74.

Raykar, Vikas C et al. (2010). “Learning from crowds”. In: _Journal of Machine
Learning Research_ 11.Apr, pp. 1297–1322.

Rubinstein, Michael, Ce Liu, and William T Freeman (2012). “Annotation propa-
gation in large image databases via dense image correspondence”. In: _European
Conference on Computer Vision_. Springer, pp. 85–99.

Russakovsky, Olga, Li-Jia Li, and Li Fei-Fei (2015). “Best of both worlds: human-
machine collaboration for object annotation”. In: _Proceedings of the IEEE Con-
ference on Computer Vision and Pattern Recognition_ , pp. 2121–2131.

Russell, Bryan C et al. (2008). “LabelMe: a database and web-based tool for image
annotation”. In: _International journal of computer vision_ 77.1-3, pp. 157–173.

Shah, Nihar B, Sivaraman Balakrishnan, and Martin J Wainwright (2016). “A
permutation-based model for crowd labeling: Optimal estimation and robust-
ness”. In: _arXiv preprint arXiv:1606.09632_.

Shah, Nihar Bhadresh and Denny Zhou (2015). “Double or nothing: Multiplicative
incentive mechanisms for crowdsourcing”. In: _Advances in Neural Information
Processing Systems_ , pp. 1–9.

Shankar Nagaraja, Naveen, Frank R Schmidt, and Thomas Brox (2015). “Video
Segmentation with Just a Few Strokes”. In: _Proceedings of the IEEE International
Conference on Computer Vision_ , pp. 3235–3243.

Sheng, Victor S, Foster Provost, and Panagiotis G Ipeirotis (2008). “Get another
label? improving data quality and data mining using multiple, noisy labelers”. In:
_Proceedings of the 14th ACM SIGKDD international conference on Knowledge
discovery and data mining_. ACM, pp. 614–622.


Siddiquie, Behjat and Abhinav Gupta (2010). “Beyond active noun tagging: Model-
ing contextual interactions for multi-class active learning”. In: _Computer Vision
and Pattern Recognition (CVPR), 2010 IEEE Conference on_. IEEE, pp. 2979–
2986.

Smyth, Padhraic et al. (1995). “Inferring ground truth from subjective labelling of
venus images”. In:

Snow, Rion et al. (2008). “Cheap and fast—but is it good?: evaluating non-expert
annotations for natural language tasks”. In: _Proceedings of the conference on
empirical methods in natural language processing_. Association for Computational
Linguistics, pp. 254–263.

Sorokin, Alexander and David Forsyth (2008). “Utility data annotation with amazon
mechanical turk”. In: _Urbana_ 51.61, p. 820.

Su, Hao, Jia Deng, and Li Fei-Fei (2012). “Crowdsourcing annotations for visual ob-
ject detection”. In: _Workshops at the Twenty-Sixth AAAI Conference on Artificial
Intelligence_.

Szegedy, Christian, Wei Liu, et al. (2015). “Going deeper with convolutions”. In:
_Proceedings of the IEEE Conference on Computer Vision and Pattern Recogni-
tion_ , pp. 1–9.

Szegedy, Christian, Scott Reed, et al. (2014). “Scalable, high-quality object detec-
tion”. In: _arXiv preprint arXiv:1412.1441_.

Tian, Tian and Jun Zhu (2015). “Max-margin majority voting for learning from
crowds”. In: _Advances in Neural Information Processing Systems_ , pp. 1621–
1629.

Van Horn, Grant et al. (2015). “Building a bird recognition app and large scale
dataset with citizen scientists: The fine print in fine-grained dataset collection”.
In: _Proceedings of the IEEE Conference on Computer Vision and Pattern Recog-
nition_ , pp. 595–604.doi:10.1109/CVPR.2015.7298658.

Vijayanarasimhan, Sudheendra and Kristen Grauman (2009a). “Multi-level active
prediction of useful image annotations for recognition”. In: _Advances in Neural
Information Processing Systems_ , pp. 1705–1712.

- (2009b). “What’s it going to cost you?: Predicting effort vs. informativeness for
    multi-label image annotations”. In: _Computer Vision and Pattern Recognition,_
    _2009. CVPR 2009. IEEE Conference on_. IEEE, pp. 2262–2269.

Vittayakorn, Sirion and James Hays (2011). “Quality Assessment for Crowdsourced
Object Annotations.” In: _BMVC_ , pp. 1–11.

Von Ahn, Luis and Laura Dabbish (2004). “Labeling images with a computer
game”. In: _Proceedings of the SIGCHI conference on Human factors in computing
systems_. ACM, pp. 319–326.


Von Ahn, Luis and Laura Dabbish (2005). “ESP: Labeling Images with a Com-
puter Game.” In: _AAAI spring symposium: Knowledge collection from volunteer
contributors_. Vol. 2.

Vondrick, Carl, Donald Patterson, and Deva Ramanan (2013). “Efficiently scaling
up crowdsourced video annotation”. In: _International Journal of Computer Vision_
101.1, pp. 184–204.

Wah, Catherine, Steve Branson, Pietro Perona, et al. (2011). “Multiclass recognition
and part localization with humans in the loop”. In: _2011 International Conference
on Computer Vision_. IEEE, pp. 2524–2531.

Wah, Catherine, Steve Branson, Peter Welinder, et al. (2011). “The caltech-ucsd
birds-200-2011 dataset”. In:

Wah, Catherine, Grant Van Horn, et al. (2014). “Similarity comparisons for inter-
active fine-grained categorization”. In: _Proceedings of the IEEE Conference on
Computer Vision and Pattern Recognition_ , pp. 859–866.

Wang, Jing, Panagiotis G Ipeirotis, and Foster Provost (2013). “Quality-based pricing
for crowdsourced workers”. In:

Welinder, Peter, Steve Branson, et al. (2010). “The multidimensional wisdom of
crowds”. In: _Advances in neural information processing systems_ , pp. 2424–2432.

Welinder, Peter and Pietro Perona (2010). “Online crowdsourcing: rating annotators
and obtaining cost-effective labels”. In:

Whitehill, Jacob et al. (2009). “Whose vote should count more: Optimal integration
of labels from labelers of unknown expertise”. In: _Advances in neural information
processing systems_ , pp. 2035–2043.

Wilber, Michael J, Iljung S Kwak, and Serge J Belongie (2014). “Cost-effective
hits for relative similarity comparisons”. In: _Second AAAI Conference on Human
Computation and Crowdsourcing_.

Yao, Angela et al. (2012). “Interactive object detection”. In: _Computer Vision and
Pattern Recognition (CVPR), 2012 IEEE Conference on_. IEEE, pp. 3242–3249.

Zhang, Chicheng and Kamalika Chaudhuri (2015). “Active learning from weak
and strong labelers”. In: _Advances in Neural Information Processing Systems_ ,
pp. 703–711.

Zhang, Yuchen et al. (2014). “Spectral methods meet EM: A provably optimal
algorithm for crowdsourcing”. In: _Advances in neural information processing
systems_ , pp. 1260–1268.

Zhou, Dengyong et al. (2015). “Regularized minimax conditional entropy for crowd-
sourcing”. In: _arXiv preprint arXiv:1503.07240_.

Zhou, Denny et al. (2012). “Learning from the wisdom of crowds by minimax
entropy”. In: _Advances in Neural Information Processing Systems_ , pp. 2195–
2203.


