# Chapter 4: Lean Multiclass Crowdsourcing

Van Horn, Grant et al. (2018). “Lean Multiclass Crowdsourcing”. In: _Proceedings
of the IEEE Conference on Computer Vision and Pattern Recognition_. Salt Lake
City, UT.doi:10.1109/cvpr.2018.00287.

### 4.1 Abstract

We introduce a method for efficiently crowdsourcing multiclass annotations in chal-
lenging, real world image datasets. Our method is designed to minimize the number
of human annotations that are necessary to achieve a desired level of confidence on
class labels. It is based on combining models of worker behavior with computer
vision. Our method is general: it can handle a large number of classes, worker labels
that come from a taxonomy rather than a flat list, and can model the dependence of
labels when workers can see a history of previous annotations. Our method may
be used as a drop-in replacement for the majority vote algorithms used in online
crowdsourcing services that aggregate multiple human annotations into a final con-
solidated label. In experiments conducted on two real-life applications, we find that
our method can reduce the number of required annotations by as much as a factor
of 5.4 and can reduce the residual annotation error by up to 90% when compared
with majority voting. Furthermore, the online risk estimates of the models may be
used to sort the annotated collection and minimize subsequent expert review effort.

### 4.2 Introduction

Multiclass crowdsourcing is emerging as an important technique in science and
industry. For example, a growing number of websites support sharing observations
(photographs) of specimens from the natural world and facilitate collaborative,
community-driven identification of those observations. Websites such as iNatu-
ralist, eBird, Mushroom Observer, HerpMapper, and LepSnap accumulate large
collections of images and identifications, often using majority voting to produce
the final species label. Ultimately, this information is aggregated into datasets
(e.g. GBIF (Ueda, 2017)) that enable global biodiversity studies (Sullivan et al.,
2014). Thus, the label accuracy of these datasets can have a direct impact on sci-
ence, conservation, and policy. Thanks to the recent dramatic improvements in our


SE
t
Snowy Egret
Great Egret
GE
©mikewitkowski
SE
t
Snowy Egret
Great Egret
GE
©mikewitkowski
Figure 4.1: **iNaturalist Community Identification.** A user uploads imagexi(top-
left) with an initial species predictionzi 1 =Great Egret (GE), one out of 1. 5 k North
American bird species. Later, two additional users (potentially alerted that a GE
has been spotted) come along and, after inspecting the image _and_ the previous
identifications, contribute their subjective identifications of the bird specieszi 2 =
GE andzi 3 =GE, agreeing with the uploader. Finally, a fourth user provides a
different identificationzi 4 =Snowy Egret (SE). In the plot below the images, two
models (red, green) integrate the information differently, with theyaxis representing
likelihood of SE vs. GE. Majority voting (yellow arrow) simply tallies the vote, and
GE is the chosen answer after four votes. Our model (blue arrow) continuously
analyzes the users’ skills across _other_ observations and is therefore capable of
updating the likelihood of the predicted label much more frequently. Knowing that
the fourth user is highly skilled on these taxa, our model overrides previous users
and predicts SE. The underlying ground truth answer is indeed SE. In this work, we
design and compare several models that estimate user skill and use it to weigh votes
appropriately. (View on iNat: https://www.inaturalist.org/observations/4599411)

field (Krizhevsky, Sutskever, and Hinton, 2012; He et al., 2015; Szegedy et al.,
2016; Huang et al., 2017), observations collected by these websites can be used
to train classification services (e.g. see merlin.allaboutbirds.org and inaturalist.org),
helping novices label their observations. The result is an even larger collection of
observations, but with potentially noisier labels, as the number of people taking
photos and submitting observations far outpaces the speed at which experts can
verify them. The benefits of a simple algorithm like majority vote are lost when the
skill of the people contributing labels is uncertain. Thus, there is need for improved
methods to integrate multiple identifications into a final label.

Figure 4.1 shows a real example of a user’s observation on iNaturalist, a sequence of
identifications from the community, and how the current species label is computed


using majority voting. The structure of these interactions present three challenges
that have not been tackled by prior work on combining multiclass annotations (Denny
Zhou et al., 2012; Karger, Oh, and D. Shah, 2013; Vempaty, L. R. Varshney, and
P. K. Varshney, 2014; Dengyong Zhou et al., 2014; Y. Zhang et al., 2014; Tian and
Zhu, 2015; J. Zhang et al., 2016): (1) iNaturalist has a tree structured taxonomy of
labels rather than a flat list, allowing users to provide labels at varying depths of
the taxonomy depending on their confidence; (2) identifiers get to see the history of
previous identifications for an observation, so their identification is _not_ independent
of previous identifiers; and (3) the number of species under consideration is huge,
currently at∼130,000 but potentially reaching 8M (Mora et al., 2011).

We propose a new method for aggregating multiple multiclass labels. Our method is
based on models of worker behavior and can replace majority vote in websites like
iNaturalist, and in more traditional data labeling services (e.g. Amazon Mechanical
Turk). We show that our models are more accurate than majority voting (reducing
error by 90% on data from iNaturalist), and when combined with a computer vision
system, can drastically reduce the number of labels required per image (e.g. by
a factor of 5. 4 on crowdsourced data). Our main contribution is a method for
_multiclass annotation_ tasks that (1) can be used in online crowdsourcing, (2) can
handle large numbers of classes, (3) can handle a taxonomy of labels allowing
workers to respond at coarser levels than leaf nodes, and (4) can handle mutually
dependent worker labels.

### 4.3 Related Work

Kovashka et al. (Kovashka et al., 2016) provide a thorough review of crowdsourcing
techniques for computer vision. The Dawid-Skene (DS) model (Dawid and Skene,
1979) is the standard probabilistic model for multiclass label inference from multiple
annotations. That model assumes each worker has a latent confusion matrix that
captures the probability of annotating a class correctly (the diagonal entries) and
the probability of confusing two classes (the off diagonal entries). The DS model
iteratively infers the reliability of each worker and updates the belief of the true
labels, using Expectation-Maximization as the inference algorithm. Alternate infer-
ence algorithms for the DS model are based on spectral methods (Ghosh, Kale, and
McAfee, 2011; Dalvi et al., 2013; Karger, Oh, and D. Shah, 2011; Karger, Oh, and
D. Shah, 2013; Karger, Oh, and D. Shah, 2014; Y. Zhang et al., 2014), belief propa-
gation (Liu, Peng, and Ihler, 2012; Ok et al., 2016), expectation maximization (Liu,
Peng, and Ihler, 2012; Y. Zhang et al., 2014), maximum entropy (Denny Zhou et al.,


2012; Dengyong Zhou et al., 2014), weighted majority voting (Littlestone and War-
muth, 1994; Li, Yu, and Dengyong Zhou, 2013), and max-margin (Tian and Zhu,
2015). Alternatives to the DS model have also been proposed (Smyth et al., 1995;
Jin and Ghahramani, 2002; Whitehill et al., 2009; Welinder et al., 2010; Raykar
et al., 2010; Tang and Lease, 2011; Kamar, Hacker, and Horvitz, 2012; J. Zhang
et al., 2016; Branson, Van Horn, and Perona, 2017; Chen et al., 2017). Further work
based on active learning tackles noisy labelers (Long, Hua, and Kapoor, 2013) and
task allocation to minimize the monetary cost of dataset construction (Karger, Oh,
and D. Shah, 2013; Karger, Oh, and D. Shah, 2014; N. B. Shah and Denny Zhou,
2015).

Multiclass tasks, as opposed to binary tasks, are explored by (Denny Zhou et al.,
2012; Karger, Oh, and D. Shah, 2013; Vempaty, L. R. Varshney, and P. K. Varshney,
2014; Dengyong Zhou et al., 2014; Y. Zhang et al., 2014; Tian and Zhu, 2015; J.
Zhang et al., 2016; Chen et al., 2017). Zhou et al. use entropy maximization to
model both worker confusions and task difficulties for multiclass Denny Zhou et al.,
2012 and ordinal (Dengyong Zhou et al., 2014) data. Similarly, Chen et al. (Chen
et al., 2017) use max-margin techniques to further improve results for ordinal tasks.
Karger et al. (Karger, Oh, and D. Shah, 2013) use an iterative algorithm by converting
k-class tasks intok− 1 binary tasks but make assumptions on the number of items
and workers. Vempaty et al. (Vempaty, L. R. Varshney, and P. K. Varshney, 2014)
also convertk-class tasks into binary tasks, but take a coding theoretic approach
to estimate labels. Zhang et al. (Y. Zhang et al., 2014) use spectral methods to
initialize the EM inference algorithm of the Dawid-Skene model, while Tian et
al. (Tian and Zhu, 2015) fuse a max-margin estimator and the Dawid-Skene model.
Zhang et al. (J. Zhang et al., 2016) create probabilistic features for each item and use
a clustering algorithm to assign them their final labels, however they do not produce
an estimate of worker skill. All of the previous approaches assume that annotations
are independent. We differentiate our work by handling both independent _and_
dependent annotations collected by sites like iNaturalist. Furthermore, we explore
the challenges of “large-scale” multiclass task modeling where the number of classes
is nearly 10×larger than the prior art has explored. Our work also handles taxonomic
modeling of the classes and non-leaf node worker annotations. See Table 4.2 for a
performance comparison of our model to prior art.

Final label quality between independent and dependent crowdsourcing tasks is
studied by Little et al. (Little et al., 2010), but without modeling workers. The


work of Branson et al. (Branson, Van Horn, and Perona, 2017) is the closest to ours,
as we adapt their framework to multiclass annotation, which they did not investigate.
Furthermore, we explore taxonomic multiclass annotations to reduce the number of
parameters. Additionally, we develop models that do not depend on the assumption
that worker annotations are independent, and we are thus able to handle mutually
dependent annotations where each worker can see previous labels.

### 4.4 Multiclass Online Crowdsourcing

Given a set of worker annotationsZfor a dataset of imagesX, the probabilistic
framework of Branson et al. (Branson, Van Horn, and Perona, 2017) jointly models
worker skillW, image difficultly D, ground truth labelsY, and computer vision
system parameters θ. A tiered prior system is used to make the system more
robust by regularizing the per worker skill and image difficulty priors. Alternating
maximization is used for parameter estimation. The Bayesian riskR(y ̄i)(see Eq.1
from (Branson, Van Horn, and Perona, 2017)) can be computed for each predicted
label, providing an intuitive online stopping criteria (i.e. the model can “retire”
images as soon as their risk is below a thresholdτ). In this work, we extend this
framework by implementing multiple models of worker skill for the task of multiclass
annotation for independent and dependent worker labels. For our experiments, we
removed the image difficulty part of the framework and focused solely on modeling
workers and their labels. Section 4.4 constructs worker skill models when the labels
Zare independent, and Section 4.4 constructs worker skill models when the labels
Zare dependent.

**Independent Labels**
Letxibe theith image, which contains an object with class labelyi ∈ { 1 ,.. .,C}
(e.g., species). Suppose a set of workersWiindependently specify their guess at
the class of imagei, such that for each j ∈ Wi,zi jis worker j’s guess atyi. In
this situation, identifiers from Figure 4.1 would not get to observe preceding users’
guesses. Letwjbe some set of parameters encoding workerj’s skill at predicting
classes. In this notation, if the classyiis unknown, we can estimate the probability
of each possible class given the setZi={zi j}j∈Wiof worker guesses:

p(yi|Zi)=
p(yi)Œj∈Wip(zi j|yi,wj)
ÕC
y= 1 p(y)
Œ

j∈Wip(zi j|y,wj)
(4.1)

wherep(yi)is the prior class probability andp(zi j|yi,wj)is a model of imperfect
human guesses. The following sections discuss possible models forp(zi j|yi,wj),


**Name**

**Interpretation**

**Model**

**Expression**

**# Params**

**# ParamsForBirds**

Flat Single Bino-mial
Probability ofbeing correct isthe same for allspecies
z=
yis binomial with
the same parameters re-gardless of
y
p(z
|y)
{ =
m
if
z=
y
(^1
−m
)p(
z)
otherwise
1
1
Flat Per Class Bi-nomial
Probability ofbeing correctfor each speciesseparately
For each value
y
=
c,
z=
yis binomial
p(z
|y)
{ =
M(
y)
if
z=
y
(^1
−M
(y))
p(z
)
otherwise
C
1,572
Flat Per ClassMultinomial
Confusionprobability overeach pair ofspecies
For each value
y=
c,
z
is multinomial
p(z
|y)
=M
(y,
z)
(^2) C
2,471,184
**Taxonomic SingleBinomial**
Probability ofbeing correct isthe same foreach species ina genus
lz
=
ly|
l−z
1
=
l−y
1
is binomial with thesame parameters re-gardless of
ly
p(z
|y)
Œ=
pl
l(z
l|y
)
p(z
l|y
l)
{ =
my
l−^1
if
lz
=
ly
(^1
−m
l−y
)p 1
l(z
)
otherwise
|N
|−
C
383
**Taxonomic PerClass Binomial**
Probability ofbeing correctfor each speciesseparately
For each value
ly
=
c,
lz
=
ly|
l−z
1 =
l−y
1 is
binomial
p(z
|y)
Œ=
pl
l(z
l|y
)
p(z
l|y
l)
=
{M
l−y
(y 1
l)
if
lz
=
ly
(^1
−M
l−y
1 (y
l))
p(z
)
otherwise
|N
|
1955
**Taxonomic PerClass Multinomial**
Confusionprobability foreach pair ofspecies in agenus
For each value
ly
=
c,
lz|
l−z
1 =
l−y
1 is multi-
nomial
p(z
|y)
Œ=
pl
l(z
l|y
)
p(z
l|y
l)
=M
l−y
1 (y
l,z
l)
Õ n∈N
|children
(n)|
2
22,472
Table 4.1: Different options for modeling worker skill given a taxonomy of classes.

N

is the set of nodes in the taxonomic tree,
C

is the
number of leaf nodes (i.e. class labels). The last column shows the number of resulting parameters when modeling the 1,572 species ofNorth American birds and their taxonomy from the iNaturalist database,
for a single worker
. Multinomial models have significantly more

parameters but can model commonly confused classes. Taxonomic methods have the benefit of supporting non-species-level humanresponses, modeling skill at certain taxa, and reducing the number of parameters for multinomial models.

which are also summarized in Table 4.1.

**Flat Models**

**Flat Single Binomial:** One simple way to model worker skills is with a single
parameter that captures the worker’s probability of providing a correct answer,
regardless of the class label. We assume that the probability of a worker being
correctmjfollows a Bernoulli distribution, with other responses having probability
proportional to class priors:

p(zi j|yi,wj)=






mj ifzi j=yi
( 1 −mj)p(zi j) otherwise
(4.2)

To prevent over fitting in low data situations, we place a beta prior Beta(nβpc,nβ( 1 −
pc))onmj, wherenβis the strength of the prior.pcrepresents the probability of any
worker providing a correct label, and is estimated by pooling all worker annotations
together. We also place a beta prior Beta(nβp,nβ( 1 −p))onpc, withpacting as our
prior belief on worker performance. Estimating the worker skills is done by counting
the number of times their response agrees with the predicted label, weighted by the
prior strength:

mj=
nβpc+Õi∈Ij 1 [zi j=y ̄i,|Wi|> 1 ]− 1
nβ+Õi∈Ij 1 [y ̄i,|Wi|> 1 ]− 2 (4.3)
where 1 [·]is the indicator function,Ijare the images labeled by workerj, andy ̄iis
our current label prediction for imagei. The pooled priorpcis estimated similarly.

**Flat Per Class Binomial:** Rather than learning a single skill parametermacross
all classes, we can learn a separate binomial model for each value ofy, resulting in
a skill vectorMjfor each worker:

p(zi j|yi,wj)=






Mj(yi) ifzi j=yi
( 1 −Mj(yi))p(zi j) otherwise
(4.4)

Similar to the single binomial model, we employ a tiered prior system by adding a
per class beta prior Beta(nβpy,nβ( 1 −py))onMj(y). We place a generic beta prior
Beta(nβp,nβ( 1 −p))onpyto encode our prior belief that a worker is correct on any
class. Estimating the worker skill parametersMj(y)and the pooled priorspyfor
classyis done in the same way as the single binomial model.


**Flat Per Class Multinomial:** A more sophisticated model ofp(zi j|yi,wj)could
assumewjencodes aC×Cconfusion matrix **M** j, where an entry **M** j(m,n)de-
notes personj’s probability of predicting classnwhen the true class ism. Here,
p(zi j|yi,wj)= **M** j(yi,zi j); the model is assumingp(zi j|yi=c,wj)is a multinomial
distribution with parametersμcj =[ **M** j(c, 1 ), ..., **M** j(c,C)]for each value ofc. We
will place Dirichlet priors Dir(nβαc)onμcj, wherenβis the strength of the prior,
andαcis estimated by pooling across all workers. We will also place a Dirichlet
prior Dir(nβα)onαc, withαacting as a global hyper-parameter that provides the
likelihood of any worker labeling a class correctly. Because the Dirichlet distribu-
tion is the conjugate prior of the multinomial distribution, the computation of each
entrykfrom 1.. .Cin the skill vectorμcjfor a single workerjand each classcis
done by counting agreements:

μcj,k=
nβαck+Õi∈Ij 1 [zi j=k,y ̄i=k,|Wi|> 1 ]− 1
nβα 0 c+Õi∈Ij 1 [y ̄i=k,|Wi|> 1 ]−C (4.5)
Whereαc 0 =

Õ

kαck. The pooled worker parametersαcare estimated in a similar
way.

**Taxonomic Models**

Multinomial models are useful because they model commonly confused classes,
however they have far more parameters than the binomial models. These models
quickly become intractable as the total number of classesCgets large. For example,
if there are 104 classes, we would be attempting to estimate a matrix **M** j with
108 entries for each workerj. This is statistically and computationally intractable.
However, when the number of classes gets large, there often exists a taxonomy used
to organize them (e.g. the Linnaean taxonomy for biological classification). We can
use this taxonomy to reduce the number of parameters in a multinomial model.

**Taxonomic Per Class Multinomial:** We will assume a taxonomy of classes that
isLlevels deep and associate a confusion matrix with each node in the taxonomy
(e.g., if we know the genus of an observation from iNaturalist, assume each worker
has a confusion matrix among species within that genus). For the taxonomic model,
letyil denote the node in the taxonomy at levell that class yi belongs to, such
thatyi^0 is the root node andyiLis the leaf node (i.e., species label). Similarly, let
zli j denote the node in the taxonomy at levell that classzi j belongs to. In this

model,p(zli j|yli,wj,yil−^1 =zli j−^1 )= **M** y

li− 1
j (yli,zi jl), where M
yli−^1
j is a confusion matrix
associated with nodeyli−^1 in the taxonomy; the assumption is that for each value of


yli,zi jl is multinomial with a vector M y
li− 1
j (yil,:)of parameters of size equal to the
number of child nodes. The termyil−^1 =zli j−^1 denotes the condition that the parent
node classification is known. Suppose, however, that workerjis wrong about both
the species and genus. We must also modelp(zli j|yil,wj,yli−^1 ,zli j−^1 ). In our model
we assume that workerjpredicts each classzli jwith some probability irrespective

of the true class (i.e. p(zli j|yil,wj,yli−^1 ,zli j−^1 )= **N** z

li j− 1
j (zli j)is multinomial with
a parameter for each possible child node). The taxonomic model results in the
following values that can be plugged into Equation 4.1:

p(zi j|yi,wj) =
÷L

l= 1
p(zli j|yil,wj), (4.6)
p(zi jl|yil,wj) =






M y
il−^1
j (yil,zli j)ifyil−^1 =zi jl−^1
N
zli j−^1
j (zi jl)otherwise
(4.7)

Note that in totality, for each nodenin the taxonomy, we have associated a confusion
matrix **M** njwith a row for each child ofn, and a vector of probabilities **N** njwith an
entry for each child. If the taxonomy is relatively balanced, this model has far
fewer parameters than the flat multinomial model (linear in the number of classes
rather than quadratic). To make estimating worker parameters more robust, we will
again make use of a tiered system of priors (e.g. Dirichlet priors on all multinomial
parameters) that are computed by pooling across all workers at each node. However,
if this is still too many parameters, we can fall back to modeling the probability that
a person is correct as a binomial distribution with a parameter per child node (i.e. the
**taxonomic per class binomial** model), or even just one parameter for all children
(i.e. the **taxonomic single binomial** model), assuming other class responseszli j,yil
have probability proportional to their priors. See Table 4.1 for an overview of all
models.

**Taxonomic Predictions**

Thus far, we have assumed that a worker always predicts a class of the finest possible
granularity (i.e. species level). An alternate UI can allow a worker to predict an
internal node in the taxonomy if unsure of the exact class, i.e. applying the “hedging
your bets” Deng et al., 2012 method to human classifiers. In Figure 4.1, this would
be akin to one of the identifiers specifying the family Ardeidae, which includes both
Snowy Egret and Great Egret. Letlevel(zi j)be the level of this prediction. Note
thatzli jis valid only forl≤level(zj). The taxonomic model in Section 4.4 works


after an update of Equation 4.6 top(zi j|yi,wj)=Œlevell= 1 (zi j)p(zi jl|yli,wj). This works
even if different workers provide different levels of taxonomic predictions.

**Dependent Labels**
In Section 4.4, we assumed each worker independently guesses the class of imagei.
We now turn to the situation described in Figure 4.1: a user submits an observation
xi and an initial identificationzi,ji 1 , where jit denotes thetth worker that labeled
imagei. A notification of the observation is sent to users that have subscribed to
the taxazi,ji 1 or to that particular geographic region (the rest of the community is
not explicitly notified but can find the observation when browsing the site). Each
subsequent identifier jit,t > 1 can see the details of the observationxi and all

identifications made by previous usersHit−^1 = {zi,ji 1 ,zi,j (^2) i, ...,zi,jti− 1 }. Users can
assess the experience of a previous identifierjby viewing all of their observations
Xj and all of their identificationsZj. Additionally, users are able to discuss the
identifications through comments.
In this setting, we can adapt Equation 4.1 to
p(yi|Zi)=p(yi|Hi|Wi|)
=
p(yi)
Œ|Wi|
t= 1 p(zi,jti|yi,Hit−^1 ,wjti)
ÕC
y= 1 p(y)
Œ|Wi|
t= 1 p(zi,jit|y,H
t− 1
i ,wjti)

(4.8)

There are many possible choices for modelingp(zi,jti|yi,Hit−^1 ,wjti). The simplest
option assumes each worker ignores all prior responses; i.e.,p(zi,jti|yi,Hit−^1 ,wjti)=
p(zi,jti|yi,wjit). In practice, however, workerjit’s response will probably be biased
toward agreeing with prior responsesHit−^1 , making a prediction combining both
evidence from analyzing prior responses and from observing the image itself. The
weight of this evidence should increase with the number of prior responses and
could vary based on workerjit’s assessment of other worker’s skill levels. In our
model, we assume that workerjtiweights each possible responsezi,jit(workerjti’s
perception of the class of imagei) with a termpjit(Hit−^1 |zi,jti)(workerjit’s perception
of the probability of prior responses given that class).p(zi,jti|yi,Hit−^1 ,wjit)can then
be expressed as:

p(zi,jti|yi,Hit−^1 ,wjti)=
p(zi,jti,Hit−^1 |yi,wjti)
p(Hit−^1 |yi,wjti)
=
p(zi,jit|yi,wjti)pjti(Hit−^1 |zi,jit,wjti)
Õ
zp(z|yi,wjti)pjti(Hit−^1 |z,wjti)
(4.9)


wherep(zi,jit|yi,wjti)is modeled using a method described in Section 4.4. Worker
jtimight choose to treat each prior response as independent sources of information
pjit(Hit−^1 |zi,jti,wjti)=Œts−=^11 pjit(zi,jis|zi,jti,wj
it
jis)where we have used the notationw

j
k
to denote parameters for workerj’s perception of workerk’s skill. Alternatively,
workerjmay choose to account for the fact that earlier responses were also biased
by prior responses using similar assumptions to those we made in Equation 4.9,
resulting in a recursive definition/computation ofpjti(Hit−^1 |zi,jit,wjit)=







pjit(zi,jit− 1 |zi,jti,wj
it
jit−^1 )pjti−^1 (H
ti− (^2) |zi,jt− 1
i ,w
jit−^1
jti−^2 )
Õzp
jti(z|zi,jti,w
jti
jit−^1 )pjit−^1 (H
it−^2 |z,wjit−^1
jti−^2 )
ift> 1
pjti(zi,jit− 1 |zi,jit,wj
ti
jti−^1 ) ift=^1

(4.10)

The last choice to make is how to model probabilities of the formpj(zk|zj,wkj)(i.e.
workerj’s perception of workerk’s responses). One model that keeps the number
of parameters low is a binomial distribution: workerjassumes other workers are
correct with probabilityρj; when they are incorrect, they respond proportionally to
class priors:

pj(zk|zj,wkj)=






ρj ifzk=zj
( 1 −ρj)p(zj) otherwise
(4.11)

Here,ρjis a learned parameter expressing workerj’s trust in the responses of other
workers.

### 4.5 Taking Pixels into Account

Rather than relying on class priorsp(yi), we can make use of a computer vision
model with parametersθthat can predict the probability of each class occurring in
each imagexi ∈ X. This results in an update to Equation 4.1, changingp(yi)to
p(yi|xi, θ). We use a computer vision model similar to the general purpose binary
computer vision system trained by Branson et al. (Branson, Van Horn, and Perona,
2017). We extract “PreLogit” featuresφ(xi)from an Inception-v3 (Szegedy et al.,
2016) CNN for each imagei, and use these features (fixed for all iterations) to train
the weightsθof a linear SVM (using a one-vs-rest strategy), followed by probability
calibration using Platt scaling (Platt et al., 1999). We use stratified cross-validation
to construct training and validation splits that contain at least one sample from each
class. This results in probability estimatesp(yi|xi, θ)=σ(γ θ·φ(xi)), whereγis the
probability calibration scalar from Platt scaling, andσ(·)is the sigmoid function.
Fine-tuning a CNN on each iteration would lead to better performance (Agrawal,


Method Label Error Rate
(%)
(Ghosh, Kale, and McAfee, 2011), (Dalvi et al., 2013) 27.78
Majority Vote 24.07
Flat Multinomial ,(Dawid and Skene, 1979), (Welinder et
al., 2010),(Karger, Oh, and D. Shah, 2013)
11.11

Flat Multinomial-CV , (Tian and Zhu, 2015), (Y. Zhang et
al., 2014)*
10.19

Table 4.2: Label error rates of different worker skill models on the binary Bluebird
dataset (Welinder et al., 2010) after receiving _all_ 4,212 annotations. Our methods
( **Flat Multinomial** , and **Flat Multinomial-CV** ) are competitive with other methods.
*(Y. Zhang et al., 2014) mistakenly reported 10.09.

Girshick, and Malik, 2014; Oquab et al., 2014; Yosinski et al., 2014) but is out of
scope.

### 4.6 Experiments

We evaluate the proposed models on data collected from paid workers through
Amazon Mechanical Turk (MTurk) and from non-paid citizen scientists who are
members of the Cornell Lab of Ornithology (Lab of O) or iNaturalist (iNat). We
follow a similar evaluation protocol to (Branson, Van Horn, and Perona, 2017) and
use Algorithm 1 from that work to run the experiments. For models that assume
worker labels are independent, we simulate multiple trials by adding worker labels
in random order. For lesion studies, we simply turn off parts of the model by
preventing those parts from updating. The tag _prob-worker_ means that a global
prior is computed across all workers, and per worker skill model was used; the tag
_online_ means that online crowdsourcing was used (with risk threshold parameter
τ=. 02 ), and the tag _cv_ means that computer vision probabilities were used instead
of class priors. **Bluebirds** To gauge the effectiveness of our model against prior
work, we run our models on the _binary_ bluebird dataset from (Welinder et al., 2010).
This dataset has a total of 108 images and 39 MTurkers labeled every image for a
total of 4,212 annotations. Table 4.2 has the final label error rates of different worker
skill models when _all_ annotations are made available. Our offline, flat multinomial
models are competitive with other offline methods.

**NABirds** This experiment was designed to test our models in a traditional dataset
collection situation where labeling tasks are posted to a crowdsourcing website and


(^1) Avg Number of Human Workers Per Image^24681020
1
Error0.1
Flat Single Binomial Model
(a)
(^1) Avg Number of Human Workers Per Image^24681020
1
Error0.1
Flat Per Class Binomial Model
majority-voteprob-workerprob-worker-cv
prob-worker-cv-online
MTurkerCTurkerCombined
(b)
(^1) Avg Number of Human Workers Per Image^24681020
1
Error0.1
Flat Per Class Multinomial Model
Combined-Prior
(c)
Figure 4.2: **Crowdsourcing Multiclass Labels with MTurkers and CTurkers:**
These figures show results from our flat models on a dataset of 69 species of birds
with labels from Amazon Mechanical Turk workers (MTukers) and citizen scientists
(CTurkers). Each model was run on a dataset that consisted of: just MTurkers
(squares), just CTurkers (triangles) or a combination of the two (circles). When
our full framework is used (prob-worker-cv-online, green lines) we can achieve the
same error as majority vote (red lines) with much fewer labels per image. When we
use our framework in an offline setting (prob-worker-cv and prob-worker, orange
and blue curves), we can achieve a lower error than majority vote with the same
number of labels. When initialized with generic priors, the single binomial model
achieves the lowest error, followed by the per class binomial and the multinomial
model. However, if domain knowledge is used to initialize the global priors to more
reasonable values, the multinomial model can achieve impressively low error (the
star lines in **(c)** ).
responses are collected independently. We constructed a labeling interface that
showed workers a sequence of 10 images and asked them to classify each image
into one of 69 different bird species by using an auto complete box or by browsing a
gallery of representative photos for each species. We used 998 images, all sampled
from either shorebird or sparrow species, from the NABirds dataset (Van Horn et
al., 2015). We collected responses from both MTurkers and citizen scientists from
the Lab of O (CTurkers). Figure 4.3a shows the contribution of annotations from
the workers. We had a total of 86 MTurkers provide 9,391 labels and a total of
202 CTurkers provide 5,300 labels. For these experiments, we made the gallery of
example images (3 to 5 images per species) available to the computer vision system
during training. This ensured that we could construct at least 3 cross validation
splits when calibrating the computer vision probabilities in the early stages of the
algorithm.
All models were initialized with uniform class priors, a probability of 0. 5 that an
MTurker will label a class correctly, and a probability of 0. 8 that a CTurker will label


(^1010010) Workers 1 102
1
102
103
Number of Annotations
Worker AnnotationsCTurker
MTurker
(a)
0.00.0 Empirical GT Probability Correct0.2 0.4 0.6 0.8 1.0
0.2
0.4
0.6
0.8
1.0
Predicted Probability Correct
CTurkerWorker Skills
MTurker
(b)
0.00.0Predicted Prob Correct on Shorebirds0.2 0.4 0.6 0.8 1.0
0.2
0.4
0.6
0.8
1.0
Predicted Prob Correct on Sparrows
Worker Taxonomic SkillsCTurker
MTurker
(c)
Figure 4.3: **MTurker and CTurker Worker Analysis:** Figure **(a)** shows the contri-
bution of labels per worker from MTurkers and CTurkers. On average we have less
than one label from each worker for each of the 69 classes, emphasizing the need to
pool data across workers for use as priors. Figure **(b)** shows the predicted probability
of a worker providing a correct labelmjplotted against the empirical ground truth
probability for the single binomial prob-worker-cv model from 4.2a. The size of
each dot is proportional to the number of annotations that worker contributed to
the dataset. Solid lines mark the priors. We can see that the model’s predictions
correlate well with the empirical ground truths. Figure **(c)** shows the predicted
worker skill for correctly labeling the species of a sparrow vs. correctly labeling
the species of a shorebird. These skill estimates came from a taxonomic binomial
model with one subtree corresponding to sparrows and the other corresponding to
shorebirds. In real applications, we can use these skill estimates to direct images to
proficient labelers.
a class correctly. This means the global Dirichlet priors (used in the multinomial
models) had a value of 0. 8 at the true class index and 0. 003 otherwise for the
CTurkers. These are highly conservative priors. For each of our three flat models
we conducted three experiments: using MTurk data only, using CTurk data only, and
using both MTurk and CTurk data together (“Combined” in the plots). Figure 4.2
shows the results. First, we note that when a computer vision system is utilized in an
online fashion (prob-worker-cv-online), we see a significant decrease in the average
number of labels per image to reach the same performance as majority vote using
all of the data (e.g. a 5.4×decrease in the single binomial combined setting). In
the offline setting (prob-worker-cv), the computer vision models decrease the final
error compared to majority vote (e.g. 25% decrease in error in the single binomial
combined setting). When considering our probabilistic model without computer
vision (prob-worker) the single binomial model consistently achieved the lowest
error, followed by the binomial per class model and then the multinomial model.
This is not unexpected, as we anticipated the larger capacity models to struggle with


the sparseness of data (i.e. on average we had 0. 75 labels per class per worker in
the combined setting). However, the fact that they approach similar performance to
the single binomial model highlights the usefulness of our tiered prior system and
the ability to pool data across all of the workers. Our global prior initializations
are purposefully on the conservative side, however in a real application setting,
a user of this framework can initialize the priors using domain knowledge or a
small amount of ground truth data. Figure 4.2c shows the dramatic effect of using
more informative priors in the combined setting (prob-worker-cv and prob-worker
in the Combined-Prior setting). These models were initialized with priors that were
computed on a small held out set of worker annotations with ground truth labels and
achieved the lowest error (0.03, for prob-worker-cv, a 79% decrease from majority
vote) on the dataset.

Figure 4.3b shows the predictedmjvalues learned by the single binomial model
plotted against the empirical ground truth in the combined setting. We can see that
the model’s predictions correlate well with the empirical estimates, with increasing
precision as the number of annotations increases (size of the dots). To further
investigate the worker skills, we constructed a simple 2 level taxonomy and placed
the shorebirds and sparrows in their own flat subtrees. By running our taxonomic
binomial model, we are able to learn a skill for each group separately, rendered in
Figure 4.3c. We can see that both MTurkers and CTurkers have a higher probability
of predicting shorebirds correctly than sparrows. In real applications, we can use
these skill estimates to direct images to proficient labelers.

**iNaturalist** This experiment was designed to test our models in a classification
situation that mimics the real world scenario of websites like iNat, see Figure 4.1.
We obtained a database export from iNat and cleaned the data using the following
three steps: (1) we select observations and identifications from a subset of the
taxonomy (e.g. species of birds); (2) for each observation, we keep only the first
identification from each user (i.e. we do not allow users to change their minds); and
(3) to facilitate experiments, we keep all observations that have a ground truth label
at the species level (i.e. leaf nodes of the taxonomy). For the experiments presented
below, after performing the previous steps, we selected a subset of 30 species of
birds and 1000 observations from each species to analyze. In this 30k image subset
we have 5,643 workers that provided a total of 98,849 labels; Figure 4.4c shows the
distribution of worker annotations. The taxonomy associated with these 30 species
consisted of 44 nodes with a max depth of 3. For these experiments we did not


(^10) 1.0^3 Avg Number of Human Workers Per Image1.5 2.0 2.5 3.0 3.5 4.0
102
101
Error
Single Binomial Model
(a)
(^10) 1.0^3 Avg Number of Human Workers Per Image1.5 2.0 2.5 3.0 3.5 4.0
102
101
Error
Per Class Multinomial Modelmajority-vote
prob-workerprob-worker-depprob-worker-tax
prob-worker-tax-dep
(b)
(^100100101) Workers 102 103
101
102
103
Number of Annotations
iNat Worker Annotations
(c)
(^101033) Empirical GT Probability of Error 102 101 100
102
101
100
Predicted Probability of Error
iNat Worker Skills
(d)
Figure 4.4: **iNaturalist Birds** Figures **(a)** and **(b)** show the errors achieved on a
dataset of 30 bird species from iNaturalist for the single binomial and multinomial
models respectively. Each model was evaluated in several configurations: “prob-
worker” assumes a flat list of species. “prob-worker-tax” takes advantage of a
taxonomy across the species, allowing workers to provide non-leaf node annotations
and reducing the number of parameters in the multinomial model from 900 to 167.
“prob-worker-dep” assumes a flat list of species, but models the dependence between
the worker labels. “prob-worker-tax-dep” uses a taxonomy across the species and
models the dependence between worker labels. All models did at least as well as
majority vote, with dependence modeling providing a significant decrease in error.
The lowest error was achieved by the multinomial prob-worker-tax-dep model that
was capable of modeling species confusions and label dependencies, decreasing
error by 90% compared to majority vote. Figure **(c)** shows the distribution of labels
per worker, emphasizing a long tail of worker contributions. Figure **(d)** shows the
predicted probability of error( 1 −mj)for each worker plotted against the empirical
ground truth probability of error for the single binomial prob-worker-dep model,
with the radius of a dot proportional to the number of annotations contributed by
that worker. The solid blue line is the global prior value. More active identifiers
are less likely to make errors, and our model skill estimates correlate well with the
empirical ground truths.
utilize a computer vision system. Class priors were initialized to be uniform, skill
priors were initialized assuming that iNat users are 80% correct. Worker labels are
added to the images sequentially by their time stamp, so only a single pass through
the data is possible.
Figures 4.4a and 4.4b show the results for our single binomial and multinomial
models respectively. For each model we used flat and taxonomic (-tax) versions,
and we turned on (-dep) and off label dependence modeling, for a total of 4 variations
of each model. We can see that all of our models are at least as good as majority
vote. Adding dependence modeling to the flat models provides a significant decrease
in error: a 59% decrease for the flat single binomial model, and an 85% decrease
for the flat multinomial model. The taxonomic single binomial model (with 14


parameters per worker) did slightly worse than the flat single binomial model (with
1 parameter per worker). However, the taxonomic multinomial model (with 167
parameters per worker) decreased error by 36% compared to the flat multinomial
model (with 900 parameters per worker). Finally, adding dependence modeling
to the taxonomic models provided a further decrease in error, with the taxonomic
multinomial model performing the best and decreasing error by 90% over majority
vote, corresponding to 28 total errors. While a majority of those errors were true
mistakes, an inspection of a few revealed errors in the ground truth labels of the
iNat dataset. Figure 4.1 is actually an example of one of those mistakes. Further,
the observation (https://tinyurl.com/ycu92cas) associated with the second “riskiest”
image (using the computed Bayes risk of the predicted labelR(y ̄i)) turned out
to be another mistake, advocating the use of these models as a way of sorting
the observations for expert review. Figure 4.4d shows the predicted probability
of a worker labeling incorrectly( 1 −mj)for the flat single binomial model with
dependence modeling from Figure 4.4a. We can see that the model’s skill predictions
correlate well with the empirical ground truth skills.

### 4.7 Conclusion

We introduced new multiclass annotation models that can be used in the online
crowdsourcing framework of Branson et al. (Branson, Van Horn, and Perona, 2017).
We explored several variants of a worker skill model using a variety of parameteriza-
tions and we showed how to harness a taxonomy to reduce the number of parameters
when the number of classes is large. As an additional benefit, our taxonomic models
are capable of processing worker labels from anywhere in the taxonomy rather than
just leaf nodes. Finally, we presented techniques for modeling the dependence of
worker labels in tasks where workers can see a prior history of identifications. Our
models consistently outperform majority vote, either reaching a similar error with far
fewer annotations or achieving a lower error with the same number of annotations.
Future work involves modeling “schools of thought” among workers and using their
skill estimates to explore human teaching.

**Acknowledgments**
This work was supported by a Google Focused Research Award. We thank Oisin
Mac Aodha for useful discussions.


**References**

Agrawal, Pulkit, Ross Girshick, and Jitendra Malik (2014). “Analyzing the per-
formance of multilayer neural networks for object recognition”. In: _European
Conference on Computer Vision_. Springer, pp. 329–344.

Branson, Steve, Grant Van Horn, and Pietro Perona (2017). “Lean Crowdsourcing:
Combining Humans and Machines in an Online System”. In: _Proceedings of the
IEEE Conference on Computer Vision and Pattern Recognition_ , pp. 7474–7483.
doi:10.1109/CVPR.2017.647.

Chen, Guangyong et al. (2017). “Learning to Aggregate Ordinal Labels by Maxi-
mizing Separating Width”. In: _International Conference on Machine Learning_ ,
pp. 787–796.

Dalvi, Nilesh et al. (2013). “Aggregating crowdsourced binary ratings”. In: _Proceed-
ings of the 22nd international conference on World Wide Web_. ACM, pp. 285–
294.

Dawid, Alexander Philip and Allan M Skene (1979). “Maximum likelihood esti-
mation of observer error-rates using the EM algorithm”. In: _Applied statistics_ ,
pp. 20–28.

Deng, Jia et al. (2012). “Hedging your bets: Optimizing accuracy-specificity trade-
offs in large scale visual recognition”. In: _Computer Vision and Pattern Recogni-
tion (CVPR), 2012 IEEE Conference on_. IEEE, pp. 3450–3457.

Ghosh, Arpita, Satyen Kale, and Preston McAfee (2011). “Who moderates the
moderators?: crowdsourcing abuse detection in user-generated content”. In: _Pro-
ceedings of the 12th ACM conference on Electronic commerce_. ACM, pp. 167–
176.

He, Kaiming et al. (2015). “Deep residual learning for image recognition”. In: _arXiv
preprint arXiv:1512.03385_.

Huang, Gao et al. (2017). “Densely connected convolutional networks”. In: _Pro-
ceedings of the IEEE Conference on Computer Vision and Pattern Recognition_.

Jin, Rong and Zoubin Ghahramani (2002). “Learning with multiple labels”. In:
_Advances in neural information processing systems_ , pp. 897–904.

Kamar, Ece, Severin Hacker, and Eric Horvitz (2012). “Combining human and
machine intelligence in large-scale crowdsourcing”. In: _Proceedings of the 11th
International Conference on Autonomous Agents and Multiagent Systems-Volume
1_. International Foundation for Autonomous Agents and Multiagent Systems,
pp. 467–474.

Karger, David R, Sewoong Oh, and Devavrat Shah (2011). “Iterative learning for
reliable crowdsourcing systems”. In: _Advances in neural information processing
systems_ , pp. 1953–1961.


Karger, David R, Sewoong Oh, and Devavrat Shah (2013). “Efficient crowdsourcing
for multi-class labeling”. In: _ACM SIGMETRICS Performance Evaluation Review_
41.1, pp. 81–92.

- (2014). “Budget-optimal task allocation for reliable crowdsourcing systems”. In:
    _Operations Research_ 62.1, pp. 1–24.

Kovashka, A. et al. (2016). “Crowdsourcing in Computer Vision”. In: _ArXiv e-
prints_. arXiv:1611.02145 [cs.CV].url:%7Bhttps://arxiv.org/abs/
1611.02145%7D.

Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E Hinton (2012). “ImageNet Clas-
sification with Deep Convolutional Neural Networks.” In: _NIPS_.

Li, Hongwei, Bin Yu, and Dengyong Zhou (2013). “Error rate analysis of labeling by
crowdsourcing”. In: _ICML Workshop: Machine Learning Meets Crowdsourcing.
Atalanta, Georgia, USA_.

Little, Greg et al. (2010). “Exploring iterative and parallel human computation pro-
cesses”. In: _Proceedings of the ACM SIGKDD workshop on human computation_.
ACM, pp. 68–76.

Littlestone, Nick and Manfred K Warmuth (1994). “The weighted majority algo-
rithm”. In: _Information and computation_ 108.2, pp. 212–261.

Liu, Qiang, Jian Peng, and Alexander T Ihler (2012). “Variational inference for
crowdsourcing”. In: _Advances in Neural Information Processing Systems_ , pp. 692–
700.

Long, Chengjiang, Gang Hua, and Ashish Kapoor (2013). “Active visual recogni-
tion with expertise estimation in crowdsourcing”. In: _Proceedings of the IEEE
International Conference on Computer Vision_ , pp. 3000–3007.

Mora, Camilo et al. (2011). “How many species are there on Earth and in the ocean?”
In: _PLoS biology_ 9.8, e1001127.

Ok, Jungseul et al. (2016). “Optimality of Belief Propagation for Crowdsourced
Classification”. In: _arXiv preprint arXiv:1602.03619_.

Oquab, Maxime et al. (2014). “Learning and transferring mid-level image repre-
sentations using convolutional neural networks”. In: _Proceedings of the IEEE
conference on computer vision and pattern recognition_ , pp. 1717–1724.

Platt, John et al. (1999). “Probabilistic outputs for support vector machines and
comparisons to regularized likelihood methods”. In: _Advances in large margin
classifiers_ 10.3, pp. 61–74.

Raykar, Vikas C et al. (2010). “Learning from crowds”. In: _Journal of Machine
Learning Research_ 11.Apr, pp. 1297–1322.

Shah, Nihar Bhadresh and Denny Zhou (2015). “Double or nothing: Multiplicative
incentive mechanisms for crowdsourcing”. In: _Advances in Neural Information
Processing Systems_ , pp. 1–9.


Smyth, Padhraic et al. (1995). “Inferring ground truth from subjective labelling of
venus images”. In:

Sullivan, Brian L et al. (2014). “The eBird enterprise: an integrated approach to
development and application of citizen science”. In: _Biological Conservation_
169, pp. 31–40.

Szegedy, Christian et al. (2016). “Rethinking the inception architecture for computer
vision”. In: _Proceedings of the IEEE Conference on Computer Vision and Pattern
Recognition_ , pp. 2818–2826.

Tang, Wei and Matthew Lease (2011). “Semi-supervised consensus labeling for
crowdsourcing”. In: _SIGIR 2011 workshop on crowdsourcing for information
retrieval (CIR)_ , pp. 1–6.

Tian, Tian and Jun Zhu (2015). “Max-margin majority voting for learning from
crowds”. In: _Advances in Neural Information Processing Systems_ , pp. 1621–
1629.

Ueda, K (2017). “iNaturalist Research-grade Observations via GBIF.org.” In:url:
https://doi.org/10.15468/ab3s5x.

Van Horn, Grant et al. (2015). “Building a bird recognition app and large scale
dataset with citizen scientists: The fine print in fine-grained dataset collection”.
In: _Proceedings of the IEEE Conference on Computer Vision and Pattern Recog-
nition_ , pp. 595–604.doi:10.1109/CVPR.2015.7298658.

Vempaty, Aditya, Lav R Varshney, and Pramod K Varshney (2014). “Reliable crowd-
sourcing for multi-class labeling using coding theory”. In: _IEEE Journal of Se-
lected Topics in Signal Processing_ 8.4, pp. 667–679.

Welinder, Peter et al. (2010). “The multidimensional wisdom of crowds”. In: _Ad-
vances in neural information processing systems_ , pp. 2424–2432.

Whitehill, Jacob et al. (2009). “Whose vote should count more: Optimal integration
of labels from labelers of unknown expertise”. In: _Advances in neural information
processing systems_ , pp. 2035–2043.

Yosinski, Jason et al. (2014). “How transferable are features in deep neural net-
works?” In: _Advances in neural information processing systems_ , pp. 3320–3328.

Zhang, Jing et al. (2016). “Multi-class ground truth inference in crowdsourcing
with clustering”. In: _IEEE Transactions on Knowledge and Data Engineering_
28.4, pp. 1080–1085.

Zhang, Yuchen et al. (2014). “Spectral methods meet EM: A provably optimal
algorithm for crowdsourcing”. In: _Advances in neural information processing
systems_ , pp. 1260–1268.

Zhou, Dengyong et al. (2014). “Aggregating Ordinal Labels from Crowds by Mini-
max Conditional Entropy.” In: _ICML_ , pp. 262–270.


Zhou, Denny et al. (2012). “Learning from the wisdom of crowds by minimax
entropy”. In: _Advances in Neural Information Processing Systems_ , pp. 2195–
2203.


