# Chapter 7: Reducing Memory & Computation Demands For Large Scale Visual Classification
Van Horn, Grant and Pietro Perona (2019). “Reducing Memory & Computation
Demands for Large Scale Visual Classification”.

### 7.1 Abstract

The computational and storage costs of state-of-the-art deep networks that are de-
signed for large scale visual classification (>1K categories) is dominated by the
fully-connected classification layers. This makes deployment problematic on mo-
bile devices, where app download size and power efficient execution is critical. In
this work we analyze different techniques aimed at reducing this bottleneck and
present a new technique, Taxonomic Parameter Sharing, that utilizes a taxonomy
to share parameters among the classes. Our experiments on the iNaturalist dataset
show that a simple tactic of jointly training a standard fully connected layer along
with a rank factorized layer can result in a 25x reduction in memory and compu-
tation in the classification layer without any loss in top-1 accuracy. The standard
fully connected layer can be discarded at test time. For a task with 8k classes, this
reduces the floating point memory requirements of the final layer from 64MB to
2.6MB when using a feature vector of size 2048. Our Taxonomic Parameter Sharing
approach is competitive in the regime where reduced parameter count is important
during both training and testing.

### 7.2 Introduction

Deep convolutional neural networks (DCNN) have dramatically improved perfor-
mance of computer vision systems. This includes decreases in error rates on aca-
demic benchmark datasets (Krizhevsky, Sutskever, and G. E. Hinton, 2012), fast and
accurate retrieval performance on consumer devices (Howard et al., 2017), niche
computer vision apps (Van Horn, Branson, et al., 2015), and rapidly improving self
driving cars. A remaining obstacle to the ubiquitous adoption of DCNNs are the
computational, energetic and memory costs of running the networks on portable
wireless devices.


Figure 7.1: **Large Scale Visual Classification** : This work is motivated by a specific
application problem: allow a user to point their phone’s camera at wildlife and
classify it in real time without requiring a network connection. The model is trained
with data collected by the citizen science website iNaturalist (https://inaturalist.org)
and the number of species grows daily, totalling over 30k in January 2019. Important
aspects of this application is the model size (which directly impacts the app download
size), the execution time (which directly impacts user interaction), model efficiency
(which directly impacts battery life) and the classification accuracy of the model
(which directly impacts user satisfaction).

Two approaches have been proposed for reducing the computational and memory
requirements of the network. The first focuses on reducing the precision of the
network weights and activations by quantizing their values (i.e. representing numbers
with fewer bits) (Asanovic and Morgan, 1991; Vanhoucke, Senior, and M. Z. Mao,
2011; Yunchao Gong et al., 2014; Courbariaux, Y. Bengio, and David, 2015; Han,
H. Mao, and Dally, 2016; Rastegari et al., 2016). Besides reducing storage, if
appropriate hardware is designed (Jouppi et al., 2017) the execution time and power
can be reduced as well.

The second focuses on reducing the number of operations in the network. This is
accomplished in four different ways. **1.** By designing efficient network architec-
tures (Mamalet and Garcia, 2012; J. Jin, Dundar, and Culurciello, 2014; Szegedy


et al., 2016; Howard et al., 2017; Sandler et al., 2018; Zhang et al., 2018; X. Jin
et al., 2018) (e.g. replacing 7x7 convolution blocks with sequences of 3x3 blocks).
Similar to this are works that employ structured efficient linear layers (Yang et al.,
2015; Cheng et al., 2015; Sindhwani, T. Sainath, and Kumar, 2015; Moczulski et al.,
2015; Hoffer, Hubara, and Soudry, 2018) that allow for fast matrix multiplication
with fewer parameters. Note that “convolutional networks” are themselves a means
of parameter efficiency when compared to fully connected layers, locally connected
features (Coates, Ng, and Lee, 2011) and tiled convolutional networks (Gregor
and LeCun, 2010). **2.** Through network pruning (Hassibi and Stork, 1993; LeCun,
Denker, and Solla, 1990; Han, Pool, et al., 2015; Guo, Yao, and Chen, 2016; Alvarez
and Salzmann, 2016; Zhou, Alvarez, and Porikli, 2016; H. Li et al., 2017), where
redundant weights (and therefore operations) are post-hoc removed from trained
network. **3.** Through knowledge distillation (Bucilua, Caruana, and Niculescu-ˇ
Mizil, 2006; Ba and Caruana, 2014; G. Hinton, Vinyals, and J. Dean, 2014) where
a smaller network is trained on the logits or softmax output of a larger network or
an ensemble of networks. **4.** Through filter factorization and decomposition tech-
niques (Masana et al., 2017; Jaderberg, Vedaldi, and Zisserman, 2014; Mamalet and
Garcia, 2012; Denton et al., 2014; Lebedev et al., 2014; Ba and Caruana, 2014) or
rank restrictions(Xue, J. Li, and Yifan Gong, 2013; Denil et al., 2013; T. N. Sainath
et al., 2013) to speed up a network and reduce memory usage.

Here we focus on the computational and storage costs of networks that classify
thousands of categories, see Figure 7.1. This regime is often referred to as “Large-
Scale Visual Classification” (LSVC). In LSVC computational and memory costs are
dominated by the fully connected classification layer, which scales linearly with the
number of classes. When the number of classes becomes sufficiently large, this
final layer becomes the memory bottleneck of the network, and its multiply-add
operations dominate computational costs. For example, consider a task with 8k
classes and a backbone DCNN architecture that produces a 2k dimension feature
vector. The fully connected layer for this setup has 16M parameters. If 32 bit floating
point values are used, then this matrix consumes 64MB of memory, and takes 16M
multiply-add operations to project the feature vector into class logits. Attempting
to quantize the representation to 8 bits will only reduce the size by a factor 4, and
will not reduce the number of multiply-add operations. Now consider the training
regime where a batch size of 32 , 64 , or 128 is used; the memory demands balloon
to over 8GB for a batch size of 128. We aim for dramatic cost reduction.


A straightforward tactic for reducing the cost is to factorize the fully connected
layer into two lower rank matrices. In our experiments we explore the performance
differences when we factorize as a post processing step, while training, and while
fine-tuning. We also investigate the effects of jointly training a full sized layer
along side a rank factorized layer. These techniques represent the simplest, easiest
to implement tactics and we find that they can result in good performance. In
addition to these strong baseline experiments, we propose a new architecture for
the fully connected last layer of the network. This new architecture is composed
of multiple fully connected layers whose outputs are parsed to navigate a taxonomy
over the classes. We call this technique Taxonomic Parameter Sharing (TPS), and we
demonstrate that it achieves competitive accuracy with much lower cost compared
to the standard fully connected layer.

### 7.3 Related Work

**Matrix Factorization**
Our analysis of factorizing the final fully connected layer is related to the works of
Sainath et al. (T. N. Sainath et al., 2013) and Denton et al. (Denton et al., 2014).
Sainath et al. (T. N. Sainath et al., 2013) explored factorizing the final matrix to make
**training** more efficient while preserving accuracy. We expand upon their work and
focus specifically on test time efficiency by investigating factorized layers that are
fine-tuned from or jointly trained with a non-factorized fully connected layer. Our
experiments reveal that a 25x reduction in the final layer parameter count can be
achieved without a loss in accuracy. Denton et al. (Denton et al., 2014) also focus on
test time efficiency and present results on factorized fully connected layers (see Table
2 in (Denton et al., 2014)). We expand upon their work by investigating various
techniques for training the factorized layers, provide a more thorough analysis of
performance, and show that the parameters in the classification layer can be reduced
by 25x, as opposed to their 8x findings.

**Large-Scale Classification**
In the realm of large scale classification (Deng, Dong, et al., 2009; Thomee et
al., 2016; Krasin et al., 2017; Van Horn, Mac Aodha, et al., 2018) our work is
related to methods that utilize a hierarchy to trade off concept specificity versus
accuracy (Deng, Berg, et al., 2010; Deng, Krause, et al., 2012; Ordonez et al., 2013)
and methods that use the hierarchy to learn experts on subsets of the classes (Yan
et al., 2015; Ahmed, Baig, and Torresani, 2016). Most relevant are methods aimed


at reducing the computational bottleneck of the softmax layer of a neural network.
These methods have been explored mainly along three directions.

The first direction is that of hierarchical models (Morin and Y. Bengio, 2005;
Mikolov et al., 2013; Yan et al., 2015) (for SVM approaches see (Griffin and Perona,
2008; Marszałek and Schmid, 2008; Gao and Koller, 2011)). In these approaches
a classifier is learned at each internal node of a taxonomic tree built on top of the
class labels, which could be as simple as a two layer “course-to-fine” hierarchy. This
increases the memory requirements of the model (the number of classes to classify
has increased) but the computational requirements decreases to that of traversing
to a leaf node. A similar approach is undertaken by (S. Bengio, Weston, and
Grangier, 2010; Deng, Satheesh, et al., 2011; Liu et al., 2013) where the authors
build a label tree over the classes, allowing for a reduction in both memory and
computation demands. Our TPS approach is different in that instead of repeatedly
dividing leaf nodes to learn a label tree we encode an existing tree structure. While
at first this seems like a step backwards, the use of a semantic tree in our approach
allows us to utilize additional training data not available to other algorithms and
to provide interpretable taxonomic predictions. Further, our method is achieves
memory reduction in addition to computation reduction.

The second direction of research utilizes Locality Sensitive Hashing (Gionis, Indyk,
Motwani, et al., 1999) to find the k-nearest rows of the weight matrix and then
approximate the softmax output by using only the dot products between those k
vectors and the feature vector. Vijayanarasimhan et al. (Vijayanarasimhan et al.,
2015), building on the work of (Yagnik et al., 2011; T. Dean et al., 2013), introduce
this hashing logic into both the training and inference executions of the model. The
related works of (Mussmann and Ermon, 2016; Mussmann, Levy, and Ermon, 2017;
Levy, Chan, and Ermon, 2018) have continued to explore this route, particularly
for natural language processing. These methods can significantly decrease the
computational cost of evaluating the softmax layer. However, the full weight matrix
still needs to be made accessible. The hashing layers and random memory accesses
that are necessary for these methods complicate GPU utilization.

The third direction of research (Krizhevsky and G. E. Hinton, 2011; Weston, S.
Bengio, and Usunier, 2011) also utilizes hashing but do so in the context of k-
nearest-neighbor search in an embedding space. These methods learn an embedding
space representation (typically using a ranking loss), project the training data into
this embedding space, and use k-nearest-neighbors via hashing to classify images


at inference time. The computation bottleneck is alleviated with these methods,
but the memory requirements is increased due to storage of the embedded training
images.

A similar theme with all of these works is that the reduction in computational com-
plexity of evaluating the softmax layer requires increasing the memory requirements
of the model or adding complex logic to the training or inference portions of the
model, or both. We contribute a new technique that can both reduce the computation
and memory requirements of the model, while still being simple.

### 7.4 Taxonomic Parameter Sharing

Figure 7.2: **Taxonomic Parameter Sharing** : In this visual example our method
converts an original classification problem over 16 classes to 3 3-way classification
problems by using a taxonomy over the original classes, a 1.7x reduction. At
each level in the original taxonomy, we construct a new classification problem by
binning the sibling nodes into distinct “buckets”. Nodes can be randomly assigned
to “buckets”, or a distance metric and a more sophisticated assignment function can
be used. Note that all sibling nodes from the original taxonomy are placed into
separate “buckets” for the new classification problem. During training, all of the
images from all of the nodes in a given “bucket” are combined together to train a
given new class. This format make it trivial to include additional inner node training
data from the original taxonomy. Each new classifier is trained jointly, receiving
the same feature vector from the backbone DCNN (so 3 classification “heads” are
trained in this example). At test time, all of the new classifiers are run, and a
prediction on the original classes is obtained in the following way: (1) The red
classifier is run, producing a prediction for either $, #, or @. The green classifier is
then run, and the most likely bucket that contains a child of the ancestor prediction
is selected. Finally this process is repeated for the blue classifier, which selects a
leaf node from the original taxonomy.


Our Taxonomic Parameter Sharing (TPS) method is a simple, greedy algorithm that
utilizes a taxonomy overNclasses to construct a collection of new classification
problemsC. The combined classification results fromCcan be used to predict the
originalNclasses. Assume we have a taxonomy withLlevels, where a node’s level
is the distance from it to the root node, and all leaf nodes are on the same level.
Traditionally, a taxonomy is utilized by training a classifier at each inner node of the
taxonomy to disambiguate that node’s children. This increases the total number of
classification tasks, but during test time we only need to traverse to a leaf node using
the classification results from L classifiers. This can be far faster than evaluating one
classifier over all leaf nodes. The motivation behind the TPS algorithm is to try to
maintain the reduction in test time computation **and** reduce the total memory usage
as well. We do this by grouping non-sibling nodes at the same level into new “super”
classes. Therefore, we have exactly 1 classifier responsible for making predictions
at each level. This is how the TPS algorithm reduces the memory demands while
still requiring only L classifications to make a prediction.

Algorithm 2 provides an overview of the process of creating the new classification
problems and Figure 7.2 provides a visual description. Given a taxonomyTour
algorithm proceeds to process each level of the taxonomy independently. For each
levell, we collect all the nodes at that level (i.e. all nodes at distancelfrom the
root node) and first determine the largest number of siblingssmax(i.e. the largest
number of nodes at levellthat share the same parent). We then construct a new
classification problem withnlclasses wherenl ≥smax. We then assign all nodes
at levellto one of the new classes. The algorithm is called Taxonomic Parameter
Sharing because we are requiring multiple nodes at a given level in the taxonomy
to share parameters. Next we provide two different methods for pickingnl and
assigning the nodes to classes. We then describe how classification results on the
new classes can be used to produce classes over the originalNclasses. And we
finish this section with additional benefits of the proposed algorithm.

**Random Assignment**
The simplest assignment strategy is to randomly assign each node at levellto one
of the new classesnl. Sibling nodes can be handled by grouping them together and
sampling class assignments without replacement, so that it is guaranteed that no two
siblings are assigned to the same class. Choosing the number of classes can be done
greedily by takingnl=smax. This produces the smallest classification problem for
this level. Increasingnlincreases the number of classes that must be classified, but


**Algorithm 2** Taxonomic Parameter Sharing
1: **input** : taxonomyTwith levelsL
2: **for** l∈L **do**
3: smax←max sibling count at levell
4: Construct new classification problemClof size|Cl| ≥smax
5: Assign each node at levellto one of the new classes such that no sibling
nodes are assigned to the same class.
6: **end for**
7: **return** C={Cl∀l} .The new classification problems.

can also make the classification problem easier.

**Facility Location Assignment**
A more sophisticated assignment strategy is to determine the node assignment and
number of classesnlfor levelljointly. This can be done by posing the problem as
a facility location problem (Erlenkotter, 1978) and using a greedy approximation
algorithm (Jain, Mahdian, and Saberi, 2002) to solve it. In this setup the cities
are the nodes at levell and the facilities that are opened to service the cities
correspond to the new classes. We can enforce that sibling nodes at levellcannot
be assigned to the same facility. Facility location problems require a notion of
distance between the cities and in our experiments we use the euclidean distance
computed between averaged training feature vectors for each class (extracted from
an ImageNet Inception-v3 model (Szegedy et al., 2016)). Alternatively, domain
specific or application specific knowledge can be used to specify the distance of the
nodes. Facility location problems also require a cost of opening a facility. This
cost value is directly related to the final size of the new classification task, with a
larger cost producing fewer opened facilities. We employed large cost values in our
experiments so that the resulting number of classes is small.

**Classification**
We can convert the classification results on the new collection of classification tasks
Cfor a test image to a classification of the originalNclasses using the taxonomy
T. We start from the root node ofTand choose the highest scoring class fromn 1.
Note that because all nodes at levell= 1 are siblings, the highest scoring class from
n 1 corresponds to a unique node in the taxonomy, call this nodet 1. We chooset 1
as our prediction for levell= 1. For levell= 2 , we examine the results ofn 2 and
consider only those classes that contain children oft 1. Because all sibling nodes


are assigned to unique classes inn 2 , we can simply choose the child corresponding
with the highest scoring class fromn 2. We repeat this process until we reach a leaf
node, which will correspond to one of the originalNclasses. Note that conditional
probabilities can be computed by multiplying the class probabilities as we traverse
from the root node to a leaf node. Also, there is nothing preventing us from
computing the conditional probability of all leaf nodes (no additional classifications
need to be done), providing the full distribution across all of the originalNclasses.

**Inner Node Training Data**
The Taxonomic Parameter Sharing algorithm makes it easy to include in the training
set data that is not labeled at the species level, but rather at the genus or other
level. These images, classified to levell, can simply be used to train all classifiers
corresponding to levels≤l. See Sec. 7.5 for details.

### 7.5 Experiments

Figure 7.3: **iNaturalist Images:** Example images from the iNaturalist 2018 com-
petition dataset. Each column contains two different species from the same genus.
These species pairs are often confused by the classifier. From left to right: A. hetzi
and A. chalcodes, P. thoas and P. rumiko, S. jello and S. barracuda, L. alleni and L.
californicus. Image credits from left to right, top to bottom: cullen, Roberto Gon-
zalez, Ian Banks, Francisco Farriols Sarabia, CK Kelly, Francisco Farriols Sarabia,
Marisa Agarwal, mbalame99

**Dataset**
We conduct experiments using an augmented version of the iNaturalist 2018 com-
petition dataset 1 , see Figure 7.3. The iNaturalist 2018 dataset consists of 8 , 142
species, however, due to a taxonomy conflict with the taxonomy hosted on inatural-
ist.org, we removed 4 species from the dataset (classes 330, 5150, 119, 120) and we

(^1) https://github.com/visipedia/inat_comp


merged class 5185 with 5188 and class 6184 with 6185, resulting in 8 , 136 species
and a taxonomy with complete ancestry paths (consisting of Kingdom, Phylum,
Class, Order, Family, and Genus ancestors for all species), see Table 7.1 for the
number of nodes and the maximum sibling counts at each taxonomic rank (we use
the terms “rank” and “level” interchangeably). Our augmented dataset contained all
of the iNaturalist 2018 dataset images (for the 8 , 136 species included), which are
all identified to species.

To explore the utility of training with non-leaf node data we augmented the dataset to
include images identified to a courser node. These additional images were chosen by
the following procedure: starting from genus nodes and continuing up to ancestors
nodes, for each nodenwe sum the images in the descendants ofnand attempt to
include additional images identified tonuntil the total number of images is 1k. Note
that when we are augmentingnwe do not include images identified to descendants
ofn, we only include those images that the iNaturalist community identified at
n. This procedure resulted in an additional 969 , 095 images added to the dataset,
for a total of 1 , 406 , 529 training images. See Table 7.1 for a break down of how
many additional images were included at each rank. Note that no additional images
identified to species were included. We report accuracy metrics using the validation
set from the iNaturalist 2018 dataset. All of the validation images are identified to
species. In the following sections, performance numbers on the validation set refer
to the percentage of images correctly identified when the model gets one guess.

Kingdom Phylum Class Order Family Genus Species
# of Nodes 6 20 54 275 1114 4420 8136
Max Siblings 6 9 7 39 73 173 28
Images 215 680 2 , 006 18 , 993 162 , 806 784 , 395 437 , 434
Total Images 1 , 406 , 529 1 , 406 , 314 1 , 405 , 634 1 , 403 , 628 1 , 384 , 635 1 , 221 , 829 437 , 434
Table 7.1: **Taxonomy & Image Statistics** : Our iNaturalist taxonomy is composed
of 8 , 136 species nodes, each with a Kingdom, Phylum, Class, Order, Family, and
Genus ancestor. The non-taxonomic experiments use only the species training
images. The taxonomic experiments use the species training images along with
additional inner node data. The TPS models can use either just species data or can
be trained with the additional inner node data as well. For the randomly assigned
TPS models, the “Max Siblings” row provides the size of the respective classification
problem at each level in the taxonomy. The “Total Images” row is a cumulative count
of training data at a particular level along with the training data at lower levels.


**Backbone Architecture**
We use an Inception-V3 (Szegedy et al., 2016) backbone architecture for all exper-
iments. Image inputs are resized to 299 x 299 and basic image augmentations are
employed. The model is trained using RMSProp with a batch size of 32 , an initial
learning rate of 0. 0045 decayed by 0. 94 every 4 epochs, batch normalization, and a
smalll 2 regularization is applied to all weights. Training is monitored by plotting
the training and validation performance and early stopping is employed. The output
feature dimension is 2048. Unless otherwise stated, all experiments started from an
ImageNet pretrained model.

Figure 7.4: **Validation Accuracy vs Test Time Parameter Count** : This plot
summarizes our experiments on the iNaturalist 2018 dataset. The open symbols
represent experiments that make use of only species-level labels, and **did not** make
use of inner node data (Sec. 7.5). The filled symbols represent experiments that **did**
make use of both species-level and inner node data (Sec. 7.5). symbols repre-
sentl 2 regularized fully connect or factorized experiments. ©symbols represent
l 1 regularized fully connected experiments. 4 symbols represent our Taxonomic
Parameter Sharing (TPS) model (Sec. 7.5). Red experiments are baselines of the re-
spective models. Blue experiments take a trained model from a red experiment and
factorize the fully connected classification layer using SVD (Sec. 7.5 and Sec. 7.5).
Green experiments train factorized matrices of the form 2048 ×kandk× 8136 from
scratch (Sec. 7.5). Orange experiments fine-tune factorized matrices of the form
2048 ×kandk× 8136 , starting from a fully trained baseline model (Sec. 7.5). Purple
experiments jointly train standard fully connected layers and factorized matrices (i.e.
multiple classification heads)(Sec. 7.5 and Sec. 7.5). Jointly training a standard fully
connected layer along with a factorized version (purple curves) provides the best
parameter reduction to accuracy loss. Jointly training a rank 64 factorized matrix
reduces the parameters in the classification layer by 25x (at test time) and results in
no loss in accuracy.


**Non-Taxonomic Models**
The models used in this section do not make use of a taxonomy nor any of the
additional training available on the inner nodes.

**Baseline**

Our baseline method simply trains a fully connected layer of equal size to the number
of species, resulting in a matrix of size 2048 × 8136 with 16 , 662 , 528 parameters.
Note that we do not include the bias in the parameter count for any of the models.
This “off the shelf” model achieves a top-1 accuracy of 60. 5 ± 1. 7. We will use
these values as baselines for the rest of the methods. This model represents an
“out-of-the-box” solution.

l 1 **Regularization**

In these experiments we took our baseline model and regularized the last fully
connected layer using anl 1 penalty. This penalty encourages weights to be 0 , so
it is the optimizer that is tasked with increasing the sparsity of the last layer. We
experimented with varying regularization strengths of 4 −^4 , 4 −^5 , and 4 −^6 , resulting in
top-1 accuracy scores of 8. 82 , 51. 21 , and 62. 12. If we clip all weight values whose
absolute value is less than 1 −^7 , then these models would produce sparse final fully
connected layers with 16 , 199 , 031 , 7 , 096 , 449 , and 13 , 149 , 137 respectively. We
can see that 4 −^4 was too strong of a regularization and resulted in a model that could
not converge, while a smallerl 1 regularization like 4 −^6 resulted in a high performing
model, but with only a (hypothetical) factor of 1. 26 x savings in memory.

**SVD**

In these experiments we took our baseline model (fully trained) and factorized
the last matrix using SVD, producing three lower rank matricesUΣVT. We then
classified each validation image using the factorized model. We experimented with
the following lower rank values: 64 , 128 , 256 , 512 , and 1024. Figure 7.4 plots the
accuracy and parameter counts of these models. We can see that small rank values
produce desired reduction in the number of parameters, but the accuracy takes a
significant hit, with a rank 64 factorization resulting in a top-1 accuracy of 37. 71.


**Matrix Factorization**

In these experiments, as opposed to doing an SVD post processing operation on a
trained matrix, we train a factorized fully connected layer, composed of two lower
rank matrices of size 2048 ×kandk× 8136. We experimented with the following
rank valuesk∈ { 16 , 32 , 64 }. Figure 7.4 plots the accuracy and parameter counts of
these models (blue curve, non-filled squares). Note that these models were trained
from an ImageNet model. We can see that this is a simple method that results in
a significant decrease in parameters while maintaining reasonable accuracy. This
result was similarly mentioned by Denton et al. (Denton et al., 2014)), see Section
5 of their work.

**Matrix Factorization Fine-tuning**

In these experiments we take the trained baseline model and replace the large
fully connected layer with randomly initialized factorized matrices (of the form
2048 ×kandk× 8136 ). We then fine-tune **only** the factorized matrices, leaving
the backbone network untouched. We experimented with the following rank values
k∈ { 16 , 32 , 64 }. Figure 7.4 plots the accuracy and parameter counts of these models
(orange curve, non-filled squares). We can see that we can effectively recover
the performance of the baseline model with much fewer parameters ( 651 , 776 vs
16 , 662 , 528 for rank 64 ). However, performance does drop for even lower rank
values ( 39. 3 top-1 accuracy for rank 16 ).

**Baseline + Matrix Factorization Joint Training**

In these experiments we jointly train a baseline model (i.e. fully connected layer
of size 2048 × 8136 ) along with a factorized fully connected layer, composed
of two lower rank matrices of size 2048 ×k andk× 8136. To be specific, the
backbone produces a feature vector of size 2048 which is fed into two different
“classification heads”, one that is a standard fully connected layer and one that
is a factorized version of a fully connected layer. Losses are computed for both
outputs and are added equally. We experimented with the following rank values
k∈ { 16 , 32 , 64 }. Figure 7.4 plots the accuracy and parameter counts of these models
(purple curve, non-filled squares). We can see that this method recovers a high
performing factorized layer for low rank factorizations, performing as well or better
than the fine-tuned factorizations. We can also see that the factorized performance
for low rank values (k = 32 andk = 16 ) still results in good performance ( 57. 5


and 53 top-1 accuracy, respectively). A rankk= 16 factorization results in a 102x
savings in weights and computations.

**Taxonomic Models**
The models in this section make use of the taxonomy for training, and some make
use of the taxonomy for classification. It will be clear from the context which models
make use of the additional training data at the inner nodes during training.

**Baseline**

Our baseline model utilizes the taxonomy by training a separate fully connected
layer for each of the 7 taxonomic ranks, see Table 7.1 for details on the number
of nodes at each rank. During training, this model uses 2048 × 14 , 025 ≈ 28 M
parameters for the classification layer. Once this model is trained, we keep only the
species classifier and remove the other classifiers for testing. Using the additional
inner node training data, our taxonomic baseline species classifier achieves a top-1
accuracy of 70. 0. Note that while at test time this model has the same number of
parameters in the classification layer as the non-taxonomic baseline, during training
this taxonomic baseline requires 1.7x more parameters.

**SVD**

In this experiment we decompose the fully connected layer of the species classifier
from the taxonomic model using SVD. We experimented with the following rank
valuesk ∈ { 64 , 128 , 256 , 512 , 1024 }. Figure 7.4 plots the accuracy and parameter
counts of these models (blue curve, filled squares). Similar to the non-taxonomic
baseline, we can see that accuracy is well maintained for large rank values (i.e.
k= 1024 ) but falls off for smaller rank values (i.e.k= 64 ) that produce the desired
large decreases in parameters.

**Baseline + Matrix Factorization Joint Training**

In this experiment we trained a taxonomic baseline model (i.e. 7 fully connected
layers for each of the ranks) plus an additional factorized species model. We then
classified the validation images use the factorized species classifier. We experi-
mented with the following rank valuesk ∈ { 32 , 64 }. Figure 7.4 plots the accuracy
and parameter counts of these models (purple curve, filled squares). Similar to
the non-taxonomic experiments, we can see the benefit of doing this joint training,


where we recover essentially the same performance of the baseline model but use
far fewer parameters.

**Taxonomic Parameter Sharing**

In these experiments we analyze the performance of TPS models. Using a random
assignment method to the fewest number of classes possible (see Table 7.1 for the
max siblings at each level), the TPS model achieves a top-1 accuracy of 51. 2 using
a total of 686 , 080 parameters (a reduction in parameters by 24.3x compared to the
baseline). This is better accuracy than the 47. 75 accuracy of the trained matrix
factorization of rankk= 64 method (with 651 , 776 parameters), with both methods
starting from an ImageNet pretrained model. We achieved similar performance using
the facility location assignment algorithm to bin the nodes into the new classification
problems. Unlike the previous methods, incorporating additional inner node training
data requires no additional parameters during training. The additional training data
increased the performance of the randomly assigned TPS model to 55. 9.

The TPS model makes predictions in a hierarchical manner, meaning that it is
affected by mistakes made at ancestor classifiers. To put the TPS model’s perfor-
mance into context, we can compare it to the baseline (full rank) taxonomic model
that makes it predictions via hierarchical predictions (i.e. predict Kingdom before
predicting Phylum, _etc._ ) as opposed to only species classifications. The baseline
taxonomic model trained **without** additional data achieves a top-1 accuracy of 54. 2
using hierarchical predictions. The baseline taxonomic model trained **with** addi-
tional data achieves a top-1 accuracy of 59. 8 using hierarchical predictions. Using
these values as upper limits of performance, we can see that the TPS models are
within 94% and 93% of these limits when trained without and with the additional
data, respectively, but use 24.3x fewer parameters.

**Observations**
We note a few observations from these experiments. First, it is advantageous
to be able to train a standard fully connected layer because the minimum found
by this over-parameterized layer is better than what can be discovered by a more
parameter constrained layer. Second, either fine-tuning a factorized layer from a
model trained with a standard fully connected layer, or (more preferably) jointly
training a factorized layer with the standard fully connected layer, enables the
factorized layer to achieve a performance comparable to the fully connected layer


but with drastically fewer parameters. Being able to jointly train the factorized layer
means that only one training pass needs to be done. Third, if the number of classes
is too large to train a standard fully connected layer, or training a standard fully
connected layer is simply too slow, then the TPS algorithm can achieve as good or
better performance than a factorized method. The TPS algorithm gets the additional
benefit of being able to include additional inner node training data at no additional
parameter expense.

### 7.6 Conclusion

In this work we analyzed several different methods for reducing the computation
and memory requirements of the classification layer, including the novel Taxonomic
Parameter Sharing algorithm. We used the large-scale iNaturalist 2018 competition
dataset to conduct the experiments and arrived at several interesting findings. Likely
the most interesting and useful for general practitioners is the fact that one can
jointly train a factorized classification matrix with the regular fully connected layer
to produce a test time model that maintains the same accuracy, yet uses 25x fewer
parameters. This is a big savings when considering the usage of DCNN in mobile
applications, where users are wary of large download sizes, and power consumption
is a top concern.

For future work we plan on exploring an even larger class space, for the natural
world this will take us to over one million species. Towards this end it will be
beneficial to take into account the work on dynamic routing in networks (McGill
and Perona, 2017). Having one layer responsible for hundreds of thousands or
millions of classes seems undesirable. Exploring how networks can organize their
knowledge into clusters and dynamically access that knowledge based on the input
seems like a more manageable way forward.

**References**

Ahmed, Karim, Mohammad Haris Baig, and Lorenzo Torresani (2016). “Network
of experts for large-scale image categorization”. In: _European Conference on
Computer Vision_. Springer, pp. 516–532.

Alvarez, Jose M and Mathieu Salzmann (2016). “Learning the number of neu-
rons in deep networks”. In: _Advances in Neural Information Processing Systems_ ,
pp. 2270–2278.

Asanovic, Krste and Nelson Morgan (1991). _Experimental determination of pre-
cision requirements for back-propagation training of artificial neural networks_.
International Computer Science Institute.


Ba, Jimmy and Rich Caruana (2014). “Do deep nets really need to be deep?” In:
_Advances in neural information processing systems_ , pp. 2654–2662.

Bengio, Samy, Jason Weston, and David Grangier (2010). “Label embedding trees
for large multi-class tasks”. In: _Advances in Neural Information Processing Sys-
tems_ , pp. 163–171.

Bucilua, Cristian, Rich Caruana, and Alexandru Niculescu-Mizil (2006). “Modelˇ
compression”. In: _Proceedings of the 12th ACM SIGKDD international confer-
ence on Knowledge discovery and data mining_. ACM, pp. 535–541.

Cheng, Yu et al. (2015). “An exploration of parameter redundancy in deep networks
with circulant projections”. In: _Proceedings of the IEEE International Conference
on Computer Vision_ , pp. 2857–2865.

Coates, Adam, Andrew Ng, and Honglak Lee (2011). “An analysis of single-layer
networks in unsupervised feature learning”. In: _Proceedings of the fourteenth
international conference on artificial intelligence and statistics_ , pp. 215–223.

Courbariaux, Matthieu, Yoshua Bengio, and Jean-Pierre David (2015). “Binarycon-
nect: Training deep neural networks with binary weights during propagations”.
In: _Advances in neural information processing systems_ , pp. 3123–3131.

Dean, Thomas et al. (2013). “Fast, accurate detection of 100,000 object classes on
a single machine”. In: _Proceedings of the IEEE Conference on Computer Vision
and Pattern Recognition_ , pp. 1814–1821.

Deng, Jia, Alexander C Berg, et al. (2010). “What does classifying more than 10,000
image categories tell us?” In: _European conference on computer vision_. Springer,
pp. 71–84.

Deng, Jia, Wei Dong, et al. (2009). “Imagenet: A large-scale hierarchical image
database”. In: _Computer Vision and Pattern Recognition, 2009. CVPR 2009.
IEEE Conference on_. IEEE, pp. 248–255.

Deng, Jia, Jonathan Krause, et al. (2012). “Hedging your bets: Optimizing accuracy-
specificity trade-offs in large scale visual recognition”. In: _Computer Vision and
Pattern Recognition (CVPR), 2012 IEEE Conference on_. IEEE, pp. 3450–3457.

Deng, Jia, Sanjeev Satheesh, et al. (2011). “Fast and balanced: Efficient label tree
learning for large scale object recognition”. In: _Advances in Neural Information
Processing Systems_ , pp. 567–575.

Denil, Misha et al. (2013). “Predicting parameters in deep learning”. In: _Advances
in neural information processing systems_ , pp. 2148–2156.

Denton, Emily L et al. (2014). “Exploiting linear structure within convolutional
networks for efficient evaluation”. In: _Advances in neural information processing
systems_ , pp. 1269–1277.

Erlenkotter, Donald (1978). “A dual-based procedure for uncapacitated facility lo-
cation”. In: _Operations Research_ 26.6, pp. 992–1009.


Gao, Tianshi and Daphne Koller (2011). “Discriminative learning of relaxed hierar-
chy for large-scale visual recognition”. In: _Computer Vision (ICCV), 2011 IEEE
International Conference on_. IEEE, pp. 2072–2079.

Gionis, Aristides, Piotr Indyk, Rajeev Motwani, et al. (1999). “Similarity search in
high dimensions via hashing”. In: _Vldb_. Vol. 99. 6, pp. 518–529.

Gong, Yunchao et al. (2014). “Compressing deep convolutional networks using
vector quantization”. In: _arXiv preprint arXiv:1412.6115_.

Gregor, Karo and Yann LeCun (2010). “Emergence of complex-like cells in a tempo-
ral product network with local receptive fields”. In: _arXiv preprint arXiv:1006.0448_.

Griffin, Gregory and Pietro Perona (2008). “Learning and using taxonomies for
fast visual categorization”. In: _Computer Vision and Pattern Recognition, 2008.
CVPR 2008. IEEE Conference on_. IEEE, pp. 1–8.

Guo, Yiwen, Anbang Yao, and Yurong Chen (2016). “Dynamic network surgery for
efficient dnns”. In: _Advances In Neural Information Processing Systems_ , pp. 1379–
1387.

Han, Song, Huizi Mao, and William J Dally (2016). “Deep compression: Com-
pressing deep neural networks with pruning, trained quantization and huffman
coding”. In: _ICLR_.

Han, Song, Jeff Pool, et al. (2015). “Learning both weights and connections for
efficient neural network”. In: _Advances in neural information processing systems_ ,
pp. 1135–1143.

Hassibi, Babak and David G Stork (1993). “Second order derivatives for network
pruning: Optimal brain surgeon”. In: _Advances in neural information processing
systems_ , pp. 164–171.

Hinton, Geoffrey, Oriol Vinyals, and Jeff Dean (2014). “Distilling the knowledge in
a neural network”. In: _NIPS Deep Learning Workshop_.

Hoffer, Elad, Itay Hubara, and Daniel Soudry (2018). “Fix your classifier: the
marginal value of training the last weight layer”. In: _arXiv preprint arXiv:1801.04540_.

Howard, Andrew G et al. (2017). “Mobilenets: Efficient convolutional neural net-
works for mobile vision applications”. In: _arXiv preprint arXiv:1704.04861_.

Jaderberg, Max, Andrea Vedaldi, and Andrew Zisserman (2014). “Speeding up
convolutional neural networks with low rank expansions”. In: _arXiv preprint
arXiv:1405.3866_.

Jain, Kamal, Mohammad Mahdian, and Amin Saberi (2002). “A new greedy ap-
proach for facility location problems”. In: _Proceedings of the thiry-fourth annual
ACM symposium on Theory of computing_. ACM, pp. 731–740.

Jin, Jonghoon, Aysegul Dundar, and Eugenio Culurciello (2014). “Flattened con-
volutional neural networks for feedforward acceleration”. In: _arXiv preprint
arXiv:1412.5474_.


Jin, Xiaojie et al. (2018). “WSNet: Compact and Efficient Networks Through Weight
Sampling”. In: _International Conference on Machine Learning_ , pp. 2357–2366.

Jouppi, Norman P et al. (2017). “In-datacenter performance analysis of a tensor
processing unit”. In: _Computer Architecture (ISCA), 2017 ACM/IEEE 44th Annual
International Symposium on_. IEEE, pp. 1–12.

Krasin, Ivan et al. (2017). “OpenImages: A public dataset for large-scale multi-label
and multi-class image classification.” In:

Krizhevsky, Alex and Geoffrey E Hinton (2011). “Using very deep autoencoders for
content-based image retrieval.” In: _ESANN_.

Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E Hinton (2012). “ImageNet Clas-
sification with Deep Convolutional Neural Networks.” In: _NIPS_.

Lebedev, Vadim et al. (2014). “Speeding-up convolutional neural networks using
fine-tuned cp-decomposition”. In: _arXiv preprint arXiv:1412.6553_.

LeCun, Yann, John S Denker, and Sara A Solla (1990). “Optimal brain damage”.
In: _Advances in neural information processing systems_ , pp. 598–605.

Levy, Daniel, Danlu Chan, and Stefano Ermon (2018). “LSH Softmax: Sub-Linear
Learning and Inference of the Softmax Layer in Deep Architectures”. In:

Li, Hao et al. (2017). “Pruning Filters for Efficient ConvNets”. In: _International
Conference on Learning Representations_.

Liu, Baoyuan et al. (2013). “Probabilistic label trees for efficient large scale image
classification”. In: _Proceedings of the IEEE conference on computer vision and
pattern recognition_ , pp. 843–850.

Mamalet, Franck and Christophe Garcia (2012). “Simplifying convnets for fast
learning”. In: _International Conference on Artificial Neural Networks_. Springer,
pp. 58–65.

Marszałek, Marcin and Cordelia Schmid (2008). “Constructing category hierarchies
for visual recognition”. In: _European conference on computer vision_. Springer,
pp. 479–491.

Masana, Marc et al. (2017). “Domain-Adaptive Deep Network Compression”. In:
_The IEEE International Conference on Computer Vision (ICCV)_.

McGill, Mason and Pietro Perona (2017). “Deciding how to decide: Dynamic routing
in artificial neural networks”. In: _Proceedings of the 34th International Confer-
ence on Machine Learning-Volume 70_. JMLR. org, pp. 2363–2372.

Mikolov, Tomas et al. (2013). “Distributed representations of words and phrases and
their compositionality”. In: _Advances in neural information processing systems_ ,
pp. 3111–3119.

Moczulski, Marcin et al. (2015). “ACDC: A structured efficient linear layer”. In:
_arXiv preprint arXiv:1511.05946_.


Morin, Frederic and Yoshua Bengio (2005). “Hierarchical probabilistic neural net-
work language model.” In: _Aistats_. Vol. 5. Citeseer, pp. 246–252.

Mussmann, Stephen and Stefano Ermon (2016). “Learning and inference via maxi-
mum inner product search”. In: _International Conference on Machine Learning_ ,
pp. 2587–2596.

Mussmann, Stephen, Daniel Levy, and Stefano Ermon (2017). “Fast amortized
inference and learning in log-linear models with randomly perturbed nearest
neighbor search”. In: _arXiv preprint arXiv:1707.03372_.

Ordonez, Vicente et al. (2013). “From large scale image categorization to entry-level
categories”. In: _Proceedings of the IEEE International Conference on Computer
Vision_ , pp. 2768–2775.

Rastegari, Mohammad et al. (2016). “Xnor-net: Imagenet classification using binary
convolutional neural networks”. In: _European Conference on Computer Vision_.
Springer, pp. 525–542.

Sainath, Tara N et al. (2013). “Low-rank matrix factorization for deep neural network
training with high-dimensional output targets”. In: _Acoustics, Speech and Signal
Processing (ICASSP), 2013 IEEE International Conference on_. IEEE, pp. 6655–
6659.

Sandler, Mark et al. (2018). “MobileNetV2: Inverted Residuals and Linear Bottle-
necks”. In: _Proceedings of the IEEE Conference on Computer Vision and Pattern
Recognition_ , pp. 4510–4520.

Sindhwani, Vikas, Tara Sainath, and Sanjiv Kumar (2015). “Structured transforms
for small-footprint deep learning”. In: _Advances in Neural Information Processing
Systems_ , pp. 3088–3096.

Szegedy, Christian et al. (2016). “Rethinking the inception architecture for computer
vision”. In: _Proceedings of the IEEE Conference on Computer Vision and Pattern
Recognition_ , pp. 2818–2826.

Thomee, Bart et al. (2016). “YFCC100M: The new data in multimedia research”.
In: _Communications of the ACM_ 59.2, pp. 64–73.

Van Horn, Grant, Steve Branson, et al. (2015). “Building a bird recognition app and
large scale dataset with citizen scientists: The fine print in fine-grained dataset
collection”. In: _Proceedings of the IEEE Conference on Computer Vision and
Pattern Recognition_ , pp. 595–604.doi:10.1109/CVPR.2015.7298658.

Van Horn, Grant, Oisin Mac Aodha, et al. (2018). “The inaturalist species clas-
sification and detection dataset”. In: _Computer Vision and Pattern Recognition
(CVPR)_.

Vanhoucke, Vincent, Andrew Senior, and Mark Z Mao (2011). “Improving the
speed of neural networks on CPUs”. In: _Proc. Deep Learning and Unsupervised
Feature Learning NIPS Workshop_. Vol. 1. Citeseer, p. 4.


Vijayanarasimhan, Sudheendra et al. (2015). “Deep networks with large output
spaces”. In: _Workshop for International Conference on Learning Representations_.

Weston, Jason, Samy Bengio, and Nicolas Usunier (2011). “Wsabie: Scaling up to
large vocabulary image annotation”. In: _IJCAI_. Vol. 11, pp. 2764–2770.

Xue, Jian, Jinyu Li, and Yifan Gong (2013). “Restructuring of deep neural network
acoustic models with singular value decomposition.” In: _Interspeech_ , pp. 2365–
2369.

Yagnik, Jay et al. (2011). “The power of comparative reasoning”. In: _Computer
Vision (ICCV), 2011 IEEE International Conference on_. IEEE, pp. 2431–2438.

Yan, Zhicheng et al. (2015). “HD-CNN: hierarchical deep convolutional neural
networks for large scale visual recognition”. In: _ICCV_.

Yang, Zichao et al. (2015). “Deep fried convnets”. In: _Proceedings of the IEEE
International Conference on Computer Vision_ , pp. 1476–1483.

Zhang, Xiangyu et al. (2018). “ShuffleNet: An Extremely Efficient Convolutional
Neural Network for Mobile Devices”. In: _Proceedings of the IEEE conference on
computer vision and pattern recognition_.

Zhou, Hao, Jose M Alvarez, and Fatih Porikli (2016). “Less is more: Towards
compact cnns”. In: _European Conference on Computer Vision_. Springer, pp. 662–
677.


