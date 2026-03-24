# Chapter 1: Introduction

Visipedia, a community-generated visual encyclopedia, is the primary motivator and
inspiration for the work in this thesis. The Visipedia project 1 has been spearheaded
by Pietro Perona’s group at Caltech and Serge Belongie’s group, first at UCSD and
then Cornell Tech. This thesis is the most recent in a series of theses (Welinder,
2012; Branson, 2012; Wah, 2014), coming out of Perona and Belongie’s respective
groups, that attempts to make Visipedia a reality. In (Perona, 2010), Perona specifies
the vision for Visipedia, defines the users and challenges of such a system, and muses
on its feasibility. He identifies two primary interfaces that Visipedia must provide.

First, Visipedia must provide an interface that allows users to ask visual questions.
Perona imagines an interface that can segment a photograph into its meaningful
component regions and then associates each of those regions with their correspond-
ing Wikipedia entry or to the same region in vast collections of photographs. This
would enable a user to photograph a rock pigeon and then click on the _operculum_
(i.e. the white, fleshy part at the base of the bill) to learn more about what purpose
that structure serves. Similarly, this type of interface would enable a user to nav-
igate to the Wikipedia page for _Amanita pantherina_ simply from a photograph of
that fungus. Similar types of interactions could be had with photographs outside
the natural world: a photograph of a painting could be annotated with the artist’s
information; a photograph of a car engine could be annotated with the engine part
names and replacement information; a photograph of a retina could be annotated
with defects and linked to similar clinical cases.

To provide answers to the visual queries discussed in the previous paragraph, Visi-
pedia must first be made aware of the visual properties of the world and their
relationships. This is the second primary interaction with Visipedia: an interface
that allows experts to share their visual knowledge of the world. Perona imagines
an easy-to-use annotation interface that allows experts to contribute their visual
knowledge by annotating a few paradigmatic images. An ornithologist could pro-
vide information on bird morphology for a few different families. A mycologist
could provide example photographs of fungi species. A Chevrolet engineer could

(^1) [http://www.visipedia.org](http://www.visipedia.org)


provide engine part schematics. An ophthalmologist could provide example retina
images along with their prognosis. Perona emphasizes the importance of making
this interface easy and quick, as experts’ time is scarce and valuable, and annotating
all of the important regions of an image is laborious and boring.

A high degree of automation is required to power the interactions that make these
two interfaces useful. A user, with a fleeting curiosity to identify the white, fleshy bit
of a pigeon, would prefer immediate results rather than waiting for a human expert
to answer. Similarly, an ornithologist would not annotate thousands of images
with the anatomical parts of a bird. Instead, we would prefer if machines could
analyze images and immediately return results and efficiently propagate information
from tens of examples to thousands or millions of photographs. This introduces
two additional types of people that would interact with Visipedia: annotators and
machine vision researchers.

Annotators, or “eye balls for hire” (Perona, 2010), are the bridge connecting the
few samples provided by an expert to the thousands of examples required to train
modern computer vision models. Annotators could be paid crowd workers (e.g.
Amazon Mechanical Turk workers), they could be motivated enthusiasts (e.g. citizen
scientists) or they could be people tasked with doing a few annotations while trying to
achieve another goal (e.g. GWAPs (Von Ahn and Dabbish, 2004) or Captchas (Von
Ahn, Maurer, et al., 2008)). In any case, their job is to propagate the expert
information to additional training data that can be used to train a computer vision
model to do the task. Machine vision researchers are responsible for designing and
implementing these computer vision models. These models are then responsible
for annotating an image with hyperlinks that allow users to answer visual questions
(i.e. clickable component regions), working with experts to efficiently incorporate
their visual knowledge, and propagating expert information to additional images
(effectively annotating the images of the web).

At this point, we have defined Visipedia as a community-generated visual encyclo-
pedia that has interfaces to answer visual questions and that enable experts to share
their visual knowledge. Annotators help propagate expert knowledge to additional
images, producing datasets that can be used to train computer vision models designed
by machine vision researchers. These same models power the question-answering
interface and interact with experts to efficiently incorporate their knowledge. In
(Perona, 2010), Perona discusses the challenges of actually building Visipedia from
the perspective of a computer vision researcher in 2009. He noted that computer


vision models at the time were not capable of performing at the level of accuracy
necessary to be useful, and that the field had not yet attempted to build such com-
plex, heterogeneous systems. In addition, he observed that self-diagnosing models
(capable of deciding when they should ask questions of humans), active incremental
learning (necessary for learning in the large scale, dynamic web environment), and
human-machine interaction were research topics largely ignored by the computer
vision research community, yet crucial for Visipedia. It has been 9 years since
Perona penned his vision, where do we stand now?

Powered by the return of convolutional neural networks (Krizhevsky, Sutskever, and
Hinton, 2012), hardware advancements and easy-to-use computational libraries (Mar-
tin Abadi et al., 2015), the computer vision field as a whole has made incredible
progress during my graduate studies on the tasks of image classification (Krizhevsky,
Sutskever, and Hinton, 2012), object detection (Ren et al., 2017), keypoint localiza-
tion (Chen et al., 2018), and image segmentation (He et al., 2017). Indeed, setting
aside the feasibility of collecting training data, the costs of training, and the size
of the resulting model, if a sufficiently large dataset can be collected for one the
previous tasks, then often the performance of the resulting convolutional neural
network model is adequate for production usage. Evidence of this progress can be
seen in the availability of computer vision-powered applications now available to
consumers. During my graduate career, I helped build two of these applications
(iNaturalist and Merlin), available through the Google Play Store and and Apple
App Store, that help users identify species in their photographs. The iNaturalist
app 2 has a server-based computer vision classifier that can help identify 25 , 000
species. The Merlin app 3 has a computer vision classifier available directly on the
phone and can help users classify 2 , 000 bird species. Besides mobile applications,
perhaps the most impressive sign of progress is the availability of self-driving cars
(albeit limited in their scope for now) becoming available to consumers.

Computer vision has entered an era of big data, where the ability to collect larger
datasets – larger in terms of the number of classes, the number of images per class,
and the level of annotation per image – appears to be paramount for continuing
performance improvement and expanding the set of solvable applications. However,
while the accuracy of computer vision models has seen a rapid improvement over the
last half-decade, our ability to collect datasets of sufficient size to train and evaluate
these models has remained essentially unchanged and presents a significant hurdle

(^2) https://www.inaturalist.org
(^3) [http://merlin.allaboutbirds.org/](http://merlin.allaboutbirds.org/)


to expanding the availability and utility of computer vision services. Indeed, the
title of this thesis is “Towards a Visipedia,” not “Visipedia: Mission Accomplished.”
So in the interim period between (Perona, 2010) and this thesis, many of the key
challenges to actually building a Visipedia (namely the challenges associated with
annotating large datasets) were still largely ignored (excluding the contributions of
the previous theses on Visipedia). The work in this thesis, however, is aimed at
reducing the burden of collecting datasets and will hopefully lay the foundation for
building a Visipedia.

In (Perona, 2010), Perona suggested that a step towards integrating a Visipedia with
all of Wikipedia would be to focus on a well-defined domain with a community
of highly motivated enthusiasts. This is precisely what we have done by engag-
ing with the birding community through the Cornell Lab of Ornithology and the
naturalist community through iNaturalist. The following chapters, each of which
is self-contained, explore dataset properties, efficient methods of collection, and
training state-of-the-art methods for deploying classification services to these two
communities. In terms of building a broader Visipedia, the following chapters con-
tain useful information for interacting with different types of annotators, modeling
the skills of annotators and vision models, and how to reliably combine informa-
tion from multiple sources (both human and machine). Taken as whole, this thesis
is an attempt to fill in the missing pieces that provide the required automation to
make Visipedia a reality. I will briefly summarize the chapters and the relevant
contributions.

In Chapter 1, we discuss the long tail property of real world datasets and the
effect this tail has on classification performance. Experiments show that state-
of-the-art methods do not share feature learning between classes and that new
training methodologies or collecting additional data in the tail is required to improve
performance.

In Chapter 2, we devise a method for online crowdsourcing of binary labels, key-
points, and multi-instance bounding box annotations. This method is capable of
estimating worker skills and jointly trains computer vision models. We present ex-
periments that show significant cost savings and improvements in dataset accuracy
by using our model instead of traditional dataset collection techniques.

In Chapter 3, we extend our online crowdsourcing method to large-scale multiclass
annotations. Our method is capable of utilizing a taxonomy across the labels,
handling a dependence between the annotations, and jointly training a computer


vision system. We present experiments that show significant accuracy gains over
traditional majority vote techniques.

In Chapter 4, we present the NABirds dataset, collected by the birding community
through the Cornell Lab of Ornithology. We present experiments comparing the
annotation performance of different groups of workers on different types of tasks.
We describe the benefit of tapping into a motivated community and how to best
harness its enthusiasm. We additionally present results on dataset noise and show
that modern state-of-the-art methods are resilient to a reasonable amount of noise.

In Chapter 5, we present the iNaturalist Species Classification and Detection Dataset,
collected by the naturalist community through iNaturalist. We describe dataset col-
lection and prepping methods and evaluate state-of-the art classifiers and detectors.
In addition, we conduct a competition to motivate the computer vision research
community to explore large-scale, fine-grained classification and detection.

In Chapter 6, we analyze multiple techniques for reducing the computational bur-
den of the final fully connected layer of traditional convolutional networks. We
experiment with a novel taxonomic approach but find that a simple factorization and
training scheme allows us to reduce the amount of computation and memory by 25x
without any loss in accuracy.

Finally, in Chapter 7, I suggest directions for future work.

**References**

Branson, Steven (2012). “Interactive learning and prediction algorithms for com-
puter vision applications”. PhD thesis. UC San Diego.

Chen, Yilun et al. (2018). “Cascaded pyramid network for multi-person pose esti-
mation”. In: _CVPR_.

He, Kaiming et al. (2017). “Mask r-cnn”. In: _Computer Vision (ICCV), 2017 IEEE
International Conference on_. IEEE, pp. 2980–2988.

Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E Hinton (2012). “ImageNet Clas-
sification with Deep Convolutional Neural Networks.” In: _NIPS_.

Martin Abadi et al. (2015). _TensorFlow: Large-Scale Machine Learning on Het-
erogeneous Systems_. Software available from tensorflow.org. url: http : / /
tensorflow.org/.

Perona, Pietro (2010). “Vision of a Visipedia”. In: _Proceedings of the IEEE_ 98.8,
pp. 1526–1534.


Ren, Shaoqing et al. (2017). “Faster r-cnn: Towards real-time object detection with
region proposal networks”. In: _PAMI_.

Von Ahn, Luis and Laura Dabbish (2004). “Labeling images with a computer
game”. In: _Proceedings of the SIGCHI conference on Human factors in computing
systems_. ACM, pp. 319–326.

Von Ahn, Luis, Benjamin Maurer, et al. (2008). “recaptcha: Human-based character
recognition via web security measures”. In: _Science_ 321.5895, pp. 1465–1468.

Wah, Catherine Lih-Lian (2014). “Leveraging Human Perception and Computer Vi-
sion Algorithms for Interactive Fine-Grained Visual Categorization”. PhD thesis.
UC San Diego.

Welinder, Nils Peter Egon (2012). “Hybrid human-machine vision systems : image
annotation using crowds, experts and machines”. PhD thesis. California Institute
of Technology.


