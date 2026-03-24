# Chapter 8: Conclusions And Future Directions

In this thesis, motivated by the desire to build Visipedia, I, along with my coauthors,
have contributed work aimed at improving our ability to collect computer vision
datasets. Using iNaturalist and the Cornell Lab of Ornithology as case studies,
we have explored how to interact with motivated enthusiasts, experts, and paid
crowdworkers to collect and combine the necessary data to train computer vision
models to answer visual questions. While the current applications help users around
the world, there is still plenty of work to be done before we can achieve the vision
of Visipedia laid out by Perona (Perona, 2010).

The Merlin and iNaturalist apps are image classification apps. They process the
whole image, or a region selected by the user, and return a list of candidate species.
This is the behavior expected by the user, but it is far from the behavior envisioned
by Perona. The image does not become interactive and therefore does not allow the
user to readily answer additional visual questions past, “What species is this?” I
would argue it is not computer vision models (Krizhevsky, Sutskever, and Hinton,
2012; Ren et al., 2017; He et al., 2017) or efficient annotation tools (Branson,
Van Horn, and Perona, 2017) that are missing, rather it is collecting the right type
of data and conveniently rendering the information for the user. Essentially, we are
still missing the two important interfaces that Visipedia must provide: an interface
that allows experts to share their knowledge, and an interface that allows users to
answer visual questions.

The expert interface must allow experts to easily contribute their knowledge. I
would imagine that this type of interface would let experts browse a taxonomy of
annotation types (imagine a taxonomy of objects and options for annotating different
parts of those objects), giving them the ability to add additional nodes and options as
needed, and encouraging them to annotate at the finest possible level. Annotations
could be simple keypoints, boxes, lines, segmentations, or any one of the standard
drawing tools now available on mapping interfaces or image editing interfaces. A
more complicated, but more powerful, annotation interface would let the expert
mark the correspondences with a 3D model of the object (in addition to allowing
them to create the 3D models). This would allow structure and constraints to be


enforced in a vision model, and would also allow future annotations to be efficiently
propagated (e.g. interpolate between two existing parts to annotate a third part that
occurs between them). Interesting academic questions that arise here include: (1)
how to select which images to ask the expert to annotate under the constraints
of time, expert cost, and expected gain; (2) when to ask the expert to refine the
taxonomy of annotations or 3D models due to ambiguity; and (3) how to propagate
information down the taxonomy when new subtrees are added.

The question-answering interface is more devious than it seems at first. In Perona’s
vision (Perona, 2010), there are many automata that can annotate an image. How
should we decide which automata to use when processing an image from a user?
Given a collection of automata’s output, how can we combine their information and
render it on the image? I believe a taxonomy will help us here too. Rather than
an image being analyzed once, I think it should be analyzed repeatedly based on
the interactions of the user. If an image contains a hummingbird eating the nectar
of a flower, rather than immediately covering the image with tens or hundreds of
clickable regions, the user should be able to provide their intention by clicking
on the bird or the flower. Then the component regions of that object should be
rendered. This process should continue until the user has clicked on the hyperlink
for a component region (taking them to Wikipedia). Interesting academic questions
that arise here include: (1) how to manage a taxonomy of automata capable of
producing component regions for an image (i.e. mitigate duplicates or conflicts);
(2) how to efficiently traverse the taxonomy of automata when processing an image;
and (3) how to assist the user when their target region of the image never becomes
clickable or the rendered components are inaccurate.

The third academic question from the previous paragraph is relevant to the current
versions of the Merlin and iNaturalist apps. Often, the list of results returned by the
classifiers contains the correct species, however it the user’s job to sift through the
example images to identify the match. This process could be drastically improved
by designing better human-machine interfaces. The human visual system is very
powerful, and can often augment the capabilities of the vision classifier, yet it is
currently ignored. Returning to a twenty questions style interface would help users
resolve ambiguous situations or continue their search in the wake of failed automata
results. Taking the work from Branson et al. (Branson, Van Horn, Wah, et al., 2014)
and updating it for use with convolutional networks and taxonomies would be a
useful step forward.


**References**

Branson, Steve, Grant Van Horn, and Pietro Perona (2017). “Lean Crowdsourcing:
Combining Humans and Machines in an Online System”. In: _Proceedings of the
IEEE Conference on Computer Vision and Pattern Recognition_ , pp. 7474–7483.
doi:10.1109/CVPR.2017.647.

Branson, Steve, Grant Van Horn, Catherine Wah, et al. (2014). “The ignorant led by
the blind: A hybrid human–machine vision system for fine-grained categoriza-
tion”. In: _International Journal of Computer Vision_ 108.1-2, pp. 3–29.

He, Kaiming et al. (2017). “Mask r-cnn”. In: _Computer Vision (ICCV), 2017 IEEE
International Conference on_. IEEE, pp. 2980–2988.

Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E Hinton (2012). “ImageNet Clas-
sification with Deep Convolutional Neural Networks.” In: _NIPS_.

Perona, Pietro (2010). “Vision of a Visipedia”. In: _Proceedings of the IEEE_ 98.8,
pp. 1526–1534.

Ren, Shaoqing et al. (2017). “Faster r-cnn: Towards real-time object detection with
region proposal networks”. In: _PAMI_.


ProQuest Number:
INFORMATION TO ALL USERS
The quality and completeness of this reproduction is dependent on the quality
and completeness of the copy made available to ProQuest.
Distributed by ProQuest LLC ( ).
Copyright of the Dissertation is held by the Author unless otherwise noted.
This work may be used in accordance with the terms of the Creative Commons license
or other rights statement, as indicated in the copyright statement or in the metadata
associated with this work. Unless otherwise specified in the copyright statement
or the metadata, all rights are reserved by the copyright holder.

This work is protected against unauthorized copying under Title 17,
United States Code and other applicable copyright laws.
Microform Edition where available © ProQuest LLC. No reproduction or digitization
of the Microform Edition is authorized without permission of ProQuest LLC.
ProQuest LLC
789 East Eisenhower Parkway
P.O. Box 1346
Ann Arbor, MI 48106 - 1346 USA
30546868
2023

