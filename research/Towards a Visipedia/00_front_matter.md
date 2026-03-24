# Towards a Visipedia: Combining Computer Vision and Communities of Experts

Thesis by
Grant Van Horn

In Partial Fulfillment of the Requirements for the
Degree of
Doctor of Philosophy
CALIFORNIA INSTITUTE OF TECHNOLOGY

Pasadena, California
2019

Defended September 7, 2018

© 2019

Grant Van Horn
ORCID: 0000-0003-2953-

All rights reserved

## Acknowledgements

First and foremost I would like to thank my parents, Mathew and Mary Van Horn.
I dedicate this work to them.

I would like to thank my advisor, Pietro Perona. I hope that at least a fraction of his
curiosity for the world has worn off on me. It has been a pleasure to be a Slacker in
his lab.

I would like to thank Serge Belongie, who I consider a joint advisor. I owe most
of my opportunities from the last decade to Serge, and owe my current path in life
to his guidance. I am forever grateful that he advised me during my undergraduate
career, my Masters, and my Ph.D.

I would like to thank Steve Branson, easily my most influential mentor. From Steve,
I learned how to identify problems, how to conduct research, and how to discuss
the findings. I owe a lot to him, and I am very proud of the work we accomplished
together.

I would like to thank Jessie Barry and the rest of her team at the Cornell Lab of
Ornithology. I would also like to thank Scott Loarie and the rest of the iNaturalist
team. I am grateful that I could work on both the Merlin and iNaturalist applications
during my graduate studies, helping me connect my passion for computer science
with my passion for the outdoors.

Finally, I owe my most enjoyable moments at Caltech to the Vision Lab, and I would
like to thank all the Slackers I overlapped with: David Hall, Oisin Mac Aodha,
Matteo Ruggero Ronchi, Joe Marino, Ron Appel, Mason McGill, Sara Beery,
Alvita Tran, Natalie Bernat, Eyrun Eyjolfsdottir, Bo Chen, Krzysztof Chałupka,
Serim Ryou, Cristina Segalin, Eli Cole, Jennifer Sun, Jan Dirk Wegner, Daniel
Laumer, Michael Maire, Xavier Burgos-Artizzu, Conchi Fernandez, Louise Naud,
Michele Damian, and Genevieve Patterson.


## Abstract

Motivated by the idea of a Visipedia, where users can search and explore by image,
this thesis presents tools and techniques for empowering expert communities through
computer vision. The collective aim of this work is to provide a scalable foundation
upon which an application like Visipedia can be built. We conduct experiments
using two highly motivated communities, the birding community and the naturalist
community, and report results and lessons on how to build the necessary components
of a Visipedia. First, we conduct experiments analyzing the behavior of state-of-the-
art computer vision classifiers on long tailed datasets. We find poor feature sharing
between classes, potentially limiting the applicability of these models and empha-
sizing the ability to intelligently direct data collection resources. Second, we devise
online crowdsourcing algorithms to make dataset collection for binary labels, multi-
class labels, keypoints, and mulit-instance bounding boxes faster, cheaper, and more
accurate. These methods jointly estimate labels, worker skills, and train computer
vision models for these tasks. Experiments show that we can achieve significant cost
savings compared to traditional data collection techniques, and that we can produce
a more accurate dataset compared to traditional data collection techniques. Third,
we present two fine-grained datasets, detail how they were constructed, and analyze
the test accuracy of state-of-the-art methods. These datasets are then used to create
applications that help users identify species in their photographs: Merlin, an app
assisting users in identifying birds species, and iNaturalist, an app that assists users
in identifying a broad variety of species. Finally, we present work aimed at reducing
the computational burden of large scale classification with the goal of creating an
application that allows users to classify tens of thousands of species in real time on
their mobile device. As a whole, the lessons learned and the techniques presented
in this thesis bring us closer to the realization of a Visipedia.


## Published Content And Contributions

Van Horn, Grant and Pietro Perona (2019). “Reducing Memory & Computation
Demands for Large Scale Visual Classification”.
G.V.H. participated in designing the project, developing the method, running the
experiments and writing the manuscript.

Van Horn, Grant, Steve Branson, Scott Loarie, et al. (2018). “Lean Multiclass
Crowdsourcing”. In: _Proceedings of the IEEE Conference on Computer Vision
and Pattern Recognition_. Salt Lake City, UT.doi:10.1109/cvpr.2018.00287.
G.V.H. participated in designing the project, developing the method, running the
experiments and writing the manuscript.

Van Horn, Grant, Oisin Mac Aodha, et al. (2018). “The iNaturalist Species Clas-
sification and Detection Dataset”. In: _Proceedings of the IEEE Conference on
Computer Vision and Pattern Recognition_. Salt Lake City, UT.doi:10.1109/
CVPR.2018.00914.
G.V.H. participated in designing the project, developing the method, running the
experiments and writing the manuscript.

Branson, Steve, Grant Van Horn, and Pietro Perona (2017). “Lean Crowdsourcing:
Combining Humans and Machines in an Online System”. In: _Proceedings of the
IEEE Conference on Computer Vision and Pattern Recognition_ , pp. 7474–7483.
doi:10.1109/CVPR.2017.647.
G.V.H. participated in designing the project, developing the method, running the
experiments and writing the manuscript.

Van Horn, Grant and Pietro Perona (2017). “The Devil is in the Tails: Fine-grained
Classification in the Wild”. In: _arXiv preprint arXiv:1709.01450_ .url:https:
//arxiv.org/abs/1709.01450.
G.V.H. participated in designing the project, developing the method, running the
experiments and writing the manuscript.

Van Horn, Grant, Steve Branson, Ryan Farrell, et al. (2015). “Building a bird
recognition app and large scale dataset with citizen scientists: The fine print
in fine-grained dataset collection”. In: _Proceedings of the IEEE Conference on
Computer Vision and Pattern Recognition_ , pp. 595–604.doi:10.1109/CVPR.
2015.7298658.
G.V.H. participated in designing the project, developing the method, running the
experiments and writing the manuscript.


## Table of Contents

Acknowledgements............................... iii
Abstract..................................... iv
Published Content and Contributions...................... v


- Chapter I: Introduction Table of Contents vi
- Chapter II: The Devil is in the Tails: Fine-grained Classification in the Wild
   - 2.1 Abstract
   - 2.2 Introduction
   - 2.3 Related Work
   - 2.4 Experiment Setup
   - 2.5 Experiments
   - 2.6 Discussion and Conclusions
   - Online System Chapter III: Lean Crowdsourcing: Combining Humans and Machines in an
   - 3.1 Abstract
   - 3.2 Introduction
   - 3.3 Related Work
   - 3.4 Method
   - 3.5 Models For Common Types of Annotations
   - 3.6 Binary Annotation
   - 3.7 Part Keypoint Annotation
   - 3.8 Multi-Object Bounding Box Annotations
   - 3.9 Experiments
   - 3.10 Conclusion
- Chapter IV: Lean Multiclass Crowdsourcing
   - 4.1 Abstract
   - 4.2 Introduction
   - 4.3 Related Work
   - 4.4 Multiclass Online Crowdsourcing
   - 4.5 Taking Pixels into Account
   - 4.6 Experiments
   - 4.7 Conclusion
   - scientists: Thefine printin fine-grained dataset collection Chapter V: Building a bird recognition app and large scale dataset with citizen
   - 5.1 Abstract
   - 5.2 Introduction
   - 5.3 Related Work
   - 5.4 Crowdsourcing with Citizen Scientists
   - 5.5 NABirds
   - 5.6 Annotator Comparison vii
   - 5.7 Measuring the Quality of Existing Datasets
   - 5.8 Effect of Annotation Quality & Quantity
   - 5.9 Conclusion
   - 5.10 Acknowledgments
- Chapter VI: The iNaturalist Species Classification and Detection Dataset
   - 6.1 Abstract
   - 6.2 Introduction
   - 6.3 Related Datasets
   - 6.4 Dataset Overview
   - 6.5 Experiments
   - 6.6 Conclusions and Future Work
   - Visual Classification Chapter VII: Reducing Memory & Computation Demands for Large Scale
   - 7.1 Abstract
   - 7.2 Introduction
   - 7.3 Related Work
   - 7.4 Taxonomic Parameter Sharing
   - 7.5 Experiments
   - 7.6 Conclusion
- Chapter VIII: Conclusions and Future Directions


