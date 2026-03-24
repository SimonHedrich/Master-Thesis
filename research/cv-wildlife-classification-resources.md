# Computer Vision for Wildlife Classification - Reading List

These papers/articles focus on computer vision applied to wildlife classification, rather than embedded ML. This is an important angle to consider as background for the thesis.

## Resources

### 1. To Crop or Not to Crop (Gadot, 2024)

- **Paper:** [To crop or not to crop: Comparing whole-image and cropped classification on a large dataset of camera trap images](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/cvi2.12284) - IET Computer Vision, Wiley Online Library
- **Notes:** Compares whole-image and cropped classification approaches on a large camera trap image dataset.

### 2. iNaturalist Competitions

- **Link:** [inat_comp/2021 at master - visipedia/inat_comp - GitHub](https://github.com/visipedia/inat_comp/tree/master/2021)
- **Notes:** iNaturalist is probably the leading platform for collecting and annotating images of wildlife and they regularly hold competitions for their datasets. Read up on the results of those competitions.

### 3. iNat Geo Model

- **Link:** [Introducing the iNaturalist Geomodel - iNaturalist](https://www.inaturalist.org/blog/84570-introducing-the-inaturalist-geomodel)
- **Notes:** Relevant for the idea of adding geographic information (e.g., one or two digits of a geohash) as model input. The iNat approach is more complex, but provides good background on incorporating location data into species classification.

### 4. Towards a Visipedia (PhD Thesis)

- **Link:** [Towards a Visipedia: Combining Computer Vision and Communities of Experts - ProQuest](https://www.proquest.com/docview/2572430408)
- **Notes:** A PhD thesis that provides excellent background for computer vision for species recognition. Not expected to read the whole thesis, but at least reading the abstract of each paper in it would give a solid background on the topic.

### 5. BioCLIP

- **Paper:** [BioCLIP: A Vision Foundation Model for the Tree of Life](https://arxiv.org/abs/2311.18803)
- **Notes:** Interesting approach for encoding hierarchical taxonomic information into a vision model.

### 6. Synthetic Examples for Rare Classes (Beery et al., WACV 2020)

- **Paper:** [Synthetic Examples Improve Generalization for Rare Classes](https://openaccess.thecvf.com/content_WACV_2020/papers/Beery_Synthetic_Examples_Improve_Generalization_for_Rare_Classes_WACV_2020_paper.pdf) - WACV 2020
- **Notes:** Explores data augmentation with synthetic examples for rare classes. Probably not as relevant if the thesis is not focusing on data augmentation, but still a cool approach.

### 7. Training with Few Example Images (PeerJ)

- **Paper:** [Accurate image-based identification of macroinvertebrate specimens using deep learning — How much training data is needed?](https://peerj.com/articles/17837/) - PeerJ
- **Notes:** Investigates how much training data is needed for accurate species identification in ecology. Relevant for understanding few-shot / low-data training scenarios.
