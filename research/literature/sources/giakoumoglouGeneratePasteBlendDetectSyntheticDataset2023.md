See discussions, stats, and author profiles for this publication at: https://www.researchgate.net/publication/371085479

## [Generate-Paste-Blend-Detect: Synthetic Dataset for Object Detection in the Agriculture Domain](https://www.researchgate.net/publication/371085479_Generate-Paste-Blend-Detect_Synthetic_Dataset_for_Object_Detection_in_the_Agriculture_Domain?enrichId=rgreq-0d285eac2ca49a03e5f478e21e79858a-XXX&enrichSource=Y292ZXJQYWdlOzM3MTA4NTQ3OTtBUzoxMTQzMTI4MTE2NTAzNDU3MUAxNjg1OTYwNjIxNDYw&el=1_x_3&_esc=publicationCoverPdf)

Article in Smart Agricultural Technology · May 2023

DOI: 10.1016/j.atech.2023.100258

CITATIONS

15

3 authors , including:

<!-- image -->

[Nikolaos Giakoumoglou](https://www.researchgate.net/profile/Nikolaos-Giakoumoglou?enrichId=rgreq-0d285eac2ca49a03e5f478e21e79858a-XXX&enrichSource=Y292ZXJQYWdlOzM3MTA4NTQ3OTtBUzoxMTQzMTI4MTE2NTAzNDU3MUAxNjg1OTYwNjIxNDYw&el=1_x_5&_esc=publicationCoverPdf)

[Imperial College London](https://www.researchgate.net/institution/ICL?enrichId=rgreq-0d285eac2ca49a03e5f478e21e79858a-XXX&enrichSource=Y292ZXJQYWdlOzM3MTA4NTQ3OTtBUzoxMTQzMTI4MTE2NTAzNDU3MUAxNjg1OTYwNjIxNDYw&el=1_x_6&_esc=publicationCoverPdf)

21 PUBLICATIONS 89 CITATIONS

[SEE PROFILE](https://www.researchgate.net/profile/Nikolaos-Giakoumoglou?enrichId=rgreq-0d285eac2ca49a03e5f478e21e79858a-XXX&enrichSource=Y292ZXJQYWdlOzM3MTA4NTQ3OTtBUzoxMTQzMTI4MTE2NTAzNDU3MUAxNjg1OTYwNjIxNDYw&el=1_x_7&_esc=publicationCoverPdf)

READS

103

<!-- image -->

Contents lists available at ScienceDirect

## Smart Agricultural Technology

journal homepage: www.journals.elsevier.com/smart-agricultural-technology

## Generate-Paste-Blend-Detect: Synthetic dataset for object detection in the agriculture domain

Nikolaos Giakoumoglou, Eleftheria Maria Pechlivani ∗ , Dimitrios Tzovaras

Information Technologies Institute, Centre for Research and Technology Hellas, Thessaloniki, Greece

## A R T I C L E I N F O

## A B S T R A C T

Editor: Spyros Fountas

Keywords: Synthetic data Deep learning Diffusion models Object detection Precision agriculture

## 1. Introduction

The rise of data-driven techniques, particularly deep learning, has made them the leading approach for solving computer vision problems in various fi elds. This has led to a need for annotated data to train these models. In agriculture, these techniques are used in applications such as yield prediction [1], disease detection [2], weed identification and management [3] among others. Each application demands training data in various forms, from simple image-level labels (classification) to complex pixel-level (segmentation) or instance-level (detection) labels.

Creating realistic synthetic datasets can aid in the application of artificial intelligence to agriculture because data annotation, especially at the pixel and instance level, is a costly and tedious process [4]. Transfer learning using these synthetic datasets can also enhance algorithm performance by modeling variations in environmental settings [5]. Moreover, working with synthetic data can unlock the potential for solving unlimited domain-specific problems, where collecting real data is either restricted, limited, or expensive [6].

Conventional methods for creating synthetic datasets in agricultural domains typically involve a lot of work, such as 3D rendering with physics modeling [7]. However, more recent developments include the use of generative models, particularly Generative Adversarial Networks

Object detection is a challenging task, hindered by the scarcity of large annotated datasets. In agriculture, the lack of annotated insect datasets often results in domain-specific models that lack generalization. Data collection and annotation can be expensive and time-consuming. This paper proposes a simple approach to generate synthetic datasets for object detection that requires only a small dataset of target objects and a larger background dataset that fi ts the desired environment. The approach named Generate-Paste-Blend-Detect uses Denoising Diffusion Probabilistic Models (DDPM) to artificially 'generate' objects, 'paste' them on a background image, 'blend' them with the environment to avoid pixel artifacts which result in poor performance for trained models, and fi nally use an object detection model to 'detect' the artificially added object instances. The proposed methodology is demonstrated in the agricultural domain to detect whiteflies achieving a mean average precision ( 𝑚𝐴𝑃 50 ) of 0 . 66 with the state-of-the-art YOLOv8 object detection model. This approach enables domain-specific detection with minimal labor and cost.

(GANs) [8], which can automatically generate realistic data. GANs can analyze the target dataset and generate similar images by taking new images from the source dataset as input. Moreover, Variational Autoencoders (VAEs) [9] have also been used to generate images by imposing constraints on the latent space where the latter representation allows VAEs to generate new images by sampling from the distribution learned. However, both GANs and VAEs require a large amount of data to learn the underlying patterns and distributions.

Recently, diffusion probabilistic models enabled image generation, sometimes better than other types of generative models [10]. In diffusion probabilistic models, a neural network is trained to denoise images blurred with Gaussian noise by learning to reverse the diffusion process [10].

In this study, a realistic synthetic dataset for the purpose of object detection is aimed to be created, based on a proposed methodology named Generate-Paste-Blend-Detect to detect insects in the agricultural domain. In this work, the proposed methodology is demonstrated for the detection of insect whitefly. The contribution of this paper is twofold and aims to: i) propose a methodology that artificially generates object instances from randomly sampled noised and adds them to a background dataset that best represents the desired environmental settings for the purpose of object detection, and ii) conduct experiments

* Corresponding author. E-mail addresses: ngiakoumoglou@iti.gr (N. Giakoumoglou), riapechl@iti.gr (E.M. Pechlivani), dimitrios.tzovaras@iti.gr (D. Tzovaras).

Received 10 April 2023; Received in revised form 22 May 2023; Accepted 23 May 2023

<!-- image -->

<!-- image -->

for detection of the artificially added objects with state-of-the-art deep learning models in the agricultural domain.

The paper is structured as follows. Section 2 presents the related work in the fi eld of synthetically generated datasets in the agriculture domain. Section 3 reviews the relevant background. Section 4 presents our approach. Section 5 exhibits the experiments with the results. The study concludes in Section 6.

## 2. Related work

In computer vision, object detection is a well-studied problem. Shetty et al. (2021) [11] give a thorough review on several existing techniques and perform comparative analyses of these techniques. The latter models often require large amount of annotated data for the purpose of object detection. This requirement is both impractical and costly. Although there are several datasets in the agriculture domain [12-15], all of them are domain-specific: they address the problem of detecting specific insects or diseases in particular fi elds.

In order to address this issue, synthetic data generation comes in handy. Cicco et al. (2017) [7] developed a method for full 3D rendering of a sugar-beet and weed dataset with physics modeling of the leaves as well. Karam et al. (2022) [16] proposed a synthetic data generation pipeline based on GANs to artificially generate images to augment small datasets and tested the pipeline on a home-collected dataset of whiteflies on different crop types. Bi et al. (2020) [17] used Wasserstein GANs (W-GANs) to enhance classification accuracy on plant diseases.

Dwibedi et al. (2017) [18] present a simple approach to dataset generation that eliminates the complexity of computer rendering or generative network training. Their approach involves creating photo-realistic images by combining real images with real backgrounds, named 'Cut, Paste and Learn'. This straightforward approach enables the construction of large datasets from a combination of background images and images of the classes they wish. To address pixel artifacts and boost detection performance, the authors made use of blending rather than just copying images into the background. Jepsen and Jeppessen (2021) [19] used a modified version of Cut, Paste and Learn [18] to generate synthetic datasets in the fi eld of agriculture (weed detection).

## 3. Background

Object Detection : Object detection is a task in computer vision where the goal is to detect and differentiate individual objects within an image or a video. It involves identifying the presence of specific objects, such as cars, people, or buildings, and determining their locations within the image. Instance detection is a more specific and fi ne-grained task than object classification, which only involves recognizing the presence of an object class in an image, without differentiating between individual instances of that class.

Dataset Collection : Dataset collection comprises of two steps: a data curation step and an annotation step: a) Data curation: this step includes the image collection from various sources such as the internet [20]. However, it is hard to fi nd datasets of particular object instances. Manually collecting images is both laborious and expensive, while requiring to ensure diversity of scenery. b) Data annotation: this step is performed usually supervised and is human-aided, while unsupervised annotation is gaining ground [21].

## 4. Approach overview

The approach requires access to two datasets: a scene dataset serving as background to paste the objects and train the detection models, named 𝐷𝑏𝑔 , and an object dataset denoted as 𝐷𝑜𝑏𝑗 with few samples for object detection to train a model to produce objects from random noise. The proposed approach requires little time and no human annotation and resembles 'Cut, Paste and Learn' [18]. The difference lies to the fact that the object instances generated from 𝐷𝑜𝑏𝑗 are not simply pasted onto the scene images from 𝐷𝑏𝑔 , but rather they are generated from random noise, allowing for greater diversity and fl exibility in the generated images. Additionally, the proposed approach can generate a large number of unique and diverse images, enabling more robust and accurate training of object detection models.

Fig. 1. Image of a whitefly with noise added for time-step 𝑡 =50 , 100 , 150 , 200 , 300 , 600 , 700 , 999 (left to right).

<!-- image -->

## 4.1. Generate: object instance generation

In the study, Denoising Diffusion Probabilistic Models (DDPM) [22] were used for image generation instead of GANs or VAEs due to the limited availability of annotated insect datasets, the difficulties in training GANs and VAEs [23], and the simplicity and straightforwardness of the training procedure provided by DDPMs. In DDPM, an image is taken from data and noise is added for 𝑇 timesteps as shown in Fig. 1. Then a model is trained to predict that noise at each step and use the model to generate images. The lower-bound formulation for sampling was not used in our approach, and Algorithm 1 from the DDPM paper was strictly followed [22], which is simpler and more straightforward.

Since DDPM generate an object of arbitrarily background, a foreground mask can be predicted that distinguishes instance pixels from background ones. This provides us the object mask that can be placed in the background.

## 4.2. Paste: paste object instances to background

The randomly generated instance objects, masked out of their background, are placed on the background images. This step can be done either manually, where human-based coordinates indicate the position where the object is placed, or automatically, leveraging annotated background datasets. For example, in the agriculture sector, insect instances can be placed arbitrarily on the pre-annotated leaf locations. Data augmentation was also added to the objects to ensure a diverse coverage: rotation, fl ip left-right, fl ip up-down which are geometrical augmentation techniques. The location where the object instance is pasted indicates the coordinates of the bounding box that will be later used for the detection task.

## 4.3. Blend: blending object instances with background

However, simply pasting objects on the background would result in pixel artifacts creating an unpleasant results. In order to create a seamless and aesthetically pleasing result, blending is exploited. In our scenarios, Poisson image blending [24] was used that utilizes generic interpolation machinery based on solving Poisson equations and Gaussian blurring where the image is blurred by a Gaussian function. The blending phase eliminates the boundary artifacts that exist between the pasted object and the background. In the case of Poisson blending, the prediction of a foreground mask can be avoided since the blending allows mixing of the object with the background.

## 4.4. Detect: object detection

Evaluating a synthetically generated dataset is an essential step in ensuring its quality and suitability for various machine learning tasks. Deep learning architectures provide a comprehensive and robust way of evaluating a synthetic dataset as they are capable of capturing complex patterns and relationships in the data. By using deep learning models as benchmarks, the quality of the synthetic data can be assessed in terms of its ability to train accurate models.

Detection Model : To comprehensively evaluate the synthetically generated dataset, the YOLOv8 [25] was adopted that builds on the success of earlier YOLO versions by introducing new features and upgrades to improve efficiency and versatility even further. YOLOv8 is real-time with low computation cost.

Evaluation Metrics : The COCO detection metrics [20] were utilized for evaluating the performance of object detection. The evaluation process includes the calculation of mean average precision (mAP) for a ranging intersection-over-union (IoU) from 0.05 to 0.95 (stride 0.05) denoted as 𝑚𝐴𝑃 , and the mAP for 𝐼𝑜𝑈 = 0 . 5 denoted as 𝑚𝐴𝑃 50 . Furthermore, precision and recall were adopted as additional performance measures.

## 5. Experiments

In this section, the proposed approach is validated in the agricultural sector for the purpose of detecting insects named whiteflies in leaf plants.

## 5.1. Datasets selection

Background Dataset : Selecting a specific dataset in agriculture is important because it can greatly impact the accuracy and effectiveness of any analysis being developed. The selection of a dataset can determine the scope of the study, the relevance of the results, and the potential applications of the fi ndings. This study aims at detecting insects named whiteflies contained in multiple leaves with different types of background conditions with varying lighting conditions. For that reason, the PlantDoc dataset [26] was selected as the background dataset denoted ( 𝐷𝑏𝑔 ) which will be used to add the insects on leaves. The latter dataset has the advantage of having annotated leaves, thus the location of the image where the insects shall be added is already known. For simplicity, | 𝐷𝑏𝑔 | = 762 images were chosen from the PlantDoc dataset to be used in the proposed pipeline, out of which objects were added on 644 of them.

Object Dataset : One challenge in the fi eld of agriculture is the lack of available images of insects. Insects are small in size that can cause significant damage to crops and can be difficult to detect and monitor. However, there is a shortage of image datasets. For that cause, | 𝐷𝑜𝑏𝑗 | = 113 images of whiteflies were selected and manually cropped to enclose the region-of-interest (insect).

## 5.2. Experimental settings

DDPM : 𝑇 = 1000 was set for all experiments as in [22]. The forward process variances were set to constants increasing linearly from 𝛽 1 = 10 -4 to 𝛽 𝑇 = 0 . 02 . To represent the reverse process, a U-Net backbone [27] was used with group normalization [28]. Parameters are shared across time, which is specified to the network using the Transformer sinusoidal position embedding [29]. Self-attention [29,30] was also used. The model was trained for 1300 epochs with a batch size of 4 . The image size was set to 64 × 64 . The AdamW optimizer [31] was chosen with a learning rate of 3 𝑒 -4 and a cosine annealing scheduler with warm-up restarts [32] every 100 epochs. The results of the generative model for every 100 epochs is shown if Fig. 2. It is observed that the model is capable of generating satisfactory results after epoch 1000 , some of which are indistinguishable from real images. Note that the model is trained with 113 images. The foreground mask is predicted using a U-Net architecture [27]. In addition to DDPM, attempts were made to use GANs and VAEs for image generation, but encountered problems such as oscillating and destabilized model parameters thus failed to converge [23], leading to the utilization of DDPM instead.

YOLOv8 : The synthetic dataset was evaluated with all YOLOv8 models (nano, small, medium, large, extra-large) [25]. Each backbone was pre-trained on COCO 2017 [20]. Each model was trained for 200 epochs, with an early stopping patience of 20 . The batch size was set to

Fig. 2. Images generated from DDPM every 100 epochs.

<!-- image -->

8 . The size of input images was fi xed to 1024 × 1024 . The AdamW optimizer [31] was used with an initial learning rate of 10 -2 and a fi nal one-cycle learning rate of 10 -2 . The momentum was set to 0 . 9 and the weight decay to 0 . 0005 . Warm-up was used for the fi rst 3 epochs. For data augmentation, the following techniques were chosen: hue augmentation, saturation augmentation, value augmentation, rotation, translation, scale, shear, fl ip left-right, fl ip top-down, and copy-paste [33]. These techniques are only used in training, and not during evaluation. The detection experiments were based on PyTorch [34] and were performed on an NVIDIA Tesla P100 with 16 GB onboard memory.

## 5.3. Results

Generated Dataset : Using the trained DDPM, | 𝐷 ′ 𝑜𝑏𝑗 | =500 instances of whiteflies were arbitrarily generated and pasted on the PlantDoc dataset. Out of the 767 images of the PlantDoc dataset, 118 were skipped where no objected were added. The objects were added on the 𝐷𝑏𝑔 dataset at a pixel size of 10 ±10% while maintaining their aspect ratio, followed by random rotation, fl ip left-right, and fl ip top-down. Three images were generated from each image: one without blending, one with Gaussian blurring, and one with Poisson blending, while the object was kept at the same coordinates for each approach for fair comparison. The fi nal dataset had 53 , 401 annotations, resulting in an average of 82 . 9 annotations per image across one class (whitefly). The number of objects per image ranged from 1 to 421 . The median image ratio was 1504 ×1504 .

Detection Results : Following [18], both Gaussian blurring and Poisson blending were used to assess the detection model, and compare them with no blending applied. The detection results for all three cases are shown in Table 1. Our fi ndings demonstrate that both Gaussian blurring and Poisson blending significantly enhance the detection results from 0 . 011 (in the case of YOLOv8s for Gaussian blurring) up to 0 . 034 points of 𝑚𝐴𝑃 50 (in the case of YOLOv8m for Gaussian blurring). Additionally, the application of blending techniques eliminates boundary artifacts between the object and the background, although it is not notable for such small object instances. Poisson blending outperformed Gaussian blurring in terms of detection results in the case of YOLOv8n, YOLOv8l, YOLOv8x, except for the YOLOv8s model where the results showed higher 𝑚𝐴𝑃 and recall with lower precision for Poisson blending. Moreover, Poisson blending significantly improves the recall to correctly detect all the relevant objects in an image. On the other side, Gaussian blurring improves the precision that measures the accuracy of the model in detecting relevant objects in an image (the decreases compared to Poisson blending are insignificant). Our results indicate that the highest accuracy was achieved using the

Table 1 Object detection results.

| Model             | 𝑚𝐴𝑃 50            |   𝑚𝐴𝑃 |   Precision |   Recall |
|-------------------|-------------------|-------|-------------|----------|
| No Blending       |                   |       |             |          |
| YOLOv8n           | 0.594             | 0.245 |       0.735 |    0.530 |
| YOLOv8s           | 0.627             | 0.262 |       0.725 |    0.558 |
| YOLOv8m           | 0.627             | 0.262 |       0.752 |    0.549 |
| YOLOv8l           | 0.624             | 0.268 |       0.744 |    0.551 |
| YOLOv8x           | 0.618             | 0.253 |       0.736 |    0.546 |
| Gaussian Blurring | Gaussian Blurring |       |             |          |
| YOLOv8n           | 0.605             | 0.256 |       0.735 |    0.542 |
| YOLOv8s           | 0.642             | 0.298 |       0.748 |    0.567 |
| YOLOv8m           | 0.661             | 0.316 |       0.762 |    0.573 |
| YOLOv8l           | 0.647             | 0.306 |       0.770 |    0.565 |
| YOLOv8x           | 0.632             | 0.285 |       0.757 |    0.563 |
| Poisson Blending  | Poisson Blending  |       |             |          |
| YOLOv8n           | 0.623             | 0.275 |       0.740 |    0.561 |
| YOLOv8s           | 0.642             | 0.300 |       0.746 |    0.575 |
| YOLOv8m           | 0.654             | 0.309 |       0.759 |    0.577 |
| YOLOv8l           | 0.652             | 0.307 |       0.757 |    0.567 |
| YOLOv8x           | 0.642             | 0.293 |       0.759 |    0.567 |

* An up green arrow indicates an increase, a down red arrow indicates a decrease, and a yellow dash represent stability in metric being analyzed (comparing Gaussian blurring and Poisson blending with each other). The rows in bold indicate the models with the highest metric.

YOLOv8m model with Gaussian blurring, resulting in 𝑚𝐴𝑃 50 =0 . 661 and 𝑚𝐴𝑃 = 0 . 316 , followed by the same model using Poisson blending, resulting in 𝑚𝐴𝑃 50 =0 . 654 and 𝑚𝐴𝑃 =0 . 309 . All the models demonstrate high precision and lower recall which indicates that many relevant objects are missed, which is expected since there are on average 82 . 9 small-sized objects on each image.

## 6. Conclusions and future work

In this paper, a method for generating synthetic datasets was proposed for object detection using the Generate-Paste-Blend-Detect technique. The effectiveness of this approach was demonstrated in the context of agriculture, specifically in detecting insect pests named whiteflies. The PlantDoc dataset [26] was combined with artificially generated instances of whiteflies using a DDPM. Gaussian blurring and Poisson blending were applied to the dataset to assess the detection model's performance and found that both techniques significantly enhance the detection results. Poisson blending outperformed Gaussian blurring in terms of recall, while Gaussian blurring improved precision. The YOLOv8m model with Gaussian blurring achieved the highest 𝑚𝐴𝑃 50 of 0 . 66 with high precision and lower recall since each image contains many small-sized instances. This method has significant potential for creating synthetic datasets for object detection in various domains, where collecting real data is either restricted, limited, or expensive. Although this work has been demonstrated on a specific dataset in the agriculture domain, further work includes extending the pipeline for larger datasets as in the case of [18], with potential use of different generative models such as GANs and VAEs and comparing their performance with DDPMs. Our scope is to contribute to the creation of a larger and more diverse synthetic dataset for object detection in the agricultural domain, which can serve as a benchmark for future studies in this fi eld.

## CRediT authorship contribution statement

Nikolaos Giakoumoglou: Conceptualization, Data curation, Formal analysis, Investigation, Methodology, Software, Visualization, Writing - original draft. Eleftheria Maria Pechlivani: Funding acquisition, Methodology, Project administration, Validation, Writing - original draft, Writing - review &amp; editing. Dimitrios Tzovaras: Supervision, Writing - review &amp; editing.

## Declaration of competing interest

The authors declare that they have no known competing fi nancial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Data availability

Data will be made available on request.

## Acknowledgement

This paper is supported by European Union's Horizon 2020 research and innovation programme under the Green Deal grant agreement No. 101037128, project PestNu. We would like to thank our colleague Georgios Pediaditis from the Centre for Research and Technology Hellas (CERTH) for his support in pasting objects to the generated dataset.

## References

- [1] A. Oikonomidis, C. Catal, A. Kassahun, Deep learning for crop yield prediction: a systematic literature review, N.Z. J. Crop Hortic. Sci. 51 (1) (2023) 1-26, https:// doi .org /10 .1080 /01140671 .2022 .2032213.
- [2] S. Ramesh, R. Hebbar, M. Niveditha, R. Pooja, N. Prasad Bhat, N. Shashank, P.V. Vinod, Plant disease detection using machine learning, in: 2018 International Conference on Design Innovations for 3Cs Compute Communicate Control (ICDI3C), 2018, pp. 41-45.
- [3] A.S.M.M. Hasan, F. Sohel, D. Diepeveen, H. Laga, M.G. Jones, A survey of deep learning techniques for weed detection from images, Comput. Electron. Agric. 184 (2021) 106067, https://doi .org /10 .1016 /j .compag .2021 .106067, https://www .sciencedirect .com /science /article /pii /S0168169921000855.
- [4] Q. Zhang, Y. Liu, C. Gong, Y. Chen, H. Yu, Applications of deep learning for dense scenes analysis in agriculture: a review, Sensors 20 (5) (2020), https://doi .org /10 . 3390 /s20051520, https://www .mdpi .com /1424 -8220 /20 /5 /1520.
- [5] S. Yang, L. Zheng, X. Chen, L. Zabawa, M. Zhang, M. Wang, Transfer learning from synthetic in-vitro soybean pods dataset for in-situ segmentation of on-branch soybean pods, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 2022, pp. 1666-1675.
- [6] A. Prakash, S. Boochoon, M. Brophy, D. Acuna, E. Cameracci, G. State, O. Shapira, S. Birchfield, Structured domain randomization: bridging the reality gap by contextaware synthetic data, CoRR, arXiv :1810 .10093 [abs], 2018.
- [7] M.D. Cicco, C. Potena, G. Grisetti, A. Pretto, Automatic model based dataset generation for fast and accurate crop and weeds detection, in: 2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), IEEE, 2017.
- [8] I.J. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, Y. Bengio, Generative adversarial networks, https://doi .org /10 .48550 / ARXIV .1406 .2661, https://arxiv .org /abs /1406 .2661, 2014.
- [9] D.P. Kingma, M. Welling, Auto-encoding variational bayes, https://doi .org /10 . 48550 /ARXIV .1312 .6114, https://arxiv .org /abs /1312 .6114, 2013.
- [10] Y. Song, S. Ermon, Improved techniques for training score-based generative models, CoRR, arXiv :2006 .09011 [abs], 2020.
- [11] A.K. Shetty, I. Saha, R.M. Sanghvi, S.A. Save, Y.J. Patel, A review: object detection models, in: 2021 6th International Conference for Convergence in Technology (I2CT), IEEE, 2021.
- [12] R. Wang, L. Liu, C. Xie, P. Yang, R. Li, M. Zhou, Agripest: a large-scale domainspecific benchmark dataset for practical agricultural pest detection in the wild, Sensors 21 (5) (2021), https://doi .org /10 .3390 /s21051601, https://www .mdpi .com / 1424 -8220 /21 /5 /1601.
- [13] J. Liu, X. Wang, Tomato diseases and pests detection based on improved yolo v3 convolutional neural network, Front. Plant Sci. 11 (Jun. 2020), https://doi .org /10 . 3389 /fpls .2020 .00898.
- [14] N. Giakoumoglou, E.M. Pechlivani, A. Sakelliou, C. Klaridopoulos, N. Frangakis, D. Tzovaras, Deep learning-based multi-spectral identification of grey mould, Smart Agricult. Technol. 4 (2023) 100174, https://doi .org /10 .1016 /j .atech .2023 .100174, https://www .sciencedirect .com /science /article /pii /S2772375523000047.
- [15] N. Giakoumoglou, E.M. Pechlivani, N. Katsoulas, D. Tzovaras, White fl ies and black aphids detection in fi eld vegetable crops using deep learning, in: 2022 IEEE 5th International Conference on Image Processing Applications and Systems (IPAS), Vol. Five, 2022, pp. 1-6.
- [16] C. Karam, M. Awad, Y.A. Jawdah, N. Ezzeddine, A. Fardoun, GAN-based semiautomated augmentation online tool for agricultural pest detection: a case study on whiteflies, Front. Plant Sci. 13 (Sep. 2022), https://doi .org /10 .3389 /fpls .2022 . 813050.
- [17] L. Bi, G. Hu, Improving image-based plant disease classification with generative adversarial network under limited training set, Front. Plant Sci. 11 (Dec. 2020), https://doi .org /10 .3389 /fpls .2020 .583438.

- [18] D. Dwibedi, I. Misra, M. Hebert, Cut, paste and learn: surprisingly easy synthesis for instance detection, https://doi .org /10 .48550 /ARXIV .1708 .01642, https://arxiv . org /abs /1708 .01642, 2017.
- [19] M.L. Jepsen, M.D. Jeppesen, Copy and paste synthetic dataset generation in agriculture, Master's thesis, Aalborg University, 2021.
- [20] T.-Y. Lin, M. Maire, S. Belongie, L. Bourdev, R. Girshick, J. Hays, P. Perona, D. Ramanan, C.L. Zitnick, P. Dollár, Microsoft coco: common objects in context, https:// doi .org /10 .48550 /ARXIV .1405 .0312, https://arxiv .org /abs /1405 .0312, 2014.
- [21] J. Cheng, X. Zhang, P. Luo, J. Huang, J. Huang, An unsupervised approach for semantic place annotation of trajectories based on the prior probability, Inf. Sci. 607 (2022) 1311-1327, https://doi .org /10 .1016 /j .ins .2022 .06 .034, https://www . sciencedirect .com /science /article /pii /S0020025522006247.
- [22] J. Ho, A. Jain, P. Abbeel, Denoising diffusion probabilistic models, CoRR, arXiv : 2006 .11239 [abs], 2020.
- [23] K.J. Liang, C. Li, G. Wang, L. Carin, Generative adversarial network training is a continual learning problem, arXiv :1811 .11083, 2018.
- [24] P. Pérez, M. Gangnet, A. Blake, Poisson image editing, in: ACM SIGGRAPH 2003 Papers, ACM, 2003.
- [25] G. Jocher, A. Chaurasia, J. Qiu, YOLO by Ultralytics, 1 2023, https://github .com / ultralytics /ultralytics.
- [26] D. Singh, N. Jain, P. Jain, P. Kayal, S. Kumawat, N. Batra, PlantDoc, in: Proceedings of the 7th ACM IKDD CoDS and 25th COMAD, ACM, 2020.
- [27] O. Ronneberger, P. Fischer, T. Brox, U-net: convolutional networks for biomedical image segmentation, https://doi .org /10 .48550 /ARXIV .1505 .04597, https:// arxiv .org /abs /1505 .04597, 2015.
- [28] Y. Wu, K. He, Group normalization, in: Proceedings of the European Conference on Computer Vision (ECCV), 2018.
- [29] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A.N. Gomez, L. u. Kaiser, I. Polosukhin, Attention is all you need, in: I. Guyon, U.V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, R. Garnett (Eds.), Advances in Neural Information Processing Systems, vol. 30, Curran Associates, Inc., 2017, https://proceedings . neurips .cc /paper /2017 /file /3f5ee243547dee91fbd053c1c4a845aa -Paper .pdf.
- [30] X. Wang, R. Girshick, A. Gupta, K. He, Non-local neural networks, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018.
- [31] I. Loshchilov, F. Hutter, Decoupled weight decay regularization, https://doi .org /10 . 48550 /ARXIV .1711 .05101, https://arxiv .org /abs /1711 .05101, 2017.
- [32] I. Loshchilov, F. Hutter, Sgdr: stochastic gradient descent with warm restarts, https://doi .org /10 .48550 /ARXIV .1608 .03983, https://arxiv .org /abs /1608 .03983, 2016.
- [33] G. Ghiasi, Y. Cui, A. Srinivas, R. Qian, T.-Y. Lin, E.D. Cubuk, Q.V. Le, B. Zoph, Simple copy-paste is a strong data augmentation method for instance segmentation, https:// doi .org /10 .48550 /ARXIV .2012 .07177, https://arxiv .org /abs /2012 .07177, 2020.
- [34] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, A. Desmaison, A. Kopf, E. Yang, Z. DeVito, M. Raison, A. Tejani, S. Chilamkurthy, B. Steiner, L. Fang, J. Bai, S. Chintala, Pytorch: An Imperative Style, High-Performance Deep Learning Library, Advances in Neural Information Processing Systems, vol. 32, Curran Associates, Inc., 2019, pp. 8024-8035.