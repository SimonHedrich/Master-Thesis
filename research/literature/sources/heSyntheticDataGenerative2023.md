## IS SYNTHETIC DATA FROM GENERATIVE MODELS READY FOR IMAGE RECOGNITION?

Ruifei He 1* Shuyang Sun 2 Xin Yu 1 Chuhui Xue 3 Wenqing Zhang 3 Philip Torr 2

Song Bai 3

† †

Xiaojuan Qi 1

1 The University of Hong Kong 2 University of Oxford 3 ByteDance

## ABSTRACT

Recent text-to-image generation models have shown promising results in generating high-fidelity photo-realistic images. Though the results are astonishing to human eyes, how applicable these generated images are for recognition tasks remains under-explored. In this work, we extensively study whether and how synthetic images generated from state-of-the-art text-to-image generation models can be used for image recognition tasks, and focus on two perspectives: synthetic data for improving classification models in data-scarce settings ( i . e . zero-shot and fewshot), and synthetic data for large-scale model pre-training for transfer learning. We showcase the powerfulness and shortcomings of synthetic data from existing generative models, and propose strategies for better applying synthetic data for recognition tasks. Code: https://github.com/CVMI-Lab/SyntheticData.

## 1 INTRODUCTION

Over the past decade, deep learning powered by large-scale annotated data has revolutionized the field of image recognition. However, it is costly and time-consuming to manually collect a largescale labeled dataset, and recent concerns about data privacy and usage rights further hinder this process. In parallel, generative models that aim to model real-data distributions can now produce high-fidelity photo-realistic images. In particular, recent text-to-image generation models (Nichol et al., 2021; Ramesh et al., 2022; Saharia et al., 2022b) have made major breakthroughs in synthesizing high-quality images from text descriptions. This promotes us to ask: is synthetic data from generative models ready for image recognition tasks?

There are a few early attempts at exploring synthetic data from generative models for image recognition tasks. Besnier et al. (2020) use a class-conditional GAN (BigGAN (Brock et al., 2018) trained for ImageNet-1000 classes) to generate images for training image classifiers. Zhang et al. (2021) leverage StyleGAN (Karras et al., 2019) to produce synthetic labeled data for object-part segmentation. Jahanian et al. (2021) manipulate the latent space of a GAN model to produce multi-view images for contrastive learning. Albeit promising, early works either address tasks on a small scale or only for a specific setting. Plus, they all focus on GAN-based models and none explore the revolutionary text-to-image generation models, which hold more promises to benefit recognition tasks.

In this paper, we present the first study on the state-of-the-art text-to-image generation models for image recognition. With the power of text-to-image generation, we could hopefully not only generate massive high-quality labeled data, but also achieve domain customization by generating synthetic data targeted for a specific label space, i . e . the label space of a downstream task. Our study is carried out on one open-sourced text-to-image generation model, GLIDE (Nichol et al., 2021) 1 . We attempt to uncover the benefits and pitfalls of synthetic data for image recognition through the lens of investigating the following two questions: 1) is synthetic data from generative models ready for improving classification models? 2) whether synthetic data can be a feasible source for transfer learning ( i . e . model pre-training)? It is worth noting that for 1), we only studied the zero-shot and few-shot settings because the positive impact of synthetic data diminishes as more shots are present. And, we build most of our investigations on the state-of-the-art method CLIP (Radford et al., 2021) with the feature extractor initialized with large-scale pre-trained weights frozen.

∗ Part of the work is done during an internship at ByteDance. Email: ruifeihe@eee.hku.hk

† Corresponding authors: songbai.site@gmail.com, xjqi@eee.hku.hk

1 At the beginning of this project, GLIDE is the only open-sourced text-to-image synthesis model that also delivers high-quality synthesis results.

Our Findings. First, in the zero-shot setting, i . e . no real-world data are available, we demonstrate that synthetic data can significantly improve classification results on 17 diverse datasets: the performance is increased by 4 . 31% in top-1 accuracy on average, and even improved by as much as 17 . 86% on the EuroSAT dataset. To better leverage synthetic data in this setting, we also investigate useful strategies to increase data diversity, reduce data noise, and enhance data reliability. This is achieved by designing diversified text prompts and measuring the correlation of text and synthesized data with CLIP features.

Second, in the few-shot setting, i . e . a few real images are available, albeit not as significant as in the zero-shot task, synthetic data are also shown to be beneficial and help us achieve a new state of the art. Our observation shows that the domain gap between synthetic data and downstream task data is one challenge on further improving the effectiveness of synthetic data on classifier learning. Fortunately, in this setting, the accessibility of real data samples can provide useful information about the data distribution of the downstream task. We thus propose to use real images as guidance in the generation process to reduce domain gaps and improve effectiveness.

Third, in large-scale model pre-training for transfer learning, our study shows that synthetic data are suitable and effective for model pre-training, delivering superior transfer learning performance and even outperforming ImageNet pre-training. Especially, synthetic data work surprisingly well in unsupervised model pre-training, and favor ViT-based backbones. We also demonstrate that by increasing the label space ( i . e . text prompts) for data generation, the enlarged data amount and diversity could further bring performance boosts. Besides, synthetic data can work collaboratively with real data ( i . e . ImageNet) where we obtain improved performance when the model is initialized with ImageNet pre-trained weights.

## 2 RELATED WORKS

Synthetic Data for Image Recognition. There are mainly two forms of synthetic data for image recognition, i . e . 1) synthetic datasets generated from a traditional simulation pipeline; 2) synthetic images output from generative models.

The first type, synthetic datasets (Dosovitskiy et al., 2015; Peng et al., 2017; Richter et al., 2016), are usually generated from a traditional pipeline with a specific data source, e . g .synthetic 2D renderings of 3D models or scenes from graphics engines. However, this traditional way of generating synthetic datasets has several drawbacks: 1) manually defined pipeline generated synthetic data may have a certain gap with real-world data; 2) taking up huge physical space to store and huge cost to share and transfer; 3) data amount and diversity bounded by the specific data source.

Compared with synthetic datasets, generative models are a more efficient means of synthetic data representation, exhibiting favorable advantages: 1) could produce high-fidelity photorealistic images closer to real data since they are trained on real-world data; 2) highly condensed compared to synthetic data itself, and take up much reduced storage space; 3) potentially unlimited synthetic data size. Only recently, few works attempt to explore synthetic data generated from generative models for image recognition. Besnier et al. (2020) use a class-conditional GAN to train classifiers of the same classes. Zhang et al. (2021) leverage the latent code of StyleGAN (Karras et al., 2019) to produce labels for object part segmentation. While they achieve promising results, both works are task-wise and only employed on a small scale. Jahanian et al. (2021) use a GAN-based generator to generate multiple views to conduct unsupervised contrastive representation learning. These works, however, explore upon the traditional GAN-based models; in contrast, our work investigates with the best released text-to-image generation model, which demonstrates new customization ability for different downstream label space.

Text-to-Image Diffusion Models. Diffusion models (Sohl-Dickstein et al., 2015; Ho et al., 2020; Nichol &amp; Dhariwal, 2021) have recently emerged as a class of promising and powerful generative models. As a likelihood-based model, the diffusion model matches the underlying data distribution q ( x 0 ) by learning to reverse a noising process, and thus novel images can be sampled from a prior Gaussian distribution via the learned reverse path. Because of the high sample quality, good mode coverage and promising training stability, diffusion models are quickly becoming a new trend in both unconditional (Ho et al., 2020; Nichol &amp; Dhariwal, 2021; Ho et al., 2022) and conditional (Dhariwal &amp; Nichol, 2021; Rombach et al., 2022; Lugmayr et al., 2022; Saharia et al., 2022a; Meng et al., 2021; Saharia et al., 2022c) image synthesis fields.

In particular, text-to-image generation can be treated as a conditional image generation task that requires the sampled image to match the given natural language description. Based upon the formulation of the diffusion model, several text-to-image models such as Stable diffusion (Rombach et al., 2022), DALL-E2 (Ramesh et al., 2022), Imagen (Saharia et al., 2022b) and GLIDE (Nichol et al., 2021) deliver unprecedented synthesis quality, largely facilitating the development of the AIfor-Art community. Despite achieving astonishing perceptual results, their potential utilization for high-level tasks is yet under-explored. In this paper, we utilize the state-of-the-art model GLIDE and showcase its powerfulness and shortcomings for synthesizing data for recognition tasks.

## 3 IS SYNTHETIC DATA READY FOR IMAGE RECOGNITION?

In the following sections, we answer the question by studying whether synthetic data can benefit recognition tasks and how to better leverage synthetic data to address different tasks. We carry out our exploration through the lens of two basic settings with three tasks: synthetic data for improving classification models in the data-scarce setting ( i . e . zero-shot and few-shot) (see Sec. 3.1 and Sec. 3.2) and synthetic data for model pre-training for transfer learning (see Sec. 3.3).

Model Setup for Data-scarce ( i . e . Zero-shot and Few-shot) Image Classification. As CLIP (Radford et al., 2021) is the state-of-the-art approach for zero-shot learning, we conduct our study for zero-shot and few-shot settings upon pre-trained CLIP models, aiming to better understand synthetic data upon strong baselines. There have been a few attempts on better tuning pre-trained CLIP for data-scarce image classification, such as CoOp (Zhou et al., 2022b), CLIP Adapter (Gao et al., 2021), and Tip Adapter (Zhang et al., 2022), where the image encoder is frozen for better preserving the pretrained feature space. We argue that different tuning methods could all be regarded as different ways of learning classifier weights, e . g . CoOp optimizes learnable prompts for better learning classifiers.

Here, we adopt a simple tuning method, Classifier Tuning ( CT ), a baseline method introduced in Wortsman et al. (2022). Concretely, for a k-way classification, we input the class names C = { c 1 , ..., c k } with prompt s i = ' a photo of a { c i } ' into the text encoder h of CLIP to obtain the text features h ( s i ) . Then the text features h ( s i ) could be used to construct classifier weights W ∈ R d × k , where d is the dimension of text features. Finally, we combine the image encoder g with the classifier weights W to obtain a classification model f ( x ) = g ( x ) T W . We empirically show that CT performs comparably with other tuning methods. Compared with complex designed tuning methods, we hope to use a simpler method for better investigating the effectiveness of synthetic data.

## 3.1 IS SYNTHETIC DATA READY FOR ZERO-SHOT IMAGE RECOGNITION?

Our aim is to investigate to what degree synthetic data are beneficial to zero-shot tasks and how to better leverage synthetic data for zero-shot learning.

Zero-shot Image Recognition. We study the inductive zero-shot learning setting where no real training images of the target categories are available. CLIP models are pre-trained with large-scale image-caption pairs, and the similarities between paired image features (from an image-encoder g ) and text features (from a text-encoder h ) are maximized during pre-training. The pre-trained feature extractor can then be used to solve zero-shot tasks where given an image, its features from g are compared with text features of different classes from h and the image is further assigned to the class that has the largest similarity in the CLIP text-image feature space.

Synthetic Data for Zero-shot Image Recognition. Though CLIP models exhibit strong zero-shot performance thanks to the large-scale vision-language dataset for pre-training, there are still several shortcomings when the model is deployed for a downstream zero-shot classification task, which may be attributed to unavoidable data noise in CLIP's pre-training data or the label space mismatch between pre-training and the zero-shot task. Hence, with a given label space for a zero-shot task, we study whether synthetic data can be used to better adapt CLIP models for zero-shot learning.

How to generate the data? Given a pre-trained text-to-image generation model, to synthesize novel samples, the basic ( B ) strategy is to use the label names of the target categories to build the language input and generate a corresponding image. Then, the paired label names and synthesized data can be employed to train the classifier with the feature extractor frozen.

How to enrich diversity? Only using the label names as inputs might limit the diversity of synthesized images and cause bottlenecks for validating the effectiveness of synthetic data. Hence, we leverage an off-the-shelf word-to-sentence T5 model (pre-trained on 'Colossal Clean Crawled Corpus' dataset (Raffel et al., 2020) and finetuned on CommonGen dataset (Lin et al., 2019)) to increase the diversity of language prompts and the generated images, namely language enhancement ( LE ), hoping to better unleash the potential of synthesized data. Concretely, we input the label name of each class to the word-to-sentence model which generates diversified sentences containing the class names as language prompts for the text-to-image generation process. For example, if the class label is 'airplane', then the enhanced language prompt from the model could be 'a white airplane hovering over a beach and a city'. The enhanced text descriptions introduce rich context descriptions.

Table 1: Main Results on Zero-shot Image Recognition. All results are top-1 accuracy on test set. o: object-level. s: scene-level. f: fine-grained. t: textures. si: satellite images. r: robustness.

| Dataset         | Task   |   CLIP-RN50 | CLIP-RN50+SYN   |   CLIP-ViT-B/16 | CLIP-ViT-B/16+SYN   |
|-----------------|--------|-------------|-----------------|-----------------|---------------------|
| CIFAR-10        | o      |       70.31 | 80.06 (+9.75)   |           90.80 | 92.37 (+1.57)       |
| CIFAR-100       | o      |       35.35 | 45.69 (+10.34)  |           68.22 | 70.71 (+2.49)       |
| Caltech101      | o      |       86.09 | 87.74 (+1.65)   |           92.98 | 94.16 (+1.18)       |
| Caltech256      | o      |       73.36 | 75.74 (+2.38)   |           80.14 | 81.43 (+1.29)       |
| ImageNet        | o      |       60.33 | 60.78 (+0.45)   |           68.75 | 69.16 (+0.41)       |
| SUN397          | s      |       58.51 | 60.07 (+1.56)   |           62.51 | 63.79 (+1.28)       |
| Aircraft        | f      |       17.34 | 21.94 (+4.60)   |           24.81 | 30.78 (+5.97)       |
| Birdsnap        | f      |       34.33 | 38.05 (+3.72)   |           41.90 | 46.84 (+4.94)       |
| Cars            | f      |       55.63 | 56.93 (+1.30)   |           65.23 | 66.86 (+1.63)       |
| CUB             | f      |       46.69 | 56.94 (+10.25)  |           55.23 | 63.79 (+8.56)       |
| Flower          | f      |       66.08 | 67.05 (+0.97)   |           71.30 | 72.60 (+1.30)       |
| Food            | f      |       80.34 | 80.35 (+0.01)   |           88.75 | 88.83 (+0.08)       |
| Pets            | f      |       85.80 | 86.81 (+1.01)   |           89.10 | 90.41 (+1.31)       |
| DTD             | t      |       42.23 | 43.19 (+0.96)   |           44.39 | 44.92 (+0.53)       |
| EuroSAT         | si     |       37.51 | 55.37 (+17.86)  |           47.77 | 59.86 (+12.09)      |
| ImageNet-Sketch | r      |       33.29 | 36.55 (+3.26)   |           46.20 | 48.47 (+2.27)       |
| ImageNet-R      | r      |       56.16 | 59.37 (+3.21)   |           74.01 | 76.41 (+2.40)       |
| Average         | /      |       55.13 | 59.47 (+4.31)   |           65.42 | 68.32 (+2.90)       |

Howto reduce noise and enhance robustness? It's unavoidable that the synthesized data may contain low-quality samples. This is even more severe in the setting with language enhancement as it may introduce undesired items into language prompts (see Figure A.2 in Appendix for visual examples). Hence, we introduce a CLIP Filter ( CF ) strategy to rule out these samples. Specifically, CLIP zero-shot classification confidence is used to assess the quality of synthesized data, and the lowconfidence ones are removed. Besides, as soft-target is more robust than hard-target in countering sample noise, we study whether soft cross-entropy loss ( SCE , see Sec. C.4 in Appendix) which uses the normalized clip scores as a target could be used to enhance robustness against data noise.

Experiment Setup . We select 17 diverse datasets covering object-level (CIFAR-10 and CIFAR-100 ((Krizhevsky et al., 2009), Caltech101 (Fei-Fei et al., 2006), Caltech256 (Griffin et al., 2007), ImageNet (Deng et al., 2009)), scene-level (SUN397 (Xiao et al., 2010)), fine-grained (Aircraft (Maji et al., 2013), Birdsnap (Berg et al., 2014), Cars (Krause et al., 2013), CUB (Wah et al., 2011), Flower (Nilsback &amp; Zisserman, 2008), Food (Bossard et al., 2014), Pets (Parkhi et al., 2012)), textures (DTD (Cimpoi et al., 2014)), satelite images (EuroSAT (Helber et al., 2019)) and robustness (ImageNetSketch (Wang et al., 2019), ImageNet-R (Hendrycks et al., 2021)) for zero-shot image classification. For synthetic data amount, we generate 2000 (study of synthetic image number in Appendix Sec. B.3) synthetic images for each class in B and LE . For LE , we generate 200 sentences for each class.

Main Results: 1) zero-shot classification results on 17 datasets; 2) study of synthetic data diversity; 3) study of synthetic data reliability; 4) study of model/classifier tuning; 5) study of the behavior of synthetic data for zero-shot classification in the training from scratch settings.

Synthetic data can significantly improve the performance of zero-shot learning. Our main studies in zero-shot settings are conducted with CLIP-RN50 (ResNet-50 (He et al., 2016) and CLIP-ViTB/16 (ViT-B/16 (Dosovitskiy et al., 2020)) as CLIP backbone), and we report results with our best strategy of LE+CF+SCE . As shown in Table 1, on 17 diverse downstream zero-shot image classification datasets, we achieve a remarkable average gain of 4.31% for CLIP-RN50 and 2.90% for CLIP-ViT-B/16 in terms of top-1 accuracy. Significantly, on the EuroSAT dataset, we achieve the largest performance boost of 17.86% for CLIP-RN50 in top-1 accuracy. We notice that the performance gain brought by synthetic data varies differently across datasets, which is mainly related to GLIDE's training data distribution. The training data distribution of the text-to-image generation model GLIDE would exhibit bias and produce different domain gaps with different datasets (see Sec. A.2 in Appendix for more analysis).

Table 2: Ablation study on Language Enhancement (LE), CLIP-based Filtering (CF), and Softtarget Cross-Entropy (SCE).

| Dataset   | CLIP   | B             | B              | LE             | LE             | LE+CF          | LE+CF          |
|-----------|--------|---------------|----------------|----------------|----------------|----------------|----------------|
| Dataset   | CLIP   | CE            | SCE            | CE             | SCE            | CE             | SCE            |
| CIFAR-10  | 70.31  | 77.39 (+7.08) | 78.23 (+7.92)  | 77.20 (+6.89)  | 77.55 (+7.24)  | 80.01 (+9.70)  | 80.06 (+9.75)  |
| CIFAR-100 | 35.35  | 43.99 (+8.64) | 44.25 (+8.90)  | 44.08 (+8.73)  | 44.91 (+9.56)  | 44.55 (+9.20)  | 45.69 (+10.34) |
| EuroSAT   | 37.51  | 45.64 (+8.13) | 48.23 (+10.72) | 53.26 (+15.75) | 54.94 (+17.43) | 54.75 (+17.24) | 55.37 (+17.86) |

Language diversity matters. By introducing more linguistic context into the text input, LE helps increase the diversity of synthetic data. As shown in Table 2, LE can achieve additional performance gains upon B in most cases (0.66 ↑ on CIFAR-100, 6.71 ↑ on EuroSAT), which demonstrates the efficacy of LE and the importance of synthetic data diversity for zero-shot classification.

Reliability matters. While LE could help increase the diversity of synthetic data, it also introduces the risks of noisy samples. Observed on CIFAR-10 in Table 2, LE sometimes even brings performance drops compared with B (0.68 % ↓ on CIFAR-10), which may attribute to the noise introduced by enhanced language prompts, e . g . the sentence extended from the class name word may contain other class names or confusing objects. Fortunately, with CF to filter out unreliable samples, LE+CF yields consistent improvement upon B . Moreover, SCE generally achieves better performance than CE , showing its better adaptation to label noise.

Classifier tuning is enough for CLIP , while tuning with the pre-trained encoder leads to degradation, mainly due to domain gaps. Here, we investigate if only tuning the final classifier is the optimal solution in our setting with synthetic data. As shown in Table 3, we tune different proportions of the full model parameters on synthetic data for EuroSAT (0.02% corresponds to our default case where only the classifier is tuned), and report the zero-shot performance on the test set of EuroSAT. The best results are obtained by only tuning the classifier, and the performance gradually decreases as we gradually incorporate more parameters in the encoder for optimization, which agrees with the traditional strategy. For understanding why synthetic data may harm pre-trained image encoder, we experiment with real-world data with domain shifts and find they behave similarly to synthetic data (Appendix Sec. B.2), which suggests that domain gap is the main reason for the phenomenon. We argue that synthetic data might have a better chance to overcome domain shifts in comparison with real-world data since we can customize and keep the label space of the synthetic data in line with the down-stream dataset and use strategies during synthesizing to alleviate domain shifts.

Table 4: Setting when training from scratch. Dataset: CIFAR-100.

| Param Tuned (%)                                            | 0                                                          | 0.02                                                       | 0.04                                                       |                                                            | 62.50 64.06                                                | 69.53                                                      | 82.81                                                      | 92.19                                                      |
|------------------------------------------------------------|------------------------------------------------------------|------------------------------------------------------------|------------------------------------------------------------|------------------------------------------------------------|------------------------------------------------------------|------------------------------------------------------------|------------------------------------------------------------|------------------------------------------------------------|
| Acc                                                        | 37.51                                                      | 55.37                                                      | 55.11                                                      | 55.28                                                      | 54.56                                                      | 54.34                                                      | 53.63                                                      | 52.09                                                      |
| Table 3: Parameters tuned v.s. Accuracy. Dataset: EuroSAT. | Table 3: Parameters tuned v.s. Accuracy. Dataset: EuroSAT. | Table 3: Parameters tuned v.s. Accuracy. Dataset: EuroSAT. | Table 3: Parameters tuned v.s. Accuracy. Dataset: EuroSAT. | Table 3: Parameters tuned v.s. Accuracy. Dataset: EuroSAT. | Table 3: Parameters tuned v.s. Accuracy. Dataset: EuroSAT. | Table 3: Parameters tuned v.s. Accuracy. Dataset: EuroSAT. | Table 3: Parameters tuned v.s. Accuracy. Dataset: EuroSAT. | Table 3: Parameters tuned v.s. Accuracy. Dataset: EuroSAT. |
| Real shot                                                  | 1                                                          | 16                                                         | 32                                                         | 64                                                         | 80                                                         | 90                                                         | 95                                                         | 100                                                        |
| Acc                                                        | 2.48                                                       | 10.4                                                       | 14.95                                                      | 21.96                                                      | 24.4                                                       | 25.52                                                      | 27.99                                                      | 29.95                                                      |

Synthetic data deliver inferior performance in the training from scratch setting and are much less data-efficient than real data. To exclude the influence of powerful CLIP initialization in our study of synthetic data, we also conduct a from-scratch setting on the CIFAR-100 dataset, where we optimize a ResNet-50 model from random initialization. Given the label space of the CIFAR-100 dataset, we generate a synthetic dataset of 50k (500 images per class) to train a ResNet-50 model from scratch for image classification. We achieve a performance of 28.74% top-1 accuracy on CIFAR-100 test set, which is much lower than the performance of the pre-trained CLIP model (see Table 1). This might be attributed to the quality and diversity of data. The CLIP model benefits from diverse realworld data. Further, we hope to investigate how many real in-domain training data can match the performance of our 50k synthetic data. As shown in Table 4, training with 95 images per category ( 95 × 100 = 9 . 5 k) will achieve comparable performance as that of 50k synthetic data. This manifests that synthetic data are not as efficient and effective as real data when solving downstream tasks. It requires around 5 times more data in order to achieve a comparable performance as that of real data. Note that we find further increasing the amount of synthetic data will not deliver further performance gains for the downstream classification task. We expect that further investigations on synthesis quality will bring new opportunities in this area which will be our future work.

Summary. Current synthetic data from text-to-image generation models could indeed bring significant performance boosts for a wide range of zero-shot image classification tasks, and is readily applicable with carefully designed strategies such as large-scale pre-trained models. Diversity and reliability matter for synthetic data when employed for zero-shot tasks. When the model is trained from scratch with synthetic data, synthetic data cannot deliver satisfactory performance and are much less data-efficient and effective for solving the classification task in comparison with real data.

## 3.2 IS SYNTHETIC DATA READY FOR FEW-SHOT IMAGE RECOGNITION?

In this section, we explore the effectiveness of synthetic data for few-shot tasks and how synthetic data impact the performance as more and more shots are included. Also, we design effective strategies to better leverage synthetic data.

Few-shot Image Recognition. We adopt the CLIP-based method as the model for few-shot image recognition due to its state-of-the-art performance (Radford et al., 2021). As discussed previously, various prompt learning based methods can be treated as tuning the classifier weights. We thus study how to tune the classifier weights with synthetic data. In an N-way M-shot case, we are given M real images of each test class, where M ∈ { 1, 2, 4, 8, 16 } in our experiments. With a total of N × M training samples, we hope to achieve favorable performance on a hold-out test set of the N classes.

Synthetic Data for Few-shot Image Recognition. While there have been a few attempts to study how to better adapt CLIP models for few-shot tasks (Zhou et al., 2022b;a; Zhang et al., 2022), they all focus on the model optimization level, and none have explored from the data level. Here, we systematically study whether and how synthetic data can be employed for solving few-shot image recognition tasks.

With the experience from synthetic data for zero-shot tasks, we adopt the best strategy ( i . e . LE+CF ) in the zero-shot setting as the basic strategy ( B ). Further, as the few-shot real samples can provide useful information on the data distribution of the classification task, we develop two new strategies leveraging the in-domain few-shot real data for better using synthetic data: 1) Real Filtering ( RF ): given synthetic data of one class c , we use the features of few-shot real samples to filter out synthetic images whose features are very close to the features of real samples that belong to other categories different from class c ; 2) Real guidance ( RG ): we use the few-shot real samples as guidance to generate synthetic images where the few-shot real samples (added noise) replace the random noise at the beginning of the generation to guide the diffusion process (details in Appendix Sec. C.3).

Experiment Setup . For datasets, we carefully select 8 image classification datasets from recent works (Zhou et al., 2022b;a; Zhang et al., 2022) that conduct few-shot learning upon CLIP: ImageNet (Deng et al., 2009), Caltech101 (Fei-Fei et al., 2006), Pets (Parkhi et al., 2012), Cars (Krause et al., 2013), Aircraft (Maji et al., 2013), SUN397 (Xiao et al., 2010), DTD (Cimpoi et al., 2014), EuroSAT (Helber et al., 2019). For synthetic image number, we generate 800 (study of synthetic image number in Appendix Sec. B.3) images per class for RG method to approximately match the number of images in B and RF .

Main Results: 1) few-shot classification results on 8 datasets; 2) ablation study of training strategy; 3) ablation study of synthetic data generation strategy; 4) ablation study of BN strategy.

Synthetic data can boost few-shot learning and the positive impact of synthetic data will gradually diminish with the increase of real data shots. As shown in Figure 1 (results of more datasets are in the Appendix Sec. B.1), with only few-shot real images for training, our implemented CT w. init (classifier weights initialized from CLIP text embeddings) performs comparably with the state-ofthe-art CLIP tuning methods Tip Adapter (Zhang et al., 2022) and CoOp (Zhou et al., 2022b). CT w. Syn represents our results of applying synthetic data with mix training, real image as guidance, and freezing BN strategies. With the help of generated synthetic data, CT w. Syn achieves noticeable performance gains upon CT w. init , and achieves a new state-of-the-art few-shot learning performance across different datasets. We argue that for data-scarce few-shot classification, synthetic data could help address the insufficient data problem to boost performance. However, we notice that the boost from synthetic data gradually diminishes as the real shot number increases. We state that the effectiveness of each sample in real data is high since there's no domain gap; in contrast, synthetic data suffer from domain gaps and perform less efficiently. In addition, the positive effects of the few-shot real data may overlap with that of synthetic data. Thus, with the increase of real data, the overlapping becomes serious and the positive impacts of synthetic data are reduced.

Mix Training fits few-shot learning with synthetic data. Now that we have two parts of data, i . e . few-shot real data and synthetic data, we could either 1) phase-wise train on each part of data with two training phases, or 2) adopt mix training that simultaneously utilizes two parts of data to update Published as a conference paper at ICLR 2023

<!-- image -->

Number of Real labeled examples per class

Figure 1: Results for few-shot image recognition. Results on all 8 datasets are provided in Appendix.

the model in each iteration. Details of phase-wise/mix training in Appendix Sec. C.5.2. We provide the results in Table 7: we study on the EuroSAT dataset and use synthetic data generated from the RG method; under different shot number settings, mix training performs consistently better than two phase-wise strategies. We suggest that mix training could help learn better classifiers since each part could function as a regularization for the other: synthetic data help alleviate instabilities brought by limited real samples, and real data help address the noise and domain gap of synthetic data.

Employing real data as guidance can alleviate domain differences and boost performance. We compare three strategies of synthetic data generation for few-shot tasks. As shown in Table 5, both RF and RG provide performance gains upon B which is the best strategy in the zero-shot setting. This demonstrates the importance of utilizing the domain knowledge from few-shot images for preparing the synthetic data. Further, RG significantly outperforms RF , yielding the best performance. This shows utilizing real data as guidance of the diffusion process help reduce the domain gap (visual illustrations in the Appendix Sec. B.8).

Table 5: Ablation for Basic strategy ( B ), Real Filtering ( RF ), Real Guidance ( RG ) on EuroSAT, 16 shot.

|    B |    RF |    RG |
|------|-------|-------|
| 87.1 | 87.33 | 88.47 |

Table 6: Frozen BN works better for 16-shot settings on EuroSAT.

| Train data Freeze   | BN? Test Acc   |       |
|---------------------|----------------|-------|
| Real                |                | 75.31 |
| Real                | ✓              | 85.63 |
| Syn                 |                | 44.73 |
| Syn                 | ✓              | 55.37 |

Table 7: Mix training works better for few-shot tasks on EuroSAT.

|   M-shot |   Phase-wise syn → real real → syn |   Phase-wise syn → real real → syn |   Mix training |
|----------|------------------------------------|------------------------------------|----------------|
|        1 |                              63.01 |                              63.32 |          64.36 |
|        2 |                              72.24 |                              72.85 |          73.62 |
|        4 |                              78.88 |                              79.21 |          79.88 |
|        8 |                              83.64 |                              83.99 |          84.57 |
|       16 |                              87.10 |                              87.44 |          88.47 |

Frozen BN works better. Lastly, we investigate batch normalization (BN) strategies for our fewshot settings with synthetic data. As shown in Table 6, for both real and synthetic data, freezing the BN layers yields much better performance. We analyze that for real data, it is hard to get a good estimation of BN statistics when the number of images is limited. As for synthetic data, we attribute this to the statistical difference between different domains. Hence, we freeze BN layers during tuning for few-shot settings.

Summary. Synthetic data from text-to-image generation models could readily benefit few-shot learning and achieve a new state-of-the-art few-shot classification performance with strategies we present in this paper. However, the positive impact of synthetic data will diminish as more shots of real data are available which further confirms our previous claim that synthetic data are still not as effective as real data in training classification models.

## 3.3 IS SYNTHETIC DATA READY FOR PRE-TRAINING?

Finally, we study whether synthetic data are effective in large-scale pre-training. We also present effective strategies to better leverage synthetic data for model pre-training.

Pre-training for Transfer Learning. Recently, it has become a common practice to first pre-train models on large-scale datasets to obtain a well-trained feature extractor and then fine-tune the models on downstream tasks with labeled data ( i . e . transfer learning). There have been various successful pre-training methods, including supervised pre-training (Joulin et al., 2016; Li et al., 2017; Mahajan et al., 2018; Sun et al., 2017; Kolesnikov et al., 2020), self-supervised pre-training (Chen et al., 2020a; He et al., 2020; Caron et al., 2020; Grill et al., 2020; Chen &amp; He, 2021; Zbontar et al., 2021; Ye et al., 2019), and semi-supervised pre-training (Xie et al., 2020; Pham et al., 2021).

Synthetic data for Pre-training. Since data amount and diversity play important roles in pretraining, we adopt the synthetic data generation strategy LE solely to maximize the scale of synthetic pre-training data. We study two settings for generating synthetic data for pre-training: 1) downstream-aware, where we have access to the label space of the downstream task, and thus we generate synthetic data according to the label space of the downstream task; 2) downstream-agnostic, where we have no access to downstream tasks in the pre-training stage, and we turn to a relatively general and diverse label space such as ImageNet-1K. For pre-training methods, we experiment with supervised pre-training and self-supervised pre-training methods.

Experiment Setup . We compare synthetic pre-trained models with models of random initialization and models of ImageNet-1K pre-training in terms of their transfer learning abilities. For downstream-aware settings: we conduct supervised pre-training on synthetic data generated according to CIFAR-100 label space and then transfer to CIFAR-100 through finetuning for evaluation.

For downstream-agnostic settings: we perform supervised pre-training and self-supervised pretraining (we adopt Moco v2 (Chen et al., 2020b) framework for its simplicity and reproducibility) on synthetic data generated from ImageNet-1K label space and evaluate the transfer performance by finetuning the pretrained models on a object detection dataset - PASCAL VOC (Everingham et al., 2010). Further, we experiment with ImageNet-2K label space (original ImageNet-1K and another non-overlapping 1K label names randomly selected from ImageNet-21K) to study the factors of data diversity and amount in synthetic pre-training. We use ResNet-50 as the default backbone when not else noted, and also experiment with a ViT-based backbone, i . e . DeiT-S (Touvron et al., 2021).

Results for Downstream-aware settings . We generate synthetic data of different sizes from CIFAR-100 label space, i . e . 1 × , 2 × , 3 × ImageNet-1K data size, concretely 1.2M, 2.4M, 3.6M. We pre-train the model on the generated synthetic labeled set in a supervised manner, and then perform evaluation after finetuning the model on CIFAR-100. As shown in Table 8, with an equivalent amount of data as that of ImageNet-1K (1.2M), synthetic data for pre-training can largely reduce the gap between training from scratch (78.83%) and ImageNet- pre-trained model (84.50%). Moreover, with 2 × and 3 × synthetic data, pre-training on synthetic data outperforms ImageNet-1K pre-training with a noticeable margin. In addition, when we initialize the model from ImageNet-1K pre-trained weights and pre-train the model on synthetic data, we obtain extra boosts upon both results.

We conclude that for downstream-aware synthetic pre-training, synthetic data deliver close performance as that of ImageNet-1K pretraining with the same amount of data, synthetic data amount helps improve the results to outperforming ImageNet-1K pre-training, and synthetic pre-training could further benefit from ImageNet-1K pre-training.

Results for Downstream-agnostic settings . We first experiment with ImageNet-1K label space with 1 × or 2 × ImageNet-1K data size, i . e . 1.2M/2.4M IN-1K Syn. We perform supervised pretraining and self-supervised pre-training ( i . e . Moco v2) on the generated synthetic data, and evaluate the pre-training results by transferring to the CIFAR-100 image classification task or the PASCAL VOCdetection task. As it is too costly to validate all settings ( e . g ., it takes more than 1 week to train Moco v2 on 4.0M synthetic data), we select several representative settings of interest to validate the effectiveness of synthetic data without hurting our conclusion.

As shown in Table 10 and 11, with 1.2M IN-1K Syn, both supervised pre-training (79.00%) and self-supervised pre-training (81.55%) could largely approach their IN-1K Real counterparts (super.:81.3%; self-super.:82.44%) and largely outperforms the result without pre-training (66.08%). When increasing the data amount to 2.4M, the transferred results further increase, and the unsupervised pre-training method, i . e . Moco v2, performs better in utilizing our synthetic data thanks to its independence of labels, yielding a 82.13% transferred performance which surpasses supervised pre-training on IN-1K Real (81.30%) and is on par with its Moco v2 counterpart at IN-1K Real (82.44%). Next, we expand the label space by adding another 1K categories, producing IN-2K Syn. The enlarged diversity and data amount further bridge the gap between synthetic pre-training results and IN-1K Real pre-training results. Noticeably, the unsupervised pre-trained model Moco v2 (82.29%) largely approaches the IN-1K Real counterpart (82.44%) with negligible performance drop of 0.15%. Furthermore, when initialized from IN-1K Real pre-trained weights, both supervised and self-supervised pre-training improve upon both pure real data and synthetic data for pre-training.

While the above results are all obtained with convolutional-based backbone i . e . ResNet50, we further explore with a recent ViT-based backbone i . e . DeiT-S (Touvron et al., 2021). Surprisingly, Table 10: Results for object detection on PASCAL VOC with downstream-agnostic supervised pre-training , all results are reported in AP 50 .

| Data     | pre-trained on IN-1k?   | Syn. images amount   | Syn. images amount   | Syn. images amount   | Syn. images amount   |
|----------|-------------------------|----------------------|----------------------|----------------------|----------------------|
| Data     | pre-trained on IN-1k?   | 0                    | 1.2M                 | 2.4M                 | 3.6M                 |
| (None)   |                         | 78.83                | -                    | -                    | -                    |
| C100 Syn |                         | -                    | 83.90                | 85.03                | 85.24                |
| (None)   | ✓                       | 84.50                | -                    | -                    | -                    |
| C100 Syn | ✓                       | -                    | 84.90                | 85.32                | 85.52                |

Table 8: Results on CIFAR-100 with downstream-aware supervised pre-training . C100: CIFAR100.

| Data      | pre-trained on IN-1k?   | Syn. images amount   | Syn. images amount   | Syn. images amount   | Syn. images amount   |
|-----------|-------------------------|----------------------|----------------------|----------------------|----------------------|
| Data      | pre-trained on IN-1k?   | 0                    | 1.2M                 | 2.4M                 | 4.0M                 |
| (None)    |                         | 66.08                | -                    | -                    | -                    |
| IN-1K Syn |                         | -                    | 79.00                | 80.00                | -                    |
| IN-2K Syn |                         | -                    | -                    | 80.54                | 80.72                |
| (None)    | ✓                       | 81.30                | -                    | -                    | -                    |
| IN-1K Syn | ✓                       | -                    | -                    | 81.78                | -                    |
| IN-2K Syn | ✓                       | -                    | -                    | 81.87                | 81.91                |

Table 9: Results on CIFAR-100 with downstream-agnostic supervised pretraining . Backbone: DeiT-S.

| Data      | pre-trained on IN-1k?   | Syn. images amount   | Syn. images amount   | Syn. images amount   | Syn. images amount   |
|-----------|-------------------------|----------------------|----------------------|----------------------|----------------------|
| Data      | pre-trained on IN-1k?   | 0                    | 1.2M                 | 2.4M                 | 4.0M                 |
| (None)    |                         | 69.29                | -                    | -                    | -                    |
| IN-1K Syn |                         | -                    | 87.98                | 88.39                | -                    |
| IN-2K Syn |                         | -                    | -                    | 88.57                | 88.91                |
| (None)    | ✓                       | -                    | 88.07                | -                    | -                    |

Table 11: Results for object detection on PASCAL VOC with downstream-agnostic selfsupervised pre-training (Moco v2) , all results are reported in AP 50 .

| Data      | pre-trained on IN-1k?   | Syn. images amount   | Syn. images amount   | Syn. images amount   | Syn. images amount   |
|-----------|-------------------------|----------------------|----------------------|----------------------|----------------------|
| Data      | pre-trained on IN-1k?   | 0                    | 1.2M                 | 2.4M                 | 4.0M                 |
| (None)    |                         | 66.08                | -                    | -                    | -                    |
| IN-1K Syn |                         | -                    | 81.55                | 82.13                | -                    |
| IN-2K Syn |                         | -                    | -                    | 82.22                | 82.29                |
| (None)    | ✓                       | 82.44                | -                    | -                    | -                    |
| IN-1K Syn | ✓                       | -                    | -                    | 82.47                | -                    |

ViT-based backbone is shown to be more advantageous compared with convolution-based backbone for synthetic pre-training: outperforming ImageNet pre-training results in the downstream-agnostic settings. Equipped with ViT-based backbone, on only 1.2M IN-1K synthetic data, we achieve comparable performance (87.98%) with ImageNet pre-training (88.07%). Further increasing the data amount (88.39%) and label space (88.57%, 88.91%) of pre-training data leads to higher performance than ImageNet pre-training. ViT-based backbones have stronger ability for learning from large-scale data and are more robust (Pinto et al., 2021), and thus could better benefit from synthetic pre-training where data are more noisy and data scale could be easily increased.

Conclusion . In terms of transfer abilities, synthetic data from text-to-image generation models show surprisingly promising results for model pre-training, which is comparable to the standard ImageNet pre-training. We conclude our findings as follows:

1. Data amount has positive impacts on synthetic pre-training; performance could be improved by increasing synthetic data size, but would gradually saturate as the amount of data increases.

2. Synthetic data for pre-training is orthogonal to real data for pre-training.

3. For downstream-aware synthetic pre-training, we significantly outperform IN-1K Real (1.2M) pre-training with 2.4M/3.6M synthetic data on CIFAR-100.

4. For downstream-agnostic synthetic pre-training, we achieve comparable results with ImageNet (IN-1k) Real pre-training; self-supervised pre-training performs better than supervised pre-training, and ViT-based backbone performs better than convolutional-based backbone. Besides, increasing the label space size could further improve the performance.

## 4 CONCLUSION

Wesystematically investigate whether synthetic data from current state-of-the-art text-to-image generation models are readily applicable for image recognition. Our extensive experiments demonstrate that synthetic data are beneficial for classifier learning in zero-shot and few-shot recognition, bringing significant performance boosts and yielding new state-of-the-art performance. Further, current synthetic data show strong potential for model pre-training, even surpassing the standard ImageNet pre-training. We also point out limitations and bottlenecks for applying synthetic data for image recognition, hoping to arouse more future research in this direction.

Limitations. In all investigated settings, we observe improved performance as the data amount and diversity (label space) increases. However, due to our limited computational resource, we are not able to further scale up data amount, which may take months to train one model. Besides, we are also not able to investigate larger model sizes and advanced architectures in the current investigation which is also worth exploring in the future. We present more discussions on limitations and future directions in the appendix.

## REFERENCES

- Thomas Berg, Jiongxin Liu, Seung Woo Lee, Michelle L. Alexander, David W. Jacobs, and Peter N. Belhumeur. Birdsnap: Large-scale fine-grained visual categorization of birds. In Proc. Conf. Computer Vision and Pattern Recognition (CVPR) , June 2014.
- Victor Besnier, Himalaya Jain, Andrei Bursuc, Matthieu Cord, and Patrick P´ erez. This dataset does not exist: training models from generated images. In ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) , pp. 1-5. IEEE, 2020.
- Lukas Bossard, Matthieu Guillaumin, and Luc Van Gool. Food-101-mining discriminative components with random forests. In European conference on computer vision , pp. 446-461. Springer, 2014.
- Andrew Brock, Jeff Donahue, and Karen Simonyan. Large scale gan training for high fidelity natural image synthesis. arXiv preprint arXiv:1809.11096 , 2018.
- Mathilde Caron, Ishan Misra, Julien Mairal, Priya Goyal, Piotr Bojanowski, and Armand Joulin. Unsupervised learning of visual features by contrasting cluster assignments. arXiv:2006.09882 , 2020.
- Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In ICML , 2020a.
- Xinlei Chen and Kaiming He. Exploring simple siamese representation learning. In CVPR , 2021.
- Xinlei Chen, Haoqi Fan, Ross Girshick, and Kaiming He. Improved baselines with momentum contrastive learning. arXiv preprint arXiv:2003.04297 , 2020b.
- Mircea Cimpoi, Subhransu Maji, Iasonas Kokkinos, Sammy Mohamed, and Andrea Vedaldi. Describing textures in the wild. In Proceedings of the IEEE conference on computer vision and pattern recognition , pp. 3606-3613, 2014.
- Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition , pp. 248-255. Ieee, 2009.
- Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in Neural Information Processing Systems , 34:8780-8794, 2021.
- A. Dosovitskiy, P. Fischer, E. Ilg, P. H¨ ausser, C. Hazırbas ¸, V . Golkov, P. v.d. Smagt, D. Cremers, and T. Brox. Flownet: Learning optical flow with convolutional networks. In IEEE International Conference on Computer Vision (ICCV) , 2015. URL http://lmb.informatik. uni-freiburg.de/Publications/2015/DFIB15 .
- Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv:2010.11929 , 2020.
- Mark Everingham, Luc Van Gool, Christopher KI Williams, John Winn, and Andrew Zisserman. The pascal visual object classes (voc) challenge. IJCV , 2010.
- Li Fei-Fei, Robert Fergus, and Pietro Perona. One-shot learning of object categories. IEEE transactions on pattern analysis and machine intelligence , 28(4):594-611, 2006.
- Peng Gao, Shijie Geng, Renrui Zhang, Teli Ma, Rongyao Fang, Yongfeng Zhang, Hongsheng Li, and Yu Qiao. Clip-adapter: Better vision-language models with feature adapters. arXiv preprint arXiv:2110.04544 , 2021.
- Gregory Griffin, Alex Holub, and Pietro Perona. Caltech-256 object category dataset. 2007.

- Jean-Bastien Grill, Florian Strub, Florent Altch´ e, Corentin Tallec, Pierre H Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Daniel Guo, Mohammad Gheshlaghi Azar, et al. Bootstrap your own latent: A new approach to self-supervised learning. arXiv:2006.07733 , 2020.
- Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR , 2016.
- Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In CVPR , 2020.
- Patrick Helber, Benjamin Bischke, Andreas Dengel, and Damian Borth. Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing , 12(7):2217-2226, 2019.
- Dan Hendrycks, Steven Basart, Norman Mu, Saurav Kadavath, Frank Wang, Evan Dorundo, Rahul Desai, Tyler Zhu, Samyak Parajuli, Mike Guo, Dawn Song, Jacob Steinhardt, and Justin Gilmer. The many faces of robustness: A critical analysis of out-of-distribution generalization. ICCV , 2021.
- Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598 , 2022.
- Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems , 33:6840-6851, 2020.
- Jonathan Ho, Chitwan Saharia, William Chan, David J Fleet, Mohammad Norouzi, and Tim Salimans. Cascaded diffusion models for high fidelity image generation. J. Mach. Learn. Res. , 23: 47-1, 2022.
- Ali Jahanian, Xavier Puig, Yonglong Tian, and Phillip Isola. Generative models as a data source for multiview representation learning. arXiv preprint arXiv:2106.05258 , 2021.
- Armand Joulin, Laurens Van Der Maaten, Allan Jabri, and Nicolas Vasilache. Learning visual features from large weakly supervised data. In ECCV , 2016.
- Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pp. 4401-4410, 2019.
- Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114 , 2013.
- Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Joan Puigcerver, Jessica Yung, Sylvain Gelly, and Neil Houlsby. Big transfer (bit): General visual representation learning. In ECCV , 2020.
- Jonathan Krause, Michael Stark, Jia Deng, and Li Fei-Fei. 3d object representations for fine-grained categorization. In 4th International IEEE Workshop on 3D Representation and Recognition (3dRR-13) , Sydney, Australia, 2013.
- Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.
- Ang Li, Allan Jabri, Armand Joulin, and Laurens van der Maaten. Learning visual n-grams from web data. In ICCV , 2017.
- Bill Yuchen Lin, Wangchunshu Zhou, Ming Shen, Pei Zhou, Chandra Bhagavatula, Yejin Choi, and Xiang Ren. Commongen: A constrained text generation challenge for generative commonsense reasoning. arXiv preprint arXiv:1911.03705 , 2019.
- Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101 , 2017.

- Andreas Lugmayr, Martin Danelljan, Andres Romero, Fisher Yu, Radu Timofte, and Luc Van Gool. Repaint: Inpainting using denoising diffusion probabilistic models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pp. 11461-11471, 2022.
- Dhruv Mahajan, Ross Girshick, Vignesh Ramanathan, Kaiming He, Manohar Paluri, Yixuan Li, Ashwin Bharambe, and Laurens Van Der Maaten. Exploring the limits of weakly supervised pretraining. In ECCV , 2018.
- S. Maji, J. Kannala, E. Rahtu, M. Blaschko, and A. Vedaldi. Fine-grained visual classification of aircraft. Technical report, 2013.
- Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations. In International Conference on Learning Representations , 2021.
- Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741 , 2021.
- Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. In International Conference on Machine Learning , pp. 8162-8171. PMLR, 2021.
- Maria-Elena Nilsback and Andrew Zisserman. Automated flower classification over a large number of classes. In 2008 Sixth Indian Conference on Computer Vision, Graphics &amp; Image Processing , pp. 722-729. IEEE, 2008.
- Omkar M Parkhi, Andrea Vedaldi, Andrew Zisserman, and CV Jawahar. Cats and dogs. In 2012 IEEE conference on computer vision and pattern recognition , pp. 3498-3505. IEEE, 2012.
- Xingchao Peng, Ben Usman, Neela Kaushik, Judy Hoffman, Dequan Wang, and Kate Saenko. Visda: The visual domain adaptation challenge. arXiv preprint arXiv:1710.06924 , 2017.
- Hieu Pham, Zihang Dai, Qizhe Xie, and Quoc V Le. Meta pseudo labels. In CVPR , 2021.
- Francesco Pinto, Philip Torr, and Puneet K Dokania. Are vision transformers always more robust than convolutional neural networks? 2021.
- Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning , pp. 8748-8763. PMLR, 2021.
- Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J Liu, et al. Exploring the limits of transfer learning with a unified text-to-text transformer. J. Mach. Learn. Res. , 21(140):1-67, 2020.
- Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical textconditional image generation with clip latents. arXiv preprint arXiv:2204.06125 , 2022.
- Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster r-cnn: Towards real-time object detection with region proposal networks. NIPS , 2015.
- Stephan R Richter, Vibhav Vineet, Stefan Roth, and Vladlen Koltun. Playing for data: Ground truth from computer games. In European conference on computer vision , pp. 102-118. Springer, 2016.
- Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨ orn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pp. 10684-10695, 2022.
- Chitwan Saharia, William Chan, Huiwen Chang, Chris Lee, Jonathan Ho, Tim Salimans, David Fleet, and Mohammad Norouzi. Palette: Image-to-image diffusion models. In ACM SIGGRAPH 2022 Conference Proceedings , pp. 1-10, 2022a.

- Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, S Sara Mahdavi, Rapha Gontijo Lopes, et al. Photorealistic text-to-image diffusion models with deep language understanding. arXiv preprint arXiv:2205.11487 , 2022b.
- Chitwan Saharia, Jonathan Ho, William Chan, Tim Salimans, David J Fleet, and Mohammad Norouzi. Image super-resolution via iterative refinement. IEEE Transactions on Pattern Analysis and Machine Intelligence , 2022c.
- Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International Conference on Machine Learning , pp. 2256-2265. PMLR, 2015.
- Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502 , 2020.
- Chen Sun, Abhinav Shrivastava, Saurabh Singh, and Abhinav Gupta. Revisiting unreasonable effectiveness of data in deep learning era. In ICCV , 2017.
- Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Herv´ e J´ egou. Training data-efficient image transformers &amp; distillation through attention. In International Conference on Machine Learning , pp. 10347-10357. PMLR, 2021.
- Catherine Wah, Steve Branson, Peter Welinder, Pietro Perona, and Serge Belongie. The caltech-ucsd birds-200-2011 dataset. 2011.
- Haohan Wang, Songwei Ge, Zachary Lipton, and Eric P Xing. Learning robust global representations by penalizing local predictive power. In Advances in Neural Information Processing Systems , pp. 10506-10518, 2019.
- Mitchell Wortsman, Gabriel Ilharco, Jong Wook Kim, Mike Li, Simon Kornblith, Rebecca Roelofs, Raphael Gontijo Lopes, Hannaneh Hajishirzi, Ali Farhadi, Hongseok Namkoong, et al. Robust fine-tuning of zero-shot models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pp. 7959-7971, 2022.
- Jianxiong Xiao, James Hays, Krista A Ehinger, Aude Oliva, and Antonio Torralba. Sun database: Large-scale scene recognition from abbey to zoo. In 2010 IEEE computer society conference on computer vision and pattern recognition , pp. 3485-3492. IEEE, 2010.
- Qizhe Xie, Minh-Thang Luong, Eduard Hovy, and Quoc V Le. Self-training with noisy student improves imagenet classification. In CVPR , 2020.
- Mang Ye, Xu Zhang, Pong C Yuen, and Shih-Fu Chang. Unsupervised embedding learning via invariant and spreading instance feature. In CVPR , 2019.
- Jure Zbontar, Li Jing, Ishan Misra, Yann LeCun, and St´ ephane Deny. Barlow twins: Self-supervised learning via redundancy reduction. arXiv:2103.03230 , 2021.
- Renrui Zhang, Zhang Wei, Rongyao Fang, Peng Gao, Kunchang Li, Jifeng Dai, Yu Qiao, and Hongsheng Li. Tip-adapter: Training-free adaption of clip for few-shot classification. arXiv preprint arXiv:2207.09519 , 2022.
- Yuxuan Zhang, Huan Ling, Jun Gao, Kangxue Yin, Jean-Francois Lafleche, Adela Barriuso, Antonio Torralba, and Sanja Fidler. Datasetgan: Efficient labeled data factory with minimal human effort. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pp. 10145-10155, 2021.
- Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. Conditional prompt learning for vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pp. 16816-16825, 2022a.
- Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. Learning to prompt for visionlanguage models. International Journal of Computer Vision , pp. 1-12, 2022b.

## A MORE ANALYSIS

## A.1 LIMITATIONS AND FUTURE DIRECTIONS

A systematic way to study the language enhancement process is needed. In our experiments, we found language enhancement to be an effective way for enlarging synthetic data diversity. In this paper, we adopt an off-the-shelf word-to-sentence language model for this job, but this may bring certain risks and generate a high volume of noisy data. Therefore, effective constraints should be considered for language enhancement, which we leave as future work.

Larger synthetic data sizes for zero/few-shot tasks should be studied. Weonly generate synthetic data of median size for zero/few-shot tasks due to our limited computational resources; we encourage future works to further explore whether increasing data diversity and data amount could benefit these data-scarce tasks.

Co-training down-stream predictors with generative models. In this paper, we propose a pre-trained off-line image generator to construct our synthetic training set. It would be interesting to investigate how to generate in-domain images by tuning the image generator together with the lateral model for down-stream tasks. This could potentially be useful for domain generalization.

Increasing scales for synthetic data and model capability for pre-training settings. We observe improved performance when the data amount and diversity (label space) increase. However, due to our limited computational resource, we cannot further scale up the investigation, which may take months to train one model. We hope our promising results could inspire future research to explore enlarged scales for synthetic data and model size (larger model capability should be better at benefiting from a larger amount of data) for pre-training settings.

## A.2 ANALYSIS OF DIFFERENCES OF PERFORMANCE BOOST ON DIFFERENT DATASETS IN ZERO-SHOT SETTINGS

The main reason is related to GLIDE's training data distribution. The training data distribution of the text-to-image generation model GLIDE would exhibit bias and produce different domain gaps with different datasets. While we do not have direct access to GLIDE's training data, we may use synthesized images from GLIDE to explore. However, how to measure data and its relation to the performance for different tasks in a quantitative manner is still an open problem to our best knowledge as different task may have different levels of difficulties and data will impact the performance in different manners. An empirical thought is related to the synthesized image quality (e.g., synthesized images for DTD dataset is relatively noisy as shown in Figure A.6, where we only achieve 0.96% gains), domain gaps (synthesized images are usually object-centric and exhibit large domain difference with scene-level dataset SUN397, where only 1.56% boost achieved), and task difficulties (e.g., ImageNet is of high difficulty where only 0.45% gains are obtained).

For further improving the results, we suggest that it would be interesting to investigate how to generate in-domain images by tuning the image generator together with the lateral model for downstream tasks. This could potentially better reducing domain gap to the downstream task.

## B MORE EXPERIMENTS

## B.1 MAIN RESULTS ON FEW-SHOT TASKS

Due to the limit of pages and space, we provide the results for few-shot tasks of all 8 datasets in Figure A.2. We observe similar results on various datasets that synthetic data could boost few-shot learning performance and achieve a new state-of-the-art performance.

## B.2 REAL-WORLD DATA WITH DOMAIN SHIFT

In order to understand why synthetic data may harm pre-trained image encoder, we experiment with real-world data with domain shifts. Specifically, we conduct experiments on pre-trained CLIP model and study two scenarios (Classifier tuning only, and end-to-end finetune) for the downstream ImageNet classification task. These experiments on conducted on three data sources: 1) in-domain real data: ImageNet data is used for training; 2) real-world data with domain shifts: ImageNetSketch data which has the same label space with ImageNet but exhibits a domain shift towards sketch are used for training; and 3) our synthetic data: synthetic data from GLIDE are used for training. The results are shown in the Table A.12. It can be clearly observed that real-world data with domain shifts behave similarly as synthetic data (i.e. end-to-end tuning cause slight harm to the pre-trained encoder) while the performance of in-domain data will benefit from end-to-end tuning. This suggests that domain gap is the main reason for harming the pre-trained image encoder.

Figure A.2: Main results for Few-shot tasks on all 8 datasets.

<!-- image -->

We argue that synthetic data might have a better chance to overcome domain shifts in comparison with real-world data since we can customize and keep the label space of the synthetic data in line with the down-stream dataset. First, synthetic data can be made to tailor to a specific label space that the downstream task requires and reduce category shifts, which yet might be very challenging for real-world data. Second, a small amount of real-data can be leveraged to guide the data synthesis process to further alleviate domain shifts (namely Real Guidance in our few-shot settings), which also worth future in-depth explorations.

| Train data          |   ImageNet |   ImageNet-Sketch |   Synthetic data |
|---------------------|------------|-------------------|------------------|
| Classifier tuning   |      70.09 |             60.50 |            60.78 |
| End-to-end finetune |      76.17 |             60.34 |            60.35 |

Table A.12: Classifier tuning v.s. End-to-end finetune on different types of data. Zero-shot performance: 60.33.

## B.3 STUDY OF SYNTHETIC IMAGE NUMBER IN ZERO/FEW-SHOT SETTING

Here, we provide the study for the number of synthetic images in the zero-shot and few-shot settings. Firstly, for zero-shot settings, we experiment with 1000/2000/4000 images per class on Cifar10 dataset. As shown in Table A.13, we found 2000 to be a sufficient number while further increasing the number to 4000 only provide limited gains. Second, for few-shot settings, we study upon the settings of Eurosat dataset with 16 shot real images, and vary the number of synthesized images for each class from 400 to 1600. As shown in Table A.14, we found 800 synthetic images for each class to be a good choice between performance and cost.

For both settings, increasing the amount of training data further beyond a certain amount will not bring significant performance gain. The reasons can be attributed to the diversity and quality of data. We found that as the increase of the training data amount, the diversity of the data might not be scaled in a similar manner. Many redundant and similar samples will appear with the increase of data amount. Effective approaches to increase data diversity and quality will help further improve model performance. For the few-shot setting, the performance reach a high value after using a small amount of synthetic data (400-shot). This is because the existence of real-data provide a strong guidance for training the classifier. And, the positive impacts of synthetic data are reduced where a small amount of synthetic data are sufficient to learn a good classifier.

| syn shot   |     0 |   1000 |   2000 |   4000 |
|------------|-------|--------|--------|--------|
| Acc        | 70.31 |  79.21 |  80.06 |  80.08 |

Table A.13: Study of synthetic image number in the zero-shot setting on CIFAR-10 dataset.

| syn shot   |     0 |   400 |   800 |   1600 |
|------------|-------|-------|-------|--------|
| Acc        | 85.83 | 88.28 | 88.47 |  88.49 |

Table A.14: Study of synthetic image number in the few-shot setting on EuroSAT dataset with 16 shot real images.

## B.4 CLAIFICATION OF ZERO-SHOT CLASSIFICATION PERFORMANCE DIFFERENCE

For all results, we use the official released CLIP model 2 . For our original paper, we conduct experiments using simple prompts 'a photo of a [CLASS]' or 'a photo of a [CLASS], a type of [dataset type]' (for fine-grained tasks) to better evaluate the performance gains from the data itself by excluding the influence of tuned prompt methods. However, in the original CLIP paper, they use specifically designed prompt ensembles for each dataset. This is the main reason that our baseline is lower than that of the original paper.

2 https://github.com/openai/CLIP

We also provide the results of using CLIP paper's prompts for 13 datasets (only these 13 datasets out of 17 have reported results from CLIP's paper) in Table A.15. After using the same prompts as the CLIP's paper, the averaged performance on 13 datasets increased from 56.14% to 56.33%, closer to the 57.03% of the CLIP's reported results. There are still slight performance differences after using the same prompts, which we suspect to be a small reproduction problem of CLIP since we could match the reported zero-shot results from CoOp (Zhou et al., 2022b) and Tip-adapter (Gao et al., 2021). We observe that on some datasets we achieve higher results than CLIP reported results (i.e., CIFAR-100, ImageNet, SUN397, Birdsnap, Flower, Pets) and on others we achieve lower results (i.e., CIFAR-10, Caltech101, Aircraft, Cars, Food, DTD, EuroSAT). The averaged zero-shot performance is similar (56.33% v.s 57.03%) and the performance boost from synthetic data of two different types of prompt is also similar (averaged performance boost on 13 datasets: 3.85% v.s 4.17%). We argue the slight performance differences do not affect the exploration of synthetic data.

Table A.15: CLIP-RN50 ∗ : original CLIP paper results. CLIP-RN50: our results using simple prompt. CLIP-RN50 † : our results using the same prompt ensemble as CLIP.

|            |   CLIP-RN50 ∗ |   CLIP-RN50 | CLIP-RN50      |   CLIP-RN50 † | CLIP-RN50 † +SYN   |
|------------|---------------|-------------|----------------|---------------|--------------------|
| CIFAR-10   |          75.6 |       70.31 | 80.06 (+9.75)  |         71.59 | 80.23 (+8.64)      |
| CIFAR-100  |          41.6 |       35.35 | 45.69 (+10.34) |         41.94 | 48.70 (+6.76)      |
| Caltech101 |          82.1 |       86.09 | 87.74 (+1.65)  |         79.99 | 82.34 (+2.35)      |
| ImageNet   |         59.60 |       60.33 | 60.78 (+0.45)  |         60.33 | 60.78 (+0.45)      |
| SUN397     |         59.60 |       58.51 | 60.07 (+1.56)  |         60.23 | 60.47 (+0.24)      |
| Aircraft   |          19.3 |       17.34 | 21.94 (+4.60)  |         17.07 | 21.78 (+4.71)      |
| Birdsnap   |          32.6 |       34.33 | 38.05 (+3.72)  |         34.33 | 38.05 (+3.72)      |
| Cars       |         55.80 |       55.63 | 56.93 (+1.30)  |         55.70 | 57.65 (+1.95)      |
| Flower     |         65.90 |       66.08 | 67.05 (+0.97)  |         66.08 | 67.05 (+0.97)      |
| Food       |         81.10 |       80.34 | 80.35 (+0.01)  |         80.34 | 80.35 (+0.01)      |
| Pets       |         85.40 |       85.80 | 86.81 (+1.01)  |         85.80 | 86.81 (+1.01)      |
| DTD        |          41.7 |       42.23 | 43.19 (+0.96)  |         41.31 | 42.21 (+0.90)      |
| EuroSAT    |          41.1 |       37.51 | 55.37 (+17.86) |         37.52 | 55.35 (+17.83)     |
| Average    |         57.03 |       56.14 | 60.31 (+4.17)  |         56.33 | 60.10 (+3.85)      |

## B.5 FID AND CLASSIFICATION ACCURACY WITH A PRE-TRAINED CLASSIFIER FOR SYNTHETIC DATA MEASUREMENT.

Frechet Inception Distance (FID) is a metric that calculates the distance between feature vectors from real and generated images, and lower scores have been shown to correlate well with higher-quality images. Here, we provide the FID scores and classification accuracy with a pre-trained CLIP ViTB/16 model for measuring diversity and quality of synthesized images. We study with the Eurosat dataset. For the FID score, we do not have access to GLIDE training data, and thus we calculate FID between the downstream task ground-truth data and synthetic data generated by different strategies. As shown in Table A.16, LE could largely reduce FID and provide large diversity while hurting class fidelity. After combing with CF, FID further reduces and we also achieve a higher class fidelity. For few shot settngs, RG could largely reduce FID and yield the best class fidelity.

Table A.16: FID and classification accuracy with CLIP-ViT-B/16. 'z' for zero-shot and 'f' for few-shot.

| syn data   |   z: B |   z: LE |   z: LE+CF |   f: RF |   f: RG |
|------------|--------|---------|------------|---------|---------|
| FID        |  84.84 |   81.15 |      90.85 |   79.02 |   37.18 |
| Acc        |  44.27 |   35.84 |      48.03 |   46.95 |   48.76 |

## B.6 EXAMPLES OF SYNTHESIZED IMAGES FROM LANGUAGE ENHANCEMENT STRATEGY

Here, we provide successful and failure examples of synthesized images from language enhancement strategy. As shown in Figure A.3 (a) ∼ (c), language enhancement could introduce more diversity into the language prompts and lead to more diversified synthesized images for each class, such as introducing 'runway' in (a), 'pond' in (b), and 'red pillow' in (c). However, we also observe failure cases after the language enhancement process. As we can see in Figure A.3 (d) ∼ (f), after introducing some other items into the language prompts, the focus of the generated images may move to other objects rather than the target class. In some extreme failure cases we show here, the generated images may even not contain the desired class object.

Figure A.3: Examples of synthesized images from Language Enhancement strategy.

<!-- image -->

## B.7 VISUALIZATION: SYNTHETIC DATA IN ZERO-SHOT SETTINGS

We provide the visual illustration of ground-truth real data and synthesized images by different strategies for the zero-shot settings, i . e . basic ( B ), language enhancement ( LE ), and language enhancement and CLIP-based filtering ( LE+CF ). Here, we take the 'Highway or road' class in EuroSAT dataset as an example. As shown in Figure A.4, we could see that LE could help increase the diversity but may introduce noisy samples, but LE+CF could further select images with higher class fidelity and yields synthetic data with reduced domain gaps.

## B.8 VISUALIZATION: SYNTHETIC DATA IN FEW-SHOT SETTINGS

We provide the visual illustration of synthesized images by different strategies for the few-shot settings, i . e . basic ( B ), real filtering ( RF ), and real guidance ( RG ) as well as real images of the same class for comparison. Here, we take the 'forest' class in EuroSAT dataset as an example. As shown in Figure A.5, both RF and RG strategies produce images with reduced domain gap from the real images of the target domain. Further, RG significantly approaches the real images better than RF , demonstrating the effectiveness of the proposed RG method.

## B.9 VISUALIZATION: SYNTHETIC DATA FOR DIFFERENT DATASETS

Here, we provide synthesized images for different datasets (i.e., CIFAR10, Caltech101, Cars, ImageNet-Sketch, DTD) in Figure A.6. All images are randomly chosen rather than human-picked. Each row consists of images of the same class. We observe that for most datasets, synthesized images from the GLIDE model are of high quality, but there also exist cases that many unsatisfactory examples are generated, such as the DTD datasets.

We state that this is a limitation of the current text-image generation model that it may produce images of low quality for certain tasks, which mainly due to the domain gap between the training data of the generation model and the task. However, with the study of future text-image generation models, the quality of synthesized images is potentially growing higher constantly. Besides, the relatively lower quality images could be used for pre-training tasks for better improving performance of different down-stream tasks.

Figure A.4: Visualization of different strategies of synthetic data in zero-shot settings.

<!-- image -->

Figure A.5: Visualization of different strategies of synthetic data in few-shot settings.

<!-- image -->

## C ADDITIONAL DETAILS

## C.1 DENOISING DIFFUSION PROBABILISTIC MODEL

Denoising diffusion probabilistic model (DDPM) learns the data distribution through introducing a series of latent variables and matching the joint distribution. Formally, given a sample from the data distribution x 0 ∼ q ( x 0 ) , a forward process q ( x 1: T | x 0 ) = ∏ T t =1 q ( x t | x t - 1 ) progressively perturbs the data with Gaussian kernels q ( x t | x t - 1 ) := N ( √ 1 - β t x t - 1 , β t I ) , producing increasingly noisy latent variables x 1 , x 2 , ..., x T . Notably, x t can be directly sampled from x 0 thanks to the closed form: √

<!-- formula-not-decoded -->

where α t := 1 - β t and ¯ α t := ∏ t s =1 α s . In general, the forward process variances β t are fixed and increased linearly from β 1 = 10 - 4 to β T = 0 . 02 . Besides, T should be large ( e . g ., 1000) enough to ensure q ( x T | x 0 ) ≈ N (0 , I ) . Diffusion model aims to model the joint distribution q ( x 0: T ) which naturally involves a tractable sampling path for the marginal distribution q ( x 0 ) .

Specifically, the candidate distribution is formulated as a Markov chain with parameterized transition kernels:

<!-- formula-not-decoded -->

The training is thus achieved by optimizing a variational bound of negative log likelihood:

<!-- formula-not-decoded -->

The loss term L can be rewritten as:

<!-- formula-not-decoded -->

In practice, the core optimization terms are L t - 1 ( t &gt; 1) that can be analytically calculated since both two terms compared in the KL divergence are Gaussians, i . e .,:

<!-- formula-not-decoded -->

where ˜ µ t ( x t , x 0 ) := √ ¯ α t - 1 β t 1 - ¯ α t x 0 + √ α t (1 - ¯ α t - 1 ) 1 - ¯ α t x t and ˜ β t := 1 - ¯ α t - 1 1 - ¯ α t β t . Ho et al. (2020) fix Σ θ ( x t , t ) = σ 2 t I during training, where σ 2 t is set to be β t or ˜ β t . Through reparameterization trick (Kingma &amp; Welling (2013)) and empirical simplification (Ho et al. (2020)), the final training term is performed as follows:

<!-- formula-not-decoded -->

where ϵ ∼ N (0 , I ) and t is uniformly sampled between 1 and T .

After training, started from an initial noise map x T ∼ p ( x T ) = N (0 , I ) , new images can be then generated via iteratively sampling from p θ ( x t - 1 | x t ) using the following equation:

<!-- formula-not-decoded -->

## C.2 TEXT-TO-IMAGE GENERATION

The text-to-image diffusion model extends the basic unconditional diffusion model by changing the target distribution q ( x 0 ) into a conditional one q ( x 0 | c ) , where c is a natural language description.

Figure A.6: Visualization of different synthetic datasets.

<!-- image -->

The derivation of the training terms and sampling procedure are similar to Sec. C.1, except that a conditioning signal c is included. Besides, following the improved DDPM (Nichol &amp; Dhariwal (2021)), Σ θ is also estimated in GLIDE (Nichol et al. (2021)).

Especially, GLIDE employs a coarse-to-fine two-stage generation framework (Nichol &amp; Dhariwal (2021); Saharia et al. (2022c)) with two guidance techniques for balancing mode coverage and sample fidelity, namely classifier guidance (Dhariwal &amp; Nichol (2021)) and classifier-free guidance (Ho &amp;Salimans (2022)). Classifier guidance mainly relies on an extra trained noise CLIP model to provide feedback at intermediate sampling steps. Classifier-free guidance, on the other hand, randomly drops the text prompt with a fixed probability p during the training, which can be viewed as a joint training of an unconditional model ϵ θ ( x t | ∅ ) ( i . e ., ϵ θ ( x t ) ) and a conditional model ϵ θ ( x t | c ) . At each sampling step, the model's output is actually performed using an extrapolation as follows:

<!-- formula-not-decoded -->

where s is a guidance scale that can trade off sampling quality and diversity. In our work, we use classifier-free guidance with default setting s = 3 for all experiments since it achieves better results than CLIP guidance. To speed up the sampling process, DDIM (Song et al. (2020)) is utilized which allows the model to produce high-quality images within few seconds. We follow the default settings in GLIDE and set T = 100 in the coarse stage and T = 27 in the upsampler stage.

## C.3 REAL GUIDANCE ( RG ) STRATEGY

We elaborate how we use few-shot in-domain real images to guide the generation process for fewshot settings. In a normal text-to-image generation process, a pure noisy image x T ∼ N (0 , I ) would be sampled first as the initialization of the reverse path. Then, the pretrained GLIDE model iteratively predicts a less noisy image x t - 1 ( t = T, T - 1 , ..., 1) using the given text prompt c and the noisy latent image x t as inputs. In our case, we add noise to a reference image x ref 0 such that the noise level corresponds to a certain time-step t ⋆ :

<!-- formula-not-decoded -->

Then, rather than sampling from time-step T , we initialize the noisy latent variable as x ref t ⋆ and begin our denoising process from time-step t ⋆ , as illustrated in Algorithm 1. Note that the GLIDE model adopts a coarse-to-fine two-stage generation framework and involves classifier-free guidance. However, we omit them in Algorithm 1 for simplicity since our image-guidance strategy only modifies the start point and leaves the other settings unchanged. In this way, the generated images can share similar in-domain properties, and thus helping to close the domain gap. While small t ⋆ could synthesis images which are more similar to the reference image, it results in low diversity, which harms the classifier's learning. In the case of a large t ⋆ , x ref t ⋆ retains too little information from x ref 0 , causing the generated image to deviate from the domain. In our experiments, we conduct different trade-offs considering different few-shot settings. Empirically, we set t ⋆ as 15, 20, 35, 40, and 50 for shot 16, 8, 4, 2, and 1, respectively.

## Algorithm 1 Real Guidance ( RG ) Strategy

```
Input: Reference image x ref 0 , text prompt c and GLIDE model ( µ θ , Σ θ ) Output: Generated image x 0 1: # Noisy variable initialization 2: Select a time-step t ⋆ ∼ 1 , 2 , 3 , ..., T and random noise ϵ ∼ N (0 , I ) 3: Obtain initial noisy image x t ⋆ := x ref t ⋆ according to Eq. 9 4: # Random Sampling (could be replaced by DDIM for speed-up) 5: for s from t ⋆ to 1 do 6: µ, Σ ← µ θ ( x s , s, c ) , Σ θ ( x s , s, c ) 7: x s - 1 ← sample from N ( µ, Σ) 8: end for 9: return x 0
```

.

## C.4 SOFT-TARGET CROSS-ENTROPY LOSS

Example code for soft-target cross-entropy loss is shown below.

```
def soft_target_cross_entropy(logits, target, labels, T=2): # T: temperature for soft targets. loss_func_CE = torch.nn.CrossEntropyLoss() CE = loss_func_CE(logits, labels) soft_targets = torch.softmax(target/T, dim=1) SCE = torch. sum (-soft_labels * F.log_softmax(x, dim=-1), dim=-1) loss = 0.5 * CE + 0.5 * SCE return loss
```

## C.5 IMPLEMENTATION DETAILS

## C.5.1 ZERO-SHOT SETTING

For text-to-image generation process, we adopt the default hyperparameters from the official GLIDE text-to-image code. The input text of the basic strategy is 'a photo of a [CLASS]', and the input text of language enhancement strategy is 'a photo of a [SENTENCE]'. For language enhancement, we adopt an off-the-shelf word-to-sentence T5 model pre-trained on 'Colossal Clean Crawled Corpus' dataset (Raffel et al., 2020) and finetuned on CommonGen dataset (Lin et al., 2019). We generate 2000 synthetic images for each class in B and LE , and use a threshold of 1/N in CF where N is the number of classes. For LE , we generate 200 sentences for each class name.

For training on synthetic data for zero-shot recognition, we use AdamW (Loshchilov &amp; Hutter, 2017) optimizer and an initial learning rate of 0.002 that is decayed by the cosine annealing rule. We train for 30 epochs, and use weight decay of 0.1 and batch size of 512. For image preprocessing, we resize the image's short side to 224 while keeping the original aspect ratio.

For datasets in the zero-shot settings, we follow previous works (Zhou et al., 2022b; Gao et al., 2021) that use 11 datasets, and we excludes UCF101 since GLIDE exclude generating 'person' related content for privacy issues. Besides, we add another 7 popular datasets for more comprehensive evaluation. We do not conduct on all CLIP's 27 datasets since our computing resources are limited, and we believe our 17 datasets are already enough to study the effectiveness of synthetic data for zero-shot settings.

## C.5.2 FEW-SHOT SETTING

For text-to-image generation process in few-shot settings, our basic strategy and Real filtering strategy both apply the same process as in the zero-shot settings; and for Real guidance strategy, the generation process is illustrated in Sec. C.3. For synthetic image number, we generate 800 images per class for RG method to approximately match the number of images in B and RF .

For training methods in the few-shot settings, we provide the implementation details of phase-wise training and mix training. For phase-wise training, we utilize synthetic data and real data in two different phases, and the order of using synthetic data and real data also yields two variants. For syn → real, we first tune the classifier on synthetic data for 30 epochs, and then tune on real data for 30 epochs; and change the order for real → syn. For mix training, in each training iteration, we get a batch of real data input into the model and obtain the loss value of real part data, and also get a batch of synthetic data input into the model and get a loss value of synthetic part data, and then add two loss values as the final loss to do back propagation.

For training in the few-shot settings, we again use AdamW optimizer, weight decay of 0.1 and the cosine annealing rule. We use batch size of 32 for few-shot real images and 512 for synthetic images. For phase-wise training, we train for 30 epochs for each stage and use an initial learning rate of 0.002. For mix training, we train for 30 epochs and use an initial learning rate of 0.001, where the loss value from real data and synthetic data are added with a 1:1 ratio in each iteration. For image preprocessing, we adopt the same strategy as zero-shot settings.

## C.5.3 PRE-TRAINING SETTING

For pre-training settings, we adopt the LE strategy in zero-shot settings for generating massive amount of diversified synthetic pre-training data. For downstream-aware settings, we generate synthetic data by language prompts constructed from CIFAR-100 label space through the word-tosentence model, and we generate synthetic data of 1.2M, 2.4M, 3.6M. For downstream-agnostic settings, we generate from a generic label space of ImageNet-1K or ImageNet-2K, and data amount of 1.2M, 2.4M, 4M.

For training details of downstream-aware synthetic pre-training on CIFAR-100 label space, we use AdamW optimizer, weight decay of 0.9, batch size of 512, training epochs of 90, and the cosine annealing rule for adjusting learning rate. For the initial learning rate, we use 1 e -4 when pre-training from random initialization, and 1 e -5 when pre-training from ImageNet pre-trained weights. For data augmentation, we adopt random cropping, resizing, and random horizontal flip. For transfering on CIFAR-100 dataset, we train for 200 epochs and use a SGD optimizer with an initial learning rate of 0.003, which is multiplied by 0.2 at 60, 120, 160 epochs. We use batch size of 128 and weight decay of 5 e -4.

For downstream-agnostic synthetic supervised pre-training with ResNet50 backbone, we train for 90 epochs and use a SGD optimizer with an initial learning rate of 0.2 for training from random initialization and 0.001 for training from ImageNet pre-trained weights. We multiply the initial learning rate by 0.1 every 30 epochs. We use batch size of 512 and weight decay of 1e-4. For data augmentation, we also adopt random cropping, resizing, and random horizontal flip.

For downstream-agnostic synthetic supervised pre-training with DeiT-S backbone, we follow the training scripts for ImageNet pre-training and CIFAR-100 transfer learning in the official DeiT codebase 3 but only replace the pre-training data with our synthetic dataset. We do not change any hyper-parameters.

For downstream-agnostic synthetic self-supervised pre-training, i . e . Moco v2, we follow the hyperparameters of the original implementation of the paper when training from random initialization. When initialized from ImageNet pre-trained weights, we use a small initial learning rate of 0.003 and keep other hyperparameters the same.

For transfer evaluation of object detection on PASCAL VOC 2012, we use the Faster R-CNN (Ren et al., 2015) detector and backbones are initialized by the pre-trained weights. All the setups follow the evaluation protocols in Moco (He et al., 2020).

3 https://github.com/facebookresearch/deit