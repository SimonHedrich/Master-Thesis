## Boosting Object Detection via Diffusion-Based Data Augmentation

Kui Song           Weihua Li           Songyan Liu



School of Information Science and Engineering, Yunnan University

## ABSTRACT

Object detection continues to be a  significant challenge in computer vision. Despite advancements made possible through deep learning, these models predominantly depend on extensive and diverse annotated training data. Such data, unfortunately, often lacks representation of many real-world scenarios. To bridge this gap, we use target images from the original dataset to train a specialized generator. The main intent behind producing these images is to mimic the appearance of targets across a broader spectrum of real-world situations. Once integrated with the primary dataset, these synthetically generated images act as an effective augmentation to the original training set, encompassing scenarios and variations previously absent. This autonomous method eliminates the need for external data sources, proving to be more practical in most situations. Our empirical findings highlight significant improvements: with the ResNet-34 backbone, the mAP for SSD rose notably from 0.185 to 0.233. Furthermore, for small objects detected by Faster R-CNN with the ResNet-101 backbone,  there  is  a  pronounced  improvement  from  0.213  to  0.225.  These  results  underscore  our  method's  efficacy, especially in enhancing detection capabilities for underrepresented scenarios and smaller objects.

Keywords: Object detection, Annotated training data, Data augmentation, Generative models.

## 1. INTRODUCTION

Object detection is a fundamental task in computer vision that aims to locate and classify the objects of interest in images or videos. Object detection has many applications in various domains, such as face recognition, autonomous driving, security,  and  medical  imaging.  Meanwhile,  object  detection  is  also  a  challenging  task  which  requires  large-scale  and diverse datasets to train effective and robust models.

Current object detection methods are predominantly based on limited datasets. While these datasets might be extensive in scale, they still present significant challenges. First and foremost, the distribution of images within these datasets is often imbalanced, leading to an overrepresentation of some target categories while others are underrepresented. Additionally, there are instances of missing annotations or even erroneous ones, which further compromise the quality of the dataset. Lastly, these datasets are unlikely to cover all real-world scenarios comprehensively, which can impact the generalization capabilities of the trained models. Such constraints suggest that relying solely on these limited datasets for object detection might not yield optimal performance.

In this paper, we put forward a groundbreaking methodology tailored to address the persistent challenges observed in the realm of object detection. Central to our approach is the integration of a conditional generative model, fundamentally anchored in the underpinnings of the denoising diffusion probabilistic model (DDPM) [1]. Our method embarks on a journey that begins with the meticulous training of the generative DDPM [1] model using specially curated target images, chosen  for  their  intricate  annotation  characteristics.  Once  seasoned,  this  model  becomes  a  powerhouse,  churning  out images that span a wide array of categories. But our diligence doesn't stop there. Each generated image is subjected to a rigorous quality assurance process, ensuring that only the most representative and high-fidelity images are retained. These elite images are then seamlessly fused with the foundational dataset images during the object detection models' training phase,  a  strategic  pasting  method  that's  both  novel  and  effective.  Alongside  this,  we  craft  meticulous  annotations corresponding to these integrated targets, ensuring a comprehensive understanding for subsequent models. As portrayed in Figure 1, our method is more than just an incremental step; it's a paradigm shift. It magnifies data volume, accentuates its diversity, and most crucially, fortifies the underrepresented categories. The outcome is a robust solution to the endemic issue of absent annotations, setting the stage for a significant enhancement in the efficacy of object detection models when they are nurtured on this enriched dataset.

 Correspond Author. Email: liusongyan@ynu.edu.cn.

Figure 1 Given a U-net structured denoising diffusion probabilistic model, our data augmentation method conceptually generates target images without a background. These images are then augmented into the original dataset using a pasting operation.

<!-- image -->

In this study, we evaluate the efficacy of our proposed method by training and testing it on two classical object detection models, SSD [2] and Faster R-CNN [3]. Using the augmented dataset, we contrast our outcomes with benchmarks set on the original dataset. Consistently, our findings underline that our augmentation technique enhances the performance of both SSD [2] and Faster R-CNN [3] across varied object categories, also fortifying their robustness. Additionally, the images generated through our method are both realistic and diverse, which enriches the dataset and significantly boosts its utility for more effective object detection tasks. (see Figure 1)

The main contributions of this article are as follows:

(1) We introduced an innovative method using the DDPM-based conditional generative model to enrich the dataset, addressing the issue of missing annotations, especially in underrepresented categories.

(2) Our experiments demonstrated clear performance enhancements in two leading object detection models, SSD [2] and Faster R-CNN [3], particularly elevating their detection capabilities for medium and small objects.

The rest of this paper is organized as follows: Section 2 reviews some related works on object detection and data augmentation  methods.  Section  3  describes  our  proposed  method  in  detail,  including  data  preprocessing,  conditional generation training and generation, target image pasting, and object detection model training and testing. Section 4 presents our experimental setup and results. Section 5 concludes this paper and summarizes our contributions.

## 2. RELATED WORK

## 2.1 Object Detection Frameworks

Object Detection, a cornerstone task in computer vision, aims to detect and locate objects of interest within images or videos.  The  primary objective  is  to  discern the  position,  boundaries, and categorize these  objects  into  various  classes, making  it  an  essential  component  in  vision  recognition  systems,  synergizing  with  tasks  like  image  classification  and retrieval. Modern methodologies lean heavily on deep convolutional neural networks [4], with state-of-the-art methods primarily splitting into two categories. One-stage methods like YOLO [5], SSD [2], and RetinaNet [6] are designed for efficiency, favoring inference speed, which makes them ideal for real-time applications. Conversely, two-stage methods such as Faster R-CNN [3], Mask R-CNN [7], and Cascade R-CNN [8] focus on detection accuracy. These models operate in two phases: an initial proposal of candidate object locations followed by a refining stage that determines the object's precise position, resulting in heightened accuracy at the expense of speed.

Diving deeper into computer vision, object detection primarily bifurcates into anchor-based and anchor-free detectors. Anchor-based detectors harness pre-defined reference boxes, or 'anchors', spanning various scales and aspect ratios to predict potential object bounding boxes. Pioneers in this approach, such as Faster R-CNN [3], SSD [2], and YOLOv2 [9], systematically iterate over these anchors to determine the optimal bounding box coordinates along with their respective class probabilities.

On the flip side, anchor-free detectors forgo the reliance on predefined anchors, ushering in heightened adaptability. As an illustration, YOLO [5] formulates bounding boxes by focusing on regions adjacent to an object's center. Taking a unique  trajectory,  CornerNet  [10]  spots  the  top-left  and  bottom-right  corners  of  prospective  bounding  boxes  and subsequently pairs these to outline the definitive object boundary. FCOS [11], a further advancement in this realm, employs a 4D vector to spatially denote the location of the bounding box. At their core, these models adopt dense sampling strategies, treating each spatial point as a plausible object candidate subject to both classification and regression operations. A pivotal post-processing phase shared by numerous models is Non-Maximum Suppression (NMS) [12], an essential technique to discard superfluous bounding boxes during inference.

The  evolution  of  object  detection  has  also  witnessed  the  inception  of  sparse  detectors.  DETR  [13],  for  instance, sidesteps traditional handcrafted components, directly formulating predictions. Sparse R-CNN [14], another avant-garde model, leverages a finite set of learnable bounding boxes, formulating object candidates and subsequently showcasing formidable performance metrics.

While there have been many improvements in object detection, these models have a couple of shared challenges. First, they heavily depend on a lot of labeled data. To train these models effectively, we need a diverse range of images with the correct labels. However, collecting and labeling such a vast amount of images is both time-consuming and hard. Secondly, these  models  often  struggle  to  detect  smaller  objects  accurately.  Additionally,  obtaining  labeled  pictures  for  some categories  can  be  tough,  leading  to  some  categories  being  underrepresented.  This  can  reduce  the  model's  accuracy, especially for those underrepresented categories. Because of these challenges, methods like data augmentation become crucial. They can help when we don't have enough data, when the data isn't diverse, or when it's not balanced.

## 2.2 Data Augmentation

Data Augmentation is a widely-used technique in machine learning and computer vision. This strategy employs various methods to increase both the quantity and diversity of data. By introducing modifications, data augmentation expands the range of examples in an original dataset, thereby increasing its size and variety. Serving as an effective regularizer, it plays a pivotal role in preventing overfitting during the training of machine learning models [15].

In computer vision, fundamental augmentation techniques such as cropping, flipping, and rotation play pivotal roles. Common techniques  span  Flipping,  Rotation,  Scaling,  Random  Cropping,  Color  Jittering,  Gaussian  Noise,  Random Blurring, and Random Erasing [16]. Beyond traditional techniques, GANs (Generative Adversarial Networks) [17] have emerged as a promising avenue for data augmentation, harnessing their capability to generate realistic synthetic data. Another strategy involves targeted transformations and random cropping tailored to the dataset's specificities. Regardless of the method's sophistication, the fundamental aim of data augmentation remains unvaried: to enhance model performance and ensure superior predictive outcomes, especially when faced with limited training data.

As we explore deeper, the significance of dataset expansion becomes increasingly evident [18]. It isn't merely about increasing numbers; there's a profound need to bolster datasets for enhancing the prowess of deep learning models. By adding more instances and diversifying the training dataset, models can achieve better generalization, reduce susceptibility to interference, and address challenges such as limited data and class imbalances.

Traditional  data  augmentation  methods,  while  helpful,  often  present  limitations.  They  might  make  superficial alterations, failing to encapsulate the nuanced variations of real-world data. Furthermore, while they offer some respite from overfitting, they may not sufficiently address data imbalance, especially in complex datasets. Resampling techniques, including  oversampling,  undersampling,  and  SMOTE  [19],  seek  to  balance  classes,  but  occasionally  at  the  cost  of overfitting or data redundancy [20].

Generative Adversarial Networks (GANs) have risen as a modern solution to generate new data samples. Yet, they come with inherent challenges like mode collapse and are resource-intensive. Here, Diffusion-based Generative Models [21]  come into  play,  offering  several  advantages  over  traditional  GANs.  Unlike  GANs,  which  involve  a  challenging adversarial  training  process,  DDPMs  provide  a more  stable  and consistent  training  regimen.  The diffusion  process  in DDPM offers a natural way to model the data distribution, reducing common issues faced in GANs.

Building on DDPM's foundational concepts, the Elucidating the Design Space of  Diffusion-Based Generative Models (EDM)  [22]  offers  an  advanced  approach.  EDM  [22]  employs  a  modular  design,  dissecting  and  examining  various components of diffusion models. This not only provides flexibility but also ensures robust performance across diverse datasets. Given its stability and efficiency, DDPM [1], and by extension EDM [22], emerges as a potent tool for data augmentation.

Our proposed method integrates these advancements. Rather than solely relying on traditional augmentation or GANgenerated synthetic data, our approach melds the strengths of conditional generative models with strategic fusion techniques. Our primary goal is to navigate the challenges of data diversity and missing annotations. By assimilating the high-fidelity images generated through our method into existing datasets, we aim to create a holistic and robust training dataset. Consequently, this rich dataset, when employed to train object detection models, promises enhanced performance across a multitude of scenarios.

Figure 2 Overview of the Methodology. A visual representation of the three-stage approach adopted in our method. It starts with the training of a conditional generative model, progresses to the generation and fusion of images with the original dataset, and culminates in the training of object detection models on the enhanced dataset.

<!-- image -->

## 3. METHOD

Figure 2 demonstrates our unique three-stage approach to enhancing object detection performance. Beginning with the training of a conditional generative model, followed by the generation and fusion of high-quality target images with the original dataset, and concluding with the training of object detection models on this enriched dataset.

Training of the Conditional Generative Model :  At  this  stage,  we train a conditional generative model using the original target images. These images are extracted from a representative training dataset based on their annotation details. Feeding them into our conditional generative model allows us to fine-tune the model, catering specifically to the dataset's requirements.

Generation &amp; Fusion with the Original Dataset : Upon obtaining a well-trained conditional generative model, we generate images from various categories. A separate Quality Check model ensures that only images of high caliber are selected. We then subject these high-quality images to elementary data augmentation techniques and strategically fuse them  with the  original  dataset.  This  fusion  occurs  by  pasting  the  generated  images  onto  the  original  dataset  images, resulting in a more robust and diverse dataset.

Training on the Augmented Dataset : The final step of our process involves training object detection models. Both the augmented and the original datasets feed into the object detection model in succession. This ensures that the model trains on a broad spectrum of data, benefiting from both original and augmented information. By using this augmented dataset, we aim to increase accuracy and stability in object detection tasks.

## 3.1 Training of the Conditional Generative Model

## 3.1.1 Data Preprocessing

To prepare the dataset for the generative model, we adopt two distinct approaches. The first method involved extracting target images from the COCO2017 [23] train dataset based on the bounding box (bbox) annotations. These images, as a result, contain the background context, providing spatial and semantic information to the generative model. In contrast, the second approach utilized mask annotations for extraction, yielding target images devoid of any background. For the purpose of training uniformity, these background-less images are all set against a black backdrop.

## 3.1.2 Generative Model Selection

We train two different generative models with different methods:

Custom U-net-based DDPM : We employ a U-Net [24] architecture tailored for conditional image generation tasks by embedding both temporal and label information. The network follows the standard encoding-decoding scheme with modifications. The encoding layers comprise max-pooling and double convolutions enriched with residual connections. An  attention  mechanism,  reminiscent  of  Transformer  models,  is  incorporated  at  various  depths  to  enhance  context awareness.

The  decoding  process  is  facilitated  using  transposed  convolutions,  supplemented  by  skip  connections  from  the encoding  layers,  preserving  spatial  resolution.  A  unique  addition  is  the  positional  encoding  aligned  with  the  Vision Transformer approach, merged with label embeddings. This integration enables the model to comprehend the relative positioning of data points, refining the generated output.

The final convolutional layers fine-tune the output, and an embedding layer conditions the model on class labels, extending its applicability beyond mere image translation.

During the image generation phase, an influence factor is introduced to modulate the contribution of noise in the generation process. By adjusting this factor, we can control the degree of noise integration, thereby influencing the visual characteristics of the generated images. This design choice provides a mechanism to strike a balance between pristine outputs  and  those  with  intentional  noise  attributes,  catering  to  diverse  application  needs  and  enhancing  the  model's versatility in producing varied image qualities. (see Figure 3)

EDM : In our study, building upon the foundation established by EDM [22], we tailored the model to fit the specifics of our custom dataset derived from the detection training set. Using mask annotations, we segmented target images and ensured uniformity by setting their backgrounds to black. These images are then resized to a 64x64 resolution, aligning our preprocessing steps with the requirements of the EDM [22] model. We initialize our model using the ImageNet pretrained version from EDM [22]. Recognizing the unique nature of our dataset, characterized by target images set against black backgrounds, we perform fine-tuning to capture our data's nuances. After multiple training epochs, our model's performance is optimized. This evaluation technique mirrors the method detailed by EDM [22].

By embracing two different techniques in data preprocessing, we not only improve the diversity of the generated images but also enable the model to learn varied contexts. This is vital, as both context-rich and isolated target images hold their unique significance in object detection. Furthermore, the dual model strategy allows us to gauge the performance of our innovative approach against a well-established standard, ensuring the credibility and effectiveness of our methodology. For visual results of the generated samples, please refer to Figure 5.

## 3.2 Augmentation, Integration, and Comprehensive Training on the Enhanced Dataset

## 3.2.1 Image Generation and Quality Check Assessment

Utilizing  our  finely-tuned  conditional  generative  model,  we  generate  a  diverse  array  of  images  spanning  multiple categories. To uphold the standard and fidelity of our dataset, we deploy a dedicated deep learning-based Quality Check model. This model's primary function isn't just classification but a stringent assessment of the generated images. It filters out images to ensure only those that meet our stringent criteria in terms of clarity, relevance, and realism are carried forward. Thus, every retained image becomes a testament to quality, primed for subsequent processing.

## 3.2.2 Augmentation and Fusion Techniques

After the vetting process, the selected images are meticulously augmented. We modify their attributes with a special emphasis on scaling to match the context and dimensions of the original dataset images, ensuring seamless integration in the upcoming fusion phase. As we move into the fusion phase, our efforts in careful preparation come to the fore. The augmented images are strategically blended into the  original dataset, with a clear emphasis on preserving the inherent ambiance of the pre-existing scenes. Each image insertion is carefully orchestrated to ensure both contextual and aesthetic harmony with the original content. Through this process, our goal is to not only increase the dataset's volume but also to imbue it with rich contextual variety and depth. Incorporating a deep learning-based Quality Check model further ensures that only high-fidelity images are included, maintaining the authenticity and credibility of our augmented dataset.

Figure 3 DDPM: Employing a trained generation model to produce images under varied influencing factors. From top to bottom, the influence of these factors grows. In our experiments, configurations from the third row are utilized.

<!-- image -->

In summary, our methodology presents a comprehensive and efficient three-stage approach, as depicted in Figure 2. We begin by meticulously training a conditional generative model using carefully curated target images. This ensures that our  model  is  highly  attuned  to  the  dataset's  specific  requirements.  Subsequently,  we  generate  images  across  varied categories, ensuring their quality through a dedicated Quality Check model. These high-quality images undergo careful augmentation and are then seamlessly blended into the original dataset. Such strategic fusion ensures a harmonious and enriched dataset, fostering both volume and contextual diversity.

The final step is the actual training of object detection models using this enhanced dataset. By ensuring the models are exposed to both original and augmented data, we provide a broader learning spectrum, maximizing their potential. Our method stands as a testament to addressing prevalent challenges in the computer vision domain, particularly the scarcity of diverse training data and the conundrum of missing annotations. By transforming the original training dataset into a richer and more diversified version, we pave the way for object detection models to achieve heightened performance, making our methodology both potent and crucial for further advancements in computer vision.

## 4. EXPERIMENT

In this section, we meticulously dissect each stage of our research methodology. We elucidate our experimental setup, decision drivers, and the ramifications of each choice, ensuring a comprehensive understanding of our endeavors.

To provide a clearer roadmap for our readers: Initially, we concentrate on the extraction of target images from the original training set, leveraging mask annotations for precise image segmentation. This acts as the foundational step to curate a tailored subset meant for the training of our generative and Quality Check models. Our Quality Check model functions as a classification model, filtering to retain only highest fidelity generated images. Following this, we employ strategic pasting strategies, integrating these curated images with the Original training set, culminating in a more diverse and augmented dataset. Our aim, achieved through these carefully orchestrated steps, is to harness this enriched dataset to boost the performance of our primary object detection models. Importantly, our pasting operations are executed during the object detection model training process. This means that each training iteration results in a uniquely augmented dataset, which further enhances the robustness of our approach.

Figure 4 Enhancing Original Dataset with DDPM-Generated Images. It can increase the accuracy of the SSD model, but the background of small images also limits its performance.

<!-- image -->

## 4.1 Dataset and Preliminary Setup

MS COCO 2017 : For our experiments, we strictly confine ourselves to the training subset of the MS COCO 2017 [23] dataset. This training set, while expansive with 80 diverse object categories and 118287 training images, remains our sole focus, and we abstain from incorporating any external datasets or the test portion of COCO 2017 [23]. The choice is strategic, ensuring that any enhancements observed could be attributed directly to our augmentation methodology and not external data influx.

## 4.2 Training, Generation, and Evaluation of Conditional Generative Models

## 4.2.1 Custom DDPM Model

For our evaluations, we utilize a specially curated subset of the MS COCO 2017 [23] dataset, which we refer to as " COCO26 ". This subset is selected with a specific focus on the categories that had the fewest bounding box annotations in the main dataset. Our primary objective is to tackle the inherent class imbalance in MS COCO by giving priority to these underrepresented categories. For visual results of the generated samples, please refer to Figure 3.

Our data preprocessing steps include resizing the images, cropping them randomly, normalizing the pixel values, and transforming the data into tensors. When it comes to training, we configure our model to process the data in batches of 16 and set the learning rate at 0.00015. To measure the model's performance, we use the Mean Squared Error (MSE) loss. Additionally, we incorporate the Wandb tool to log the performance metrics, and to ensure a more consistent training trajectory, we apply the Exponential Moving Average (EMA) to the model weights.

Throughout the training process, we periodically evaluate the model's capability in conditional generation, placing an emphasis on the underrepresented categories, such as sunflowers, roses, and tulips. These evaluations provide us valuable insights into how the model is evolving and its proficiency in generating images.

Once training concludes, we identify the model that showcased the lowest validation MSE as our best performer. We then harness this model for a range of image generation tasks. This included generating images conditioned on specific labels and creating images with consistent visual attributes using a predetermined noise seed.

## 4.2.2 Referenced EDM Model

For our experimental framework, we employ a multi-GPU training strategy, harnessing the computational prowess of eight graphics cards. Our dataset is derived meticulously from the COCO 2017 [23] collection. By leveraging the Mask annotations, we segment the target images. To ensure quality, only high-fidelity images are retained, thereby filtering out less significant ones. Post this refinement process, our dataset is composed of approximately 200,000 images.

In our training procedure, the model is set to a conditional generative training mode. We employ the ADM architecture to underpin our model's structure. A total training duration of 2500 units is allocated, and during this process, we feed data to the model in batches of 4096 images. An optimized learning rate of 1e-4 is applied to ensure steady and efficient learning. Throughout the training, we apply an exponential moving average with a factor of 50 to smoothen our model's performance. A dropout rate of 10% is introduced to minimize overfitting, and we opt against data augmentation, keeping it deactivated. For enhanced computational efficiency and to expedite the training process, we use 16-bit floating-point precision.

Figure 5 Generation Results of the EDM Model. The generated images are assessed for quality using the Frechet Inception Distance (FID) metric. An impressive score of 5.77 highlights the high caliber of our produced images.

<!-- image -->

Figure 6 The outcomes of our Quality Check model. On the left, the displayed images are those that have passed through our Quality Check model, demonstrating the efficacy of the generated images in being correctly recognized during training. On the right, we see images that have not been vetted. It's evident that the unfiltered set contains some low-quality images that the model struggles to correctly identify, thereby adversely affecting its performance.

<!-- image -->

These hyperparameters and training configurations are judiciously chosen, aiming to optimize the performance and achieve the most accurate results for our model.

Upon the completion of our training process, we leverage the trained generative model to synthesize images. We generate a substantial volume of images, totaling around 200,000. These images, showcasing distinct target objects  set against a black backdrop, are then subjected to a performance evaluation using the Frechet Inception Distance (FID) [25] metric.  The  outcome  is  commendable:  our  model  yields  an  FID  score  of  5.77,  indicating  a  high-quality  generation consistent with the reference dataset. This low score emphasizes the model's capability to replicate the statistical properties and intricate details of the reference images, showcasing the effectiveness of our training process.

## 4.3 Custom Quality Check Model

To ensure the integration of high-quality images into our augmentation pipeline, we introduce a custom Quality Check model. This model, essential for evaluating the produced images, is built upon a modified ResNet50 [26] architecture and serves the purpose of coarse-grained categorization. In the absence of this quality-checking module, certain low-quality images from the generative process could adversely affect the training. This negative impact is qualitatively illustrated in Figure 6. For a quantitative perspective on how these low-quality images can alter model performance, one can refer to the data presented in Table 2.

| Table 1                | SSD [2] Model Performance with Augmented Dataset. The impact of different pasting times.   | SSD [2] Model Performance with Augmented Dataset. The impact of different pasting times.   | SSD [2] Model Performance with Augmented Dataset. The impact of different pasting times.   | SSD [2] Model Performance with Augmented Dataset. The impact of different pasting times.   | SSD [2] Model Performance with Augmented Dataset. The impact of different pasting times.   | SSD [2] Model Performance with Augmented Dataset. The impact of different pasting times.   | SSD [2] Model Performance with Augmented Dataset. The impact of different pasting times.   |
|------------------------|--------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| Dataset Configuration  | IoU area                                                                                   | 50 ： 90 All                                                                                | 50 All                                                                                     | 75 All                                                                                     | 50 ： 95 Small                                                                              | 50 ： 95 Medium                                                                             | 50 ： 95 Large                                                                              |
| Original COCO2017 [23] | Original COCO2017 [23]                                                                     | 0.185                                                                                      | 0.338                                                                                      | 0.188                                                                                      | 0.042                                                                                      | 0.203                                                                                      | 0.306                                                                                      |
| Aug. (26-URC) * 1      | Aug. (26-URC) * 1                                                                          | 0.197                                                                                      | 0.348                                                                                      | 0.203                                                                                      | 0.045                                                                                      | 0.212                                                                                      | 0.324                                                                                      |
| Aug. (26-URC) * 2      | Aug. (26-URC) * 2                                                                          | 0.233                                                                                      | 0.396                                                                                      | 0.241                                                                                      | 0.059                                                                                      | 0.256                                                                                      | 0.380                                                                                      |
| Aug. (26-URC) * 3      | Aug. (26-URC) * 3                                                                          | 0.223                                                                                      | 0.378                                                                                      | 0.233                                                                                      | 0.054                                                                                      | 0.245                                                                                      | 0.364                                                                                      |
| Aug. (26-URC) * 4      | Aug. (26-URC) * 4                                                                          | 0.232                                                                                      | 0.390                                                                                      | 0.241                                                                                      | 0.056                                                                                      | 0.253                                                                                      | 0.379                                                                                      |
| Aug. (26-URC) * 5      | Aug. (26-URC) * 5                                                                          | 0.223                                                                                      | 0.376                                                                                      | 0.234                                                                                      | 0.052                                                                                      | 0.245                                                                                      | 0.366                                                                                      |

The model's architecture is a variant of the ResNet50 [26], where we retain the primary features from the pre-trained ResNet50  [26]  but  replace  the  topmost  fully  connected  layer.  The  revised  top  layer  is  designed  to  suit  our  specific classification requirement of 80 classes.

For  training,  we  utilize  a  curated  dataset  of  high-quality  images  from  the  COCO  2017  [23]  train  dataset.  Data preprocessing includes resizing images to a uniform dimension of 64x64. The model is trained using the Adam optimizer with a learning rate of 0.001 and the Cross-Entropy Loss criterion. A batch size of 2048 is used for training, while validation employs a batch size of 1024. We train the model over 10 epochs, evaluating its performance on the validation set at the end of each epoch to ensure optimal convergence and robustness. The progress of the model is tracked using both the training loss and validation accuracy metrics.

By utilizing this classifier post-generation, we could effectively filter and include only those images that adhered to a predefined quality threshold in our augmented dataset.

## 4.4 Performance Evaluation on Augmented Dataset

## 4.4.1 SSD Model Performance with Augmented Dataset

For  our  experiments,  the  SSD  [2]  model  with  a  ResNet34  [26]  backbone  is  trained  using  the  PaddleDetection framework [27]. The training spans 70 epochs starting with a learning rate of 0.05, which decays at the 48th and 60th epochs. The  Momentum  optimizer is  employed alongside  an L2 regularization.  Data augmentations,  such as  random cropping and flipping, are incorporated during training with a batch size of 64. For evaluation and testing, images undergo resizing to 300x300 and are processed either in batches or individually.

Table 1 presents the mean average precision (mAP) of the SSD [2] model across different dataset configurations. The "Aug. (26-URC) * n" configuration pertains to the enhanced dataset with target images from the 26 least represented COCO 2017 [23] train dataset categories. These images, measuring 64x64, are pasted n times onto original images.

The  results  highlight  that  integrating  augmented  data,  particularly  from  underrepresented  categories,  positively impacts the SSD [2] model. As the pasting frequency rises, the performance enhances, but only up to a point. After two pastings, there's a performance plateau, suggesting that excessive augmentation doesn't necessarily yield better detection.

The improvements aren't limited to mAP metrics. There's also a noticeable boost in detecting medium-sized targets, consistent with our augmentation goal of emphasizing medium and lesser-represented categories.

However, augmentation has its limits. While adding generated images enhances category representation, they don't perfectly mirror the original dataset's distribution. This hints at an upper limit to augmentation benefits.

In conclusion, data augmentation can significantly boost model performance, but there's a balance to maintain. Overaugmentation risks overfitting or overshadowing other dataset elements. Therefore, it's crucial to optimize augmentation techniques, balancing robustness and generalization.

## 4.4.2 Faster R-CNN Model Performance with Augmented Dataset

Pasting images with backgrounds can affect how real the images look. (see Figure 4) To keep images looking natural, we paste images without backgrounds. We also try other ways to improve our images, such as using only high-quality images, adjusting the size of the images we paste, and varying how often we paste images. We also check if blending the new images with the original dataset improves the training results. In addition, we conduct tests using a more accurate Faster R-CNN [3] model.

Table 2 Performance Evaluation of Faster R-CNN [3] with Various Augmentation Strategies and Backbones. "Augmentation (5x5 ~ 32x32)" means target images are resized within this range. "Fine." shows if models are fine-tuned on the original data after augmented training. "BK." denotes the backbone used. Images without "No Filter" passed our Quality Check, but all are generated using the EDM [22] model.

| Fine.                | Augmentation (resize)   | IoU area   | Bk.    |   50 ： 90 All |   50 All |   75 All |   50 ： 95 Small |   50 ： 95 Medium |   50 ： 95 Large |
|----------------------|-------------------------|------------|--------|---------------|----------|----------|-----------------|------------------|-----------------|
|                      | Baseline                |            | Rs.50  |         0.367 |    0.570 |    0.392 |           0.205 |            0.415 |           0.496 |
|                     | 64*64                   | Aug * 2    | Rs.50  |         0.365 |    0.564 |    0.390 |           0.202 |            0.415 |           0.498 |
|                     | 64*64(No Filter)        | Aug * 2    | Rs.50  |         0.364 |    0.562 |    0.386 |           0.194 |            0.415 |           0.495 |
|                     | 64*64                   | Aug * 3    | Rs.50  |         0.365 |    0.563 |    0.387 |           0.197 |            0.419 |           0.495 |
|                     | 64*64                   | Aug * 3    | Rs.50  |         0.370 |    0.571 |    0.395 |           0.203 |            0.420 |           0.500 |
|                     | 32*32                   | Aug * 3    | Rs.50  |         0.370 |    0.572 |    0.392 |           0.211 |            0.419 |           0.495 |
|                     | 5*5 ~ 32*32             | Aug * 2    | Rs.50  |         0.371 |    0.573 |    0.393 |           0.211 |            0.421 |           0.496 |
|                     | 5*5 ~ 32*32             | Aug * 3    | Rs.50  |         0.372 |    0.573 |    0.394 |           0.210 |            0.419 |           0.498 |
|                      | Baseline                |            | Rs.101 |         0.390 |    0.594 |    0.416 |           0.213 |            0.442 |           0.530 |
|                     | 5*5 ~ 32*32             | Aug * 2    | Rs.101 |         0.398 |    0.599 |    0.426 |           0.225 |            0.447 |           0.536 |
|                     | 5*5 ~ 32*32             | Aug * 3    | Rs.101 |         0.397 |    0.597 |    0.424 |           0.222 |            0.446 |           0.539 |
| Combine Ori with Aug | Combine Ori with Aug    | Aug * 3    | Rs.101 |         0.378 |    0.573 |    0.404 |           0.213 |            0.428 |           0.511 |

Our Faster R-CNN [3] model, utilizing either ResNet50 or ResNet101 [26] backbone, trains under the guidance of the PaddleDetection framework [27]. The model is trained for 12 epochs with an initial learning rate of 0.01, which decays at the 8th and 11th epochs. The Momentum optimizer is used with L2 regularization, featuring a decay factor of 0.0001.

During training, the dataset undergoes multiple transformations: images are randomly resized to dimensions within the range [640, 1333] to [800, 1333], and a random flip is applied with a  50% probability. Subsequently, the images are normalized and permuted. Each training batch consists of a single image, ensuring an equal representation across batches.

For evaluation and testing, images are resized to 800x1333 pixels, maintaining the original aspect ratio. The dataset transformations are consistent across both evaluation and testing, including normalization, permutation, and padding. Both evaluation and test datasets are processed in single-image batches, with no shuffling applied to ensure consistency in results.

As show in Table 2, one notable observation from the table is that the performance witnesses a more pronounced enhancement when the resizing range for the pasted images is set between 5x5 to 32x32. This optimized resizing range seems to strike a balance, ensuring that the pasted images neither dominate the scenes nor become inconspicuous, offering maximum benefit to the detection model.

Furthermore, a crucial takeaway from the table is the consistent enhancement in performance when models trained on the augmented dataset are further finetuned on the original COCO 2017 [23] dataset. Without this finetuning step, models demonstrated a tendency to be skewed by the augmented data, potentially disrupting the spatial distribution inherent to the original dataset.

Interestingly, the data point labeled "Combine Ori with Aug" indicates that simply combining the original with the augmented dataset and training  alternately  doesn't  boost  performance.  One  plausible  reason  might  be  the  abrupt  and alternate shifts in the data distribution during training, which may hinder the model's ability to converge optimally. Instead of gradually learning and adapting, the model faces starkly contrasting data sets in rapid succession, leading to potential oscillation in the optimization landscape and preventing stable convergence.

The Faster R-CNN [3] model, especially with a ResNet101 [26] backbone, exhibits significant enhancements on our augmented dataset due to our unique augmentation methods. Most notably, the box Average Precision (box AP) rises by 1.0 (percentage points) at a stringent IoU threshold of 0.75. Additionally, for small-sized targets, which are inherently difficult to detect due to their sparse representation, the box AP increases by 1.2 (percentage points). These results not only highlight the effectiveness of our methodology in addressing challenges in object detection but also underscore its superiority in boosting accuracy for smaller and medium-sized objects.

## 5. CONCLUSION

In this research, we turn our attention to make a transform on the original training set to relieve the data-unbalance problem  in  object  detection.  By  innovatively  employing  a  DDPM-based  conditional  generative  model,  we  enrich  the dataset,  effectively  addressing the annotation  shortfalls.  This  augmentation, underpinned  by our  rigorous  experiments, manifests in significant performance boosts in SSD and Faster R-CNN models, especially elevating the detection accuracy for  medium  and  small  objects.  Our  cohesive  approach  to  data  augmentation,  balancing  quality  with  context,  offers  a tangible solution to enduring challenges in object detection, holding implications for both academic exploration and realworld applications.

Acknowledgments This work was supported by the Yunnan Provincial Research Foundation for Basic Research, China [Grant No. 202201AU070024] and the Yunnan Provincial Foundation for Leaders of Disciplines in Science and Technology, China [Grant No. 202305AC160014].

## REFERENCES

- [1] Ho, J., Jain, A., and Abbeel, P., ' Denoising Diffusion Probabilistic Models,' Proc. Adv. Neural Inf. Process. Syst. 33, 6840-6851(2020).
- [2] Liu, W., Anguelov, D., Erhan, D., Szegedy, C., Reed, S., Fu, C.-Y., and Berg, A.C., ' SSD: Single Shot MultiBox Detector,' Proc. Computer Vision - ECCV 2016, 21-37 (2016).
- [3] Ren, S., He, K., Girshick, R., and Sun, J., ' Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks,' Proc. Adv. Neural Inf. Process. Syst. 28, (2015).
- [4] Krizhevsky, A., Sutskever, I., and Hinton, G.E., ' ImageNet Classification with Deep Convolutional Neural Networks,' Proc. Adv. Neural Inf. Process. Syst. 25, (2012).
- [5] Redmon, J., Divvala, S., Girshick, R., and Farhadi, A., 'You Only Look Once: Unified, Real-Time Object Detection,' Proc. IEEE Conf. Comput. Vision Pattern Recogn., 779-788, (2016).
- [6] Lin, T.-Y., Goyal, P., Girshick, R., He, K., and Dollár, P., 'Focal loss for dense object detection,' Proc. IEEE Int. Conf. Comput. Vision, 2980-2988, (2017).
- [7] He, K., Gkioxari, G., Dollá r, P., and Girshick, R., 'Mask R-CNN,' arXiv:1703.06870, (2018).
- [8] Cai, Z., and Vasconcelos, N., 'Cascade R-CNN: Delving Into High Quality Object Detection,' Proc. IEEE/CVF Conf. Comput. Vision Pattern Recogn., 6154-6162, (2018).
- [9] Redmon, J., and Farhadi, A., 'YOLO9000: Better, Faster, Stronger,' Proc. IEEE Conf. Comput. Vision Pattern Recogn., 6517-6525, (2017).
- [10] Duan, K., Xie, L., Qi, H., Bai, S., Huang, Q., and Tian, Q., 'Corner Proposal Network for Anchor-Free, TwoStage Object Detection,' Proc. Computer Vision - ECCV 2020, 399-416, (2020).
- [11] Tian, Z., Shen, C., Chen, H., and He, T., 'FCOS: Fully Convolutional One-Stage Object Detection,' Proc. IEEE/CVF Int. Conf. Comput. Vision, 9626-9635, (2019).
- [12] Bodla, N., Singh, B., Chellappa, R., and Davis, L.S., 'Soft-NMS - Improving Object Detection with One Line of Code,' Proc. IEEE Int. Conf. Comput. Vision, 5562-5570, (2017).
- [13] Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., and Zagoruyko, S., 'End-to-End Object Detection with Transformers,' Proc. Computer Vision - ECCV 2020, 213-229, (2020).
- [14] Sun, P., Zhang, R., Jiang, Y., Kong, T., Xu, C., Zhan, W., Tomizuka, M., Li, L., Yuan, Z., et al., 'Sparse RCNN: End-to-End Object Detection with Learnable Proposals,' Proc. IEEE/CVF Conf. Comput. Vision Pattern Recogn., 14449-14458, (2021).
- [15] Shorten, C., and Khoshgoftaar, T.M., 'A survey on Image Data Augmentation for Deep Learning,' J. Big Data 6(1), 60, (2019).
- [16] Kisantal, M., Wojna, Z., Murawski, J., Naruniec, J., and Cho, K., 'Augmentation for small object detection,' arXiv:1902.07296, (2019).
- [17] Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., and Bengio, Y., 'Generative adversarial networks,' Commun. ACM 63(11), 139-144, (2020).
- [18] Sun, C., Shrivastava, A., Singh, S., and Gupta, A., 'Revisiting unreasonable effectiveness of data in deep learning era,' Proc. IEEE Int. Conf. Comput. Vision, 843-852, (2017).
- [19] Chawla, N.V., Bowyer, K.W., Hall, L.O., and Kegelmeyer, W.P., 'SMOTE: Synthetic Minority Over-sampling Technique,' J. Artif. Intell. Res. 16, 321-357, (2002).
- [20] Zhang, C., Bengio, S., Hardt, M., Recht, B., and Vinyals, O., 'Understanding deep learning (still) requires rethinking generalization,' Commun. ACM 64(3), 107-115, (2021).
- [21] Nichol, A., and Dhariwal, P., 'Improved Denoising Diffusion Probabilistic Models,' arXiv:2102.09672, (2021).
- [22] Karras, T., Aittala, M., Aila, T., and Laine, S., 'Elucidating the Design Space of Diffusion-Based Generative Models,' arXiv:2206.00364, (2022).
- [23] Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., and Zitnick, C.L., 'Microsoft coco: Common objects in context,' Proc. Computer Vision-ECCV 2014: 13th European Conference, 740-755, (2014).
- [24] Ronneberger, O., Fischer, P., and Brox, T., 'U-net: Convolutional networks for biomedical image segmentation,' Proc. Med. Image Comput. Comput.-Assist. Interv. - MICCAI 2015: 18th Int. Conf., 234-241, (2015).
- [25] Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., and Hochreiter, S., 'GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium,' Proc. Adv. Neural Inf. Process. Syst. 30, (2017).
- [26] He, K., Zhang, X., Ren, S., and Sun, J., 'Deep residual learning for image recognition,' Proc. IEEE Conf. Comput. Vision Pattern Recogn., 770-778, (2016).
- [27] PaddlePaddle Authors ,'PaddleDetection, Object detection and instance segmentation toolkit based on PaddlePaddle,' https://github.com/PaddlePaddle/PaddleDetection, (2019).