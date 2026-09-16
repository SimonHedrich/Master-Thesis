## DEMYSTIFYING MMD GANS

Mikołaj Bi´ nkowski ∗ Department of Mathematics Imperial College London mikbinkowski@gmail.com

## Danica J. Sutherland ∗ , Michael Arbel &amp; Arthur Gretton

Gatsby Computational Neuroscience Unit

University College London

{danica.j.sutherland,michael.n.arbel,arthur.gretton}@gmail.com

## ABSTRACT

We investigate the training and performance of generative adversarial networks using the Maximum Mean Discrepancy (MMD) as critic, termed MMD GANs. As our main theoretical contribution, we clarify the situation with bias in GAN loss functions raised by recent work: we show that gradient estimators used in the optimization process for both MMD GANs and Wasserstein GANs are unbiased, but learning a discriminator based on samples leads to biased gradients for the generator parameters. We also discuss the issue of kernel choice for the MMD critic, and characterize the kernel corresponding to the energy distance used for the Cramér GAN critic. Being an integral probability metric, the MMD benefits from training strategies recently developed for Wasserstein GANs. In experiments, the MMDGANisable to employ a smaller critic network than the Wasserstein GAN, resulting in a simpler and faster-training algorithm with matching performance. We also propose an improved measure of GAN convergence, the Kernel Inception Distance , and show how to use it to dynamically adapt learning rates during GAN training.

## 1 INTRODUCTION

Generative Adversarial Networks (GANs; Goodfellow et al., 2014) provide a powerful method for general-purpose generative modeling of datasets. Given examples from some distribution, a GAN attempts to learn a generator function, which maps from some fixed noise distribution to samples that attempt to mimic a reference or target distribution. The generator is trained to trick a discriminator , or critic , which tries to distinguish between generated and target samples.

This alternative to standard maximum likelihood approaches for training generative models has brought about a rush of interest over the past several years. Likelihoods do not necessarily correspond well to sample quality (Theis et al., 2016), and GAN-type objectives focus much more on producing plausible samples, as illustrated particularly directly by Danihelka et al. (2017). This class of models has recently led to many impressive examples of image generation (e.g. Huang et al., 2017a;b; Jin et al., 2017; Zhu et al., 2017).

GANs are, however, notoriously tricky to train (Salimans et al., 2016). This might be understood in terms of the discriminator class. Goodfellow et al. (2014) showed that, when the discriminator is trained to optimality among a rich enough function class, the generator network attempts to minimize the Jensen-Shannon divergence between the generator and target distributions. This result has been extended to general f -divergences by Nowozin et al. (2016). According to Arjovsky &amp; Bottou (2017), however, it is likely that both the GAN and reference probability measures are supported on manifolds within a larger space, as occurs for the set of images in the space of possible pixel values. These manifolds might not intersect at all, or at best might intersect on sets of measure zero. In this case, the Jensen-Shannon divergence is constant, and the KL and reverse-KL divergences are infinite, meaning that they provide no useful gradient for the generator to follow. This helps to explain some of the instability of GAN training.

∗ These authors contributed equally.

The lack of sensitivity to distance, meaning that nearby but non-overlapping regions of high probability mass are not considered similar, is a long-recognized problem for KL divergence-based discrepancy measures (e.g. Gneiting &amp; Raftery, 2007, Section 4.2). It is natural to address this problem using Integral Probability Metrics (IPMs; Müller, 1997): these measure the distance between probability measures via the largest discrepancy in expectation over a class of 'well behaved' witness functions. Thus, IPMs are able to signal proximity in the probability mass of the generator and reference distributions. (Section 2 describes this framework in more detail.)

Arjovsky et al. (2017) proposed to use the Wasserstein distance between distributions as the discriminator, which is an integral probability metric constructed from the witness class of 1-Lipschitz functions. To implement the Wasserstein critic, Arjovsky et al. originally proposed weight clipping of the discriminator network, to enforce k -Lipschitz smoothness. Gulrajani et al. (2017) improved on this result by directly constraining the gradient of the discriminator network at points between the generator and reference samples. This new Wasserstein GAN implementation, called WGAN-GP, is more stable and easier to train.

A second integral probability metric used in GAN variants is the maximum mean discrepancy (MMD), for which the witness function class is a unit ball in a reproducing kernel Hilbert space (RKHS). Generative adversarial models based on minimizing the MMD were first considered by Li et al. (2015) and Dziugaite et al. (2015). These works optimized a generator to minimize the MMDwith a fixed kernel, either using a generic kernel on image pixels or by modeling autoencoder representations instead of images directly. Sutherland et al. (2017) instead minimized the statistical power of an MMD-based test with a fixed kernel. Such approaches struggle with complex natural images, where pixel distances are of little value, and fixed representations can easily be tricked, as in the adversarial examples of Szegedy et al. (2014).

Adversarial training of the MMD loss is thus an obvious choice to advance these methods. Here the kernel MMD is defined on the output of a convolutional network, which is trained adversarially. Recent notable work has made use of the IPM representation of the MMD to employ the same witness function regularization strategies as Arjovsky et al. (2017) and Gulrajani et al. (2017), effectively corresponding to an additional constraint on the MMD function class. Without such constraints, the convolutional features are unstable and difficult to train (Sutherland et al., 2017). Li et al. (2017b) essentially used the weight clipping strategy of Arjovsky et al., with additional constraints to encourage the kernel distribution embeddings to be injective. 1 In light of the observations by Gulrajani et al., however, we use a gradient constraint on the MMD witness function in the present work (see Sections 2.1 and 2.2). 2 Bellemare et al. (2017)'s method, the Cramér GAN, also used the gradient constraint strategy of Gulrajani et al. in their discriminator network. As we discuss in Section 2.3, the Cramér GAN discriminator is related to the energy distance, which is an instance of the MMD (Sejdinovic et al., 2013), and which can therefore use a gradient constraint on the witness function. Note, however, that there are important differences between the Cramér GAN critic and the energy distance, which make it more akin to the optimization of a scoring rule: we provide further details in Appendix A. Weight clipping and gradient constraints are not the only approaches possible: variance features (Mroueh et al., 2017) and constraints (Mroueh &amp; Sercu, 2017) can work, as can other optimization strategies (Berthelot et al., 2017; Li et al., 2017a).

Given that both the Wasserstein distance and the MMD are integral probability metrics, it is of interest to consider how they differ when used in GAN training. Bellemare et al. (2017) showed that optimizing the empirical Wasserstein distance can lead to biased gradients for the generator, and gave an explicit example where optimizing with these biased gradients leads the optimizer to incorrect parameter values, even in expectation. They then claim that the energy distance does not suffer from these problems. As our main theoretical contribution, we substantially clarify the bias situation in Section 3. First, we show (Theorem 1) that the natural maximum mean discrepancy estimator, including the estimator of energy distance, has unbiased gradients when used 'on top' of a fixed deep network representation. The generator gradients obtained from a trained representation, however, will be biased relative to the desired gradients of the optimal critic based on infinitely many samples. This situation is exactly analogous to WGANs: the generator's gradients with a fixed critic are unbiased, but gradients from a learned critic are biased with respect to the supremum over critics.

1 When distribution embeddings are injective, the critic is guaranteed to be able to distinguish any two distributions, given an infinite number of samples.

2 Li et al. also did this in a later revision of their paper, independent of this work.

MMD GANs, though, do have some advantages over Wasserstein GANs. Certainly we would not expect the MMD on its own to perform well on raw image data, since these data lie on a low dimensional manifold embedded in a higher dimensional pixel space. Once the images are mapped through appropriately trained convolutional layers, however, they can follow a much simpler distribution with broader support across the mapped domain: a phenomenon also observed in autoencoders (Bengio et al., 2013). In this setting, the MMD with characteristic kernels (Sriperumbudur et al., 2010) shows strong discriminative performance between distributions. To achieve comparable performance, a WGAN without the advantage of a kernel on the transformed space requires many more convolutional filters in the critic. In our experiments (Section 5), we find that MMD GANs achieve the same generator performance as WGAN-GPs with smaller discriminator networks, resulting in GANs with fewer parameters and computationally faster training. Thus, the MMD GAN discriminator can be understood as a hybrid model that plays to the strengths of both the initial convolutional mappings and the kernel layer that sits on top.

## 2 LOSSES AND WITNESS FUNCTIONS

We begin with a review of the MMD and relate it to the loss functions used by other GAN variants. Through its interpretation as an integral probability metric, we show that the gradient penalty of Gulrajani et al. (2017) applies to the MMD GAN.

## 2.1 MAXIMUM MEAN DISCREPANCY AND WITNESS FUNCTIONS

Weconsider a random variable X with probability measure P , which we associate with the generator, and a second random variable Y with probability measure Q , which we associate with the reference sample that we wish to learn. Our goal is to measure the distance from P to Q using samples drawn independently from each distribution.

The maximum mean discrepancy is a metric on probability measures (Gretton et al., 2012), which falls within the family of integral probability metrics (Müller, 1997); this family includes the Wasserstein and Kolmogorov metrics, but not for instance the KL or χ 2 divergences. Integral probability metrics make use of a class of witness functions to distinguish between P and Q , choosing the function with the largest discrepancy in expectation over P , Q ,

<!-- formula-not-decoded -->

The particular witness function class F determines the probability metric. 3 For example, the Wasserstein-1 metric is defined using the 1-Lipschitz functions, the total variation by functions with absolute value bounded by 1, and the Kolmogorov metric using the functions of bounded variation 1 . For more on this family of distances, see e.g. Sriperumbudur et al. (2009b).

In this work, our witness function class F will be the unit ball in a reproducing kernel Hilbert space H , with positive definite kernel k ( x, x ′ ) . The key aspect of a reproducing kernel Hilbert space is the reproducing property: for all f ∈ H , f ( x ) = 〈 f, k ( x, · ) 〉 H . We define the mean embedding of the probability measure P as the element µ P ∈ H such that E P f ( X ) = 〈 f, µ P 〉 H ; it is given by µ P = E X ∼ P k ( · , X ) . 4

The maximum mean discrepancy (MMD) is defined as the IPM (1) with F the unit ball in H ,

<!-- formula-not-decoded -->

The witness function f ∗ that attains the supremum has a straightforward expression (Gretton et al., 2012, Section 2.3),

<!-- formula-not-decoded -->

3 We assume throughout that if f ∈ F , we also have - f ∈ F , so that D F is symmetric.

4 This is well defined for Bochner-integrable kernels (Steinwart &amp; Christmann, 2008, Definition A.5.20), for which E P ‖ k ( x, · ) ‖ H &lt; ∞ for the class of probability measures P being considered. For bounded kernels, the condition always holds, but for unbounded kernels, additional conditions on the moments might apply.

Given samples X = { x i } i m =1 drawn i.i.d. from P , and Y = { y j } n j =1 drawn i.i.d. from Q , the empirical witness function is

<!-- formula-not-decoded -->

and an unbiased estimator of the squared MMD is (Gretton et al., 2012, Lemma 6)

̸

̸

<!-- formula-not-decoded -->

When the kernel is characteristic (Sriperumbudur et al., 2010; 2011), the embedding µ P is injective (i.e., associated uniquely with P ). Perhaps the best-known characteristic kernel is the exponentiated quadratic kernel, also known as the Gaussian RBF kernel,

<!-- formula-not-decoded -->

Both the kernel and its derivatives decay exponentially, however, causing significant problems in high dimensions, and especially when used in gradient-based representation learning. The rational quadratic kernel

<!-- formula-not-decoded -->

with α &gt; 0 corresponds to a scaled mixture of exponentiated quadratic kernels, with a Gamma( α, 1) prior on the inverse lengthscale (Rasmussen &amp; Williams, 2006, Section 4.2). This kernel will be the mainstay of our experiments, as its tail behaviour is much superior to that of the exponentiated quadratic kernel; it is also characteristic.

## 2.2 WITNESS FUNCTION AND GRADIENT PENALTIES

The MMD has been a popular choice for the role of a critic in a GAN. This idea was proposed simultaneously by Dziugaite et al. (2015) and Li et al. (2015), with numerous recent follow-up works (Sutherland et al., 2017; Liu, 2017; Li et al., 2017b; Bellemare et al., 2017). As a key strategy in these recent works, the MMD of (4) is not computed directly on the samples; rather, the samples first pass through a mapping function h , generally a convolutional network. Note that we can think of this either as the MMD with kernel k on features h ( x ) , or simply as the MMD with kernel κ ( x, y ) = k ( h ( x ) , h ( y )) . The challenge is to learn the features h so as to maximize the MMD, without causing the critic to collapse to a trivial answer early in training.

Bearing in mind that the MMD is an integral probability metric, strategies developed for training the Wasserstein GAN critic can be directly adopted for training the MMD critic. Li et al. (2017b) employed the weight clipping approach of Arjovsky et al. (2017), though they motivated it using different considerations. Gulrajani et al. (2017) found a number of issues with weight clipping, however: it oversimplifies the loss functions given standard architectures, the gradient decays exponentially as we move up the network, and it seems to require the use of slower optimizers such as RMSProp rather than standard approaches such as Adam (Kingma &amp; Ba, 2015).

It thus seems preferable to adopt Gulrajani et al.'s proposal of regularising the critic witness (3) by constraining its gradient norm to be nearly 1 along randomly chosen convex combinations of generator and reference points, αx i + (1 - α ) y j for α ∼ Uniform(0 , 1) . This was motivated by the observation that the Wasserstein witness satisfies this property (their Lemma 1), but perhaps its main benefit is one of regularization: if the critic function becomes too flat anywhere between the samples, the generator cannot easily follow its gradient. We will thus follow this approach, as did Bellemare et al. (2017), whose model we describe next. 5

5 By doing so, we implicitly change the definition of the distance being approximated; we leave study of the differences to future work. By analogy, Liu et al. (2017) give some basic properties for the distance used by Gulrajani et al. (2017).

## 2.3 THE ENERGY DISTANCE AND ASSOCIATED MMD

Liu (2017) and Bellemare et al. (2017, Section 4) proposed to use the energy distance as the critic in an adversarial network. The energy distance (Székely &amp; Rizzo, 2004; Lyons, 2013) is a measure of divergence between two probability measures, defined as

<!-- formula-not-decoded -->

where E P ρ ( X,X ′ ) is an expectation over two independent samples from the generator P (likewise, Y and Y ′ are independent samples from the reference Q ), and ρ ( x, y ) is a semimetric of negative type. 6 We will focus on the case ρ β ( x, y ) = ‖ x - y ‖ β for 0 &lt; β ≤ 2 . When β ≤ 1 , ρ β is a metric.

Sejdinovic et al. (2013, Lemma 12) showed that the energy distance is an instance of the maximum mean discrepancy, where the corresponding distance-induced kernel family for the distance (7) is

<!-- formula-not-decoded -->

for any choice of z 0 ∈ X . (Often z 0 = 0 is chosen to simplify notation, which corresponds to a fractional Brownian motion kernel; Sejdinovic et al., 2013, Example 15.) Note that the resulting kernel is not translation invariant. It is characteristic (when β &lt; 2 ), and the MMD is well-defined for a class of distributions P that satisfy a moment condition (Sejdinovic et al., 2013, Remark 21). 7

To apply the regularization strategy of Gulrajani et al. (2017) in training the critic of an adversarial network, we need to compute the form taken by the witness function (2) given the kernel (8). Bearing in mind that

<!-- formula-not-decoded -->

where the second term of the above expression is constant, and substituting into (2), we have

<!-- formula-not-decoded -->

This is in agreement with Bellemare et al.'s function f ∗ (their page 5), via a different argument (though note that the function g ∗ in their footnote 4 is missing the constant terms).

We now turn to the divergence implemented by the critic in Bellemare et al.'s Algorithm 1, which is somewhat different from the energy distance (7). The Cramér GAN witness function is defined as

<!-- formula-not-decoded -->

which is regularized using Gulrajani et al.'s gradient constraint. The expected surrogate loss associated with this witness function, and used for the Cramér critic, is

<!-- formula-not-decoded -->

̸

In brief, Y ′ in (7) is replaced by the origin: it is explained that this is necessary for instance in the conditional case, where two independent reference samples Y, Y ′ are not available. Unfortunately, following this change, it becomes straightforward to define P and Q which are different, yet have an expected D c ( P , Q ) loss of zero. For example, if P is a point mass at the origin in R , and Q is a point mass a distance t from the origin, then P = Q and yet D c ( P , Q ) = 0 because

<!-- formula-not-decoded -->

Nevertheless, good empirical performance has been obtained in practice for the Cramér critic, both by Bellemare et al. (2017) and in our experiments of Section 5. Our Appendix A provides some insight into this behavior by considering the Cramér critic's relationship to the score function associated with the energy distance.

6 ρ must satisfy the properties of a metric besides the triangle inequality, and for all n ≥ 2 , x 1 , . . . , x n ∈ X , and a 1 , . . . , a n ∈ R with ∑ i a i = 0 , it must hold that ∑ n i =1 ∑ n j =1 a i a j ρ ( x i , x j ) ≤ 0 .

7 Namely, there must exist z 0 ∈ X such that ∫ ρ ( z, z 0 ) d P ( z ) &lt; ∞ for all P ∈ P .

## 2.4 OTHER RELATED MODELS

Many other GAN variants fall into the framework of IPMs (e.g. Mroueh et al., 2017; Mroueh &amp; Sercu, 2017; Berthelot et al., 2017). Notably, although Goodfellow et al. (2014) motivated GANs as estimating the Jensen-Shannon divergence, they can also be viewed as minimizing the IPM defined by the classifier family (Arora et al., 2017; Liu et al., 2017), thus motivating applying the gradient penalty to original GANs (Fedus et al., 2018). Liu et al. (2017) in particular study properties of these distances.

## 3 GRADIENT BIAS

The issue of biased gradients in GANs was brought to prominence by Bellemare et al. (2017, Section 3), who showed bias in the gradients of the empirical Wasserstein distance for finite sample sizes, and demonstrated cases where this bias could lead to serious problems in stochastic gradient descent, even in expectation. They then claimed that the energy distance used in the Cramér GAN critic does not suffer from these problems. We will now both formalize and clarify these results.

First, Bellemare et al.'s proof that the gradient of the energy distance is unbiased was incomplete: the essential step in the reasoning, the exchange in the order of the expectations and the derivatives, is simply assumed. 8 We show that one can exchange the order of expectations and derivatives, under very mild assumptions about the distributions in question, the form of the network, and the kernel:

Theorem 1. Let G ψ : Z → X and h θ : X → R d be deep networks, with parameters ψ ∈ R m ψ and θ ∈ R m θ , of the form defined in Appendix C.1 and satisfying Assumptions C and D (in Appendix C.2). This includes almost all feedforward networks used in practice, in particular covering convolutions, max pooling, and ReLU activations.

Let P be a distribution on X such that E [ ‖ X ‖ 2 ] exists, and likewise Z a distribution on Z such that E [ ‖ Z ‖ 2 ] exists. P and Z need not have densities.

Let k : R d × R d → R be a kernel function satisfying the growth assumption Assumption E for some α ∈ [1 , 2] . All kernels considered in this paper satisfy this assumption; see the discussion after Corollary 3.

For µ -almost all ( ψ, θ ) ∈ R m ψ + m θ , where µ is the Lebesgue measure, the function

<!-- formula-not-decoded -->

is differentiable at ( ψ, θ ) , and moreover

<!-- formula-not-decoded -->

Thus for µ -almost all ( ψ, θ ) ,

<!-- formula-not-decoded -->

This result is shown in Appendix C, specifically as Corollary 3 to Theorem 5, which is a quite general result about interchanging expectations and derivatives of functions of deep networks. The proof is more complex than a typical proof that derivatives and integrals can be exchanged, due to the non-differentiability of ReLU-like functions used in deep networks.

But this unbiasedness result is not the whole story. In WGANs, the generator attempts to minimize the loss function

<!-- formula-not-decoded -->

based on an estimate ˆ W ( X , Y ) : first critic parameters θ are estimated on a 'training set' X tr , Y tr , i.e. all points seen in the optimization process thus far, and then the distance is estimated on the remaining 'test set' X te , Y te , i.e. the current minibatch, as

<!-- formula-not-decoded -->

8 Suppose ˆ f ( x ) is an unbiased estimator of a function f ( x ) , so that E ˆ f ( x ) = f ( x ) . Then, if we can exchange expectations and gradients, it is immediate that E ∇ ˆ f ( x ) = ∇ E ˆ f ( x ) = ∇ f ( x ) .

(After the first pass through the training set, these two sets will not be quite independent, but for large datasets they should be approximately so.) Theorem 2 (in Appendix B.2) shows that this estimator ˆ W is biased; Appendix B.4 further gives an explicit numerical example. This almost certainly implies that ∇ ψ ˆ W is biased as well, by Theorem 4 (Appendix B.3). 9 Yet, for fi xed θ , Corollary 1 shows that the estimator (13) has unbiased gradients; it is only the procedure which first selects a θ based on training samples and then evaluates (13) which is a biased estimator of (12).

The situation with MMD GANs, including energy distance-based GANs, is exactly analogous. We have (11): for almost all particular critic representations h θ , the estimator of MMD 2 is unbiased. But the population divergence the generator attempts to minimize is actually

<!-- formula-not-decoded -->

a distance previously studied by Sriperumbudur et al. (2009a) as well as Li et al. (2017b). An MMD GAN's effective estimator of ˆ η is also biased by Theorem 2 (see particularly Appendix B.5); by Theorem 4, its gradients are also almost certainly biased.

In both cases, the bias vanishes as the selection of θ becomes better; in particular, no bias is introduced by the use of a fixed (and potentially small) minibatch size, but rather by the optimization procedure for θ and the total number of samples seen in training the discriminator.

Yet there is at least some sense in which MMD GANs might be considered 'less biased' than WGANs. Optimizing the generator parameters of a WGAN while holding the critic parameters fixed is not sensible: consider, for example, P a point mass at 0 ∈ R and Q a point mass at q ∈ R . If q &gt; 0 , an optimal θ might correspond to the witness function f ( t ) = t ; if we hold this witness function f fi xed, the optimal q is at -∞ , rather than at the correct value of 0 . But if we hold an MMD GAN's critic fixed and optimize the generator, we obtain the GMMN model (Li et al., 2015; Dziugaite et al., 2015). Here, because the witness function still adapts to the observed pair of distributions, the correct distribution P = Q will always be optimal. Bad solutions might also seem to be optimal, but they can never seem arbitrarily better . Thus unbiased gradients of MMD 2 u might somehow be more meaningful to the optimization process than unbiased gradients of (13); exploring and formalizing this intuition is an intriguing area for future work.

## 4 EVALUATION METRICS

One challenge in comparing GAN models, as we will do in the next section, is that quantitative comparisons are difficult. Some insight can be gained by visually examining samples, but we also consider the following approaches to evaluate GAN methods.

Inception score This metric, proposed by Salimans et al. (2016), is based on the classification output p ( y | x ) of the Inception model (Szegedy et al., 2016). Defined as exp( E x KL( p ( y | x ) ‖ p ( y ))) , it is highest when each image's predictive distribution has low entropy, but the marginal predictive distribution p ( y ) = E x p ( y | x ) has high entropy. This score correlates somewhat with human judgement of sample quality on natural images, but it has some issues, especially when applied to domains which do not represent a variety of the types of classes in ImageNet. In particular, it knows nothing about the desired distribution for the model.

FID The Fréchet Inception Distance , proposed by Heusel et al. (2017), avoids some of the problems of Inception by measuring the similarity of the samples' representations in the Inception architecture (at the pool3 layer, of dimension 2048 ) to those of samples from the target distribution. The FID fits a Gaussian distribution to the hidden activations for each distribution and then computes the Fréchet distance, also known as the Wasserstein-2 distance, between those Gaussians. Heusel et al. show that unlike the Inception score, the FID worsens monotonically as various types of artifacts are added to CelebA images - though in our Appendix E we found the Inception score to be more mono-

9 Bellemare et al.'s Appendix A.2 rather showed gradient bias in a different situation: ∇W ( ˆ P m , Q ) , where Q is a known distribution and ˆ P m is the empirical distribution of m samples from a distribution which changes as m increases.

(a) KID estimates are unbiased, and standard deviations shrink quickly even for small n .

<!-- image -->

<!-- image -->

n

(b) FID estimates exhibit strong bias for n even up to 10 000 . All standard deviations are less than 0 . 5 .

Figure 1: Estimates of distances between the CIFAR-10 train and test sets. Each point is based on 100 samples, estimating with replacement; sampling without replacement, and/or using the full training set, gives similar results. Lines show means, error bars standard deviations, dark colored regions a 2 3 coverage interval of the samples, light colored regions a 95% interval. Note the differing n axes.

tonic than did Heusel et al., so this property may not be very robust to small changes in evaluation methods. Note also that the estimator of FID is biased; 10 we will discuss this issue shortly.

KID We propose a metric similar to the FID, the Kernel Inception Distance , to be the squared MMD between Inception representations. We use a polynomial kernel, k ( x, y ) = ( 1 d x T y +1 ) 3 where d is the representation dimension, to avoid correlations with the objective of MMD GANs as well as to avoid tuning any kernel parameters. 11 This can also be viewed as an MMD directly on input images with the kernel K ( x, y ) = k ( φ ( x ) , φ ( y )) , with φ the function mapping images to Inception representations. Compared to the FID, the KID has several advantages. First, it does not assume a parametric form for the distribution of activations. This is particularly sensible since the representations have ReLU activations, and therefore are not only never negative, but do not even have a density: about 2% of components in Inception representations are typically exactly zero. With the cubic kernel we use here, the KID compares skewness as well as the mean and variance. Also, unlike the FID, the KID has a simple unbiased estimator. 12 It also shares the behavior of the FID as artifacts are added to images (Appendix E).

Figure 1 demonstrates the empirical bias of the FID and the unbiasedness of the KID by comparing the CIFAR-10 train and test sets. The KID (Figure 1a) converges quickly to its presumed true value of 0; even for very small n , simple Monte Carlo estimates of the variance provide a reasonable measure of uncertainty. By contrast, the FID estimate (Figure 1b) does not behave so nicely: at n = 2000 , when the KID estimator is essentially always 0, the FID estimator is still quite large. Even at n = 10000 , the full size of the CIFAR test set, the FID still seems to be decreasing from its estimate of about 8.1 towards zero, showing the strong persistence of bias. This highlights that FID scores can only be compared to one another with the same value of n .

Yet even for the same value of n , there is no particular reason to think that the bias in the FID estimator will be the same when comparing different pairs of distributions. In Appendix D, we demonstrate two situations where FID ( P 1 , Q ) &lt; FID ( P 2 , Q ) , but for insufficent numbers of samples the estimator usually gives the other ordering. This can happen even where all distributions in question are one-dimensional Gaussians, as Appendix D.1 shows analytically. Appendix D.2 also empirically demonstrates this on distributions more like the ones used for FID in practice, giving a simple example with d = 2048 where even estimating with n = 50000 samples reliably gives the wrong ordering between the models. Moreover, Monte Carlo estimates of the variance are extremely small even when the estimate is very far from its asymptote, so it is difficult to judge the reliability of an estimate, and practitioners may be misled by the very low variance into thinking that they have obtained the true value. Thus comparing FID estimates bears substantial risks. KID estimates, by contrast, are unbiased and asymptotically normal.

10 This is easily seen when the true FID is 0 : here the estimator may be positive, but can never be negative. Note also that in fact no unbiased estimator of the FID exists; see Appendix D.3.

11 k is the default polynomial kernel in scikit-learn (Pedregosa et al., 2011).

12 Because the computation of the MMD estimator scales like O ( n 2 d ) , we recommend using a relatively small n and averaging over several estimates; this is closely related to the block estimator of Zaremba et al. (2013). The FID estimator, for comparison, takes time O ( nd 2 + d 3 ) , and is substantially slower for d = 2048 .

For models on MNIST, we replace the Inception featurization with features from a LeNet-like convolutional classifier 13 (LeCun et al., 1998), but otherwise compute the scores in the same way.

We also considered the diagnostic test of Arora &amp; Zhang (2017), which estimates the approximate number of 'distinct' images produced by a GAN. The amount of subjectivity in what constitutes a duplicate image, however, makes it hard to reliably compare models based on this diagnostic. Comparisons likely need to be performed both with a certain notion of duplication in mind and by a user who does not know which models are being compared, to avoid subconscious biases; we leave further exploration of this intriguing procedure to future work.

## 4.1 LEARNING RATE ADAPTATION

In supervised deep learning, it is common practice to dynamically reduce the learning rate of an optimizer when it has stopped improving the metric on a validation set. So far, this does not seem to be common in GAN-type models, so that learning rate schedules must be tuned by hand. We propose instead using an adaptive scheme, based on comparing the KID score for samples from a previous iteration to that from the current iteration.

To avoid setting an explicit threshold on the change in the numerical value of the score, we use a p -value obtained from the relative similarity test of Bounliphone et al. (2016). If the test does not indicate that our current model is closer to the validation set than the model from a certain number of iterations ago at a given significance level, we mark it as a failure; when a given number of failures occur in a row, we decrease the learning rate. Bounliphone et al.'s test is for the hypothesis MMD ( P 1 , Q ) &lt; MMD ( P 2 , Q ) , and since the KID can be viewed as an MMD on image inputs, we can apply it directly. 14

## 5 EXPERIMENTS

We compare the quality of samples generated by MMD GAN using various kernels with samples obtained by WGAN-GP (Gulrajani et al., 2017) and Cramér GAN (Bellemare et al., 2017) on four standard benchmark datasets: the MNIST dataset of 28 × 28 handwritten digits 15 , the CIFAR-10 dataset of 32 × 32 photos (Krizhevsky, 2009), the LSUN dataset of bedroom pictures resized to 64 × 64 (Yu et al., 2015), and the CelebA dataset of celebrity face images resized and cropped to 160 × 160 (Liu et al., 2015).

For most experiments, except for those with the CelebA dataset, we used the DCGAN architecture (Radford et al., 2016) for both generator and critic. For MMD losses, we used only 16 top-layer neurons in the critic; more did not seem to improve performance, except for the distance kernel for which 256 neurons in the top layer was advantageous. As Bellemare et al. (2017) advised to use at least 256-dimensional critic output, this enabled exact comparison between Cramér GAN and energy distance MMD, which are directly related (Section 2.3). For the generator we used the standard number of convolutional filters (64 in the second-to-last layer); for the critic, we compared networks with 16 and 64 filters in the first convolutional layer. 16

[13 github.com/tensorflow/models/blob/master/tutorials/image/mnist/convolutional.py](https://github.com/tensorflow/models/blob/master/tutorials/image/mnist/convolutional.py)

14 We use the slight corrections to the asymptotic distribution of the MMD estimator given by Sutherland et al. (2017) in this test.

[15 yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)

16 In the DCGAN architecture the number of filers doubles in each consecutive layer, so an f -filter critic has f , 2 f , 4 f and 8 f convolutional filters in layers 1-4, respectively.

For the higher-resolution model for the CelebA dataset, we used a 5-layer DCGAN critic and a 10-layer ResNet generator 17 , with 64 convolutional filters in the last/first layer. This allows us to compare the performance of MMD GANs with a more complex architecture.

Models with smaller critics run considerably faster: on our systems, the 16-filter DCGAN networks typically ran at about twice the speed of the 64-filter ones. Note that the critic size is far more important to training runtime than the generator size: we update the critic 5 times for each generator step, and moreover the critic network is run on two batches each time we use it, one from P and one from Q . Given the same architecture, all models considered here run at about the same speed.

We evaluate several MMD GAN kernel functions in our experiments. 18 The simplest is the linear kernel: k dot ( x, y ) = 〈 x, y 〉 , whose MMD corresponds to the distance between means (this is somewhat similar to the feature matching idea of Salimans et al., 2016). We also use the exponentiated quadratic (5) and rational quadratic (6) functions, with mixtures of lengthscales,

<!-- formula-not-decoded -->

where Σ = { 2 , 5 , 10 , 20 , 40 , 80 } , A = { . 2 , . 5 , 1 , 2 , 5 } . For the latter, however, we found it advantageous to add a linear kernel to the mixture, resulting in the mixed RQ-dot kernel k rq ∗ = k rq + k dot . Lastly we use the distance-induced kernel k dist ρ 1 , 0 of (8), using the Euclidean distance ρ 1 so that the MMD is the energy distance. 19 We also considered Cramér GANs, with the surrogate critic (10), and WGAN-GPs.

Each model was trained with a batch size of 64, and 5 discriminator updates per generator update. For CIFAR-10, LSUN and CelebA we trained for 150 000 generator updates, while for MNIST we used 50 000 . The initial learning rate was set to 10 - 4 and followed the adaptive scheme described in Section 4.1, with KID compared between the current model and the model 20 000 generator steps earlier ( 5 000 for MNIST), every 2 000 steps ( 500 for MNIST). After 3 consecutive failures to improve, the learning rate was halved. This approach allowed us to avoid manually picking a different learning rate for each of the considered models.

We scaled the gradient penalty by 1 , instead of the 10 recommended by Gulrajani et al. (2017) and Bellemare et al. (2017); we found this to usually work slightly better with MMD models. With the distance kernel, however, we scale the penalty by 10 to allow direct comparison with Cramér GAN.

Quantitative scores are estimated based on 25 000 generator samples ( 100 000 for MNIST), and compared to 25 000 dataset elements (for LSUN and CelebA) or the standard test set ( 10 000 images held out from training for MNIST and CIFAR-10). Inception and FID scores were computed using 10 bootstrap resamplings of the given images; the KID score was estimated based on 100 repetitions of sampling 1 000 elements without replacement.

[Code for our models is available at github.com/mbinkowski/MMD-GAN .](https://github.com/mbinkowski/MMD-GAN)

MNIST All of the models achieved good results, measured both visually and in quantitative scores; full results are in Appendix F. Figure 2, however, shows the evolution of our quantitative criteria throughout the training process for several models. This shows that the linear kernel dot and rbf kernel rbf are clearly worse than the other models at the beginning of the training process, but both improve eventually. rbf , however, never fully catches up with the other models. There is also some evidence that dist , and perhaps WGAN-GP, converge more slowly than rq and Cramér GAN. Given their otherwise similar properties, we thus recommend the use of rq kernels over rbf in MMD GANs and limit experiments for other datasets to rq and dist kernels.

CIFAR-10 Full results are shown in Appendix F. Small-critic MMD GAN models approximately match large-critic WGAN-GP models, at substantially reduced computational cost.

17 As in Gulrajani et al. (2017), we use a linear layer, 4 residual blocks and one convolutional layer.

18 Because these higher-resolution experiments were slower to run, for CelebA we trained MMD GAN with only one type of kernel.

19 We also found it helpful to add an activation penalty to the critic representation network in certain MMD models. Otherwise the representations h θ sometimes chose very large values, which for most kernels does not change the theoretical loss (defined only in terms of distances) but leads to floating-point precision issues. We use a combined L 2 penalty on activations across all critic layers, with a factor of 1 for rq ∗ and 0 . 0001 for dist .

Figure 2: Score estimates over the learning process for MNIST training.

<!-- image -->

LSUN Bedrooms Table 1 presents scores for models trained on the LSUN Bedrooms dataset; samples from most of these models are shown in Figure 3. Comparing the models' Inception scores with the one achieved by the test set makes clear that this measure is not meaningful for this dataset - not surprisingly, given the drastic difference in domain from ImageNet class labels.

In terms of KID and FID, MMD GANs outperform Cramér and WGAN-GP for each critic size. Although results with the smaller critic are worse than with the large one for each considered model, small-critic MMD GANs still produce reasonably good samples, which certainly is not the case for WGAN-GP. Although a small-critic Cramér GAN produces relatively good samples, the separate objects in these pictures often seem less sharp than the MMD rq * samples. With a large critic, both Cramér GAN and MMD rq * give good quality samples, many of which are hardly distinguishable from the test set by eye.

Table 1: Mean (standard deviation) of score evaluations for the LSUN models. Inception scores do not seem meaningful for this dataset.

|            | critic size   | critic size   |             |               |               |
|------------|---------------|---------------|-------------|---------------|---------------|
| loss       | filters       | top layer     | Inception   | FID           | KID           |
| rq         | 16            | 16            | 3.13 (0.01) | 86.47 (0.29)  | 0.091 (0.002) |
| rq         | 64            | 16            | 2.80 (0.01) | 31.95 (0.28)  | 0.028 (0.002) |
| dist       | 16            | 256           | 3.42 (0.01) | 104.85 (0.32) | 0.109 (0.002) |
| dist       | 64            | 256           | 2.79 (0.01) | 35.28 (0.21)  | 0.032 (0.001) |
| Cramér GAN | 16            | 256           | 3.46 (0.02) | 122.03 (0.41) | 0.132 (0.002) |
| Cramér GAN | 64            | 256           | 3.44 (0.02) | 54.18 (0.39)  | 0.050 (0.002) |
| WGAN-GP    | 16            | 1             | 2.40 (0.01) | 292.77 (0.35) | 0.370 (0.003) |
| WGAN-GP    | 64            | 1             | 3.12 (0.01) | 41.39 (0.25)  | 0.039 (0.002) |
| test set   | -             | -             | 2.36 (0.01) | 2.49 (0.02)   | 0.000 (0.000) |

CelebA Scores for the CelebA dataset are shown in Table 2; MMD GAN with rq* kernel outperforms both WGAN-GP and Cramér GAN in KID and FID. Samples in Figure 4 show that for each of the models there are many visually pleasing pictures among the generated ones, yet unrealistic images are more common for WGAN-GP and Cramér.

Table 2: Mean (standard deviation) of score evaluations for the CelebA dataset.

|          | critic size   | critic size   |             |              |               |
|----------|---------------|---------------|-------------|--------------|---------------|
| loss     | filters       | top layer     | Inception   | FID          | KID           |
| rq       | 64            | 16            | 2.61 (0.01) | 20.55 (0.25) | 0.013 (0.001) |
| Cramér   | 64            | 256           | 2.86 (0.01) | 31.30 (0.17) | 0.025 (0.001) |
| WGAN-GP  | 64            | 1             | 2.72 (0.01) | 29.24 (0.22) | 0.022 (0.001) |
| test set | -             | -             | 3.76 (0.02) | 2.25 (0.04)  | 0.000 (0.000) |

These results illustrate the benefits of using the MMD on deep convolutional feaures as a GAN critic. In this hybrid system, the initial convolutional layers map the generator and reference image distributions to a simpler representation, which is well suited to comparison via the MMD. The

<!-- image -->

(d) MMD

rq

, critic size 64

(e) WGAN-GP, critic size 64

(f) Test set

Figure 3: Comparison of samples for the 64 × 64 LSUN Bedroom database.

<!-- image -->

(a) MMD

rq

*

<!-- image -->

(b) WGAN-GP

(c) Cramér GAN

<!-- image -->

Figure 4: Comparison of samples with a ResNet generator for the 160 × 160 CelebA dataset.

MMD in turn employs an infinite dimensional feature space to compare the outputs of these convolutional layers. By comparison, WGAN-GP requires a larger discriminator network to achieve similar performance. It is interesting to consider the question of kernel choice: the distance kernel and RQ kernel are both characteristic (Sriperumbudur et al., 2010), and neither suffers from the fast decay of the exponentiated quadratic kernel, yet the RQ kernel performs slightly better in our experiments. The relative merits of different kernel families for GAN training will be an interesting topic for further study.

## REFERENCES

M. Arjovsky and L. Bottou. Towards principled methods for training generative adversarial networks. In ICLR , 2017. arXiv:1701.04862.

M. Arjovsky, S. Chintala, and L. Bottou. Wasserstein generative adversarial networks. In ICML , 2017. arXiv:1701.07875.

- S. Arora and Y. Zhang. Do GANs actually learn the distribution? An empirical study, 2017. arXiv:1706.08224.
- S. Arora, R. Ge, Y. Liang, T. Ma, and Y. Zhang. Generalization and equilibrium in generative adversarial nets (GANs). In ICML , 2017. arXiv:1703.00573.
- M. G. Bellemare, I. Danihelka, W. Dabney, S. Mohamed, B. Lakshminarayanan, S. Hoyer, and R. Munos. The Cramer distance as a solution to biased Wasserstein gradients, 2017. arXiv:1705.10743.
- Y. Bengio, G. Mesnil, Y. Dauphin, and S. Rifai. Better mixing via deep representations. In ICML , 2013. arXiv:1207.4404.
- D. Berthelot, T. Schumm, and L. Metz. BEGAN: Boundary equilibrium generative adversarial networks, 2017. arXiv:1703.10717.
- P. J. Bickel and E. L. Lehmann. Unbiased estimation in convex families. The Annals of Mathematical Statistics , 40(5):1523-1535, 1969.
- D. Bouchacourt, P. K. Mudigonda, and S. Nowozin. DISCO nets: DISsimilarity COefficients networks. In NIPS , pp. 352-360. 2016.
- W. Bounliphone, E. Belilovsky, M. B. Blaschko, I. Antonoglou, and A. Gretton. A test of relative similarity for model selection in generative models. In ICLR , 2016. arXiv:1511.04581.
9. D.-A. Clevert, T. Unterthiner, and S. Hochreiter. Fast and accurate deep network learning by exponential linear units (ELUs). In ICLR , 2016. arXiv:1511.07289.
- I. Danihelka, B. Lakshminarayanan, B. Uria, D. Wierstra, and P. Dayan. Comparison of maximum likelihood and GAN-based training of Real NVPs, 2017. arXiv:1705.05263.
- G. K. Dziugaite, D. M. Roy, and Z. Ghahramani. Training generative neural networks via maximum mean discrepancy optimization. In UAI , 2015. arXiv:1505.03906.
- W. Fedus, M. Rosca, B. Lakshminarayanan, A. M. Dai, S. Mohamed, and I. Goodfellow. Many paths to equilibrium: GANs do not need to decrease a divergence at every step. In ICLR , 2018. arXiv:1710.08446.
- T. Gneiting and A. E. Raftery. Strictly proper scoring rules, prediction, and estimation. JASA , 102 (477):359-378, 2007.
- I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio. Generative adversarial nets. In NIPS , 2014. arXiv:1406.2661.
- A. Gretton, K. M. Borgwardt, M. J. Rasch, B. Schölkopf, and A. J. Smola. A kernel two-sample test. JMLR , 13, 2012.
- I. Gulrajani, F. Ahmed, M. Arjovsky, V. Dumoulin, and A. Courville. Improved training of Wasserstein GANs. In NIPS , 2017. arXiv:1704.00028.
- M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, G. Klambauer, and S. Hochreiter. GANs trained by a two time-scale update rule converge to a Nash equilibrium. In NIPS , 2017. arXiv:1706.08500.
- R. Huang, S. Zhang, T. Li, and R. He. Beyond face rotation: Global and local perception GAN for photorealistic and identity preserving frontal view synthesis. In ICCV , 2017a. arXiv:1704.04086.
- X. Huang, Y. Li, O. Poursaeed, J. Hopcroft, and S. Belongie. Stacked generative adversarial networks. In CVPR , 2017b. arXiv:1612.04357.
- Y. Jin, K. Zhang, M. Li, Y. Tian, H. Zhu, and Z. Fang. Towards the automatic anime characters creation with generative adversarial networks, 2017. arXiv:1708.05509.
- D. Kingma and J. Ba. Adam: A method for stochastic optimization. In ICLR , 2015. arXiv:1412.6980.

- A. Klenke. Probability Theory: A Comprehensive Course . World Publishing Corporation, 2008.
- A. Krizhevsky. Learning multiple layers of features from tiny images, 2009.
- Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE , 1998.
- C. Li, D. Alvarez-Melis, K. Xu, S. Jegelka, and S. Sra. Distributional adversarial networks, 2017a. arXiv:1706.09549.
5. C.-L. Li, W.-C. Chang, Y. Cheng, Y. Yang, and B. Póczos. MMD GAN: Towards deeper understanding of moment matching network. In NIPS , 2017b. arXiv:1705.08584.
- Y. Li, K. Swersky, and R. Zemel. Generative moment matching networks. In ICML , 2015. arXiv:1502.02761.
- L. Liu. On the two-sample statistic approach to generative adversarial networks. Master's thesis, University of Princeton Senior Thesis, April 2017. URL http://arks.princeton.edu/ ark:/88435/dsp0179408079v .
- S. Liu, O. Bousquet, and K. Chaudhuri. Approximation and convergence properties of generative adversarial learning. In NIPS , 2017. arXiv:1705.08991.
- Z. Liu, P. Luo, X. Wang, and X. Tang. Deep learning face attributes in the wild. In ICCV , 2015.
- D. Lopez-Paz and M. Oquab. Revisiting classifier two-sample tests. In ICLR , 2017. arXiv:1610.06545.
- R. Lyons. Distance covariance in metric spaces. The Annals of Probability , 41(5):3051-3696, 2013.
- B. Mityagin. The zero set of a real analytic function, 2015. arXiv:1512.07276.
- Y. Mroueh and T. Sercu. Fisher GAN. In NIPS , 2017. arXiv:1705.09675.
- Y. Mroueh, T. Sercu, and V. Goel. McGan: Mean and covariance feature matching GAN. In ICML , 2017. arXiv:1702.08398.
- A. Müller. Integral probability metrics and their generating classes of functions. Advances in Applied Probability , 29(2):429-443, 1997.
- S. Nowozin, B. Cseke, and R. Tomioka. f-GAN: Training generative neural samplers using variational divergence minimization. In NIPS , 2016. arXiv:1606.00709.
- F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay. Scikit-learn: Machine learning in Python. JMLR , 12:2825-2830, 2011.
- G. Piranian. The Set of Nondifferentiability of a Continuous Function. The American Mathematical Monthly , 73(4):57-61, 1966.
- A. Radford, L. Metz, and S. Chintala. Unsupervised representation learning with deep convolutional generative adversarial networks. In ICLR , 2016. arXiv:1511.06434.
- C. E. Rasmussen and C. K. I. Williams. Gaussian Processes for Machine Learning . MIT Press, Cambridge, MA, 2006.
- S. Rosenbaum. Moments of a truncated bivariate normal distribution. JRSS B , 23:405-408, 1961.
- T. Salimans, I. Goodfellow, W. Zaremba, V. Cheung, A. Radford, and X. Chen. Improved techniques for training GANs. In NIPS , 2016. arXiv:1606.03498.
- D. Sejdinovic, B. K. Sriperumbudur, A. Gretton, and K. Fukumizu. Equivalence of distance-based and RKHS-based statistics in hypothesis testing. The Annals of Stastistics , 41(5):2263-2291, 2013. arXiv:1207.6076.

- B. K. Sriperumbudur, K. Fukumizu, A. Gretton, G. R. G. Lanckriet, and B. Schölkopf. Kernel choice and classifiability for RKHS embeddings of probability distributions. In NIPS , 2009a.
- B. K. Sriperumbudur, K. Fukumizu, A. Gretton, B. Schölkopf, and G. R. G. Lanckriet. On integral probability metrics, phi-divergences and binary classification, 2009b. arXiv:0901.2698.
- B. K. Sriperumbudur, A. Gretton, K. Fukumizu, G. R. G. Lanckriet, and B. Schölkopf. Hilbert space embeddings and metrics on probability measures. JMLR , 11:1517-1561, 2010. arXiv:0907.5309.
- B. K. Sriperumbudur, K. Fukumizu, and G. R. G. Lanckriet. Universality, characteristic kernels and RKHS embedding of measures. JMLR , 12:2389-2410, 2011. arXiv:1003.0887.
- B. K. Sriperumbudur, K. Fukumizu, A. Gretton, B. Schölkopf, and G. R. G. Lanckriet. On the empirical estimation of integral probability metrics. Electronic Journal of Statistics , 6:15501599, 2012.
- I. Steinwart and A. Christmann. Support Vector Machines . Information Science and Statistics. Springer, 2008.
- D. J. Sutherland. What are the mean and variance of a 0-censored multivariate normal? Cross Validated answer, 2018. URL https://stats.stackexchange.com/q/326347 .
- D. J. Sutherland, H.-Y. Tung, H. Strathmann, S. De, A. Ramdas, A. Smola, and A. Gretton. Generative models and model criticism via optimized maximum mean discrepancy. In International Conference on Learning Representations , 2017. arXiv:1611.04488.
- C. Szegedy, W. Zaremba, I. Sutskever, J. Bruna, D. Erhan, I. Goodfellow, and R. Fergus. Intriguing properties of neural networks. In ICLR , 2014. arXiv:1312.6199.
- C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna. Rethinking the Inception architecture for computer vision. In CVPR , 2016. arXiv:1512.00567.
- G. Székely and M. Rizzo. Testing for equal distributions in high dimension. InterStat , 5, 2004.
- L. Theis, A. van den Oord, and M. Bethge. A note on the evaluation of generative models. In ICLR , 2016. arXiv:1511.01844.
- F. Yu, Y. Zhang, S. Song, A. Seff, and J. Xiao. LSUN: Construction of a large-scale image dataset using deep learning with humans in the loop, 2015. arXiv:1506.03365.
- Z. Zahorski. Sur l'ensemble des points de non-dérivabilité d'une fonction continue. Bulletin de la Société mathématique de France , 2:147-178, 1946.
- W. Zaremba, A. Gretton, and M. B. Blaschko. B-tests: Low variance kernel two-sample tests. In NIPS , 2013. arXiv:1307.1954.
16. J.-Y. Zhu, T. Park, P. Isola, and A. A. Efros. Unpaired image-to-image translation using cycleconsistent adversarial networks. In ICCV , 2017. arXiv:1703.10593.

## A SCORE FUNCTIONS, DIVERGENCES, AND THE CRAMÉR GAN

It is not immediately obvious how to interpret the surrogate loss (10). An insight comes from considering the score function associated with the energy distance, which we now briefly review (Gneiting &amp; Raftery, 2007). A scoring rule is a function S ( P , y ) , which is the loss incurred when a forecaster makes prediction P , and the event y is observed. The expected score is the expectation under Q of the score,

<!-- formula-not-decoded -->

If a score is proper , then the expected score obtained when P = Q is greater or equal than the expected score for P = Q ,

<!-- formula-not-decoded -->

̸

A strictly proper scoring rule shows an equality only when P and Q agree. We can define a divergence measure based on this score,

<!-- formula-not-decoded -->

Bearing in mind the definition of the divergence (15), it is easy to see (Gneiting &amp; Raftery, 2007, eq. 22) that the energy distance (7) arises from the score function

<!-- formula-not-decoded -->

The interpretation is straightforward: the score of a reference sample y is determined by comparing its average distance to a generator sample with the average distance among independent generator samples, E P ρ ( X,X ′ ) . If we take an expectation over Y ∼ Q , we recover the scoring rule optimized by the DISCO Nets algorithm (Bouchacourt et al., 2016, Section 3.3).

As discussed earlier, the Cramér GAN critic does not use the energy distance (7) directly on the samples, but first maps the samples through a function h , for instance a convolutional network; this should be chosen to maximize the discriminative performance of the critic. Writing this mapping as h , we break the energy distance down as D e ( P , Q ) = S ( Q , Q ) - S ( P , Q ) , where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and

When training the discriminator, the goal is to maximize the divergence by learning h , and so both (16) and (17) change: in other words, divergence maximization is not possible without two independent samples Y, Y ′ from the reference distribution Q .

An alternative objective in light of the score interpretation, however, is to simply optimize the average score (17). In other words, we would find features h that make the average distance from generator to reference samples much larger than the average distance between pairs of generator samples. We no longer control the term encoding the 'variability' due to Q , E Q ρ ( h ( Y ) , h ( Y ′ )) , which might therefore explode: for instance, h might cause h ( Y ) to disperse broadly, and far from the support of P , assuming sufficient flexibility to keep E P ρ ( h ( X ) , h ( X ′ )) under control. We can mitigate this by controlling the expected norm E Q ρ ( h ( Y ) , 0) , which has the advantage of only requiring a single sample to compute. For example, we could maximize

-

<!-- formula-not-decoded -->

This resembles the Cramér GAN critic (10), but the generator-to-generator distance is scaled differently, and there is an additional term: E P ρ ( h ( X ) , 0) is being maximized in (10), which is more difficult to interpret. An argument has been made (in personal communication with Bellemare et al.) that this last term is required if the function f c in (9) is to be a witness of an integral probability metric (1), although the asymmetry of this witness in P vs Q needs to be analyzed further.

## B BIAS OF GENERALIZED IPM ESTIMATORS

We will now show that all estimators of IPM-like distances and their gradients are biased. Appendix B.1 defines a slight generalization of IPMs, used to analyze MMD GANs in the same framework as WGANs, and a class of estimators that are a natural model for the estimator used in GAN models. Appendix B.2 both shows that not only are this form of estimators invariably biased in nontrivial cases, and moreover no unbiased estimator can possibly exist; Appendix B.3 then demonstrates that any estimator with non-constant bias yields a biased gradient estimator. Appendices B.4 and B.5 demonstrate specific examples of this bias for the Wasserstein and maximized-MMD distances.

## B.1 GENERALIZED IPMS AND DATA-SPLITTING ESTIMATORS

We will first define a slight generalization of IPMs: we will use this added generality to help analyze MMDGANsin Appendix B.5.

Definition 1 (Generalized IPM) . Let X be some domain, with M a class of probability measures on X . 20 Let F be some parameter set, and J : F × M × M → R an objective functional. The generalized IPM D : M×M→ R is then given by

<!-- formula-not-decoded -->

For example, if F is a class of functions f : X → R , and J is given by

<!-- formula-not-decoded -->

then we obtain integral probability metrics (1). Given samples X ∼ P m and Y ∼ Q n , let ˆ P denote the empirical distribution of X (an equal mixture of point masses at each X i ∈ X ), and similarly ˆ Q for Y . Then we have a simple estimator of (19) which is unbiased for fixed f :

<!-- formula-not-decoded -->

Definition 2 (Data-splitting estimator of a generalized IPM) . Consider the distance (18) , with objective J and parameter class F . Suppose we observe iid samples X ∼ P m , Y ∼ Q n , for any two distributions P , Q . A data-splitting estimator is a function ˆ D ( X , Y ) which first randomly splits the sample X into X tr , X te and Y into Y tr , Y te , somehow selects a critic function ˆ f X tr , Y tr ∈ F independently of X te , Y te , and then returns a result of the form

<!-- formula-not-decoded -->

where ˆ J ( f, X , Y ) is an estimator of J ( f, P , Q ) .

These estimators are defined by three components: the choice of relative sizes of the train-test split, the selection procedure for ˆ f X tr , Y tr , and the estimator ˆ J . The most obvious selection procedure is

<!-- formula-not-decoded -->

though of course one could use regularization or other techniques to select a different f ∈ F , and in practice one will use an approximate optimizer. Lopez-Paz &amp; Oquab (2017) used an estimator of exactly this form in a two-sample testing setting.

As noted in Section 3, this training/test split is a reasonable match for the GAN training process. As we optimize a WGAN-type model, we compute the loss (or its gradients) on a minibatch, while the current parameters of the critic are based only on data seen in previous iterations. We can view the current minibatch as X te , Y te , all previously-seen data as X tr , Y tr , and the current critic function as ˆ f X tr , Y tr . Thus, at least in the first pass over the training set, WGAN-type approaches exactly fit the data-splitting form of Definition 2; in later passes, the difference from this setup should be relatively small unless the model is substantially overfitting.

## B.2 ESTIMATOR BIAS

We first show, in Theorem 2, that data-splitting estimators are biased downwards. Although this provides substantial intuition about the situation in GANs, it leaves open the question of whether some other unbiased estimator might exist; Theorem 3 shows that this is not the case.

Theorem 2. Consider a data-splitting estimator (Definition 2) of the generalized IPM D (Definition 1) based on an unbiased estimator ˆ J of J : for any fixed f ∈ F ,

<!-- formula-not-decoded -->

Then either the selection procedure is almost surely perfect,

<!-- formula-not-decoded -->

or else the estimator has a downward bias:

<!-- formula-not-decoded -->

20 All results in this section could be trivially extended to support P and Q over different domains X and Y , if desired.

Proof. Since X tr , Y tr are independent of X te , Y te ,

<!-- formula-not-decoded -->

Define the suboptimality of ˆ f X tr , Y tr as

<!-- formula-not-decoded -->

so that E ˆ D ( X , Y ) = D ( P , Q ) - E [ ε ] . Note that ε ≥ 0 , since D ( P , Q ) = sup f ∈F J ( f, P , Q ) and so for any f ∈ F we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Theorem 2 makes clear that as ˆ f X tr , Y tr converges to its optimum, the bias of ˆ D should vanish (as in Bellemare et al., 2017, Theorem 3). Moreover, in the GAN setting the minibatch size only directly determines X te , Y te , which do not contribute to this bias; bias is due rather to the training procedure and the number of samples seen through the training process. As long as ˆ f X tr , Y tr is not optimal, however, the estimator will remain biased.

Many estimators of IPMs do not actually perform this data splitting procedure, instead estimating D ( P , Q ) with the distance between empirical distributions D ( ˆ P , ˆ Q ) . The standard biased estimator of the MMD (Gretton et al., 2012, Equation 5), the IPM estimators of Sriperumbudur et al. (2012), and the empirical Wasserstein estimator studied by Bellemare et al. (2017) are all of this form. These estimators, as well as any other conceivable estimator, are also biased:

̸

Theorem 3. Let P be a class of distributions such that { (1 - α ) P 0 + α P 1 : 0 ≤ α ≤ 1 } ⊆ P , where P 0 = P 1 are two fixed distributions. Let D be an IPM (1) . There does not exist any estimator of D which is unbiased on P .

Proof. We use a technique inspired by Bickel &amp; Lehmann (1969). Suppose there is an unbiased estimator ˆ D ( X , Y ) of D : for some finite m and n , if X = { X 1 , . . . , X m } ∼ P m , Y ∼ Q n , then E [ ˆ D ( X , Y )] = D ( P , Q ) .

Fix P 0 , P 1 , and Q ∈ P , and consider the function

<!-- formula-not-decoded -->

Thus R ( α ) is a polynomial in α of degree at most m .

But taking 1 1

<!-- formula-not-decoded -->

where (23) used our general assumption about IPMs that if f ∈ F , we also have - f ∈ F . But R ( α ) is not a polynomial with any finite degree. Thus no such unbiased estimator ˆ D exists. □

Note that the proof of Theorem 3 does not readily extend to generalized IPMs, and so does not tell us whether an unbiased estimator of the MMD GAN objective (14) can exist. Also, attempting to apply the same argument to squared IPMs would give the square of (24), which is a quadratic function in α . Thus tells us that although no unbiased estimator for a squared IPM can exist with only m = 1 sample point, one can exist for m ≥ 2 , as indeed (4) does for the squared MMD.

## B.3 GRADIENT ESTIMATOR BIAS

We will now show that biased estimators, except for estimators with a constant bias, must also have biased gradients.

Assume that, as in the GAN setting, Q is given by a generator network G ψ with parameter ψ and inputs Z ∼ Z , so that Y = G ψ ( Z ) ∼ Q ψ . The generalized IPM of (18) is now a function of ψ , which we will denote as

<!-- formula-not-decoded -->

Consider an estimator ˆ D ( ψ ) of D ( ψ ) . Theorem 4 shows that when ˆ D ( ψ ) and D ( ψ ) are differentiable, the gradient ∇ ψ ˆ D ( ψ ) is an unbiased estimator for ∇ ψ D ( ψ ) only if the bias of ˆ D ( ψ ) doesn't depend on ψ . This is exceedingly unlikely to happen for the biased estimator ˆ D ( W ) defined in Theorem 2, and indeed Theorem 3 shows cannot happen for any IPM estimator.

Theorem 4. Let D : Ψ → R be a function on a parameter space Ψ ⊆ R d , with a random estimator ˆ D : Ψ → R which is almost surely differentiable. Suppose that ˆ D has unbiased gradients:

<!-- formula-not-decoded -->

Then, for each connected component of Ψ ,

<!-- formula-not-decoded -->

where the constant can vary only across distinct connected components.

Proof. Let ψ 1 and ψ 2 be an arbitrary pair of parameter values in Ψ , connected by some smooth path r : [0 , 1] → Ψ with r (0) = ψ 1 , r (1) = ψ 2 . For example, if Ψ is convex, then paths of the form r ( t ) = tψ 1 +(1 - t ) ψ 2 are sufficient. Using Fubini's theorem and standard results about path integrals, we have that

<!-- formula-not-decoded -->

This implies that E [ ˆ D ( ψ )] = D ( ψ ) + const for all ψ in the same connected component of Ψ .

## B.4 WGANS

Theorems 2 and 3 hold for the original WGANs, whose critic functions are exactly L -Lipschitz, considering F as the set of L -Lipschitz functions so that D F is L times the Wasserstein distance. They also hold for either WGANs or WGAN-GPs with F the actual set of functions attainable by the critic architecture, so that D is the 'neural network distance' of Arora et al. (2017) or the 'adversarial divergence' of Liu et al. (2017).

It should be obvious that for nontrivial distributions P and Q and reasonable selection criteria for ˆ f X tr , Y tr , (21) does not hold, and thus (22) does (so that the estimate is biased downwards). Theorem 3 also shows this is the case on reasonable families of input distributions, and moreover that the bias is not constant, so that gradients are biased by Theorem 4.

Example For an explicit demonstration, consider the Wasserstein case, F the set of 1 -Lipschitz functions, with P = N (1 , 1) and Q = N (0 , 1) . Here D F ( P , Q ) = 1 ; the only critic functions f ∈ F which achieve this are f ( t ) = t + C for C ∈ R .

If we observe only one training pair X tr ∼ P and Y tr ∼ Q , when X tr &gt; Y tr , f 1 ( t ) = t is a maximizer of (20), leading to the expected estimate J ( f 1 , P , Q ) = 1 . But with probability Φ ( - 1 / √ 2 ) ≈ 0 . 24 it happens that X tr &lt; Y tr . In such cases, (20) could give e.g. f - 1 ( t ) = - t , giving the expected response J ( f - 1 , P , Q ) = - 1 ; the overall expected estimate of the estimator using this critic selection procedure is then E ˆ D F ( X , Y ) = ( 1 - Φ ( - 1 / √ 2 )) - Φ ( - 1 / √ 2 ) ≈ 0 . 52 .

The only way to achieve E ˆ D F ( X , Y ) = 1 would be a 'stubborn' selection procedure which chooses f 1 + C no matter the given inputs. This would have the correct output E ˆ D F ( X , Y ) = 1 for this ( P , Q ) pair. Applying this same procedure to P = N ( - 1 , 1) and Q = N (0 , 1) , however, would then give E ˆ D F ( X , Y ) = - 1 , when it should also be 1 .

## B.5 MAXIMAL MMD ESTIMATOR

Recall the distance η ( P , Q ) = sup θ MMD 2 ( h θ ( P ) , h θ ( Q )) defined by (14). MMD GANs can be viewed as estimating η according to the scheme of Theorem 2, with F the set of possible parameters θ , J ( θ, P , Q ) = MMD 2 ( h θ ( P ) , h θ ( Q )) , and ˆ J ( θ, X , Y ) = MMD 2 u ( h θ ( X ) , h θ ( Y )) . Clearly our optimization scheme for θ does not almost surely yield perfect answers, and so again we have

<!-- formula-not-decoded -->

As m tr , n tr → ∞ , as for Wasserstein it should be the case that ˆ η → η . This is shown for certain kernels, along with the rate of convergence, by Sriperumbudur et al. (2009a, Section 4).

It should also be clear that in nontrivial situations, this bias is not constant, and hence gradients are biased by Theorem 4.

Example For a particular demonstration, consider

<!-- formula-not-decoded -->

with h θ : R 2 → R given by h θ ( x ) = θ T x , ‖ θ ‖ = 1 , so that h θ chooses a one-dimensional projection of the two-dimensional data. Then use the linear kernel k dot , so that the MMD is simply the difference in means between projected features:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and

Clearly η ( P , Q ) = 1 , which is obtained by θ ∈ { ( - 1 , 0) , (1 , 0) } ; any other valid θ will yield a strictly smaller value of MMD 2 ( h θ ( P ) , h θ ( Q )) .

The MMD GAN estimator of η , if the optimum is achieved, uses

̸

̸

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where A X tr , Y tr ∈ R 2 × 2 . ˆ θ X tr , Y tr is then the normalized first eigenvector of A X tr , Y tr . For Gaussian X tr , Y tr with finite m tr , n tr , ˆ θ X tr , Y tr is a continuous random variable and so does not almost surely lie in { ( - 1 , 0) , (1 , 0) } ; thus by Theorem 2, E ˆ η ( X , Y ) &lt; η ( P , Q ) = 1 . (A numerical simulation for the former gives a value around 0 . 6 when m tr = n tr = 2 .)

## C PROOF OF UNBIASED GRADIENTS

We now proceed to prove Theorem 1 as a corollary to the Theorem 5, our main result about exchanging gradients and expectations of deep networks.

Exchanging the gradient and the expectation can often be guaranteed using a standard result in measure theory (see Proposition 1), as a corollary of the Dominated Convergence theorem (Proposition 2). This result, however, requires the property Proposition 1.(ii): for almost all inputs X , the mapping is differentiable on the entirety of a neighborhood around θ . This order of quantifiers is important: it allows the use of the mean value theorem to control the average rate of change of the function, and the result then follows from Proposition 2.

̸

For a neural network with the ReLU activation function, however, this assumption doesn't hold in general. For instance, if θ = ( θ 1 , θ 2 ) ∈ R 2 with θ 2 = 0 and X ∈ R , one can consider this very simple function: h θ ( X ) = max(0 , θ 1 + θ 2 X ) . For any fixed value of θ , the function h θ ( X ) is differentiable in θ for all X in R except for X θ = - θ 1 /θ 2 . However, if we consider a ball of possible θ values B ( θ, r ) , the function is not differentiable on the set {- θ ′ 1 /θ ′ 2 ∈ R | θ ′ ∈ B ( θ, r ) } , which can have positive measure for many possible distributions for X .

In Theorem 5, we provide a proof that derivatives and expectations can be exchanged for all parameter values outside of a 'bad set' Θ P , without relying on Proposition 1.(ii). This can be done using Lemma 1, which takes advantage of the particular structure of neural networks to control the average rate of change without using the mean value theorem. Dominated convergence (Proposition 2) can then be applied directly.

We also show in Proposition 3 that the set Θ P , of parameter values where Theorem 5 might not hold, has zero Lebesgue measure. This relies on the standard Fubini theorem (Klenke, 2008, Theorem 14.16) and Lemma 4, which ensures that the network θ ↦→ h θ ( X ) is differentiable for almost all parameter values θ when X is fixed. Although Lemma 4 might at first sight seem obvious, it requires some technical considerations in topology and differential geometry.

Proposition 1 (Differentiation Lemma (e.g. Klenke, 2008, Theorem 6.28)) . Let V be a nontrivial open set in R m and let P be a probability distribution on R d . Define a map h : R d × V ↦→ R n with the following properties:

- (i) For any θ ∈ V , E P [ ‖ h θ ( X ) ‖ ] &lt; ∞ .
- (ii) For P -almost all X ∈ R d , the map V → R n , θ ↦→ h θ ( X ) is differentiable.
- (iii) There exists a P -integrable function g : R d ↦→ R such that ‖ ∂ θ h θ ( X ) ‖ ≤ g ( X ) for all θ ∈ V .

Then, for any θ ∈ V , E P [ ‖ ∂ θ h θ ( X ) ‖ ] &lt; ∞ and the function θ ↦→ E P [ h θ ( X )] is differentiable with differential:

<!-- formula-not-decoded -->

Proposition 2 (Dominated Convergence Theorem (e.g. Klenke, 2008, Corollary 6.26)) . Let P be a probability distribution on R d and f a measurable function. Let ( f n ) n ∈ N be a sequence of of integrable functions such that for P -almost all X ∈ R d , f n ( X ) → f ( X ) as n goes to ∞ . Assume that there is a dominating function g : ‖ f n ( X ) ‖ ≤ g ( X ) for P -almost all X ∈ R d for all n ∈ N , and E P [ g ( X )] &lt; ∞ . Then f is P -integrable, and E P [ f n ( X )] → E P [ f ( X )] as n goes to ∞ .

## C.1 NETWORK DEFINITION

We would like to consider general feed-forward networks with a directed acyclic computation graph G . Here, G consists of L + 1 nodes, with a root node i = 0 and a leaf node i = L . We denote by π ( i ) the set of parent nodes of i . The nodes are sorted according to a topological order: if j is a parent node of i , then j &lt; i . Each node i for i &gt; 0 computes a function f i , which outputs a vector in R d i based on its input in R d π ( i ) , the concatenation of the outputs of each layer in π ( i ) . Here d π ( i ) = ∑ j ∈ π ( i ) d j , and d 0 = d &gt; 0 .

We define the feed-forward network that factorizes according the graph G and with functions f i recursively:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where h π ( i ) is the concatenation of the vectors h j for j ∈ π ( i ) . The functions f i can be of two types:

- Affine transform (Linear Module) :

<!-- formula-not-decoded -->

W i is a vector of dimension m i . The function g i : R m i → R d i × ( d π ( i ) +1) is a known linear operator on the weights W i , which can account for convolutions and similar linear operations. We will sometimes use ˜ Y to denote the augmented vector [ Y 1 ] , which accounts

for bias terms.

- Non-linear : These f i have no learnable weights. f i can potentially be non-differentiable, such as max pooling, ReLU, and so on. Some conditions on f i will be required (see Assumption D ); the usual functions used in practice satisfy these conditions.

Denote by C the set of nodes i such that f i is non-linear. θ is the concatenation of parameters of all linear modules: θ = ( W k ) k ∈ C c , where C c is the complement of C in [ L ] = { 1 , ..., L } . Call the total number of parameters m = ∑ i ∈ C c m i , so that θ ∈ R m . The feature vector of the network corresponds to the output of the last node L and will be denoted h θ := h L θ ∈ R d L . The subscript θ stands for the parameters of the network. We will sometimes use h θ ( X ) to denote explicit dependence on X , or omit it when X is fixed.

Also define a 'top-level function' to be applied to h θ , K : R d L → R . This function might simply be K ( U ) = U , as in Corollaries 1 and 2. But it also allows us to represent the kernel function of an MMDGANinCorollary 3: here we take X to be the two inputs to the kernel stacked together, apply the network to each of the two inputs with the same parameters in parallel, and then compute the kernel value between the two representations with K . K will have different smoothness assumptions than the preceding layers (Assumption B ).

## C.2 ASSUMPTIONS

We will need the following assumptions at various points, where α ≥ 1 :

- A (Moments) E P [ ‖ X ‖ α ] &lt; ∞ .
- B The function K is continuously differentiable, and satisfies the following growth conditions where C 0 and C 1 are constants:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- C (Lipschitz nonlinear layers) For each i ∈ C , f i is M -Lipschitz.
- D (Analytic pieces) For each nonlinear layer f i , i ∈ C , there are K i functions ( f k i ) k ∈ [ K i ] , each real analytic on R d π ( i ) , which agree with f i on the closure of a set D k i :

<!-- formula-not-decoded -->

These sets D k i are disjoint, and cover the whole input space: ⋃ K i k =1 ¯ D k i = R d π ( i ) . Moreover, each D k i is defined by S i,k real analytic functions G i,k,s : R d π ( i ) → R as

<!-- formula-not-decoded -->

<!-- image -->

(a) Domains of analyticity for the ReLU function in R 2 .

(b) Domains of analyticity for max-pooling in R 2 .

<!-- image -->

Figure 5: Illustrations of the domains of analyticity for two functions used in deep learning. Panel a represents the domains of analyticity for the ReLU function in R 2 . Each of the four domains D 1 , D 2 , D 3 and D 4 can be described by two inequalities of the form G k, 1 ( Y ) &gt; 0 and G k, 2 ( Y ) &gt; 0 , where k is the index of each domain. Here, G k, 1 ( Y ) = ± Y 1 and G k, 2 ( Y ) = ± Y 2 which are analytic functions. Panel b represents the domains of analyticity for the max-pooling function in R 2 . D 1 and D 2 are each one described by one inequality of the form G k, 1 ( Y ) &gt; 0 with G 1 , 1 ( Y ) = Y 1 - Y 2 and G 2 , 1 ( Y ) = Y 2 - Y 1 , which are both analytic functions.

Note that Assumption B is satisfied by the function K ( U ) = U , used in Corollaries 1 and 2, with α = 1 . It is also satisfied by the top-level functions of an MMD GAN with each of the kernels we consider in this work; see Corollary 3.

Assumptions C and D are satisfied by the vast majority of deep networks used in practice.

For example, if f i computes the ReLU activation function on two inputs, then we have K i = 4 , with each D k i corresponding to a quadrant of the real plane (see Figure 5a). These quadrants might each be defined by S i,k = 2 inequalities of the form G i,k, 1 ( Y ) &gt; 0 and G i,k, 2 ( Y ) &gt; 0 , where G i,k,s ( Y ) = ± Y 1 and G i,k,s ( Y ) = ± Y 2 are analytic. Moreover, on each of these domains f i coincides with an analytic function:

<!-- formula-not-decoded -->

Another example is when f i computes max-pooling on two inputs. In that case we have K i = 2 , and each domain D k i corresponds to a half plane (see Figure 5b ). Each domain is defined by one inequality G i,k, 1 ( Y ) &gt; 0 with G i, 1 , 1 ( Y ) = Y 1 - Y 2 and G i, 2 , 1 ( Y ) = Y 2 - Y 1 . Again, G i,k, 1 are analytic functions and f i coincides with an analytic function on each of the domains:

<!-- formula-not-decoded -->

When f i is analytic on the whole space, D i = R d π ( i ) , we can choose K i = 1 to get D 1 i = R d π ( i ) , which can be defined by a single function ( S i,k = 1 ) of G i, 1 , 1 ( Y ) = 1 . This case corresponds to most of the differentiable functions used in deep learning, such as the softmax, sigmoid, hyperbolic tangent, and batch normalization functions.

Other activation functions, such as the ELU (Clevert et al., 2016), are piecewise-analytic and also satisfy Assumptions C and D .

## C.3 MAIN RESULTS

We first state the main result, which implies Theorem 1 via Corollaries 1 to 3. The proof depends on various intermediate results which will be established afterwards.

Theorem 5. Under Assumptions A to D , for µ -almost all θ 0 ∈ R m the function θ ↦→ E P [ K ( h θ ( X ))] is differentiable at θ 0 , and

<!-- formula-not-decoded -->

where µ is the Lebesgue measure.

Proof. Let θ 0 be such that the function θ ↦→ h θ ( X ) is differentiable at θ 0 for P -almost all X . By Proposition 3, this is the case for µ -almost all θ 0 in R m .

Consider a sequence ( θ n ) n ∈ N that converges to θ 0 ; there is then an R &gt; 0 such that ‖ θ n - θ 0 ‖ &lt; R for all n ∈ N . Letting X be in R d , Lemma 2 gives that

<!-- formula-not-decoded -->

with E P [ F ( X )] &lt; ∞ . It also follows that:

<!-- formula-not-decoded -->

for P -almost all X ∈ R d . The sequence M n ( X ) defined by:

<!-- formula-not-decoded -->

converges point-wise to 0 and is bounded by the integrable function 2 F ( X ) . Therefore by the dominated convergence theorem (Proposition 2) it follows that

<!-- formula-not-decoded -->

Finally we define the sequence

<!-- formula-not-decoded -->

which is upper-bounded by E P [ M n ( X )] and therefore converges to 0 . By the sequential characterization of limits in Lemma 3, it follows that E P [ K ( h θ ( X ))] is differentiable at θ 0 , and its differential is given by E P [ ∂ θ K ( h θ ( X ))] .

These corollaries of Theorem 5 apply it to specific GAN architectures. Here we use the distribution Z to represent the noise distribution.

Corollary 1 (WGANs) . Let P and Z be two distributions, on X and Z respectively, each satisfying Assumption A for α = 1 . Let G ψ : Z → X be a generator network and D θ : X → R a critic network, each satisfying Assumptions C and D . Then, for µ -almost all ( θ, ψ ) , we have that

<!-- formula-not-decoded -->

Proof. By linearity, we only need the following two results:

<!-- formula-not-decoded -->

The first follows immediately from Theorem 5, using the function K ( U ) = U (which clearly satisfies Assumption B for α = 1 ). The latter does as well by considering that the augmented network h ( θ,ψ ) ( Z ) = D θ ( G ψ ( Z )) still satisifes the conditions of Theorem 5.

Corollary 2 (Original GANs) . Let P and Z be two distributions, on X and Z respectively, each satisfying Assumption A for α = 1 . Let G ψ : Z → X be a generator network, and D θ : X → R a discriminator network, each satisfying Assumptions C and D . Further assume that the output of D is almost surely bounded: there is some γ &gt; 0 such that for µ -almost all ( θ, ψ ) ,

<!-- formula-not-decoded -->

Then we have the following:

<!-- formula-not-decoded -->

Thus, by linearity, gradients of all the loss functions given in Goodfellow et al. (2014, Section 3) are unbiased.

Proof. The log function is real analytic and ( 1 /γ )-Lipschitz on ( γ, 1 - γ ) . The claim therefore follows from Theorem 5, using the networks log ◦ D θ , log ◦ [ x ↦→ (1 - x )] ◦ D θ ◦ G ψ , and log ◦ D θ ◦ G ψ with K ( U ) = U .

The following assumption about a kernel k implies Assumption B when used as a top-level function K :

E Suppose k is a kernel such that there are constants C 0 , C 1 where

<!-- formula-not-decoded -->

Corollary 3 (MMD GANs) . Let P and Z be two distributions, on X and Z respectively, each satisfying Assumption A for some α ≥ 1 . Let k be a kernel satisfying Assumption E . Let G ψ : Z → X be a generator network and D θ : X → R a critic representation network each satisfying Assumptions C and D . Then

<!-- formula-not-decoded -->

Proof. Consider the following augmented networks:

<!-- formula-not-decoded -->

h (1) has inputs distributed as P × Z , which satisfies Assumption A with the same α as P and Z , and h (1) satisfies Assumptions C and D . The same is true of h (2) and h (3) . Moreover, the function

<!-- formula-not-decoded -->

satisfies Assumption B . Thus Theorem 5 applies to each of h (1) , h (2) , and h (3) . Considering the form of MMD 2 u (4), the result follows by linearity and the fact that MMD 2 u is unbiased (Gretton et al., 2012, Lemma 6).

Each of the kernels considered in this paper satisfies Assumption E with α at most 2:

- k dot ( x, y ) = 〈 x, y 〉 works with α = 2 , C 0 = 1 , C 1 = 1 .
- k rq α ′ of (6) works with α = 2 , C 0 = 1 , C 1 = √ 2 .
- k rbf σ of (5) works with α = 2 , C 0 = 1 , C 1 = √ 2 σ - 2 .
- k dist ρ β , 0 of (8), using ρ β ( x, y ) = ‖ x - y ‖ β with 1 ≤ β ≤ 2 , works with α = β , C 0 = 3 , C 1 = 4 β .

Since the existence of a moment implies the existence of all lower-order moments by Jensen's inequality, this finalizes the proof of Theorem 1.

## C.4 BOUNDS ON NETWORK GROWTH

The following lemmas were used in the proof of Theorem 5. We start by stating a result on the growth and Lipschitz properties of the network.

Lemma 1. Under Assumption C , there exist continuous functions θ ↦→ b ( θ ) , a ( θ ) and ( θ, θ ′ ) ↦→ ( α ( θ, θ ′ ) , β ( θ, θ ′ )) such that:

<!-- formula-not-decoded -->

for all X in R d and all θ, θ ′ in R m .

Proof. Let X in R d and θ, θ ′ in R m . We proceed by recursion on the nodes of the network. For i = 0 the inequalities hold trivially b 0 = 0 , a 0 = 1 , β 0 = 0 and α 0 = 0 . Assume now that:

<!-- formula-not-decoded -->

where a π ( i ) ( θ ) , b π ( i ) ( θ ) , α π ( i ) ( θ, θ ′ ) and β π ( i ) ( θ, θ ′ ) are continuous functions. If i is a linear layer then:

<!-- formula-not-decoded -->

with a i ( θ ) = ‖ g i ‖‖ W i ‖ a π ( i ) ( θ ) and b i ( θ ) = ‖ g i ‖‖ W i ‖ b π ( i ) ( θ ) . Moreover, we have that:

<!-- formula-not-decoded -->

with:

<!-- formula-not-decoded -->

When i is not a linear layer, then by Assumption C f i is M -Lipschitz. Thus we can directly get the needed functions by recursion: α i = Mα π ( i ) , β i = Mβ π ( i ) , a i = Ma π ( i ) and b i = Mb π ( i ) .

Lemma 2. Let R be a positive constant and θ ∈ R m . Under Assumptions A to C , the following hold for all θ ′ ∈ B ( θ, R ) and all X in R d :

<!-- formula-not-decoded -->

with E P [ F ( X )] &lt; ∞ .

Proof. We will first prove the following inequality:

<!-- formula-not-decoded -->

for all U and V in R L .

Let t be in [0 , 1] and define the function f by

<!-- formula-not-decoded -->

Then f (0) = K ( V ) and f (1) = K ( U ) . Moreover, f is differentiable and its derivative is given by:

<!-- formula-not-decoded -->

Using Assumption B one has that:

<!-- formula-not-decoded -->

The conclusion follows using the mean value theorem. Now choosing U = h θ ′ ( X ) and V = h θ ( X ) one gets the following:

<!-- formula-not-decoded -->

Under Assumption C , it follows by Lemma 1 that:

<!-- formula-not-decoded -->

The functions a , b , α , β defined in Lemma 1 are continuous, and hence all bounded on the ball B ( θ, R ); choose D &gt; 0 to be a bound on all of these functions. It follows after some algebra that

<!-- formula-not-decoded -->

Set F ( X ) = C 1 ( D α ( R +1) α - 1 (1 + ‖ X ‖ ) α + D (1 + ‖ X ‖ )) . Since α ≥ 1 , t ↦→ (1 + t 1 /α ) α is concave on t ≥ 0 , and so we have that

<!-- formula-not-decoded -->

via Jensen's inequality and Assumption A . We also have E [1 + ‖ X ‖ ] &lt; ∞ by the same assumption. Thus F ( X ) is integrable.

Lemma 3. Let f : R m → R be a real valued function and g a vector in R m such that:

<!-- formula-not-decoded -->

̸

for all sequences ( θ n ) n ∈ N converging towards θ 0 with θ n = θ 0 . Then f is differentiable at θ 0 , and its differential is g .

Proof. Recall the definition of a differential: g is the differential of f at θ 0 if

<!-- formula-not-decoded -->

The result directly follows from the sequential characterization of limits.

## C.5 CRITICAL PARAMETERS HAVE ZERO MEASURE

The last result required for the proof of Theorem 5 is Proposition 3. We will first need some additional notation.

For a given node i , we will use the following sets of indices to denote 'paths' through the network's computational graph:

<!-- formula-not-decoded -->

Note that ∂i ⊆ ¬ i , and that ¬ i = ∂i ∪ ¬ π ( i ) .

If a ( i ) is the set of ancestors of node i , we define a backward trajectory starting from node i as an element q of the form:

<!-- formula-not-decoded -->

where k j are integers in [ K j ] . We call T ( i ) the set of such trajectories for node i .

For p ∈ P of the form p = ( i, k, s ) , the set of parameters for which we lie on the boundary of p is

<!-- formula-not-decoded -->

We also denote by ∂S p the boundary of the set S p . If Q is a subset of P , we use the following notation for convenience:

<!-- formula-not-decoded -->

For a given θ 0 ∈ R m , the set of input vectors X ∈ R d such that h θ 0 is not differentiable is

<!-- formula-not-decoded -->

Consider a random variable X in the input space R d , following the distribution P . For a given distribution P , we introduce the following set of "critical" parameters:

<!-- formula-not-decoded -->

This is the set of parameters θ where the network is not differentiable for a non-negligible set of datasets X .

Finally, for a given X ∈ R d , set of parameters for which the network is not differentiable is

<!-- formula-not-decoded -->

We are now ready to state and prove the remaining result.

Proposition 3. Under Assumption D , the set Θ P has 0 Lebesgue measure for any distribution P .

Proof. Consider the following two sets:

<!-- formula-not-decoded -->

By virtue of Theorem I in Zahorski (1946); Piranian (1966), it follows that the set of nondifferentiability of continuous functions is measurable. It is easy to see then, that D and Q are also measurable sets since the network is continuous. Note that we have the inclusion D ⊆ Q . We endow the two sets with the product measure ν := µ × P , where µ is the Lebesgue measure. Therefore ν ( D ) ≤ ν ( Q ) . On one hand, Fubini's theorem tells us:

<!-- formula-not-decoded -->

By Lemma 4, we have that µ (Θ X ) = 0 ; therefore ν ( Q ) = 0 and hence ν ( D ) = 0 . On the other hand, we use again Fubini's theorem for ν ( D ) to write:

<!-- formula-not-decoded -->

For all θ ∈ Θ P , we have P ( N ( θ )) &gt; 0 by definition. Thus ν ( D ) = 0 implies that µ (Θ P ) = 0 .

Lemma4. Under Assumption D , for any X in R d , the set Θ X has 0 Lebesgue measure: µ (Θ X ) = 0 .

Proof. We first show that Θ X ⊆ ∂S P , which was defined by (25).

Let θ 0 be in Θ X . By Assumption D , it follows that θ 0 ∈ S P . Assume for the sake of contradiction that θ 0 / ∈ ∂S P . Then applying Lemma 5 to the output layer, i = L , implies that there is a real analytic function f ( θ ) which agrees with h θ on all θ ∈ B ( θ 0 , η ) for some η &gt; 0 . Therefore the network is differentiable at θ 0 , contradicting the fact that θ 0 ∈ Θ X . Thus Θ X ⊆ ∂S P .

Lemma 6 then establishes that µ ( ∂S P ) = 0 , and hence µ (Θ X ) = 0 .

Lemma 5. Let i be a node in the graph. Under Assumption D , if θ ∈ R m \ ∂S ¬ i , then there exist η &gt; 0 and a trajectory q ∈ T ( i ) such that h i θ ′ = f q ( θ ′ ) for all θ ′ in the ball B ( θ, η ) . Here f q is the real analytic function on R m defined with the same structure as h θ , but replacing each nonlinear f j with the analytic function f k j j for ( j, k j ) ∈ q .

Proof. We proceed by recursion on the nodes of the network. If i = 0 , we trivially have h 0 θ = X , which is real analytic on R m . Assume the result for ¬ π ( i ) and let θ ∈ R m \ ∂S ¬ i . In particular θ ∈ R m \ ∂S ¬ π ( p ) . By the recursion assumption, we get:

<!-- formula-not-decoded -->

with f q real analytic in R m .

If θ / ∈ S ∂i , then there is some sufficiently small η ′ &gt; 0 such that B ( θ, η ′ ) does not intersect S ∂i . Therefore, by Assumption D , there is some k ∈ [ K i ] such that h i θ ′ = f k i ( h π ( i ) θ ′ ) for all θ ′ ∈ B ( θ, η ′ ) , where f k i is one of the real analytic functions defining f i . By (26) we then have

<!-- formula-not-decoded -->

Otherwise, θ ∈ S ∂i . Then, noting that by assumption θ / ∈ ∂S ∂i , it follows that for small enough η ′ &gt; 0 , we have B ( θ, η ′ ) ⊆ S ∂i . Denote by A the set of index triples p ∈ ∂i such that θ ∈ S p ; A is nonempty since θ ∈ S ∂i . Therefore θ ∈ ⋂ p ∈ A S p , and θ / ∈ ⋃ p ∈ A c S p . We will show that for η ′ small enough, B ( θ, η ′ ) ⊆ ⋂ p ∈ A S p . Assume for the sake of contradiction that there exists a sequence of (parameter, index-triple) pairs ( θ n , p n ) such that p n ∈ A c , θ n ∈ S p n , and θ n → θ . p n is drawn from a finite set and thus has a constant subsequence, so we can assume without loss of generality that p n = p 0 for some p 0 ∈ A c . Since S p 0 is a closed set by continuity of the network and G p 0 , it follows that θ ∈ S p 0 by taking the limit. This contradicts the fact that θ / ∈ ⋃ p ∈ A c S p . Hence, for η ′ small enough, B ( θ, η ′ ) ⊆ ⋂ p ∈ A S p . Again, by Assumption D there is a k ∈ [ K i ] satisfying (27).

By setting f q 0 = f k i ( f q ) with q 0 = (( i, k ) ⊕ q ) , where ⊕ denotes concatenation, it finally follows that h i θ ′ = f q 0 ( θ ′ ) for all θ ′ in B ( θ, min( η, η ′ )) , and f q 0 is the real analytic function on R m as described.

Lemma 6. Under Assumption D ,

<!-- formula-not-decoded -->

Proof. We will proceed by recursion. For i = 0 we trivially have ∂S ¬ 0 = ∅ , thus µ ( ∂S ¬ 0 ) = 0 . Thus assume that

<!-- formula-not-decoded -->

For s = ( p, q ) , the pair of an index triple p ∈ ∂i and a trajectory q ∈ T ( i ) , define the set

<!-- formula-not-decoded -->

where f q is the real analytic function defined in Lemma 5 which locally agrees with h π ( i ) θ .

We will now prove that for any θ in ∂S ∂i \ ∂S ¬ π ( i ) , there exists s ∈ ∂i × T ( i ) such that θ ∈ M s and µ ( M s ) = 0 . We proceed by contradiction.

Let θ ∈ ∂S ∂i \ ∂S ¬ π ( i ) ; then for small enough η &gt; 0 , B ( θ, η ) ⊆ R m \ ∂S ¬ π ( i ) . By Lemma 5, there is a trajectory q ∈ T ( i ) such that

<!-- formula-not-decoded -->

Moreover, since θ ∈ ∂S ∂i , there exists p ∈ ∂i such that G p ( h π ( i ) θ ) = 0 . This means that for s = ( p, q ) , we have θ ∈ M s . If µ ( M s ) &gt; 0 , then by Lemma 7 M s = R m , hence we would have B ( θ, η ) ⊆ M s . By (28) it would then follow that B ( θ, η ) ⊆ S ∂i . This contradicts the fact that θ is in ∂S ∂i , and hence µ ( M s ) = 0 .

We have shown that ∂S ∂i \ ∂S ¬ π ( i ) ⊆ ⋃ s ∈ A M s , where the sets M s have zero Lebesgue measure and A ⊆ P × ⋃ L j =0 T ( j ) is finite. This implies:

<!-- formula-not-decoded -->

Using the recursion assumption µ ( ∂S ¬ π ( i ) ) = 0 , one concludes that µ ( ∂S ¬ i ) = 0 . Hence for the last node L , recalling that ¬ L = P one gets µ ( ∂S P ) = 0 .

Lemma 7. Let θ ↦→ F ( θ ) : R m → R be a real analytic function on R m and define the set:

<!-- formula-not-decoded -->

Then either µ ( M ) = 0 or F is identically zero.

Proof. This result is shown e.g. as Proposition 0 of Mityagin (2015).

## D FID ESTIMATOR BIAS

We now further study the bias behavior of the FID estimator (Heusel et al., 2017) mentioned in Section 4.

We will refer to the Fréchet Inception Distance between two distributions, letting µ P denote the mean of a distribution P and Σ P its covariance matrix, as

<!-- formula-not-decoded -->

This is motivated because it coincides with the Fréchet (Wasserstein-2) distance between normal distributions. Although the Inception coding layers to which the FID is applied are not normally distributed, the FID remains a well-defined pseudometric between arbitrary distributions whose first two moments exist.

The usual estimator of the FID based on samples { X i } i m =1 ∼ P m and { Y j } n j =1 ∼ P n is the plug-in estimator. First, estimate the mean and covariance with the standard estimators:

<!-- formula-not-decoded -->

Letting ˆ P X be a distribution matching these moments, e.g. N ( ˆ µ X , ˆ Σ X ) , the estimator is given by

<!-- formula-not-decoded -->

In Appendices D.1 and D.2, we exhibit two examples where FID ( P 1 , Q ) &lt; FID ( P 2 , Q ) , but the estimator FID ( ˆ P 1 , Q ) is usually greater than FID ( ˆ P 2 , Q ) with an equal number of samples m from P 1 and P 2 , for a reasonable number of samples. (As m →∞ , of course, the estimator is consistent, and so the order will eventually be correct.) We assume here an infinite number of samples n from Q for simplicity; this reversal of ordering is even easier to obtain when n = m . It is also trivial to achieve when the number of samples from P 1 and P 2 differ, as demonstrated by Figure 1b.

Note that Appendices D.1 and D.2 only apply to this plug-in estimator of the FID; it remains conceivable that there would be some other estimator for the FID which is unbiased. Appendix D.3 shows that this is not the case: there is no unbiased estimator of the FID.

## D.1 ANALYTIC EXAMPLE WITH ONE-DIMENSIONAL NORMALS

We will first show that the estimator can behave poorly even with very simple distributions.

When P = N ( µ P , Σ P ) and Q = N ( µ Q , Σ Q ) , it is well-known that

<!-- formula-not-decoded -->

where W is the Wishart distribution. Then we have

<!-- formula-not-decoded -->

The remaining term E Tr ( ( ˆ Σ X ˆ Σ Y ) 1 2 ) is more difficult to evaluate, because we must consider the correlations across dimensions of the two estimators. But if the distributions in question are one-dimensional, denoting Σ P = σ 2 P and ˆ Σ X = ˆ σ 2 X , the matrix square root becomes simple:

<!-- formula-not-decoded -->

Since √ m - 1 σ P ˆ σ X ∼ χ m - 1 , we get that

<!-- formula-not-decoded -->

Thus the expected estimator for one-dimensional normals becomes

<!-- formula-not-decoded -->

Now, consider the particular case

<!-- formula-not-decoded -->

Clearly

<!-- formula-not-decoded -->

But letting n →∞ in (29) gives

<!-- formula-not-decoded -->

where the inequality follows because 1 m 2 &lt; 1 m and d m &lt; 1 for all m ≥ 2 . Thus we have the undesirable situation

<!-- formula-not-decoded -->

## D.2 EMPIRICAL EXAMPLE WITH HIGH-DIMENSIONAL CENSORED NORMALS

The example of Appendix D.1, though indicative in that the estimator can behave poorly even with very simple distributions, is somewhat removed from the situations in which we actually apply the FID. Thus we now empirically consider a more realistic setup.

First, as noted previously, the hidden codes of an Inception coding network are not well-modeled by a normal distribution. They are, however, reasonably good fits to a censored normal distribution ReLU( X ) , where X ∼ N ( µ, Σ) and ReLU( X ) i = max(0 , X i ) . Using results of Rosenbaum (1961), it is straightforward to derive the mean and variance of ReLU( X ) (Sutherland, 2018), and hence to find the population value of FID (ReLU( X ) , ReLU( Y )) .

Let d = 2048 , matching the Inception coding layer, and consider

<!-- formula-not-decoded -->

where Σ = 4 d CC T , with C a d × d matrix whose entries are chosen iid standard normal. For one particular random draw of C , we found that FID ( P 1 , Q ) ≈ 1123 . 0 &gt; 1114 . 8 ≈ FID ( P 2 , Q ) . Yet with m = 50000 samples, FID ( ˆ P 1 , Q ) ≈ 1133 . 7 (sd 0 . 2 ) &lt; 1136 . 2 (sd 0 . 5 ) ≈ FID ( ˆ P 2 , Q ) . The variance in each estimate was small enough that of 100 evaluations, the largest FID ( ˆ P 1 , Q ) estimate was less than the smallest FID ( ˆ P 2 , Q ) estimate. At m = 100000 samples, however, the ordering of the estimates was correct in each of 100 trials, with FID ( ˆ P 1 , Q ) ≈ 1128 . 0 (sd 0 . 1 ) and FID ( ˆ P 2 , Q ) ≈ 1126 . 4 (sd 0 . 4 ). This behavior was similar for other random draws of C .

This example thus gives a case where, for the dimension and sample sizes at which we actually apply the FID and for somewhat-realistic distributions, comparing two models based on their FID estimates will not only not reliably give the right ordering - with relatively close true values and high dimensions, this is not too surprising - but, more distressingly, will reliably give the wrong answer , with misleadingly small variance. This emphasizes that unbiased estimators, like the natural KID estimator, are important for model comparison.

## D.3 NON-EXISTENCE OF AN UNBIASED ESTIMATOR

We can also show, using the reasoning of Bickel &amp; Lehmann (1969) that we also employed in Theorem 3, that there is no estimator of the FID which is unbiased for all distributions.

̸

Fix a target distribution Q , and define the quantity F ( P ) = FID ( P , Q ) . Also fix two distributions P 0 = P 1 . Suppose there exists some estimator ˆ F ( X ) based on a sample of size n for which

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now consider the function

<!-- formula-not-decoded -->

This function R ( α ) is therefore a polynomial in α of degree at most n .

But let's consider the following one-dimensional case:

<!-- formula-not-decoded -->

The mean and variance of (1 - α ) P 0 + α P 1 can be written as

<!-- formula-not-decoded -->

Thus

<!-- formula-not-decoded -->

̸

Note that ( µ α - µ ) 2 + σ 2 α + σ 2 is a quadratic function of α . However, σ α is polynomial in α only in the trivial case when P 0 = P 1 . Thus R ( α ) is not a polynomial when P 0 = P 1 , and so no estimator of the FID to an arbitrary fixed normal distribution Q can be unbiased on any class of distributions which includes two-component Gaussian mixtures.

There is also no unbiased estimator is available in the two-sample setting, where Q is also unknown, by the same trivial extension to this argument as in Theorem 3.

Unfortunately, this type of analysis can tell us nothing about whether there exists an estimator which is unbiased on normal distributions. Given that the distributions used for the FID in practice are clearly not normal, however, a practical unbiased estimator of the FID is impossible.

Figure 6: Gaussian noise; α is the mixture weight of the Gaussian noise.

<!-- image -->

Figure 7: Gaussian blur; α is the standard deviation of the Gaussian filter.

<!-- image -->

## E COMPARISON OF EVALUATION METRICS' RESILIENCE TO NOISE

We replicate here the experiments of Heusel et al.'s Appendix 1, which examines the behavior of the Inception and FID scores as images are increasingly 'disturbed,' and additionally consider the KID. As the 'disturbance level' α is increased, images are altered more from the reference distribution. Figures 6 to 11 show the FID, KID, and negative (for comparability) Inception score for both CelebA (left) and CIFAR-10 (right); each score is scaled to [0 , 1] to be plotted on one axis, with minimal and maximal values shown in the legend.

Note that Heusel et al. compared means and variances computed on 50 000 random disturbed CelebA images to those computed on the full 200 000 dataset; we instead use the standard traintest split, computing the disturbances on the 160 000 -element training set and comparing to the 20 000 -element test set. In this (very slightly) different setting, we find the Inception score to be monotonic with increasing noise on more of the disturbance types than did Heusel et al. (2017). We also found similar behavior on the CIFAR-10 dataset, again comparing the noised training set (size 50 000 ) to the test set (size 10 000 ). This perhaps means that the claimed non-monotonicity of the Inception score is quite sensitive to the exact experimental setting; further investigation into this phenomenon would be intriguing for future work.

## F SAMPLES AND DETAILED RESULTS FOR MNIST AND CIFAR-10

MNIST After training for 50 000 generator iterations, all variants achieved reasonable results. Among MMD models, only the distance kernel saw an improvement with more neurons in the top layer. Table 3 shows the quantitative measures, computed on the basis of a LeNet model. All have achieved KIDs of essentially zero, and FIDs around the same as that of the test set, with Inception scores slightly lower. Model samples are shown in Figure 12.

Figure 8: Black rectangles; α is the portion of the image size each rectangle contains.

<!-- image -->

Figure 9: Swirl: α is the strength of the swirl effect.

<!-- image -->

Figure 10: Salt and pepper noise: α is the portion of pixels which are noised.

<!-- image -->

Figure 11: ImageNet contamination: α is the portion of images replaced by ImageNet samples.

<!-- image -->

Table 3: Mean (standard deviation) of score evaluations for the MNIST models.

|            | critic size   | critic size   |             |              |               |
|------------|---------------|---------------|-------------|--------------|---------------|
| loss       | filters       | top layer     | Inception   | FID          | KID           |
| rq         | 16            | 16            | 9.11 (0.01) | 4.206 (0.05) | 0.005 (0.004) |
| rbf        | 16            | 16            | 8.98 (0.02) | 8.264 (0.02) | 0.011 (0.006) |
| dot        | 16            | 16            | 8.86 (0.02) | 6.245 (0.06) | 0.006 (0.004) |
| dist       | 16            | 256           | 9.13 (.004) | 6.179 (0.05) | 0.005 (0.004) |
| Cramér GAN | 16            | 256           | 9.25 (0.02) | 3.385 (0.10) | 0.006 (0.005) |
| WGAN-GP    | 16            | 1             | 9.12 (0.02) | 6.915 (0.10) | 0.009 (0.004) |
| test set   | -             | -             | 9.78 (0.02) | 4.305 (0.16) | 0.003 (0.003) |

Examining samples during training, we observed that rbf more frequently produces extremely 'blurry' outputs, which can persist for a substantial amount of time before eventually resolving. This makes sense, given the very fast gradient decay of the rbf kernel: when generator samples are extremely far away from the reference samples, slight improvements yield very little reward for the generator, and so bad samples can stay bad for a long time.

CIFAR-10 Scores for various models trained on CIFAR-10 are shown in Table 4. The scores for rq with a small critic network approximately match those of WGAN-GP with a large critic network, at substantially reduced computational cost. With a small critic, WGAN-GP, Cramér GAN and the distance kernel all performed very poorly. Samples from these models are presented in Figure 13.

Table 4: Mean (standard deviation) of score evaluations for the CIFAR-10 models.

|            | critic size   | critic size   |              |               |               |
|------------|---------------|---------------|--------------|---------------|---------------|
| loss       | filters       | top layer     | Inception    | FID           | KID           |
| rq         | 16            | 16            | 5.86 (0.06)  | 48.10 (0.16)  | 0.032 (0.001) |
| rq         | 64            | 16            | 6.51 (0.03)  | 39.90 (0.29)  | 0.027 (0.001) |
| dist       | 16            | 256           | 4.53 (0.03)  | 80.48 (0.19)  | 0.061 (0.001) |
| dist       | 64            | 256           | 6.39 (0.04)  | 40.25 (0.19)  | 0.028 (0.001) |
| Cramér GAN | 16            | 256           | 4.67 (0.02)  | 74.93 (0.32)  | 0.060 (0.001) |
| Cramér GAN | 64            | 256           | 6.39 (0.01)  | 40.27 (0.15)  | 0.028 (0.001) |
| WGAN-GP    | 16            | 1             | 3.15 (0.01)  | 147.09 (0.31) | 0.116 (0.002) |
| WGAN-GP    | 64            | 1             | 6.53 (0.02)  | 37.52 (0.19)  | 0.026 (0.001) |
| test set   | -             | -             | 11.21 (0.13) | 6.11 (0.05)   | 0.000 (0.000) |

(a) MMD rq

<!-- image -->

<!-- image -->

<!-- image -->

(b) MMD rbf

(c) MMD dot

<!-- image -->

2 4 8 9 7 9 6 7 9 6 2 h 6 88 4 0 2 1 2 7 2 6 3 5 9 q0 3

6 8 7 2 8 8 7 à 9 8 7 0 5 8 1 5 5 6 7 8 8 3 1 9 8 5 7 5 h h 3 8 3 9 a 6 h

(d) Cramer GAN

(e) WGAN-GP

(f) MNIST test set

Figure 12: Samples from the models listed in Table 3. Rational-quadratic and Gaussian kernels obtain retain sample quality despite reduced discriminator complexity. Each of these models generates good quality samples with the standard DCGAN discriminator (critic size 64).

<!-- image -->

(b) WGAN-GP, critic size 16

<!-- image -->

(a) MMD rq *, critic size 16

<!-- image -->

(c) Cramér GAN, critic size 16

<!-- image -->

(e) Cramér GAN, critic size 64

<!-- image -->

(d) MMD rq *, critic size 64

<!-- image -->

(f) Test set

Figure 13: Comparison of samples from various models, as well as true samples from the test set. WGAN-GP samples with critic size 16 are quite bad. Cramér GAN samples with critic size 16 are more appealing to the eye, but seem to have pixel-level issues. Large-critic Cramér and MMD rq * GAN are of similar quality.