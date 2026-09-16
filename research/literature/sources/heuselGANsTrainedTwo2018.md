## GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium

Martin Heusel Hubert Ramsauer Thomas Unterthiner Bernhard Nessler

## Sepp Hochreiter

LIT AI Lab &amp; Institute of Bioinformatics, Johannes Kepler University Linz

A-4040 Linz, Austria

{mhe,ramsauer,unterthiner,nessler,hochreit}@bioinf.jku.at

## Abstract

Generative Adversarial Networks (GANs) excel at creating realistic images with complex models for which maximum likelihood is infeasible. However, the convergence of GAN training has still not been proved. We propose a two time-scale update rule (TTUR) for training GANs with stochastic gradient descent on arbitrary GAN loss functions. TTUR has an individual learning rate for both the discriminator and the generator. Using the theory of stochastic approximation, we prove that the TTUR converges under mild assumptions to a stationary local Nash equilibrium. The convergence carries over to the popular Adam optimization, for which we prove that it follows the dynamics of a heavy ball with friction and thus prefers flat minima in the objective landscape. For the evaluation of the performance of GANs at image generation, we introduce the 'Fréchet Inception Distance' (FID) which captures the similarity of generated images to real ones better than the Inception Score. In experiments, TTUR improves learning for DCGANs and Improved Wasserstein GANs (WGAN-GP) outperforming conventional GAN training on CelebA, CIFAR-10, SVHN, LSUN Bedrooms, and the One Billion Word Benchmark.

## Introduction

Generative adversarial networks (GANs) [18] have achieved outstanding results in generating realistic images [51, 36, 27, 3, 6] and producing text [23]. GANs can learn complex generative models for which maximum likelihood or a variational approximations are infeasible. Instead of the likelihood, a discriminator network serves as objective for the generative model, that is, the generator. GAN learning is a game between the generator, which constructs synthetic data from random variables, and the discriminator, which separates synthetic data from real world data. The generator's goal is to construct data in such a way that the discriminator cannot tell them apart from real world data. Thus, the discriminator tries to minimize the synthetic-real discrimination error while the generator tries to maximize this error. Since training GANs is a game and its solution is a Nash equilibrium, gradient descent may fail to converge [53, 18, 20]. Only local Nash equilibria are found, because gradient descent is a local optimization method. If there exists a local neighborhood around a point in parameter space where neither the generator nor the discriminator can unilaterally decrease their respective losses, then we call this point a local Nash equilibrium.

5000

10000

15000

Iteration

onvergence of deterministic algorithm under different step sizes.

0.4

0.2

0

10 0

10 1

10 2

10 3

10 4

10 5

Iteration

Fig. 4. Convergence under noisy feedback (the biased case).

<!-- image -->

Iteration

onvergence under noisy feedback (the unbiased case). e convergence to a neighborhood is the best we can Fig. 5. 'Zoomed-in' convergence behavior of the iterates in Figure 4. V. STOCHASTIC STABILITY OF TWO TIME-SCALE ALGORITHM UNDER NOISY FEEDBACK Figure 1: Left: Original vs. TTUR GAN training on CelebA. Right: Figure from Zhang 2007 [61] which shows the distance of the parameter from the optimum for a one time-scale update of a 4 node network flow problem. When the upper bounds on the errors ( α, β ) are small, the iterates oscillate and repeatedly return to a neighborhood of the optimal solution (see also Appendix Section A2.3).

on the { α s , β ( i,j ) } are small, the iterates return to orhood of the optimal solution. However, when the n errors are large, the recurrent behavior of the may not occur, and the iterates may diverge. This ates the theoretical analysis. We can further observe . 5 that the smaller the upper-bound is, the smaller the tion region' A η becomes, indicating that the iterates 'closer' to the optimal points. Ξ 2 : maximize { m s ≤ x s ≤ M s , p } ∑ s U s ( x s ) subject to ∑ s : l ∈L ( s ) x s ≤ c l , ∀ l c l = h l ( p ) , ∀ l p ∈ H , (39) where the link capacities { c l } are functions of specific MAC parameters p (for instance, p can be transmission probabilities Recently actor-critic learning has been analyzed using stochastic approximation. Prasad et al. [50] showed that a two time-scale update rule ensures that training reaches a stationary local Nash equilibrium if the critic learns faster than the actor. Convergence was proved via an ordinary differential equation (ODE), whose stable limit points coincide with stationary local Nash equilibria. Wefollow the same approach. We prove that GANs converge to a local Nash equilibrium when trained by a two time-scale update rule (TTUR), i.e., when discriminator and generator have separate learning rates. This also leads to better results in experiments. The main premise is that the discriminator

hereas by using diminishing step sizes, convergence bability one to the optimal points is made possible. bility of The Stochastic Algorithm: The Biased Case: at when the gradient estimation error is biased, we ope to obtain almost sure convergence to the optimal . Instead, we have shown that provided that the biased asymptotically uniformly bounded, the iterates return ntraction region' infinitely often. In this example, we hat α s ( n ) = β ( i,j ) ( n ) and are uniformly bounded by a positive value. We also assume that ζ s ( n ) ∼ N (0 , 1) ) ( n ) ∼ N (0 , 1) , for all s and ( i, j ) . lot the iterates (using the relative distance to the points) in Fig. 4, which is further 'zoomed in' in t can be observed from Fig. 4 that when the upperIn the previous sections, we have applied the dual decomposition method to Problem (1) and devised the primal-dual algorithm, which is a single time-scale algorithm. As noted in Section I, there are many other decomposition methods. In particular, the primal decomposition method is a useful machinery for problem with coupled variables [31]; and when some of the variables are fixed, the rest of the problem may decouple into several subproblems. This naturally yields multiple time-scale algorithms. It is also of great interest to examine the stability of the multiple time-scale algorithms in the presence of noisy feedback, and compare with the single time-scale algorithms, in terms of complexity and robustness. To get a more concrete sense of the two time-scale algorithms based on primal decomposition, we consider the following NUM problem: However, when the upper bounds on the errors are large, the iterates typically diverge. To characterize the convergence properties of training general GANs is still an open challenge [19, 20]. For special GAN variants, convergence can be proved under certain assumptions [39, 22, 56]. A prerequisit for many convergence proofs is local stability [35] which was shown for GANs by Nagarajan and Kolter [46] for a min-max GAN setting. However, Nagarajan and Kolter require for their proof either rather strong and unrealistic assumptions or a restriction to a linear discriminator. Recent convergence proofs for GANs hold for expectations over training samples or for the number of examples going to infinity [37, 45, 40, 4], thus do not consider mini-batch learning which leads to a stochastic gradient [57, 25, 42, 38].

10 converges to a local minimum when the generator is fixed. If the generator changes slowly enough, then the discriminator still converges, since the generator perturbations are small. Besides ensuring convergence, the performance may also improve since the discriminator must first learn new patterns before they are transferred to the generator. In contrast, a generator which is overly fast, drives the discriminator steadily into new regions without capturing its gathered information. In recent GAN implementations, the discriminator often learned faster than the generator. A new objective slowed down the generator to prevent it from overtraining on the current discriminator [53]. The Wasserstein GAN algorithm uses more update steps for the discriminator than for the generator [3]. We compare TTUR and standard GAN training. Fig. 1 shows at the left panel a stochastic gradient example on CelebA for original GAN training (orig), which often leads to oscillations, and the TTUR. On the right panel an example of a 4 node network flow problem of Zhang et al. [61] is shown. The distance between the actual parameter and its optimum for an one time-scale update rule is shown across iterates. When the upper bounds on the errors are small, the iterates return to a neighborhood of the optimal solution, while for large errors the iterates may diverge (see also Appendix Section A2.3).

## Our novel contributions in this paper are:

- The two time-scale update rule for GANs,
- The description of Adam as heavy ball with friction and the resulting second order differential equation,
- We proof that GANs trained with TTUR converge to a stationary local Nash equilibrium,
- The convergence of GANs trained with TTUR and Adam to a stationary local Nash equilibrium,
- We introduce the 'Fréchet Inception Distance' (FID) to evaluate GANs, which is more consistent than the Inception Score.

## Two Time-Scale Update Rule for GANs

We consider a discriminator D ( . ; w ) with parameter vector w and a generator G ( . ; θ ) with parameter vector θ . Learning is based on a stochastic gradient ˜ g ( θ , w ) of the discriminator's loss function L D and a stochastic gradient ˜ h ( θ , w ) of the generator's loss function L G . The loss functions L D and L G can be the original as introduced in Goodfellow et al. [18], its improved versions [20], or recently proposed losses for GANs like the Wasserstein GAN [3]. Our setting is not restricted to min-max GANs, but is valid for all other, more general GANs for which the discriminator's loss function L D is not necessarily related to the generator's loss function L G . The gradients ˜ g ( θ , w ) and ˜ h ( θ , w ) are stochastic, since they use mini-batches of m real world samples x ( i ) , 1 ⩽ i ⩽ m and m synthetic samples z ( i ) , 1 ⩽ i ⩽ m which are randomly chosen. If the true gradients are g ( θ , w ) = ∇ w L D and h ( θ , w ) = ∇ θ L G , then we can define ˜ g ( θ , w ) = g ( θ , w )+ M ( w ) and ˜ h ( θ , w ) = h ( θ , w )+ M ( θ ) with random variables M ( w ) and M ( θ ) . Thus, the gradients ˜ g ( θ , w ) and ˜ h ( θ , w ) are stochastic approximations to the true gradients. Consequently, we analyze convergence of GANs by two time-scale stochastic approximations algorithms. For a two time-scale update rule (TTUR), we use the learning rates b ( n ) and a ( n ) for the discriminator and the generator update, respectively:

<!-- formula-not-decoded -->

For more details on the following convergence proof and its assumptions see Appendix Section A2.1. To prove convergence of GANs learned by TTUR, we make the following assumptions (The actual assumption is ended by ◀ , the following text are just comments and explanations):

- (A1) The gradients h and g are Lipschitz. ◀ Consequently, networks with Lipschitz smooth activation functions like ELUs ( α = 1 ) [13] fulfill the assumption but not ReLU networks.
- (A3) The stochastic gradient errors { M ( θ ) n } and { M ( w ) n } are martingale difference sequences w.r.t. the increasing σ -field F n = σ ( θ l , w l , M ( θ ) l , M ( w ) l , l ⩽ n ) , n ⩾ 0 with E [ ‖ M ( θ ) n ‖ 2 | F ( θ ) n ] ⩽ B 1 and E [ ‖ M ( w ) n ‖ 2 | F ( w ) n ] ⩽ B 2 , where B 1 and B 2 are positive deterministic constants. ◀ The original Assumption (A3) from Borkar 1997 follows from Lemma 2 in [7] (see also [52]). The assumption is fulfilled in the Robbins-Monro setting, where mini-batches are randomly sampled and the gradients are bounded.

<!-- formula-not-decoded -->

- (A4) For each θ , the ODE ˙ w ( t ) = g ( θ , w ( t ) ) has a local asymptotically stable attractor λ ( θ ) within a domain of attraction G θ such that λ is Lipschitz. The ODE ˙ θ ( t ) = h ( θ ( t ) , λ ( θ ( t )) ) has a local asymptotically stable attractor θ ∗ within a domain of attraction. ◀ The discriminator must converge to a minimum for fixed generator parameters and the generator, in turn, must converge to a minimum for this fixed discriminator minimum. Borkar 1997 required unique global asymptotically stable equilibria [9]. The assumption of global attractors was relaxed to local attractors via Assumption (A6) and Theorem 2.7 in Karmakar &amp; Bhatnagar [28]. See for more details Assumption (A6) in the Appendix Section A2.1.3. Here, the GAN objectives may serve as Lyapunov functions. These assumptions of locally stable ODEs can be ensured by an additional weight decay term in the loss function which increases the eigenvalues of the Hessian. Therefore, problems with a region-wise constant discriminator that has zero second order derivatives are avoided. For further discussion see Appendix Section A2 (C3).
- (A5) sup n ‖ θ n ‖ &lt; ∞ and sup n ‖ w n ‖ &lt; ∞ . ◀ Typically ensured by the objective or a weight decay term.

The next theorem has been proved in the seminal paper of Borkar 1997 [9].

Theorem 1 (Borkar) . If the assumptions are satisfied, then the updates Eq. (1) converge to ( θ ∗ , λ ( θ ∗ )) a.s.

The solution ( θ ∗ , λ ( θ ∗ )) is a stationary local Nash equilibrium [50], since θ ∗ as well as λ ( θ ∗ ) are local asymptotically stable attractors with g ( θ ∗ , λ ( θ ∗ ) ) = 0 and h ( θ ∗ , λ ( θ ∗ ) ) = 0 . An alternative approach to the proof of convergence using the Poisson equation for ensuring a solution to the fast update rule can be found in the Appendix Section A2.1.2. This approach assumes a linear update function in the fast update rule which, however, can be a linear approximation to a nonlinear gradient [30, 32]. For the rate of convergence see Appendix Section A2.2, where Section A2.2.1 focuses on linear and Section A2.2.2 on non-linear updates. For equal time-scales it can only be proven that the updates revisit an environment of the solution infinitely often, which, however, can be very large [61, 14]. For more details on the analysis of equal time-scales see Appendix Section A2.3. The main idea of the proof of Borkar [9] is to use ( T, δ ) perturbed ODEs according to Hirsch 1989 [24] (see also Appendix Section C of Bhatnagar, Prasad, &amp; Prashanth 2013 [8]). The proof relies on the fact that there eventually is a time point when the perturbation of the slow update rule is small enough (given by δ ) to allow the fast update rule to converge. For experiments with TTUR, we aim at finding learning rates such that the slow update is small enough to allow the fast to converge. Typically, the slow update is the generator and the fast update the discriminator. We have to adjust the two learning rates such that the generator does not affect discriminator learning in a undesired way and perturb it too much. However, even a larger learning rate for the generator than for the discriminator may ensure that the discriminator has low perturbations. Learning rates cannot be translated directly into perturbation since the perturbation of the discriminator by the generator is different from the perturbation of the generator by the discriminator.

## Adam Follows an HBF ODE and Ensures TTUR Convergence

In our experiments, we aim at using Adam stochastic approximation to avoid mode collapsing. GANs suffer from 'mode collapsing' where large masses of probability are mapped onto a few modes that cover only small regions. While these regions represent meaningful samples, the variety of the

Figure 2: Heavy Ball with Friction, where the ball with mass overshoots the local minimum θ + and settles at the flat minimum θ ∗ .

<!-- image -->

real world data is lost and only few prototype samples are generated. Different methods have been proposed to avoid mode collapsing [11, 43]. We obviate mode collapsing by using Adam stochastic approximation [29]. Adam can be described as Heavy Ball with Friction (HBF) (see below), since it averages over past gradients. This averaging corresponds to a velocity that makes the generator resistant to getting pushed into small regions. Adam as an HBF method typically overshoots small local minima that correspond to mode collapse and can find flat minima which generalize well [26]. Fig. 2 depicts the dynamics of HBF, where the ball settles at a flat minimum. Next, we analyze whether GANs trained with TTUR converge when using Adam. For more details see Appendix Section A3.

We recapitulate the Adam update rule at step n , with learning rate a , exponential averaging factors β 1 for the first and β 2 for the second moment of the gradient ∇ f ( θ n - 1 ) :

where following operations are meant componentwise: the product ⊙ , the square root √ . , and the division / in the last line. Instead of learning rate a , we introduce the damping coefficient a ( n ) with a ( n ) = an - τ for τ ∈ (0 , 1] . Adam has parameters β 1 for averaging the gradient and β 2 parametrized by a positive α for averaging the squared gradient. These parameters can be considered as defining a memory for Adam. To characterize β 1 and β 2 in the following, we define the exponential memory r ( n ) = r and the polynomial memory r ( n ) = r/ ∑ n l =1 a ( l ) for some positive constant r . The next theorem describes Adam by a differential equation, which in turn allows to apply the idea of ( T, δ ) perturbed ODEs to TTUR. Consequently, learning GANs with TTUR and Adam converges.

<!-- formula-not-decoded -->

Theorem 2. If Adam is used with β 1 = 1 - a ( n +1) r ( n ) , β 2 = 1 - αa ( n +1) r ( n ) and with ∇ f as the full gradient of the lower bounded, continuously differentiable objective f , then for stationary second moments of the gradient, Adam follows the differential equation for Heavy Ball with Friction

(HBF):

Adam converges for gradients ∇ f that are L -Lipschitz.

<!-- formula-not-decoded -->

Proof. Gadat et al. derived a discrete and stochastic version of Polyak's Heavy Ball method [49], the Heavy Ball with Friction (HBF) [17]:

<!-- formula-not-decoded -->

These update rules are the first moment update rules of Adam [29]. The HBF can be formulated as the differential equation Eq. (3) [17]. Gadat et al. showed that the update rules Eq. (4) converge for loss functions f with at most quadratic grow and stated that convergence can be proofed for ∇ f that are L -Lipschitz [17]. Convergence has been proved for continuously differentiable f that is quasiconvex (Theorem 3 in Goudou &amp; Munier [21]). Convergence has been proved for ∇ f that is L -Lipschitz and bounded from below (Theorem 3.1 in Attouch et al. [5]). Adam normalizes the average m n by the second moments v n of of the gradient g n : v n = E[ g n ⊙ g n ] . m n is componentwise divided by the square root of the components of v n . We assume that the second moments of g n are stationary, i.e., v = E[ g n ⊙ g n ] . In this case the normalization can be considered as additional noise since the normalization factor randomly deviates from its mean. In the HBF interpretation the normalization by √ v corresponds to introducing gravitation. We obtain

<!-- formula-not-decoded -->

For a stationary second moment v and β 2 = 1 - αa ( n +1) r ( n ) , we have ∆ v n ∝ a ( n +1) r ( n ) . We use a componentwise linear approximation to Adam's second moment normalization 1 / √ v +∆ v n ≈ 1 / √ v - (1 / (2 v ⊙ √ v )) ⊙ ∆ v n +O(∆ 2 v n ) , where all operations are meant componentwise. If we set M ( v ) n +1 = - ( m n ⊙ ∆ v n ) / (2 v ⊙ √ v a ( n + 1) r ( n )) , then m n / √ v n ≈ m n / √ v + a ( n + 1) r ( n ) M ( v ) n +1 and E [ M ( v ) n +1 ] = 0 , since E[ g l ⊙ g l - v ] = 0 . For a stationary second moment v , the random variable { M ( v ) n } is a martingale difference sequence with a bounded second moment. Therefore { M ( v ) n +1 } can be subsumed into { M n +1 } in update rules Eq. (4). The factor 1 / √ v can be componentwise incorporated into the gradient g which corresponds to rescaling the parameters without changing the minimum. □

According to Attouch et al. [5] the energy, that is, a Lyapunov function, is E ( t ) = 1 / 2 | ˙ θ ( t ) | 2 + f ( θ ( t )) and ˙ E ( t ) = - a | ˙ θ ( t ) | 2 &lt; 0 . Since Adam can be expressed as differential equation and has a Lyapunov function, the idea of ( T, δ ) perturbed ODEs [9, 24, 10] carries over to Adam. Therefore the convergence of Adam with TTUR can be proved via two time-scale stochastic approximation analysis like in Borkar [9] for stationary second moments of the gradient.

In the Appendix we further discuss the convergence of two time-scale stochastic approximation algorithms with additive noise, linear update functions depending on Markov chains, nonlinear update functions, and updates depending on controlled Markov processes. Futhermore, the Appendix presents work on the rate of convergence for both linear and nonlinear update rules using similar techniques as the local stability analysis of Nagarajan and Kolter [46]. Finally, we elaborate more on equal time-scale updates, which are investigated for saddle point problems and actor-critic learning.

## Experiments

Performance Measure. Before presenting the experiments, we introduce a quality measure for models learned by GANs. The objective of generative learning is that the model produces data which matches the observed data. Therefore, each distance between the probability of observing real world data p w ( . ) and the probability of generating model data p ( . ) can serve as performance measure for generative models. However, defining appropriate performance measures for generative models is difficult [55]. The best known measure is the likelihood, which can be estimated by annealed importance sampling [59]. However, the likelihood heavily depends on the noise assumptions for the real data and can be dominated by single samples [55]. Other approaches like density estimates have drawbacks, too [55]. A well-performing approach to measure the performance of GANs is the 'Inception Score' which correlates with human judgment [53]. Generated samples are fed into an inception model that was trained on ImageNet. Images with meaningful objects are supposed to have low label (output) entropy, that is, they belong to few object classes. On the other hand, the entropy across images should be high, that is, the variance over the images should be large. Drawback of the Inception Score is that the statistics of real world samples are not used and compared to the statistics of synthetic samples. Next, we improve the Inception Score. The equality p ( . ) = p w ( . ) holds except for a non-measurable set if and only if ∫ p ( . ) f ( x ) dx = ∫ p w ( . ) f ( x ) dx for a basis f ( . ) spanning the function space in which p ( . ) and p w ( . ) live. These equalities of expectations are used to describe distributions by moments or cumulants, where f ( x ) are polynomials of the data x . We generalize these polynomials by replacing x by the coding layer of an inception model in order to obtain vision-relevant features. For practical reasons we only consider the first two polynomials, that is, the first two moments: mean and covariance. The Gaussian is the maximum entropy distribution for given mean and covariance, therefore we assume the coding units to follow a multidimensional Gaussian. The difference of two Gaussians (synthetic and real-world images) is measured by the Fréchet distance [16] also known as Wasserstein-2 distance [58]. We call the Fréchet distance d ( ., . ) between the Gaussian with mean ( m , C ) obtained from p ( . ) and the Gaussian with mean ( m w , C w ) obtained from p w ( . ) the 'Fréchet Inception Distance' (FID), which is given by [15]:

Figure 3: FID is evaluated for upper left: Gaussian noise, upper middle: Gaussian blur, upper right: implanted black rectangles, lower left: swirled images, lower middle: salt and pepper noise, and lower right: CelebA dataset contaminated by ImageNet images. The disturbance level rises from zero and increases to the highest level. The FID captures the disturbance level very well by monotonically increasing.

<!-- image -->

<!-- formula-not-decoded -->

Next we show that the FID is consistent with increasing disturbances and human judgment. Fig. 3 evaluates the FID for Gaussian noise, Gaussian blur, implanted black rectangles, swirled images, salt and pepper noise, and CelebA dataset contaminated by ImageNet images. The FID captures the disturbance level very well. In the experiments we used the FID to evaluate the performance of GANs. For more details and a comparison between FID and Inception Score see Appendix Section A1, where we show that FID is more consistent with the noise level than the Inception Score.

Model Selection and Evaluation. We compare the two time-scale update rule (TTUR) for GANs with the original GAN training to see whether TTUR improves the convergence speed and performance of GANs. We have selected Adam stochastic optimization to reduce the risk of mode collapsing. The advantage of Adam has been confirmed by MNIST experiments, where Adam indeed considerably reduced the cases for which we observed mode collapsing. Although TTUR ensures that the discriminator converges during learning, practicable learning rates must be found for each experiment. We face a trade-off since the learning rates should be small enough (e.g. for the generator) to ensure convergence but at the same time should be large enough to allow fast learning. For each of the experiments, the learning rates have been optimized to be large while still ensuring stable training which is indicated by a decreasing FID or Jensen-Shannon-divergence (JSD). We further fixed the time point for stopping training to the update step when the FID or Jensen-Shannon-divergence of the best models was no longer decreasing. For some models, we observed that the FID diverges or starts to increase at a certain time point. An example of this behaviour is shown in Fig. 5. The performance of generative models is evaluated via the Fréchet Inception Distance (FID) introduced above. For the One Billion Word experiment, the normalized JSD served as performance measure. For computing the FID, we propagated all images from the training dataset through the pretrained Inception-v3 model following the computation of the Inception Score [53], however, we use the last pooling layer as coding layer. For this coding layer, we calculated the mean m w and the covariance matrix C w . Thus, we approximate the first and second central moment of the function given by the Inception coding layer under the real world distribution. To approximate these moments for the model distribution, we generate 50,000 images, propagate them through the Inception-v3 model, and then compute the mean m and the covariance matrix C . For computational efficiency, we evaluate the FID every 1,000 DCGAN mini-batch updates, every 5,000 WGAN-GP outer iterations for the image experiments, and every 100 outer iterations for the WGAN-GP language model. For the one time-scale updates a WGAN-GP outer iteration for the image model consists of five discriminator mini-batches and ten discriminator mini-batches for the language model, where we follow the original implementation. For TTUR however, the discriminator is updated only once per iteration. We repeat the training for each single time-scale (orig) and TTUR learning rate eight times for the image datasets and ten times for the language benchmark. Additionally to the mean FID training progress we show the minimum and maximum FID over all runs at each evaluation time-step. For more details, implementations and further results see Appendix Section A4 and A6.

Simple Toy Data. We first want to demonstrate the difference between a single time-scale update rule and TTUR on a simple toy min/max problem where a saddle point should be found. The objective f ( x, y ) = (1 + x 2 )(100 - y 2 ) in Fig. 4 (left) has a saddle point at ( x, y ) = (0 , 0) and fulfills assumption A4. The norm ‖ ( x, y ) ‖ measures the distance of the parameter vector ( x, y ) to the saddle point. We update ( x, y ) by gradient descent in x and gradient ascent in y using additive Gaussian noise in order to simulate a stochastic update. The updates should converge to the saddle point ( x, y ) = (0 , 0) with objective value f (0 , 0) = 100 and the norm 0 . In Fig. 4 (right), the first two rows show one time-scale update rules. The large learning rate in the first row diverges and has large fluctuations. The smaller learning rate in the second row converges but slower than the TTUR in the third row which has slow x -updates. TTUR with slow y -updates in the fourth row also converges but slower.

Figure 4: Left: Plot of the objective with a saddle point at (0 , 0) . Right: Training progress with equal learning rates of 0 . 01 (first row) and 0 . 001 (second row)) for x and y , TTUR with a learning rate of 0 . 0001 for x vs. 0 . 01 for y (third row) and a larger learning rate of 0 . 01 for x vs. 0 . 0001 for y (fourth row). The columns show the function values (left), norms (middle), and ( x, y ) (right). TTUR (third row) clearly converges faster than with equal time-scale updates and directly moves to the saddle point as shown by the norm and in the ( x, y ) -plot.

<!-- image -->

DCGAN on Image Data. We test TTUR for the deep convolutional GAN (DCGAN) [51] at the CelebA, CIFAR-10, SVHN and LSUN Bedrooms dataset. Fig. 5 shows the FID during learning with the original learning method (orig) and with TTUR. The original training method is faster at the beginning, but TTUR eventually achieves better performance. DCGAN trained TTUR reaches constantly a lower FID than the original method and for CelebA and LSUN Bedrooms all one time-scale runs diverge. For DCGAN the learning rate of the generator is larger then that of the discriminator, which, however, does not contradict the TTUR theory (see the Appendix Section A5). In Table 1 we report the best FID with TTUR and one time-scale training for optimized number of updates and learning rates. TTUR constantly outperforms standard training and is more stable.

Figure 5: Mean FID (solid line) surrounded by a shaded area bounded by the maximum and the minimum over 8 runs for DCGAN on CelebA, CIFAR-10, SVHN, and LSUN Bedrooms. TTUR learning rates are given for the discriminator b and generator a as: 'TTUR b a '. Top Left: CelebA. Top Right: CIFAR-10, starting at mini-batch update 10k for better visualisation. Bottom Left: SVHN. Bottom Right: LSUN Bedrooms. Training with TTUR (red) is more stable, has much lower variance, and leads to a better FID.

<!-- image -->

WGAN-GP on Image Data. We used the WGAN-GP image model [23] to test TTUR with the CIFAR-10 and LSUN Bedrooms datasets. In contrast to the original code where the discriminator is trained five times for each generator update, TTUR updates the discriminator only once, therefore we align the training progress with wall-clock time. The learning rate for the original training was optimized to be large but leads to stable learning. TTUR can use a higher learning rate for the discriminator since TTUR stabilizes learning. Fig. 6 shows the FID during learning with the original learning method and with TTUR. Table 1 shows the best FID with TTUR and one time-scale training for optimized number of iterations and learning rates. Again TTUR reaches lower FIDs than one time-scale training.

Figure 6: Mean FID (solid line) surrounded by a shaded area bounded by the maximum and the minimum over 8 runs for WGAN-GP on CelebA, CIFAR-10, SVHN, and LSUN Bedrooms. TTUR learning rates are given for the discriminator b and generator a as: 'TTUR b a '. Left: CIFAR-10, starting at minute 20. Right: LSUN Bedrooms. Training with TTUR (red) has much lower variance and leads to a better FID.

<!-- image -->

Figure 7: Performance of WGAN-GP models trained with the original (orig) and our TTUR method on the One Billion Word benchmark. The performance is measured by the normalized JensenShannon-divergence based on 4-gram ( left ) and 6-gram ( right ) statistics averaged (solid line) and surrounded by a shaded area bounded by the maximum and the minimum over 10 runs, aligned to wall-clock time and starting at minute 150. TTUR learning (red) clearly outperforms the original one time-scale learning.

<!-- image -->

WGAN-GP on Language Data. Finally the One Billion Word Benchmark [12] serves to evaluate TTUR on WGAN-GP. The character-level generative language model is a 1D convolutional neural network (CNN) which maps a latent vector to a sequence of one-hot character vectors of dimension 32 given by the maximum of a softmax output. The discriminator is also a 1D CNN applied to sequences of one-hot vectors of 32 characters. Since the FID criterium only works for images, we measured the performance by the Jensen-Shannon-divergence (JSD) between the model and the real world distribution as has been done previously [23]. In contrast to the original code where the critic is trained ten times for each generator update, TTUR updates the discriminator only once, therefore we align the training progress with wall-clock time. The learning rate for the original training was optimized to be large but leads to stable learning. TTUR can use a higher learning rate for the discriminator since TTUR stabilizes learning. We report for the 4 and 6-gram word evaluation the normalized mean JSD for ten runs for original training and TTUR training in Fig. 7. In Table 1 we report the best JSD at an optimal time-step where TTUR outperforms the standard training for both measures. The improvement of TTUR on the 6-gram statistics over original training shows that TTUR enables to learn to generate more subtle pseudo-words which better resembles real words.

Table 1: The performance of DCGAN and WGAN-GP trained with the original one time-scale update rule and with TTUR on CelebA, CIFAR-10, SVHN, LSUN Bedrooms and the One Billion Word Benchmark. During training we compare the performance with respect to the FID and JSD for optimized number of updates. TTUR exhibits consistently a better FID and a better JSD.

| DCGAN Image      | DCGAN Image      |            |         |      |        |       |         | FID   |
|------------------|------------------|------------|---------|------|--------|-------|---------|-------|
| dataset          | method           | b, a       | updates | FID  | method | b = a | updates |       |
| CelebA           | TTUR             | 1e-5, 5e-4 | 225k    | 12.5 | orig   | 5e-4  | 70k     | 21.4  |
| CIFAR-10         | TTUR             | 1e-4, 5e-4 | 75k     | 36.9 | orig   | 1e-4  | 100k    | 37.7  |
| SVHN             | TTUR             | 1e-5, 1e-4 | 165k    | 12.5 | orig   | 5e-5  | 185k    | 21.4  |
| LSUN             | TTUR             | 1e-5, 1e-4 | 340k    | 57.5 | orig   | 5e-5  | 70k     | 70.4  |
| WGAN-GP Image    | WGAN-GP Image    |            |         |      |        |       |         |       |
| dataset          | method           | b, a       | time(m) | FID  | method | b = a | time(m) | FID   |
| CIFAR-10         | TTUR             | 3e-4, 1e-4 | 700     | 24.8 | orig   | 1e-4  | 800     | 29.3  |
| LSUN             | TTUR             | 3e-4, 1e-4 | 1900    | 9.5  | orig   | 1e-4  | 2010    | 20.5  |
| WGAN-GP Language | WGAN-GP Language |            |         |      |        |       |         |       |
| n-gram           | method           | b, a       | time(m) | JSD  | method | b = a | time(m) | JSD   |
| 4-gram           | TTUR             | 3e-4, 1e-4 | 1150    | 0.35 | orig   | 1e-4  | 1040    | 0.38  |
| 6-gram           | TTUR             | 3e-4, 1e-4 | 1120    | 0.74 | orig   | 1e-4  | 1070    | 0.77  |

## Conclusion

For learning GANs, we have introduced the two time-scale update rule (TTUR), which we have proved to converge to a stationary local Nash equilibrium. Then we described Adam stochastic optimization as a heavy ball with friction (HBF) dynamics, which shows that Adam converges and that Adam tends to find flat minima while avoiding small local minima. A second order differential equation describes the learning dynamics of Adam as an HBF system. Via this differential equation, the convergence of GANs trained with TTUR to a stationary local Nash equilibrium can be extended to Adam. Finally, to evaluate GANs, we introduced the 'Fréchet Inception Distance' (FID) which captures the similarity of generated images to real ones better than the Inception Score. In experiments we have compared GANs trained with TTUR to conventional GAN training with a one time-scale update rule on CelebA, CIFAR-10, SVHN, LSUN Bedrooms, and the One Billion Word Benchmark. TTUR outperforms conventional GAN training consistently in all experiments.

## Acknowledgment

This work was supported by NVIDIA Corporation, Bayer AG with Research Agreement 09/2017, Zalando SE with Research Agreement 01/2016, Audi.JKU Deep Learning Center, Audi Electronic Venture GmbH, IWT research grant IWT150865 (Exaptation), H2020 project grant 671555 (ExCAPE) and FWF grant P 28660-N31.

## References

The references are provided after Section A6.

## Appendix

## Contents

| A1              | Fréchet Inception Distance (FID) . . . . . . . . . . . . . . . . . . . . . . . . . . . .                                           | . . 11   |
|-----------------|------------------------------------------------------------------------------------------------------------------------------------|----------|
| A2              | Two Time-Scale Stochastic Approximation Algorithms . . . . . . . . . . . . . . . . .                                               | . 16     |
|                 | A2.1 Convergence of Two Time-Scale Stochastic Approximation Algorithms . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . | . 16     |
|                 | A2.1.1 Additive Noise . . . . . .                                                                                                  | . 16     |
|                 | A2.1.2 Linear Update, Additive Noise, and Markov Chain . . . . . . . . . . . .                                                     | . 18     |
|                 | A2.1.3 Additive Noise and Controlled Markov Processes                                                                              | . 20     |
|                 | . . . . . . . . . . . . . A2.2 Rate of Convergence of Two Time-Scale Stochastic Approximation Algorithms .                         | . 23     |
|                 | A2.2.1 Linear Update Rules . . . . . . . . . . . . . . . . . . . . . . . . . .                                                     | . 23     |
|                 | . . A2.2.2 Nonlinear Update Rules . . . . . . . . . . . . . . . . . . . . . . . . . .                                              | . 25     |
|                 | A2.3 Equal Time-Scale Stochastic Approximation Algorithms . . . . . . . . . . . . .                                                | . 27     |
|                 | A2.3.1 Equal Time-Scale for Saddle Point Iterates . . . . . . . . . . . . . . . .                                                  | . 27     |
|                 | A2.3.2 Equal Time Step for Actor-Critic Method . . . . . . . . . . . . . . .                                                       | . 28     |
| A3              | . . ADAM Optimization as Stochastic Heavy Ball with Friction . . . . . . . . . . . . . .                                           | . 30     |
| A4              | Experiments: Additional Information . . . . . . . . . . . . . . . . . . . . . . . . . .                                            | . 32     |
|                 | A4.1 WGAN-GP on Image Data. . . . . . . . . . . . . . . . . .                                                                      | . 32     |
|                 | . . . . . . . . . . . A4.2 WGAN-GP on the One Billion Word Benchmark. . . . . . . . . . . . . . . . .                              | . 33     |
|                 | A4.3 BEGAN . . . . . . . . . . . . . . . . . . . . . . .                                                                           | . 33     |
| A5              | . . . . . . . . . . . . . . . Discriminator vs. Generator Learning Rate . . . . . . . . .                                          | . 34     |
| A6              | . . . . . . . . . . . . . . Used Software, Datasets, Pretrained Models, and Implementations . . . . . . . . . . .                  | . 34     |
| List of Figures | . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                                                  | . 38     |

List of Tables

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

38

## A1 Fréchet Inception Distance (FID)

We improve the Inception score for comparing the results of GANs [53]. The Inception score has the disadvantage that it does not use the statistics of real world samples and compare it to the statistics of synthetic samples. Let p ( . ) be the distribution of model samples and p w ( . ) the distribution of the samples from real world. The equality p ( . ) = p w ( . ) holds except for a non-measurable set if and only if ∫ p ( . ) f ( x ) dx = ∫ p w ( . ) f ( x ) dx for a basis f ( . ) spanning the function space in which p ( . ) and p w ( . ) live. These equalities of expectations are used to describe distributions by moments or cumulants, where f ( x ) are polynomials of the data x . We replacing x by the coding layer of an Inception model in order to obtain vision-relevant features and consider polynomials of the coding unit functions. For practical reasons we only consider the first two polynomials, that is, the first two moments: mean and covariance. The Gaussian is the maximum entropy distribution for given mean and covariance, therefore we assume the coding units to follow a multidimensional Gaussian. The difference of two Gaussians is measured by the Fréchet distance [16] also known as Wasserstein-2 distance [58]. The Fréchet distance d ( ., . ) between the Gaussian with mean and covariance ( m , C ) obtained from p ( . ) and the Gaussian ( m w , C w ) obtained from p w ( . ) is called the 'Fréchet Inception Distance' (FID), which is given by [15]:

<!-- formula-not-decoded -->

Next we show that the FID is consistent with increasing disturbances and human judgment on the CelebA dataset. We computed the ( m w , C w ) on all CelebA images, while for computing ( m , C ) we used 50,000 randomly selected samples. We considered following disturbances of the image X :

1. Gaussian noise : We constructed a matrix N with Gaussian noise scaled to [0 , 255] . The noisy image is computed as (1 - α ) X + α N for α ∈ { 0 , 0 . 25 , 0 . 5 , 0 . 75 } . The larger α is, the larger is the noise added to the image, the larger is the disturbance of the image.
2. Gaussian blur : The image is convolved with a Gaussian kernel with standard deviation α ∈ { 0 , 1 , 2 , 4 } . The larger α is, the larger is the disturbance of the image, that is, the more the image is smoothed.
3. Black rectangles : To an image five black rectangles are are added at randomly chosen locations. The rectangles cover parts of the image. The size of the rectangles is α imagesize with α ∈ { 0 , 0 . 25 , 0 . 5 , 0 . 75 } . The larger α is, the larger is the disturbance of the image, that is, the more of the image is covered by black rectangles.
4. Swirl : Parts of the image are transformed as a spiral, that is, as a swirl (whirlpool effect). Consider the coordinate ( x, y ) in the noisy (swirled) image for which we want to find the color. Towards this end we need the reverse mapping for the swirl transformation which gives the location which is mapped to ( x, y ) . We first compute polar coordinates relative to a center ( x 0 , y 0 ) given by the angle θ = arctan(( y - y 0 ) / ( x - x 0 )) and the radius r = √ ( x - x 0 ) 2 +( y - y 0 ) 2 . We transform them according to θ ′ = θ + αe - 5 r/ (ln 2 ρ ) . Here α is a parameter for the amount of swirl and ρ indicates the swirl extent in pixels. The original coordinates, where the color for ( x, y ) can be found, are x org = x 0 + r cos( θ ′ ) and y org = y 0 + r sin( θ ′ ) . We set ( x 0 , y 0 ) to the center of the image and ρ = 25 . The disturbance level is given by the amount of swirl α ∈ { 0 , 1 , 2 , 4 } . The larger α is, the larger is the disturbance of the image via the amount of swirl.
5. Salt and pepper noise : Some pixels of the image are set to black or white, where black is chosen with 50% probability (same for white). Pixels are randomly chosen for being flipped to white or black, where the ratio of pixel flipped to white or black is given by the noise level α ∈ { 0 , 0 . 1 , 0 . 2 , 0 . 3 } . The larger α is, the larger is the noise added to the image via flipping pixels to white or black, the larger is the disturbance level.
6. ImageNet contamination : From each of the 1,000 ImageNet classes, 5 images are randomly chosen, which gives 5,000 ImageNet images. The images are ensured to be RGB and to have a minimal size of 256x256. A percentage of α ∈ { 0 , 0 . 25 , 0 . 5 , 0 . 75 } of the CelebA images has been replaced by ImageNet images. α = 0 means all images are from CelebA, α = 0 . 25 means that 75% of the images are from CelebA and 25% from ImageNet etc. The larger α is, the larger is the disturbance of the CelebA dataset by contaminating it by ImageNet images. The larger the disturbance level is, the more the dataset deviates from the reference real world dataset.

We compare the Inception Score [53] with the FID. The Inception Score with m samples and K classes is

<!-- formula-not-decoded -->

The FID is a distance, while the Inception Score is a score. To compare FID and Inception Score, we transform the Inception Score to a distance, which we call 'Inception Distance' (IND). This transformation to a distance is possible since the Inception Score has a maximal value. For zero probability p ( y k | X i ) = 0 , we set the value p ( y k | X i ) log p ( y k | X i ) p ( y k ) = 0 . We can bound the log -term by

<!-- formula-not-decoded -->

Using this bound, we obtain an upper bound on the Inception Score:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The upper bound is tight and achieved if m ⩽ K and every sample is from a different class and the sample is classified correctly with probability 1. The IND is computed 'IND = m - Inception Score', therefore the IND is zero for a perfect subset of the ImageNet with m&lt;K samples, where each sample stems from a different class. Therefore both distances should increase with increasing disturbance level. In Figure A8 we present the evaluation for each kind of disturbance. The larger the disturbance level is, the larger the FID and IND should be. In Figure A9, A10, A11, and A11 we show examples of images generated with DCGAN trained on CelebA with FIDs 500, 300, 133, 100, 45, 13, and FID 3 achieved with WGAN-GP on CelebA.

Figure A8: Left: FID and right: Inception Score are evaluated for fi rst row: Gaussian noise, second row: Gaussian blur, third row: implanted black rectangles, fourth row: swirled images, fi fth row. salt and pepper noise, and sixth row: the CelebA dataset contaminated by ImageNet images. Left is the smallest disturbance level of zero, which increases to the highest level at right. The FID captures the disturbance level very well by monotonically increasing whereas the Inception Score fluctuates, stays flat or even, in the worst case, decreases.

<!-- image -->

Figure A9: Samples generated from DCGAN trained on CelebA with different FIDs. Left: FID 500 and Right: FID 300.

<!-- image -->

Figure A10: Samples generated from DCGAN trained on CelebA with different FIDs. Left: FID 133 and Right: FID 100.

<!-- image -->

Figure A11: Samples generated from DCGAN trained on CelebA with different FIDs. Left: FID 45 and Right: FID 13.

<!-- image -->

Figure A12: Samples generated from WGAN-GP trained on CelebA with a FID of 3.

<!-- image -->

## A2 Two Time-Scale Stochastic Approximation Algorithms

Stochastic approximation algorithms are iterative procedures to find a root or a stationary point (minimum, maximum, saddle point) of a function when only noisy observations of its values or its derivatives are provided. Two time-scale stochastic approximation algorithms are two coupled iterations with different step sizes. For proving convergence of these interwoven iterates it is assumed that one step size is considerably smaller than the other. The slower iterate (the one with smaller step size) is assumed to be slow enough to allow the fast iterate converge while being perturbed by the the slower. The perturbations of the slow should be small enough to ensure convergence of the faster.

The iterates map at time step n ⩾ 0 the fast variable w n ∈ R k and the slow variable θ n ∈ R m to their new values:

<!-- formula-not-decoded -->

The iterates use

<!-- formula-not-decoded -->

- h ( . ) ∈ R m : mapping for the slow iterate Eq. (13),
- a ( n ) : step size for the slow iterate Eq. (13),
- g ( . ) ∈ R k : mapping for the fast iterate Eq. (14),
- b ( n ) : step size for the fast iterate Eq. (14),
- M ( w ) n : additive random Markov process for the fast iterate Eq. (14),
- M ( θ ) n : additive random Markov process for the slow iterate Eq. (13),
- Z ( θ ) n : random Markov process for the slow iterate Eq. (13),
- Z ( w ) n : random Markov process for the fast iterate Eq. (14).

## A2.1 Convergence of Two Time-Scale Stochastic Approximation Algorithms

## A2.1.1 Additive Noise

The first result is from Borkar 1997 [9] which was generalized in Konda and Borkar 1999 [31]. Borkar considered the iterates:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Assumptions. We make the following assumptions:

- (A1) Assumptions on the update functions: The functions h : R k + m ↦→ R m and g : R k + m ↦→ R k are Lipschitz.
- (A2) Assumptions on the learning rates:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- (A3) Assumptions on the noise: For the increasing σ -field

<!-- formula-not-decoded -->

the sequences of random variables ( M ( θ ) n , F n ) and ( M ( w ) n , F n ) satisfy

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- (A4) Assumption on the existence of a solution of the fast iterate: For each θ ∈ R m , the ODE

has a unique global asymptotically stable equilibrium λ ( θ ) such that λ : R m ↦→ R k is Lipschitz.

<!-- formula-not-decoded -->

- (A5) Assumption on the existence of a solution of the slow iterate: The ODE

has a unique global asymptotically stable equilibrium θ ∗ .

<!-- formula-not-decoded -->

- (A6) Assumption of bounded iterates:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Convergence Theorem The next theorem is from Borkar 1997 [9].

Theorem 3 (Borkar) . If the assumptions are satisfied, then the iterates Eq. (15) and Eq. (16) converge to ( θ ∗ , λ ( θ ∗ )) a.s.

## Comments

- (C1) According to Lemma 2 in [7] Assumption (A3) is fulfilled if { M ( θ ) n } is a martingale difference sequence w.r.t F n with

and { M ( w ) n } is a martingale difference sequence w.r.t F n with

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where B 1 and B 2 are positive deterministic constants.

- (C2) Assumption (A3) holds for mini-batch learning which is the most frequent case of stochastic gradient. The batch gradient is G n := ∇ θ ( 1 N ∑ N i =1 f ( x i , θ )) , 1 ⩽ i ⩽ N and the minibatch gradient for batch size s is h n := ∇ θ ( 1 s ∑ s i =1 f ( x u i , θ )) , 1 ⩽ u i ⩽ N , where the indexes u i are randomly and uniformly chosen. For the noise M ( θ ) n := h n - G n we have E[ M ( θ ) n ] = E[ h n ] - G n = G n - G n = 0 . Since the indexes are chosen without knowing past events, we have a martingale difference sequence. For bounded gradients we have bounded ‖ M ( θ ) n ‖ 2 .
- (C3) We address assumption (A4) with weight decay in two ways: (I) Weight decay avoids problems with a discriminator that is region-wise constant and, therefore, does not have a locally stable generator. If the generator is perfect, then the discriminator is 0.5 everywhere. For generator with mode collapse, (i) the discriminator is 1 in regions without generator examples, (ii) 0 in regions with generator examples only, (iii) is equal to the local ratio

of real world examples for regions with generator and real world examples. Since the discriminator is locally constant, the generator has gradient zero and cannot improve. Also the discriminator cannot improve, since it has minimal error given the current generator. However, without weight decay the Nash Equilibrium is not stable since the second order derivatives are zero, too. (II) Weight decay avoids that the generator is driven to infinity with unbounded weights. For example a linear discriminator can supply a gradient for the generator outside each bounded region.

- (C4) The main result used in the proof of the theorem relies on work on perturbations of ODEs according to Hirsch 1989 [24].
- (C5) Konda and Borkar 1999 [31] generalized the convergence proof to distributed asynchronous update rules.
- (C6) Tadi´ c relaxed the assumptions for showing convergence [54]. In particular the noise assumptions (Assumptions A2 in [54]) do not have to be martingale difference sequences and are more general than in [9]. In another result the assumption of bounded iterates is not necessary if other assumptions are ensured [54]. Finally, Tadi´ c considers the case of non-additive noise [54]. Tadi´ c does not provide proofs for his results. We were not able to find such proofs even in other publications of Tadi´ c.

## A2.1.2 Linear Update, Additive Noise, and Markov Chain

In contrast to the previous subsection, we assume that an additional Markov chain influences the iterates [30, 32]. The Markov chain allows applications in reinforcement learning, in particular in actor-critic setting where the Markov chain is used to model the environment. The slow iterate is the actor update while the fast iterate is the critic update. For reinforcement learning both the actor and the critic observe the environment which is driven by the actor actions. The environment observations are assumed to be a Markov chain. The Markov chain can include eligibility traces which are modeled as explicit states in order to keep the Markov assumption.

The Markov chain is the sequence of observations of the environment which progresses via transition probabilities. The transitions are not affected by the critic but by the actor.

Konda et al. considered the iterates [30, 32]:

<!-- formula-not-decoded -->

H n is a random process that drives the changes of θ n . We assume that H n is a slow enough process. We have a linear update rule for the fast iterate using the vector function g ( . ) ∈ R k and the matrix function G ( . ) ∈ R k × k .

<!-- formula-not-decoded -->

Assumptions. We make the following assumptions:

- (A1) Assumptions on the Markov process, that is, the transition kernel: The stochastic process Z ( w ) n takes values in a Polish (complete, separable, metric) space Z with the Borel σ -field

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For every measurable set A ⊂ Z and the parametrized transition kernel P( . ; θ n ) we have:

We define for every measurable function f

- (A2) Assumptions on the learning rates:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for some d &gt; 0 .

- (A3) Assumptions on the noise: The sequence M ( w ) n is a k × k -matrix valued F n -martingale difference with bounded moments:

<!-- formula-not-decoded -->

We assume slowly changing θ , therefore the random process H n satisfies

<!-- formula-not-decoded -->

- (A4) Assumption on the existence of a solution of the fast iterate: We assume the existence of a solution to the Poisson equation for the fast iterate. For each θ ∈ R m , there exist functions ¯ g ( θ ) ∈ R k , ¯ G ( θ ) ∈ R k × k , ˆ g ( z ; θ ) : Z → R k , and ˆ G ( z ; θ ) : Z → R k × k that satisfy the Poisson equations:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- (A5) Assumptions on the update functions and solutions to the Poisson equation:
- (a) Boundedness of solutions: For some constant C and for all θ :

<!-- formula-not-decoded -->

- (b) Boundedness in expectation: All moments are bounded. For any d &gt; 0 , there exists C d &gt; 0 such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- (c) Lipschitz continuity of solutions: For some constant C &gt; 0 and for all θ , ¯ θ ∈ R m :
- ∥ ∥ (d) Lipschitz continuity in expectation: There exists a positive measurable function C ( . ) on Z such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Function C ( . ) gives the Lipschitz constant for every z :

- ∥ ∥ (e) Uniform positive definiteness: There exists some α &gt; 0 such that for all w ∈ R k and θ ∈ R m :

<!-- formula-not-decoded -->

Convergence Theorem. We report Theorem 3.2 (see also Theorem 7 in [32]) and Theorem 3.13 from [30]:

Theorem 4 (Konda &amp; Tsitsiklis) . If the assumptions are satisfied, then for the iterates Eq. (26) and Eq. (27) holds:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## Comments.

- (C1) The proofs only use the boundedness of the moments of H n [30, 32], therefore H n may depend on w n . In his PhD thesis [30], Vijaymohan Konda used this framework for the actor-critic learning, where H n drives the updates of the actor parameters θ n . However, the actor updates are based on the current parameters w n of the critic.
- (C2) The random process Z ( w ) n can affect H n as long as boundedness is ensured.
- (C3) Nonlinear update rule. g ( Z ( w ) n ; θ n ) + G ( Z ( w ) n ; θ n ) w n can be viewed as a linear approximation of a nonlinear update rule. The nonlinear case has been considered in [30] where additional approximation errors due to linearization were addressed. These errors are treated in the given framework [30].

## A2.1.3 Additive Noise and Controlled Markov Processes

The most general iterates use nonlinear update functions g and h , have additive noise, and have controlled Markov processes [28].

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Required Definitions. Marchaud Map : A set-valued map h : R l →{ subsets of R k } is called a Marchaud map if it satisfies the following properties:

- (i) For each θ ∈ R l , h ( θ ) is convex and compact.
- (iii) h is an upper-semicontinuous map. We say that h is upper-semicontinuous, if given sequences { θ n } n ≥ 1 (in R l ) and { y n } n ≥ 1 (in R k ) with θ n → θ , y n → y and y n ∈ h ( θ n ) , n ≥ 1 , y ∈ h ( θ ) . In other words, the graph of h , { ( x , y ) : y ∈ h ( x ) , x ∈ R l } , is closed in R l × R k .
- (ii) (point-wise boundedness) For each θ ∈ R l , sup w ∈ h ( θ ) ‖ w ‖ &lt; K (1 + ‖ θ ‖ ) for some K &gt; 0 .

If the set-valued map H : R m →{ subsets of R m } is Marchaud, then the differential inclusion (DI) given by

<!-- formula-not-decoded -->

Invariant Set : M ⊆ R m is invariant if for every θ ∈ M there exists a trajectory, Θ , entirely in M with Θ (0) = θ . In other words, Θ ∈ Σ with Θ ( t ) ∈ M , for all t ≥ 0 .

is guaranteed to have at least one solution that is absolutely continuous. If Θ is an absolutely continuous map satisfying Eq. (52) then we say that Θ ∈ Σ .

Internally Chain Transitive Set : M ⊂ R m is said to be internally chain transitive if M is compact and for every θ , y ∈ M , ϵ &gt; 0 and T &gt; 0 we have the following: There exist Φ 1 , . . . , Φ n that are n solutions to the differential inclusion ˙ θ ( t ) ∈ h ( θ ( t )) , a sequence θ 1 (= θ ) , . . . , θ n +1 (= y ) ⊂ M and n real numbers t 1 , t 2 , . . . , t n greater than T such that: Φ i t i ( θ i ) ∈ N ϵ ( θ i +1 ) where N ϵ ( θ ) is the open ϵ -neighborhood of θ and Φ i [0 ,t i ] ( θ i ) ⊂ M for 1 ≤ i ≤ n . The sequence ( θ 1 (= θ ) , . . . , θ n +1 (= y )) is called an ( ϵ, T ) chain in M from θ to y .

Assumptions. We make the following assumptions [28]:

- (A1) Assumptions on the controlled Markov processes: The controlled Markov process { Z ( w ) n } takes values in a compact metric space S ( w ) . The controlled Markov process { Z ( θ ) n } takes values in a compact metric space S ( θ ) . Both processes are controlled by the iterate sequences { θ n } and { w n } . Furthermore { Z ( w ) n } is additionally controlled by a random process { A ( w ) n } taking values in a compact metric space U ( w ) and { Z ( θ ) n } is additionally controlled by a random process { A ( θ ) n } taking values in a compact metric space U ( θ ) . The { Z ( θ ) n } dynamics is

<!-- formula-not-decoded -->

for B ( θ ) Borel in S ( θ ) . The { Z ( w ) n } dynamics is

<!-- formula-not-decoded -->

for B ( w ) Borel in S ( w ) .

- (A2) Assumptions on the update functions: h : R m + k × S ( θ ) → R m is jointly continuous as well as Lipschitz in its first two arguments uniformly w.r.t. the third. The latter condition means that

<!-- formula-not-decoded -->

Note that the Lipschitz constant L ( θ ) does not depend on z ( θ ) .

g : R k + m × S ( w ) → R k is jointly continuous as well as Lipschitz in its first two arguments uniformly w.r.t. the third. The latter condition means that

<!-- formula-not-decoded -->

Note that the Lipschitz constant L ( w ) does not depend on z ( w ) .

- (A3) Assumptions on the additive noise: { M ( θ ) n } and { M ( w ) n } are martingale difference sequence with second moments bounded by K (1 + ‖ θ n ‖ 2 + ‖ w n ‖ 2 ) . More precisely, { M ( θ ) n } is a martingale difference sequence w.r.t. increasing σ -fields

satisfying

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for n ⩾ 0 and a given constant K &gt; 0 .

{ M ( w ) n } is a martingale difference sequence w.r.t. increasing σ -fields

satisfying

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for n ⩾ 0 and a given constant K &gt; 0 .

- (A4) Assumptions on the learning rates:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Furthermore, a ( n ) , b ( n ) , n ⩾ 0 are non-increasing.

- (A5) Assumptions on the controlled Markov processes, that is, the transition kernels: The stateaction map

<!-- formula-not-decoded -->

and the state-action map

<!-- formula-not-decoded -->

are continuous.

- (A6) Assumptions on the existence of a solution:

We consider occupation measures which give for the controlled Markov process the probability or density to observe a particular state-action pair from S × U for given θ and a given control policy π . We denote by D ( w ) ( θ , w ) the set of all ergodic occupation measures for the prescribed θ and w on state-action space S ( w ) × U ( θ ) for the controlled Markov process Z ( w ) with policy π ( w ) . Analogously we denote, by D ( θ ) ( θ , w ) the set of all ergodic occupation measures for the prescribed θ and w on state-action space S ( θ ) × U ( θ ) for the controlled Markov process Z ( θ ) with policy π ( θ ) . Define

<!-- formula-not-decoded -->

for ν a measure on S ( w ) × U ( w ) and the Marchaud map

<!-- formula-not-decoded -->

We assume that the set D ( w ) ( θ , w ) is singleton, that is, ˆ g ( θ , w ) contains a single function and we use the same notation for the set and its single element. If the set is not a singleton, the assumption of a solution can be expressed by the differential inclusion ˙ w ( t ) ∈ ˆ g ( θ , w ( t )) [28].

∀ θ ∈ R m , the ODE

- (A7) Assumption of bounded iterates:

<!-- formula-not-decoded -->

has an asymptotically stable equilibrium λ ( θ ) with domain of attraction G θ where λ : R m → R k is a Lipschitz map with constant K . Moreover, the function V : G → [0 , ∞ ) is continuously differentiable where V ( θ , . ) is the Lyapunov function for λ ( θ ) and G = { ( θ , w ) : w ∈ G θ , θ ∈ R m } . This extra condition is needed so that the set { ( θ , λ ( θ )) : θ ∈ R m } becomes an asymptotically stable set of the coupled ODE

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Convergence Theorem. The following theorem is from Karmakar &amp; Bhatnagar [28]:

Theorem 5 (Karmakar &amp; Bhatnagar) . Under above assumptions if for all θ ∈ R m , with probability 1, { w n } belongs to a compact subset Q θ (depending on the sample point) of G θ 'eventually', then

where A 0 = ∩ t ⩾ 0 { ¯ θ ( s ) : s ⩾ t } which is almost everywhere an internally chain transitive set of the differential inclusion

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ˆ h ( θ ) = { ˜ h ( θ , λ ( θ ) , ν ) : ν ∈ D ( w ) ( θ , λ ( θ )) } .

## Comments.

- (C1) This framework allows to show convergence for gradient descent methods beyond stochastic gradient like for the ADAM procedure where current learning parameters are memorized and updated. The random processes Z ( w ) and Z ( θ ) may track the current learning status for the fast and slow iterate, respectively.
- (C2) Stochastic regularization like dropout is covered via the random processes A ( w ) and A ( θ ) .

## A2.2 Rate of Convergence of Two Time-Scale Stochastic Approximation Algorithms

## A2.2.1 Linear Update Rules

First we consider linear iterates according to the PhD thesis of Konda [30] and Konda &amp; Tsitsiklis [33].

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Assumptions. We make the following assumptions:

- (A1) The random variables ( M ( θ ) n , M ( w ) n ) , n = 0 , 1 , . . . , are independent of w 0 , θ 0 and of each other. The have zero mean: E[ M ( θ ) n ] = 0 and E[ M ( w ) n ] = 0 . The covariance is

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- (A2) The learning rates are deterministic, positive, nondecreasing and satisfy with ϵ ⩽ 0 :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We often consider the case ϵ = 0 .

- (A3) Convergence of the iterates: We define

<!-- formula-not-decoded -->

A matrix is Hurwitz if the real part of each eigenvalue is strictly negative. We assume that the matrices - A 22 and - ∆ are Hurwitz.

(A4) Convergence rate remains simple:

- (a) There exists a constant ¯ a ⩽ 0 such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

-

<!-- formula-not-decoded -->

- (b) If ϵ = 0 , then
- (c) The matrix

is Hurwitz.

Rate of Convergence Theorem. The next theorem is taken from Konda [30] and Konda &amp; Tsitsiklis [33].

Let θ ∗ ∈ R m and w ∗ ∈ R k be the unique solution to the system of linear equations

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Theorem 6 (Konda &amp; Tsitsiklis) . Under above assumptions and when the constant ϵ is sufficiently small, the limit matrices

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

exist. Furthermore, the matrix

<!-- formula-not-decoded -->

is the unique solution to the following system of equations

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For each n , let

Finally,

<!-- formula-not-decoded -->

The next theorems shows that the asymptotic covariance matrix of a ( n ) - 1 / 2 θ n is the same as that of a ( n ) - 1 / 2 ¯ θ n , where ¯ θ n evolves according to the single time-scale stochastic iteration:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The next theorem combines Theorem 2.8 of Konda &amp; Tsitsiklis and Theorem 4.1 of Konda &amp; Tsitsiklis:

Theorem 7 (Konda &amp; Tsitsiklis 2nd) . Under above assumptions

<!-- formula-not-decoded -->

If the assumptions hold with ϵ = 0 , then a ( n ) - 1 / 2 ˆ θ n converges in distribution to N ( 0 , Σ (0) 11 ) .

## Comments.

- (C1) In his PhD thesis [30] Konda extended the analysis to the nonlinear case. Konda makes a linearization of the nonlinear function h and g with

<!-- formula-not-decoded -->

There are additional errors due to linearization which have to be considered. However, only a sketch of a proof is provided but not a complete proof.

- (C2) Theorem 4.1 of Konda &amp; Tsitsiklis is important to generalize to the nonlinear case.
- (C3) The convergence rate is governed by A 22 for the fast and ∆ for the slow iterate. ∆ in turn is affected by the interaction effects captured by A 21 and A 12 together with the inverse of A 22 .

## A2.2.2 Nonlinear Update Rules

The rate of convergence for nonlinear update rules according to Mokkadem &amp; Pelletier is considered [44].

The iterates are

<!-- formula-not-decoded -->

with the increasing σ -fields

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The terms Z ( θ ) n and Z ( w ) n can be used to address the error through linearization, that is, the difference of the nonlinear functions to their linear approximation.

Assumptions. We make the following assumptions:

- (A1) Convergence is ensured:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- (A2) Linear approximation and Hurwitz:

There exists a neighborhood U of ( θ ∗ , w ∗ ) such that, for all ( θ , w ) ∈ U

We define

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

A matrix is Hurwitz if the real part of each eigenvalue is strictly negative. We assume that the matrices A 22 and ∆ are Hurwitz.

- (A3) Assumptions on the learning rates:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where a 0 &gt; 0 and b 0 &gt; 0 and 1 / 2 &lt; β &lt; α ⩽ 1 . If α = 1 , then a 0 &gt; 1 / (2 e min ) with e min as the absolute value of the largest eigenvalue of ∆ (the eigenvalue closest to 0).

(A4) Assumptions on the noise and error:

- (a) martingale difference sequences:

<!-- formula-not-decoded -->

- (b) existing second moments:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- (c) bounded moments:

<!-- formula-not-decoded -->

There exist l &gt; 2 /β such that

- (d) bounded error:

with

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Rate of Convergence Theorem. We report a theorem and a proposition from Mokkadem &amp; Pelletier [44]. However, first we have to define the covariance matrices Σ θ and Σ w which govern the rate of convergence.

First we define

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We now define the asymptotic covariance matrices Σ θ and Σ w :

<!-- formula-not-decoded -->

Σ θ and Σ w are solutions of the Lyapunov equations:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Theorem 8 (Mokkadem &amp; Pelletier: Joint weak convergence) . Under above assumptions:

Theorem 9 (Mokkadem &amp; Pelletier: Strong convergence) . Under above assumptions:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## Comments.

- (C1) Besides the learning steps a ( n ) and b ( n ) , the convergence rate is governed by A 22 for the fast and ∆ for the slow iterate. ∆ in turn is affected by interaction effects which are captured by A 21 and A 12 together with the inverse of A 22 .

## A2.3 Equal Time-Scale Stochastic Approximation Algorithms

In this subsection we consider the case when the learning rates have equal time-scale.

## A2.3.1 Equal Time-Scale for Saddle Point Iterates

If equal time-scales assumed then the iterates revisit infinite often an environment of the solution [61]. In Zhang 2007, the functions of the iterates are the derivatives of a Lagrangian with respect to the dual and primal variables [61]. The iterates are

<!-- formula-not-decoded -->

with the increasing σ -fields

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The terms Z ( θ ) n and Z ( w ) n subsum biased estimation errors.

## Assumptions. We make the following assumptions:

- (A1) Assumptions on update function: h and g are continuous, differentiable, and bounded. The Jacobians

<!-- formula-not-decoded -->

are Hurwitz. A matrix is Hurwitz if the real part of each eigenvalue is strictly negative. This assumptions corresponds to the assumption in [61] that the Lagrangian is concave in w and convex in θ .

- (A2) Assumptions on noise:

{ M ( θ ) n } and { M ( w ) n } are a martingale difference sequences w.r.t. the increasing σ -fields F n . Furthermore they are mutually independent.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Bounded second moment:

- (A3) Assumptions on the learning rate:

<!-- formula-not-decoded -->

- (A4) Assumption on the biased error:

Boundedness:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Theorem. Define the 'contraction region' A η as follows:

<!-- formula-not-decoded -->

Theorem 10 (Zhang) . Under above assumptions the iterates return to A η infinitely often with probability one (a.s.).

## Comments.

- (C1) The proof of the theorem in [61] does not use the saddle point condition and not the fact that the functions of the iterates are derivatives of the same function.
- (C2) For the unbiased case, Zhang showed in Theorem 3.1 of [61] that the iterates converge. However, he used the saddle point condition of the Lagrangian. He considered iterates with functions that are the derivatives of a Lagrangian with respect to the dual and primal variables [61].

## A2.3.2 Equal Time Step for Actor-Critic Method

If equal time-scales assumed then the iterates revisit infinite often an environment of the solution of DiCastro &amp; Meir [14]. The iterates of DiCastro &amp; Meir are derived for actor-critic learning.

To present the actor-critic update iterates, we have to define some functions and terms. µ ( u | x , θ ) is the policy function parametrized by θ ∈ R m with observations x ∈ X and actions u ∈ U . A Markov chain given by P( y | x , u ) gives the next observation y using the observation x and the action u . In each state x the agent receives a reward r ( x ) .

The average reward per stage is for the recurrent state x ∗ :

<!-- formula-not-decoded -->

The estimate of ˜ η is denoted by η .

The differential value function is

<!-- formula-not-decoded -->

The temporal difference is

<!-- formula-not-decoded -->

The estimate of ˜ d is denoted by d .

The likelihood ratio derivative Ψ ∈ R m is

<!-- formula-not-decoded -->

The value function ˜ h is approximated by

<!-- formula-not-decoded -->

where φ ( x ) ∈ R k . We define Φ ∈ R |X|× k

and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For TD( λ ) we have an eligibility trace:

<!-- formula-not-decoded -->

We define the approximation error with optimal parameter w ∗ ( θ ) :

<!-- formula-not-decoded -->

where π ( θ ) is an projection operator into the span of Φ w . We bound this error by

<!-- formula-not-decoded -->

We denoted by ˜ η , ˜ d , and ˜ h the exact functions and used for their approximation η , d , and h , respectively. We have learning rate adjustments Γ η and Γ w for the critic.

The update rules are:

Critic:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Actor:

<!-- formula-not-decoded -->

Assumptions. We make the following assumptions:

- (A1) Assumption on rewards:

The rewards { r ( x ) } x ∈X are uniformly bounded by a finite constant B r .

- (A2) Assumption on the Markov chain:

Each Markov chain for each θ is aperiodic, recurrent, and irreducible.

- (A3) Assumptions on the policy function:

The conditional probability function µ ( u | x , θ ) is twice differentiable. Moreover, there exist positive constants, B µ 1 and B µ 2 , such that for all x ∈ X , u ∈ U , θ ∈ R m and 1 ⩽ l 1 , l 2 ⩽ m we have

- (A4) Assumption on the likelihood ratio derivative:

For all x ∈ X , u ∈ U , and θ ∈ R m , there exists a positive constant B Ψ , such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ‖ . ‖ 2 is the Euclidean L 2 norm.

- (A5) Assumptions on the approximation space given by Φ :

The columns of the matrix Φ are independent, that is, the form a basis of dimension k . The norms of the columns vectors of the matrix Φ are bounded above by 1 , that is, ‖ φ l ‖ 2 ⩽ 1 for 1 ⩽ l ⩽ k .

- (A6) Assumptions on the learning rate:

<!-- formula-not-decoded -->

Theorem. The algorithm converged if ∇ θ ˜ η ( θ ) = 0 , since the actor reached a stationary point where the updates are zero. We assume that ‖∇ θ ˜ η ( θ ) ‖ hints at how close we are to the convergence point.

The next theorem from DiCastro &amp; Meir [14] implies that the trajectory visits a neighborhood of a local maximum infinitely often. Although it may leave the local vicinity of the maximum, it is guaranteed to return to it infinitely often.

Theorem 11 (DiCastro &amp; Meir) . Define

<!-- formula-not-decoded -->

where B ∆ td 1 , B ∆ td 2 , and B ∆ td 3 are finite constants depending on the Markov decision process and the agent parameters.

Under above assumptions

The trajectory visits a neighborhood of a local maximum infinitely often.

## Comments.

- (C1) The larger the critic learning rates Γ w and Γ η are, the smaller is the region around the local maximum.
- (C2) The results are in agreement with those of Zhang 2007 [61].
- (C3) Even if the results are derived for a special actor-critic setting, they carry over to a more general setting of the iterates.

## A3 ADAMOptimization as Stochastic Heavy Ball with Friction

The Nesterov Accelerated Gradient Descent (NAGD) [47] has raised considerable interest due to its numerical simplicity and its low complexity. Previous to NAGD and its derived methods there was Polyak's Heavy Ball method [49]. The idea of the Heavy Ball is a ball that evolves over the graph of a function f with damping (due to friction) and acceleration. Therefore, this second-order dynamical system can be described by the ODE for the Heavy Ball with Friction (HBF) [17]:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where a ( n ) is the damping coefficient with a ( n ) = a n β for β ∈ (0 , 1] . This ODE is equivalent to the integro-differential equation

where k and h are two memory functions related to a ( t ) . For polynomially memoried HBF we have k ( t ) = t α +1 and h ( t ) = ( α +1) t α for some positive α , and for exponentially memoried HBF we have k ( t ) = λ exp( λ t ) and h ( t ) = exp( λ t ) . For the sum of the learning rates, we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where γ = 0 . 5772156649 is the Euler-Mascheroni constant.

Gadat et al. derived a discrete and stochastic version of the HBF [17]:

where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This recursion can be rewritten as

<!-- formula-not-decoded -->

m n +1 = ( 1 - a ( n +1) r ( n ) ) m n + a ( n +1) r ( n ) ( ∇ f ( θ n ) + M n +1 ) . (167) The recursion Eq. (166) is the first moment update of ADAM [29].

For the term r ( n ) a ( n ) we obtain for the polynomial memory the approximations

<!-- formula-not-decoded -->

Gadat et al. showed that the recursion Eq. (164) converges for functions with at most quadratic grow [17]. The authors mention that convergence can be proofed for functions f that are L -smooth, that is, the gradient is L -Lipschitz.

Kingma et al. [29] state in Theorem 4.1 convergence of ADAM while assuming that β 1 , the first moment running average coefficient, decays exponentially. Furthermore they assume that β 2 1 √ β 2 &lt; 1 and the learning rate α t decays with α t = α √ t .

ADAMdivides m n of the recursion Eq. (166) by the bias-corrected second raw moment estimate. Since the bias-corrected second raw moment estimate changes slowly, we consider it as an error.

<!-- formula-not-decoded -->

ADAM assumes the second moment E [ g 2 ] to be stationary with its approximation v n :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore

<!-- formula-not-decoded -->

We are interested in the difference of actual stochastic v n to the true stationary v :

<!-- formula-not-decoded -->

For a stationary second moment of m n and β 2 = 1 - αa ( n + 1) r ( n ) , we have ∆ v n ∝ a ( n + 1) r ( n ) . We use a linear approximation to ADAM's second moment normalization 1 / √ v +∆ v n ≈ 1 / √ v - 1 / (2 v √ v )∆ v n +O(∆ 2 v n ) . If we set M ( v ) n +1 = - ( m n ∆ v n ) / (2 v √ va ( n +1) r ( n )) , then m n / √ v n ≈ m n / √ v + a ( n + 1) r ( n ) M ( v ) n +1 and E [ M ( v ) n +1 ] = 0 , since E [ g 2 l - v ] = 0 . For a stationary second moment of m n , { M ( v ) n } is a martingale difference sequence with a bounded second moment. Therefore { M ( v ) n +1 } can be subsumed into { M n +1 } in update rules Eq. (166). The factor 1 / √ v can be incorporated into a ( n +1) and r ( n ) .

## A4 Experiments: Additional Information

## A4.1 WGAN-GP on Image Data.

Table A2: The performance of WGAN-GP trained with the original procedure and with TTUR on CIFAR-10 and LSUN Bedrooms. We compare the performance with respect to the FID at the optimal number of iterations during training and wall-clock time in minutes.

| dataset   | method   | b, a       | iter   |   time(m) |   FID | method   |   b = a | iter   |   time(m) |   FID |
|-----------|----------|------------|--------|-----------|-------|----------|---------|--------|-----------|-------|
| CIFAR-10  | TTUR     | 3e-4, 1e-4 | 168k   |       700 |  24.8 | orig     |    1e-4 | 53k    |       800 |  29.3 |
| LSUN      | TTUR     | 3e-4, 1e-4 | 80k    |      1900 |   9.5 | orig     |    1e-4 | 23k    |      2010 |  20.5 |

## A4.2 WGAN-GP on the One Billion Word Benchmark.

Table A3: Samples generated by WGAN-GP trained on fhe One Billion Word benchmark with TTUR (left) the original method (right).

Dry Hall Sitning tven the concer There are court phinchs hasffort He scores a supponied foutver il Bartfol reportings ane the depor Seu hid , it 's watter 's remold Later fasted the store the inste Indiwezal deducated belenseous K Starfers on Rbama 's all is lead Inverdick oper , caldawho 's non She said , five by theically rec RichI , Learly said remain .'''' Reforded live for they were like The plane was git finally fuels The skip lifely will neek by the SEW McHardy Berfect was luadingu But I pol rated Franclezt is the No say that tent Franstal at Bra Caulh Paphionars tven got corfle Resumaly , braaky facting he at On toipe also houd , aid of sole When Barrysels commono toprel to The Moster suprr tent Elay diccu The new vebators are demases to Many 's lore wockerssaow 2 2 ) A Andly , has le wordd Uold steali But be the firmoters is no 200 s Jermueciored a noval wan 't mar Onles that his boud-park , the g ISLUN , The crather wilh a them Fow 22o2 surgeedeto , theirestra Make Sebages of intarmamates , a Gullla " has cautaria Thoug ly t

Table A4: The performance of WGAN-GP trained with the original procedure and with TTUR on the One Billion Word Benchmark. We compare the performance with respect to the JSD at the optimal number of iterations and wall-clock time in minutes during training. WGAN-GP trained with TTUR exhibits consistently a better FID.

| n-gram   | method   | b, a       | iter   |   time(m) |   JSD | method   |   b = a | iter   |   time(m) |   JSD |
|----------|----------|------------|--------|-----------|-------|----------|---------|--------|-----------|-------|
| 4-gram   | TTUR     | 3e-4, 1e-4 | 98k    |      1150 |  0.35 | orig     |    1e-4 | 33k    |      1040 |  0.38 |
| 6-gram   | TTUR     | 3e-4, 1e-4 | 100k   |      1120 |  0.74 | orig     |    1e-4 | 32k    |      1070 |  0.77 |

## A4.3 BEGAN

The Boundary Equilibrium GAN (BEGAN) [6] maintains an equilibrium between the discriminator and generator loss (cf. Section 3.3 in [6])

<!-- formula-not-decoded -->

which, in turn, also leads to a fixed relation between the two gradients, therefore, a two time-scale update is not ensured by solely adjusting the learning rates. Indeed, for stable learning rates, we see no differences in the learning progress between orig and TTUR as depicted in Figure A13.

Figure A13: Mean, maximum and minimum FID over eight runs for BEGAN training on CelebA and LSUN Bedrooms. TTUR learning rates are given as pairs ( b, a ) of discriminator learning rate b and generator learning rate a : 'TTUR b a '. Left: CelebA, starting at mini-batch 10k for better visualisation. Right: LSUN Bedrooms. Orig and TTUR behave similar. For BEGAN we cannot ensure TTUR by adjusting learning rates.

<!-- image -->

## A5 Discriminator vs. Generator Learning Rate

The convergence proof for learning GANs with TTUR assumes that the generator learning rate will eventually become small enough to ensure convergence of the discriminator learning. At some time point, the perturbations of the discriminator updates by updates of the generator parameters are sufficient small to assure that the discriminator converges. Crucial for discriminator convergence is the magnitude of the perturbations which the generator induces into the discriminator updates. These perturbations are not only determined by the generator learning rate but also by its loss function, current value of the loss function, optimization method, size of the error signals that reach the generator (vanishing or exploding gradient), complexity of generator's learning task, architecture of the generator, regularization, and others. Consequently, the size of generator learning rate does not solely determine how large the perturbations of the discriminator updates are but serve to modulate them. Thus, the generator learning rate may be much larger than the discriminator learning rate without inducing large perturbation into the discriminator learning.

Even the learning dynamics of the generator is different from the learning dynamics of the discriminator, though they both have the same learning rate. Figure A14 shows the loss of the generator and the discriminator for an experiment with DCGAN on CelebA, where the learning rate was 0.0005 for both the discriminator and the generator. However, the discriminator loss is decreasing while the generator loss is increasing. This example shows that the learning rate neither determines the perturbations nor the progress in learning for two coupled update rules. The choice of the learning rate for the generator should be independent from choice for the discriminator. Also the search ranges of discriminator and generator learning rates should be independent from each other, but adjusted to the corresponding architecture, task, etc.

<!-- image -->

mini-batch x 1k

Figure A14: The respective losses of the discriminator and the generator show the different learning dynamics of the two networks.

## A6 Used Software, Datasets, Pretrained Models, and Implementations

We used the following datasets to evaluate GANs: The Large-scale CelebFaces Attributes (CelebA) dataset, aligned and cropped [41], the training dataset of the bedrooms category of the large scale image database (LSUN) [60], the CIFAR-10 training dataset [34], the Street View House Numbers training dataset (SVHN) [48], and the One Billion Word Benchmark [12].

All experiments rely on the respective reference implementations for the corresponding GAN model. The software framework for our experiments was Tensorflow 1.3 [1, 2] and Python 3.6. We used following software, datasets and pretrained models:

- BEGAN in Tensorflow, https://github.com/carpedm20/BEGAN-tensorflow , Fixed random seeds removed. Accessed: 2017-05-30

- DCGAN in Tensorflow, https://github.com/carpedm20/DCGAN-tensorflow , Fixed random seeds removed. Accessed: 2017-04-03
- Improved Training of Wasserstein GANs, image model, https://github.com/igul222/ improved\_wgan\_training/blob/master/gan\_64x64.py , Accessed: 2017-06-12
- Improved Training of Wasserstein GANs, language model, https://github.com/ igul222/improved\_wgan\_training/blob/master/gan\_language.py , Accessed: 2017-06-12
- [Inception-v3 pretrained, http://download.tensorflow.org/models/image/ imagenet/inception-2015-12-05.tgz , Accessed: 2017-05-02](http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz)

Implementations are available at

- [https://github.com/bioinf-jku/TTUR](https://github.com/bioinf-jku/TTUR)

## References

- [1] M. Abadi, A. Agarwal, P. Barham, E. Brevdo, Z. Chen, C. Citro, G. S. Corrado, A. Davis, J. Dean, M. Devin, S. Ghemawat, I. J. Goodfellow, A. Harp, G. Irving, M. Isard, Y. Jia, R. Józefowicz, L. Kaiser, M. Kudlur, J. Levenberg, D. Mané, R. Monga, S. Moore, D. G. Murray, C. Olah, M. Schuster, J. Shlens, B. Steiner, I. Sutskever, K. Talwar, P. A. Tucker, V. Vanhoucke, V. Vasudevan, F. B. Viégas, O. Vinyals, P. Warden, M. Wattenberg, M. Wicke, Y. Yu, and X. Zheng. Tensorflow: Large-scale machine learning on heterogeneous distributed systems. arXiv e-prints , arXiv:1603.04467, 2016.
- [2] M. Abadi, P. Barham, J. Chen, Z. Chen, A. Davis, J. Dean, M. Devin, S. Ghemawat, G. Irving, M. Isard, M. Kudlur, J. Levenberg, R. Monga, S. Moore, D. G. Murray, B. Steiner, P. Tucker, V. Vasudevan, P. Warden, M. Wicke, Y. Yu, and X. Zheng. Tensorflow: A system for largescale machine learning. In 12th USENIX Symposium on Operating Systems Design and Implementation (OSDI 16) , pages 265-283, 2016.
- [3] M. Arjovsky, S. Chintala, and L. Bottou. Wasserstein GAN. arXiv e-prints , arXiv:1701.07875, 2017.
- [4] S. Arora, R. Ge, Y. Liang, T. Ma, and Y. Zhang. Generalization and equilibrium in generative adversarial nets (GANs). In D. Precup and Y. W. Teh, editors, Proceedings of the 34th International Conference on Machine Learning , Proceedings of Machine Learning Research, vol. 70, pages 224-232, 2017.
- [5] H. Attouch, X. Goudou, and P. Redont. The heavy ball with friction method, I. the continuous dynamical system: Global exploration of the local minima of a real-valued function by asymptotic analysis of a dissipative dynamical system. Communications in Contemporary Mathematics , 2(1):1-34, 2000.
- [6] D. Berthelot, T. Schumm, and L. Metz. BEGAN: Boundary equilibrium generative adversarial networks. arXiv e-prints , arXiv:1703.10717, 2017.
- [7] D. P. Bertsekas and J. N. Tsitsiklis. Gradient convergence in gradient methods with errors. SIAM Journal on Optimization , 10(3):627-642, 2000.
- [8] S. Bhatnagar, H. L. Prasad, and L. A. Prashanth. Stochastic Recursive Algorithms for Optimization . Lecture Notes in Control and Information Sciences. Springer-Verlag London, 2013.
- [9] V. S. Borkar. Stochastic approximation with two time scales. Systems &amp; Control Letters , 29(5):291-294, 1997.
- [10] V. S. Borkar and S. P. Meyn. The O.D.E. method for convergence of stochastic approximation and reinforcement learning. SIAM Journal on Control and Optimization , 38(2):447-469, 2000.
- [11] T. Che, Y. Li, A. P. Jacob, Y. Bengio, and W. Li. Mode regularized generative adversarial networks. In Proceedings of the International Conference on Learning Representations (ICLR) , 2017. arXiv:1612.02136.
- [12] C. Chelba, T. Mikolov, M. Schuster, Q. Ge, T. Brants, P. Koehn, and T. Robinson. One billion word benchmark for measuring progress in statistical language modeling. arXiv e-prints , arXiv:1312.3005, 2013.

- [13] D.-A. Clevert, T. Unterthiner, and S. Hochreiter. Fast and accurate deep network learning by exponential linear units (ELUs). In Proceedings of the International Conference on Learning Representations (ICLR) , 2016. arXiv:1511.07289.
- [14] D. DiCastro and R. Meir. A convergent online single time scale actor critic algorithm. J. Mach. Learn. Res. , 11:367-410, 2010.
- [15] D. C. Dowson and B. V. Landau. The Fréchet distance between multivariate normal distributions. Journal of Multivariate Analysis , 12:450-455, 1982.
- [16] M. Fréchet. Sur la distance de deux lois de probabilité. C. R. Acad. Sci. Paris , 244:689-692, 1957.
- [17] S. Gadat, F. Panloup, and S. Saadane. Stochastic heavy ball. arXiv e-prints , arXiv:1609.04228, 2016.
- [18] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio. Generative adversarial nets. In Z. Ghahramani, M. Welling, C. Cortes, N. D. Lawrence, and K. Q. Weinberger, editors, Advances in Neural Information Processing Systems 27 , pages 2672-2680, 2014.
- [19] I. J. Goodfellow. On distinguishability criteria for estimating generative models. In Workshop at the International Conference on Learning Representations (ICLR) , 2015. arXiv:1412.6515.
- [20] I. J. Goodfellow. NIPS 2016 tutorial: Generative adversarial networks. arXiv e-prints , arXiv:1701.00160, 2017.
- [21] X. Goudou and J. Munier. The gradient and heavy ball with friction dynamical systems: the quasiconvex case. Mathematical Programming , 116(1):173-191, 2009.
- [22] P. Grnarova, K. Y. Levy, A. Lucchi, T. Hofmann, and A. Krause. An online learning approach to generative adversarial networks. arXiv e-prints , arXiv:1706.03269, 2017.
- [23] I. Gulrajani, F. Ahmed, M. Arjovsky, V. Dumoulin, and A. Courville. Improved training of Wasserstein GANs. arXiv e-prints , arXiv:1704.00028, 2017. Advances in Neural Information Processing Systems 31 (NIPS 2017).
- [24] M. W. Hirsch. Convergent activation dynamics in continuous time networks. Neural Networks , 2(5):331-349, 1989.
- [25] R. D. Hjelm, A. P. Jacob, T. Che, K. Cho, and Y. Bengio. Boundary-seeking generative adversarial networks. arXiv e-prints , arXiv:1702.08431, 2017.
- [26] S. Hochreiter and J. Schmidhuber. Flat minima. Neural Computation , 9(1):1-42, 1997.
- [27] P. Isola, J.-Y . Zhu, T. Zhou, and A. A. Efros. Image-to-image translation with conditional adversarial networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition , 2017. arXiv:1611.07004.
- [28] P. Karmakar and S. Bhatnagar. Two time-scale stochastic approximation with controlled Markov noise and off-policy temporal-difference learning. Mathematics of Operations Research , 2017.
- [29] D. P. Kingma and J. L. Ba. Adam: A method for stochastic optimization. In Proceedings of the International Conference on Learning Representations (ICLR)) , 2015. arXiv:1412.6980.
- [30] V. R. Konda. Actor-Critic Algorithms . PhD thesis, Department of Electrical Engineering and Computer Science, Massachusetts Institute of Technology, 2002.
- [31] V. R. Konda and V. S. Borkar. Actor-critic-type learning algorithms for Markov decision processes. SIAM J. Control Optim. , 38(1):94-123, 1999.
- [32] V. R. Konda and J. N. Tsitsiklis. Linear stochastic approximation driven by slowly varying Markov chains. Systems &amp; Control Letters , 50(2):95-102, 2003.
- [33] V. R. Konda and J. N. Tsitsiklis. Convergence rate of linear two-time-scale stochastic approximation. The Annals of Applied Probability , 14(2):796-819, 2004.
- [34] A. Krizhevsky, I. Sutskever, and G. E. Hinton. ImageNet classification with deep convolutional neural networks. In Proceedings of the 25th International Conference on Neural Information Processing Systems , pages 1097-1105, 2012.
- [35] H. J. Kushner and G. G. Yin. Stochastic Approximation Algorithms and Recursive Algorithms and Applications . Springer-Verlag New York, second edition, 2003.

- [36] C. Ledig, L. Theis, F. Huszar, J. Caballero, A. P. Aitken, A. Tejani, J. Totz, Z. Wang, and W. Shi. Photo-realistic single image super-resolution using a generative adversarial network. arXiv e-prints , arXiv:1609.04802, 2016.
- [37] C.-L. Li, W.-C. Chang, Y. Cheng, Y. Yang, and B. Póczos. MMD GAN: Towards deeper understanding of moment matching network. In Advances in Neural Information Processing Systems 31 (NIPS 2017) , 2017. arXiv:1705.08584.
- [38] J. Li, A. Madry, J. Peebles, and L. Schmidt. Towards understanding the dynamics of generative adversarial networks. arXiv e-prints , arXiv:1706.09884, 2017.
- [39] J. H. Lim and J. C. Ye. Geometric GAN. arXiv e-prints , arXiv:1705.02894, 2017.
- [40] S. Liu, O. Bousquet, and K. Chaudhuri. Approximation and convergence properties of generative adversarial learning. In Advances in Neural Information Processing Systems 31 (NIPS 2017) , 2017. arXiv:1705.08991.
- [41] Z. Liu, P. Luo, X. Wang, and X. Tang. Deep learning face attributes in the wild. In Proceedings of International Conference on Computer Vision (ICCV) , 2015.
- [42] L. M. Mescheder, S. Nowozin, and A. Geiger. The numerics of GANs. In Advances in Neural Information Processing Systems 31 (NIPS 2017) , 2017. arXiv:1705.10461.
- [43] L. Metz, B. Poole, D. Pfau, and J. Sohl-Dickstein. Unrolled generative adversarial networks. In Proceedings of the International Conference on Learning Representations (ICLR) , 2017. arXiv:1611.02163.
- [44] A. Mokkadem and M. Pelletier. Convergence rate and averaging of nonlinear two-time-scale stochastic approximation algorithms. The Annals of Applied Probability , 16(3):1671-1702, 2006.
- [45] Y. Mroueh and T. Sercu. Fisher GAN. In Advances in Neural Information Processing Systems 31 (NIPS 2017) , 2017. arXiv:1705.09675.
- [46] V. Nagarajan and J. Z. Kolter. Gradient descent GAN optimization is locally stable. arXiv e-prints , arXiv:1706.04156, 2017. Advances in Neural Information Processing Systems 31 (NIPS 2017).
- [47] Y. Nesterov. A method of solving a convex programming problem with convergence rate o (1 /k 2 ) . Soviet Mathematics Doklady , 27:372-376, 1983.
- [48] Y. Netzer, T. Wang, A. Coates, A. Bissacco, B. Wu, and A. Y. Ng. Reading digits in natural images with unsupervised feature learning. In NIPS Workshop on Deep Learning and Unsupervised Feature Learning 2011 , 2011.
- [49] B. T. Polyak. Some methods of speeding up the convergence of iteration methods. USSR Computational Mathematics and Mathematical Physics , 4(5):1-17, 1964.
- [50] H. L. Prasad, L. A. Prashanth, and S. Bhatnagar. Two-timescale algorithms for learning Nash equilibria in general-sum stochastic games. In Proceedings of the 2015 International Conference on Autonomous Agents and Multiagent Systems (AAMAS '15) , pages 1371-1379, 2015.
- [51] A. Radford, L. Metz, and S. Chintala. Unsupervised representation learning with deep convolutional generative adversarial networks. In Proceedings of the International Conference on Learning Representations (ICLR) , 2016. arXiv:1511.06434.
- [52] A. Ramaswamy and S. Bhatnagar. Stochastic recursive inclusion in two timescales with an application to the lagrangian dual problem. Stochastics , 88(8):1173-1187, 2016.
- [53] T. Salimans, I. J. Goodfellow, W. Zaremba, V. Cheung, A. Radford, and X. Chen. Improved techniques for training GANs. In D. D. Lee, M. Sugiyama, U. V. Luxburg, I. Guyon, and R. Garnett, editors, Advances in Neural Information Processing Systems 29 , pages 2234-2242, 2016.
- [54] V. B. Tadi´ c. Almost sure convergence of two time-scale stochastic approximation algorithms. In Proceedings of the 2004 American Control Conference , volume 4, pages 3802-3807, 2004.
- [55] L. Theis, A. van den Oord, and M. Bethge. A note on the evaluation of generative models. In Proceedings of the International Conference on Learning Representations (ICLR) , 2016. arXiv:1511.01844.

| [56] I. Tolstikhin, S. Gelly, O. Bousquet, C.-J. Simon-Gabriel, and B. Schölkopf. AdaGAN: Boosting generative models. arXiv e-prints , arXiv:1701.02386, 2017. Advances in Neural Information Processing Systems 31 (NIPS 2017).   | [56] I. Tolstikhin, S. Gelly, O. Bousquet, C.-J. Simon-Gabriel, and B. Schölkopf. AdaGAN: Boosting generative models. arXiv e-prints , arXiv:1701.02386, 2017. Advances in Neural Information Processing Systems 31 (NIPS 2017).    | [56] I. Tolstikhin, S. Gelly, O. Bousquet, C.-J. Simon-Gabriel, and B. Schölkopf. AdaGAN: Boosting generative models. arXiv e-prints , arXiv:1701.02386, 2017. Advances in Neural Information Processing Systems 31 (NIPS 2017).   |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [57]                                                                                                                                                                                                                               | R. Wang, A. Cully, H. J. Chang, and Y. Demiris. MAGAN: margin adaptation for generative adversarial networks. arXiv e-prints , arXiv:1704.03817, 2017.                                                                              | 2                                                                                                                                                                                                                                  |
| [58]                                                                                                                                                                                                                               | L. N. Wasserstein. Markov processes over denumerable products of spaces describing systems of automata. Probl. Inform. Transmission , 5:47-52, 1969.                                                                                | large                                                                                                                                                                                                                              |
| [59]                                                                                                                                                                                                                               | Y. Wu, Y. Burda, R. Salakhutdinov, and R. B. Grosse. On the quantitative analysis of based generative models. In Proceedings of the International Conference on Learning sentations (ICLR) , 2017. arXiv:1611.04273.                | decoder- Repre-                                                                                                                                                                                                                    |
| [60]                                                                                                                                                                                                                               | F. Yu, Y. Zhang, S. Song, A. Seff, and J. Xiao. LSUN: construction dataset using deep learning with humans in the loop. arXiv e-prints , arXiv:1506.03365,                                                                          | of a large-scale image 2015.                                                                                                                                                                                                       |
| [61]                                                                                                                                                                                                                               | J. Zhang, D. Zheng, and M. Chiang. The impact of stochastic noisy feedback on distributed network utility maximization. In IEEE INFOCOM 2007 - 26th IEEE International Conference on Computer Communications , pages 222-230, 2007. |                                                                                                                                                                                                                                    |
| List of Figures                                                                                                                                                                                                                    | List of Figures                                                                                                                                                                                                                     | List of Figures                                                                                                                                                                                                                    |
| 1                                                                                                                                                                                                                                  | Oscillation in GAN training . . . . . . . . . . . . . . . . . . . .                                                                                                                                                                 | . . . . . . . . . .                                                                                                                                                                                                                |
| 2                                                                                                                                                                                                                                  | Heavy Ball with Friction . . . . . . . . . . . . . .                                                                                                                                                                                | . . . . . . . . . . 4                                                                                                                                                                                                              |
| 3                                                                                                                                                                                                                                  | . . . . . . . FID evaluated for different disturbances . . . . . . . . . . . . .                                                                                                                                                    | . . . . . . . . . . 6                                                                                                                                                                                                              |
| 4                                                                                                                                                                                                                                  | TTUR and single time-scale update with toy data. . . .                                                                                                                                                                              | . . . . . 7                                                                                                                                                                                                                        |
| 5                                                                                                                                                                                                                                  | . . . . . . . . . . FID for DCGAN on CelebA, CIFAR-10, SVHN, and LSUN Bedrooms. .                                                                                                                                                   | . . . . . 8                                                                                                                                                                                                                        |
| 6                                                                                                                                                                                                                                  | FID for WGAN-GP trained on CIFAR-10 and LSUN Bedrooms. . . .                                                                                                                                                                        | . . . . . . . 8                                                                                                                                                                                                                    |
| 7                                                                                                                                                                                                                                  | Performance of WGAN-GP on One Billion Word. . . . . . . . . . . . . . .                                                                                                                                                             | . . . . . . . . . . 9                                                                                                                                                                                                              |
| A8                                                                                                                                                                                                                                 | FID and Inception Score Comparison . . . . . . . . .                                                                                                                                                                                | . . . . . . . . . . 13                                                                                                                                                                                                             |
| A9                                                                                                                                                                                                                                 | CelebA Samples with FID 500 and 300 . . . . . . . . . . . .                                                                                                                                                                         | . . . . . . . . . . 14                                                                                                                                                                                                             |
| A10                                                                                                                                                                                                                                | . . CelebA Samples with FID 133 and 100 . . . . . . . . . . . . . . . . . . . .                                                                                                                                                     | . . . . . . . . . . 14                                                                                                                                                                                                             |
| A11                                                                                                                                                                                                                                | CelebA Samples with FID 45 and 13 . . . . . . . . . . . .                                                                                                                                                                           | . . . . . . . . . . 15                                                                                                                                                                                                             |
| A12                                                                                                                                                                                                                                | CelebA Samples with FID 3 . . . . . . . . . . . . . . . .                                                                                                                                                                           | . . . . . . . . . . 15                                                                                                                                                                                                             |
| A13                                                                                                                                                                                                                                | FID for BEGAN trained on CelebA and LSUN Bedrooms. . . . . . . . . . . . .                                                                                                                                                          | . . . . . . . . . . 33                                                                                                                                                                                                             |
| A14                                                                                                                                                                                                                                |                                                                                                                                                                                                                                     | . . . . . . . . . . 34                                                                                                                                                                                                             |
|                                                                                                                                                                                                                                    | Learning dynamics of two networks. . . . . . .                                                                                                                                                                                      |                                                                                                                                                                                                                                    |
| List of Tables                                                                                                                                                                                                                     | List of Tables                                                                                                                                                                                                                      | List of Tables                                                                                                                                                                                                                     |
| 1                                                                                                                                                                                                                                  | Results DCGAN and WGAN-GP . . . . . . . . . . . . . .                                                                                                                                                                               | . . . . . . . . . . 9                                                                                                                                                                                                              |
| A2                                                                                                                                                                                                                                 | . . . Results WGAN-GP on Image Data . . . . . . . . . . . . . . . .                                                                                                                                                                 | . . . . . . . . . . 32                                                                                                                                                                                                             |
| A3                                                                                                                                                                                                                                 | Samples of the One Billion Word benchmark generated by WGAN-GP. . . . . . . . . . . . . . .                                                                                                                                         | . . . . . . . 33                                                                                                                                                                                                                   |
| A4                                                                                                                                                                                                                                 | Results WGAN-GP on One Billion Word . .                                                                                                                                                                                             | . . . . . . . 33                                                                                                                                                                                                                   |