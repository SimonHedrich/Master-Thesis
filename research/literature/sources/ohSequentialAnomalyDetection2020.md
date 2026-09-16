## Sequential Anomaly Detection using Inverse Reinforcement Learning

Min-hwan Oh Columbia University New York, New York m.oh@columbia.edu Garud Iyengar Columbia University New York, New York garud@ieor.columbia.edu

## 1 INTRODUCTION

Anomaly detection, or outlier detection refers to (automatic) identification of unforeseen or abnormal phenomena embedded in a large amount of normal data. However, anomaly detection is a challenging topic, mainly because of the insufficient knowledge and inaccurate representative of the so-called anomaly for a given system. Often, there are no examples from which distinct features of the anomaly could be learned. Furthermore, in many cases, even when the available data is large, there is the limited availability of labels to model anomaly detection as a discriminative classification task. In most common practices, one must rather learn a precise generative model of normal patterns and detect anomalies as cases that are not sufficiently explained by this model.

One of the most interesting application scenarios in anomaly detection is when sequential data are targeted. For example, in a safety-critical environment, it is crucial to have an automatic detection system to screen the streaming data gathered by monitoring sensors, and to report abnormal observations if detected in real-time. Oftentimes, stakes are much higher when these potential anomalies are intentional or goal-oriented. Various types and large amounts of sequential data are available due to increasing pervasiveness of mobile devices, surveillance systems or other monitoring devices, which pose new challenges both computationally and statistically, and thus require novel approaches in discovering useful patterns. As opposed to static anomalies which are objects that exhibit abnormal behavior at a single snap shot of time, sequential anomalies are objects or instances that exhibit abnormal behavior over several time periods (and can potentially be intentional). Hence, detecting sequential anomalies is more complex, as one needs to capture the objects or instances that deviate from normal behavior by establishing continuity of actions in potentially multiple dimensions

We propose an end-to-end framework for sequential anomaly detection using inverse reinforcement learning (IRL). The objective of IRL is to determine the decision making agent's underlying reward function from its behavior data [45]. A reward function incentivizes an agent to act in a certain way, and hence describes the preferences of the agent whose objective is to collect as much reward as possible (see Section 3 for more detail). The significance of IRL has emerged from problems in diverse research areas. In robotics [3], IRL provides a framework for making robots learn to imitate the demonstrator's behavior using the inferred reward function. In human and animal behavior studies [32], the agent's behavior could be understood by the reward function since the reward function reflects the agent's objectives and preferences.

The proposed method takes the trajectories of target agents as input. In addition to the sequence of coordinates, our method

## ABSTRACT

One of the most interesting application scenarios in anomaly detection is when sequential data are targeted. For example, in a safety-critical environment, it is crucial to have an automatic detection system to screen the streaming data gathered by monitoring sensors, and to report abnormal observations if detected in realtime. Oftentimes, stakes are much higher when these potential anomalies are intentional or goal-oriented. We propose an endto-end framework for sequential anomaly detection using inverse reinforcement learning (IRL), whose objective is to determine the decision-making agent's underlying function which triggers his/her behavior. The proposed method takes the sequence of actions of a target agent (and possibly other meta information) as input. The agent's normal behavior is then understood by the reward function which is inferred via IRL. We use a neural network to represent a reward function. Using a learned reward function, we evaluate whether a new observation from the target agent follows a normal pattern. In order to construct a reliable anomaly detection method and take into consideration the confidence of the predicted anomaly score, we adopt a Bayesian approach for IRL. The empirical study on publicly available real-world data shows that our proposed method is effective in identifying anomalies.

## CCS CONCEPTS

· Computingmethodologies → Inverse reinforcement learning ; · Information systems → Spatial-temporal systems .

## KEYWORDS

anomaly detection; outlier detection; inverse reinforcement learning; neural networks; bootstrap; spatial-temporal data

## ACMReference Format:

Min-hwan Oh and Garud Iyengar. 2019. Sequential Anomaly Detection using Inverse Reinforcement Learning. In The 25th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '19), August 4-8, 2019, Anchorage, AK, USA. ACM, New York, NY, USA, 11 pages. https: //doi.org/10.1145/3292500.3330932

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

KDD '19, August 4-8, 2019, Anchorage, AK, USA

© 2019 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 978-1-4503-6201-6/19/08...$15.00 https://doi.org/10.1145/3292500.3330932

can also incorporate other meta information, e.g. date, time, and possibly the features of landmarks if available, as additional input. The agent's normal behavior is then understood by the reward function which is inferred via inverse reinforcement learning. We use a neural network to represent a reward function for each target agent. Using a learned reward function, we can evaluate whether a new observation from the target agent follows a normal pattern. In other words, if the new observation gives a low reward, then it implies that the observation is not explained by the preferences of the agent that we have learned so far, and that it can be considered as a potential anomaly.

In order to construct a reliable anomaly detection method, we need to also take into consideration the confidence of the predicted anomaly score. Rather than blindly taking the estimated reward as a normality score only, we also consider model confidence of the predicted values. This is crucial especially when false alarms can incur costs (which is typical in many applications). For this reason, we consider a Bayesian approach for IRL in this work. Bayesian IRL formulates the reward preference as the prior and the behavior compatibility as the likelihood, and find the posterior distribution of the reward function [42]. Most of the existing IRL methods with non-linear function approximation are in non-Bayesian (point estimation) settings [12, 13, 53]; hence cannot incorporate model uncertainty. There are a number of advantages of Bayesian approaches in learning perspectives: We do not need a completely specified optimal policy as input to the IRL agent, nor do we need to assume that the trajectories from the target agent has no anomalies, hence allowing a small portion of anomalies to be present in the trajectories when learning the normal behavior; Also, we can incorporate external information about specific IRL problems into the prior of the model. Furthermore, with model confidence at hand we can treat uncertain inputs and special cases explicitly. For example, the reward function might return a value with high uncertainty. In this case we might decide to pass the input to a human for validation.

The main contributions of this paper are as follows:

- To the best of our knowledge, this is the first time that IRL is used for anomaly detection, which is a natural way of understanding the underlying motive of observed behaviors when anomalous actions are potentially intentional or goaloriented.
- We incorporate the model uncertainty in the IRL problem in a very simple and scalable manner, which may be of independent interest. This model uncertainty is shown to help identifying an anomaly.
- Ourmethodcanincorporate trajectories with varying lengths as input and can be used for real-time detection.
- The empirical study on publicly available real-world data shows that our proposed method is effective in identifying anomalies.

The organization of the paper is as follows: In Section 2, we discuss literature in sequential anomaly detection. Note that we defer the discussion (and literature review) on IRL entirely to Section 3 due to the limited space. Section 3 gives a brief overview of IRL and its Bayesian extension. Section 4 presents our algorithm and anomaly detection procedure. Section 5 discusses the empirical study of our proposed method. Finally, Section 6 concludes our paper.

## 2 RELATED WORK

Anomaly detection has its roots in the more general problem of outlier detection. In many cases the data is static, rather than evolving over time. There are a number of different methods available for outlier detection, including supervised approaches [1], distance-based [2, 23], density-based [7], model-based [18] and isolation-based methods [26].

Sequential anomaly detection methods constitute an important category of anomaly detection methods, which usually uses objecttrajectories of tracked persons, or cars in traffic scenes for example [10, 19, 25, 41, 48], but also temporally shorter and dense trajectory representations, such as representative trajectories for crowd flow obtained from clustering particle trajectories in [52]. [48] presents a method that uses cubic spline curves to parametrize trajectories and an incremental one-class learning approach using Gaussian mixture models. [25] introduces trajectory sparse reconstruction analysis (SRA) that constructs a normal dictionary set which is used to reconstruct test trajectories. In [24], a trajectory is split into various partitions (at equal intervals) and a hybrid of distance and density based approaches is used to classify each partition as anomalous or not. In [15], the authors compute a score based on the evolving moving direction and density of trajectories, and make use of a decay function to include previous scores. [8] present a method for monitoring anomalies over continuous trajectory streams by using the local continuity characteristics of trajectories to cluster trajectories on a local level; anomalies are then identified by a pruning mechanism. However, many of these previous methods require feature engineering to perform analysis or preprocessing of the data.

There is also a line of work that utilize clustering approaches. [19] uses a hierarchical clustering of trajectories depending on spatial and temporal information and a chain of Gaussian distributions to represent motion patterns. In [41], single-class support vector machine (SVM) clustering is used to identify anomalous trajectories in traffic scenes. [51] distinguish regular trajectories and anomalous trajectories by applying adaptive hierarchical clustering based on an optimal number of clusters. However, many of these methods are designed to be used as batch-based method, and cannot be used in real-time in online settings.

Recent advances in artificial neural network and availability of larger datasets allowed the use of deep learning in the domain of sequential anomaly detection. [46] uses convolutional neural network (CNN) transferred from a pre-trained supervised network and ensures the detection of (global) anomalies in video scenes. However, this is only limited to a fixed location (fixed camera view point). Recurrent neural network (RNN) based approaches have been proposed for anomaly detection [30, 39]. These approaches learn a model to reconstruct the normal data (e.g. when a system is in perfect health) such that the learned model could reconstruct the sub-sequences which belong to normal behavior. The learned model leads to high reconstruction error for anomalous sub-sequences, since it has not seen such data during training. However, their evaluation uses either only univariate time series or time series with more regularity such as bounded sequences and periodic patterns. Also, these method typically requires the fixed length of sequences (or fixed window-size).

The majority of methods discussed so far do not necessarily address the scenario where the target agent may be intentionally malicious or goal-directed. [10] addresses this issue and uses inexplicability scores to measure the extent to which a trajectory can be regarded as goal-directed. While our work and [10] share a similar motivation, their method however still requires feature-engineering and limited to be used in a online setting.

There are also other work which use (forward) reinforcement learning to detect anomalies [28] or certain patterns of interest [21] in dynamic systems. However, these methods require the predefined notion of reward signals whereas our approach is to infer such reward functions since it is unknown.

## 3 PRELIMINARIES

## 3.1 Forward and Inverse Reinforcement Learning (RL) Basics

We assume that the environment is modeled as a Markov decision process (MDP) ⟨ S , A , T , R , γ , p 0 ⟩ where S is the finite set of states; A is the finite set of actions; T ( s , a , s ′ ) is the state transition probability of changing to state s ′ from state s when action a is taken; r ( s , a ) is the immediate reward of executing action a in state s ; γ ∈ [ 0 , 1 ) is the discount factor; p 0 ( s ) denotes the probability of starting in state s . For notational convenience, we use the vector r = [ r 1 , ..., r D ] to denote the reward function.

A policy is a mapping π : S → A . The value of policy π is the expected discounted return of executing the policy, defined as V π = E [ ∞ t = 0 γ t r ( s t , a t )| p 0 , π ] . The value function of policy π for state s is computed by V π ( s ) = r ( s , π ( s )) + γ s ′ ∈ S T ( s , π ( s ) , s ′ ) V π ( s ′ ) so that the value is calculated by V π = s ∈ S p 0 ( s ) V π ( s ) . Given an MDP, the agent's objective is to execute an optimal policy π ∗ that maximizes the value function for all the states, which should satisfy the Bellman optimality equation: V ∗ ( s ) = max a ∈ A [ r ( s , a ) + γ s ′ ∈ S T ( s , a , s ′ ) V ∗ ( s ′ )] .

In (forward) RL, π ∗ is the policy to be learned. In IRL on the other hand, the reward function is not explicitly given. Hence, the goal of IRL is to learn a reward function r ∗ (·) that that explains the demonstrator's behavior:

<!-- formula-not-decoded -->

where π ∗ is the demonstrator's policy which is optimal (or nearoptimal) with respect to the unknown reward function r ∗ ( s t , a t ) .

However, the optimal policy π ∗ is not always explicitly given in most cases. Instead, demonstrations or samples from π ∗ are given. In other words, we assume that the agent's demonstration is generated by executing an optimal policy π ∗ with some unknown reward function R ∗ , given as the set T of M trajectories where the m -th trajectory is a sequence of state-action pairs: τ m = {( s m , 1 , a m , 1 ) , ( s m , 2 , a m , 2 ) ..., ( s m , H , a m , H ))} where H is a horizon length, which can vary from a trajectory to another. Also, we sometimes overload the term π and denote the trajectory-wise policy as π ( τ ) to represent a policy being followed to generate an entire trajectory following π ( s ) sequentially.

## 3.2 Maximum Entropy IRL

Maximum entropy IRL framework [56] models the demonstrations using a Boltzmann distribution, where the energy is given by the trajectory-wise reward function R ( τ | θ ) :

<!-- formula-not-decoded -->

where R ( τ | θ ) = ( s , a )∈ τ r θ ( s , a ) and r θ ( s , a ) is a learned reward function parametrized by θ , and the partition function Z is the integral of exp ( R ( τ | θ )) over all trajectories consistent with the MDP's dynamics. Under this model, trajectories with equal reward are assigned the same likelihood, and higher-reward trajectories are exponentially more preferred by the demonstrator. The parameters θ are chosen to maximize the likelihood of the (given) demonstrated trajectories:

<!-- formula-not-decoded -->

Computing the gradient with respect to θ gives the following:

<!-- formula-not-decoded -->

Here π ∗ is the demonstrator's policy and π θ is a (soft) optimal policy under reward parameter θ [56]. This derivation is applicable to any differentiable reward functions. [56] use linear reward functions in their work, and they show that the gradient implies that the optimal policy under θ ∗ matches the feature expectation of the demonstrator's policy.

Computing the second expectation in (2) requires finding the soft optimal policy under the current reward parameter θ and computing its expected state visitation frequencies using a variant of the value iteration algorithm for (forward) RL. However, for large or continuous domains, this becomes intractable, since this computation scales exponentially with the dimensionality of the state space. This method also requires repeatedly solving an MDP in the inner loop of an iterative optimization, further increasing the computational difficulty of larger systems. [12, 13] extend maximum entropy IRL to sample-based IRL where dynamics of MDP is not given and a reward function is parametrized with non-linear function, e.g. neural networks; hence in order to estimate Z , they generate background sample trajectories (see [13] for more detail). The sample-based IRL [13] is closely related to other sample-based maximum entropy methods, including relative entropy IRL by [6] and path integral IRL by [20], which can also handle unknown dynamics. However, unlike these prior methods, [13] adapt the sampling distribution using policy optimization.

## 3.3 Bayesian framework for IRL

The main idea of Bayesian IRL (BIRL) is to use a prior to encode the reward preference and to formulate the compatibility with the demonstrator's policy as a likelihood in order to derive a probability distribution over the space of reward functions, from which the demonstrator's reward function is extracted [42]. Assuming that the reward function entries are independently distributed, the prior is defined as

<!-- formula-not-decoded -->

Various distributions can be used as the prior. For example, the uniform prior can be used if we have no knowledge about the reward function other than its range, and a Gaussian or a Laplacian prior can be used if we prefer rewards to be close to some specific values. The likelihood in BIRL is defined as an independent exponential distribution analogous to the softmax function:

<!-- formula-not-decoded -->

The posterior over the reward function is then formulated by combining the prior and the likelihood, using Bayes theorem:

<!-- formula-not-decoded -->

The reward function can be inferred from the model by computing the posterior mean using a Markov chain Monte Carlo (MCMC) algorithm [42].

MAP Inference. [9] show that using the posterior mean may not be a good idea since it may yield a reward function whose corresponding optimal policy is inconsistent with the demonstrator's behaviour. In other words, the posterior mean integrates the error over the entire space of reward functions by including (possibly) infinitely many rewards that induce policies that are inconsistent with the demonstration trajectories. Instead, the maximum-a-posteriori (MAP) estimate could be a better solution for IRL. Then IRL can be formulated as a posterior optimization problem and [9] propose a gradient method to calculate the MAP estimate that is based on the (sub)differentiability of the posterior distribution. Most of the non-Bayesian IRL algorithms in the literature [35, 36, 43, 50, 56] can be cast as searching for the MAP reward function in BIRL with different priors and different ways of encoding the compatibility with the demonstrator's policy.

Note that both BIRL [42] and MAP framework for IRL [9] have been limited to tabular settings, i.e. without function approximation. Hence, it cannot be directly applied to our setting. In the next section, we introduce our method and discuss how we can combine the notion of uncertainty with sampled-based IRL.

## 4 METHOD

Our proposed method extends the sample-based maximum entropy IRL [13] to Bayesian framework to approximate a distribution over reward functions. The main idea of the proposed method is the following. We first assume the parameters of the reward function come from a prior distribution. At the start of each training iteration, we sample a single reward function from its approximate posterior. We then follow a sample generating policy for the duration of the iteration to generate background trajectories (recall that we need background sample trajectories to estimate the partition function Z in Eq. (1) since we do not know the underlying dynamics of MDP), and improve the policy with respect to the sampled reward function. Then, using the generated background trajectories along with the demonstrated trajectories from the target agent, we estimate the gradient and update the (sampled) reward function. We repeat this process until convergence.

We parametrize our reward functions as neural networks, expanding their expressive power without hand-engineered features. Previous work has shown that an affine reward function is not expressive enough to learn complex behaviors [13, 53]. While the expressive power of nonlinear reward functions provide a range of benefits, they introduce significant model complexity to an already underspecified IRL objective and additionally learning the distribution over this nonlinear reward function adds more complexity. In the following sections, we present how we approximate the posterior distribution over reward functions and how we can remedy increased computational complexity. Then, we show how we approximate the gradient for the reward function updates. Finally, we present how we operationalize the entire algorithm.

## 4.1 Approximate Posterior Distribution

Many previous literature on Bayesian neural network studied uncertainty quantification founded on parametric Bayesian inference [5, 14]. (For more detailed review on Bayesian neural network, see Section C in the appendix.) In this work, we consider a nonparametric bootstrap of functions.

4.1.1 Bootstrap ensemble. Bootstrap is a simple technique for producing a distribution over functions with theoretical guarantees [4, 11]. It is also general in terms of the class of models that we can accommodate. In its most common form, a bootstrap method takes as input a data set D and a function f θ . We can transform the original dataset D into K different data sets {D k } K k = 1 's of cardinality equal to that of the original data D that is sampled uniformly with replacement. Then we train K different models. For each model f θ k , we train the model on the data set D k . So each of these models is trained on data from the same distribution but on a different data set. Then if we want to approximate sampling from the distribution of functions, we sample uniformly an integer k from 1 to K and use the corresponding function f θ .

k

Figure 1: Bootstrap ensemble sampling. each base model is trained on randomly perturbed data

<!-- image -->

In cases of using neural networks as base models f θ k , bootstrap ensemble maintains a set of K neural networks { f θ k } K k = 1 independently on K different bootstrapped subsets of the data. It treats each network as independent samples from the weight distribution. In contrast to traditional Bayesian approaches discussed earlier, bootstrapping is a frequentist method, but with the use of the prior distribution, it could approximate the posterior in a simple manner. Also it scales nicely to high-dimensional spaces, since it only requires point estimates of the weights. However, one major drawback is that computational load increase linearly with respect to the number of base models. In the following section, we discuss how to mitigate this issue and still maintain a reasonable uncertainty estimates.

Figure 2: Illustration of bootstrap network with K output heads. In each iteration, an output head is randomly chosen

<!-- image -->

4.1.2 Single network with K output heads. Training and maintaining a multiple independent neural networks is computationally expensive especially when each base network is a large and deep neural network. In order to remedy this issue, we adopt a single network framework which is scalable for generating bootstrap samples from a large and deep neural network [37, 38]. The network consists of a shared architecture with K bootstrapped heads branching off independently (as shown in Figure 2). Each head is trained only on its bootstrapped sub-sample of the data as described in Section 4.1.1. The shared network learns a joint feature representation across all the data, which can provide significant computational advantages at the cost of lower diversity between heads. This type of bootstrap can be trained efficiently in a single forward/backward pass.

To capture model uncertainty, we assume a prior distribution over its weights, e.g. with a Gaussian prior [ θ s , θ 1 , ..., θ K ] ∼ N( 0 , σ 2 ) for fixed variance σ 2 , where θ s is the parameter of the shared network and θ 1 , ..., θ K are the parameters of bootstrap heads 1 , .., K . For brevity of notations, we overload the term θ k = [ θ k , θ s ] since θ s is shared across all samples. For each iteration of training procedure, we sample the model parameter ˆ θ k ∼ q ( θ ) where q ( θ ) is a bootstrap distribution. In other words, at each iteration we randomly choose which head to use to predict a reward score and update the parameters. In the following section, we present how we update the reward parameter θ .

## 4.2 Parameter Update

Suppose we sampled the current reward parameter, i.e. sampled an output head from K heads, and denote the sampled parameter as θ (instead of θ k ) for the notational brevity in this section. Hence, our reward function is expressed as r θ ( s , a ) . Following MAP inference arguments in Section 3.3 and Eq. 3.3, we can then reformulate the IRL problem into the posterior optimization problem, which is finding θ MAP that maximizes the (log unnormalized) posterior:

<!-- formula-not-decoded -->

where the distribution p ( τ | θ ) is defined as in Eq. (1) and p ( θ ) is a prior distribution of θ . We can compute the gradient with respect to θ as follows:

<!-- formula-not-decoded -->

where D is the distribution of the observed trajectories. We can then write ∂ ∂ θ log Z θ as E τ ′ ∼ π ( s , a )∈ τ ′ ∂ ∂ θ r θ ( s , a ) where π = π ( τ ) is a sample-generating policy discussed in Section 3.2. Now, we approximate the gradient with finite samples:

<!-- formula-not-decoded -->

where T d is a set of demonstrated trajectories from the target agent, and T s is a set of background trajectories the policy π generates. Now, note that this policy π is being improved as we update reward parameters. Hence, trajectories in T s are the samples collected for policy improvement and are also used to update the reward function. However, trajectories sampled from these intermediate policies are biased samples since those are not sampled from the soft optimal policy under the current reward as shown in Eq. (2). To address this issue, we adopt a technique used in [13] to use importance sampling to re-weight the samples to make the ones with higher reward more likely (or ones that are unlikely from the current policy more likely).

Define w j : = exp ( R ( τ j | θ )) q ( τ j ) where R ( τ j | θ ) = ( s , a )∈ τ j r θ ( s , a ) , and W : = j w j . The gradient is then given by:

<!-- formula-not-decoded -->

Since in our case the reward function r θ ( s , a ) is represented by a neural network, this gradient can be computed efficiently by backpropagating - w j W for each background trajectory τ j ∈ T s and 1 | T d | for each demonstration trajectory τ ∈ T d . The algorithm alternates between optimizing the reward function r θ (·) using this gradient estimate, and optimizing the policy π ( τ ) with respect to the current reward function. We also use a neural network to represent policy π ( τ ) . For policy optimization, we use trust-region policy optimization (TRPO) [47], a state-of-the-art policy gradient algorithm.

## 4.3 Algorithm Overview

Suppose we are interested in individual target agent's behavior, hence we learn a single reward function for a given target agent and T d contains trajectories from a single target agent. It is important to note that we can also consider learning a reward function for a group of individuals to capture an underlying common behaviors of the group in other applications. Our method can also be used even in that scenario. However, we focus on learning a reward function for a single target agent in this section for the ease of the presentation.

We initialize the reward network parameters θ and policy network π , e.g. with a Gaussian distribution with mean µ = 0 and variance σ 2 = 0 . 1. We are given a set of trajectories from a target agent T d = { τ 1 , ..., τ M } . For each iteration, we sample uniformly at random k ∈ { 1 , ..., K } to choose an output head k , hence a reward function r θ k (·) . Given this reward function, we follow π ( τ ) to generate a new set of trajectories T traj which are added to the background samples T s to estimate the partition function Z in maximum entropy formulation in Eq. (1). Using newly updated background samples T s , we update the reward parameter θ k using Eq. (3). Note that the reward function is updated using all samples collected thus far. Then based on the current reward function, we update the policy π using TRPO (or any policy gradient method). Then we repeat this process until convergence. This procedure returns a learned reward function r θ (·) and also a trajectory generating policy π ( τ ) as a byproduct. Our algorithm is summarized in Algorithm 1 which uses Algorithm 2 as a sub-routine.

| Algorithm 1:   | 1 IRL with bootstrapped reward Obtain trajectories T d from the target agent   |
|----------------|--------------------------------------------------------------------------------|
| 2:             | Initialize policy π and reward functions { r θ k } K = 1                       |
| 3:             | k for iterations n = 1 to N do                                                 |
| 4:             | Sample r θ uniformly at random                                                 |
| 5:             | k Generate samples T traj from π                                               |
| 6:             | Append samples: T s ←T s ∪ T traj                                              |
| 7:             | Update r θ using Alogrithm 2 with T s and θ k                                  |
| 8:             | k Update π with respect to r θ k using TRPO [47]                               |
| 9:             | end for                                                                        |
| 10:            | return optimized reward parameters { θ k } K = 1 and policy π                  |

## Algorithm 2 Reward function update

| Require: θ k , T d , T s   | Require: θ k , T d , T s                                  |
|----------------------------|-----------------------------------------------------------|
| 1:                         | for iterations j = 1 to J do                              |
| 2:                         | Sample background batch ˆ T s ⊂ T s                       |
| 3:                         | Sample demonstration batch ˆ T d ⊂ T d                    |
| 4:                         | Append demonstration to background: ˆ T s ← ˆ T s ∪ ˆ T d |
| 5:                         | Estimate d L( θ k ) dθ k using ˆ T d and ˆ T s            |
| 6:                         | Update parameters θ using gradient d L( θ k ) in Eq. (3)  |
| 7:                         | dθ k end for                                              |
| 8:                         | return optimized reward parameters θ k                    |

## 4.4 Normality Score

Once the reward function is learned on the demonstrated trajectories, we can compute the normality score of a new observation ( s , a ) , which we define as

<!-- formula-not-decoded -->

r

where ¯ r and ¯ σ r are an average reward and standard deviation for the entire observations, and ¯ r θ ( s , a ) = 1 K k ¯ r θ k ( s , a ) is the mean reward for the new observation. Note that during training we sample a single output head to compute a gradient and update reward parameters, but in test time prediction, we output from all output heads to compute the mean reward. Also, it is important to note that a target agent that has trajectories with more regular patterns will have a smaller σ r than ones with more diverse trajectories. This normalization helps balance out this effect when deciding an anomaly.

Also we define N ( τ ) : = 1 | τ | ( s , a )∈ τ n ( s , a ) to be a trajectory-wise normality score. Then, an observation (either state-action pair observation ( s , a ) or a trajectory τ ) can be considered as an anomaly if it has a low normality score, e.g. a normality score is below some threshold ϵ . Now, the exact value of threshold ϵ depends on specific applications. In some applications, one can allow the threshold to change dynamically depending on time, or other meta information Or, one may use different thresholds for different target agents. Either way, this score indicates how much an observation deviates from the normal score.

## 4.5 Incorporating Model Uncertainty when Deciding an Anomaly

In addition to the predicted reward for a new observation (and the normality score based on it), we also have the predicted variance over the K reward estimates:

<!-- formula-not-decoded -->

which represent how uncertain the model is regarding this normality measure. Hence, we can incorporate this information when deciding an anomaly. For example, once normality score finds a potential anomaly with n ( s , a ) ≤ ϵ , one can then put another threshold σ r ( s , a ) ≤ γ for some fixed γ , i.e. the model has to be certain that an observation is an anomaly when assigning an anomaly. This may be useful when false positives (false alarms) incur a higher cost, hence one can be conservative in deciding an anomaly based on how small γ is.

## 5 EXPERIMENTS

In our experiments, we aim to answer two questions:

- (1) Can our proposed method accurately identify anomalies in trajectory data?
- (2) Is having an estimate of model uncertainty helpful in identifying anomalies?

To answer (1), we evaluate our method on publicly available dataset and compare with baseline models based on standard classification metrics. To answer (2), instead of taking the normality score only, we incorporate the predictive variance when assigning anomalies which is discussed in Section 4.5. We denote our proposed model with the uncertainty consideration as IRL-ADU (which stands for IRL Anomaly Detector with Uncertainty) and our model without the uncertainty consideration as IRL-AD. We compare these two methods with other baseline methods.

## 5.1 Datasets

We report on a comprehensive empirical study using real-world multidimensional time series datasets which are publicly available: GeoLife GPS trajectory dataset [54, 55] and Taxi Service Trajectory (TST) dataset [33]. The datasets do not include labeled anomalies. 1 Hence, we define what anomalies are in each dataset.

1 To the best of our knowledge, there is no publicly available GPS trajectory datasets with labeled anomalies. Perhaps, the closest dataset is UAH-DriveSet [44] which contains normal, aggressive, and drowsy driving behaviors with labels. However, the data size is very small with only 6 drivers and two trajectories per driver for each category.

GeoLife GPS Dataset [54, 55] contains GPS trajectories collected from 182 individual users over the period of three years and contains more than 17,000 trajectories. each trajectory contains the information of latitude, longitude and altitude. These trajectories were recorded by different GPS loggers and GPS-phones, and have a variety of sampling rates. 91 percent of the trajectories are logged every 1-5 seconds or every 5-10 meters per point. The length of trajectory varies significantly, ranging from below 100 to over 3000. The number of trajectories per user also varies from 28 to 400. This dataset recorded a broad range of users' outdoor movements, including not only life routines like go home and go to work but also some entertainments and sports activities, such as shopping, sightseeing, dining, hiking, and cycling. With this dataset, we chose 10 individuals (with larger number of trajectories recorded) to be our target agents, for each of whom we learn a separate reward function. We injected trajectories which do not belong to these individuals and define them as anomalies. Additionally, we handlabeled some of the existing abnormal trajectories that are already present in those individuals trajectories (e.g. see the trajectory in the center in Figure 3)

TST Dataset [33] contains GPS trajectories collected from 442 taxi drivers in the city of Porto, in Portugal, and contains more than 1 million trips. Each trajectory contains the information of latitude and longitude, along with other meta information including how a service initiated (dispatch from central, taxi stand, or random streets), day type (workdays/weekends) and the location of an origin stand, etc. Number of trips per driver varies from 100 to 6751. Here, we are interested in learning the aggregated normal behavior of the taxi drivers; hence we learn a single reward function for the entire pool of drivers, and we define longer trips - trips with cruise time greater than 50 minutes, which are mostly cross-town trips - as anomalies. To prevent detection methods from being able to identify these anomalies by trivially looking at the length of the trips, we crop the anomalous trips from the end so that they are of similar lengths. Hence, the detection methods look at the earlier portion of these long trips and need to distinguish from other normal inner-city trips.

Data Preparation. In the evaluations on both datasets, we define state s t to be a vector of the location (longitude and latitude) at time t concatenated with the initial location of the trajectory and the time duration since leaving the initial location. We define action a t to be the 2-dimensional vector representing a velocity at time t . One important thing to note is that a training dataset may also contain anomalous trajectories, however our model can deal with this suboptimality since we adopt a Bayesian approach. In particular, this is crucial when deployed to real-world applications, since in practice we rarely have a clean scenario, free of any anomalies. Hence, for this reason, we can use our method in online settings where data arrive continuously and one can constantly update the reward function with a new stream of data.

## 5.2 Comparison with Other Methods

As baselines, we consider two classic anomaly detection models: Local Outlier Factor (LOF) [7] and One-class Support Vector Machines (One-Class SVM) [31] to compare with our proposed model. Both baselines are state-of-the-art anomaly detection algorithms which can be used for sequential data [17]. The settings for both baselines are set to default of Scikit-learn [40] in Python library. Thus, both methods are able to return an anomaly score for each observation in a trajectory. Additionally, we also compare with neural network autoencoder based detectors, similar to more recently work [30, 39, 46]. We consider both a feed-forward network autoencoder (FNN-AE) and a recurrent autoencoder, specifically Long short-term memory (LSTM-AE) model. The architecture and implementation details are deferred to the appendix.

## 5.3 Evaluation Metrics

For evaluation, we compared predicted anomaly decisions to the ground truth, Boolean anomaly labels. Since the proposed method and baselines are able to identify anomaly observations, we are able to measure F 1-score, recall, and precision against the ground truth anomaly labels. The higher the values of F 1-score, recall, and precision are, the more accurate the given anomaly detection method is.

## 5.4 Results

We first discuss the sample results shown in Figure 3. The left figure shows a normal trajectory which is a return trip from the source to Destination 1 and a return to Destination 2. The predicted normality scores suggest that the entire trajectory can be considered normal. Recall that the normality score is normalized in Eq. (4), hence we are only interested in negative deviations from zero when searching for anomalies - and more positive normality scores represent potentially more frequently appearing trajectories than the average. The center figure in Figure 3 shows a trajectory with a partial anomaly. This trajectory is one of the examples we chose to hand-label as anomaly after inspections due to the unusual detour which is highlighted as a red color trace on the figure. The predicted normality scores suggest the detour is highly likely an anomaly because of the low predicted normality score along that detour path. 2 An interesting observation is that on the return path which is in fact a normal path, the normality score returns back to a normal range. This observation is very promising since it shows the proposed method can predict where/when the anomaly starts and where/wen it ends. This will be very useful when we deal with continuous streams of data sequences without the discrete notion of beginning and end of sequences. It suggests that our method can be used for real-time predictions. Lastly, the right figure shows the trajectory-wise anomaly which in fact belongs to someone else other than the target agent but was injected as the target agent's trajectory in the test time. Hence, this is an completely unobserved new sequence to the learned reward function. Hence, as expected, the predicted normality scores are very low for the entire duration. Note that on the third graph the y-axis scales are lower than the other two plots, and the scores are significantly lower than those of the other two trajectories; hence diagnosing this entire trajectory as an anomaly.

Based on the observations, we chose the threshold ϵ = - 2 and γ = 1 . 5. One could potentially fine-tune these threshold hyperparameters. However, for the evaluations we performed in this study, we fixed the threshold values. For the number of output heads in the reward network, we chose K = 10 for all experiments. Table 1 summarizes the average performance on the GeoLife GPS and TST datasets based on F 1, recall, and precision. Note that the performances on GeoLife GPS were averaged over the 10 users, for whom we learn a separate reward function, and their detailed results are reported in Tables 2 and 3 in the appendix. For TST dataset, we learned a single reward function for the entire pool of the taxi drivers; hence learning a common behavior. The performance results show that both of our proposed methods, IRL-AD and IRL-ADU outperform other baseline methods. The results demonstrate that the proposed methods are effective at identifying outliers. While these two methods show comparable results, IRL-ADU yields a higher precision overall and IRL-AD yield a slightly higher recall.

2 Note that this does not mean all detours are anomalous. This trajectory happens to be a detour and anomalous with respect to normal behavior of the target agent.

Figure 3: Sample trajectories from GeoLife GPS dataset and their corresponding predicted normality scores

<!-- image -->

## 6 DISCUSSION

We propose a end-to-end framework for anomaly detection on sequential data. Our method utilizes IRL framework. To the best of our knowledge, this is the first attempt to use IRL framework in sequential anomaly detection problems. As stated earlier, this adaptation appears to be very natural and intuitive. We believe this is an important result that shows viability of this line of work, which can lead further to bridging between the rich field of reinforcement learning and the practical application of anomaly detection.

We also propose a scalable and simple technique to extend the state-of-art sample-based IRL method to approximate the posterior distribution of reward functions at a very low additional computational cost. This extension may be of independent interest, which can also be used to solve a general IRL problems.

## 7 ACKNOWLEDGEMENTS

The authors would like to thank Naoki Abe for helpful discussions.

Table 1: Average F 1 , recall, and precision on real-world datasets and comparison with baselines

| Method         |           |   GeoLife |   TST |
|----------------|-----------|-----------|-------|
| LOF            | Recall    |     0.610 | 0.483 |
| LOF            | Precision |     0.154 | 0.478 |
| LOF            | F 1 score |     0.245 | 0.481 |
| One-Class SVM  | Recall    |     0.620 | 0.491 |
| One-Class SVM  | Precision |     0.168 | 0.474 |
| One-Class SVM  | F 1 score |     0.263 | 0.482 |
| FNN-AE         | Recall    |     0.660 | 0.521 |
| FNN-AE         | Precision |     0.173 | 0.873 |
| FNN-AE         | F 1 score |     0.274 | 0.652 |
| LSTM-AE        | Recall    |     0.720 | 0.555 |
| LSTM-AE        | Precision |     0.199 | 0.871 |
| LSTM-AE        | F 1 score |     0.312 | 0.678 |
| IRL-AD (Ours)  | Recall    |     0.850 | 0.566 |
| IRL-AD (Ours)  | Precision |     0.314 | 0.924 |
| IRL-AD (Ours)  | F 1 score |     0.458 | 0.702 |
| IRL-ADU (Ours) | Recall    |     0.840 | 0.534 |
|                | Precision |     0.353 | 0.936 |
|                | F 1 score |     0.495 | 0.680 |

## REFERENCES

- [1] Naoki Abe, Bianca Zadrozny, and John Langford. 2006. Outlier detection by active learning. In Proceedings of the 12th ACM SIGKDD international conference on Knowledge discovery and data mining . ACM, 504-509.
- [2] Fabrizio Angiulli and Fabio Fassetti. 2009. Dolphin: An efficient algorithm for mining distance-based outliers in very large datasets. ACM Transactions on Knowledge Discovery from Data (TKDD) 3, 1 (2009), 4.
- [3] Brenna D Argall, Sonia Chernova, Manuela Veloso, and Brett Browning. 2009. A survey of robot learning from demonstration. Robotics and autonomous systems 57, 5 (2009), 469-483.
- [4] Peter J Bickel and David A Freedman. 1981. Some asymptotic theory for the bootstrap. The Annals of Statistics (1981), 1196-1217.
- [5] Charles Blundell, Julien Cornebise, Koray Kavukcuoglu, and Daan Wierstra. 2015. Weight uncertainty in neural networks. arXiv preprint arXiv:1505.05424 (2015).
- [6] Abdeslam Boularias, Jens Kober, and Jan Peters. 2011. Relative entropy inverse reinforcement learning. In Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics . 182-189.
- [7] Markus M Breunig, Hans-Peter Kriegel, Raymond T Ng, and Jörg Sander. 2000. LOF: identifying density-based local outliers. In ACMsigmod record , Vol. 29. ACM, 93-104.
- [8] Yingyi Bu, Lei Chen, Ada Wai-Chee Fu, and Dawei Liu. 2009. Efficient anomaly monitoring over moving object trajectory streams. In Proceedings of the 15th ACM SIGKDD international conference on Knowledge discovery and data mining . ACM, 159-168.
- [9] Jaedeug Choi and Kee-Eung Kim. 2011. Map inference for bayesian inverse reinforcement learning. In Advances in Neural Information Processing Systems . 1989-1997.
- [10] Hannah M Dee and David C Hogg. 2004. Detecting inexplicable behaviour.. In BMVC . 1-10.
- [11] Bradley Efron and Robert J Tibshirani. 1994. An introduction to the bootstrap . CRC press.
- [12] Chelsea Finn, Paul Christiano, Pieter Abbeel, and Sergey Levine. 2016. A connection between generative adversarial networks, inverse reinforcement learning, and energy-based models. arXiv preprint arXiv:1611.03852 (2016).
- [13] Chelsea Finn, Sergey Levine, and Pieter Abbeel. 2016. Guided cost learning: Deep inverse optimal control via policy optimization. In International Conference on Machine Learning . 49-58.
- [14] Yarin Gal and Zoubin Ghahramani. 2016. Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. In international conference on machine learning . 1050-1059.
- [15] Yong Ge, Hui Xiong, Zhi-hua Zhou, Hasan Ozdemir, Jannite Yu, and Kuo Chu Lee. 2010. Top-eye: Top-k evolving trajectory outlier detection. In Proceedings of the 19th ACM international conference on Information and knowledge management . ACM, 1733-1736.
- [16] Alex Graves. 2011. Practical variational inference for neural networks. In Advances in neural information processing systems . 2348-2356.
- [17] Manish Gupta, Jing Gao, Charu C Aggarwal, and Jiawei Han. 2014. Outlier detection for temporal data: A survey. IEEE Transactions on Knowledge and Data Engineering 26, 9 (2014), 2250-2267.
- [18] Zengyou He, Xiaofei Xu, and Shengchun Deng. 2003. Discovering cluster-based local outliers. Pattern Recognition Letters 24, 9-10 (2003), 1641-1650.
- [19] Weiming Hu, Xuejuan Xiao, Zhouyu Fu, Dan Xie, Tieniu Tan, and Steve Maybank. 2006. A system for learning statistical motion patterns. IEEE transactions on pattern analysis and machine intelligence 28, 9 (2006), 1450-1464.
- [20] Mrinal Kalakrishnan, Peter Pastor, Ludovic Righetti, and Stefan Schaal. 2013. Learning objective functions for manipulation. In Robotics and Automation (ICRA), 2013 IEEE International Conference on . IEEE, 1331-1336.
- [21] Hiroki Kanezashi, Toyotaro Suzumura, Dario Garcia-Gasulla, Min-hwan Oh, and Satoshi Matsuoka. 2018. Adaptive Pattern Matching with Reinforcement Learning for Dynamic Graphs. In 2018 IEEE 25th International Conference on High Performance Computing (HiPC) . IEEE, 92-101.
- [22] Diederik P Kingma, Tim Salimans, and Max Welling. 2015. Variational dropout and the local reparameterization trick. In Advances in Neural Information Processing Systems . 2575-2583.
- [23] Edwin M Knorr, Raymond T Ng, and Vladimir Tucakov. 2000. Distance-based outliers: algorithms and applications. The VLDB JournalâĂŤThe International Journal on Very Large Data Bases 8, 3-4 (2000), 237-253.
- [24] Jae-Gil Lee, Jiawei Han, and Xiaolei Li. 2008. Trajectory outlier detection: A partition-and-detect framework. In Data Engineering, 2008. ICDE 2008. IEEE 24th International Conference on . IEEE, 140-149.
- [25] Ce Li, Zhenjun Han, Qixiang Ye, and Jianbin Jiao. 2013. Visual abnormal behavior detection based on trajectory sparse reconstruction analysis. Neurocomputing 119 (2013), 94-100.
- [26] Fei Tony Liu, Kai Ming Ting, and Zhi-Hua Zhou. 2008. Isolation forest. In Data Mining, 2008. ICDM'08. Eighth IEEE International Conference on . IEEE, 413-422.
- [27] Christos Louizos and Max Welling. 2017. Multiplicative normalizing flows for variational bayesian neural networks. arXiv preprint arXiv:1703.01961 (2017).
- [28] Huimin Lu, Yujie Li, Shenglin Mu, Dong Wang, Hyoungseop Kim, and Seiichi Serikawa. 2017. Motor anomaly detection for unmanned aerial vehicles using reinforcement learning. IEEE internet of things journal 5, 4 (2017), 2315-2322.
- [29] David JC MacKay. 1992. A practical Bayesian framework for backpropagation networks. Neural computation 4, 3 (1992), 448-472.
- [30] Pankaj Malhotra, Anusha Ramakrishnan, Gaurangi Anand, Lovekesh Vig, Puneet Agarwal, and Gautam Shroff. 2016. Lstm-based encoder-decoder for multi-sensor

anomaly detection. arXiv preprint arXiv:1607.00148 (2016).

- [31] Larry M Manevitz and Malik Yousef. 2001. One-class SVMs for document classification. Journal of machine Learning research 2, Dec (2001), 139-154.
- [32] P Read Montague and Gregory S Berns. 2002. Neural economics and the biological substrates of valuation. Neuron 36, 2 (2002), 265-284.
- [33] Luis Moreira-Matias, Joao Gama, Michel Ferreira, Joao Mendes-Moreira, and Luis Damas. 2013. Predicting taxi-passenger demand using streaming data. IEEE Transactions on Intelligent Transportation Systems 14, 3 (2013), 1393-1402.
- [34] Radford M Neal. 1993. Bayesian learning via stochastic dynamics. In Advances in neural information processing systems . 475-482.
- [35] Gergely Neu and Csaba Szepesvári. 2012. Apprenticeship learning using inverse reinforcement learning and gradient methods. arXiv preprint arXiv:1206.5264 (2012).
- [36] Andrew Y Ng, Stuart J Russell, et al. 2000. Algorithms for inverse reinforcement learning.. In Proceedings of the 17th international conference on Machine learning . 663-670.
- [37] Min-hwan Oh, Peder A Olsen, and Karthikeyan Natesan Ramamurthy. 2019. Crowd Counting with Decomposed Uncertainty. arXiv preprint arXiv:1903.07427 (2019).
- [38] Ian Osband, Charles Blundell, Alexander Pritzel, and Benjamin Van Roy. 2016. Deep exploration via bootstrapped DQN. In Advances in Neural Information Processing Systems . 4026-4034.
- [39] Timothy J O'Shea, T Charles Clancy, and Robert W McGwier. 2016. Recurrent Neural Radio Anomaly Detection. arXiv preprint arXiv:1611.00301 (2016).
- [40] Fabian Pedregosa, Gaël Varoquaux, Alexandre Gramfort, Vincent Michel, Bertrand Thirion, Olivier Grisel, Mathieu Blondel, Peter Prettenhofer, Ron Weiss, Vincent Dubourg, et al. 2011. Scikit-learn: Machine learning in Python. Journal of machine learning research 12, Oct (2011), 2825-2830.
- [41] Claudio Piciarelli, Christian Micheloni, and Gian Luca Foresti. 2008. Trajectorybased anomalous event detection. IEEE Transactions on Circuits and Systems for video Technology 18, 11 (2008), 1544-1554.
- [42] Deepak Ramachandran and Eyal Amir. 2007. Bayesian inverse reinforcement learning. Urbana 51, 61801 (2007), 1-4.
- [43] Nathan D Ratliff, J Andrew Bagnell, and Martin A Zinkevich. 2006. Maximum margin planning. In Proceedings of the 23rd international conference on Machine learning . ACM, 729-736.
- [44] Eduardo Romera, Luis M Bergasa, and Roberto Arroyo. 2016. Need data for driver behaviour analysis? Presenting the public UAH-DriveSet. In Intelligent Transportation Systems (ITSC), 2016 IEEE 19th International Conference on . IEEE, 387-392.
- [45] Stuart Russell. 1998. Learning agents for uncertain environments. In Proceedings of the eleventh annual conference on Computational learning theory . ACM, 101103.
- [46] Mohammad Sabokrou, Mohsen Fayyaz, Mahmood Fathy, Zahra Moayed, and Reinhard Klette. 2018. Deep-anomaly: Fully convolutional neural network for fast anomaly detection in crowded scenes. Computer Vision and Image Understanding (2018).
- [47] John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz. 2015. Trust region policy optimization. In Proceedings of the 32nd International Conference on Machine Learning (ICML-15) . 1889-1897.
- [48] Rowland R Sillito and Robert B Fisher. 2008. Semi-supervised Learning for Anomalous Trajectory Detection.. In BMVC , Vol. 1. 035-1.
- [49] Nitish Srivastava, Geoffrey E Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. 2014. Dropout: a simple way to prevent neural networks from overfitting. Journal of machine learning research 15, 1 (2014), 1929-1958.
- [50] Umar Syed, Michael Bowling, and Robert E Schapire. 2008. Apprenticeship learning using linear programming. In Proceedings of the 25th international conference on Machine learning . ACM, 1032-1039.
- [51] Yulong Wang, Kun Qin, Yixiang Chen, and Pengxiang Zhao. 2018. Detecting anomalous trajectories and behavior patterns using hierarchical clustering from taxi GPS data. ISPRS International Journal of Geo-Information 7, 1 (2018), 25.
- [52] S Wu, B Moore, and M Shah. 2010. Chaotic invariants of lagrangian particle trajectories for anomaly detection in crowded scenes, in CVPR. (2010).
- [53] Markus Wulfmeier, Peter Ondruska, and Ingmar Posner. 2015. Maximum entropy deep inverse reinforcement learning. arXiv preprint arXiv:1507.04888 (2015).
- [54] Yu Zheng, Xing Xie, and Wei-Ying Ma. 2010. Geolife: A collaborative social networking service among user, location and trajectory. IEEE Data Eng. Bull. 33, 2 (2010), 32-39.
- [55] Yu Zheng, Lizhu Zhang, Xing Xie, and Wei-Ying Ma. 2009. Mining interesting locations and travel sequences from GPS trajectories. In Proceedings of the 18th international conference on World wide web . ACM, 791-800.
- [56] Brian D Ziebart, Andrew L Maas, J Andrew Bagnell, and Anind K Dey. 2008. Maximum Entropy Inverse Reinforcement Learning.. In AAAI , Vol. 8. Chicago, IL, USA, 1433-1438.

## A NEURAL NETWORK DETAILS

FNN-AE is a fully-connected feed forward autoencoder with 3 hidden layers with dimensions 64, 16, and 64. We use Relu activation functions for all hidden layers and linear activation at the output. LSTM-AE is a hierarchical model that consists of an encoder and a decoder, each of which is an LSTM. For our reward function, we use two hidden layers of 64 and 16, the network branches out independently to K fully connected output nodes. In all experiments, we used K = 10. For policy optimization, we use TRPO method [47] for which we used openAI baseline model.

## B PER USER EXPERIMENTS ON GEOLIFE-GPS

We picked 10 users with largest number of trajectories. Among them, we filtered trajectories that are shorter than 100 time-steps. After training, for each user, we injected (and/or identified) anomalies so that the total proportions of anomalies in the test datasets are 5% (Table 2) and 10% (Table 3) respectively in each independent experiments. The results are shown in Tables 2 and 3.

## C BAYESIAN NEURAL NETWORK

In this section, we briefly discuss the work on Bayesian neural network. Let D = {( x i , y i )} N i = 1 be a collection of realizations of i.i.d random variables, where x i is an image, y i is a corresponding density map, and N denotes the sample size. In Bayesian neural network framework, rather than thinking of the weights of the network as fixed parameters to be optimized over, it treats them as random variables, and so we place a prior distribution p ( θ ) over the weights of the network θ ∈ θ . This results in the posterior distribution

<!-- formula-not-decoded -->

While this formalization is simple, the learning is often challenging because calculating the posterior p ( θ |D) requires an integration with respect to the entire parameter space Θ for which a closed form often does not exist. [29] proposed a Laplace approximation of the posterior. [34] introduced the Hamiltonian Monte Carlo, a Markov Chain Monte Carlo (MCMC) sampling approach using Hamiltonian dynamics, to learn Bayesian neural networks. This yields a principled set of posterior samples without direct calculation of the posterior but it is computationally prohibitive. Another Bayesian method is variational inference [5, 16, 27] which approximates the posterior distribution by a tractable variational distribution q η ( θ ) indexed by a variational parameter η . The optimal variational distribution is the closest distribution to the posterior among the pre-determined family Q = { q η ( θ )} . The closeness is often measured by the Kullback-Leibler (KL) divergence between q η ( θ ) and p ( θ |D) . While these Bayesian neural networks are the state of art at estimating predictive uncertainty, these require significant modifications to the training procedure and are computationally expensive compared to standard (non-Bayesian) neural networks

[14] proposed using Monte Carlo dropout to estimate predictive uncertainty by using dropout at test time. There has been work on approximate Bayesian interpretation of dropout [14, 22]. Specifically, [14] showed that Monte Carlo dropout is equivalent to a variational approximation in a Bayesian neural network. With this justification, they proposed a method to estimate predictive uncertainty through variational distribution. Monte Carlo dropout is relatively simple to implement leading to its popularity in practice. Interestingly, dropout may also be interpreted as ensemble model combination [49] where the predictions are averaged over an ensemble of neural networks. The ensemble interpretation seems more plausible particularly in the scenario where the dropout rates are not tuned based on the training data, since any sensible approximation to the true Bayesian posterior distribution has to depend on the training data. This interpretation motivates the investigation of ensembles as an alternative solution for estimating predictive uncertainty. Despite the simplicity of dropout implementation, we were not able to produce satisfying confidence interval for our crowd counting problem. Hence we consider a simple non-parametric bootstrap of functions which we discuss in the following section.

Table 2: Evaluation on GeoLife-GPS dataset with anomaly rate 5%

|      | LOF   | LOF   | LOF   | One-class SVM   | One-class SVM   | One-class SVM   | FNN-AE   | FNN-AE   | FNN-AE   | LSTM-AE   | LSTM-AE   | LSTM-AE   | IRL-AD   | IRL-AD   | IRL-AD   | IRL-ADU   | IRL-ADU   | IRL-ADU   |
|------|-------|-------|-------|-----------------|-----------------|-----------------|----------|----------|----------|-----------|-----------|-----------|----------|----------|----------|-----------|-----------|-----------|
| Data | R     | P     | F 1   | R               | P               | F 1             | R        | P        | F 1      | R         | P         | F 1       | R        | P        | F 1      | R         | P         | F 1       |
| U 1  | 0.500 | 0.071 | 0.125 | 0.500           | 0.076           | 0.132           | 0.700    | 0.100    | 0.175    | 0.700     | 0.113     | 0.194     | 0.800    | 0.148    | 0.250    | 0.800     | 0.178     | 0.291     |
| U 2  | 0.700 | 0.091 | 0.161 | 0.600           | 0.078           | 0.138           | 0.800    | 0.113    | 0.198    | 0.600     | 0.120     | 0.200     | 0.700    | 0.149    | 0.246    | 0.700     | 0.167     | 0.269     |
| U 3  | 0.800 | 0.095 | 0.170 | 0.800           | 0.114           | 0.200           | 0.500    | 0.082    | 0.141    | 0.700     | 0.135     | 0.226     | 0.800    | 0.131    | 0.225    | 0.800     | 0.157     | 0.262     |
| U 4  | 0.600 | 0.081 | 0.143 | 0.600           | 0.072           | 0.129           | 0.600    | 0.094    | 0.162    | 0.700     | 0.113     | 0.194     | 0.800    | 0.163    | 0.271    | 0.800     | 0.157     | 0.262     |
| U 5  | 0.200 | 0.037 | 0.063 | 0.300           | 0.037           | 0.065           | 0.700    | 0.104    | 0.182    | 0.800     | 0.114     | 0.200     | 0.900    | 0.170    | 0.286    | 0.900     | 0.220     | 0.352     |
| U 6  | 0.600 | 0.083 | 0.146 | 0.700           | 0.119           | 0.203           | 0.500    | 0.069    | 0.122    | 0.600     | 0.086     | 0.150     | 0.800    | 0.174    | 0.286    | 0.800     | 0.182     | 0.296     |
| U 7  | 0.800 | 0.098 | 0.174 | 0.700           | 0.097           | 0.171           | 0.900    | 0.127    | 0.222    | 0.900     | 0.145     | 0.250     | 1.000    | 0.175    | 0.299    | 1.000     | 0.217     | 0.357     |
| U 8  | 0.600 | 0.080 | 0.141 | 0.600           | 0.074           | 0.132           | 0.700    | 0.113    | 0.194    | 0.700     | 0.101     | 0.177     | 0.800    | 0.174    | 0.286    | 0.800     | 0.174     | 0.286     |
| U 9  | 0.800 | 0.118 | 0.205 | 0.500           | 0.068           | 0.120           | 0.500    | 0.094    | 0.159    | 0.600     | 0.102     | 0.174     | 0.700    | 0.140    | 0.233    | 0.700     | 0.171     | 0.275     |
| U 10 | 0.600 | 0.074 | 0.132 | 0.400           | 0.058           | 0.101           | 0.600    | 0.092    | 0.160    | 0.600     | 0.090     | 0.155     | 0.800    | 0.157    | 0.262    | 0.800     | 0.160     | 0.267     |
| Avg  | 0.620 | 0.083 | 0.146 | 0.570           | 0.079           | 0.139           | 0.650    | 0.099    | 0.171    | 0.690     | 0.112     | 0.192     | 0.810    | 0.158    | 0.264    | 0.810     | 0.178     | 0.292     |

Table 3: Evaluation on GeoLife-GPS dataset with anomaly rate 10%

|      | LOF   | LOF   | LOF   | One-class SVM   | One-class SVM   | One-class SVM   | FNN-AE   | FNN-AE   | FNN-AE   | LSTM-AE   | LSTM-AE   | LSTM-AE   | IRL-AD   | IRL-AD   | IRL-AD   | IRL-ADU   | IRL-ADU   | IRL-ADU   |
|------|-------|-------|-------|-----------------|-----------------|-----------------|----------|----------|----------|-----------|-----------|-----------|----------|----------|----------|-----------|-----------|-----------|
| Data | R     | P     | F 1   | R               | P               | F 1             | R        | P        | F 1      | R         | P         | F 1       | R        | P        | F 1      | R         | P         | F 1       |
| U 1  | 0.500 | 0.119 | 0.192 | 0.500           | 0.139           | 0.217           | 0.500    | 0.125    | 0.200    | 0.600     | 0.162     | 0.255     | 0.800    | 0.296    | 0.432    | 0.800     | 0.308     | 0.444     |
| U 2  | 0.500 | 0.139 | 0.217 | 0.600           | 0.146           | 0.235           | 0.600    | 0.158    | 0.250    | 0.700     | 0.179     | 0.286     | 0.900    | 0.300    | 0.450    | 0.900     | 0.346     | 0.500     |
| U 3  | 0.700 | 0.194 | 0.304 | 0.700           | 0.156           | 0.255           | 0.800    | 0.186    | 0.302    | 0.800     | 0.250     | 0.381     | 0.800    | 0.364    | 0.500    | 0.800     | 0.421     | 0.551     |
| U 4  | 0.700 | 0.167 | 0.269 | 0.600           | 0.162           | 0.255           | 0.700    | 0.156    | 0.255    | 0.700     | 0.171     | 0.275     | 0.900    | 0.360    | 0.514    | 0.800     | 0.348     | 0.485     |
| U 5  | 0.300 | 0.086 | 0.133 | 0.500           | 0.106           | 0.175           | 0.500    | 0.139    | 0.217    | 0.500     | 0.179     | 0.263     | 0.700    | 0.259    | 0.378    | 0.700     | 0.259     | 0.378     |
| U 6  | 0.600 | 0.130 | 0.214 | 0.600           | 0.182           | 0.279           | 0.500    | 0.135    | 0.213    | 0.700     | 0.194     | 0.304     | 0.800    | 0.308    | 0.444    | 0.800     | 0.381     | 0.516     |
| U 7  | 0.800 | 0.211 | 0.333 | 0.700           | 0.212           | 0.326           | 0.800    | 0.250    | 0.381    | 0.900     | 0.237     | 0.375     | 0.900    | 0.310    | 0.462    | 0.900     | 0.321     | 0.474     |
| U 8  | 0.700 | 0.171 | 0.275 | 0.700           | 0.212           | 0.326           | 0.800    | 0.195    | 0.314    | 0.800     | 0.222     | 0.348     | 0.900    | 0.333    | 0.486    | 0.900     | 0.429     | 0.581     |
| U 9  | 0.800 | 0.195 | 0.314 | 0.800           | 0.222           | 0.348           | 0.900    | 0.243    | 0.383    | 0.900     | 0.250     | 0.391     | 1.000    | 0.370    | 0.541    | 1.000     | 0.417     | 0.588     |
| U 10 | 0.500 | 0.125 | 0.200 | 0.500           | 0.139           | 0.217           | 0.500    | 0.143    | 0.222    | 0.600     | 0.150     | 0.240     | 0.800    | 0.242    | 0.372    | 0.800     | 0.296     | 0.432     |
| Avg  | 0.610 | 0.154 | 0.245 | 0.620           | 0.168           | 0.263           | 0.660    | 0.173    | 0.274    | 0.720     | 0.199     | 0.312     | 0.850    | 0.314    | 0.458    | 0.840     | 0.353     | 0.495     |