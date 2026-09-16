# Progress: Section B2 — Evaluating Synthetic Image Quality & Long-Tail/Rare-Class

## sec:lit_synthetic_evaluation (Evaluating Synthetic Image Quality)
- [x] zhouHYPEBenchmarkHuman2019 — HYPE, structured/blind human perceptual evaluation protocol. Read in full. Key facts: two variants HYPEtime (adaptive staircase, perceptual-threshold ms) and HYPE∞ (error rate on 50 real/50 fake, no time limit, cheaper); blind (model identity hidden), multi-rater (30 MTurk evaluators/model), reliability via bootstrapped 95% CI + ANOVA/Tukey separability tests; found HYPE scores NOT correlated with FID (Spearman ρ=-0.029) on faces, only low-moderate correlation with KID/FID on ImageNet-5 -- i.e. paper's own evidence that automatic proxies and human judgment can diverge, motivating triangulation across axes.
- [x] heuselGANsTrainedTwo2018 — FID. Read in full. Computes Fréchet distance between two multivariate Gaussians fit to Inception-v3 pool3 (2048-d) activations of real vs. generated image sets: FID=||mu_r-mu_g||^2 + Tr(C_r+C_g-2(C_r C_g)^{1/2}). Introduced alongside TTUR; shown more sensitive than IS to mode collapse, added noise/blur/artifacts (monotonic degradation curves), and consistent with increasing disturbance level. Known caveat (from paper itself): needs a sufficiently large sample size to estimate the Gaussian statistics reliably (biased at small n).
- [x] binkowskiDemystifyingMMDGANs2021 — KID. Read in full. Defines Maximum Mean Discrepancy between Inception representations using a polynomial kernel; unlike FID, the MMD^2 estimator is unbiased and does not assume a Gaussian representation, so it does not require large sample sizes to be reliable -- explicit motivation given is fixing FID's large-sample-size assumption/bias.
- [x] hesselCLIPScoreReferencefreeEvaluation2021 — CLIPScore. Read in full. Reference-free image-caption/prompt alignment: cosine similarity between CLIP image embedding and CLIP text embedding of the prompt/caption, rescaled (2.5x, clipped at 0) -- CLIPScore(I,C)=2.5*max(cos(E_I,E_C),0). Correlates well with human judgment of image-caption compatibility without needing reference captions/images, unlike prior reference-based metrics.
- [x] ravuriClassificationAccuracyScore2019 — Classification Accuracy Score (CAS). Read in full. Trains a classifier ONLY on generated (class-conditional) samples, evaluates on real held-out test set (TSTR pattern); score = real-test top-1/top-5 accuracy of the classifier trained purely on synthetic data. Found CAS can diverge sharply from FID/IS rankings (a BigGAN-deep variant with excellent FID/IS scored much worse on CAS), demonstrating downstream utility is a distinct signal not captured by distributional/diversity metrics alone -- directly supports thesis's inclusion of a downstream axis rather than relying on FID/CLIP alone.
- [x] salimansImprovedTechniquesTraining2016 — Inception Score. Read in full. IS = exp(E_x[KL(p(y|x) || p(y))]) using Inception model's conditional label distribution -- rewards both confident per-image predictions (low entropy p(y|x)) and diverse predictions across the sample set (high entropy marginal p(y)); introduced as a way to avoid relying on human MTurk annotators, computed over >=50k samples; predecessor to FID (paper's own Sec 6.2 already flags that directly optimizing IS produces adversarial-example-like artifacts rather than genuine quality).

## sec:lit_synthetic_long_tail (Synthetic Data for Long-Tail and Rare-Class Problems)
- [x] chawlaSMOTESyntheticMinority2002 — SMOTE. Read in full. Over-samples minority class by synthesizing new points along line segments joining a minority sample to its k (=5) nearest minority-class neighbors IN FEATURE SPACE (interpolation between existing feature vectors), combined with majority-class under-sampling; evaluated via ROC/AUC on C4.5, Ripper, Naive Bayes on tabular/pixel data (mammography, etc.), not images. Contrast case for the thesis: interpolates within the existing feature manifold rather than synthesizing new pixel-level appearance content, so it cannot supply novel visual information (pose/background/lighting variation) the way generative image synthesis can -- it only rebalances what's already observed.
- [x] cuiClassBalancedLossBased2019 — Class-Balanced Loss. Read in full. Defines "effective number of samples" E_n=(1-beta^n)/(1-beta) via a random-covering argument (diminishing marginal information from near-duplicate/overlapping samples), then reweights any loss (softmax CE, sigmoid CE, focal) by (1-beta)/(1-beta^{n_y}) for ground-truth class y; beta in [0,1) interpolates between no reweighting (beta=0) and inverse-frequency reweighting (beta->1). Explicitly a re-weighting alternative to re-sampling; validated on long-tailed CIFAR, ImageNet, and iNaturalist.
- [x] kangDecouplingRepresentationClassifier2020 — Decoupling. Read in full. Key finding: decoupling representation learning from classifier learning shows instance-balanced sampling learns the BEST representations, and long-tail performance is recovered by re-adjusting only the classifier afterward (re-training classifier with class-balanced sampling, nearest-class-mean, or tau-normalization of classifier weight norms) -- outperforms jointly-trained re-sampling/re-weighting on ImageNet-LT, Places-LT, iNaturalist. Supports framing re-sampling/re-weighting as intervening at the classifier/decision-boundary level rather than fixing the underlying data-scarcity problem.
- [x] caoLearningImbalancedDatasets2019 — LDAM. Read in full. Derives a per-class margin from a generalization-error bound (margin gamma_j should scale ~ n_j^{-1/4}), yielding a label-distribution-aware margin loss that enforces larger margins for minority classes; combined with a "deferred re-weighting" schedule (train first with plain loss, re-weight only in a later stage) to avoid re-weighting's early-training optimization difficulties. Explicitly a re-weighting/re-margining approach operating purely on the loss function -- orthogonal to re-sampling, and to any strategy (like synthetic supplementation) that changes what data exists rather than how it's weighted.
- [x] zhangDeepLongTailedLearning2023 — Long-tail survey (TPAMI 2023). Read in full. Taxonomizes methods into class re-balancing (re-sampling, class-sensitive re-weighting/re-margining, logit adjustment), information augmentation (transfer learning; data augmentation), and module improvement. Directly relevant: explicitly distinguishes "non-transfer augmentation" (SMOTE-style feature-space interpolation, mixup, Gaussian-prior feature synthesis e.g. FASA/MetaSAug) from "head-to-tail transfer augmentation" (M2m: perturbation-translating head-class samples into tail-class ones; RSG: feature-center displacement) -- both operate in feature space on top of existing samples, NOT by synthesizing new raw images from an external generative prior. Explicitly flags "how to better conduct data augmentation for long-tailed learning is still an open question" as unresolved. Strong grounding for framing image-level generative synthesis as a distinct, complementary axis to these existing re-balancing/augmentation families.
- [~] weiFineGrainedImageAnalysis2022 — DROPPED for this subsection. Re-checked the source file for long-tail/imbalance/data-scarcity/augmentation content: zero in-text matches for "long-tail", "imbalance", "scarcity", "few-shot", "augmentation", "synthetic", "generative" anywhere in the extracted markdown (740 lines) despite two long-tailed-recognition papers appearing in its reference list ([211] BBN, [212] large-scale long-tailed recognition) -- those refs are never actually cited in the extracted body text, likely because the citing passage (a "future directions"-type discussion) was not captured by the PDF extraction. Since the mandatory-grounding rule requires the full text to actually support the claim, and it does not here, this citation is NOT used in sec:lit_synthetic_long_tail (it was already used in Section A for a different, properly-grounded claim).

## Bib verification (grep, all confirmed present in references.bib)
All 12 keys confirmed present via grep of `research/literature/references.bib`:
zhouHYPEBenchmarkHuman2019 (L465), heuselGANsTrainedTwo2018 (L480), binkowskiDemystifyingMMDGANs2021 (L494),
hesselCLIPScoreReferencefreeEvaluation2021 (L508), ravuriClassificationAccuracyScore2019 (L525),
salimansImprovedTechniquesTraining2016 (L539), chawlaSMOTESyntheticMinority2002 (L583),
cuiClassBalancedLossBased2019 (L953), kangDecouplingRepresentationClassifier2020 (L969),
caoLearningImbalancedDatasets2019 (L983), zhangDeepLongTailedLearning2023 (L997), weiFineGrainedImageAnalysis2022 (L935).
All 12 have source files in research/literature/sources/.

## Internal docs checked
- docs/synthetic-model-comparison/06_evaluation-methodology.md — FOUND, read in full (grounds three-axis structure).
- thesis/manuscript/chapters/3-Methods_and_Implementation/33-Synthetic_Data_Supplementation.tex — read for band-structure context.
- research/literature/drafts/B1-generative-and-simtoreal.tex — read for style consistency (not edited).

## Status
Both subsections written to `research/literature/drafts/B2-evaluation-and-longtail.tex`:
- `\subsection{Evaluating Synthetic Image Quality}` (sec:lit_synthetic_evaluation) -- DONE
- `\subsection{Synthetic Data for Long-Tail and Rare-Class Problems}` (sec:lit_synthetic_long_tail) -- DONE

## Citekeys used

sec:lit_synthetic_evaluation (6 citekeys, all verified present in references.bib and grounded against their source .md files):
- salimansImprovedTechniquesTraining2016 (Inception Score)
- heuselGANsTrainedTwo2018 (FID)
- binkowskiDemystifyingMMDGANs2021 (KID)
- zhouHYPEBenchmarkHuman2019 (HYPE, blind multi-rater human protocol)
- hesselCLIPScoreReferencefreeEvaluation2021 (CLIPScore)
- ravuriClassificationAccuracyScore2019 (Classification Accuracy Score / TSTR-style downstream utility)

sec:lit_synthetic_long_tail (6 citekeys, all verified present in references.bib and grounded against their source .md files):
- zhangDeepLongTailedLearning2023 (long-tail survey; cited twice, once via \citeauthor)
- chawlaSMOTESyntheticMinority2002 (SMOTE, contrast case)
- cuiClassBalancedLossBased2019 (class-balanced loss, re-weighting)
- caoLearningImbalancedDatasets2019 (LDAM, re-margining)
- kangDecouplingRepresentationClassifier2020 (decoupling, classifier-level re-balancing)

## Softened / dropped claims
- Inter-rater agreement (Cohen's/Fleiss' kappa) statistics: NOT cited anywhere, per the prior agent's confirmed-absent finding (no matching bib entries for Cohen 1960/Fleiss 1971/Landis & Koch 1977/Krippendorff). The evaluation subsection describes HYPE's actual reliability apparatus instead (bootstrapped 95% CIs, ANOVA/Tukey separability tests), which is what the source paper actually reports, rather than naming an uncited statistic.
- weiFineGrainedImageAnalysis2022: considered for sec:lit_synthetic_long_tail per the task brief ("reuse if genuinely relevant") but DROPPED after re-reading the full source file -- it contains zero in-text discussion of long-tail/imbalance/data-scarcity/augmentation (verified by grep for "long-tail", "imbalance", "scarcity", "few-shot", "augmentation", "synthetic", "generative" against the full 740-line extraction); two long-tail papers appear only in its reference list ([211], [212]) but are never cited in the extracted body text. Not used in this subsection to avoid an ungrounded claim.
- No numeric claims from the CAS, HYPE, FID, or KID papers were altered from what the source files report; all inline numbers (e.g. CAS's 27.9%/41.6% accuracy gaps, HYPE's 50.7% HYPE∞ and rho=-0.029, CLIPScore's tau_c=51.2 vs 44.9, LDAM's gamma_j ~ n_j^{-1/4}) were checked against the source .md files before use.

## Unresolved / flagged for the caller
- None. Both subsections are complete and fully grounded in verified citekeys.
