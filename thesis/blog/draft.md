# Too Few Photos of the Binturong? Training a Wildlife Detector for a Binocular with AI-Generated Images

<!--
Draft v2, 2026-11-11. About 3,900 words of body text (v1: about 2,900 by the same count).
Label: Artificial Intelligence Expert. Tags: Computer Vision, Generative AI, Edge AI, Object Detection.
[Figure N: file] blocks are placeholders. media.md holds the WordPress title, alt text and
caption of each one; visuals.md the concept. All visuals are static images (no clips).
Every number is traced in the "Sources" table at the end of this file (not part of the article).

Writing rules of v2 (after the review of v1):
1. A claim is as strong as its evidence. No "nothing", "never", "no trend" from a handful of
   models or single runs; say how many models and runs a statement rests on.
2. Headings and bold lines state the finding. No slogans that the article's own numbers contradict.
3. What we did and why is told as it happened. No invented surprises, mistakes or motives.
4. Every point ends in something a reader can use. An observation without a consequence is cut
   or attached to the point it illustrates.
5. Confounds are named where the claim is made, not in a later caveat.
6. TL;DR, body and takeaways say the same thing.
-->

**TL;DR: Key findings**

- **Both the generator and the prompt decide how useful generated images are.** In a controlled comparison, images from API models trained a clearly better detector than images from models we ran locally. Cutting the prompt of two API models to 75 tokens cost each of them almost half of that accuracy.
- **Generated images replaced half of a small training set at almost no cost, but not all of it.** Classes trained on 100 real and 100 generated images reached an mAP of 0.588 on real photos, against 0.591 for classes with 200 real images. Classes trained on generated images alone reached 0.300.
- **Scores on generated test images are far too high.** The classes that score 0.99 on generated test images score 0.30 on real photos.
- **One nano-scale detector can do the work of two networks.** A YOLO26n with 2.7 million parameters is about 13 times faster than a detector-plus-classifier pipeline and about 83 times faster than the large reference ensemble, and loses 0.07 mAP against that ensemble on real photos.

(mAP, mean average precision, is the usual accuracy measure for object detectors. It runs from 0 to 1 and rewards finding the animal, drawing the box in the right place and naming the right species.)

---

A brown shape moves at the edge of the forest, two hundred metres away. You lift the binoculars and get a good look for maybe ten seconds: long body, dark fur, a tail as long as the animal itself. What *is* that?

The [AX Visio](https://www.swarovskioptik.com/) by Swarovski Optik is a binocular built to answer that question. It has a camera and a neural processor on board, identifies what you are looking at, and shows the name in the eyepiece, without a phone and without a network connection. For birds this already works. A model for mammals would have to tell 225 species apart, on a [Qualcomm QCS605](https://www.qualcomm.com/products/technology/processors/application-processors/qcs605) chip from 2018, while the animal is still in view.

In my Master's thesis, carried out at inovex together with Swarovski Optik, I asked how to build that detector under two constraints that show up in many computer vision projects: the hardware is small, and for a good part of the classes there are hardly any photos.

[Figure 1: figures/hero/hero_b.jpg (featured image)]

## Two problems: rare animals and a small chip

The first problem is data. Lions and red foxes have been photographed millions of times. The [binturong](https://en.wikipedia.org/wiki/Binturong), a shaggy tree-dwelling mammal from Southeast Asia, has not. When we started with the openly licensed images on [GBIF](https://www.gbif.org/), 114 of our 225 classes had fewer than 100 photos, and 30 had none at all.

The second problem is compute. The fastest route to a working product is a pipeline of existing models: a fine-tuned [YOLOv5s](https://github.com/ultralytics/yolov5) detects a single "animal" class and hands each crop to [SpeciesNet](https://github.com/google/cameratrapai), Google's open classifier for camera-trap images. That works, but two networks run one after the other, and the classifier runs once per animal. A herd of zebras costs more than a single fox.

So the thesis question was: **can one small detector do both jobs, finding the animal and naming the species, if part of its training data is generated?**

## Collecting real photos, and what you throw away

Before generating anything, we collected what exists. Five sources with licences that allow commercial use ([iNaturalist](https://www.inaturalist.org/), GBIF, [Wikimedia Commons](https://commons.wikimedia.org/), [Open Images](https://storage.googleapis.com/openimages/web/index.html) and stock imagery) yielded about 580,000 candidate images, 400,000 of them from iNaturalist alone.

A folder named after a species holds a lot of things that are not that species. Our filter chain threw out a Tutankhamun wall painting filed under "cheetah", a university's tiger mascot, a wooden aardvark mask from a museum and a range map of the African buffalo. After metadata checks, blur heuristics, an animal detector, a vision-language model as a second opinion and a species classifier, 465,242 images were left, and the per-class caps reduced them to a final set of 222,073 real photos. We held back 63,802 of them as the test set.

[Figure 2: figures/rejects.jpg]

More sources helped the middle of the distribution, not the tail. 29 classes still had fewer than 100 images. The 50 rarest classes averaged about 67 usable photos each, which is not enough to train on and barely enough to test with. The binturong is one of them, with 36.

## Which image generator? A controlled comparison

The obvious idea in 2026 is to generate the missing photos. In our first attempts, the choice of generator made a visible difference. Asked for a binturong, three open-weight models running on our own GPU all drew something closer to a small bear. Gemini, called through an API, drew a binturong.

[Figure 3: figures/binturong.jpg]

By eye, the verdict was clear: the API model was far better than the local ones. We wanted to back that impression with numbers, for two reasons. An impression is hard to defend for animals that the person judging has never seen. And it does not tell you how much better one generator is than another, or which of two generators is better when both produce convincing images. So we set up a comparison in which the generator was the only thing that changed:

- ten generator configurations: two API models, [gpt-image-2](https://platform.openai.com/docs/guides/image-generation) and [Gemini](https://ai.google.dev/gemini-api/docs/image-generation), in two variants each, and six open-weight models running locally, among them [Stable Diffusion 3.5](https://huggingface.co/stabilityai/stable-diffusion-3.5-large), [FLUX.2 klein](https://huggingface.co/black-forest-labs) and [HiDream-I1](https://huggingface.co/HiDream-ai/HiDream-I1-Full),
- the same twelve species, from lion to aye-aye, with 100 images each,
- the same [YOLO26n](https://docs.ultralytics.com/models/yolo26/) detector trained on each set of 1,200 images with the same schedule, two or three times per set,
- scored on the same 9,742 **real** photos.

[Figure 4: plots/out/ranking.png]

The numbers confirm the impression. With its full prompt, every API model beat every local model, and by a wide margin: the weakest API configuration reached an mAP of 0.095, the best local model, Stable Diffusion 3.5 Large, 0.055. (These absolute values are low because each detector saw only 1,200 images. They rank the generators and say nothing about the accuracy of the final model.)

The numbers also show what the eye could not. Among the API models, the two quality settings of gpt-image-2 lead with 0.134 and 0.126, ahead of Gemini 3.1 Flash Image with 0.105 and Gemini 3.1 Flash-Lite with 0.095. The differences within each pair are smaller than the spread between training runs, so the comparison cannot separate the two variants of either model. gpt-image-2 is ahead of Gemini, clearly so against Flash-Lite. The three runs of Gemini Flash spread widely (between 0.064 and about 0.13), which makes its distance to gpt-image-2 less certain.

One consequence for the rest of this article: our complete synthetic dataset was generated with Gemini 3.1 Flash Image, which was the easiest to integrate into our setup. In this comparison gpt-image-2 scored about a quarter higher on average. We did not repeat the later experiments with it, but the results below were probably not obtained with the best generator available, and gpt-image-2 would likely have done somewhat better.

Three further observations are worth passing on.

### 1. Price is a rough guide, not a precise one

The coarse pattern holds: the paid API models gave better training images than the free local ones. Beyond that, a higher price did not buy a better result in the cases we could compare. The medium quality setting of gpt-image-2 costs about twice as much per image as the low setting ($0.033 against $0.017 at batch prices), and the two are tied. The same is true of Gemini Flash and the cheaper Flash-Lite.

For the local models, the price is GPU time. HiDream-I1 took the longest, almost 48 GPU-hours for its 1,200 images, and came second to last. The fastest model, at under a second per image, came last. The four in between took 24 to 54 seconds per image, and the two best of them differ by a factor of two in time and not at all in result. Six models are too few to say whether there is a relationship. They are enough to say that generation time alone would not have picked the best local model.

For a project this means: price narrows the field, and the final choice between similar candidates needs a measurement.

### 2. A long prompt and a strong model both matter

Our prompts were long. Each one described the species, its distinguishing marks, the habitat, the pose and the camera position, at about 5,900 tokens on average. The API models read all of it. The text encoders of the local models stop at 75 to 512 tokens. So the ranking above compares two things at once: the models and the amount of description they were able to take in.

To separate the two, we gave two of the API models the same short prompt of at most 75 tokens. Both lost almost half of their accuracy: gpt-image-2 dropped from 0.134 to 0.074, Gemini Flash-Lite from 0.095 to 0.049. Confusion between the three zebra species nearly doubled at the same time. The long prompt evidently carries details that make a Grévy's zebra a Grévy's zebra.

The model still carries a lot of weight. With the short prompt, gpt-image-2 stays ahead of every local model, including those that read up to 512 tokens, while Gemini Flash-Lite lands in the middle of the local field. Part of the gap between API and local models is therefore prompt capacity, and part is the model. We tested the short prompt on two models and could not test a long one on the local models, so we cannot put a number on the two shares.

[Figure 5: figures/lion_prompts.jpg]

For a project this means: choose a generator that can read a detailed description, and then write one.

### 3. For the rare species, the comparison cannot rank the generators

The ranking comes almost entirely from the well-photographed species among the twelve, such as the American black bear and the zebras. For five of the six rare species, every generator scores below 0.05, and the differences between them are too small to interpret. This is a limit of the measurement and not a verdict on the images: 100 generated images per species and a small real test set do not give enough resolution.

The kinkajou shows what that means in practice. It is a golden-brown relative of the raccoon with a plain tail. FLUX.2 klein consistently drew it with a ringed tail and a spotted coat, closer to a genet, while two other local models drew it correctly from the same prompt. All three score close to zero on the kinkajou. The measurement does not register an error that anyone with a reference photo sees at once.

[Figure 6: figures/kinkajou.jpg]

So the numbers and the eye cover different ground. The numbers rank generators on species with enough real test photos, and they can separate candidates that all look good. For the rare species, which are the reason for generating images in the first place, we had nothing better than comparing samples with reference photos. A structured version of that check, with several raters who do not know which generator made which image, was planned and not carried out. It is what we would add first.

One more observation concerns the method. After a single training run per generator, Gemini Flash stood at 0.064, well behind its cheaper sibling. With a second run its average rose to 0.090 and with a third to 0.105, and the order reversed. With single runs we would have reported a difference that is not there.

The complete synthetic dataset, 23,850 images of all 225 classes, cost about $785 at list price.

## How much synthetic data is too much? The band experiment

To measure what generated images are worth in training, we sorted the 225 classes by the number of real photos available and cut them into four bands. The 50 classes with the fewest photos form Band A. Their few real photos are all needed for testing, so Band A trains on 200 generated images per class and nothing else. The next 26 classes form Band B, which trains on 100 real and 100 generated images. Band C, the 26 classes after that, trains on 200 real photos. Bands A to C thus share the same budget of 200 training images per class and differ in where those images come from. They also contain different species, which limits how far the comparison can be pushed. The remaining 122 well-photographed classes form Band D, which trains on up to 1,500 real photos per class and serves as the reference.

[Figure 7: plots/out/bands.png]

On real test photos, the YOLO26n detector reaches an mAP of 0.588 on Band B and 0.591 on Band C. Band A, trained on generated images only, reaches 0.300, and the data-rich Band D 0.742.

For YOLO26n, the classes with half of their training images generated did as well as the classes with real images only. The large reference model showed the same pattern (0.638 against 0.639). The older YOLOv5s did lose something (0.341 against 0.386). Each model was trained once and the two bands hold different species, so we read this as good evidence that generated images can stand in for a part of a small real training set, and not as proof that they are equivalent.

Band A reaches about half the score of Bands B and C. Two things differ here at once: the training images are all generated, and the species are the rarest ones, with few real test photos. We cannot say how much of the drop each one explains. What the result does show is that generated images alone teach the detector something that transfers to real photos. Without them, these 50 classes would be missing from the model.

Inside Band A the spread is large. The striped hyena, trained on generated images only, reaches an AP of 0.48 on real photos. The mangabeys, a genus of monkeys with several quite different-looking species, stay near zero. One possible explanation is that animals with a uniform appearance are easier to generate convincingly than varied ones. We did not test it.

## A generated test set scores far too high

Rare classes are hard to evaluate for the same reason they are hard to train: some of ours have six real test photos. To get a stable number for them, we generated a balanced test set of 50 images per class in addition to the real one, and compared what the two say.

[Figure 8: plots/out/synthetic_vs_real.png]

On the generated test images, Band A scores 0.99. On real photos, the same classes score 0.30. For YOLO26n the generated score is higher in every band, and the gap is widest where the training images were generated too. Even the off-the-shelf reference model, which was not trained on any of our images, scores about 0.2 mAP higher on the generated set.

We see two likely reasons. The generated test images are easier: they show one clearly visible animal each, while the real photos contain 1.44 animals on average, along with backlight and bushes. And a model trained on images from one generator is tested here on more images from the same generator.

A generated test set therefore measures how well the model handles generated images, and that is not what the binocular will see. We did not judge any model on generated images alone. The thesis reports the real-photo result next to every figure that includes generated test images, and all numbers in this article are real-photo numbers unless stated otherwise.

## Making it small: one nano detector instead of two networks

With the dataset fixed, three candidates competed. The first is the large reference: [MegaDetector](https://github.com/agentmorris/MegaDetector), an open detector built for camera traps, finds the animals, and SpeciesNet names each crop. Together they have about 193 million parameters. We ran it once off the shelf and once with SpeciesNet fine-tuned on our 225 classes. The other two candidates are single detectors that find and name the animal in one pass, both trained on the same data with all 225 classes: YOLOv5s with 7.6 million parameters, and the newer YOLO26n with 2.7 million.

To measure speed, we ran every model on a [Raspberry Pi 400](https://www.raspberrypi.com/products/raspberry-pi-400/), whose CPU is a slightly slower stand-in for the CPU cores of the QCS605. All models ran in [ONNX Runtime](https://onnxruntime.ai/) with four threads at full 32-bit precision, and the clock ran from the JPEG file to the final boxes.

[Figure 9: plots/out/latency_vs_map.png]

The chart puts both measurements together. Up means more accurate, left means faster, and the time axis is logarithmic: each gridline to the right is about three times slower. The best place is the top left.

The fine-tuned ensemble is the most accurate model with an mAP of 0.599, but it needs about 35 seconds per frame, an estimate composed from the measured times of its two networks. MegaDetector alone takes 31 of them. By the time it answers, the animal is gone. Its score is also favoured by the test set, because MegaDetector drew the boxes in our test labels. YOLO26n answers in 423 ms with 284 MB of memory and reaches 0.529. That makes it about **83 times faster** than the ensemble with 71 times fewer parameters, at a cost of 0.07 mAP. It is more accurate than the ensemble as it comes off the shelf (0.445).

YOLOv5s sits below and to the right of YOLO26n: at 949 ms it is slower, and at 0.407 it is clearly less accurate, although it has nearly three times as many parameters. Each model was trained once, but a gap of 0.12 mAP together with half the runtime makes the newer architecture the clear choice here.

The chart leaves out the two-stage pipeline from the beginning, a YOLOv5s that only finds animals, followed by SpeciesNet for each crop. With SpeciesNet running on 1.7 crops per image on average, it adds up to an estimated 5.4 seconds per frame. The single YOLO26n is about **13 times faster** than that.

A side note from the benchmark: decoding and resizing one native 4192 × 3120 camera frame took 256–275 ms on the same device, more than half of the model's own runtime. On a device like this, pre-processing belongs in the latency budget from the start.

### Distillation did not help in our setup

A common way to make a small model better is [knowledge distillation](https://en.wikipedia.org/wiki/Knowledge_distillation), in which the big ensemble teaches the small detector by passing on its full class probabilities instead of a single label. We tried it. The distilled YOLO26n reached 0.479, the directly trained one 0.529, and direct training was ahead in every band.

A plausible reason: our species labels came from the same classifier pipeline that later acted as the teacher, so the teacher's probabilities largely repeated what the labels already said, and differed from them mostly where the teacher was wrong. We tested one configuration in single runs. That speaks against this setup and says little about distillation in general. If your labels come from the model you want to distil from, expect less from it.

## Where it still fails: look-alikes

*Finding* the animal is the easier part. If the detector only has to say "there is an animal", it reaches an mAP of 0.782 on real photos. *Naming* it brings the score down to 0.529.

Part of that gap sits between species that look alike. We grouped such species (three zebras, two elephants, six dogs and wolves, and so on) and counted how often the detector finds the animal, lands in the right group and still picks the wrong member. Across all 30 groups this happens in about one of five cases.

[Figure 10: figures/elephants.jpg]

The groups differ enormously. The zebras, the classic look-alike case, are confused in fewer than 10 % of cases. The elephants are wrong every second time, and almost always in one direction: 79 % of the Asian elephants are named African elephants, and 0.2 % the other way round. The model had seen 1,500 real photos of African elephants and only 100 real plus 100 generated ones of Asian elephants. The imbalance is the most likely cause, although the generated half of the Asian elephant images may play a part. Two fixes follow from that and are still to be tried: more balanced data for exactly these pairs, or a model that may answer "elephant" when it is not sure.

## What's next

Everything above ran at full precision on a CPU. The QCS605 has a DSP that runs 8-bit integer models much faster, and quantisation is the documented way to get there. We did not measure it, so how close the detector gets to real time on the actual chip is still open. It is the obvious next step.

The generators will also keep changing, and a ranking like ours ages quickly. The procedure can be repeated with any new candidate: train the same small model on each generator's images, several times, and score it on real photos.

## Conclusion

A single detector with 2.7 million parameters can find and name 225 mammal species at an accuracy 0.07 mAP below an ensemble 71 times its size, and it runs about 83 times faster on a small CPU. Generated images brought 50 species into the model that had too few photos to train on. Where they supplemented real photos, we measured almost no loss. Where they had to replace real photos entirely, the classes reached about half the accuracy. Which generator draws the rarest animals best is a question our comparison could not answer. What it did show is that the generator and the prompt both matter, and that the results here probably leave room for improvement with a stronger generator.

### Takeaways for your next project

1. **Look at the images, then measure.** Our visual impression that the API model beat the local models held up in the numbers. The measurement added what the eye could not: how large the gap is, and that gpt-image-2 did better than the generator we used.
2. **Pick a generator that reads long prompts, and write detailed ones.** A short prompt cost two API models almost half their accuracy, and the stronger model stayed ahead of the local models even then.
3. **Use price to narrow the field, not to decide.** The paid API models were better than the free local ones. Between the price tiers of one model we found no difference.
4. **Use generated data to fill gaps before you rely on it alone.** Half generated did about as well as all real in our setup. All generated reached about half.
5. **Do not evaluate on generated data alone.** Report the real-only number next to every figure that includes generated test images.
6. **On edge hardware, count every network and every pre-processing step.** Decoding the frame cost more than half as much as the model.
7. **Repeat training runs before you believe small differences.** One of our orderings reversed with the third run.

### Read more

- [Enhancing Object Detection with Synthetic Data](https://www.inovex.de/de/blog/enhancing-object-detection-with-synthetic-data/), the predecessor of this article, on generated training data for small and distant objects
- [Tracking on the Edge: A Case Study on Real-Time Object Tracking](https://www.inovex.de/de/blog/tracking-on-the-edge-a-case-study-on-real-time-object-tracking/)
- [Android Snapdragon Integration: Tensor Interpretation with Kotlin](https://www.inovex.de/de/blog/android-snapdragon-integration-tensor-interpretation-with-kotlin/)

---

<!-- Not part of the article -->

## Sources

`R` = `thesis/manuscript/chapters/4-Results.tex`, `M3x` = the Methods chapter files,
`data` = `thesis/presentation/slides/data.py`. Checked on 2026-11-11.

| Claim | Value | Source |
|---|---|---|
| Prompt ablation | 0.134 → 0.074 (−45 %), 0.095 → 0.049 (−48 %) | R `tab:prompt-length-ablation` |
| Price of the gpt-image-2 tiers | $20.21 vs $39.47 per 1,200 images (batch) = $0.017 vs $0.033 | M35 `tab:generator-api-cost`, `data.GENERATOR_COST`. **R l.80 says $10.11 vs $29.36 (factor 2.9); the manuscript disagrees with itself here.** |
| Band mAP on real photos | A 0.300, B 0.588, C 0.591, D 0.742 | R `tab:band-grid-real`, `data.BAND_REAL` |
| Ensemble and YOLOv5s, B vs C | 0.638 / 0.639, 0.341 / 0.386 | R `tab:band-grid-real` |
| Band A on the synthetic test set | 0.991 | R `tab:kd-vs-direct-bands`, column B (direct), synthetic. Same per-band scoring as `tab:band-grid-real`. The outline used 0.931 vs 0.202 from `tab:domain-shift-delta`, which scores per class on the full test domain; the draft keeps one scoring throughout. |
| Off-the-shelf ensemble higher on synthetic by ≈ 0.2 | 0.205–0.266 | R `tab:domain-shift-delta` |
| Animals per real test image | 1.44 | R l.313 |
| GBIF start, after five sources | 114 classes < 100, 30 with none; 29 still < 100 | `data.gbif_counts`, talk script v3 slide 8 |
| Candidates, filtered, final | 580,817; 465,242; 222,073 real images, 63,802 test | M31 `tab:source-funnel`, M32 l.158 |
| Band A real photos per class | ≈ 67 | M33 l.31 |
| Comparison setup | 12 species × 100 images, 9,742 real test photos | R l.10 |
| Ranking | 0.134, 0.126, 0.105, 0.095, 0.055 … | R `tab:generator-ranking` |
| HiDream, rank 11 | ≈ 48 GPU-hours | R l.71 |
| gpt-image-2 against the production generator | 0.134 vs 0.105, production generator ≈ 28 % behind (gpt-image-2 ≈ 1.28×) | R `sec:results_practicability_verdict` |
| Ties within the API pairs | gpt-image-2: gap 0.008 vs combined SD ≈ 0.018; Gemini: 0.105 ± 0.036 vs 0.095 ± 0.006 | R `sec:results_cost_vs_quality`, `tab:generator-ranking` |
| Gemini Flash, spread of the three runs | single-seed 0.064; mean 0.105 ± 0.036, so the third run is ≈ 0.13 (derived, check against the run logs) | R l.150, `tab:generator-ranking` |
| Runs per generator set | 3 for the API cells, 2 for the local cells | R `tab:generator-ranking` |
| Local generation time | 54 / 24 / 25 / 31 / 144 / 0.9 s per image | `data.GENERATOR_COST` |
| Short-prompt cells against the local models | 0.074 above SD 3.5 Large 0.055; 0.049 between SD 3.5 Medium 0.054 and Turbo 0.043 | R `tab:generator-ranking` |
| Where the ranking comes from | data-rich classes and zebras; black bear 0.072–0.326 | R `sec:results_per_class_structure` |
| Blind rater rubric planned, not run | | R `sec:results_rubric_not_executed` |
| Binturong | Band A, 36 real photos | `eval_per_class.csv` of the model exports |
| Animals per generated test image | 11,264 boxes on 11,250 images | R l.155 |
| Distillation, likely reason | labels and teacher from the same classifier pipeline | Ch. 5 l.56 |
| Thesis headline = mixed test set, real-only alongside | | `CLAUDE.md`, R `sec:results_headline_mixed` |
| Prompt lengths | ≈ 5,900 tokens; 75–512 tokens | `data.GENERATOR_PROMPT` |
| Zebra confusion with the short prompt | 0.253 → 0.457, 0.218 → 0.405 | R l.64 |
| Rare classes | five below 0.05, water deer peaks at 0.066 | R l.85 |
| Seeds | 0.064 → 0.090 → 0.105 | R l.150 |
| Synthetic dataset | 23,850 images, ≈ $785 | M33, Ch. 5 l.49 |
| Hyena and mangabeys | AP 0.48 and ≈ 0.01 | talk script v3 slide 16, `data.per_class_real_ap` |
| Model table | params, real mAP | R `tab:headline-cross-model` |
| Latency | 423 ms, 949 ms, 31 s; ≈ 35 s, ≈ 5.4 s; 83×, 13×, 71× | R `tab:embedded-latency`, l.452–454 |
| Memory | 284 MB | R `tab:embedded-latency` |
| Decode of a native frame | 256–275 ms | R l.467 |
| Pipeline estimate | detector pass + 1.7 classifier crops per image on average | R l.452 |
| Distillation | 0.479 vs 0.529 | R `tab:kd-vs-direct` |
| Detection vs species mAP | 0.782 vs 0.529 | R `tab:headline-cross-model` |
| Within-group confusion, real photos | 19.7 % overall, zebras 9.6 %, elephants 52.3 % | `data.WITHIN_GROUP_CONFUSION_REAL*` (the manuscript's 19.3 / 9.5 / 50.8 % are on the mixed set) |
| Elephant direction | 1,545 of 1,959 (78.9 %) vs 2 of 998 (0.2 %) | `data.SPECIES_CONFUSION_REAL` |
| Elephant training images | 100 + 100 vs 1,500 | `data.SPECIES_TRAIN` |
