# No Photos of the Binturong? Training a Wildlife Detector for a Binocular with AI-Generated Images

<!--
Draft v1, 2026-11-11. About 2,550 words of body text.
Label: Artificial Intelligence Expert. Tags: Computer Vision, Generative AI, Edge AI, Object Detection.
[Figure N: file] blocks are placeholders. media.md holds the WordPress title, alt text and
caption of each one; visuals.md the concept. All visuals are static images (no clips).
Every number is traced in the "Sources" table at the end of this file (not part of the article).
-->

**TL;DR: Key findings**

- **The prompt matters more than the generator.** Cutting the image prompt down to 75 tokens cost 45–48 % of the downstream accuracy. Paying twice as much per image for a higher quality tier bought nothing.
- **Generated images can replace half of a small training set, but not all of it.** Classes trained on 100 real and 100 generated images reached an mAP of 0.588 on real photos, against 0.591 for 200 real images. Classes trained on generated images alone reached 0.300.
- **A generated test set flatters the model.** The same classes that score 0.99 on generated test images score 0.30 on real photos.
- **One nano-scale detector replaces two networks.** A YOLO26n with 2.7 million parameters is about 13 times faster than a detector-plus-classifier pipeline and 83 times faster than the large reference ensemble, for 0.07 mAP less on real photos.

---

A brown shape moves at the edge of the forest, two hundred metres away. You lift the binoculars and get a good look for maybe ten seconds: long body, dark fur, a tail as long as the animal itself. What *is* that?

The [AX Visio](https://www.swarovskioptik.com/) by Swarovski Optik is a binocular built to answer that question. It has a camera and a neural processor on board, identifies what you are looking at, and shows the name in the eyepiece, without a phone and without a network connection. For birds this already works. A model for mammals would have to tell 225 species apart, on a Qualcomm QCS605 chip from 2018, while the animal is still in view.

In my Master's thesis, carried out at inovex together with Swarovski Optik, I asked how to build that detector under two constraints that show up in many computer vision projects: the hardware is small, and for a good part of the classes there are hardly any photos.

[Figure 1: figures/hero/hero_b.jpg (featured image)]

## Two problems: rare animals and a small chip

The first problem is data. Lions and red foxes have been photographed millions of times. The binturong, a shaggy tree-dwelling mammal from Southeast Asia, has not. When we started with the openly licensed images on GBIF, 114 of our 225 classes had fewer than 100 photos, and 30 had none at all.

The second problem is compute. The fastest route to a working product is a pipeline of existing models: a fine-tuned YOLOv5s detects a single "animal" class and hands each crop to SpeciesNet, Google's open classifier for camera-trap images. That works, but two networks run one after the other, and the classifier runs once per animal. A herd of zebras costs more than a single fox.

So the thesis question was: **can one small detector do both jobs, finding the animal and naming the species, if part of its training data is generated?**

## Collecting real photos, and what you throw away

Before generating anything, we collected what exists. Five sources with licences that allow commercial use (iNaturalist, GBIF, Wikimedia Commons, Open Images and stock imagery) yielded about 580,000 candidate images, 400,000 of them from iNaturalist alone.

A folder named after a species holds a lot of things that are not that species. Our filter chain threw out a Tutankhamun wall painting filed under "cheetah", a university's tiger mascot, a wooden aardvark mask from a museum and a range map of the African buffalo. After metadata checks, blur heuristics, an animal detector, a vision-language model as a second opinion and a species classifier, 465,242 images were left, and the per-class caps reduced them to a final set of 222,073 real photos. We held back 63,802 of them as the test set.

[Figure 2: figures/rejects.jpg]

More sources helped the middle of the distribution, not the tail. 29 classes still had fewer than 100 images. The 50 rarest classes averaged about 67 usable photos each, which is not enough to train on and barely enough to test with.

## Which image generator? Twelve candidates compared

The obvious idea in 2026 is to generate the missing photos. The first attempts showed why this is harder than it sounds. Asked for a binturong, three open-weight models running on our own GPU all drew something closer to a small bear. Gemini, called through an API, drew a binturong.

[Figure 3: figures/binturong.jpg]

But "looks right to me" is a weak criterion, especially for animals that the person judging has never seen. We wanted to know which images a detector actually *learns* from. So we set up a comparison in which the generator was the only thing that changed:

- ten image generators, four behind an API and six running locally,
- the same twelve species, from lion to aye-aye, with 100 images each,
- the same YOLO26n detector trained on each set of 1,200 images with the same schedule,
- scored on the same 9,742 **real** photos.

[Figure 4: plots/out/ranking.png]

Every API model beat every local model. The cheapest tier of gpt-image-2 came first with an mAP of 0.134. Gemini 3.1 Flash Image, the model we had picked by eye months earlier and used for the production dataset, came third with 0.105. The best local model, Stable Diffusion 3.5 Large, reached 0.055. (These absolute values are low because each detector saw only 1,200 images. They rank the generators and say nothing about the accuracy of the final model.)

Three things in this ranking surprised us.

### 1. Price predicts nothing

The medium quality tier of gpt-image-2 costs about twice as much per image as the low tier ($0.033 against $0.017 at batch prices) and scored slightly *worse*, within the noise. Among the local models, HiDream-I1 needed almost 48 GPU-hours for its 1,200 images and came second to last. The cheapest local model, at under a second per image, came last. In between there is no trend at all.

### 2. The prompt is worth more than the model

Our prompts were long. Each one described the species, its distinguishing marks, the habitat, the pose and the camera position, at about 5,900 tokens on average. The API models read all of it. The text encoders of the local models stop at 75 to 512 tokens.

To separate the model from the prompt, we gave two of the API models the same short prompt of at most 75 tokens. Both lost almost half of their accuracy: gpt-image-2 dropped from 0.134 to 0.074, Gemini Flash-Lite from 0.095 to 0.049. That is a bigger step than most differences between generators, and it moved the winner down to fifth place. Confusion between the three zebra species nearly doubled at the same time. The long prompt carries the details that make a Grévy's zebra a Grévy's zebra.

[Figure 5: figures/lion_prompts.jpg]

### 3. A generator can draw the wrong animal and get away with it

The kinkajou is a golden-brown relative of the raccoon with a plain tail. FLUX.2 klein consistently drew it with a ringed tail and a spotted coat, closer to a genet. Two other local models got it right from the same prompt. In the numbers, this error is invisible. All three score close to zero on the kinkajou, because 100 generated images are too few for any of them.

[Figure 6: figures/kinkajou.jpg]

This is the uncomfortable part of the result. Our measure can rank generators on well-photographed species. On the rare species, the ones we wanted the generated images for, every generator scores below 0.07 and they cannot be told apart. A generator only draws well what it has seen often enough, and that is the same long tail we started with.

One more lesson came for free. With a single training run, Gemini scored 0.064, behind its own cheaper sibling. A second seed made it 0.090, a third 0.105, and the order flipped. If a comparison rests on one run, differences of a few points mean nothing.

The complete synthetic dataset, 23,850 images of all 225 classes, cost about $785 at list price.

## How much synthetic data is too much? The band experiment

To measure what generated images are worth in training, we sorted the 225 classes by the number of real photos available and cut them into four bands. Bands A to C get the same budget of 200 training images per class and differ only in where those images come from:

| Band | Classes | Training images per class |
|------|---------|---------------------------|
| A | 50 | 200 generated |
| B | 26 | 100 real + 100 generated |
| C | 26 | 200 real |
| D | 122 | up to 1,500 real |

The YOLO26n detector then scores as follows on real test photos:

[Figure 7: plots/out/bands.png]

Band B, half generated, reaches an mAP of 0.588. Band C, all real, reaches 0.591. Band A, all generated, reaches 0.300, and the data-rich Band D 0.742.

Replacing half of a small real training set with generated images cost nothing measurable. The large reference model showed the same pattern (0.638 against 0.639), the older YOLOv5s lost a little (0.341 against 0.386). Replacing *all* of it halved the accuracy. That is still far better than nothing, since those 50 classes would otherwise be missing from the model entirely, but it is a ceiling to plan for.

Inside Band A the spread is large. The striped hyena, trained on generated images only, reaches an AP of 0.48 on real photos. The mangabeys, a genus of monkeys with several quite different-looking species, stay near zero. Our working hypothesis: animals that look uniform are easy to generate convincingly, varied ones are not.

Two caveats belong here. Each model was trained once, and the bands contain different species, so "band B equals band C" is a strong hint and not a law.

## Don't trust the synthetic test set

Rare classes are hard to evaluate for the same reason they are hard to train: some of ours have six real test photos. Generating a balanced test set of 50 images per class looked like an elegant fix. It was a mistake to rely on it, and a useful one.

[Figure 8: plots/out/synthetic_vs_real.png]

On the generated test images, Band A scores 0.99. On real photos, the same classes score 0.30. The gap exists in every band and for every model. Even the off-the-shelf reference model, which has never seen a generated image, scores about 0.2 mAP higher on them. Generated test images show one well-lit animal in the centre of the frame. Real photos have backlight, bushes and, on average, 1.44 animals per image. On top of that, a model trained on images from one generator is tested on more images from the same generator.

**If you generate your test data, you measure your generator, not your model.** We kept the generated test set as a secondary figure and report every result on real photos. All numbers in this article are real-photo numbers unless stated otherwise.

## Making it small: one nano detector instead of two networks

With the dataset fixed, three candidates competed:

| Model | Parameters | mAP on real photos | Time per frame |
|-------|-----------:|-------------------:|---------------:|
| MegaDetector + SpeciesNet, classifier fine-tuned | ≈ 193 M | 0.599 | ≈ 35 s |
| MegaDetector + SpeciesNet, off the shelf | ≈ 193 M | 0.445 | ≈ 35 s |
| YOLOv5s, fine-tuned | 7.63 M | 0.407 | 0.95 s |
| **YOLO26n, fine-tuned** | **2.71 M** | **0.529** | **0.42 s** |

The first two rows are the big reference ensemble that also produced our labels, so its score is an upper bound more than a competitor. YOLO26n is the smallest model in the table, and it beats the nearly three times larger YOLOv5s by 0.12 mAP on the same data and schedule. Two generations of architecture progress outweighed anything we could have tuned on the older model.

We measured speed on a Raspberry Pi 400, whose CPU is a slightly slower stand-in for the CPU cores of the QCS605: ONNX Runtime, four threads, full 32-bit precision, timed from JPEG to final boxes. YOLO26n needs 423 ms per frame and 284 MB of memory. YOLOv5s needs 949 ms. MegaDetector alone needs 31 seconds. Put together, the reference ensemble lands at about 35 seconds per frame and a YOLOv5s-plus-SpeciesNet pipeline at about 5.4 seconds.

That makes the single nano detector **83 times faster** than the ensemble and **13 times faster** than the two-stage pipeline, with 71 times fewer parameters than the former, at a cost of 0.07 mAP.

[Figure 9: plots/out/latency_vs_map.png]

A side note from the benchmark: decoding and resizing one native 4192 × 3120 camera frame took 256–275 ms on the same device, more than half of the model's own runtime. On the edge, the pre-processing is part of the model.

### The negative result: distillation did not help

The textbook move would be knowledge distillation, in which the big ensemble teaches the small detector by passing on its full class probabilities instead of a single label. We tried it. The distilled YOLO26n reached 0.479, the directly trained one 0.529, and direct training won in every band.

The likely reason is mundane. Our species labels came from the very classifier that later acted as the teacher, so the teacher had little to say that the labels had not already said. We tested one configuration in single runs, so this rules out that setup and not distillation in general. But it is a reminder that a teacher is only useful if it knows something the labels don't.

## Where it still fails: look-alikes

*Finding* the animal is the easy part. If the detector only has to say "there is an animal", it reaches an mAP of 0.782 on real photos. *Naming* it brings the score down to 0.529.

Part of that gap sits between species that look alike. We grouped such species (three zebras, two elephants, six dogs and wolves, and so on) and counted how often the detector finds the animal, lands in the right group and still picks the wrong member. Across all 30 groups this happens in about one of five cases.

[Figure 10: figures/elephants.jpg]

The groups differ enormously. The zebras, which we had worried about most, are confused in fewer than 10 % of cases. The elephants are wrong every second time, and almost always in one direction: 79 % of the Asian elephants are named African elephants, and 0.2 % the other way round. The model had seen 1,500 real photos of African elephants and only 100 real plus 100 generated ones of Asian elephants. When in doubt, it names the species it knows better. That points to the next fix, more balanced data for exactly these pairs, or a model that may answer "elephant" when it is not sure.

## What's next

Everything above ran at full precision on a CPU. The QCS605 has a DSP that runs 8-bit integer models much faster, and quantisation is the documented way to get there. We did not measure it, so how close the detector gets to real time on the actual chip is still open. It is the obvious next step.

The generators will also keep changing. The ranking in this article will be outdated in a few months. The protocol will not be: train the same small model on each candidate's images, and score it on real photos.

## Conclusion

A single detector with 2.7 million parameters can find and name 225 mammal species at an accuracy close to an ensemble 71 times its size, and it runs 83 times faster on a small CPU. Generated images made that possible for classes with too few photos, as long as they supplemented real photos. Where they had to replace real photos entirely, accuracy halved, and no generator we tested could draw the rarest animals reliably. The most valuable outcome was arguably not the model but the measuring: a comparison that ranks generators by what a detector learns from them, and a real-photo test set that kept us honest when the generated one looked too good.

### Takeaways for your next project

1. **Judge a generator by what it teaches your model**, measured on real data, not by how its images look. Our pick by eye came third.
2. **Spend effort on the prompt before you spend money on the generator.** A short prompt cost us half the accuracy, a pricier tier bought none.
3. **Use generated data to fill gaps, not to replace real data.** Half generated worked as well as all real. All generated did not.
4. **Never evaluate on generated data alone.** Report the real-only number next to every headline.
5. **On edge hardware, count every network and every pre-processing step.** Decoding the frame can cost as much as the model.
6. **Repeat runs before you believe small differences.** One of our rankings flipped with the third seed.

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
| Distillation | 0.479 vs 0.529 | R `tab:kd-vs-direct` |
| Detection vs species mAP | 0.782 vs 0.529 | R `tab:headline-cross-model` |
| Within-group confusion, real photos | 19.7 % overall, zebras 9.6 %, elephants 52.3 % | `data.WITHIN_GROUP_CONFUSION_REAL*` (the manuscript's 19.3 / 9.5 / 50.8 % are on the mixed set) |
| Elephant direction | 1,545 of 1,959 (78.9 %) vs 2 of 998 (0.2 %) | `data.SPECIES_CONFUSION_REAL` |
| Elephant training images | 100 + 100 vs 1,500 | `data.SPECIES_TRAIN` |
