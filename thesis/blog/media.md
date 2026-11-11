# Blog Article: Media for WordPress

The ten images of `draft.md` in article order, with the three text fields WordPress asks
for on upload. All are static images.

- **Title** is the name in the media library (short, no file extension).
- **Alternative text** describes the image for screen readers and for readers who cannot
  load it. For charts it states the values, since they are not repeated in a table.
- **Caption** is shown under the image. It says what to see and carries the attribution.

"Exists" means the file is in the repo. "To build" means the text describes the planned
image (`visuals.md`) and has to be checked against the finished one. `[attribution]` marks
real photos whose author and licence still have to be filled in.

## Figure 1: featured image

- **File:** `figures/hero/hero_b.jpg` (exists; candidate, not yet chosen)
- **Title:** Generated wildlife images, a small chip and a detected binturong
- **Alternative text:** Illustration on a blue background: a stack of AI-generated animal photos with a zebra on top is connected by a line to a small computer chip, which is connected to a photo of a binturong on a branch with a green detection box around it.
- **Caption:** none (featured images are shown without a caption).

If another candidate is chosen:

- `hero_a.jpg`. **Title:** A wall of generated wildlife images. **Alternative text:** Illustration on a blue background: twelve AI-generated animal photos, among them a lion, zebras, a fox and a bear, stand as a wall of cards on a round platform. Four of them carry a green detection box.
- `hero_c.jpg`. **Title:** A binturong seen through a binocular. **Alternative text:** Illustration on a blue background: the two overlapping circles of a binocular view show an AI-generated binturong walking along a mossy branch, with a green detection box around it. Four small generated animal photos float around the view.

## Figure 2: rejected images

- **File:** `figures/rejects.jpg` (to build)
- **Title:** Rejected images from scraped species folders
- **Alternative text:** Four images that a scraper filed under animal species and that show no living animal: an ancient Egyptian wall painting filed under cheetah, a person in a tiger mascot costume, a carved wooden aardvark mask in a museum, and a map of Africa showing the range of the African buffalo.
- **Caption:** Filed under "cheetah", "tiger", "aardvark" and "African buffalo": four of the roughly 115,000 images our filter chain threw out. Images: `[attribution]`, via Wikimedia Commons.

## Figure 3: binturong

- **File:** `figures/binturong.jpg` (to build)
- **Title:** Binturong: real photos and four image generators
- **Alternative text:** Grid of fifteen images in five columns of three. First column: real photos of a binturong, a dark shaggy animal with a long tail in a tree. Columns two to four, from the generators RealVisXL, SD 3.5 Medium and FLUX.1 schnell: animals that look like small black bears. Fifth column, from Gemini: animals that match the real binturong.
- **Caption:** Asked for a binturong, three models running locally drew something closer to a bear. Only Gemini (right) matched the real animal (left). Real photos: `[attribution]`; all other images are AI-generated.

## Figure 4: generator ranking

- **File:** `plots/out/ranking.png` (exists)
- **Title:** Generator ranking by detector accuracy on real photos
- **Alternative text:** Bar chart titled "Which generator teaches a detector best?", showing mAP on 9,742 real photos after training on 1,200 generated images. API models with prompts of about 5,900 tokens lead: gpt-image-2 low 0.134, gpt-image-2 medium 0.126, Gemini 3.1 Flash Image 0.105, Gemini 3.1 Flash-Lite 0.095. With a short prompt of at most 75 tokens, gpt-image-2 low drops to 0.074 and Gemini Flash-Lite to 0.049. Models that run locally, with prompts of 75 to 512 tokens: SD 3.5 Large 0.055, SD 3.5 Medium 0.054, SD 3.5 Large Turbo 0.043, FLUX.2 klein 0.037, HiDream-I1 0.029, RealVisXL Lightning 0.023.
- **Caption:** With its full prompt, every API model (blue) beats every local model (grey). The hatched bars are the same two API models with a short prompt: they lose almost half of their accuracy. Lines show the spread across training runs.

## Figure 5: one lion, four prompt lengths

- **File:** `figures/lion_prompts.jpg` (to build)
- **Title:** The same lion prompt at four lengths
- **Alternative text:** Four AI-generated images of a male lion in dry grassland, each above a sheet of text showing its prompt. From left to right the prompt shrinks from 4,642 words (Gemini) to 380 words (FLUX.2 klein), 149 words (SD 3.5 Medium) and 20 words (RealVisXL), and the sheet of text shrinks from a full page to two lines.
- **Caption:** One lion, four prompt lengths. The API model reads the full description of 4,642 words, the local models stop after 512, 256 or 75 tokens. All images are AI-generated.

## Figure 6: kinkajou

- **File:** `figures/kinkajou.jpg` (to build)
- **Title:** Kinkajou: real photo and three generators
- **Alternative text:** Four images side by side. A real photo of a kinkajou, a small golden-brown animal with a plain tail. Next to it three AI-generated kinkajous: the one from FLUX.2 klein has a spotted coat and a ringed tail, which is marked with a circle. The ones from SD 3.5 Large and HiDream-I1 have plain coats and tails like the real animal.
- **Caption:** A real kinkajou has a plain tail. FLUX.2 klein gives it rings and spots, closer to a genet. All three generators score near zero on this species, so the measurement does not register the error. Real photo: `[attribution]`; the other three images are AI-generated.

## Figure 7: bands

- **File:** `plots/out/bands.png` (exists)
- **Title:** Detector accuracy by share of generated training images
- **Alternative text:** Bar chart titled "Half generated works. All generated does not.", showing the mAP of YOLO26n on real test photos for four groups of classes. Band A, 50 classes trained on 200 generated images each: 0.300. Band B, 26 classes with 100 real and 100 generated images: 0.588. Band C, 26 classes with 200 real images: 0.591. Band D, 122 classes with up to 1,500 real images: 0.742.
- **Caption:** With the same budget of 200 training images per class, the half-generated classes (Band B) score as high as the all-real ones (Band C). The all-generated classes (Band A) reach about half of that. The bands hold different species.

## Figure 8: generated against real test set

- **File:** `plots/out/synthetic_vs_real.png` (exists)
- **Title:** Scores on generated and on real test images per band
- **Alternative text:** Paired bar chart titled "The generated test set flatters every band", showing the mAP of YOLO26n for the same classes on generated test images and on real test photos. Band A: 0.99 generated, 0.30 real. Band B: 0.99 and 0.59. Band C: 0.93 and 0.59. Band D: 0.87 and 0.74.
- **Caption:** The same model, the same classes: on generated test images (amber) every band looks nearly solved, on real photos (pink) it does not. The gap is widest where the training images were generated too.

## Figure 9: accuracy against time per frame

- **File:** `plots/out/latency_vs_map.png` (exists)
- **Title:** Accuracy against time per frame on a Raspberry Pi 400
- **Alternative text:** Scatter plot titled "83× faster for 0.07 mAP", showing species mAP on real photos against time per frame on a Raspberry Pi 400, on a logarithmic time axis. YOLO26n with 2.7 million parameters: 0.42 seconds and mAP 0.529. YOLOv5s with 7.6 million parameters: 0.95 seconds and 0.407. MegaDetector plus SpeciesNet with 193 million parameters: about 35 seconds, with 0.599 fine-tuned and 0.445 off the shelf. An arrow leads from the fine-tuned ensemble to YOLO26n.
- **Caption:** The nano detector (green) gives up 0.07 mAP against the fine-tuned ensemble and answers in 0.42 instead of 35 seconds. Measured on the CPU at full precision; the ensemble time is a composed estimate.

## Figure 10: elephants

- **File:** `figures/elephants.jpg` (to build)
- **Title:** Asian and African elephant with confusion rates
- **Alternative text:** Two photos side by side: an Asian elephant in a forest, labelled "100 real + 100 generated training images", and an African elephant in a savanna, labelled "1,500 real training images". A thick arrow from the Asian to the African elephant is labelled 79 %, a thin arrow in the other direction 0.2 %.
- **Caption:** 79 % of the Asian elephants are named African elephants, 0.2 % the other way round. The model saw 1,500 real photos of one species and 100 of the other. `[attribution, or "Both images are AI-generated."]`
