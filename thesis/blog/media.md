# Blog Article: Media for WordPress

The ten images of `draft.md` in article order, with the three text fields WordPress asks
for on upload. All are static images.

- **Title** is the name in the media library (short, no file extension).
- **Alternative text** describes the image for screen readers and for readers who cannot
  load it. For charts it states the values, since they are not repeated in a table.
- **Caption** is shown under the image. It says what to see and carries the attribution.

"Exists" means the file is in the repo. "To build" means the text describes the planned
image (`visuals.md`) and has to be checked against the finished one. Real photos carry their
author and licence in the caption; the **Sources** table under the figure has the file pages
to link to.

## Figure 1: featured image

- **File:** `figures/hero/hero.jpg` (exists; generated with Gemini by `figures/hero/generate_hero.py`)
- **Title:** A binocular detects a binturong made from AI-generated photos
- **Alternative text:** Illustration on a blue background: on the right, a white binocular looks along a beam of light at a black binturong standing on a mossy branch. A green detection box frames the animal. Above it float five photo cards of binturongs that are partly still made of pixels, with green sparkles around them, the sign for AI-generated images.
- **Caption:** none (featured images are shown without a caption).

If the alternative is chosen:

- `hero_alt.jpg`. **Title:** A binocular looks at a half-generated binturong. **Alternative text:** Illustration on a blue background: on the left, a white binocular looks along a beam of light at a black binturong walking on a mossy branch inside a green detection box. The rear half of the animal and its tail break up into small square pixels and green sparkles, the sign for an AI-generated image.

## Figure 2: rejected images

- **File:** `figures/rejects.jpg` (exists; built by `figures/make_figures.py`)
- **Title:** Rejected images from scraped species folders
- **Alternative text:** Four images that a scraper filed under animal species and that show no living animal: an ancient Egyptian wall painting filed under cheetah, a Roman floor mosaic of an elephant, a carved wooden aardvark mask in a museum, and a map of Africa showing the range of the African buffalo.
- **Caption:** Filed under "cheetah", "African elephant", "aardvark" and "African buffalo": four of the roughly 115,000 images our filter chain threw out. Images via Wikimedia Commons, from left: tomb painting from the tomb of Tutankhamun (public domain); photo by Marie-Lan Nguyen (CC BY 2.5); photo by Daderot (CC0); map by Altaileopard (CC BY 3.0).
- **Sources** (link each credit in the caption to its file page; the local files are byte-identical to the Commons originals, the figure only scales them down):

  | Image | Commons file page | Author | Licence |
  |---|---|---|---|
  | Tomb painting | [Opening of the Mouth - Tutankhamun and Aja.jpg](https://commons.wikimedia.org/wiki/File:Opening_of_the_Mouth_-_Tutankhamun_and_Aja.jpg) | ancient Egyptian artist; faithful reproduction of a two-dimensional work | public domain |
  | Elephant mosaic | [14 Piazzale delle Corporazioni Ostia Antica 2006-09-08.jpg](https://commons.wikimedia.org/wiki/File:14_Piazzale_delle_Corporazioni_Ostia_Antica_2006-09-08.jpg) | Marie-Lan Nguyen (User:Jastrow) | [CC BY 2.5](https://creativecommons.org/licenses/by/2.5) |
  | Aardvark mask | [Aardvark mask and costume, Cameroon grassland, wood - Naturhistorisches Museum Nürnberg - DSC03948.jpg](https://commons.wikimedia.org/wiki/File:Aardvark_mask_and_costume,_Cameroon_grassland,_wood_-_Naturhistorisches_Museum_N%C3%BCrnberg_-_Nuremberg,_Germany_-_DSC03948.jpg) | Daderot | [CC0](https://creativecommons.org/publicdomain/zero/1.0/deed.en) |
  | Buffalo range map | [Syncerus range map.png](https://commons.wikimedia.org/wiki/File:Syncerus_range_map.png) | Altaileopard | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0) |

## Figure 3: binturong

- **File:** `figures/binturong.jpg` (exists; built by `figures/make_figures.py`)
- **Title:** Binturong: real photos and four image generators
- **Alternative text:** Grid of fifteen images in five columns of three. First column: real photos of a binturong, a dark shaggy animal with a long tail in a tree. Columns two to four, from the generators RealVisXL, SD 3.5 Medium and FLUX.1 schnell: animals that look like small black bears. Fifth column, from Gemini: animals that match the real binturong.
- **Caption:** Asked for a binturong, three models running locally drew something closer to a bear. Only Gemini (right) matched the real animal (left). Real photos, cropped, via Wikimedia Commons, from top: sheilaellen (CC BY 2.0), Tassilo Rau (CC BY-SA 3.0), Terence Ong (CC BY 2.5); all other images are AI-generated.
- **Sources** (link each credit in the caption to its file page; the three photos were identified by comparing them with every 4:3 and same-size file in the Commons category *Arctictis binturong*):

  | Image | Commons file page | Author | Licence |
  |---|---|---|---|
  | `binturong_real_1.jpg` (top) | [Binturong.jpg](https://commons.wikimedia.org/wiki/File:Binturong.jpg) | sheilaellen ([Flickr original](https://www.flickr.com/photos/sheilaellen/1122131130/)) | [CC BY 2.0](https://creativecommons.org/licenses/by/2.0) |
  | `binturong_real_2.jpg` (middle) | [Binturong in Overloon.jpg](https://commons.wikimedia.org/wiki/File:Binturong_in_Overloon.jpg) | Tassilo Rau | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) |
  | `binturong_real_3.jpg` (bottom) | [Arctictis binturong, Singapore Zoo - 20061122.jpg](https://commons.wikimedia.org/wiki/File:Arctictis_binturong,_Singapore_Zoo_-_20061122.jpg) | Terence Ong | [CC BY 2.5](https://creativecommons.org/licenses/by/2.5) |

  The middle photo is ShareAlike and is cropped in the figure. Either state under the figure that the cropped photo stays under CC BY-SA 3.0, or swap it for a CC BY photo.

## Figure 4: generator ranking

- **File:** `plots/out/ranking.png` (exists)
- **Title:** Generator ranking by detector accuracy on real photos
- **Alternative text:** Bar chart titled "Which generator teaches a detector best?", showing mAP in percent on 9,742 real photos after training on 1,200 generated images. API models with prompts of about 5,900 tokens lead: gpt-image-2 low 13.4 %, gpt-image-2 medium 12.6 %, Gemini 3.1 Flash Image 10.5 %, Gemini 3.1 Flash-Lite 9.5 %. With a short prompt of at most 75 tokens, gpt-image-2 low drops to 7.4 % and Gemini Flash-Lite to 4.9 %. Models that run locally, with prompts of 75 to 512 tokens: SD 3.5 Large 5.5 %, SD 3.5 Medium 5.4 %, SD 3.5 Large Turbo 4.3 %, FLUX.2 klein 3.7 %, HiDream-I1 2.9 %, RealVisXL Lightning 2.3 %.
- **Caption:** With its full prompt, every API model (blue) beats every local model (grey). The hatched bars are the same two API models with a short prompt: they lose almost half of their accuracy. Lines show the spread across training runs.

## Figure 5: one lion, four prompt lengths

- **File:** `figures/lion_prompts.jpg` (exists; built by `figures/make_figures.py`)
- **Title:** The same lion prompt at four lengths
- **Alternative text:** Four AI-generated images of a male lion in dry grassland, each above a sheet of text showing its prompt. From left to right the prompt shrinks from 4,642 words (Gemini) to 380 words (FLUX.2 klein), 149 words (SD 3.5 Medium) and 20 words (RealVisXL), and the sheet of text shrinks from a full page to two lines.
- **Caption:** One lion, four prompt lengths. The API model reads the full description of 4,642 words, the local models stop after 512, 256 or 75 tokens. All images are AI-generated.

## Figure 6: kinkajou

- **File:** `figures/kinkajou.jpg` (exists; built by `figures/make_figures.py`)
- **Title:** Kinkajou: real photo and three generators
- **Alternative text:** Four images side by side. A real photo of a kinkajou, a small golden-brown animal with a plain tail. Next to it three AI-generated kinkajous: the one from FLUX.2 klein has a spotted coat and a ringed tail, which is marked with a circle. The ones from SD 3.5 Large and HiDream-I1 have plain coats and tails like the real animal.
- **Caption:** A real kinkajou has a plain tail. FLUX.2 klein gives it rings and spots, closer to a genet. All three generators score near zero on this species, so the measurement does not register the error. Real photo: Kristof Zyskowski via iNaturalist (CC BY 4.0), cropped; the other three images are AI-generated.
- **Source** (link the credit in the caption to the observation; identified by comparing the photo with all GBIF images of *Potos flavus*, a single exact match):

  | Image | Source page | Author | Licence |
  |---|---|---|---|
  | `kinkajou_real.jpg` | [iNaturalist observation 44818134](https://www.inaturalist.org/observations/44818134), [photo 71048382](https://www.inaturalist.org/photos/71048382); GBIF occurrence [2620029702](https://www.gbif.org/occurrence/2620029702). Magdalena, Colombia, 2015-01-04 | Kristof Zyskowski | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

## Figure 7: bands

- **File:** `plots/out/bands.png` (exists)
- **Title:** Detector accuracy by share of generated training images
- **Alternative text:** Bar chart titled "Half generated works. All generated does not.", showing the mAP of YOLO26n in percent on real test photos for four groups of classes. Band A, 50 classes trained on 200 generated images each: 30.0 %. Band B, 26 classes with 100 real and 100 generated images: 58.8 %. Band C, 26 classes with 200 real images: 59.1 %. Band D, 122 classes with up to 1,500 real images: 74.2 %.
- **Caption:** With the same budget of 200 training images per class, the half-generated classes (Band B) score as high as the all-real ones (Band C). The all-generated classes (Band A) reach about half of that. The bands hold different species.

## Figure 8: generated against real test set

- **File:** `plots/out/synthetic_vs_real.png` (exists)
- **Title:** Scores on generated and on real test images per band
- **Alternative text:** Paired bar chart titled "The generated test set flatters every band", showing the mAP of YOLO26n in percent for the same classes on generated test images and on real test photos. Band A: 99 % generated, 30 % real. Band B: 99 % and 59 %. Band C: 93 % and 59 %. Band D: 87 % and 74 %.
- **Caption:** The same model, the same classes: on generated test images (amber) every band looks nearly solved, on real photos (pink) it does not. The gap is widest where the training images were generated too.

## Figure 9: accuracy against time per frame

- **File:** `plots/out/latency_vs_map.png` (exists)
- **Title:** Accuracy against time per frame on a Raspberry Pi 400
- **Alternative text:** Scatter plot titled "83× faster for 7 percentage points of mAP", showing species mAP in percent on real photos against time per frame on a Raspberry Pi 400, on a logarithmic time axis. YOLO26n with 2.7 million parameters: 42 % seconds and mAP 52.9 %. YOLOv5s with 7.6 million parameters: 95 % seconds and 40.7 %. MegaDetector plus SpeciesNet with 193 million parameters: about 35 seconds, with 59.9 % fine-tuned and 44.5 % off the shelf. An arrow leads from the fine-tuned ensemble to YOLO26n.
- **Caption:** The nano detector (green) gives up 7 percentage points of mAP against the fine-tuned ensemble and answers in 0.42 instead of 35 seconds. Measured on the CPU at full precision; the ensemble time is a composed estimate.

## Figure 10: elephants

- **File:** `figures/elephants.jpg` (exists; built by `figures/make_figures.py`)
- **Title:** Asian and African elephant with confusion rates
- **Alternative text:** Two AI-generated images side by side: an Asian elephant in a forest, labelled "100 real + 100 generated training images", and an African elephant in a savanna, labelled "1,500 real training images". A thick arrow from the Asian to the African elephant is labelled 79 %, a thin arrow in the other direction 0.2 %.
- **Caption:** 79 % of the Asian elephants are named African elephants, 0.2 % the other way round. The model saw 1,500 real photos of one species and 100 of the other. Both images are AI-generated.
