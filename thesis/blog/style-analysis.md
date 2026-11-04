# inovex Blog: Style Analysis

Analysis of 9 inovex blog posts (2019–2026), done on 2026-11-04 to set the frame for the
thesis article. Word counts are estimates from the reading-time labels and page text.

## The bachelor article (the prequel)

[Enhancing Object Detection with Synthetic Data](https://www.inovex.de/de/blog/enhancing-object-detection-with-synthetic-data/), 06.06.2025

- **Language and label:** English body on the German site. Labelled "Artificial Intelligence
  Expert", 17 min. Tags: Artificial Intelligence, Computer Vision, Data Science, Deep Learning,
  Object Detection.
- **Length:** about 2,500–2,800 words.
- **TL;DR:** one paragraph of prose.
- **Structure:**
  - Motivation (why object detection, the problem, KITTI vs Stanford Cars, object size)
  - The idea (outpainting, generative models)
  - Research question and results
  - Beyond Cars (medical, manufacturing, retail)
  - Conclusion
- **Hook:** opens on application areas, then asks "But what happens when the objects we need
  to detect are small or distant?"
- **inovex:** one sentence: "The thesis was carried out with the support from inovex."
- **Visuals:** 8 photos and collages, 2 short label captions. No charts, tables, code or video.
- **Results:** given inline in prose ("0.39 (from 0.21) which marks a 86 % improvement").
- **Not present:** links, references, bold text or lists.

**Lessons for the sequel:**
- Keep what worked: a concrete problem, photo-led, readable prose.
- Add what was missing:
  - charts for the quantitative results;
  - captions on every visual;
  - a bullet TL;DR;
  - takeaways;
  - links to related posts.

## Comparison posts

| Post | Type | Length | Notable |
|---|---|---|---|
| [Data-Driven Volleyball](https://www.inovex.de/de/blog/data-driven-volleyball-creating-player-statistics-using-computer-vision/) (2024) | master thesis | ~1,500 w / 9 min | H2s follow the pipeline. 8 images (YOLO detections, charts, an architecture diagram), **5 embedded videos** with short captions, 1 table. Ends with "Further Work". |
| [AI Coding Assistants' Impact on Developer Workflows](https://www.inovex.de/en/blog/ai-coding-assistants-impact-on-developer-workflows/) (10/2026) | master thesis | ~850 w / 4 min | Bullet TL;DR, personal "I", opens with an anecdote, 3 practical takeaways. |
| [Anomaly Detection: Do Tabular Foundation Models Help?](https://www.inovex.de/en/blog/anomaly-detection-do-tabular-foundation-models-help/) (09/2026) | study | ~2,600 w / 14 min | TL;DR with "Key Findings" bullets, "hype vs. reality" findings as numbered H3s, ends with "Takeaways for Your Next Project". |
| [Active Learning for Object Detection](https://www.inovex.de/de/blog/active-learning-for-object-detection-using-uncertainty-aggregation/) (2023) | research project | ~1,850 w / 14 min | Formal academic "we", diagrams and result charts, inline citations. |
| [Tracking on the Edge](https://www.inovex.de/de/blog/tracking-on-the-edge-a-case-study-on-real-time-object-tracking/) (2019) | case study | ~2,400 w / 21 min | Raspberry Pi + Coral TPU. Setup photo, **inference-time charts with a real-time line**, 3 videos. The only post with an author-written CTA (portfolio link). |
| [Android Snapdragon Integration](https://www.inovex.de/de/blog/android-snapdragon-integration-tensor-interpretation-with-kotlin/) (2023) | tech spike | ~850 w / 9 min | Qualcomm SNPE + YOLO. Lessons learned, links to earlier edge posts. |
| [TypeSafe AI Jev Review](https://www.inovex.de/en/blog/typesafe-ai-jev-review-how-good-is-the-new-model-for-ai-classification/) (09/2026) | review | ~6,000 w / 31 min | Rare long format, with tables, code and references. |
| [What Does AI Really Cost?](https://www.inovex.de/en/blog/what-does-ai-really-cost-an-attempt-to-make-emissions-visible/) (2026) | opinion/tool | ~1,350 w | Ends with a "Conclusion" plus a "What Users Can Still Do" list. |

## House style

- **Language:** English for expert and thesis posts; the German site carries the same body.
- **Length:** 1,500–2,800 words (9–17 min) for thesis and research posts.
- **Structure:**
  1. TL;DR (bullets / "Key Findings" since 2026)
  2. Hook: a real-world problem or anecdote, then a question
  3. Thesis framing in 1–2 sentences
  4. Problem / Context
  5. Idea / Approach
  6. Findings, often as numbered H3s
  7. Discussion / beyond this use case
  8. Conclusion, plus "Takeaways for your next project" or "Further Work"
- **Headings:** descriptive, often phrased as questions or with a subtitle after a colon.
- **Tone:**
  - "We" for the work, "I" to frame the thesis or tell an anecdote, "you" mainly in the
    takeaways.
  - Professional and accessible, with light touches ("Data scientists know the struggle.").
- **Depth:** concrete numbers in prose (mAP, ms). Methods explained conceptually; few
  formulas, 0–2 code blocks.
- **Visuals:**
  - 4–10 per post, captions of one sentence (newer posts) or a short label.
  - Edge posts use latency charts and embedded MP4s.
- **inovex and CTA:**
  - inovex appears at most once in the body.
  - Expertise shows through the use case, honest limitations and transferable takeaways.
  - The site template adds the training box, related posts, newsletter and contacts.
  - Authors add at most links to related inovex posts or the service page.
- **Recurring elements:**
  - TL;DR: very common.
  - Conclusion: always.
  - Takeaways list: increasingly common.
  - References: optional (inline links are fine).
  - Author box: name + photo.

## Rules for this article

1. **Format:** English. About 2,500 words. Label "Artificial Intelligence Expert". Tags
   Computer Vision, Generative AI, Edge AI, Object Detection.
2. **Opening:** a TL;DR with 4 "Key findings" bullets, each holding one number.
3. **Hook:** a real moment (an animal in the binocular, "what is that?"), then the question,
   then one sentence of framing that names inovex and Swarovski Optik.
4. **Headings:** H2s as questions or "claim: subtitle". The findings follow the reader's
   questions, not the thesis chapters.
5. **Voice:** "we" for the work, "I" sparingly, "you" only in the takeaways.
6. **Visuals:** every one has a one-sentence caption that states what to see, plus alt text.
   Real photos carry their CC attribution.
7. **Show the expertise implicitly:**
   - a careful evaluation (real-only breakout, the synthetic test set that flatters scores);
   - negative results stated plainly (distillation);
   - the limits named (look-alikes, FP32 only);
   - lessons that carry over to other projects.
8. **Ending:** "Conclusion" plus "Takeaways for your next project", then links to the
   bachelor article, Tracking on the Edge, the Snapdragon post, and at most one link to
   inovex's AI/computer-vision offering. No sales copy.
