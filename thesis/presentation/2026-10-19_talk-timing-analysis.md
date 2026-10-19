# Final talk 2026-10-19 — time per slide

How long each slide of the recorded final presentation took, compared with the plan in
`final-talk-script-v2.md`. The deck shown was `slides/talk_v2.py` (25 slides).

**Total: 37:58 against a planned 19:15–19:30, about twice as long.**

## Method

Slide titles were read off the recording every 5 seconds, so every start time is accurate
to about ±5 s. Slides without a visible title (product page, hedgehogs, zebras, "23,850
synthetic images") were checked against single frames. The slide changes match the
transitions in `2026-10-19_abschlusspraesentation_transcript.txt`.

## Per slide

| # | Slide | Start | Actual | Plan | Over |
|---|---|---|---|---|---|
| 1 | Title | 0:00 | 0:10 | 0:15 | −0:05 |
| 2 | Product (AX Visio) | 0:10 | 0:45 | 0:15 | +0:30 |
| 3 | BirdID | 0:55 | 0:55 | 0:45 | +0:10 |
| 4 | MammalID | 1:50 | 1:20 | 0:30 | +0:50 |
| 5 | Model Ensemble | 3:10 | 1:10 | 1:15 | −0:05 |
| 6 | Three challenges | 4:20 | 1:00 | 0:30 | +0:30 |
| 7 | SpeciesNet | 5:20 | 1:20 | 0:40 | +0:40 |
| 8 | Hedgehogs | 6:40 | 0:40 | 0:25 | +0:15 |
| 9 | 225 Custom Classes | 7:20 | 0:35 | 0:20 | +0:15 |
| 10 | Images per class | 7:55 | 2:25 | 1:10 | +1:15 |
| 11 | **Data Quality** | 10:20 | **3:20** | 0:50 | **+2:30** |
| 12 | Binturong | 13:40 | 1:20 | 0:50 | +0:30 |
| 13 | Three zebras | 15:00 | 1:15 | 0:35 | +0:40 |
| 14 | **Prompt** | 16:15 | **2:25** | 0:50 | **+1:35** |
| 15 | Training Images Comparison | 18:40 | 2:40 | 1:20 | +1:20 |
| 16 | Rare Animals | 21:20 | 1:25 | 0:50 | +0:35 |
| 17 | 23,850 synthetic images | 22:45 | 0:30 | 0:20 | +0:10 |
| 18 | Final Dataset | 23:15 | 2:15 | 1:15 | +1:00 |
| 19 | Recognition Results | 25:30 | 1:15 | 1:30 | −0:15 |
| 20 | Test Data | 26:45 | 1:50 | 0:45 | +1:05 |
| 21 | Three candidates | 28:35 | 0:40 | 0:20 | +0:20 |
| 22 | Accuracy on real photos | 29:15 | 2:35 | 1:10 | +1:25 |
| 23 | Similar Species | 31:50 | 1:35 | 1:00 | +0:35 |
| 24 | **Knowledge Distillation** | 33:25 | **2:55** | 0:45 | **+2:10** |
| 25 | Summary | 36:20 | 1:38 | 0:50 | +0:48 |

## Per act

| Act | Slides | Actual | Plan |
|---|---|---|---|
| 1 Use case | 1–6 | 5:20 | 3:30 |
| 2 Data | 7–11 | 8:20 | 3:25 |
| 3 Generated images | 12–16 | 9:05 | 4:25 |
| 4 Bands | 17–20 | 5:50 | 3:50 |
| 5 Models | 21–24 | 7:45 | 3:20 |
| 6 Summary | 25 | 1:38 | 0:50 (1:00 in the act table) |

## Where the time went

- **More words, spoken slower.** The transcript has about 3,970 words against about 2,520
  in the script, 57 % more text. The pace was about 105 words per minute, where the plan
  assumes about 130. Together these explain the doubling.
- **Three slides account for about a third of the overrun:**
  - **Data Quality** (+2:30): every filter step explained in detail — metadata, blur and
    white-pixel checks, MegaDetector, the vision language model with the ceramic-elephant
    example, the SpeciesNet check and the manual pass. Images per class had already covered
    the licence filtering and the junk images from Wikipedia.
  - **Knowledge Distillation** (+2:10): the failure explained twice, first from the numbers
    (the off-the-shelf ensemble is already worse), then from the labels (the direct labels
    are more accurate).
  - **Prompt** (+1:35): from about 17:20 the talk already describes the setup of the
    generator experiment (twelve species, 100 images each, the professor's request). That
    content belongs to slide 15, which then ran 1:20 over as well.
- **Only Model Ensemble and Recognition Results stayed on time or under.** The results
  slide the deck was built around got less time than planned; most of the extra time went
  to method slides.
- **Slides 1–9 were each only modestly over**, about 3 minutes in total.

To fit about 20 minutes, the biggest savings are Data Quality, Knowledge Distillation, and
the experiment setup on the Prompt slide.
