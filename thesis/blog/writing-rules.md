# Blog Article: Writing Rules

Read this before writing or editing `draft.md`. It records the review of draft v1
(2026-11-11): what went wrong, why, and how to check for it. `style-analysis.md` covers
form (length, structure, tone of the inovex blog); this file covers whether the text says
what the experiments show.

## What went wrong in v1

Draft v1 was written to be punchy. Each finding was sharpened into a surprise or a rule,
and in the process several statements ended up saying more than, or something different
from, what the thesis measured. The author's review of one section found six such cases;
the same patterns were then found in every other section.

| v1 text | Why it was wrong | Pattern |
|---|---|---|
| '"Looks right to me" is a weak criterion' | The visual judgement (Gemini far better than the local models) was confirmed by the experiment. The experiment was run to back the judgement with numbers, not because the judgement was distrusted. | Disparaging something to make the next step look necessary |
| "Gemini, the model we had picked by eye months earlier" / "Our pick by eye came third" | Gemini was compared by eye with local models only, never with the whole line-up. It was used because the setup was easier. The real consequence (gpt-image-2 scored higher, so later results probably leave headroom) was missing. | Compressing the history into a neater story |
| "Price predicts nothing" | Paid API models beat free local ones throughout. Price is informative, just not precise. | Absolute claim |
| "In between there is no trend at all" | Six local models. Too few to assert a trend or its absence. | Confident claim from a handful of data points |
| "The prompt is worth more than the model" | gpt-image-2 with the short prompt still beat every local model. The article's own numbers contradict the heading. | Slogan heading |
| "A generator can draw the wrong animal and get away with it" | An observation with no lesson: three models scored near zero and one of them also made a visible error. The usable point is that the measurement cannot rank generators on rare species. | Anecdote without a consequence |
| "It was a mistake to rely on [the generated test set]" | Nobody relied on it. The evaluation design never judged a model on generated images alone. | Invented drama |
| "If you generate your test data, you measure your generator, not your model" | Catchy, and stronger than the result (a large gap with two likely causes). | Slogan |
| "A generator only draws well what it has seen often enough" | The near-zero rare-class scores are a resolution limit of the measurement (100 images, few test photos), which the thesis says explicitly. | Explaining a result with a cause that was not tested |
| "Two generations of architecture progress outweighed anything we could have tuned" | No tuning was compared. | Claim about an experiment that was not run |
| "[Band A's score] is a ceiling to plan for" | One run, and Band A differs from B and C in species as well as in data source. | Generalising from a confounded single result |

## Rules

1. **A claim is as strong as its evidence.** Before writing "never", "nothing", "no trend",
   "always", "every", count what the statement rests on: how many models, how many runs,
   how many classes. If the count is small, say the count instead of the superlative.
2. **Headings and bold lines state the finding.** Test each one against the numbers in its
   own section. If a single number there contradicts it, it is a slogan. A finding with two
   sides gets a heading with two sides ("A long prompt and a strong model both matter").
3. **Tell what was done and why as it happened.** Do not add surprise ("this surprised
   us"), mistakes ("it was a mistake to") or motives that the author did not report. When
   the reason for a decision is not in the repo, ask or leave it out.
4. **Do not disparage one method to motivate another.** Visual inspection and measurement
   complement each other here: the eye was right about API against local models, and it is
   the only instrument for the rare species. Say what each one adds.
5. **Every point ends in something the reader can use.** For each subsection, write down
   the sentence that starts with "For a project this means". If there is none, cut the
   point or attach the observation to the point it illustrates.
6. **Name confounds where the claim is made.** Known ones in this thesis:
   - API models read the full prompt, local models a truncated one, so the API/local gap
     mixes model and prompt capacity.
   - The bands contain different species; Band A is both all-generated and the rarest.
   - Every final model was trained once. The generator comparison has two or three runs.
   - The 35 s and 5.4 s per-frame times are composed estimates.
   - MegaDetector drew the test boxes, which favours the ensemble.
   - Asian elephant is under-represented and half generated.
7. **A cause that was not tested is a hypothesis.** Write "a likely reason" or "one possible
   explanation", and say that it was not tested.
8. **Follow each finding to its consequences.** If gpt-image-2 beat the production
   generator, the article has to say what that means for every later result.
9. **TL;DR, body and takeaways say the same thing.** After any change to a finding, grep
   for it in all three and in `media.md` (captions and alt texts).
10. **The thesis is the reference, and it is more careful than a first blog draft.** Where
    `4-Results.tex` hedges ("a demonstrated effect rather than a general law", "a resolution
    limit of the evaluation axis"), the article keeps the hedge in plain words.

## How to check a draft

Go through the text once per question, not once for all:

- Search for absolutes: `nothing`, `never`, `no `, `always`, `every`, `all`, `only`,
  `clearly`, `proves`. Justify or soften each hit.
- Read only the headings, bold lines, TL;DR and takeaways. Each must survive on its own as
  a true statement.
- For every "because", "so" and "that is why": was the cause measured?
- For every sentence about what "we" thought, expected, picked or got wrong: is it
  documented, or was it written for effect?
- For every subsection: what does the reader do differently after reading it?

## What is fine

Vivid writing is not the problem. The opening scene, the rejected Tutankhamun painting and
"by the time it answers, the animal is gone" are fine, because they do not claim a result.
The title "No Photos of the Binturong?" is fine for the same reason: it asks the reader to
imagine the situation and does not assert that the binturong had no photos (it has 36).
