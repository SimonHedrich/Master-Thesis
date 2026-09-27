# Review instructions

You are reviewing paragraphs of a Master's thesis manuscript (LaTeX) after a
DeepL Write "rephrase" pass. Each entry below shows the paragraph as it stands
in the manuscript (`--- original`), the rephrased version with its LaTeX
restored mechanically (`--- rephrased`), the section it belongs to, and flags
the tooling raised. Your job, per entry: write the paragraph that should end
up in the manuscript into `--- final`, and one line into `--- verdict`.

The reader of the thesis is one examiner who knows machine learning but not
this project, reading under time pressure, who wants the decision, the reason,
the evidence and the limit. Plain, exact prose serves that reader. Decoration
does not.

## What to keep and what to take back

Keep DeepL's wording wherever it reads better **and says the same thing**.
Take the original sentence back wherever the rephrasing changed any of:

- the meaning or direction of a claim ("excludes" -> "includes", "exercises
  both problems" -> "addresses both problems", a negation dropped or added),
- an epistemic hedge or scope condition ("on the mixed test set", "for Band A",
  "measured on the proxy device", "roughly", "at least"). These calibrate a
  claim against its evidence and are mandatory. Only tone-softening padding
  ("arguably", "somewhat", "it is worth noting") may go,
- a number, a unit, a count, a date, or a proper name,
- a term of art: "fine-grained" (not "finely grained"), "nano-scale",
  "long-tailed", "look-alike", "trade-off", "held-out", "real-only",
  "mixed test set" / "real test set" / "synthetic test set", "Band A..D" (never
  "tier", "group", "regime" for the bands), "teacher model" / "student model",
  "knowledge distillation", "step distillation" (never bare "distillation" for
  the diffusion sense), "prompt regime", "generator", "cell",
- who acts: the script, the filter, the model, the stage, not the author and
  never "we" / "I". "This work" is the self-reference, not "this thesis"
  (except when the document itself is meant).

Mixing at sentence level is expected: take DeepL's sentence 1 and 3 and the
original sentence 2 if that is what reads best and stays true. Do not add
content of your own, and do not rewrite beyond choosing between the two
versions and repairing citation placement or formatting.

## Hard rules (the apply step rejects a final that breaks one)

- Every LaTeX span of the original must appear in the final and none may be
  added: every `\cite{...}` key, `\Cref{...}`, `\hyperref`, `\textit{...}`,
  `\texttt{...}`, `\enquote{...}`, `$...$` math, `\%`, and any other command.
  Reposition them, never drop or invent them. `\cite{a} ... \cite{b}` and
  `\cite{a,b}` count as the same.
- No parenthetical dash (` -- ` or `---`) and no semicolon in prose. Split the
  sentence instead. Numeric ranges like `12--15` are fine.
- No `\citeauthor`, no `\textcite`, no "Smith et al. showed". A citation is a
  tag hung on a claim, never a word in the sentence and never its subject:
  wrong: "\cite{key} trained a small model"; right: "Early work trained a small
  model to reproduce a large ensemble \cite{key}."
- The final is one paragraph on one line, exactly as it goes into the `.tex`,
  with a leading `\item` if the original had one and any trailing `% comment`
  kept.

## Citation placement

State the finding, hang the citation at the end of the sentence that makes
the claim. The tooling already moved every citation to the end of the
sentence its claim landed in after rephrasing (flag `cite-moved`); check that
the citation still tags the right claim. If DeepL merged two cited sentences
into one, the citations were merged into one `\cite{a,b}` (`cite-merged`,
`cite-merged-sentences`): confirm both sources support the merged sentence,
or split the sentence again and give each its citation. When the existence
of prior work is the point, an impersonal subject is right ("Previous work
trained...", "It has been shown that...") and the citation still goes at the
end. A named work, method or tool may be the subject ("\textit{YOLOv5}
\cite{...} is commercially usable only up to..."), people never.

## House style (warnings only, but fix them when you see them)

- American spelling: behavior, artifact, judgment, color, labeled, analyze,
  -ize, -ization.
- Numbers in math mode: `$225$`, `$28\%$`, `$1{,}200$`. DeepL sometimes
  spells a number out or writes "28 percent"; put the original `$...$` back.
- `\textit{}` on a tool, dataset or model name only at its first mention in the
  chapter; the original shows which mentions carry it.
- Sentences target 20-30 words; over 55 is a rewrite. Prefer two sentences.
- "rather than" is an authorial tic; do not add new ones.
- German text (the German abstract) follows the same rules; keep the German
  terminology as written ("Knowledge Distillation", "Fine-Tuning",
  "Objektdetektor").

## Flags

- `unchanged`: DeepL returned the text unchanged; nothing to do.
- `cite-moved`: a citation that sat mid-sentence now sits at the sentence end.
- `cite-merged` / `cite-merged-sentences`: citations were joined into one call
  because their sentences were merged.
- `cite-lowconf`: the tooling was unsure which rephrased sentence the cited
  claim moved to. Check the placement with care.
- `format-lost: <span>`: a `\textit`, `$...$` or similar could not be found in
  the rephrased text (DeepL changed the wording inside it). The rephrased
  version shows the plain text; restore the span from the original.
- `wrap-ambiguous` / `wrap-reordered` / `wrap-case`: a span was re-wrapped
  where the tooling thought it belonged; verify the placement.
- `token-lost`: DeepL dropped a cross-reference or math token; the rephrased
  text was discarded and the original is shown. Keep the original.
- `dash`, `semicolon`, `british`, `bare-number`, `long-sentence`,
  `this-thesis`, `rather-than`, `cite-subject`: the named house-style problem
  appeared in the rephrased text. Fix it in the final.

## How to fill an entry

Write the complete paragraph into the `--- final` block (one line) and one
line into `--- verdict`: `rephrased`, `original`, or `merged`, followed by a
short reason ("kept original s2: hedge dropped"). Leave `--- final` empty to
keep the original paragraph unchanged. Skip entries whose `--- final` block is
already filled. Edit only these blocks; change nothing else in this file.
