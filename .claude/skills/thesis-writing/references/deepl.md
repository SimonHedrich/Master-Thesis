# The DeepL pass

Every new or rewritten paragraph goes through DeepL Write before it lands in the manuscript.
Write is a suggestion engine, not an editor: it reads better than a first draft about half
the time and quietly reverses a claim or drops a hedge the other half. So the pass has three
parts that never collapse into one: send plain prose, take back only the wording that
survives review, and refit the citations and markup by hand.

The client is `scripts/thesis/deepl_write.py`; `scripts/thesis/README.md` holds the
measurements behind every rule below. Run everything from the repo root, no container needed.

## 1. When

- After a paragraph is drafted under `construction.md` and before it is pasted into the
  `.tex`. The same for any existing paragraph rewritten by more than a sentence.
- Per paragraph or per section, not per file. The bulk pass over whole chapters is
  `rephrase_manuscript.py` (README §"rephrase_manuscript"), and it already ran over the
  whole manuscript on 2026-09-27. Re-running it on a file to improve one new paragraph
  re-rephrases everything else in that file.
- Never for the research-question text, which must stay byte-identical between Chapter 1
  and Chapter 5 (`mechanics.md` §6). Headings, labels and captions under one sentence are
  not worth the round trip either.

## 2. Write the plain-prose twin

Keep two scratch files in the session scratchpad: the LaTeX draft, and a plain-text twin
of it with one paragraph per line and a blank line between paragraphs. The twin is what
gets sent. Flatten the markup the way `rephrase_manuscript.py` does, so the words DeepL
sees are the words the reader will see:

| in the draft | in the twin |
|---|---|
| `\cite{…}` in any form | removed. Note the key and the sentence it belongs to. |
| `\textit{X}`, `\texttt{X}`, `\emph{X}`, `\textbf{X}` | `X` |
| `\enquote{X}` | `"X"` |
| `$28\%$`, `$1{,}200$`, `$512 \times 512$`, `$193\,\mathrm{M}$` | `28%`, `1,200`, `512 × 512`, `193 M` |
| `\Cref{sec:results}`, `\Cref{tab:foo}` | a plain noun phrase: "the results chapter", "the cost table" |
| `\gls{cnn}` | the expanded term as it reads on the page |
| `\%`, `\&`, `\_` | the literal character |
| a leading `\item`, a trailing `% comment` | dropped |

Why not `--latex` on the draft directly: that mode holds back every sentence that carries
any markup, and in a thesis paragraph that is most of them. It is the right mode for a
cheap first look at a whole file that already exists (`--latex --diff <file>`); it is the
wrong mode for a fresh paragraph, where every sentence should get looked at.

Never send a `.tex` file without `--latex`. Sent raw, Write deletes `\cite{}`, strips
`\textit{}`, and rewrites `\Cref{}` into an invented section number. Masking the commands
behind placeholders is no better, because Write treats the placeholder as a noun and
restructures the sentence around it.

## 3. Run it

```
uv run python -m scripts.thesis.deepl_write rephrase -t en-US --paragraphs --diff -f <twin>.txt
```

- `rephrase`, not `correct`. `correct` inserts commas before citations and has inverted a
  technical claim ("excludes" came back as "includes").
- No `--style` and no `--tone`. The bare call is what the DeepL Write web app does by
  default, and it is the register the supervisor asked for. `academic` over-complicates
  plain sentences.
- `-t en-US`, never plain `en`. Plain `en` accepts no options and `en-GB` re-introduces the
  British spellings `construction.md` §5.2 removes.
- Never `--in-place` on anything under `thesis/manuscript/`. The working tree routinely
  carries someone else's uncommitted edits in those files.
- The word-level diff goes to stderr and the improved text to stdout, so `> <twin>.out.txt`
  captures only the text. A thesis paragraph costs roughly 700 to 1,000 characters against
  the shared 1,000,000 a month; `deepl_write usage` reports the running total.

## 4. Review the diff, sentence by sentence

DeepL's version is the default and the draft is the fallback. The author compared both on
a whole chapter (2026-09-28) and found the rephrased text much easier to read, so a
reviewer who "keeps the original when in doubt" undoes the point of the pass. Take a single
sentence back, never the whole paragraph, and only where one of these changed:

- the direction of a claim: "exercises both problems" became "addresses both problems",
  "whose budget excludes" became "includes", "understate the cost" became "the actual cost";
- a hedge or a scope condition, which are mandatory when they calibrate a claim
  (`conflicts.md` §2);
- a number, a name, or a term of art: "fine-grained" became "finely grained", "nano-scale"
  became "nanoscale", and Band A to D, teacher and student model, step distillation, and
  "this work" all have one spelling (`terminology.md`);
- the actor. Rule 2 names the script, the model, the filter. DeepL turns "the generator
  was asked" into "they asked" and "drew on additional AI assistance" into "was required".

Mixing at sentence level is fine: one sentence from DeepL, the next from the draft. Add no
content that was not in the draft, however natural the added clause sounds. Expect DeepL
to open sentences with "In addition," "Furthermore," and "However,", to join two sentences
with a semicolon, and to lengthen a short sentence for rhythm. All three get cut on the
way back, per `conflicts.md` §3 and rule 3.

## 5. Refit the citations and the markup

Port the surviving wording into the LaTeX draft by hand, then walk the paragraph once
against the table in §2 in the opposite direction. The checks, in the order they fail:

1. **Every citation key noted in §2 is back**, at the end of the sentence that now carries
   its claim (`scope.md` §3). When DeepL merged two cited sentences, the keys merge into one
   `\cite{a,b}`. When it split one sentence, each half gets the key for the claim it makes.
   A key that no longer has a sentence to sit on means the claim was dropped: restore the
   sentence, do not orphan the citation.
2. **No `\cite` as a noun and no author name as a subject** (`mechanics.md` §4). DeepL
   sometimes promotes "previous work found" to "the authors found"; revert it.
3. **Numbers back in math mode** with `\%`, `{,}` and `\times`, and no "28 percent" or
   "twenty-eight" spelled out where the draft had a figure (`mechanics.md` §1).
4. **Every wrapped term re-wrapped where it now sits.** `\textit{}` on the first mention
   only, `\texttt{}` on identifiers, `\enquote{}` for quotes, `\Cref{}` for every float and
   section the noun phrase stood in for, `\gls{}` on the first use of an acronym. DeepL
   moves terms within a sentence, so re-wrap the word where it landed, not where it was.
5. **House style survived the port:** no `--` or `---`, no `;`, no British spellings, no
   sentence above 55 words, no `\citeauthor` or `\textcite` (`conflicts.md`,
   `construction.md` §5.2, `mechanics.md` §4).
6. **The span inventory matches.** The multiset of `\cite`, `\Cref`, `\gls`, `\enquote` and
   `$…$` spans in the final paragraph equals the one in the draft, with `\cite{a}` plus
   `\cite{b}` counting as `\cite{a,b}`. This is the same fail-closed check
   `rephrase_manuscript.py apply` performs mechanically. Do it by eye here, because nothing
   runs it for a hand-ported paragraph.

Only then does the paragraph go into the chapter file.

## 6. Verify

```
uv run python -m scripts.thesis.check_manuscript
git diff thesis/manuscript/
```

`check_manuscript` catches broken LaTeX, the word budget, and severed references. It does
not catch a reversed claim or a lost hedge, because none of that breaks the LaTeX. The
diff read in §4 is the only check for those.
