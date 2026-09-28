# `scripts/thesis` — manuscript tooling

Five scripts: an Overleaf sync bridge, a structural checker that stands in for
the LaTeX compile this container cannot run, a DeepL Write client for language
passes over prose, a paragraph-level rephrase pipeline on top of it whose
output is reviewed by subagents before anything reaches the manuscript, and a
list-price calculator for the synthetic-image cost figures the manuscript quotes.

## Overleaf sync

Two-way sync between `thesis/manuscript/` and the Overleaf project that compiles
it. The repo stays the source of truth and keeps the version history; Overleaf
stays the editor and the only place that actually produces a PDF — no LaTeX
toolchain is installed in this repo's container (`thesis/README.md`).

```
make overleaf               # sync, whichever way it needs to go
```

That is the one to reach for. It works out which side moved and does that:

| Overleaf moved | Manuscript moved | `sync` does |
|---|---|---|
| no | no | nothing — reports both sides identical |
| no | **yes** | **push** |
| **yes** | no | **pull**, leaving it uncommitted for review |
| **yes** | **yes** | **stops** and explains the two ways out |

The explicit targets are there when you want to force a direction, or just to
look:

```
make overleaf-status        # what does each side have? read-only
make overleaf-pull          # Overleaf edits -> working tree
make overleaf-push          # working tree -> Overleaf
make overleaf-push OVERLEAF_MSG="rewrote 3.3 intro"
make overleaf-force-push    # working tree -> Overleaf, local always wins
```

`make overleaf-force-push` is the escape hatch for when `push` refuses: it
overwrites Overleaf with the local manuscript even if Overleaf has commits that
were never pulled, and it resets a clone left dirty by a failed run. It is not a
`git push --force` — the clone is fast-forwarded first and the overwrite lands as
an ordinary commit on top, so the Overleaf history keeps the replaced versions
and `make overleaf-pull` from an older repo state can still get them back.

Or call the script directly, which is the same thing:

```
uv run python -m scripts.thesis.sync_overleaf sync|status|pull|push [--force]
```

Run from the repo root. No container needed. `OVERLEAF_ARGS` passes extra flags
through the make targets, e.g. `make overleaf-status OVERLEAF_ARGS=--allow-any-branch`.

## Why a mirror and not a git remote

Overleaf maps a project to the **root** of its git repo, and our manuscript is a
subdirectory of a repo whose `.git` is 2.6 GB — far past Overleaf's ~100 MB and
~100-files-per-commit guidance. So `git remote add overleaf … && git push` is not
available, and `git subtree push --prefix=thesis/manuscript` would need a strict
always-pull-before-push invariant plus a force-push to bootstrap, which the
Overleaf docs do not promise works.

Instead the script mirrors the file tree between `thesis/manuscript/` and a
dedicated clone of the Overleaf project. The two histories stay independent —
which costs nothing, because this repo's history is the one that counts.

## Setup

Both steps are already done on this machine; this is the record for a new one.

1. **Token.** Overleaf → Account Settings → Project Synchronization → Git
   Integration → generate a token (starts with `olp_`, shown once). Put it in
   `.env` as `OVERLEAF_TOKEN=…`. `.env` is gitignored (`.gitignore:2`) and is
   already where `HF_TOKEN` and `OPENROUTER_API_KEY` live.

   The script hands the token to git through an in-memory credential helper, so
   it never reaches `~/.git-credentials`, the clone's config, or a process
   argument list. `.env` stays the only copy on disk.

2. **Clone.** Create the Overleaf project, delete its default `main.tex`, then:

   ```
   git clone https://git.overleaf.com/<project-id> ~/overleaf/master-thesis
   ```

   Username `git`, password the `olp_` token. Override the location with
   `$OVERLEAF_CLONE` if you put it elsewhere.

3. In Overleaf, set Menu → Compiler = **pdfLaTeX**. Biber is picked up
   automatically from `\usepackage[backend=biber]{biblatex}`.

Current project: `https://git.overleaf.com/6aaec9482e7b434d43e2c4b6`.

## The designated branch

All sync runs happen on **`thesis/overleaf-sync`**. The Overleaf git bridge has
no branch support — its single `main` must correspond to exactly one branch here,
or the manuscript silently forks in two. `status`, `pull` and `push` all refuse
to run on any other branch; `--allow-any-branch` overrides this if you know why
you want to.

Merge `thesis/overleaf-sync` into `main` like any other branch.

## Safety model

Nothing here force-pushes or rewrites Overleaf's history, and every guard fails
closed unless you ask for `--force`:

- **`pull` refuses if `thesis/manuscript/` is dirty.** It overwrites the working
  tree, so it only ever runs into a clean one. Commit or stash first.
- **`pull` does not commit.** Incoming Overleaf edits are left as working-tree
  changes; review them with `git diff thesis/manuscript` and commit deliberately.
- **`push` refuses if Overleaf has commits you have not pulled**, rather than
  overwriting someone else's edits. It tells you to `pull` first.
- **`push` refuses if the clone's working tree is dirty**, which means an earlier
  run died part-way and needs looking at.
- **`push --force` (`make overleaf-force-push`) deliberately lifts those last two
  guards** — it is the only way to overwrite unpulled Overleaf edits. It still
  does not rewrite history: the clone is fast-forwarded to Overleaf's tip before
  the manuscript is mirrored over it, so the replaced content stays reachable in
  the Overleaf log. It prints the commits it is about to supersede.
- Mirroring compares **checksums**, not size+mtime. rsync's default heuristic
  would silently skip a one-character LaTeX edit that keeps the file size and
  lands in the same second as the previous write.

## Caveats

- **Track changes and comments do not survive git round-trips** — Overleaf's own
  docs advise against mixing them with git. Read advisor comments in the Overleaf
  editor, then apply them as ordinary edits.
- The bridge supports no branches, tags, symlinks or LFS.
- `main.tex` loads `\usepackage{svg}`, which needs shell-escape at `\includesvg`
  time. Currently unused — everything in `figures/plots/` is PNG — but verify on
  Overleaf before adding an SVG figure.
- `glossaries` + `\makeglossaries`: confirm Overleaf runs `makeglossaries` as
  expected on the first compile.
- The `.gitkeep` files under `figures/images/` show up as empty folders.

## Preconditions the manuscript already satisfies

Worth re-checking if the structure changes, since only `thesis/manuscript/` is
synced and anything outside it is invisible to Overleaf:

- Every `\input`, `\include`, `\addbibresource` and `\graphicspath` entry in
  `main.tex` is relative and stays inside `thesis/manuscript/`.
- All citekeys resolve in `bibliography/references.bib`. Nothing depends on the
  second bibliography at `research/literature/references.bib`, which is outside
  the synced tree.
- 30 files, 1.4 MB — comfortably inside Overleaf's project limits.


## `check_manuscript.py` — structural and length check

```
uv run python -m scripts.thesis.check_manuscript        # from the repo root
uv run python -m scripts.thesis.check_manuscript -v     # list every warning
```

No LaTeX toolchain is installed here, so nothing local catches a broken
manuscript before it reaches Overleaf. This script covers the failure modes that
editing prose actually causes, and measures every file against the word budget in
`.claude/skills/thesis-writing/references/scope.md` §9.

Errors (exit code 1):

- a `\ref`, `\Cref` or `\hyperref` with no `\label` defining it,
- a `\cite` key that resolves in neither file under `bibliography/`,
- a budgeted file that has gone missing.

Warnings (reported, never fatal, since each needs a judgment call):

- labels nobody references, files more than 10% over their target, paragraphs
  over 180 words, acronyms declared in `preamble/acronyms.tex` but never used,
  and image files in `figures/` that no chapter includes.

It also reports the count of sentences over 55 words, which is
`construction.md` §2.1's rewrite threshold.

Run it before and after any large edit. The baseline immediately before the
September 2026 shortening pass was 0 dangling references, 0 unresolved keys,
58 orphaned labels, 19 paragraphs over 180 words, and 56,606 words against a
31,600-word budget.

## `synthetic_cost_estimate.py` — list-price cost of the synthetic images

```
uv run python -m scripts.thesis.synthetic_cost_estimate
```

Prints the two tables behind the cost figures in the Synthetic Data
Supplementation section (production corpus) and the Generator Comparison
section (the four API cells). No invoice exists for the Gemini runs, so every
figure is measured token usage per request times the vendors' published rates,
standard and batch. Usage is read from the comparison cells' batch-output files
(`usageMetadata` / `usage`) when present and falls back to the values measured
on 2026-09-28. The production runs kept no usage records; their prompt tokens
are estimated from prompt length at the tokens-per-character ratio of the
comparison cell, and their image and text-output tokens are taken from that
cell, which used the same model, modalities, size and aspect ratio. The price
constants at the top of the file carry the pricing-page URLs and the check date;
re-verify them before submission.


## `deepl_write.py` — DeepL Write API client

Improves prose in the language it is already in. Write is not a translator: the
text and `target_lang` must be the same language.

```
uv run python -m scripts.thesis.deepl_write rephrase -t en-US "text to improve"
uv run python -m scripts.thesis.deepl_write rephrase -t en-US --diff -f draft.txt
uv run python -m scripts.thesis.deepl_write rephrase -t en-US --latex --diff \
    thesis/manuscript/chapters/1-Introduction.tex
uv run python -m scripts.thesis.deepl_write correct  -t en-US "som text to fix"
uv run python -m scripts.thesis.deepl_write usage
uv run python -m scripts.thesis.deepl_write languages
```

Run from the repo root. No container needed.

### `rephrase` with no style is the default, and the one to use

`rephrase` with neither `--style` nor `--tone` is exactly what the DeepL Write
web app does by default, and it is the right setting for this manuscript. The
named styles are heavier handed: `academic` over-complicates plain sentences,
which is the opposite of what the thesis register calls for. They exist on the
flag for completeness, not as a recommendation.

`correct` is the other endpoint: spelling and grammar only, no rewording.

| | `correct` | `rephrase` |
|---|---|---|
| Fixes spelling and grammar | yes | yes |
| Rewords sentences | no | yes |
| Takes `--style` / `--tone` | no | yes |

`--diff` prints a before/after to stderr while the improved text goes to stdout,
so `> out.txt` captures only the text. On the same input:

```
correct  -> I could really use some help with edits on this text!
rephrase -> I would appreciate some assistance with editing this text.
```

Text comes from positional arguments, `-f/--file`, or stdin. `--paragraphs`
splits file or stdin input on blank lines. Batching is automatic: DeepL caps a
request body at 10 KiB, and the client packs texts up to a 9 KiB budget,
preserving order so each result stays paired with its own input.

### `--latex` — and why LaTeX may not simply be sent

**Never send raw LaTeX to Write.** Measured on one real sentence from
`1-Introduction.tex`, `rephrase` stripped `\textit{}`, deleted `\cite{AXVisio}`
outright, dropped the `$…$` delimiters, and rewrote
`\Cref{sec:results_granularity_gaps}` into the literal text "see section 3.2.1"
— a section number it invented, and not the one that label resolves to.

Masking the markup behind placeholders is not sufficient either. Placeholders do
survive (`<0>` and `[[0]]` get deleted; a bare-word token comes back intact), but
the service then treats the placeholder as **a noun in the sentence** and
restructures around it. On the real Introduction that produced
`\textit{inovex GmbH} and \cite{inovex}, IT consultancies`,
`\textit{SpeciesNet}/\cite{gadotCropNotCrop2024a}`, and
`ChatGPT (\cite{ChatGPT})`. `correct` is no safer: it inserted a comma before
nearly every `\cite{}`, turned `\item` into `\item:`, and in one paragraph
inverted a technical claim — "whose compute budget **excludes** the model scales"
came back as "**includes**".

So `--latex` works at **sentence** granularity and sends only sentences that
contain no LaTeX at all:

- a markup-free sentence has nothing to corrupt, so its improvement is applied,
- a sentence carrying any `\cite`, `\Cref`, `\textit`, `$…$` or other span is
  **never transmitted** and is left byte-identical,
- headings, labels, comments, and the bodies of tables, figures, tikz pictures
  and display math are never prose and never sent; `\item` lines are prose.

On `1-Introduction.tex` that is 29 sentences improved and 46 held back, with all
103 LaTeX spans, every non-prose line, and the line count unchanged. The held-back
count is high because this manuscript cites densely — that is the price of the
guarantee, and the alternative is a corrupted bibliography.

```
--latex --diff <file>          # show what would change, write nothing (default)
--latex <file>                 # improved document to stdout
--latex --in-place <file>      # rewrite the file; review with `git diff`
```

`--in-place` is opt-in and only valid with `--latex`.

**Read the diff before keeping it.** The markup guarantee is mechanical and
tested; the prose judgment is not. Rephrasing drifts: on the Introduction it
turned "exercises both problems" into "addresses both problems" (which reverses
the meaning), "fine-grained" into "finely grained" (losing the term of art),
"drew on additional AI assistance" into "was required", and introduced
"nanoscale" alongside the manuscript's "nano-scale". It is a suggestion engine,
not an editor. `check_manuscript.py` will not catch any of this, because none of
it breaks the LaTeX.

### Setup and cost

`DEEPL_API_KEY` in `.env` at the repo root, alongside `OVERLEAF_TOKEN`. Get one
at https://www.deepl.com/your-account/keys.

The host follows the key: a key ending in `:fx` is Free tier and uses
`api-free.deepl.com`, anything else uses `api.deepl.com`. Sending a key to the
wrong host is a 403, so `--api-base` is there for completeness and should not
normally be touched.

DeepL's Write quickstart states Write is Pro-only. That is not what the API does:
the Free key here serves both `/v2/write/rephrase` and `/v2/write/correct` with
HTTP 200. Verified 2026-09-27 by the live tests below.

Write characters come out of the same 1,000,000-character monthly quota as
translation. `usage` reports it. A full `--latex` pass over the Introduction
costs roughly 6,000 characters.

To check the quota without the script — useful for confirming the key and host
themselves, independently of any Python:

```sh
export $(grep -E '^DEEPL_API_KEY=' .env | xargs)   # from the repo root
curl --request GET \
  --url https://api-free.deepl.com/v2/usage \
  --header "Authorization: DeepL-Auth-Key $DEEPL_API_KEY"
```

```json
{"character_count":40641,"character_limit":1000000}
```

Two things to keep right in that command:

- **`api-free.deepl.com` is correct only for a `:fx` key.** A Pro key needs
  `api.deepl.com`. The wrong pairing returns 403 with
  `"Wrong endpoint. Use https://api-free.deepl.com."`, not a usage figure — which
  is a useful way to confirm which tier a key is on.
- **Pass the key as `$DEEPL_API_KEY`, never inline.** A literal key typed into the
  command lands in shell history, and one pasted into a file in this repo is a
  single `git push` from being public. `.env` stays the only copy on disk.

The same check through the script, which picks the host from the key for you:

```
uv run python -m scripts.thesis.deepl_write usage
```

Stdlib HTTP only, deliberately — the official `deepl` package would have to go
into the shared lockfile and be synced into a venv that is routinely mid-training.
`X-Trace-ID` is logged to stderr on every call, since it is the only handle DeepL
support has on a request. 429 and 5xx retry with exponential backoff, honouring
`Retry-After`; 400 and 456 (quota exhausted) never retry, because neither
improves by waiting.

### Style and tone, if ever wanted

`--style` (`academic`, `business`, `casual`, `simple`) and `--tone`
(`confident`, `diplomatic`, `enthusiastic`, `friendly`) are **mutually
exclusive**, and `rephrase`-only. Not every target language supports them: `de`,
`en-GB`, `en-US`, `es`, `fr`, `it`, `pt`, `pt-BR` and `pt-PT` do; plain `en`,
`ja`, `ko` and `zh` do not, and a bare value there is a 400, while the
`prefer_`-prefixed variants fall back to the default instead. `deepl_write
languages` reports the live feature set. Note that plain `en` accepts no style or
tone — use `en-US` or `en-GB`, which also converts between the two variants.

### Tests

```
uv run pytest scripts/thesis/tests/                          # offline, no quota
uv run pytest scripts/thesis/tests/ -m live                  # hits the real API
uv run pytest scripts/thesis/tests/ -m "live or not live"    # both
```

108 offline tests and 11 live ones. The offline set drives a stub transport and
covers request shaping, host selection, batching, retry and backoff, the
HTTP-status-to-exception mapping, masking and sentence splitting, and the
`--latex` guarantees — no network, no credentials, no quota. The live set is
deselected by default (`addopts = "-m 'not live'"` in `pyproject.toml`) because it
spends quota.

Two invariants carry the most weight:

- **Losslessness**, parametrized over every real manuscript `.tex` file: line
  splitting, sentence splitting and masking all round-trip byte-for-byte, so no
  construct anywhere in the manuscript can be silently mangled.
- **The markup guarantee**, asserted both offline and live against the real
  Introduction: the multiset of LaTeX spans is identical before and after, every
  non-prose line comes back character-identical, the line count is unchanged, and
  nothing containing a backslash or a `$` was ever transmitted.

Live tests assert documented contracts rather than exact output strings, so they
survive DeepL rephrasing its own output. One thing they deliberately do not
assert: that `correct` fixes *grammar*. It leaves some non-standard grammar ("we
was going home") untouched, so testing that would be testing DeepL's editorial
judgment rather than this client.


## `rephrase_manuscript.py` — paragraph-level rephrase pass, reviewed before it lands

`deepl_write --latex` is the conservative mode: it sends only sentences that
contain no markup at all, and in this manuscript that holds back most of the
text. This pipeline sends **whole paragraphs** as plain prose and puts the LaTeX
back afterwards, with a reviewer between the service and the manuscript.

```
uv run python -m scripts.thesis.rephrase_manuscript prepare --dry-run <tex…>      # count, no request
uv run python -m scripts.thesis.rephrase_manuscript prepare <tex…>                # fetch + review files
uv run python -m scripts.thesis.rephrase_manuscript prepare -t de thesis/manuscript/preamble/abstract_ger.tex
uv run python -m scripts.thesis.rephrase_manuscript apply reports/deepl_rephrase/<stem>            # diff only
uv run python -m scripts.thesis.rephrase_manuscript apply --in-place reports/deepl_rephrase/<stem> # write
uv run python -m scripts.thesis.rephrase_manuscript status
```

Run from the repo root. No container needed. `--out-dir` (before the
subcommand) moves the bundle root; the default is `reports/deepl_rephrase/`,
which is committed: it is the audit trail of what the service and the
reviewers did to the manuscript.

### What is sent

One request per paragraph (one prose line of the `.tex`, which is how this
manuscript is written). Before sending, `latex_prose.flatten` gives every
LaTeX span one of three treatments:

| span | sent as | back as |
|---|---|---|
| `\cite{…}` (any form) | removed | re-attached at the end of the sentence its claim moved to |
| `\textit{X}`, `\textbf`, `\texttt`, `\emph`, `\textsc` | `X` | re-wrapped where `X` reappears |
| `\enquote{X}` | `"X"` | re-wrapped |
| `$225$`, `$28\%$`, `$1{,}200$`, `$193\,\mathrm{M}$`, `$512 \times 512$` | `225`, `28%`, `1,200`, `193 M`, `512 × 512` | re-wrapped |
| `\%`, `\&`, `\_`, `\#` | the literal | re-wrapped |
| `\Cref`, `\hyperref`, `\gls`, `\url`, `\footnote`, non-numeric math, spacing, any other command | `ZQX<n>ZQX` noun token | unmasked, fail-closed |
| a leading `\item`, a trailing `% comment` | not sent | put back verbatim |

Math glued to a word (`v$4.0.2$a`, `$30$th`) stays a token, because its plain
form could not be found again.

**One paragraph per physical line is assumed.** `30-Overview.tex` is hard-wrapped,
so each line there is a sentence fragment; DeepL turns fragments into standalone
sentences and the reviewers rightly kept the originals. Re-flow such a file to
one paragraph per line before running it through this pipeline. Headings, labels, comments, and the bodies of
tables, figures, tikz pictures and display math are never prose and never
sent. A prose line that also occurs verbatim in another manuscript file is
held back (`verbatim-echo`): that is the four research questions, which
`mechanics.md` §6 requires identical in Chapters 1 and 5.

Citations move to the end of their sentence by design (`scope.md` §3: the
citation is a tag on the claim, never a word in the sentence). Sentences are
aligned old-to-new by content-word overlap, monotonically, so a rephrasing
that merges or splits sentences still places each citation on the sentence
its claim landed in. Two `\cite{}` calls landing on one sentence merge into
`\cite{a,b}`. Every one of these decisions is flagged for the reviewer
(`cite-moved`, `cite-merged`, `cite-merged-sentences`, `cite-lowconf`).

Re-wrapping searches for each plain string in order, choosing among repeats
by the words around it in the sent text. A string the service reworded
("28 percent", "193 million") is flagged `format-lost` and the paragraph is
handed to the reviewer with the plain text in place; a dropped token is
`token-lost` and the rephrasing is discarded outright. Automatic house-style
checks on the restored text add `dash`, `semicolon`, `british`,
`bare-number`, `long-sentence`, `this-thesis`, `rather-than` and
`cite-subject`.

### The review

`prepare` writes `reports/deepl_rephrase/<stem>/chunk-NN.review.txt`, twelve
paragraphs per file, each headed by the reviewer instructions
(`rephrase_review_header.md`). Every entry shows the original, the restored
rephrasing, the section title, the flags, and two empty blocks:

```
#### P007  1-Introduction.tex:16  flags: cite-moved: \cite{…}, long-sentence: 58 words
## section: Motivation
--- original
…
--- rephrased
…
--- final

--- verdict

```

The review is done by Claude Code subagents, one per chunk file, in parallel,
with the prompt "read this file, follow its header, fill every empty
`--- final` and `--- verdict` block in place". Sonnet is the default: the
judgments are semantic (a reversed claim, a dropped hedge, a citation now on
the wrong sentence), and DeepL produced all three on the Introduction. The
header is the only context a reviewer gets. It tells them to keep the
rephrasing where it reads better and says the same thing, to take single
sentences back from the original where meaning, a hedge, a number or a term
of art changed, and it condenses the citation-placement and house-style rules
from `.claude/skills/thesis-writing/`. An empty final keeps the original.
Reviewers edit the files on disk paragraph by paragraph, so a reviewer that
dies mid-chunk loses nothing; re-run it on the same file and it skips filled
entries.

### `apply` fails closed

Per paragraph, a final is written only if its LaTeX span inventory equals the
original's (every non-citation span as a multiset, every citation key as a
multiset, so `\cite{a} … \cite{b}` and `\cite{a,b}` are equal), it contains no
prose dash or semicolon the original did not have, no `\citeauthor` or
`\textcite`, no citation used as a sentence subject, balanced braces, and the
same `\item` prefix. Anything else keeps the original line and is named in the
summary with its reason. The whole file is refused if the `.tex` changed since
`prepare` ran (the manifest carries its hash); `prepare --resume` regenerates
the bundle from the log at no cost. House-style checks on the final are
warnings, printed but applied.

Run `check_manuscript` and read `git diff` after every `apply --in-place`.

### Quota, the log, and a second key

Write bills the characters of the *sent* text, which is the flattened prose:
a full pass over both abstracts and all chapters is **165k characters**
(`prepare --dry-run` over the file list), against the 1,000,000-character
monthly quota. Measured cost of the first live run: 14,064 characters for the
two abstracts and the Introduction (24 paragraphs).

- `prepare` calls `/v2/usage` first and refuses a run the remainder cannot
  cover (exit 3); `--force-quota` fetches what fits.
- Every response is appended to `reports/deepl_rephrase/api_log.jsonl` and
  fsynced **before** it is used: `{sha, file, line, target_lang, sent,
  received, trace_id, chars, timestamp}`. That file is the source of truth;
  the manifests and review files are derived from it and `prepare --resume`
  rebuilds them without a request.
- A 456 mid-run stops with exit 2 and says how many paragraphs are unfetched.
  Put a second key in `.env` as `DEEPL_API_KEY_2` and re-run with `--resume`:
  when the first key cannot cover the remainder, the second one is used, host
  derived from its `:fx` suffix as usual.
- `status` reports per bundle what is fetched, reviewed and applied.
- `--limit N` sends only the first N paragraphs, for a smoke test.

### Tests

`uv run pytest scripts/thesis/tests/test_rephrase_manuscript.py` (offline, a
stub plays DeepL): flatten/restore identity round-trip on every manuscript
file, citation re-attachment across reworded, merged and split sentences,
`format-lost` and `token-lost`, `\item` and comment handling, echo hold-back,
one request per paragraph, the log line on disk before use and `--resume`
sending nothing, pre-flight and mid-run quota handling, the second key, the
review file round trip, and every `apply` rejection rule. `-m live` adds one
test that sends three real Introduction paragraphs.

