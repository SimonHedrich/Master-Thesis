# Overleaf sync for the manuscript

**Date:** 2026-09-19
**Status:** implemented and in use
**Implements:** `scripts/thesis/sync_overleaf.py` · **Operating manual:** `scripts/thesis/README.md`

## Why this exists

The manuscript is written in this repo but compiled nowhere in it. No LaTeX
toolchain is installed in the training container — `pdflatex`, `latexmk` and
`biber` are all absent — so before this change the only route to a PDF was
hand-copying files into the Overleaf web editor, and any edit made *in* Overleaf
(by me or by a supervisor) had no way back into git. Two consequences:

1. Every structural check on the manuscript — citekeys resolving, `\label`/`\Cref`
   consistency, environment balance, figure paths — was done by inspection,
   because nothing ever compiled the document.
2. The repo's copy and Overleaf's copy could drift apart silently, with no
   mechanism to detect it and no record of which was newer.

The goal was a scripted two-way sync: the repo stays the source of truth and
keeps the version history, Overleaf stays the editor and the only thing that
actually produces a PDF.

## What Overleaf offers

Three mechanisms exist, all of them **premium** on Overleaf Cloud (this account
has access; many universities also grant it free via Overleaf Commons):

| Mechanism | How it works | Verdict |
|---|---|---|
| **Git integration** | Each project exposes a real git remote at `https://git.overleaf.com/<project-id>`; authenticate with username `git` and an `olp_` token from Account Settings → Project Synchronization. | **Chosen.** |
| **GitHub synchronization** | Links a project to a github.com repo, synced manually from the Integrations menu. | Rejected — cannot connect an *existing* project to an *existing* repo, maps the project to the repo root, and would have needed a third mirror repo. |
| **Dropbox sync** | Folder-level sync. | Rejected — no version history, no review step. |

Free-tier fallbacks exist (cookie-based CLIs such as the `overleaf-sync` forks,
or manual zip upload/download) but were not needed and are not maintained by
Overleaf.

## The constraint that shaped the design

**Overleaf maps a project to the root of its git repo.** Our manuscript is at
`thesis/manuscript/`, a subdirectory of a repo whose `.git` is **2.6 GB**
(datasets, model exports, benchmark outputs) — far past Overleaf's guidance of
under ~100 files per commit and under ~100 MB per project.

So the obvious `git remote add overleaf … && git push` was never available. The
sync had to *project* `thesis/manuscript/` onto the Overleaf project root.

### Two ways to do that, and why the mirror won

**`git subtree push/pull --prefix=thesis/manuscript`** preserves real history
linkage and detects conflicts as genuine git merges. It was rejected because:

- It requires a strict **always-pull-before-push** invariant. `git subtree push`
  pushes the split head; if the Overleaf side has moved, the split is behind and
  the push is rejected — recoverable, but a footgun in routine use.
- Bootstrapping it into a project that already has a commit needs either a
  force-push or an `--allow-unrelated-histories` merge whose root files land in
  the wrong place. **Overleaf's docs do not state whether the bridge accepts a
  force-push**, and this is not a good thing to discover experimentally on the
  repository holding the thesis.
- The benefit it buys — shared history — is worth little here, because the
  authoritative history is this repo's, and the Overleaf side is a working copy.

**A dedicated clone plus tree mirroring** was chosen instead. It never
force-pushes, never rewrites history, and surfaces every incoming Overleaf change
as an ordinary `git diff` in this repo, reviewed before it is committed. The two
histories stay independent, which costs nothing.

As it turned out, the Overleaf project was created empty, so the first push was a
plain fast-forward and the force-push question never arose at all.

## The design

```
thesis/manuscript/  ←—— mirror ——→  ~/overleaf/master-thesis  ←—— git ——→  Overleaf
     (30 files, 1.4 MB)                  (clone, branch `main`)
```

- **Clone:** `~/overleaf/master-thesis`, remote
  `https://git.overleaf.com/6aaec9482e7b434d43e2c4b6`, default branch **`main`**
  (not `master`). Override the path with `$OVERLEAF_CLONE`.
- **Script:** `scripts/thesis/sync_overleaf.py`, run from the repo root with the
  standard project convention — no container needed, `python-dotenv` plus stdlib
  only, no new dependencies.

  ```
  make overleaf                              # the usual one — picks a direction
  make overleaf-status                       # read-only
  make overleaf-pull
  make overleaf-push OVERLEAF_MSG="message"
  ```

  The targets are thin wrappers over the script, which is equally callable
  directly as `uv run python -m scripts.thesis.sync_overleaf status|pull|push`.
  `OVERLEAF_ARGS` forwards extra flags (e.g. `--allow-any-branch`), and
  `OVERLEAF_CLONE` overrides the clone location.

### `make overleaf`: one command, four outcomes

`make overleaf` is the everyday entry point, and it is a **fourth subcommand
rather than a chain of the other two**. Chaining could not work: `pull` refuses
into a dirty tree, so `overleaf: overleaf-pull overleaf-push` would fail exactly
when you had been writing — which is every time you would want to run it.

Instead `sync` decides. The clone's working tree is kept clean, so it *is* the
last synced state; any difference between it and `thesis/manuscript/` means this
side moved, while commits on `origin/main` mean the other side did:

| Overleaf moved | Manuscript moved | `sync` does |
|---|---|---|
| no | no | nothing — reports both sides identical |
| no | **yes** | **push** |
| **yes** | no | **pull**, leaving it uncommitted for review |
| **yes** | **yes** | **stops** and explains the two ways out |

The fourth row is the point of the command. Two-way sync tools usually resolve
divergence by guessing — last-writer-wins, or a merge that mangles LaTeX. This
one stops, names both sides, and offers the two honest resolutions: commit and
pull and reapply, or push and let your version win. Refusing is the feature.

One subtlety in row three. `pull` normally requires a clean tree because it
overwrites the working tree, but `sync` reaches that row only after proving the
two trees are byte-identical — at which point a pull can only rewrite files
Overleaf actually changed, so uncommitted local edits cannot be lost. `sync`
passes that proof to `cmd_pull` as `_trees_identical`, which is the only thing
that relaxes the guard, and it can never be set from the command line.

### The `make` interface

The subcommands are also `Makefile` targets, which is the documented entry
point — it saves remembering the module path and keeps the sync alongside the
repo's other operational commands:

```make
OVERLEAF_MSG  ?=
OVERLEAF_ARGS ?=
_OVERLEAF := uv run python -m scripts.thesis.sync_overleaf

overleaf:
	$(_OVERLEAF) sync $(if $(OVERLEAF_MSG),-m "$(OVERLEAF_MSG)") $(OVERLEAF_ARGS)

overleaf-status:
	$(_OVERLEAF) status $(OVERLEAF_ARGS)

overleaf-pull:
	$(_OVERLEAF) pull $(OVERLEAF_ARGS)

overleaf-push:
	$(_OVERLEAF) push $(if $(OVERLEAF_MSG),-m "$(OVERLEAF_MSG)") $(OVERLEAF_ARGS)
```

Three details worth recording:

- **They live in their own `─── Thesis: Overleaf ───` section, deliberately
  separate from the existing `─── Sync ───` block.** That block is unrelated —
  it rsyncs gitignored data to `gpu.local`, the NAS and `ics-server`. Two things
  called "sync" in one Makefile is exactly the sort of collision that gets the
  wrong one run at 2 a.m., so the names are `overleaf-*` throughout and the
  sections do not touch.
- **`OVERLEAF_MSG` and `OVERLEAF_ARGS` follow the existing `YOLOV5S_ARGS` /
  `SPECIESNET_ARGS` convention** rather than inventing a new one. The
  `$(if ...)` guard means a bare `make overleaf-push` falls through to the
  script's generated default message (`sync: manuscript from <branch>@<sha>`),
  while `make overleaf-push OVERLEAF_MSG="…"` overrides it.
- **They run on the host, not in the training container**, and rely on `uv` being
  on `PATH` — which it is via pyenv global `thesis`, the same assumption the
  existing `yolov5s-train` target already makes.

### Token handling

`OVERLEAF_TOKEN` lives in `.env` and nowhere else. `.env` is gitignored
(`.gitignore:2`) and already holds `HF_TOKEN` and `OPENROUTER_API_KEY`, so this
follows the existing convention rather than inventing a second secret store — a
copy in `~/.git-credentials` would only widen the exposure.

The script loads it with `load_dotenv()` and hands it to git through an
**in-memory credential helper** passed per-invocation:

```
git -c credential.helper= \
    -c 'credential.helper=!f() { test "$1" = get && { echo username=git; echo "password=$OVERLEAF_TOKEN"; }; }; f' \
    -c credential.https://git.overleaf.com.username=git …
```

The first `-c` clears the globally configured `store` helper for that call. The
token is supplied in the subprocess environment with `GIT_TERMINAL_PROMPT=0`, so
it never reaches disk, the clone's config, an argv, or shell history.

### The designated branch

All sync runs happen on **`thesis/overleaf-sync`**, branched from
`feat/embedded-benchmark-rpi400`. It merges into `main` like any other branch.

This is not bookkeeping preference: the Overleaf git bridge **has no branch
support**, so its single `main` must correspond to exactly one branch here. If
two branches both synced, the manuscript would fork in two with no signal that it
had happened. All three subcommands refuse to run elsewhere;
`--allow-any-branch` overrides.

### Safety model

Every guard fails closed, and nothing force-pushes or rewrites history:

| Guard | Prevents |
|---|---|
| `pull` refuses if `thesis/manuscript/` is dirty | Overwriting uncommitted local edits — a pull only ever runs into a clean tree. |
| `pull` does not commit | Incoming edits are left in the working tree for `git diff` review, so nothing lands unseen. |
| `push` refuses if Overleaf has unpulled commits | Clobbering a supervisor's edits; it directs you to `pull` first. |
| `push` refuses if the clone's tree is dirty | Pushing the wreckage of an earlier run that died part-way. |
| Both refuse off `thesis/overleaf-sync` | Forking the manuscript across branches. |
| Mirroring compares **checksums** | Silently dropping a real edit — see below. |
| `sync` refuses when both sides moved | Guessing a winner on a genuine divergence — it names both sides and stops. |

## Verification (2026-09-19)

| Check | Result |
|---|---|
| Initial import | Commit `86409a0`, **30 files, 4 898 insertions** — all chapters, preamble, bibliography, appendix and 6 PNG plots. |
| Trees match | `diff -r --exclude=.git thesis/manuscript ~/overleaf/master-thesis` → identical. |
| Idempotence | Re-running `push` with no edits → "Nothing to push". |
| `pull` into dirty tree | Refused, exit 1, listing the 9 offending files. |
| `push` with Overleaf ahead | Refused against the **live remote** (an empty commit was pushed from the clone to create the condition). |
| Wrong branch | Refused on `main`. |
| `mirror()` in isolation | Both the rsync and the stdlib-fallback paths correctly add, overwrite, delete stale files, and preserve `dst/.git`. |
| Token containment | `~/.git-credentials` absent; no credential entries in the clone's config; the literal token value appears only in `.env` — not in the clone, `~/.gitconfig` or shell history. |
| `make` targets | All four documented invocations expand to the right command (`make -n`); `make overleaf-push` runs clean end-to-end. |
| `make` guard propagation | `make overleaf-status` on `main` refuses **and** exits non-zero, so a guard cannot be swallowed by make. |
| `make` on a clean shell | Verified in a login shell rather than this session, whose forced `PYENV_VERSION` hid `uv`. |
| `make overleaf`, all four rows | Exercised against a **throwaway local remote**, not the real project, so its history stays clean: in-sync → no-op; local-only → push; Overleaf-only → pull; both → refused, non-zero exit. `appendices/appendix.tex` restored byte-for-byte afterwards. |
| `pull` change reporting | Reports the files the mirror actually touched, verified to show **1 file** for a one-line incoming edit against an already-dirty tree. |

### Three bugs found during verification

**1. rsync silently skipped same-size edits.**

The first implementation mirrored with `rsync -a --delete`, whose default
comparison is **size + mtime**. The isolation test caught it immediately: a file
changed from `v1` to `v2` was *not* copied, because the edit preserved the file
size and landed in the same second as the previous write.

For LaTeX this is not a hypothetical — most edits are a word or a character and
leave the byte count unchanged. The fix is `--checksum`, which costs nothing on a
1.4 MB tree and removes the entire failure class. The stdlib fallback path was
never affected (`shutil.copy2` always overwrites) but is tested the same way.

**2. `--allow-any-branch` only parsed before the subcommand.** Adding the make
targets exposed it. The flag was defined on the top-level `argparse` parser only,
so `sync_overleaf.py status --allow-any-branch` died with `unrecognized
arguments` — and since `OVERLEAF_ARGS` is appended *after* the subcommand, that
was precisely the form the README and this document had already recommended. The
escape hatch was documented and broken at the same time.

The fix is a shared parent parser carrying the flag into each subparser, with
`default=argparse.SUPPRESS` on that copy so the subparser's default cannot
clobber a value given up front. Both orderings now work, and the guard still
refuses without the flag.

**3. `pull` over-reported what it had changed.** Testing `make overleaf`'s
Overleaf-moved row surfaced it: `pull` printed `git diff --stat`, which compares
the working tree to git HEAD, not to what the pull brought in. With nine files
already dirty, a one-line incoming edit was reported as *"10 files changed, 3145
insertions(+), 3038 deletions(-)"*. Harmless to the data, but for a tool whose
whole job is not losing work, a report that cannot be trusted is close to no
report. It now diffs the tree digest either side of the mirror and names the
files actually touched.

The pattern in all three bugs is the same: the failure was in the path *around* the
feature — the comparison heuristic, the argument position — not in the feature
itself, and only an explicit test of the documented invocation caught it.

## Known caveats

- **Track changes and comments do not survive git round-trips.** Overleaf's own
  docs advise against mixing them with git. Read supervisor comments in the
  Overleaf editor and apply them as ordinary edits; do not expect them to reach
  the repo.
- The bridge supports no branches, tags, symlinks or LFS.
- `main.tex` loads `\usepackage{svg}`, which needs shell-escape at `\includesvg`
  time. Currently unused — everything in `figures/plots/` is PNG — but this needs
  verifying on Overleaf before an SVG figure is added.
- `glossaries` + `\makeglossaries`: confirm Overleaf runs `makeglossaries` as
  expected on the first compile.
- The `.gitkeep` files under `figures/images/` appear as empty folders in Overleaf.
- Overleaf history now contains one **empty** commit, `ca8b0db "simulated
  Overleaf-side edit"`, left by the push-guard test. It changes no files;
  removing it would require the force-push this design deliberately avoids.

## Preconditions the manuscript satisfies

Only `thesis/manuscript/` is synced — anything outside it is invisible to
Overleaf. Re-check these if the structure changes:

- Every `\input`, `\include`, `\addbibresource` and the `\graphicspath` entry in
  `main.tex` is relative and stays inside `thesis/manuscript/`.
- All **98 citekeys** used across `chapters/`, `appendices/` and `preamble/`
  resolve in `bibliography/references.bib` (219 entries). Nothing depends on the
  second bibliography at `research/literature/references.bib`, which lies outside
  the synced tree.
- Engine is pdfLaTeX-compatible: `scrreprt` + `inputenc utf8`, no `fontspec`.
  Bibliography is `biblatex` with `backend=biber`, which Overleaf detects
  automatically. Set Menu → Compiler = **pdfLaTeX**.
- 30 files, 1.4 MB — comfortably inside Overleaf's project limits.

## Open items

1. **The live `pull` round-trip is not yet verified end-to-end.**
   `thesis/manuscript/` had 9 pre-existing uncommitted changes at implementation
   time, and `pull` refuses into a dirty tree by design. `mirror()` was verified
   in isolation and `git pull --ff-only` is exercised by every `push`, but the
   full path should be confirmed once the tree is clean: edit one line in the
   Overleaf editor, run `pull`, check that exact line appears in
   `git diff thesis/manuscript` and nothing else changed.
2. **The manuscript has never actually been compiled.** The first Overleaf
   compile is the real test — expect genuine LaTeX errors rather than sync
   errors. The two to watch are the `glossaries` run and all 98 citations
   resolving without `[?]`.

## Sources

- [Git integration](https://docs.overleaf.com/integrations-and-add-ons/git-integration-and-github-synchronization/git-integration)
- [Git integration authentication tokens](https://docs.overleaf.com/integrations-and-add-ons/git-integration-and-github-synchronization/git-integration/git-integration-authentication-tokens)
- [Advanced Git operations](https://docs.overleaf.com/integrations-and-add-ons/git-integration-and-github-synchronization/git-integration/advanced-git-operations)
- [GitHub synchronization](https://docs.overleaf.com/integrations-and-add-ons/git-integration-and-github-synchronization/github-synchronization)
