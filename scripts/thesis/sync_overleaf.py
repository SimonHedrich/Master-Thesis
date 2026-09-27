"""Two-way sync between `thesis/manuscript/` and the Overleaf project's git repo.

Usage:
    uv run python -m scripts.thesis.sync_overleaf sync          (make overleaf)
    uv run python -m scripts.thesis.sync_overleaf status
    uv run python -m scripts.thesis.sync_overleaf pull
    uv run python -m scripts.thesis.sync_overleaf push -m "message"
    uv run python -m scripts.thesis.sync_overleaf push --force   (make overleaf-force-push)

Run from the repo root (no container needed), on the designated sync branch
`thesis/overleaf-sync`. Requires `OVERLEAF_TOKEN` in `.env` and a clone of the
Overleaf project at `~/overleaf/master-thesis` (override with `$OVERLEAF_CLONE`).

Overleaf maps a project to the *root* of its git repo, so the manuscript cannot be
pushed as a subdirectory of this repo. This script mirrors the file tree between
`thesis/manuscript/` and a dedicated clone instead. Incoming Overleaf edits land in
the working tree and are left uncommitted so they can be reviewed with `git diff`.

See scripts/thesis/README.md.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
MANUSCRIPT = REPO_ROOT / "thesis" / "manuscript"
MANUSCRIPT_REL = "thesis/manuscript"
SYNC_BRANCH = "thesis/overleaf-sync"
DEFAULT_CLONE = Path.home() / "overleaf" / "master-thesis"

# Feeds the token to git from the environment only: never written to disk, never in argv.
_CRED_HELPER = (
    '!f() { test "$1" = get && { echo username=git; echo "password=$OVERLEAF_TOKEN"; }; }; f'
)


class SyncError(RuntimeError):
    """Aborts the run with a user-facing message and no traceback."""


def clone_path() -> Path:
    return Path(os.environ.get("OVERLEAF_CLONE", str(DEFAULT_CLONE))).expanduser()


def git(*args: str, cwd: Path, auth: bool = False, check: bool = True) -> str:
    """Run git and return stdout. `auth` injects the Overleaf credential helper."""
    cmd = ["git"]
    env = os.environ.copy()
    if auth:
        token = env.get("OVERLEAF_TOKEN")
        if not token:
            raise SyncError(
                "OVERLEAF_TOKEN is not set. Add it to .env (it is gitignored) — see "
                "scripts/thesis/README.md."
            )
        cmd += [
            "-c",
            "credential.helper=",  # drop any globally configured helper for this call
            "-c",
            f"credential.helper={_CRED_HELPER}",
            "-c",
            "credential.https://git.overleaf.com.username=git",
        ]
        env["GIT_TERMINAL_PROMPT"] = "0"
    result = subprocess.run(
        cmd + list(args), cwd=cwd, env=env, capture_output=True, text=True
    )
    if check and result.returncode != 0:
        raise SyncError(
            f"git {' '.join(args)} failed in {cwd}:\n{result.stderr.strip() or result.stdout.strip()}"
        )
    return result.stdout.strip()


def mirror(src: Path, dst: Path) -> None:
    """Make `dst` match `src` exactly, leaving `dst/.git` untouched."""
    if shutil.which("rsync"):
        # --checksum, not the default size+mtime heuristic: a one-character LaTeX edit
        # keeps the file size, and a same-second mtime would then be skipped silently.
        subprocess.run(
            ["rsync", "-a", "--checksum", "--delete", "--exclude", ".git/", f"{src}/", f"{dst}/"],
            check=True,
        )
        return
    for entry in dst.iterdir():
        if entry.name == ".git":
            continue
        shutil.rmtree(entry) if entry.is_dir() else entry.unlink()
    for entry in src.iterdir():
        if entry.name == ".git":
            continue
        target = dst / entry.name
        shutil.copytree(entry, target) if entry.is_dir() else shutil.copy2(entry, target)


def tree_digest(root: Path) -> dict[str, str]:
    """Content hash per file under `root`, ignoring `.git`. Two equal digests mean
    the trees are byte-identical, which is what `sync` needs to tell the sides apart."""
    digest: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if path.is_dir() or ".git" in rel.parts:
            continue
        digest[str(rel)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def check_setup(allow_any_branch: bool) -> Path:
    clone = clone_path()
    if not (clone / ".git").is_dir():
        raise SyncError(
            f"No Overleaf clone at {clone}.\n"
            "  git clone https://git.overleaf.com/<project-id> ~/overleaf/master-thesis\n"
            "(username `git`, password the olp_ token from .env)"
        )
    branch = git("branch", "--show-current", cwd=REPO_ROOT)
    if branch != SYNC_BRANCH and not allow_any_branch:
        raise SyncError(
            f"On branch '{branch}', but the Overleaf bridge has no branch support — one\n"
            f"Overleaf project must map to exactly one branch here, or the manuscript forks.\n"
            f"  git switch {SYNC_BRANCH}\n"
            "Pass --allow-any-branch if you really mean to sync from elsewhere."
        )
    return clone


def manuscript_dirty() -> str:
    return git("status", "--porcelain", "--", MANUSCRIPT_REL, cwd=REPO_ROOT)


def cmd_status(args: argparse.Namespace) -> int:
    clone = check_setup(args.allow_any_branch)
    git("fetch", "origin", cwd=clone, auth=True)
    local = git("rev-parse", "HEAD", cwd=clone)
    remote = git("rev-parse", "@{u}", cwd=clone)
    behind, ahead = git("rev-list", "--left-right", "--count", "@{u}...HEAD", cwd=clone).split()

    print(f"repo branch    : {git('branch', '--show-current', cwd=REPO_ROOT)}")
    print(f"overleaf clone : {clone}")
    print(f"  remote       : {git('remote', 'get-url', 'origin', cwd=clone)}")
    print(f"  HEAD         : {git('log', '-1', '--format=%h %s', cwd=clone)}")
    if local == remote:
        print("  vs Overleaf  : up to date")
    else:
        print(f"  vs Overleaf  : {behind} commit(s) to pull, {ahead} to push")
    local_changes = git("status", "--porcelain", cwd=clone)
    if local_changes:
        print("  WARNING: clone working tree is dirty (leftover from a failed run?):")
        print("\n".join(f"    {line}" for line in local_changes.splitlines()))

    dirty = manuscript_dirty()
    print(f"{MANUSCRIPT_REL}: {'dirty — ' + str(len(dirty.splitlines())) + ' change(s)' if dirty else 'clean'}")
    return 0


def cmd_pull(args: argparse.Namespace, _trees_identical: bool = False) -> int:
    clone = check_setup(args.allow_any_branch)
    dirty = manuscript_dirty()
    # `sync` sets _trees_identical when it has already proved the two trees match
    # byte for byte. A pull can then only rewrite files Overleaf actually changed,
    # so uncommitted local edits — already present on both sides — cannot be lost.
    if dirty and not _trees_identical:
        raise SyncError(
            f"{MANUSCRIPT_REL} has uncommitted changes, which a pull would overwrite.\n"
            + "\n".join(f"  {line}" for line in dirty.splitlines())
            + "\nCommit or stash them first, or push them to Overleaf instead."
        )
    before = git("rev-parse", "HEAD", cwd=clone)
    git("pull", "--ff-only", cwd=clone, auth=True)
    after = git("rev-parse", "HEAD", cwd=clone)
    if before == after:
        print("Overleaf has no new commits; mirroring anyway to confirm the trees match.")

    # Report what the mirror actually did, rather than `git diff` against HEAD:
    # the tree may already have been dirty, and conflating that with the incoming
    # changes would make a one-line edit look like a ten-file rewrite.
    was = tree_digest(MANUSCRIPT)
    mirror(clone, MANUSCRIPT)
    now = tree_digest(MANUSCRIPT)

    touched = (
        [f"  added    {k}" for k in sorted(now.keys() - was.keys())]
        + [f"  deleted  {k}" for k in sorted(was.keys() - now.keys())]
        + [f"  changed  {k}" for k in sorted(was.keys() & now.keys()) if was[k] != now[k]]
    )
    if not touched:
        print(f"Nothing changed — {MANUSCRIPT_REL} already matches Overleaf.")
        return 0

    print(f"Pulled Overleaf {before[:7]}..{after[:7]} — {len(touched)} file(s) in {MANUSCRIPT_REL}:\n")
    print("\n".join(touched))
    print(f"\nReview with `git diff {MANUSCRIPT_REL}`, then commit when you are happy.")
    return 0


def cmd_push(args: argparse.Namespace) -> int:
    force = getattr(args, "force", False)
    clone = check_setup(args.allow_any_branch)
    if git("status", "--porcelain", cwd=clone):
        if not force:
            raise SyncError(
                f"The Overleaf clone at {clone} has uncommitted changes — a previous run\n"
                "probably failed part-way. Inspect it and reset before pushing again."
            )
        # --force means "the manuscript wins", so a half-finished previous run in the
        # clone is just noise: discard it rather than making the user clean up by hand.
        print(f"--force: discarding uncommitted changes in {clone}.")
        git("reset", "--hard", cwd=clone)
        git("clean", "-fd", cwd=clone)
    before = git("rev-parse", "HEAD", cwd=clone)
    git("pull", "--ff-only", cwd=clone, auth=True)
    after = git("rev-parse", "HEAD", cwd=clone)
    if before != after:
        incoming = git("rev-list", "--count", f"{before}..{after}", cwd=clone)
        if not force:
            raise SyncError(
                f"Overleaf has {incoming} new commit(s) that are not in this repo yet.\n"
                "Pushing now would overwrite them. Run this first, review, and commit:\n"
                "  uv run python -m scripts.thesis.sync_overleaf pull"
            )
        # Fast-forwarding first and then mirroring over the result keeps the Overleaf
        # history linear — the edits stay in its log, they are just superseded by the
        # commit this makes. No history is rewritten, so the bridge accepts the push.
        print(
            f"--force: Overleaf has {incoming} new commit(s) that this repo has not "
            f"pulled.\nTheir content will be overwritten by {MANUSCRIPT_REL}:\n"
        )
        print(git("log", "--format=  %h %s", f"{before}..{after}", cwd=clone))
        print()

    mirror(MANUSCRIPT, clone)
    git("add", "-A", cwd=clone)
    if not git("diff", "--cached", "--name-only", cwd=clone):
        print(f"Nothing to push — Overleaf already matches {MANUSCRIPT_REL}.")
        return 0

    message = args.message or (
        f"{'force-sync' if force else 'sync'}: manuscript from "
        f"{git('branch', '--show-current', cwd=REPO_ROOT)}@"
        f"{git('rev-parse', '--short', 'HEAD', cwd=REPO_ROOT)}"
    )
    print(git("diff", "--cached", "--stat", cwd=clone))
    git("commit", "-m", message, cwd=clone)
    git("push", cwd=clone, auth=True)
    print(f"\nPushed to Overleaf: {git('log', '-1', '--format=%h %s', cwd=clone)}")
    return 0


def cmd_sync(args: argparse.Namespace) -> int:
    """Work out which side moved and do that. Refuses when both did."""
    clone = check_setup(args.allow_any_branch)
    if git("status", "--porcelain", cwd=clone):
        raise SyncError(
            f"The Overleaf clone at {clone} has uncommitted changes — a previous run\n"
            "probably failed part-way. Inspect it and reset before syncing again."
        )
    git("fetch", "origin", cwd=clone, auth=True)
    incoming = int(git("rev-list", "--count", "HEAD..@{u}", cwd=clone))
    # The clone's tree is clean, so it *is* the last synced state: any difference
    # against it means this side moved. `incoming` means the other side moved.
    local_changed = tree_digest(MANUSCRIPT) != tree_digest(clone)

    if incoming and local_changed:
        raise SyncError(
            f"Both sides changed — Overleaf has {incoming} new commit(s) and\n"
            f"{MANUSCRIPT_REL} has edits of its own. Merging them is a judgement call,\n"
            "so this stops rather than picking a winner. Either:\n"
            "  - commit or stash your edits, `make overleaf-pull`, then reapply them; or\n"
            "  - `make overleaf-push` to let your version win (discards the Overleaf edits)."
        )
    if incoming:
        print(f"Overleaf moved ({incoming} new commit(s)) and {MANUSCRIPT_REL} did not — pulling.\n")
        return cmd_pull(args, _trees_identical=not local_changed)
    if local_changed:
        print(f"{MANUSCRIPT_REL} moved and Overleaf did not — pushing.\n")
        return cmd_push(args)

    print(f"Already in sync — {MANUSCRIPT_REL} and Overleaf are identical.")
    return 0


def main() -> int:
    load_dotenv(REPO_ROOT / ".env")
    branch_help = f"sync from a branch other than {SYNC_BRANCH} (risks forking the manuscript)"
    # Accepted on either side of the subcommand, so `push --allow-any-branch` works as
    # readily as `--allow-any-branch push`. SUPPRESS on the shared copy stops the
    # subparser's default from clobbering the flag when it was given up front.
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--allow-any-branch",
        action="store_true",
        default=argparse.SUPPRESS,
        help=branch_help,
    )

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--allow-any-branch", action="store_true", help=branch_help)
    sub = parser.add_subparsers(dest="command", required=True)
    sync = sub.add_parser(
        "sync",
        parents=[shared],
        help="pull or push, whichever the two sides call for; refuse if both moved",
    )
    sync.add_argument("-m", "--message", help="commit message, if this turns out to be a push")
    sub.add_parser(
        "status",
        parents=[shared],
        help="report what each side has, without changing anything",
    )
    sub.add_parser(
        "pull",
        parents=[shared],
        help="bring Overleaf edits into thesis/manuscript/ for review",
    )
    push = sub.add_parser("push", parents=[shared], help="send thesis/manuscript/ to Overleaf")
    push.add_argument("-m", "--message", help="commit message for the Overleaf-side commit")
    push.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="overwrite Overleaf with the local manuscript even if Overleaf is ahead",
    )

    args = parser.parse_args()
    handler = {
        "sync": cmd_sync,
        "status": cmd_status,
        "pull": cmd_pull,
        "push": cmd_push,
    }[args.command]
    try:
        return handler(args)
    except SyncError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
