"""Structural integrity and length check for the thesis manuscript.

No LaTeX toolchain is installed in this environment, so this script covers the
failure modes that prose deletion actually causes -- severed cross-references,
citation keys left behind, labels nobody points at -- and measures every file
against the standing word budget in
`.claude/skills/thesis-writing/references/scope.md` section 9.

Run from the repository root:

    uv run python -m scripts.thesis.check_manuscript

Exits non-zero if any integrity check fails. Length overruns are reported but
do not fail the run, since a file over target needs a judgment call, not a gate.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

MANUSCRIPT = Path("thesis/manuscript")

# Target word counts per file, from scope.md section 9. These are ceilings to hold,
# not targets to shrink toward: they record the allocation the September 2026
# shortening pass actually reached, after each file had been cut to the point where
# the next cut would have removed evidence rather than noise.
BUDGET: dict[str, int] = {
    "chapters/1-Introduction.tex": 2000,
    "chapters/2-Literature_Review.tex": 6800,
    "chapters/3-Methods_and_Implementation/30-Overview.tex": 170,
    "chapters/3-Methods_and_Implementation/31-Data_Sourcing_and_Taxonomy.tex": 1400,
    "chapters/3-Methods_and_Implementation/32-Data_Quality_and_Curation.tex": 2100,
    "chapters/3-Methods_and_Implementation/33-Synthetic_Data_Supplementation.tex": 1700,
    "chapters/3-Methods_and_Implementation/34-Data_Augmentation.tex": 800,
    "chapters/3-Methods_and_Implementation/35-Synthetic_Generator_Comparison.tex": 2550,
    "chapters/3-Methods_and_Implementation/36-Model_Training_and_Experiment_Tracking.tex": 1500,
    "chapters/3-Methods_and_Implementation/37-Evaluation_Framework.tex": 1550,
    "chapters/4-Results.tex": 7400,
    "chapters/5-Discussion_and_Conclusion.tex": 3000,
    "appendices/appendix.tex": 5500,
}

LABEL_RE = re.compile(r"\\label\{([^}]*)\}")
REF_RE = re.compile(r"\\(?:auto|name|c|C|v|V|full|page)?ref\*?\{([^}]*)\}")
CREF_RE = re.compile(r"\\[Cc]refrange\{([^}]*)\}\{([^}]*)\}")
HYPERREF_RE = re.compile(r"\\hyperref\[([^]]*)\]")
CITE_RE = re.compile(r"\\(?:cite|textcite|parencite|citeauthor|citeyear|footcite)\*?"
                     r"(?:\[[^]]*\])*\{([^}]*)\}")
BIBKEY_RE = re.compile(r"^\s*@[A-Za-z]+\s*\{\s*([^,\s]+)\s*,", re.MULTILINE)
# \newacronym{key}{SHORT}{long expansion} -- the short form is what appears in prose.
ACRONYM_RE = re.compile(r"\\newacronym\{[^}]*\}\{([^}]*)\}")
GRAPHIC_RE = re.compile(r"\\includegraphics(?:\[[^]]*\])?\{([^}]*)\}")

# Comment stripping: a % not preceded by a backslash starts a comment.
COMMENT_RE = re.compile(r"(?<!\\)%.*$", re.MULTILINE)
# Environments whose bodies are not prose and should not count as words.
FLOAT_RE = re.compile(
    r"\\begin\{(table|tabular|tabularx|figure|lstlisting|verbatim|equation|align|tikzpicture)\*?\}"
    r".*?\\end\{\1\*?\}",
    re.DOTALL,
)
MACRO_RE = re.compile(r"\\[A-Za-z@]+\*?(?:\[[^]]*\])*")


@dataclass
class Findings:
    errors: list[str] = field(default_factory=list)
    warnings: dict[str, list[str]] = field(default_factory=dict)

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, category: str, msg: str) -> None:
        self.warnings.setdefault(category, []).append(msg)

    def report(self, verbose: bool) -> None:
        if self.warnings:
            total = sum(len(v) for v in self.warnings.values())
            print(f"\n{total} warning(s)")
            for category, msgs in self.warnings.items():
                print(f"  {category}: {len(msgs)}")
                shown = msgs if verbose else msgs[:3]
                for msg in shown:
                    print(f"    - {msg}")
                if not verbose and len(msgs) > len(shown):
                    print(f"    ... {len(msgs) - len(shown)} more (pass -v to list)")
        if self.errors:
            print(f"\n{len(self.errors)} error(s)")
            for msg in self.errors:
                print(f"  ERROR: {msg}")


def tex_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.tex") if ".git" not in p.parts)


def strip_comments(text: str) -> str:
    return COMMENT_RE.sub("", text)


def raw_words(text: str) -> int:
    """What `wc -w` counts. The word budget in scope.md section 9 is stated in these."""
    return len(text.split())


def prose_words(text: str) -> int:
    """Approximate the prose word count: no comments, floats, or macro names."""
    body = strip_comments(text)
    body = FLOAT_RE.sub(" ", body)
    body = MACRO_RE.sub(" ", body)
    body = re.sub(r"[{}$&~^_\\]", " ", body)
    return len([w for w in body.split() if any(c.isalnum() for c in w)])


def collect_labels_and_refs(files: list[Path]) -> tuple[dict[str, Path], dict[str, list[Path]]]:
    labels: dict[str, Path] = {}
    refs: dict[str, list[Path]] = {}
    for path in files:
        text = strip_comments(path.read_text(encoding="utf-8"))
        for name in LABEL_RE.findall(text):
            labels.setdefault(name.strip(), path)
        targets = [m.strip() for m in REF_RE.findall(text)]
        targets += [m.strip() for m in HYPERREF_RE.findall(text)]
        for start, end in CREF_RE.findall(text):
            targets += [start.strip(), end.strip()]
        for group in targets:
            for name in (n.strip() for n in group.split(",")):
                if name:
                    refs.setdefault(name, []).append(path)
    return labels, refs


def collect_cites(files: list[Path]) -> dict[str, list[Path]]:
    cites: dict[str, list[Path]] = {}
    for path in files:
        text = strip_comments(path.read_text(encoding="utf-8"))
        for group in CITE_RE.findall(text):
            for key in (k.strip() for k in group.split(",")):
                if key:
                    cites.setdefault(key, []).append(path)
    return cites


def bib_keys(root: Path) -> set[str]:
    keys: set[str] = set()
    for bib in sorted((root / "bibliography").glob("*.bib")):
        keys |= set(BIBKEY_RE.findall(bib.read_text(encoding="utf-8")))
    return keys


def check_references(files: list[Path], f: Findings) -> None:
    labels, refs = collect_labels_and_refs(files)
    dangling = sorted(set(refs) - set(labels))
    for name in dangling:
        where = ", ".join(sorted({str(p.relative_to(MANUSCRIPT)) for p in refs[name]}))
        f.error(f"dangling reference: \\ref{{{name}}} in {where} -- no \\label defines it")
    orphans = sorted(set(labels) - set(refs))
    for name in orphans:
        f.warn("orphaned label", f"{name} ({labels[name].relative_to(MANUSCRIPT)})")
    print(f"  {len(labels)} labels, {len(refs)} referenced targets, "
          f"{len(dangling)} dangling, {len(orphans)} orphaned")


def check_citations(files: list[Path], f: Findings) -> None:
    cites = collect_cites(files)
    known = bib_keys(MANUSCRIPT)
    unresolved = sorted(set(cites) - known)
    for key in unresolved:
        where = ", ".join(sorted({str(p.relative_to(MANUSCRIPT)) for p in cites[key]}))
        f.error(f"unresolved citation key: {key} (cited in {where})")
    calls = sum(len(v) for v in cites.values())
    print(f"  {len(cites)} distinct keys over {calls} calls against {len(known)} bib entries, "
          f"{len(unresolved)} unresolved")


def check_acronyms(files: list[Path], f: Findings) -> None:
    acr_file = MANUSCRIPT / "preamble" / "acronyms.tex"
    if not acr_file.exists():
        return
    declared = ACRONYM_RE.findall(strip_comments(acr_file.read_text(encoding="utf-8")))
    body = "\n".join(
        strip_comments(p.read_text(encoding="utf-8"))
        for p in files
        if p != acr_file
    )
    for short in declared:
        # The chapters spell abbreviations out directly rather than via \gls.
        if not re.search(rf"\b{re.escape(short)}\b", body):
            f.warn("unused acronym", f"{short} is declared in acronyms.tex but never used")


def check_graphics(files: list[Path], f: Findings) -> None:
    fig_root = MANUSCRIPT / "figures"
    if not fig_root.is_dir():
        return
    used: set[str] = set()
    for path in files:
        for g in GRAPHIC_RE.findall(strip_comments(path.read_text(encoding="utf-8"))):
            used.add(Path(g).stem)
    on_disk = {p.stem for p in fig_root.rglob("*") if p.is_file() and p.suffix.lower()
               in {".png", ".jpg", ".jpeg", ".pdf", ".svg", ".eps"}}
    for stem in sorted(on_disk - used):
        f.warn("unused figure file", f"{stem} exists in figures/ but is included nowhere")


def check_lengths(f: Findings) -> int:
    total = prose_total = 0
    print(f"  {'file':<58}{'words':>7}{'target':>8}{'delta':>8}{'prose':>8}")
    for rel, target in BUDGET.items():
        path = MANUSCRIPT / rel
        if not path.exists():
            f.error(f"budgeted file missing: {rel}")
            continue
        text = path.read_text(encoding="utf-8")
        words, prose = raw_words(text), prose_words(text)
        total += words
        prose_total += prose
        delta = words - target
        flag = "" if abs(delta) <= 0.10 * target else ("  OVER" if delta > 0 else "  under")
        short = rel.replace("chapters/3-Methods_and_Implementation/", "3-.../")
        short = short.replace("chapters/", "").replace("appendices/", "")
        print(f"  {short:<58}{words:>7}{target:>8}{delta:>+8}{prose:>8}{flag}")
        if delta > 0.10 * target:
            f.warn("over budget", f"{short}: {words} words, {delta:+} against a target of {target}")
    budget_total = sum(BUDGET.values())
    print(f"  {'TOTAL':<58}{total:>7}{budget_total:>8}{total - budget_total:>+8}{prose_total:>8}")
    return total


def check_prose_shape(files: list[Path], f: Findings) -> None:
    """construction.md: rewrite sentences above 55 words, paragraphs above 180."""
    long_sentences = 0
    long_paragraphs = 0
    for path in files:
        body = FLOAT_RE.sub(" ", strip_comments(path.read_text(encoding="utf-8")))
        for para in re.split(r"\n\s*\n", body):
            if para.lstrip().startswith("\\") and "." not in para:
                continue
            text = MACRO_RE.sub(" ", para)
            text = re.sub(r"[{}$&~^_\\]", " ", text)
            words = [w for w in text.split() if any(c.isalnum() for c in w)]
            if len(words) > 180:
                long_paragraphs += 1
                f.warn("paragraph over 180 words",
                       f"{path.relative_to(MANUSCRIPT)}: {len(words)} words -- "
                       f"'{' '.join(words[:9])}...'")
            for sentence in re.split(r"(?<=[.!?])\s+", text):
                n = len([w for w in sentence.split() if any(c.isalnum() for c in w)])
                if n > 55:
                    long_sentences += 1
    print(f"  {long_sentences} sentences over 55 words, {long_paragraphs} paragraphs over 180")


def main() -> int:
    if not MANUSCRIPT.is_dir():
        print(f"error: run from the repository root; {MANUSCRIPT} not found", file=sys.stderr)
        return 2

    files = tex_files(MANUSCRIPT)
    f = Findings()

    print(f"\nmanuscript check -- {len(files)} .tex files under {MANUSCRIPT}\n")
    print("cross-references")
    check_references(files, f)
    print("\ncitations")
    check_citations(files, f)
    print("\nprose shape")
    check_prose_shape(files, f)
    print("\nlength against budget")
    check_lengths(f)

    check_acronyms(files, f)
    check_graphics(files, f)

    f.report(verbose="-v" in sys.argv or "--verbose" in sys.argv)
    if f.errors:
        return 1
    print("\nno integrity errors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
