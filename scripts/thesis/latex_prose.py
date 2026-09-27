"""Extract improvable prose from LaTeX, and put improved prose back.

DeepL Write has no idea what LaTeX is. Sent a raw manuscript sentence it strips
`\\textit{}`, silently deletes `\\cite{}`, drops `$...$` delimiters, and — the
failure that matters — *invents* cross-references: `\\Cref{sec:results_granularity_gaps}`
came back as the literal text "see section 3.2.1", a section number that is not
the one that label resolves to. Anything that reaches the API must therefore have
its markup removed first, and restored afterwards.

Two jobs here:

`iter_blocks`
    Split a document into blank-line-separated blocks and mark which are prose.
    Headings, labels, comments, and the bodies of tables, figures, tikz pictures
    and display math are not prose and are never sent anywhere.
`mask` / `unmask`
    Replace every LaTeX span inside a prose block with an opaque placeholder,
    then restore them. `unmask` **fails closed**: if a placeholder did not come
    back exactly once, it raises rather than emit LaTeX that lost a citation.

Placeholders are `ZQX<n>ZQX`. The shape is not arbitrary — it was picked by
testing what survives a round trip. `<0>` and `[[0]]` are deleted outright (and
their absence is filled in with invented numbers); a bare-word token like
`ZQX0ZQX` comes back intact and gets treated as a noun, which is what keeps the
surrounding sentence grammatical.

Text-bearing commands (`\\textit`, `\\texttt`, `\\emph`, …) are masked *whole*,
contents included, rather than having their wrapper masked around exposed text.
In this manuscript they hold proper nouns and code identifiers — "AX Visio",
"inovex", "Qualcomm QCS605" — which is exactly the material that must come back
character-identical instead of being rephrased.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

PLACEHOLDER_TEMPLATE = "ZQX{}ZQX"
PLACEHOLDER_RE = re.compile(r"ZQX(\d+)ZQX")

# Commands masked together with all their arguments. Two groups, same treatment:
# references and citations, whose arguments are keys that must survive verbatim;
# and text-bearing commands, whose arguments are proper nouns or code.
MASKED_COMMANDS: frozenset[str] = frozenset({
    # references, citations, labels
    "cite", "textcite", "parencite", "citeauthor", "citeyear", "footcite",
    "ref", "cref", "Cref", "crefrange", "Crefrange", "autoref", "nameref",
    "hyperref", "label", "eqref", "pageref",
    # text-bearing: proper nouns, code identifiers, quoted matter
    "textit", "textbf", "texttt", "emph", "enquote", "textsc", "textsuperscript",
    "acrshort", "acrlong", "acrfull", "gls", "Gls", "glspl",
    # inclusions and spacing that carry no prose
    "includegraphics", "input", "include", "url", "href", "footnote",
    "si", "num", "SI", "qty",
})

# Escaped literals: the backslash is load-bearing and DeepL drops it.
ESCAPED_LITERAL_RE = re.compile(r"\\[%&_#$~^{}]")

# A control sequence with no arguments, e.g. \newpage, \midrule, \hfill, \pm.
BARE_COMMAND_RE = re.compile(r"\\[A-Za-z@]+\*?")

# Whole lines that are structure, not prose.
SKIP_LINE_RE = re.compile(
    r"^\s*(?:%"
    r"|\\(?:chapter|section|subsection|subsubsection|paragraph|subparagraph)\*?\s*\{"
    r"|\\(?:label|input|include|newpage|cleardoublepage|clearpage|pagebreak"
    r"|graphicspath|addbibresource|appendix|listoffigures|listoftables|tableofcontents)\b"
    r"|\\(?:begin|end)\s*\{"
    r")"
)

# Environments whose bodies are never prose to improve.
SKIP_ENVIRONMENTS: frozenset[str] = frozenset({
    "table", "table*", "tabular", "tabularx", "tabular*", "longtable",
    "figure", "figure*", "subfigure", "wrapfigure", "minipage",
    "tikzpicture", "lstlisting", "verbatim", "Verbatim", "listing", "minted",
    "equation", "equation*", "align", "align*", "gather", "gather*",
    "eqnarray", "eqnarray*", "displaymath", "math",
    "thebibliography", "tcolorbox",
})

BEGIN_RE = re.compile(r"\\begin\s*\{([^}]*)\}")
END_RE = re.compile(r"\\end\s*\{([^}]*)\}")


class PlaceholderLost(RuntimeError):
    """A masked span did not survive the round trip, so the text is unsafe to use.

    Raised instead of returning LaTeX that has lost a citation or gained an
    invented cross-reference. The caller is expected to keep the original.
    """


@dataclass
class Masked:
    """Prose with every LaTeX span replaced by a placeholder."""

    text: str
    spans: list[str] = field(default_factory=list)

    @property
    def placeholders(self) -> list[str]:
        return [PLACEHOLDER_TEMPLATE.format(i) for i in range(len(self.spans))]

    @property
    def has_prose(self) -> bool:
        """Whether anything but placeholders and punctuation is left to improve."""
        stripped = PLACEHOLDER_RE.sub("", self.text)
        return bool(re.search(r"[A-Za-z]{2,}", stripped))


@dataclass
class Line:
    """One physical line of a document, flagged as improvable prose or not.

    Lines, not paragraphs, because a heading and the paragraph it introduces are
    adjacent lines of the same blank-line-separated block: `\\paragraph{...}` must
    stay verbatim while the prose under it is improved. This manuscript writes one
    paragraph per line, so a prose line is a paragraph anyway.
    """

    text: str          # without its line ending
    ending: str        # "\n", "\r\n", or "" on a final line with no newline
    is_prose: bool
    number: int        # 1-based, for error messages

    @property
    def raw(self) -> str:
        return self.text + self.ending


def _match_balanced(text: str, start: int, open_ch: str, close_ch: str) -> int:
    """Return the index just past a balanced `open_ch`…`close_ch` group at `start`.

    Returns `start` unchanged if there is no group there or it never closes, so a
    malformed document degrades to "mask less" rather than raising.
    """
    if start >= len(text) or text[start] != open_ch:
        return start
    depth = 0
    i = start
    while i < len(text):
        char = text[i]
        if char == "\\":  # an escaped brace is not a delimiter
            i += 2
            continue
        if char == open_ch:
            depth += 1
        elif char == close_ch:
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return start


def _match_inline_math(text: str, start: int) -> int:
    """Return the index just past an inline `$…$` span at `start`, else `start`."""
    if not text.startswith("$", start):
        return start
    if text.startswith("$$", start):  # display math: let the caller skip the block
        return start
    i = start + 1
    while i < len(text):
        if text[i] == "\\":
            i += 2
            continue
        if text[i] == "$":
            return i + 1
        i += 1
    return start


def mask(text: str) -> Masked:
    """Replace every LaTeX span in `text` with a `ZQX<n>ZQX` placeholder."""
    out: list[str] = []
    spans: list[str] = []
    i = 0
    length = len(text)

    def emit(span: str) -> None:
        out.append(PLACEHOLDER_TEMPLATE.format(len(spans)))
        spans.append(span)

    while i < length:
        char = text[i]

        if char == "$":
            end = _match_inline_math(text, i)
            if end > i:
                emit(text[i:end])
                i = end
                continue

        if char == "\\":
            literal = ESCAPED_LITERAL_RE.match(text, i)
            if literal:
                emit(literal.group(0))
                i = literal.end()
                continue

            command = BARE_COMMAND_RE.match(text, i)
            if command:
                name = command.group(0).lstrip("\\").rstrip("*")
                end = command.end()
                if name in MASKED_COMMANDS:
                    # Consume this command's optional and mandatory arguments.
                    while end < length:
                        if text[end] == "[":
                            nxt = _match_balanced(text, end, "[", "]")
                        elif text[end] == "{":
                            nxt = _match_balanced(text, end, "{", "}")
                        else:
                            break
                        if nxt == end:
                            break
                        end = nxt
                emit(text[i:end])
                i = end
                continue

        out.append(char)
        i += 1

    return Masked(text="".join(out), spans=spans)


def unmask(text: str, masked: Masked, *, strict: bool = True) -> str:
    """Put the LaTeX spans back, verifying none was lost, duplicated or invented.

    With `strict`, raises `PlaceholderLost` unless every placeholder appears
    exactly once and no unknown placeholder appeared. That check is the whole
    point: a dropped `\\cite{}` or an invented `\\Cref{}` must never reach the
    manuscript silently.
    """
    if strict:
        _verify(text, masked)

    def substitute(match: re.Match[str]) -> str:
        index = int(match.group(1))
        return masked.spans[index] if index < len(masked.spans) else match.group(0)

    return PLACEHOLDER_RE.sub(substitute, text)


def _verify(text: str, masked: Masked) -> None:
    found = [int(n) for n in PLACEHOLDER_RE.findall(text)]
    expected = set(range(len(masked.spans)))

    unknown = sorted(set(found) - expected)
    if unknown:
        raise PlaceholderLost(
            f"the service invented placeholder(s) {unknown} that were never sent"
        )

    missing = sorted(expected - set(found))
    if missing:
        lost = ", ".join(repr(masked.spans[i]) for i in missing)
        raise PlaceholderLost(f"{len(missing)} LaTeX span(s) did not come back: {lost}")

    duplicated = sorted({n for n in found if found.count(n) > 1})
    if duplicated:
        dupes = ", ".join(repr(masked.spans[i]) for i in duplicated)
        raise PlaceholderLost(f"{len(duplicated)} LaTeX span(s) came back more than once: {dupes}")


def reordered_spans(text: str, masked: Masked) -> list[str]:
    """Spans the service moved relative to each other, for the caller to report.

    Not an error: rephrasing legitimately moves a citation within its sentence.
    Worth surfacing, because a citation that crosses a sentence boundary now
    supports a different claim.
    """
    found = [int(n) for n in PLACEHOLDER_RE.findall(text)]
    if found == sorted(found):
        return []
    return [masked.spans[i] for i in found if i < len(masked.spans)]


DISPLAY_MATH_MARKERS = ("$$", r"\[", r"\]")

# Abbreviations whose period does not end a sentence. Without these, "e.g." and
# "et al." split mid-sentence and the two halves are rephrased as if they were
# whole sentences.
ABBREVIATIONS: frozenset[str] = frozenset({
    "e.g", "i.e", "et al", "cf", "vs", "etc", "resp", "approx", "ca", "incl",
    "Fig", "Tab", "Sec", "Eq", "Ch", "App", "No", "no", "pp", "p", "Nr",
    "Dr", "Prof", "Mr", "Ms", "Mrs", "St", "vol", "ed", "eds", "al",
})

# A sentence boundary: .!? then closing quotes/brackets, then space, then
# something that can start a sentence.
_SENTENCE_BOUNDARY_RE = re.compile(r'([.!?][)"\'\]]*)(\s+)(?=[A-Z(\\$"\'])')


def split_sentences(text: str) -> list[str]:
    """Split `text` into sentences, each keeping its trailing whitespace.

    `"".join(split_sentences(t)) == t` for any input, so a caller can replace
    individual sentences and reassemble the paragraph byte-exactly.
    """
    parts: list[str] = []
    start = 0
    for match in _SENTENCE_BOUNDARY_RE.finditer(text):
        head = text[start:match.end(1)]
        # Do not split after an abbreviation or a single-letter initial.
        tail_word = re.search(r"([A-Za-z.]+)\.$", head)
        if tail_word:
            word = tail_word.group(1).rstrip(".")
            if word in ABBREVIATIONS or len(word) == 1:
                continue
        parts.append(text[start:match.end(2)])
        start = match.end(2)
    if start < len(text):
        parts.append(text[start:])
    return parts


def is_markup_free(sentence: str) -> bool:
    """Whether `sentence` contains no LaTeX at all, so it is safe to send as-is.

    This is the gate that makes automatic application safe: a sentence with no
    markup has nothing an improvement pass could corrupt. Both Write endpoints
    were measured mangling markup-adjacent text — inserting a comma before a
    `\\cite{}`, joining a `\\textit{}` to it with "and", turning `\\item` into
    "\\item:" — so markup-bearing sentences are never sent for in-place use.
    """
    return not mask(sentence).spans


def iter_lines(document: str) -> list[Line]:
    """Split `document` into lines, flagging which carry improvable prose.

    A line is not prose when it is blank, structural (a comment, a heading, a
    label, an `\\input`, an environment delimiter), part of display math, or
    inside a skipped environment such as a table, figure or tikz picture.

    `"".join(line.raw for line in iter_lines(doc)) == doc` for any input, so a
    caller can rebuild the document by replacing only the prose lines.
    """
    lines: list[Line] = []
    depth = 0  # nesting depth inside skipped environments
    for number, raw in enumerate(document.splitlines(keepends=True), start=1):
        text = raw.rstrip("\r\n")
        ending = raw[len(text):]

        # A line inside a skipped environment is not prose, and neither is the
        # \begin line that opened it — so evaluate depth before applying \begin.
        inside = depth > 0
        for match in BEGIN_RE.finditer(text):
            if match.group(1) in SKIP_ENVIRONMENTS:
                depth += 1
        for match in END_RE.finditer(text):
            if match.group(1) in SKIP_ENVIRONMENTS:
                depth = max(0, depth - 1)

        is_prose = not (
            inside
            or not text.strip()
            or SKIP_LINE_RE.match(text)
            or any(marker in text for marker in DISPLAY_MATH_MARKERS)
        )
        lines.append(Line(text=text, ending=ending, is_prose=is_prose, number=number))
    return lines
