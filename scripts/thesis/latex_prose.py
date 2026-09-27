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

import collections
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

# Escaped literals: the backslash is load-bearing and DeepL drops it. Includes
# the spacing commands (`\,` `\;` `\ ` `\/` `\-`), which are punctuation-like
# control symbols rather than named commands.
ESCAPED_LITERAL_RE = re.compile(r"\\[%&_#$~^{},;/ \-]")

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


# ── Flattening: plain prose out, markup back in ──────────────────────────────
#
# `mask` hides every span behind a noun token. That is safe, but the service then
# treats each token as a noun and restructures around it. `flatten` goes one
# step further for the spans that *have* a natural plain-text reading: a
# `\textit{AX Visio}` is sent as "AX Visio", a `$225$` as "225", a `\cite{}` is
# removed altogether. The paragraph the service sees is ordinary prose. On the
# way back, `restore` re-wraps the plain strings, re-inserts the citations at the
# end of the sentence they belonged to, and reports every span it could not
# place, so the caller can hand that paragraph to a reviewer instead of the
# manuscript.

# Wrappers whose argument is prose-like enough to send bare.
FLATTEN_COMMANDS: frozenset[str] = frozenset({
    "textit", "textbf", "texttt", "emph", "textsc", "enquote",
})

# Citation commands: stripped from the sent text, re-attached afterwards.
CITE_COMMANDS: frozenset[str] = frozenset({
    "cite", "textcite", "parencite", "citeauthor", "citeyear", "footcite",
})

# Escaped literals that read naturally as their bare character.
FLATTEN_LITERALS: dict[str, str] = {r"\%": "%", r"\&": "&", r"\_": "_", r"\#": "#"}

# Inline-math tokens with an obvious plain rendering. Anything else in a math
# span (a variable, a superscript, a \frac) keeps the span masked as a token.
_MATH_TOKEN_RENDERING: tuple[tuple[str, str], ...] = (
    ("{,}", ","), (r"\%", "%"), (r"\,", " "), (r"\times", " × "), (r"\pm", " ± "),
    (r"\approx", " ≈ "), (r"\leq", " ≤ "), (r"\geq", " ≥ "), (r"\sim", " ~ "),
)
_MATH_UNIT_RE = re.compile(r"\\(?:mathrm|text|textrm)\{([A-Za-z%]{1,4})\}")
_NUMERIC_PLAIN_RE = re.compile(
    r"^[-+]?\d[\d.,]*(?:\s?%|\s?[×±≈≤≥~x]\s?\d[\d.,]*%?)*(?:\s?[A-Za-z]{1,4})?$"
)

_CITE_KEYS_RE = re.compile(
    r"\\(?:cite|textcite|parencite|citeauthor|citeyear|footcite)\*?(?:\[[^\]]*\])*\{([^}]*)\}"
)

# An unescaped % starts a comment that runs to the end of the line.
_TRAILING_COMMENT_RE = re.compile(r"(?<!\\)(\s*%.*)$")

_ITEM_PREFIX_RE = re.compile(r"^(\\item\s+)")


def render_numeric_math(span: str) -> str | None:
    """The plain-text form of a simple numeric `$…$` span, or None if it has none.

    `$225$` -> "225", `$28\\%$` -> "28%", `$1{,}200$` -> "1,200",
    `$193\\,\\mathrm{M}$` -> "193 M", `$512 \\times 512$` -> "512 × 512".
    `$T$`, `$10^{-4}$` and `$\\Delta_{\\text{fine}}$` -> None: mask them instead.
    """
    inner = span[1:-1] if span.startswith("$") and span.endswith("$") else span
    plain = _MATH_UNIT_RE.sub(r"\1", inner)
    for token, rendering in _MATH_TOKEN_RENDERING:
        plain = plain.replace(token, rendering)
    plain = re.sub(r"\s+", " ", plain).strip()
    if "\\" in plain or any(ch in plain for ch in "{}^_"):
        return None
    return plain if _NUMERIC_PLAIN_RE.match(plain) else None


@dataclass(frozen=True)
class FlatSpan:
    """A span sent as plain text, to be re-wrapped on the way back.

    `before` and `after` are the few characters around it in the sent text, so
    that a plain string occurring more than once in the improved text ("and",
    "2") can be put back where it belongs.
    """

    plain: str
    latex: str
    before: str = ""
    after: str = ""


@dataclass(frozen=True)
class Citation:
    """A citation removed from the sent text, with where it came from."""

    latex: str
    sentence: int         # index into split_sentences(flattened text)
    sentence_final: bool  # nothing but closing punctuation followed it

    @property
    def keys(self) -> list[str]:
        match = _CITE_KEYS_RE.match(self.latex)
        return [k.strip() for k in match.group(1).split(",")] if match else []


@dataclass
class Flattened:
    """A paragraph reduced to plain prose plus everything needed to rebuild it."""

    text: str                              # what gets sent
    item_prefix: str = ""                  # a leading "\item " peeled off, if any
    comment_suffix: str = ""               # a trailing "% …" comment, kept verbatim
    masked: Masked = field(default_factory=lambda: Masked(text=""))
    flat_spans: list[FlatSpan] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)

    @property
    def has_prose(self) -> bool:
        stripped = PLACEHOLDER_RE.sub("", self.text)
        return bool(re.search(r"[A-Za-z]{2,}", stripped))


def _glued(text: str, start: int, end: int, out: list[str]) -> bool:
    """Whether the span text[start:end] touches a word character on either side.

    `v$4.0.2$a` and `$30$th` have no plain form that can be found again, so such
    spans stay masked.
    """
    prev = out[-1][-1] if out and out[-1] else ""
    nxt = text[end] if end < len(text) else ""
    return bool(re.match(r"\w", prev or " ")) or bool(re.match(r"\w", nxt or " "))


def flatten(text: str) -> Flattened:
    """Reduce `text` to plain prose: strip citations, unwrap text commands,
    render numeric math, and mask whatever remains as a noun token."""
    prefix = ""
    match = _ITEM_PREFIX_RE.match(text)
    if match:
        prefix = match.group(1)
        text = text[match.end():]
    suffix = ""
    comment = _TRAILING_COMMENT_RE.search(text)
    if comment and not text.lstrip().startswith("%"):
        suffix = comment.group(1)
        text = text[: comment.start(1)]

    out: list[str] = []
    masked_spans: list[str] = []
    # (offset in flattened text, plain, latex); contexts are filled in at the end.
    flat_positions: list[tuple[int, str, str]] = []
    cite_positions: list[tuple[int, str]] = []
    i = 0
    length = len(text)

    def current_length() -> int:
        return sum(len(part) for part in out)

    def emit_masked(span: str) -> None:
        out.append(PLACEHOLDER_TEMPLATE.format(len(masked_spans)))
        masked_spans.append(span)

    def emit_flat(plain: str, span: str) -> None:
        flat_positions.append((current_length(), plain, span))
        out.append(plain)

    while i < length:
        char = text[i]

        if char == "$":
            end = _match_inline_math(text, i)
            if end > i:
                span = text[i:end]
                plain = render_numeric_math(span)
                if plain is not None and not _glued(text, i, end, out):
                    emit_flat(plain, span)
                else:
                    emit_masked(span)
                i = end
                continue

        if char == "\\":
            literal = ESCAPED_LITERAL_RE.match(text, i)
            if literal:
                span = literal.group(0)
                if span in FLATTEN_LITERALS:
                    emit_flat(FLATTEN_LITERALS[span], span)
                else:
                    emit_masked(span)
                i = literal.end()
                continue

            command = BARE_COMMAND_RE.match(text, i)
            if command:
                name = command.group(0).lstrip("\\").rstrip("*")
                end = command.end()
                if name in MASKED_COMMANDS or name in FLATTEN_COMMANDS or name in CITE_COMMANDS:
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
                span = text[i:end]

                if name in CITE_COMMANDS:
                    # Drop the citation together with the space that led into
                    # it: "large \cite{x}, which" -> "large, which".
                    if out and out[-1].endswith(" "):
                        out[-1] = out[-1][:-1]
                        if not out[-1]:
                            out.pop()
                    elif end < length and text[end] == " ":
                        end += 1  # "(\cite{x} and" -> "(and"
                    cite_positions.append((current_length(), span))
                    i = end
                    continue

                if name in FLATTEN_COMMANDS:
                    inner_start = span.find("{")
                    inner = span[inner_start + 1:-1] if inner_start >= 0 and span.endswith("}") else ""
                    # A wrapper whose argument itself carries markup is not
                    # plain prose; keep it a token.
                    if inner and "\\" not in inner and "$" not in inner and not _glued(text, i, end, out):
                        plain = f'"{inner}"' if name == "enquote" else inner
                        emit_flat(plain, span)
                        i = end
                        continue

                emit_masked(span)
                i = end
                continue

        out.append(char)
        i += 1

    flat_text = "".join(out)

    flat_spans = [
        FlatSpan(
            plain=plain, latex=latex,
            before=flat_text[max(0, offset - 24):offset],
            after=flat_text[offset + len(plain):offset + len(plain) + 24],
        )
        for offset, plain, latex in flat_positions
    ]

    sentences = split_sentences(flat_text)
    bounds: list[tuple[int, int]] = []
    pos = 0
    for sentence in sentences:
        bounds.append((pos, pos + len(sentence)))
        pos += len(sentence)
    citations: list[Citation] = []
    for offset, latex in cite_positions:
        index = len(bounds) - 1
        for k, (start, end) in enumerate(bounds):
            if offset < end:
                index = k
                break
        remainder = flat_text[offset:bounds[index][1]] if bounds else ""
        final = not re.search(r"[A-Za-z0-9]", remainder)
        citations.append(Citation(latex=latex, sentence=max(index, 0), sentence_final=final))

    return Flattened(
        text=flat_text,
        item_prefix=prefix,
        comment_suffix=suffix,
        masked=Masked(text=flat_text, spans=masked_spans),
        flat_spans=flat_spans,
        citations=citations,
    )


@dataclass
class Restored:
    """The rebuilt paragraph and everything that did not go back cleanly."""

    text: str
    flags: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not any(f.startswith(("format-lost", "token-lost")) for f in self.flags)


_STOPWORDS = frozenset(
    "the a an and or of to in on for with that this those these is are was were be "
    "been by as at it its from than which who whose into not no but if then so such".split()
)


def _content_words(sentence: str) -> set[str]:
    return {
        w for w in re.findall(r"[a-z][a-z\-]{2,}", sentence.lower()) if w not in _STOPWORDS
    }


def align_sentences(original: list[str], improved: list[str]) -> list[tuple[int, float]]:
    """For each original sentence, the improved sentence it best matches.

    Monotonic and greedy: sentence i maps to some j at or after the mapping of
    sentence i-1, so a rephrasing that merges or splits sentences still maps
    every original sentence somewhere sensible. Returns (index, similarity).
    """
    result: list[tuple[int, float]] = []
    improved_words = [_content_words(s) for s in improved]
    floor = 0
    for sentence in original:
        words = _content_words(sentence)
        best_j, best_score = floor, 0.0
        for j in range(floor, len(improved)):
            union = words | improved_words[j]
            score = len(words & improved_words[j]) / len(union) if union else 0.0
            if score > best_score:
                best_j, best_score = j, score
        result.append((min(best_j, max(len(improved) - 1, 0)), best_score))
        floor = best_j
    return result


def _boundary_pattern(plain: str, flags: int = 0) -> re.Pattern[str]:
    """A regex for `plain` that will not match inside a longer word or number."""
    head = r"(?<![\w.,])" if re.match(r"\w", plain) else ""
    tail = r"(?![\w]|[.,]\d)" if re.search(r"\w$", plain) else ""
    return re.compile(head + re.escape(plain) + tail, flags)


def _last_word(text: str) -> str:
    words = re.findall(r"[\w%]+", text)
    return words[-1].lower() if words else ""


def _first_word(text: str) -> str:
    words = re.findall(r"[\w%]+", text)
    return words[0].lower() if words else ""


def _pick(candidates: list[re.Match[str]], text: str, span: FlatSpan) -> tuple[re.Match[str], bool]:
    """The candidate whose surroundings best match the sent text, and whether
    that choice was unique."""
    if len(candidates) == 1:
        return candidates[0], True
    scored = []
    for m in candidates:
        score = 0
        if _last_word(text[:m.start()]) and _last_word(text[:m.start()]) == _last_word(span.before):
            score += 1
        if _first_word(text[m.end():]) and _first_word(text[m.end():]) == _first_word(span.after):
            score += 1
        scored.append((score, m))
    best = max(s for s, _ in scored)
    winners = [m for s, m in scored if s == best]
    return winners[0], len(winners) == 1


_TERMINAL_RE = re.compile(r"[.!?]+[\"')\]]*\s*$")


def _attach(sentence: str, citation_latex: str) -> str:
    """Put a citation at the end of a sentence, before its closing punctuation."""
    body = sentence.rstrip()
    trailing_ws = sentence[len(body):]
    match = _TERMINAL_RE.search(body)
    if match:
        return body[: match.start()] + " " + citation_latex + body[match.start():] + trailing_ws
    return body + " " + citation_latex + trailing_ws


def restore(improved: str, flat: Flattened, *, low_confidence: float = 0.5) -> Restored:
    """Rebuild a LaTeX paragraph from the service's improved plain prose."""
    flags: list[str] = []

    # 1. Noun tokens back. A token the service dropped makes the text unusable.
    try:
        text = unmask(improved, flat.masked)
    except PlaceholderLost as exc:
        return Restored(text="", flags=[f"token-lost: {exc}"])
    for span in reordered_spans(improved, flat.masked):
        flags.append(f"token-reordered: {span}")

    # 2. Flattened spans re-wrapped, in order, choosing among repeats by context.
    replacements: list[tuple[int, int, str]] = []
    cursor = 0
    for span in flat.flat_spans:
        pattern = _boundary_pattern(span.plain)
        taken = lambda m: any(s < m.end() and m.start() < e for s, e, _ in replacements)  # noqa: E731
        candidates = [m for m in pattern.finditer(text, cursor) if not taken(m)]
        if not candidates:
            candidates = [m for m in pattern.finditer(text) if not taken(m)]
            if candidates:
                flags.append(f"wrap-reordered: {span.plain}")
        if not candidates:
            ci = [m for m in _boundary_pattern(span.plain, re.IGNORECASE).finditer(text) if not taken(m)]
            if ci:
                candidates = ci
                flags.append(f"wrap-case: {span.plain} -> {ci[0].group(0)}")
        if not candidates:
            flags.append(f"format-lost: {span.latex}")
            continue
        match, unique = _pick(candidates, text, span)
        if not unique:
            flags.append(f"wrap-ambiguous: {span.plain}")
        replacements.append((match.start(), match.end(), span.latex))
        cursor = max(cursor, match.end())
    for start, end, latex in sorted(replacements, reverse=True):
        text = text[:start] + latex + text[end:]

    # 3. Citations back, each at the end of the sentence its claim moved to.
    if flat.citations:
        original_sentences = split_sentences(flat.text)
        improved_sentences = split_sentences(text) or [text]
        mapping = align_sentences(original_sentences, improved_sentences)
        per_sentence: dict[int, list[Citation]] = collections.defaultdict(list)
        cited_targets: dict[int, set[int]] = collections.defaultdict(set)
        for citation in flat.citations:
            j, score = mapping[citation.sentence] if citation.sentence < len(mapping) else (len(improved_sentences) - 1, 0.0)
            per_sentence[j].append(citation)
            cited_targets[j].add(citation.sentence)
            if score < low_confidence:
                flags.append(f"cite-lowconf: {citation.latex} ({score:.2f})")
            if not citation.sentence_final:
                flags.append(f"cite-moved: {citation.latex}")
        if any(len(origins) > 1 for origins in cited_targets.values()):
            flags.append("cite-merged-sentences")
        for j, citations in per_sentence.items():
            # Plain \cite{} commands landing on one sentence merge into one call.
            simple = [c for c in citations if c.latex.startswith("\\cite{")]
            others = [c for c in citations if not c.latex.startswith("\\cite{")]
            if len(simple) > 1:
                merged_latex = "\\cite{" + ",".join(k for c in simple for k in c.keys) + "}"
                flags.append(f"cite-merged: {merged_latex}")
                calls = [merged_latex] + [c.latex for c in others]
            else:
                calls = [c.latex for c in simple] + [c.latex for c in others]
            for latex in calls:
                improved_sentences[j] = _attach(improved_sentences[j], latex)
        text = "".join(improved_sentences)

    return Restored(text=flat.item_prefix + text + flat.comment_suffix, flags=flags)


def span_inventory(text: str) -> tuple[list[str], list[str]]:
    """(non-citation spans, citation keys), both sorted, for equality checks.

    Two texts with the same inventory carry the same LaTeX: nothing lost,
    nothing invented. Citation keys are compared as a multiset so that
    `\\cite{a} … \\cite{b}` and `\\cite{a,b}` count as the same.
    """
    spans: list[str] = []
    keys: list[str] = []
    for span in mask(text).spans:
        match = _CITE_KEYS_RE.match(span)
        if match:
            keys.extend(k.strip() for k in match.group(1).split(","))
        else:
            spans.append(span)
    return sorted(spans), sorted(keys)
