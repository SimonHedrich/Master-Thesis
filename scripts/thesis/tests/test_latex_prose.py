"""Tests for the LaTeX prose extractor (`scripts/thesis/latex_prose.py`).

Run command
-----------
    uv run pytest scripts/thesis/tests/test_latex_prose.py

All offline: this module never talks to a network. The invariants here are what
make `deepl_write --latex` safe to point at the manuscript, so they are asserted
against real manuscript constructs rather than toy strings.
"""

from __future__ import annotations

import collections

import pytest

from scripts.thesis import latex_prose as lp

# A real paragraph shape from chapters/1-Introduction.tex.
REAL_PARAGRAPH = (
    r"The \textit{AX Visio} binocular \cite{AXVisio} runs a species-recognition model "
    r"on board. Of the $225$ classes, $50$ retain fewer than $150$ usable real "
    r"photographs each (\Cref{sec:results_granularity_gaps}). Roughly $80\%$ of the "
    r"time went into dataset construction."
)

REAL_DOCUMENT = "\n".join([
    r"\newpage",
    r"\chapter{Introduction}\label{chapter:introduction}",
    "",
    r"This work investigates detectors. The budget is tight.",
    "",
    r"%%%%%%%%%%%%%%%%%%%%",
    r"\section{Motivation}\label{sec:motivation}",
    "",
    r"\paragraph{The Production Baseline}",
    r"The device ships a pipeline \cite{AXVisio}. It works well.",
    "",
    r"\begin{table}[ht]",
    r"    \centering",
    r"    Raw tabular content that is not prose.",
    r"\end{table}",
    "",
    r"\begin{enumerate}",
    r"    \item Does distillation help the student model?",
    r"\end{enumerate}",
    "",
])


# ── Masking ──────────────────────────────────────────────────────────────────

def test_mask_replaces_every_latex_span_and_unmask_is_the_inverse():
    masked = lp.mask(REAL_PARAGRAPH)
    # 1 textit, 1 cite, 1 Cref, 4 inline-math ($225$, $50$, $150$, $80\%$ — the
    # escaped percent sits inside the math span, not beside it).
    assert len(masked.spans) == 7
    assert "\\cite" not in masked.text
    assert "\\Cref" not in masked.text
    assert "$" not in masked.text
    assert lp.unmask(masked.text, masked) == REAL_PARAGRAPH


@pytest.mark.parametrize("source, expected_span", [
    (r"see \cite{key2024} here", r"\cite{key2024}"),
    (r"see \Cref{sec:foo} here", r"\Cref{sec:foo}"),
    (r"see \Cref{sec:a,sec:b} here", r"\Cref{sec:a,sec:b}"),
    (r"the \textit{AX Visio} device", r"\textit{AX Visio}"),
    (r"the \texttt{yolo26n} model", r"\texttt{yolo26n}"),
    (r"about $225$ classes", "$225$"),
    (r"about $193\,\mathrm{M}$ params", r"$193\,\mathrm{M}$"),
    (r"exactly 80\% of it", r"\%"),
    (r"a \hyperref[chapter:results]{\textit{Results}} chapter",
     r"\hyperref[chapter:results]{\textit{Results}}"),
    (r"an \includegraphics[width=0.5\textwidth]{fig.png} float",
     r"\includegraphics[width=0.5\textwidth]{fig.png}"),
])
def test_individual_constructs_are_masked_whole(source, expected_span):
    masked = lp.mask(source)
    assert expected_span in masked.spans
    assert lp.unmask(masked.text, masked) == source


def test_nested_braces_are_matched_not_truncated():
    source = r"see \textit{\texttt{nested} inner} done"
    masked = lp.mask(source)
    assert masked.spans == [r"\textit{\texttt{nested} inner}"]
    assert lp.unmask(masked.text, masked) == source


def test_bare_commands_are_masked_but_take_no_arguments():
    # \item must not swallow the sentence that follows it.
    masked = lp.mask(r"\item Does distillation help?")
    assert masked.spans == [r"\item"]
    assert "Does distillation help?" in masked.text


def test_unbalanced_braces_degrade_to_masking_less_rather_than_raising():
    source = r"broken \textit{unclosed and more text"
    masked = lp.mask(source)
    assert lp.unmask(masked.text, masked) == source


def test_placeholders_are_the_shape_that_survives_a_round_trip():
    # Tokens like <0> and [[0]] are deleted by the service; a bare-word token is
    # not. Pinned because changing the shape silently breaks the whole approach.
    masked = lp.mask(r"a \cite{k} b")
    assert masked.spans and "ZQX0ZQX" in masked.text


def test_has_prose_distinguishes_real_text_from_markup_only_lines():
    assert lp.mask("A real sentence here.").has_prose is True
    assert lp.mask(r"\label{sec:x}").has_prose is False
    assert lp.mask(r"$225$ \cite{k}").has_prose is False


# ── unmask fails closed ──────────────────────────────────────────────────────

def test_unmask_raises_when_a_span_did_not_come_back():
    masked = lp.mask(r"text \cite{lost} more")
    with pytest.raises(lp.PlaceholderLost, match=r"did not come back"):
        lp.unmask("text more", masked)


def test_unmask_raises_when_a_span_came_back_twice():
    masked = lp.mask(r"text \cite{dup} more")
    with pytest.raises(lp.PlaceholderLost, match="more than once"):
        lp.unmask("text ZQX0ZQX and ZQX0ZQX more", masked)


def test_unmask_raises_when_the_service_invented_a_placeholder():
    masked = lp.mask(r"text \cite{k} more")
    with pytest.raises(lp.PlaceholderLost, match="invented"):
        lp.unmask("text ZQX0ZQX and ZQX7ZQX more", masked)


def test_unmask_can_be_asked_not_to_verify():
    masked = lp.mask(r"text \cite{k} more")
    assert lp.unmask("text more", masked, strict=False) == "text more"


def test_reordered_spans_reports_movement_without_failing():
    masked = lp.mask(r"\cite{a} then \cite{b}")
    assert lp.reordered_spans("ZQX0ZQX then ZQX1ZQX", masked) == []
    assert lp.reordered_spans("ZQX1ZQX then ZQX0ZQX", masked) == [r"\cite{b}", r"\cite{a}"]


# ── Sentence splitting ───────────────────────────────────────────────────────

def test_split_sentences_joins_back_exactly():
    assert "".join(lp.split_sentences(REAL_PARAGRAPH)) == REAL_PARAGRAPH


def test_split_sentences_finds_the_boundaries():
    parts = [p.strip() for p in lp.split_sentences("First one. Second one! Third one?")]
    assert parts == ["First one.", "Second one!", "Third one?"]


@pytest.mark.parametrize("text", [
    "Models such as e.g. YOLO are small. A second sentence follows.",
    "Work by Horn et al. 2018 is used here. A second sentence follows.",
    "See Fig. 3 for details. A second sentence follows.",
    "The value is approx. 5 percent. A second sentence follows.",
    "Work by A. Author is cited. A second sentence follows.",
])
def test_abbreviations_and_initials_do_not_split_a_sentence(text):
    parts = lp.split_sentences(text)
    assert len(parts) == 2, parts
    assert parts[1].strip() == "A second sentence follows."
    assert "".join(parts) == text


def test_split_sentences_preserves_leading_indentation():
    text = "    \\item First question? Second question?"
    parts = lp.split_sentences(text)
    assert "".join(parts) == text
    assert parts[0].startswith("    ")


def test_is_markup_free_gates_what_may_be_sent():
    assert lp.is_markup_free("The classification must happen in view.") is True
    assert lp.is_markup_free(r"The device \cite{AXVisio} runs a model.") is False
    assert lp.is_markup_free(r"Of the $225$ classes, few remain.") is False


# ── Line classification ──────────────────────────────────────────────────────

def test_iter_lines_reassembles_the_document_byte_for_byte():
    assert "".join(line.raw for line in lp.iter_lines(REAL_DOCUMENT)) == REAL_DOCUMENT


def test_iter_lines_handles_crlf_and_a_missing_final_newline():
    for document in ("a\r\nb\r\n", "no trailing newline", "trailing\n"):
        assert "".join(line.raw for line in lp.iter_lines(document)) == document


def prose_text(document: str) -> list[str]:
    return [line.text.strip() for line in lp.iter_lines(document) if line.is_prose]


def test_headings_labels_and_comments_are_not_prose():
    prose = prose_text(REAL_DOCUMENT)
    assert not any(line.startswith("\\chapter") for line in prose)
    assert not any(line.startswith("\\section") for line in prose)
    assert not any(line.startswith("\\paragraph") for line in prose)
    assert not any(line.startswith("%") for line in prose)
    assert not any(line.startswith("\\newpage") for line in prose)


def test_table_bodies_are_not_prose_but_enumerate_items_are():
    prose = prose_text(REAL_DOCUMENT)
    assert "Raw tabular content that is not prose." not in prose
    assert r"\item Does distillation help the student model?" in prose


def test_real_paragraphs_are_prose():
    prose = prose_text(REAL_DOCUMENT)
    assert "This work investigates detectors. The budget is tight." in prose
    assert r"The device ships a pipeline \cite{AXVisio}. It works well." in prose


def test_nested_skipped_environments_close_correctly():
    document = "\n".join([
        r"\begin{figure}",
        r"\begin{subfigure}{0.5\textwidth}",
        r"inner caption text",
        r"\end{subfigure}",
        r"\end{figure}",
        r"Prose after the float.",
        "",
    ])
    assert prose_text(document) == ["Prose after the float."]


def test_display_math_lines_are_not_prose():
    document = "Before.\n$$ a = b $$\nAfter.\n"
    assert prose_text(document) == ["Before.", "After."]


# ── The invariant that matters, on the real manuscript ───────────────────────

def manuscript_files() -> list[str]:
    import glob
    from pathlib import Path

    root = Path(lp.__file__).resolve().parents[2] / "thesis" / "manuscript"
    return sorted(glob.glob(str(root / "**" / "*.tex"), recursive=True))


@pytest.mark.parametrize("path", manuscript_files())
def test_every_manuscript_file_round_trips_exactly(path):
    """Line splitting, sentence splitting and masking must all be lossless.

    Parametrized over the real manuscript because that is the input the tool is
    pointed at, and a single lossy construct anywhere would corrupt the file.
    """
    source = open(path, encoding="utf-8").read()
    lines = lp.iter_lines(source)
    assert "".join(line.raw for line in lines) == source

    for line in lines:
        if not line.is_prose:
            continue
        sentences = lp.split_sentences(line.text)
        assert "".join(sentences) == line.text
        for sentence in sentences:
            masked = lp.mask(sentence)
            assert lp.unmask(masked.text, masked) == sentence


def test_manuscript_span_inventory_is_stable_under_mask_unmask():
    """The multiset of LaTeX spans must be identical after a mask/unmask cycle."""
    def inventory(text: str) -> collections.Counter:
        return collections.Counter(
            span for line in lp.iter_lines(text) for span in lp.mask(line.text).spans
        )

    for path in manuscript_files():
        source = open(path, encoding="utf-8").read()
        rebuilt = "".join(
            (lp.unmask(lp.mask(line.text).text, lp.mask(line.text)) + line.ending)
            for line in lp.iter_lines(source)
        )
        assert inventory(rebuilt) == inventory(source), path
