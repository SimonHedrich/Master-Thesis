"""Tests for the paragraph-level rephrase pipeline (`scripts/thesis/rephrase_manuscript.py`)
and the flatten/restore layer in `latex_prose.py` it relies on.

Run command
-----------
    uv run pytest scripts/thesis/tests/test_rephrase_manuscript.py
    uv run pytest scripts/thesis/tests/test_rephrase_manuscript.py -m live   # spends quota

Offline by default: a stub transport plays DeepL. The invariants that matter
are the ones that keep the manuscript safe — nothing is applied that lost a
LaTeX span, every response is on disk before it is used, and a resumed run
sends nothing twice.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts.thesis import deepl_write as dw
from scripts.thesis import latex_prose as lp
from scripts.thesis import rephrase_manuscript as rm

MANUSCRIPT_FILES = sorted((rm.MANUSCRIPT_DIR).rglob("*.tex"))

PARAGRAPH = (
    r"The \textit{AX Visio} binocular \cite{AXVisio} runs a model on board. "
    r"Of the $225$ classes, $50$ retain fewer than $150$ photographs (\Cref{sec:x}). "
    r"The most accurate models are large \cite{zou}, which makes it hard. "
    r"Roughly $80\%$ of the time went into it \cite{a} \cite{b}."
)


# ── flatten / restore ────────────────────────────────────────────────────────

def test_flatten_sends_plain_prose_and_remembers_everything():
    flat = lp.flatten(PARAGRAPH)
    assert flat.text == (
        "The AX Visio binocular runs a model on board. Of the 225 classes, 50 retain "
        "fewer than 150 photographs (ZQX0ZQX). The most accurate models are large, which "
        "makes it hard. Roughly 80% of the time went into it."
    )
    assert [s.plain for s in flat.flat_spans] == ["AX Visio", "225", "50", "150", "80%"]
    assert flat.masked.spans == [r"\Cref{sec:x}"]
    assert [(c.latex, c.sentence, c.sentence_final) for c in flat.citations] == [
        (r"\cite{AXVisio}", 0, False), (r"\cite{zou}", 2, False),
        (r"\cite{a}", 3, True), (r"\cite{b}", 3, True),
    ]


def test_restore_of_the_unchanged_text_keeps_every_span_and_moves_citations_to_sentence_ends():
    flat = lp.flatten(PARAGRAPH)
    restored = lp.restore(flat.text, flat)
    assert restored.clean
    assert lp.span_inventory(restored.text) == lp.span_inventory(PARAGRAPH)
    assert restored.text == (
        r"The \textit{AX Visio} binocular runs a model on board \cite{AXVisio}. "
        r"Of the $225$ classes, $50$ retain fewer than $150$ photographs (\Cref{sec:x}). "
        r"The most accurate models are large, which makes it hard \cite{zou}. "
        r"Roughly $80\%$ of the time went into it \cite{a,b}."
    )
    assert "cite-moved: \\cite{AXVisio}" in restored.flags
    assert "cite-merged: \\cite{a,b}" in restored.flags


def test_restore_follows_the_claim_when_sentences_are_reworded():
    flat = lp.flatten(PARAGRAPH)
    improved = (
        "The AX Visio binocular runs a model on board. Of the 225 classes, 50 keep fewer "
        "than 150 photographs (ZQX0ZQX). Because the most accurate models are large, an "
        "embedded target is hard to reach. About 80% of the time went into it."
    )
    restored = lp.restore(improved, flat)
    assert restored.clean
    assert lp.span_inventory(restored.text) == lp.span_inventory(PARAGRAPH)
    assert r"hard to reach \cite{zou}." in restored.text
    assert r"About $80\%$ of the time went into it \cite{a,b}." in restored.text


def test_restore_flags_a_span_whose_wording_the_service_changed():
    flat = lp.flatten(r"Roughly $80\%$ of the time went into it.")
    restored = lp.restore("Roughly 80 percent of the time went into it.", flat)
    assert not restored.clean
    assert restored.flags == [r"format-lost: $80\%$"]
    # The plain text stays so the reviewer sees what happened; apply will reject it.
    assert lp.span_inventory(restored.text) != lp.span_inventory(r"Roughly $80\%$ of the time went into it.")


def test_restore_picks_the_right_repeat_by_context():
    text = r"A \emph{and} B is stronger than A or B, \emph{and} it is cheaper."
    flat = lp.flatten(text)
    assert flat.text == "A and B is stronger than A or B, and it is cheaper."
    restored = lp.restore("A and B is stronger than A or B, and it is cheaper.", flat)
    assert restored.text == text
    assert restored.flags == []


def test_restore_fails_closed_when_a_token_is_dropped():
    flat = lp.flatten(r"The setup is described in \Cref{sec:setup} and works.")
    restored = lp.restore("The setup is described and works.", flat)
    assert restored.text == ""
    assert restored.flags and restored.flags[0].startswith("token-lost")


def test_glued_math_and_trailing_comments_stay_verbatim():
    text = r"Version v$4.0.2$a ranked $30$th of all.  % reports/x.csv"
    flat = lp.flatten(text)
    assert flat.comment_suffix == "  % reports/x.csv"
    assert "ZQX0ZQX" in flat.text and "ZQX1ZQX" in flat.text
    assert "%" not in flat.text
    assert lp.restore(flat.text, flat).text == text


def test_item_prefix_is_peeled_and_put_back():
    text = r"\item Does distillation help the student model \cite{k}?"
    flat = lp.flatten(text)
    assert flat.item_prefix == "\\item "
    assert flat.text == "Does distillation help the student model?"
    restored = lp.restore("Does distillation help the student?", flat)
    assert restored.text == r"\item Does distillation help the student \cite{k}?"


@pytest.mark.parametrize("span, plain", [
    (r"$225$", "225"), (r"$28\%$", "28%"), (r"$1{,}200$", "1,200"), (r"$0.599$", "0.599"),
    (r"$193\,\mathrm{M}$", "193 M"), (r"$512 \times 512$", "512 × 512"),
    (r"$T$", None), (r"$10^{-4}$", None), (r"$\Delta_{\text{fine}}$", None),
])
def test_numeric_math_rendering(span, plain):
    assert lp.render_numeric_math(span) == plain


@pytest.mark.parametrize("path", MANUSCRIPT_FILES, ids=lambda p: p.name)
def test_identity_round_trip_on_every_manuscript_file(path):
    """With the service returning its input, every span comes back and the
    only change is citations moving to the end of their sentence."""
    for line in lp.iter_lines(path.read_text(encoding="utf-8")):
        if not line.is_prose:
            continue
        text = line.text.strip()
        flat = lp.flatten(text)
        if not flat.has_prose:
            continue
        assert "\\" not in flat.text and "$" not in flat.text, (path.name, line.number)
        restored = lp.restore(flat.text, flat)
        assert restored.clean, (path.name, line.number, restored.flags)
        assert lp.span_inventory(restored.text) == lp.span_inventory(text), (path.name, line.number)
        if all(c.sentence_final for c in flat.citations):
            assert restored.text == text, (path.name, line.number)


def test_span_inventory_treats_merged_citations_as_equal():
    assert lp.span_inventory(r"A \cite{a}. B \cite{b}.") == lp.span_inventory(r"A. B \cite{b,a}.")
    assert lp.span_inventory(r"A \cite{a}.") != lp.span_inventory(r"A.")


# ── automatic checks ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("candidate, expected", [
    ("The filter has four stages -- each runs alone.", "dash"),
    ("The filter has four stages; each runs alone.", "semicolon"),
    ("The colour of the behaviour was analysed.", "british"),
    ("The set holds 225 classes.", "bare-number"),
    ("This thesis argues the point.", "this-thesis"),
    ("A rather than B, and C rather than D.", "rather-than"),
    (r"\cite{k} showed that it works.", "cite-subject"),
])
def test_style_flags_detect_each_house_rule(candidate, expected):
    original = r"The set holds $225$ classes and it is fine \cite{k}."
    assert any(f.startswith(expected) for f in rm.style_flags(original, candidate)), rm.style_flags(original, candidate)


def test_style_flags_ignore_problems_the_original_already_had():
    original = "Ranges like 12--15 and the premise are fine; so is this."
    assert rm.style_flags(original, original) == []
    assert "british" not in " ".join(rm.style_flags("x", "The premise is precise and otherwise wise."))


# ── prepare / apply end to end, offline ───────────────────────────────────────

DOC = "\n".join([
    r"\chapter{Introduction}\label{chapter:intro}",
    "",
    r"The device ships a pipeline \cite{AXVisio}. It works quite well indeed.",
    "",
    r"\section{Objective}\label{sec:objective}",
    r"\begin{enumerate}",
    r"    \item Does knowledge distillation from a teacher into a student help at all?",
    r"\end{enumerate}",
    "",
    r"\begin{table}[ht]",
    r"    tabular content here",
    r"\end{table}",
    "",
    r"A wholly markup free sentence here. Another one follows it.  % note",
    "",
])
ECHO_DOC = "\n".join([
    r"\chapter{Conclusion}",
    "",
    r"    \item Does knowledge distillation from a teacher into a student help at all?",
    "",
])


class Service:
    """A stub DeepL: rewrites by a callable, counts requests, can run out of quota."""

    def __init__(self, rewrite, quota_after: int | None = None, usage=(40_000, 1_000_000)):
        self.rewrite = rewrite
        self.calls: list[str] = []
        self.quota_after = quota_after
        self.usage = usage

    def __call__(self, url, payload, headers):
        if url.endswith("/v2/usage"):
            body = {"character_count": self.usage[0], "character_limit": self.usage[1]}
            return 200, json.dumps(body).encode(), {}
        texts = json.loads(payload.decode())["text"]
        assert len(texts) == 1, "one paragraph per request"
        if self.quota_after is not None and len(self.calls) >= self.quota_after:
            return 456, json.dumps({"message": "Quota exceeded"}).encode(), {}
        self.calls.append(texts[0])
        return 200, json.dumps({"improvements": [
            {"text": self.rewrite(texts[0]), "detected_source_language": "en", "target_language": "en-US"}
        ]}).encode(), {"X-Trace-ID": f"trace-{len(self.calls)}"}


def make_client(service):
    return dw.DeepLWriteClient("k:fx", transport=service, sleep=lambda _: None, log=lambda _: None)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A manuscript-like tree with two files sharing one research question."""
    manuscript = tmp_path / "manuscript"
    (manuscript / "chapters").mkdir(parents=True)
    intro = manuscript / "chapters" / "1-Introduction.tex"
    intro.write_text(DOC, encoding="utf-8")
    (manuscript / "chapters" / "5-Conclusion.tex").write_text(ECHO_DOC, encoding="utf-8")
    monkeypatch.setattr(rm, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(rm, "MANUSCRIPT_DIR", manuscript)
    return tmp_path, intro, tmp_path / "out"


def fill_entry(chunk: Path, pid: str, final: str, verdict: str) -> None:
    """Write into one entry's empty final/verdict blocks the way a reviewer would."""
    text = chunk.read_text(encoding="utf-8")
    start = text.index(f"#### {pid}")
    head, tail = text[:start], text[start:]
    tail = tail.replace("--- final\n", f"--- final\n{final}\n", 1)
    tail = tail.replace("--- verdict\n", f"--- verdict\n{verdict}\n", 1)
    chunk.write_text(head + tail, encoding="utf-8")


def run_prepare(workspace, service, **kwargs):
    root, intro, out = workspace
    errors = []

    class Err:
        def write(self, s):
            errors.append(s)

    code = rm.prepare([intro], out_dir=out, client=make_client(service),
                      echoes=rm.manuscript_echo_lines(root / "manuscript"), stderr=Err(), **kwargs)
    return code, "".join(errors)


def test_prepare_sends_one_request_per_paragraph_and_holds_back_echoes(workspace):
    service = Service(lambda t: t.replace("quite well indeed", "well"))
    code, err = run_prepare(workspace, service)
    assert code == 0
    assert service.calls == [
        "The device ships a pipeline. It works quite well indeed.",
        "A wholly markup free sentence here. Another one follows it.",
    ]
    _, _, out = workspace
    manifest = rm.load_manifest(out / "1-Introduction")
    statuses = {p.line: p.status for p in manifest.paragraphs}
    assert statuses == {3: "fetched", 7: "held-back", 14: "unchanged"}
    assert manifest.paragraphs[0].restored == r"The device ships a pipeline \cite{AXVisio}. It works well."
    assert "verbatim-echo" in manifest.paragraphs[1].flags
    assert "1 to review" in err and "1 held back" in err


def test_every_response_is_logged_before_it_is_used_and_resume_sends_nothing(workspace):
    service = Service(lambda t: t.upper())
    run_prepare(workspace, service)
    _, _, out = workspace
    log_lines = (out / rm.LOG_NAME).read_text().splitlines()
    assert len(log_lines) == 2
    entry = json.loads(log_lines[0])
    assert entry["received"] == entry["sent"].upper()
    assert entry["trace_id"] == "trace-1" and entry["chars"] == len(entry["sent"])

    # Delete the bundle; a resumed run rebuilds it from the log without a request.
    for path in (out / "1-Introduction").iterdir():
        path.unlink()
    again = Service(lambda t: t.upper())
    code, _ = run_prepare(workspace, again, resume=True)
    assert code == 0 and again.calls == []
    assert (out / "1-Introduction" / rm.MANIFEST_NAME).is_file()
    assert "cached" in rm.load_manifest(out / "1-Introduction").paragraphs[0].flags


def test_dry_run_makes_no_request_and_writes_nothing(workspace):
    service = Service(lambda t: t)
    code, err = run_prepare(workspace, service, dry_run=True)
    _, _, out = workspace
    assert code == 0 and service.calls == [] and not out.exists()
    assert "2 to send" in err and "would send" in err


def test_preflight_refuses_a_run_the_quota_cannot_cover(workspace):
    service = Service(lambda t: t, usage=(999_990, 1_000_000))
    code, err = run_prepare(workspace, service)
    assert code == rm.EXIT_PREFLIGHT and service.calls == []
    assert "only 10 remain" in err

    code, _ = run_prepare(workspace, service, force_quota=True)
    assert code == 0 and len(service.calls) == 2


def test_quota_exhaustion_mid_run_keeps_the_log_and_exits_2(workspace):
    service = Service(lambda t: t.upper(), quota_after=1)
    code, err = run_prepare(workspace, service)
    _, _, out = workspace
    assert code == rm.EXIT_QUOTA
    assert "1 still unfetched" in err
    assert len((out / rm.LOG_NAME).read_text().splitlines()) == 1
    manifest = rm.load_manifest(out / "1-Introduction")
    assert [p.status for p in manifest.paragraphs] == ["fetched", "held-back", "pending"]


def test_second_key_is_used_on_resume_when_the_first_is_spent(workspace, monkeypatch):
    monkeypatch.setenv("DEEPL_API_KEY", "first:fx")
    monkeypatch.setenv(rm.SECOND_KEY_ENV, "second:fx")
    keys_seen: list[str] = []
    calls: list[str] = []

    def transport(url, payload, headers):
        keys_seen.append(headers["Authorization"].split()[-1])
        if url.endswith("/v2/usage"):
            spent = 1_000_000 if "first" in keys_seen[-1] else 0
            return 200, json.dumps({"character_count": spent, "character_limit": 1_000_000}).encode(), {}
        text = json.loads(payload.decode())["text"][0]
        calls.append(text)
        return 200, json.dumps({"improvements": [{"text": text, "detected_source_language": "en", "target_language": "en-US"}]}).encode(), {}

    monkeypatch.setattr(dw.DeepLWriteClient, "_urllib_transport", lambda self, *a: transport(*a))
    root, intro, out = workspace
    code = rm.prepare([intro], out_dir=out, resume=True, echoes=rm.manuscript_echo_lines(root / "manuscript"), stderr=open(os.devnull, "w"))
    assert code == 0 and len(calls) == 2
    assert "second:fx" in keys_seen and keys_seen[-1] == "second:fx"


def test_review_file_round_trip_and_apply(workspace):
    service = Service(lambda t: t.replace("quite well indeed", "well").replace("wholly markup free", "plain"))
    run_prepare(workspace, service)
    root, intro, out = workspace
    bundle = out / "1-Introduction"
    chunk = bundle / "chunk-01.review.txt"
    text = chunk.read_text(encoding="utf-8")
    assert text.startswith("# Review instructions")
    entries = rm.parse_review_file(chunk)
    assert set(entries) == {"P001", "P003"}
    assert entries["P001"]["original"] == r"The device ships a pipeline \cite{AXVisio}. It works quite well indeed."
    assert entries["P001"]["rephrased"] == r"The device ships a pipeline \cite{AXVisio}. It works well."
    assert entries["P001"]["final"] == "" and entries["P003"]["final"] == ""

    # A reviewer merges: DeepL's first sentence, the original second one.
    fill_entry(chunk, "P001", "The device ships a pipeline \\cite{AXVisio}. It works quite\nwell.", "merged: kept s1, trimmed s2")
    fill_entry(chunk, "P003", "A plain sentence here. Another one follows it.  % note", "rephrased")

    errors = []

    class Err:
        def write(self, s):
            errors.append(s)

    code = rm.apply(bundle, in_place=True, stderr=Err())
    assert code == 0
    err = "".join(errors)
    assert "2 applied" in err and "joined with spaces" in err
    new_doc = intro.read_text(encoding="utf-8")
    assert r"The device ships a pipeline \cite{AXVisio}. It works quite well." in new_doc
    assert "A plain sentence here. Another one follows it.  % note" in new_doc
    assert "tabular content here" in new_doc and new_doc.count("\n") == DOC.count("\n")
    assert rm.load_manifest(bundle).applied_at

    # A second apply refuses: the source moved past the manifest's hash.
    # (The manifest hash was updated on write, so re-running is a no-op instead.)
    code = rm.apply(bundle, in_place=True, stderr=Err())
    assert code == 0 and "nothing to write" in "".join(errors)


@pytest.mark.parametrize("final, reason", [
    (r"The device ships a pipeline. It works well.", "inventory changed"),
    (r"The device ships a pipeline \cite{AXVisio} (\Cref{sec:x}). It works well.", "inventory changed"),
    (r"The device ships a pipeline \cite{AXVisio}; it works well.", "semicolon"),
    (r"The device ships a pipeline \cite{AXVisio} -- it works well.", "dash"),
    (r"\cite{AXVisio} showed the device ships a pipeline. It works well.", "subject"),
    (r"The device ships a pipeline \cite{AXVisio. It works well.", "braces"),
])
def test_apply_rejects_a_final_that_breaks_a_hard_rule(final, reason):
    original = r"The device ships a pipeline \cite{AXVisio}. It works quite well indeed."
    rejections, _ = rm.validate_final(original, final)
    assert rejections and any(reason in r for r in rejections), rejections


def test_apply_accepts_a_clean_merged_final_with_a_style_warning():
    original = r"The device ships a pipeline \cite{AXVisio}. It works quite well indeed."
    final = r"The device ships a pipeline rather than a model \cite{AXVisio}. It works well."
    rejections, warnings = rm.validate_final(original, final)
    assert rejections == [] and warnings == ["rather-than"]


def test_apply_refuses_when_the_source_changed_since_prepare(workspace):
    run_prepare(workspace, Service(lambda t: t.upper()))
    root, intro, out = workspace
    intro.write_text(DOC.replace("quite well", "very well"), encoding="utf-8")
    errors = []

    class Err:
        def write(self, s):
            errors.append(s)

    assert rm.apply(out / "1-Introduction", in_place=True, stderr=Err()) == 1
    assert "changed since" in "".join(errors)


def test_unreviewed_and_original_verdicts_keep_the_line(workspace):
    run_prepare(workspace, Service(lambda t: t.upper()))
    root, intro, out = workspace
    bundle = out / "1-Introduction"
    chunk = bundle / "chunk-01.review.txt"
    fill_entry(chunk, "P001", "", "original: shouting")
    errors = []

    class Err:
        def write(self, s):
            errors.append(s)

    rm.apply(bundle, in_place=True, stderr=Err())
    assert "1 kept by verdict, 1 unreviewed" in "".join(errors)
    assert intro.read_text() == DOC


# ── live ──────────────────────────────────────────────────────────────────────

live = pytest.mark.skipif(
    not (dw.REPO_ROOT / ".env").is_file() and not os.environ.get("DEEPL_API_KEY"),
    reason="no DeepL credentials",
)


@pytest.mark.live
@live
def test_live_prepare_restores_every_span_on_real_introduction_paragraphs(tmp_path):
    intro = rm.MANUSCRIPT_DIR / "chapters" / "1-Introduction.tex"
    out = tmp_path / "out"
    code = rm.prepare([intro], out_dir=out, limit=3, stderr=open(os.devnull, "w"))
    assert code == 0
    manifest = rm.load_manifest(out / "1-Introduction")
    fetched = [p for p in manifest.paragraphs if p.status in ("fetched", "unchanged")]
    assert len(fetched) == 3
    for p in fetched:
        assert p.restored
        if not any(f.startswith(("format-lost", "token-lost")) for f in p.flags):
            assert lp.span_inventory(p.restored) == lp.span_inventory(p.original), (p.id, p.flags)
    assert len((out / rm.LOG_NAME).read_text().splitlines()) == 3
