"""Tests for the DeepL Write client (`scripts/thesis/deepl_write.py`).

Run command
-----------
    uv run pytest scripts/thesis/tests/                      # offline only
    uv run pytest scripts/thesis/tests/ -m live              # hits the real API
    uv run pytest scripts/thesis/tests/ -m "live or not live" -v   # everything

Two tiers. The offline tests use a stub transport, so they assert request
shaping, batching, retry and error mapping without a network or a single
character of quota — they are the ones that run by default. The `live` tests
(deselected unless `-m live` is given) verify the same client against the real
endpoint and are the actual proof the credentials and request shapes work; they
spend a few hundred characters of the quota.
"""

from __future__ import annotations

import collections
import json
import os

import pytest

from scripts.thesis import deepl_write as dw
from scripts.thesis import latex_prose

TYPO_TEXT = "I could relly use sum help with edits on thiss text !"


# ── Stub transport ────────────────────────────────────────────────────────────

class StubTransport:
    """Canned HTTP responses, recording every request the client makes.

    `responses` is a list of `(status, body, headers)`; the last one repeats once
    exhausted, so a retry test can queue two failures and one success.
    """

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def __call__(self, url, payload, headers):
        self.calls.append({
            "url": url,
            "headers": headers,
            "body": json.loads(payload.decode("utf-8")) if payload else None,
            "raw_len": len(payload) if payload else 0,
        })
        status, body, response_headers = (
            self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        )
        raw = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        return status, raw, response_headers


def ok(*texts, source="en", target="en-US"):
    """A 200 `improvements` payload, one entry per text."""
    return (200, {"improvements": [
        {"text": t, "detected_source_language": source, "target_language": target}
        for t in texts
    ]}, {"X-Trace-ID": "trace-abc"})


def make_client(responses, **kwargs):
    """A client wired to a stub transport, with sleeping and logging silenced."""
    transport = StubTransport(responses)
    client = dw.DeepLWriteClient(
        "test-key:fx", transport=transport,
        sleep=lambda _: None, log=lambda _: None, **kwargs,
    )
    return client, transport


# ── Host selection from the key ───────────────────────────────────────────────

@pytest.mark.parametrize("key, expected", [
    ("abc-123:fx", dw.FREE_HOST),
    ("abc-123:fx  ", dw.FREE_HOST),   # trailing whitespace from .env must not flip the host
    ("abc-123", dw.PRO_HOST),
    ("fx:abc-123", dw.PRO_HOST),      # ':fx' counts only as a suffix
])
def test_api_base_follows_key_suffix(key, expected):
    assert dw.api_base_for_key(key) == expected


def test_client_derives_host_from_key_and_allows_override():
    client, _ = make_client([ok("x")])
    assert client.api_base == dw.FREE_HOST
    override, _ = make_client([ok("x")], api_base="https://example.test/")
    assert override.api_base == "https://example.test"  # trailing slash stripped


def test_resolve_api_key_prefers_explicit_over_environment(monkeypatch):
    monkeypatch.setenv("DEEPL_API_KEY", "from-env")
    assert dw.resolve_api_key("explicit") == "explicit"
    assert dw.resolve_api_key(None) == "from-env"


def test_resolve_api_key_errors_when_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("DEEPL_API_KEY", raising=False)
    # Point the .env lookup at an empty directory so a real key cannot leak in.
    monkeypatch.setattr(dw, "REPO_ROOT", tmp_path)
    with pytest.raises(dw.DeepLError, match="DEEPL_API_KEY is not set"):
        dw.resolve_api_key(None)


# ── Request shaping ──────────────────────────────────────────────────────────

def test_rephrase_sends_expected_url_headers_and_body():
    client, transport = make_client([ok("improved")])
    client.rephrase(TYPO_TEXT, "en-US", writing_style="academic")

    call = transport.calls[0]
    assert call["url"] == f"{dw.FREE_HOST}/v2/write/rephrase"
    assert call["headers"]["Authorization"] == "DeepL-Auth-Key test-key:fx"
    assert call["headers"]["Content-Type"] == "application/json"
    assert call["body"] == {
        "text": [TYPO_TEXT], "target_lang": "en-US", "writing_style": "academic",
    }


def test_correct_hits_the_correct_endpoint_and_sends_no_style():
    client, transport = make_client([ok("fixed")])
    client.correct([TYPO_TEXT], "en-US")

    call = transport.calls[0]
    assert call["url"].endswith("/v2/write/correct")
    assert call["body"] == {"text": [TYPO_TEXT], "target_lang": "en-US"}


def test_tone_is_sent_and_string_input_is_wrapped():
    client, transport = make_client([ok("improved")])
    results = client.rephrase("a single string", "de", tone="confident")
    assert transport.calls[0]["body"]["tone"] == "confident"
    assert transport.calls[0]["body"]["text"] == ["a single string"]
    assert len(results) == 1


def test_usage_and_languages_are_get_requests():
    client, transport = make_client([(200, {"character_count": 42, "character_limit": 500}, {})])
    usage = client.usage()
    assert (usage.character_count, usage.character_limit) == (42, 500)
    assert usage.fraction_used == pytest.approx(0.084)
    assert transport.calls[0]["body"] is None  # no payload -> GET

    client, transport = make_client([(200, [{"lang": "de", "features": {"tone": {}}}], {})])
    assert client.supports_style_or_tone("de") is True
    assert transport.calls[0]["url"].endswith("/v3/languages?resource=write")


def test_supports_style_or_tone_is_false_for_languages_without_the_feature():
    client, _ = make_client([(200, [
        {"lang": "en-US", "features": {"tone": {}, "writing_style": {}}},
        {"lang": "ja", "features": {"auto_detection": {}}},
    ], {})])
    assert client.supports_style_or_tone("ja") is False


# ── Client-side validation: reject before spending quota ─────────────────────

def test_style_and_tone_are_mutually_exclusive():
    client, transport = make_client([ok("x")])
    with pytest.raises(dw.DeepLBadRequest, match="mutually exclusive"):
        client.rephrase(TYPO_TEXT, "en-US", writing_style="business", tone="friendly")
    assert transport.calls == []  # never left the process


@pytest.mark.parametrize("kwargs, match", [
    ({"writing_style": "bogus"}, "Unknown writing_style"),
    ({"tone": "bogus"}, "Unknown tone"),
])
def test_unknown_style_and_tone_values_are_rejected(kwargs, match):
    client, transport = make_client([ok("x")])
    with pytest.raises(dw.DeepLBadRequest, match=match):
        client.rephrase(TYPO_TEXT, "en-US", **kwargs)
    assert transport.calls == []


def test_unsupported_target_lang_is_rejected():
    client, transport = make_client([ok("x")])
    with pytest.raises(dw.DeepLBadRequest, match="not a Write target language"):
        client.rephrase(TYPO_TEXT, "en-AU")
    assert transport.calls == []


def test_blank_text_is_rejected_and_empty_list_is_a_no_op():
    client, transport = make_client([ok("x")])
    with pytest.raises(dw.DeepLBadRequest, match="Empty or whitespace-only"):
        client.correct(["fine", "   "], "en-US")
    assert client.correct([], "en-US") == []
    assert transport.calls == []


def test_oversized_single_request_is_refused_before_sending():
    client, transport = make_client([ok("x")])
    # One text over the 10 KiB cap: iter_batches yields it alone, then _request
    # refuses it rather than letting DeepL answer 413.
    with pytest.raises(dw.DeepLBadRequest, match="over DeepL's"):
        client.correct(["word " * 4000], "en-US")
    assert transport.calls == []


# ── Batching ─────────────────────────────────────────────────────────────────

def test_iter_batches_packs_to_the_budget_and_preserves_order():
    texts = [f"text-{i}" for i in range(20)]
    batches = list(dw.iter_batches(texts, budget=60))
    assert len(batches) > 1
    assert [t for batch in batches for t in batch] == texts
    for batch in batches:
        assert len(json.dumps(batch).encode("utf-8")) <= 80  # budget plus envelope slack


def test_iter_batches_yields_an_oversized_text_alone():
    batches = list(dw.iter_batches(["tiny", "x" * 500, "tiny"], budget=50))
    assert batches == [["tiny"], ["x" * 500], ["tiny"]]


def test_large_input_is_split_across_requests_and_results_stay_aligned():
    # ~15 KiB total, comfortably over the 9 KiB batch budget, so this must split.
    texts = [f"sentence number {i} in a draft that needs improving" for i in range(300)]
    client, transport = make_client([ok("improved")])

    # Replace the transport with one that mirrors the request back, which is what
    # actually proves each Improvement is paired to its own input.
    def echo(url, payload, headers):
        body = json.loads(payload.decode("utf-8"))
        transport.calls.append({"url": url, "headers": headers, "body": body,
                                "raw_len": len(payload)})
        return 200, json.dumps({"improvements": [
            {"text": t.upper(), "detected_source_language": "en",
             "target_language": "en-US"} for t in body["text"]
        ]}).encode("utf-8"), {}

    client._transport = echo
    results = client.correct(texts, "en-US")

    assert len(transport.calls) > 1, "input should have needed more than one request"
    assert all(call["raw_len"] <= dw.MAX_REQUEST_BYTES for call in transport.calls)
    assert [r.original for r in results] == texts
    assert [r.text for r in results] == [t.upper() for t in texts]


def test_mismatched_improvement_count_raises_instead_of_mispairing():
    client, _ = make_client([ok("only-one")])
    with pytest.raises(dw.DeepLError, match="pairing to the inputs"):
        client.correct(["first", "second"], "en-US")


# ── Error mapping ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("status, expected", [
    (400, dw.DeepLBadRequest),
    (401, dw.DeepLAuthError),
    (403, dw.DeepLAuthError),
    (456, dw.DeepLQuotaExceeded),
    (413, dw.DeepLError),
])
def test_statuses_map_to_exception_types(status, expected):
    client, _ = make_client([(status, {"message": "nope"}, {"X-Trace-ID": "t-1"})])
    with pytest.raises(expected) as excinfo:
        client.correct(TYPO_TEXT, "en-US")
    assert excinfo.value.status == status
    assert excinfo.value.trace_id == "t-1"
    assert "nope" in str(excinfo.value)
    assert "t-1" in str(excinfo.value)  # trace ID surfaces in the message


def test_quota_and_bad_request_are_never_retried():
    for status in (400, 456):
        client, transport = make_client([(status, {"message": "stop"}, {})])
        with pytest.raises(dw.DeepLError):
            client.correct(TYPO_TEXT, "en-US")
        assert len(transport.calls) == 1, f"HTTP {status} must not be retried"


def test_non_json_error_body_still_produces_a_message():
    client, _ = make_client([(502, b"<html>bad gateway</html>", {})], max_retries=0)
    with pytest.raises(dw.DeepLError, match="bad gateway"):
        client.correct(TYPO_TEXT, "en-US")


def test_trace_id_header_lookup_is_case_insensitive():
    client, _ = make_client([(400, {"message": "bad"}, {"x-trace-id": "lower-case"})])
    with pytest.raises(dw.DeepLBadRequest) as excinfo:
        client.correct(TYPO_TEXT, "en-US")
    assert excinfo.value.trace_id == "lower-case"


# ── Retry and backoff ────────────────────────────────────────────────────────

def test_429_then_success_retries_with_exponential_backoff():
    delays: list[float] = []
    transport = StubTransport([
        (429, {"message": "slow down"}, {}),
        (429, {"message": "slow down"}, {}),
        ok("improved")[0:3],
    ])
    client = dw.DeepLWriteClient(
        "k:fx", transport=transport, sleep=delays.append, log=lambda _: None,
        backoff_base=1.0,
    )
    results = client.correct(TYPO_TEXT, "en-US")
    assert results[0].text == "improved"
    assert len(transport.calls) == 3
    assert delays == [1.0, 2.0]  # base * 2**attempt


def test_retries_are_capped_and_the_last_error_is_raised():
    client, transport = make_client([(503, {"message": "unavailable"}, {})], max_retries=2)
    with pytest.raises(dw.DeepLError, match="unavailable"):
        client.correct(TYPO_TEXT, "en-US")
    assert len(transport.calls) == 3  # initial attempt plus 2 retries


def test_retry_after_header_overrides_the_backoff_schedule():
    delays: list[float] = []
    transport = StubTransport([(429, {"message": "wait"}, {"Retry-After": "7"}), ok("done")[0:3]])
    client = dw.DeepLWriteClient(
        "k:fx", transport=transport, sleep=delays.append, log=lambda _: None,
    )
    client.correct(TYPO_TEXT, "en-US")
    assert delays == [7.0]


def test_unparseable_retry_after_falls_back_to_backoff():
    delays: list[float] = []
    transport = StubTransport([
        (429, {"message": "wait"}, {"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}),
        ok("done")[0:3],
    ])
    client = dw.DeepLWriteClient(
        "k:fx", transport=transport, sleep=delays.append, log=lambda _: None, backoff_base=0.5,
    )
    client.correct(TYPO_TEXT, "en-US")
    assert delays == [0.5]


def test_trace_id_is_logged_on_success():
    logged: list[str] = []
    transport = StubTransport([ok("improved")])
    client = dw.DeepLWriteClient(
        "k:fx", transport=transport, sleep=lambda _: None, log=logged.append,
    )
    client.correct(TYPO_TEXT, "en-US")
    assert any("trace-abc" in line for line in logged)


# ── Improvement helpers ──────────────────────────────────────────────────────

def test_improvement_tracks_whether_the_text_changed():
    unchanged = dw.Improvement("same text", "same text", "en", "en-US")
    changed = dw.Improvement("teh cat", "the cat", "en", "en-US")
    assert unchanged.changed is False
    assert changed.changed is True
    diff = changed.unified_diff()
    assert "-teh" in diff and "+the" in diff


# ── CLI ──────────────────────────────────────────────────────────────────────

def test_cli_usage_reports_the_quota(monkeypatch, capsys):
    monkeypatch.setenv("DEEPL_API_KEY", "k:fx")
    monkeypatch.setattr(
        dw.DeepLWriteClient, "usage", lambda self: dw.Usage(1500, 1_000_000)
    )
    assert dw.main(["usage"]) == 0
    assert "1,500 / 1,000,000 characters" in capsys.readouterr().out


def test_cli_correct_prints_improved_text_from_a_file(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("DEEPL_API_KEY", "k:fx")
    draft = tmp_path / "draft.txt"
    draft.write_text("first paragraph with a typo\n\nsecond paragraph\n", encoding="utf-8")

    seen: dict = {}

    def fake_correct(self, texts, target_lang):
        seen["texts"], seen["target"] = list(texts), target_lang
        return [dw.Improvement(t, t.upper(), "en", target_lang) for t in texts]

    monkeypatch.setattr(dw.DeepLWriteClient, "correct", fake_correct)
    assert dw.main(["correct", "-t", "en-US", "-f", str(draft), "--paragraphs"]) == 0

    assert seen["texts"] == ["first paragraph with a typo", "second paragraph"]
    assert seen["target"] == "en-US"
    out = capsys.readouterr().out
    assert "FIRST PARAGRAPH WITH A TYPO" in out and "SECOND PARAGRAPH" in out


def test_cli_rejects_style_and_tone_together(monkeypatch, capsys):
    monkeypatch.setenv("DEEPL_API_KEY", "k:fx")
    assert dw.main(
        ["rephrase", "-t", "en-US", "--style", "business", "--tone", "friendly", "x"]
    ) == 1
    assert "mutually exclusive" in capsys.readouterr().err


def test_cli_reports_api_errors_as_exit_code_1(monkeypatch, capsys):
    monkeypatch.setenv("DEEPL_API_KEY", "k:fx")

    def boom(self):
        raise dw.DeepLQuotaExceeded("quota used up", status=456, trace_id="t-9")

    monkeypatch.setattr(dw.DeepLWriteClient, "usage", boom)
    assert dw.main(["usage"]) == 1
    err = capsys.readouterr().err
    assert "quota used up" in err and "t-9" in err


def test_cli_rejects_an_unsupported_target_lang_at_parse_time(monkeypatch):
    monkeypatch.setenv("DEEPL_API_KEY", "k:fx")
    with pytest.raises(SystemExit) as excinfo:
        dw.main(["correct", "-t", "en-AU", "text"])
    assert excinfo.value.code == 2  # argparse choices, before any network call


# ── LaTeX mode: the safety guarantee ─────────────────────────────────────────

LATEX_DOC = "\n".join([
    r"\chapter{Introduction}\label{chapter:intro}",
    "",
    r"The device ships a pipeline \cite{AXVisio}. It works quite well indeed.",
    "",
    r"\begin{table}[ht]",
    r"    tabular content here",
    r"\end{table}",
    "",
    r"A wholly markup free sentence here.",
    "",
])


def upper_echo_client():
    """A client whose transport uppercases every text it is given."""
    sent: list[list[str]] = []

    def echo(url, payload, headers):
        body = json.loads(payload.decode("utf-8"))
        sent.append(body["text"])
        return 200, json.dumps({"improvements": [
            {"text": t.upper(), "detected_source_language": "en",
             "target_language": "en-US"} for t in body["text"]
        ]}).encode("utf-8"), {}

    client = dw.DeepLWriteClient(
        "k:fx", transport=echo, sleep=lambda _: None, log=lambda _: None,
    )
    return client, sent


def test_latex_mode_sends_only_markup_free_sentences():
    client, sent = upper_echo_client()
    client.improve_latex(LATEX_DOC, "en-US")

    flat = [text for batch in sent for text in batch]
    assert "It works quite well indeed." in flat
    assert "A wholly markup free sentence here." in flat
    # The guarantee: nothing containing LaTeX is ever transmitted.
    assert all("\\" not in text and "$" not in text for text in flat), flat
    assert not any("AXVisio" in text for text in flat)
    assert not any("tabular content" in text for text in flat)
    assert not any("chapter" in text for text in flat)


def test_latex_mode_leaves_markup_bearing_sentences_byte_identical():
    client, _ = upper_echo_client()
    improved, report = client.improve_latex(LATEX_DOC, "en-US")

    assert r"The device ships a pipeline \cite{AXVisio}." in improved
    assert report.held_back == 1
    # And the sentence beside it, in the same paragraph, was improved.
    assert "IT WORKS QUITE WELL INDEED." in improved


def test_latex_mode_preserves_non_prose_lines_and_line_count():
    client, _ = upper_echo_client()
    improved, _ = client.improve_latex(LATEX_DOC, "en-US")

    assert r"\chapter{Introduction}\label{chapter:intro}" in improved
    assert r"\begin{table}[ht]" in improved
    assert "    tabular content here" in improved
    assert len(improved.splitlines()) == len(LATEX_DOC.splitlines())


def test_latex_mode_span_inventory_is_unchanged():
    client, _ = upper_echo_client()
    improved, _ = client.improve_latex(LATEX_DOC, "en-US")

    def inventory(text):
        return collections.Counter(
            span for line in latex_prose.iter_lines(text)
            for span in latex_prose.mask(line.text).spans
        )

    assert inventory(improved) == inventory(LATEX_DOC)


def test_latex_mode_returns_the_document_unchanged_when_nothing_improves():
    # A transport that echoes input verbatim must produce a byte-identical file.
    def identity(url, payload, headers):
        body = json.loads(payload.decode("utf-8"))
        return 200, json.dumps({"improvements": [
            {"text": t, "detected_source_language": "en", "target_language": "en-US"}
            for t in body["text"]
        ]}).encode("utf-8"), {}

    client = dw.DeepLWriteClient(
        "k:fx", transport=identity, sleep=lambda _: None, log=lambda _: None,
    )
    improved, report = client.improve_latex(LATEX_DOC, "en-US")
    assert improved == LATEX_DOC
    assert report.improved == 0
    assert report.unchanged == 2


def test_latex_mode_spends_nothing_on_a_document_with_no_sendable_prose():
    client, sent = upper_echo_client()
    document = "\\section{Only}\\label{sec:only}\n\n\\begin{table}\nrows\n\\end{table}\n"
    improved, report = client.improve_latex(document, "en-US")
    assert improved == document
    assert sent == []
    assert report.improved == 0


def test_latex_report_summary_mentions_what_was_held_back():
    report = dw.LatexReport(improved=3, unchanged=1, skipped=8, held_back=5)
    summary = report.summary()
    assert "3 sentence(s) improved" in summary
    assert "5 held back (contain LaTeX)" in summary


def test_cli_latex_writes_in_place_only_when_asked(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("DEEPL_API_KEY", "k:fx")
    path = tmp_path / "chapter.tex"
    path.write_text(LATEX_DOC, encoding="utf-8")

    def fake(self, document, target_lang, **kwargs):
        return document.replace("quite well indeed", "very well"), dw.LatexReport(improved=1)

    monkeypatch.setattr(dw.DeepLWriteClient, "improve_latex", fake)

    # Without --in-place the file must not be touched.
    assert dw.main(["rephrase", "-t", "en-US", "--latex", str(path)]) == 0
    assert path.read_text(encoding="utf-8") == LATEX_DOC
    assert "very well" in capsys.readouterr().out

    assert dw.main(["rephrase", "-t", "en-US", "--latex", "--in-place", str(path)]) == 0
    assert "very well" in path.read_text(encoding="utf-8")


def test_cli_latex_requires_a_real_file(monkeypatch, capsys):
    monkeypatch.setenv("DEEPL_API_KEY", "k:fx")
    assert dw.main(["rephrase", "-t", "en-US", "--latex", "/nonexistent/x.tex"]) == 1
    assert "is not a file" in capsys.readouterr().err


def test_cli_in_place_without_latex_is_refused(monkeypatch, capsys):
    monkeypatch.setenv("DEEPL_API_KEY", "k:fx")
    assert dw.main(["rephrase", "-t", "en-US", "--in-place", "some text"]) == 1
    assert "--in-place applies only with --latex" in capsys.readouterr().err


# ── Live tests: real endpoint, real key, real quota ──────────────────────────

live = pytest.mark.skipif(
    not (os.environ.get("DEEPL_API_KEY") or (dw.REPO_ROOT / ".env").exists()),
    reason="needs DEEPL_API_KEY in the environment or .env",
)


@pytest.fixture(scope="module")
def live_client():
    return dw.DeepLWriteClient(log=lambda _: None)


@pytest.mark.live
@live
def test_live_usage_is_within_the_quota(live_client):
    usage = live_client.usage()
    assert usage.character_limit > 0
    assert 0 <= usage.character_count <= usage.character_limit


@pytest.mark.live
@live
def test_live_languages_include_the_documented_write_targets(live_client):
    langs = {entry["lang"] for entry in live_client.languages()}
    assert {"de", "en-US", "en-GB", "es", "fr", "it"} <= langs
    # The style/tone feature set is language-specific, and the client's own
    # helper must agree with what the API reports.
    assert live_client.supports_style_or_tone("en-US") is True
    assert live_client.supports_style_or_tone("ja") is False


@pytest.mark.live
@live
def test_live_correct_fixes_typos_without_rewording(live_client):
    (result,) = live_client.correct(TYPO_TEXT, "en-US")
    assert result.detected_source_language.lower().startswith("en")
    assert result.target_language == "en-US"
    for typo in ("relly", "sum help", "thiss"):
        assert typo not in result.text
    assert "really" in result.text
    # Corrections-only keeps the author's phrasing: "edits on this text", not a rewrite.
    assert "edits" in result.text


@pytest.mark.live
@live
def test_live_rephrase_rewords_more_than_correct_does(live_client):
    (rephrased,) = live_client.rephrase(TYPO_TEXT, "en-US")
    (corrected,) = live_client.correct(TYPO_TEXT, "en-US")
    assert "relly" not in rephrased.text and "thiss" not in rephrased.text
    # The documented contrast between the endpoints: rephrase may restructure the
    # sentence, correct may not. Assert they are not the same string rather than
    # pinning exact wording, which the service is free to change.
    assert rephrased.text != corrected.text


@pytest.mark.live
@live
def test_live_writing_style_changes_the_output(live_client):
    plain, business = (
        live_client.rephrase(TYPO_TEXT, "en-US", writing_style=style)[0].text
        for style in (None, "business")
    )
    assert plain != business


@pytest.mark.live
@live
def test_live_batch_returns_one_improvement_per_input_in_order(live_client):
    # Each input carries an unambiguous misspelling, which corrections-only always
    # fixes. Grammar is deliberately not relied on here: the service leaves some
    # non-standard grammar ("we was going home") untouched, so asserting on it
    # would test DeepL's editorial judgment rather than this client's batching.
    texts = ["the fox jumpd over teh dog", "she dont knows the anser", "a buatiful sunsett"]
    results = live_client.correct(texts, "en-US")

    assert [r.original for r in results] == texts, "results must stay aligned to inputs"
    assert all(r.changed for r in results)
    assert "jumped" in results[0].text
    assert "answer" in results[1].text
    assert "beautiful" in results[2].text


@pytest.mark.live
@live
def test_live_bare_style_on_an_unsupporting_language_is_a_400(live_client):
    # Japanese supports no writing styles: the bare value is an error, and the
    # `prefer_` variant silently falls back. This is the contract the client's
    # enum lists document, verified against the service.
    with pytest.raises(dw.DeepLBadRequest) as excinfo:
        live_client.rephrase("これはテストです", "ja", writing_style="business")
    assert excinfo.value.status == 400

    (fallback,) = live_client.rephrase("これはテストです", "ja", writing_style="prefer_business")
    assert fallback.target_language == "ja"


@pytest.mark.live
@live
def test_live_latex_mode_leaves_every_span_byte_identical(live_client):
    """The guarantee, against the real service: markup in, same markup out.

    This is the test that matters for pointing the tool at the manuscript. The
    service demonstrably mangles markup-adjacent text when it sees any, so the
    assertion is that it never sees any.
    """
    improved, report = live_client.improve_latex(LATEX_DOC, "en-US")

    def inventory(text):
        return collections.Counter(
            span for line in latex_prose.iter_lines(text)
            for span in latex_prose.mask(line.text).spans
        )

    assert inventory(improved) == inventory(LATEX_DOC)
    assert r"\cite{AXVisio}" in improved
    assert r"\chapter{Introduction}\label{chapter:intro}" in improved
    assert "    tabular content here" in improved
    assert len(improved.splitlines()) == len(LATEX_DOC.splitlines())
    assert report.held_back == 1


@pytest.mark.live
@live
def test_live_latex_mode_on_a_real_manuscript_chapter(live_client):
    """Same guarantee on real manuscript input, which is denser than any fixture."""
    path = dw.REPO_ROOT / "thesis" / "manuscript" / "chapters" / "1-Introduction.tex"
    if not path.is_file():
        pytest.skip("manuscript not present in this checkout")
    source = path.read_text(encoding="utf-8")

    improved, report = live_client.improve_latex(source, "en-US")

    original_lines = latex_prose.iter_lines(source)
    improved_lines = latex_prose.iter_lines(improved)
    assert len(original_lines) == len(improved_lines)

    def inventory(lines):
        return collections.Counter(
            span for line in lines for span in latex_prose.mask(line.text).spans
        )

    assert inventory(improved_lines) == inventory(original_lines)
    # Every non-prose line must be returned untouched, character for character.
    for before, after in zip(original_lines, improved_lines):
        if not before.is_prose:
            assert before.raw == after.raw
    assert report.improved > 0, "expected at least some markup-free prose to improve"
    assert path.read_text(encoding="utf-8") == source, "must not write to the manuscript"


@pytest.mark.live
@live
def test_live_wrong_host_for_the_key_is_an_auth_error():
    key = dw.resolve_api_key()
    wrong_host = dw.PRO_HOST if key.endswith(dw.FREE_KEY_SUFFIX) else dw.FREE_HOST
    client = dw.DeepLWriteClient(key, api_base=wrong_host, log=lambda _: None)
    with pytest.raises(dw.DeepLAuthError) as excinfo:
        client.usage()
    assert excinfo.value.status == 403


@pytest.mark.live
@live
def test_live_invalid_key_is_rejected():
    client = dw.DeepLWriteClient("definitely-not-a-valid-key:fx", log=lambda _: None)
    with pytest.raises(dw.DeepLAuthError) as excinfo:
        client.usage()
    assert excinfo.value.status in (401, 403)
