"""DeepL Write API client — rephrase and corrections-only passes over prose.

Write improves text *within* one language: the source text and ``target_lang``
must be the same language. It is not a translator. Two endpoints:

``rephrase``
    Broad improvement: fixes spelling and grammar, and rewords sentences so the
    prose reads better. This is what the DeepL Write web app does, and with no
    ``--style`` or ``--tone`` it behaves exactly as that default does. Leave it
    that way unless there is a reason not to — the named styles are heavier
    handed, and ``academic`` in particular over-complicates plain sentences.
``correct``
    Minimal spelling and grammar pass, matching "Corrections Only" in the DeepL
    Translator UI. Changes nothing else, so it is the conservative option when
    the wording is already deliberate.

Run from the repository root:

    uv run python -m scripts.thesis.deepl_write rephrase -t en-US "text to improve"
    uv run python -m scripts.thesis.deepl_write rephrase -t en-US --diff -f draft.txt
    uv run python -m scripts.thesis.deepl_write rephrase -t en-US --latex \
        thesis/manuscript/chapters/1-Introduction.tex
    uv run python -m scripts.thesis.deepl_write correct  -t en-US "som text to fix"
    uv run python -m scripts.thesis.deepl_write usage
    uv run python -m scripts.thesis.deepl_write languages

``--latex`` is required for any ``.tex`` input. Sent raw LaTeX, Write strips
``\\textit{}``, deletes ``\\cite{}``, and rewrites ``\\Cref{sec:foo}`` into an
invented section number. Masking the markup is not enough either — the service
treats a placeholder as a noun and restructures the sentence around it. So
``--latex`` improves only sentences that contain no markup at all, and leaves
every sentence carrying a citation, reference or math span untransmitted and
byte-identical. See scripts/thesis/README.md for the measurements behind that.

Reads ``DEEPL_API_KEY`` from ``.env`` (gitignored, alongside ``OVERLEAF_TOKEN``).
Get a key at https://www.deepl.com/your-account/keys. A key ending in ``:fx`` is
a Free-tier key and selects the ``api-free.deepl.com`` host automatically; any
other key selects ``api.deepl.com``. Sending a key to the wrong host is a 403,
so never override ``--api-base`` without a reason.

Stdlib HTTP only, deliberately: the official ``deepl`` package would have to go
into the shared lockfile and be synced into a venv that is routinely mid-training.

Write characters count against the same quota as translation characters
(``usage`` reports it).

The same quota check without going through this script, when you want to confirm
the key and the host directly::

    export $(grep -E '^DEEPL_API_KEY=' .env | xargs)
    curl --request GET \
      --url https://api-free.deepl.com/v2/usage \
      --header "Authorization: DeepL-Auth-Key $DEEPL_API_KEY"

    {"character_count":40641,"character_limit":1000000}

Note ``api-free.deepl.com`` — that host is correct only for a ``:fx`` key; a Pro
key needs ``api.deepl.com``, and the wrong pairing answers 403 rather than any
usage figure. Keep the key in the ``$DEEPL_API_KEY`` variable as shown instead of
pasting it into the command: a literal key in shell history or in a file here is
one ``git push`` away from being public.

See scripts/thesis/README.md.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from dotenv import load_dotenv

from scripts.thesis import latex_prose

REPO_ROOT = Path(__file__).resolve().parents[2]

FREE_HOST = "https://api-free.deepl.com"
PRO_HOST = "https://api.deepl.com"
FREE_KEY_SUFFIX = ":fx"

# DeepL caps a Write request body at 10 KiB. `iter_batches` packs texts up to a
# slightly lower ceiling so the JSON envelope and escaping cannot push a batch
# that measured as fitting over the real limit.
MAX_REQUEST_BYTES = 10 * 1024
BATCH_BUDGET_BYTES = 9 * 1024

# From the OpenAPI spec's TargetLanguageWrite enum. Validated client-side so a
# typo costs nothing instead of a round trip.
WRITE_TARGET_LANGS: tuple[str, ...] = (
    "de", "en", "en-GB", "en-US", "es", "fr", "it", "ja",
    "ko", "pt", "pt-BR", "pt-PT", "zh", "zh-Hans",
)

# WritingStyle / WritingTone enums. A `prefer_`-prefixed value falls back to
# `default` when the target language supports no styles/tones; the bare value
# is a 400 in that case. Mutually exclusive: one request takes style or tone.
WRITING_STYLES: tuple[str, ...] = (
    "academic", "business", "casual", "default", "simple",
    "prefer_academic", "prefer_business", "prefer_casual", "prefer_simple",
)
WRITING_TONES: tuple[str, ...] = (
    "confident", "default", "diplomatic", "enthusiastic", "friendly",
    "prefer_confident", "prefer_diplomatic", "prefer_enthusiastic", "prefer_friendly",
)

# Retry 429 and 5xx with exponential backoff. 456 (quota exhausted) and 400
# (the request itself is wrong) are never retried — neither improves by waiting.
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
QUOTA_STATUS = 456
DEFAULT_MAX_RETRIES = 4
DEFAULT_BACKOFF_BASE = 1.0


class DeepLError(RuntimeError):
    """A DeepL API call failed. Carries the trace ID for DeepL's own logs."""

    def __init__(self, message: str, *, status: int | None = None,
                 code: str | None = None, trace_id: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.trace_id = trace_id

    def __str__(self) -> str:
        parts = [super().__str__()]
        if self.status is not None:
            parts.append(f"HTTP {self.status}")
        if self.code:
            parts.append(f"code={self.code}")
        if self.trace_id:
            parts.append(f"X-Trace-ID={self.trace_id}")
        return " | ".join(parts)


class DeepLAuthError(DeepLError):
    """401/403 — key missing, wrong, or sent to the wrong host."""


class DeepLBadRequest(DeepLError):
    """400 — the request is invalid. Not retried."""


class DeepLQuotaExceeded(DeepLError):
    """456 — the account's character quota is used up. Not retried."""


@dataclass(frozen=True)
class Improvement:
    """One improved text, paired with the input it came from."""

    original: str
    text: str
    detected_source_language: str
    target_language: str

    @property
    def changed(self) -> bool:
        return self.original != self.text

    def unified_diff(self, context: int = 1) -> str:
        """Word-level before/after, for eyeballing what an endpoint actually did."""
        return "\n".join(difflib.unified_diff(
            self.original.split(), self.text.split(),
            fromfile="original", tofile="improved", lineterm="", n=context,
        ))


@dataclass
class LatexReport:
    """What `improve_latex` changed, and what it deliberately left alone."""

    improved: int = 0        # sentences rewritten
    unchanged: int = 0       # sentences sent back identical
    skipped: int = 0         # non-prose lines never considered
    held_back: int = 0       # sentences containing LaTeX, never sent
    # (line number, before, after) — for review.
    changes: list[tuple[int, str, str]] = field(default_factory=list)

    def summary(self) -> str:
        parts = [
            f"{self.improved} sentence(s) improved",
            f"{self.unchanged} already fine",
            f"{self.held_back} held back (contain LaTeX)",
            f"{self.skipped} non-prose line(s) skipped",
        ]
        return ", ".join(parts)


@dataclass(frozen=True)
class Usage:
    """Characters consumed against the current billing period's limit."""

    character_count: int
    character_limit: int

    @property
    def fraction_used(self) -> float:
        return self.character_count / self.character_limit if self.character_limit else 0.0


def resolve_api_key(explicit: str | None = None) -> str:
    """Return the API key from `explicit`, then the environment, then `.env`."""
    if explicit:
        return explicit.strip()
    load_dotenv(REPO_ROOT / ".env")
    key = (os.environ.get("DEEPL_API_KEY") or "").strip()
    if not key:
        raise DeepLError(
            "DEEPL_API_KEY is not set. Put it in .env at the repository root "
            "(DEEPL_API_KEY=...), or pass --api-key. Keys: "
            "https://www.deepl.com/your-account/keys"
        )
    return key


def api_base_for_key(api_key: str) -> str:
    """Free keys end in ``:fx`` and must use the free host; anything else is Pro."""
    return FREE_HOST if api_key.rstrip().endswith(FREE_KEY_SUFFIX) else PRO_HOST


def iter_batches(texts: Sequence[str], budget: int = BATCH_BUDGET_BYTES) -> Iterator[list[str]]:
    """Split `texts` into request-sized batches, preserving order.

    A single text larger than the budget is yielded alone rather than dropped or
    truncated: splitting mid-sentence would change what the API is asked to
    improve, so let DeepL reject it with a 413 the caller can see.
    """
    batch: list[str] = []
    size = 0
    for text in texts:
        cost = len(json.dumps(text).encode("utf-8")) + 1  # + separator
        if batch and size + cost > budget:
            yield batch
            batch, size = [], 0
        batch.append(text)
        size += cost
    if batch:
        yield batch


class DeepLWriteClient:
    """Thin client over the two ``/v2/write`` endpoints plus usage and languages.

    `transport` exists for tests: it takes ``(url, payload_or_None, headers)`` and
    returns ``(status, body_bytes, response_headers)``, so request shaping, retry
    and error mapping can be exercised without a network or quota.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        api_base: str | None = None,
        timeout: float = 30.0,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_base: float = DEFAULT_BACKOFF_BASE,
        transport: Any | None = None,
        sleep: Any = time.sleep,
        log: Any = None,
    ) -> None:
        self.api_key = resolve_api_key(api_key)
        self.api_base = (api_base or api_base_for_key(self.api_key)).rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self._transport = transport or self._urllib_transport
        self._sleep = sleep
        self._log = log if log is not None else (lambda msg: print(msg, file=sys.stderr))

    # ── HTTP ──────────────────────────────────────────────────────────────────

    def _urllib_transport(
        self, url: str, payload: bytes | None, headers: dict[str, str]
    ) -> tuple[int, bytes, dict[str, str]]:
        request = urllib.request.Request(
            url, data=payload, headers=headers, method="POST" if payload else "GET"
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.status, response.read(), dict(response.headers)
        except urllib.error.HTTPError as exc:  # an HTTP error is still a response
            return exc.code, exc.read(), dict(exc.headers or {})

    def _request(self, path: str, body: dict[str, Any] | None = None) -> Any:
        url = f"{self.api_base}{path}"
        headers = {
            "Authorization": f"DeepL-Auth-Key {self.api_key}",
            "User-Agent": "master-thesis-deepl-write/1.0",
        }
        payload: bytes | None = None
        if body is not None:
            payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
            if len(payload) > MAX_REQUEST_BYTES:
                raise DeepLBadRequest(
                    f"Request body is {len(payload)} bytes, over DeepL's "
                    f"{MAX_REQUEST_BYTES}-byte Write limit. Split the input."
                )

        last_error: DeepLError | None = None
        for attempt in range(self.max_retries + 1):
            status, raw, response_headers = self._transport(url, payload, headers)
            trace_id = _header(response_headers, "X-Trace-ID")

            if 200 <= status < 300:
                # Logged by default: it is the only handle DeepL support has on a request.
                self._log(f"deepl {path} -> HTTP {status} (X-Trace-ID={trace_id or 'n/a'})")
                return json.loads(raw.decode("utf-8")) if raw else None

            error = _error_for(status, raw, trace_id)
            if status not in RETRY_STATUSES or attempt == self.max_retries:
                raise error

            delay = self._retry_delay(response_headers, attempt)
            self._log(
                f"deepl {path} -> HTTP {status}, retrying in {delay:.1f}s "
                f"(attempt {attempt + 1}/{self.max_retries}, X-Trace-ID={trace_id or 'n/a'})"
            )
            self._sleep(delay)
            last_error = error

        raise last_error or DeepLError(f"{path} failed with no response")

    def _retry_delay(self, headers: dict[str, str], attempt: int) -> float:
        """Honour ``Retry-After`` when DeepL sends one, else exponential backoff."""
        retry_after = _header(headers, "Retry-After")
        if retry_after:
            try:
                return max(0.0, float(retry_after))
            except ValueError:
                pass
        return self.backoff_base * (2 ** attempt)

    # ── Endpoints ─────────────────────────────────────────────────────────────

    def rephrase(
        self,
        texts: str | Sequence[str],
        target_lang: str,
        *,
        writing_style: str | None = None,
        tone: str | None = None,
    ) -> list[Improvement]:
        """Improve texts, optionally steering style or tone (never both)."""
        if writing_style and tone:
            raise DeepLBadRequest(
                "writing_style and tone are mutually exclusive — one request may set "
                "only one of them."
            )
        if writing_style is not None and writing_style not in WRITING_STYLES:
            raise DeepLBadRequest(
                f"Unknown writing_style {writing_style!r}. Supported: "
                f"{', '.join(WRITING_STYLES)}"
            )
        if tone is not None and tone not in WRITING_TONES:
            raise DeepLBadRequest(
                f"Unknown tone {tone!r}. Supported: {', '.join(WRITING_TONES)}"
            )
        extra: dict[str, Any] = {}
        if writing_style is not None:
            extra["writing_style"] = writing_style
        if tone is not None:
            extra["tone"] = tone
        return self._improve("/v2/write/rephrase", texts, target_lang, extra)

    def correct(self, texts: str | Sequence[str], target_lang: str) -> list[Improvement]:
        """Fix spelling and grammar with minimal changes to the wording."""
        return self._improve("/v2/write/correct", texts, target_lang, {})

    def _improve(
        self, path: str, texts: str | Sequence[str], target_lang: str, extra: dict[str, Any]
    ) -> list[Improvement]:
        items = [texts] if isinstance(texts, str) else list(texts)
        if not items:
            return []
        if any(not t.strip() for t in items):
            raise DeepLBadRequest("Empty or whitespace-only text cannot be improved.")
        if target_lang not in WRITE_TARGET_LANGS:
            raise DeepLBadRequest(
                f"target_lang {target_lang!r} is not a Write target language. "
                f"Supported: {', '.join(WRITE_TARGET_LANGS)}"
            )

        results: list[Improvement] = []
        for batch in iter_batches(items):
            body = {"text": batch, "target_lang": target_lang, **extra}
            response = self._request(path, body)
            improvements = (response or {}).get("improvements") or []
            if len(improvements) != len(batch):
                raise DeepLError(
                    f"{path} returned {len(improvements)} improvements for "
                    f"{len(batch)} texts — the pairing to the inputs would be wrong."
                )
            # DeepL documents that improvements come back in request order.
            results.extend(
                Improvement(
                    original=original,
                    text=item.get("text", ""),
                    detected_source_language=item.get("detected_source_language", ""),
                    target_language=item.get("target_language", target_lang),
                )
                for original, item in zip(batch, improvements)
            )
        return results

    def improve_latex(
        self,
        document: str,
        target_lang: str,
        *,
        endpoint: str = "rephrase",
        writing_style: str | None = None,
        tone: str | None = None,
    ) -> tuple[str, LatexReport]:
        """Improve the prose lines of a LaTeX document, leaving its markup alone.

        Every LaTeX span is masked before the text is sent and verified on return.
        A line whose markup did not survive is **kept exactly as it was** and named
        in the report — a paragraph silently missing a citation is a far worse
        outcome than a paragraph that went unimproved.
        """
        lines = latex_prose.iter_lines(document)
        report = LatexReport()

        # Sentence, not line, granularity: a paragraph usually mixes markup-free
        # sentences with sentences carrying a citation. Only the former are sent.
        sentences: list[list[str]] = []
        targets: list[tuple[int, int, str]] = []  # (line index, sentence index, text)

        for line_index, line in enumerate(lines):
            if not line.is_prose:
                sentences.append([])
                continue
            parts = latex_prose.split_sentences(line.text)
            sentences.append(parts)
            for part_index, part in enumerate(parts):
                body = part.strip()
                if not body:
                    continue
                if not latex_prose.is_markup_free(body):
                    report.held_back += 1
                    continue
                if not re.search(r"[A-Za-z]{2,}", body):
                    continue
                targets.append((line_index, part_index, body))

        report.skipped = sum(1 for line in lines if not line.is_prose and line.text.strip())
        if not targets:
            return document, report

        improve = self.rephrase if endpoint == "rephrase" else self.correct
        kwargs = {"writing_style": writing_style, "tone": tone} if endpoint == "rephrase" else {}
        improvements = improve([text for _, _, text in targets], target_lang, **kwargs)

        for (line_index, part_index, body), improvement in zip(targets, improvements):
            if improvement.text.strip() == body:
                report.unchanged += 1
                continue
            original = sentences[line_index][part_index]
            # Preserve the sentence's own leading and trailing whitespace so the
            # paragraph reassembles exactly.
            lead = original[: len(original) - len(original.lstrip())]
            trail = original[len(original.rstrip()):]
            sentences[line_index][part_index] = lead + improvement.text.strip() + trail
            report.improved += 1
            report.changes.append((lines[line_index].number, body, improvement.text.strip()))

        rebuilt = [
            ("".join(sentences[i]) + line.ending) if sentences[i] else line.raw
            for i, line in enumerate(lines)
        ]
        return "".join(rebuilt), report

    def usage(self) -> Usage:
        """Characters used this billing period. Write shares the translation quota."""
        data = self._request("/v2/usage") or {}
        return Usage(
            character_count=int(data.get("character_count", 0)),
            character_limit=int(data.get("character_limit", 0)),
        )

    def languages(self) -> list[dict[str, Any]]:
        """Write's target languages, with per-language tone/style support."""
        return self._request("/v3/languages?resource=write") or []

    def supports_style_or_tone(self, target_lang: str) -> bool:
        """Whether `target_lang` accepts a bare (non-``prefer_``) style or tone."""
        for entry in self.languages():
            if entry.get("lang") == target_lang:
                features = entry.get("features") or {}
                return "writing_style" in features or "tone" in features
        return False


def _header(headers: dict[str, str], name: str) -> str | None:
    """Case-insensitive header lookup — HTTP header case is not guaranteed."""
    target = name.lower()
    for key, value in headers.items():
        if key.lower() == target:
            return value
    return None


def _error_for(status: int, raw: bytes, trace_id: str | None) -> DeepLError:
    """Map an HTTP status and JSON error body onto the right exception type."""
    message, code = f"HTTP {status}", None
    try:
        parsed = json.loads(raw.decode("utf-8"))
        if isinstance(parsed, dict):
            message = parsed.get("message") or message
            code = parsed.get("code")
    except (ValueError, UnicodeDecodeError):
        text = raw.decode("utf-8", "replace").strip()
        if text:
            message = text[:500]

    if status in (401, 403):
        cls: type[DeepLError] = DeepLAuthError
    elif status == QUOTA_STATUS:
        cls = DeepLQuotaExceeded
    elif status == 400:
        cls = DeepLBadRequest
    else:
        cls = DeepLError
    return cls(message, status=status, code=code, trace_id=trace_id)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _read_inputs(args: argparse.Namespace) -> list[str]:
    """Texts from positional args, ``-f/--file``, or stdin when piped.

    ``--paragraphs`` splits a file on blank lines, which is what a LaTeX or
    Markdown draft wants: one improvement per paragraph, batched automatically.
    """
    if args.text:
        return list(args.text)
    if args.file:
        raw = Path(args.file).read_text(encoding="utf-8")
    elif not sys.stdin.isatty():
        raw = sys.stdin.read()
    else:
        raise DeepLError("No input. Pass text as arguments, -f FILE, or on stdin.")
    if args.paragraphs:
        return [block.strip() for block in raw.split("\n\n") if block.strip()]
    return [raw.strip()] if raw.strip() else []


def _print_improvements(improvements: Iterable[Improvement], *, show_diff: bool) -> None:
    for index, improvement in enumerate(improvements):
        if index:
            print()
        print(improvement.text)
        if show_diff:
            # stdout is block-buffered when piped; without this the stderr diff
            # lands before the text it annotates.
            sys.stdout.flush()
            marker = "changed" if improvement.changed else "unchanged"
            print(
                f"  [{marker} | {improvement.detected_source_language} -> "
                f"{improvement.target_language}]",
                file=sys.stderr,
            )
            if improvement.changed:
                for line in improvement.unified_diff().splitlines():
                    print(f"  {line}", file=sys.stderr)


def _run_latex(client: DeepLWriteClient, args: argparse.Namespace) -> int:
    """Improve a LaTeX file's prose, reporting anything it refused to touch."""
    source = args.file or (args.text[0] if args.text else None)
    if not source:
        raise DeepLError("--latex needs a file: pass it with -f, or as the argument.")
    path = Path(source)
    if not path.is_file():
        raise DeepLError(f"{path} is not a file. --latex operates on a .tex file.")

    original = path.read_text(encoding="utf-8")
    improved, report = client.improve_latex(
        original,
        args.target_lang,
        endpoint=args.command,
        writing_style=getattr(args, "style", None),
        tone=getattr(args, "tone", None),
    )

    print(report.summary(), file=sys.stderr)
    if report.held_back:
        print(
            f"  note: {report.held_back} sentence(s) carry LaTeX and were left alone. "
            "Both Write endpoints corrupt markup-adjacent text, so they are never sent. "
            "Improve those by hand, or paste the plain prose into the DeepL Write web app.",
            file=sys.stderr,
        )

    if args.in_place:
        if improved == original:
            print(f"{path} unchanged.", file=sys.stderr)
            return 0
        path.write_text(improved, encoding="utf-8")
        print(f"{path} rewritten — review with `git diff {path}`.", file=sys.stderr)
        return 0

    if args.diff:
        # A line-level diff is what a reviewer wants here, not word-level: the
        # unit that changed is the paragraph.
        for line in difflib.unified_diff(
            original.splitlines(), improved.splitlines(),
            fromfile=str(path), tofile=f"{path} (improved)", lineterm="", n=0,
        ):
            print(line, file=sys.stderr)
    else:
        sys.stdout.write(improved)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deepl_write",
        description="Improve text in place with the DeepL Write API.",
    )
    parser.add_argument("--api-key", help="Override DEEPL_API_KEY from .env.")
    parser.add_argument(
        "--api-base",
        help="Override the host. Derived from the key by default "
             "(':fx' suffix -> free host); a mismatch is a 403.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name, help_text in (
        ("rephrase", "Improve fluency and readability; may reword sentences."),
        ("correct", "Fix spelling and grammar only, keeping the wording."),
    ):
        sub = subparsers.add_parser(name, help=help_text)
        sub.add_argument("text", nargs="*", help="Text to improve.")
        sub.add_argument("-f", "--file", help="Read the text from this file instead.")
        sub.add_argument(
            "-t", "--target-lang", required=True, choices=WRITE_TARGET_LANGS,
            help="Language of the text. Write improves within one language.",
        )
        sub.add_argument(
            "--paragraphs", action="store_true",
            help="Split file/stdin input on blank lines and improve each separately.",
        )
        sub.add_argument(
            "--diff", action="store_true",
            help="Also print a word-level before/after to stderr.",
        )
        sub.add_argument(
            "--latex", action="store_true",
            help="Treat the input as LaTeX: improve only sentences that contain no "
                 "markup, leaving every other sentence byte-identical. Required "
                 "for .tex input.",
        )
        sub.add_argument(
            "--in-place", action="store_true",
            help="With --latex, rewrite the file instead of printing to stdout.",
        )
        if name == "rephrase":
            sub.add_argument(
                "--style", choices=WRITING_STYLES,
                help="Writing style. Omit for the DeepL Write web default, which is "
                     "the recommended setting.",
            )
            sub.add_argument("--tone", choices=WRITING_TONES, help="Tone. Excludes --style.")

    subparsers.add_parser("usage", help="Characters used against the quota.")
    subparsers.add_parser("languages", help="Write target languages and their features.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        client = DeepLWriteClient(args.api_key, api_base=args.api_base)

        if args.command == "usage":
            usage = client.usage()
            print(
                f"{usage.character_count:,} / {usage.character_limit:,} characters "
                f"({usage.fraction_used:.1%} of the billing period's quota)"
            )
            return 0

        if args.command == "languages":
            for entry in client.languages():
                features = sorted((entry.get("features") or {}).keys())
                target = "target" if entry.get("usable_as_target") else "source-only"
                print(f"{entry['lang']:<8} {entry['name']:<22} {target:<12} "
                      f"{', '.join(features) or '-'}")
            return 0

        if args.latex:
            return _run_latex(client, args)

        if args.in_place:
            raise DeepLError("--in-place applies only with --latex.")

        texts = _read_inputs(args)
        if not texts:
            print("Nothing to improve.", file=sys.stderr)
            return 0

        if args.command == "rephrase":
            improvements = client.rephrase(
                texts, args.target_lang, writing_style=args.style, tone=args.tone
            )
        else:
            improvements = client.correct(texts, args.target_lang)
        _print_improvements(improvements, show_diff=args.diff)
        return 0

    except DeepLError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
