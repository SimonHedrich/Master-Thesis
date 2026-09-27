"""Paragraph-level DeepL Write pass over the manuscript, reviewed before it lands.

Three stages. Only the first spends quota, and everything it fetches is on
disk before it is used, so nothing has to be fetched twice.

``prepare``
    For every prose paragraph of the given ``.tex`` files: flatten it to plain
    prose (citations stripped, ``\\textit{}`` and numeric math unwrapped, the
    rest masked as noun tokens — see ``latex_prose.flatten``), send it to
    ``/v2/write/rephrase`` **one paragraph per request**, append the response to
    ``reports/deepl_rephrase/api_log.jsonl`` at once, then restore the markup
    and re-attach the citations (``latex_prose.restore``). Writes a manifest and
    review files of twelve paragraphs each under
    ``reports/deepl_rephrase/<file-stem>/``.
``review``
    Not a subcommand: the ``chunk-NN.review.txt`` files are handed to reviewer
    subagents, one file each, who fill the ``--- final`` block of every entry
    in place. The instructions they follow are the header of every review file
    (``rephrase_review_header.md``).
``apply``
    Read the review files back and write the finals into the ``.tex`` — after
    checking, per paragraph, that the LaTeX span inventory is unchanged, that
    no prose dash or semicolon crept in, and that the source file has not moved
    since ``prepare``. Anything that fails a check keeps its original line and
    is named in the summary.
``status``
    What is fetched, reviewed and applied per bundle.

Run from the repository root:

    uv run python -m scripts.thesis.rephrase_manuscript prepare --dry-run \\
        thesis/manuscript/chapters/1-Introduction.tex
    uv run python -m scripts.thesis.rephrase_manuscript prepare \\
        thesis/manuscript/preamble/abstract_eng.tex thesis/manuscript/chapters/1-Introduction.tex
    uv run python -m scripts.thesis.rephrase_manuscript prepare -t de \\
        thesis/manuscript/preamble/abstract_ger.tex
    uv run python -m scripts.thesis.rephrase_manuscript apply reports/deepl_rephrase/1-Introduction
    uv run python -m scripts.thesis.rephrase_manuscript apply --in-place reports/deepl_rephrase/1-Introduction
    uv run python -m scripts.thesis.rephrase_manuscript status

Quota: ``prepare`` checks ``/v2/usage`` first and refuses to start a run the
remaining quota cannot cover (``--force-quota`` overrides). A 456 mid-run stops
cleanly with exit code 2; ``prepare --resume`` continues from the log, on
``DEEPL_API_KEY_2`` if that is set and the first key is spent.

See scripts/thesis/README.md.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import difflib
import hashlib
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from scripts.thesis import deepl_write, latex_prose

REPO_ROOT = Path(__file__).resolve().parents[2]
MANUSCRIPT_DIR = REPO_ROOT / "thesis" / "manuscript"
DEFAULT_OUT_DIR = REPO_ROOT / "reports" / "deepl_rephrase"
LOG_NAME = "api_log.jsonl"
MANIFEST_NAME = "manifest.json"
HEADER_PATH = Path(__file__).with_name("rephrase_review_header.md")

DEFAULT_TARGET_LANG = "en-US"
DEFAULT_CHUNK_SIZE = 12
SECOND_KEY_ENV = "DEEPL_API_KEY_2"

# A prose line this long that also occurs verbatim in another manuscript file
# is a deliberate echo (the research questions, restated in Chapter 5) and must
# stay identical in both places.
ECHO_MIN_CHARS = 40

EXIT_QUOTA = 2
EXIT_PREFLIGHT = 3

_HEADING_RE = re.compile(
    r"\\(?:chapter|section|subsection|subsubsection|paragraph)\*?\s*\{((?:[^{}]|\{[^{}]*\})*)\}"
)
_BRITISH_RE = re.compile(
    r"\b(?:\w+is(?:e|ed|es|ing|ation|ations)|behaviours?|artefacts?|colours?|judgements?"
    r"|labelled|labelling|favourable|penalis\w*|analys(?:e|ed|es|ing)|modelling|neighbours?"
    r"|grey|centre|metres?|optimis\w*|minimis\w*|maximis\w*|summaris\w*|characteris\w*)\b",
    re.IGNORECASE,
)
_BRITISH_FALSE_POSITIVES = frozenset({
    "premise", "premises", "promise", "promised", "promises", "raise", "raised", "raises",
    "rise", "rises", "arise", "arises", "wise", "precise", "concise", "noise", "exercise",
    "exercised", "exercises", "otherwise", "likewise", "expertise", "franchise", "advise",
    "advised", "revise", "revised", "revises", "revision", "devise", "devised", "supervise",
    "supervised", "supervises", "comprise", "comprised", "comprises", "surprise", "surprised",
    "surprises", "disguise", "compromise", "compromised", "compromises", "enterprise",
    "paradise", "merchandise", "treatise", "chemise", "cruise", "bruise", "poise", "practise",
    "practised", "practises", "reprise", "demise", "excise", "incise", "incised", "despise",
    "despised", "arises", "arising", "rising", "raising", "promising", "exercising",
    "advising", "revising", "supervising", "comprising", "surprising", "compromising",
})
_CITE_SUBJECT_RE = re.compile(
    r"\\cite\{[^}]*\}\s+(?:show|shows|showed|shown|find|finds|found|propose|proposes|proposed"
    r"|introduce|introduces|introduced|report|reports|reported|train|trains|trained|use|uses"
    r"|used|argue|argues|argued|demonstrate|demonstrates|demonstrated|present|presents"
    r"|presented|generalize|generalized|describe|describes|described|observe|observes"
    r"|observed|note|notes|noted|measure|measures|measured|evaluate|evaluates|evaluated"
    r"|compare|compares|compared|achieve|achieves|achieved|is|are|was|were|has|have|had)\b"
)
_ET_AL_SUBJECT_RE = re.compile(r"\bet al\.\s+(?:show|found|propose|introduce|report|train|argue|demonstrate|present)\w*\b")
_LONG_SENTENCE_WORDS = 55


# ── Data ──────────────────────────────────────────────────────────────────────

@dataclass
class Paragraph:
    """One prose line of a manuscript file, through the pipeline."""

    id: str                       # "P007"
    line: int                     # 1-based line number in the source file
    section: str                  # nearest heading above it, for reviewer context
    indent: str                   # leading whitespace of the line, restored on apply
    original: str                 # the line, stripped
    sent: str = ""                # flattened text that went to the API ("" if not sent)
    received: str | None = None   # the API's improved plain text
    restored: str | None = None   # received with the LaTeX put back
    flags: list[str] = field(default_factory=list)
    status: str = "pending"       # pending | fetched | unchanged | held-back | skipped
    chars: int = 0

    @property
    def needs_review(self) -> bool:
        return self.status == "fetched"


@dataclass
class Manifest:
    source: str                   # path relative to the repo root
    source_sha: str
    target_lang: str
    created: str
    paragraphs: list[Paragraph]
    applied_at: str | None = None

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Manifest":
        paragraphs = [Paragraph(**p) for p in data["paragraphs"]]
        return cls(
            source=data["source"], source_sha=data["source_sha"],
            target_lang=data["target_lang"], created=data["created"],
            paragraphs=paragraphs, applied_at=data.get("applied_at"),
        )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def request_key(target_lang: str, sent: str) -> str:
    """The cache key of one request: language and exact text."""
    return sha256_text(f"{target_lang}\n{sent}")


# ── The append-only API log ───────────────────────────────────────────────────

class ApiLog:
    """``api_log.jsonl``: every response, written the moment it arrives.

    This file, not the manifest, is the source of truth. A manifest or review
    file can be regenerated from it at no cost with ``prepare --resume``.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._cache: dict[str, dict[str, Any]] = {}
        if path.is_file():
            for raw in path.read_text(encoding="utf-8").splitlines():
                raw = raw.strip()
                if not raw:
                    continue
                entry = json.loads(raw)
                self._cache[entry["sha"]] = entry

    def lookup(self, target_lang: str, sent: str) -> dict[str, Any] | None:
        return self._cache.get(request_key(target_lang, sent))

    def record(self, *, source: str, line: int, target_lang: str, sent: str,
               received: str, detected_source_language: str, trace_id: str | None) -> dict[str, Any]:
        entry = {
            "sha": request_key(target_lang, sent),
            "file": source,
            "line": line,
            "target_lang": target_lang,
            "sent": sent,
            "received": received,
            "detected_source_language": detected_source_language,
            "trace_id": trace_id,
            "chars": len(sent),
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._cache[entry["sha"]] = entry
        return entry

    def __len__(self) -> int:
        return len(self._cache)


# ── Automatic checks on a restored paragraph ──────────────────────────────────

def _prose_only(text: str) -> str:
    """The text with every LaTeX span removed, for checks that must ignore markup."""
    return latex_prose.PLACEHOLDER_RE.sub(" ", latex_prose.mask(text).text)


def _british_hits(prose: str) -> list[str]:
    return [
        m.group(0) for m in _BRITISH_RE.finditer(prose)
        if m.group(0).lower() not in _BRITISH_FALSE_POSITIVES
    ]


def _digit_runs_outside_math(text: str) -> int:
    return len(re.findall(r"\d+", _prose_only(text)))


def _longest_sentence(text: str) -> int:
    return max(
        (len(s.split()) for s in latex_prose.split_sentences(_prose_only(text))), default=0
    )


def style_flags(original: str, candidate: str) -> list[str]:
    """House-style problems the candidate has that the original did not."""
    flags: list[str] = []
    o_prose, c_prose = _prose_only(original), _prose_only(candidate)

    if re.search(r"\s---?\s|---", c_prose) and not re.search(r"\s---?\s|---", o_prose):
        flags.append("dash")
    if ";" in c_prose and ";" not in o_prose:
        flags.append("semicolon")
    british = [w for w in _british_hits(c_prose) if w not in _british_hits(o_prose)]
    if british:
        flags.append("british: " + ", ".join(sorted(set(british))))
    if _digit_runs_outside_math(candidate) > _digit_runs_outside_math(original):
        flags.append("bare-number")
    longest = _longest_sentence(candidate)
    if longest > _LONG_SENTENCE_WORDS and longest > _longest_sentence(original):
        flags.append(f"long-sentence: {longest} words")
    if len(re.findall(r"\b(?:this|the) thesis\b", c_prose, re.I)) > len(re.findall(r"\b(?:this|the) thesis\b", o_prose, re.I)):
        flags.append("this-thesis")
    if c_prose.count("rather than") > o_prose.count("rather than"):
        flags.append("rather-than")
    if _CITE_SUBJECT_RE.search(candidate) or _ET_AL_SUBJECT_RE.search(candidate) \
            or re.search(r"\\(?:citeauthor|textcite)\b", candidate):
        flags.append("cite-subject")
    return flags


# ── Extraction ────────────────────────────────────────────────────────────────

def manuscript_echo_lines(manuscript_dir: Path = MANUSCRIPT_DIR) -> dict[str, set[Path]]:
    """Every prose line of every manuscript file, mapped to the files it is in."""
    seen: dict[str, set[Path]] = collections.defaultdict(set)
    for path in sorted(manuscript_dir.rglob("*.tex")):
        for line in latex_prose.iter_lines(path.read_text(encoding="utf-8")):
            if line.is_prose:
                text = line.text.strip()
                if len(text) >= ECHO_MIN_CHARS:
                    seen[text].add(path.resolve())
    return seen


def extract_paragraphs(path: Path, document: str, echoes: dict[str, set[Path]] | None = None) -> list[Paragraph]:
    """All prose lines of `document`, flattened, with echoes held back."""
    paragraphs: list[Paragraph] = []
    section = ""
    for line in latex_prose.iter_lines(document):
        heading = _HEADING_RE.search(line.text)
        if heading:
            section = heading.group(1)
        if not line.is_prose:
            continue
        stripped = line.text.strip()
        indent = line.text[: len(line.text) - len(line.text.lstrip())]
        paragraph = Paragraph(
            id=f"P{len(paragraphs) + 1:03d}", line=line.number, section=section,
            indent=indent, original=stripped,
        )
        flat = latex_prose.flatten(stripped)
        if not flat.has_prose:
            paragraph.status = "skipped"
            paragraph.flags.append("no-prose")
        elif echoes and len(echoes.get(stripped, set()) - {path.resolve()}) > 0:
            paragraph.status = "held-back"
            paragraph.flags.append("verbatim-echo")
        else:
            paragraph.sent = flat.text
            paragraph.chars = len(flat.text)
        paragraphs.append(paragraph)
    return paragraphs


def restore_paragraph(paragraph: Paragraph) -> None:
    """Fill `restored`, `flags` and `status` from `received`."""
    assert paragraph.received is not None
    flat = latex_prose.flatten(paragraph.original)
    kept = [f for f in paragraph.flags if f == "cached"]
    if paragraph.received.strip() == paragraph.sent.strip():
        paragraph.restored = paragraph.original
        paragraph.flags = kept + ["unchanged"]
        paragraph.status = "unchanged"
        return
    restored = latex_prose.restore(paragraph.received, flat)
    if not restored.text:
        paragraph.restored = paragraph.original
        paragraph.flags = kept + list(restored.flags)
        paragraph.status = "fetched"
        return
    paragraph.restored = restored.text
    paragraph.flags = kept + list(restored.flags) + style_flags(paragraph.original, restored.text)
    paragraph.status = "fetched"


# ── Review files ──────────────────────────────────────────────────────────────

_ENTRY_RE = re.compile(r"^#### (P\d{3})\s+(\S+):(\d+)\s+flags:\s*(.*)$")
_BLOCK_NAMES = ("original", "rephrased", "final", "verdict")


def review_header() -> str:
    return HEADER_PATH.read_text(encoding="utf-8").rstrip() + "\n"


def write_review_files(bundle_dir: Path, manifest: Manifest, chunk_size: int) -> list[Path]:
    """Split the paragraphs that need review into chunk files. Existing files
    with filled finals are kept so a re-run never discards reviewer work."""
    to_review = [p for p in manifest.paragraphs if p.needs_review]
    source_name = Path(manifest.source).name
    written: list[Path] = []
    header = review_header()
    for index in range(0, len(to_review), chunk_size):
        chunk = to_review[index:index + chunk_size]
        path = bundle_dir / f"chunk-{index // chunk_size + 1:02d}.review.txt"
        existing = parse_review_file(path) if path.is_file() else {}
        parts = [header, "", f"<!-- {source_name}: {len(chunk)} paragraphs -->", ""]
        for p in chunk:
            previous = existing.get(p.id)
            final = previous["final"] if previous and previous.get("original") == p.original else ""
            verdict = previous["verdict"] if previous and previous.get("original") == p.original else ""
            parts += [
                f"#### {p.id}  {source_name}:{p.line}  flags: {', '.join(p.flags) or '-'}",
                f"## section: {p.section or '-'}",
                "--- original",
                p.original,
                "--- rephrased",
                p.restored or "",
                "--- final",
                final,
                "--- verdict",
                verdict,
                "",
            ]
        path.write_text("\n".join(parts) + "\n", encoding="utf-8")
        written.append(path)
    return written


def parse_review_file(path: Path) -> dict[str, dict[str, str]]:
    """Entries of a review file by paragraph id. Blocks are stripped of
    surrounding blank lines; a multi-line final is joined with single spaces."""
    entries: dict[str, dict[str, str]] = {}
    current: dict[str, Any] | None = None
    block: str | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        entry = _ENTRY_RE.match(raw)
        if entry:
            current = {"id": entry.group(1), "line": int(entry.group(3)), "flags": entry.group(4)}
            for name in _BLOCK_NAMES:
                current[name] = []
            entries[current["id"]] = current
            block = None
            continue
        if current is None:
            continue
        if raw.startswith("--- ") and raw[4:].strip() in _BLOCK_NAMES:
            block = raw[4:].strip()
            continue
        if raw.startswith("## section:"):
            continue
        if block is not None:
            current[block].append(raw)
    for entry in entries.values():
        for name in _BLOCK_NAMES:
            lines = [line for line in entry[name]]
            while lines and not lines[0].strip():
                lines.pop(0)
            while lines and not lines[-1].strip():
                lines.pop()
            entry[name] = " ".join(line.strip() for line in lines) if name != "verdict" else " ".join(lines).strip()
            if name != "verdict":
                entry[name + "_lines"] = len(lines)
    return entries


# ── Bundle I/O ────────────────────────────────────────────────────────────────

def bundle_dir_for(source: Path, out_dir: Path) -> Path:
    return out_dir / source.stem


def save_manifest(bundle_dir: Path, manifest: Manifest) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest.to_json(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )


def load_manifest(bundle_dir: Path) -> Manifest:
    path = bundle_dir / MANIFEST_NAME
    if not path.is_file():
        raise SystemExit(f"error: {path} does not exist; run `prepare` first.")
    return Manifest.from_json(json.loads(path.read_text(encoding="utf-8")))


def relative_source(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


# ── prepare ───────────────────────────────────────────────────────────────────

class QuotaStop(RuntimeError):
    """The key's quota ran out mid-run; the log holds everything fetched so far."""


def _client_for(api_key: str | None, api_base: str | None, log_messages: list[str], env_var: str = "DEEPL_API_KEY") -> deepl_write.DeepLWriteClient:
    key = deepl_write.resolve_api_key(api_key, env_var=env_var)
    return deepl_write.DeepLWriteClient(key, api_base=api_base, log=log_messages.append)


def _trace_id(log_messages: list[str]) -> str | None:
    for message in reversed(log_messages):
        match = re.search(r"X-Trace-ID=(\S+)\)", message)
        if match:
            return None if match.group(1) == "n/a" else match.group(1)
    return None


def prepare(
    sources: Sequence[Path],
    *,
    target_lang: str = DEFAULT_TARGET_LANG,
    out_dir: Path = DEFAULT_OUT_DIR,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    dry_run: bool = False,
    resume: bool = False,
    limit: int | None = None,
    force_quota: bool = False,
    api_key: str | None = None,
    api_base: str | None = None,
    client: deepl_write.DeepLWriteClient | None = None,
    echoes: dict[str, set[Path]] | None = None,
    stderr: Any = None,
) -> int:
    err = stderr or sys.stderr
    log = ApiLog(out_dir / LOG_NAME)
    if echoes is None:
        echoes = manuscript_echo_lines()

    # Extract everything first so the pre-flight can count what is uncached.
    jobs: list[tuple[Path, str, Manifest]] = []
    to_send: list[tuple[Manifest, Paragraph]] = []
    for source in sources:
        document = source.read_text(encoding="utf-8")
        paragraphs = extract_paragraphs(source, document, echoes)
        manifest = Manifest(
            source=relative_source(source), source_sha=sha256_text(document),
            target_lang=target_lang,
            created=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            paragraphs=paragraphs,
        )
        jobs.append((source, document, manifest))
        for p in paragraphs:
            if p.status != "pending":
                continue
            cached = log.lookup(target_lang, p.sent)
            if cached is not None:
                p.received = cached["received"]
                p.flags.append("cached")
                restore_paragraph(p)
            else:
                to_send.append((manifest, p))

    if limit is not None:
        for _, p in to_send[limit:]:
            p.status = "skipped"
            p.flags.append("over-limit")
        to_send = to_send[:limit]

    chars_needed = sum(p.chars for _, p in to_send)
    held = sum(1 for _, _, m in jobs for p in m.paragraphs if p.status == "held-back")
    cached_count = sum(1 for _, _, m in jobs for p in m.paragraphs if "cached" in p.flags)
    print(
        f"{len(jobs)} file(s): {sum(len(m.paragraphs) for _, _, m in jobs)} paragraph(s), "
        f"{len(to_send)} to send ({chars_needed:,} characters), {cached_count} already in the log, "
        f"{held} held back as verbatim echoes",
        file=err,
    )
    for _, _, m in jobs:
        for p in m.paragraphs:
            if p.status == "held-back":
                print(f"  held back  {Path(m.source).name}:{p.line}  {p.original[:70]}…", file=err)

    if dry_run:
        for _, _, m in jobs:
            for p in m.paragraphs:
                if p.status == "pending":
                    print(f"  would send {Path(m.source).name}:{p.line}  {p.chars:5d} chars  {p.sent[:70]}…", file=err)
        return 0

    log_messages: list[str] = []
    if client is None:
        client = _client_for(api_key, api_base, log_messages)
    else:
        client._log = log_messages.append  # capture trace ids from an injected client

    if to_send:
        usage = client.usage()
        remaining = usage.character_limit - usage.character_count
        print(f"quota before: {usage.character_count:,} / {usage.character_limit:,} used, {remaining:,} remaining", file=err)
        if chars_needed > remaining:
            second = os.environ.get(SECOND_KEY_ENV, "").strip()
            if resume and second and not api_key:
                print(f"first key cannot cover {chars_needed:,} characters; switching to {SECOND_KEY_ENV}", file=err)
                client = _client_for(None, api_base, log_messages, env_var=SECOND_KEY_ENV)
                usage = client.usage()
                remaining = usage.character_limit - usage.character_count
                print(f"quota before ({SECOND_KEY_ENV}): {usage.character_count:,} / {usage.character_limit:,} used, {remaining:,} remaining", file=err)
            if chars_needed > remaining and not force_quota:
                print(
                    f"error: this run needs {chars_needed:,} characters but only {remaining:,} remain. "
                    f"Add a second key as {SECOND_KEY_ENV} in .env and re-run with --resume, or pass "
                    f"--force-quota to fetch what fits and resume later.",
                    file=err,
                )
                return EXIT_PREFLIGHT

    exit_code = 0
    fetched = 0
    try:
        for manifest, p in to_send:
            try:
                improvements = client.rephrase([p.sent], target_lang)
            except deepl_write.DeepLQuotaExceeded as exc:
                raise QuotaStop(str(exc)) from exc
            improvement = improvements[0]
            log.record(
                source=manifest.source, line=p.line, target_lang=target_lang, sent=p.sent,
                received=improvement.text, detected_source_language=improvement.detected_source_language,
                trace_id=_trace_id(log_messages),
            )
            p.received = improvement.text
            restore_paragraph(p)
            fetched += 1
    except QuotaStop as exc:
        pending = sum(1 for _, p in to_send if p.received is None)
        print(
            f"quota exhausted after {fetched} paragraph(s) ({exc}); {pending} still unfetched. "
            f"Everything fetched is in {log.path}. Add {SECOND_KEY_ENV} to .env and re-run with --resume.",
            file=err,
        )
        exit_code = EXIT_QUOTA

    for source, _, manifest in jobs:
        bundle_dir = bundle_dir_for(source, out_dir)
        save_manifest(bundle_dir, manifest)
        files = write_review_files(bundle_dir, manifest, chunk_size)
        counts = collections.Counter(p.status for p in manifest.paragraphs)
        flag_counts = collections.Counter(f.split(":")[0] for p in manifest.paragraphs for f in p.flags if p.status == "fetched")
        print(
            f"{manifest.source}: {counts.get('fetched', 0)} to review in {len(files)} chunk file(s), "
            f"{counts.get('unchanged', 0)} unchanged, {counts.get('held-back', 0)} held back, "
            f"{counts.get('skipped', 0)} skipped, {counts.get('pending', 0)} unfetched -> {bundle_dir}",
            file=err,
        )
        if flag_counts:
            print("  flags: " + ", ".join(f"{k} ×{v}" for k, v in sorted(flag_counts.items())), file=err)

    if fetched:
        try:
            usage = client.usage()
            print(f"quota after: {usage.character_count:,} / {usage.character_limit:,} used", file=err)
        except deepl_write.DeepLError as exc:  # the run itself succeeded; do not fail on this
            print(f"quota after: unavailable ({exc})", file=err)
    return exit_code


# ── apply ─────────────────────────────────────────────────────────────────────

@dataclass
class Decision:
    paragraph: Paragraph
    outcome: str                  # applied | kept | unreviewed | rejected | not-reviewable
    final: str | None = None
    verdict: str = ""
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_final(original: str, final: str) -> tuple[list[str], list[str]]:
    """(rejections, warnings) for a reviewer-written final against its original."""
    rejections: list[str] = []
    if not final.strip():
        return ["empty final"], []
    if latex_prose.span_inventory(final) != latex_prose.span_inventory(original):
        o_spans, o_keys = latex_prose.span_inventory(original)
        f_spans, f_keys = latex_prose.span_inventory(final)
        missing = sorted((collections.Counter(o_spans) - collections.Counter(f_spans)).elements())
        added = sorted((collections.Counter(f_spans) - collections.Counter(o_spans)).elements())
        missing_keys = sorted((collections.Counter(o_keys) - collections.Counter(f_keys)).elements())
        added_keys = sorted((collections.Counter(f_keys) - collections.Counter(o_keys)).elements())
        detail = []
        if missing:
            detail.append("missing " + ", ".join(missing))
        if added:
            detail.append("added " + ", ".join(added))
        if missing_keys:
            detail.append("missing cite keys " + ", ".join(missing_keys))
        if added_keys:
            detail.append("added cite keys " + ", ".join(added_keys))
        rejections.append("LaTeX span inventory changed: " + "; ".join(detail))
    prose = _prose_only(final)
    if re.search(r"\s---?\s|---", prose) and not re.search(r"\s---?\s|---", _prose_only(original)):
        rejections.append("parenthetical dash in prose")
    if ";" in prose and ";" not in _prose_only(original):
        rejections.append("semicolon in prose")
    if re.search(r"\\(?:citeauthor|textcite)\b", final):
        rejections.append("\\citeauthor or \\textcite")
    unescaped = re.sub(r"\\[{}]", "", final)
    if unescaped.count("{") != unescaped.count("}"):
        rejections.append("unbalanced braces")
    if original.startswith("\\item") != final.startswith("\\item"):
        rejections.append("\\item prefix mismatch")
    warnings = [f for f in style_flags(original, final) if f != "cite-subject"]
    if "cite-subject" in style_flags(original, final):
        rejections.append("citation used as a sentence subject")
    return rejections, warnings


def decide(manifest: Manifest, entries: dict[str, dict[str, str]]) -> list[Decision]:
    decisions: list[Decision] = []
    for p in manifest.paragraphs:
        if not p.needs_review:
            decisions.append(Decision(p, "not-reviewable"))
            continue
        entry = entries.get(p.id)
        if entry is None or not entry["final"].strip():
            outcome = "kept" if entry and entry["verdict"].strip().lower().startswith("original") else "unreviewed"
            decisions.append(Decision(p, outcome, verdict=entry["verdict"] if entry else ""))
            continue
        final = entry["final"].strip()
        rejections, warnings = validate_final(p.original, final)
        if entry.get("final_lines", 1) > 1:
            warnings.append(f"final was {entry['final_lines']} lines, joined with spaces")
        if final == p.original:
            decisions.append(Decision(p, "kept", final=final, verdict=entry["verdict"]))
        elif rejections:
            decisions.append(Decision(p, "rejected", final=final, verdict=entry["verdict"], reasons=rejections))
        else:
            decisions.append(Decision(p, "applied", final=final, verdict=entry["verdict"], warnings=warnings))
    return decisions


def apply_decisions(document: str, decisions: Iterable[Decision]) -> str:
    lines = latex_prose.iter_lines(document)
    by_line = {d.paragraph.line: d for d in decisions if d.outcome == "applied"}
    rebuilt: list[str] = []
    for line in lines:
        decision = by_line.get(line.number)
        if decision is None:
            rebuilt.append(line.raw)
        else:
            rebuilt.append(decision.paragraph.indent + (decision.final or "") + line.ending)
    return "".join(rebuilt)


def apply(bundle_dir: Path, *, in_place: bool = False, stderr: Any = None) -> int:
    err = stderr or sys.stderr
    manifest = load_manifest(bundle_dir)
    source = REPO_ROOT / manifest.source
    document = source.read_text(encoding="utf-8")
    if sha256_text(document) != manifest.source_sha:
        print(
            f"error: {manifest.source} changed since `prepare` ran. Re-run "
            f"`prepare --resume` on it (cached, costs nothing) and review again.",
            file=err,
        )
        return 1

    entries: dict[str, dict[str, str]] = {}
    for path in sorted(bundle_dir.glob("chunk-*.review.txt")):
        entries.update(parse_review_file(path))
    decisions = decide(manifest, entries)

    counts = collections.Counter(d.outcome for d in decisions)
    name = Path(manifest.source).name
    print(
        f"{name}: {counts['applied']} applied, {counts['kept']} kept by verdict, "
        f"{counts['unreviewed']} unreviewed, {counts['rejected']} rejected, "
        f"{counts['not-reviewable']} not reviewable (unchanged/held back/skipped)",
        file=err,
    )
    for d in decisions:
        if d.outcome == "rejected":
            print(f"  REJECTED {d.paragraph.id} {name}:{d.paragraph.line}: " + "; ".join(d.reasons), file=err)
        elif d.outcome == "unreviewed":
            print(f"  unreviewed {d.paragraph.id} {name}:{d.paragraph.line}", file=err)
    for d in decisions:
        if d.outcome == "applied" and d.warnings:
            print(f"  warn {d.paragraph.id} {name}:{d.paragraph.line}: " + "; ".join(d.warnings), file=err)

    improved = apply_decisions(document, decisions)
    if improved == document:
        print(f"{name}: nothing to write.", file=err)
        return 0
    if in_place:
        source.write_text(improved, encoding="utf-8")
        manifest.source_sha = sha256_text(improved)
        manifest.applied_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        save_manifest(bundle_dir, manifest)
        print(f"{manifest.source} rewritten ({counts['applied']} paragraph(s)); review with `git diff`.", file=err)
    else:
        for line in difflib.unified_diff(
            document.splitlines(), improved.splitlines(),
            fromfile=manifest.source, tofile=f"{manifest.source} (reviewed)", lineterm="", n=0,
        ):
            print(line)
    return 0


# ── status ────────────────────────────────────────────────────────────────────

def status(out_dir: Path = DEFAULT_OUT_DIR, stderr: Any = None) -> int:
    err = stderr or sys.stderr
    log_path = out_dir / LOG_NAME
    log = ApiLog(log_path)
    print(f"{log_path}: {len(log)} response(s) logged", file=err)
    for bundle_dir in sorted(p for p in out_dir.iterdir() if p.is_dir()) if out_dir.is_dir() else []:
        if not (bundle_dir / MANIFEST_NAME).is_file():
            continue
        manifest = load_manifest(bundle_dir)
        counts = collections.Counter(p.status for p in manifest.paragraphs)
        chunks = sorted(bundle_dir.glob("chunk-*.review.txt"))
        reviewed = total = 0
        for path in chunks:
            for entry in parse_review_file(path).values():
                total += 1
                reviewed += bool(entry["final"].strip() or entry["verdict"].strip())
        applied = f"applied {manifest.applied_at}" if manifest.applied_at else "not applied"
        print(
            f"  {manifest.source}: {len(manifest.paragraphs)} paragraphs, "
            f"{counts.get('fetched', 0)} to review, {counts.get('pending', 0)} unfetched, "
            f"{counts.get('unchanged', 0)} unchanged, {counts.get('held-back', 0)} held back; "
            f"{reviewed}/{total} reviewed across {len(chunks)} chunk(s); {applied}",
            file=err,
        )
    return 0


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rephrase_manuscript",
        description="DeepL Write rephrase pass over manuscript paragraphs, with reviewed apply.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Bundle root (default reports/deepl_rephrase).")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="Fetch rephrasings and write review files.")
    p.add_argument("sources", nargs="+", type=Path, help=".tex files to process.")
    p.add_argument("-t", "--target-lang", default=DEFAULT_TARGET_LANG, choices=deepl_write.WRITE_TARGET_LANGS)
    p.add_argument("--dry-run", action="store_true", help="Count and list what would be sent; no request.")
    p.add_argument("--resume", action="store_true", help="Reuse logged responses; allow the second key.")
    p.add_argument("--limit", type=int, help="Send at most N paragraphs (smoke test).")
    p.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    p.add_argument("--force-quota", action="store_true", help="Start even if the quota cannot cover the run.")
    p.add_argument("--api-key", help="Override the key from .env.")
    p.add_argument("--api-base", help="Override the host (derived from the key by default).")

    a = sub.add_parser("apply", help="Write reviewed finals back into the .tex.")
    a.add_argument("bundle", type=Path, help="Bundle directory, e.g. reports/deepl_rephrase/1-Introduction.")
    a.add_argument("--in-place", action="store_true", help="Rewrite the .tex instead of printing a diff.")

    sub.add_parser("status", help="Fetched / reviewed / applied per bundle.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            return prepare(
                args.sources, target_lang=args.target_lang, out_dir=args.out_dir,
                chunk_size=args.chunk_size, dry_run=args.dry_run, resume=args.resume,
                limit=args.limit, force_quota=args.force_quota, api_key=args.api_key,
                api_base=args.api_base,
            )
        if args.command == "apply":
            return apply(args.bundle, in_place=args.in_place)
        return status(args.out_dir)
    except deepl_write.DeepLError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
