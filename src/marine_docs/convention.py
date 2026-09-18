"""
Agent: Convention Detector (ingest-time, once per new manual)

Input: sampled page header/footer strings from ~15-20 pages (start/middle/end +
        at least one diagram-heavy page). Optional rendered page is not required.
Output: CitationConvention TypedDict stored on manuals.citation_convention.
Model: configured via marine_docs.llm / settings.llm_model (LiteLLM). Overridable.
Failure mode: if the LLM call fails or returns unparseable JSON, fall back to
  heuristic_detect() on the same samples. If that also finds nothing structured,
  return the page-number fallback so citation_key = "p." + printed page always works.

This replaces manufacturer-specific regexes in parser code. Extraction reads the
JSON stored for THIS manual; it does not branch on manufacturer name.
"""

from __future__ import annotations

import logging
import re
from typing import Any, TypedDict

import fitz

from marine_docs.llm import llm_ready

logger = logging.getLogger(__name__)

CONVENTION_DETECTOR = "convention_detector"

SEE_VERBS = ("see", "refer to", "as described in", "as shown in", "according to")
_WS_RE = re.compile(r"\s+")


def _normalize(value: str | None) -> str:
    return _WS_RE.sub(" ", (value or "").replace("\x00", "")).strip()


class CitationConvention(TypedDict, total=False):
    citation_field: str
    pattern: str
    revision_field: str | None
    revision_granularity: str
    label_words: list[str]
    quote_hint: str | None
    has_printed_page: bool
    source: str
    notes: str


FALLBACK: CitationConvention = {
    "citation_field": "page_number_printed",
    "pattern": r"p\.\d+",
    "revision_field": None,
    "revision_granularity": "per_document",
    "label_words": [],
    "quote_hint": None,
    "has_printed_page": True,
    "source": "fallback",
    "notes": "No structured citation convention detected; quote printed page.",
}


def sample_page_bands(pdf_path: str, *, count: int = 18) -> list[dict[str, Any]]:
    """Spread samples across the PDF; prefer at least one drawing-dense page."""
    doc = fitz.open(pdf_path)
    try:
        n = doc.page_count
        if n <= 0:
            return []
        indexes = sorted(set(
            int(round(i * (n - 1) / max(count - 1, 1))) for i in range(min(count, n))
        ))
        dense = max(
            range(n),
            key=lambda i: len(doc[i].get_drawings()),
            default=None,
        )
        if dense is not None and dense not in indexes:
            indexes[-1] = dense
        out: list[dict[str, Any]] = []
        for i in indexes:
            page = doc[i]
            h = page.rect.height
            header = _normalize(page.get_text("text", clip=fitz.Rect(0, 0, page.rect.width, min(80, h * 0.12))))
            footer = _normalize(page.get_text("text", clip=fitz.Rect(0, h * 0.88, page.rect.width, h)))
            out.append(
                {
                    "page": i + 1,
                    "header": (header or "")[:500],
                    "footer": (footer or "")[:500],
                    "drawings": len(page.get_drawings()),
                }
            )
        return out
    finally:
        doc.close()


def heuristic_detect(samples: list[dict[str, Any]]) -> CitationConvention:
    """Pattern-category detector. No manufacturer names, no one-PDF regex monopoly."""
    blob = "\n".join(
        f"{s.get('header') or ''}\n{s.get('footer') or ''}" for s in samples
    )
    has_page = bool(re.search(r"\bpage\s+\d+", blob, re.I))

    quote_proc = re.findall(
        r"quote\s+(Procedure|Data|Plate|Section|Chapter|Drawing)\s+(\S+)\s+(?:Edition|Rev(?:ision)?|Rev\.?)\s+(\S+)",
        blob,
        re.I,
    )
    section_rev = re.findall(
        r"\bSection\s+(\d+(?:\.\d+)+)\s+Rev(?:ision)?\.?\s*([A-Z0-9]+)\b",
        blob,
        re.I,
    )
    chapter_dwg = re.findall(
        r"\bChapter\s+(\d+)\b.*?\bDrawing\s+([A-Z0-9][\w.-]*)",
        blob,
        re.I | re.S,
    )
    dotted = re.findall(r"\b(\d{3}-\d+(?:\.\d+)?)\b", blob)

    if quote_proc:
        labels = sorted({m[0].lower() for m in quote_proc})
        proc_codes = [
            m[1] for m in quote_proc if m[0].lower() in ("procedure", "section", "chapter")
        ]
        all_codes = [m[1] for m in quote_proc]
        pattern = (
            _infer_pattern(proc_codes)
            or _infer_pattern(all_codes)
            or r"\d{3}-\d+(?:\.\d+)?"
        )
        # Manuals often quote a document code (M90201) while headers also
        # carry a dotted section number (902-1.3). Keep both in the pattern
        # so query-time exact-match ranking sees either form.
        if pattern == r"[A-Z]\d{4,6}":
            pattern = r"(?:[A-Z]\d{4,6}|\d{3}-\d+(?:\.\d+)?)"
        return {
            "citation_field": "citation_key",
            "pattern": pattern,
            "revision_field": "edition",
            "revision_granularity": "per_section",
            "label_words": labels,
            "quote_hint": "quote {label} {citation_key} Edition/Rev {revision}",
            "has_printed_page": has_page,
            "source": "heuristic",
            "notes": "Footer tells the reader which code to quote.",
        }
    if section_rev:
        return {
            "citation_field": "citation_key",
            "pattern": r"\d+(?:\.\d+)+",
            "revision_field": "revision",
            "revision_granularity": "per_section",
            "label_words": ["section"],
            "quote_hint": "Section {citation_key} Rev {revision}",
            "has_printed_page": has_page,
            "source": "heuristic",
            "notes": "Dotted section numbers with a revision letter/digit.",
        }
    if chapter_dwg:
        return {
            "citation_field": "citation_key",
            "pattern": r"[A-Z]{2,5}-?\d[\w.-]*",
            "revision_field": None,
            "revision_granularity": "per_document",
            "label_words": ["chapter", "drawing"],
            "quote_hint": "Chapter {n} / Drawing {citation_key}",
            "has_printed_page": has_page,
            "source": "heuristic",
            "notes": "Chapter plus drawing-number citations.",
        }
    if len(dotted) >= 3:
        return {
            "citation_field": "citation_key",
            "pattern": r"\d{3}-\d+(?:\.\d+)?",
            "revision_field": "edition",
            "revision_granularity": "per_section",
            "label_words": ["procedure"],
            "quote_hint": None,
            "has_printed_page": has_page,
            "source": "heuristic",
            "notes": "Repeated NNN-N section codes in headers/footers.",
        }
    fallback = dict(FALLBACK)
    fallback["has_printed_page"] = has_page
    return fallback  # type: ignore[return-value]


def _infer_pattern(codes: list[str]) -> str | None:
    cleaned = [re.sub(r"[^A-Za-z0-9.-]", "", c) for c in codes if c]
    if not cleaned:
        return None
    candidates = [
        (r"\d{3}-\d+(?:\.\d+)?", re.compile(r"^\d{3}-\d+(?:\.\d+)?$")),
        (r"[A-Z]\d{4,6}", re.compile(r"^[A-Z]\d{4,6}$", re.I)),
        (r"P\d{4,6}", re.compile(r"^P\d{4,6}$", re.I)),
        (r"\d+(?:\.\d+)+", re.compile(r"^\d+(?:\.\d+)+$")),
        (r"[A-Z]{2,5}-?\d[\w.-]*", re.compile(r"^[A-Z]{2,5}-?\d[\w.-]*$")),
    ]
    best_pat = None
    best_n = 0
    for pat, cre in candidates:
        n = sum(1 for c in cleaned if cre.match(c))
        if n > best_n:
            best_n = n
            best_pat = pat
    if best_pat and best_n >= max(2, (len(cleaned) + 2) // 3):
        return best_pat
    return None


def _pattern_fits(pattern: str, blob: str) -> bool:
    """A learned regex must match the codes readers are told to quote, not doc-codes nearby."""
    try:
        compiled = re.compile(pattern)
    except re.error:
        return False
    quote_codes = [
        m.group(2)
        for m in re.finditer(
            r"quote\s+(\w+)\s+(\S+)\s+(?:Edition|Rev)",
            blob,
            re.I,
        )
    ]
    if quote_codes:
        hits = sum(1 for c in quote_codes if compiled.search(re.sub(r"[^A-Za-z0-9.-]", "", c)))
        return hits >= max(1, len(quote_codes) // 2)
    return len(compiled.findall(blob)) >= 2


def detect_convention(pdf_path: str, *, allow_llm: bool = True) -> CitationConvention:
    samples = sample_page_bands(pdf_path)
    heuristic = heuristic_detect(samples)
    if not allow_llm or not llm_ready():
        return heuristic
    try:
        from marine_docs.agents import llm_agents
        from marine_docs.agents.jsonutil import parse_json_object

        catalog = "\n".join(
            f"p.{s['page']} drawings={s['drawings']}\nHEADER:\n{s['header']}\nFOOTER:\n{s['footer']}"
            for s in samples[:20]
        )
        parsed = parse_json_object(
            llm_agents.ask(
                llm_agents.CONVENTION_DETECTOR,
                "Sampled headers/footers from one technical manual:\n\n" + catalog,
                max_tokens=500,
            )
        )
        pattern = str(parsed.get("pattern") or "").strip()
        if parsed.get("no_structured_convention") or not pattern:
            return heuristic
        re.compile(pattern)
        blob = "\n".join(f"{s.get('header') or ''}\n{s.get('footer') or ''}" for s in samples)
        if not _pattern_fits(pattern, blob):
            logger.warning("Convention agent pattern %r does not match samples; using heuristic", pattern)
            return heuristic
        return {
            "citation_field": str(parsed.get("citation_field") or "citation_key"),
            "pattern": pattern,
            "revision_field": parsed.get("revision_field"),
            "revision_granularity": str(
                parsed.get("revision_granularity") or "per_section"
            ),
            "label_words": [str(x).lower() for x in (parsed.get("label_words") or [])],
            "quote_hint": parsed.get("quote_hint"),
            "has_printed_page": bool(parsed.get("has_printed_page", True)),
            "source": "llm",
            "notes": str(parsed.get("notes") or "")[:240],
        }
    except Exception:
        logger.exception("Convention detector agent failed; using heuristic")
        return heuristic


def extract_citation(
    text: str,
    convention: CitationConvention | None,
    *,
    printed_page: str | None = None,
    pdf_page: int | None = None,
) -> dict[str, Any]:
    """Apply a stored convention to one page's text. Never assumes a manufacturer."""
    conv = convention or FALLBACK
    blob = text or ""
    revision = None
    key = None
    pattern = conv.get("pattern") or ""
    try:
        compiled = re.compile(pattern)
    except re.error:
        compiled = None

    quote = re.search(
        r"(?:quote|referring to this page)[^\n]{0,80}",
        blob,
        re.I,
    )
    window = quote.group(0) if quote else blob[:800]
    if compiled:
        match = compiled.search(window) or compiled.search(blob)
        if match:
            key = match.group(0)

    rev_field = conv.get("revision_field")
    if rev_field:
        rev_m = re.search(
            r"(?:Edition|Rev(?:ision)?|Rev\.?)\s+([A-Za-z0-9._-]+)",
            blob,
            re.I,
        )
        if rev_m:
            revision = re.sub(r"\s+", "", rev_m.group(1))

    page_m = re.search(r"\bPage\s+(\d+)\s*(?:\((\d+)\))?", blob, re.I)
    printed = printed_page or (page_m.group(1) if page_m else None)
    if not key:
        if printed:
            key = f"p.{printed}"
        elif pdf_page:
            key = f"p.{pdf_page}"
    return {
        "citation_key": key,
        "revision": revision,
        "page_number_printed": printed,
        "fields": {
            "revision": revision,
            "label_words": list(conv.get("label_words") or []),
        },
    }


def citation_key_regex(convention: CitationConvention | None) -> re.Pattern[str]:
    pattern = (convention or FALLBACK).get("pattern") or r"p\.\d+"
    try:
        return re.compile(pattern)
    except re.error:
        return re.compile(r"p\.\d+")


def see_reference_regex(convention: CitationConvention | None) -> re.Pattern[str]:
    """Generic 'see / refer to' + this manual's own citation pattern (spec §7)."""
    inner = (convention or FALLBACK).get("pattern") or r"[A-Za-z0-9][\w.-]*"
    verbs = "|".join(re.escape(v) for v in SEE_VERBS)
    labels = "|".join(
        re.escape(w) for w in (convention or {}).get("label_words") or
        ["procedure", "section", "chapter", "plate", "data", "drawing", "appendix"]
    )
    try:
        return re.compile(
            rf"(?:{verbs})\s+(?:(?P<kind>{labels})\s+)?(?P<ref>{inner})?",
            re.IGNORECASE,
        )
    except re.error:
        return re.compile(
            r"(?:see|refer to)\s+(?P<kind>\w+)?\s*(?P<ref>\S+)?",
            re.I,
        )


def convention_from_keys(keys: list[str]) -> CitationConvention:
    """Infer a convention from already-extracted citation_key values (no re-ingest)."""
    cleaned = [str(k).strip() for k in keys if k and not str(k).lower().startswith("p.")]
    if not cleaned:
        return dict(FALLBACK)  # type: ignore[return-value]
    pattern = _infer_pattern(cleaned)
    if not pattern:
        return dict(FALLBACK)  # type: ignore[return-value]
    return {
        "citation_field": "citation_key",
        "pattern": pattern,
        "revision_field": "edition",
        "revision_granularity": "per_section",
        "label_words": [],
        "quote_hint": None,
        "has_printed_page": True,
        "source": "backfill",
        "notes": "Inferred from stored citation_key values.",
    }


def persist_convention(manual_id: Any, convention: CitationConvention) -> None:
    from psycopg.types.json import Jsonb

    from marine_docs.db import connect

    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE manuals SET citation_convention = %s WHERE id = %s",
            (Jsonb(dict(convention)), manual_id),
        )
        conn.commit()
