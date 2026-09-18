"""Per-page citation metadata from headers and footers.

The quote format is not assumed. Ingest stores a CitationConvention on the manual
(spec §5); this module applies that convention to each page. If nothing structured
is found, citation_key falls back to p.<printed-or-pdf-page>.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import fitz

from marine_docs.convention import CitationConvention, extract_citation

# The PDF encodes hyphens and a few ligatures as control bytes.
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

QUOTE_RE = re.compile(
    r"quote\s+(?P<kind>Procedure|Data|Plate|Maintenance Schedules)\s+"
    r"(?P<code>[A-Z]\d[\w.]*)\s+Edition\s+(?P<edition>[A-Za-z0-9. ]+?)[^A-Za-z0-9]*$",
    re.IGNORECASE | re.MULTILINE,
)
PRINTED_PAGE_RE = re.compile(r"\bPage\s+(?P<n>\d+)\s*\((?P<total>\d+)\)", re.IGNORECASE)
PROCEDURE_NO_RE = re.compile(r"^\d{3}-\d+(?:\.\d+)?$")
PLATE_NO_RE = re.compile(r"^P\d{4,6}$", re.IGNORECASE)

# Header noise that is never a component or action title.
_HEADER_NOISE_RE = re.compile(
    r"^(Page\s+\d+.*|\d{3}-\d+(\.\d+)?|"
    r"[A-Z]\d{4,6}(-[\w.]+)?|[-\s.]*)$",
    re.IGNORECASE,
)

# The header prints component then action, but plate pages sometimes print the
# action word first ("Plate | Cylinder Cover Panel").
ACTION_TITLES = frozenset(
    {
        "data", "plate", "dismantling", "mounting", "overhaul", "checking",
        "inspection", "removal", "replacement", "tools", "adjustment",
        "description", "operation", "maintenance", "dismantling and mounting",
    }
)

KIND_MAP = {"D": "data", "M": "procedure", "P": "plate", "A": "schedule"}

HEADER_MAX_Y = 62.0  # header band in PDF points
FOOTER_MIN_Y_FRAC = 0.90


@dataclass
class PageMeta:
    page: int
    doc_code: str | None = None
    edition: str | None = None
    procedure_no: str | None = None
    plate_no: str | None = None
    page_number_printed: str | None = None
    component_title: str | None = None
    action_title: str | None = None
    kind: str = "other"
    citation_key: str | None = None
    citation_fields: dict[str, Any] | None = None

    @property
    def primary_citation(self) -> str | None:
        """Primary reference this page wants quoted."""
        if self.citation_key:
            return (
                f"{self.citation_key} Ed.{self.edition}"
                if self.edition
                else self.citation_key
            )
        primary = self.plate_no or self.procedure_no or self.doc_code
        if not primary:
            return None
        return f"{primary} Ed.{self.edition}" if self.edition else primary


def normalize_text(value: str | None) -> str:
    """Replace the control bytes this PDF family uses for hyphens/ligatures."""
    return _CTRL_RE.sub("-", value or "")


def scan_pdf(
    pdf_path: str, convention: CitationConvention | None = None
) -> dict[int, PageMeta]:
    doc = fitz.open(pdf_path)
    try:
        return {
            i + 1: scan_page(doc[i], i + 1, convention=convention)
            for i in range(doc.page_count)
        }
    finally:
        doc.close()


def scan_page(
    page: fitz.Page,
    page_no: int,
    convention: CitationConvention | None = None,
) -> PageMeta:
    meta = PageMeta(page=page_no)
    text = normalize_text(page.get_text("text"))
    extracted = extract_citation(text, convention, pdf_page=page_no)
    meta.citation_key = extracted.get("citation_key")
    meta.citation_fields = extracted.get("fields") or {}
    if extracted.get("revision") and not meta.edition:
        meta.edition = extracted["revision"]
    if extracted.get("page_number_printed"):
        meta.page_number_printed = extracted["page_number_printed"]
    key = (meta.citation_key or "").strip()
    if re.fullmatch(r"P\d{4,6}", key, re.I):
        meta.plate_no = key.upper()
        meta.kind = "plate"
    elif re.fullmatch(r"\d{3}-\d+(?:\.\d+)?", key):
        meta.procedure_no = key

    quote = QUOTE_RE.search(text)
    if quote:
        meta.doc_code = quote.group("code").strip()
        meta.edition = re.sub(r"\s+", "", quote.group("edition"))
        kind_word = (quote.group("kind") or "").lower()
        if "plate" in kind_word:
            meta.kind = "plate"
        elif "data" in kind_word:
            meta.kind = "data"
        elif "procedure" in kind_word:
            meta.kind = "procedure"
        elif "schedule" in kind_word:
            meta.kind = "schedule"
        else:
            meta.kind = KIND_MAP.get(meta.doc_code[0].upper(), "other")
        if meta.kind == "plate" or meta.doc_code.upper().startswith("P"):
            meta.plate_no = meta.doc_code

    printed = PRINTED_PAGE_RE.search(text)
    if printed:
        meta.page_number_printed = printed.group("n")

    titles: list[str] = []
    for span in _header_spans(page):
        value = span.strip()
        if PROCEDURE_NO_RE.match(value):
            meta.procedure_no = value
            continue
        if PLATE_NO_RE.match(value) and not meta.plate_no:
            meta.plate_no = value.upper()
            continue
        if _HEADER_NOISE_RE.match(value) or len(value) < 3:
            continue
        if value not in titles:
            titles.append(value)

    if len(titles) > 1 and _is_action(titles[0]) and not _is_action(titles[1]):
        titles[0], titles[1] = titles[1], titles[0]
    if titles:
        meta.component_title = titles[0][:200]
    if len(titles) > 1:
        meta.action_title = titles[1][:200]
    if not meta.citation_key or str(meta.citation_key).startswith("p."):
        richer = meta.plate_no or meta.procedure_no or meta.doc_code
        if richer:
            meta.citation_key = richer
    if not meta.citation_key:
        meta.citation_key = f"p.{meta.page_number_printed or page_no}"
    return meta


def _is_action(title: str) -> bool:
    return title.strip().lower() in ACTION_TITLES


def _header_spans(page: fitz.Page) -> list[str]:
    """Header spans in reading order (top band only, page number column first)."""
    spans: list[tuple[float, float, str]] = []
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                x0, y0, _x1, y1 = span["bbox"]
                if y1 > HEADER_MAX_Y:
                    continue
                value = normalize_text(span.get("text"))
                if value.strip():
                    spans.append((y0, x0, value))
    spans.sort(key=lambda s: (round(s[0] / 12), s[1]))
    return [s[2] for s in spans]
