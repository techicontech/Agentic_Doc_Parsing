"""Build section tree from PDF bookmarks + page header/footer codes."""

from __future__ import annotations

import re
from collections import defaultdict

import fitz

from marine_docs.models import SectionDraft

QUOTE_RE = re.compile(
    r"quote\s+(?:Procedure|Data|Plate|Maintenance Schedules)\s+"
    r"(?P<code>[A-Z]\d[\w.]*)\s+Edition\s+(?P<edition>[\w.]+)",
    re.IGNORECASE,
)

# M90101-0249 Cylinder Cover  |  D10101-0S48 Cylinder Cover-Data  |  P90151-0199 ...
TOC_CODE_RE = re.compile(
    r"^(?P<code>[A-Z]\d{4,6})(?:-(?P<edition>[\w.]+))?\s*(?P<title>.*)$"
)

KIND_MAP = {
    "D": "data",
    "M": "procedure",
    "P": "plate",
    "A": "schedule",
}


def extract_sections(pdf_path: str) -> list[SectionDraft]:
    doc = fitz.open(pdf_path)
    try:
        chapter_sections = _sections_from_toc(doc)
        page_codes = _scan_page_codes(doc)
        return _merge_sections(chapter_sections, page_codes, doc.page_count)
    finally:
        doc.close()


def _sections_from_toc(doc: fitz.Document) -> list[SectionDraft]:
    toc = doc.get_toc()
    chapters: list[SectionDraft] = []
    current_chapter: str | None = None

    for _level, title, page in toc:
        clean = _clean_title(title)
        if not clean:
            continue
        if _is_junk_bookmark(clean):
            chapter_name = _chapter_name_from_path(clean)
            if chapter_name:
                current_chapter = chapter_name
                chapters.append(
                    SectionDraft(
                        path=[chapter_name],
                        title=chapter_name,
                        page_start=max(1, page),
                        section_kind="other",
                    )
                )
            continue

        doc_code = None
        edition = None
        section_title = clean
        kind = "other"
        m = TOC_CODE_RE.match(clean)
        if m and m.group("code")[0] in KIND_MAP:
            doc_code = m.group("code")
            edition = m.group("edition")
            section_title = (m.group("title") or clean).strip(" -_\t") or clean
            kind = KIND_MAP[doc_code[0]]

        path = [current_chapter, section_title] if current_chapter else [section_title]
        chapters.append(
            SectionDraft(
                path=path,
                title=section_title,
                page_start=max(1, page),
                doc_code=doc_code,
                edition=edition,
                section_kind=kind,
                parent_path=[current_chapter] if current_chapter else None,
            )
        )
    return chapters


def _scan_page_codes(doc: fitz.Document) -> dict[int, dict]:
    found: dict[int, dict] = {}
    for i in range(doc.page_count):
        text = doc[i].get_text("text") or ""
        m = QUOTE_RE.search(text)
        if not m:
            continue
        code = m.group("code").strip()
        edition = m.group("edition").strip()
        kind = KIND_MAP.get(code[0], "other")
        found[i + 1] = {"doc_code": code, "edition": edition, "kind": kind}
    return found


def _merge_sections(
    toc_sections: list[SectionDraft],
    page_codes: dict[int, dict],
    page_count: int,
) -> list[SectionDraft]:
    by_code: dict[str, list[SectionDraft]] = defaultdict(list)
    for s in toc_sections:
        if s.doc_code:
            by_code[s.doc_code].append(s)

    for page, meta in page_codes.items():
        code = meta["doc_code"]
        if code in by_code:
            for s in by_code[code]:
                if s.edition is None:
                    s.edition = meta["edition"]
                if s.section_kind == "other":
                    s.section_kind = meta["kind"]
        else:
            chapter = _chapter_for_page(toc_sections, page)
            title = code
            path = [chapter, title] if chapter else [title]
            toc_sections.append(
                SectionDraft(
                    path=path,
                    title=title,
                    page_start=page,
                    page_end=page,
                    doc_code=code,
                    edition=meta["edition"],
                    section_kind=meta["kind"],
                    parent_path=[chapter] if chapter else None,
                )
            )

    ordered = sorted(
        [s for s in toc_sections if s.page_start is not None],
        key=lambda s: (s.page_start or 0, len(s.path)),
    )
    for i, s in enumerate(ordered):
        if s.page_end is not None:
            continue
        next_start = None
        for nxt in ordered[i + 1 :]:
            if (nxt.page_start or 0) > (s.page_start or 0):
                next_start = nxt.page_start
                break
        s.page_end = (next_start - 1) if next_start else page_count

    unique: dict[tuple[str, ...], SectionDraft] = {}
    for s in ordered:
        key = tuple(s.path)
        if key not in unique:
            unique[key] = s
            continue
        cur = unique[key]
        if not cur.edition and s.edition:
            cur.edition = s.edition
        if not cur.doc_code and s.doc_code:
            cur.doc_code = s.doc_code
        if cur.page_end is None or (s.page_end and s.page_end > cur.page_end):
            cur.page_end = s.page_end
    return list(unique.values())


def _chapter_for_page(sections: list[SectionDraft], page: int) -> str | None:
    chapter = None
    for s in sections:
        if len(s.path) == 1 and s.page_start is not None and s.page_start <= page:
            chapter = s.path[0]
    return chapter


def _clean_title(title: str) -> str:
    return title.replace("\u200e", "").replace("\r", " ").replace("\n", " ").strip()


def _is_junk_bookmark(title: str) -> bool:
    lower = title.lower()
    return (
        ".pdf" in lower
        or lower.startswith("d:\\")
        or lower.startswith("c:\\")
        or ("\\" in title and "MAN" in title)
    )


def _chapter_name_from_path(title: str) -> str | None:
    m = re.search(r"(\d{3})_vol2", title, re.IGNORECASE)
    if m:
        return f"Chapter {m.group(1)}"
    if "vol2" in title.lower():
        m = re.search(r"(\d{3})", title)
        if m:
            return f"Chapter {m.group(1)}"
    return None
