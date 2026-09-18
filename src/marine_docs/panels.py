"""Panel-level figure extraction (spec Section 6).

Detect bordered/boxed sub-regions from layout (drawn boxes, whitespace, ruled
lines) — not from a manufacturer-specific numbering pattern. Never assume one
figure per page; never assume panels are numbered.

If a panel label and a step label share the same identifier, set
linked_step_number. If they do not, still store the panel with panel_index and
leave linked_step_number null rather than guessing a pairing.

This is a Tool, not an agent — no LLM is involved in finding panels.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import fitz

from marine_docs.pagemeta import normalize_text

logger = logging.getLogger(__name__)

EXTRACTOR_NAME = "panel_geometry"
EXTRACTOR_VERSION = "pymupdf-maximal-rect"

# A page frame covers most of the page; a panel does not.
MAX_PANEL_AREA_FRAC = 0.40
MIN_PANEL_AREA_FRAC = 0.02
MIN_PANEL_SIDE = 50.0
# A rect is "inside" a kept panel when most of its area overlaps it.
NESTED_OVERLAP_FRAC = 0.70
# Step labels sit at the panel's top-left, left of the panel's right edge.
STEP_LEFT_SLACK = 70.0
STEP_ABOVE_SLACK = 28.0
STEP_BELOW_SLACK = 45.0
# Drawing codes are printed vertically just outside the panel border.
CODE_SIDE_SLACK = 30.0
# More than this many "panels" is inner drawing detail, not real borders.
MAX_PANELS_PER_PAGE = 8

# Numeric or lettered step markers; pairing only uses a shared identifier.
STEP_RE = re.compile(r"^([0-9]{1,2}|[A-Za-z])[.)]$")
DRAWING_CODE_RE = re.compile(r"[A-Z]{1,3}\s?\d{3,6}[-\s]\d{1,4}[\w\s-]{0,8}$")


@dataclass
class PanelDraft:
    page: int
    panel_index: int
    bbox: list[float]
    drawing_code: str | None
    linked_step_number: int | None
    image_png: bytes | None = None


def split_is_messy(panels: list[PanelDraft]) -> bool:
    """True when geometry failed and the page should fall back to whole-page vision."""
    return not panels or len(panels) > MAX_PANELS_PER_PAGE


def extract_page_panels(
    pdf_path: str,
    pages_1based: list[int],
    *,
    zoom: float = 2.0,
) -> dict[int, list[PanelDraft]]:
    """Find and crop every diagram panel on the requested pages."""
    out: dict[int, list[PanelDraft]] = {}
    doc = fitz.open(pdf_path)
    try:
        for page_no in pages_1based:
            if not (1 <= page_no <= doc.page_count):
                continue
            try:
                out[page_no] = _panels_for_page(doc[page_no - 1], page_no, zoom=zoom)
            except Exception:
                logger.exception("Panel extraction failed on page %s", page_no)
                out[page_no] = []
    finally:
        doc.close()
    return out


def _panels_for_page(page: fitz.Page, page_no: int, *, zoom: float) -> list[PanelDraft]:
    boxes = _panel_boxes(page)
    if not boxes:
        return []

    steps = _step_labels(page)
    code_by_box = _assign_codes(boxes, _drawing_codes(page))
    matrix = fitz.Matrix(zoom, zoom)

    panels: list[PanelDraft] = []
    for index, box in enumerate(boxes, start=1):
        png: bytes | None = None
        try:
            pix = page.get_pixmap(matrix=matrix, clip=box, alpha=False)
            png = pix.tobytes("png")
        except Exception:
            logger.exception("Panel crop failed on page %s panel %s", page_no, index)
        panels.append(
            PanelDraft(
                page=page_no,
                panel_index=index,
                bbox=[round(v, 2) for v in (box.x0, box.y0, box.x1, box.y1)],
                drawing_code=code_by_box.get(index),
                linked_step_number=_step_for_box(box, steps),
                image_png=png,
            )
        )
    return panels


def _panel_boxes(page: fitz.Page) -> list[fitz.Rect]:
    """Maximal drawing rects: panel borders minus the page frame and inner detail."""
    page_area = page.rect.get_area()
    if page_area <= 0:
        return []

    candidates: list[fitz.Rect] = []
    for drawing in page.get_drawings():
        rect = drawing.get("rect")
        if rect is None or rect.is_empty:
            continue
        frac = rect.get_area() / page_area
        if frac > MAX_PANEL_AREA_FRAC or frac < MIN_PANEL_AREA_FRAC:
            continue
        if rect.width < MIN_PANEL_SIDE or rect.height < MIN_PANEL_SIDE:
            continue
        candidates.append(fitz.Rect(rect))

    kept: list[fitz.Rect] = []
    for rect in sorted(candidates, key=lambda r: -r.get_area()):
        if any(_mostly_inside(rect, bigger) for bigger in kept):
            continue
        kept.append(rect)

    kept.sort(key=lambda r: (round(r.y0 / 20), r.x0))
    return kept


def _mostly_inside(inner: fitz.Rect, outer: fitz.Rect) -> bool:
    overlap = fitz.Rect(inner) & outer
    if overlap.is_empty:
        return False
    area = inner.get_area()
    return area > 0 and overlap.get_area() / area >= NESTED_OVERLAP_FRAC


def _step_labels(page: fitz.Page) -> list[tuple[fitz.Rect, int]]:
    labels: list[tuple[fitz.Rect, int]] = []
    for rect, text, _vertical in _spans(page):
        match = STEP_RE.match(text)
        if match and match.group(1).isdigit():
            # Letter labels are detected so we do not assume numeric-only manuals,
            # but linked_step_number is INTEGER — store a pair only for digits.
            labels.append((rect, int(match.group(1))))
    return labels


def _drawing_codes(page: fitz.Page) -> list[tuple[fitz.Rect, str]]:
    """Codes printed vertically along a panel border, e.g. M90201-0285D06."""
    codes: list[tuple[fitz.Rect, str]] = []
    for rect, text, vertical in _spans(page, join_lines=True):
        if not vertical:
            continue
        if DRAWING_CODE_RE.match(text) or re.match(r"^[A-Z]{1,3}\s?\d{3,6}-", text):
            codes.append((rect, text))
    return codes


def _spans(page: fitz.Page, *, join_lines: bool = False):
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            direction = line.get("dir") or (1.0, 0.0)
            vertical = abs(direction[0]) < 0.5
            if join_lines:
                text = normalize_text("".join(s.get("text", "") for s in line.get("spans", []))).strip()
                if text:
                    yield fitz.Rect(line["bbox"]), text, vertical
                continue
            for span in line.get("spans", []):
                text = normalize_text(span.get("text")).strip()
                if text:
                    yield fitz.Rect(span["bbox"]), text, vertical


def _step_for_box(box: fitz.Rect, steps: list[tuple[fitz.Rect, int]]) -> int | None:
    """Step label nearest the panel's top-left corner, inside the panel's columns."""
    best: tuple[float, int] | None = None
    for rect, number in steps:
        if rect.x0 < box.x0 - STEP_LEFT_SLACK or rect.x0 > box.x1:
            continue
        if rect.y0 < box.y0 - STEP_ABOVE_SLACK or rect.y0 > box.y0 + STEP_BELOW_SLACK:
            continue
        distance = abs(rect.y0 - box.y0) + abs(rect.x0 - box.x0) * 0.1
        if best is None or distance < best[0]:
            best = (distance, number)
    return best[1] if best else None


def _assign_codes(
    boxes: list[fitz.Rect],
    codes: list[tuple[fitz.Rect, str]],
) -> dict[int, str]:
    """Give each code to one panel only — codes sit beside the panel they label.

    Neighbouring panels are a few points apart, so a plain nearest-neighbour match
    hands the same code to two panels. Preferring the panel that actually contains
    the code's centre keeps them distinct.
    """
    scored: list[tuple[int, float, int, str]] = []
    for code_index, (rect, text) in enumerate(codes):
        centre = fitz.Point((rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2)
        for panel_index, box in enumerate(boxes, start=1):
            padded = fitz.Rect(box) + (
                -CODE_SIDE_SLACK, -CODE_SIDE_SLACK, CODE_SIDE_SLACK, CODE_SIDE_SLACK
            )
            if not padded.contains(centre):
                continue
            tier = 0 if box.contains(centre) else 1
            distance = abs(centre.y - (box.y0 + box.y1) / 2) + abs(centre.x - box.x0)
            scored.append((tier, distance, panel_index, text))
            _ = code_index

    assigned: dict[int, str] = {}
    used_codes: set[str] = set()
    for _tier, _distance, panel_index, text in sorted(scored, key=lambda s: (s[0], s[1])):
        if panel_index in assigned or text in used_codes:
            continue
        assigned[panel_index] = text[:120]
        used_codes.add(text)
    return assigned
