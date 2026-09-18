"""
Agent: Diagram/plate OCR + captioning (ingest-time vision)

Input: PNG bytes of a whole page or one cropped panel, plus a text prompt.
Output: markdown text (and parsed DRAWING_CODE / STEP lines for panels).
  Downstream persist turns this into ParsedPage / ParsedElement records.
Model: settings.ocr_vision_model via LiteLLM (default claude-haiku). Overridable.
Failure mode: on LLM error the caller retries; if still failing the page is
  stored as an image-only figure rather than inventing OCR text.

Panel detection itself is geometric (marine_docs.panels) — this agent interprets
the cropped image. Layout boxes, not numbering format, decide what a panel is.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Callable

import fitz

from marine_docs.agents import llm_agents
from marine_docs.config import get_settings
from marine_docs.models import ElementType, ParsedElement, ParsedPage, Route

logger = logging.getLogger(__name__)

EXTRACTOR_NAME = "vision_ocr_agent"
EXTRACTOR_VERSION = "adk-litellm-vision"

ProgressCb = Callable[[str, dict[str, Any]], None]

_PAGE_PROMPT = "Extract every readable element of this manual page."

_PANEL_PROMPT = """This image is ONE diagram panel cropped from a marine engine manual.
Extract:
1. The drawing code printed on the panel border (example: M90201-0285D06).
2. The procedure step number this panel illustrates, if visible.
3. Every callout number and its label, plus any dimensions.

Start with exactly these two lines when you can read them:
DRAWING_CODE: <code or UNKNOWN>
STEP: <integer or UNKNOWN>

Then the rest as markdown. Never invent text."""

_DRAWING_LINE = re.compile(r"^DRAWING_CODE:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_STEP_LINE = re.compile(r"^STEP:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_DRAWING_IN_TEXT = re.compile(r"\b([A-Z]{1,3}\d{3,6}-\d{3,5}[A-Z]?\d{0,3})\b")


@dataclass
class PanelOcr:
    text: str
    drawing_code: str | None
    linked_step_number: int | None
    caption: str | None


def parse_pages_with_vision(
    pdf_path: str,
    pages_1based: list[int],
    progress: ProgressCb | None = None,
) -> list[ParsedPage]:
    """Whole-page vision OCR — fallback when panel geometry cannot split the page."""
    settings = get_settings()
    if not (settings.litellm_api_base and settings.litellm_api_key):
        raise RuntimeError("Set LITELLM_API_BASE + LITELLM_API_KEY for Claude vision OCR.")

    model = settings.ocr_vision_model or settings.llm_model
    results: list[ParsedPage] = []
    total = len(pages_1based)

    for idx, page_no in enumerate(pages_1based, start=1):
        png = _render_page_png(pdf_path, page_no)
        try:
            text = _vision_ocr_with_retry(png, _PAGE_PROMPT)
            elements: list[ParsedElement] = []
            if text.strip():
                elements.append(
                    ParsedElement(
                        type=ElementType.PARAGRAPH,
                        page=page_no,
                        text=text.strip(),
                        extractor_name=EXTRACTOR_NAME,
                        extractor_version=EXTRACTOR_VERSION,
                    )
                )
            elements.append(
                ParsedElement(
                    type=ElementType.FIGURE,
                    page=page_no,
                    figure_id=f"page-{page_no}",
                    caption=_first_line(text) or f"Page {page_no}",
                    extractor_name=EXTRACTOR_NAME,
                    extractor_version=EXTRACTOR_VERSION,
                )
            )
            results.append(
                ParsedPage(
                    page=page_no,
                    route=Route.MISTRAL,
                    elements=elements,
                    page_image_png=png,
                    notes={"via": "vision_ocr_agent", "model": model, "mode": "full_page"},
                )
            )
        except Exception:
            logger.exception("Vision OCR agent failed on page %s", page_no)
            results.append(
                ParsedPage(
                    page=page_no,
                    route=Route.MISTRAL,
                    page_image_png=png,
                    notes={"error": "vision_ocr_agent_failed", "model": model, "mode": "full_page"},
                    elements=[
                        ParsedElement(
                            type=ElementType.FIGURE,
                            page=page_no,
                            figure_id=f"page-{page_no}",
                            caption=f"Page {page_no} figure (OCR failed)",
                            extractor_name=EXTRACTOR_NAME,
                            extractor_version=EXTRACTOR_VERSION,
                        )
                    ],
                )
            )
        if progress:
            progress("mistral", {"done": idx, "total": total, "page": page_no})
    return results


def ocr_panel_png(png: bytes) -> PanelOcr:
    """Vision OCR one geometrically cropped panel."""
    text = _vision_ocr_with_retry(png, _PANEL_PROMPT)
    return parse_panel_ocr(text)


def parse_panel_ocr(text: str) -> PanelOcr:
    drawing: str | None = None
    step: int | None = None
    match = _DRAWING_LINE.search(text or "")
    if match:
        value = match.group(1).strip()
        if value.upper() not in {"UNKNOWN", "NONE", "N/A", "-", ""}:
            drawing = value[:120]
    match = _STEP_LINE.search(text or "")
    if match:
        digits = re.search(r"\d+", match.group(1))
        if digits:
            step = int(digits.group(0))
    if not drawing:
        found = _DRAWING_IN_TEXT.search(text or "")
        if found:
            drawing = found.group(1)
    return PanelOcr(
        text=(text or "").strip(),
        drawing_code=drawing,
        linked_step_number=step,
        caption=_first_line(text),
    )


def _vision_ocr_with_retry(png: bytes, prompt: str, attempts: int = 3) -> str:
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            return llm_agents.ask(
                llm_agents.VISION_OCR,
                prompt,
                image_png=png,
                max_tokens=2500,
            )
        except Exception as exc:
            last_exc = exc
            logger.warning("vision OCR attempt %s/%s failed: %s", i + 1, attempts, exc)
            time.sleep(1.5 * (i + 1))
    assert last_exc is not None
    raise last_exc


def _render_page_png(pdf_path: str, page_1based: int, zoom: float = 1.5) -> bytes:
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_1based - 1]
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        return pix.tobytes("png")
    finally:
        doc.close()


def _first_line(text: str) -> str | None:
    for line in (text or "").splitlines():
        stripped = line.strip().lstrip("#").strip()
        if not stripped:
            continue
        if stripped.upper().startswith(("DRAWING_CODE:", "STEP:")):
            continue
        return stripped[:240]
    return None
