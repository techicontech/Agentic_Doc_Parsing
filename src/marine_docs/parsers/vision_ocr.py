"""Vision OCR via Claude on LiteLLM proxy (no Mistral key needed)."""

from __future__ import annotations

import base64
import logging
import time
from typing import Any, Callable

import fitz

from marine_docs.config import get_settings
from marine_docs.llm import chat_completion
from marine_docs.models import ElementType, ParsedElement, ParsedPage, Route

logger = logging.getLogger(__name__)

EXTRACTOR_NAME = "litellm_vision_ocr"
EXTRACTOR_VERSION = "claude-vision"

ProgressCb = Callable[[str, dict[str, Any]], None]

_PROMPT = (
    "You are OCR for a marine technical manual page (diagrams, plates, tables). "
    "Extract ALL readable text exactly. Preserve labels, part numbers, codes, "
    "dimensions, and table structure in markdown. "
    "If a region is a drawing with callouts, list each callout number and its label. "
    "Do not invent text. Reply with markdown only."
)


def parse_pages_with_vision(
    pdf_path: str,
    pages_1based: list[int],
    progress: ProgressCb | None = None,
) -> list[ParsedPage]:
    settings = get_settings()
    if not (settings.litellm_api_base and settings.litellm_api_key):
        raise RuntimeError("Set LITELLM_API_BASE + LITELLM_API_KEY for Claude vision OCR.")

    model = settings.ocr_vision_model or settings.llm_model
    results: list[ParsedPage] = []
    total = len(pages_1based)

    for idx, page_no in enumerate(pages_1based, start=1):
        png = _render_page_png(pdf_path, page_no)
        data_url = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
        try:
            text = _vision_ocr_with_retry(model, data_url)
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
                    notes={"via": "claude_vision", "model": model},
                )
            )
        except Exception:
            logger.exception("Claude vision OCR failed on page %s", page_no)
            results.append(
                ParsedPage(
                    page=page_no,
                    route=Route.MISTRAL,
                    page_image_png=png,
                    notes={"error": "claude_vision_ocr_failed", "model": model},
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


def _vision_ocr_with_retry(model: str, data_url: str, attempts: int = 3) -> str:
    last_exc: Exception | None = None
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": _PROMPT},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
    ]
    for i in range(attempts):
        try:
            return chat_completion(messages, model=model, temperature=0.0, max_tokens=2500)
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
        line = line.strip().lstrip("#").strip()
        if line:
            return line[:240]
    return None
