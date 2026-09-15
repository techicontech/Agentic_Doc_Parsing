"""Mistral OCR via LiteLLM (not the Mistral SDK directly)."""

from __future__ import annotations

import base64
import logging
import os
import time
from typing import Any, Callable

import fitz

from marine_docs.config import get_settings
from marine_docs.models import ElementType, ParsedElement, ParsedPage, Route

logger = logging.getLogger(__name__)

EXTRACTOR_NAME = "litellm_ocr"
EXTRACTOR_VERSION = "mistral-ocr"

ProgressCb = Callable[[str, dict[str, Any]], None]


def parse_pages_with_mistral(
    pdf_path: str,
    pages_1based: list[int],
    progress: ProgressCb | None = None,
) -> list[ParsedPage]:
    settings = get_settings()
    if not settings.ocr_ready:
        raise RuntimeError(
            "OCR is not configured. Set either:\n"
            "  - LITELLM_API_BASE + LITELLM_API_KEY (LiteLLM proxy), or\n"
            "  - MISTRAL_API_KEY (LiteLLM direct → Mistral OCR)"
        )

    try:
        from litellm import ocr
    except ImportError as exc:
        raise RuntimeError("litellm is not installed. Run: pip install -r requirements.txt") from exc

    _configure_litellm_env(settings)

    results: list[ParsedPage] = []
    total = len(pages_1based)
    for idx, page_no in enumerate(pages_1based, start=1):
        png = _render_page_png(pdf_path, page_no)
        data_url = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
        try:
            response = _ocr_with_retry(ocr, settings, data_url)
            parsed = _page_from_ocr_response(page_no, response, png)
            results.append(parsed)
        except Exception:
            logger.exception("LiteLLM OCR failed on page %s", page_no)
            results.append(
                ParsedPage(
                    page=page_no,
                    route=Route.MISTRAL,
                    page_image_png=png,
                    notes={"error": "litellm_ocr_failed"},
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


def _configure_litellm_env(settings) -> None:
    """Point LiteLLM at proxy and/or provider keys — app never calls providers directly."""
    if settings.litellm_api_base:
        os.environ["LITELLM_API_BASE"] = settings.litellm_api_base
    if settings.litellm_api_key:
        os.environ["LITELLM_API_KEY"] = settings.litellm_api_key
    if settings.mistral_api_key:
        os.environ["MISTRAL_API_KEY"] = settings.mistral_api_key
    if settings.anthropic_api_key:
        os.environ["ANTHROPIC_API_KEY"] = settings.anthropic_api_key


def _ocr_endpoint(settings) -> str:
    base = (settings.litellm_api_base or "").strip().rstrip("/")
    if not base:
        raise RuntimeError("LITELLM_API_BASE is required for OCR via proxy")
    if not base.endswith("/ocr"):
        # Accept .../v1 or host root
        base = f"{base}/ocr" if base.endswith("/v1") else f"{base}/v1/ocr"
    return base


def _ocr_with_retry(ocr_fn, settings, data_url: str, attempts: int = 3) -> Any:
    """Call OCR through LiteLLM proxy, preserving the exact model name.

    The litellm.ocr() SDK strips ``mistral/`` and sends ``mistral-ocr-latest``,
    which virtual keys that allow ``mistral/mistral-ocr-latest`` reject with 403.
    So we POST /v1/ocr ourselves when a proxy base is configured.
    """
    last_exc: Exception | None = None
    document = {"type": "image_url", "image_url": data_url}
    model = settings.mistral_ocr_model

    if settings.litellm_api_base and settings.litellm_api_key:
        import httpx

        url = _ocr_endpoint(settings)
        headers = {
            "Authorization": f"Bearer {settings.litellm_api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": model, "document": document}
        for i in range(attempts):
            try:
                resp = httpx.post(url, headers=headers, json=payload, timeout=180.0)
                if resp.status_code >= 400:
                    raise RuntimeError(f"OCR HTTP {resp.status_code}: {resp.text[:500]}")
                return _ocr_response_from_dict(resp.json())
            except Exception as exc:
                last_exc = exc
                logger.warning("OCR attempt %s/%s failed: %s", i + 1, attempts, exc)
                time.sleep(1.5 * (i + 1))
        assert last_exc is not None
        raise last_exc

    # Direct LiteLLM SDK path (no proxy)
    kwargs: dict[str, Any] = {"model": model, "document": document}
    if settings.mistral_api_key:
        kwargs["api_key"] = settings.mistral_api_key
    elif settings.litellm_api_key:
        kwargs["api_key"] = settings.litellm_api_key

    for i in range(attempts):
        try:
            return ocr_fn(**kwargs)
        except Exception as exc:
            last_exc = exc
            time.sleep(1.5 * (i + 1))
    assert last_exc is not None
    raise last_exc


def _ocr_response_from_dict(data: dict[str, Any]) -> Any:
    """Normalize proxy JSON into attribute-style objects for _page_from_ocr_response."""
    from types import SimpleNamespace

    pages = []
    for p in data.get("pages") or []:
        if isinstance(p, dict):
            blocks = p.get("blocks") or []
            norm_blocks = []
            for b in blocks:
                if isinstance(b, dict):
                    norm_blocks.append(SimpleNamespace(**b))
                else:
                    norm_blocks.append(b)
            pages.append(
                SimpleNamespace(
                    markdown=p.get("markdown"),
                    blocks=norm_blocks or None,
                    **{k: v for k, v in p.items() if k not in {"markdown", "blocks"}},
                )
            )
        else:
            pages.append(p)
    return SimpleNamespace(pages=pages, raw=data)


def _render_page_png(pdf_path: str, page_1based: int, zoom: float = 2.0) -> bytes:
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_1based - 1]
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        return pix.tobytes("png")
    finally:
        doc.close()


def _page_from_ocr_response(page_no: int, response: Any, png: bytes) -> ParsedPage:
    elements: list[ParsedElement] = []
    pages = getattr(response, "pages", None) or []
    page_obj = pages[0] if pages else None

    markdown = getattr(page_obj, "markdown", None) if page_obj else None
    blocks = getattr(page_obj, "blocks", None) if page_obj else None

    if blocks:
        for block in blocks:
            elements.append(_element_from_block(page_no, block))
    elif markdown and str(markdown).strip():
        elements.append(
            ParsedElement(
                type=ElementType.PARAGRAPH,
                page=page_no,
                text=str(markdown).strip(),
                extractor_name=EXTRACTOR_NAME,
                extractor_version=EXTRACTOR_VERSION,
            )
        )

    elements.append(
        ParsedElement(
            type=ElementType.FIGURE,
            page=page_no,
            figure_id=f"page-{page_no}",
            caption=_guess_caption(markdown, elements),
            extractor_name=EXTRACTOR_NAME,
            extractor_version=EXTRACTOR_VERSION,
        )
    )

    return ParsedPage(
        page=page_no,
        route=Route.MISTRAL,
        elements=elements,
        page_image_png=png,
        notes={"via": "litellm.ocr", "model": EXTRACTOR_VERSION},
    )


def _element_from_block(page_no: int, block: Any) -> ParsedElement:
    btype = str(getattr(block, "type", None) or getattr(block, "label", "") or "other").lower()
    text = getattr(block, "text", None) or getattr(block, "content", None) or getattr(block, "markdown", None)
    if text is not None and not isinstance(text, str):
        text = str(text)

    conf = getattr(block, "confidence", None)
    if conf is None:
        conf = getattr(block, "confidence_score", None)

    bbox = getattr(block, "bbox", None) or getattr(block, "bounding_box", None)
    bbox_list = None
    if isinstance(bbox, dict):
        bbox_list = [
            float(bbox.get("x1", bbox.get("l", 0))),
            float(bbox.get("y1", bbox.get("t", 0))),
            float(bbox.get("x2", bbox.get("r", 0))),
            float(bbox.get("y2", bbox.get("b", 0))),
        ]
    elif isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        bbox_list = [float(x) for x in bbox]

    if "table" in btype:
        etype = ElementType.TABLE
    elif "heading" in btype or "title" in btype:
        etype = ElementType.HEADING
    elif "figure" in btype or "image" in btype or "picture" in btype:
        etype = ElementType.FIGURE
    elif "list" in btype:
        etype = ElementType.LIST
    elif "caption" in btype:
        etype = ElementType.CAPTION
    else:
        etype = ElementType.PARAGRAPH

    return ParsedElement(
        type=etype,
        page=page_no,
        text=text.strip() if isinstance(text, str) and text.strip() else None,
        bbox=bbox_list,
        confidence=float(conf) if conf is not None else None,
        figure_id=f"page-{page_no}" if etype == ElementType.FIGURE else None,
        extractor_name=EXTRACTOR_NAME,
        extractor_version=EXTRACTOR_VERSION,
    )


def _guess_caption(markdown: str | None, elements: list[ParsedElement]) -> str | None:
    for el in elements:
        if el.type in (ElementType.HEADING, ElementType.CAPTION) and el.text:
            return el.text[:240]
    if markdown:
        line = str(markdown).strip().splitlines()[0].strip()
        return line[:240] if line else None
    return None
