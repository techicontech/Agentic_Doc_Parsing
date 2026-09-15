"""Docling parser for pages with a usable PDF text layer."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Callable

import fitz

from marine_docs.config import get_settings
from marine_docs.device import configure_docling_accelerator
from marine_docs.models import ElementType, ParsedElement, ParsedPage, Route

logger = logging.getLogger(__name__)

DOCLING_VERSION = "docling"

ProgressCb = Callable[[str, dict[str, Any]], None]


def parse_pages_with_docling(
    pdf_path: str,
    pages_1based: list[int],
    progress: ProgressCb | None = None,
) -> list[ParsedPage]:
    """Parse selected pages via Docling with OCR disabled (text-layer only)."""
    if not pages_1based:
        return []

    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError as exc:
        raise RuntimeError(
            "docling is not installed. Run: pip install -r requirements.txt"
        ) from exc

    settings = get_settings()
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True
    pipeline_options.generate_page_images = False
    # Prefer embedded text layer for marine manuals that already have OCR text.
    if hasattr(pipeline_options, "force_backend_text"):
        pipeline_options.force_backend_text = True

    device = configure_docling_accelerator(pipeline_options, settings.docling_device)
    logger.info("Docling using device=%s batch_size=%s", device, settings.docling_batch_size)

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    results: list[ParsedPage] = []
    batch_size = max(1, settings.docling_batch_size)
    total = len(pages_1based)
    done = 0
    for start in range(0, total, batch_size):
        batch = pages_1based[start : start + batch_size]
        batch_pdf = _extract_page_subset(pdf_path, batch)
        try:
            conv = converter.convert(str(batch_pdf))
            exported = conv.document.export_to_dict()
            mapped = _elements_from_docling(exported, batch)
            results.extend(mapped)
        except Exception:
            logger.exception("Docling failed on pages %s; falling back to PyMuPDF text", batch)
            results.extend(_pymupdf_fallback(pdf_path, batch))
        finally:
            batch_pdf.unlink(missing_ok=True)
        done = min(total, start + len(batch))
        if progress:
            progress(
                "docling",
                {"done": done, "total": total, "batch": batch, "device": device},
            )
    return results


def _extract_page_subset(pdf_path: str, pages_1based: list[int]) -> Path:
    import os

    src = fitz.open(pdf_path)
    dst = fitz.open()
    try:
        for p in pages_1based:
            dst.insert_pdf(src, from_page=p - 1, to_page=p - 1)
        fd, name = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)  # Windows cannot overwrite an open handle
        tmp = Path(name)
        dst.save(tmp)
        return tmp
    finally:
        src.close()
        dst.close()


def _elements_from_docling(exported: dict, pages_1based: list[int]) -> list[ParsedPage]:
    """Map Docling export into per-page ParsedPage objects.

    Docling page numbers in a subset PDF restart at 1; remap to original pages.
    """
    by_page: dict[int, ParsedPage] = {
        p: ParsedPage(page=p, route=Route.DOCLING) for p in pages_1based
    }

    texts = exported.get("texts") or []
    tables = exported.get("tables") or []
    pictures = exported.get("pictures") or []

    def remap_page(local_page: int | None) -> int | None:
        if local_page is None:
            return None
        idx = int(local_page) - 1
        if 0 <= idx < len(pages_1based):
            return pages_1based[idx]
        return None

    for item in texts:
        page = _page_from_prov(item, remap_page)
        if page is None or page not in by_page:
            continue
        label = (item.get("label") or item.get("type") or "").lower()
        text = item.get("text") or item.get("orig") or ""
        if not text.strip():
            continue
        etype = ElementType.HEADING if "title" in label or "section" in label else ElementType.PARAGRAPH
        if "list" in label:
            etype = ElementType.LIST
        if "caption" in label:
            etype = ElementType.CAPTION
        by_page[page].elements.append(
            ParsedElement(
                type=etype,
                page=page,
                text=text.strip(),
                bbox=_bbox_from_prov(item),
                extractor_name="docling",
                extractor_version=DOCLING_VERSION,
            )
        )

    for item in tables:
        page = _page_from_prov(item, remap_page)
        if page is None or page not in by_page:
            continue
        table_json = item.get("data") or item.get("export") or item
        text = item.get("text") or ""
        if not text.strip():
            text = _flatten_table_text(table_json)
        by_page[page].elements.append(
            ParsedElement(
                type=ElementType.TABLE,
                page=page,
                text=text.strip() or None,
                table_json=table_json if isinstance(table_json, (dict, list)) else {"raw": table_json},
                bbox=_bbox_from_prov(item),
                extractor_name="docling",
                extractor_version=DOCLING_VERSION,
            )
        )

    for item in pictures:
        page = _page_from_prov(item, remap_page)
        if page is None or page not in by_page:
            continue
        caption = item.get("text") or item.get("caption") or None
        by_page[page].elements.append(
            ParsedElement(
                type=ElementType.FIGURE,
                page=page,
                text=caption,
                caption=caption,
                figure_id=f"docling-fig-p{page}",
                bbox=_bbox_from_prov(item),
                extractor_name="docling",
                extractor_version=DOCLING_VERSION,
            )
        )

    return [by_page[p] for p in pages_1based]


def _page_from_prov(item: dict, remap) -> int | None:
    prov = item.get("prov") or []
    if prov:
        return remap(prov[0].get("page_no") or prov[0].get("page"))
    return remap(item.get("page_no") or item.get("page"))


def _bbox_from_prov(item: dict) -> list[float] | None:
    prov = item.get("prov") or []
    if not prov:
        return None
    bbox = prov[0].get("bbox")
    if not bbox:
        return None
    if isinstance(bbox, dict):
        return [
            float(bbox.get("l", bbox.get("x0", 0))),
            float(bbox.get("t", bbox.get("y0", 0))),
            float(bbox.get("r", bbox.get("x1", 0))),
            float(bbox.get("b", bbox.get("y1", 0))),
        ]
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        return [float(x) for x in bbox]
    return None


def _flatten_table_text(table_json) -> str:
    """Pull cell strings out of Docling table structures for FTS / ILIKE."""
    cells: list[str] = []

    def walk(node):
        if isinstance(node, dict):
            if "text" in node and isinstance(node["text"], str) and node["text"].strip():
                cells.append(node["text"].strip())
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(table_json)
    # Preserve order, drop exact consecutive dupes
    out: list[str] = []
    for c in cells:
        if not out or out[-1] != c:
            out.append(c)
    return " | ".join(out)


def _pymupdf_fallback(pdf_path: str, pages_1based: list[int]) -> list[ParsedPage]:
    doc = fitz.open(pdf_path)
    out: list[ParsedPage] = []
    try:
        for p in pages_1based:
            page = doc[p - 1]
            text = (page.get_text("text") or "").strip()
            elements: list[ParsedElement] = []
            if text:
                elements.append(
                    ParsedElement(
                        type=ElementType.PARAGRAPH,
                        page=p,
                        text=text,
                        extractor_name="pymupdf_fallback",
                        extractor_version=fitz.version[0],
                    )
                )
            out.append(ParsedPage(page=p, route=Route.DOCLING, elements=elements, notes={"fallback": True}))
    finally:
        doc.close()
    return out
