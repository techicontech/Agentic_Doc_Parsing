"""Classify PDF pages into Docling vs Mistral OCR vs skip."""

from __future__ import annotations

import fitz

from marine_docs.models import PageClassification, Route


def classify_pages(pdf_path: str) -> list[PageClassification]:
    """Route image-dominant / plate-like pages to Mistral; text-layer pages to Docling.

    Heuristics tuned on MAN B&W Vol II:
    - P-plates often have header OCR (~100-200 chars) but thousands of vector drawings
      and little usable prose.
    - Empty raster pages (0 chars, embedded images) need OCR.
    - Blank separator pages are skipped.
    """
    doc = fitz.open(pdf_path)
    results: list[PageClassification] = []
    try:
        for i in range(doc.page_count):
            page = doc[i]
            text = page.get_text("text") or ""
            char_count = len(text.strip())
            image_count = len(page.get_images(full=True))
            drawing_count = len(page.get_drawings())

            route, reason = _decide(char_count, image_count, drawing_count)
            results.append(
                PageClassification(
                    page=i + 1,
                    route=route,
                    char_count=char_count,
                    image_count=image_count,
                    drawing_count=drawing_count,
                    reason=reason,
                )
            )
    finally:
        doc.close()
    return results


def _decide(char_count: int, image_count: int, drawing_count: int) -> tuple[Route, str]:
    if char_count == 0 and image_count == 0 and drawing_count < 5:
        return Route.SKIP, "blank_page"

    # Raster-only / scan-like pages with no usable text layer
    if char_count == 0 and image_count >= 1:
        return Route.MISTRAL, "image_only_no_text_layer"

    # Plate / diagram pages: sparse text + dense vector drawings
    if drawing_count >= 200 and char_count < 400:
        return Route.MISTRAL, "drawing_dense_sparse_text"

    # Sparse text with embedded images (diagram panels)
    if image_count >= 3 and char_count < 300:
        return Route.MISTRAL, "multi_image_sparse_text"

    # Very little text overall — safer to OCR for structure/bbox fidelity
    if char_count < 80 and (image_count >= 1 or drawing_count >= 20):
        return Route.MISTRAL, "low_text_visual_content"

    return Route.DOCLING, "clean_text_layer"
