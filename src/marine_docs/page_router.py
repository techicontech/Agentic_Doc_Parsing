"""
Agent: Page Router (ingest-time, ambiguous pages only)

Input: rendered page image is optional; the live path sends a text-density
  heuristic (character count, drawing count, image count) plus a text-layer sample.
Output: PageClassification(route: Literal["docling","vision_ocr","skip"] via Route,
  confidence is implicit in `reason`). Route.MISTRAL is the vision-OCR bucket.
Model: configured via marine_docs.llm / settings.llm_model (LiteLLM). Overridable.
  Default is the same chat model as other agents (typically claude-haiku).
Failure mode: on LLM error, unparseable JSON, or low-confidence/no-decision,
  default to vision OCR (Route.MISTRAL). Safer to over-send to the more
  expensive/accurate path than to under-extract silently.

Heuristic first. The LLM agent is consulted only where the heuristic is genuinely
ambiguous — a page with both substantial prose and dense drawings. Clear pages
never reach the model. No manufacturer-specific layout rules.
"""

from __future__ import annotations

import logging

import fitz

from marine_docs.config import get_settings
from marine_docs.models import PageClassification, Route
from marine_docs.pagemeta import normalize_text

logger = logging.getLogger(__name__)

# Ambiguity band: enough text to parse, enough drawings to matter.
AMBIGUOUS_MIN_CHARS = 300
AMBIGUOUS_MAX_CHARS = 1200
AMBIGUOUS_MIN_DRAWINGS = 150
# A broken text layer looks like text but reads as control bytes / mojibake.
GARBLED_RATIO = 0.25


def classify_pages(pdf_path: str, *, allow_llm: bool = True) -> list[PageClassification]:
    """Route image-dominant / diagram pages to OCR; text-layer pages to Docling.

    Layout heuristics (manufacturer-agnostic):
    - Sparse text + dense vector drawings -> vision OCR.
    - Empty raster pages (0 chars, embedded images) need OCR.
    - Blank separator pages are skipped.
    """
    settings = get_settings()
    use_llm = allow_llm and settings.page_router_llm_enabled

    doc = fitz.open(pdf_path)
    results: list[PageClassification] = []
    escalated = 0
    try:
        for i in range(doc.page_count):
            page = doc[i]
            text = page.get_text("text") or ""
            char_count = len(text.strip())
            image_count = len(page.get_images(full=True))
            drawing_count = len(page.get_drawings())

            route, reason = _decide(char_count, image_count, drawing_count, text)
            if reason == "ambiguous_mixed_content":
                if use_llm and escalated < settings.page_router_llm_max_pages:
                    route, reason = _ask_router_agent(i + 1, text, drawing_count, image_count)
                    escalated += 1
                else:
                    route, reason = Route.MISTRAL, "ambiguous_default_vision"

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
    if escalated:
        logger.info("Page router escalated %s ambiguous pages to the LLM agent", escalated)
    return results


def _decide(
    char_count: int,
    image_count: int,
    drawing_count: int,
    text: str = "",
) -> tuple[Route, str]:
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

    # A text layer that decodes to control bytes is not usable prose.
    if char_count and _garbled_ratio(text) > GARBLED_RATIO:
        return Route.MISTRAL, "garbled_text_layer"

    # Substantial prose *and* dense drawings — the heuristic cannot decide.
    if (
        AMBIGUOUS_MIN_CHARS <= char_count <= AMBIGUOUS_MAX_CHARS
        and drawing_count >= AMBIGUOUS_MIN_DRAWINGS
    ):
        return Route.DOCLING, "ambiguous_mixed_content"

    return Route.DOCLING, "clean_text_layer"


def _garbled_ratio(text: str) -> float:
    sample = text[:4000]
    if not sample:
        return 0.0
    replaced = normalize_text(sample)
    control = sum(1 for a, b in zip(sample, replaced) if a != b)
    return control / len(sample)


def _ask_router_agent(
    page_no: int,
    text: str,
    drawing_count: int,
    image_count: int,
) -> tuple[Route, str]:
    """Agent (LLM): judgment call on a mixed text+diagram page."""
    from marine_docs.agents import llm_agents
    from marine_docs.agents.jsonutil import parse_json_object

    sample = normalize_text(text)[:1500]
    try:
        parsed = parse_json_object(
            llm_agents.ask(
                llm_agents.PAGE_ROUTER,
                f"Page {page_no}: {len(text.strip())} characters of text layer, "
                f"{drawing_count} vector drawings, {image_count} raster images.\n\n"
                f"Text layer sample:\n{sample}",
                max_tokens=150,
            )
        )
        route = str(parsed.get("route") or "").strip().lower()
        reason = str(parsed.get("reason") or "llm_router")[:120]
        if route == "vision":
            return Route.MISTRAL, f"llm:{reason}"
        if route == "docling":
            return Route.DOCLING, f"llm:{reason}"
    except Exception:
        logger.exception("Page router agent failed on page %s; defaulting to vision OCR", page_no)
    return Route.MISTRAL, "ambiguous_default_vision"
