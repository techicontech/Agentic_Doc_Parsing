"""Milestone 1 ingest: hybrid Docling + Mistral OCR → Postgres + MinIO."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import fitz
from rich.console import Console
from rich.table import Table

from marine_docs.config import get_settings
from marine_docs.knowledge import builder
from marine_docs.log_setup import progress_log
from marine_docs.models import ElementType, ParsedElement, ParsedPage, Route
from marine_docs.page_router import classify_pages
from marine_docs.pagemeta import PageMeta, normalize_text, scan_pdf
from marine_docs.panels import EXTRACTOR_NAME as PANEL_EXTRACTOR
from marine_docs.panels import EXTRACTOR_VERSION as PANEL_EXTRACTOR_VERSION
from marine_docs.panels import extract_page_panels, split_is_messy
from marine_docs.parsers.docling_parser import parse_pages_with_docling
from marine_docs.parsers.mistral_ocr import parse_pages_with_mistral
from marine_docs.parsers.vision_ocr import ocr_panel_png, parse_pages_with_vision
from marine_docs.structure import extract_sections

console = Console()
logger = logging.getLogger("ingest")

ProgressCb = Callable[[str, dict[str, Any]], None]

# Overall progress budget (weights sum to 100)
_W_CLASSIFY = 5
_W_SECTIONS = 5
_W_DOCLING = 55
_W_MISTRAL = 20
_W_PANELS = 5
_W_PERSIST = 10


def run_ingest(
    *,
    pdf: Path | None = None,
    limit_pages: int | None = None,
    pages: str | None = None,
    skip_mistral: bool = False,
    skip_docling: bool = False,
    dry_run: bool = False,
    progress: ProgressCb | None = None,
) -> dict[str, Any]:
    """Programmatic ingest used by CLI and UI API."""

    def emit(stage: str, percent: int, message: str, **extra: Any) -> None:
        percent = max(0, min(100, int(percent)))
        payload = {"stage": stage, "percent": percent, "message": message, **extra}
        progress_log(f"[{percent:3d}%] {message}")
        logger.info("ingest stage=%s percent=%s %s", stage, percent, extra)
        if progress:
            progress(stage, payload)

    settings = get_settings()
    pdf_path = Path(pdf).resolve() if pdf else settings.resolved_pdf_path
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    emit("opening_pdf", 0, f"Opening PDF: {pdf_path.name}", pdf=str(pdf_path))
    doc = fitz.open(pdf_path)
    page_count = doc.page_count
    doc.close()

    emit("classifying", 1, f"Classifying {page_count} pages…", page_count=page_count)
    from marine_docs.convention import detect_convention

    convention = detect_convention(str(pdf_path))
    emit(
        "convention",
        2,
        f"Citation convention: {convention.get('source')} "
        f"pattern={convention.get('pattern')}",
        convention=convention,
    )
    classifications = classify_pages(str(pdf_path))
    selected = _select_pages(classifications, page_count, limit_pages, pages)
    route_counts = dict(Counter(c.route.value for c in selected))
    emit(
        "classified",
        _W_CLASSIFY,
        f"Routed: Docling={route_counts.get('docling', 0)} "
        f"Mistral={route_counts.get('mistral', 0)} "
        f"Skip={route_counts.get('skip', 0)}",
        routes=route_counts,
        selected_pages=len(selected),
    )

    docling_pages = [c.page for c in selected if c.route == Route.DOCLING]
    mistral_pages = [c.page for c in selected if c.route == Route.MISTRAL]
    skipped_pages = [c.page for c in selected if c.route == Route.SKIP]

    if skip_docling:
        docling_pages = []
    if skip_mistral:
        mistral_pages = []

    # Reallocate unused stage weight so % still ends cleanly after skip stages
    w_docling = _W_DOCLING if docling_pages and settings.docling_enabled else 0
    w_mistral = _W_MISTRAL if mistral_pages else 0

    if mistral_pages and not settings.ocr_ready:
        raise RuntimeError(
            "OCR is not configured. Set LITELLM_API_BASE + LITELLM_API_KEY or MISTRAL_API_KEY."
        )

    base = _W_CLASSIFY
    emit("extracting_sections", base + 1, "Extracting section tree…")
    sections = extract_sections(str(pdf_path), convention=convention)
    page_metas = scan_pdf(str(pdf_path), convention=convention)
    base = _W_CLASSIFY + _W_SECTIONS
    emit(
        "sections_ready",
        base,
        f"Sections ready ({len(sections)}); "
        f"{sum(1 for m in page_metas.values() if m.procedure_no)} pages carry a procedure no.",
        section_count=len(sections),
    )

    parsed_pages = []
    if docling_pages and settings.docling_enabled and not skip_docling:

        def on_docling(_stage: str, payload: dict[str, Any]) -> None:
            done = int(payload.get("done") or 0)
            total = max(1, int(payload.get("total") or 1))
            frac = done / total
            pct = base + int(w_docling * frac)
            emit(
                "docling",
                pct,
                f"Docling {done}/{total} pages",
                done=done,
                total=total,
                device=payload.get("device"),
            )

        emit("docling_start", base, f"Docling starting ({len(docling_pages)} pages)…")
        parsed_pages.extend(
            parse_pages_with_docling(str(pdf_path), docling_pages, progress=on_docling)
        )
        base = _W_CLASSIFY + _W_SECTIONS + w_docling
        emit("docling_done", base, f"Docling done ({len(docling_pages)} pages)")
    else:
        base = _W_CLASSIFY + _W_SECTIONS + w_docling

    if mistral_pages:
        backend = (settings.ocr_backend or "claude").strip().lower()

        def on_ocr(_stage: str, payload: dict[str, Any]) -> None:
            done = int(payload.get("done") or 0)
            total = max(1, int(payload.get("total") or 1))
            frac = done / total
            pct = base + int(w_mistral * frac)
            emit(
                "mistral",
                pct,
                payload.get("message") or f"Diagram OCR {done}/{total} pages",
                done=done,
                total=total,
                page=payload.get("page"),
                backend=backend,
                mode=payload.get("mode"),
            )

        emit(
            "mistral_ocr_start",
            base,
            f"Diagram pages: geometry split then vision per crop "
            f"({len(mistral_pages)} pages, backend={backend})…",
        )
        parsed_pages.extend(
            _parse_diagram_pages(
                str(pdf_path),
                mistral_pages,
                page_metas,
                backend=backend,
                progress=on_ocr,
            )
        )
        base = _W_CLASSIFY + _W_SECTIONS + w_docling + w_mistral
        emit(
            "mistral_ocr_done",
            base,
            f"Diagram OCR done ({len(mistral_pages)} pages, backend={backend})",
        )
    else:
        base = _W_CLASSIFY + _W_SECTIONS + w_docling + w_mistral

    emit("panels_start", base, "Extracting diagram panels…")
    panel_count = _attach_panels(
        str(pdf_path),
        parsed_pages,
        page_metas,
        progress=lambda done, total: emit(
            "panels",
            base + int(_W_PANELS * (done / max(1, total))),
            f"Panels {done}/{total} pages",
            done=done,
            total=total,
        ),
    )
    base += _W_PANELS
    emit("panels_done", base, f"Extracted {panel_count} diagram panels", panels=panel_count)

    if dry_run:
        emit("completed", 100, "Dry run complete")
        return {
            "status": "dry_run",
            "parsed_pages": len(parsed_pages),
            "elements": sum(len(p.elements) for p in parsed_pages),
            "routes": route_counts,
        }

    emit("persisting", base + 1, "Saving to Postgres + MinIO…")
    run_id = builder.start_ingest_run(str(pdf_path), page_count)
    try:
        identity = _infer_manual_identity(pdf_path)
        manual_id, revision_id = builder.create_manual(
            title=identity["title"],
            equipment_type=identity["equipment_type"],
            vessel_class=None,
            revision=identity["revision"],
            source_pdf_path=str(pdf_path),
            manufacturer=identity.get("manufacturer"),
            citation_convention=dict(convention),
        )
        fleet_id = builder.seed_fleet_registry(
            manual_id,
            equipment_id=identity["equipment_id"],
            manufacturer=identity["manufacturer"],
            model=identity["model"],
        )
        path_to_id = builder.insert_sections(manual_id, sections)
        counts = builder.persist_parsed_pages(
            manual_id=manual_id,
            revision_id=revision_id,
            source_pdf_path=str(pdf_path),
            sections=sections,
            path_to_id=path_to_id,
            pages=parsed_pages,
            page_metas=page_metas,
        )
        counts["plate_pages"] = builder.ensure_plate_page_elements(
            manual_id=manual_id,
            source_pdf_path=str(pdf_path),
            sections=sections,
            path_to_id=path_to_id,
            page_metas=page_metas,
        )
        counts["panels"] = panel_count
        counts["structural_relationships"] = builder.rebuild_structural_relationships(manual_id)
        counts["cross_ref_relationships"] = builder.rebuild_cross_refs(manual_id)
        counts["step_links"] = builder.rebuild_step_links(manual_id)
        from marine_docs.editions import reset_cache

        reset_cache()

        builder.finish_ingest_run(
            run_id,
            manual_id=manual_id,
            status="completed",
            docling_pages=len(docling_pages),
            mistral_pages=len(mistral_pages),
            skipped_pages=len(skipped_pages),
            notes={"counts": counts, "fleet_registry_id": str(fleet_id)},
        )
    except Exception:
        logger.exception("Ingest failed")
        builder.finish_ingest_run(
            run_id,
            manual_id=None,
            status="failed",
            docling_pages=len(docling_pages),
            mistral_pages=len(mistral_pages),
            skipped_pages=len(skipped_pages),
            notes={"error": "see logs"},
        )
        raise

    result = {
        "status": "completed",
        "manual_id": str(manual_id),
        "run_id": str(run_id),
        "counts": counts,
        "routes": route_counts,
        "docling_pages": len(docling_pages),
        "mistral_pages": len(mistral_pages),
        "skipped_pages": len(skipped_pages),
    }
    emit("completed", 100, "Ingest complete — you can chat now", **result)
    return result


def main(argv: list[str] | None = None) -> int:
    from marine_docs.log_setup import setup_logging

    log_path = setup_logging()
    parser = argparse.ArgumentParser(description="Ingest marine manual into Postgres knowledge model")
    parser.add_argument("--pdf", type=Path, default=None, help="Path to PDF (default from .env)")
    parser.add_argument("--limit-pages", type=int, default=None, help="Only first N pages (smoke test)")
    parser.add_argument("--pages", type=str, default=None, help="Comma list / ranges, e.g. 1-20,405,557")
    parser.add_argument("--classify-only", action="store_true", help="Print page routing and exit")
    parser.add_argument("--skip-mistral", action="store_true", help="Skip Mistral OCR (dev only)")
    parser.add_argument("--skip-docling", action="store_true", help="Skip Docling (dev only)")
    parser.add_argument("--dry-run", action="store_true", help="Parse but do not write DB/MinIO")
    args = parser.parse_args(argv)

    settings = get_settings()
    pdf_path = Path(args.pdf).resolve() if args.pdf else settings.resolved_pdf_path
    if not pdf_path.exists():
        console.print(f"[red]PDF not found:[/red] {pdf_path}")
        return 1

    progress_log(f"Full logs -> {log_path}")

    if args.classify_only:
        classifications = classify_pages(str(pdf_path))
        _print_route_summary(classifications)
        out = settings.artifacts_dir / "page_classification.json"
        settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps([c.model_dump() for c in classifications], indent=2),
            encoding="utf-8",
        )
        console.print(f"Wrote {out}")
        return 0

    try:
        result = run_ingest(
            pdf=pdf_path,
            limit_pages=args.limit_pages,
            pages=args.pages,
            skip_mistral=args.skip_mistral,
            skip_docling=args.skip_docling,
            dry_run=args.dry_run,
        )
    except Exception as exc:
        console.print(f"[red]Ingest failed:[/red] {exc}")
        progress_log(f"See full traceback in {log_path}")
        return 1

    console.print("[bold green]Ingest complete[/bold green]")
    console.print(result)
    return 0


def _parse_diagram_pages(
    pdf_path: str,
    pages: list[int],
    page_metas: dict[int, PageMeta],
    *,
    backend: str,
    progress: Callable[[str, dict[str, Any]], None] | None = None,
) -> list[ParsedPage]:
    """Geometry split → vision per crop; whole-page vision only if the split is messy.

    Plate pages have no numbered panels, so they always take the whole-page path.
    """
    total = len(pages)
    parsed: list[ParsedPage] = []
    native_text = _native_text_by_page(pdf_path, pages)

    split_candidates = [
        page
        for page in pages
        if not (page_metas.get(page) and page_metas[page].plate_no)
    ]
    panels_by_page = extract_page_panels(pdf_path, split_candidates) if split_candidates else {}

    for idx, page_no in enumerate(pages, start=1):
        meta = page_metas.get(page_no)
        is_plate = bool(meta and meta.plate_no)
        panels = [] if is_plate else (panels_by_page.get(page_no) or [])
        messy = is_plate or split_is_messy(panels)
        mode = "full_page_plate" if is_plate else ("full_page" if messy else "panel_crop")

        if messy or backend == "mistral":
            if backend == "mistral" and not messy:
                # Mistral OCR is whole-page only; geometry crops attach afterwards.
                parsed.extend(parse_pages_with_mistral(pdf_path, [page_no]))
            elif backend == "skip":
                parsed.append(_skip_diagram_page(pdf_path, page_no, panels, meta, native_text.get(page_no)))
            else:
                parsed.extend(parse_pages_with_vision(pdf_path, [page_no]))
            logger.info("Diagram page %s → %s (%s panels)", page_no, mode, len(panels))
        else:
            parsed.append(
                _page_from_panel_vision(
                    page_no, panels, meta, native_text.get(page_no), backend=backend
                )
            )
            logger.info("Diagram page %s → vision per crop (%s panels)", page_no, len(panels))

        if progress:
            progress(
                "mistral",
                {
                    "done": idx,
                    "total": total,
                    "page": page_no,
                    "mode": mode,
                    "message": (
                        f"Diagram {idx}/{total}: page {page_no} "
                        f"{'full-page vision' if messy else f'{len(panels)} panel crops'}"
                    ),
                },
            )
    return parsed


def _page_from_panel_vision(
    page_no: int,
    panels: list,
    meta: PageMeta | None,
    native: str | None,
    *,
    backend: str,
) -> ParsedPage:
    """One diagram page: PDF text layer (if any) plus vision OCR of each crop."""
    elements: list[ParsedElement] = []
    snippet = (native or "").strip()
    if len(snippet) >= 40:
        elements.append(
            ParsedElement(
                type=ElementType.PARAGRAPH,
                page=page_no,
                text=snippet[:4000],
                extractor_name="pdf_text_layer",
                extractor_version="pymupdf",
            )
        )

    for panel in panels:
        ocr_text = None
        drawing = panel.drawing_code
        step = panel.linked_step_number
        if panel.image_png and backend != "skip":
            try:
                ocr = ocr_panel_png(panel.image_png)
                ocr_text = ocr.text
                drawing = drawing or ocr.drawing_code
                step = step or ocr.linked_step_number
            except Exception:
                logger.exception(
                    "Panel vision OCR failed on page %s panel %s",
                    page_no,
                    panel.panel_index,
                )
        panel.drawing_code = drawing
        panel.linked_step_number = step
        elements.append(
            ParsedElement(
                type=ElementType.FIGURE,
                page=page_no,
                bbox=panel.bbox,
                text=ocr_text,
                figure_id=drawing or f"page-{page_no}-panel-{panel.panel_index}",
                caption=_panel_caption(panel, meta) or (ocr_text[:240] if ocr_text else None),
                panel_index=panel.panel_index,
                drawing_code=drawing,
                linked_step_number=step,
                image_png=panel.image_png,
                extractor_name=PANEL_EXTRACTOR,
                extractor_version=PANEL_EXTRACTOR_VERSION,
            )
        )

    return ParsedPage(
        page=page_no,
        route=Route.MISTRAL,
        elements=elements,
        notes={
            "via": "panel_geometry+vision",
            "backend": backend,
            "panels": len(panels),
            "panels_attached": True,
        },
    )


def _skip_diagram_page(
    pdf_path: str,
    page_no: int,
    panels: list,
    meta: PageMeta | None,
    native: str | None,
) -> ParsedPage:
    """OCR_BACKEND=skip: store crops (or the page image) without a model call."""
    if panels and not split_is_messy(panels):
        return _page_from_panel_vision(page_no, panels, meta, native, backend="skip")
    from marine_docs.parsers.vision_ocr import _render_page_png

    png = _render_page_png(pdf_path, page_no)
    return ParsedPage(
        page=page_no,
        route=Route.MISTRAL,
        page_image_png=png,
        notes={"backend": "skip", "mode": "full_page"},
        elements=[
            ParsedElement(
                type=ElementType.FIGURE,
                page=page_no,
                figure_id=f"page-{page_no}",
                caption=f"Page {page_no}",
                extractor_name="skip",
                extractor_version="skip",
            )
        ],
    )


def _native_text_by_page(pdf_path: str, pages: list[int]) -> dict[int, str]:
    out: dict[int, str] = {}
    doc = fitz.open(pdf_path)
    try:
        for page_no in pages:
            if not (1 <= page_no <= doc.page_count):
                continue
            out[page_no] = normalize_text(doc[page_no - 1].get_text("text") or "")
    finally:
        doc.close()
    return out


def _attach_panels(
    pdf_path: str,
    parsed_pages: list[ParsedPage],
    page_metas: dict[int, PageMeta],
    *,
    progress: Callable[[int, int], None] | None = None,
) -> int:
    """Attach geometrically cropped panels to Docling pages (spec §6.2).

    Diagram pages that already ran vision-per-crop are left alone. Full-page plates
    have no numbered panels and are stored as a single 'plate' element instead.
    """
    candidates = [
        p.page
        for p in parsed_pages
        if not p.notes.get("panels_attached")
        and not (page_metas.get(p.page) and page_metas[p.page].plate_no)
    ]
    total = len(candidates)
    panels_by_page: dict[int, list] = {}
    if candidates:
        step = max(1, total // 20)
        for index, page in enumerate(candidates, start=1):
            panels_by_page.update(extract_page_panels(pdf_path, [page]))
            if progress and (index % step == 0 or index == total):
                progress(index, total)
    elif progress:
        progress(0, 1)

    attached = 0
    for parsed in parsed_pages:
        if parsed.notes.get("panels_attached"):
            attached += sum(1 for el in parsed.elements if el.panel_index)
            continue
        panels = panels_by_page.get(parsed.page) or []
        if split_is_messy(panels):
            continue
        parsed.elements = [
            el for el in parsed.elements if el.type not in (ElementType.FIGURE, ElementType.PLATE)
        ]
        meta = page_metas.get(parsed.page)
        for panel in panels:
            parsed.elements.append(
                ParsedElement(
                    type=ElementType.FIGURE,
                    page=parsed.page,
                    bbox=panel.bbox,
                    figure_id=panel.drawing_code
                    or f"page-{parsed.page}-panel-{panel.panel_index}",
                    caption=_panel_caption(panel, meta),
                    panel_index=panel.panel_index,
                    drawing_code=panel.drawing_code,
                    linked_step_number=panel.linked_step_number,
                    image_png=panel.image_png,
                    extractor_name=PANEL_EXTRACTOR,
                    extractor_version=PANEL_EXTRACTOR_VERSION,
                )
            )
            attached += 1
        parsed.notes["panels_attached"] = True
    return attached


def _infer_manual_identity(pdf_path: Path) -> dict[str, str | None]:
    """Title / fleet fields from the PDF itself — not a hardcoded engine family."""
    stem = pdf_path.stem
    title = re.sub(r"[\-_]+", " ", stem).strip() or pdf_path.name
    manufacturer: str | None = None
    first_page = ""
    try:
        doc = fitz.open(pdf_path)
        meta = doc.metadata or {}
        pdf_title = (meta.get("title") or "").strip()
        if pdf_title and len(pdf_title) > 3 and not pdf_title.lower().endswith(".pdf"):
            title = pdf_title
        author = (meta.get("author") or "").strip()
        if author:
            manufacturer = author[:200]
        if doc.page_count:
            first_page = (doc[0].get_text("text") or "")[:2000]
        doc.close()
    except Exception:
        logger.exception("Could not read PDF metadata from %s", pdf_path)

    blob = f"{title} {stem} {first_page}"
    model_hits = re.findall(
        r"\b([A-Z]{1,3}\d{2,5}[A-Z]{1,8}(?:-[A-Z0-9]+)?|"
        r"\d{1,2}[A-Z]\d{1,4}[A-Z]{1,8}(?:-[A-Z0-9]+)?)\b",
        blob,
        flags=re.IGNORECASE,
    )
    skip_letters = {"MM", "NM", "KG", "BAR", "KPA", "MPA", "RPM", "DEG"}
    model = title[:80]
    for hit in model_hits:
        letters = re.sub(r"[^A-Z]", "", hit.upper())
        if len(letters) >= 2 and letters not in skip_letters:
            model = hit.upper()[:80]
            break
    slug = re.sub(r"[^A-Za-z0-9]+", "-", stem).strip("-").upper()[:80] or "INGESTED-MANUAL"
    return {
        "title": title[:300],
        "equipment_type": "Technical Manual",
        "revision": "source PDF revision stamps retained per section",
        "equipment_id": slug,
        "manufacturer": manufacturer,
        "model": model,
    }


def _panel_caption(panel, meta: PageMeta | None) -> str:
    parts = []
    if meta and meta.component_title:
        parts.append(meta.component_title)
    if meta and meta.action_title:
        parts.append(meta.action_title)
    if panel.linked_step_number:
        parts.append(f"step {panel.linked_step_number}")
    if panel.drawing_code:
        parts.append(panel.drawing_code)
    return " - ".join(parts) or f"Panel {panel.panel_index} on page {panel.page}"


def _select_pages(classifications, page_count, limit_pages, pages_arg):
    if pages_arg:
        wanted = _parse_page_spec(pages_arg, page_count)
        return [c for c in classifications if c.page in wanted]
    if limit_pages:
        return [c for c in classifications if c.page <= limit_pages]
    return classifications


def _parse_page_spec(spec: str, page_count: int) -> set[int]:
    pages: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            start, end = int(a), int(b)
            pages.update(range(start, end + 1))
        else:
            pages.add(int(part))
    return {p for p in pages if 1 <= p <= page_count}


def _print_route_summary(selected) -> None:
    counts = Counter(c.route.value for c in selected)
    table = Table(title="Page routing summary")
    table.add_column("Route")
    table.add_column("Pages", justify="right")
    for route, n in sorted(counts.items()):
        table.add_row(route, str(n))
    console.print(table)


if __name__ == "__main__":
    sys.exit(main())
