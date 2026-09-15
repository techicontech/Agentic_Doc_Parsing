"""Milestone 1 ingest: hybrid Docling + Mistral OCR → Postgres + MinIO."""

from __future__ import annotations

import argparse
import json
import logging
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
from marine_docs.parsers.docling_parser import parse_pages_with_docling
from marine_docs.parsers.mistral_ocr import parse_pages_with_mistral
from marine_docs.parsers.vision_ocr import parse_pages_with_vision
from marine_docs.structure import extract_sections

console = Console()
logger = logging.getLogger("ingest")

ProgressCb = Callable[[str, dict[str, Any]], None]

# Overall progress budget (weights sum to 100)
_W_CLASSIFY = 5
_W_SECTIONS = 5
_W_DOCLING = 60
_W_MISTRAL = 20
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
    sections = extract_sections(str(pdf_path))
    base = _W_CLASSIFY + _W_SECTIONS
    emit("sections_ready", base, f"Sections ready ({len(sections)})", section_count=len(sections))

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
            label = {
                "claude": "Claude vision OCR",
                "mistral": "Mistral OCR",
                "skip": "Diagram pages",
            }.get(backend, "OCR")
            emit(
                "mistral",
                pct,
                f"{label} {done}/{total} pages",
                done=done,
                total=total,
                page=payload.get("page"),
                backend=backend,
            )

        if backend == "skip":
            emit("mistral_ocr_start", base, f"Skipping OCR ({len(mistral_pages)} diagram pages as images)…")
            # Keep page images only via empty parse that still stores figures downstream
            from marine_docs.parsers.vision_ocr import _render_page_png

            skip_pages: list = []
            total = len(mistral_pages)
            for idx, page_no in enumerate(mistral_pages, start=1):
                png = _render_page_png(str(pdf_path), page_no)
                skip_pages.append(
                    ParsedPage(
                        page=page_no,
                        route=Route.MISTRAL,
                        page_image_png=png,
                        notes={"backend": "skip"},
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
                )
                on_ocr("mistral", {"done": idx, "total": total, "page": page_no})
            parsed_pages.extend(skip_pages)
        elif backend == "mistral":
            emit("mistral_ocr_start", base, f"Mistral OCR starting ({len(mistral_pages)} pages)…")
            parsed_pages.extend(
                parse_pages_with_mistral(str(pdf_path), mistral_pages, progress=on_ocr)
            )
        else:
            emit(
                "mistral_ocr_start",
                base,
                f"Claude vision OCR starting ({len(mistral_pages)} pages)…",
            )
            parsed_pages.extend(
                parse_pages_with_vision(str(pdf_path), mistral_pages, progress=on_ocr)
            )
        base = _W_CLASSIFY + _W_SECTIONS + w_docling + w_mistral
        emit("mistral_ocr_done", base, f"Hard-page OCR done ({len(mistral_pages)} pages, backend={backend})")
    else:
        base = _W_CLASSIFY + _W_SECTIONS + w_docling + w_mistral

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
        manual_id, revision_id = builder.create_manual(
            title="MAN B&W S50MC-C Volume II Maintenance",
            equipment_type="Main Engine",
            vessel_class=None,
            revision="Vol II / S50MC-C (source PDF revision stamps retained per section)",
            source_pdf_path=str(pdf_path),
        )
        fleet_id = builder.seed_fleet_registry(manual_id)
        path_to_id = builder.insert_sections(manual_id, sections)
        counts = builder.persist_parsed_pages(
            manual_id=manual_id,
            revision_id=revision_id,
            source_pdf_path=str(pdf_path),
            sections=sections,
            path_to_id=path_to_id,
            pages=parsed_pages,
        )
        plate_figs = builder.ensure_page_figures_for_plates(
            manual_id=manual_id,
            source_pdf_path=str(pdf_path),
            sections=sections,
            path_to_id=path_to_id,
        )
        counts["plate_page_figures"] = plate_figs
        counts["structural_relationships"] = builder.rebuild_structural_relationships(manual_id)
        counts["cross_ref_relationships"] = builder.rebuild_cross_refs(manual_id)

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

    progress_log(f"Full logs → {log_path}")

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
