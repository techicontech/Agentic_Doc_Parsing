"""Persist manuals, sections, elements, figures, relationships into Postgres + MinIO."""

from __future__ import annotations

import logging
import re
from typing import Iterable
from uuid import UUID

import fitz
from psycopg.types.json import Jsonb

from marine_docs import minio_client
from marine_docs.db import connect
from marine_docs.models import ElementType, ParsedPage, SectionDraft

logger = logging.getLogger(__name__)

CROSS_REF_RE = re.compile(
    r"(?:See|Refer to|see|refer to)\s+(?:Procedures?|Data|Plate|Chapter)?\s*"
    r"(?P<ref>\d{3}-\d+(?:\.\d+)?|[A-Z]\d[\w.\-]*|Fig\.?\s*[\d.]+)",
    re.IGNORECASE,
)


def create_manual(
    *,
    title: str,
    equipment_type: str,
    vessel_class: str | None,
    revision: str,
    source_pdf_path: str,
) -> tuple[UUID, UUID]:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO manuals (title, equipment_type, vessel_class, revision, source_pdf_path)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (title, equipment_type, vessel_class, revision, source_pdf_path),
            )
            manual_id = cur.fetchone()["id"]
            cur.execute(
                """
                INSERT INTO manual_revisions (manual_id, revision_label, change_summary)
                VALUES (%s, %s, %s)
                RETURNING id
                """,
                (manual_id, revision, "Initial Milestone 1 ingest of Vol II"),
            )
            revision_id = cur.fetchone()["id"]
        conn.commit()
    return manual_id, revision_id


def seed_fleet_registry(manual_id: UUID) -> UUID:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO fleet_registry (
                    vessel_imo, vessel_class, equipment_id, manufacturer, model,
                    serial_number, manual_id, revision_status
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (equipment_id, manual_id) DO UPDATE
                    SET revision_status = EXCLUDED.revision_status
                RETURNING id
                """,
                (
                    None,
                    None,
                    "MAN-BW-S50MC-C-VOL2",
                    "MAN B&W Diesel A/S",
                    "S50MC-C",
                    None,
                    manual_id,
                    "current",
                ),
            )
            row = cur.fetchone()
        conn.commit()
    return row["id"]


def insert_sections(manual_id: UUID, drafts: list[SectionDraft]) -> dict[tuple[str, ...], UUID]:
    """Insert sections parents-first; return path -> id map."""
    path_to_id: dict[tuple[str, ...], UUID] = {}
    ordered = sorted(drafts, key=lambda s: (len(s.path), s.page_start or 0))

    with connect() as conn:
        with conn.cursor() as cur:
            for draft in ordered:
                parent_id = None
                if len(draft.path) > 1:
                    parent_key = tuple(draft.path[:-1])
                    parent_id = path_to_id.get(parent_key)
                key = tuple(draft.path)
                cur.execute(
                    """
                    INSERT INTO sections (
                        manual_id, parent_section_id, path, title,
                        page_start, page_end, doc_code, edition, section_kind
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (manual_id, path) DO UPDATE SET
                        page_start = COALESCE(EXCLUDED.page_start, sections.page_start),
                        page_end = COALESCE(EXCLUDED.page_end, sections.page_end),
                        doc_code = COALESCE(EXCLUDED.doc_code, sections.doc_code),
                        edition = COALESCE(EXCLUDED.edition, sections.edition),
                        section_kind = COALESCE(EXCLUDED.section_kind, sections.section_kind)
                    RETURNING id
                    """,
                    (
                        manual_id,
                        parent_id,
                        list(draft.path),
                        draft.title,
                        draft.page_start,
                        draft.page_end,
                        draft.doc_code,
                        draft.edition,
                        draft.section_kind,
                    ),
                )
                path_to_id[key] = cur.fetchone()["id"]
        conn.commit()
    return path_to_id


def section_id_for_page(
    page: int,
    sections: list[SectionDraft],
    path_to_id: dict[tuple[str, ...], UUID],
) -> UUID | None:
    """Pick the most specific section covering this page."""
    candidates = [
        s
        for s in sections
        if s.page_start is not None
        and s.page_end is not None
        and s.page_start <= page <= s.page_end
    ]
    if not candidates:
        return None
    # Prefer deepest path, then narrowest page span
    candidates.sort(key=lambda s: (-len(s.path), (s.page_end or 0) - (s.page_start or 0)))
    return path_to_id.get(tuple(candidates[0].path))


def persist_parsed_pages(
    *,
    manual_id: UUID,
    revision_id: UUID,
    source_pdf_path: str,
    sections: list[SectionDraft],
    path_to_id: dict[tuple[str, ...], UUID],
    pages: Iterable[ParsedPage],
) -> dict[str, int]:
    counts = {"elements": 0, "figures": 0, "provenance": 0, "relationships": 0}
    element_ids_by_page: dict[int, list[UUID]] = {}
    figure_element_ids: list[tuple[UUID, int, UUID | None]] = []

    with connect() as conn:
        with conn.cursor() as cur:
            for parsed in pages:
                section_id = section_id_for_page(parsed.page, sections, path_to_id)
                element_ids_by_page.setdefault(parsed.page, [])

                # Ensure page image stored for visual path when available or for figure pages.
                image_key = None
                if parsed.page_image_png:
                    image_key = f"manuals/{manual_id}/pages/page-{parsed.page:04d}.png"
                    minio_client.upload_png(image_key, parsed.page_image_png)

                for el in parsed.elements:
                    cur.execute(
                        """
                        INSERT INTO elements (
                            section_id, type, page, bbox, text, table_json, figure_id,
                            extractor_name, extractor_version, confidence
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                        """,
                        (
                            section_id,
                            el.type.value,
                            el.page,
                            Jsonb(el.bbox) if el.bbox else None,
                            el.text,
                            Jsonb(el.table_json) if el.table_json is not None else None,
                            el.figure_id,
                            el.extractor_name,
                            el.extractor_version,
                            el.confidence,
                        ),
                    )
                    element_id = cur.fetchone()["id"]
                    element_ids_by_page[parsed.page].append(element_id)
                    counts["elements"] += 1

                    cur.execute(
                        """
                        INSERT INTO element_provenance (
                            element_id, source_pdf_path, revision_id, page, bbox,
                            extractor_name, extractor_version
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            element_id,
                            source_pdf_path,
                            revision_id,
                            el.page,
                            Jsonb(el.bbox) if el.bbox else None,
                            el.extractor_name,
                            el.extractor_version,
                        ),
                    )
                    counts["provenance"] += 1

                    if el.type == ElementType.FIGURE:
                        if image_key is None:
                            # Render on demand for Docling figure pages without prior PNG.
                            image_key = f"manuals/{manual_id}/pages/page-{parsed.page:04d}.png"
                            png = _render_page_png(source_pdf_path, parsed.page)
                            minio_client.upload_png(image_key, png)
                        cur.execute(
                            """
                            INSERT INTO figures (
                                element_id, manual_id, page, section_id, image_path, caption, figure_label
                            )
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (manual_id, page, image_path) DO UPDATE SET
                                caption = COALESCE(EXCLUDED.caption, figures.caption),
                                figure_label = COALESCE(EXCLUDED.figure_label, figures.figure_label),
                                element_id = COALESCE(EXCLUDED.element_id, figures.element_id),
                                section_id = COALESCE(EXCLUDED.section_id, figures.section_id)
                            RETURNING id
                            """,
                            (
                                element_id,
                                manual_id,
                                parsed.page,
                                section_id,
                                image_key,
                                el.caption or el.text,
                                el.figure_id,
                            ),
                        )
                        counts["figures"] += 1
                        figure_element_ids.append((element_id, parsed.page, section_id))

            # belongs_to edges: every element with a section
            cur.execute(
                """
                SELECT e.id AS element_id, e.section_id
                FROM elements e
                JOIN sections s ON s.id = e.section_id
                WHERE s.manual_id = %s AND e.section_id IS NOT NULL
                """,
                (manual_id,),
            )
            # belongs_to is element->section conceptually; schema is element->element.
            # Represent as relationship among elements in same section using first heading/paragraph as anchor,
            # and also store illustrates/describes/cross_ref below.
            # For Milestone 1 we also add cross_ref edges between elements.

            counts["relationships"] += _insert_cross_refs(cur, manual_id)
            # describes/illustrates are built after plate figures exist (see rebuild_structural_relationships)

        conn.commit()
    return counts


def rebuild_structural_relationships(manual_id: UUID) -> int:
    """Create describes / illustrates edges after all section elements exist."""
    with connect() as conn:
        with conn.cursor() as cur:
            inserted = _insert_illustrates_and_describes(cur, manual_id)
        conn.commit()
    return inserted


def rebuild_cross_refs(manual_id: UUID) -> int:
    with connect() as conn:
        with conn.cursor() as cur:
            inserted = _insert_cross_refs(cur, manual_id)
        conn.commit()
    return inserted


def _insert_cross_refs(cur, manual_id: UUID) -> int:
    cur.execute(
        """
        SELECT e.id, e.text, e.page, e.section_id
        FROM elements e
        JOIN sections s ON s.id = e.section_id
        WHERE s.manual_id = %s AND e.text IS NOT NULL
        """,
        (manual_id,),
    )
    rows = cur.fetchall()
    inserted = 0
    # Build doc_code -> representative element
    cur.execute(
        """
        SELECT s.doc_code, e.id
        FROM sections s
        JOIN elements e ON e.section_id = s.id
        WHERE s.manual_id = %s AND s.doc_code IS NOT NULL
        ORDER BY e.page, e.id
        """,
        (manual_id,),
    )
    code_to_element: dict[str, UUID] = {}
    for row in cur.fetchall():
        code_to_element.setdefault(row["doc_code"], row["id"])

    for row in rows:
        text = row["text"] or ""
        for m in CROSS_REF_RE.finditer(text):
            ref = m.group("ref").strip()
            # Map procedure-style 909-5.2 -> approximate M90905 / section search
            target_id = _resolve_ref(ref, code_to_element, cur, manual_id)
            if not target_id or target_id == row["id"]:
                continue
            cur.execute(
                """
                INSERT INTO element_relationships (from_element_id, to_element_id, relationship_type)
                VALUES (%s, %s, 'cross_ref')
                ON CONFLICT DO NOTHING
                """,
                (row["id"], target_id),
            )
            inserted += 1
    return inserted


def _resolve_ref(ref: str, code_to_element: dict[str, UUID], cur, manual_id: UUID) -> UUID | None:
    if ref in code_to_element:
        return code_to_element[ref]
    # 909-11.1 -> look for sections with doc_code like M90911
    m = re.match(r"^(?P<a>\d{3})-(?P<b>\d+)", ref)
    if m:
        chapter = m.group("a")
        proc = m.group("b")
        guess = f"M{chapter}{int(proc):02d}" if len(proc) <= 2 else f"M{chapter}{proc}"
        # Try common patterns M90911
        for code, eid in code_to_element.items():
            if code.endswith(f"{chapter}{proc.zfill(2)}") or code.endswith(f"{chapter}{proc}"):
                return eid
            if guess in code:
                return eid
        cur.execute(
            """
            SELECT e.id
            FROM sections s
            JOIN elements e ON e.section_id = s.id
            WHERE s.manual_id = %s AND s.doc_code ILIKE %s
            ORDER BY e.page
            LIMIT 1
            """,
            (manual_id, f"%{chapter}%{proc}%"),
        )
        hit = cur.fetchone()
        return hit["id"] if hit else None
    return None


def _insert_illustrates_and_describes(cur, manual_id: UUID) -> int:
    """Link P-plates (illustrates) and D-data sheets (describes) to M-procedures.

    MAN B&W coding: D10101 (data) and M90101 (procedure) share component stem 0101;
    chapter family maps D1xx → M9xx (Cylinder Cover data 101 ↔ procedure 901).
    """
    inserted = 0
    cur.execute(
        """
        SELECT s.id AS section_id, s.doc_code, s.section_kind, s.page_start, s.page_end, s.path,
               (SELECT e.id FROM elements e WHERE e.section_id = s.id ORDER BY e.page, e.id LIMIT 1) AS element_id
        FROM sections s
        WHERE s.manual_id = %s AND s.doc_code IS NOT NULL
        """,
        (manual_id,),
    )
    sections = [r for r in cur.fetchall() if r["element_id"]]
    procs = [s for s in sections if s["section_kind"] == "procedure"]
    plates = [s for s in sections if s["section_kind"] == "plate"]
    data = [s for s in sections if s["section_kind"] == "data"]

    def component_stem(code: str) -> str:
        digits = re.sub(r"\D", "", code)
        return digits[-4:] if len(digits) >= 4 else digits

    def chapter_family(code: str) -> str | None:
        digits = re.sub(r"\D", "", code)
        if len(digits) < 3:
            return None
        if code.startswith(("M", "P", "A")):
            return digits[:3]
        if code.startswith("D") and digits[0] == "1":
            return "9" + digits[1:3]
        return digits[:3]

    for plate in plates:
        p_chapter = chapter_family(plate["doc_code"])
        for proc in procs:
            if p_chapter and p_chapter == chapter_family(proc["doc_code"]):
                cur.execute(
                    """
                    INSERT INTO element_relationships (from_element_id, to_element_id, relationship_type)
                    VALUES (%s, %s, 'illustrates')
                    ON CONFLICT DO NOTHING
                    """,
                    (plate["element_id"], proc["element_id"]),
                )
                if cur.rowcount:
                    inserted += 1
                break

    for d in data:
        d_stem = component_stem(d["doc_code"])
        d_chapter = chapter_family(d["doc_code"])
        matched = False
        for proc in procs:
            if d_stem and d_stem == component_stem(proc["doc_code"]):
                cur.execute(
                    """
                    INSERT INTO element_relationships (from_element_id, to_element_id, relationship_type)
                    VALUES (%s, %s, 'describes')
                    ON CONFLICT DO NOTHING
                    """,
                    (d["element_id"], proc["element_id"]),
                )
                if cur.rowcount:
                    inserted += 1
                matched = True
                break
        if matched:
            continue
        for proc in procs:
            if d_chapter and d_chapter == chapter_family(proc["doc_code"]):
                cur.execute(
                    """
                    INSERT INTO element_relationships (from_element_id, to_element_id, relationship_type)
                    VALUES (%s, %s, 'describes')
                    ON CONFLICT DO NOTHING
                    """,
                    (d["element_id"], proc["element_id"]),
                )
                if cur.rowcount:
                    inserted += 1
                break
    return inserted


def ensure_page_figures_for_plates(
    *,
    manual_id: UUID,
    source_pdf_path: str,
    sections: list[SectionDraft],
    path_to_id: dict[tuple[str, ...], UUID],
) -> int:
    """Minimal visual path: every plate section page gets a stored page image + figure row."""
    created = 0
    plate_pages = set()
    for s in sections:
        if s.section_kind == "plate" and s.page_start and s.page_end:
            for p in range(s.page_start, s.page_end + 1):
                plate_pages.add(p)

    with connect() as conn:
        with conn.cursor() as cur:
            for page in sorted(plate_pages):
                section_id = section_id_for_page(page, sections, path_to_id)
                image_key = f"manuals/{manual_id}/pages/page-{page:04d}.png"
                # Skip if already present
                cur.execute(
                    "SELECT 1 FROM figures WHERE manual_id = %s AND page = %s LIMIT 1",
                    (manual_id, page),
                )
                if cur.fetchone():
                    continue
                png = _render_page_png(source_pdf_path, page)
                minio_client.upload_png(image_key, png)
                cur.execute(
                    """
                    INSERT INTO elements (
                        section_id, type, page, text, figure_id,
                        extractor_name, extractor_version
                    )
                    VALUES (%s, 'figure', %s, %s, %s, 'page_render', 'pymupdf')
                    RETURNING id
                    """,
                    (section_id, page, f"Plate page {page}", f"page-{page}"),
                )
                element_id = cur.fetchone()["id"]
                cur.execute(
                    """
                    INSERT INTO figures (
                        element_id, manual_id, page, section_id, image_path, caption, figure_label
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        element_id,
                        manual_id,
                        page,
                        section_id,
                        image_key,
                        f"Plate page {page}",
                        f"page-{page}",
                    ),
                )
                created += 1
        conn.commit()
    return created


def _render_page_png(pdf_path: str, page_1based: int, zoom: float = 2.0) -> bytes:
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_1based - 1]
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        return pix.tobytes("png")
    finally:
        doc.close()


def start_ingest_run(source_pdf_path: str, page_count: int) -> UUID:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ingest_runs (source_pdf_path, status, page_count)
                VALUES (%s, 'running', %s)
                RETURNING id
                """,
                (source_pdf_path, page_count),
            )
            run_id = cur.fetchone()["id"]
        conn.commit()
    return run_id


def finish_ingest_run(
    run_id: UUID,
    *,
    manual_id: UUID | None,
    status: str,
    docling_pages: int,
    mistral_pages: int,
    skipped_pages: int,
    notes: dict,
) -> None:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE ingest_runs
                SET manual_id = %s,
                    status = %s,
                    docling_pages = %s,
                    mistral_pages = %s,
                    skipped_pages = %s,
                    notes = %s,
                    finished_at = now()
                WHERE id = %s
                """,
                (
                    manual_id,
                    status,
                    docling_pages,
                    mistral_pages,
                    skipped_pages,
                    Jsonb(notes),
                    run_id,
                ),
            )
        conn.commit()
