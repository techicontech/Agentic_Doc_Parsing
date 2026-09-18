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
from marine_docs.convention import see_reference_regex
from marine_docs.pagemeta import PageMeta, normalize_text

logger = logging.getLogger(__name__)

# Default verbs; the actual citation-key pattern comes from citation_convention (§7).
CROSS_REF_RE = re.compile(
    r"(?:See|Refer to|as described in)\s+(?P<kind>Procedures?|Data|Plate|Section|Chapter|Drawing|Appendix)?\s*"
    r"(?P<ref>\d{3}-\d+(?:\.\d+)?|[A-Z]\d[\w.-]*|\d+(?:\.\d+)+)?",
    re.IGNORECASE,
)
STEP_LEAD_RE = re.compile(r"^\s*(\d{1,2})\.\s")


def _step_from_text(text: str | None) -> int | None:
    if not text:
        return None
    match = STEP_LEAD_RE.match(text)
    return int(match.group(1)) if match else None


def create_manual(
    *,
    title: str,
    equipment_type: str,
    vessel_class: str | None,
    revision: str,
    source_pdf_path: str,
    manufacturer: str | None = None,
    citation_convention: dict | None = None,
) -> tuple[UUID, UUID]:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO manuals (
                    title, equipment_type, vessel_class, revision, source_pdf_path,
                    manufacturer, citation_convention
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    title,
                    equipment_type,
                    vessel_class,
                    revision,
                    source_pdf_path,
                    manufacturer,
                    Jsonb(citation_convention) if citation_convention else None,
                ),
            )
            manual_id = cur.fetchone()["id"]
            cur.execute(
                """
                INSERT INTO manual_revisions (manual_id, revision_label, change_summary)
                VALUES (%s, %s, %s)
                RETURNING id
                """,
                (manual_id, revision, "Initial ingest"),
            )
            revision_id = cur.fetchone()["id"]
        conn.commit()
    return manual_id, revision_id


def seed_fleet_registry(
    manual_id: UUID,
    *,
    equipment_id: str,
    manufacturer: str | None,
    model: str,
) -> UUID:
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
                    SET revision_status = EXCLUDED.revision_status,
                        manufacturer = EXCLUDED.manufacturer,
                        model = EXCLUDED.model
                RETURNING id
                """,
                (
                    None,
                    None,
                    equipment_id,
                    manufacturer,
                    model,
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
                        page_start, page_end, doc_code, edition, section_kind,
                        procedure_no, plate_no, component_title, action_title, citation_key
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (manual_id, path) DO UPDATE SET
                        page_start = COALESCE(EXCLUDED.page_start, sections.page_start),
                        page_end = COALESCE(EXCLUDED.page_end, sections.page_end),
                        doc_code = COALESCE(EXCLUDED.doc_code, sections.doc_code),
                        edition = COALESCE(EXCLUDED.edition, sections.edition),
                        section_kind = COALESCE(EXCLUDED.section_kind, sections.section_kind),
                        procedure_no = COALESCE(EXCLUDED.procedure_no, sections.procedure_no),
                        plate_no = COALESCE(EXCLUDED.plate_no, sections.plate_no),
                        component_title = COALESCE(EXCLUDED.component_title, sections.component_title),
                        action_title = COALESCE(EXCLUDED.action_title, sections.action_title),
                        citation_key = COALESCE(EXCLUDED.citation_key, sections.citation_key)
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
                        draft.procedure_no,
                        draft.plate_no,
                        draft.component_title,
                        draft.action_title,
                        draft.citation_key or draft.plate_no or draft.procedure_no or draft.doc_code,
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
    page_metas: dict[int, PageMeta] | None = None,
) -> dict[str, int]:
    counts = {"elements": 0, "figures": 0, "provenance": 0, "relationships": 0}
    element_ids_by_page: dict[int, list[UUID]] = {}
    figure_element_ids: list[tuple[UUID, int, UUID | None]] = []
    metas = page_metas or {}

    with connect() as conn:
        with conn.cursor() as cur:
            for parsed in pages:
                section_id = section_id_for_page(parsed.page, sections, path_to_id)
                element_ids_by_page.setdefault(parsed.page, [])
                meta = metas.get(parsed.page)

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
                            extractor_name, extractor_version, confidence,
                            procedure_no, plate_no, edition, page_number_printed,
                            panel_index, drawing_code, linked_step_number,
                            citation_key, citation_fields
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                                %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                            meta.procedure_no if meta else None,
                            meta.plate_no if meta else None,
                            meta.edition if meta else None,
                            meta.page_number_printed if meta else None,
                            el.panel_index,
                            el.drawing_code,
                            el.linked_step_number or _step_from_text(el.text),
                            (meta.citation_key if meta else None)
                            or (meta.plate_no if meta else None)
                            or (meta.procedure_no if meta else None),
                            Jsonb(meta.citation_fields) if meta and meta.citation_fields else None,
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

                    if el.type in (ElementType.FIGURE, ElementType.PLATE):
                        # Panels get their own crop; whole-page figures/plates reuse
                        # the page render (spec §6.2 / §6.5).
                        if el.image_png and el.panel_index:
                            figure_key = (
                                f"manuals/{manual_id}/panels/"
                                f"page-{parsed.page:04d}-p{el.panel_index:02d}.png"
                            )
                            minio_client.upload_png(figure_key, el.image_png)
                        else:
                            if image_key is None:
                                # Render on demand for figure pages without a prior PNG.
                                image_key = f"manuals/{manual_id}/pages/page-{parsed.page:04d}.png"
                                minio_client.upload_png(
                                    image_key, _render_page_png(source_pdf_path, parsed.page)
                                )
                            figure_key = image_key
                        cur.execute(
                            """
                            INSERT INTO figures (
                                element_id, manual_id, page, section_id, image_path, caption,
                                figure_label, panel_index, drawing_code, linked_step_number
                            )
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (manual_id, page, image_path) DO UPDATE SET
                                caption = COALESCE(EXCLUDED.caption, figures.caption),
                                figure_label = COALESCE(EXCLUDED.figure_label, figures.figure_label),
                                element_id = COALESCE(EXCLUDED.element_id, figures.element_id),
                                section_id = COALESCE(EXCLUDED.section_id, figures.section_id),
                                panel_index = COALESCE(EXCLUDED.panel_index, figures.panel_index),
                                drawing_code = COALESCE(EXCLUDED.drawing_code, figures.drawing_code),
                                linked_step_number = COALESCE(
                                    EXCLUDED.linked_step_number, figures.linked_step_number
                                )
                            RETURNING id
                            """,
                            (
                                element_id,
                                manual_id,
                                parsed.page,
                                section_id,
                                figure_key,
                                el.caption or el.text,
                                el.drawing_code or el.figure_id,
                                el.panel_index,
                                el.drawing_code,
                                el.linked_step_number,
                            ),
                        )
                        counts["figures"] += 1
                        figure_element_ids.append((element_id, parsed.page, section_id))

        conn.commit()
    # Cross-references, describes/illustrates and step links are built afterwards, once
    # plate elements exist too — see rebuild_cross_refs / rebuild_step_links.
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
    """Resolve 'see / refer to' + this manual's citation_key pattern (spec §7).

    Typed as references_section / references_data so ranking can auto-resolve one hop.
    """
    conv = None
    try:
        cur.execute(
            "SELECT citation_convention FROM manuals WHERE id = %s",
            (manual_id,),
        )
        row = cur.fetchone() or {}
        conv = row.get("citation_convention")
    except Exception:
        conv = None
    xref_re = see_reference_regex(conv) if conv else CROSS_REF_RE
    index = _reference_index(cur, manual_id)
    cur.execute(
        """
        SELECT e.id, e.text, e.page, e.section_id, e.procedure_no,
               s.component_title, s.doc_code
        FROM elements e
        JOIN sections s ON s.id = e.section_id
        WHERE s.manual_id = %s AND e.text IS NOT NULL
        """,
        (manual_id,),
    )
    rows = cur.fetchall()

    inserted = 0
    for row in rows:
        text = normalize_text(row["text"])
        for match in xref_re.finditer(text):
            target_id, rel_type = _resolve_reference(match, row, index)
            if not target_id or target_id == row["id"]:
                continue
            cur.execute(
                """
                INSERT INTO element_relationships (from_element_id, to_element_id, relationship_type)
                VALUES (%s, %s, %s)
                ON CONFLICT DO NOTHING
                """,
                (row["id"], target_id, rel_type),
            )
            inserted += cur.rowcount
    return inserted


def _reference_index(cur, manual_id: UUID) -> dict[str, dict[str, UUID]]:
    """Lookup tables for reference targets, keyed by the manual's own codes."""
    cur.execute(
        """
        SELECT e.id, e.page, e.procedure_no, s.doc_code, s.section_kind, s.component_title
        FROM elements e
        JOIN sections s ON s.id = e.section_id
        WHERE s.manual_id = %s AND e.type IN ('paragraph', 'heading', 'table')
        ORDER BY e.page, e.id
        """,
        (manual_id,),
    )
    by_procedure: dict[str, UUID] = {}
    by_code: dict[str, UUID] = {}
    data_by_component: dict[str, UUID] = {}
    for row in cur.fetchall():
        if row["procedure_no"]:
            by_procedure.setdefault(row["procedure_no"], row["id"])
        if row["doc_code"]:
            by_code.setdefault(row["doc_code"].upper(), row["id"])
        if row["section_kind"] == "data" and row["component_title"]:
            data_by_component.setdefault(row["component_title"].strip().lower(), row["id"])
    return {
        "procedure": by_procedure,
        "code": by_code,
        "data_component": data_by_component,
    }


def _resolve_reference(
    match: re.Match,
    row: dict,
    index: dict[str, dict[str, UUID]],
) -> tuple[UUID | None, str]:
    kind = (match.group("kind") or "").strip().lower()
    ref = (match.group("ref") or "").strip().upper()

    if not ref:
        # Bare "See Data" points at this component's own data sheet.
        if kind == "data" and row.get("component_title"):
            component = row["component_title"].strip().lower()
            return index["data_component"].get(component), "references_data"
        return None, "cross_ref"

    if re.match(r"^\d{3}-\d+", ref):
        target = index["procedure"].get(ref)
        if target is None and "." not in ref:
            # "See Procedure 903-1" can name a group; take its first sub-procedure.
            target = next(
                (
                    eid
                    for key, eid in sorted(index["procedure"].items())
                    if key.startswith(f"{ref}.")
                ),
                None,
            )
        rel = "references_data" if kind == "data" else "references_section"
        return target, rel

    target = index["code"].get(ref)
    if ref.startswith("D") or kind == "data":
        return target, "references_data"
    if ref.startswith(("M", "A")):
        return target, "references_section"
    return target, "cross_ref"


def rebuild_step_links(manual_id: UUID) -> int:
    """Pair each diagram panel to the numbered step it illustrates (spec §7.2)."""
    with connect() as conn:
        with conn.cursor() as cur:
            inserted = _insert_illustrates_step(cur, manual_id)
        conn.commit()
    return inserted


def _insert_illustrates_step(cur, manual_id: UUID) -> int:
    cur.execute(
        """
        SELECT e.id, e.page, e.linked_step_number
        FROM elements e
        JOIN sections s ON s.id = e.section_id
        WHERE s.manual_id = %s
          AND e.type IN ('figure', 'plate')
          AND e.linked_step_number IS NOT NULL
        """,
        (manual_id,),
    )
    panels = cur.fetchall()
    if not panels:
        return 0

    cur.execute(
        """
        SELECT e.id, e.page, e.text
        FROM elements e
        JOIN sections s ON s.id = e.section_id
        WHERE s.manual_id = %s AND e.type IN ('paragraph', 'heading', 'list')
          AND e.text IS NOT NULL
        ORDER BY e.page, e.id
        """,
        (manual_id,),
    )
    text_by_page: dict[int, list[dict]] = {}
    for row in cur.fetchall():
        text_by_page.setdefault(row["page"], []).append(row)

    inserted = 0
    for panel in panels:
        step = panel["linked_step_number"]
        candidates = text_by_page.get(panel["page"]) or []
        step_re = re.compile(rf"(?:^|\n)\s*{step}\.\s", re.MULTILINE)
        target = next(
            (c for c in candidates if step_re.search(normalize_text(c["text"]))),
            None,
        )
        if target is None:
            # Panels always belong with the page's prose; fall back to the longest block.
            target = max(candidates, key=lambda c: len(c["text"] or ""), default=None)
        if target is None or target["id"] == panel["id"]:
            continue
        cur.execute(
            """
            UPDATE elements
            SET linked_step_number = COALESCE(linked_step_number, %s)
            WHERE id = %s
            """,
            (step, target["id"]),
        )
        cur.execute(
            """
            INSERT INTO element_relationships (from_element_id, to_element_id, relationship_type)
            VALUES (%s, %s, 'illustrates_step')
            ON CONFLICT DO NOTHING
            """,
            (panel["id"], target["id"]),
        )
        inserted += cur.rowcount
    return inserted


def _insert_illustrates_and_describes(cur, manual_id: UUID) -> int:
    """Link plates (illustrates) and data sheets (describes) to procedures.

    Code-stem matching covers manuals that number data/procedure/plate families
    together. Component-title matching covers manuals that do not.
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

    inserted += _describe_by_component_title(cur, manual_id)
    return inserted


def _describe_by_component_title(cur, manual_id: UUID) -> int:
    """Pair data sheets with checking procedures that share a component title.

    Works on any IETM that splits spec tables from how-to text, including manuals
    that do not use D/M/P code stems.
    """
    cur.execute(
        """
        INSERT INTO element_relationships (from_element_id, to_element_id, relationship_type)
        SELECT DISTINCT ON (d.id) d.id, p.id, 'describes'
        FROM elements d
        JOIN sections ds ON ds.id = d.section_id
        JOIN sections ps ON ps.manual_id = ds.manual_id
          AND lower(trim(ps.component_title)) = lower(trim(ds.component_title))
          AND ps.section_kind = 'procedure'
        JOIN elements p ON p.section_id = ps.id
        WHERE ds.manual_id = %s
          AND ds.section_kind = 'data'
          AND ds.component_title IS NOT NULL
          AND length(trim(ds.component_title)) > 2
          AND d.type IN ('paragraph', 'table', 'heading')
          AND p.type IN ('paragraph', 'table', 'heading')
        ORDER BY d.id,
          CASE WHEN ps.action_title ~* 'check|inspect|evaluat|criteri|accept' THEN 0 ELSE 1 END,
          length(coalesce(p.text, '')) DESC
        ON CONFLICT DO NOTHING
        """,
        (manual_id,),
    )
    return cur.rowcount or 0


def ensure_plate_page_elements(
    *,
    manual_id: UUID,
    source_pdf_path: str,
    sections: list[SectionDraft],
    path_to_id: dict[tuple[str, ...], UUID],
    page_metas: dict[int, PageMeta] | None = None,
) -> int:
    """Full-page plates: one 'plate' element per page, cited by Plate No. (spec §6.5).

    Plates have no numbered panels, so they are never split the way procedure pages
    are — the whole page is the retrievable unit.
    """
    created = 0
    metas = page_metas or {}
    plate_pages = {
        page
        for s in sections
        if s.section_kind == "plate" and s.page_start and s.page_end
        for page in range(s.page_start, s.page_end + 1)
    }
    plate_pages.update(page for page, meta in metas.items() if meta.plate_no)

    with connect() as conn:
        with conn.cursor() as cur:
            for page in sorted(plate_pages):
                section_id = section_id_for_page(page, sections, path_to_id)
                meta = metas.get(page)
                image_key = f"manuals/{manual_id}/pages/page-{page:04d}.png"
                cur.execute(
                    """
                    SELECT 1 FROM elements
                    WHERE page = %s AND type = 'plate'
                      AND section_id IN (SELECT id FROM sections WHERE manual_id = %s)
                    LIMIT 1
                    """,
                    (page, manual_id),
                )
                if cur.fetchone():
                    continue
                minio_client.upload_png(image_key, _render_page_png(source_pdf_path, page))
                label = (meta.plate_no if meta else None) or f"page-{page}"
                caption = (
                    " - ".join(
                        part
                        for part in (
                            meta.component_title if meta else None,
                            meta.action_title if meta else None,
                        )
                        if part
                    )
                    if meta
                    else None
                ) or f"Plate page {page}"
                cur.execute(
                    """
                    INSERT INTO elements (
                        section_id, type, page, text, figure_id,
                        extractor_name, extractor_version,
                        procedure_no, plate_no, edition, page_number_printed
                    )
                    VALUES (%s, 'plate', %s, %s, %s, 'page_render', 'pymupdf', %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        section_id,
                        page,
                        caption,
                        label,
                        meta.procedure_no if meta else None,
                        meta.plate_no if meta else None,
                        meta.edition if meta else None,
                        meta.page_number_printed if meta else None,
                    ),
                )
                element_id = cur.fetchone()["id"]
                cur.execute(
                    """
                    INSERT INTO figures (
                        element_id, manual_id, page, section_id, image_path, caption, figure_label
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (manual_id, page, image_path) DO UPDATE SET
                        element_id = EXCLUDED.element_id,
                        caption = COALESCE(EXCLUDED.caption, figures.caption)
                    """,
                    (element_id, manual_id, page, section_id, image_key, caption, label),
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
