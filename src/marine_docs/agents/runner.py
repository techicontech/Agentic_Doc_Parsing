"""Run one chat turn through the pipeline and shape it for the API/UI."""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import quote

from marine_docs.agents.pipeline import execute_pipeline
from marine_docs.db import connect
from marine_docs.retrieval import RetrievalResult, dict_to_evidence, get_fleet_context

logger = logging.getLogger(__name__)


def run_agentic_query(
    query: str,
    *,
    equipment_context: str | None = None,
    log: bool = True,
):
    from marine_docs.chat import ChatResponse

    state = execute_pipeline(query, equipment=equipment_context)
    hits = state.get("ranked_hits") or []
    verification = state.get("verification") or {}
    fleet = get_fleet_context() or {}
    priority = state.get("priority_notes") or {}

    retrieval = RetrievalResult(
        evidences=[dict_to_evidence(h) for h in hits if h.get("element_id")],
        fleet_model=fleet.get("model") or verification.get("fleet_model"),
        manual_title=fleet.get("title") or verification.get("manual_title"),
        revision=fleet.get("revision") or verification.get("revision"),
        notes={
            "paths_fired": state.get("router_labels") or [],
            "router_reason": state.get("router_reason"),
            "fusion_method": "rrf",
            "orchestration": state.get("orchestration"),
            "exact_identifier_hits": priority.get("exact_matches", 0),
            "panels_pulled": priority.get("panels_pulled", 0),
            "cross_refs_resolved": priority.get("cross_refs_resolved", 0),
            "diagram_count": sum(1 for h in hits if h.get("figure_image_path")),
        },
    )

    passed = bool(verification.get("passed"))
    resp = ChatResponse(
        answer=state.get("final_answer") or "No answer produced.",
        abstained=not passed,
        citations=[_citation(h, retrieval) for h in _citation_hits(hits, query)],
        diagrams=_diagrams(hits, query),
        verification={
            "passed": passed,
            "checks": verification.get("checks") or {},
            "failed": verification.get("failed") or [],
            "reason": verification.get("reason"),
            "details": verification.get("details") or {},
        },
        retrieval_notes=retrieval.notes,
    )
    if log:
        _log_query(query, hits, retrieval, resp)
    return resp


def _citation_hits(hits: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    from marine_docs.ranking import preferred_evidence_hits

    return preferred_evidence_hits(hits, query, limit=8)


def _citation(h: dict[str, Any], retrieval: RetrievalResult) -> dict[str, Any]:
    """Procedure/plate code first; page numbers are supplementary (spec §6.1)."""
    return {
        "manual": retrieval.manual_title,
        "ref": h.get("citation_ref"),
        "procedure_no": h.get("procedure_no"),
        "plate_no": h.get("plate_no"),
        "citation_key": h.get("citation_key"),
        "edition": h.get("edition"),
        "doc_code": h.get("doc_code"),
        "page_printed": h.get("page_number_printed"),
        "page": h.get("page"),
        "section": h.get("section_path"),
        "component": h.get("component_title"),
        "action": h.get("action_title"),
        "panel_index": h.get("panel_index"),
        "drawing_code": h.get("drawing_code"),
        "step": h.get("linked_step_number"),
        "source": h.get("source"),
        "rule": h.get("priority_rule"),
        "image_path": h.get("figure_image_path"),
        "snippet": (h.get("text") or "")[:240],
    }


def _diagrams(hits: list[dict[str, Any]], query: str = "") -> list[dict[str, Any]]:
    from marine_docs.ranking import (
        _hit_matches_negative,
        _negative_tokens,
        _positive_query_text,
        _procedure_stem,
        _query_tokens,
        _title_overlap,
        query_references,
    )

    source = _positive_query_text(query)
    qtoks = _query_tokens(source)
    neg = _negative_tokens(query)
    refs = query_references(query)
    named = {
        *(refs.get("plate_nos") or []),
        *(refs.get("drawing_codes") or []),
        *(refs.get("doc_codes") or []),
        *(refs.get("citation_keys") or []),
    }
    named = {str(n).upper() for n in named if n}
    named_stems = {
        stem
        for stem in (_procedure_stem(p) for p in (refs.get("procedure_nos") or []))
        if stem
    }
    named_chaps = {s[:3] for s in named_stems}

    preferred = []
    by_title = sorted(hits, key=lambda h: -_title_overlap(h, qtoks))
    for h in by_title[:12]:
        preferred.append((h.get("plate_no") or "").upper())
        preferred.append((h.get("doc_code") or "").upper())
        preferred.append((h.get("citation_key") or "").upper())
    preferred = [p for p in preferred if p]

    code_overlap: dict[str, int] = {}
    for h in hits:
        code = (h.get("plate_no") or h.get("doc_code") or h.get("citation_key") or "").upper()
        if len(code) >= 5:
            code_overlap[code] = max(code_overlap.get(code, 0), _title_overlap(h, qtoks))

    def _plate_code(h: dict[str, Any]) -> str:
        return (h.get("plate_no") or h.get("doc_code") or h.get("citation_key") or "").upper()

    def _key(h: dict[str, Any]) -> tuple:
        blob = " ".join(
            [
                h.get("drawing_code") or "",
                h.get("plate_no") or "",
                h.get("doc_code") or "",
                h.get("citation_key") or "",
                h.get("component_title") or "",
                " ".join(h.get("section_path") or []),
                h.get("citation_ref") or "",
            ]
        )
        blob_u = blob.upper().replace(" ", "").replace("-", "")
        named_hit = 1 if any(n.replace(" ", "").replace("-", "") in blob_u for n in named if len(n) >= 5) else 0
        pref_hit = 1 if any(p.replace(" ", "").replace("-", "") in blob_u for p in preferred) else 0
        code = _plate_code(h)
        overlap = max(_title_overlap(h, qtoks), code_overlap.get(code, 0))
        named_rule = 1 if h.get("priority_rule") in {"exact_identifier", "plate_figure"} else 0
        return (-named_hit, -named_rule, -overlap, -pref_hit, 0 if h.get("pinned") else 1)

    scored = [
        h
        for h in hits
        if h.get("figure_image_path") and not _hit_matches_negative(h, neg)
    ]
    filtered: list[dict[str, Any]] = []
    for h in scored:
        proc = _procedure_stem(h.get("procedure_no") or h.get("citation_key") or "")
        if (
            named_stems
            and proc
            and proc[:3] in named_chaps
            and proc not in named_stems
        ):
            continue
        filtered.append(h)
    scored = filtered or scored
    scored.sort(key=_key)
    if scored:
        best_code = _plate_code(scored[0])
        if best_code:
            scored = [h for h in scored if _plate_code(h) == best_code] + [
                h for h in scored if _plate_code(h) != best_code
            ]
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for h in scored:
        key = h.get("figure_image_path")
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "label": h.get("drawing_code")
                or h.get("plate_no")
                or h.get("doc_code")
                or h.get("figure_label")
                or h.get("citation_ref")
                or f"page-{h.get('page')}",
                "ref": h.get("citation_ref"),
                "page": h.get("page"),
                "panel_index": h.get("panel_index"),
                "step": h.get("linked_step_number"),
                "section": " > ".join(h.get("section_path") or []),
                "image_path": key,
                "url": f"/api/figures?key={quote(key, safe='')}",
            }
        )
        if len(out) >= 6:
            break
    return out


def _log_query(
    query: str,
    hits: list[dict[str, Any]],
    retrieval: RetrievalResult,
    resp,
) -> None:
    try:
        from uuid import UUID

        from psycopg.types.json import Jsonb

        ids = []
        for h in hits:
            try:
                ids.append(UUID(str(h["element_id"])))
            except Exception:
                continue
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO query_log (
                        query_text, router_labels, retrieved_element_ids,
                        fusion_method, verification_result, failed_checks,
                        final_answer, abstained
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        query,
                        list(retrieval.notes.get("paths_fired") or []),
                        ids or None,
                        "rrf+section7",
                        Jsonb(resp.verification),
                        resp.verification.get("failed") or [],
                        (resp.answer or "")[:8000],
                        resp.abstained,
                    ),
                )
            conn.commit()
    except Exception:
        logger.exception("Failed to write query_log")
