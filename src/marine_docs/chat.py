"""Answer / abstain orchestration for the chatbot UI / API."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

from marine_docs.db import connect
from marine_docs.llm import chat_completion, llm_ready
from marine_docs.retrieval import RetrievalResult, retrieve
from marine_docs.verify import VerificationResult, verify

logger = logging.getLogger(__name__)


@dataclass
class ChatResponse:
    answer: str
    abstained: bool
    citations: list[dict[str, Any]] = field(default_factory=list)
    diagrams: list[dict[str, Any]] = field(default_factory=list)
    verification: dict[str, Any] = field(default_factory=dict)
    retrieval_notes: dict[str, Any] = field(default_factory=dict)


SYSTEM_PROMPT = """You are a technical manual assistant for ingested maintenance documents.
You help with specs, procedures, troubleshooting, diagrams, and any question answerable from the evidence.
Answer ONLY from the provided evidence. If evidence is insufficient, say you cannot verify.
Cite the manual's own reference: Procedure or Plate number with its Edition. Printed and
PDF page numbers are supplementary context, never the primary citation.
Be precise with numbers, units, and procedure steps. Do not invent values.
If diagrams/plates are listed in evidence, mention them so the user can inspect the images.

Reading rules that apply to any manual:
- A value written as N above/below a named reference (adjustment sheet, recorded value,
  as-fitted, original) is a delta, not an absolute clearance or limit.
- When both a numeric data sheet and a checking procedure for the same component are
  in evidence, apply the procedure if/then first. A limits table alone is not the full decision.
- Unmarked checklist boxes are not requirements; only ticked/selected items apply.
- If two numeric values for the same quantity disagree, report both with citations.
- Arithmetic honesty: never say A exceeds B if A is less than B. If a measured
  value is below an inspect/replace threshold, do not apply the exceeded-branch
  action. Do not feed a measurement into a check that uses a different quantity.
- If a criteria table maps a measurement band to a replacement type, that mapping
  is the decision. A clearance stated as N above/below a named reference is a
  delta: compare N only to the procedure's delta threshold, not to min/max bands.
- Unmarked checklist lines are not required; do not copy ticks across components.
  Read only the checklist under the citation for the component that was asked about.
- If evidence equates two names, they are the same item. Do not invert that.
- Do not abstain when the measured value is below a threshold — that is keep/OK.
- Apply the if/then list row whose band contains the stated measurement.
- Prefer the component whose title shares the most words with the question. A
  shared table-row code on a different component is not the source.
- Quote the named plate/drawing and its OCR callouts; a sibling plate is not the diagram.
"""


def answer_query(
    query: str,
    *,
    equipment_context: str | None = None,
    log: bool = True,
) -> ChatResponse:
    try:
        from marine_docs.config import get_settings
        from marine_docs.agents.runner import run_agentic_query

        if get_settings().agentic_enabled:
            return run_agentic_query(
                query, equipment_context=equipment_context, log=log
            )
    except Exception:
        logger.exception("Agentic pipeline failed; using Milestone-1 retrieve")
    retrieval = retrieve(query)
    verification = verify(query, retrieval, query_equipment=equipment_context)

    citations = [_citation(ev, retrieval) for ev in retrieval.evidences[:10]]
    diagrams = _diagrams(retrieval)

    if not verification.passed:
        resp = ChatResponse(
            answer=_abstain_message(verification, retrieval),
            abstained=True,
            citations=citations,
            diagrams=diagrams,
            verification={
                "passed": False,
                "checks": verification.checks,
                "failed": verification.failed,
                "reason": verification.reason,
                "details": verification.details,
            },
            retrieval_notes=retrieval.notes,
        )
        if log:
            _log_query(query, retrieval, verification, resp)
        return resp

    if not llm_ready():
        blocks = []
        for ev in retrieval.evidences[:5]:
            path = " > ".join(ev.section_path or [])
            blocks.append(
                f"[p.{ev.page} | {ev.doc_code or '-'} | {path}]\n{(ev.text or '')[:700]}"
            )
        answer = (
            "LLM credentials not configured — showing retrieved evidence only.\n\n"
            + "\n\n---\n\n".join(blocks)
        )
        resp = ChatResponse(
            answer=answer,
            abstained=False,
            citations=citations,
            diagrams=diagrams,
            verification={
                "passed": True,
                "checks": verification.checks,
                "failed": [],
                "details": verification.details,
            },
            retrieval_notes=retrieval.notes,
        )
        if log:
            _log_query(query, retrieval, verification, resp)
        return resp

    evidence_blob = _format_evidence(retrieval)
    diagram_note = ""
    if diagrams:
        diagram_note = (
            "\nDiagrams available to the user (already attached in UI):\n"
            + "\n".join(
                f"- {d.get('label') or d.get('doc_code')} page {d.get('page')} ({d.get('section')})"
                for d in diagrams
            )
        )

    conflict_note = ""
    if verification.details.get("conflicts"):
        conflict_note = (
            "\nNumeric values in evidence disagree: "
            + "; ".join(verification.details["conflicts"][:6])
            + ". Report each value with its citation.\n"
        )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Equipment context: {equipment_context or retrieval.fleet_model}\n"
                f"Manual: {retrieval.manual_title} ({retrieval.revision})\n\n"
                f"Question: {query}\n\n"
                f"Evidence:\n{evidence_blob}\n"
                f"{diagram_note}{conflict_note}\n"
                "Write a concise cited answer. Mention relevant diagrams by page/doc_code."
            ),
        },
    ]
    try:
        answer = chat_completion(messages)
    except Exception as exc:
        logger.exception("LiteLLM completion failed")
        answer = f"Model call failed via LiteLLM: {exc}"

    resp = ChatResponse(
        answer=answer,
        abstained=False,
        citations=citations,
        diagrams=diagrams,
        verification={
            "passed": True,
            "checks": verification.checks,
            "failed": [],
            "details": verification.details,
        },
        retrieval_notes=retrieval.notes,
    )
    if log:
        _log_query(query, retrieval, verification, resp)
    return resp


def _citation(ev, retrieval: RetrievalResult) -> dict[str, Any]:
    """Procedure/plate code first; page numbers are supplementary (spec §6.1)."""
    return {
        "manual": retrieval.manual_title,
        "ref": ev.citation_ref,
        "procedure_no": ev.procedure_no,
        "plate_no": ev.plate_no,
        "edition": ev.edition,
        "doc_code": ev.doc_code,
        "page_printed": ev.page_number_printed,
        "page": ev.page,
        "section": ev.section_path,
        "component": ev.component_title,
        "action": ev.action_title,
        "panel_index": ev.panel_index,
        "drawing_code": ev.drawing_code,
        "step": ev.linked_step_number,
        "source": ev.source,
        "image_path": ev.figure_image_path,
        "snippet": (ev.text or "")[:240],
    }


def _diagrams(retrieval: RetrievalResult) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for ev in retrieval.evidences:
        if not ev.figure_image_path:
            continue
        key = ev.figure_image_path
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "label": ev.drawing_code or ev.figure_label or ev.citation_ref,
                "ref": ev.citation_ref,
                "page": ev.page,
                "panel_index": ev.panel_index,
                "step": ev.linked_step_number,
                "section": " > ".join(ev.section_path or []),
                "image_path": key,
                "url": f"/api/figures?key={quote(key, safe='')}",
            }
        )
    return out[:6]


def _format_evidence(retrieval: RetrievalResult) -> str:
    parts = []
    for i, ev in enumerate(retrieval.evidences[:8], 1):
        path = " > ".join(ev.section_path or [])
        fig = f" drawing={ev.drawing_code}" if ev.figure_image_path else ""
        parts.append(
            f"[{i}] {ev.citation_ref} | kind={ev.section_kind} | {path} "
            f"| printed page={ev.page_number_printed or '-'} | pdf page={ev.page}{fig}\n{ev.text}"
        )
    return "\n\n".join(parts)


def _abstain_message(verification: VerificationResult, retrieval: RetrievalResult) -> str:
    return (
        "ABSTAIN — I cannot provide a verified answer.\n\n"
        f"Reason: {verification.reason}\n"
        f"Failed checks: {', '.join(verification.failed) or 'none'}\n"
        f"Registered equipment: {retrieval.fleet_model or 'unknown'}\n"
        f"Corpus: {retrieval.manual_title or 'none'}"
    )


def _log_query(
    query: str,
    retrieval: RetrievalResult,
    verification: VerificationResult,
    resp: ChatResponse,
) -> None:
    try:
        from psycopg.types.json import Jsonb

        ids = [ev.element_id for ev in retrieval.evidences]
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
                        "lexical+structural+figure",
                        Jsonb(resp.verification),
                        verification.failed,
                        resp.answer[:8000],
                        resp.abstained,
                    ),
                )
            conn.commit()
    except Exception:
        logger.exception("Failed to write query_log")
