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


SYSTEM_PROMPT = """You are a marine technical manual assistant for MAN B&W engine maintenance docs.
You help with specs, procedures, troubleshooting, diagrams, and any question answerable from the evidence.
Answer ONLY from the provided evidence. If evidence is insufficient, say you cannot verify.
Always include citations as: Manual, page, section path, doc_code when available.
Be precise with numbers, units, and procedure steps. Do not invent values.
If diagrams/plates are listed in evidence, mention them so the user can inspect the images."""


def answer_query(
    query: str,
    *,
    equipment_context: str | None = "S50MC-C",
    log: bool = True,
) -> ChatResponse:
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
            verification={"passed": True, "checks": verification.checks, "failed": []},
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

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Equipment context: {equipment_context or retrieval.fleet_model}\n"
                f"Manual: {retrieval.manual_title} ({retrieval.revision})\n\n"
                f"Question: {query}\n\n"
                f"Evidence:\n{evidence_blob}\n"
                f"{diagram_note}\n\n"
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
        verification={"passed": True, "checks": verification.checks, "failed": []},
        retrieval_notes=retrieval.notes,
    )
    if log:
        _log_query(query, retrieval, verification, resp)
    return resp


def _citation(ev, retrieval: RetrievalResult) -> dict[str, Any]:
    return {
        "manual": retrieval.manual_title,
        "revision": retrieval.revision,
        "page": ev.page,
        "section": ev.section_path,
        "doc_code": ev.doc_code,
        "edition": ev.edition,
        "source": ev.source,
        "figure": ev.figure_label,
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
                "label": ev.figure_label or ev.doc_code or f"page-{ev.page}",
                "page": ev.page,
                "doc_code": ev.doc_code,
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
        fig = f" figure={ev.figure_label}" if ev.figure_image_path else ""
        parts.append(
            f"[{i}] page={ev.page} doc_code={ev.doc_code} kind={ev.section_kind} "
            f"path={path} source={ev.source}{fig}\n{ev.text}"
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
