"""Pipeline steps for chat (spec §3, §4).

Agents (LLM, via ADK + LiteLLM) — judgment is required:
  * query router      — multi-label intent classification
  * tree navigator    — decides which sections/procedures apply
  * answer synthesis  — composes a grounded, cited answer

Tools (plain functions) — deterministic, reproducible, auditable:
  * lexical retrieval, visual retrieval over the figures table
  * candidate fusion (RRF) and the Section 7 ranking rules
  * rerank (a lightweight scoring model call, not an agent)
  * the verification gate

Fusion and the verification gate must never become agents: they are what make
abstention auditable and reproducible.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from marine_docs.agents import llm_agents
from marine_docs.agents.jsonutil import parse_json_object
from marine_docs.llm import chat_completion, llm_ready
from marine_docs.ranking import apply_priority, finalize_order
from marine_docs.retrieval import (
    dict_to_evidence,
    get_fleet_context,
    lexical_path,
    list_document_tree,
    overlap_score,
    rrf_fuse,
    structural_path_from_picks,
    visual_path,
)
from marine_docs.verify import verify

logger = logging.getLogger(__name__)

VALID_LABELS = ("lexical", "structural", "visual")
_VISUAL_HINTS = (
    "diagram", "figure", "plate", "drawing", "panel", "callout", "sketch",
    "show me", "show the", "illustration",
)

RERANK_PROMPT = """You rerank retrieved manual elements for a technical question.
Score only on whether the snippet can answer the question.
For keep/replace, wear-limit, or criteria questions, keep both the numeric data
and the checking/inspection procedure for the same component near the top.
When the question mentions shut-off, precautions, mass, or checklists, keep those
rows even if they are short. When it mentions hours or overhaul interval, keep
maintenance-schedule rows. Do not fill the list with one procedure code.
If the question names a plate, drawing, panel, or callout, keep that figure and
any OCR/callout table for it near the top. Prefer the component whose title
shares the most words with the question; a sibling plate or data sheet for a
different component is worse even if it shares a table-row code.
Return ONLY JSON: {"order": [1, 4, 2]}
where numbers are candidate indices (1-based), best first. Include every index once."""


# --- Agent: query router -------------------------------------------------------

def route_query(query: str) -> dict[str, Any]:
    """
    Agent: Query Router (chat-time, multi-label)

    Input: user question string.
    Output: {"labels": list[Literal["lexical","structural","visual"]], "reason": str}
    Model: settings.llm_model via marine_docs.llm / llm_agents (LiteLLM). Overridable.
    Failure mode: on LLM error or unparseable labels, fire all three retrieval
      paths rather than silently returning zero paths.
    """
    labels = list(VALID_LABELS)
    reason = "default_all_paths"
    if llm_ready():
        try:
            parsed = parse_json_object(
                llm_agents.ask(llm_agents.QUERY_ROUTER, query, max_tokens=200)
            )
            got = [
                str(x).lower()
                for x in (parsed.get("labels") or [])
                if str(x).lower() in VALID_LABELS
            ]
            if got:
                labels = list(dict.fromkeys(got))
                reason = str(parsed.get("reason") or "router")
        except Exception:
            logger.exception("query router agent failed; firing all paths")
    q = query.lower()
    if any(h in q for h in _VISUAL_HINTS) and "visual" not in labels:
        labels = list(labels) + ["visual"]
        reason = (reason + "+visual_hint")[:240]
    return {"labels": labels, "reason": reason}


# --- Tool: lexical retrieval ---------------------------------------------------

def run_lexical(query: str, labels: list[str]) -> list[dict[str, Any]]:
    if "lexical" not in labels:
        return []
    return lexical_path(query)


# --- Agent: structural tree navigation ----------------------------------------

def run_structural(query: str, labels: list[str]) -> list[dict[str, Any]]:
    """
    Agent: Structural reasoning (tree / citation-key navigation)

    Input: question + compact section catalog from list_document_tree().
    Output: list of evidence dicts (same shape as lexical hits).
    Model: settings.llm_model via LiteLLM. Overridable.
    Failure mode: on LLM error, return [] for this path; fusion still runs on
      whatever lexical/visual returned (router already defaulted to all paths).
    """
    if "structural" not in labels:
        return []
    codes: list[str] = []
    titles: list[str] = []
    procedures: list[str] = []
    tree = list_document_tree()
    if llm_ready() and tree:
        catalog = "\n".join(
            f"- {row.get('doc_code') or '-'} | proc {row.get('procedure_no') or '-'} "
            f"| {row.get('kind')} | {row.get('component') or ' > '.join(row.get('path') or [])} "
            f"| {row.get('action') or '-'} | p.{row.get('pages')}"
            for row in tree[:250]
        )
        try:
            parsed = parse_json_object(
                llm_agents.ask(
                    llm_agents.TREE_NAVIGATOR,
                    f"Question: {query}\n\nCatalog:\n{catalog}",
                    max_tokens=400,
                )
            )
            codes = [str(c) for c in (parsed.get("doc_codes") or [])][:8]
            titles = [str(t) for t in (parsed.get("titles") or [])][:8]
            procedures = [str(p) for p in (parsed.get("procedure_nos") or [])][:8]
        except Exception:
            logger.exception("tree navigator agent failed; using heuristic structural search")
    return structural_path_from_picks(
        query, doc_codes=codes, titles=titles, procedure_nos=procedures
    )


# --- Tool: visual retrieval over the figures table (ColPali is Phase 2) --------

def run_visual(query: str, labels: list[str]) -> list[dict[str, Any]]:
    q = query.lower()
    if "visual" not in labels and not any(h in q for h in _VISUAL_HINTS):
        return []
    return visual_path(query)


# --- Tool: candidate fusion (RRF). Never an agent. ----------------------------

def fuse_paths(
    lexical: list[dict[str, Any]],
    structural: list[dict[str, Any]],
    visual: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    lists = [lst for lst in (lexical, structural, visual) if lst]
    if not lists:
        return []
    return rrf_fuse(lists, k=60, limit=24)


# --- Tool: Section 7 ranking priority ----------------------------------------

def prioritize(query: str, fused: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fleet = get_fleet_context() or {}
    return apply_priority(
        query,
        fused,
        manual_id=fleet.get("manual_id"),
        convention=fleet.get("citation_convention"),
        limit=24,
    )


# --- Tool: rerank (lightweight model call, not an agent) ---------------------

def rerank_hits(query: str, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(hits) <= 1 or not llm_ready():
        return hits
    lines = []
    for i, item in enumerate(hits[:16], 1):
        path = " > ".join(item.get("section_path") or [])
        snippet = (item.get("text") or "")[:280].replace("\n", " ")
        lines.append(
            f"{i}. {item.get('citation_ref') or ''} {item.get('section_kind')} {path}\n{snippet}"
        )
    try:
        parsed = parse_json_object(
            chat_completion(
                [
                    {"role": "system", "content": RERANK_PROMPT},
                    {
                        "role": "user",
                        "content": f"Question: {query}\n\nCandidates:\n" + "\n\n".join(lines),
                    },
                ],
                temperature=0.0,
                max_tokens=400,
            )
        )
        indexed = {i: hits[i - 1] for i in range(1, min(17, len(hits) + 1))}
        ranked: list[dict[str, Any]] = []
        seen: set[int] = set()
        for raw in parsed.get("order") or []:
            try:
                n = int(raw)
            except (TypeError, ValueError):
                continue
            if n in indexed and n not in seen:
                ranked.append(indexed[n])
                seen.add(n)
        ranked.extend(indexed[n] for n in indexed if n not in seen)
        ranked.extend(hits[16:])
        return ranked
    except Exception:
        logger.exception("rerank call failed; keeping fused order")
        return hits


# --- Tool: verification gate. Never an agent. --------------------------------

def verify_hits(
    query: str,
    hits: list[dict[str, Any]],
    *,
    equipment: str | None,
) -> dict[str, Any]:
    from marine_docs.retrieval import RetrievalResult

    fleet = get_fleet_context() or {}
    retrieval = RetrievalResult(
        evidences=[dict_to_evidence(h) for h in hits if h.get("element_id")],
        fleet_model=fleet.get("model"),
        manual_title=fleet.get("title"),
        revision=fleet.get("revision"),
    )
    result = verify(query, retrieval, query_equipment=equipment)
    return {
        "passed": result.passed,
        "checks": result.checks,
        "failed": result.failed,
        "reason": result.reason,
        "details": result.details,
        "fleet_model": retrieval.fleet_model,
        "manual_title": retrieval.manual_title,
        "revision": retrieval.revision,
    }


# --- Agent: answer synthesis (only reached when verification passes) ---------

def synthesize_answer(
    query: str,
    hits: list[dict[str, Any]],
    verification: dict[str, Any],
    *,
    equipment: str | None,
) -> str:
    """
    Agent: Answer synthesis (chat-time)

    Input: question + ranked evidence dicts + verification result. Never called
      for generation when the verification gate failed (abstain string instead).
    Output: grounded answer string that cites evidence citation_key values.
    Model: settings.llm_model via LiteLLM. Overridable.
    Failure mode: if the LLM call fails, return an evidence-only dump rather
      than an ungrounded guess.
    """
    if not verification.get("passed"):
        return abstain_message(verification)
    if not llm_ready():
        blocks = [
            f"[{h.get('citation_ref') or 'page ' + str(h.get('page'))}]\n{(h.get('text') or '')[:700]}"
            for h in hits[:5]
        ]
        return (
            "LLM credentials not configured — showing retrieved evidence only.\n\n"
            + "\n\n---\n\n".join(blocks)
        )

    evidence = []
    q_l = (query or "").lower()
    ordered = list(hits)
    if any(k in q_l for k in ("minute", "hour", "drop")):
        lists = [h for h in hits if (h.get("element_type") or "").lower() == "list"]
        rest = [h for h in hits if (h.get("element_type") or "").lower() != "list"]
        ordered = lists + rest
    if any(k in q_l for k in ("plate", "diagram", "drawing", "callout", "show the", "show me")):
        visual = [
            h
            for h in ordered
            if (h.get("section_kind") or "").lower() == "plate"
            or h.get("drawing_code")
            or h.get("figure_image_path")
            or "callout" in (h.get("text") or "").lower()
        ]
        rest = [h for h in ordered if h not in visual]
        ordered = visual + rest
    qtoks = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", q_l)
    dist = [t for t in qtoks if len(t) >= 5]
    ordered.sort(
        key=lambda h: (
            0 if h.get("pinned") else 1,
            -overlap_score(
                h.get("text") or "",
                qtoks,
                distinctive=dist,
                element_type=h.get("element_type"),
            ),
        )
    )
    for i, h in enumerate(ordered[:12], 1):
        path = " > ".join(h.get("section_path") or [])
        printed = h.get("page_number_printed")
        panel = ""
        if h.get("panel_index"):
            panel = (
                f" panel={h.get('panel_index')} drawing={h.get('drawing_code') or '-'}"
                f" illustrates_step={h.get('linked_step_number') or '-'}"
            )
        evidence.append(
            f"[{i}] {h.get('citation_ref')} | kind={h.get('section_kind')} "
            f"| component={h.get('component_title') or '-'} "
            f"| action={h.get('action_title') or '-'} | {path}"
            f" | printed page={printed or '-'} | pdf page={h.get('page')}{panel}\n"
            f"{h.get('text') or ''}"
        )
    diagrams = [
        f"- {h.get('drawing_code') or h.get('figure_label') or h.get('citation_ref')}"
        f" (step {h.get('linked_step_number') or '-'}, pdf page {h.get('page')})"
        for h in hits
        if h.get("figure_image_path")
    ]
    diagram_note = "\nDiagram panels attached in the UI:\n" + "\n".join(diagrams[:6]) if diagrams else ""

    conflict_note = ""
    details = verification.get("details") or {}
    if details.get("conflicts"):
        conflict_note = (
            "\nNumeric values in evidence disagree: "
            + "; ".join(details["conflicts"][:6])
            + ". Report each value with its citation.\n"
        )

    try:
        return llm_agents.ask(
            llm_agents.ANSWER_SYNTHESIZER,
            f"Equipment context: {equipment or verification.get('fleet_model')}\n"
            f"Manual: {verification.get('manual_title')}\n\n"
            f"Question: {query}\n\n"
            f"Evidence:\n" + "\n\n".join(evidence) + f"\n{diagram_note}{conflict_note}\n"
            + (
                "Before you write exceeds/below/within, subtract the two numbers. "
                "If A − B is negative, A does not exceed B. "
                "A 50% reduction of original L means remaining L is still acceptable when "
                "it is greater than 0.5 × original. "
                "If several time-band rows are in evidence, use the row that contains the "
                "stated time, not a neighboring row. "
                "A unit number is a location, not a component type; follow the named "
                "symptoms (rotation, spring air, hard-faced seat) to the matching procedure. "
                "The first-line decision must match the if/then you computed: do not "
                "headline replace/scrap/overhaul if the numbers you compared are still "
                "inside the keep/OK band. "
                "Prefer evidence whose component title shares the most words with the "
                "question. A data sheet that only shares a table-row code (D13-01 style) "
                "with a different component is not the answer. "
                "If the question names a plate, drawing, or callout, quote that panel's "
                "OCR/callout table. The drawing caption's action "
                "(checking/overhaul/dismantling/mounting) is the action of that panel. "
                "Do not attach a sibling component's plate as the diagram."
            ),
            max_tokens=1200,
        )
    except Exception as exc:
        logger.exception("answer synthesis agent failed")
        return f"Model call failed via LiteLLM: {exc}"


def abstain_message(verification: dict[str, Any]) -> str:
    failed = ", ".join(verification.get("failed") or []) or "none"
    return (
        "ABSTAIN — I cannot provide a verified answer.\n\n"
        f"Reason: {verification.get('reason')}\n"
        f"Failed checks: {failed}\n"
        f"Registered equipment: {verification.get('fleet_model') or 'unknown'}\n"
        f"Corpus: {verification.get('manual_title') or 'none'}"
    )


__all__ = [
    "VALID_LABELS",
    "abstain_message",
    "finalize_order",
    "fuse_paths",
    "prioritize",
    "rerank_hits",
    "route_query",
    "run_lexical",
    "run_structural",
    "run_visual",
    "synthesize_answer",
    "verify_hits",
]
