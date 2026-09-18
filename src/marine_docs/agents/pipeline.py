"""Deterministic chat pipeline that calls agents at fixed points (spec §3, §4).

Control flow is Python, not an orchestrator agent: the sequence of steps is the same
for every query, and only the steps that need judgment are LLM agents. That is what
makes an abstention reproducible — rerun the query, get the same evidence set and the
same verification result.

    Router (agent)
      -> Lexical (tool) | Structural (agent) | Visual (tool)   [parallel]
      -> RRF fusion (tool)
      -> Section 7 ranking rules 1-3 (tool)
      -> Rerank (lightweight model call)
      -> Section 7 rule 4: pinned evidence stays pinned (tool)
      -> Verification gate (tool, 6 checks)
      -> Answer synthesis (agent) or Abstain
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from marine_docs.agents.nodes import (
    finalize_order,
    fuse_paths,
    prioritize,
    rerank_hits,
    route_query,
    run_lexical,
    run_structural,
    run_visual,
    synthesize_answer,
    verify_hits,
)

logger = logging.getLogger(__name__)

APP_NAME = "marine_docs_intelligence"


def execute_pipeline(query: str, *, equipment: str | None) -> dict[str, Any]:
    routed = route_query(query)
    labels = routed["labels"]

    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="retrieval") as pool:
        lexical_future = pool.submit(run_lexical, query, labels)
        structural_future = pool.submit(run_structural, query, labels)
        visual_future = pool.submit(run_visual, query, labels)
        lexical = _resolve(lexical_future, "lexical")
        structural = _resolve(structural_future, "structural")
        visual = _resolve(visual_future, "visual")

    fused = fuse_paths(lexical, structural, visual)
    prioritized, priority_notes = prioritize(query, fused)
    reranked = rerank_hits(query, prioritized)
    ranked = finalize_order(reranked, query=query)
    verification = verify_hits(query, ranked, equipment=equipment)
    answer = synthesize_answer(query, ranked, verification, equipment=equipment)

    return {
        "query": query,
        "equipment": equipment,
        "router_labels": labels,
        "router_reason": routed["reason"],
        "lexical_hits": lexical,
        "structural_hits": structural,
        "visual_hits": visual,
        "fused_hits": fused,
        "ranked_hits": ranked,
        "priority_notes": priority_notes,
        "verification": verification,
        "final_answer": answer,
        "abstained": not bool(verification.get("passed")),
        "orchestration": "deterministic_pipeline_with_adk_agents",
    }


def _resolve(future, label: str) -> list[dict[str, Any]]:
    try:
        return future.result()
    except Exception:
        logger.exception("%s path failed; continuing with remaining paths", label)
        return []
