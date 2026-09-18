"""Chat pipeline: a deterministic Python sequence that calls LLM agents where
judgment is needed (spec §3, §4)."""

from marine_docs.agents.pipeline import APP_NAME, execute_pipeline
from marine_docs.agents.runner import run_agentic_query

__all__ = ["APP_NAME", "execute_pipeline", "run_agentic_query"]
