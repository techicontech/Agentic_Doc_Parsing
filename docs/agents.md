# Agents and tools

The pipeline is a **deterministic Python sequence that calls agents at fixed points** — not a root orchestrator agent making its own control-flow decisions. Every model call goes through LiteLLM; the ADK layer is `LlmAgent` + the `LiteLlm` wrapper. No Vertex AI, no Google-hosted Agent Engine.

## Agent vs tool

A step is an **agent** only when it needs judgment. Everything reproducible is a plain Python function.

| Step | Type | Where |
|------|------|-------|
| Convention detector (ingest) | Agent (LLM), once per manual | `convention.detect_convention` |
| Page router (ingest) | Agent — ambiguous pages only | `page_router._ask_router_agent` |
| Docling extraction | Tool | `parsers/docling_parser.py` |
| Diagram/plate OCR + captioning | Agent (vision) | `parsers/vision_ocr.py` |
| Panel geometry + step pairing | Tool | `panels.py` |
| Persistence (Postgres/MinIO) | Tool | `knowledge/builder.py` |
| Query router | Agent | `agents/nodes.route_query` |
| Lexical retrieval (FTS) | Tool | `retrieval.lexical_path` |
| Structural tree navigation | Agent | `agents/nodes.run_structural` |
| Visual retrieval | Tool in v1 (figures table) | `retrieval.visual_path` |
| Candidate fusion (RRF) | **Tool — never an agent** | `retrieval.rrf_fuse` |
| Ranking priority | Tool | `ranking.apply_priority` |
| Rerank | Lightweight model call, non-agentic | `agents/nodes.rerank_hits` |
| Verification gate | **Tool — never an agent** | `verify.verify` |
| Answer synthesis | Agent | `agents/nodes.synthesize_answer` |

Fusion and the verification gate must never be wrapped as agents. They make abstention auditable: rerun a query, get the same evidence set and the same pass/fail.

## Chat sequence

```
Router (agent)
  -> Lexical (tool) | Structural (agent) | Visual (tool)   [parallel threads]
  -> RRF fusion (tool)
  -> Ranking: exact code pin, component phrases, kind pairs, siblings,
     contrast clauses, step-panel pull, one cross-ref hop (tool)
  -> Rerank (model call)
  -> Pinned evidence stays pinned (tool)
  -> Verification gate (tool, 6 checks)
  -> Answer synthesis (agent)  OR  Abstain
```

Implemented in `agents/pipeline.execute_pipeline`. Retrieval paths run in a thread pool; a failing path is logged and skipped. The API layer (`agents/runner.py`) then builds the citation window and diagram strip from ranked hits.

## Ingest sequence

Ingest stays a deterministic batch process. Chat changes never require a re-ingest; schema or parser changes do.

```
PDF -> convention detector (agent once; heuristic fallback)
    -> page router (heuristic; agent only on ambiguous pages)
    -> Docling (clean text pages)
    -> diagram pages: geometry split → vision per crop
       (whole-page vision only if the split is messy, or the page is a plate)
    -> panel geometry on Docling pages (drawing codes, step pairing)
    -> typed cross-references
    -> Postgres + MinIO
```

## Page router cost control

A model on every page of a 600–1000 page manual costs more than it resolves. The heuristic decides first; the agent runs only for genuinely mixed pages. `PAGE_ROUTER_LLM_MAX_PAGES` caps escalations; `PAGE_ROUTER_LLM_ENABLED=false` disables them.

## What “chunks” are

Retrieved items are **`elements` rows**: paragraph, table, figure panel or plate, each with a page, optional bbox, citation keys, and the extractor that produced it. There is no sliding token-window chunker.

## Provenance

- `elements.extractor_name`, `extractor_version`, `bbox`
- `element_provenance`: source PDF, revision, page, bbox, extractor version, timestamp

## Visual path

v1 uses the `figures` table plus MinIO panel crops, linked to sections and steps. ColPali/ColQwen visual semantic search is Phase 2.

## Fallback

If `google-adk` is unavailable or an agent run fails, the same prompt goes through `marine_docs.llm.chat_completion` (LiteLLM). Set `AGENTIC_ENABLED=false` to force the older single-retrieve path in `chat.answer_query`.
