# Architecture

## Goals

Turn large technical PDFs (maintenance manuals, parts books, procedure volumes) into a **structured, citable knowledge base**. The chat assistant must quote the manual's own identifiers and abstain when evidence is missing — it must not invent numbers or steps.

Manufacturer layout is **data**, not code. A convention detector runs once per new PDF and stores the citation pattern on that manual.

## End-to-end flow

### Ingest — deterministic batch

1. **Upload** — PDF stored under `uploads/` (gitignored). The UI ingest path **clears the previous corpus** then parses the new file.
2. **Convention detector** — samples headers/footers across the document (LLM, heuristic fallback). Writes `manuals.citation_convention` JSONB: citation regex, revision granularity (`per_section` or `per_document`), optional quote hint. Parsers do not branch on manufacturer name.
3. **Page routing** — heuristics classify text-heavy vs diagram/plate pages; the LLM page-router is consulted only for mixed pages, capped by `PAGE_ROUTER_LLM_MAX_PAGES`.
4. **Parse**
   - **Docling** — layout, tables, body text (`DOCLING_DEVICE=cuda` when configured).
   - **Diagram/plate pages** — geometry splits bordered panels; the vision OCR agent (LiteLLM) reads **each crop**. Whole-page vision only when the split is messy or the page is a full plate.
5. **Structure** — hierarchical sections, `component_title` / `action_title`, `section_kind` (data / procedure / plate / schedule / other). `citation_key` is filled from procedure, plate, or doc code; fallback is `p.` + printed page.
6. **Panels** — each independently bordered diagram becomes its own figure, with drawing code and the procedure step it illustrates. Never one figure per page when panels exist.
7. **Cross-references** — “see procedure / data / plate …” resolved into typed relationships.
8. **Persist**
   - **Postgres** — manuals, sections, elements (FTS), figures, relationships, provenance, fleet registry, query log.
   - **MinIO** — page renders and panel crops.

Chat/ranking code changes never require a re-ingest. Schema or parser/OCR changes do.

### Chat — deterministic pipeline calling agents

Implemented in `agents/pipeline.execute_pipeline`:

```
Router (agent)
  -> Lexical (tool) | Structural (agent) | Visual (tool)   [parallel]
  -> RRF fusion (tool) — merges duplicate element_ids and keeps figure_image_path
  -> Ranking rules (tool)
  -> Rerank (lightweight model call)
  -> Pinned evidence stays pinned
  -> Verification gate (tool, 6 checks)
  -> Answer synthesis (agent)  OR  Abstain
```

The API/UI then shapes **citations** (preferred component window) and **diagrams** (named plate/drawing, contrast clauses stripped). Fusion and the gate are plain functions. See [agents.md](agents.md) and [retrieval.md](retrieval.md).

## Knowledge model (Postgres)

| Entity | Role |
|--------|------|
| `manuals` | Identity, title, revision, `manufacturer`, `citation_convention` |
| `manual_revisions` | Revision labels and supersession |
| `fleet_registry` | Current equipment ↔ manual binding |
| `sections` | Path, `doc_code`, `section_kind`, `procedure_no`, `plate_no`, titles, `citation_key`, `citation_fields` |
| `elements` | Paragraph / table / figure / list rows with FTS, citation keys, panel fields |
| `figures` | MinIO path, caption, panel index, drawing code, step link |
| `element_relationships` | `references_procedure`, `references_data`, `illustrates_step`, … |
| `element_provenance` | PDF → revision → page → bbox → extractor version |
| `query_log` | Question, router labels, evidence ids, verification, answer |

Schema files (applied in order by `scripts/init_db.py`):

- [`sql/001_schema.sql`](../sql/001_schema.sql) — base tables
- [`sql/002_milestone2.sql`](../sql/002_milestone2.sql) — procedure/plate/edition/panel fields
- [`sql/003_generic_citations.sql`](../sql/003_generic_citations.sql) — `citation_convention`, `citation_key`, `citation_fields` + backfill

An already-ingested corpus does not need a full re-ingest after `003`; keys are backfilled from existing procedure/plate/doc codes.

### What the model encodes (manufacturer-agnostic)

These properties showed up on the S50MC-C test manual and are stored as fields, not as MAN-only `if` branches:

- **The quote key is a procedure/plate/doc code, not a PDF page number.** Page numbers are supplementary.
- **Several diagram panels per page is normal.** Each panel is its own figure.
- **Revision can be per section.** Two procedures in one PDF may have different valid editions. The gate fails only when the user *names* a different edition than the one stored on that procedure.
- **Cross-references are dense**, so they are resolved at ingest into typed edges.

## Retrieval strategy

No vector database in v1. Structure-aware retrieval, lexical FTS, identifier ranking, and a small amount of LLM judgment fit identifier-dense manuals better as the primary path. Dense retrieval is an optional later signal (`pgvector` in this same Postgres) only if paraphrased questions miss lexical/structural recall.

Ranking boosts are generic (intent tokens, section kind, title overlap, contrast/negation in the question). There are no hard-coded question→page maps.

Details: [retrieval.md](retrieval.md).

## LLM boundary

All model calls go through **LiteLLM** (`LITELLM_API_BASE` + `LITELLM_API_KEY`):

- Chat completions for routing, structural nav, rerank, answers, convention detection.
- Vision/OCR for plate pages when `OCR_BACKEND=claude` (or Mistral OCR when configured).

Direct Anthropic/Mistral SDK keys are not required if the proxy is fully configured.

## UI

React + Vite (`web/`):

- **Ingest** — PDF upload, progress %, status poll. Upload replaces the previous corpus.
- **Chat** — answer or abstain, citations, diagram strip (MinIO via `/api/figures`).

API: FastAPI (`marine_docs.api`) on `http://127.0.0.1:8000`.

## Non-goals (current)

- **No claim of 100% accuracy.** Grounded retrieval with citations and abstention is the product.
- **No root orchestrator agent.** Control flow stays deterministic Python.
- ColPali / ColQwen visual semantic search: Phase 2. v1 visual path is the `figures` table.
- A production cross-encoder reranker: optional later.
- Embedding / vector search (addable without changing knowledge tables).
- Writing back into OEM manuals.
