# Architecture

## Goals

Turn large technical PDFs (maintenance manuals, parts books, procedure volumes) into a **structured, citable knowledge base** that a chat assistant can query without inventing numbers or steps.

## End-to-end flow

1. **Upload** — PDF stored under `uploads/` (gitignored).
2. **Page routing** — Classify pages as text-heavy vs diagram/plate (sparse text + images).
3. **Parse**
   - **Docling** — layout, tables, body text (CUDA when configured).
   - **Vision OCR** (Claude via LiteLLM) or **Mistral OCR** — diagram/plate pages.
4. **Structure** — Detect chapters, titles, and MAN-style codes (`D#####` data, `M#####` procedure, `P#####` plate, `A#####` schedule).
5. **Persist**
   - **Postgres** — manuals, sections, elements (with FTS), fleet registry, query logs.
   - **MinIO** — figure/plate images for the UI.
6. **Chat**
   - Retrieve (lexical FTS + structural title/phrase + optional figures).
   - Verify (equipment / scope / light conflict checks).
   - Answer via LiteLLM **or abstain** with citations + diagram thumbnails.

## Knowledge model (Postgres)

| Entity | Role |
|--------|------|
| `manuals` | Document identity, title, revision |
| `fleet_registry` | Current equipment ↔ manual binding |
| `sections` | Hierarchical path, `doc_code`, `section_kind` |
| `elements` | Text / table cells with `tsv` for FTS |
| `figures` | Image path in MinIO, caption, page |
| `query_logs` | Optional audit of asks + retrieval |

Schema: [`sql/001_schema.sql`](../sql/001_schema.sql).

## Retrieval strategy

No vector DB in Milestone 1. Retrieval is **hybrid symbolic**:

- **Lexical** — Postgres `tsvector` / `tsquery` over element text; phrase ILIKE on titles.
- **Structural** — Match component phrases (e.g. `main bearing`, `cylinder cover`) to section titles; prefer `data` vs `procedure` by query intent.
- **Figures** — Attached when the question needs diagrams/how-to, not for pure numeric lookups.

Ranking boosts are **generic** (intent tokens, section kind, phrase hits) — not hard-coded per sample question.

Details: [retrieval.md](retrieval.md).

## LLM boundary

All model calls go through **LiteLLM** (`LITELLM_API_BASE` + `LITELLM_API_KEY`):

- Chat completions for answers.
- Vision/OCR for plate pages when `OCR_BACKEND=claude` (or Mistral OCR when configured).

The app never requires direct Anthropic/Mistral SDK keys if the proxy is fully configured.

## UI

React + Vite (`web/`):

- **Ingest** — upload, progress %, clear-previous option.
- **Chat** — answer, abstain messaging, citations, diagram strip.

API: FastAPI (`marine_docs.api`).

## Non-goals (current)

- Full multi-agent ADK orchestration (future milestone).
- Embedding / vector search (can be added without changing the knowledge tables).
- Writing back into OEM manuals.
