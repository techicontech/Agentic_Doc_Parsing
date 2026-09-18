# Configuration

Copy `.env.example` → `.env` and fill in values. All settings are read via `marine_docs.config`.

## LiteLLM / models

| Variable | Description | Example |
|----------|-------------|---------|
| `LITELLM_API_BASE` | Proxy OpenAI-compatible base URL | `https://proxy.example/v1` |
| `LITELLM_API_KEY` | Proxy API key | *(secret)* |
| `LLM_MODEL` | Chat model id on the proxy | `claude-haiku` |
| `OCR_BACKEND` | `claude` \| `mistral` \| `skip` | `claude` |
| `OCR_VISION_MODEL` | Vision model for diagram OCR | `claude-haiku` |
| `MISTRAL_OCR_MODEL` | Used only if `OCR_BACKEND=mistral` | `mistral/mistral-ocr-latest` |
| `AGENTIC_ENABLED` | Use the agentic chat pipeline | `true` |
| `PAGE_ROUTER_LLM_ENABLED` | Page-router agent on ambiguous pages | `true` |
| `PAGE_ROUTER_LLM_MAX_PAGES` | Cap on router escalations per ingest | `60` |

## Docling

| Variable | Description | Example |
|----------|-------------|---------|
| `DOCLING_ENABLED` | Enable Docling parse path | `true` |
| `DOCLING_DEVICE` | `cuda` \| `cpu` \| `auto` | `cuda` |
| `DOCLING_BATCH_SIZE` | Pages per Docling batch | `10` |

## PostgreSQL

| Variable | Description | Default (dev) |
|----------|-------------|----------------|
| `POSTGRES_HOST` | Host | `localhost` |
| `POSTGRES_PORT` | Port | `5433` |
| `POSTGRES_DB` | Database name | `marine_docs` |
| `POSTGRES_USER` | User | `marine` |
| `POSTGRES_PASSWORD` | Password | `marine_dev` |

## MinIO / S3

| Variable | Description | Default (dev) |
|----------|-------------|----------------|
| `MINIO_ENDPOINT` | Host:port | `localhost:9000` |
| `MINIO_ACCESS_KEY` | Access key | `minioadmin` |
| `MINIO_SECRET_KEY` | Secret key | `minioadmin` |
| `MINIO_BUCKET` | Bucket for figures | `marine-figures` |
| `MINIO_SECURE` | HTTPS | `false` |

## Notes

- Dev defaults in `.env.example` are for local only — change passwords before any shared deployment.
- UI and API must share the same Postgres + MinIO instance that ingest wrote to.
- Restart API/UI after changing `.env`. **Re-ingest only** if parse/OCR settings or the PDF changed.

## Schema migrations

`scripts/init_db.py` applies, in order:

1. `sql/001_schema.sql` — only if `manuals` is missing
2. `sql/002_milestone2.sql` — procedure/plate/edition/panel columns (idempotent)
3. `sql/003_generic_citations.sql` — `citation_convention`, `citation_key`, `citation_fields` + backfill (idempotent)

Clearing the corpus from the UI and re-uploading a PDF rebuilds knowledge rows against the current schema; it does not drop the schema itself.
