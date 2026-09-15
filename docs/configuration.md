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

## Docling

| Variable | Description | Example |
|----------|-------------|---------|
| `DOCLING_ENABLED` | Enable Docling parse path | `true` |
| `DOCLING_DEVICE` | `cuda` \| `cpu` \| `auto` | `cuda` |

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
- The UI and API both expect the same Postgres + MinIO instance that ingest wrote to.
- Restart API/UI after changing `.env`; **re-ingest only** if parse/OCR settings or the PDF changed.
