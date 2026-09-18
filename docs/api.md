# HTTP API

Base URL (dev): `http://127.0.0.1:8000`

Interactive docs: `http://127.0.0.1:8000/docs` (FastAPI Swagger).

The React UI talks only to these routes (Vite proxies `/api` in dev).

## Health / status

### `GET /api/health`

Corpus readiness, current ingest job snippet, OCR/LiteLLM readiness.

### `GET /api/ingest/status`

Full ingest job: `status` (`idle` | `running` | `completed` | `failed`), `stage`, `percent`, `message`, `has_data`, `log_file`.

## Ingest

### `POST /api/ingest/upload`

Multipart PDF upload. Saves under `uploads/`, **clears any previous corpus**, then starts ingest on a background thread.

Returns 409 if ingest is already running.

Progress is polled via `/api/ingest/status`. File logs go under `logs/` (gitignored).

## Chat

### `POST /api/chat`

Blocked with 409 while ingest is running, and 400 if no corpus exists.

```json
{
  "message": "What is the cylinder liner bore diameter?",
  "equipment": "S50MC-C"
}
```

`equipment` is optional; blank uses the ingested fleet model.

Response shape:

```json
{
  "answer": "...",
  "abstained": false,
  "citations": [
    {
      "manual": "…",
      "ref": "Procedure 903-1.1 Edition 0286",
      "procedure_no": "903-1.1",
      "plate_no": null,
      "citation_key": "903-1.1",
      "edition": "0286",
      "doc_code": "D10301",
      "page_printed": "1",
      "page": 73,
      "section": ["Chapter 903", "Cylinder Liner"],
      "component": "Cylinder Liner",
      "action": "Data",
      "drawing_code": null,
      "step": null,
      "rule": "exact_identifier",
      "snippet": "..."
    }
  ],
  "diagrams": [
    {
      "label": "M90201-0285D06",
      "ref": "Procedure 902-1.2 Edition 0286",
      "page": 38,
      "panel_index": 2,
      "step": 5,
      "url": "/api/figures?key=..."
    }
  ],
  "verification": {
    "passed": true,
    "checks": {
      "equipment": true,
      "revision": true,
      "condition": true,
      "units": true,
      "scope": true,
      "conflicts": true
    },
    "failed": [],
    "details": {}
  },
  "retrieval_notes": {
    "paths_fired": ["lexical", "structural", "visual"],
    "fusion_method": "rrf",
    "orchestration": "deterministic_pipeline_with_adk_agents"
  }
}
```

`ref` / `citation_key` is the primary citation — the reference the manual asks readers to quote. `page` (PDF) and `page_printed` are supplementary. `rule` records which ranking rule brought an element into evidence.

When verification fails, `abstained` is `true`, `verification.failed` lists the failing checks, and `answer` explains what was missing. The answer model is not called.

## Figures

### `GET /api/figures?key=...`

PNG bytes from MinIO for the chat diagram strip. Keys come from `diagrams[].url`.

## CLI helpers

| Script | Purpose |
|--------|---------|
| `scripts/run_api.py` | FastAPI on `:8000` |
| `scripts/run_ingest.py` | CLI ingest |
| `scripts/init_db.py` | Apply `sql/001`–`003` + MinIO bucket |
| `scripts/reset_db.py` | Wipe tables (destructive, `--yes`) |
| `scripts/ask.py` | One-shot ask on the same chat path |
