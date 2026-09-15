# HTTP API

Base URL (dev): `http://127.0.0.1:8000`

Interactive docs: `http://127.0.0.1:8000/docs` (FastAPI Swagger).

## Health / status

### `GET /status`

Ingest and corpus readiness (whether data is present, last job progress).

## Ingest

### `POST /upload`

Multipart upload of a PDF. Stores under `uploads/` and returns a path/id for ingest.

### `POST /ingest`

Start (or resume) ingest for the uploaded manual. Body/query may include options such as clear-previous.

### `GET /ingest/progress`

Percent complete and phase (classify / Docling / OCR / knowledge build). Mirrored in UI and `logs/`.

## Chat

### `POST /ask`

```json
{
  "query": "What is the cylinder liner bore diameter?",
  "equipment_context": "S50MC-C"
}
```

Response (shape):

```json
{
  "answer": "...",
  "abstained": false,
  "citations": [
    {
      "manual": "...",
      "page": 73,
      "doc_code": "D10301",
      "section": "Chapter 903 > Cylinder Liner"
    }
  ],
  "diagrams": [
    {
      "page": 65,
      "url": "/figures/...",
      "label": "..."
    }
  ],
  "verification": { "passed": true, "checks": {} },
  "retrieval_notes": {}
}
```

When verification fails, `abstained` is `true` and `answer` explains what evidence was missing.

## Figures

Figure URLs returned in chat point at API/MinIO-backed paths served for the UI diagram strip.

## CLI helpers

| Script | Purpose |
|--------|---------|
| `scripts/run_api.py` | Start uvicorn |
| `scripts/run_ingest.py` | CLI ingest |
| `scripts/init_db.py` | Apply schema |
| `scripts/reset_db.py` | Wipe tables (destructive) |
| `scripts/ask.py` | One-shot ask from terminal |
| `scripts/run_ui.py` | Optional Gradio/legacy UI launcher |
