# Agentic Doc Parsing

**Technical-manual document intelligence** — ingest large engineering PDFs, structure them into a queryable knowledge model, and answer maintenance questions with citations, diagrams, and safe abstention when evidence is missing.

Built for marine / industrial manuals (e.g. MAN B&W Volume II Maintenance). Works with **any** uploaded PDF through the same pipeline.

[Repository](https://github.com/techicontech/Agentic_Doc_Parsing)

---

## Highlights

| Capability | What you get |
|------------|----------------|
| **Hybrid ingest** | Docling (layout/tables/text) + vision OCR for diagram plates |
| **Structured corpus** | Chapters, doc codes (`D`/`M`/`P`/`A`), procedures vs data vs plates |
| **Grounded chat** | Lexical + structural retrieval → verification gate → LLM answer or abstain |
| **Evidence UI** | Page citations + diagram thumbnails from MinIO |
| **GPU-ready** | Docling on CUDA when available; LLM/OCR via LiteLLM proxy |

---

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  PDF upload │────▶│  Page router     │────▶│ Docling (GPU)   │
│  (Vol II…)  │     │  text vs diagram │     │ + Vision OCR     │
└─────────────┘     └──────────────────┘     └────────┬────────┘
                                                       │
                       ┌───────────────────────────────▼────────┐
                       │  Postgres knowledge model + MinIO figs │
                       │  sections · elements · FTS · figures   │
                       └───────────────────────────────┬────────┘
                                                       │
┌─────────────┐     ┌──────────────────┐     ┌─────────▼────────┐
│  React UI   │◀───▶│  FastAPI         │◀───▶│ Retrieve → Verify│
│  Chat+Ingest│     │  /ask /ingest    │     │ → LiteLLM /abstain│
└─────────────┘     └──────────────────┘     └──────────────────┘
```

See [docs/architecture.md](docs/architecture.md) for details.

---

## Quick start

### Prerequisites

- Python **3.10+**
- Node.js **18+** (React UI)
- PostgreSQL (local or remote)
- Docker (for MinIO) — or any S3-compatible store
- LiteLLM proxy with chat + vision/OCR models configured
- NVIDIA GPU optional but recommended for Docling (`DOCLING_DEVICE=cuda`)

### 1. Clone & configure

```bash
git clone https://github.com/techicontech/Agentic_Doc_Parsing.git
cd Agentic_Doc_Parsing

cp .env.example .env
# Edit .env — set LITELLM_API_BASE, LITELLM_API_KEY, Postgres, MinIO
```

### 2. Python environment

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

For CUDA Docling, install a CUDA build of PyTorch from [pytorch.org](https://pytorch.org) that matches your driver, then keep `DOCLING_DEVICE=cuda` in `.env`.

### 3. Infrastructure

```bash
docker compose up -d          # MinIO on :9000 / console :9001
python scripts/init_db.py     # apply sql/001_schema.sql
```

### 4. Run API + UI

```bash
# Terminal 1 — API (default http://127.0.0.1:8000)
python scripts/run_api.py

# Terminal 2 — UI (http://127.0.0.1:5173)
cd web && npm install && npm run dev
```

1. Open the UI → upload a maintenance PDF → start ingest (progress % + file logs under `logs/`).
2. When ingest completes → ask grounded questions in Chat.

CLI ingest (optional):

```bash
python scripts/run_ingest.py --pdf path/to/manual.pdf
```

---

## Documentation

| Doc | Description |
|-----|-------------|
| [docs/architecture.md](docs/architecture.md) | Pipeline, data model, retrieval |
| [docs/setup.md](docs/setup.md) | Full install, GPU, troubleshooting |
| [docs/configuration.md](docs/configuration.md) | Environment variables |
| [docs/api.md](docs/api.md) | HTTP endpoints |
| [docs/retrieval.md](docs/retrieval.md) | How questions are grounded |

---

## Project layout

```
├── src/marine_docs/     # Ingest, retrieval, chat, FastAPI
├── scripts/             # init_db, run_api, run_ingest, reset_db
├── sql/                 # Schema + sanity SQL
├── web/                 # React + Vite chat/ingest UI
├── docs/                # Architecture & ops docs
├── docker-compose.yml   # MinIO
├── .env.example         # Safe template (no secrets)
└── requirements.txt
```

---

## Safety & grounding

- Answers are produced **only from retrieved evidence**.
- A verification gate can **abstain** when evidence is missing or conflicting.
- Citations include page, section path, and `doc_code` when available.
- **Do not commit** API keys, `.env`, uploaded PDFs, or ingest logs (see `.gitignore`).

---

## License

Proprietary — © Techicon. All rights reserved unless otherwise agreed.
Contact the maintainers for licensing.

---

## Maintainers

**Techicon** — [techicontech/Agentic_Doc_Parsing](https://github.com/techicontech/Agentic_Doc_Parsing)
