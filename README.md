# Marine & Shipping Technical Manual Intelligence

Ingest large marine-engine and shipping-equipment maintenance manuals (any manufacturer) into Postgres + MinIO, then answer questions with **procedure/plate citations**, **diagram panels**, and **controlled abstention**.

This is not a general chatbot and it does not claim 100% accuracy. Answers are produced only from retrieved evidence. If the verification gate fails, the system abstains instead of guessing.

The MAN B&W S50MC-C Volume II PDF used in development is a **test case**, not a hard-coded format. Citation layout, section numbering, and revision conventions are detected per document at ingest and stored as `manuals.citation_convention`.

[Repository](https://github.com/techicontech/Agentic_Doc_Parsing)

---

## End-to-end flow

```
INGEST (batch, deterministic)
  PDF
    -> convention detector (LLM once per manual; heuristic fallback)
    -> page router (heuristic first; LLM only on ambiguous mixed pages)
    -> Docling on text/table pages (CUDA when configured)
    -> vision OCR on diagram/plate pages (LiteLLM; panel crops, not one image per page)
    -> structure (sections, component/action titles, citation_key)
    -> typed cross-references
    -> Postgres + MinIO

CHAT (deterministic control flow; agents only where judgment is needed)
  Query
    -> Router agent (lexical / structural / visual labels)
    -> Parallel retrieval:
         Lexical FTS tool  |  Structural tree agent  |  Visual figures tool
    -> RRF fusion (tool — never an agent)
    -> Ranking rules (tool — identifiers, kind pairs, siblings, contrast clauses)
    -> Lightweight rerank (not an orchestrator agent)
    -> Pinned hits stay pinned
    -> Verification gate (6 checks, tool — never an agent)
    -> Answer synthesis (LLM)  OR  Abstain
```

Live code is under `src/marine_docs/`. Fusion is `rrf_fuse` / `fuse_paths`. Ranking is `ranking.apply_priority`. The gate is `verify` in `marine_docs.verify`. Orchestration is `agents/pipeline.execute_pipeline`.

See [docs/architecture.md](docs/architecture.md), [docs/retrieval.md](docs/retrieval.md), and [docs/agents.md](docs/agents.md).

---

## Agent vs tool

| Step | Type | Why |
|---|---|---|
| Convention detector (ingest) | Agent (LLM), once per new manual | Unfamiliar header/footer layout |
| Page router (ingest) | Agent — ambiguous pages only | Mixed text + diagram pages |
| Docling extraction | Tool | Deterministic parser |
| Diagram/plate OCR + captioning | Agent (vision via LiteLLM) | Reading a drawing |
| Panel geometry + step pairing | Tool | Layout boxes |
| Persistence | Tool | Postgres / MinIO writes |
| Query router | Agent | Intent labels |
| Lexical retrieval (SQL/FTS) | Tool | Deterministic |
| Structural tree navigation | Agent | Which section applies |
| Visual retrieval | Tool (figures table; ColPali = Phase 2) | Panel rows |
| Candidate fusion (RRF) | **Tool — never an agent** | Pure merge |
| Ranking priority | Tool | Reproducible evidence window |
| Rerank | Lightweight model call | Non-agentic |
| Verification gate | **Tool — never an agent** | Auditable abstention |
| Answer synthesis | Agent | Grounded prose |

Control flow is Python, not a root orchestrator agent choosing its own steps.

---

## Quick start

### Prerequisites

- Python **3.10+**
- Node.js **18+** (React UI)
- PostgreSQL 14+
- Docker (MinIO) or any S3-compatible store
- LiteLLM proxy with chat + vision models
- NVIDIA GPU optional, recommended for Docling (`DOCLING_DEVICE=cuda`)

### 1. Clone and configure

```bash
git clone https://github.com/techicontech/Agentic_Doc_Parsing.git
cd Agentic_Doc_Parsing

cp .env.example .env
```

Edit `.env`: `LITELLM_API_BASE`, `LITELLM_API_KEY`, `LLM_MODEL`, `OCR_VISION_MODEL`, `OCR_BACKEND`, `DB_*`, `MINIO_*`. Full list: [docs/configuration.md](docs/configuration.md).

### 2. Python environment

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

For CUDA Docling, install a CUDA PyTorch build from [pytorch.org](https://pytorch.org) that matches your driver, then set `DOCLING_DEVICE=cuda`.

### 3. Infrastructure

```bash
docker compose up -d
python scripts/init_db.py
```

`init_db.py` applies `sql/001_schema.sql` (if empty), then `sql/002_milestone2.sql` and `sql/003_generic_citations.sql` (idempotent).

### 4. Run API + UI

Windows (two terminals, repo root):

```powershell
# Terminal 1 — API  http://127.0.0.1:8000
.\.venv\Scripts\activate
python scripts\run_api.py

# Terminal 2 — UI   http://127.0.0.1:5173
.\.venv\Scripts\activate
cd web
npm install
npm run dev
```

macOS / Linux:

```bash
python scripts/run_api.py
# other terminal
cd web && npm install && npm run dev
```

1. Open the UI → upload a maintenance PDF. A new upload **clears the previous corpus** and starts ingest. Progress is in the UI and in gitignored files under `logs/`.
2. When ingest completes, ask questions in Chat.

CLI ingest (same pipeline, no UI):

```bash
python scripts/run_ingest.py --pdf path/to/manual.pdf
```

One-shot CLI ask:

```bash
python scripts/ask.py "Show plate P91066 callout 011"
```

Restarting the API or UI does **not** wipe ingested data. Re-ingest only if the PDF or parse/OCR settings changed. Ranking/chat code changes do not need a re-ingest.

---

## Adding a new manufacturer's manual

No manufacturer-specific code paths.

1. Upload the PDF (UI) or `python scripts/run_ingest.py --pdf path/to/manual.pdf`.
2. The **convention detector** samples headers/footers and writes `citation_convention` (pattern, revision granularity, quote hint).
3. Parsers and retrieval read that JSON. `citation_key` is always populated; fallback is `p.` + printed page.
4. If a new layout is missed, fix the convention-detector prompt — do not add `if manufacturer == ...`.

Cross-references use the stored citation pattern plus generic verbs (`see`, `refer to`, …). Multi-panel figures come from layout boxes.

---

## Example questions

After ingest, try questions that name a component **and** an identifier, and that distinguish siblings:

- Show the air-cooler lifting-tools plate: what is callout 011, and which plate number should I quote?
- Stay-bolt work: quote hydraulic pressure for mounting vs dismantling. Do not use the cylinder-cover data sheet.
- Connecting-rod mounting: to what crank angle do you turn before shifting tackle A?
- Show piston checking step 1, drawing M90201-0285O01. Which procedure owns that panel?

---

## Known limitations

- **Phase 2:** ColPali visual retrieval, a dedicated cross-encoder reranker, and dense vectors (`pgvector` in this same Postgres only if evaluation shows they help). v1 is vectorless: structure + lexical + ranking + reasoning.
- **Abstention:** six-check gate (equipment, revision, condition, units, scope, conflicts). If any check fails, the answer model is not called.
- **No 100% accuracy claim** in code, docs, or UI.
- Diagram OCR uses vision via LiteLLM on drawing-heavy pages; cost scales with plate count.

---

## Documentation

| Doc | Description |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Ingest + chat flow, data model |
| [docs/retrieval.md](docs/retrieval.md) | Paths, ranking rules, citations |
| [docs/agents.md](docs/agents.md) | Agents vs tools |
| [docs/setup.md](docs/setup.md) | Install, GPU, troubleshooting |
| [docs/configuration.md](docs/configuration.md) | Environment variables |
| [docs/api.md](docs/api.md) | HTTP endpoints |

---

## Project layout

```
├── src/marine_docs/     # Ingest, ADK agents, ranking, FastAPI
├── scripts/             # init_db, run_api, run_ingest, reset_db, ask
├── sql/                 # 001 schema, 002 section fields, 003 citation_key
├── tests/               # Convention detector + verification gate
├── web/                 # React + Vite ingest/chat UI
├── docs/
├── docker-compose.yml   # MinIO
├── .env.example
└── requirements.txt
```

Runtime only (gitignored): `uploads/`, `logs/`, `artifacts/`, `.env`.

---

## Safety and grounding

- Answers come **only from retrieved evidence**.
- The verification gate **abstains** when evidence is missing, out of scope, or conflicting.
- Revision checks use the granularity stored on that manual (per-section or per-document).
- Citations lead with `citation_key` (plus revision when present). Printed page numbers are supplementary.
- Do not commit API keys, `.env`, uploaded PDFs, or ingest logs.

---

## License

Proprietary — © Techicon. All rights reserved unless otherwise agreed.

## Maintainers

**Techicon** — [techicontech/Agentic_Doc_Parsing](https://github.com/techicontech/Agentic_Doc_Parsing)
