# Contributing

Thanks for helping improve **Agentic Doc Parsing**.

## Ground rules

- Do **not** commit secrets (`.env`, API keys), uploaded PDFs, logs, or MinIO/Postgres dumps. See `.gitignore`.
- Prefer small, focused PRs with a clear problem statement.
- Retrieval and ranking must stay **generic** (intent, section kind, identifiers, contrast clauses) — no hard-coded question → page maps and no `if manufacturer == ...`.
- Keep answers grounded: prefer abstain over inventing values.
- Fusion (`rrf_fuse`) and the verification gate (`verify.verify`) are **tools**, never agents.

## Docs for a new developer

Start here, in order:

1. [README.md](README.md) — what it is, end-to-end flow, how to run
2. [docs/setup.md](docs/setup.md) — install, GPU, troubleshooting
3. [docs/architecture.md](docs/architecture.md) — ingest + chat pipeline and data model
4. [docs/retrieval.md](docs/retrieval.md) — ranking, citations, diagrams, re-ingest vs restart
5. [docs/agents.md](docs/agents.md) — which steps are agents vs tools
6. [docs/api.md](docs/api.md) — HTTP routes the UI uses
7. [docs/configuration.md](docs/configuration.md) — `.env` and schema `sql/001`–`003`

## Dev workflow

1. Fork / branch from `main`.
2. Follow [docs/setup.md](docs/setup.md).
3. Run API + UI locally; exercise ingest + chat on a sample PDF.
4. `python -m unittest tests.test_convention_detector tests.test_verification_gate`
5. Open a PR against [techicontech/Agentic_Doc_Parsing](https://github.com/techicontech/Agentic_Doc_Parsing) with:
   - What changed and why
   - How you tested
   - Any `.env` / infra notes (without secrets)

Restarting API/UI does not wipe the corpus. Re-ingest only if the PDF or parse/OCR settings changed.

## Code layout

- `src/marine_docs/` — ingest, ranking, ADK agents, FastAPI
- `web/` — React + Vite ingest/chat UI
- `sql/` — `001_schema.sql`, `002_milestone2.sql`, `003_generic_citations.sql`
- `scripts/` — `init_db`, `run_api`, `run_ingest`, `reset_db`, `ask` (no one-off experiment scripts)
- `tests/` — convention detector + verification gate
- `docs/` — architecture and ops

## Style

- Match existing Python / React patterns in the repo.
- Avoid drive-by refactors unrelated to the PR.
