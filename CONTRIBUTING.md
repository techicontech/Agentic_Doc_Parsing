# Contributing

Thanks for helping improve **Agentic Doc Parsing**.

## Ground rules

- Do **not** commit secrets (`.env`, API keys), uploaded PDFs, logs, or MinIO/Postgres dumps.
- Prefer small, focused PRs with a clear problem statement.
- Retrieval changes must stay **generic** (intent/section ranking) — no hard-coded question → page maps.
- Keep answers grounded: prefer abstain over inventing values.

## Dev workflow

1. Fork / branch from `main`.
2. Follow [docs/setup.md](docs/setup.md).
3. Run API + UI locally; exercise ingest + chat on a sample PDF.
4. Open a PR against [techicontech/Agentic_Doc_Parsing](https://github.com/techicontech/Agentic_Doc_Parsing) with:
   - What changed and why
   - How you tested
   - Any `.env` / infra notes (without secrets)

## Code layout

- `src/marine_docs/` — core library + API
- `web/` — React UI
- `docs/` — architecture and ops
- `scripts/` — entrypoints only (no one-off experiment scripts in `main`)

## Style

- Match existing Python / React patterns in the repo.
- Avoid drive-by refactors unrelated to the PR.
