# Setup guide

## System requirements

| Component | Requirement |
|-----------|-------------|
| OS | Windows 10/11, Linux, or macOS |
| Python | 3.10+ |
| Node.js | 18+ (UI only) |
| Postgres | 14+ recommended |
| Docker | For MinIO (optional if you bring your own S3) |
| GPU | Optional NVIDIA + CUDA PyTorch for faster Docling |

## Step-by-step

### 1. Repository

```bash
git clone https://github.com/techicontech/Agentic_Doc_Parsing.git
cd Agentic_Doc_Parsing
cp .env.example .env
```

Edit `.env` — see [configuration.md](configuration.md).

### 2. Python

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

### 3. GPU (optional)

1. Install NVIDIA driver.
2. Install CUDA-enabled PyTorch from https://pytorch.org.
3. Set `DOCLING_DEVICE=cuda` in `.env`.
4. Verify:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

### 4. MinIO

```bash
docker compose up -d
```

Console: http://localhost:9001 (default `minioadmin` / `minioadmin` — change in production).

### 5. Postgres

Create database/user matching `.env`, then:

```bash
python scripts/init_db.py
```

Optional wipe of **tables** (not the schema files):

```bash
python scripts/reset_db.py --yes
```

### 6. Services

Two terminals from the repo root:

```bash
python scripts/run_api.py
```

```bash
cd web
npm install
npm run dev
```

- API: http://127.0.0.1:8000
- UI: http://127.0.0.1:5173
- Swagger: http://127.0.0.1:8000/docs

Restarting these processes does **not** delete ingested data.

## Ingest notes

- Full manuals (500+ pages) take a long time on CPU; GPU is strongly preferred.
- UI upload **replaces** the previous corpus, then ingests the new PDF.
- Progress is written to `logs/` (gitignored).
- Uploaded PDFs stay in `uploads/` (gitignored) — never commit OEM manuals.

## Troubleshooting

| Symptom | Check |
|---------|--------|
| LiteLLM 403 | Wrong `LITELLM_API_KEY` or model not enabled on the proxy |
| OCR 504 / missing Mistral | Use `OCR_BACKEND=claude` or configure Mistral on the proxy |
| Docling slow / CPU | Install CUDA torch; set `DOCLING_DEVICE=cuda` |
| Chat abstains wrongly | Usually retrieval ranking — see [retrieval.md](retrieval.md); **not** a re-ingest unless OCR text is wrong |
| Empty fleet / no data | Finish ingest; confirm `init_db` + MinIO healthy |
| Sibling plate in the diagram strip | Question named the wrong plate in a contrast clause; ranking strips those identifiers from the positive clause |

## Security

- Never commit `.env` or real API keys.
- Rotate keys if they appear in shell history or screenshots.
- Treat uploaded manuals as confidential; keep them outside git.
