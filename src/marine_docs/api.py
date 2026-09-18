"""FastAPI backend — upload PDF, ingest, chat."""

from __future__ import annotations

import logging
import re
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

from marine_docs.chat import answer_query
from marine_docs.config import ROOT
from marine_docs.corpus import clear_corpus, corpus_status
from marine_docs.ingest import run_ingest
from marine_docs.log_setup import progress_log, setup_logging
from marine_docs.minio_client import download_bytes
from marine_docs.retrieval import get_fleet_context

logger = logging.getLogger(__name__)

_LOG_PATH = setup_logging()
progress_log(f"Full logs -> {_LOG_PATH}")

app = FastAPI(title="Marine Docs Chat API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = ROOT / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

_job_lock = threading.Lock()
_job: dict[str, Any] = {
    "status": "idle",  # idle | uploading | running | completed | failed
    "stage": None,
    "message": "Upload a PDF to begin",
    "percent": 0,
    "done": None,
    "total": None,
    "pdf_name": None,
    "result": None,
    "error": None,
    "log_file": str(_LOG_PATH),
}


def _set_job(**kwargs: Any) -> None:
    with _job_lock:
        _job.update(kwargs)


def _safe_filename(name: str) -> str:
    base = Path(name).name
    base = re.sub(r"[^\w.\- ]+", "_", base).strip() or "manual.pdf"
    if not base.lower().endswith(".pdf"):
        base += ".pdf"
    return base


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    equipment: str | None = None


class ChatApiResponse(BaseModel):
    answer: str
    abstained: bool
    citations: list[dict[str, Any]]
    diagrams: list[dict[str, Any]]
    verification: dict[str, Any]
    retrieval_notes: dict[str, Any]


@app.get("/api/health")
def health() -> dict[str, Any]:
    status = corpus_status()
    fleet = get_fleet_context()
    with _job_lock:
        job = {
            k: _job[k]
            for k in ("status", "stage", "message", "percent", "pdf_name", "error", "log_file")
        }
    return {
        "ok": True,
        "has_data": status.get("has_data"),
        "manual": (fleet or {}).get("title"),
        "ocr_ready": status.get("ocr_ready"),
        "agentic": True,
        "job": job,
    }


@app.post("/api/chat", response_model=ChatApiResponse)
def chat(req: ChatRequest) -> ChatApiResponse:
    if _job["status"] == "running":
        raise HTTPException(status_code=409, detail="Ingest still running. Wait until it finishes.")
    status = corpus_status()
    if not status.get("has_data"):
        raise HTTPException(status_code=400, detail="No manual ingested yet. Upload a PDF first.")
    try:
        resp = answer_query(req.message.strip(), equipment_context=req.equipment or None)
    except Exception as exc:
        logger.exception("chat failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return ChatApiResponse(
        answer=resp.answer,
        abstained=resp.abstained,
        citations=resp.citations,
        diagrams=resp.diagrams,
        verification=resp.verification,
        retrieval_notes=resp.retrieval_notes,
    )


@app.get("/api/figures")
def get_figure(key: str = Query(..., min_length=1)) -> Response:
    if ".." in key or key.startswith("/"):
        raise HTTPException(status_code=400, detail="invalid key")
    try:
        data = download_bytes(key)
    except Exception as exc:
        logger.exception("figure fetch failed for %s", key)
        raise HTTPException(status_code=404, detail=f"figure not found: {exc}") from exc
    return Response(content=data, media_type="image/png")


@app.get("/api/ingest/status")
def ingest_status() -> dict[str, Any]:
    with _job_lock:
        job = dict(_job)
    job["has_data"] = corpus_status().get("has_data")
    return job


@app.post("/api/ingest/upload")
async def ingest_upload(file: UploadFile = File(...)) -> dict[str, Any]:
    """Upload any PDF, clear previous corpus, and start ingest."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    with _job_lock:
        if _job["status"] == "running":
            raise HTTPException(status_code=409, detail="Ingest already running")
        _job.update(
            {
                "status": "running",
                "stage": "uploading",
                "message": "Saving PDF…",
                "percent": 0,
                "done": None,
                "total": None,
                "pdf_name": file.filename,
                "result": None,
                "error": None,
                "log_file": str(_LOG_PATH),
            }
        )

    try:
        safe = _safe_filename(file.filename)
        dest = UPLOAD_DIR / safe
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty file")
        dest.write_bytes(data)
    except HTTPException:
        _set_job(status="failed", stage="failed", message="Upload failed", percent=0, error="Empty or invalid file")
        raise
    except Exception as exc:
        _set_job(status="failed", stage="failed", message="Upload failed", percent=0, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    def worker(pdf_path: Path, pdf_name: str) -> None:
        try:
            _set_job(
                stage="clearing",
                message="Clearing previous data…",
                percent=0,
                pdf_name=pdf_name,
            )
            if corpus_status().get("has_data"):
                clear_corpus()

            def on_progress(stage: str, payload: dict[str, Any]) -> None:
                _set_job(
                    stage=stage,
                    message=payload.get("message") or stage.replace("_", " "),
                    percent=int(payload.get("percent") or 0),
                    done=payload.get("done"),
                    total=payload.get("total"),
                )

            result = run_ingest(pdf=pdf_path, progress=on_progress)
            _set_job(
                status="completed",
                stage="completed",
                message="Ingest complete — you can chat now",
                percent=100,
                result=result,
                error=None,
                pdf_name=pdf_name,
            )
        except Exception as exc:
            logger.exception("upload ingest failed")
            _set_job(
                status="failed",
                stage="failed",
                message="Ingest failed",
                error=str(exc),
            )

    threading.Thread(target=worker, args=(dest, file.filename), daemon=True).start()
    return {"started": True, "pdf_name": file.filename, "saved_as": str(dest), "log_file": str(_LOG_PATH)}
