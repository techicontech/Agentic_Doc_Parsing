"""Corpus status + clear for fresh re-ingest."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from marine_docs.config import get_settings
from marine_docs.db import connect, execute_sql_file
from marine_docs.minio_client import ensure_bucket, get_s3_client

TABLES = [
    "query_log",
    "ingest_runs",
    "element_relationships",
    "element_provenance",
    "figures",
    "elements",
    "sections",
    "fleet_registry",
    "manual_revisions",
    "manuals",
]


def corpus_status() -> dict[str, Any]:
    settings = get_settings()
    counts = {
        "manuals": 0,
        "sections": 0,
        "elements": 0,
        "figures": 0,
        "relationships": 0,
        "fleet_rows": 0,
    }
    latest_run = None
    try:
        with connect() as conn:
            with conn.cursor() as cur:
                for table, key in [
                    ("manuals", "manuals"),
                    ("sections", "sections"),
                    ("elements", "elements"),
                    ("figures", "figures"),
                    ("element_relationships", "relationships"),
                    ("fleet_registry", "fleet_rows"),
                ]:
                    cur.execute(f"SELECT count(*) AS n FROM {table}")
                    counts[key] = cur.fetchone()["n"]
                cur.execute(
                    """
                    SELECT id, status, page_count, docling_pages, mistral_pages,
                           skipped_pages, started_at, finished_at, notes
                    FROM ingest_runs
                    ORDER BY started_at DESC
                    LIMIT 1
                    """
                )
                latest_run = cur.fetchone()
    except Exception as exc:
        return {
            "has_data": False,
            "counts": counts,
            "pdf_path": str(settings.resolved_pdf_path),
            "pdf_exists": settings.resolved_pdf_path.exists(),
            "ocr_ready": settings.ocr_ready,
            "error": str(exc),
            "latest_run": None,
        }

    has_data = any(counts[k] > 0 for k in ("manuals", "sections", "elements", "figures"))
    return {
        "has_data": has_data,
        "counts": counts,
        "pdf_path": str(settings.resolved_pdf_path),
        "pdf_exists": settings.resolved_pdf_path.exists(),
        "ocr_ready": settings.ocr_ready,
        "latest_run": _serialize_run(latest_run),
    }


def clear_corpus() -> dict[str, Any]:
    """Drop knowledge tables and recreate schema; clear MinIO figure objects."""
    root = Path(__file__).resolve().parents[2]

    with connect() as conn:
        with conn.cursor() as cur:
            for t in TABLES:
                cur.execute(f"DROP TABLE IF EXISTS {t} CASCADE")
        conn.commit()
    for migration in sorted((root / "sql").glob("0*.sql")):
        if "sanity" in migration.name:
            continue
        execute_sql_file(str(migration))

    deleted_objects = _clear_minio_prefix("manuals/")
    return {"cleared": True, "minio_objects_deleted": deleted_objects}


def _clear_minio_prefix(prefix: str) -> int:
    try:
        bucket = ensure_bucket()
        client = get_s3_client()
        deleted = 0
        continuation = None
        while True:
            kwargs = {"Bucket": bucket, "Prefix": prefix}
            if continuation:
                kwargs["ContinuationToken"] = continuation
            resp = client.list_objects_v2(**kwargs)
            objs = [{"Key": o["Key"]} for o in resp.get("Contents") or []]
            if objs:
                client.delete_objects(Bucket=bucket, Delete={"Objects": objs})
                deleted += len(objs)
            if not resp.get("IsTruncated"):
                break
            continuation = resp.get("NextContinuationToken")
        return deleted
    except Exception:
        return 0


def _serialize_run(row: dict | None) -> dict | None:
    if not row:
        return None
    return {
        "id": str(row["id"]),
        "status": row["status"],
        "page_count": row["page_count"],
        "docling_pages": row["docling_pages"],
        "mistral_pages": row["mistral_pages"],
        "skipped_pages": row["skipped_pages"],
        "started_at": row["started_at"].isoformat() if row.get("started_at") else None,
        "finished_at": row["finished_at"].isoformat() if row.get("finished_at") else None,
        "notes": row.get("notes"),
    }
