#!/usr/bin/env python
"""Apply schema (if needed), start stack helpers, and print connection info."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from marine_docs.config import get_settings
from marine_docs.db import connect, execute_sql_file
from marine_docs.minio_client import ensure_bucket


def main() -> int:
    settings = get_settings()
    schema = ROOT / "sql" / "001_schema.sql"
    print(f"Connecting to {settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}")
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 AS ok")
            print("Postgres OK:", cur.fetchone())
    # Schema is normally applied by docker-entrypoint-initdb.d on first boot.
    # This re-applies safely only when tables are missing.
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) AS n FROM information_schema.tables "
                "WHERE table_schema='public' AND table_name='manuals'"
            )
            exists = cur.fetchone()["n"] > 0
    if not exists:
        print("Applying schema...")
        execute_sql_file(str(schema))
    else:
        print("Schema already present")

    for extra in ("002_milestone2.sql", "003_generic_citations.sql"):
        path = ROOT / "sql" / extra
        if path.exists():
            print(f"Applying {extra}...")
            execute_sql_file(str(path))

    bucket = ensure_bucket()
    print(f"MinIO bucket ready: {bucket}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
