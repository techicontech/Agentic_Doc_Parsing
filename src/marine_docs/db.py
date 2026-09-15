"""PostgreSQL helpers."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import psycopg
from psycopg.rows import dict_row

from marine_docs.config import get_settings


def connect() -> psycopg.Connection:
    settings = get_settings()
    return psycopg.connect(settings.database_url, row_factory=dict_row)


@contextmanager
def db_cursor() -> Iterator[psycopg.Cursor]:
    with connect() as conn:
        with conn.cursor() as cur:
            yield cur
        conn.commit()


def execute_sql_file(path: str) -> None:
    sql = open(path, encoding="utf-8").read()
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
