"""Per-section or per-document revision checks (spec Section 9).

Granularity comes from manuals.citation_convention (per_section vs per_document),
not from a manufacturer hardcoded as per-section. With a single ingested volume,
the printed revision *is* the current one. The verification gate therefore passes
unless the user names a different revision than the one stored on the retrieved
citation_key.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any

from marine_docs.db import connect

logger = logging.getLogger(__name__)

NAMED_EDITION_RE = re.compile(
    r"\b(?:edition|ed\.?|rev(?:ision)?\.?)\s*[:#]?\s*([A-Za-z0-9]{1,12})\b",
    re.IGNORECASE,
)


def normalize_edition(value: str | None) -> str:
    """Compare editions on alphanumerics only.

    A safety check must not abstain because one page's footer picked up a stray
    character during extraction ("0S48_" vs "0S48").
    """
    return re.sub(r"[^A-Za-z0-9]", "", (value or "")).upper()


def named_edition(query: str) -> str | None:
    """Edition the user named in the question, if any."""
    match = NAMED_EDITION_RE.search(query or "")
    if not match:
        return None
    return normalize_edition(match.group(1)) or None


@lru_cache(maxsize=8)
def current_editions() -> dict[str, str]:
    """Map procedure_no / plate_no / doc_code to its highest recorded edition."""
    editions: dict[str, set[str]] = {}
    try:
        with connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT e.procedure_no, e.plate_no, s.doc_code, e.citation_key, e.edition
                FROM elements e
                LEFT JOIN sections s ON s.id = e.section_id
                WHERE e.edition IS NOT NULL
                GROUP BY e.procedure_no, e.plate_no, s.doc_code, e.citation_key, e.edition
                """
            )
            rows = cur.fetchall()
    except Exception:
        logger.exception("Could not load per-procedure editions")
        return {}

    for row in rows:
        edition = normalize_edition(row["edition"])
        if not edition:
            continue
        for key in (row["procedure_no"], row["plate_no"], row["doc_code"], row.get("citation_key")):
            if key:
                editions.setdefault(key.strip(), set()).add(edition)

    return {key: max(values) for key, values in editions.items()}


def reset_cache() -> None:
    current_editions.cache_clear()


def check_named_mismatch(query: str, evidences: list[Any]) -> tuple[bool, list[str]]:
    """Fail only when a named identifier's edition disagrees with the query.

    If the user names an edition but no procedure/plate/doc code, a mixed
    evidence pile must not fail — stray footers from other sections are common.
    """
    wanted = named_edition(query)
    if not wanted:
        return True, []

    named_ids = _identifiers_in_query(query)
    if not named_ids:
        return True, []

    mismatches: list[str] = []
    for ev in evidences:
        edition = normalize_edition(_field(ev, "edition"))
        if not edition or edition == wanted:
            continue
        keys = [
            _field(ev, "procedure_no"),
            _field(ev, "plate_no"),
            _field(ev, "doc_code"),
            _field(ev, "citation_key"),
        ]
        ev_ids = {str(k).strip().upper() for k in keys if k}
        if not (ev_ids & named_ids):
            continue
        key = next((k for k in keys if k), f"page {_field(ev, 'page') or '?'}")
        mismatches.append(f"{key} is Edition {edition}, query asked for {wanted}")
    return not mismatches, mismatches


def _identifiers_in_query(query: str) -> set[str]:
    found: set[str] = set()
    for match in re.finditer(r"\b(\d{3}-\d+(?:\.\d+)?)\b", query or ""):
        found.add(match.group(1).strip().upper())
    for match in re.finditer(r"\b([A-Z]{1,3}\d{4,8})\b", query or "", re.IGNORECASE):
        found.add(match.group(1).strip().upper())
    return found


def _field(ev: Any, name: str) -> Any:
    if isinstance(ev, dict):
        return ev.get(name)
    return getattr(ev, name, None)


def check_revision(evidences: list[Any]) -> tuple[bool, list[str]]:
    """True when every cited procedure is at its own current edition.

    Informational for logs/details. The gate itself uses check_named_mismatch so a
    single ingested volume does not abstain just because two procedures sit at
    different, independently valid editions.
    """
    registry = current_editions()
    if not registry:
        return True, []

    superseded: list[str] = []
    for ev in evidences:
        edition = normalize_edition(getattr(ev, "edition", None))
        key = (
            getattr(ev, "citation_key", None)
            or getattr(ev, "procedure_no", None)
            or getattr(ev, "plate_no", None)
            or getattr(ev, "doc_code", None)
        )
        if not edition or not key:
            continue
        current = registry.get(str(key).strip())
        if current and edition != current:
            superseded.append(f"{key} Edition {edition} (current is {current})")
    return not superseded, superseded
