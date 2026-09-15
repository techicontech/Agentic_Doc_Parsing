"""Milestone-1/2 retrieval over Postgres knowledge model (no vector DB)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from marine_docs.db import connect


def _flatten_table_json(table_json) -> str:
    if not table_json:
        return ""
    cells: list[str] = []

    def walk(node):
        if isinstance(node, dict):
            t = node.get("text")
            if isinstance(t, str) and t.strip():
                cells.append(t.strip())
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(table_json)
    out: list[str] = []
    for c in cells:
        if not out or out[-1] != c:
            out.append(c)
    return " | ".join(out)


def _row_text(row: dict) -> str:
    text = (row.get("text") or "").strip()
    if text:
        return text[:2000]
    flat = _flatten_table_json(row.get("table_json"))
    return flat[:2000] if flat else ""


@dataclass
class Evidence:
    element_id: UUID
    page: int
    text: str
    section_path: list[str] | None
    doc_code: str | None
    edition: str | None
    section_kind: str | None
    score: float
    source: str  # lexical | structural | figure
    figure_image_path: str | None = None
    figure_label: str | None = None


@dataclass
class RetrievalResult:
    evidences: list[Evidence] = field(default_factory=list)
    fleet_model: str | None = None
    manual_title: str | None = None
    revision: str | None = None
    notes: dict[str, Any] = field(default_factory=dict)


DOC_CODE_RE = re.compile(r"\b([DMPA]\d{4,6})\b", re.IGNORECASE)
SECTION_RE = re.compile(r"\b(\d{3}-\d+(?:\.\d+)?)\b")

# Generic English / question filler — never used as sole section match keys
STOPWORDS = frozenset(
    {
        "the", "and", "for", "with", "what", "show", "how", "does", "did", "are",
        "is", "was", "were", "can", "could", "would", "should", "which", "where",
        "when", "who", "whom", "why", "from", "into", "onto", "about", "than",
        "then", "that", "this", "these", "those", "have", "has", "had", "been",
        "being", "used", "using", "use", "please", "tell", "give", "find",
        "need", "want", "also", "any", "all", "both", "each", "vs", "versus",
        "between", "allowed", "recommended", "required", "still", "measured",
        "new", "old", "complete", "section", "manual", "page", "chapter",
    }
)

# Spec / numeric lookup intent
DATA_HINTS = frozenset(
    {
        "clearance", "diameter", "torque", "pressure", "wear", "limit", "limits",
        "weight", "kg", "mm", "bar", "nm", "hydraulic", "radial", "vertical",
        "gap", "height", "width", "deviation", "criteria", "data", "spec",
        "specification", "value", "values", "max", "min", "maximum", "minimum",
        "mounting", "dismantling", "tightening",
    }
)

# Procedure / how-to intent
PROC_HINTS = frozenset(
    {
        "remove", "removal", "dismantle", "dismantling", "install", "installation",
        "mount", "mounting", "replace", "replacement", "procedure", "steps",
        "step", "how", "open", "opening", "overhaul", "check", "checking",
        "inspect", "inspection", "safety", "precaution", "precautions",
    }
)

FIGURE_HINTS = frozenset(
    {
        "diagram", "figure", "plate", "drawing", "panel", "wiring", "piping",
        "illustration", "image", "picture",
    }
)


def _singularize(token: str) -> str:
    t = token.lower()
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 3 and t.endswith("ses"):
        return t[:-2]
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        return t[:-1]
    return t


def _content_tokens(query: str) -> list[str]:
    # Split hyphenated compounds (top-clearance → top, clearance)
    raw = re.findall(r"[A-Za-z][A-Za-z0-9]*", query)
    out: list[str] = []
    seen: set[str] = set()
    for t in raw:
        low = t.lower()
        if low in STOPWORDS or len(low) < 3:
            continue
        stem = _singularize(low)
        if stem in STOPWORDS or stem in seen:
            continue
        seen.add(stem)
        out.append(stem)
    return out


def _phrases(tokens: list[str]) -> list[str]:
    """Bigrams first (component names), then trigrams — dynamic, not question-specific."""
    phrases: list[str] = []
    for n in (2, 3):
        for i in range(len(tokens) - n + 1):
            phrases.append(" ".join(tokens[i : i + n]))
    return phrases


def _fts_or_query(tokens: list[str]) -> str:
    """OR-query so long questions don't fail plainto_tsquery's implicit AND."""
    parts: list[str] = []
    for t in tokens[:10]:
        safe = re.sub(r"[^a-z0-9]", "", t.lower())
        if len(safe) >= 3:
            parts.append(safe)
    return " | ".join(parts) if parts else ""


def _query_intent(query: str) -> dict[str, Any]:
    q = query.lower()
    tokens = _content_tokens(query)
    token_set = set(tokens)
    wants_figure = any(h in q for h in FIGURE_HINTS) or "show me" in q
    wants_data = bool(token_set & DATA_HINTS) or bool(
        re.search(r"\b\d+(?:\.\d+)?\s*(mm|bar|nm|kg)\b", q)
    )
    # "hydraulic pressure, dismantling" is data-sheet wording — only treat as
    # procedure when how-to verbs dominate or question is explicitly how-to.
    proc_hits = token_set & PROC_HINTS
    data_hits = token_set & DATA_HINTS
    wants_proc = bool(proc_hits) and (
        q.strip().startswith("how ")
        or any(v in token_set for v in ("remove", "replace", "install", "open", "step", "procedure", "safety", "precaution"))
        or (not wants_data)
        or len(proc_hits - {"dismantling", "mounting", "dismantle", "mount"}) > 0
    )
    component = [t for t in tokens if t not in DATA_HINTS and t not in PROC_HINTS]
    data_toks = [t for t in tokens if t in DATA_HINTS]
    fts_tokens = (component[:4] + data_toks[:4]) or tokens[:8]
    return {
        "tokens": tokens,
        "phrases": _phrases(tokens),
        "component_tokens": component,
        "wants_figure": wants_figure,
        "wants_data": wants_data,
        "wants_proc": wants_proc,
        "codes": [m.group(1).upper() for m in DOC_CODE_RE.finditer(query)],
        "sections": [m.group(1) for m in SECTION_RE.finditer(query)],
        "fts_query": _fts_or_query(fts_tokens),
        "fts_plain": " ".join(fts_tokens) or query,
    }


def get_fleet_context() -> dict[str, Any] | None:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT fr.model, fr.equipment_id, fr.revision_status,
                       m.id AS manual_id, m.title, m.revision
                FROM fleet_registry fr
                JOIN manuals m ON m.id = fr.manual_id
                WHERE fr.revision_status = 'current'
                ORDER BY fr.id
                LIMIT 1
                """
            )
            return cur.fetchone()


def retrieve(query: str, *, limit: int = 12) -> RetrievalResult:
    fleet = get_fleet_context()
    result = RetrievalResult(
        fleet_model=fleet["model"] if fleet else None,
        manual_title=fleet["title"] if fleet else None,
        revision=fleet["revision"] if fleet else None,
    )
    if not fleet:
        result.notes["error"] = "no_fleet_registry"
        return result

    manual_id = fleet["manual_id"]
    intent = _query_intent(query)
    evidences: list[Evidence] = []
    evidences.extend(_lexical_search(manual_id, query, intent, limit=limit))
    evidences.extend(_structural_search(manual_id, intent, limit=max(6, limit // 2)))
    evidences.extend(_figure_search(manual_id, query, intent, limit=4))

    for ev in evidences:
        _apply_ranking_boosts(ev, intent)

    # Deduplicate by element_id, keep best score
    best: dict[UUID, Evidence] = {}
    for ev in evidences:
        prev = best.get(ev.element_id)
        if prev is None or ev.score > prev.score:
            best[ev.element_id] = ev

    ranked = sorted(best.values(), key=lambda e: e.score, reverse=True)[:limit]

    # Attach related plates only when diagrams/procedures are relevant
    if intent["wants_figure"] or intent["wants_proc"]:
        related = _related_figures(manual_id, ranked, limit=4)
        for fig in related:
            if fig.element_id not in {e.element_id for e in ranked}:
                ranked.append(fig)
        ranked = sorted(ranked, key=lambda e: e.score, reverse=True)[: limit + 4]
    elif intent["wants_data"]:
        # Keep only plates already ranked high; drop low-score figure noise
        ranked = [
            e
            for e in ranked
            if not (
                e.source == "figure"
                and e.section_kind == "plate"
                and e.score < 3.0
            )
        ][:limit]

    result.evidences = ranked
    result.notes["paths_fired"] = sorted({e.source for e in ranked})
    result.notes["diagram_count"] = sum(1 for e in ranked if e.figure_image_path)
    result.notes["intent"] = {
        "wants_data": intent["wants_data"],
        "wants_proc": intent["wants_proc"],
        "wants_figure": intent["wants_figure"],
        "phrases": intent["phrases"][:6],
        "tokens": intent["tokens"][:10],
    }
    return result


def _apply_ranking_boosts(ev: Evidence, intent: dict[str, Any]) -> None:
    kind = (ev.section_kind or "").lower()
    title_blob = " ".join(ev.section_path or []).lower()
    code = (ev.doc_code or "").upper()
    text_l = (ev.text or "").lower()
    blob = f"{title_blob} {code.lower()} {text_l}"

    # Exact doc_code / section refs in the user query
    for c in intent["codes"]:
        if c == code or c in (ev.text or "").upper():
            ev.score += 4.0
    for sec in intent["sections"]:
        if sec in title_blob or sec in text_l:
            ev.score += 2.0

    # Multi-word component phrases in title/path beat bare "main"/"engine" noise
    for phrase in intent["phrases"]:
        if phrase in title_blob:
            ev.score += 5.0
            break
        elif phrase in text_l[:400]:
            ev.score += 2.5
            break

    # Single-token title hits (weaker than phrases)
    title_hits = sum(1 for t in intent["tokens"][:8] if t in title_blob)
    if title_hits:
        ev.score += min(2.5, 0.8 * title_hits)

    # Prefer data sheets for numeric/spec questions; procedures for how-to
    if intent["wants_data"] and kind == "data":
        ev.score += 3.0
        # Prefer denser rows that look like value tables, not tool lists
        if len(ev.text or "") > 120 and any(
            k in text_l
            for k in ("clearance", "diameter", "torque", "pressure", "wear", "mm", "bar", "nm", "kg", "max", "min")
        ):
            ev.score += 2.0
        elif "plate" in text_l and "item no" in text_l:
            ev.score -= 1.5
    if intent["wants_proc"] and kind == "procedure":
        ev.score += 3.5
    if intent["wants_proc"] and kind == "data" and not intent["wants_data"]:
        # Safety checklist on data sheets is useful, but don't bury procedure steps
        ev.score += 0.5
    if kind == "plate" and not intent["wants_figure"]:
        ev.score -= 2.0
    if kind in {"other", "schedule"} and (intent["wants_data"] or intent["wants_proc"]):
        ev.score -= 1.5

    # Torque / clearance / hydraulic cues in evidence text
    if intent["wants_data"]:
        if any(k in text_l for k in ("clearance", "hydraulic", "torque", "diameter", "wear")):
            ev.score += 1.0
        if "nm" in text_l and "torque" in blob:
            ev.score += 0.8


def _related_figures(manual_id: UUID, evidences: list[Evidence], *, limit: int) -> list[Evidence]:
    """Pull plate images for chapters touched by top evidence (troubleshooting needs diagrams)."""
    chapters: list[str] = []
    for e in evidences:
        if e.section_path and e.section_path[0] not in chapters:
            chapters.append(e.section_path[0])
    if not chapters:
        return []

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT e.id, f.page, coalesce(e.text, f.caption, '') AS text,
                       s.path, s.doc_code, s.edition, s.section_kind,
                       f.image_path, f.figure_label
                FROM figures f
                JOIN sections s ON s.id = f.section_id
                LEFT JOIN elements e ON e.id = f.element_id
                WHERE f.manual_id = %s
                  AND s.section_kind = 'plate'
                  AND s.path[1] = ANY(%s)
                ORDER BY f.page
                LIMIT %s
                """,
                (manual_id, chapters[:3], limit),
            )
            rows = cur.fetchall()
    return [_figure_evidence(r, score=2.3) for r in rows if r.get("id")]


def _figure_evidence(r: dict, *, score: float) -> Evidence:
    return Evidence(
        element_id=r["id"],
        page=r["page"],
        text=(r["text"] or f"Figure/plate on page {r['page']}")[:2000],
        section_path=r["path"],
        doc_code=r["doc_code"],
        edition=r["edition"],
        section_kind=r["section_kind"],
        score=score,
        source="figure",
        figure_image_path=r["image_path"],
        figure_label=r["figure_label"],
    )


def _lexical_search(
    manual_id: UUID, query: str, intent: dict[str, Any], *, limit: int
) -> list[Evidence]:
    fts_or = intent["fts_query"]
    fts_plain = intent["fts_plain"]
    with connect() as conn:
        with conn.cursor() as cur:
            rows: list[dict] = []
            if fts_or:
                cur.execute(
                    """
                    SELECT e.id, e.page, coalesce(e.text, '') AS text, e.table_json,
                           s.path, s.doc_code, s.edition, s.section_kind,
                           ts_rank(e.tsv, to_tsquery('english', %s)) AS rank
                    FROM elements e
                    LEFT JOIN sections s ON s.id = e.section_id
                    WHERE s.manual_id = %s
                      AND e.tsv @@ to_tsquery('english', %s)
                    ORDER BY
                      CASE WHEN s.section_kind = 'data' THEN 0
                           WHEN s.section_kind = 'procedure' THEN 1
                           ELSE 2 END,
                      rank DESC,
                      length(coalesce(e.text, '')) DESC,
                      e.page
                    LIMIT %s
                    """,
                    (fts_or, manual_id, fts_or, limit),
                )
                rows.extend(cur.fetchall())

            # Also try a shorter plain query (AND of fewer terms) for precision
            if fts_plain:
                cur.execute(
                    """
                    SELECT e.id, e.page, coalesce(e.text, '') AS text, e.table_json,
                           s.path, s.doc_code, s.edition, s.section_kind,
                           ts_rank(e.tsv, plainto_tsquery('english', %s)) + 0.5 AS rank
                    FROM elements e
                    LEFT JOIN sections s ON s.id = e.section_id
                    WHERE s.manual_id = %s
                      AND e.tsv @@ plainto_tsquery('english', %s)
                    ORDER BY
                      CASE WHEN s.section_kind = 'data' THEN 0
                           WHEN s.section_kind = 'procedure' THEN 1
                           ELSE 2 END,
                      rank DESC,
                      length(coalesce(e.text, '')) DESC,
                      e.page
                    LIMIT %s
                    """,
                    (fts_plain, manual_id, fts_plain, max(4, limit // 2)),
                )
                seen_ids = {r["id"] for r in rows}
                for r in cur.fetchall():
                    if r["id"] not in seen_ids:
                        rows.append(r)
                        seen_ids.add(r["id"])
            else:
                seen_ids = {r["id"] for r in rows}

            # Phrase / title-anchored pulls (avoids "main" → "Main Engine" noise)
            for phrase in intent["phrases"][:10]:
                pat = f"%{phrase}%"
                cur.execute(
                    """
                    SELECT e.id, e.page, coalesce(e.text, '') AS text, e.table_json,
                           s.path, s.doc_code, s.edition, s.section_kind,
                           2.5 AS rank
                    FROM elements e
                    LEFT JOIN sections s ON s.id = e.section_id
                    WHERE s.manual_id = %s
                      AND (
                        s.title ILIKE %s
                        OR array_to_string(s.path, ' ') ILIKE %s
                        OR e.text ILIKE %s
                      )
                    ORDER BY
                      CASE WHEN s.section_kind = 'data' THEN 0
                           WHEN s.section_kind = 'procedure' THEN 1
                           ELSE 2 END,
                      length(coalesce(e.text, '')) DESC,
                      e.page
                    LIMIT %s
                    """,
                    (manual_id, pat, pat, pat, limit),
                )
                for r in cur.fetchall():
                    if r["id"] not in seen_ids:
                        rows.append(r)
                        seen_ids.add(r["id"])

            # Exact identifier / code hits
            extra_tokens = list(intent["codes"]) + [
                t for t in re.findall(r"\b\d+\s*Nm\b", query, flags=re.I)
            ]
            for token in extra_tokens:
                pat = f"%{token}%"
                cur.execute(
                    """
                    SELECT e.id, e.page, coalesce(e.text, '') AS text, e.table_json,
                           s.path, s.doc_code, s.edition, s.section_kind,
                           1.5 AS rank
                    FROM elements e
                    LEFT JOIN sections s ON s.id = e.section_id
                    WHERE s.manual_id = %s
                      AND (
                        e.text ILIKE %s
                        OR e.table_json::text ILIKE %s
                        OR s.doc_code ILIKE %s
                      )
                    ORDER BY e.page
                    LIMIT %s
                    """,
                    (manual_id, pat, pat, pat, limit),
                )
                for r in cur.fetchall():
                    if r["id"] not in seen_ids:
                        rows.append(r)
                        seen_ids.add(r["id"])

    return [
        Evidence(
            element_id=r["id"],
            page=r["page"],
            text=_row_text(r),
            section_path=r["path"],
            doc_code=r["doc_code"],
            edition=r["edition"],
            section_kind=r["section_kind"],
            score=float(r["rank"] or 0) + 1.0,
            source="lexical",
        )
        for r in rows
    ]


def _structural_search(
    manual_id: UUID, intent: dict[str, Any], *, limit: int
) -> list[Evidence]:
    codes = intent["codes"]
    phrases = intent["phrases"]
    tokens = intent["tokens"]
    # Collect extra candidates; final retrieve() ranker picks the winners
    fetch_n = max(limit * 5, 20)
    kind_order = (
        "CASE WHEN s.section_kind = 'procedure' THEN 0 "
        "WHEN s.section_kind = 'data' THEN 1 ELSE 2 END"
        if intent["wants_proc"] and not intent["wants_data"]
        else "CASE WHEN s.section_kind = 'data' THEN 0 "
        "WHEN s.section_kind = 'procedure' THEN 1 ELSE 2 END"
    )

    with connect() as conn:
        with conn.cursor() as cur:
            rows: list[dict] = []
            if codes:
                cur.execute(
                    f"""
                    SELECT e.id, e.page, coalesce(e.text, '') AS text, e.table_json,
                           s.path, s.doc_code, s.edition, s.section_kind
                    FROM sections s
                    JOIN elements e ON e.section_id = s.id
                    WHERE s.manual_id = %s AND s.doc_code = ANY(%s)
                    ORDER BY {kind_order},
                      length(coalesce(e.text, '')) DESC,
                      e.page, e.id
                    LIMIT %s
                    """,
                    (manual_id, codes, fetch_n),
                )
                rows.extend(cur.fetchall())

            for phrase in phrases[:10]:
                cur.execute(
                    f"""
                    SELECT e.id, e.page, coalesce(e.text, '') AS text, e.table_json,
                           s.path, s.doc_code, s.edition, s.section_kind
                    FROM sections s
                    JOIN elements e ON e.section_id = s.id
                    WHERE s.manual_id = %s AND s.title ILIKE %s
                    ORDER BY {kind_order},
                      length(coalesce(e.text, '')) DESC,
                      e.page, e.id
                    LIMIT %s
                    """,
                    (manual_id, f"%{phrase}%", fetch_n),
                )
                rows.extend(cur.fetchall())

            comp = intent.get("component_tokens") or tokens
            has_useful = any(
                (r.get("section_kind") or "") in {"data", "procedure"} for r in rows
            )
            if len(comp) >= 2 and not has_useful:
                like_parts = " AND ".join(["s.title ILIKE %s"] * min(len(comp), 2))
                params: list[Any] = [manual_id]
                params.extend([f"%{t}%" for t in comp[:2]])
                params.append(fetch_n)
                cur.execute(
                    f"""
                    SELECT e.id, e.page, coalesce(e.text, '') AS text, e.table_json,
                           s.path, s.doc_code, s.edition, s.section_kind
                    FROM sections s
                    JOIN elements e ON e.section_id = s.id
                    WHERE s.manual_id = %s
                      AND s.section_kind IN ('data', 'procedure')
                      AND ({like_parts})
                    ORDER BY {kind_order},
                      length(coalesce(e.text, '')) DESC,
                      e.page, e.id
                    LIMIT %s
                    """,
                    params,
                )
                rows.extend(cur.fetchall())
            elif comp and not rows:
                cur.execute(
                    f"""
                    SELECT e.id, e.page, coalesce(e.text, '') AS text, e.table_json,
                           s.path, s.doc_code, s.edition, s.section_kind
                    FROM sections s
                    JOIN elements e ON e.section_id = s.id
                    WHERE s.manual_id = %s
                      AND s.section_kind IN ('data', 'procedure')
                      AND s.title ILIKE %s
                    ORDER BY {kind_order},
                      length(coalesce(e.text, '')) DESC, e.page, e.id
                    LIMIT %s
                    """,
                    (manual_id, f"%{comp[0]}%", fetch_n),
                )
                rows.extend(cur.fetchall())

    out: list[Evidence] = []
    seen: set[UUID] = set()
    for r in rows:
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        kind = r["section_kind"] or ""
        base = 3.0 if kind == "procedure" and intent["wants_proc"] else (
            2.8 if kind == "data" else 2.0
        )
        out.append(
            Evidence(
                element_id=r["id"],
                page=r["page"],
                text=_row_text(r),
                section_path=r["path"],
                doc_code=r["doc_code"],
                edition=r["edition"],
                section_kind=r["section_kind"],
                score=base,
                source="structural",
            )
        )
        if len(out) >= fetch_n:
            break
    return out


def _figure_search(
    manual_id: UUID, query: str, intent: dict[str, Any], *, limit: int
) -> list[Evidence]:
    wants_figure = intent["wants_figure"]
    # For data lookups, skip broad figure search (noise); structural/lexical cover plates if needed
    if intent["wants_data"] and not wants_figure and not intent["wants_proc"]:
        return []

    phrase = intent["phrases"][0] if intent["phrases"] else (intent["tokens"][0] if intent["tokens"] else query)
    with connect() as conn:
        with conn.cursor() as cur:
            if wants_figure:
                cur.execute(
                    """
                    SELECT e.id, f.page, coalesce(e.text, f.caption, '') AS text,
                           s.path, s.doc_code, s.edition, s.section_kind,
                           f.image_path, f.figure_label
                    FROM figures f
                    JOIN sections s ON s.id = f.section_id
                    LEFT JOIN elements e ON e.id = f.element_id
                    WHERE f.manual_id = %s
                      AND (
                        s.title ILIKE %s
                        OR s.doc_code ILIKE %s
                        OR coalesce(f.caption,'') ILIKE %s
                        OR s.section_kind = 'plate'
                      )
                    ORDER BY
                      CASE WHEN s.title ILIKE %s OR s.doc_code ILIKE %s THEN 0 ELSE 1 END,
                      f.page
                    LIMIT %s
                    """,
                    (
                        manual_id,
                        f"%{phrase}%",
                        f"%{phrase}%",
                        f"%{phrase}%",
                        f"%{phrase}%",
                        f"%{phrase}%",
                        limit,
                    ),
                )
            else:
                cur.execute(
                    """
                    SELECT e.id, f.page, coalesce(e.text, f.caption, '') AS text,
                           s.path, s.doc_code, s.edition, s.section_kind,
                           f.image_path, f.figure_label
                    FROM figures f
                    JOIN sections s ON s.id = f.section_id
                    LEFT JOIN elements e ON e.id = f.element_id
                    WHERE f.manual_id = %s
                      AND (
                        s.title ILIKE %s
                        OR s.doc_code ILIKE %s
                        OR coalesce(f.caption,'') ILIKE %s
                      )
                    ORDER BY f.page
                    LIMIT %s
                    """,
                    (manual_id, f"%{phrase}%", f"%{phrase}%", f"%{phrase}%", limit),
                )
            rows = cur.fetchall()

    score = 2.5 if wants_figure else 1.2
    return [
        Evidence(
            element_id=r["id"],
            page=r["page"],
            text=(r["text"] or f"Figure/plate on page {r['page']}")[:2000],
            section_path=r["path"],
            doc_code=r["doc_code"],
            edition=r["edition"],
            section_kind=r["section_kind"],
            score=score,
            source="figure",
            figure_image_path=r["image_path"],
            figure_label=r["figure_label"],
        )
        for r in rows
        if r["id"] is not None
    ]
