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
    # Primary citation keys (spec §6.1) plus panel pairing fields (§6.2)
    procedure_no: str | None = None
    plate_no: str | None = None
    page_number_printed: str | None = None
    element_type: str | None = None
    panel_index: int | None = None
    drawing_code: str | None = None
    linked_step_number: int | None = None
    component_title: str | None = None
    action_title: str | None = None
    citation_key: str | None = None

    @property
    def citation_ref(self) -> str:
        """How the manual asks to be quoted: its citation_key plus revision if present."""
        primary = self.plate_no or self.procedure_no or self.citation_key or self.doc_code
        if not primary:
            return f"page {self.page}"
        label = "Plate" if self.plate_no else "Procedure"
        ref = f"{label} {primary}"
        return f"{ref} Edition {self.edition}" if self.edition else ref


# Shared column list for element queries that join `sections s`.
ELEMENT_COLUMNS = """
    e.id, e.page, coalesce(e.text, '') AS text, e.table_json, e.type AS element_type,
    e.procedure_no, e.plate_no, coalesce(e.edition, s.edition) AS edition,
    coalesce(e.citation_key, e.plate_no, e.procedure_no, s.doc_code) AS citation_key,
    e.page_number_printed, e.panel_index, e.drawing_code, e.linked_step_number,
    s.path, s.doc_code, s.section_kind, s.component_title, s.action_title
"""


def row_to_evidence(row: dict, *, score: float, source: str) -> Evidence:
    return Evidence(
        element_id=row["id"],
        page=row["page"],
        text=_row_text(row) or f"Figure/plate on page {row['page']}",
        section_path=row.get("path"),
        doc_code=row.get("doc_code"),
        edition=row.get("edition"),
        section_kind=row.get("section_kind"),
        score=score,
        source=source,
        figure_image_path=row.get("image_path"),
        figure_label=row.get("figure_label") or row.get("drawing_code"),
        procedure_no=row.get("procedure_no"),
        plate_no=row.get("plate_no"),
        page_number_printed=row.get("page_number_printed"),
        element_type=row.get("element_type"),
        panel_index=row.get("panel_index"),
        drawing_code=row.get("drawing_code"),
        linked_step_number=row.get("linked_step_number"),
        component_title=row.get("component_title"),
        action_title=row.get("action_title"),
        citation_key=row.get("citation_key"),
    )


@dataclass
class RetrievalResult:
    evidences: list[Evidence] = field(default_factory=list)
    fleet_model: str | None = None
    manual_title: str | None = None
    revision: str | None = None
    notes: dict[str, Any] = field(default_factory=dict)


DOC_CODE_RE = re.compile(r"\b([A-Z]{1,3}\d{4,8})\b", re.IGNORECASE)
SECTION_RE = re.compile(r"\b(\d{3}-\d+(?:\.\d+)?)\b")
DRAWING_CODE_RE = re.compile(
    r"\b([A-Z]{1,3}\d{3,6}-\d{1,5}(?:\.\d+)?(?:[A-Z]\d{0,4})?)\b",
    re.IGNORECASE,
)

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
        "mounting", "dismantling", "tightening", "hours", "interval", "schedule",
        "mass",
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

# Keep / replace / scrap / "is this valid" — needs the checking procedure AND the
# limits table. Not OEM-specific: any IETM that splits spec sheets from how-to text.
DECISION_HINTS = frozenset(
    {
        "replace", "reuse", "reinstall", "keep", "scrap", "discard",
        "valid", "oversize", "dummy", "walk", "decision",
        "criteria", "accept", "reject",
    }
)

_PRECISION_STOP = frozenset(
    {
        "after", "before", "which", "what", "does", "with", "from", "that", "this",
        "have", "been", "were", "while", "where", "when", "your", "need", "must",
        "still", "same", "both", "each", "unit", "work", "quote", "include",
        "engine", "manual", "procedure", "actually", "required", "complete",
        "whether", "decision", "compare", "state", "walk", "give", "cite",
        "there", "their", "about", "would", "could", "should", "using", "during",
    }
)


def overlap_score(
    text: str,
    tokens: list[str],
    *,
    distinctive: list[str] | None = None,
    element_type: str | None = None,
) -> float:
    """Boost short criterion sentences that reuse several query tokens."""
    blob = re.sub(r"[-_/]", " ", (text or "").lower())
    if not blob or not tokens:
        return 0.0
    qtoks = [t for t in tokens if len(t) >= 4 or t[:1].isdigit()]
    dist = [t for t in (distinctive or []) if len(t) >= 5]
    th = sum(1 for t in qtoks if t in blob)
    dh = sum(1 for t in dist if t in blob)
    boost = 0.0
    if dh:
        boost += 4.0 * dh
        if len(blob) < 400:
            boost += 6.0 * dh
    for t in dist:
        if t[:1].isdigit() and t in blob:
            boost += 8.0
    if (element_type or "") == "list" and any(
        k in blob for k in ("minute", "hour", "must not exceed", "keep under")
    ):
        boost += 6.0
    if th >= 2:
        if len(blob) < 500:
            boost += 3.0 + 1.5 * th
        else:
            boost += min(8.0, 0.6 * th)
    if (element_type or "") in {"list", "heading"} and (dh or th >= 2):
        boost += 4.0
    return boost


# Generic keep/replace / delta-from-reference phrasing (not OEM- or question-specific)
DECISION_PHRASES = (
    "keep or", "replace or", "still valid", "walk me", "step by step",
    "if deviation", "checking procedure", "acceptance criteria",
    "which of", "actions required", "walk the", "last mounting",
    "above the", "below the", "above adjustment", "below adjustment",
    "as fitted", "as-fitted", "recorded value", "adjustment sheet",
    "shut off", "lifting plan", "same keep", "same decision", "compare two",
)

FIGURE_HINTS = frozenset(
    {
        "diagram", "figure", "plate", "drawing", "panel", "wiring", "piping",
        "illustration", "image", "picture", "callout", "sketch",
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
    # Split hyphenated compounds (top-clearance → top, clearance). Keep numbers so
    # "0.10 mm" can match the paragraph that states that threshold.
    q = (query or "").replace(",", "")
    raw = re.findall(r"[A-Za-z][A-Za-z0-9]*|\d+(?:\.\d+)?", q)
    out: list[str] = []
    seen: set[str] = set()
    for t in raw:
        low = t.lower()
        if re.fullmatch(r"\d+(?:\.\d+)?", low):
            if "." in low or len(low) >= 2:
                if low not in seen:
                    seen.add(low)
                    out.append(low)
            continue
        if low in STOPWORDS:
            continue
        if len(low) < 3 and low not in DATA_HINTS:
            continue
        stem = _singularize(low)
        if stem in STOPWORDS or stem in seen:
            continue
        seen.add(stem)
        out.append(stem)
    return out


def _phrases(tokens: list[str]) -> list[str]:
    """Bigrams first (component names), then skip-grams, then trigrams."""
    bigrams: list[str] = []
    trigrams: list[str] = []
    skips: list[str] = []
    for i in range(len(tokens) - 1):
        bigrams.append(" ".join(tokens[i : i + 2]))
    for i in range(len(tokens) - 2):
        trigrams.append(" ".join(tokens[i : i + 3]))
        a, b = tokens[i], tokens[i + 2]
        if len(a) >= 5 and len(b) >= 3:
            skips.append(f"{a} {b}")
    return bigrams[:8] + skips + bigrams[8:] + trigrams


def _fts_or_query(tokens: list[str]) -> str:
    """OR-query so long questions don't fail plainto_tsquery's implicit AND."""
    parts: list[str] = []
    for t in tokens[:10]:
        safe = re.sub(r"[^a-z0-9]", "", t.lower())
        if len(safe) >= 3:
            parts.append(safe)
    return " | ".join(parts) if parts else ""


def _kind_order_sql(intent: dict[str, Any]) -> str:
    """SQL sort for section_kind. When the question is a decision, do not bury procedures."""
    if intent.get("wants_schedule"):
        return (
            "CASE WHEN s.section_kind IN ('schedule', 'data', 'procedure') THEN 0 ELSE 1 END"
        )
    if intent.get("wants_proc") and intent.get("wants_data"):
        return (
            "CASE WHEN s.section_kind IN ('data', 'procedure') THEN 0 ELSE 1 END"
        )
    if intent.get("wants_proc") and not intent.get("wants_data"):
        return (
            "CASE WHEN s.section_kind = 'procedure' THEN 0 "
            "WHEN s.section_kind = 'data' THEN 1 ELSE 2 END"
        )
    return (
        "CASE WHEN s.section_kind = 'data' THEN 0 "
        "WHEN s.section_kind = 'procedure' THEN 1 ELSE 2 END"
    )


def _query_intent(query: str) -> dict[str, Any]:
    q = query.lower()
    tokens = _content_tokens(query)
    token_set = set(tokens)
    wants_figure = (
        any(h in q for h in FIGURE_HINTS)
        or "show me" in q
        or "show the" in q
        or bool(re.search(r"\bP\d{4,6}\b", query, re.I))
        or bool(re.search(r"\b[A-Z]{1,3}\d{3,6}-\d", query, re.I))
    )
    wants_data = bool(token_set & DATA_HINTS) or bool(
        re.search(r"\b\d+(?:\.\d+)?\s*(mm|bar|nm|kg)\b", q)
    )
    proc_hits = token_set & PROC_HINTS
    wants_decision = bool(token_set & DECISION_HINTS) or any(p in q for p in DECISION_PHRASES)
    if not wants_decision:
        wants_decision = any(
            re.search(rf"\b{re.escape(h)}\b", q) for h in DECISION_HINTS
        )
    if not wants_decision and re.search(
        r"\b(above|below)\b.{0,48}\b(sheet|recorded|fitted|original|reference|as[- ]fitted)\b"
        r"|\b(sheet|recorded|fitted|original|reference)\b.{0,48}\b(above|below)\b",
        q,
    ):
        wants_decision = True
    wants_proc = wants_decision or (
        bool(proc_hits)
        and (
            q.strip().startswith("how ")
            or any(
                v in token_set
                for v in (
                    "remove", "replace", "install", "open", "step",
                    "procedure", "safety", "precaution",
                )
            )
            or (not wants_data)
            or len(proc_hits - {"dismantling", "mounting", "dismantle", "mount"}) > 0
        )
    )
    if wants_decision:
        wants_data = True
        wants_proc = True
    # A named procedure/plate plus numbers still needs the complementary kind.
    named = bool(DOC_CODE_RE.search(query) or SECTION_RE.search(query))
    if named and wants_data:
        wants_proc = True
    if named and wants_proc:
        wants_data = True
    if re.search(r"\bwhich\b", q) and wants_data:
        wants_decision = True
        wants_proc = True
        wants_data = True
    wants_schedule = bool(
        token_set & {"hours", "overhaul", "interval", "schedule"}
    ) or bool(re.search(r"\b\d[\d,]*\s*h(?:ours?)?\b", q)) or bool(
        re.search(r"\b(overhaul|interval|schedule|running check)\b", q)
    )
    if wants_schedule:
        wants_proc = True
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
        "wants_decision": wants_decision,
        "wants_schedule": wants_schedule,
        "codes": [m.group(1).upper() for m in DOC_CODE_RE.finditer(query)],
        "sections": [m.group(1) for m in SECTION_RE.finditer(query)],
        "drawing_codes": [m.group(1).upper() for m in DRAWING_CODE_RE.finditer(query)],
        "fts_query": _fts_or_query(fts_tokens),
        "fts_plain": " ".join(fts_tokens) or query,
    }


def get_fleet_context() -> dict[str, Any] | None:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SAVEPOINT fleet_ctx")
            try:
                cur.execute(
                    """
                    SELECT fr.model, fr.equipment_id, fr.revision_status,
                           m.id AS manual_id, m.title, m.revision,
                           m.source_pdf_path, m.citation_convention, m.manufacturer
                    FROM fleet_registry fr
                    JOIN manuals m ON m.id = fr.manual_id
                    WHERE fr.revision_status = 'current'
                    ORDER BY fr.id
                    LIMIT 1
                    """
                )
            except Exception:
                cur.execute("ROLLBACK TO SAVEPOINT fleet_ctx")
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

    from marine_docs.ranking import fetch_kind_pairs

    extra = fetch_kind_pairs(
        [evidence_to_dict(e) for e in ranked],
        manual_id=manual_id,
        exclude={str(e.element_id) for e in ranked},
    )
    if extra:
        extra_ids = {item["element_id"] for item in extra}
        ranked = [dict_to_evidence(item) for item in extra] + [
            e for e in ranked if str(e.element_id) not in extra_ids
        ]
        result.notes["kind_pairs"] = len(extra)

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
        "wants_decision": intent.get("wants_decision"),
        "wants_figure": intent["wants_figure"],
        "phrases": intent["phrases"][:6],
        "tokens": intent["tokens"][:10],
    }
    return result


def _apply_ranking_boosts(ev: Evidence, intent: dict[str, Any]) -> None:
    kind = (ev.section_kind or "").lower()
    title_blob = " ".join(
        (ev.section_path or []) + [ev.component_title or "", ev.action_title or ""]
    ).lower()
    code = (ev.doc_code or "").upper()
    text_l = (ev.text or "").lower()
    blob = f"{title_blob} {code.lower()} {text_l}"

    # Exact citation keys named in the query (spec §7.1 handles final pinning)
    for c in intent["codes"]:
        if c == code or c in (ev.text or "").upper():
            ev.score += 4.0
    for sec in intent["sections"]:
        if sec == (ev.procedure_no or ""):
            ev.score += 6.0
        elif sec in title_blob or sec in text_l:
            ev.score += 2.0

    # Multi-word component phrases in title/path beat bare "main"/"engine" noise
    phrase_in_title = False
    for phrase in intent["phrases"]:
        if phrase in title_blob:
            ev.score += 8.0
            phrase_in_title = True
            break
        elif phrase in text_l[:400]:
            ev.score += 2.5
            break

    distinctive = [t for t in (intent.get("component_tokens") or []) if len(t) >= 5]
    numeric = [
        t
        for t in (intent.get("tokens") or [])
        if t[:1].isdigit() and ("." in t or len(t) >= 3)
    ]
    distinctive = list(dict.fromkeys(distinctive + numeric))
    title_component_hits = sum(1 for t in distinctive if t in title_blob)
    if title_component_hits:
        ev.score += min(6.0, 2.0 * title_component_hits)
    elif distinctive and (ev.component_title or "").strip() and not phrase_in_title:
        if not any(t in title_blob for t in distinctive):
            ev.score -= 10.0

    # Single-token title hits (weaker than phrases)
    title_hits = sum(1 for t in intent["tokens"][:8] if t in title_blob)
    if title_hits:
        ev.score += min(2.5, 0.8 * title_hits)

    # Prefer data sheets for numeric/spec questions; procedures for how-to
    if intent["wants_data"] and kind == "data":
        ev.score += 3.0
        # Prefer denser rows that look like value tables, not tool lists —
        # but not so hard that checking-procedure text cannot compete.
        if not intent.get("wants_proc") and len(ev.text or "") > 120 and any(
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
        if kind == "schedule" and intent.get("wants_schedule"):
            ev.score += 4.0
        else:
            ev.score -= 1.5
    if intent.get("wants_data") or "shut" in " ".join(intent.get("tokens") or []):
        if any(k in text_l for k in ("shut off", "precaution", "safety check", "lubricating oil")):
            ev.score += 2.5

    # Torque / clearance / hydraulic cues in evidence text
    if intent["wants_data"]:
        if any(k in text_l for k in ("clearance", "hydraulic", "torque", "diameter", "wear")):
            ev.score += 1.0
        if "nm" in text_l and "torque" in blob:
            ev.score += 0.8

    ev.score += overlap_score(
        ev.text or "",
        intent.get("tokens") or [],
        distinctive=distinctive,
        element_type=ev.element_type,
    )
    action = (ev.action_title or "").lower()
    qjoin = " ".join(intent.get("tokens") or [])
    title_words = [
        w
        for w in (ev.component_title or "").lower().split()
        if len(w) >= 5 and w not in {"bearing", "engine", "complete"}
    ]
    if any(p in qjoin for p in ("spring air", "hard faced", "hard-faced")) and "fuel" not in qjoin:
        if "fuel" in title_blob:
            ev.score -= 20.0
    if intent.get("wants_decision") or any(k in qjoin for k in ("stay", "keep", "reuse", "scrap")):
        if "check" in action:
            ev.score += 3.0
        if "dismantl" in action and "dismantl" not in qjoin:
            ev.score -= 5.0
    if any(k in qjoin for k in ("mount", "landed", "landing")):
        if "mount" in text_l[:220] or "mount" in action:
            ev.score += 3.5
        if "dismantl" in action and "dismantl" not in qjoin:
            ev.score -= 3.0


# Figure queries read the caption from `figures` when the element itself has no text.
FIGURE_COLUMNS = """
    e.id, f.page, coalesce(e.text, f.caption, '') AS text, e.table_json,
    e.type AS element_type, e.procedure_no, e.plate_no,
    coalesce(e.edition, s.edition) AS edition, e.page_number_printed,
    coalesce(e.panel_index, f.panel_index) AS panel_index,
    coalesce(e.drawing_code, f.drawing_code) AS drawing_code,
    coalesce(e.linked_step_number, f.linked_step_number) AS linked_step_number,
    s.path, s.doc_code, s.section_kind, s.component_title, s.action_title,
    coalesce(e.citation_key, e.plate_no, e.procedure_no, s.doc_code) AS citation_key,
    f.image_path, f.figure_label
"""


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
                f"""
                SELECT {FIGURE_COLUMNS}
                FROM figures f
                JOIN sections s ON s.id = f.section_id
                LEFT JOIN elements e ON e.id = f.element_id
                WHERE f.manual_id = %s
                  AND s.section_kind = 'plate'
                  AND s.path[1] = ANY(%s)
                ORDER BY f.page, f.panel_index
                LIMIT %s
                """,
                (manual_id, chapters[:3], limit),
            )
            rows = cur.fetchall()
    return [row_to_evidence(r, score=2.3, source="figure") for r in rows if r.get("id")]


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
                    f"""
                    SELECT {ELEMENT_COLUMNS},
                           ts_rank(e.tsv, to_tsquery('english', %s)) AS rank
                    FROM elements e
                    LEFT JOIN sections s ON s.id = e.section_id
                    WHERE s.manual_id = %s
                      AND e.tsv @@ to_tsquery('english', %s)
                    ORDER BY
                      {_kind_order_sql(intent)},
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
                    f"""
                    SELECT {ELEMENT_COLUMNS},
                           ts_rank(e.tsv, plainto_tsquery('english', %s)) + 0.5 AS rank
                    FROM elements e
                    LEFT JOIN sections s ON s.id = e.section_id
                    WHERE s.manual_id = %s
                      AND e.tsv @@ plainto_tsquery('english', %s)
                    ORDER BY
                      {_kind_order_sql(intent)},
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
            for phrase in intent["phrases"][:14]:
                pat = f"%{phrase}%"
                cur.execute(
                    f"""
                    SELECT {ELEMENT_COLUMNS}, 2.5 AS rank
                    FROM elements e
                    LEFT JOIN sections s ON s.id = e.section_id
                    WHERE s.manual_id = %s
                      AND (
                        s.title ILIKE %s
                        OR array_to_string(s.path, ' ') ILIKE %s
                        OR e.text ILIKE %s
                        OR s.component_title ILIKE %s
                      )
                    ORDER BY
                      {_kind_order_sql(intent)},
                      length(coalesce(e.text, '')) ASC,
                      e.page
                    LIMIT %s
                    """,
                    (manual_id, pat, pat, pat, pat, 8),
                )
                for r in cur.fetchall():
                    if r["id"] not in seen_ids:
                        rows.append(r)
                        seen_ids.add(r["id"])

            # Exact identifier / code hits, including the manual's own citation keys
            extra_tokens = (
                list(intent["codes"])
                + list(intent["sections"])
                + list(intent.get("drawing_codes") or [])
                + [t for t in re.findall(r"\b\d+\s*Nm\b", query, flags=re.I)]
            )
            for token in extra_tokens:
                pat = f"%{token}%"
                cur.execute(
                    f"""
                    SELECT {ELEMENT_COLUMNS}, 1.5 AS rank
                    FROM elements e
                    LEFT JOIN sections s ON s.id = e.section_id
                    WHERE s.manual_id = %s
                      AND (
                        e.text ILIKE %s
                        OR e.table_json::text ILIKE %s
                        OR s.doc_code ILIKE %s
                        OR e.procedure_no = %s
                        OR e.plate_no = %s
                        OR e.drawing_code ILIKE %s
                      )
                    ORDER BY e.page
                    LIMIT %s
                    """,
                    (manual_id, pat, pat, pat, token, token.upper(), pat, limit),
                )
                for r in cur.fetchall():
                    if r["id"] not in seen_ids:
                        rows.append(r)
                        seen_ids.add(r["id"])

            # Short criterion sentences (drop-time bands, "aftmost = …") lose to
            # long tables in FTS. Pull them by distinctive token overlap.
            distinctive = [
                t
                for t in (intent.get("tokens") or [])
                if (len(t) >= 5 and t not in _PRECISION_STOP)
                or (t[:1].isdigit() and ("." in t or len(t) >= 3))
            ][:10]
            overlap_toks = [
                t
                for t in (intent.get("tokens") or [])
                if len(t) >= 3 and t not in _PRECISION_STOP
            ][:12]
            if distinctive:
                likes = " OR ".join(["e.text ILIKE %s"] * len(distinctive))
                score_sql = " + ".join(["(e.text ILIKE %s)::int" for _ in overlap_toks] or ["0"])
                params = [manual_id, *[f"%{t}%" for t in distinctive]]
                if overlap_toks:
                    params.extend(f"%{t}%" for t in overlap_toks)
                else:
                    score_sql = "0"
                cur.execute(
                    f"""
                    SELECT {ELEMENT_COLUMNS}, 3.8 AS rank
                    FROM elements e
                    LEFT JOIN sections s ON s.id = e.section_id
                    WHERE s.manual_id = %s
                      AND length(coalesce(e.text, '')) BETWEEN 24 AND 2500
                      AND ({likes})
                    ORDER BY ({score_sql}) DESC, length(e.text) ASC
                    LIMIT 16
                    """,
                    params,
                )
                for r in cur.fetchall():
                    if r["id"] not in seen_ids:
                        rows.append(r)
                        seen_ids.add(r["id"])

    return [
        row_to_evidence(r, score=float(r["rank"] or 0) + 1.0, source="lexical")
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
    kind_order = _kind_order_sql(intent)

    with connect() as conn:
        with conn.cursor() as cur:
            rows: list[dict] = []
            if codes or intent["sections"]:
                cur.execute(
                    f"""
                    SELECT {ELEMENT_COLUMNS}
                    FROM sections s
                    JOIN elements e ON e.section_id = s.id
                    WHERE s.manual_id = %s
                      AND (
                        s.doc_code = ANY(%s)
                        OR e.procedure_no = ANY(%s)
                        OR s.procedure_no = ANY(%s)
                      )
                    ORDER BY {kind_order},
                      length(coalesce(e.text, '')) DESC,
                      e.page, e.id
                    LIMIT %s
                    """,
                    (manual_id, codes, intent["sections"], intent["sections"], fetch_n),
                )
                rows.extend(cur.fetchall())

            for phrase in phrases[:10]:
                cur.execute(
                    f"""
                    SELECT {ELEMENT_COLUMNS}
                    FROM sections s
                    JOIN elements e ON e.section_id = s.id
                    WHERE s.manual_id = %s
                      AND (s.title ILIKE %s OR s.component_title ILIKE %s)
                    ORDER BY {kind_order},
                      length(coalesce(e.text, '')) DESC,
                      e.page, e.id
                    LIMIT %s
                    """,
                    (manual_id, f"%{phrase}%", f"%{phrase}%", fetch_n),
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
                    SELECT {ELEMENT_COLUMNS}
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
                    SELECT {ELEMENT_COLUMNS}
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
        out.append(row_to_evidence(r, score=base, source="structural"))
        if len(out) >= fetch_n:
            break
    return out


def _backfill_figure_ocr(rows: list[dict]) -> None:
    """Plate figure rows often store only a caption; OCR callouts live on a sibling."""
    if not rows:
        return
    pages: list[int] = []
    codes: list[str] = []
    for r in rows:
        text = (r.get("text") or "").strip()
        if len(text) >= 160 and "callout" in text.lower():
            continue
        if r.get("page"):
            pages.append(int(r["page"]))
        if r.get("doc_code"):
            codes.append(str(r["doc_code"]))
    if not pages and not codes:
        return
    try:
        with connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT s.doc_code, e.page, e.text
                FROM elements e
                JOIN sections s ON s.id = e.section_id
                WHERE e.type IN ('paragraph', 'table', 'list')
                  AND length(coalesce(e.text, '')) > 40
                  AND (
                    e.page = ANY(%s)
                    OR s.doc_code = ANY(%s)
                  )
                ORDER BY
                  CASE WHEN e.text ~* 'callout|item no|item description' THEN 0 ELSE 1 END,
                  length(e.text) DESC
                LIMIT 40
                """,
                (pages or [0], codes or [""]),
            )
            extras = cur.fetchall()
    except Exception:
        return
    by_code: dict[str, str] = {}
    by_page: dict[int, str] = {}
    for x in extras:
        text = x.get("text") or ""
        code = x.get("doc_code")
        if code and code not in by_code:
            by_code[code] = text
        page = x.get("page")
        if page is not None and int(page) not in by_page:
            by_page[int(page)] = text
    for r in rows:
        current = (r.get("text") or "").strip()
        if len(current) >= 160 and "callout" in current.lower():
            continue
        fill = by_code.get(r.get("doc_code") or "") or by_page.get(int(r.get("page") or 0))
        if not fill:
            continue
        if current and current not in fill:
            r["text"] = (current + "\n" + fill)[:2000]
        else:
            r["text"] = fill[:2000]


def _figure_search(
    manual_id: UUID, query: str, intent: dict[str, Any], *, limit: int
) -> list[Evidence]:
    wants_figure = intent["wants_figure"]
    # For data lookups, skip broad figure search (noise); structural/lexical cover plates if needed
    if intent["wants_data"] and not wants_figure and not intent["wants_proc"]:
        return []

    phrases = list(intent.get("phrases") or [])
    if not phrases:
        seed = intent["tokens"][0] if intent.get("tokens") else query
        phrases = [seed]
    drawing_codes = list(intent.get("drawing_codes") or [])
    plate_nos = [m.group(0).upper() for m in re.finditer(r"\bP\d{4,6}\b", query, re.I)]
    tokens = intent.get("tokens") or []

    with connect() as conn:
        with conn.cursor() as cur:
            rows: list[dict] = []
            seen_fig: set = set()

            def add_rows(fetched) -> None:
                for r in fetched:
                    if r.get("id") and r["id"] not in seen_fig:
                        rows.append(r)
                        seen_fig.add(r["id"])

            for tok in drawing_codes + plate_nos:
                p2 = f"%{tok}%"
                cur.execute(
                    f"""
                    SELECT {FIGURE_COLUMNS}
                    FROM figures f
                    JOIN sections s ON s.id = f.section_id
                    LEFT JOIN elements e ON e.id = f.element_id
                    WHERE f.manual_id = %s
                      AND (
                        coalesce(f.drawing_code,'') ILIKE %s
                        OR coalesce(f.caption,'') ILIKE %s
                        OR s.doc_code ILIKE %s
                        OR coalesce(s.plate_no,'') ILIKE %s
                        OR coalesce(e.drawing_code,'') ILIKE %s
                      )
                    ORDER BY f.page, f.panel_index
                    LIMIT 8
                    """,
                    (manual_id, p2, p2, p2, p2, p2),
                )
                add_rows(cur.fetchall())

            for phrase in phrases[:8]:
                pat = f"%{phrase}%"
                if wants_figure:
                    cur.execute(
                        f"""
                        SELECT {FIGURE_COLUMNS}
                        FROM figures f
                        JOIN sections s ON s.id = f.section_id
                        LEFT JOIN elements e ON e.id = f.element_id
                        WHERE f.manual_id = %s
                          AND (
                            s.title ILIKE %s
                            OR s.component_title ILIKE %s
                            OR s.doc_code ILIKE %s
                            OR coalesce(f.caption,'') ILIKE %s
                            OR coalesce(f.drawing_code,'') ILIKE %s
                          )
                        ORDER BY
                          CASE WHEN s.section_kind = 'plate' THEN 0 ELSE 1 END,
                          CASE WHEN s.title ILIKE %s OR s.component_title ILIKE %s THEN 0 ELSE 1 END,
                          f.page, f.panel_index
                        LIMIT %s
                        """,
                        (manual_id, pat, pat, pat, pat, pat, pat, pat, max(6, limit)),
                    )
                else:
                    cur.execute(
                        f"""
                        SELECT {FIGURE_COLUMNS}
                        FROM figures f
                        JOIN sections s ON s.id = f.section_id
                        LEFT JOIN elements e ON e.id = f.element_id
                        WHERE f.manual_id = %s
                          AND (
                            s.title ILIKE %s
                            OR s.component_title ILIKE %s
                            OR s.doc_code ILIKE %s
                            OR coalesce(f.caption,'') ILIKE %s
                          )
                        ORDER BY f.page, f.panel_index
                        LIMIT %s
                        """,
                        (manual_id, pat, pat, pat, pat, max(4, limit // 2)),
                    )
                add_rows(cur.fetchall())

    _backfill_figure_ocr(rows)

    def _norm(value: str) -> str:
        return re.sub(r"[\s_]+", "", (value or "").upper())

    named = {_norm(c) for c in drawing_codes + plate_nos if c}

    def rank_key(r: dict) -> tuple:
        blob = " ".join(
            [
                r.get("component_title") or "",
                r.get("action_title") or "",
                r.get("doc_code") or "",
                r.get("plate_no") or "",
                r.get("drawing_code") or "",
                r.get("text") or "",
                r.get("figure_label") or "",
            ]
        ).lower()
        dc = _norm(r.get("drawing_code") or "") + _norm(r.get("doc_code") or "") + _norm(
            r.get("plate_no") or ""
        )
        named_hit = 1 if any(c and c in dc for c in named) else 0
        overlap = sum(1 for t in tokens if len(t) >= 4 and t in blob)
        plate = 0 if (r.get("section_kind") or "") == "plate" else 1
        return (-named_hit, -overlap, plate, int(r.get("page") or 0))

    rows.sort(key=rank_key)
    rows = rows[:limit]
    out: list[Evidence] = []
    for r in rows:
        if r.get("id") is None:
            continue
        named_hit = rank_key(r)[0] == -1
        overlap = -rank_key(r)[1]
        base = 2.5 if wants_figure else 1.2
        score = base + (8.0 if named_hit else 0.0) + min(6.0, 1.2 * overlap)
        out.append(row_to_evidence(r, score=score, source="figure"))
    return out


def evidence_to_dict(ev: Evidence) -> dict[str, Any]:
    return {
        "element_id": str(ev.element_id),
        "page": ev.page,
        "text": ev.text,
        "section_path": ev.section_path,
        "doc_code": ev.doc_code,
        "edition": ev.edition,
        "section_kind": ev.section_kind,
        "score": ev.score,
        "source": ev.source,
        "figure_image_path": ev.figure_image_path,
        "figure_label": ev.figure_label,
        "procedure_no": ev.procedure_no,
        "plate_no": ev.plate_no,
        "page_number_printed": ev.page_number_printed,
        "element_type": ev.element_type,
        "panel_index": ev.panel_index,
        "drawing_code": ev.drawing_code,
        "linked_step_number": ev.linked_step_number,
        "component_title": ev.component_title,
        "action_title": ev.action_title,
        "citation_key": ev.citation_key,
        "citation_ref": ev.citation_ref,
    }


def dict_to_evidence(d: dict[str, Any]) -> Evidence:
    return Evidence(
        element_id=UUID(str(d["element_id"])),
        page=int(d.get("page") or 0),
        text=d.get("text") or "",
        section_path=d.get("section_path"),
        doc_code=d.get("doc_code"),
        edition=d.get("edition"),
        section_kind=d.get("section_kind"),
        score=float(d.get("score") or 0),
        source=d.get("source") or "lexical",
        figure_image_path=d.get("figure_image_path"),
        figure_label=d.get("figure_label"),
        procedure_no=d.get("procedure_no"),
        plate_no=d.get("plate_no"),
        page_number_printed=d.get("page_number_printed"),
        element_type=d.get("element_type"),
        panel_index=d.get("panel_index"),
        drawing_code=d.get("drawing_code"),
        linked_step_number=d.get("linked_step_number"),
        component_title=d.get("component_title"),
        action_title=d.get("action_title"),
        citation_key=d.get("citation_key"),
    )


def list_document_tree(*, limit: int = 280) -> list[dict[str, Any]]:
    """Compact section catalog for PageIndex-style tree navigation (no embeddings)."""
    fleet = get_fleet_context()
    if not fleet:
        return []
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT s.id, s.path, s.title, s.doc_code, s.section_kind,
                       s.procedure_no, s.plate_no, s.edition,
                       s.component_title, s.action_title,
                       s.page_start, s.page_end
                FROM sections s
                WHERE s.manual_id = %s
                ORDER BY s.page_start NULLS LAST, s.path
                LIMIT %s
                """,
                (fleet["manual_id"], limit),
            )
            rows = cur.fetchall()
    out = []
    for r in rows:
        out.append(
            {
                "id": str(r["id"]),
                "path": r["path"],
                "title": r["title"],
                "doc_code": r["doc_code"],
                "kind": r["section_kind"],
                "procedure_no": r["procedure_no"],
                "plate_no": r["plate_no"],
                "edition": r["edition"],
                "component": r["component_title"],
                "action": r["action_title"],
                "pages": f"{r['page_start'] or '?'}-{r['page_end'] or '?'}",
            }
        )
    return out


def lexical_path(query: str, *, limit: int = 20) -> list[dict[str, Any]]:
    fleet = get_fleet_context()
    if not fleet:
        return []
    intent = _query_intent(query)
    hits = _lexical_search(fleet["manual_id"], query, intent, limit=limit)
    for ev in hits:
        _apply_ranking_boosts(ev, intent)
    hits.sort(key=lambda e: e.score, reverse=True)
    return [evidence_to_dict(e) for e in hits[:limit]]


def visual_path(query: str, *, limit: int = 12) -> list[dict[str, Any]]:
    fleet = get_fleet_context()
    if not fleet:
        return []
    intent = {**_query_intent(query), "wants_figure": True}
    hits = _figure_search(fleet["manual_id"], query, intent, limit=limit)
    for ev in hits:
        _apply_ranking_boosts(ev, intent)
    hits.sort(key=lambda e: e.score, reverse=True)
    return [evidence_to_dict(e) for e in hits[:limit]]


def structural_path_from_picks(
    query: str,
    *,
    doc_codes: list[str] | None = None,
    titles: list[str] | None = None,
    procedure_nos: list[str] | None = None,
    limit: int = 16,
) -> list[dict[str, Any]]:
    """Fetch typed elements for LLM-chosen sections (still rows, not token windows)."""
    fleet = get_fleet_context()
    if not fleet:
        return []
    intent = _query_intent(query)
    codes = [c.upper() for c in (doc_codes or []) if c]
    titles = [t for t in (titles or []) if t]
    procedures = [p.strip() for p in (procedure_nos or []) if p]
    if not codes and not titles and not procedures:
        hits = _structural_search(fleet["manual_id"], intent, limit=limit)
        for ev in hits:
            _apply_ranking_boosts(ev, intent)
        return [evidence_to_dict(e) for e in hits[:limit]]

    kind_order = _kind_order_sql(intent)
    with connect() as conn:
        with conn.cursor() as cur:
            rows: list[dict] = []
            if codes or procedures:
                cur.execute(
                    f"""
                    SELECT {ELEMENT_COLUMNS}
                    FROM sections s
                    JOIN elements e ON e.section_id = s.id
                    WHERE s.manual_id = %s
                      AND (
                        s.doc_code = ANY(%s)
                        OR e.procedure_no = ANY(%s)
                        OR s.procedure_no = ANY(%s)
                      )
                    ORDER BY {kind_order},
                      length(coalesce(e.text, '')) DESC,
                      e.page
                    LIMIT %s
                    """,
                    (fleet["manual_id"], codes, procedures, procedures, limit * 3),
                )
                rows.extend(cur.fetchall())
            for title in titles[:8]:
                cur.execute(
                    f"""
                    SELECT {ELEMENT_COLUMNS}
                    FROM sections s
                    JOIN elements e ON e.section_id = s.id
                    WHERE s.manual_id = %s
                      AND (s.title ILIKE %s OR s.component_title ILIKE %s)
                    ORDER BY {kind_order},
                      length(coalesce(e.text, '')) DESC,
                      e.page
                    LIMIT %s
                    """,
                    (fleet["manual_id"], f"%{title}%", f"%{title}%", limit * 2),
                )
                rows.extend(cur.fetchall())

    seen: set[UUID] = set()
    out: list[Evidence] = []
    for r in rows:
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        ev = row_to_evidence(r, score=3.0, source="structural")
        _apply_ranking_boosts(ev, intent)
        out.append(ev)
        if len(out) >= limit:
            break
    return [evidence_to_dict(e) for e in out]


def rrf_fuse(
    ranked_lists: list[list[dict[str, Any]]],
    *,
    k: int = 60,
    limit: int = 16,
) -> list[dict[str, Any]]:
    """Reciprocal rank fusion over element_id (architecture §6.4)."""
    scores: dict[str, float] = {}
    by_id: dict[str, dict[str, Any]] = {}
    for lst in ranked_lists:
        for rank, item in enumerate(lst, start=1):
            eid = str(item.get("element_id") or "")
            if not eid:
                continue
            scores[eid] = scores.get(eid, 0.0) + 1.0 / (k + rank)
            prev = by_id.get(eid)
            if prev is None:
                by_id[eid] = dict(item)
                continue
            if float(item.get("score") or 0) > float(prev.get("score") or 0):
                merged = dict(item)
                if not merged.get("figure_image_path"):
                    merged["figure_image_path"] = prev.get("figure_image_path")
                if not merged.get("drawing_code"):
                    merged["drawing_code"] = prev.get("drawing_code")
                by_id[eid] = merged
            elif item.get("figure_image_path") and not prev.get("figure_image_path"):
                prev["figure_image_path"] = item.get("figure_image_path")
                prev["figure_label"] = item.get("figure_label") or prev.get("figure_label")
                prev["drawing_code"] = prev.get("drawing_code") or item.get("drawing_code")
    fused = []
    for eid, rrf in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
        item = by_id[eid]
        item["score"] = rrf
        item["source"] = item.get("source") or "rrf"
        fused.append(item)
        if len(fused) >= limit:
            break
    return fused
