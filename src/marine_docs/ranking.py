"""Retrieval ranking priority, applied before the final rerank (spec Section 8).

A Tool, not an agent: the rules below are deterministic and run identically on
every query, so the evidence set an answer was built from is reproducible.

  1. Exact citation_key match to something named in the query -> near-deterministic
     top rank. (procedure_no / plate_no remain aliases of citation_key.)
  2. If the top-ranked text element has a linked_step_number, pull its matching
     panel (same page) into evidence regardless of that panel's own rank.
  3. Auto-resolve one hop on references_section / references_data (and the older
     references_procedure alias) before answer synthesis.
  4. Only then does the rerank score break remaining ties. Pinned hits stay pinned.
"""

from __future__ import annotations

import logging
import re
from typing import Any
from uuid import UUID

from marine_docs.db import connect
from marine_docs.retrieval import (
    ELEMENT_COLUMNS,
    FIGURE_COLUMNS,
    evidence_to_dict,
    overlap_score,
    row_to_evidence,
)

logger = logging.getLogger(__name__)

PROCEDURE_NO_RE = re.compile(r"\b(\d{3}-\d+(?:\.\d+)?)\b")
PLATE_NO_RE = re.compile(r"\b(P\d{4,6})\b", re.IGNORECASE)
DOC_CODE_RE = re.compile(r"\b([A-Z]{1,3}\d{4,8})\b", re.IGNORECASE)
DRAWING_CODE_RE = re.compile(
    r"\b([A-Z]{1,3}\d{3,6}-\d{1,5}(?:\.\d+)?(?:[A-Z]\d{0,4})?)\b",
    re.IGNORECASE,
)
VISUAL_QUERY_RE = re.compile(
    r"plate|diagram|drawing|callout|sketch|show the|show me|illustration|\bpanel\b",
    re.I,
)
NEGATED_COMPONENT_RE = re.compile(
    r"(?:do not use|don't use|not the|not from|ignore|avoid|even if)\s+"
    r"(?:the\s+)?([a-z0-9][a-z0-9\s\-]{2,80}?)(?:\s+data|\s+sheet|\s+even|\s+if|\s+as the|\s*[.?;,):]|$)",
    re.I,
)
_NOT_COMPONENT_RE = re.compile(
    r"(?:^|[,\s;(])not\s+(?:the\s+)?([a-z0-9]+(?:-[a-z0-9]+)+)",
    re.I,
)
_IS_THAT_A_RE = re.compile(
    r"\bis that a ([a-z0-9]+(?:-[a-z0-9]+)+)\b",
    re.I,
)
_COMPARE_QUERY_RE = re.compile(
    r"\b(?:compare|versus|both sources|same rule|if they differ|quote both)\b",
    re.I,
)
_PRIMARY_SPLIT_RE = re.compile(
    r",\s+and is\b|\beven if\b|\bdo not\b|\bdon't\b|\bis that a\b|\? quote\b",
    re.I,
)
_WEAK_COMPONENT = frozenset(
    {
        "bearing", "engine", "diesel", "manual", "complete", "after", "before",
        "remaining", "length", "value", "point", "tool", "tools", "above", "below",
        "sheet", "adjustment", "spindle", "plate", "panel", "callout", "quote", "number",
        "show", "which", "pipe", "pressure", "flange", "torque", "row", "correct",
        "data", "work", "procedure", "checking", "mounting", "dismantling",
        "adjusting", "walk", "through",
    }
)
STEP_LEAD_RE = re.compile(r"^\s*(\d{1,2})\.\s")
CHECKING_ACTION_RE = re.compile(
    r"check|inspect|evaluat|criteri|accept|limit|wear|decision",
    re.IGNORECASE,
)

EXACT_MATCH_BOOST = 40.0
KIND_PAIR_BOOST = 10.0
COMPONENT_PHRASE_BOOST = 18.0
PANEL_PULL_BOOST = 12.0
CROSS_REF_BOOST = 6.0
TABLE_REF_BOOST = 8.0
SECTION_MATE_BOOST = 7.0
MAX_PANEL_PULLS = 4
MAX_KIND_PAIRS = 8
MAX_CROSS_REF_HOPS = 8
MAX_TABLE_REF_HOPS = 4
MAX_SECTION_MATES = 8
TABLE_CODE_RE = re.compile(r"\b([A-Z]\d{2}-\d{2})\b")
_STEM_SUFFIX_RE = re.compile(
    r"[\s\-:]*(?:data|checking|dismantling|mounting|overhaul|plate|inspection)$",
    re.I,
)


def _procedure_stem(value: str) -> str | None:
    match = re.match(r"(\d{3}-\d+)", (value or "").strip())
    return match.group(1) if match else None


def _component_stem(title: str) -> str:
    t = (title or "").strip().lower()
    return _STEM_SUFFIX_RE.sub("", t).strip(" -:")


def _norm_code(value: str) -> str:
    return re.sub(r"[\s_]+", "", (value or "").upper())


def _drawing_named(stored: str, codes: list[str]) -> bool:
    s = _norm_code(stored)
    if not s:
        return False
    for c in codes:
        cc = _norm_code(c)
        if len(cc) >= 5 and cc in s:
            return True
    return False


def _hit_blob(hit: dict[str, Any]) -> str:
    return " ".join(
        [
            hit.get("component_title") or "",
            hit.get("action_title") or "",
            " ".join(hit.get("section_path") or []),
            hit.get("doc_code") or "",
            hit.get("plate_no") or "",
            hit.get("drawing_code") or "",
            (hit.get("text") or "")[:500],
        ]
    ).lower()


def _norm_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()


def _query_tokens(query: str) -> list[str]:
    return [t for t in _norm_text(query).split() if t]


def _title_overlap(hit: dict[str, Any], tokens: list[str]) -> int:
    blob = _norm_text(
        " ".join(
            [
                hit.get("component_title") or "",
                hit.get("action_title") or "",
                " ".join(hit.get("section_path") or []),
                hit.get("doc_code") or "",
                hit.get("plate_no") or "",
                hit.get("drawing_code") or "",
                hit.get("citation_key") or "",
                hit.get("citation_ref") or "",
            ]
        )
    )
    score = 0
    seen: set[str] = set()
    for t in tokens:
        for part in _norm_text(t).split() or [t.lower()]:
            if part in seen or len(part) < 4 or part in _WEAK_COMPONENT:
                continue
            if part in blob:
                seen.add(part)
                score += 1
    return score


def _negative_tokens(query: str) -> set[str]:
    """Tokens the question is ruling out (sibling families, contrast codes)."""
    out: set[str] = set()

    def _absorb(chunk: str) -> None:
        out.update(
            t
            for t in re.findall(r"[a-z]{4,}", chunk.lower())
            if t not in _WEAK_COMPONENT
        )
        out.update(p.lower() for p in PROCEDURE_NO_RE.findall(chunk))
        out.update(p.lower() for p in PLATE_NO_RE.findall(chunk))
        out.update(p.lower() for p in TABLE_CODE_RE.findall(chunk))

    for match in NEGATED_COMPONENT_RE.finditer(query or ""):
        _absorb(match.group(1))
    for match in _NOT_COMPONENT_RE.finditer(query or ""):
        _absorb(match.group(1))
    for match in _IS_THAT_A_RE.finditer(query or ""):
        _absorb(match.group(1))
    for match in TABLE_CODE_RE.finditer(query or ""):
        span_start = max(0, match.start() - 24)
        window = (query or "")[span_start : match.end() + 56].lower()
        if any(k in window for k in ("even if", "correct", "looks familiar", "do not")):
            out.add(match.group(1).lower())
            gloss = re.search(r"\(([^)]{3,80})\)", (query or "")[match.end() : match.end() + 80])
            if gloss:
                _absorb(gloss.group(1))
    return out


def _primary_query_tokens(query: str) -> list[str]:
    primary = _PRIMARY_SPLIT_RE.split(query or "", maxsplit=1)[0]
    neg = _negative_tokens(query)
    return [t for t in _query_tokens(primary) if t not in neg and t.lower() not in neg]


def _positive_query_text(query: str) -> str:
    """Strip contrast/negation clauses so named identifiers there are not pinned."""
    q = query or ""
    q = re.sub(
        r"(?:do not use|don't use|not the|even if|ignore|avoid)\b.{0,90}",
        " ",
        q,
        flags=re.I,
    )
    q = re.sub(r"(?:,\s+not\b|\bnot\s+[a-z0-9]+-[a-z0-9]+).{0,80}", " ", q, flags=re.I)
    q = re.sub(r"\bis that a [a-z0-9\-]+.{0,50}", " ", q, flags=re.I)
    q = re.sub(r"\band is it\b.{0,70}", " ", q, flags=re.I)
    q = re.sub(r"\bis data [A-Z]?\d[\w.-]*.{0,60}?correct.{0,20}", " ", q, flags=re.I)
    return q


def _positive_query_tokens(query: str) -> list[str]:
    neg = _negative_tokens(query)
    return [t for t in _query_tokens(query) if t not in neg and t.lower() not in neg]


def _hit_matches_negative(hit: dict[str, Any], neg: set[str]) -> bool:
    if not neg:
        return False
    raw = " ".join(
        [
            hit.get("component_title") or "",
            hit.get("action_title") or "",
            hit.get("procedure_no") or "",
            hit.get("doc_code") or "",
            hit.get("citation_key") or "",
            hit.get("plate_no") or "",
            hit.get("drawing_code") or "",
        ]
    ).lower()
    blob_toks = set(_norm_text(raw).split())
    title_toks = set(
        _norm_text(
            " ".join(
                [
                    hit.get("component_title") or "",
                    hit.get("action_title") or "",
                ]
            )
        ).split()
    )
    alpha_neg = [
        tok
        for tok in neg
        if len(tok) >= 4 and tok.isalpha() and tok not in _WEAK_COMPONENT
    ]
    matched = [tok for tok in alpha_neg if tok in title_toks or tok in blob_toks]
    if len(alpha_neg) == 1 and matched:
        return True
    if len(matched) >= 2:
        return True
    compact = re.sub(r"[^a-z0-9]+", "", raw)
    for tok in neg:
        if len(tok) < 5 or not any(ch.isdigit() for ch in tok):
            continue
        ct = re.sub(r"[^a-z0-9]+", "", tok)
        if len(ct) >= 5 and ct in compact:
            return True
    return False


def preferred_evidence_hits(
    hits: list[dict[str, Any]],
    query: str,
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Citation window: keep the components the question actually names."""
    if not hits:
        return []
    qtoks = _primary_query_tokens(query)
    neg = _negative_tokens(query)
    refs = query_references(query)
    named_stems = {
        stem
        for stem in (_procedure_stem(p) for p in (refs.get("procedure_nos") or []))
        if stem
    }
    named_chaps = {s[:3] for s in named_stems}
    compare = bool(_COMPARE_QUERY_RE.search(query or ""))
    pos = _positive_query_text(query)
    prefer_data = bool(re.search(r"\bdata\b", pos, re.I)) and not bool(
        re.search(r"\b(procedure|step|mounting|dismantling|checking)\b", pos, re.I)
    )
    scored = [(_title_overlap(h, qtoks), h) for h in hits]
    best = max((s for s, _ in scored), default=0)
    stem_score: dict[str, int] = {}
    for s, h in scored:
        if _hit_matches_negative(h, neg):
            continue
        stem = _component_stem(h.get("component_title") or "") or (h.get("doc_code") or "")
        if stem:
            stem_score[stem] = max(stem_score.get(stem, 0), s)
    keep_floor = 1 if compare and best >= 1 else best
    top_stems = {
        stem
        for stem, sc in stem_score.items()
        if sc >= keep_floor or (compare and sc >= max(1, best - 1))
    }
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for s, h in sorted(
        scored,
        key=lambda x: (-x[0], 0 if x[1].get("pinned") else 1, -float(x[1].get("score") or 0)),
    ):
        if _hit_matches_negative(h, neg):
            continue
        kind = (h.get("section_kind") or "").lower()
        if prefer_data and kind == "procedure":
            continue
        proc = _procedure_stem(h.get("procedure_no") or h.get("citation_key") or "")
        if (
            named_stems
            and proc
            and proc[:3] in named_chaps
            and proc not in named_stems
            and h.get("priority_rule") != "exact_identifier"
        ):
            continue
        stem = _component_stem(h.get("component_title") or "") or (h.get("doc_code") or "")
        if top_stems and stem and stem not in top_stems:
            continue
        key = (
            (h.get("citation_key") or h.get("procedure_no") or h.get("doc_code") or "")
            + "|"
            + (h.get("section_kind") or "")
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(h)
        if len(out) >= limit:
            break
    return out or hits[:limit]


def query_references(
    query: str,
    convention: dict[str, Any] | None = None,
) -> dict[str, list[str]]:
    """Identifiers the user named explicitly, including this manual's citation_key pattern."""
    source = _positive_query_text(query)
    drawings = [m.group(1).upper() for m in DRAWING_CODE_RE.finditer(source)]
    citation_keys: list[str] = []
    conv = convention if isinstance(convention, dict) else {}
    pat = conv.get("pattern") if conv else None
    too_broad = (not pat) or pat in (r"[A-Za-z0-9][\w.-]*", r"\w+", r".*", r"[\\w.-]*")
    if pat and not too_broad:
        try:
            citation_keys = [m.group(0) for m in re.finditer(pat, source)]
        except re.error:
            citation_keys = []
    procedure_nos = [m.group(1) for m in PROCEDURE_NO_RE.finditer(source)]
    # KN905-1.1 / GN909-6.4 encode the procedure number after the drawing prefix.
    for drawing in drawings:
        implied = re.search(r"(\d{3}-\d+(?:\.\d+)?)", drawing)
        if implied:
            procedure_nos.append(implied.group(1))
    return {
        "procedure_nos": list(dict.fromkeys(procedure_nos)),
        "plate_nos": [m.group(1).upper() for m in PLATE_NO_RE.finditer(source)],
        "doc_codes": [m.group(1).upper() for m in DOC_CODE_RE.finditer(source)],
        "drawing_codes": list(dict.fromkeys(drawings)),
        "citation_keys": list(dict.fromkeys(citation_keys)),
        "table_codes": [m.group(1).upper() for m in TABLE_CODE_RE.finditer(source)],
    }


def apply_priority(
    query: str,
    hits: list[dict[str, Any]],
    *,
    manual_id: UUID | None = None,
    convention: dict[str, Any] | None = None,
    limit: int = 24,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run ranking rules and return (hits, notes). Pinned hits carry `pinned=True`."""
    notes: dict[str, Any] = {
        "exact_matches": 0,
        "kind_pairs": 0,
        "component_pins": 0,
        "panels_pulled": 0,
        "cross_refs_resolved": 0,
        "table_ref_hops": 0,
        "section_mates": 0,
    }
    if not hits:
        return hits, notes

    from marine_docs.retrieval import _query_intent

    if convention is None:
        try:
            from marine_docs.retrieval import get_fleet_context

            convention = (get_fleet_context() or {}).get("citation_convention")
        except Exception:
            convention = None
    if isinstance(convention, str):
        import json

        try:
            convention = json.loads(convention)
        except Exception:
            convention = None

    refs = query_references(query, convention)
    intent = _query_intent(query)
    working = [dict(h) for h in hits]

    # Rule 1 — an explicitly named procedure or plate resolves near-deterministically.
    for hit in working:
        if _matches_named_reference(hit, refs):
            hit["score"] = float(hit.get("score") or 0) + EXACT_MATCH_BOOST
            hit["pinned"] = True
            hit["priority_rule"] = "exact_identifier"
            notes["exact_matches"] += 1

    if (
        refs["procedure_nos"]
        or refs["plate_nos"]
        or refs["doc_codes"]
        or refs["drawing_codes"]
        or refs.get("citation_keys")
    ):
        working.extend(
            _fetch_named_elements(
                refs,
                manual_id,
                exclude={h["element_id"] for h in working},
                query=query,
            )
        )
        _merge_figure_fields(
            working,
            _fetch_named_drawings(
                refs,
                manual_id,
                exclude=set(),
            ),
        )
        notes["exact_matches"] = sum(1 for h in working if h.get("pinned"))

    # Rule 1b — query component phrases beat a similar numeric rule in another chapter.
    notes["component_pins"] = _apply_component_priority(working, intent)

    plate_figs = _fetch_pinned_plate_figures(
        working, exclude=set(), query=query
    )
    if plate_figs:
        _merge_figure_fields(working, plate_figs)

    # Rule 2 — pair data sheets with the checking procedure for the same component.
    pair_sources = [h for h in working if h.get("pinned")]
    if len(pair_sources) < 8:
        seen_ids = {h["element_id"] for h in pair_sources}
        pair_sources.extend(h for h in working if h["element_id"] not in seen_ids)
        pair_sources = pair_sources[:8]
    pairs = fetch_kind_pairs(
        pair_sources or working,
        manual_id=manual_id,
        exclude={h["element_id"] for h in working},
    )
    if pairs:
        working.extend(pairs)
        notes["kind_pairs"] = len(pairs)
    _pin_existing_pairs(working)

    mates = _fetch_section_mates(
        working, exclude={h["element_id"] for h in working}, query=query
    )
    if mates:
        working.extend(mates)
        notes["section_mates"] = len(mates)

    lists = _fetch_list_siblings(working, exclude={h["element_id"] for h in working})
    if lists:
        working.extend(lists)
        notes["section_mates"] = int(notes.get("section_mates") or 0) + len(lists)

    table_hops = _fetch_table_code_siblings(
        working, exclude={h["element_id"] for h in working}, query=query
    )
    if table_hops:
        working.extend(table_hops)
        notes["table_ref_hops"] = len(table_hops)

    # Rule 3 — pair step text with the panel that illustrates it.
    panels = _fetch_step_panels(working, exclude={h["element_id"] for h in working})
    if panels:
        working.extend(panels)
        notes["panels_pulled"] = len(panels)

    # Rule 4 — one hop along cross-references found in the evidence.
    hops = _fetch_cross_refs(working, exclude={h["element_id"] for h in working})
    if hops:
        working.extend(hops)
        notes["cross_refs_resolved"] = len(hops)

    dist = [t for t in (intent.get("component_tokens") or []) if len(t) >= 5]
    dist.extend(
        t
        for t in (intent.get("tokens") or [])
        if t[:1].isdigit() and ("." in t or len(t) >= 2)
    )
    table_codes = list(refs.get("table_codes") or []) or [
        m.group(1).upper() for m in TABLE_CODE_RE.finditer(query)
    ]
    nums = [t for t in (intent.get("tokens") or []) if t[:1].isdigit()]
    q_l = (query or "").lower()
    named_stems = {
        stem
        for stem in (_procedure_stem(p) for p in (refs.get("procedure_nos") or []))
        if stem
    }
    named_chapters = {s[:3] for s in named_stems}
    negated = _negative_tokens(query)
    pos_toks = _positive_query_tokens(query)
    phrases = [p for p in (intent.get("phrases") or []) if len(p.split()) >= 2][:8]
    for hit in working:
        hit["score"] = float(hit.get("score") or 0) + overlap_score(
            hit.get("text") or "",
            intent.get("tokens") or [],
            distinctive=dist,
            element_type=hit.get("element_type"),
        )
        text = hit.get("text") or ""
        text_l = text.lower()
        et = (hit.get("element_type") or "").lower()
        title_pts = _title_overlap(hit, pos_toks)
        negated_hit = _hit_matches_negative(hit, negated)
        if et == "table" and (
            any(n in text for n in nums) or any(c in text.upper() for c in table_codes)
        ) and not negated_hit:
            hit["pinned"] = True
            hit.setdefault("priority_rule", "numeric_table")
        elif (
            "hydraulic" in q_l
            and "hydraulic pressure" in text_l
            and title_pts >= 1
            and not negated_hit
        ):
            hit["score"] = float(hit.get("score") or 0) + 14.0
            hit["pinned"] = True
            hit.setdefault("priority_rule", "numeric_table")
        elif et in {"paragraph", "list"} and any(n in text for n in nums if len(str(n)) >= 2):
            hit["score"] = float(hit.get("score") or 0) + 8.0
            if len(text) < 280:
                hit["pinned"] = True
                hit.setdefault("priority_rule", "numeric_span")
        if "shut" in q_l and "shut off" in text_l:
            hit["pinned"] = True
            hit.setdefault("priority_rule", "checklist")
        if any(p in text_l for p in phrases) and len(text) < 900:
            hit["score"] = float(hit.get("score") or 0) + 10.0
            if any(k in text_l for k in ("must not", "do not", "shall not", "not be used")):
                hit["pinned"] = True
                hit.setdefault("priority_rule", "phrase_rule")
        if negated_hit:
            hit["score"] = float(hit.get("score") or 0) - 25.0
            if hit.get("priority_rule") != "exact_identifier":
                hit["pinned"] = False
        if named_stems:
            hit_stem = _procedure_stem(
                hit.get("procedure_no") or hit.get("citation_key") or ""
            )
            hit_chap = (hit_stem or "")[:3]
            if (
                hit_stem
                and hit_chap in named_chapters
                and hit_stem not in named_stems
                and hit.get("priority_rule") != "exact_identifier"
            ):
                hit["score"] = float(hit.get("score") or 0) - 22.0
                hit["pinned"] = False

    best_overlap = max((_title_overlap(h, pos_toks) for h in working), default=0)
    if best_overlap >= 2:
        for hit in working:
            ov = _title_overlap(hit, pos_toks)
            if ov == 0 and hit.get("priority_rule") != "exact_identifier":
                hit["score"] = float(hit.get("score") or 0) - 18.0
                hit["pinned"] = False

    working.sort(key=lambda h: (0 if h.get("pinned") else 1, -float(h.get("score") or 0)))
    working = _diversify_hits(working, limit=limit, query=query)
    return working, notes


def _merge_figure_fields(working: list[dict[str, Any]], extras: list[dict[str, Any]]) -> None:
    """Keep image paths when the same element was retrieved as text-only elsewhere."""
    by_id = {str(h.get("element_id")): h for h in working}
    for item in extras:
        eid = str(item.get("element_id") or "")
        if not eid:
            continue
        if eid in by_id:
            dest = by_id[eid]
            if item.get("figure_image_path") and not dest.get("figure_image_path"):
                dest["figure_image_path"] = item["figure_image_path"]
            if item.get("drawing_code") and not dest.get("drawing_code"):
                dest["drawing_code"] = item["drawing_code"]
            if item.get("figure_label") and not dest.get("figure_label"):
                dest["figure_label"] = item["figure_label"]
            continue
        working.append(item)
        by_id[eid] = item


def finalize_order(
    ranked: list[dict[str, Any]],
    *,
    limit: int = 24,
    query: str = "",
) -> list[dict[str, Any]]:
    """Pinned hits stay ahead of rerank, then kinds are mixed so one code cannot fill the window."""
    pinned = [h for h in ranked if h.get("pinned")]
    rest = [h for h in ranked if not h.get("pinned")]
    return _diversify_hits(pinned + rest, limit=limit, query=query)


def _diversify_hits(
    hits: list[dict[str, Any]],
    *,
    limit: int,
    query: str = "",
) -> list[dict[str, Any]]:
    """Keep complementary kinds in the window so one procedure code cannot starve data."""
    if not hits:
        return []
    q = (query or "").lower()
    wants_safety = any(
        k in q
        for k in ("shut off", "shut-off", "lubricat", "precaution", "safety", "mass", "lifting")
    )
    wants_schedule = any(k in q for k in ("hours", "overhaul", "interval", "schedule"))
    wants_visual = bool(VISUAL_QUERY_RE.search(query or ""))

    def sort_key(h: dict[str, Any]) -> tuple:
        text = (h.get("text") or "").lower()
        kind = (h.get("section_kind") or "").lower()
        bonus = 0
        if wants_safety and any(
            k in text for k in ("shut off", "precaution", "safety check", "lubricating oil")
        ):
            bonus -= 2
        if wants_schedule and kind == "schedule":
            bonus -= 2
        return (0 if h.get("pinned") else 1, bonus, -float(h.get("score") or 0))

    ordered = sorted(hits, key=sort_key)
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    per_code: dict[str, int] = {}
    seen_stem_kind: set[tuple[str, str]] = set()

    def _code(h: dict[str, Any]) -> str:
        return (h.get("doc_code") or h.get("procedure_no") or "").upper()

    def _take(h: dict[str, Any]) -> None:
        eid = h.get("element_id")
        if not eid or eid in seen:
            return
        selected.append(h)
        seen.add(eid)
        code = _code(h)
        if code:
            per_code[code] = per_code.get(code, 0) + 1

    qtoks = _query_tokens(query)
    if wants_visual:
        fig_hits = [
            h
            for h in ordered
            if h.get("figure_image_path")
            or (h.get("section_kind") or "").lower() == "plate"
            or h.get("drawing_code")
        ]
        fig_hits.sort(
            key=lambda h: (
                0 if h.get("figure_image_path") else 1,
                0 if h.get("pinned") else 1,
                0 if (h.get("priority_rule") in {"exact_identifier", "plate_figure"}) else 1,
                -_title_overlap(h, qtoks),
                -float(h.get("score") or 0),
            )
        )
        images_taken = 0
        for h in fig_hits:
            if not h.get("figure_image_path"):
                continue
            if images_taken >= 6:
                break
            before = len(selected)
            _take(h)
            if len(selected) > before:
                images_taken += 1
        for h in fig_hits:
            if h.get("figure_image_path"):
                continue
            if len(selected) >= 8:
                break
            _take(h)

    pinned_codes = {
        (h.get("plate_no") or h.get("doc_code") or "").upper()
        for h in selected
        if (h.get("plate_no") or (h.get("section_kind") or "").lower() == "plate")
    }
    have_img = {
        (h.get("plate_no") or h.get("doc_code") or "").upper()
        for h in selected
        if h.get("figure_image_path")
    }
    if pinned_codes:
        for h in ordered:
            code = (h.get("plate_no") or h.get("doc_code") or "").upper()
            if (
                h.get("figure_image_path")
                and code in pinned_codes
                and code not in have_img
            ):
                _take(h)
                have_img.add(code)

    # Pass 1: one data + one procedure (+ schedule) per top component stem.
    top_stems: list[str] = []
    for h in ordered:
        stem = _component_stem(h.get("component_title") or "")
        if stem and stem not in top_stems:
            top_stems.append(stem)
        if len(top_stems) >= 4:
            break
    top_stem_set = set(top_stems)
    for h in ordered:
        if len(selected) >= limit:
            break
        stem = _component_stem(h.get("component_title") or "")
        kind = (h.get("section_kind") or "").lower()
        if stem not in top_stem_set or kind not in {"data", "procedure", "schedule"}:
            continue
        key = (stem, kind)
        if key in seen_stem_kind:
            continue
        seen_stem_kind.add(key)
        _take(h)

    # Pass 2: remaining by score, cap 3 snippets per document code.
    for h in ordered:
        if len(selected) >= limit:
            break
        eid = h.get("element_id")
        if not eid or eid in seen:
            continue
        code = _code(h)
        if code and per_code.get(code, 0) >= 3:
            extra_list = (h.get("element_type") or "").lower() == "list"
            if not extra_list or per_code.get(code, 0) >= 7:
                continue
        _take(h)

    # Pass 3: fill leftover without the per-code cap.
    for h in ordered:
        if len(selected) >= limit:
            break
        _take(h)
    return selected[:limit]


def fetch_kind_pairs(
    hits: list[dict[str, Any]],
    *,
    manual_id: UUID | None = None,
    exclude: set[str] | None = None,
    limit: int = MAX_KIND_PAIRS,
) -> list[dict[str, Any]]:
    """Pull the complementary data sheet or checking procedure for the same component.

    Pairing uses component_title and `describes` edges already stored at ingest —
    not OEM code prefixes and not the wording of any particular question.
    """
    exclude = set(exclude or ())
    titles: list[str] = []
    need_kinds: set[str] = set()
    source_ids: list[UUID] = []
    for hit in hits[:12]:
        title = (hit.get("component_title") or "").strip()
        kind = (hit.get("section_kind") or "").lower()
        if title:
            titles.append(title)
        if kind == "data":
            need_kinds.add("procedure")
        elif kind == "procedure":
            need_kinds.add("data")
        else:
            need_kinds.update({"data", "procedure"})
        try:
            source_ids.append(UUID(str(hit["element_id"])))
        except Exception:
            continue
    if not titles and not source_ids:
        return []
    if not need_kinds:
        need_kinds = {"data", "procedure"}

    out: list[dict[str, Any]] = []
    seen = set(exclude)
    if titles:
        out.extend(
            _query_kind_pairs_by_title(
                titles, sorted(need_kinds), manual_id, seen, limit=limit
            )
        )
        seen.update(item["element_id"] for item in out)
    if source_ids and len(out) < limit:
        out.extend(
            _query_kind_pairs_by_describes(
                source_ids, seen, limit=limit - len(out)
            )
        )
    return out[:limit]


def _pin_existing_pairs(hits: list[dict[str, Any]]) -> None:
    """Pin data+procedure hits that already share a component stem."""
    by_stem: dict[str, set[str]] = {}
    for hit in hits:
        stem = _component_stem(hit.get("component_title") or "")
        kind = (hit.get("section_kind") or "").lower()
        if stem and kind in {"data", "procedure"}:
            by_stem.setdefault(stem, set()).add(kind)
    paired = {t for t, kinds in by_stem.items() if "data" in kinds and "procedure" in kinds}
    if not paired:
        return
    pinned_per_stem: dict[str, set[str]] = {}
    for hit in hits:
        stem = _component_stem(hit.get("component_title") or "")
        kind = (hit.get("section_kind") or "").lower()
        if stem not in paired or kind not in {"data", "procedure"}:
            continue
        used = pinned_per_stem.setdefault(stem, set())
        if kind in used and not hit.get("pinned"):
            continue
        used.add(kind)
        hit["pinned"] = True
        hit.setdefault("priority_rule", "kind_pair")


def _query_kind_pairs_by_title(
    titles: list[str],
    kinds: list[str],
    manual_id: UUID | None,
    exclude: set[str],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    lowered = list(dict.fromkeys(t.strip().lower() for t in titles if t and t.strip()))
    stems = list(dict.fromkeys(_component_stem(t) for t in lowered if _component_stem(t)))
    if not lowered:
        return []
    try:
        with connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT {ELEMENT_COLUMNS}
                FROM elements e
                LEFT JOIN sections s ON s.id = e.section_id
                WHERE (%s::uuid IS NULL OR s.manual_id = %s)
                  AND (
                    lower(trim(s.component_title)) = ANY(%s)
                    OR regexp_replace(
                         lower(trim(coalesce(s.component_title, ''))),
                         '[\s\-:]*(data|checking|dismantling|mounting|overhaul|plate|inspection)$',
                         ''
                       ) = ANY(%s)
                  )
                  AND s.section_kind = ANY(%s)
                  AND e.type IN ('paragraph', 'table', 'heading', 'list')
                ORDER BY
                  CASE WHEN e.text ~* 'hydraulic pressure|clearance [ABC]|tightening torque|D[0-9]{2}-[0-9]{2}' THEN 0 ELSE 1 END,
                  CASE WHEN e.type = 'table' THEN 0 ELSE 1 END,
                  CASE WHEN s.action_title ~* %s THEN 0 ELSE 1 END,
                  length(coalesce(e.text, '')) DESC,
                  e.page
                LIMIT %s
                """,
                (
                    manual_id,
                    manual_id,
                    lowered,
                    stems or lowered,
                    kinds,
                    CHECKING_ACTION_RE.pattern,
                    max(limit, 8),
                ),
            )
            rows = cur.fetchall()
    except Exception:
        logger.exception("Kind-pair lookup by component_title failed")
        return []
    return _rows_to_pair_hits(rows, exclude, "kind_pair")


def _query_kind_pairs_by_describes(
    source_ids: list[UUID],
    exclude: set[str],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    try:
        with connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT {ELEMENT_COLUMNS}
                FROM elements e
                LEFT JOIN sections s ON s.id = e.section_id
                WHERE e.section_id IN (
                    SELECT DISTINCT other_id FROM (
                        SELECT b.section_id AS other_id
                        FROM element_relationships r
                        JOIN elements a ON a.id = r.from_element_id
                        JOIN elements b ON b.id = r.to_element_id
                        WHERE r.relationship_type = 'describes'
                          AND a.section_id IN (
                              SELECT section_id FROM elements WHERE id = ANY(%s)
                          )
                        UNION
                        SELECT a.section_id AS other_id
                        FROM element_relationships r
                        JOIN elements a ON a.id = r.from_element_id
                        JOIN elements b ON b.id = r.to_element_id
                        WHERE r.relationship_type = 'describes'
                          AND b.section_id IN (
                              SELECT section_id FROM elements WHERE id = ANY(%s)
                          )
                    ) x
                )
                  AND e.type IN ('paragraph', 'table', 'heading', 'list')
                ORDER BY
                  CASE WHEN s.action_title ~* %s THEN 0 ELSE 1 END,
                  CASE WHEN e.type = 'table' THEN 0 ELSE 1 END,
                  length(coalesce(e.text, '')) DESC
                LIMIT %s
                """,
                (source_ids, source_ids, CHECKING_ACTION_RE.pattern, max(limit, 8)),
            )
            rows = cur.fetchall()
    except Exception:
        logger.exception("Kind-pair lookup via describes failed")
        return []
    return _rows_to_pair_hits(rows, exclude, "describes")


def _rows_to_pair_hits(
    rows: list[dict[str, Any]],
    exclude: set[str],
    rule: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        item = evidence_to_dict(row_to_evidence(row, score=KIND_PAIR_BOOST, source="structural"))
        if item["element_id"] in exclude:
            continue
        item["pinned"] = True
        item["priority_rule"] = rule
        out.append(item)
        exclude.add(item["element_id"])
    return out


def _apply_component_priority(hits: list[dict[str, Any]], intent: dict[str, Any]) -> int:
    """Pin titles that match the query's component phrases; demote other chapters."""
    phrases = [p for p in (intent.get("phrases") or []) if len(p.split()) >= 2]
    distinctive = [
        t
        for t in (intent.get("tokens") or [])
        if len(t) >= 5 and t not in _WEAK_COMPONENT
    ]
    qtoks = {t.lower() for t in (intent.get("tokens") or [])}
    if not phrases and not distinctive:
        return 0
    pinned = 0
    for hit in hits:
        title = (hit.get("component_title") or "").lower()
        blob = " ".join(
            [
                title,
                " ".join(hit.get("section_path") or []),
                hit.get("action_title") or "",
            ]
        ).lower()
        phrase_hit = False
        for p in phrases[:12]:
            if p not in title and p not in blob:
                continue
            extra = [t for t in distinctive if t not in p.split()]
            if extra and not any(t in blob for t in extra):
                continue
            phrase_hit = True
            break
        token_hits = sum(1 for t in distinctive if t in blob)
        title_hits = sum(1 for t in distinctive if t in title)
        extra_title = [
            w
            for w in re.findall(r"[a-z]{5,}", title)
            if w not in qtoks and w not in _WEAK_COMPONENT
        ]
        unmatched = [t for t in distinctive if t not in blob]
        if extra_title and title_hits < 3 and not hit.get("pinned"):
            hit["score"] = float(hit.get("score") or 0) - 14.0
        strong = phrase_hit or (
            title_hits >= 2 and (not unmatched or title_hits >= 3)
        )
        if strong:
            hit["score"] = float(hit.get("score") or 0) + COMPONENT_PHRASE_BOOST
            hit["pinned"] = True
            hit.setdefault("priority_rule", "component_phrase")
            pinned += 1
        elif title_hits == 1:
            hit["score"] = float(hit.get("score") or 0) + 8.0
        elif token_hits:
            hit["score"] = float(hit.get("score") or 0) + 6.0 * token_hits
        elif title and distinctive and not hit.get("pinned"):
            if not any(t in blob for t in distinctive):
                loose = any(
                    t in title
                    for t in (intent.get("tokens") or [])
                    if len(t) >= 4 and t not in _WEAK_COMPONENT
                )
                if not loose:
                    hit["score"] = float(hit.get("score") or 0) - 12.0
    return pinned


def _fetch_section_mates(
    hits: list[dict[str, Any]],
    *,
    exclude: set[str],
    query: str = "",
) -> list[dict[str, Any]]:
    """Pull the rest of a data/procedure sheet once one row from it scored."""
    q = (query or "").lower()
    kinds = {"data", "procedure", "schedule"}
    if VISUAL_QUERY_RE.search(query or "") or PLATE_NO_RE.search(query or ""):
        kinds.add("plate")
    source_ids: list[UUID] = []
    ranked_src = sorted(
        hits, key=lambda h: (0 if h.get("pinned") else 1, -float(h.get("score") or 0))
    )
    for hit in ranked_src[:12]:
        if (hit.get("section_kind") or "").lower() not in kinds:
            continue
        try:
            source_ids.append(UUID(str(hit["element_id"])))
        except Exception:
            continue
    if not source_ids:
        return []
    prefer_checklist = any(
        k in q
        for k in ("shut off", "shut-off", "lubricat", "precaution", "safety", "mass", "lifting")
    )
    prefer_visual = "plate" in kinds
    if prefer_visual:
        order_sql = """
                ORDER BY
                  CASE WHEN e.text ~* 'callout|item no|item description' THEN 0 ELSE 1 END,
                  CASE WHEN e.type IN ('paragraph', 'table', 'list') THEN 0 ELSE 1 END,
                  length(coalesce(e.text, '')) DESC
        """
    elif prefer_checklist:
        order_sql = """
                ORDER BY
                  CASE WHEN e.text ~* 'shut off|precaution|safety check|lubricating oil|mass|kg' THEN 0 ELSE 1 END,
                  CASE WHEN e.type IN ('paragraph', 'list') THEN 0 ELSE 1 END,
                  length(coalesce(e.text, '')) DESC
        """
    else:
        order_sql = """
                ORDER BY
                  CASE WHEN e.type = 'table' THEN 0 ELSE 1 END,
                  length(coalesce(e.text, '')) DESC
        """
    try:
        with connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT {ELEMENT_COLUMNS}
                FROM elements e
                LEFT JOIN sections s ON s.id = e.section_id
                WHERE e.section_id IN (
                    SELECT e2.section_id FROM elements e2 WHERE e2.id = ANY(%s)
                )
                  AND e.type IN ('paragraph', 'table', 'heading', 'list')
                {order_sql}
                LIMIT %s
                """,
                (source_ids, 32),
            )
            rows = cur.fetchall()
    except Exception:
        logger.exception("Same-section mate lookup failed")
        return []
    qtoks = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", (query or "").lower())
    dist = [t for t in qtoks if len(t) >= 5]
    out: list[dict[str, Any]] = []
    for row in rows:
        item = evidence_to_dict(
            row_to_evidence(row, score=SECTION_MATE_BOOST, source="structural")
        )
        if item["element_id"] in exclude:
            continue
        item["priority_rule"] = "section_mate"
        item["score"] = SECTION_MATE_BOOST + overlap_score(
            item.get("text") or "",
            qtoks,
            distinctive=dist,
            element_type=item.get("element_type"),
        )
        out.append(item)
        exclude.add(item["element_id"])
    out.sort(key=lambda h: -float(h.get("score") or 0))
    picked = out[:6]
    seen_pick = {h["element_id"] for h in picked}
    for h in out:
        if len(picked) >= MAX_SECTION_MATES:
            break
        if h["element_id"] in seen_pick:
            continue
        if h.get("element_type") == "table":
            picked.append(h)
            seen_pick.add(h["element_id"])
    return picked[:MAX_SECTION_MATES]


def _fetch_list_siblings(
    hits: list[dict[str, Any]],
    *,
    exclude: set[str],
) -> list[dict[str, Any]]:
    """Criterion lists (time bands, OK/observe/overhaul) are split into short rows."""
    procs: list[str] = []
    for hit in hits[:12]:
        kind = (hit.get("element_type") or "").lower()
        text = (hit.get("text") or "").lower()
        if kind in {"list", "heading"} or any(
            k in text for k in ("minutes", "hour", "must not exceed", "keep under")
        ):
            proc = (hit.get("procedure_no") or "").strip()
            if proc:
                procs.append(proc)
    procs = list(dict.fromkeys(procs))[:6]
    if not procs:
        return []
    try:
        with connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT {ELEMENT_COLUMNS}
                FROM elements e
                LEFT JOIN sections s ON s.id = e.section_id
                WHERE e.procedure_no = ANY(%s)
                  AND e.type IN ('list', 'heading')
                ORDER BY length(coalesce(e.text, '')) ASC
                LIMIT 16
                """,
                (procs,),
            )
            rows = cur.fetchall()
    except Exception:
        logger.exception("List-sibling lookup failed")
        return []
    out: list[dict[str, Any]] = []
    for row in rows:
        item = evidence_to_dict(
            row_to_evidence(row, score=SECTION_MATE_BOOST + 2.0, source="structural")
        )
        if item["element_id"] in exclude:
            continue
        item["priority_rule"] = "list_sibling"
        text_l = (item.get("text") or "").lower()
        if any(k in text_l for k in ("minute", "hour", "must not exceed", "keep under")):
            item["pinned"] = True
            item["score"] = float(item.get("score") or 0) + 20.0
        out.append(item)
        exclude.add(item["element_id"])
        if len(out) >= 8:
            break
    return out


def _fetch_table_code_siblings(
    hits: list[dict[str, Any]],
    *,
    exclude: set[str],
    query: str = "",
) -> list[dict[str, Any]]:
    """Same table-row code can be restated on another data sheet — pull those rows."""
    codes: list[str] = [m.group(1).upper() for m in TABLE_CODE_RE.finditer(query or "")]
    named_from_query = set(codes)
    for hit in hits[:8]:
        codes.extend(TABLE_CODE_RE.findall(hit.get("text") or ""))
    codes = list(dict.fromkeys(c.upper() for c in codes))[:8]
    if not codes:
        return []
    patterns = [f"%{c}%" for c in codes]
    try:
        with connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT {ELEMENT_COLUMNS}
                FROM elements e
                LEFT JOIN sections s ON s.id = e.section_id
                WHERE e.type IN ('paragraph', 'table', 'heading')
                  AND (
                    e.text ILIKE ANY(%s)
                    OR e.table_json::text ILIKE ANY(%s)
                  )
                ORDER BY length(coalesce(e.text, '')) DESC
                LIMIT %s
                """,
                (patterns, patterns, MAX_TABLE_REF_HOPS + 10),
            )
            rows = cur.fetchall()
    except Exception:
        logger.exception("Table-code sibling lookup failed")
        return []
    qtoks = _positive_query_tokens(query)
    neg = _negative_tokens(query)
    named_from_query = {c for c in named_from_query if c.lower() not in neg}
    compare = bool(_COMPARE_QUERY_RE.search(query or ""))
    source_stems = {
        _component_stem(h.get("component_title") or "")
        for h in hits[:8]
        if (h.get("component_title") or "").strip()
    }
    scored: list[tuple[int, dict[str, Any]]] = []
    for row in rows:
        item = evidence_to_dict(
            row_to_evidence(row, score=TABLE_REF_BOOST, source="structural")
        )
        if item["element_id"] in exclude:
            continue
        if _hit_matches_negative(item, neg):
            continue
        stem = _component_stem(item.get("component_title") or "")
        if source_stems and stem and stem not in source_stems and not named_from_query:
            continue
        title_pts = _title_overlap(item, qtoks)
        item["score"] = TABLE_REF_BOOST + 3.0 * title_pts
        item["priority_rule"] = "table_code"
        if named_from_query and any(c in (item.get("text") or "").upper() for c in named_from_query):
            item["pinned"] = True
        scored.append((title_pts, item))
        exclude.add(item["element_id"])
    best = max((s for s, _ in scored), default=0)
    min_keep = 0
    if best >= 2:
        min_keep = 1 if compare else best
    elif named_from_query:
        min_keep = 1
    out = [item for s, item in scored if s >= min_keep]
    out.sort(key=lambda h: -float(h.get("score") or 0))
    return out[: MAX_TABLE_REF_HOPS + (2 if named_from_query else 0)]


def _matches_named_reference(hit: dict[str, Any], refs: dict[str, list[str]]) -> bool:
    procedure_no = (hit.get("procedure_no") or "").strip()
    plate_no = (hit.get("plate_no") or "").upper()
    doc_code = (hit.get("doc_code") or "").upper()
    drawing_code = (hit.get("drawing_code") or "").upper()
    citation_key = (hit.get("citation_key") or "").strip()
    named_drawings = refs.get("drawing_codes") or []
    named_keys = [k.upper() for k in (refs.get("citation_keys") or [])]
    return (
        (procedure_no and procedure_no in refs["procedure_nos"])
        or (plate_no and plate_no in refs["plate_nos"])
        or (doc_code and doc_code in refs["doc_codes"])
        or (citation_key and citation_key.upper() in named_keys)
        or _drawing_named(drawing_code, named_drawings)
        or _drawing_named(doc_code, named_drawings)
        or _drawing_named(plate_no, refs.get("plate_nos") or [])
    )


def _fetch_named_elements(
    refs: dict[str, list[str]],
    manual_id: UUID | None,
    *,
    exclude: set[str],
    query: str = "",
) -> list[dict[str, Any]]:
    """Pull the named procedure/plate directly, even if no path retrieved it."""
    procedure_nos = list(dict.fromkeys(refs["procedure_nos"]))
    plate_nos = list(dict.fromkeys(refs["plate_nos"]))
    doc_codes = list(dict.fromkeys(refs["doc_codes"]))
    drawing_codes = list(dict.fromkeys(refs.get("drawing_codes") or []))
    citation_keys = [str(k) for k in dict.fromkeys(refs.get("citation_keys") or [])]
    if not (procedure_nos or plate_nos or doc_codes or drawing_codes or citation_keys):
        return []
    drawing_pats = [f"%{c}%" for c in drawing_codes] or ["__none__"]

    try:
        with connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT {ELEMENT_COLUMNS}
                FROM elements e
                LEFT JOIN sections s ON s.id = e.section_id
                WHERE (%s::uuid IS NULL OR s.manual_id = %s)
                  AND (
                    e.procedure_no = ANY(%s)
                    OR upper(e.plate_no) = ANY(%s)
                    OR upper(s.doc_code) = ANY(%s)
                    OR s.procedure_no = ANY(%s)
                    OR coalesce(e.drawing_code, '') ILIKE ANY(%s)
                    OR upper(coalesce(e.citation_key, '')) = ANY(%s)
                    OR upper(coalesce(s.citation_key, '')) = ANY(%s)
                  )
                ORDER BY
                  CASE WHEN e.text ~* 'callout|item no' THEN 0 ELSE 1 END,
                  CASE WHEN e.type IN ('paragraph', 'table', 'heading', 'list') THEN 0 ELSE 1 END,
                  length(coalesce(e.text, '')) DESC,
                  e.page
                LIMIT 16
                """,
                (
                    manual_id,
                    manual_id,
                    procedure_nos,
                    plate_nos,
                    doc_codes,
                    procedure_nos,
                    drawing_pats,
                    [k.upper() for k in citation_keys] or ["__NONE__"],
                    [k.upper() for k in citation_keys] or ["__NONE__"],
                ),
            )
            rows = cur.fetchall()
    except Exception:
        logger.exception("Exact-identifier lookup failed")
        return []

    qtoks = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", (query or "").lower())
    dist = [t for t in qtoks if len(t) >= 5]
    out: list[dict[str, Any]] = []
    for row in rows:
        item = evidence_to_dict(row_to_evidence(row, score=0.0, source="structural"))
        if item["element_id"] in exclude:
            continue
        item["score"] = EXACT_MATCH_BOOST + overlap_score(
            item.get("text") or "",
            qtoks,
            distinctive=dist,
            element_type=item.get("element_type"),
        )
        item["pinned"] = True
        item["priority_rule"] = "exact_identifier"
        out.append(item)
        exclude.add(item["element_id"])
    out.sort(key=lambda h: -float(h.get("score") or 0))
    return out[:10]


def _fetch_named_drawings(
    refs: dict[str, list[str]],
    manual_id: UUID | None,
    *,
    exclude: set[str],
    rule: str = "exact_identifier",
) -> list[dict[str, Any]]:
    """Pull the figure row for a drawing/plate the user named, including OCR text."""
    needles = list(
        dict.fromkeys(
            (refs.get("drawing_codes") or [])
            + (refs.get("plate_nos") or [])
            + (refs.get("citation_keys") or [])
        )
    )
    if not needles:
        return []
    pats = [f"%{n}%" for n in needles]
    try:
        with connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT {FIGURE_COLUMNS}
                FROM figures f
                JOIN sections s ON s.id = f.section_id
                LEFT JOIN elements e ON e.id = f.element_id
                WHERE (%s::uuid IS NULL OR f.manual_id = %s)
                  AND (
                    coalesce(f.drawing_code, '') ILIKE ANY(%s)
                    OR coalesce(e.drawing_code, '') ILIKE ANY(%s)
                    OR s.doc_code ILIKE ANY(%s)
                    OR coalesce(s.plate_no, '') ILIKE ANY(%s)
                    OR coalesce(s.citation_key, '') ILIKE ANY(%s)
                    OR coalesce(e.citation_key, '') ILIKE ANY(%s)
                    OR coalesce(f.caption, '') ILIKE ANY(%s)
                  )
                ORDER BY
                  CASE WHEN s.section_kind = 'plate' THEN 0 ELSE 1 END,
                  f.page, f.panel_index
                LIMIT 8
                """,
                (manual_id, manual_id, pats, pats, pats, pats, pats, pats, pats),
            )
            rows = cur.fetchall()
    except Exception:
        logger.exception("Named-drawing lookup failed")
        return []
    from marine_docs.retrieval import _backfill_figure_ocr

    _backfill_figure_ocr(rows)
    out: list[dict[str, Any]] = []
    for row in rows:
        if row.get("id") is None:
            continue
        item = evidence_to_dict(
            row_to_evidence(row, score=EXACT_MATCH_BOOST + 8.0, source="figure")
        )
        if item["element_id"] in exclude:
            continue
        item["pinned"] = True
        item["priority_rule"] = rule
        out.append(item)
        exclude.add(item["element_id"])
    return out


def _fetch_pinned_plate_figures(
    hits: list[dict[str, Any]],
    *,
    exclude: set[str],
    query: str = "",
) -> list[dict[str, Any]]:
    """Attach figure rows for the plate that best matches the query title tokens."""
    qtoks = _query_tokens(query)
    named_plates = [m.group(1).upper() for m in PLATE_NO_RE.finditer(query or "")]
    best_code = ""
    best_score = -1
    for hit in hits:
        kind = (hit.get("section_kind") or "").lower()
        code = (
            hit.get("plate_no")
            or hit.get("citation_key")
            or hit.get("doc_code")
            or ""
        ).upper()
        if kind != "plate" and not hit.get("plate_no") and not str(code).startswith("P"):
            continue
        if named_plates and code in named_plates:
            score = 100 + _title_overlap(hit, qtoks)
        else:
            score = _title_overlap(hit, qtoks)
            if hit.get("pinned"):
                score += 2
        if score > best_score:
            best_score = score
            best_code = code
    if not best_code and named_plates:
        best_code = named_plates[0]
    if not best_code:
        return []
    refs = {
        "procedure_nos": [],
        "plate_nos": [best_code],
        "doc_codes": [best_code],
        "drawing_codes": [],
        "citation_keys": [best_code],
    }
    return _fetch_named_drawings(refs, None, exclude=exclude, rule="plate_figure")


def _fetch_step_panels(
    hits: list[dict[str, Any]],
    *,
    exclude: set[str],
) -> list[dict[str, Any]]:
    """For top text hits with a step number, add the panel on the same page."""
    wanted: list[tuple[int, int]] = []
    text_ids: list[UUID] = []
    for hit in hits[:6]:
        if hit.get("figure_image_path"):
            continue
        page = hit.get("page")
        step = hit.get("linked_step_number")
        if not step:
            match = STEP_LEAD_RE.match(hit.get("text") or "")
            if match:
                step = int(match.group(1))
        if step and page:
            wanted.append((int(page), int(step)))
        try:
            text_ids.append(UUID(str(hit["element_id"])))
        except Exception:
            continue

    out: list[dict[str, Any]] = []
    seen = set(exclude)
    if wanted:
        out.extend(_query_panels_by_step(wanted, seen))
        seen.update(item["element_id"] for item in out)
    if text_ids and len(out) < MAX_PANEL_PULLS:
        out.extend(_query_panels_by_relationship(text_ids, seen, limit=MAX_PANEL_PULLS - len(out)))
    return out[:MAX_PANEL_PULLS]


def _query_panels_by_step(
    wanted: list[tuple[int, int]],
    exclude: set[str],
) -> list[dict[str, Any]]:
    try:
        with connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT {ELEMENT_COLUMNS}, f.image_path, f.figure_label
                FROM elements e
                LEFT JOIN sections s ON s.id = e.section_id
                LEFT JOIN figures f ON f.element_id = e.id
                WHERE e.type IN ('figure', 'plate')
                  AND (e.page, e.linked_step_number) IN (
                      SELECT * FROM unnest(%s::int[], %s::int[])
                  )
                ORDER BY e.page, e.panel_index
                LIMIT %s
                """,
                (
                    [p for p, _ in wanted],
                    [s for _, s in wanted],
                    MAX_PANEL_PULLS,
                ),
            )
            rows = cur.fetchall()
    except Exception:
        logger.exception("Panel pull failed")
        return []

    return _rows_to_panel_hits(rows, exclude, "step_panel")


def _query_panels_by_relationship(
    text_ids: list[UUID],
    exclude: set[str],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    try:
        with connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT {ELEMENT_COLUMNS}, f.image_path, f.figure_label
                FROM element_relationships r
                JOIN elements e ON e.id = r.from_element_id
                LEFT JOIN sections s ON s.id = e.section_id
                LEFT JOIN figures f ON f.element_id = e.id
                WHERE r.to_element_id = ANY(%s)
                  AND r.relationship_type = 'illustrates_step'
                  AND e.type IN ('figure', 'plate')
                ORDER BY e.page, e.panel_index
                LIMIT %s
                """,
                (text_ids, limit),
            )
            rows = cur.fetchall()
    except Exception:
        logger.exception("Panel pull via illustrates_step failed")
        return []

    return _rows_to_panel_hits(rows, exclude, "illustrates_step")


def _rows_to_panel_hits(
    rows: list[dict[str, Any]],
    exclude: set[str],
    rule: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        item = evidence_to_dict(row_to_evidence(row, score=PANEL_PULL_BOOST, source="figure"))
        if item["element_id"] in exclude:
            continue
        item["priority_rule"] = rule
        out.append(item)
        exclude.add(item["element_id"])
    return out


def _fetch_cross_refs(
    hits: list[dict[str, Any]],
    *,
    exclude: set[str],
) -> list[dict[str, Any]]:
    """Resolve one hop of references_procedure / references_data."""
    source_ids: list[UUID] = []
    pinned_ids: list[UUID] = []
    for hit in hits:
        try:
            uid = UUID(str(hit["element_id"]))
        except Exception:
            continue
        if hit.get("pinned"):
            pinned_ids.append(uid)
        elif len(source_ids) < 8:
            source_ids.append(uid)
    source_ids = pinned_ids + [i for i in source_ids if i not in pinned_ids]
    if not source_ids:
        return []

    try:
        with connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT {ELEMENT_COLUMNS}, r.relationship_type
                FROM element_relationships r
                JOIN elements e ON e.id = r.to_element_id
                LEFT JOIN sections s ON s.id = e.section_id
                WHERE r.from_element_id = ANY(%s)
                  AND r.relationship_type IN (
                    'references_procedure', 'references_section', 'references_data'
                  )
                ORDER BY length(coalesce(e.text, '')) DESC
                LIMIT %s
                """,
                (source_ids, MAX_CROSS_REF_HOPS),
            )
            rows = cur.fetchall()
    except Exception:
        logger.exception("Cross-reference hop failed")
        return []

    out: list[dict[str, Any]] = []
    for row in rows:
        item = evidence_to_dict(row_to_evidence(row, score=CROSS_REF_BOOST, source="structural"))
        if item["element_id"] in exclude:
            continue
        item["priority_rule"] = row["relationship_type"]
        out.append(item)
    return out
