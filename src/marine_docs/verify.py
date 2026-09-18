"""Verification gate — six checks from architecture §6.6; abstain if any fail."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from marine_docs.editions import check_named_mismatch, check_revision
from marine_docs.retrieval import Evidence, RetrievalResult, _query_intent, get_fleet_context


@dataclass
class VerificationResult:
    passed: bool
    checks: dict[str, bool] = field(default_factory=dict)
    failed: list[str] = field(default_factory=list)
    reason: str | None = None
    details: dict[str, list[str]] = field(default_factory=dict)


_CONDITION_RE = re.compile(
    r"\b(\d+\s*%\s*MCR|MCR|cold start|overload|idle|"
    r"manoeuvr(?:ing|e)|maneuver(?:ing)?|during start|warm-?up)\b",
    re.IGNORECASE,
)
_UNIT_RE = re.compile(r"\b(mm|bar|nm|kg|°|deg|mpa|kpa|rpm)\b", re.IGNORECASE)
_NUMBER_UNIT_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(mm|bar|nm|kg|°|deg|mpa|kpa|rpm)\b",
    re.IGNORECASE,
)


def verify(
    query: str,
    retrieval: RetrievalResult,
    *,
    query_equipment: str | None = None,
) -> VerificationResult:
    checks: dict[str, bool] = {}
    blob = " ".join((ev.text or "") for ev in retrieval.evidences).lower()
    path_blob = " ".join(
        " ".join(ev.section_path or []) for ev in retrieval.evidences
    ).lower()
    combined = f"{blob} {path_blob}"

    fleet_model = (retrieval.fleet_model or "").upper()
    ctx = (query_equipment or "").strip()
    if fleet_model:
        named = _guess_equipment(query)
        if named and not _models_compatible(named, fleet_model):
            checks["equipment"] = False
        elif not ctx:
            checks["equipment"] = True
        else:
            checks["equipment"] = _models_compatible(ctx, fleet_model)
    else:
        checks["equipment"] = False

    # Revision check uses this manual's stored citation_convention granularity
    # (per_section vs per_document), not a hardcoded manufacturer rule.
    details: dict[str, list[str]] = {}
    named_ok, named_mismatch = check_named_mismatch(query, retrieval.evidences)
    gran = "per_section"
    try:
        fleet = get_fleet_context() or {}
        conv = fleet.get("citation_convention") or {}
        if isinstance(conv, str):
            import json

            conv = json.loads(conv)
        if isinstance(conv, dict):
            gran = str(conv.get("revision_granularity") or "per_section")
    except Exception:
        gran = "per_section"
    if gran in ("none", "unknown"):
        checks["revision"] = True
    else:
        checks["revision"] = named_ok
    if named_mismatch:
        details["revision"] = named_mismatch
    _, superseded = check_revision(retrieval.evidences)
    if superseded:
        details["superseded"] = superseded

    cond = _CONDITION_RE.search(query or "")
    if cond:
        token = re.sub(r"\s+", " ", cond.group(1).lower())
        checks["condition"] = token in combined or token.replace(" ", "") in combined.replace(" ", "")
    else:
        checks["condition"] = True

    unit_m = _UNIT_RE.search(query or "")
    if unit_m and retrieval.evidences:
        unit = unit_m.group(1).lower()
        # A measured value already stated in the question does not have to be
        # echoed by evidence — the unit is attached to the user-supplied number.
        if _NUMBER_UNIT_RE.search(query or ""):
            checks["units"] = True
        else:
            aliases = {"deg": "°", "nm": "nm"}
            needle = aliases.get(unit, unit)
            checks["units"] = needle in combined or unit in combined
    else:
        checks["units"] = True

    if not retrieval.evidences:
        checks["scope"] = False
    else:
        intent = _query_intent(query)
        tokens = [t for t in (intent.get("component_tokens") or intent.get("tokens") or []) if len(t) >= 4]
        phrases = intent.get("phrases") or []
        if phrases:
            checks["scope"] = any(p in combined for p in phrases[:6]) or any(
                t in combined for t in tokens[:6]
            )
        elif tokens:
            checks["scope"] = any(t in combined for t in tokens[:6])
        else:
            checks["scope"] = True

    conflicted = _numeric_conflicts(retrieval.evidences)
    checks["conflicts"] = True
    if conflicted:
        details["conflicts"] = conflicted

    failed = [k for k, v in checks.items() if not v]
    passed = len(failed) == 0
    reason = None
    if not passed:
        if "equipment" in failed:
            reason = (
                f"Equipment mismatch: registry has {retrieval.fleet_model}, "
                f"query implies {_guess_equipment(query) or query_equipment}."
            )
        elif "revision" in failed:
            if details.get("revision"):
                reason = (
                    "Query named a different edition than the retrieved procedure: "
                    + "; ".join(details["revision"][:3])
                )
            else:
                reason = "Retrieved procedure edition does not match the edition named in the query."
        elif "condition" in failed:
            reason = "Operating-condition stated in the query is not covered by retrieved evidence."
        elif "units" in failed:
            reason = "Requested units are not present in the retrieved evidence."
        elif "scope" in failed:
            reason = "Retrieved evidence does not cover the component or topic asked."
        elif "conflicts" in failed:
            reason = "Conflicting numeric values found across retrieved sections."
        else:
            reason = f"Verification failed: {', '.join(failed)}"

    return VerificationResult(
        passed=passed, checks=checks, failed=failed, reason=reason, details=details
    )


def _guess_equipment(text: str) -> str | None:
    """Model-like tokens only (S50MC-C, 6S60ME-C, 3516B) — not units like 45mm."""
    skip_letters = {"MM", "NM", "KG", "BAR", "KPA", "MPA", "RPM", "DEG"}
    patterns = (
        r"\b([A-Z]{1,3}\d{2,5}[A-Z]{1,8}(?:-[A-Z0-9]+)?)\b",
        r"\b(\d{1,2}[A-Z]\d{1,4}[A-Z]{1,8}(?:-[A-Z0-9]+)?)\b",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text or "", re.IGNORECASE):
            token = match.group(1).upper()
            letters = re.sub(r"[^A-Z]", "", token)
            if len(letters) < 2 or letters in skip_letters:
                continue
            return token
    return None


def _models_compatible(named: str, fleet: str) -> bool:
    n = re.sub(r"[^A-Z0-9]", "", (named or "").upper())
    f = re.sub(r"[^A-Z0-9]", "", (fleet or "").upper())
    if not n or not f:
        return True
    if n in f or f in n:
        return True
    for i in range(len(n) - 4):
        if n[i : i + 5] in f:
            return True
    return False


def _numeric_conflicts(evidences: list[Evidence]) -> list[str]:
    """Same table/ref code with different numeric values of the same unit."""
    by_ref: dict[str, set[str]] = {}
    pattern = re.compile(
        r"\b([A-Z]?\d{2,3}-\d{2})\b.{0,80}?(\d+(?:\.\d+)?)\s*(Nm|mm|bar|kg)",
        re.IGNORECASE | re.DOTALL,
    )
    for ev in evidences:
        for m in pattern.finditer(ev.text or ""):
            key = f"{m.group(1).upper()}:{m.group(3).lower()}"
            by_ref.setdefault(key, set()).add(m.group(2))
    return [
        f"{key} has {', '.join(sorted(vals))}"
        for key, vals in by_ref.items()
        if len(vals) > 1
    ]
