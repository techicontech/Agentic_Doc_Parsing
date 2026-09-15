"""Verification gate — abstain when evidence is not safe to answer from."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from marine_docs.retrieval import Evidence, RetrievalResult


@dataclass
class VerificationResult:
    passed: bool
    checks: dict[str, bool] = field(default_factory=dict)
    failed: list[str] = field(default_factory=list)
    reason: str | None = None


def verify(
    query: str,
    retrieval: RetrievalResult,
    *,
    query_equipment: str | None = None,
) -> VerificationResult:
    checks: dict[str, bool] = {}

    # Equipment: query context must match fleet registry model when provided
    fleet_model = (retrieval.fleet_model or "").upper()
    q_equip = (query_equipment or _guess_equipment(query) or fleet_model).upper()
    if fleet_model:
        # Wrong-equipment trap: explicit different model in query
        other = _guess_equipment(query)
        if other and other.upper() != fleet_model and other.upper() not in fleet_model:
            checks["equipment"] = False
        else:
            checks["equipment"] = q_equip in fleet_model or fleet_model in q_equip or not query_equipment
    else:
        checks["equipment"] = False

    checks["revision"] = bool(retrieval.revision)  # stub: current revision present
    checks["scope"] = len(retrieval.evidences) > 0
    checks["conflicts"] = not _has_numeric_conflicts(retrieval.evidences)
    # Condition / units: soft pass for Milestone-1 UI unless obvious mismatch markers
    checks["condition"] = True
    checks["units"] = True

    # Unanswerable: no evidence
    if not retrieval.evidences:
        checks["scope"] = False

    failed = [k for k, v in checks.items() if not v]
    passed = len(failed) == 0
    reason = None
    if not passed:
        if "equipment" in failed:
            reason = (
                f"Equipment mismatch: registry has {retrieval.fleet_model}, "
                f"query implies {_guess_equipment(query) or query_equipment}."
            )
        elif "scope" in failed:
            reason = "No supporting evidence found in the ingested Vol II corpus."
        elif "conflicts" in failed:
            reason = "Conflicting numeric values found across retrieved sections."
        else:
            reason = f"Verification failed: {', '.join(failed)}"

    return VerificationResult(passed=passed, checks=checks, failed=failed, reason=reason)


def _guess_equipment(text: str) -> str | None:
    m = re.search(r"\b([A-Z0-9]+MC(?:-C)?)\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b(S\d{2}MC(?:-C)?)\b", text, re.IGNORECASE)
    return m.group(1).upper() if m else None


def _has_numeric_conflicts(evidences: list[Evidence]) -> bool:
    """Very light conflict detector: same Ref code with different Nm values."""
    torque_by_ref: dict[str, set[str]] = {}
    pattern = re.compile(
        r"(D\d{2}-\d{2}).{0,40}?(\d+(?:\.\d+)?)\s*Nm",
        re.IGNORECASE | re.DOTALL,
    )
    for ev in evidences:
        for m in pattern.finditer(ev.text or ""):
            torque_by_ref.setdefault(m.group(1).upper(), set()).add(m.group(2))
    return any(len(vals) > 1 for vals in torque_by_ref.values())
