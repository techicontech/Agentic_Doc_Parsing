"""Verification gate is a plain tool; the abstain path must be explicit."""

from __future__ import annotations

import unittest
from uuid import uuid4

from marine_docs.retrieval import Evidence, RetrievalResult
from marine_docs.verify import verify


def _ev(**kwargs) -> Evidence:
    defaults = dict(
        element_id=uuid4(),
        page=10,
        text="Piston clearance 0.45 mm at 100% MCR. Procedure 902-1.3 Edition 0286.",
        section_path=["Cylinder Cover", "Piston"],
        doc_code="M90201",
        edition="0286",
        section_kind="procedure",
        score=0.0,
        source="lexical",
        procedure_no="902-1.3",
        citation_key="902-1.3",
        component_title="Piston",
        action_title="Checking",
    )
    defaults.update(kwargs)
    return Evidence(**defaults)


class VerificationGateTests(unittest.TestCase):
    def test_abstain_when_no_evidence(self) -> None:
        retrieval = RetrievalResult(evidences=[], fleet_model="S50MC-C")
        result = verify("what is the piston clearance in mm?", retrieval, query_equipment="S50MC-C")
        self.assertFalse(result.passed)
        self.assertIn("scope", result.failed)

    def test_abstain_on_equipment_mismatch(self) -> None:
        retrieval = RetrievalResult(
            evidences=[_ev()],
            fleet_model="S50MC-C",
        )
        result = verify(
            "On a Wartsila 6L32, what is the piston clearance in mm?",
            retrieval,
            query_equipment="6L32",
        )
        self.assertFalse(result.passed)
        self.assertIn("equipment", result.failed)

    def test_pass_when_evidence_covers_query(self) -> None:
        retrieval = RetrievalResult(
            evidences=[_ev()],
            fleet_model="S50MC-C",
        )
        result = verify(
            "Piston clearance in mm at 100% MCR?",
            retrieval,
            query_equipment="S50MC-C",
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.failed, [])


if __name__ == "__main__":
    unittest.main()
