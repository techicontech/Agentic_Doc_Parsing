"""Convention detector must not overfit to the MAN B&W test-case layout."""

from __future__ import annotations

import unittest

from marine_docs.convention import (
    FALLBACK,
    extract_citation,
    heuristic_detect,
    see_reference_regex,
)


def _samples(*lines: str) -> list[dict]:
    return [{"page": i + 1, "header": line, "footer": line, "drawings": 0} for i, line in enumerate(lines)]


class ConventionDetectorTests(unittest.TestCase):
    def test_man_like_quote_procedure_edition(self) -> None:
        samples = _samples(
            "Please quote Procedure 902-1.3 Edition 0286 when referring to this page",
            "Please quote Procedure 901-1.1 Edition 0249 when referring to this page",
            "Please quote Data 903-1 Edition 0312 when referring to this page",
        )
        conv = heuristic_detect(samples)
        self.assertNotEqual(conv.get("source"), "fallback")
        self.assertIn("citation_key", conv.get("citation_field", "citation_key"))
        extracted = extract_citation(samples[0]["footer"], conv, printed_page="12")
        self.assertEqual(extracted["citation_key"], "902-1.3")
        self.assertTrue(extracted.get("revision"))

    def test_second_manufacturer_section_rev(self) -> None:
        """Synthetic Wartsila/Sulzer-style: Section 4.2.1 Rev C — not NNN-N.N."""
        samples = _samples(
            "Section 4.2.1 Rev C   Cooling water pump",
            "Section 4.2.2 Rev C   Impeller clearance",
            "Section 7.1.0 Rev D   Fuel injection",
            "Section 7.1.1 Rev D   Nozzle inspection",
        )
        conv = heuristic_detect(samples)
        self.assertEqual(conv.get("source"), "heuristic")
        self.assertEqual(conv.get("revision_field"), "revision")
        self.assertIn("section", conv.get("label_words") or [])
        extracted = extract_citation(samples[0]["header"], conv, printed_page="88")
        self.assertEqual(extracted["citation_key"], "4.2.1")
        self.assertEqual((extracted.get("revision") or "").upper(), "C")
        xref = see_reference_regex(conv)
        match = xref.search("refer to Section 7.1.0 for the nozzle")
        self.assertIsNotNone(match)

    def test_unstructured_falls_back_to_printed_page(self) -> None:
        samples = _samples(
            "Introduction",
            "This chapter describes general safety.",
            "Page 14",
        )
        conv = heuristic_detect(samples)
        self.assertEqual(conv.get("source"), "fallback")
        extracted = extract_citation("body text with no codes", conv, printed_page="14")
        self.assertEqual(extracted["citation_key"], "p.14")
        self.assertEqual(FALLBACK["citation_field"], "page_number_printed")

    def test_extraction_reads_convention_data_not_manufacturer_name(self) -> None:
        conv = {
            "citation_field": "citation_key",
            "pattern": r"DWG-\d+",
            "revision_field": None,
            "revision_granularity": "per_document",
            "label_words": ["drawing"],
            "source": "heuristic",
        }
        extracted = extract_citation(
            "See Drawing DWG-4471 in Chapter 7",
            conv,
            printed_page="3",
        )
        self.assertEqual(extracted["citation_key"], "DWG-4471")


if __name__ == "__main__":
    unittest.main()
