#!/usr/bin/env python
"""Quick CLI ask against the knowledge model (no UI)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from marine_docs.chat import answer_query


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("query", nargs="+")
    p.add_argument("--equipment", default="S50MC-C")
    args = p.parse_args()
    q = " ".join(args.query)
    resp = answer_query(q, equipment_context=args.equipment)
    print(resp.answer)
    print("\n--- citations ---")
    print(json.dumps(resp.citations, indent=2, default=str))
    print("\n--- verification ---")
    print(json.dumps(resp.verification, indent=2))
    return 0 if not resp.abstained else 2


if __name__ == "__main__":
    raise SystemExit(main())
