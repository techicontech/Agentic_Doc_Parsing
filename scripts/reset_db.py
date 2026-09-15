#!/usr/bin/env python
"""Drop and recreate knowledge tables (keeps Docker volumes). Use before full re-ingest."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from marine_docs.corpus import clear_corpus


def main() -> int:
    confirm = "--yes" in sys.argv
    if not confirm:
        print("This deletes all ingested manuals/elements. Re-run with --yes to confirm.")
        return 1
    result = clear_corpus()
    print("Schema reset complete.", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
