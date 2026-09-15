#!/usr/bin/env python
"""Run FastAPI chat backend on http://127.0.0.1:8000"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import uvicorn


def main() -> None:
    # Logging (quiet console + full file) is configured when marine_docs.api loads.
    uvicorn.run(
        "marine_docs.api:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="warning",
        access_log=True,
    )


if __name__ == "__main__":
    main()
