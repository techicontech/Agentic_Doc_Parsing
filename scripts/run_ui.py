#!/usr/bin/env python
"""Launch the basic Gradio chatbot UI."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from marine_docs.ui_app import main

if __name__ == "__main__":
    main()
