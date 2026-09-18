"""Quiet console + full logs to a gitignored text file."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

from marine_docs.config import ROOT

LOG_DIR = ROOT / "logs"
_NOISY = (
    "docling",
    "httpx",
    "httpcore",
    "urllib3",
    "openai",
    "litellm",
    # LiteLLM's own loggers are capitalised and attach their own handlers.
    "LiteLLM",
    "LiteLLM Proxy",
    "LiteLLM Router",
    "google_adk",
    "transformers",
    "torch",
    "PIL",
    "filelock",
    "fsspec",
)


class _StatusPollFilter(logging.Filter):
    """Hide noisy UI status polls from the terminal access log."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "/api/ingest/status" in msg:
            return False
        return True


def setup_logging(*, console_level: int = logging.WARNING) -> Path:
    """Route verbose logs to logs/ingest_*.txt; keep terminal mostly clean.

    Returns path to the active log file.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"ingest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(console_level)
    console.setFormatter(logging.Formatter("%(message)s"))
    console.addFilter(_StatusPollFilter())
    root.addHandler(console)

    # Dedicated progress stream → always visible on console + file
    progress = logging.getLogger("marine_docs.progress")
    progress.setLevel(logging.INFO)
    progress.propagate = False
    progress.handlers.clear()
    ph_console = logging.StreamHandler(sys.stdout)
    ph_console.setLevel(logging.INFO)
    ph_console.setFormatter(logging.Formatter("%(message)s"))
    progress.addHandler(ph_console)
    ph_file = logging.FileHandler(log_path, encoding="utf-8")
    ph_file.setLevel(logging.INFO)
    ph_file.setFormatter(fmt)
    progress.addHandler(ph_file)

    for name in _NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)

    # Uvicorn access: keep errors, drop status polls via filter on access logger
    access = logging.getLogger("uvicorn.access")
    access.addFilter(_StatusPollFilter())

    logging.getLogger(__name__).info("Full logs -> %s", log_path)
    return log_path


def progress_log(message: str) -> None:
    logging.getLogger("marine_docs.progress").info(message)
