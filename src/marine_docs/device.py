"""Device / accelerator helpers for local Docling models."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def resolve_torch_device(requested: str = "auto") -> str:
    """Return 'cuda', 'mps', or 'cpu'.

    Notes:
    - Mistral OCR via LiteLLM runs on Mistral's servers — your local GPU is unused there.
    - Docling layout/table models run locally and benefit from CUDA when available.
    - Current env may have CPU-only torch; install a CUDA build to enable GPU.
    """
    req = (requested or "auto").strip().lower()
    if req in {"cpu", "cuda", "mps"}:
        if req == "cuda" and not _cuda_available():
            logger.warning("DOCLING_DEVICE=cuda requested but CUDA unavailable; falling back to CPU")
            return "cpu"
        if req == "mps" and not _mps_available():
            logger.warning("DOCLING_DEVICE=mps requested but MPS unavailable; falling back to CPU")
            return "cpu"
        return req

    if _cuda_available():
        return "cuda"
    if _mps_available():
        return "mps"
    return "cpu"


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _mps_available() -> bool:
    try:
        import torch

        return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception:
        return False


def configure_docling_accelerator(pipeline_options, device: str) -> str:
    """Attach AcceleratorOptions to Docling PdfPipelineOptions when supported."""
    resolved = resolve_torch_device(device)
    try:
        from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions

        device_map = {
            "cuda": AcceleratorDevice.CUDA,
            "cpu": AcceleratorDevice.CPU,
            "mps": getattr(AcceleratorDevice, "MPS", AcceleratorDevice.CPU),
        }
        pipeline_options.accelerator_options = AcceleratorOptions(
            device=device_map.get(resolved, AcceleratorDevice.CPU)
        )
        logger.info("Docling accelerator device=%s", resolved)
    except Exception as exc:
        # Older Docling versions / missing enum values
        logger.warning("Could not set Docling accelerator_options (%s); device=%s", exc, resolved)
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0" if resolved == "cuda" else "")
    return resolved
