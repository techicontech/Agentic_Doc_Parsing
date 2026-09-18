"""LiteLLM wrapper — sole model access path for OCR + chat."""

from __future__ import annotations

import logging
import os
from typing import Any

from marine_docs.config import get_settings

logger = logging.getLogger(__name__)


def configure_litellm() -> None:
    settings = get_settings()
    # LiteLLM's logger inherits our root DEBUG level and prints every request body.
    os.environ.setdefault("LITELLM_LOG", "ERROR")
    for name in ("LiteLLM", "LiteLLM Proxy", "LiteLLM Router"):
        logging.getLogger(name).setLevel(logging.WARNING)
    if settings.litellm_api_base:
        os.environ["LITELLM_API_BASE"] = _normalize_api_base(settings.litellm_api_base)
    if settings.litellm_api_key:
        os.environ["LITELLM_API_KEY"] = settings.litellm_api_key
    if settings.mistral_api_key:
        os.environ["MISTRAL_API_KEY"] = settings.mistral_api_key
    if settings.anthropic_api_key:
        os.environ["ANTHROPIC_API_KEY"] = settings.anthropic_api_key


def _normalize_api_base(base: str) -> str:
    """LiteLLM appends /v1/... itself — strip trailing /v1 to avoid /v1/v1/..."""
    b = base.strip().rstrip("/")
    if b.endswith("/v1"):
        b = b[:-3].rstrip("/")
    return b


def chat_completion(
    messages: list[dict[str, Any]],
    *,
    model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 1200,
) -> str:
    configure_litellm()
    settings = get_settings()
    model_name = model or settings.llm_model

    try:
        from litellm import completion
    except ImportError as exc:
        raise RuntimeError("litellm is not installed") from exc

    kwargs: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if settings.litellm_api_base:
        # Talk to LiteLLM / OpenAI-compatible proxy
        kwargs["api_base"] = _normalize_api_base(settings.litellm_api_base)
        kwargs["api_key"] = settings.litellm_api_key or settings.anthropic_api_key or "sk-none"
        # Force OpenAI-compatible chat/completions path on the proxy
        kwargs["custom_llm_provider"] = "openai"
        # Keep original model id as the proxy's model name
        if "/" in model_name:
            # Prefer passing the full litellm model string the proxy expects
            kwargs["model"] = model_name
    elif settings.anthropic_api_key and model_name.startswith("anthropic/"):
        kwargs["api_key"] = settings.anthropic_api_key
    elif settings.litellm_api_key:
        kwargs["api_key"] = settings.litellm_api_key

    response = completion(**kwargs)
    return response.choices[0].message.content or ""


def llm_ready() -> bool:
    settings = get_settings()
    if settings.litellm_api_base and (settings.litellm_api_key or settings.anthropic_api_key):
        return True
    return bool(settings.anthropic_api_key or settings.litellm_api_key)
