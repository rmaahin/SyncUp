"""LLM factory — centralised Groq-backed model construction for SyncUp agents.

Provides ``get_high_tier_llm`` and ``get_low_tier_llm`` that return
``ChatGroq`` instances configured from environment variables.
"""

from __future__ import annotations

import os

from langchain_groq import ChatGroq


def _get_api_key() -> str:
    """Return the Groq API key from the environment.

    Raises:
        RuntimeError: If ``GROQ_API_KEY`` is not set or empty.
    """
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY environment variable is not set. "
            "Set it in your .env file or export it before running."
        )
    return key


def get_high_tier_llm(temperature: float = 0.7) -> ChatGroq:
    """High-tier LLM for complex reasoning agents.

    Uses ``LLM_HIGH_TIER_MODEL`` env var (default ``llama-3.3-70b-versatile``).
    """
    return ChatGroq(
        model=os.environ.get("LLM_HIGH_TIER_MODEL", "llama-3.3-70b-versatile"),
        api_key=_get_api_key(),  # type: ignore[arg-type]
        temperature=temperature,
    )


def get_low_tier_llm(temperature: float = 0.7) -> ChatGroq:
    """Low-tier LLM for structured / simple agents.

    Uses ``LLM_LOW_TIER_MODEL`` env var (default ``llama-3.1-8b-instant``).
    """
    return ChatGroq(
        model=os.environ.get("LLM_LOW_TIER_MODEL", "llama-3.1-8b-instant"),
        api_key=_get_api_key(),  # type: ignore[arg-type]
        temperature=temperature,
    )
