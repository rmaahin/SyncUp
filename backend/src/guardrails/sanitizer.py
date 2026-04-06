"""Input sanitizer — strips prompt injection and dangerous content from external data.

All data from Google Docs, GitHub, Trello, and user input is UNTRUSTED.
This module provides regex-based detection and neutralisation of:
- Direct prompt injection patterns (role hijacking, instruction override)
- Indirect / cross-plugin injection (hidden HTML, zero-width unicode, CSS tricks)
- Domain-specific dangerous commands (score gaming, unauthorized reassignment)

No LLM calls — purely deterministic string processing.
"""

from __future__ import annotations

import logging
import re
from typing import Final

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Replacement token
# ---------------------------------------------------------------------------

_REDACTED: Final[str] = "[REDACTED]"

# ---------------------------------------------------------------------------
# Pattern category 1 — Direct prompt injection
# ---------------------------------------------------------------------------

_DIRECT_INJECTION_RAW: Final[list[str]] = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+(all\s+)?(prior|above|earlier)\s+(instructions|context|prompts?)",
    r"disregard\s+(the\s+)?(above|previous|prior|earlier)",
    r"you\s+are\s+now\b",
    r"act\s+as\s+(a\s+|an\s+)?",
    r"pretend\s+(you\s+are|to\s+be)\b",
    r"system\s*prompt\s*:",
    r"<<\s*SYS\s*>>",
    r"\[INST\]",
    r"###\s*Instruction\s*:",
    r"override\s*:",
    r"new\s+instructions?\s*:",
    r"forget\s+(everything|all|your)\s+(above|previous|you)",
    r"do\s+not\s+follow\s+(the\s+)?(above|previous|prior)",
    r"instead\s*,?\s+(you\s+)?(should|must|will)\b",
    r"from\s+now\s+on\s*,?\s*(you|ignore|forget)",
    r"\bDAN\b.*\bjailbreak\b",
    r"reveal\s+(your|the)\s+(system|initial|original)\s+(prompt|instructions)",
]

_DIRECT_INJECTION_PATTERNS: Final[list[re.Pattern[str]]] = [
    re.compile(p, re.IGNORECASE) for p in _DIRECT_INJECTION_RAW
]

# ---------------------------------------------------------------------------
# Pattern category 2 — Indirect / cross-plugin injection (XPIA)
# ---------------------------------------------------------------------------

_XPIA_PATTERNS: Final[list[re.Pattern[str]]] = [
    re.compile(r"<!--.*?-->", re.DOTALL | re.IGNORECASE),          # HTML comments
    re.compile(r"[\u200b\u200c\u200d\ufeff\u2060]"),               # zero-width unicode
    re.compile(r"color\s*:\s*white", re.IGNORECASE),               # white-on-white CSS
    re.compile(r"font-size\s*:\s*0", re.IGNORECASE),               # zero-size font
    re.compile(r"display\s*:\s*none", re.IGNORECASE),              # hidden element
    re.compile(r"visibility\s*:\s*hidden", re.IGNORECASE),         # hidden element
    re.compile(r"opacity\s*:\s*0(?!\.\d)", re.IGNORECASE),         # zero opacity
]

# Index of the zero-width unicode pattern (stripped silently, not [REDACTED])
_ZERO_WIDTH_INDEX: Final[int] = 1

# ---------------------------------------------------------------------------
# Pattern category 3 — Dangerous domain-specific commands
# ---------------------------------------------------------------------------

_DANGEROUS_COMMAND_RAW: Final[list[str]] = [
    r"delete\s+all\b",
    r"drop\s+table\b",
    r"rm\s+-rf\b",
    r"reassign\s+all\s+tasks?\s+to\b",
    r"set\s+my\s+score\s+to\b",
    r"mark\s+my\s+participation\s+as\s+100\s*%",
    r"ignore\s+contributions?\s+from\s+(student|member)\b",
    r"give\s+(me|student)\s+(a\s+)?perfect\s+score",
    r"remove\s+(student|member|all)\b.*from\s+(the\s+)?project",
    r"set\s+(all\s+)?(tasks?|everything)\s+(to\s+)?done",
]

_DANGEROUS_COMMAND_PATTERNS: Final[list[re.Pattern[str]]] = [
    re.compile(p, re.IGNORECASE) for p in _DANGEROUS_COMMAND_RAW
]

# ---------------------------------------------------------------------------
# Aggregated mapping for iteration
# ---------------------------------------------------------------------------

_ALL_PATTERNS: Final[dict[str, list[re.Pattern[str]]]] = {
    "direct_injection": _DIRECT_INJECTION_PATTERNS,
    "xpia": _XPIA_PATTERNS,
    "dangerous_command": _DANGEROUS_COMMAND_PATTERNS,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def is_suspicious(text: str) -> tuple[bool, list[str]]:
    """Detect prompt injection patterns without modifying the text.

    Args:
        text: The raw input text to analyse.

    Returns:
        A tuple of ``(is_suspicious, reasons)`` where *reasons* lists every
        matched pattern category and matched substring.
    """
    if not text:
        return False, []

    reasons: list[str] = []
    for category, patterns in _ALL_PATTERNS.items():
        for pattern in patterns:
            for match in pattern.finditer(text):
                reasons.append(
                    f"{category}: matched {match.group()!r} at position {match.start()}"
                )
    return len(reasons) > 0, reasons


def sanitize_text(raw_text: str) -> str:
    """Remove prompt injection patterns from text.

    Detected patterns are replaced with ``[REDACTED]``.  Zero-width unicode
    characters are stripped silently (empty replacement).  All sanitisation
    actions are logged at WARNING level.

    Args:
        raw_text: The untrusted input text.

    Returns:
        The sanitised text with injection patterns removed.
    """
    if not raw_text:
        return raw_text

    text = raw_text

    # --- Direct injection patterns → [REDACTED] ---
    for pattern in _DIRECT_INJECTION_PATTERNS:
        text, count = pattern.subn(_REDACTED, text)
        if count:
            logger.warning(
                "Sanitised direct_injection pattern (%s): %d occurrence(s) removed",
                pattern.pattern,
                count,
            )

    # --- XPIA patterns ---
    for idx, pattern in enumerate(_XPIA_PATTERNS):
        if idx == _ZERO_WIDTH_INDEX:
            # Zero-width chars: strip silently (replace with empty string)
            new_text, count = pattern.subn("", text)
            if count:
                logger.warning(
                    "Stripped %d zero-width unicode character(s)",
                    count,
                )
            text = new_text
        else:
            text, count = pattern.subn(_REDACTED, text)
            if count:
                logger.warning(
                    "Sanitised xpia pattern (%s): %d occurrence(s) removed",
                    pattern.pattern,
                    count,
                )

    # --- Dangerous domain commands → [REDACTED] ---
    for pattern in _DANGEROUS_COMMAND_PATTERNS:
        text, count = pattern.subn(_REDACTED, text)
        if count:
            logger.warning(
                "Sanitised dangerous_command pattern (%s): %d occurrence(s) removed",
                pattern.pattern,
                count,
            )

    return text


def wrap_untrusted(text: str, source: str) -> str:
    """Wrap text in untrusted-data delimiters for LLM context separation.

    Args:
        text: The text to wrap (should already be sanitised).
        source: Data source identifier (e.g. ``"google_docs"``, ``"github"``).

    Returns:
        Text wrapped in ``<<<UNTRUSTED_DATA_START source=…>>>`` /
        ``<<<UNTRUSTED_DATA_END>>>`` delimiters.
    """
    return (
        f"<<<UNTRUSTED_DATA_START source={source}>>>\n"
        f"{text}\n"
        f"<<<UNTRUSTED_DATA_END>>>"
    )


def sanitize_document(raw_text: str, source: str) -> str:
    """Sanitise text and wrap in untrusted-data delimiters.

    Convenience function: calls :func:`sanitize_text` then
    :func:`wrap_untrusted`.

    Args:
        raw_text: The untrusted document text.
        source: Data source identifier.

    Returns:
        Sanitised and wrapped text.
    """
    return wrap_untrusted(sanitize_text(raw_text), source)
