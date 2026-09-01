"""Shared deterministic text helpers.

Extracted from ``agent.py`` so the retrieval path and the state manager
(``starter/state.py``) tokenize identically without a circular import.
No behavior change: same regex, same stopword set, same functions.
"""

from __future__ import annotations

import re

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}


def flatten_text(value: object) -> str:
    """Render a catalog field (str / list / dict / None) as one string."""
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def terms(text: str) -> list[str]:
    """Lowercased content tokens: length > 1, not a stopword."""
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]
