"""String helpers for producing safe IDs and namespace keys."""

from __future__ import annotations

import re
from pathlib import Path

GERMAN_MAP = {"ä": "ae", "ö": "oe", "ü": "ue", "Ä": "Ae", "Ö": "Oe", "Ü": "Ue", "ß": "ss"}


def translit_german(text: str) -> str:
    """Return ``text`` with German characters transliterated to ASCII equivalents."""
    return "".join(GERMAN_MAP.get(char, char) for char in text)


def pinecone_safe_slug(name: str, max_len: int = 48) -> str:
    """Normalise ``name`` into a Pinecone-friendly slug with length bounds."""
    path = Path(name)
    base = translit_german(path.stem).lower()
    base = re.sub(r"[^a-z0-9._:-]+", "-", base)
    base = re.sub(r"-{2,}", "-", base).strip("-")
    return (base or "doc")[:max_len]


def sanitize_namespace(ns: str) -> str:
    """Normalise Pinecone namespaces to safe characters."""
    namespace = translit_german(ns).lower()
    namespace = re.sub(r"[^a-z0-9._:-]+", "-", namespace)
    namespace = re.sub(r"-{2,}", "-", namespace).strip("-")
    return namespace or "__default__"
