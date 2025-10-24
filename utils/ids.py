import os
import re

GERMAN_MAP = {"ä":"ae", "ö":"oe", "ü":"ue", "Ä":"Ae", "Ö":"Oe", "Ü":"Ue", "ß":"ss"}

def translit_german(s: str) -> str:
    return "".join(GERMAN_MAP.get(ch, ch) for ch in s)

def pinecone_safe_slug(name: str, max_len: int = 48) -> str:
    base = os.path.splitext(os.path.basename(name))[0]
    base = translit_german(base).lower()
    base = re.sub(r"[^a-z0-9._:-]+", "-", base)
    base = re.sub(r"-{2,}", "-", base).strip("-")
    return (base or "doc")[:max_len]

def sanitize_namespace(ns: str) -> str:
    ns = translit_german(ns).lower()
    ns = re.sub(r"[^a-z0-9._:-]+", "-", ns)
    ns = re.sub(r"-{2,}", "-", ns).strip("-")
    return ns or "__default__"
