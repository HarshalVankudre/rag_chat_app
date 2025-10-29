"""Utility helpers for splitting text into overlapping chunks."""


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[str]:
    """Return overlapping segments of ``text`` respecting the configured size."""
    if chunk_size <= 0:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap if chunk_overlap > 0 else end
        if start < 0:
            start = 0
    return [c.strip() for c in chunks if c and c.strip()]
