"""Document ingestion and vectorisation helpers."""

from __future__ import annotations

import io
import logging
import uuid
from collections.abc import Iterable
from datetime import datetime as dt
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

try:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError
except ImportError:  # pragma: no cover - optional dependency
    PdfReader = None
    PdfReadError = RuntimeError
try:
    import docx
except ImportError:
    docx = None

from openai import OpenAI
from pinecone import Index

from utils.chunk import chunk_text
from utils.ids import pinecone_safe_slug, sanitize_namespace

from .pinecone_utils import embed_texts

PYPDF_MISSING_MSG = "pypdf is not installed"
DOCX_MISSING_MSG = "python-docx is not installed"


@runtime_checkable
class Uploadable(Protocol):
    """Minimal protocol for uploaded files handled by Streamlit."""

    name: str

    def read(self) -> bytes:
        """Return the raw file contents."""


logger = logging.getLogger(__name__)


def extract_text_units(uploaded_file: Uploadable) -> list[tuple[str, str]]:
    """Extract text units from an uploaded file for downstream chunking."""
    name = uploaded_file.name
    suffix = Path(name).suffix.lower()
    if suffix in [".txt", ".md", ".csv", ".log"]:
        raw = uploaded_file.read()
        try:
            text = raw.decode("utf-8", errors="ignore")
        except UnicodeDecodeError:
            text = raw.decode("latin-1", errors="ignore")
        return [(text, f"{name}")]
    if suffix == ".pdf":
        if PdfReader is None:
            raise RuntimeError(PYPDF_MISSING_MSG)
        reader = PdfReader(io.BytesIO(uploaded_file.read()))
        units = []
        for i, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text() or ""
            except PdfReadError as exc:
                logger.warning(
                    "Failed to extract text from PDF page %s of %s",
                    i + 1,
                    name,
                    exc_info=exc,
                )
                page_text = ""
            if page_text.strip():
                units.append((page_text, f"{name}#page={i+1}"))
        return units or [("", f"{name}")]
    if suffix == ".docx":
        if docx is None:
            raise RuntimeError(DOCX_MISSING_MSG)
        document = docx.Document(io.BytesIO(uploaded_file.read()))
        paras = [p.text for p in document.paragraphs if p.text and p.text.strip()]
        text = "\n".join(paras)
        return [(text, f"{name}")]
    # Fallback for unknown types
    raw = uploaded_file.read()
    try:
        text = raw.decode("utf-8", errors="ignore")
    except UnicodeDecodeError:
        text = raw.decode("latin-1", errors="ignore")
    return [(text, f"{name}")]


def build_context(
    matches: Iterable[dict[str, Any]] | None,
    text_key: str,
    source_key: str,
    max_chars: int,
) -> dict[str, Any]:
    """Build a context blob from retrieved vector matches."""
    contexts: list[str] = []
    sources: list[dict[str, Any]] = []
    total = 0
    for match in matches or []:
        metadata = match.get("metadata") or {}
        chunk = str(metadata.get(text_key, ""))
        if not chunk:
            continue
        if total + len(chunk) > max_chars:
            break
        total += len(chunk)
        contexts.append(chunk)
        src = metadata.get(source_key)
        sources.append(
            {
                "id": match.get("id"),
                "score": match.get("score"),
                "source": str(src) if src else None,
            }
        )
    return {"context_text": "\n\n---\n\n".join(contexts), "sources": sources}


def upsert_chunks(
    client: OpenAI,
    index: Index,
    *,
    embedding_model: str,
    filename: str,
    text_units: list[tuple[str, str]],
    namespace: str | None,
    chunk_size: int,
    chunk_overlap: int,
    md_text_key: str,
    md_source_key: str,
) -> dict[str, Any]:
    """Chunk, embed, and upsert document text into Pinecone."""
    now = dt.utcnow().isoformat() + "Z"
    doc_id = f"{pinecone_safe_slug(filename)}-{uuid.uuid4().hex[:8]}"
    all_chunks: list[str] = []
    all_sources: list[str] = []
    vec_ids: list[str] = []
    for unit_text, unit_src in text_units:
        for chunk in chunk_text(
            unit_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        ):
            all_chunks.append(chunk)
            all_sources.append(unit_src)
    embeddings: list[list[float]] = []
    if not all_chunks:
        logger.warning("No text chunks found for %s, skipping upsert.", filename)
        return {
            "doc_id": doc_id,
            "filename": filename,
            "namespace": namespace,
            "vector_count": 0,
            "vector_ids": [],
            "uploaded_at": now,
        }

    for i in range(0, len(all_chunks), 64):
        sub = all_chunks[i : i + 64]
        vecs = embed_texts(client, sub, embedding_model)
        embeddings.extend(vecs)

    def json_safe(meta: dict[str, Any]) -> dict[str, Any]:
        """Coerce metadata to JSON-friendly primitives."""
        out: dict[str, Any] = {}
        for key, value in meta.items():
            if isinstance(value, (str | int | float | bool)) or value is None:
                out[key] = value
            else:
                out[key] = str(value)
        return out

    vectors: list[dict[str, Any]] = []
    ns = sanitize_namespace(namespace or "__default__")
    for idx, (embedding, source, chunk_text_value) in enumerate(
        zip(embeddings, all_sources, all_chunks, strict=False)
    ):
        vid = f"{doc_id}::chunk-{idx:04d}"
        meta = json_safe(
            {
                md_text_key: chunk_text_value,
                md_source_key: source,
                "doc_id": doc_id,
                "filename": filename,
                "uploaded_at": now,
                "chunk_index": idx,
            }
        )
        vectors.append({"id": vid, "values": [float(x) for x in embedding], "metadata": meta})
        vec_ids.append(vid)
    for i in range(0, len(vectors), 1000):
        index.upsert(vectors=vectors[i : i + 1000], namespace=ns)
    return {
        "doc_id": doc_id,
        "filename": filename,
        "namespace": ns,
        "vector_count": len(vectors),
        "vector_ids": vec_ids,
        "uploaded_at": now,
    }

