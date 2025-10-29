"""Document ingestion and vectorisation helpers."""

from __future__ import annotations

import io
import logging
import uuid
from collections.abc import Iterable
from datetime import datetime as dt
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Protocol, runtime_checkable

try:
    from docling.document_converter import DocumentConverter
except ImportError:  # pragma: no cover - optional dependency
    DocumentConverter = None
from typing import Any, Protocol, runtime_checkable

try:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError
except ImportError:  # pragma: no cover - optional dependency
    PdfReader = None
    PdfReadError = RuntimeError
try:
    import docx
except ImportError:  # pragma: no cover - optional dependency
    docx = None

from openai import OpenAI
from pinecone import Index

from utils.chunk import chunk_text
from utils.ids import pinecone_safe_slug, sanitize_namespace

from .pinecone_utils import embed_texts

PYPDF_MISSING_MSG = "pypdf is not installed"
DOCX_MISSING_MSG = "python-docx is not installed"
DOCLING_MISSING_MSG = "docling is not installed"
DOCLING_EMPTY_DOC_MSG = "Docling conversion returned no document"


@runtime_checkable
class Uploadable(Protocol):
    """Minimal protocol for uploaded files handled by Streamlit."""

    name: str

    def read(self) -> bytes:
        """Return the raw file contents."""


logger = logging.getLogger(__name__)

_docling_converter: DocumentConverter | None = None


def _get_docling_converter() -> DocumentConverter:
    """Return a shared Docling converter instance."""
    if DocumentConverter is None:  # pragma: no cover - optional dependency
        raise RuntimeError(DOCLING_MISSING_MSG)

    global _docling_converter
    if _docling_converter is None:
        _docling_converter = DocumentConverter()
    return _docling_converter


def _extract_with_docling(name: str, suffix: str, payload: bytes) -> list[tuple[str, str]]:
    """Extract text from arbitrary documents using Docling when available."""
    converter = _get_docling_converter()
    with NamedTemporaryFile(suffix=suffix or ".bin", delete=False) as tmp:
        tmp.write(payload)
        tmp_path = Path(tmp.name)
    try:
        result = converter.convert(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    document = getattr(result, "document", None)
    if document is None:
        raise RuntimeError(DOCLING_EMPTY_DOC_MSG)

    text: str | None = None
    for attr in ("export_to_markdown", "export_to_text", "export_to_plaintext"):
        exporter = getattr(document, attr, None)
        if callable(exporter):
            try:
                candidate = exporter()
            except Exception as exc:  # noqa: BLE001  # pragma: no cover - defensive fallback
                logger.debug("Docling exporter %s failed for %s", attr, name, exc_info=exc)
                continue
            if isinstance(candidate, str) and candidate.strip():
                text = candidate
                break

    if not text:
        text = str(document)

    return [(text, f"{name}#docling")]


def extract_text_units(uploaded_file: Uploadable) -> list[tuple[str, str]]:
    """Extract text units from an uploaded file for downstream chunking."""
    name = uploaded_file.name
    suffix = Path(name).suffix.lower()
    payload = uploaded_file.read() or b""

    if payload and DocumentConverter is not None:
        try:
            return _extract_with_docling(name, suffix, payload)
        except RuntimeError as exc:
            logger.info("Docling could not process %s: %s", name, exc)
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - defensive fallback
            logger.warning("Docling processing crashed for %s", name, exc_info=exc)

    if suffix in [".txt", ".md", ".csv", ".log"]:
        try:
            text = payload.decode("utf-8", errors="ignore")
        except UnicodeDecodeError:
            text = payload.decode("latin-1", errors="ignore")
        return [(text, f"{name}")]

    if suffix == ".pdf":
        if PdfReader is None:
            raise RuntimeError(PYPDF_MISSING_MSG)
        reader = PdfReader(io.BytesIO(payload))
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
        document = docx.Document(io.BytesIO(payload))
        document = docx.Document(io.BytesIO(uploaded_file.read()))
        paras = [p.text for p in document.paragraphs if p.text and p.text.strip()]
        text = "\n".join(paras)
        return [(text, f"{name}")]

    # Fallback for unknown types
    try:
        text = payload.decode("utf-8", errors="ignore")
    except UnicodeDecodeError:
        text = payload.decode("latin-1", errors="ignore")
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
                "index": (
                    str(metadata.get("index_name"))
                    if metadata.get("index_name")
                    else None
                ),
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
    index_name: str | None,
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
                "index_name": index_name,
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
        "index_name": index_name,
        "vector_count": len(vectors),
        "vector_ids": vec_ids,
        "uploaded_at": now,
    }

