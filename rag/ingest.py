"""Document ingestion and vectorisation helpers (with RapidOCR GPU + rich logging)."""

from __future__ import annotations
import io
import logging
import os
import time
import uuid
from collections.abc import Iterable
from datetime import datetime as dt, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

# ---- Optional Docling converter import (kept lazy-safe) ----
try:
    from docling.document_converter import DocumentConverter
except ImportError:  # pragma: no cover - optional dependency
    DocumentConverter = None  # type: ignore[assignment]

# ---- Optional deps for fallbacks ----
try:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError
except ImportError:  # pragma: no cover
    PdfReader = None  # type: ignore[assignment]
    PdfReadError = RuntimeError  # type: ignore[assignment]

try:
    import docx  # python-docx
except ImportError:  # pragma: no cover
    docx = None  # type: ignore[assignment]

# ---- ONNX Runtime probe (for logging) ----
try:
    import onnxruntime as ort  # type: ignore
except Exception:  # pragma: no cover
    ort = None  # type: ignore

from openai import OpenAI

# ---- v4-safe type-only import for Pinecone Index (runtime uses Any fallback) ----
if TYPE_CHECKING:
    from pinecone import Index  # type: ignore[attr-defined]
else:
    Index = Any  # type: ignore[misc,assignment]

from utils.chunk import chunk_text
from utils.ids import pinecone_safe_slug, sanitize_namespace
from rag.pinecone_utils import embed_texts

# ------------------ Constants & Messages ------------------

PYPDF_MISSING_MSG = "pypdf is not installed"
DOCX_MISSING_MSG = "python-docx is not installed"
DOCLING_MISSING_MSG = "docling is not installed"
DOCLING_EMPTY_DOC_MSG = "Docling conversion returned no document"

# Default folder where you placed your three ONNX model files
DEFAULT_MODEL_DIR = Path("ocr-models")

logger = logging.getLogger(__name__)
if not logger.handlers:
    # Safe default formatter; main app can reconfigure
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)

_docling_converter: DocumentConverter | None = None


# ------------------ Small helpers ------------------


def _pydantic_field_names(model_cls: Any) -> set[str]:
    """Return declared field names for Pydantic v1/v2 model classes."""
    names: set[str] = set()
    try:
        # Pydantic v2
        fields = getattr(model_cls, "model_fields", None)
        if fields:
            names.update(fields.keys())
    except Exception:
        pass
    try:
        # Pydantic v1
        fields_v1 = getattr(model_cls, "__fields__", None)
        if fields_v1:
            names.update(fields_v1.keys())
    except Exception:
        pass
    return names


def _probe_onnxruntime() -> tuple[str, list[str]]:
    """Return (device, providers) for log visibility."""
    if ort is None:
        return ("N/A", [])
    device = "UNKNOWN"
    providers: list[str] = []
    try:
        device = ort.get_device()
    except Exception:
        pass
    try:
        providers = list(ort.get_available_providers() or [])
    except Exception:
        pass
    return (device, providers)


def _maybe_enable_tensorrt_cache() -> None:
    """Enable TensorRT engine caching if TRT is present and env not set."""
    if ort is None:
        return
    try:
        providers = ort.get_available_providers()
    except Exception:
        providers = []
    if "TensorrtExecutionProvider" in providers:
        os.environ.setdefault("ORT_TENSORRT_ENGINE_CACHE_ENABLE", "1")
        os.environ.setdefault("ORT_TENSORRT_ENGINE_CACHE_PATH", ".ort_trt_cache")


# ------------------ Docling converter (GPU RapidOCR) ------------------


def _get_docling_converter() -> DocumentConverter:
    """Return a shared Docling converter configured for RapidOCR (ONNXRuntime, GPU)."""
    if DocumentConverter is None:  # pragma: no cover - optional dependency
        raise RuntimeError(DOCLING_MISSING_MSG)

    global _docling_converter
    if _docling_converter is not None:
        return _docling_converter

    # Probe ORT once (prints in logs so you can confirm GPU is available)
    device, providers = _probe_onnxruntime()
    logger.info("ONNXRuntime device: %s", device)
    logger.info("ONNXRuntime providers (available): %s", providers)

    # Prefer CUDA over TensorRT by default (TRT can be slow on first engine build)
    prefer_cuda = os.environ.get("RAPID_OCR_PREFER_CUDA", "1") not in ("0", "false", "False")
    providers_kw: dict[str, Any] = {}
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
        from docling.document_converter import PdfFormatOption
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Docling OCR pipeline options are unavailable. Ensure a compatible Docling version is installed."
        ) from exc

    # Detect which fields this Docling build supports
    rapid_fields = _pydantic_field_names(RapidOcrOptions)
    pdf_fields = _pydantic_field_names(PdfPipelineOptions)

    # Optional: pass providers in a feature-detected way
    if "providers" in rapid_fields and providers:
        if prefer_cuda and "CUDAExecutionProvider" in providers:
            providers_kw["providers"] = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            # Keep ORT's default order
            providers_kw["providers"] = providers

    # Build RapidOCR options (only include fields guaranteed by your version)
    model_dir = DEFAULT_MODEL_DIR
    rapid_kwargs: dict[str, Any] = {
        "backend": "onnxruntime",  # <- IMPORTANT: this is the accepted literal
        "det_model_path": str(model_dir / "ch_PP-OCRv4_det_server_infer.onnx"),
        "rec_model_path": str(model_dir / "ch_PP-OCRv4_rec_server_infer.onnx"),
        "cls_model_path": str(model_dir / "ch_ppocr_mobile_v2.0_cls_infer.onnx"),
        **providers_kw,
    }
    # Some builds allow 'use_angle_cls' and 'precision'
    if "use_angle_cls" in rapid_fields:
        rapid_kwargs["use_angle_cls"] = True
    if "precision" in rapid_fields:
        rapid_kwargs["precision"] = "fp32"
    # (We intentionally skip det_db_* and languages unless your build exposes them.)

    ocr_opts = RapidOcrOptions(**rapid_kwargs)

    # Pdf pipeline options (feature-detected)
    pdf_kwargs: dict[str, Any] = {"ocr_options": ocr_opts}
    # Try to set DPI to 400 for a good speed/quality balance, if supported
    if "render_dpi" in pdf_fields:
        dpi = int(os.environ.get("DOCLING_RENDER_DPI", "400"))
        pdf_kwargs["render_dpi"] = dpi
        logger.info("PDF render DPI set to %s (env DOCLING_RENDER_DPI)", dpi)
    # Force OCR if the field exists
    if "do_ocr" in pdf_fields:
        pdf_kwargs["do_ocr"] = True

    pdf_opts = PdfPipelineOptions(**pdf_kwargs)

    # Optional: enable TRT engine caching if TRT is around
    _maybe_enable_tensorrt_cache()

    # Build DocumentConverter
    t0 = time.time()
    _docling_converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)}
    )
    logger.info("DocumentConverter built in %.2fs", time.time() - t0)
    return _docling_converter


# ------------------ Doc extraction (with timings) ------------------


@runtime_checkable
class Uploadable(Protocol):
    """Minimal protocol for uploaded files handled by Streamlit."""

    name: str

    def read(self) -> bytes: ...


def _extract_with_docling(name: str, suffix: str, payload: bytes) -> list[tuple[str, str]]:
    """Extract text from arbitrary documents using Docling with OCR; logs timings."""
    converter = _get_docling_converter()

    # Best-effort page count (for progress-style logging)
    page_count: int | None = None
    if suffix == ".pdf" and PdfReader is not None:
        try:
            page_count = len(PdfReader(io.BytesIO(payload)).pages)
        except Exception:
            page_count = None

    logger.info(
        "Starting Docling convert: %s | pages=%s",
        name,
        page_count if page_count is not None else "?",
    )

    with NamedTemporaryFile(suffix=suffix or ".bin", delete=False) as tmp:
        tmp.write(payload)
        tmp_path = Path(tmp.name)

    try:
        t0 = time.time()
        result = converter.convert(tmp_path)
        t1 = time.time()
        logger.info("Docling.convert finished in %.2fs", t1 - t0)

        document = getattr(result, "document", None)
        if document is None:
            raise RuntimeError(DOCLING_EMPTY_DOC_MSG)

        # Export text/markdown with timing
        text: str | None = None
        for attr in ("export_to_markdown", "export_to_text", "export_to_plaintext"):
            exporter = getattr(document, attr, None)
            if callable(exporter):
                try:
                    t_exp0 = time.time()
                    candidate = exporter()
                    t_exp1 = time.time()
                    logger.info("Document.%s finished in %.2fs", attr, t_exp1 - t_exp0)
                except (RuntimeError, ValueError, TypeError, AttributeError) as exc:  # defensive
                    logger.debug("Docling exporter %s failed for %s", attr, name, exc_info=exc)
                    continue
                if isinstance(candidate, str) and candidate.strip():
                    text = candidate
                    break

        if not text:
            text = str(document)

        logger.info("Extraction completed: %s | total=%.2fs", name, time.time() - t0)
        return [(text, f"{name}#docling")]

    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def extract_text_units(uploaded_file: Uploadable) -> list[tuple[str, str]]:
    """Extract text units from an uploaded file for downstream chunking."""
    name = uploaded_file.name
    suffix = Path(name).suffix.lower()
    payload = uploaded_file.read() or b""

    # Prefer Docling (with RapidOCR) for any non-trivial type
    if payload and DocumentConverter is not None:
        try:
            return _extract_with_docling(name, suffix, payload)
        except RuntimeError as exc:
            logger.info("Docling could not process %s: %s", name, exc)
        except (OSError, ValueError, TypeError) as exc:
            logger.warning("Docling processing crashed for %s", name, exc_info=exc)

    # Simple text-like fallbacks
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
        units: list[tuple[str, str]] = []
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
        paras = [p.text for p in document.paragraphs if p.text and p.text.strip()]
        text = "\n".join(paras)
        return [(text, f"{name}")]

    # Fallback for unknown types
    try:
        text = payload.decode("utf-8", errors="ignore")
    except UnicodeDecodeError:
        text = payload.decode("latin-1", errors="ignore")
    return [(text, f"{name}")]


# ------------------ Context build & upsert (unchanged except logging) ------------------


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
                "index": (str(metadata.get("index_name")) if metadata.get("index_name") else None),
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
    t0 = time.time()
    now = dt.now(timezone.utc).isoformat()
    doc_id = f"{pinecone_safe_slug(filename)}-{uuid.uuid4().hex[:8]}"
    all_chunks: list[str] = []
    all_sources: list[str] = []
    vec_ids: list[str] = []

    # Chunk the input units
    for unit_text, unit_src in text_units:
        for chunk in chunk_text(unit_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap):
            all_chunks.append(chunk)
            all_sources.append(unit_src)

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

    # Embed in small batches to avoid rate limits
    embeddings: list[list[float]] = []
    for i in range(0, len(all_chunks), 64):
        sub = all_chunks[i : i + 64]
        t_embed0 = time.time()
        vecs = embed_texts(client, sub, embedding_model)
        t_embed1 = time.time()
        logger.info("Embedded %d chunks in %.2fs", len(sub), t_embed1 - t_embed0)
        embeddings.extend(vecs)

    def json_safe(meta: dict[str, Any]) -> dict[str, Any]:
        """Coerce metadata to JSON-friendly primitives; drop None (Pinecone rejects null)."""
        out: dict[str, Any] = {}
        for key, value in meta.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                out[key] = value
            else:
                out[key] = str(value)
        return out

    ns = sanitize_namespace(namespace or "__default__")
    vectors: list[tuple[str, list[float], dict[str, Any]]] = []
    for idx, (embedding, source, chunk_text_value) in enumerate(
        zip(embeddings, all_sources, all_chunks, strict=False)
    ):
        vid = f"{doc_id}::chunk-{idx:04d}"
        meta_raw: dict[str, Any] = {
            md_text_key: chunk_text_value,
            md_source_key: source,
            "doc_id": doc_id,
            "filename": filename,
            "uploaded_at": now,
            "chunk_index": idx,
        }
        if index_name:  # only include when present (avoid null metadata)
            meta_raw["index_name"] = index_name

        md = json_safe(meta_raw)
        vectors.append((vid, [float(x) for x in embedding], md))
        vec_ids.append(vid)

    # Upsert in chunks; Index.upsert accepts list[tuple[id, values, metadata]]
    t_up0 = time.time()
    for i in range(0, len(vectors), 1000):
        batch = vectors[i : i + 1000]
        index.upsert(vectors=batch, namespace=ns)
        logger.info(
            "Upserted %d/%d vectors (namespace=%s)", min(i + 1000, len(vectors)), len(vectors), ns
        )
    logger.info("Upsert finished in %.2fs", time.time() - t_up0)

    logger.info("Full ingest for %s completed in %.2fs", filename, time.time() - t0)
    return {
        "doc_id": doc_id,
        "filename": filename,
        "namespace": ns,
        "index_name": index_name,
        "vector_count": len(vectors),
        "vector_ids": vec_ids,
        "uploaded_at": now,
    }
