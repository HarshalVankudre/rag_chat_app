"""Document ingestion and vectorisation helpers (Docling + RapidOCR on CPU, optimized for low-resource servers).

This version is optimized for CPU-only deployments on servers with limited resources (4 cores, 4GB RAM):
- Uses ONNXRuntime CPU execution provider (no GPU required)
- Uses lightweight mobile OCR models for better CPU performance
- Lower default rasterisation DPI (150) for faster processing
- Optimized thread settings for 4-core systems
- Angle classifier disabled by default for speed
- Optional page-chunk parallelism for very large PDFs (configurable workers)
- Environment-variable overrides for chunking and embedding batch size

Environment knobs (all optional):
- DOCLING_RENDER_DPI=150                # PDF render DPI for OCR; 150-200 is optimal for CPU
- RAPID_OCR_MODEL_FLAVOR=mobile         # mobile (default) or server - mobile is lighter for CPU
- RAPID_OCR_LANGS="en"                  # languages (default: en only)
- RAPID_OCR_ANGLE=0                     # default 0 (disable angle classifier for speed)
- RAPID_OCR_MAX_PAGES=<int>             # optional hard cap on pages
- DOCLING_PAR_PAGES=<int>               # split big PDFs into subsets of N pages (e.g., 10)
- DOCLING_PAR_WORKERS=<int>             # number of parallel workers (default: 2 for 4 cores)
- CHUNK_SIZE=<int>                      # override chunk size without changing callers
- CHUNK_OVERLAP=<int>                   # override chunk overlap
- EMBED_BATCH=<int>                     # override embedding batch size (default: 64 for limited RAM)
- OMP_NUM_THREADS=<int>                 # OpenMP threads (default: 4)
- TMPDIR=/dev/shm                       # optional on Linux to leverage RAM disk for temps
"""

from __future__ import annotations
import io
import json
import logging
import os
import time
import uuid
import hashlib
from collections.abc import Iterable
from datetime import datetime as dt, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

# ---- Optional Docling converter import (kept lazy-safe) ----
try:
    from docling.document_converter import DocumentConverter
except ImportError:  # pragma: no cover - optional dependency
    DocumentConverter = None  # type: ignore[assignment]

# ---- Optional deps for helpers ----
try:
    from pypdf import PdfReader, PdfWriter
    from pypdf.errors import PdfReadError
except ImportError:  # pragma: no cover
    PdfReader = None  # type: ignore[assignment]
    PdfWriter = None  # type: ignore[assignment]
    PdfReadError = RuntimeError  # type: ignore[assignment]

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
DOCLING_MISSING_MSG = "docling is not installed"
DOCLING_EMPTY_DOC_MSG = "Docling conversion returned no document"

DEFAULT_MODEL_DIR = Path("ocr-models")  # only used if you disable defaults

# Optimize thread settings for 4-core CPU systems
_cpu_count = min(os.cpu_count() or 4, 4)  # Cap at 4 cores for resource-constrained servers
os.environ.setdefault("OMP_NUM_THREADS", str(_cpu_count))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_cpu_count))
os.environ.setdefault("MKL_NUM_THREADS", str(_cpu_count))
# Disable GPU fallback attempts
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

logger = logging.getLogger(__name__)
if not logger.handlers:
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
        fields = getattr(model_cls, "model_fields", None)  # Pydantic v2
        if fields:
            names.update(fields.keys())
    except Exception:
        pass
    try:
        fields_v1 = getattr(model_cls, "__fields__", None)  # Pydantic v1
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




# ------------------ Docling converter (RapidOCR on CPU) ------------------

def _get_docling_converter() -> DocumentConverter:
    """Return a shared Docling converter configured for RapidOCR (ONNXRuntime, CPU-only)."""
    if DocumentConverter is None:  # pragma: no cover - optional dependency
        raise RuntimeError(DOCLING_MISSING_MSG)

    global _docling_converter
    if _docling_converter is not None:
        return _docling_converter

    # Probe ORT / Providers
    device, providers = _probe_onnxruntime()
    logger.info("ONNXRuntime device: %s", device)
    logger.info("ONNXRuntime providers (available): %s", providers)

    # Force CPU-only execution
    providers_kw: dict[str, Any] = {}
    if providers:
        # Use CPU provider only, filter out GPU providers
        cpu_providers = [p for p in providers if "CPU" in p]
        if cpu_providers:
            providers_kw["providers"] = cpu_providers
        else:
            # Fallback to default CPU provider if available
            if "CPUExecutionProvider" in providers:
                providers_kw["providers"] = ["CPUExecutionProvider"]
            else:
                logger.warning("No CPU execution provider found, using default providers")
                providers_kw["providers"] = providers

    # Import Docling options lazily and feature-detect fields
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions,
            RapidOcrOptions,
        )
        from docling.document_converter import PdfFormatOption
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Docling OCR pipeline options are unavailable. Ensure a compatible Docling version is installed."
        ) from exc

    rapid_fields = _pydantic_field_names(RapidOcrOptions)
    pdf_fields = _pydantic_field_names(PdfPipelineOptions)

    # Use lightweight mobile models for CPU performance
    rapid_kwargs: dict[str, Any] = {
        "backend": "onnxruntime",  # RapidOCR in Docling uses ORT
    }
    # Only attach explicit providers if this Docling version supports it
    if "providers" in rapid_fields and providers_kw.get("providers"):
        rapid_kwargs["providers"] = providers_kw["providers"]
    else:
        if providers_kw.get("providers"):
            logger.info("RapidOcrOptions.providers not supported by this Docling version; relying on onnxruntime CPU defaults.")

    # Force mobile models for CPU optimization (lighter and faster)
    use_default_models = os.environ.get("RAPID_OCR_USE_DEFAULT_MODELS", "1").lower() in (
        "1",
        "true",
        "yes",
    )
    
    if not use_default_models:
        # Use mobile flavor by default for CPU (lighter models)
        flavor = os.environ.get("RAPID_OCR_MODEL_FLAVOR", "mobile").lower()  # mobile|server
        det_name = f"ch_PP-OCRv4_det_{flavor}_infer.onnx"
        rec_name = f"ch_PP-OCRv4_rec_{flavor}_infer.onnx"
        cls_name = "ch_ppocr_mobile_v2.0_cls_infer.onnx"
        rapid_kwargs.update(
            {
                "det_model_path": str(DEFAULT_MODEL_DIR / det_name),
                "rec_model_path": str(DEFAULT_MODEL_DIR / rec_name),
                "cls_model_path": str(DEFAULT_MODEL_DIR / cls_name),
            }
        )
    else:
        # Even with default models, try to prefer mobile if available
        flavor = os.environ.get("RAPID_OCR_MODEL_FLAVOR", "mobile").lower()
        logger.info("Using default RapidOCR models with mobile flavor preference for CPU optimization")

    # Optimize for CPU: disable angle classifier by default (faster)
    if "use_angle_cls" in rapid_fields:
        rapid_kwargs["use_angle_cls"] = os.environ.get("RAPID_OCR_ANGLE", "0").lower() in (
            "1",
            "true",
            "yes",
        )
    
    # CPU doesn't need fp16 precision (that's GPU optimization)
    # Remove precision setting for CPU execution
    
    # Limit languages to reduce memory usage (default: English only)
    if "languages" in rapid_fields:
        langs = os.environ.get("RAPID_OCR_LANGS", "en")
        if langs:
            rapid_kwargs["languages"] = [s.strip() for s in langs.split(",") if s.strip()]

    ocr_opts = RapidOcrOptions(**rapid_kwargs)

    # Pdf pipeline options - lower DPI for CPU performance
    pdf_kwargs: dict[str, Any] = {"ocr_options": ocr_opts}
    if "render_dpi" in pdf_fields:
        # Lower DPI (150) for faster CPU processing, balance between quality and speed
        dpi = int(os.environ.get("DOCLING_RENDER_DPI", "150"))
        pdf_kwargs["render_dpi"] = dpi
        logger.info("PDF render DPI set to %s (env DOCLING_RENDER_DPI) - optimized for CPU", dpi)
    if "do_ocr" in pdf_fields:
        pdf_kwargs["do_ocr"] = True

    pdf_opts = PdfPipelineOptions(**pdf_kwargs)

    t0 = time.time()
    _docling_converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)}
    )
    logger.info("DocumentConverter built in %.2fs", time.time() - t0)
    return _docling_converter


# ------------------ Parallel helpers for big PDFs ------------------

def _split_pdf_to_subsets(payload: bytes, pages_per_chunk: int) -> list[Path]:
    if PdfReader is None or PdfWriter is None:
        raise RuntimeError(PYPDF_MISSING_MSG)
    reader = PdfReader(io.BytesIO(payload))
    subsets: list[Path] = []
    total = len(reader.pages)
    for start in range(0, total, pages_per_chunk):
        writer = PdfWriter()
        for i in range(start, min(start + pages_per_chunk, total)):
            writer.add_page(reader.pages[i])
        tmp = NamedTemporaryFile(suffix=".pdf", delete=False)
        writer.write(tmp)
        tmp.flush(); tmp.close()
        subsets.append(Path(tmp.name))
    return subsets


def _convert_subset(path: Path) -> str:
    # NOTE: Runs in a separate process when parallelism is enabled
    conv = _get_docling_converter()
    res = conv.convert(path)
    document = getattr(res, "document", None)
    if document is None:
        raise RuntimeError(DOCLING_EMPTY_DOC_MSG)
    # Prefer export_to_text for speed; fall back if not available
    for attr in ("export_to_text", "export_to_markdown", "export_to_plaintext"):
        exporter = getattr(document, attr, None)
        if callable(exporter):
            try:
                return exporter()
            except Exception:
                continue
    return str(document)


# ------------------ Doc extraction (always Docling) ------------------

@runtime_checkable
class Uploadable(Protocol):
    """Minimal protocol for uploaded files handled by Streamlit."""

    name: str

    def read(self) -> bytes: ...


def _extract_with_docling(name: str, suffix: str, payload: bytes) -> list[tuple[str, str]]:
    """Extract text from arbitrary documents using Docling with OCR; logs timings."""
    # Optional: page cap BEFORE running conversion
    if suffix == ".pdf" and os.environ.get("RAPID_OCR_MAX_PAGES") and PdfReader is not None:
        try:
            max_pages = int(os.environ["RAPID_OCR_MAX_PAGES"])
            reader = PdfReader(io.BytesIO(payload))
            if len(reader.pages) > max_pages:
                writer = PdfWriter()
                for i in range(min(max_pages, len(reader.pages))):
                    writer.add_page(reader.pages[i])
                with NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_sub:
                    writer.write(tmp_sub)
                    tmp_path = Path(tmp_sub.name)
                logger.info("Using first %d pages for OCR (RAPID_OCR_MAX_PAGES).", max_pages)
                # Replace payload with subset file path
                # Convert via Docling
                t0 = time.time()
                converter = _get_docling_converter()
                result = converter.convert(tmp_path)
                document = getattr(result, "document", None)
                if document is None:
                    raise RuntimeError(DOCLING_EMPTY_DOC_MSG)
                text: str | None = None
                for attr in ("export_to_text", "export_to_markdown", "export_to_plaintext"):
                    exporter = getattr(document, attr, None)
                    if callable(exporter):
                        try:
                            text = exporter()
                            break
                        except Exception:
                            continue
                if not text:
                    text = str(document)
                logger.info("Extraction completed: %s | total=%.2fs", name, time.time() - t0)
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass
                return [(text, f"{name}#docling")]
        except Exception:
            pass

    # Optional: parallel subsets for big PDFs - optimized for 4 cores
    # Default to 2 workers to leave resources for other processes
    par_pages = int(os.environ.get("DOCLING_PAR_PAGES", "0") or 0)
    par_workers = int(os.environ.get("DOCLING_PAR_WORKERS", "2") or 2)  # Default 2 for 4-core system
    if suffix == ".pdf" and par_pages > 0 and par_workers > 0 and PdfReader is not None and PdfWriter is not None:
        subsets = _split_pdf_to_subsets(payload, par_pages)
        texts: list[tuple[str, str]] = []
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=par_workers) as ex:
            futs = [ex.submit(_convert_subset, p) for p in subsets]
            for fu in as_completed(futs):
                txt = fu.result()
                if txt and txt.strip():
                    texts.append((txt, f"{name}#docling"))
        # Cleanup temp subset files
        for p in subsets:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
        logger.info("Parallel extraction completed: %s | total=%.2fs (workers=%s, pages/chunk=%s)",
                    name, time.time() - t0, par_workers, par_pages)
        return texts if texts else [("", f"{name}")]

    # Single-pass Docling conversion
    with NamedTemporaryFile(suffix=suffix or ".bin", delete=False) as tmp:
        tmp.write(payload)
        tmp_path = Path(tmp.name)

    try:
        t0 = time.time()
        converter = _get_docling_converter()
        result = converter.convert(tmp_path)
        logger.info("Docling.convert finished in %.2fs", time.time() - t0)

        document = getattr(result, "document", None)
        if document is None:
            raise RuntimeError(DOCLING_EMPTY_DOC_MSG)

        text: str | None = None
        for attr in ("export_to_text", "export_to_markdown", "export_to_plaintext"):
            exporter = getattr(document, attr, None)
            if callable(exporter):
                try:
                    t_exp0 = time.time()
                    candidate = exporter()
                    t_exp1 = time.time()
                    logger.info("Document.%s finished in %.2fs", attr, t_exp1 - t_exp0)
                except (RuntimeError, ValueError, TypeError, AttributeError) as exc:
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
    """Extract text units from an uploaded file for downstream chunking (always Docling when available)."""
    name = uploaded_file.name
    suffix = Path(name).suffix.lower()
    payload = uploaded_file.read() or b""

    if payload and DocumentConverter is not None:
        try:
            return _extract_with_docling(name, suffix, payload)
        except RuntimeError as exc:
            logger.info("Docling could not process %s: %s", name, exc)
        except (OSError, ValueError, TypeError) as exc:
            logger.warning("Docling processing crashed for %s", name, exc_info=exc)

    # Fallbacks only if Docling unavailable
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
                    "Failed to extract text from PDF page %s of %s", i + 1, name, exc_info=exc
                )
                page_text = ""
            if page_text.strip():
                units.append((page_text, f"{name}#page={i+1}"))
        return units or [("", f"{name}")]

    # Plaintext-ish
    try:
        text = payload.decode("utf-8", errors="ignore")
    except UnicodeDecodeError:
        text = payload.decode("latin-1", errors="ignore")
    return [(text, f"{name}")]


# ------------------ Context build & upsert ------------------

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

    # Allow env overrides for chunking (no caller changes needed)
    eff_chunk_size = int(os.environ.get("CHUNK_SIZE", str(chunk_size)))
    eff_chunk_overlap = int(os.environ.get("CHUNK_OVERLAP", str(chunk_overlap)))
    eff_chunk_overlap = max(0, min(eff_chunk_overlap, max(0, eff_chunk_size - 1)))

    # Chunk the input units
    for unit_text, unit_src in text_units:
        for chunk in chunk_text(unit_text, chunk_size=eff_chunk_size, chunk_overlap=eff_chunk_overlap):
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

    # Embed in batches - reduced default for 4GB RAM systems
    batch = int(os.environ.get("EMBED_BATCH", "64"))  # Reduced from 96 to 64 for limited RAM
    embeddings: list[list[float]] = []
    for i in range(0, len(all_chunks), batch):
        sub = all_chunks[i : i + batch]
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
        batch_vecs = vectors[i : i + 1000]
        index.upsert(vectors=batch_vecs, namespace=ns)
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
