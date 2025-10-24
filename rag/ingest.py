import io, os, uuid, datetime as dt
from typing import Dict, List, Optional, Tuple
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None
try:
    import docx
except Exception:
    docx = None

from .pinecone_utils import embed_texts
from utils.ids import pinecone_safe_slug, sanitize_namespace
from utils.chunk import chunk_text

def extract_text_units(uploaded_file) -> List[Tuple[str, str]]:
    name = uploaded_file.name
    suffix = os.path.splitext(name)[1].lower()
    if suffix in [".txt",".md",".csv",".log"]:
        raw = uploaded_file.read()
        try: text = raw.decode("utf-8", errors="ignore")
        except Exception: text = raw.decode("latin-1", errors="ignore")
        return [(text, f"{name}")]
    if suffix == ".pdf":
        if PdfReader is None: raise RuntimeError("pypdf is not installed.")
        reader = PdfReader(io.BytesIO(uploaded_file.read())); units = []
        for i, page in enumerate(reader.pages):
            try: page_text = page.extract_text() or ""
            except Exception: page_text = ""
            if page_text.strip(): units.append((page_text, f"{name}#page={i+1}"))
        return units or [("", f"{name}")]
    if suffix == ".docx":
        if docx is None: raise RuntimeError("python-docx is not installed.")
        document = docx.Document(io.BytesIO(uploaded_file.read()))
        paras = [p.text for p in document.paragraphs if p.text and p.text.strip()]
        text = "\n".join(paras); return [(text, f"{name}")]
    raw = uploaded_file.read()
    try: text = raw.decode("utf-8", errors="ignore")
    except Exception: text = raw.decode("latin-1", errors="ignore")
    return [(text, f"{name}")]

def build_context(matches, text_key: str, source_key: str, max_chars: int):
    contexts, total, sources = [], 0, []
    for m in matches or []:
        md = m.get("metadata") or {}
        chunk = str(md.get(text_key, ""))
        if not chunk: continue
        if total + len(chunk) > max_chars: break
        total += len(chunk); contexts.append(chunk)
        src = md.get(source_key); sources.append({"id": m.get("id"), "score": m.get("score"), "source": str(src) if src else None})
    return {"context_text": "\n\n---\n\n".join(contexts), "sources": sources}

def upsert_chunks(client, index, *, embedding_model: str, filename: str, text_units: List[Tuple[str, str]], namespace: Optional[str], chunk_size: int, chunk_overlap: int, md_text_key: str, md_source_key: str) -> Dict:
    now = dt.datetime.utcnow().isoformat() + "Z"
    doc_id = f"{pinecone_safe_slug(filename)}-{uuid.uuid4().hex[:8]}"
    all_chunks, all_sources, vec_ids = [], [], []
    for unit_text, unit_src in text_units:
        for ch in chunk_text(unit_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap):
            all_chunks.append(ch); all_sources.append(unit_src)
    embeddings = []
    for i in range(0, len(all_chunks), 64):
        sub = all_chunks[i:i+64]; vecs = embed_texts(client, sub, embedding_model); embeddings.extend(vecs)
    def json_safe(meta: dict) -> dict:
        out = {}
        for k, v in meta.items():
            if isinstance(v, (str, int, float, bool)) or v is None: out[k] = v
            else: out[k] = str(v)
        return out
    vectors = []
    ns = sanitize_namespace(namespace or "__default__")
    for idx, (emb, src, ch_text) in enumerate(zip(embeddings, all_sources, all_chunks)):
        vid = f"{doc_id}::chunk-{idx:04d}"
        meta = json_safe({md_text_key: ch_text, md_source_key: src, "doc_id": doc_id, "filename": filename, "uploaded_at": now, "chunk_index": idx})
        vectors.append({"id": vid, "values": [float(x) for x in emb], "metadata": meta})
        vec_ids.append(vid)
    for i in range(0, len(vectors), 1000):
        index.upsert(vectors=vectors[i:i+1000], namespace=ns)
    return {"doc_id": doc_id, "filename": filename, "namespace": ns, "vector_count": len(vectors), "vector_ids": vec_ids, "uploaded_at": now}
