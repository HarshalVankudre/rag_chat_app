# ui/ingest.py
import streamlit as st
from typing import Dict, Any
from pydantic import ValidationError

from models.settings import AppSettings, DEFAULT_CHAT_SYSTEM_PROMPT
from config.env import load_env_doc
from db.mongo import COL_INGEST
from rag.pinecone_utils import (
    get_openai_client,
    get_pinecone_index,
)
from rag.ingest import (
    extract_text_units,
    upsert_chunks,
)

def ingest_panel(db, env_doc: Dict[str, Any]):
    st.subheader("📥 Upload & Ingest Documents to Pinecone")

    # Always reload latest env from Mongo so we don't use stale settings
    env_now = load_env_doc(db) if db is not None else env_doc
    try:
        settings = AppSettings(
            openai_api_key=env_now.get("openai_api_key",""),
            openai_base_url=(env_now.get("openai_base_url") or None),
            openai_model=env_now.get("openai_model","gpt-4o-mini"),
            embedding_model=env_now.get("embedding_model","text-embedding-3-small"),
            pinecone_api_key=env_now.get("pinecone_api_key",""),
            pinecone_index_name=(env_now.get("pinecone_index_name") or None),
            pinecone_host=(env_now.get("pinecone_host") or None),
            pinecone_namespace=(env_now.get("pinecone_namespace") or None),
            top_k=int(env_now.get("top_k",5)),
            temperature=float(env_now.get("temperature",0.2)),
            max_context_chars=int(env_now.get("max_context_chars",8000)),
            metadata_text_key=env_now.get("metadata_text_key","text"),
            metadata_source_key=env_now.get("metadata_source_key","source"),
            system_prompt=env_now.get("system_prompt", DEFAULT_CHAT_SYSTEM_PROMPT),
            mongo_uri=env_now.get("mongo_uri"),
            mongo_db=env_now.get("mongo_db","rag_chat"),
        )
    except ValidationError as e:
        st.error("Environment not configured. Open Admin → Environment and save your keys.")
        st.exception(e)
        return

    with st.form("ingest_form"):
        files = st.file_uploader(
            "Choose documents",
            type=["txt","md","pdf","docx","csv","log"],
            accept_multiple_files=True
        )
        ns_override = st.text_input(
            "Namespace (leave blank to use default)",
            value=env_now.get("pinecone_namespace","")
        )
        chunk_size = st.number_input("Chunk size", 300, 4000, 1000, 100)
        chunk_overlap = st.number_input("Chunk overlap", 0, 1000, 200, 50)

        submitted = st.form_submit_button("🚀 Embed & Upsert")

    if submitted:
        if not files:
            st.warning("Select files first.")
        else:
            try:
                client = get_openai_client(settings.openai_api_key, settings.openai_base_url)
                index = get_pinecone_index(settings.pinecone_api_key, settings.pinecone_host, settings.pinecone_index_name)
                ns = (ns_override or settings.pinecone_namespace or "__default__").strip()
                progress = st.progress(0)

                for i, f in enumerate(files, start=1):
                    with st.status(f"Ingesting **{f.name}** …", expanded=True):
                        try:
                            units = extract_text_units(f)
                            if not units or all(not u[0].strip() for u in units):
                                st.warning("No text found; skipping.")
                            else:
                                rec = upsert_chunks(
                                    client,
                                    index,
                                    embedding_model=settings.embedding_model,
                                    filename=f.name,
                                    text_units=units,
                                    namespace=ns,
                                    chunk_size=int(chunk_size),
                                    chunk_overlap=int(chunk_overlap),
                                    md_text_key=settings.metadata_text_key,
                                    md_source_key=settings.metadata_source_key,
                                )
                                db[COL_INGEST].insert_one(rec)
                                st.success(
                                    f"Uploaded {rec['vector_count']} chunks to `{rec['namespace']}` "
                                    f"(doc_id: `{rec['doc_id']}`)"
                                )
                        except Exception as exf:
                            st.error(f"Failed: {exf}")
                    progress.progress(int(100 * i / len(files)))
                st.success("✅ Done")
            except Exception as ex:
                st.error(f"Ingestion failed: {ex}")

    st.divider()
    st.markdown("### 📚 Ingested documents")
    # Simple list + delete controls
    if db is not None:
        cursor = db[COL_INGEST].find({}).sort("uploaded_at", -1)
        found = False
        for rec in cursor:
            found = True
            with st.expander(f"📄 {rec.get('filename','(no name)')} • ns={rec.get('namespace')} • vectors={rec.get('vector_count')}"):
                st.code({"doc_id": rec.get("doc_id"), "namespace": rec.get("namespace"), "vector_count": rec.get("vector_count")})
                col_a, col_b = st.columns([1, 3])
                with col_a:
                    if st.button("🗑️ Delete vectors", key=f"delvec-{rec['_id']}"):
                        try:
                            index = get_pinecone_index(settings.pinecone_api_key, settings.pinecone_host, settings.pinecone_index_name)
                            # Delete by stored vector IDs in this manifest entry
                            vec_ids = rec.get("vector_ids") or []
                            index.delete(ids=vec_ids, namespace=rec.get("namespace"))
                            # Optionally also remove the manifest row:
                            db[COL_INGEST].delete_one({"_id": rec["_id"]})
                            st.success("Vectors deleted and manifest removed.")
                            st.experimental_rerun() if hasattr(st, "experimental_rerun") else st.rerun()
                        except Exception as exd:
                            st.error(f"Delete failed: {exd}")
                with col_b:
                    st.caption("Deletes the vectors from Pinecone and removes this manifest entry.")
        if not found:
            st.info("No ingested documents yet. Upload some above.")
