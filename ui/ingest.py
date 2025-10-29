"""Streamlit panel for document ingestion."""

import logging
from collections.abc import Mapping
from typing import Any

import streamlit as st
from pinecone.core.client.exceptions import PineconeException
from pydantic import ValidationError
from pymongo.database import Database
from pymongo.errors import PyMongoError

from config.env import load_env_doc
from db.mongo import COL_INGEST
from models.settings import DEFAULT_CHAT_SYSTEM_PROMPT, AppSettings
from rag.ingest import (
    extract_text_units,
    upsert_chunks,
)
from rag.pinecone_utils import (
    get_openai_client,
    get_pinecone_index,
)

logger = logging.getLogger(__name__)


def ingest_panel(
    db: Database | None, env_doc: dict[str, Any], lang: Mapping[str, str]
) -> None:
    """Render the ingestion workflow for uploading and indexing documents."""
    st.subheader(lang["ingest_title"])

    env_now = load_env_doc(db) if db is not None else env_doc
    try:
        settings = AppSettings(
            openai_api_key=env_now.get("openai_api_key", ""),
            openai_base_url=(env_now.get("openai_base_url") or None),
            openai_model=env_now.get("openai_model", "gpt-4o-mini"),
            embedding_model=env_now.get("embedding_model", "text-embedding-3-small"),
            pinecone_api_key=env_now.get("pinecone_api_key", ""),
            pinecone_index_name=(env_now.get("pinecone_index_name") or None),
            pinecone_host=(env_now.get("pinecone_host") or None),
            pinecone_namespace=(env_now.get("pinecone_namespace") or None),
            top_k=int(env_now.get("top_k", 5)),
            temperature=float(env_now.get("temperature", 0.2)),
            max_context_chars=int(env_now.get("max_context_chars", 8000)),
            metadata_text_key=env_now.get("metadata_text_key", "text"),
            metadata_source_key=env_now.get("metadata_source_key", "source"),
            system_prompt=env_now.get("system_prompt", DEFAULT_CHAT_SYSTEM_PROMPT),
            mongo_uri=env_now.get("mongo_uri"),
            mongo_db=env_now.get("mongo_db", "rag_chat"),
        )
    except ValidationError as exc:
        logger.exception("Ingestion panel failed to load settings")
        st.error(lang["ingest_error_env_not_configured"])
        st.exception(exc)
        return None

    with st.form("ingest_form"):
        files = st.file_uploader(
            lang["ingest_file_uploader"],
            type=["txt", "md", "pdf", "docx", "csv", "log"],
            accept_multiple_files=True,
        )
        ns_override = st.text_input(
            lang["ingest_namespace_label"], value=env_now.get("pinecone_namespace", "")
        )
        chunk_size = st.number_input(lang["ingest_chunk_size"], 300, 4000, 1000, 100)
        chunk_overlap = st.number_input(
            lang["ingest_chunk_overlap"], 0, 1000, 200, 50
        )

        submitted = st.form_submit_button(lang["ingest_submit_button"])

    if submitted:
        if not files:
            st.warning(lang["ingest_warn_no_files"])
        else:
            try:
                client = get_openai_client(
                    settings.openai_api_key, settings.openai_base_url
                )
                index = get_pinecone_index(
                    settings.pinecone_api_key,
                    settings.pinecone_host,
                    settings.pinecone_index_name,
                )
                ns = (ns_override or settings.pinecone_namespace or "__default__").strip()
                progress = st.progress(0)

                for i, f in enumerate(files, start=1):
                    with st.status(
                        lang["ingest_status_ingesting"].format(f_name=f.name),
                        expanded=True,
                    ):
                        try:
                            units = extract_text_units(f)
                            if not units or all(not u[0].strip() for u in units):
                                st.warning(lang["ingest_status_no_text"])
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
                                if rec.get("vector_count", 0) > 0 and db is not None:
                                    db[COL_INGEST].insert_one(rec)
                                    st.success(
                                        lang["ingest_status_success"].format(
                                            rec_vector_count=rec["vector_count"],
                                            rec_namespace=rec["namespace"],
                                            rec_doc_id=rec["doc_id"],
                                        )
                                    )
                                else:
                                    st.warning(
                                        f"No vectors were generated for {f.name}, "
                                        "skipping database entry."
                                    )
                        except (ValueError, RuntimeError, OSError) as file_exc:
                            logger.exception("Failed to ingest file %s", f.name)
                            st.error(f"{lang['ingest_status_failed']}: {file_exc}")
                    progress.progress(int(100 * i / len(files)))
                st.success(lang["ingest_status_done"])
            except (PineconeException, PyMongoError, RuntimeError, ValueError) as exc:
                logger.exception("Ingestion process failed")
                st.error(f"{lang['ingest_error_failed']}: {exc}")

    st.divider()
    st.markdown(lang["ingest_docs_header"])
    if db is not None:
        try:
            cursor = db[COL_INGEST].find({}).sort("uploaded_at", -1)
            found = False
            for rec in cursor:
                found = True
                expander_label = lang["ingest_docs_expander_label"].format(
                    filename=rec.get("filename", "(no name)"),
                    namespace=rec.get("namespace"),
                    vector_count=rec.get("vector_count"),
                )
                with st.expander(expander_label):
                    st.code(
                        {
                            "doc_id": rec.get("doc_id"),
                            "namespace": rec.get("namespace"),
                            "vector_count": rec.get("vector_count"),
                        }
                    )
                    col_a, col_b = st.columns([1, 3])
                    with col_a:
                        if st.button(
                            lang["ingest_delete_button"], key=f"delvec-{rec['_id']}"
                        ):
                            try:
                                index = get_pinecone_index(
                                    settings.pinecone_api_key,
                                    settings.pinecone_host,
                                    settings.pinecone_index_name,
                                )
                                vec_ids = rec.get("vector_ids") or []
                                if vec_ids:
                                    index.delete(
                                        ids=vec_ids, namespace=rec.get("namespace")
                                    )
                                db[COL_INGEST].delete_one({"_id": rec["_id"]})
                                st.success(lang["ingest_delete_success"])
                                st.experimental_rerun() if hasattr(
                                    st, "experimental_rerun"
                                ) else st.rerun()
                            except (PineconeException, PyMongoError, ValueError) as exc:
                                logger.exception("Failed to delete vectors")
                                st.error(f"{lang['ingest_delete_failed']}: {exc}")
                    with col_b:
                        st.caption(lang["ingest_delete_caption"])
            if not found:
                st.info(lang["ingest_docs_none"])
        except PyMongoError as exc:
            logger.exception("Failed to fetch ingested docs from Mongo")
            st.error(f"Failed to fetch ingested docs: {exc}")
