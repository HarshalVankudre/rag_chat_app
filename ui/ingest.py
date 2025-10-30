"""Streamlit panel for document ingestion."""

import logging
from collections.abc import Mapping
from typing import Any

import streamlit as st
from pinecone.exceptions import PineconeException
from pydantic import ValidationError
from pymongo.database import Database
from pymongo.errors import PyMongoError

from config.env import load_env_doc
from db.mongo import COL_INGEST
from models.settings import AppSettings
from rag.ingest import (
    extract_text_units,
    upsert_chunks,
)
from rag.pinecone_utils import (
    get_openai_client,
    get_pinecone_index,
)

logger = logging.getLogger(__name__)


def ingest_panel(db: Database | None, env_doc: dict[str, Any], lang: Mapping[str, str]) -> None:
    """Render the ingestion workflow for uploading and indexing documents."""
    st.subheader(lang["ingest_title"])

    env_now = load_env_doc(db) if db is not None else env_doc
    try:
        settings = AppSettings.from_env(env_now)
    except (ValidationError, ValueError) as exc:
        logger.exception("Ingestion panel failed to load settings")
        st.error(lang["ingest_error_env_not_configured"])
        st.exception(exc)
        return None

    available_indexes = settings.pinecone_index_names
    index_hint = ", ".join(available_indexes)
    default_index_value = settings.pinecone_index_name or ""

    with st.form("ingest_form"):
        files = st.file_uploader(
            lang["ingest_file_uploader"],
            type=["txt", "md", "pdf", "docx", "csv", "log"],
            accept_multiple_files=True,
        )
        ns_override = st.text_input(
            lang["ingest_namespace_label"], value=env_now.get("pinecone_namespace", "")
        )
        index_help = (
            lang["ingest_target_index_help"].format(index_list=index_hint)
            if index_hint
            else lang["ingest_target_index_help_empty"]
        )
        target_index_input = st.text_input(
            lang["ingest_target_index_label"],
            value=default_index_value,
            help=(lang["ingest_target_index_help_host"] if settings.pinecone_host else index_help),
            disabled=bool(settings.pinecone_host),
        )
        chunk_size = st.number_input(lang["ingest_chunk_size"], 300, 4000, 1000, 100)
        chunk_overlap = st.number_input(lang["ingest_chunk_overlap"], 0, 1000, 200, 50)

        submitted = st.form_submit_button(lang["ingest_submit_button"])

    if submitted:
        if not files:
            st.warning(lang["ingest_warn_no_files"])
        else:
            target_index = (target_index_input or default_index_value or "").strip() or None
            try:
                client = get_openai_client(settings.openai_api_key, settings.openai_base_url)
                index = get_pinecone_index(
                    settings.pinecone_api_key,
                    settings.pinecone_host,
                    target_index,
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
                                    index_name=target_index,
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
                        if st.button(lang["ingest_delete_button"], key=f"delvec-{rec['_id']}"):
                            try:
                                index = get_pinecone_index(
                                    settings.pinecone_api_key,
                                    settings.pinecone_host,
                                    rec.get("index_name") or settings.pinecone_index_name,
                                )
                                vec_ids = rec.get("vector_ids") or []
                                if vec_ids:
                                    index.delete(ids=vec_ids, namespace=rec.get("namespace"))
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
