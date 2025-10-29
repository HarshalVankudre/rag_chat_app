import logging
from typing import Any, Dict

import streamlit as st
from pydantic import ValidationError

from config.env import save_env_doc
from db.mongo import add_user, delete_user
from models.settings import DEFAULT_CHAT_SYSTEM_PROMPT, AppSettings
from rag.pinecone_utils import embed_texts, get_openai_client, get_pinecone_index

from .ingest import ingest_panel

logger = logging.getLogger(__name__)


def admin_dashboard(db, env_doc: Dict[str, Any], lang: dict):
    st.header(lang["admin_dashboard_title"])

    # --- Sidebar navigation for admin sections ---
    admin_view_options = [lang["admin_env"], lang["admin_users"], lang["admin_ingest"]]
    admin_sub_view = st.sidebar.radio(
        "Admin Panel",
        admin_view_options,
        key="admin_sub_view",
        label_visibility="collapsed",
    )
    st.sidebar.divider()
    # --------------------------------------------------

    if admin_sub_view == lang["admin_env"]:
        st.subheader(lang["admin_env_subheader"])
        with st.form("env_form"):
            # ... (all existing OpenAI and Pinecone fields) ...
            env_doc["openai_api_key"] = st.text_input(
                lang["admin_env_openai_api_key"],
                value=env_doc.get("openai_api_key", ""),
                type="password",
            )
            env_doc["openai_base_url"] = st.text_input(
                lang["admin_env_openai_base_url"],
                value=env_doc.get("openai_base_url", ""),
            )
            env_doc["openai_model"] = st.text_input(
                lang["admin_env_openai_model"],
                value=env_doc.get("openai_model", "gpt-4o-mini"),
            )
            env_doc["embedding_model"] = st.text_input(
                lang["admin_env_embedding_model"],
                value=env_doc.get("embedding_model", "text-embedding-3-small"),
            )

            env_doc["pinecone_api_key"] = st.text_input(
                lang["admin_env_pinecone_api_key"],
                value=env_doc.get("pinecone_api_key", ""),
                type="password",
            )
            env_doc["pinecone_host"] = st.text_input(
                lang["admin_env_pinecone_host"], value=env_doc.get("pinecone_host", "")
            )
            env_doc["pinecone_index_name"] = st.text_input(
                lang["admin_env_pinecone_index_name"],
                value=env_doc.get("pinecone_index_name", ""),
            )
            env_doc["pinecone_namespace"] = st.text_input(
                lang["admin_env_pinecone_namespace"],
                value=env_doc.get("pinecone_namespace", ""),
            )

            # ... (all existing RAG fields) ...
            env_doc["top_k"] = st.number_input(
                lang["admin_env_top_k"], 1, 100, int(env_doc.get("top_k", 5))
            )
            env_doc["temperature"] = st.slider(
                lang["admin_env_temperature"],
                0.0,
                2.0,
                float(env_doc.get("temperature", 0.2)),
                0.1,
            )
            env_doc["max_context_chars"] = st.number_input(
                lang["admin_env_max_context_chars"],
                500,
                100000,
                int(env_doc.get("max_context_chars", 8000)),
                500,
            )
            env_doc["metadata_text_key"] = st.text_input(
                lang["admin_env_metadata_text_key"],
                value=env_doc.get("metadata_text_key", "text"),
            )
            env_doc["metadata_source_key"] = st.text_input(
                lang["admin_env_metadata_source_key"],
                value=env_doc.get("metadata_source_key", "source"),
            )
            env_doc["system_prompt"] = st.text_area(
                lang["admin_env_system_prompt"],
                value=env_doc.get("system_prompt", DEFAULT_CHAT_SYSTEM_PROMPT),
                height=150,
            )

            # ... (all existing Mongo fields) ...
            env_doc["mongo_uri"] = st.text_input(
                lang["admin_env_mongo_uri"], value=env_doc.get("mongo_uri", "")
            )
            env_doc["mongo_db"] = st.text_input(
                lang["admin_env_mongo_db"], value=env_doc.get("mongo_db", "rag_chat")
            )

            # --- NEW FIELDS ADDED HERE ---
            st.divider()
            env_doc["auth_secret_key"] = st.text_input(
                lang["admin_env_auth_secret_key"],
                value=env_doc.get("auth_secret_key", ""),
                type="password",
            )
            env_doc["auth_cookie_expiry_days"] = st.number_input(
                lang["admin_env_auth_cookie_expiry"],
                1,
                365,
                int(env_doc.get("auth_cookie_expiry_days", 30)),
            )
            # ---------------------------

            if st.form_submit_button(lang["admin_env_save_button"]):
                save_env_doc(env_doc, db)
                st.success(lang["admin_env_save_success"])
                st.rerun()

        # ... (rest of the file is unchanged) ...
        if st.button(lang["admin_env_test_button"]):
            try:
                settings = AppSettings(
                    openai_api_key=env_doc.get("openai_api_key", ""),
                    openai_base_url=(env_doc.get("openai_base_url") or None),
                    openai_model=env_doc.get("openai_model", "gpt-4o-mini"),
                    embedding_model=env_doc.get(
                        "embedding_model", "text-embedding-3-small"
                    ),
                    pinecone_api_key=env_doc.get("pinecone_api_key", ""),
                    pinecone_index_name=(env_doc.get("pinecone_index_name") or None),
                    pinecone_host=(env_doc.get("pinecone_host") or None),
                    pinecone_namespace=(env_doc.get("pinecone_namespace") or None),
                    top_k=int(env_doc.get("top_k", 5)),
                    temperature=float(env_doc.get("temperature", 0.2)),
                    max_context_chars=int(env_doc.get("max_context_chars", 8000)),
                    metadata_text_key=env_doc.get("metadata_text_key", "text"),
                    metadata_source_key=env_doc.get("metadata_source_key", "source"),
                    system_prompt=env_doc.get(
                        "system_prompt", DEFAULT_CHAT_SYSTEM_PROMPT
                    ),
                    mongo_uri=env_doc.get("mongo_uri"),
                    mongo_db=env_doc.get("mongo_db", "rag_chat"),
                    # --- ADDED NEW AUTH FIELDS TO VALIDATION ---
                    auth_secret_key=env_doc.get("auth_secret_key"),
                    auth_cookie_expiry_days=int(
                        env_doc.get("auth_cookie_expiry_days")
                    ),
                )
                client = get_openai_client(
                    settings.openai_api_key, settings.openai_base_url
                )
                _ = embed_texts(client, ["ping"], settings.embedding_model)
                idx = get_pinecone_index(
                    settings.pinecone_api_key,
                    settings.pinecone_host,
                    settings.pinecone_index_name,
                )
                stats = idx.describe_index_stats()
                st.success(lang["admin_env_test_success"])
                st.json(stats)
            except (ValidationError, Exception) as ex:
                logger.exception(f"Connection test failed: {ex}")
                st.error(f"{lang['admin_env_test_failed']}: {ex}")

    elif admin_sub_view == lang["admin_users"]:
        # ... (rest of the file is unchanged) ...
        st.subheader(lang["admin_users_subheader"])
        with st.form("add_user_form"):
            new_u = st.text_input(lang["admin_users_form_username"])
            new_e = st.text_input(lang["admin_users_form_email"])
            new_p = st.text_input(lang["admin_users_form_password"], type="password")
            role = st.selectbox(
                lang["admin_users_form_role"],
                [
                    lang["admin_users_form_role_user"],
                    lang["admin_users_form_role_admin"],
                ],
                index=0,
            )

            if st.form_submit_button(lang["admin_users_form_add_button"]):
                if not new_u.strip() or not new_p.strip() or not new_e.strip():
                    st.warning(lang["admin_users_form_warn_all_fields"])
                elif db["users"].find_one({"username": new_u}):
                    st.warning(lang["admin_users_form_warn_user_exists"])
                else:
                    role_key = (
                        "user" if role == lang["admin_users_form_role_user"] else "admin"
                    )
                    msg = add_user(db, new_u.strip(), new_p.strip(), new_e.strip(), role_key)
                    if msg == "ok":
                        st.success(lang["admin_users_form_add_success"].format(new_u=new_u))
                    else:
                        st.error(f"{lang['admin_users_form_add_failed']}: {msg}")

        st.write(lang["admin_users_list_header"])
        cols = st.columns([2, 3, 1, 2, 1])
        cols[0].markdown(lang["admin_users_col_username"])
        cols[1].markdown(lang["admin_users_col_email"])
        cols[2].markdown(lang["admin_users_col_role"])
        cols[3].markdown(lang["admin_users_col_created"])
        cols[4].markdown(lang["admin_users_col_action"])

        for u in db["users"].find({}).sort("username"):
            cols = st.columns([2, 3, 1, 2, 1])
            cols[0].write(f"{u['username']}")
            cols[1].write(u.get("email", "—"))
            cols[2].write(u.get("role", "user"))
            cols[3].caption(u.get("created_at", "")[:10] if u.get("created_at") else "—")

            if u["username"] != "admin":
                if cols[4].button(
                    lang["admin_users_list_delete_button"],
                    key=f"del-{u['username']}",
                    type="secondary",
                ):
                    msg = delete_user(db, u["username"])
                    if msg == "ok":
                        st.success(lang["admin_users_list_delete_success"])
                        st.rerun()
                    else:
                        st.error(f"{lang['admin_users_form_add_failed']}: {msg}")
            else:
                cols[4].write("—")

    elif admin_sub_view == lang["admin_ingest"]:
        ingest_panel(db, env_doc, lang)

