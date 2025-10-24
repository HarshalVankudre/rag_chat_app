import streamlit as st
from typing import Dict, Any
from models.settings import AppSettings, DEFAULT_CHAT_SYSTEM_PROMPT
from config.env import load_env_doc, save_env_doc
from db.mongo import add_user, delete_user, COL_USERS, COL_CONV, COL_MSG
from rag.pinecone_utils import get_openai_client, get_pinecone_index, embed_texts
from .ingest import ingest_panel  # <-- add this import


def admin_dashboard(db, env_doc: Dict[str, Any]):
    st.header("🛡️ Admin Dashboard")
    tabs = st.tabs(["⚙️ Environment", "👥 Users", "📥 Ingest"])

    with tabs[0]:
        st.subheader("Environment Settings (stored in MongoDB)")
        with st.form("env_form"):
            env_doc["openai_api_key"] = st.text_input("OpenAI API Key", value=env_doc.get("openai_api_key",""), type="password")
            env_doc["openai_base_url"] = st.text_input("OpenAI Base URL (optional)", value=env_doc.get("openai_base_url",""))
            env_doc["openai_model"] = st.text_input("Chat model", value=env_doc.get("openai_model","gpt-4o-mini"))
            env_doc["embedding_model"] = st.text_input("Embedding model", value=env_doc.get("embedding_model","text-embedding-3-small"))

            env_doc["pinecone_api_key"] = st.text_input("Pinecone API Key", value=env_doc.get("pinecone_api_key",""), type="password")
            env_doc["pinecone_host"] = st.text_input("Pinecone Host (recommended)", value=env_doc.get("pinecone_host",""))
            env_doc["pinecone_index_name"] = st.text_input("Pinecone Index name (dev)", value=env_doc.get("pinecone_index_name",""))
            env_doc["pinecone_namespace"] = st.text_input("Default Namespace", value=env_doc.get("pinecone_namespace",""))

            env_doc["top_k"] = st.number_input("Top K", 1, 100, int(env_doc.get("top_k", 5)))
            env_doc["temperature"] = st.slider("Temperature", 0.0, 2.0, float(env_doc.get("temperature", 0.2)), 0.1)
            env_doc["max_context_chars"] = st.number_input("Max context chars", 500, 100000, int(env_doc.get("max_context_chars", 8000)), 500)
            env_doc["metadata_text_key"] = st.text_input("Metadata key: chunk text", value=env_doc.get("metadata_text_key","text"))
            env_doc["metadata_source_key"] = st.text_input("Metadata key: source", value=env_doc.get("metadata_source_key","source"))
            env_doc["system_prompt"] = st.text_area("System prompt", value=env_doc.get("system_prompt", DEFAULT_CHAT_SYSTEM_PROMPT), height=150)

            env_doc["mongo_uri"] = st.text_input("Mongo URI", value=env_doc.get("mongo_uri",""))
            env_doc["mongo_db"] = st.text_input("Mongo DB name", value=env_doc.get("mongo_db","rag_chat"))

            if st.form_submit_button("💾 Save settings"):
                save_env_doc(env_doc, db); st.success("Saved to Mongo."); st.rerun()

        if st.button("🔌 Test connections"):
            try:
                settings = AppSettings(
                    openai_api_key=env_doc.get("openai_api_key",""),
                    openai_base_url=(env_doc.get("openai_base_url") or None),
                    openai_model=env_doc.get("openai_model","gpt-4o-mini"),
                    embedding_model=env_doc.get("embedding_model","text-embedding-3-small"),
                    pinecone_api_key=env_doc.get("pinecone_api_key",""),
                    pinecone_index_name=(env_doc.get("pinecone_index_name") or None),
                    pinecone_host=(env_doc.get("pinecone_host") or None),
                    pinecone_namespace=(env_doc.get("pinecone_namespace") or None),
                    top_k=int(env_doc.get("top_k",5)),
                    temperature=float(env_doc.get("temperature",0.2)),
                    max_context_chars=int(env_doc.get("max_context_chars",8000)),
                    metadata_text_key=env_doc.get("metadata_text_key","text"),
                    metadata_source_key=env_doc.get("metadata_source_key","source"),
                    system_prompt=env_doc.get("system_prompt", DEFAULT_CHAT_SYSTEM_PROMPT),
                    mongo_uri=env_doc.get("mongo_uri"),
                    mongo_db=env_doc.get("mongo_db","rag_chat"),
                )
                client = get_openai_client(settings.openai_api_key, settings.openai_base_url)
                _ = embed_texts(client, ["ping"], settings.embedding_model)
                idx = get_pinecone_index(settings.pinecone_api_key, settings.pinecone_host, settings.pinecone_index_name)
                stats = idx.describe_index_stats()
                st.success("✅ OpenAI + Pinecone connections look good!")
                st.json(stats)
            except Exception as ex:
                st.error(f"Connection test failed: {ex}")

    with tabs[1]:
        st.subheader("User Management (MongoDB)")
        with st.form("add_user_form"):
            new_u = st.text_input("New username")
            new_p = st.text_input("New password", type="password")
            role = st.selectbox("Role", ["user", "admin"], index=0)
            if st.form_submit_button("➕ Add user"):
                if not new_u.strip() or not new_p.strip(): st.warning("Provide both username and password.")
                elif db["users"].find_one({"username": new_u}): st.warning("User already exists.")
                else:
                    msg = add_user(db, new_u, new_p, role)
                    st.success(f"User '{new_u}' added.") if msg == "ok" else st.error(f"Failed: {msg}")

        st.write("### Current users")
        for u in db["users"].find({}):
            cols = st.columns([3, 2, 2, 2])
            cols[0].markdown(f"**{u['username']}**")
            cols[1].write(u.get("role", "user"))
            cols[2].write(u.get("created_at", ""))
            if u["username"] != "admin":
                if cols[3].button("Delete", key=f"del-{u['username']}"):
                    msg = delete_user(db, u["username"])
                    if msg == "ok": st.success("Deleted."); st.rerun()
                    else: st.error(f"Failed: {msg}")
            else: cols[3].write("—")

    with tabs[2]:
        ingest_panel(db, env_doc) 