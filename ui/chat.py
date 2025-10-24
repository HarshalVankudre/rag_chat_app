import re
from typing import Dict, Any, Optional

import streamlit as st
from pydantic import ValidationError

from models.settings import AppSettings, DEFAULT_CHAT_SYSTEM_PROMPT
from config.env import load_env_doc
from db.mongo import (
    list_conversations,
    create_conversation,
    rename_conversation,
    delete_conversation,
    add_message,
    get_messages,
)
from rag.pinecone_utils import (
    get_openai_client,
    get_pinecone_index,
    embed_texts,
    retrieve_chunks,
    stream_chat_completion,
)
from rag.ingest import build_context

# ---- Configuration for memory window ----
MAX_TURNS = 25  # Maximum conversation turns (user + assistant message pairs) to include in context

# ---- Pattern for meta-question: "what was my last question?" (DE/EN) ----
LAST_Q_PATTERN = re.compile(
    r"(what\s+was\s+my\s+last\s+question\??|was\s+war\s+meine\s+letzte\s+frage\??)",
    re.IGNORECASE,
)


def find_last_user_question(history_docs):
    """
    Return the most recent *user* message content BEFORE the current one.
    Assumes history_docs are sorted ascending by created_at.
    """
    if not history_docs:
        return None
    # Exclude the very last element (current prompt we just added)
    for msg in reversed(history_docs[:-1]):
        if msg.get("role") == "user":
            return (msg.get("content") or "").strip()
    return None


def conversations_sidebar(db, username: str):
    """
    Renders the conversation list and controls in the sidebar.
    """
    with st.sidebar:
        st.header("🗂️ Your Conversations")

        # --- New Chat Button ---
        if st.button("➕ New chat", use_container_width=True):
            cid = create_conversation(db, username, "New chat")
            st.session_state["current_conv_id"] = cid
            # Clear any action states
            st.session_state.pop("editing_conv_id", None)
            st.session_state.pop("confirm_delete_id", None)
            st.rerun()

        st.divider()

        # --- Get conversations and ensure one is selected ---
        convs = list_conversations(db, username)

        if "current_conv_id" not in st.session_state:
            if not convs:
                # If no convs, create one
                cid = create_conversation(db, username, "New chat")
                st.session_state["current_conv_id"] = cid
                st.rerun()
            else:
                st.session_state["current_conv_id"] = convs[0]["id"]

        current_conv_id = st.session_state.get("current_conv_id")

        # Handle case where current_conv_id might be stale (e.g., deleted)
        conv_ids = [c["id"] for c in convs]
        if current_conv_id not in conv_ids:
            current_conv_id = convs[0]["id"] if convs else None
            st.session_state["current_conv_id"] = current_conv_id

        # Get state for inline actions
        editing_id = st.session_state.get("editing_conv_id")
        confirm_delete_id = st.session_state.get("confirm_delete_id")

        # --- List conversations ---
        if not convs:
            st.info("No conversations yet.")

        for c in convs:
            c_id = c["id"]
            c_title = c.get('title', 'Untitled')

            if editing_id == c_id:
                # --- RENDER EDITING UI ---
                with st.form(key=f"form_edit_{c_id}"):
                    st.text_input("Rename:", value=c_title, key=f"rename_input_{c_id}")
                    col_a, col_b = st.columns(2)
                    if col_a.form_submit_button("Save", use_container_width=True, type="primary"):
                        new_title = st.session_state[f"rename_input_{c_id}"]
                        if new_title.strip():
                            rename_conversation(db, c_id, username, new_title.strip())
                            st.session_state.pop("editing_conv_id", None)
                            st.rerun()
                        else:
                            st.warning("Title cannot be empty")
                    if col_b.form_submit_button("Cancel", use_container_width=True):
                        st.session_state.pop("editing_conv_id", None)
                        st.rerun()

            elif confirm_delete_id == c_id:
                # --- RENDER CONFIRM DELETE UI ---
                st.error(f"Delete '{c_title}'?")
                col_a, col_b = st.columns(2)
                if col_a.button("DELETE", key=f"confirm_del_btn_{c_id}", use_container_width=True, type="primary"):
                    delete_conversation(db, c_id, username)
                    st.session_state.pop("confirm_delete_id", None)
                    if st.session_state.get("current_conv_id") == c_id:
                        st.session_state["current_conv_id"] = None  # Clear it
                    st.rerun()
                if col_b.button("Cancel", key=f"cancel_del_btn_{c_id}", use_container_width=True):
                    st.session_state.pop("confirm_delete_id", None)
                    st.rerun()

            else:
                # --- RENDER NORMAL ROW UI (Title + Popover Menu) ---
                col1, col2 = st.columns([0.85, 0.15])
                with col1:
                    is_active = (c_id == current_conv_id)
                    if st.button(
                            f"• {c_title[:32]}",
                            key=f"select_{c_id}",
                            type=("primary" if is_active else "secondary"),
                            use_container_width=True
                    ):
                        st.session_state["current_conv_id"] = c_id
                        # Clear other states just in case
                        st.session_state.pop("editing_conv_id", None)
                        st.session_state.pop("confirm_delete_id", None)
                        st.rerun()

                with col2:
                    # This is the "..." button that opens the menu
                    with st.popover("...", use_container_width=False):
                        if st.button("Rename", key=f"rename_pop_{c_id}", use_container_width=True):
                            st.session_state["editing_conv_id"] = c_id
                            st.session_state["current_conv_id"] = c_id  # Select it
                            st.session_state.pop("confirm_delete_id", None)
                            st.rerun()

                        if st.button("Delete", key=f"delete_pop_{c_id}", use_container_width=True, type="primary"):
                            st.session_state["confirm_delete_id"] = c_id
                            st.session_state["current_conv_id"] = c_id  # Select it
                            st.session_state.pop("editing_conv_id", None)
                            st.rerun()


def render_chat_ui(db, username: str) -> Optional[str]:
    """
    Renders the chat title and existing messages for the current conversation.
    Also renders the sidebar.
    Returns the current_conv_id.
    """
    # --- Render sidebar with conversation list ---
    conversations_sidebar(db, username)

    # --- Render main chat UI ---
    st.title("💬 Chat")

    # Ensure we have an active conversation
    current_conv_id = st.session_state.get("current_conv_id")
    if not current_conv_id:
        st.info("Select a conversation or start a new one.")
        if st.button("Reload conversations"):
            st.rerun()
        return None

    # Render existing messages
    existing_msgs = get_messages(db, current_conv_id, limit=200)
    for m in existing_msgs:
        with st.chat_message("user" if m["role"] == "user" else "assistant"):
            st.markdown(m["content"])

    return current_conv_id


def process_new_message(db, env_doc: Dict[str, Any], username: str, current_conv_id: str, prompt: str):
    """
    Processes a new user prompt, performs RAG, gets a response, and saves to DB.
    """
    if not prompt or not current_conv_id:
        return  # Should not happen if called correctly

    # Save and render the new user message
    add_message(db, current_conv_id, username, "user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    # Reload history (now includes the message we just added)
    history_docs = get_messages(db, current_conv_id, limit=400)
    history = [{"role": h["role"], "content": h["content"]} for h in history_docs]
    recent_history = history[-MAX_TURNS:]

    # Deterministic fallback for "what was my last question?"
    if LAST_Q_PATTERN.search(prompt):
        last_q = find_last_user_question(history_docs)
        with st.chat_message("assistant"):
            if last_q:
                msg = f"Your last question was:\n\n> {last_q}"
            else:
                msg = "I could not find a previous question."
            st.markdown(msg)
            add_message(db, current_conv_id, username, "assistant", msg)
        return

    # Always reload the freshest env from Mongo
    env_doc = load_env_doc(db) if db is not None else env_doc
    try:
        # Validate settings
        settings = AppSettings(
            openai_api_key=env_doc.get("openai_api_key", ""),
            openai_base_url=(env_doc.get("openai_base_url") or None),
            openai_model=env_doc.get("openai_model", "gpt-4o-mini"),
            embedding_model=env_doc.get("embedding_model", "text-embedding-3-small"),
            pinecone_api_key=env_doc.get("pinecone_api_key", ""),
            pinecone_index_name=(env_doc.get("pinecone_index_name") or None),
            pinecone_host=(env_doc.get("pinecone_host") or None),
            pinecone_namespace=(env_doc.get("pinecone_namespace") or None),
            top_k=int(env_doc.get("top_k", 5)),
            temperature=float(env_doc.get("temperature", 0.2)),
            max_context_chars=int(env_doc.get("max_context_chars", 8000)),
            metadata_text_key=env_doc.get("metadata_text_key", "text"),
            metadata_source_key=env_doc.get("metadata_source_key", "source"),
            system_prompt=env_doc.get("system_prompt", DEFAULT_CHAT_SYSTEM_PROMPT),
            mongo_uri=env_doc.get("mongo_uri"),
            mongo_db=env_doc.get("mongo_db", "rag_chat"),
            # NEW knobs (present in default_env; Pydantic will accept them via defaults if missing)
            allow_general_answers=env_doc.get("allow_general_answers", True),
            rag_min_context_chars=int(env_doc.get("rag_min_context_chars", 600)),
            rag_min_matches=int(env_doc.get("rag_min_matches", 1)),
            rag_min_score=float(env_doc.get("rag_min_score", 0.0)),
        )
    except ValidationError as e:
        st.error("Environment not configured. Open Admin → Environment and save your keys.")
        st.exception(e)
        return

    try:
        # Clients
        client = get_openai_client(settings.openai_api_key, settings.openai_base_url)

        # -------- Retrieval (RAG) --------
        qvec = embed_texts(client, [prompt], settings.embedding_model)[0]
        index = get_pinecone_index(
            settings.pinecone_api_key, settings.pinecone_host, settings.pinecone_index_name
        )
        res = retrieve_chunks(index, qvec, settings.top_k, settings.pinecone_namespace)

        # Normalize matches and collect best score
        raw_matches = getattr(res, "matches", None) or []
        matches = []
        best_score = None
        for m in raw_matches:
            if isinstance(m, dict):
                m_norm = m
            else:
                m_norm = {
                    "id": getattr(m, "id", None),
                    "score": getattr(m, "score", None),
                    "metadata": getattr(m, "metadata", {}),
                }
            matches.append(m_norm)
            sc = m_norm.get("score")
            if sc is not None:
                try:
                    scf = float(sc)
                    best_score = scf if best_score is None else max(best_score, scf)
                except Exception:
                    pass

        built = build_context(
            matches,
            settings.metadata_text_key,
            settings.metadata_source_key,
            settings.max_context_chars,
        )
        context_text = built["context_text"] or ""
        num_matches = len(matches)

        # Decide if RAG context is sufficient
        rag_sufficient = (
                num_matches >= settings.rag_min_matches
                and len(context_text) >= settings.rag_min_context_chars
                and (best_score is None or best_score >= settings.rag_min_score)
        )

        # Build messages for the model
        if rag_sufficient:
            # ✅ RAG path (grounded)
            messages = [{"role": "system", "content": settings.system_prompt}]
            messages.extend(recent_history)
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Use the information sources as follows:\n"
                        "- CHAT HISTORY: for conversation continuity.\n"
                        "- CONTEXT: for factual answers from documents (do not invent citations).\G\n"
                        f"CONTEXT:\n{context_text}\n\n"
                        f"USER QUESTION:\n{prompt}\n"
                    ),
                }
            )
            show_sources = True
        else:
            # 🤝 General knowledge fallback (still uses history)
            if not settings.allow_general_answers:
                with st.chat_message("assistant"):
                    msg = "I do not have enough information in the provided documents to answer that."
                    st.markdown(msg)
                    add_message(db, current_conv_id, username, "assistant", msg)
                return

            messages = [{"role": "system", "content": settings.system_prompt}]
            messages.extend(recent_history)
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "The retrieved CONTEXT is insufficient or empty. "
                        "Answer the USER QUESTION from general knowledge. "
                        "Do NOT invent citations.\n\n"
                        f"USER QUESTION:\n{prompt}\n"
                    ),
                }
            )
            show_sources = False

        # -------- Stream answer --------
        with st.chat_message("assistant"):
            chunks = []

            def gen():
                for t in stream_chat_completion(
                        client, settings.openai_model, messages, settings.temperature
                ):
                    chunks.append(t)
                    yield t

            full = st.write_stream(gen())
            if not full:
                full = "".join(chunks)

            add_message(db, current_conv_id, username, "assistant", full)

            # Only show sources on RAG path
            if show_sources and built["sources"]:
                with st.expander("🔎 Sources used"):
                    for i, s in enumerate(built["sources"], start=1):
                        src_label = s.get("source") or "(no source metadata)"
                        score = s.get("score")
                        try:
                            score_s = f"{float(score):.4f}" if score is not None else "n/a"
                        except Exception:
                            score_s = str(score)
                        st.markdown(
                            f"**{i}.** `{src_label}` — score: `{score_s}` — id: `{s.get('id')}`"
                        )

    except Exception as e:
        with st.chat_message("assistant"):
            st.error(f"Error: {e}")