import re
from typing import Dict, Any

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
    st.sidebar.header("🗂️ Your Conversations")

    # Initialize current_conv_id if missing
    if "current_conv_id" not in st.session_state:
        convs = list_conversations(db, username)
        if not convs:
            cid = create_conversation(db, username, "New chat")
            st.session_state["current_conv_id"] = cid
        else:
            st.session_state["current_conv_id"] = convs[0]["id"]

    if st.sidebar.button("➕ New chat"):
        cid = create_conversation(db, username, "New chat")
        st.session_state["current_conv_id"] = cid
        st.rerun()

    # List and select conversations
    convs = list_conversations(db, username)
    for c in convs:
        label = f"• {c.get('title','(untitled)')[:32]}"
        if st.sidebar.button(label, key=f"conv-{c['id']}"):
            st.session_state["current_conv_id"] = c["id"]
            st.rerun()

    with st.sidebar.expander("✏️ Rename / 🗑️ Delete"):
        new_title = st.text_input("New title")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Rename"):
                if st.session_state.get("current_conv_id"):
                    rename_conversation(
                        db,
                        st.session_state["current_conv_id"],
                        username,
                        new_title or "Untitled",
                    )
                    st.rerun()
        with col_b:
            if st.button("Delete conversation"):
                if st.session_state.get("current_conv_id"):
                    delete_conversation(db, st.session_state["current_conv_id"], username)
                    st.session_state["current_conv_id"] = None
                    st.rerun()


def chat_window(db, env_doc: Dict[str, Any], username: str):
    """
    Chat view that (1) reloads env from Mongo, (2) loads recent conversation history,
    (3) answers "what was my last question?" deterministically from Mongo,
    and (4) prefers RAG but falls back to general knowledge when context is weak.
    """
    # Always reload the freshest env from Mongo
    env_doc = load_env_doc(db) if db is not None else env_doc

    # Validate settings (includes optional RAG knobs with defaults in default_env)
    try:
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

    # Sidebar + title
    conversations_sidebar(db, username)
    st.title("💬 Chat")

    # Ensure we have an active conversation
    current_conv_id = st.session_state.get("current_conv_id")
    if not current_conv_id:
        current_conv_id = create_conversation(db, username, "New chat")
        st.session_state["current_conv_id"] = current_conv_id

    # Render existing messages
    existing_msgs = get_messages(db, current_conv_id, limit=200)
    for m in existing_msgs:
        with st.chat_message("user" if m["role"] == "user" else "assistant"):
            st.markdown(m["content"])

    # Chat input
    prompt = st.chat_input("Ask about your documents or anything...")
    if not prompt:
        return

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
                msg = f"Deine letzte Frage war:\n\n> {last_q}"
            else:
                msg = "Ich konnte keine vorherige Frage finden."
            st.markdown(msg)
            add_message(db, current_conv_id, username, "assistant", msg)
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
                        "- CONTEXT: for factual answers from documents (do not invent citations).\n\n"
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
                    msg = "Dazu habe ich keine ausreichend belegten Informationen im Kontext."
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
