"""Streamlit chat UI implementation."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

import streamlit as st
from pydantic import ValidationError
from pymongo.database import Database

from config.env import load_env_doc
from db.mongo import (
    add_message,
    create_conversation,
    delete_conversation,
    get_messages,
    list_conversations,
    rename_conversation,
)
from models.settings import AppSettings
from rag.ingest import build_context
from rag.pinecone_utils import (
    embed_texts,
    get_openai_client,
    get_pinecone_index,
    retrieve_chunks,
    stream_chat_completion,
)

logger = logging.getLogger(__name__)

# ---- Configuration for memory window ----
MAX_TURNS = 25

# ---- Pattern for meta-question: "what was my last question?" (DE/EN) ----
LAST_Q_PATTERN = re.compile(
    r"(what\s+was\s+my\s+last\s+question\??|was\s+war\s+meine\s+letzte\s+frage\??)",
    re.IGNORECASE,
)


def find_last_user_question(history_docs: Iterable[dict[str, Any]]) -> str | None:
    """Return the previous user message within the supplied history."""
    if not history_docs:
        return None
    history_list = list(history_docs)
    for msg in reversed(history_list[:-1]):
        if msg.get("role") == "user":
            return (msg.get("content") or "").strip()
    return None


def conversations_sidebar(db: Database, username: str, lang: Mapping[str, str]) -> None:
    """Render the conversation list and controls in the sidebar."""
    with st.sidebar:
        st.header(lang["conv_header"])

        if st.button(lang["conv_new_chat"], use_container_width=True):
            cid = create_conversation(db, username, "New chat")
            st.session_state["current_conv_id"] = cid
            st.session_state.pop("editing_conv_id", None)
            st.session_state.pop("confirm_delete_id", None)
            st.rerun()

        st.divider()

        convs = list_conversations(db, username)

        if "current_conv_id" not in st.session_state:
            if not convs:
                cid = create_conversation(db, username, "New chat")
                st.session_state["current_conv_id"] = cid
                st.rerun()
            else:
                st.session_state["current_conv_id"] = convs[0]["id"]

        current_conv_id = st.session_state.get("current_conv_id")

        conv_ids = [c["id"] for c in convs]
        if current_conv_id not in conv_ids:
            current_conv_id = convs[0]["id"] if convs else None
            st.session_state["current_conv_id"] = current_conv_id

        editing_id = st.session_state.get("editing_conv_id")
        confirm_delete_id = st.session_state.get("confirm_delete_id")

        if not convs:
            st.info(lang["conv_no_convs"])

        for c in convs:
            c_id = c["id"]
            c_title = c.get("title", "Untitled")

            if editing_id == c_id:
                with st.form(key=f"form_edit_{c_id}"):
                    st.text_input(
                        lang["conv_rename_label"], value=c_title, key=f"rename_input_{c_id}"
                    )
                    col_a, col_b = st.columns(2)
                    if col_a.form_submit_button(
                        lang["conv_save"], use_container_width=True, type="primary"
                    ):
                        new_title = st.session_state[f"rename_input_{c_id}"]
                        if new_title.strip():
                            rename_conversation(
                                db, c_id, username, new_title.strip()
                            )
                            st.session_state.pop("editing_conv_id", None)
                            st.rerun()
                        else:
                            st.warning(lang["conv_warn_empty_title"])
                    if col_b.form_submit_button(
                        lang["conv_cancel"], use_container_width=True
                    ):
                        st.session_state.pop("editing_conv_id", None)
                        st.rerun()

            elif confirm_delete_id == c_id:
                st.error(lang["conv_confirm_delete"].format(c_title=c_title))
                col_a, col_b = st.columns(2)
                if col_a.button(
                    lang["conv_delete"].upper(),
                    key=f"confirm_del_btn_{c_id}",
                    use_container_width=True,
                    type="primary",
                ):
                    delete_conversation(db, c_id, username)
                    st.session_state.pop("confirm_delete_id", None)
                    if st.session_state.get("current_conv_id") == c_id:
                        st.session_state["current_conv_id"] = None
                    st.rerun()
                if col_b.button(
                    lang["conv_cancel"],
                    key=f"cancel_del_btn_{c_id}",
                    use_container_width=True,
                ):
                    st.session_state.pop("confirm_delete_id", None)
                    st.rerun()

            else:
                col1, col2 = st.columns([0.85, 0.15])
                with col1:
                    is_active = c_id == current_conv_id
                    if st.button(
                        f"• {c_title[:32]}",
                        key=f"select_{c_id}",
                        type=("primary" if is_active else "secondary"),
                        use_container_width=True,
                    ):
                        st.session_state["current_conv_id"] = c_id
                        st.session_state.pop("editing_conv_id", None)
                        st.session_state.pop("confirm_delete_id", None)
                        st.rerun()

                with col2, st.popover("...", use_container_width=False):
                    if st.button(
                        lang["conv_popover_rename"],
                        key=f"rename_pop_{c_id}",
                        use_container_width=True,
                    ):
                        st.session_state["editing_conv_id"] = c_id
                        st.session_state["current_conv_id"] = c_id
                        st.session_state.pop("confirm_delete_id", None)
                        st.rerun()

                    if st.button(
                        lang["conv_popover_delete"],
                        key=f"delete_pop_{c_id}",
                        use_container_width=True,
                        type="primary",
                    ):
                        st.session_state["confirm_delete_id"] = c_id
                        st.session_state["current_conv_id"] = c_id
                        st.session_state.pop("editing_conv_id", None)
                        st.rerun()
def render_chat_ui(
    db: Database,
    username: str,
    lang: Mapping[str, str],
    render_main_chat: bool = True,
    render_conv_sidebar: bool = True,
) -> str | None:
    """Render the chat sidebar and optionally the main message area."""
    if render_conv_sidebar:
        conversations_sidebar(db, username, lang)  # --- PASS lang ---

    current_conv_id = st.session_state.get("current_conv_id")

    if render_main_chat:
        st.title(lang["chat_title"])

        if not current_conv_id:
            st.info(lang["chat_no_conv_selected"])
            if st.button(lang["chat_reload_convs"]):
                st.rerun()
            return None

        existing_msgs = get_messages(db, current_conv_id, limit=200)
        for m in existing_msgs:
            with st.chat_message("user" if m["role"] == "user" else "assistant"):
                st.markdown(m["content"])

    return current_conv_id
def process_new_message(
    db: Database,
    env_doc: dict[str, Any],
    username: str,
    current_conv_id: str,
    prompt: str,
    lang: Mapping[str, str],
) -> None:
    """Process a user prompt, generate a response, and persist the interaction."""
    if not prompt or not current_conv_id:
        return None

    add_message(db, current_conv_id, username, "user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)  # User prompt is NOT translated

    history_docs = get_messages(db, current_conv_id, limit=400)
    history = [{"role": h["role"], "content": h["content"]} for h in history_docs]
    recent_history = history[-MAX_TURNS:]

    if LAST_Q_PATTERN.search(prompt):
        last_q = find_last_user_question(history_docs)
        with st.chat_message("assistant"):
            if last_q:
                msg = lang["chat_last_q_prefix"].format(last_q=last_q)
            else:
                msg = lang["chat_last_q_not_found"]
            st.markdown(msg)
            add_message(db, current_conv_id, username, "assistant", msg)
        return

    env_doc = load_env_doc(db) if db is not None else env_doc
    try:
        settings = AppSettings.from_env(env_doc)
    except ValidationError as exc:
        logger.exception("Environment settings are invalid")
        st.error(lang["chat_error_env_not_configured"])
        st.exception(exc)
        return None

    try:
        client = get_openai_client(
            settings.openai_api_key, settings.openai_base_url
        )

        qvec = embed_texts(client, [prompt], settings.embedding_model)[0]
        index = get_pinecone_index(
            settings.pinecone_api_key,
            settings.pinecone_host,
            settings.pinecone_index_name,
        )
        res = retrieve_chunks(index, qvec, settings.top_k, settings.pinecone_namespace)

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
                except (ValueError, TypeError):
                    pass  # Ignore non-floatable scores

        built = build_context(
            matches,
            settings.metadata_text_key,
            settings.metadata_source_key,
            settings.max_context_chars,
        )
        context_text = built["context_text"] or ""
        num_matches = len(matches)

        rag_sufficient = (
            num_matches >= settings.rag_min_matches
            and len(context_text) >= settings.rag_min_context_chars
            and (best_score is None or best_score >= settings.rag_min_score)
        )

        if rag_sufficient:
            messages = [{"role": "system", "content": settings.system_prompt}]
            messages.extend(recent_history)
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Use the information sources as follows:\n"
                        "- CHAT HISTORY: for conversation continuity.\n"
                        "- CONTEXT: for factual answers from documents (do not invent citations).\n"
                        f"CONTEXT:\n{context_text}\n\n"
                        f"USER QUESTION:\n{prompt}\n"
                    ),
                }
            )
            show_sources = True
        else:
            if not settings.allow_general_answers:
                with st.chat_message("assistant"):
                    msg = lang["chat_no_info_fallback"]
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

        with st.chat_message("assistant"):
            chunks = []

            def gen() -> Iterator[str]:
                for t in stream_chat_completion(
                    client, settings.openai_model, messages, settings.temperature
                ):
                    chunks.append(t)
                    yield t

            full = st.write_stream(gen())
            if not full:
                full = "".join(chunks)

            add_message(
                db, current_conv_id, username, "assistant", full
            )  # AI response is NOT translated

            if show_sources and built["sources"]:
                with st.expander(lang["chat_sources_expander"]):
                    for i, s in enumerate(built["sources"], start=1):
                        src_label = (
                            s.get("source") or lang["chat_sources_label_no_source"]
                        )
                        score = s.get("score")
                        try:
                            score_s = f"{float(score):.4f}" if score is not None else "n/a"
                        except (ValueError, TypeError):
                            score_s = str(score)
                        st.markdown(
                            f"**{i}.** `{src_label}` — score: `{score_s}` — id: `{s.get('id')}`"
                        )

    except Exception as exc:  # pragma: no cover - runtime safeguard
        logger.exception("Error processing new message")
        with st.chat_message("assistant"):
            st.error(f"{lang['chat_error']}: {exc}")

