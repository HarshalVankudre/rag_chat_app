"""Main Streamlit application file for the RAG Chat App."""

from __future__ import annotations

import logging
from typing import Any

import streamlit as st
import streamlit_authenticator as stauth
from pymongo import errors as mongo_errors
from pymongo.database import Database

from config.env import save_env_doc
from config.i18n import get_lang
from config.logging import setup_logging
from db.mongo import (
    COL_USERS,
    ensure_indexes,
    fetch_all_users_for_auth,
    get_mongo,
    seed_admin_if_empty,
)
from services.bootstrap import AppBootstrapResult, bootstrap_application
from ui.admin import admin_dashboard
from ui.chat import process_new_message, render_chat_ui
from ui.password import change_password_page

APP_TITLE = "💬 RükoGPT"
logger = logging.getLogger(__name__)


def configure_page() -> None:
    """Configure the Streamlit page before rendering any UI."""

    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def initialize_language() -> dict[str, Any]:
    """Persist and resolve the active UI language."""

    if "lang_code" not in st.session_state:
        st.session_state["lang_code"] = "en"

    with st.sidebar:
        st.selectbox(
            label="Language / Sprache",
            options=[("en", "🇬🇧 English"), ("de", "🇩🇪 Deutsch")],
            format_func=lambda x: x[1],
            key="lang_select_key",
            on_change=lambda: st.session_state.update(
                lang_code=st.session_state.lang_select_key[0],
            ),
        )

    return get_lang(st.session_state.get("lang_code", "en"))


def setup_screen(lang_dict: dict[str, Any]) -> None:
    """First-run setup to enter Mongo URI/DB."""

    st.title(lang_dict["setup_title"])
    st.write(lang_dict["setup_descr"])

    uri = st.text_input(
        lang_dict["setup_mongo_uri"],
        value="",
        type="password",
        placeholder=lang_dict["setup_mongo_uri_placeholder"],
        help=lang_dict["setup_mongo_uri_help"],
    )
    db_name = st.text_input(
        lang_dict["setup_mongo_db"],
        value="rag_chat",
        placeholder="rag_chat",
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button(lang_dict["setup_button"], use_container_width=True):
            test_uri = uri.strip()
            if not test_uri:
                st.error(lang_dict["setup_error_no_uri"])
                return

            client = get_mongo(test_uri)
            if client is None:
                st.error(lang_dict["setup_error_connect_failed"])
                return

            save_env_doc(
                {"mongo_uri": test_uri, "mongo_db": db_name.strip() or "rag_chat"},
                db=None,
            )

            db = client[(db_name.strip() or "rag_chat")]
            try:
                ensure_indexes(db)
                seed_status = seed_admin_if_empty(db)
                logger.info("Setup seeding status: %s", seed_status)
                save_env_doc({"mongo_uri": test_uri, "mongo_db": db.name}, db=db)
            except mongo_errors.PyMongoError:
                logger.exception("Error during setup bootstrap")
                st.error(lang_dict["setup_error_connect_failed"])
                return

            st.success(lang_dict["setup_success"])
            st.rerun()
    with col2:
        st.caption(lang_dict["setup_caption"])


@st.cache_resource(ttl=600)
def get_auth_credentials(_db: Database) -> dict[str, Any]:
    """Wrap the fetch call in Streamlit's cache."""
    logger.info(
        "Refreshing authentication credentials cache for database '%s'.",
        _db.name,
    )
    creds = fetch_all_users_for_auth(_db)
    logger.info("Loaded %d user credential entries.", len(creds.get("usernames", {})))
    return creds


def _resolve_user_role(db: Database, username: str) -> str:
    """Lookup the latest role for a user."""

    user_doc_live = db[COL_USERS].find_one({"username": username}) or {}
    return user_doc_live.get("role", "user")


def _render_authenticated_app(context: AppBootstrapResult, lang: dict[str, Any]) -> None:
    """Render the authenticated portion of the UI."""

    db = context.db
    if db is None:
        setup_screen(lang)
        st.stop()

    credentials = get_auth_credentials(db)

    default_key = "your_strong_secret_key_here"
    secret_key = context.env_doc.get("auth_secret_key", default_key)
    expiry_days = int(context.env_doc.get("auth_cookie_expiry_days", 30))

    if secret_key == default_key:
        st.warning(
            "Warning: Using default auth_secret_key. "
            "Please set a strong secret key in your environment for security.",
        )

    authenticator = stauth.Authenticate(
        credentials,
        "rag_chat_cookie_v1",
        secret_key,
        cookie_expiry_days=expiry_days,
    )

    st.session_state.setdefault("auth_user", None)
    st.session_state.setdefault("is_admin", False)

    authenticator.login(
        fields={
            "Form name": lang["login_title"],
            "Username": lang["username"],
            "Password": lang["password"],
            "Login": lang["login_button"],
        },
    )

    if st.session_state.get("authentication_status"):
        username = st.session_state["username"]
        st.session_state["auth_user"] = username

        user_role = _resolve_user_role(db, username)
        st.session_state["is_admin"] = user_role == "admin"

        with st.sidebar:
            st.markdown(f"### {APP_TITLE}")
            admin_label = f"({lang['sidebar_admin']})" if st.session_state.get("is_admin") else ""
            st.success(
                f"{lang['sidebar_signed_in_as']} **{username}** {admin_label}",
            )
            authenticator.logout(lang["sidebar_logout"], "sidebar")

            # Navigation for all users
            nav_options = [lang["sidebar_nav_chat"], lang["sidebar_nav_password"]]
            if st.session_state.get("is_admin"):
                nav_options.insert(1, lang["sidebar_nav_admin"])
            
            st.radio(
                "Navigation",
                nav_options,
                key="user_view",
                label_visibility="collapsed",
                index=0,
            )
            st.divider()

        user_selection = st.session_state.get("user_view", lang["sidebar_nav_chat"])
        current_conv_id = None
        prompt = None

        if user_selection == lang["sidebar_nav_password"]:
            change_password_page(db, username, lang)
        elif st.session_state.get("is_admin") and user_selection == lang["sidebar_nav_admin"]:
            current_conv_id = render_chat_ui(
                db,
                username,
                lang,
                render_main_chat=False,
                render_conv_sidebar=False,
            )
            admin_dashboard(db, context.env_doc, lang)
        else:
            current_conv_id = render_chat_ui(
                db,
                username,
                lang,
                render_main_chat=True,
                render_conv_sidebar=True,
            )
            prompt = st.chat_input(
                lang.get("chat_input_placeholder", "Ask about your documents..."),
            )

        if prompt:
            if current_conv_id:
                process_new_message(db, context.env_doc, username, current_conv_id, prompt, lang)
                st.rerun()
            else:
                st.error(lang["chat_no_conv_selected"])

    elif st.session_state.get("authentication_status") is False:
        st.error(lang["login_error_incorrect"])

    else:
        st.info(lang["login_prompt"])


def main() -> None:
    """Entry point used by Streamlit to render the application."""

    setup_logging()
    configure_page()
    lang = initialize_language()
    context = bootstrap_application()

    if context.db is None:
        setup_screen(lang)
        st.stop()

    _render_authenticated_app(context, lang)


if __name__ == "__main__":
    main()
