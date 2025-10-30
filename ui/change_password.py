"""Streamlit view for the change password workflow."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import streamlit as st
from pymongo.database import Database

from db.mongo import change_user_password


def render_change_password_page(
    db: Database,
    username: str,
    lang: Mapping[str, str],
    *,
    on_success: Callable[[], Any] | None = None,
    min_length: int = 8,
) -> None:
    """Render the change password form for authenticated users."""

    st.title(lang["change_password_title"])
    st.write(lang["change_password_help"].format(min_length=min_length))

    with st.form("change-password-form", clear_on_submit=False):
        current_password = st.text_input(
            lang["change_password_current"],
            type="password",
        )
        new_password = st.text_input(
            lang["change_password_new"],
            type="password",
        )
        confirm_password = st.text_input(
            lang["change_password_confirm"],
            type="password",
        )

        submitted = st.form_submit_button(
            lang["change_password_submit"],
            type="primary",
            use_container_width=True,
        )

    if not submitted:
        return

    if not current_password or not new_password or not confirm_password:
        st.error(lang["change_password_error_required"])
        return

    if len(new_password) < min_length:
        st.error(lang["change_password_error_length"].format(min_length=min_length))
        return

    if new_password != confirm_password:
        st.error(lang["change_password_error_mismatch"])
        return

    success, reason = change_user_password(db, username, current_password, new_password)

    if success:
        st.success(lang["change_password_success"])
        if on_success is not None:
            on_success()
        return

    error_messages = {
        "db_fetch_error": lang["change_password_error_db_fetch"],
        "user_not_found": lang["change_password_error_user_not_found"],
        "no_existing_password": lang["change_password_error_no_password"],
        "incorrect_current_password": lang["change_password_error_incorrect"],
        "password_same_as_current": lang["change_password_error_same_password"],
        "db_update_error": lang["change_password_error_db_update"],
    }

    st.error(error_messages.get(reason, lang["change_password_error_generic"].format(reason=reason)))
