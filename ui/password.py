"""Streamlit change password page for all users."""

from collections.abc import Mapping
from typing import Any

import streamlit as st
from pymongo.database import Database

from db.mongo import update_password


def change_password_page(db: Database, username: str, lang: Mapping[str, str]) -> None:
    """Render the change password page."""
    st.title(lang["password_change_title"])
    st.write(lang["password_change_description"])

    with st.form("change_password_form"):
        current_password = st.text_input(
            lang["password_change_current_password"],
            type="password",
            help=lang.get("password_change_current_password_help", ""),
        )
        new_password = st.text_input(
            lang["password_change_new_password"],
            type="password",
            help=lang.get("password_change_new_password_help", ""),
        )
        confirm_password = st.text_input(
            lang["password_change_confirm_password"],
            type="password",
            help=lang.get("password_change_confirm_password_help", ""),
        )

        submitted = st.form_submit_button(
            lang["password_change_submit_button"], use_container_width=True
        )

        if submitted:
            # Validation
            if not current_password.strip():
                st.error(lang["password_change_error_current_required"])
                return

            if not new_password.strip():
                st.error(lang["password_change_error_new_required"])
                return

            if len(new_password) < 6:
                st.error(lang["password_change_error_min_length"])
                return

            if new_password != confirm_password:
                st.error(lang["password_change_error_mismatch"])
                return

            if current_password == new_password:
                st.error(lang["password_change_error_same_password"])
                return

            # Update password
            result = update_password(db, username, current_password, new_password)

            if result == "ok":
                st.success(lang["password_change_success"])
                st.rerun()
            else:
                error_msg = lang.get("password_change_error_generic", result)
                if "Current password is incorrect" in result:
                    error_msg = lang["password_change_error_incorrect"]
                elif "Database error" in result:
                    error_msg = lang["password_change_error_database"]
                elif "User not found" in result:
                    error_msg = lang["password_change_error_user_not_found"]
                st.error(error_msg)
