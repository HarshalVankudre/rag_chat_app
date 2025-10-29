import streamlit as st
import streamlit_authenticator as stauth

from config.env import load_env_doc, save_env_doc
from db.mongo import get_mongo, ensure_indexes, seed_admin_if_empty, fetch_all_users_for_auth
from ui.chat import render_chat_ui, process_new_message
from ui.admin import admin_dashboard
from config.i18n import get_lang

APP_TITLE = "💬 RükoGPT"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'lang_code' not in st.session_state:
    st.session_state['lang_code'] = 'en'

with st.sidebar:
    st.selectbox(
        label="Language / Sprache",
        options=[('en', '🇬🇧 English'), ('de', '🇩🇪 Deutsch')],
        format_func=lambda x: x[1],
        key='lang_select_key',
        on_change=lambda: st.session_state.update(lang_code=st.session_state.lang_select_key[0])
    )

lang = get_lang(st.session_state.get('lang_code', 'en'))


def setup_screen(lang: dict):
    # ... (This function is unchanged)
    st.title(lang["setup_title"])
    st.write(lang["setup_descr"])
    uri = st.text_input(lang["setup_mongo_uri"], value="", type="password",
                        placeholder=lang["setup_mongo_uri_placeholder"], help=lang["setup_mongo_uri_help"])
    db_name = st.text_input(lang["setup_mongo_db"], value="rag_chat", placeholder="rag_chat")
    col1, col2 = st.columns(2)
    with col1:
        if st.button(lang["setup_button"], use_container_width=True):
            test_uri = uri.strip()
            if not test_uri: st.error(lang["setup_error_no_uri"]); return
            client = get_mongo(test_uri)
            if client is None: st.error(lang["setup_error_connect_failed"]); return
            save_env_doc({"mongo_uri": test_uri, "mongo_db": db_name.strip() or "rag_chat"}, db=None)
            db = client[(db_name.strip() or "rag_chat")]
            try:
                ensure_indexes(db)
                # --- UPDATED TO CAPTURE RETURN VALUE ---
                seed_status = seed_admin_if_empty(db)
                print(f"--- DEBUG (Setup): Seeding status: {seed_status}")
                # -------------------------------------
                save_env_doc({"mongo_uri": test_uri, "mongo_db": db.name}, db=db)
            except Exception as e:
                print(f"--- DEBUG (Setup): Error during seeding: {e}")
                pass
            st.success(lang["setup_success"]);
            st.rerun()
    with col2:
        st.caption(lang["setup_caption"])


# -------- Bootstrap: read config and try connecting --------
env_local = load_env_doc(db=None)
mongo_client = get_mongo(env_local.get("mongo_uri"))
db = None

if mongo_client is not None:
    db = mongo_client[env_local.get("mongo_db", "rag_chat")]

    # --- ADDED LOGS ---
    print("\n" + "=" * 50)
    print(f"--- DEBUG: Connected to MongoDB. Using database: '{db.name}'")
    # ------------------

    try:
        ensure_indexes(db)
        # --- UPDATED TO CAPTURE AND PRINT RETURN VALUE ---
        seed_status = seed_admin_if_empty(db)
        print(f"--- DEBUG: Seeding status: {seed_status}")
        # -----------------------------------------------
    except Exception as e:
        # --- ADDED LOG ---
        print(f"--- DEBUG: Error during startup indexing/seeding: {e}")
        # -----------------
        pass

env_doc = load_env_doc(db) if db is not None else env_local

if db is None:
    setup_screen(lang)
    st.stop()

# --- Authenticator Setup ---
credentials = fetch_all_users_for_auth(db)

# --- ADDED LOG ---
print(f"--- DEBUG: Fetched credentials for authenticator: {credentials}")
print("=" * 50 + "\n")
# -----------------

default_key = "your_strong_secret_key_here"
secret_key = env_doc.get("auth_secret_key", default_key)
expiry_days = int(env_doc.get("auth_cookie_expiry_days", 30))

if secret_key == default_key:
    st.warning(
        "Warning: Using default auth_secret_key. Please set a strong secret key in your environment for security.")

authenticator = stauth.Authenticate(
    credentials,
    "rag_chat_cookie_v1",
    secret_key,
    cookie_expiry_days=expiry_days
)

# ... (rest of the file is unchanged) ...
st.session_state["auth_user"] = None
st.session_state["is_admin"] = False

authenticator.login(fields={
    'Form name': lang["login_title"],
    'Username': lang["username"],
    'Password': lang["password"],
    'Login': lang["login_button"]
})

if st.session_state.get("authentication_status"):
    username = st.session_state["username"]
    st.session_state["auth_user"] = username

    user_data = credentials.get("usernames", {}).get(username, {})
    user_role = user_data.get("role", "user")
    st.session_state["is_admin"] = (user_role == "admin")

    with st.sidebar:
        st.markdown(f"### {APP_TITLE}")
        admin_label = f"({lang['sidebar_admin']})" if st.session_state.get('is_admin') else ''
        st.success(
            f"{lang['sidebar_signed_in_as']} **{username}** {admin_label}"
        )
        authenticator.logout(lang["sidebar_logout"], "sidebar")

        if st.session_state.get("is_admin"):
            st.radio(
                "Admin Navigation",
                [lang["sidebar_nav_chat"], lang["sidebar_nav_admin"]],
                key="admin_view",
                label_visibility="collapsed",
                index=0
            )
            st.divider()

    current_conv_id = None
    prompt = None

    if st.session_state.get("is_admin"):
        admin_view_selection = st.session_state.get("admin_view", lang["sidebar_nav_chat"])
        if admin_view_selection == lang["sidebar_nav_chat"]:
            current_conv_id = render_chat_ui(db, username, lang, render_main_chat=True, render_conv_sidebar=True)
            prompt = st.chat_input(lang.get("chat_input_placeholder", "Ask about your documents or anything..."))
        elif admin_view_selection == lang["sidebar_nav_admin"]:
            current_conv_id = render_chat_ui(db, username, lang, render_main_chat=False, render_conv_sidebar=False)
            admin_dashboard(db, env_doc, lang)
    else:
        current_conv_id = render_chat_ui(db, username, lang, render_main_chat=True, render_conv_sidebar=True)
        prompt = st.chat_input(lang.get("chat_input_placeholder", "Ask about your documents or anything..."))

    if prompt:
        if current_conv_id:
            process_new_message(db, env_doc, username, current_conv_id, prompt, lang)
            st.rerun()
        else:
            st.error(lang["chat_no_conv_selected"])

elif st.session_state["authentication_status"] is False:
    st.error(lang["login_error_incorrect"])

elif st.session_state["authentication_status"] is None:
    st.info(lang["login_prompt"])