import streamlit as st

from config.env import load_env_doc, save_env_doc
from db.mongo import get_mongo, ensure_indexes, seed_admin_if_empty, verify_user
# Import the new functions
from ui.chat import render_chat_ui, process_new_message
from ui.admin import admin_dashboard

# --- UPDATED APP TITLE ---
APP_TITLE = "💬 RükoGPT"
# -------------------------

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)


def setup_screen():
    """First-run setup to enter Mongo URI/DB. Does NOT prefill or show secrets."""
    st.title("🛠️ First-run setup: MongoDB")
    st.write(
        "Enter your MongoDB connection string and database name. "
        "Values are stored in Mongo (collection `env`) after a successful connect."
    )

    uri = st.text_input(
        "Mongo URI",
        value="",
        type="password",
        placeholder="mongodb+srv://user:pass@cluster.mongodb.net/?retryWrites=true&w=majority&appName=chatbot",
        help="The value is masked and not displayed back. It will be saved to Mongo after connection."
    )
    db_name = st.text_input("Mongo DB name", value="rag_chat", placeholder="rag_chat")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save & Connect", use_container_width=True):
            test_uri = uri.strip()
            if not test_uri:
                st.error("Please paste a Mongo URI.")
                return

            # Try connect first (no secrets persisted yet)
            client = get_mongo(test_uri)
            if client is None:
                st.error(
                    "Could not connect to MongoDB.\n\n"
                    "- Check username/password\n"
                    "- Ensure Atlas Network Access allows this host (Streamlit Cloud egress IPs)\n"
                    "- Verify the connection string"
                )
                return

            # On success: write to local file (dev) so next boot reuses; also seed DB
            save_env_doc({"mongo_uri": test_uri, "mongo_db": db_name.strip() or "rag_chat"}, db=None)

            db = client[(db_name.strip() or "rag_chat")]
            try:
                ensure_indexes(db)
                seed_admin_if_empty(db)  # seeds admin if empty
                # Persist full env doc into Mongo so cloud runs pull from DB thereafter
                save_env_doc({"mongo_uri": test_uri, "mongo_db": db.name}, db=db)
            except Exception:
                pass

            st.success("MongoDB connected. Re-running the app…")
            st.rerun()
    with col2:
        st.caption("Tip: If Streamlit Cloud can't reach Mongo, check Atlas IP allowlist.")


def login_screen(db):
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Sign in"):
        if db is None:
            st.error("MongoDB is not configured yet.")
        else:
            if verify_user(db, username, password):
                user_doc = db["users"].find_one({"username": username}) or {}
                st.session_state["auth_user"] = username
                st.session_state["is_admin"] = (user_doc.get("role") == "admin")
                st.rerun()
            else:
                st.error("Invalid credentials.")


# -------- Bootstrap: read config and try connecting --------
env_local = load_env_doc(db=None)
mongo_client = get_mongo(env_local.get("mongo_uri"))
db = None
if mongo_client is not None:
    db = mongo_client[env_local.get("mongo_db", "rag_chat")]
    try:
        ensure_indexes(db)
        seed_admin_if_empty(db)
    except Exception:
        pass

env_doc = load_env_doc(db) if db is not None else env_local

# If still no DB connection, show setup page
if db is None:
    setup_screen()
    st.stop()

# -------- Sidebar session controls --------
with st.sidebar:
    st.markdown(f"### {APP_TITLE}")
    if "auth_user" not in st.session_state:
        st.session_state["auth_user"] = None
    if "is_admin" not in st.session_state:
        st.session_state["is_admin"] = False

    if st.session_state["auth_user"]:
        st.success(
            f"Signed in as **{st.session_state['auth_user']}** "
            f"{'(admin)' if st.session_state.get('is_admin') else ''}"
        )
        if st.button("Log out"):
            for k in ["auth_user", "is_admin", "current_conv_id"]:
                st.session_state.pop(k, None)
            st.rerun()

# -------- Main routing --------
if not st.session_state["auth_user"]:
    login_screen(db)
else:
    # State for chat
    current_conv_id = None
    username = st.session_state["auth_user"]

    if st.session_state.get("is_admin"):
        tab_chat, tab_admin = st.tabs(["💬 Chat", "🛡️ Admin"])
        with tab_chat:
            # Render chat messages and sidebar
            current_conv_id = render_chat_ui(db, username)
        with tab_admin:
            admin_dashboard(db, env_doc)
    else:
        # Render chat messages and sidebar
        current_conv_id = render_chat_ui(db, username)

    # --- Place st.chat_input at the top level ---
    prompt = st.chat_input("Ask about your documents or anything...")

    if prompt:
        if current_conv_id:
            # We can process the message regardless of the tab,
            # as long as a conversation is selected.
            process_new_message(db, env_doc, username, current_conv_id, prompt)

            # Rerun only if we are in the admin tab view to refresh the chat
            # This is a small hack to make sure the tab content updates
            if st.session_state.get("is_admin"):
                st.rerun()
        else:
            # This case might happen if no chat is created yet, etc.
            st.error("No active conversation selected.")