import streamlit as st

from models.settings import default_env
from config.env import load_env_doc, save_env_doc
from db.mongo import get_mongo, ensure_indexes, seed_admin_if_empty, verify_user
from ui.chat import chat_window
from ui.admin import admin_dashboard


APP_TITLE = "🔐 RAG Chat (Admins & Users, MongoDB, Pinecone)"
st.set_page_config(page_title=APP_TITLE, page_icon="🔐", layout="wide")


def setup_screen():
    """First-run setup screen shown when MongoDB is not configured yet."""
    st.title("🛠️ First-run setup: MongoDB")
    st.write(
        "Paste your MongoDB connection string and database name. "
        "This saves locally so the app can connect and seed the admin user."
    )

    # Prefill with your known values to make this easy
    uri = st.text_input(
        "Mongo URI",
        value=(
            "mongodb+srv://harshalvankudre_db_user:QYJ7qsxDnQg6OS8x"
            "@chatbot-1.acaznw5.mongodb.net/?retryWrites=true&w=majority&appName=chatbot-1"
        ),
    )
    db_name = st.text_input("Mongo DB name", value="rag_chat")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save & Connect"):
            # Save to local env JSON so the very next run can connect to Mongo
            save_env_doc({"mongo_uri": uri.strip(), "mongo_db": db_name.strip()}, db=None)
            st.success("Saved. Re-running the app to connect…")
            st.rerun()
    with col2:
        st.caption("You can change these later in Admin → Environment.")


def login_screen(db):
    """Username/password login. Admin is auto-seeded in Mongo on first connect."""
    st.title("🔐 Login")

    col1, col2 = st.columns([2, 3])
    with col1:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Sign in"):
            if db is None:
                st.error(
                    "MongoDB is not configured yet. Use the first-run setup to save your Mongo URI."
                )
            else:
                if verify_user(db, username, password):
                    u = db["users"].find_one({"username": username})
                    st.session_state["auth_user"] = username
                    st.session_state["is_admin"] = (u.get("role") == "admin")
                    st.rerun()
                else:
                    st.error("Invalid credentials.")
    with col2:
        st.info(
            "Default admin credentials (auto-seeded on first Mongo connect):\n"
            "- **user**: `admin`\n"
            "- **password**: `password 123`"
        )


# -----------------------------
# Bootstrap: load env + connect
# -----------------------------
# 1) Load local env first (may only contain Mongo URI/DB on first run)
env_local = load_env_doc(db=None)

# 2) Try connecting to Mongo using whatever URI is there right now
mongo_client = get_mongo(env_local.get("mongo_uri"))
db = None
if mongo_client is not None:
    db = mongo_client[env_local.get("mongo_db", "rag_chat")]
    try:
        ensure_indexes(db)
        seed_admin_if_empty(db)  # ensures admin/admin-cred exists
    except Exception:
        pass

# 3) Load the latest env from Mongo if available (merged with defaults)
env_doc = load_env_doc(db) if db is not None else env_local

# If no DB connection yet, show setup screen and stop
if db is None:
    setup_screen()
    st.stop()


# -----------------------------
# Sidebar auth controls
# -----------------------------
with st.sidebar:
    if "auth_user" not in st.session_state:
        st.session_state["auth_user"] = None
    if "is_admin" not in st.session_state:
        st.session_state["is_admin"] = False

    st.markdown(f"### {APP_TITLE}")
    if st.session_state.get("auth_user"):
        st.success(
            f"Signed in as **{st.session_state['auth_user']}** "
            f"{'(admin)' if st.session_state.get('is_admin') else ''}"
        )
        if st.button("Log out"):
            for k in ["auth_user", "is_admin", "current_conv_id"]:
                st.session_state.pop(k, None)
            st.rerun()


# -----------------------------
# Main routing
# -----------------------------
if not st.session_state["auth_user"]:
    login_screen(db)
else:
    if st.session_state.get("is_admin"):
        tab_chat, tab_admin = st.tabs(["💬 Chat", "🛡️ Admin"])
        with tab_chat:
            # Chat window always reloads fresh ENV from Mongo internally
            chat_window(db, env_doc, st.session_state["auth_user"])
        with tab_admin:
            # Admin dashboard reruns after saving env to avoid stale settings
            admin_dashboard(db, env_doc)
    else:
        chat_window(db, env_doc, st.session_state["auth_user"])
