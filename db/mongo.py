import datetime as dt
from typing import Any, Dict, List, Optional
from pymongo import MongoClient, ASCENDING
from bson import ObjectId
import hashlib
import streamlit_authenticator as stauth

COL_USERS = "users"
COL_ENV = "env"
COL_CONV = "conversations"
COL_MSG = "messages"
COL_INGEST = "ingest_manifest"

def sha256(s: str) -> str:
    import hashlib
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def get_mongo(uri: Optional[str]):
    if not uri: return None
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        _ = client.server_info()
        return client
    except Exception:
        return None

def ensure_indexes(db):
    db[COL_USERS].create_index([("username", ASCENDING)], unique=True)
    db[COL_USERS].create_index([("email", ASCENDING)], unique=True)
    db[COL_CONV].create_index([("user", ASCENDING), ("created_at", ASCENDING)])
    db[COL_MSG].create_index([("conv_id", ASCENDING), ("created_at", ASCENDING)])

# --- UPDATED FUNCTION ---
def seed_admin_if_empty(db) -> str:
    """Checks if the user collection is empty and seeds an admin user if it is. Returns a status message."""
    count = db[COL_USERS].count_documents({})
    if count == 0:
        # --- THIS LINE IS THE FIX ---
        hashed_password = stauth.Hasher().hash("password 123")
        # ---------------------------
        db[COL_USERS].insert_one({
            "username": "admin",
            "password_hash": hashed_password,
            "role": "admin",
            "email": "admin@example.com",
            "created_at": dt.datetime.utcnow().isoformat() + "Z",
        })
        return "User collection was empty. Seeded default 'admin' user."
    else:
        return f"User collection not empty (found {count} users). Skipping seed."

def fetch_all_users_for_auth(db) -> dict:
    """Fetches users and formats them for streamlit-authenticator."""
    credentials = {"usernames": {}}
    users = db[COL_USERS].find({})
    for user in users:
        username = user["username"]
        credentials["usernames"][username] = {
            "name": user.get("email", username),
            "email": user.get("email", ""),
            "password": user.get("password_hash", ""),
            "role": user.get("role", "user")
        }
    return credentials

def list_conversations(db, username: str) -> List[Dict[str, Any]]:
    convs = list(db[COL_CONV].find({"user": username}).sort("updated_at", -1))
    for c in convs:
        c["id"] = str(c["_id"])
    return convs

def create_conversation(db, username: str, title: Optional[str] = None) -> str:
    now = dt.datetime.utcnow().isoformat() + "Z"
    doc = {"user": username, "title": title or "New chat", "created_at": now, "updated_at": now}
    res = db[COL_CONV].insert_one(doc)
    return str(res.inserted_id)

def rename_conversation(db, conv_id: str, username: str, new_title: str):
    db[COL_CONV].update_one({"_id": ObjectId(conv_id), "user": username}, {"$set": {"title": new_title, "updated_at": dt.datetime.utcnow().isoformat()+"Z"}})

def delete_conversation(db, conv_id: str, username: str):
    db[COL_MSG].delete_many({"conv_id": conv_id})
    db[COL_CONV].delete_one({"_id": ObjectId(conv_id), "user": username})

def add_message(db, conv_id: str, username: str, role: str, content: str) -> str:
    now = dt.datetime.utcnow().isoformat() + "Z"
    doc = {"conv_id": conv_id, "user": username, "role": role, "content": content, "created_at": now}
    res = db[COL_MSG].insert_one(doc)
    db[COL_CONV].update_one({"_id": ObjectId(conv_id)}, {"$set": {"updated_at": now}})
    return str(res.inserted_id)

def get_messages(db, conv_id: str, limit: int = 80) -> List[Dict[str, Any]]:
    msgs = list(db[COL_MSG].find({"conv_id": conv_id}).sort("created_at", 1).limit(limit))
    for m in msgs: m["id"] = str(m["_id"])
    return msgs

# --- UPDATED FUNCTION ---
def add_user(db, username: str, password: str, email: str, role: str = "user") -> Optional[str]:
    """ --- UPDATED function to use .hash() --- """
    try:
        # --- THIS LINE IS THE FIX ---
        hashed_password = stauth.Hasher().hash(password)
        # ---------------------------
        db[COL_USERS].insert_one({
            "username": username,
            "password_hash": hashed_password,
            "email": email,
            "role": role,
            "created_at": dt.datetime.utcnow().isoformat() + "Z",
        })
        return "ok"
    except Exception as e:
        if "E11000 duplicate key error" in str(e) and "email" in str(e):
            return "Email address already exists."
        return str(e)

def delete_user(db, username: str) -> Optional[str]:
    try:
        convs = list(db[COL_CONV].find({"user": username}, {"_id": 1}))
        conv_ids = [str(c["_id"]) for c in convs]
        if conv_ids:
            db[COL_MSG].delete_many({"conv_id": {"$in": conv_ids}})
            db[COL_CONV].delete_many({"_id": {"$in": [c["_id"] for c in convs]}})
        db[COL_USERS].delete_one({"username": username})
        return "ok"
    except Exception as e:
        return str(e)