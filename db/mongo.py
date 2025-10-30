"""MongoDB persistence helpers used by the application."""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any

import streamlit_authenticator as stauth
from bson import ObjectId
from pymongo import ASCENDING, MongoClient
from pymongo.database import Database
from pymongo.errors import (
    ConnectionFailure,
    DuplicateKeyError,
    OperationFailure,
    PyMongoError,
)

logger = logging.getLogger(__name__)

COL_USERS = "users"
COL_ENV = "env"
COL_CONV = "conversations"
COL_MSG = "messages"
COL_INGEST = "ingest_manifest"


def get_mongo(uri: str | None) -> MongoClient | None:
    """Return a connected :class:`MongoClient` if the URI is provided."""
    if not uri:
        return None
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.server_info()  # Test connection
    except ConnectionFailure:
        logger.exception("MongoDB connection failed")
        return None
    logger.info("MongoDB connection successful.")
    return client


def ensure_indexes(db: Database) -> None:
    """Create the indexes required by the application collections."""
    try:
        db[COL_USERS].create_index([("username", ASCENDING)], unique=True)
        db[COL_USERS].create_index([("email", ASCENDING)], unique=True)
        db[COL_CONV].create_index([("user", ASCENDING), ("created_at", ASCENDING)])
        db[COL_MSG].create_index([("conv_id", ASCENDING), ("created_at", ASCENDING)])
    except OperationFailure:
        logger.exception("Could not create indexes (this may be fine)")
    else:
        logger.info("Database indexes ensured.")


def seed_admin_if_empty(db: Database) -> str:
    """Seed a default admin user if the users collection is empty."""
    try:
        count = db[COL_USERS].count_documents({})
    except PyMongoError:
        logger.exception("Failed to count users while seeding admin user")
        return "Failed to seed admin user: database error"

    if count != 0:
        return f"User collection not empty (found {count} users). Skipping seed."

    hashed_password = stauth.Hasher().hash("password 123")
    try:
        db[COL_USERS].insert_one(
            {
                "username": "admin",
                "password_hash": hashed_password,
                "role": "admin",
                "email": "admin@example.com",
                "created_at": dt.datetime.utcnow().isoformat() + "Z",
            }
        )
    except PyMongoError:
        logger.exception("Failed to insert default admin user")
        return "Failed to seed admin user: database error"
    return "User collection was empty. Seeded default 'admin' user."


def fetch_all_users_for_auth(db: Database) -> dict[str, dict[str, dict[str, str]]]:
    """Return credentials structured for ``streamlit-authenticator``."""
    credentials: dict[str, dict[str, dict[str, str]]] = {"usernames": {}}
    try:
        users = db[COL_USERS].find({})
    except PyMongoError:
        logger.exception("Failed to fetch users for auth")
        return credentials

    for user in users:
        username = user["username"]
        credentials["usernames"][username] = {
            "name": user.get("email", username),
            "email": user.get("email", ""),
            "password": user.get("password_hash", ""),
            "role": user.get("role", "user"),
        }
    return credentials


def list_conversations(db: Database, username: str) -> list[dict[str, Any]]:
    """Return conversations for a user ordered by recent activity."""
    try:
        convs = list(db[COL_CONV].find({"user": username}).sort("updated_at", -1))
    except PyMongoError:
        logger.exception("Failed to list conversations", extra={"username": username})
        return []

    for conv in convs:
        conv["id"] = str(conv["_id"])
    return convs


def create_conversation(db: Database, username: str, title: str | None = None) -> str:
    """Create a new conversation for the supplied user."""
    now = dt.datetime.utcnow().isoformat() + "Z"
    doc = {
        "user": username,
        "title": title or "New chat",
        "created_at": now,
        "updated_at": now,
    }
    result = db[COL_CONV].insert_one(doc)
    return str(result.inserted_id)


def rename_conversation(db: Database, conv_id: str, username: str, new_title: str) -> None:
    """Rename a conversation owned by ``username``."""
    db[COL_CONV].update_one(
        {"_id": ObjectId(conv_id), "user": username},
        {"$set": {"title": new_title, "updated_at": dt.datetime.utcnow().isoformat() + "Z"}},
    )


def delete_conversation(db: Database, conv_id: str, username: str) -> None:
    """Delete a conversation and its messages."""
    db[COL_MSG].delete_many({"conv_id": conv_id})
    db[COL_CONV].delete_one({"_id": ObjectId(conv_id), "user": username})


def add_message(db: Database, conv_id: str, username: str, role: str, content: str) -> str:
    """Persist a chat message and update conversation metadata."""
    now = dt.datetime.utcnow().isoformat() + "Z"
    doc = {
        "conv_id": conv_id,
        "user": username,
        "role": role,
        "content": content,
        "created_at": now,
    }
    result = db[COL_MSG].insert_one(doc)
    db[COL_CONV].update_one(
        {"_id": ObjectId(conv_id)},
        {"$set": {"updated_at": now}},
    )
    return str(result.inserted_id)


def get_messages(db: Database, conv_id: str, limit: int = 80) -> list[dict[str, Any]]:
    """Return messages for a conversation up to ``limit`` items."""
    try:
        messages = list(db[COL_MSG].find({"conv_id": conv_id}).sort("created_at", 1).limit(limit))
    except PyMongoError:
        logger.exception("Failed to fetch messages", extra={"conv_id": conv_id})
        return []

    for message in messages:
        message["id"] = str(message["_id"])
    return messages


def add_user(
    db: Database, username: str, password: str, email: str, role: str = "user"
) -> str | None:
    """Create a new user, returning an error string on failure."""
    hashed_password = stauth.Hasher().hash(password)
    document = {
        "username": username,
        "password_hash": hashed_password,
        "email": email,
        "role": role,
        "created_at": dt.datetime.utcnow().isoformat() + "Z",
    }
    try:
        db[COL_USERS].insert_one(document)
    except DuplicateKeyError as exc:
        logger.warning("Failed to add user due to duplicate key", exc_info=exc)
        if "email" in str(exc):
            return "Email address already exists."
        return "Username already exists."
    except PyMongoError:
        logger.exception("Failed to add user", extra={"username": username})
        return "Database error while creating user."
    return "ok"


def delete_user(db: Database, username: str) -> str | None:
    """Delete a user and any related conversations."""
    try:
        convs = list(db[COL_CONV].find({"user": username}, {"_id": 1}))
    except PyMongoError:
        logger.exception("Failed to fetch conversations for deletion", extra={"username": username})
        return "Database error while fetching user conversations."

    conv_ids = [str(conv["_id"]) for conv in convs]
    if conv_ids:
        db[COL_MSG].delete_many({"conv_id": {"$in": conv_ids}})
        db[COL_CONV].delete_many({"_id": {"$in": [conv["_id"] for conv in convs]}})

    try:
        db[COL_USERS].delete_one({"username": username})
    except PyMongoError:
        logger.exception("Failed to delete user", extra={"username": username})
        return "Database error while deleting user."
    return "ok"
