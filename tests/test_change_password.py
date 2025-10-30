"""Tests for the change password workflow."""

from __future__ import annotations

import pathlib
import sys

import streamlit_authenticator as stauth
from pymongo.errors import PyMongoError

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from db.mongo import COL_USERS, change_user_password  # noqa: E402


class FakeUsersCollection:
    """Simple in-memory stand-in for the users collection."""

    def __init__(self, user_doc: dict[str, str] | None) -> None:
        self._user_doc = user_doc
        self.update_calls: list[tuple[dict[str, str], dict[str, dict[str, str]]]] = []

    def find_one(self, query: dict[str, str]) -> dict[str, str] | None:
        if self._user_doc and query == {"username": self._user_doc["username"]}:
            return dict(self._user_doc)
        return None

    def update_one(self, query: dict[str, str], update: dict[str, dict[str, str]]) -> None:
        self.update_calls.append((query, update))


class FakeDB:
    """Dictionary-like adapter exposing the users collection."""

    def __init__(self, users_collection: FakeUsersCollection) -> None:
        self._users_collection = users_collection

    def __getitem__(self, name: str) -> FakeUsersCollection:
        if name != COL_USERS:
            raise KeyError(name)
        return self._users_collection


def test_change_user_password_successfully_updates_hash() -> None:
    """A successful password change stores a new hash and timestamps."""

    old_hash = stauth.Hasher.hash("old-secret-1")
    user_doc = {"_id": "user-1", "username": "alice", "password_hash": old_hash}
    collection = FakeUsersCollection(user_doc)
    db = FakeDB(collection)

    ok, reason = change_user_password(db, "alice", "old-secret-1", "new-secret-2")

    assert ok is True
    assert reason == "ok"
    assert collection.update_calls, "Expected the password hash to be updated"

    _, update_payload = collection.update_calls[0]
    new_hash = update_payload["$set"]["password_hash"]
    assert stauth.Hasher.check_pw("new-secret-2", new_hash)
    assert update_payload["$set"]["password_changed_at"].endswith("Z")


def test_change_user_password_rejects_incorrect_current_password() -> None:
    """The helper returns the correct error code when the old password is wrong."""

    old_hash = stauth.Hasher.hash("old-secret-1")
    user_doc = {"_id": "user-1", "username": "alice", "password_hash": old_hash}
    collection = FakeUsersCollection(user_doc)
    db = FakeDB(collection)

    ok, reason = change_user_password(db, "alice", "wrong-password", "new-secret-2")

    assert ok is False
    assert reason == "incorrect_current_password"
    assert not collection.update_calls


def test_change_user_password_requires_different_new_password() -> None:
    """Users must choose a password that differs from the current one."""

    old_hash = stauth.Hasher.hash("same-secret")
    user_doc = {"_id": "user-1", "username": "alice", "password_hash": old_hash}
    collection = FakeUsersCollection(user_doc)
    db = FakeDB(collection)

    ok, reason = change_user_password(db, "alice", "same-secret", "same-secret")

    assert ok is False
    assert reason == "password_same_as_current"
    assert not collection.update_calls


def test_change_user_password_handles_database_fetch_errors() -> None:
    """Database errors while fetching the user return a stable code."""

    class FailingCollection(FakeUsersCollection):
        def __init__(self) -> None:
            super().__init__(None)

        def find_one(self, query: dict[str, str]) -> dict[str, str]:  # type: ignore[override]
            raise PyMongoError("boom")

    collection = FailingCollection()
    db = FakeDB(collection)

    ok, reason = change_user_password(db, "alice", "old", "new")

    assert ok is False
    assert reason == "db_fetch_error"
    assert not collection.update_calls
