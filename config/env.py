"""Environment persistence and override helpers."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any

from pymongo.database import Database
from pymongo.errors import PyMongoError

from models.settings import default_env

from .paths import ENV_FILE

logger = logging.getLogger(__name__)

# We DO NOT use streamlit.secrets here.
# Precedence (lowest -> highest):
#   default_env()  <  env_settings.json (local dev)
#   < Mongo 'env' doc  <  OS env vars (optional override)

_ENV_MAP: Mapping[str, str] = {
    "openai_api_key": "OPENAI_API_KEY",
    "openai_base_url": "OPENAI_BASE_URL",
    "pinecone_api_key": "PINECONE_API_KEY",
    "pinecone_host": "PINECONE_HOST",
    "pinecone_index_name": "PINECONE_INDEX_NAME",
    "pinecone_namespace": "PINECONE_NAMESPACE",
    "mongo_uri": "MONGO_URI",
    "mongo_db": "MONGO_DB",
}


def _overrides_from_os_env() -> dict[str, Any]:
    """Collect non-empty overrides supplied via environment variables."""
    out: dict[str, Any] = {}
    for key, env_key in _ENV_MAP.items():
        value = os.getenv(env_key)
        if value is not None and str(value).strip():
            out[key] = str(value).strip()
    return out


def _resolve_env_file() -> Path:
    """Return the configured environment file path as a :class:`Path`."""
    return Path(ENV_FILE)


def load_env_doc(db: Database | None = None) -> dict[str, Any]:
    """Return the merged environment document from defaults and overrides."""
    base: dict[str, Any] = default_env()

    # Local JSON (dev convenience)
    file_doc: dict[str, Any] = {}
    env_file = _resolve_env_file()
    if env_file.exists():
        try:
            file_doc = json.loads(env_file.read_text(encoding="utf-8")) or {}
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not load local env_settings.json", exc_info=exc)
            file_doc = {}

    # Mongo (authoritative store for cloud/production)
    mongo_doc: dict[str, Any] = {}
    if db is not None:
        try:
            mongo_doc = db["env"].find_one({"_id": "global"}) or {}
        except PyMongoError as exc:
            logger.exception("Could not fetch env doc from MongoDB", exc_info=exc)
            mongo_doc = {}

    # Optional OS overrides (Docker, CI, etc.)
    top = _overrides_from_os_env()

    # Merge
    return {**base, **file_doc, **mongo_doc, **top}


def save_env_doc(env_doc: Mapping[str, Any], db: Database | None = None) -> None:
    """Persist environment overrides to MongoDB or the local JSON file."""
    payload: MutableMapping[str, Any] = {**default_env(), **(env_doc or {})}
    if db is not None:
        try:
            payload["_id"] = "global"
            db["env"].find_one_and_replace({"_id": "global"}, payload, upsert=True)
            logger.info("Saved environment settings to MongoDB.")
        except PyMongoError as exc:
            logger.exception("Failed to save env doc to MongoDB", exc_info=exc)
        return

    # Local file (dev)
    env_file = _resolve_env_file()
    try:
        env_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Saved environment settings to local env_settings.json.")
    except OSError as exc:
        logger.warning("Failed to save env doc to local JSON", exc_info=exc)

