# config/env.py
import json
import logging
import os
from typing import Any, Dict

from models.settings import default_env

from .paths import ENV_FILE

logger = logging.getLogger(__name__)

# We DO NOT use streamlit.secrets here.
# Precedence (lowest -> highest):
#   default_env()  <  env_settings.json (local dev)  <  Mongo 'env' doc  <  OS env vars (optional override)

_ENV_MAP = {
    "openai_api_key": "OPENAI_API_KEY",
    "openai_base_url": "OPENAI_BASE_URL",
    "pinecone_api_key": "PINECONE_API_KEY",
    "pinecone_host": "PINECONE_HOST",
    "pinecone_index_name": "PINECONE_INDEX_NAME",
    "pinecone_namespace": "PINECONE_NAMESPACE",
    "mongo_uri": "MONGO_URI",
    "mongo_db": "MONGO_DB",
}


def _overrides_from_os_env() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, env_key in _ENV_MAP.items():
        v = os.getenv(env_key)
        if v is not None and str(v).strip():
            out[key] = str(v).strip()
    return out


def load_env_doc(db=None) -> Dict[str, Any]:
    base = default_env()

    # Local JSON (dev convenience)
    file_doc: Dict[str, Any] = {}
    if os.path.exists(ENV_FILE):
        try:
            with open(ENV_FILE, encoding="utf-8") as f:
                file_doc = json.load(f) or {}
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Could not load local env_settings.json: {e}")
            file_doc = {}

    # Mongo (authoritative store for cloud/production)
    mongo_doc: Dict[str, Any] = {}
    if db is not None:
        try:
            mongo_doc = db["env"].find_one({"_id": "global"}) or {}
        except Exception as e:
            logger.error(f"Could not fetch env doc from MongoDB: {e}")
            mongo_doc = {}

    # Optional OS overrides (Docker, CI, etc.)
    top = _overrides_from_os_env()

    # Merge
    return {**base, **file_doc, **mongo_doc, **top}


def save_env_doc(env_doc: Dict[str, Any], db=None) -> None:
    """Save config to Mongo if available; otherwise to local JSON (dev only)."""
    payload = {**default_env(), **(env_doc or {})}
    if db is not None:
        try:
            payload["_id"] = "global"
            db["env"].find_one_and_replace({"_id": "global"}, payload, upsert=True)
            logger.info("Saved environment settings to MongoDB.")
        except Exception as e:
            logger.exception(f"Failed to save env doc to MongoDB: {e}")
        return
    # Local file (dev)
    try:
        with open(ENV_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info("Saved environment settings to local env_settings.json.")
    except OSError as e:
        logger.warning(f"Failed to save env doc to local JSON: {e}")

