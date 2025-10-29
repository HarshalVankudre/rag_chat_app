"""Utilities that assemble the dependencies required by the Streamlit app."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from dataclasses import dataclass
from typing import Any, Mapping

from pydantic import ValidationError
from pymongo import errors as mongo_errors
from pymongo.database import Database

from config.env import load_env_doc
from db.mongo import ensure_indexes, get_mongo, seed_admin_if_empty
from models.settings import AppSettings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AppBootstrapResult:
    """Container with the state required to run the Streamlit application."""

    db: Database | None
    env_doc: dict[str, Any]
    settings: AppSettings | None


def _should_validate_settings(env_doc: Mapping[str, Any]) -> bool:
    """Return True when the provided document contains API credentials."""

    required_keys = ("openai_api_key", "pinecone_api_key")
    for key in required_keys:
        value = env_doc.get(key)
        if not value or not str(value).strip():
            return False
    return True


def bootstrap_application() -> AppBootstrapResult:
    """Load environment configuration and prepare the MongoDB database."""

    env_local = load_env_doc(db=None)
    mongo_client = get_mongo(env_local.get("mongo_uri"))
    db: Database | None = None
    env_doc = env_local

    if mongo_client is not None:
        db = mongo_client[env_local.get("mongo_db", "rag_chat")]
        logger.info("Connected to MongoDB database '%s'.", db.name)
        try:
            ensure_indexes(db)
            seed_status = seed_admin_if_empty(db)
            logger.info("Database bootstrap completed: %s", seed_status)
        except mongo_errors.PyMongoError:
            logger.warning("Encountered a MongoDB error while bootstrapping.", exc_info=True)
        except OSError:
            logger.exception("Encountered an OS level error while bootstrapping the database.")

        env_doc = load_env_doc(db)

    settings: AppSettings | None = None
    if _should_validate_settings(env_doc):
        try:
            settings = AppSettings.from_env(env_doc)
        except ValidationError:
            logger.exception("Environment settings failed validation during bootstrap.")

    return AppBootstrapResult(db=db, env_doc=env_doc, settings=settings)
