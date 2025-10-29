"""Application configuration models and helpers."""
from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

DEFAULT_CHAT_SYSTEM_PROMPT = (
    "You are a helpful assistant.\n"
    "You have two information sources:\n"
    "1) CHAT HISTORY (previous turns)\n"
    "2) CONTEXT (retrieved document chunks)\n"
    "Guidelines:\n"
    "- Prefer answers grounded in CONTEXT whenever it is sufficient.\n"
    "- If CONTEXT is insufficient or empty and general answers are allowed, "
    "answer from general knowledge (do NOT invent citations).\n"
    "- For meta-questions about the conversation (e.g., 'what was my last question?'), "
    "use CHAT HISTORY.\n"
    "- If you truly don't know, say so clearly.\n"
)


def default_env() -> dict[str, Any]:
    """Return the baseline environment document used throughout the app."""
    return {
        "_id": "global",
        # OpenAI
        "openai_api_key": "",
        "openai_base_url": "",
        "openai_model": "gpt-4o-mini",
        "embedding_model": "text-embedding-3-small",
        # Pinecone
        "pinecone_api_key": "",
        "pinecone_host": "",
        "pinecone_index_name": "",
        "pinecone_index_names": [],
        "pinecone_namespace": "",
        # Retrieval / generation
        "top_k": 5,
        "temperature": 0.2,
        "max_context_chars": 8000,
        "metadata_text_key": "text",
        "metadata_source_key": "source",
        "system_prompt": DEFAULT_CHAT_SYSTEM_PROMPT,
        # Mongo
        "mongo_uri": (
            "mongodb+srv://harshalvankudre_db_user:<db_password>"
            "@chatbot-1.acaznw5.mongodb.net/?appName=chatbot-1"
        ),
        "mongo_db": "rag_chat",
        # RAG vs general fallback
        "allow_general_answers": True,
        "rag_min_context_chars": 600,
        "rag_min_matches": 1,
        "rag_min_score": 0.0,
        # --- NEW: Auth settings ---
        # IMPORTANT: Change this in your environment.
        "auth_secret_key": "your_strong_secret_key_here",
        "auth_cookie_expiry_days": 30,
    }


MISSING_OPENAI_KEY_MSG = "openai_api_key cannot be blank"
MISSING_PINECONE_KEY_MSG = "pinecone_api_key cannot be blank"


class AppSettings(BaseModel):
    """Strongly-typed settings derived from the persisted environment document."""
    # OpenAI
    openai_api_key: str
    openai_base_url: str | None = None
    openai_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # Pinecone
    pinecone_api_key: str
    pinecone_index_name: str | None = None
    pinecone_index_names: list[str] = Field(default_factory=list)
    pinecone_host: str | None = None
    pinecone_namespace: str | None = None

    # Retrieval / generation
    top_k: int = 5
    temperature: float = 0.2
    max_context_chars: int = 8000
    metadata_text_key: str = "text"
    metadata_source_key: str = "source"

    # Prompt
    system_prompt: str = DEFAULT_CHAT_SYSTEM_PROMPT

    # Mongo
    mongo_uri: str | None = None
    mongo_db: str = "rag_chat"

    # RAG vs general fallback
    allow_general_answers: bool = True
    rag_min_context_chars: int = 600
    rag_min_matches: int = 1
    rag_min_score: float = 0.0

    # --- NEW: Auth settings ---
    auth_secret_key: str = "your_strong_secret_key_here"
    auth_cookie_expiry_days: int = 30

    @model_validator(mode="after")
    def _validate_required_keys(self) -> AppSettings:
        """Validate secrets and normalise optional string fields."""
        if not (self.openai_api_key and self.openai_api_key.strip()):
            raise ValueError(MISSING_OPENAI_KEY_MSG)
        if not (self.pinecone_api_key and self.pinecone_api_key.strip()):
            raise ValueError(MISSING_PINECONE_KEY_MSG)

        # --- NEW: Check for default secret key ---
        if self.auth_secret_key == "your_strong_secret_key_here":
            logger.warning(
                "Using default auth_secret_key. Please set a strong secret key in your environment."
            )

        if self.openai_base_url:
            self.openai_base_url = self.openai_base_url.strip() or None
        if self.pinecone_index_name:
            self.pinecone_index_name = self.pinecone_index_name.strip() or None
        if self.pinecone_host:
            self.pinecone_host = self.pinecone_host.strip() or None
        if self.pinecone_namespace:
            self.pinecone_namespace = self.pinecone_namespace.strip() or None

        cleaned_indexes: list[str] = []
        seen: set[str] = set()
        for raw in self.pinecone_index_names:
            candidate = str(raw).strip()
            if candidate and candidate not in seen:
                cleaned_indexes.append(candidate)
                seen.add(candidate)
        self.pinecone_index_names = cleaned_indexes

        if not self.pinecone_index_name and self.pinecone_index_names:
            self.pinecone_index_name = self.pinecone_index_names[0]
        if self.mongo_uri:
            self.mongo_uri = self.mongo_uri.strip() or None
        self.mongo_db = (self.mongo_db or "rag_chat").strip()
        return self

    @classmethod
    def from_env(cls, env_doc: Mapping[str, Any]) -> AppSettings:
        """Build an :class:`AppSettings` instance from a persisted env document."""
        raw_indexes = env_doc.get("pinecone_index_names", [])
        if isinstance(raw_indexes, str):
            raw_indexes = [
                seg.strip()
                for chunk in raw_indexes.splitlines()
                for seg in chunk.split(",")
                if seg.strip()
            ]
        elif isinstance(raw_indexes, list):
            raw_indexes = list(raw_indexes)
        elif raw_indexes:
            raw_indexes = [str(raw_indexes)]
        else:
            raw_indexes = []
        return cls(
            openai_api_key=env_doc.get("openai_api_key", ""),
            openai_base_url=(env_doc.get("openai_base_url") or None),
            openai_model=env_doc.get("openai_model", "gpt-4o-mini"),
            embedding_model=env_doc.get("embedding_model", "text-embedding-3-small"),
            pinecone_api_key=env_doc.get("pinecone_api_key", ""),
            pinecone_index_name=(env_doc.get("pinecone_index_name") or None),
            pinecone_index_names=[str(item) for item in raw_indexes],
            pinecone_host=(env_doc.get("pinecone_host") or None),
            pinecone_namespace=(env_doc.get("pinecone_namespace") or None),
            top_k=int(env_doc.get("top_k", 5)),
            temperature=float(env_doc.get("temperature", 0.2)),
            max_context_chars=int(env_doc.get("max_context_chars", 8000)),
            metadata_text_key=env_doc.get("metadata_text_key", "text"),
            metadata_source_key=env_doc.get("metadata_source_key", "source"),
            system_prompt=env_doc.get("system_prompt", DEFAULT_CHAT_SYSTEM_PROMPT),
            mongo_uri=env_doc.get("mongo_uri"),
            mongo_db=env_doc.get("mongo_db", "rag_chat"),
            allow_general_answers=env_doc.get("allow_general_answers", True),
            rag_min_context_chars=int(env_doc.get("rag_min_context_chars", 600)),
            rag_min_matches=int(env_doc.get("rag_min_matches", 1)),
            rag_min_score=float(env_doc.get("rag_min_score", 0.0)),
            auth_secret_key=env_doc.get("auth_secret_key", "your_strong_secret_key_here"),
            auth_cookie_expiry_days=int(env_doc.get("auth_cookie_expiry_days", 30)),
        )

