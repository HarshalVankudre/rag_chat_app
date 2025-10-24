# models/settings.py
from typing import Optional
from pydantic import BaseModel, model_validator

DEFAULT_CHAT_SYSTEM_PROMPT = (
    "You are a helpful assistant.\n"
    "You have two information sources:\n"
    "1) CHAT HISTORY (previous turns)\n"
    "2) CONTEXT (retrieved document chunks)\n"
    "Guidelines:\n"
    "- Prefer answers grounded in CONTEXT whenever it is sufficient.\n"
    "- If CONTEXT is insufficient or empty and general answers are allowed, answer from general knowledge (do NOT invent citations).\n"
    "- For meta-questions about the conversation (e.g., 'what was my last question?'), use CHAT HISTORY.\n"
    "- If you truly don't know, say so clearly.\n"
)

def default_env() -> dict:
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
        "pinecone_namespace": "",
        # Retrieval / generation
        "top_k": 5,
        "temperature": 0.2,
        "max_context_chars": 8000,
        "metadata_text_key": "text",
        "metadata_source_key": "source",
        "system_prompt": DEFAULT_CHAT_SYSTEM_PROMPT,
        # Mongo
        "mongo_uri": "mongodb+srv://harshalvankudre_db_user:<db_password>@chatbot-1.acaznw5.mongodb.net/?appName=chatbot-1",
        "mongo_db": "rag_chat",
        # NEW knobs for RAG vs general fallback
        "allow_general_answers": True,
        "rag_min_context_chars": 600,  # if built context shorter than this → use general fallback
        "rag_min_matches": 1,          # minimum matches required
        "rag_min_score": 0.0,          # optional score gate
    }

class AppSettings(BaseModel):
    # OpenAI
    openai_api_key: str
    openai_base_url: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # Pinecone
    pinecone_api_key: str
    pinecone_index_name: Optional[str] = None
    pinecone_host: Optional[str] = None
    pinecone_namespace: Optional[str] = None

    # Retrieval / generation
    top_k: int = 5
    temperature: float = 0.2
    max_context_chars: int = 8000
    metadata_text_key: str = "text"
    metadata_source_key: str = "source"

    # Prompt
    system_prompt: str = DEFAULT_CHAT_SYSTEM_PROMPT

    # Mongo
    mongo_uri: Optional[str] = None
    mongo_db: str = "rag_chat"

    # RAG vs general fallback
    allow_general_answers: bool = True
    rag_min_context_chars: int = 600
    rag_min_matches: int = 1
    rag_min_score: float = 0.0

    # ---- Pydantic v2-friendly validation (no field_validator needed) ----
    @model_validator(mode="after")
    def _validate_required_keys(self):
        # Only enforce these when this config is actually used (e.g., in Chat/Test)
        # Strip and ensure non-empty
        if not (self.openai_api_key and self.openai_api_key.strip()):
            raise ValueError("openai_api_key cannot be blank")
        if not (self.pinecone_api_key and self.pinecone_api_key.strip()):
            raise ValueError("pinecone_api_key cannot be blank")
        # Normalize strings
        if self.openai_base_url:
            self.openai_base_url = self.openai_base_url.strip() or None
        if self.pinecone_index_name:
            self.pinecone_index_name = self.pinecone_index_name.strip() or None
        if self.pinecone_host:
            self.pinecone_host = self.pinecone_host.strip() or None
        if self.pinecone_namespace:
            self.pinecone_namespace = self.pinecone_namespace.strip() or None
        if self.mongo_uri:
            self.mongo_uri = self.mongo_uri.strip() or None
        self.mongo_db = (self.mongo_db or "rag_chat").strip()
        return self
