"""Thin wrappers around OpenAI and Pinecone clients."""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from collections.abc import Iterator
from typing import Any

from openai import OpenAI
from pinecone import Index, Pinecone
from pinecone.core.client.exceptions import PineconeException

logger = logging.getLogger(__name__)


def get_openai_client(api_key: str, base_url: str | None = None) -> OpenAI:
    """Return an OpenAI client configured with the provided credentials."""
    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url.strip()
    return OpenAI(**kwargs)


def get_pinecone_index(api_key: str, host: str | None, index_name: str | None) -> Index:
    """Return a Pinecone index using either the HTTP host or the index name."""
    pc = Pinecone(api_key=api_key)
    if host and host.strip():
        return pc.Index(host=host.strip())
    if index_name and index_name.strip():
        return pc.Index(index_name.strip())
    msg = "Provide Pinecone index host or index name."
    raise ValueError(msg)


def get_pinecone_indexes(
    api_key: str,
    host: str | None,
    index_name: str | None,
    extra_index_names: Sequence[str] | None,
) -> list[tuple[str, Index]]:
    """Return Pinecone index handles for retrieval across multiple indexes."""
    pc = Pinecone(api_key=api_key)
    if host and host.strip():
        host_clean = host.strip()
        return [(host_clean, pc.Index(host=host_clean))]

    names: list[str] = []
    if index_name and index_name.strip():
        names.append(index_name.strip())
    for extra in extra_index_names or []:
        extra_clean = str(extra).strip()
        if extra_clean and extra_clean not in names:
            names.append(extra_clean)

    if not names:
        msg = "Provide at least one Pinecone index name or host."
        raise ValueError(msg)

    return [(name, pc.Index(name)) for name in names]


def embed_texts(client: OpenAI, texts: list[str], model: str) -> list[list[float]]:
    """Generate embeddings for ``texts`` using the supplied model."""
    resp = client.embeddings.create(model=model, input=texts)
    return [list(map(float, data.embedding)) for data in resp.data]


def retrieve_chunks(
    index: Index, query_vec: list[float], top_k: int, namespace: str | None
) -> dict[str, Any]:
    """Query Pinecone for the closest ``top_k`` vectors."""
    return index.query(
        vector=[float(x) for x in query_vec],
        top_k=int(top_k),
        include_values=False,
        include_metadata=True,
        namespace=(namespace or "__default__"),
    )


def retrieve_from_indexes(
    indexes: Sequence[tuple[str, Index]],
    query_vec: list[float],
    top_k: int,
    namespace: str | None,
) -> list[dict[str, Any]]:
    """Collect matches from all provided indexes and return the top results."""
    all_matches: list[dict[str, Any]] = []
    for label, index in indexes:
        try:
            response = retrieve_chunks(index, query_vec, top_k, namespace)
        except PineconeException as exc:  # pragma: no cover - defensive runtime path
            logger.warning("Pinecone query failed for index %s", label, exc_info=exc)
            continue
        matches_obj = getattr(response, "matches", None)
        if matches_obj is None and isinstance(response, dict):
            matches_obj = response.get("matches", [])
        if matches_obj is None:
            matches_obj = []
        for raw in matches_obj:
            if isinstance(raw, dict):
                match = raw
            else:
                match = {
                    "id": getattr(raw, "id", None),
                    "score": getattr(raw, "score", None),
                    "metadata": getattr(raw, "metadata", {}) or {},
                }
            metadata = match.setdefault("metadata", {})
            metadata.setdefault("index_name", label)
            all_matches.append(match)

    def _score(match: dict[str, Any]) -> float:
        try:
            return float(match.get("score", 0) or 0)
        except (TypeError, ValueError):  # pragma: no cover - defensive fallback
            return 0.0

    all_matches.sort(key=_score, reverse=True)
    return all_matches[: int(top_k)]


def stream_chat_completion(
    client: OpenAI, model: str, messages: list[dict[str, Any]], temperature: float
) -> Iterator[str]:
    """Yield chunks from a streaming chat completion response."""
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        stream=True,
    )
    for event in stream:
        delta = event.choices[0].delta.content or ""
        if delta:
            yield delta
