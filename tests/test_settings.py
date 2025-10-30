"""Unit tests for the configuration models."""

from __future__ import annotations

import pathlib
import sys

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.settings import AppSettings, default_env  # noqa: E402
from models.settings import AppSettings, default_env


def _minimal_env() -> dict[str, str]:
    env = default_env()
    env.update(
        {
            "openai_api_key": "test-openai-key",
            "pinecone_api_key": "test-pinecone-key",
        }
    )
    return env


def test_app_settings_from_env_normalises_strings() -> None:
    """Ensure whitespace-only values are stripped when constructing settings."""
    env = _minimal_env()
    env.update(
        {
            "openai_base_url": " https://api.example.com ",
            "pinecone_index_name": "  index-name  ",
            "pinecone_index_names": ["  index-name  ", "second-index", "second-index"],
            "pinecone_namespace": " namespace ",
            "pinecone_host": " https://pc.example.com ",
            "mongo_uri": " mongodb://localhost ",
            "mongo_db": " rag_chat ",
        }
    )

    settings = AppSettings.from_env(env)

    assert settings.openai_base_url == "https://api.example.com"
    assert settings.pinecone_index_name == "index-name"
    assert settings.pinecone_index_names == ["index-name", "second-index"]
    assert settings.pinecone_namespace == "namespace"
    assert settings.pinecone_host == "https://pc.example.com"
    assert settings.mongo_uri == "mongodb://localhost"
    assert settings.mongo_db == "rag_chat"


def test_app_settings_requires_api_keys() -> None:
    """Verify that required API keys cannot be blank."""
    env = default_env()

    with pytest.raises(ValueError):
        AppSettings.from_env(env)


def test_app_settings_parses_string_index_list() -> None:
    """Ensure comma or newline separated index lists are normalised."""
    env = _minimal_env()
    env["pinecone_index_name"] = "primary"
    env["pinecone_index_names"] = "primary, secondary\n tertiary "

    settings = AppSettings.from_env(env)

    assert settings.pinecone_index_name == "primary"
    assert settings.pinecone_index_names == ["primary", "secondary", "tertiary"]
