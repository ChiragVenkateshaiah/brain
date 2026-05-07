"""Tests for brain.embed."""

from types import SimpleNamespace

import httpx
import numpy as np
import ollama
import pytest

from brain.embed import EmbedError, _default_model, embed


def _fake_client(embeddings: list[list[float]]):
    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def embed(self, *, model, input, **kwargs):
            return SimpleNamespace(embeddings=embeddings)

    return FakeClient


def test_default_model_fallback(monkeypatch):
    monkeypatch.delenv("BRAIN_EMBED_MODEL", raising=False)
    assert _default_model() == "nomic-embed-text"


def test_default_model_from_env(monkeypatch):
    monkeypatch.setenv("BRAIN_EMBED_MODEL", "my-model")
    assert _default_model() == "my-model"


def test_embed_uses_default_model(monkeypatch):
    monkeypatch.delenv("BRAIN_EMBED_MODEL", raising=False)
    called_with = {}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def embed(self, *, model, input, **kwargs):
            called_with["model"] = model
            return SimpleNamespace(embeddings=[[0.1] * 3])

    monkeypatch.setattr("brain.embed.ollama.Client", FakeClient)
    embed(["hello"])
    assert called_with["model"] == "nomic-embed-text"


def test_embed_env_var_overrides_default(monkeypatch):
    monkeypatch.setenv("BRAIN_EMBED_MODEL", "env-model")
    called_with = {}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def embed(self, *, model, input, **kwargs):
            called_with["model"] = model
            return SimpleNamespace(embeddings=[[0.1] * 3])

    monkeypatch.setattr("brain.embed.ollama.Client", FakeClient)
    embed(["hello"])
    assert called_with["model"] == "env-model"


def test_embed_explicit_model_overrides_env(monkeypatch):
    monkeypatch.setenv("BRAIN_EMBED_MODEL", "env-model")
    called_with = {}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def embed(self, *, model, input, **kwargs):
            called_with["model"] = model
            return SimpleNamespace(embeddings=[[0.1] * 3])

    monkeypatch.setattr("brain.embed.ollama.Client", FakeClient)
    embed(["hello"], model="explicit-model")
    assert called_with["model"] == "explicit-model"


def test_embed_returns_ndarray_per_text(monkeypatch):
    fake = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    monkeypatch.setattr("brain.embed.ollama.Client", _fake_client(fake))
    result = embed(["a", "b"])
    assert len(result) == 2
    assert all(isinstance(r, np.ndarray) for r in result)
    assert all(r.dtype == np.float32 for r in result)
    np.testing.assert_array_almost_equal(result[0], [0.1, 0.2, 0.3])


def test_embed_empty_list_returns_empty(monkeypatch):
    called = []

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def embed(self, **kwargs):
            called.append(True)
            return SimpleNamespace(embeddings=[])

    monkeypatch.setattr("brain.embed.ollama.Client", FakeClient)
    result = embed([])
    assert result == []
    assert not called


def test_embed_wraps_response_error(monkeypatch):
    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def embed(self, **kwargs):
            raise ollama.ResponseError("model not found")

    monkeypatch.setattr("brain.embed.ollama.Client", FakeClient)
    with pytest.raises(EmbedError):
        embed(["hello"])


def test_embed_wraps_connect_error(monkeypatch):
    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def embed(self, **kwargs):
            raise httpx.ConnectError("connection refused")

    monkeypatch.setattr("brain.embed.ollama.Client", FakeClient)
    with pytest.raises(EmbedError):
        embed(["hello"])
