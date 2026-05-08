"""Tests for brain.qa."""

import sqlite3
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from brain import storage
from brain.chunker import chunk_text
from brain.qa import Answer, QAError, ask


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.execute("PRAGMA foreign_keys = ON")
    storage.init_db(c)
    return c


def _fake_embed_zeros(texts, **kw):
    return [np.zeros(768, dtype=np.float32) for _ in texts]


def _seed_corpus(conn, files: dict[str, str]) -> None:
    for path, text in files.items():
        doc_id = storage.upsert_document(conn, path, 0.0)
        chunks = chunk_text(text)
        embeddings = [np.zeros(768, dtype=np.float32) for _ in chunks]
        storage.insert_chunks(conn, doc_id, list(zip(chunks, embeddings)))


def _make_chat_client(tokens: list[str]):
    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def chat(self, *, model, messages, stream=False, **kwargs):
            for tok in tokens:
                yield SimpleNamespace(message=SimpleNamespace(content=tok))

    return FakeClient


def test_ask_empty_corpus_returns_no_context_answer(conn, monkeypatch):
    chat_called = []

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def chat(self, **kwargs):
            chat_called.append(True)
            return iter([])

    monkeypatch.setattr("brain.qa.ollama.Client", FakeClient)
    monkeypatch.setattr("brain.search.embed.embed", _fake_embed_zeros)

    answer = ask(conn, "anything?")
    assert answer.citations == []
    text = "".join(answer.text_stream)
    assert "don't know" in text.lower()
    assert not chat_called


def test_ask_builds_context_and_citations(conn, monkeypatch):
    _seed_corpus(conn, {
        "/a/file1.md": "alpha content alpha",
        "/b/file2.md": "beta content beta",
    })

    captured = {}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def chat(self, *, model, messages, stream=False, **kwargs):
            captured["messages"] = messages
            return iter([SimpleNamespace(message=SimpleNamespace(content="answer"))])

    monkeypatch.setattr("brain.qa.ollama.Client", FakeClient)
    monkeypatch.setattr("brain.search.embed.embed", _fake_embed_zeros)

    answer = ask(conn, "what is alpha?", top_k=5)
    list(answer.text_stream)  # consume

    assert set(answer.citations) == {"/a/file1.md", "/b/file2.md"}

    system_content = captured["messages"][0]["content"]
    assert "[1]" in system_content
    assert "[2]" in system_content


def test_ask_dedupes_citations_from_same_file(conn, monkeypatch):
    # Two chunks from the same file should share one citation number.
    _seed_corpus(conn, {
        "/a/file1.md": "word " * 600,  # long enough to produce multiple chunks
    })

    captured = {}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def chat(self, *, model, messages, stream=False, **kwargs):
            captured["messages"] = messages
            return iter([SimpleNamespace(message=SimpleNamespace(content="ok"))])

    monkeypatch.setattr("brain.qa.ollama.Client", FakeClient)
    monkeypatch.setattr("brain.search.embed.embed", _fake_embed_zeros)

    answer = ask(conn, "word?", top_k=10)
    list(answer.text_stream)

    assert answer.citations == ["/a/file1.md"]
    assert "[2]" not in captured["messages"][0]["content"]


def test_ask_stream_yields_tokens_in_order(conn, monkeypatch):
    _seed_corpus(conn, {"/doc.md": "hello world " * 50})

    tokens = ["Hello", ", ", "world", "!"]
    monkeypatch.setattr("brain.qa.ollama.Client", _make_chat_client(tokens))
    monkeypatch.setattr("brain.search.embed.embed", _fake_embed_zeros)

    answer = ask(conn, "say hello")
    result = "".join(answer.text_stream)
    assert result == "Hello, world!"


def test_ask_raises_qaerror_on_connection_failure(conn, monkeypatch):
    _seed_corpus(conn, {"/doc.md": "some content " * 50})

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def chat(self, **kwargs):
            def _failing():
                raise httpx.ConnectError("refused")
                yield  # noqa: unreachable — makes this a generator

            return _failing()

    monkeypatch.setattr("brain.qa.ollama.Client", FakeClient)
    monkeypatch.setattr("brain.search.embed.embed", _fake_embed_zeros)

    answer = ask(conn, "question")
    with pytest.raises(QAError):
        list(answer.text_stream)


def test_ask_uses_env_model_override(conn, monkeypatch):
    _seed_corpus(conn, {"/doc.md": "content " * 50})

    called_with = {}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def chat(self, *, model, messages, stream=False, **kwargs):
            called_with["model"] = model
            return iter([SimpleNamespace(message=SimpleNamespace(content="ok"))])

    monkeypatch.setattr("brain.qa.ollama.Client", FakeClient)
    monkeypatch.setattr("brain.search.embed.embed", _fake_embed_zeros)
    monkeypatch.setenv("BRAIN_CHAT_MODEL", "env-model")

    answer = ask(conn, "q")
    list(answer.text_stream)
    assert called_with["model"] == "env-model"


def test_ask_explicit_model_overrides_env(conn, monkeypatch):
    _seed_corpus(conn, {"/doc.md": "content " * 50})

    called_with = {}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def chat(self, *, model, messages, stream=False, **kwargs):
            called_with["model"] = model
            return iter([SimpleNamespace(message=SimpleNamespace(content="ok"))])

    monkeypatch.setattr("brain.qa.ollama.Client", FakeClient)
    monkeypatch.setattr("brain.search.embed.embed", _fake_embed_zeros)
    monkeypatch.setenv("BRAIN_CHAT_MODEL", "env-model")

    answer = ask(conn, "q", model="explicit-model")
    list(answer.text_stream)
    assert called_with["model"] == "explicit-model"
