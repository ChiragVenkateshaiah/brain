"""Tests for brain.search."""

import sqlite3

import numpy as np
import pytest

from brain import storage
from brain.embed import EmbedError
from brain.search import SearchResult, search


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.execute("PRAGMA foreign_keys = ON")
    storage.init_db(c)
    return c


def _insert(conn, path: str, chunks: list[tuple[str, np.ndarray]]) -> int:
    doc_id = storage.upsert_document(conn, path, 1.0)
    storage.insert_chunks(conn, doc_id, chunks)
    return doc_id


def _unit(v: list[float]) -> np.ndarray:
    a = np.array(v, dtype=np.float32)
    return (a / np.linalg.norm(a)).astype(np.float32)


def test_search_empty_corpus_returns_empty(conn, monkeypatch):
    monkeypatch.setattr("brain.search.embed.embed", lambda texts, **kw: [np.zeros(768, dtype=np.float32)])
    assert search(conn, "anything") == []


def test_search_top_k_orders_by_cosine(conn, monkeypatch):
    # chunk 2 is most similar to query
    e1 = _unit([1.0, 0.0, 0.0])
    e2 = _unit([0.9, 0.4, 0.0])   # closest to query
    e3 = _unit([0.0, 1.0, 0.0])
    e4 = _unit([0.0, 0.0, 1.0])
    emb = lambda dim: np.pad(e1, (0, dim - len(e1)), constant_values=0.0)

    # Pad to 768 dims
    def pad(v):
        return np.pad(v, (0, 768 - len(v)), constant_values=0.0).astype(np.float32)

    doc_id = storage.upsert_document(conn, "/doc.md", 1.0)
    storage.insert_chunks(conn, doc_id, [
        ("chunk one",   pad(e1)),
        ("chunk two",   pad(e2)),
        ("chunk three", pad(e3)),
        ("chunk four",  pad(e4)),
    ])

    query_vec = pad(_unit([0.85, 0.5, 0.0]))
    monkeypatch.setattr("brain.search.embed.embed", lambda texts, **kw: [query_vec])

    results = search(conn, "q", top_k=4)
    assert results[0].text == "chunk two"


def test_search_respects_top_k_limit(conn, monkeypatch):
    e = np.zeros(768, dtype=np.float32)
    _insert(conn, "/a.md", [(f"chunk {i}", e) for i in range(5)])
    monkeypatch.setattr("brain.search.embed.embed", lambda texts, **kw: [e])
    results = search(conn, "q", top_k=2)
    assert len(results) == 2


def test_search_top_k_greater_than_n_returns_all(conn, monkeypatch):
    e = np.zeros(768, dtype=np.float32)
    _insert(conn, "/a.md", [(f"chunk {i}", e) for i in range(3)])
    monkeypatch.setattr("brain.search.embed.embed", lambda texts, **kw: [e])
    results = search(conn, "q", top_k=10)
    assert len(results) == 3


def test_search_scores_are_true_cosine(conn, monkeypatch):
    v = _unit([1.0, 0.0, 0.0])
    orth = _unit([0.0, 1.0, 0.0])

    def pad(x):
        return np.pad(x, (0, 768 - len(x)), constant_values=0.0).astype(np.float32)

    doc_id = storage.upsert_document(conn, "/a.md", 1.0)
    storage.insert_chunks(conn, doc_id, [("same", pad(v)), ("orth", pad(orth))])

    query_vec = pad(v)
    monkeypatch.setattr("brain.search.embed.embed", lambda texts, **kw: [query_vec])

    results = search(conn, "q", top_k=2)
    scores = {r.text: r.score for r in results}
    assert scores["same"] == pytest.approx(1.0, abs=1e-5)
    assert scores["orth"] == pytest.approx(0.0, abs=1e-5)
    assert all(-1.0 <= r.score <= 1.0 for r in results)


def test_search_returns_frozen_dataclass(conn, monkeypatch):
    e = np.ones(768, dtype=np.float32) / 768**0.5
    _insert(conn, "/a.md", [("text", e)])
    monkeypatch.setattr("brain.search.embed.embed", lambda texts, **kw: [e])
    results = search(conn, "q")
    assert isinstance(results[0], SearchResult)
    with pytest.raises(Exception):
        results[0].score = 0.0  # frozen


def test_search_propagates_embed_error(conn, monkeypatch):
    e = np.zeros(768, dtype=np.float32)
    _insert(conn, "/a.md", [("text", e)])

    def boom(texts, **kw):
        raise EmbedError("connection refused")

    monkeypatch.setattr("brain.search.embed.embed", boom)
    with pytest.raises(EmbedError):
        search(conn, "q")
