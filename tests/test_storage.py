"""Tests for brain.storage."""

import sqlite3

import numpy as np
import pytest

from brain import storage


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.execute("PRAGMA foreign_keys = ON")
    storage.init_db(c)
    return c


def test_init_db_creates_tables(conn):
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert "documents" in tables
    assert "chunks" in tables


def test_init_db_creates_index(conn):
    indexes = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()
    }
    assert "idx_chunks_document" in indexes


def test_init_db_is_idempotent(conn):
    storage.init_db(conn)  # second call should not raise


def test_upsert_document_returns_id(conn):
    doc_id = storage.upsert_document(conn, "/a/b.md", 1234.0)
    assert isinstance(doc_id, int)
    assert doc_id > 0


def test_upsert_document_conflict_updates_mtime(conn):
    id1 = storage.upsert_document(conn, "/a/b.md", 1000.0)
    id2 = storage.upsert_document(conn, "/a/b.md", 2000.0)
    assert id1 == id2
    row = conn.execute("SELECT mtime FROM documents WHERE id = ?", (id1,)).fetchone()
    assert row[0] == 2000.0


def test_get_document_returns_none_when_missing(conn):
    assert storage.get_document(conn, "/no/such/file.md") is None


def test_get_document_returns_id_and_mtime(conn):
    doc_id = storage.upsert_document(conn, "/a/b.md", 999.5)
    result = storage.get_document(conn, "/a/b.md")
    assert result == (doc_id, 999.5)


def test_insert_chunks_returns_count(conn):
    doc_id = storage.upsert_document(conn, "/x.md", 1.0)
    emb = np.zeros(768, dtype=np.float32)
    count = storage.insert_chunks(conn, doc_id, [("hello", emb), ("world", emb)])
    assert count == 2


def test_insert_chunks_assigns_ord_in_order(conn):
    doc_id = storage.upsert_document(conn, "/x.md", 1.0)
    emb = np.zeros(768, dtype=np.float32)
    storage.insert_chunks(conn, doc_id, [("a", emb), ("b", emb), ("c", emb)])
    rows = conn.execute(
        "SELECT ord, text FROM chunks WHERE document_id = ? ORDER BY ord", (doc_id,)
    ).fetchall()
    assert [(r[0], r[1]) for r in rows] == [(0, "a"), (1, "b"), (2, "c")]


def test_insert_chunks_roundtrips_embedding(conn):
    doc_id = storage.upsert_document(conn, "/x.md", 1.0)
    emb = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    storage.insert_chunks(conn, doc_id, [("text", emb)])
    blob = conn.execute("SELECT embedding FROM chunks WHERE document_id = ?", (doc_id,)).fetchone()[0]
    recovered = np.frombuffer(blob, dtype=np.float32)
    np.testing.assert_array_almost_equal(recovered, emb)


def test_delete_chunks_for_document(conn):
    doc_id = storage.upsert_document(conn, "/x.md", 1.0)
    emb = np.zeros(768, dtype=np.float32)
    storage.insert_chunks(conn, doc_id, [("a", emb), ("b", emb)])
    storage.delete_chunks_for_document(conn, doc_id)
    count = conn.execute("SELECT COUNT(*) FROM chunks WHERE document_id = ?", (doc_id,)).fetchone()[0]
    assert count == 0


def test_cascade_delete_removes_chunks(conn):
    doc_id = storage.upsert_document(conn, "/x.md", 1.0)
    emb = np.zeros(768, dtype=np.float32)
    storage.insert_chunks(conn, doc_id, [("a", emb)])
    conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    conn.commit()
    count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    assert count == 0


def test_foreign_keys_enabled(conn):
    # Confirm the connection has foreign keys on — cascade delete only works with it
    result = conn.execute("PRAGMA foreign_keys").fetchone()[0]
    assert result == 1


def test_corpus_stats_empty_db(conn, tmp_path, monkeypatch):
    monkeypatch.setenv("BRAIN_EMBED_MODEL", "test-model")
    db_path = str(tmp_path / "corpus.db")
    # Write something so the file exists for getsize
    real_conn = storage.connect(db_path)
    storage.init_db(real_conn)
    stats = storage.corpus_stats(real_conn, db_path)
    assert stats["documents"] == 0
    assert stats["chunks"] == 0
    assert stats["approx_tokens"] == 0
    assert stats["embed_model"] == "test-model"
    assert stats["disk_size_bytes"] >= 0
    real_conn.close()


def test_corpus_stats_populated_db(conn, tmp_path, monkeypatch):
    monkeypatch.setenv("BRAIN_EMBED_MODEL", "nomic-embed-text")
    db_path = str(tmp_path / "corpus.db")
    real_conn = storage.connect(db_path)
    storage.init_db(real_conn)
    doc_id = storage.upsert_document(real_conn, "/a.md", 1.0)
    emb = np.zeros(768, dtype=np.float32)
    storage.insert_chunks(real_conn, doc_id, [("hello world", emb), ("foo bar", emb)])
    stats = storage.corpus_stats(real_conn, db_path)
    assert stats["documents"] == 1
    assert stats["chunks"] == 2
    assert stats["approx_tokens"] == (len("hello world") + len("foo bar")) // 4
    real_conn.close()
