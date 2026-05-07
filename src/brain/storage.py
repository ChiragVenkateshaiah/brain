"""SQLite schema and CRUD for the corpus."""

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def connect(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS documents (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            path        TEXT UNIQUE NOT NULL,
            mtime       REAL NOT NULL,
            ingested_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            ord         INTEGER NOT NULL,
            text        TEXT NOT NULL,
            embedding   BLOB NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
    """)
    conn.commit()


def get_document(conn: sqlite3.Connection, path: str) -> tuple[int, float] | None:
    row = conn.execute(
        "SELECT id, mtime FROM documents WHERE path = ?", (path,)
    ).fetchone()
    return (row[0], row[1]) if row else None


def upsert_document(conn: sqlite3.Connection, path: str, mtime: float) -> int:
    ingested_at = datetime.now(timezone.utc).isoformat()
    row = conn.execute(
        """
        INSERT INTO documents (path, mtime, ingested_at)
        VALUES (?, ?, ?)
        ON CONFLICT(path) DO UPDATE SET
            mtime = excluded.mtime,
            ingested_at = excluded.ingested_at
        RETURNING id
        """,
        (path, mtime, ingested_at),
    ).fetchone()
    conn.commit()
    return row[0]


def delete_chunks_for_document(conn: sqlite3.Connection, doc_id: int) -> None:
    conn.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
    conn.commit()


def insert_chunks(
    conn: sqlite3.Connection,
    doc_id: int,
    chunks: list[tuple[str, np.ndarray]],
) -> int:
    rows = [
        (doc_id, ord_, text, embedding.astype(np.float32).tobytes())
        for ord_, (text, embedding) in enumerate(chunks)
    ]
    conn.executemany(
        "INSERT INTO chunks (document_id, ord, text, embedding) VALUES (?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    return len(rows)


def get_all_chunks_with_embeddings(
    conn: sqlite3.Connection,
) -> list[tuple[int, str, str, np.ndarray]]:
    """Return (chunk_id, document_path, chunk_text, embedding) for every chunk."""
    rows = conn.execute(
        """
        SELECT c.id, d.path, c.text, c.embedding
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        ORDER BY c.id
        """
    ).fetchall()
    return [(row[0], row[1], row[2], np.frombuffer(row[3], dtype=np.float32)) for row in rows]


def corpus_stats(conn: sqlite3.Connection, db_path: str) -> dict:
    from brain.embed import _default_model

    doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    approx_tokens = conn.execute(
        "SELECT COALESCE(SUM(LENGTH(text)), 0) FROM chunks"
    ).fetchone()[0] // 4
    disk_size = os.path.getsize(db_path) if os.path.exists(db_path) else 0

    return {
        "db_path": db_path,
        "documents": doc_count,
        "chunks": chunk_count,
        "approx_tokens": approx_tokens,
        "embed_model": _default_model(),
        "disk_size_bytes": disk_size,
    }
