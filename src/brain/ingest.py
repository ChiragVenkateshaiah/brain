"""Per-file ingest pipeline: parse → chunk → embed → store."""

import sqlite3
from pathlib import Path

from brain import embed, storage
from brain.chunker import chunk_markdown, chunk_text
from brain.parsers import ParseError, parse_html, parse_pdf

SUPPORTED_EXTENSIONS = frozenset({".md", ".txt", ".pdf", ".html", ".htm"})


def extract_text(path: Path) -> str:
    """Dispatch on extension. Raises ParseError or OSError."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(path)
    if suffix in {".html", ".htm"}:
        return parse_html(path)
    return path.read_text(encoding="utf-8", errors="replace")


def chunk_for(path: Path, text: str) -> list[str]:
    if path.suffix.lower() == ".md":
        return chunk_markdown(text)
    return chunk_text(text)


def ingest_file(conn: sqlite3.Connection, path: Path) -> int:
    """Ingest one file. Returns number of chunks inserted.

    Caller handles mtime-skip and progress reporting.
    Raises ParseError, OSError, or embed.EmbedError on failure.
    """
    abs_path = str(path.resolve())
    text = extract_text(path)
    chunks = chunk_for(path, text)

    if chunks:
        embeddings = embed.embed(chunks)
    else:
        embeddings = []

    existing = storage.get_document(conn, abs_path)
    if existing:
        storage.delete_chunks_for_document(conn, existing[0])
    doc_id = storage.upsert_document(conn, abs_path, path.stat().st_mtime)
    if chunks:
        storage.insert_chunks(conn, doc_id, list(zip(chunks, embeddings)))
    return len(chunks)
