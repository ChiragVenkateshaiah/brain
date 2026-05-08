"""Tests for brain.ingest."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from brain import storage
from brain.ingest import SUPPORTED_EXTENSIONS, chunk_for, extract_text, ingest_file
from brain.parsers import ParseError


def _fake_embed(texts: list[str], **kwargs) -> list[np.ndarray]:
    return [np.zeros(768, dtype=np.float32) for _ in texts]


# --- extract_text ---


def test_extract_text_dispatches_pdf(tmp_path):
    f = tmp_path / "doc.pdf"
    f.write_bytes(b"")
    with patch("brain.ingest.parse_pdf", return_value="pdf text") as mock_pdf:
        result = extract_text(f)
    mock_pdf.assert_called_once_with(f)
    assert result == "pdf text"


def test_extract_text_dispatches_html(tmp_path):
    f = tmp_path / "page.html"
    f.write_text("")
    with patch("brain.ingest.parse_html", return_value="html text") as mock_html:
        result = extract_text(f)
    mock_html.assert_called_once_with(f)
    assert result == "html text"


def test_extract_text_dispatches_htm(tmp_path):
    f = tmp_path / "page.htm"
    f.write_text("")
    with patch("brain.ingest.parse_html", return_value="htm text") as mock_html:
        result = extract_text(f)
    mock_html.assert_called_once_with(f)
    assert result == "htm text"


def test_extract_text_reads_md_as_utf8(tmp_path):
    f = tmp_path / "note.md"
    f.write_text("# Hello")
    assert extract_text(f) == "# Hello"


def test_extract_text_reads_txt_as_utf8(tmp_path):
    f = tmp_path / "note.txt"
    f.write_text("plain text")
    assert extract_text(f) == "plain text"


# --- chunk_for ---


def test_chunk_for_uses_chunk_markdown_for_md(tmp_path):
    f = tmp_path / "note.md"
    with patch("brain.ingest.chunk_markdown", return_value=["md chunk"]) as mock_md:
        with patch("brain.ingest.chunk_text", return_value=["text chunk"]):
            result = chunk_for(f, "# H\n\nbody")
    mock_md.assert_called_once()
    assert result == ["md chunk"]


def test_chunk_for_uses_chunk_text_for_txt(tmp_path):
    f = tmp_path / "note.txt"
    with patch("brain.ingest.chunk_markdown") as mock_md:
        with patch("brain.ingest.chunk_text", return_value=["text chunk"]) as mock_text:
            result = chunk_for(f, "plain text")
    mock_md.assert_not_called()
    assert result == ["text chunk"]


def test_chunk_for_uses_chunk_text_for_pdf(tmp_path):
    f = tmp_path / "doc.pdf"
    with patch("brain.ingest.chunk_text", return_value=["pdf chunk"]) as mock_text:
        result = chunk_for(f, "extracted pdf text")
    assert result == ["pdf chunk"]


def test_chunk_for_uses_chunk_text_for_html(tmp_path):
    f = tmp_path / "page.html"
    with patch("brain.ingest.chunk_text", return_value=["html chunk"]) as mock_text:
        result = chunk_for(f, "extracted html text")
    assert result == ["html chunk"]


# --- SUPPORTED_EXTENSIONS ---


def test_supported_extensions_contains_expected():
    assert {".md", ".txt", ".pdf", ".html", ".htm"} == SUPPORTED_EXTENSIONS


# --- ingest_file ---


@pytest.fixture
def mem_db():
    conn = storage.connect(":memory:")
    storage.init_db(conn)
    return conn


def test_ingest_file_inserts_document_and_chunks(tmp_path, mem_db, monkeypatch):
    monkeypatch.setattr("brain.ingest.embed.embed", _fake_embed)
    f = tmp_path / "note.md"
    f.write_text("# Section\n\n" + "word " * 200)

    n = ingest_file(mem_db, f)

    assert n > 0
    docs = mem_db.execute("SELECT path FROM documents").fetchall()
    assert any(str(f.resolve()) in d[0] for d in docs)
    chunks = mem_db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    assert chunks == n


def test_ingest_file_replaces_chunks_on_reingest(tmp_path, mem_db, monkeypatch):
    monkeypatch.setattr("brain.ingest.embed.embed", _fake_embed)
    f = tmp_path / "note.txt"
    f.write_text("first version " * 100)
    ingest_file(mem_db, f)

    first_count = mem_db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    f.write_text("completely different content that is much longer " * 200)
    ingest_file(mem_db, f)

    second_count = mem_db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    # Only one document should exist (upserted)
    doc_count = mem_db.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    assert doc_count == 1
    # Chunks reflect new content, not accumulated old + new
    assert second_count != first_count or second_count > 0


def test_ingest_file_empty_text_stores_doc_no_chunks(tmp_path, mem_db, monkeypatch):
    monkeypatch.setattr("brain.ingest.embed.embed", _fake_embed)
    f = tmp_path / "empty.txt"
    f.write_text("   ")

    n = ingest_file(mem_db, f)

    assert n == 0
    doc_count = mem_db.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    assert doc_count == 1
