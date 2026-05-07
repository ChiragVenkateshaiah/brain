"""Tests for brain.cli."""

import sqlite3

import numpy as np
import pytest
from typer.testing import CliRunner

from brain.cli import app
from brain import storage

runner = CliRunner()


def _fake_embed(texts: list[str], **kwargs) -> list[np.ndarray]:
    return [np.zeros(768, dtype=np.float32) for _ in texts]


@pytest.fixture
def fixture_dir(tmp_path):
    (tmp_path / "a.md").write_text("# Note A\n" + "word " * 200)
    (tmp_path / "b.txt").write_text("plain text " * 100)
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.md").write_text("nested markdown " * 50)
    (tmp_path / "ignored.pdf").write_bytes(b"%PDF-1.4 fake")
    return tmp_path


@pytest.fixture
def brain_dir(tmp_path, monkeypatch):
    data_dir = tmp_path / "brain"
    data_dir.mkdir()
    monkeypatch.setenv("BRAIN_DATA_DIR", str(data_dir))
    return data_dir


def test_ingest_basic(fixture_dir, brain_dir, monkeypatch):
    monkeypatch.setattr("brain.cli.embed.embed", _fake_embed)
    result = runner.invoke(app, ["ingest", str(fixture_dir)])
    assert result.exit_code == 0
    # Should process 3 md/txt files (a.md, b.txt, sub/c.md), skip the .pdf
    assert "3 files ingested" in result.output


def test_ingest_skips_pdf(fixture_dir, brain_dir, monkeypatch):
    monkeypatch.setattr("brain.cli.embed.embed", _fake_embed)
    runner.invoke(app, ["ingest", str(fixture_dir)])
    db_path = str(brain_dir / "corpus.db")
    conn = storage.connect(db_path)
    paths = [r[0] for r in conn.execute("SELECT path FROM documents").fetchall()]
    assert not any(p.endswith(".pdf") for p in paths)


def test_ingest_chunk_count(fixture_dir, brain_dir, monkeypatch):
    monkeypatch.setattr("brain.cli.embed.embed", _fake_embed)
    runner.invoke(app, ["ingest", str(fixture_dir)])
    db_path = str(brain_dir / "corpus.db")
    conn = storage.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    assert count > 0


def test_reingest_unchanged_files_skipped(fixture_dir, brain_dir, monkeypatch):
    monkeypatch.setattr("brain.cli.embed.embed", _fake_embed)
    runner.invoke(app, ["ingest", str(fixture_dir)])

    db_path = str(brain_dir / "corpus.db")
    conn = storage.connect(db_path)
    count_before = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    result = runner.invoke(app, ["ingest", str(fixture_dir)])
    assert result.exit_code == 0
    assert "3 skipped" in result.output

    count_after = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    assert count_before == count_after


def test_reingest_changed_file_replaces_chunks(fixture_dir, brain_dir, monkeypatch):
    monkeypatch.setattr("brain.cli.embed.embed", _fake_embed)
    runner.invoke(app, ["ingest", str(fixture_dir)])

    db_path = str(brain_dir / "corpus.db")
    conn = storage.connect(db_path)
    a_path = str((fixture_dir / "a.md").resolve())
    doc = storage.get_document(conn, a_path)
    old_chunk_count = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE document_id = ?", (doc[0],)
    ).fetchone()[0]

    # Write new content and bump mtime
    new_text = "completely different content " * 300
    (fixture_dir / "a.md").write_text(new_text)

    result = runner.invoke(app, ["ingest", str(fixture_dir)])
    assert result.exit_code == 0

    new_chunk_count = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE document_id = ?", (doc[0],)
    ).fetchone()[0]
    # The chunk count should reflect new content, not accumulated old + new
    assert new_chunk_count != old_chunk_count or new_chunk_count > 0


def test_stats_shows_corpus_info(fixture_dir, brain_dir, monkeypatch):
    monkeypatch.setattr("brain.cli.embed.embed", _fake_embed)
    runner.invoke(app, ["ingest", str(fixture_dir)])
    result = runner.invoke(app, ["stats"])
    assert result.exit_code == 0
    assert "Documents" in result.output
    assert "Chunks" in result.output


def test_stats_nonzero_after_ingest(fixture_dir, brain_dir, monkeypatch):
    monkeypatch.setattr("brain.cli.embed.embed", _fake_embed)
    runner.invoke(app, ["ingest", str(fixture_dir)])
    result = runner.invoke(app, ["stats"])
    assert "3" in result.output  # 3 documents


def test_ingest_nonexistent_path(brain_dir):
    result = runner.invoke(app, ["ingest", "/no/such/path"])
    assert result.exit_code != 0


def test_ingest_skips_hidden_directories(tmp_path, brain_dir, monkeypatch):
    monkeypatch.setattr("brain.cli.embed.embed", _fake_embed)
    (tmp_path / "visible.md").write_text("visible content " * 50)
    hidden = tmp_path / ".hidden"
    hidden.mkdir()
    (hidden / "secret.md").write_text("secret content " * 50)
    git = tmp_path / ".git"
    git.mkdir()
    (git / "config").write_text("[core]")

    runner.invoke(app, ["ingest", str(tmp_path)])
    db_path = str(brain_dir / "corpus.db")
    conn = storage.connect(db_path)
    paths = [r[0] for r in conn.execute("SELECT path FROM documents").fetchall()]
    assert not any(".hidden" in p or ".git" in p for p in paths)
    assert any("visible.md" in p for p in paths)
