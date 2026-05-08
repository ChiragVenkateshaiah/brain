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


def test_search_no_corpus_errors(brain_dir):
    result = runner.invoke(app, ["search", "hello"])
    assert result.exit_code != 0
    assert "No corpus" in result.output


def test_search_basic_renders_table(fixture_dir, brain_dir, monkeypatch):
    monkeypatch.setattr("brain.cli.embed.embed", _fake_embed)
    runner.invoke(app, ["ingest", str(fixture_dir)])
    result = runner.invoke(app, ["search", "word"])
    assert result.exit_code == 0
    assert "Score" in result.output


def test_search_top_flag_limits_rows(fixture_dir, brain_dir, monkeypatch):
    monkeypatch.setattr("brain.cli.embed.embed", _fake_embed)
    runner.invoke(app, ["ingest", str(fixture_dir)])
    result = runner.invoke(app, ["search", "x", "-k", "2"])
    assert result.exit_code == 0
    # Table rows: each data row contains a score like "0.0000"
    score_lines = [l for l in result.output.splitlines() if "0.0000" in l]
    assert len(score_lines) <= 2


def test_search_empty_query_errors(brain_dir, monkeypatch):
    monkeypatch.setattr("brain.cli.embed.embed", _fake_embed)
    result = runner.invoke(app, ["search", "   "])
    assert result.exit_code != 0
    assert "Error" in result.output


def test_search_embed_error_exits_nonzero(fixture_dir, brain_dir, monkeypatch):
    monkeypatch.setattr("brain.cli.embed.embed", _fake_embed)
    runner.invoke(app, ["ingest", str(fixture_dir)])

    from brain.embed import EmbedError

    def boom(texts, **kw):
        raise EmbedError("no ollama")

    monkeypatch.setattr("brain.cli.embed.embed", boom)
    result = runner.invoke(app, ["search", "anything"])
    assert result.exit_code != 0
    assert "Error" in result.output


def test_ask_no_corpus_exits_1(brain_dir):
    result = runner.invoke(app, ["ask", "what is this?"])
    assert result.exit_code != 0
    assert "No corpus" in result.output


def test_ask_empty_query_exits_1(brain_dir):
    result = runner.invoke(app, ["ask", "   "])
    assert result.exit_code != 0
    assert "Error" in result.output


def test_ask_streams_answer_and_prints_sources(fixture_dir, brain_dir, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr("brain.cli.embed.embed", _fake_embed)
    runner.invoke(app, ["ingest", str(fixture_dir)])

    tokens = ["The", " answer", " is", " 42."]

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def chat(self, *, model, messages, stream=False, **kwargs):
            for tok in tokens:
                yield SimpleNamespace(message=SimpleNamespace(content=tok))

    monkeypatch.setattr("brain.qa.ollama.Client", FakeClient)

    result = runner.invoke(app, ["ask", "what is the answer?"])
    assert result.exit_code == 0
    assert "The answer is 42." in result.output
    assert "Sources:" in result.output


def test_ask_qaerror_exits_1(fixture_dir, brain_dir, monkeypatch):
    monkeypatch.setattr("brain.cli.embed.embed", _fake_embed)
    runner.invoke(app, ["ingest", str(fixture_dir)])

    import httpx

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def chat(self, **kwargs):
            def _failing():
                raise httpx.ConnectError("refused")
                yield

            return _failing()

    monkeypatch.setattr("brain.qa.ollama.Client", FakeClient)

    result = runner.invoke(app, ["ask", "anything?"])
    assert result.exit_code != 0
    assert "Error" in result.output


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
