# brain

A personal "second brain" CLI: ingests your documents, embeds them locally,
and answers questions with citations. All local, no API keys.

## Stack

- Python 3.12+
- `httpx` — HTTP client (sync; only async if a feature genuinely needs it)
- `typer` — CLI framework
- `rich` — terminal output formatting (tables, progress bars, streaming)
- `sqlite3` — stdlib, no ORM
- `numpy` — vector math (dot products, normalization)
- `ollama` — local LLM + embedding model client
- `beautifulsoup4` — HTML parsing (later ships)
- `pypdf` — PDF parsing (later ships)
- `pytest` — testing

## Conventions

- Type hints on all public functions (no `Any` unless justified)
- Functions over classes unless state is clearly required
- Errors: raise specific exceptions, don't swallow with bare `except`
- No `print()` in library code — use rich's console or return values
- Tests live in `tests/` mirroring `src/` structure
- All tests use `:memory:` databases or mocked I/O — no network or disk
  in CI

## Models

- **Embedding model:** `nomic-embed-text` (768-dim, ~270MB, fast)
- **Chat model:** `llama3.2:3b` (small, fast, runs on most laptops)

Both run via local Ollama daemon at `http://localhost:11434`. Override with
env vars `BRAIN_EMBED_MODEL` and `BRAIN_CHAT_MODEL`.

## Storage

- All data under `~/.brain/` by default (override with `BRAIN_DATA_DIR`)
- Single SQLite file: `~/.brain/corpus.db`
- Embeddings stored as `BLOB` columns (numpy arrays serialized via
  `np.frombuffer`)
- No separate vector index file; index is built in-memory at search time
  (we'll address scale at Ship 2 if needed)

## Project layout

```
src/
  brain/
    __init__.py
    storage.py      # SQLite schema and CRUD
    chunker.py      # Document → chunks
    ingest.py       # File walker + chunker + embedder pipeline
    embed.py        # Ollama embedding wrapper
    search.py       # Vector similarity (Ship 2)
    qa.py           # Retrieval + LLM Q&A (Ship 3)
    cli.py          # typer entry point
tests/
  test_storage.py
  test_chunker.py
  test_ingest.py
  ...
```

## Hard rules — don't

- Don't add an ORM (SQLAlchemy, Peewee). Raw `sqlite3` is fine at this scale
- Don't add async unless a feature genuinely needs concurrency
- Don't add a vector database (Chroma, Qdrant, FAISS). The lesson is to
  understand what those tools do — a real index will go directly into
  SQLite if needed, with `sqlite-vec` as the reach goal
- Don't catch broad `Exception` — be specific or let it propagate
- Don't reach for caching layers, queues, background jobs, web frameworks.
  This is a single-user local CLI

## Running

```bash
# Tests
pytest

# CLI (during development, before pip install -e .)
python -m brain <command>

# After install
brain ingest ~/Documents/notes/
brain stats
brain search "context window"
brain ask "what did I conclude about X?"
```

## Environment

- Ollama daemon must be running at `http://localhost:11434`
- Required models: `ollama pull nomic-embed-text` and
  `ollama pull llama3.2:3b`
- `BRAIN_DATA_DIR` — optional, defaults to `~/.brain/`
- `BRAIN_EMBED_MODEL` — optional, defaults to `nomic-embed-text`
- `BRAIN_CHAT_MODEL` — optional, defaults to `llama3.2:3b`

## Context-management notes for Claude

This project is being built to practice (a) agentic engineering with Claude
Code and (b) the context-management discipline from a prior project
(`linkstash`). The human will sometimes:

- `/clear` between ships even when context has headroom (task boundary
  matters more than token math)
- Spawn subagents for doc-reading or unfamiliar-API research (don't resist;
  this is intentional)
- Ask you to write `debug-notes.md` mid-debugging (this is a deliberate
  document-and-clear pattern, not a sign that you're failing)

When asked to read external docs, default to spawning a subagent and
returning a syntax cheatsheet rather than pulling full docs into the main
conversation.

## Ship plan

The project ships in five small, independently-useful pieces:

1. **Ingestion + storage** — walk a folder, chunk .md/.txt files, store in
   SQLite with embeddings. `brain stats` shows corpus size.
2. **Vector search** — `brain search` returns top-K matching chunks.
3. **Q&A with citations** — `brain ask` does retrieval + LLM with cited
   answers. Streaming output.
4. **More formats + smart chunking** — PDF, HTML, section-aware chunking.
5. **Eval harness** — known-answer Q&A pairs, accuracy reporting.

Each ship ends with a working command and passing tests. Don't
forward-design for later ships — keep the current ship's code minimal.
