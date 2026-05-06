# Ship 1 — Ingestion + Storage

The smallest version of `brain` that's actually useful: feed it a folder
of text and Markdown files, get back a searchable corpus with embeddings
ready for retrieval.

After this ship, you'll have:

```bash
$ brain ingest ~/Documents/notes
Walking ~/Documents/notes...
  ✓ 47 files found (.md, .txt)
  ✓ 312 chunks created
  ✓ 312 embeddings generated (took 18s)

$ brain stats
Corpus: ~/.brain/corpus.db
  Documents: 47
  Chunks:    312
  Total tokens (approx): 41,200
  Embedding model: nomic-embed-text
  Disk size: 4.2 MB
```

No search, no Q&A yet. Just: walk → chunk → embed → store, plus a
sanity-check command. That's the whole ship.

---

## Why ship this first

Three reasons:

1. **It's the foundation.** Search and Q&A are useless without a corpus.
2. **It exercises the full pipeline minus the LLM.** You'll touch the
   ingestion walker, the chunker, the embedding API, and the storage
   layer all in one ship.
3. **It's testable.** Every component except the embedding API call can
   be tested without network access. The embedder gets mocked.

---

## Scope — exactly four modules

### `src/brain/storage.py`

SQLite schema + CRUD. Two tables:

```sql
CREATE TABLE documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    path        TEXT UNIQUE NOT NULL,    -- absolute path
    mtime       REAL NOT NULL,            -- file modification time (epoch)
    ingested_at TEXT NOT NULL             -- ISO 8601
);

CREATE TABLE chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ord         INTEGER NOT NULL,         -- position within doc, 0-indexed
    text        TEXT NOT NULL,
    embedding   BLOB NOT NULL             -- numpy float32 array, 768-dim
);

CREATE INDEX idx_chunks_document ON chunks(document_id);
```

Functions:

- `init_db(path: str) -> None`
- `upsert_document(path: str, mtime: float) -> int` — returns doc id
- `delete_chunks_for_document(doc_id: int) -> None`
- `insert_chunks(doc_id: int, chunks: list[tuple[str, np.ndarray]]) -> int`
  — list of (text, embedding) pairs; returns count inserted
- `corpus_stats() -> dict` — counts and disk size for `brain stats`

### `src/brain/chunker.py`

One function for now:

- `chunk_text(text: str, *, target_size: int = 500, overlap: int = 50) -> list[str]`

Naive sliding-window chunker by character count. Don't try to be clever
about sentence boundaries yet — that's Ship 4. The point of Ship 1 is to
prove the pipeline works end-to-end, not to optimize retrieval quality.

### `src/brain/embed.py`

Ollama embedding wrapper:

- `embed(texts: list[str], *, model: str | None = None) -> list[np.ndarray]`
  — batch interface, returns list of float32 arrays
- Default model from `BRAIN_EMBED_MODEL` env var or `nomic-embed-text`
- Raises `EmbedError` on connection or model errors

### `src/brain/cli.py`

Two commands:

- `ingest <path>` — walks the directory, processes `.md` and `.txt` files,
  shows a `rich.progress` bar during embedding. Idempotent: re-running on
  the same folder skips files where `mtime` hasn't changed.
- `stats` — prints corpus stats from `corpus_stats()`.

---

## The skip-unchanged-files behavior — important

When a user re-runs `brain ingest` on the same folder, we should not
re-embed everything. Embeddings are slow (~50ms each on a laptop). The
mtime check enables incremental ingestion:

- File path not in DB → ingest fresh
- File path in DB, mtime matches → skip entirely (no chunking, no embedding)
- File path in DB, mtime newer → delete old chunks, re-ingest

This is a small bit of logic but it's the difference between "useful tool"
and "frustrating tool" — running ingest on a 2000-file folder should
take seconds the second time, not minutes.

---

## Tests to write (you should expect ~25–35 tests)

- `test_storage.py` — schema creation, document upsert, chunk insertion,
  cascade delete, stats. All `:memory:`.
- `test_chunker.py` — short text (one chunk), exact boundary, overlap
  behavior, empty input, very long text.
- `test_embed.py` — mock the ollama client; verify model selection,
  batch handling, error wrapping.
- `test_cli.py` — using `tmp_path` and a temp SQLite file: ingest a
  small fixture folder, verify chunk count, verify re-ingest skips
  unchanged files. Use `monkeypatch` to stub the embedder.

---

## What's deliberately out of scope

Don't let Claude over-build this. **Reject scope creep aggressively.**
The following are NOT in Ship 1:

- Search/retrieval (Ship 2)
- Q&A (Ship 3)
- PDF/HTML support (Ship 4)
- Smart chunking by sentence/section (Ship 4)
- Eval harness (Ship 5)
- Configuration files (env vars only)
- Logging frameworks (rich console only)
- Async ingestion (sync is fine for ship 1)
- Multi-corpus support (one corpus, one DB file)
- Embedding cache outside of SQLite

If Claude proposes any of these "while we're here," reject it. Ship 1's
job is to be the smallest thing that earns its keep.

---

## Success criteria

1. All tests pass.
2. `brain ingest <some-folder>` actually works end-to-end against your
   real Ollama daemon and produces a real `~/.brain/corpus.db`.
3. `brain stats` shows nonzero values.
4. Re-running `brain ingest` on the same folder takes <1s and doesn't
   re-embed anything.

---

## Predictions to log before starting

(Same habit from linkstash — predict, then verify.)

1. Will Claude flag the design choice between "store mtime as epoch float"
   vs "ISO string"? Either is fine; the question is whether it's discussed.
2. Will Claude propose using async for the embedding loop? It's a
   reasonable instinct (network I/O in a loop), but the CLAUDE.md says
   sync. Predict: does it follow the rule, or push back?
3. Token budget for Ship 1: predict the % of context after this ship
   completes. Linkstash Session 1 was ~12% messages.

---

## Decision points to anticipate

- **Start of Ship 1**: fresh project, fresh session. No `/clear`
  decision yet — there's nothing to clear.
- **Mid-Ship 1**: probably no decisions. This is "continue" territory.
- **End of Ship 1 → start of Ship 2**: the Decision 1 trap re-runs.
  Storage and chunking context might feel relevant to vector search,
  but the *math* of similarity search is genuinely different from CRUD.
  We'll evaluate when we get there.
