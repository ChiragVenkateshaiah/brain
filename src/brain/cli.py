"""Typer entry point for the `brain` command."""

import os
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from brain import embed, qa, storage
from brain import search as search_mod
from brain.ingest import SUPPORTED_EXTENSIONS, ingest_file
from brain.parsers import ParseError

app = typer.Typer(add_completion=False)
console = Console()

_HIDDEN_DIRS = {".git", ".venv", "node_modules", "__pycache__"}


def _data_dir() -> Path:
    p = Path(os.environ.get("BRAIN_DATA_DIR", Path.home() / ".brain")).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _db_path() -> Path:
    return _data_dir() / "corpus.db"


def _walk_files(root: Path):
    for p in root.rglob("*"):
        if any(part.startswith(".") or part in _HIDDEN_DIRS for part in p.parts[len(root.parts):]):
            continue
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield p


@app.command()
def ingest(path: Path = typer.Argument(..., help="Directory to ingest")) -> None:
    """Walk a directory and ingest .md and .txt files into the corpus."""
    if not path.exists() or not path.is_dir():
        console.print(f"[red]Error:[/red] {path} is not a directory")
        raise typer.Exit(1)

    db_path = _db_path()
    conn = storage.connect(str(db_path))
    storage.init_db(conn)

    files = list(_walk_files(path))
    if not files:
        console.print(f"No supported files found in {path}")
        return

    processed = 0
    skipped = 0
    total_chunks = 0
    start = time.monotonic()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Ingesting", total=len(files))

        for file_path in files:
            abs_path = str(file_path.resolve())
            try:
                mtime = file_path.stat().st_mtime
            except OSError as e:
                console.print(f"[yellow]Warning:[/yellow] could not stat {file_path}: {e}")
                progress.advance(task)
                continue

            existing = storage.get_document(conn, abs_path)
            if existing and existing[1] == mtime:
                skipped += 1
                progress.advance(task)
                continue

            try:
                n_chunks = ingest_file(conn, file_path)
            except OSError as e:
                console.print(f"[yellow]Warning:[/yellow] could not read {file_path}: {e}")
                progress.advance(task)
                continue
            except ParseError as e:
                console.print(f"[yellow]Warning:[/yellow] {e}")
                progress.advance(task)
                continue
            except embed.EmbedError as e:
                console.print(f"[yellow]Warning:[/yellow] embedding failed for {file_path}: {e}")
                progress.advance(task)
                continue

            total_chunks += n_chunks
            processed += 1
            progress.advance(task)

    elapsed = time.monotonic() - start
    console.print(
        f"[green]Done[/green] — {processed} files ingested, {skipped} skipped, "
        f"{total_chunks} chunks created ({elapsed:.1f}s)"
    )


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    top: int = typer.Option(5, "--top", "-k", help="Number of results"),
) -> None:
    """Search the corpus for chunks matching the query."""
    if not query.strip():
        console.print("[red]Error:[/red] empty query")
        raise typer.Exit(1)

    db_path = _db_path()
    if not db_path.exists():
        console.print("No corpus found. Run [bold]brain ingest[/bold] first.")
        raise typer.Exit(1)

    conn = storage.connect(str(db_path))
    storage.init_db(conn)

    try:
        results = search_mod.search(conn, query, top_k=top)
    except embed.EmbedError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    if not results:
        console.print("No results.")
        return

    table = Table(show_header=True, box=None, padding=(0, 2))
    table.add_column("#", style="bold")
    table.add_column("Score")
    table.add_column("Document")
    table.add_column("Snippet")

    for rank, r in enumerate(results, 1):
        snippet = " ".join(r.text.split())
        if len(snippet) > 117:
            snippet = snippet[:117] + "…"
        table.add_row(str(rank), f"{r.score:.4f}", r.path, snippet)

    console.print(table)


@app.command()
def ask(
    query: str = typer.Argument(..., help="Question to answer"),
    top: int = typer.Option(5, "--top", "-k", help="Chunks to retrieve"),
    model: str | None = typer.Option(None, "--model", "-m", help="Override chat model"),
) -> None:
    """Answer a question using retrieved chunks from the corpus."""
    if not query.strip():
        console.print("[red]Error:[/red] empty question")
        raise typer.Exit(1)

    db_path = _db_path()
    if not db_path.exists():
        console.print("No corpus found. Run [bold]brain ingest[/bold] first.")
        raise typer.Exit(1)

    conn = storage.connect(str(db_path))
    storage.init_db(conn)

    try:
        answer = qa.ask(conn, query, top_k=top, model=model)
        for token in answer.text_stream:
            console.print(token, end="", soft_wrap=True)
        console.print()

        if answer.citations:
            console.print()
            console.print("[bold]Sources:[/bold]")
            for i, path in enumerate(answer.citations, 1):
                console.print(f"  [{i}] {path}")
    except (embed.EmbedError, qa.QAError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def stats() -> None:
    """Show corpus statistics."""
    db_path = _db_path()
    if not db_path.exists():
        console.print("No corpus found. Run [bold]brain ingest[/bold] first.")
        raise typer.Exit(1)

    conn = storage.connect(str(db_path))
    storage.init_db(conn)
    s = storage.corpus_stats(conn, str(db_path))

    size_bytes = s["disk_size_bytes"]
    if size_bytes >= 1024 * 1024:
        size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        size_str = f"{size_bytes / 1024:.1f} KB"

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="bold")
    table.add_column("Value")
    table.add_row("Corpus", s["db_path"])
    table.add_row("Documents", str(s["documents"]))
    table.add_row("Chunks", str(s["chunks"]))
    table.add_row("Total tokens (approx)", str(s["approx_tokens"]))
    table.add_row("Embedding model", s["embed_model"])
    table.add_row("Disk size", size_str)

    console.print(table)
