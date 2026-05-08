"""Retrieval + LLM Q&A with citations."""

import os
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass

import httpx
import ollama

from brain import search as search_mod
from brain.search import SearchResult


class QAError(Exception):
    pass


@dataclass
class Answer:
    text_stream: Iterator[str]
    citations: list[str]


def _default_model() -> str:
    return os.environ.get("BRAIN_CHAT_MODEL") or "llama3.2:3b"


def _build_context(results: list[SearchResult]) -> tuple[str, list[str]]:
    citations: list[str] = []
    path_to_num: dict[str, int] = {}
    lines: list[str] = []

    for r in results:
        if r.path not in path_to_num:
            path_to_num[r.path] = len(citations) + 1
            citations.append(r.path)
        num = path_to_num[r.path]
        lines.append(f"[{num}] {r.text}")

    return "\n\n".join(lines), citations


def _build_messages(query: str, context: str) -> list[dict]:
    system = (
        "You answer questions using only the numbered context snippets below. "
        "Cite sources inline as [N] matching the snippet numbers. "
        'If the context does not contain the answer, say '
        '"I don\'t know based on the indexed corpus." Do not invent facts.\n\n'
        f"Context:\n{context}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": query},
    ]


def ask(
    conn: sqlite3.Connection,
    query: str,
    *,
    top_k: int = 5,
    model: str | None = None,
) -> Answer:
    """Retrieve top_k chunks for `query`, ask the LLM, return a streaming Answer.

    Raises QAError on LLM connection or model errors.
    """
    results = search_mod.search(conn, query, top_k=top_k)
    if not results:
        def _no_context() -> Iterator[str]:
            yield "I don't know based on the indexed corpus."
        return Answer(text_stream=_no_context(), citations=[])

    context, citations = _build_context(results)
    messages = _build_messages(query, context)
    resolved_model = model or _default_model()

    try:
        stream = ollama.Client().chat(
            model=resolved_model, messages=messages, stream=True
        )
    except ollama.ResponseError as e:
        raise QAError(str(e)) from e
    except httpx.ConnectError as e:
        raise QAError(str(e)) from e

    def _tokens() -> Iterator[str]:
        try:
            for chunk in stream:
                yield chunk.message.content
        except ollama.ResponseError as e:
            raise QAError(str(e)) from e
        except httpx.ConnectError as e:
            raise QAError(str(e)) from e

    return Answer(text_stream=_tokens(), citations=citations)
