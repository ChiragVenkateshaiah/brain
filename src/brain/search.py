"""Vector similarity search over the stored corpus."""

import sqlite3
from dataclasses import dataclass

import numpy as np

from brain import embed, storage


@dataclass(frozen=True)
class SearchResult:
    chunk_id: int
    path: str
    text: str
    score: float


def search(
    conn: sqlite3.Connection,
    query: str,
    *,
    top_k: int = 5,
) -> list[SearchResult]:
    """Embed `query`, rank all chunks by cosine similarity, return top_k."""
    q = embed.embed([query])[0]

    rows = storage.get_all_chunks_with_embeddings(conn)
    if not rows:
        return []

    M = np.vstack([r[3] for r in rows])
    q_norm = np.linalg.norm(q)
    m_norms = np.linalg.norm(M, axis=1)
    scores = (M @ q) / (m_norms * q_norm + 1e-12)

    k = min(top_k, len(rows))
    idx = np.argsort(scores)[-k:][::-1]

    return [
        SearchResult(
            chunk_id=rows[i][0],
            path=rows[i][1],
            text=rows[i][2],
            score=float(scores[i]),
        )
        for i in idx
    ]
