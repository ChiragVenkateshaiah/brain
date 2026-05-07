"""Ollama embedding-model client."""

import os

import httpx
import numpy as np
import ollama


class EmbedError(Exception):
    pass


def _default_model() -> str:
    return os.environ.get("BRAIN_EMBED_MODEL") or "nomic-embed-text"


def embed(texts: list[str], *, model: str | None = None) -> list[np.ndarray]:
    if not texts:
        return []

    resolved_model = model or _default_model()
    try:
        response = ollama.Client().embed(model=resolved_model, input=texts)
    except ollama.ResponseError as e:
        raise EmbedError(str(e)) from e
    except httpx.ConnectError as e:
        raise EmbedError(str(e)) from e

    return [np.array(row, dtype=np.float32) for row in response.embeddings]
