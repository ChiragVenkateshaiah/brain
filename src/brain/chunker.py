"""Document → chunk splitting."""


def chunk_text(
    text: str,
    *,
    target_size: int = 500,
    overlap: int = 50,
) -> list[str]:
    if target_size <= 0:
        raise ValueError(f"target_size must be > 0, got {target_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    if overlap >= target_size:
        raise ValueError(f"overlap ({overlap}) must be < target_size ({target_size})")

    text = text.strip()
    if not text:
        return []

    if len(text) <= target_size:
        return [text]

    stride = target_size - overlap
    chunks: list[str] = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + target_size])
        start += stride
    return chunks
