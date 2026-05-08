"""Document → chunk splitting."""

import re


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


_HEADING_RE = re.compile(r"^#{1,6}\s+\S", re.MULTILINE)


def chunk_markdown(
    text: str,
    *,
    target_size: int = 500,
    overlap: int = 50,
) -> list[str]:
    """Section-aware chunking for markdown.

    Splits on ATX headings, prepends the heading to every chunk from that
    section so retrieval sees section context. Falls back to chunk_text when
    there are no headings.
    """
    if target_size <= 0:
        raise ValueError(f"target_size must be > 0, got {target_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    if overlap >= target_size:
        raise ValueError(f"overlap ({overlap}) must be < target_size ({target_size})")

    text = text.strip()
    if not text:
        return []

    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return chunk_text(text, target_size=target_size, overlap=overlap)

    sections: list[tuple[str, str]] = []
    # Text before the first heading (no section prefix)
    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append(("", preamble))

    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[m.start() : end]
        heading_end = section_text.index("\n") if "\n" in section_text else len(section_text)
        heading = section_text[:heading_end].strip()
        body = section_text[heading_end:].strip()
        sections.append((heading, body))

    result: list[str] = []
    for heading, body in sections:
        if not body:
            if heading:
                result.append(heading)
            continue
        body_chunks = chunk_text(body, target_size=target_size, overlap=overlap)
        for chunk in body_chunks:
            result.append(f"{heading}\n\n{chunk}" if heading else chunk)
    return result
